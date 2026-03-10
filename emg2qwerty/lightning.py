# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar
import hydra

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform
from emg2qwerty.decoder import CTCGreedyDecoder
import torch.nn.functional as F
import math
import os
from emg2qwerty.vis_transformer import EMGFeatureExtractor, TemporalTransformerEncoder
from emg2qwerty.comformer import EMGConformerCTC


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.metrics["train_metrics"].compute(), sync_dist=True)
        self.metrics["train_metrics"].reset()

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

class TransformerCTCModule(pl.LightningModule):
    def __init__(
            self, 
            in_features: int = 528,
            mlp_features: int = [384],
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 515,
            dropout: float = 0.1,
            optimizer: DictConfig = None,
            lr_scheduler: DictConfig = None,
            decoder: Any = None,
            tds_checkpoint: str = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.debug = kwargs.get("debug", False)

        self.feature_extractor = EMGFeatureExtractor(in_features= in_features, mlp_features= mlp_features)
        d_model = self.feature_extractor.num_features

        self.transformer = TemporalTransformerEncoder(
            d_model= d_model,
            nhead= nhead,
            num_layers= num_layers,
            dim_feedforward= dim_feedforward,
            dropout= dropout,
        )

        self.classifier = nn.Linear(d_model, charset().num_classes)
        nn.init.normal_(self.classifier.weight, mean=0, std=0.001)
        nn.init.constant_(self.classifier.bias, -math.log(charset().num_classes))
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        # Hardcode for now to avoid Hydra parsing issues
        # TDS_CKPT = "/home/danielluzzatto2/sEMG2QWERTY/logs/2026-03-08/05-46-34/checkpoints/epoch=9-step=4800.ckpt"

        # if tds_checkpoint is not None:
        #     self._load_tds_feature_extractor(tds_checkpoint)
        # elif os.path.exists(TDS_CKPT):
        #     self._load_tds_feature_extractor(TDS_CKPT)
        
        
        if decoder == "greedy":
            self.decoder = CTCGreedyDecoder()
        elif isinstance(decoder, (dict, DictConfig)):
            self.decoder = hydra.utils.instantiate(decoder)
        else:
            self.decoder = decoder

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def _load_tds_feature_extractor(self, checkpoint_path: str):
        print(f"Loading TDS feature extractor from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state = ckpt['state_dict']

        mapping = {}
        for k, v in state.items():
            if k.startswith('model.0.'):
                mapping[k.replace('model.0.', 'norm.')] = v
            elif k.startswith('model.1.'):
                mapping[k.replace('model.1.', 'mlp.')] = v

        missing, unexpected = self.feature_extractor.load_state_dict(mapping, strict=False)
        print(f"  Missing keys: {missing}")
        print(f"  Unexpected keys: {unexpected}")

        # Freeze both feature extractor and classifier — only transformer trains initially
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        print("  Feature extractor and classifier frozen.")

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log('train/lr', lr, on_epoch=True, prog_bar=False)
        # Unfreeze classifier at epoch 3 so it can adapt to transformer outputs
        if self.current_epoch == 3:
            for param in self.classifier.parameters():
                param.requires_grad = True
            print("  Classifier unfrozen.")
        # Unfreeze feature extractor at epoch 6 for full fine-tuning
        if self.current_epoch == 6:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            print("  Feature extractor unfrozen.")
    
    def forward(self, x,chunk_size = 2000):
        x = self.feature_extractor(x)   # (T, N, 768)

        if not self.training and x.size(0) > chunk_size:
            chunks = x.split(chunk_size, dim=0)
            x = torch.cat([self.transformer(chunk) for chunk in chunks], dim=0)
        else:
            x = self.transformer(x) # (T, N, 768)
        x = self.classifier(x)           # (T, N, 99)
        return F.log_softmax(x, dim=-1)
    
    def _step(self, phase, batch, batch_idx):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        # Filter zero-length targets
        valid = target_lengths > 0
        if valid.sum() == 0:
            return None

        inputs = inputs[:, valid]
        targets = targets[:, valid]
        input_lengths = input_lengths[valid]
        target_lengths = target_lengths[valid]
        N = valid.sum().item()

        emissions = self.forward(inputs)  # (T, N, 99)
        
        # blank_lp = emissions[:, :, charset().null_class].exp()  # (T, N)
        # blank_penalty = blank_lp.mean()

        T = emissions.size(0)
        input_lengths_clamped = input_lengths.clamp(max=T).cpu().long()
        target_lengths_cpu = target_lengths.cpu().long()

        loss = self.ctc_loss(
            emissions,
            targets.transpose(0, 1),  # (N, S)
            input_lengths_clamped,
            target_lengths_cpu,
        )

        self.log(f"{phase}/loss", loss, batch_size=N, prog_bar=True, sync_dist=True)

        # if phase == "train" and batch_idx == 1:
        #     # print(f"  emissions std at batch 1: {emissions.std().item():.6f}")
        #     # print(f"  blank prob at batch 1: {emissions[:,:,self.charset.null_class].exp().mean().item():.4f}")
        #     for name, param in self.named_parameters():
        #         if param.grad is not None:
        #             print(f"  grad {name}: {param.grad.norm().item():.6f}")

        if self.debug and batch_idx % 250 == 0:
            preds = emissions.argmax(dim=-1)
            blank_ratio = (preds == charset().null_class).float().mean().item()
            # current_lr = self.optimizers().param_groups[0]['lr']
            # print(f"[{phase}] batch {batch_idx}: loss={loss.item():.3f}  blank={blank_ratio:.3f}  max_lp={emissions.max().item():.3f}, mean_lp={emissions.mean().item():.3f}, lr={current_lr:.5f}")

        if self.decoder and phase != "train" and target_lengths_cpu.sum() > 0:
            emissions_np = emissions.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            metrics = self.metrics[f"{phase}_metrics"]
            for i in range(N):
                tgt_len = int(target_lengths_cpu[i])
                elen = int(input_lengths_clamped[i])
                self.decoder.reset()
                pred = self.decoder.decode(
                    emissions=emissions_np[:elen, i, :],
                    timestamps=np.arange(elen),
                )
                target_data = LabelData.from_labels(targets_np[:tgt_len, i])
                metrics.update(prediction=pred, target=target_data)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def on_validation_epoch_end(self):
        computed = self.metrics["val_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["val_metrics"].reset()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if self.hparams.lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.scheduler,
                optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.lr_scheduler.get("interval", "epoch"),
                    "monitor": self.hparams.lr_scheduler.get("monitor", None),
                }
            }
        return optimizer


    def on_train_epoch_end(self):
        computed = self.metrics["train_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["train_metrics"].reset()

    def on_load_checkpoint(self, checkpoint):
    # Replace pe buffer with correct size if it doesn't match
        pe_key = 'transformer.pos_encoding.pe'
        if pe_key in checkpoint['state_dict']:
            if checkpoint['state_dict'][pe_key].shape[0] != 200000:
                del checkpoint['state_dict'][pe_key]
                # It will be filled by the model's own buffer after loading
                # We need to add it back with the right shape
                checkpoint['state_dict'][pe_key] = self.transformer.pos_encoding.pe
    
    def on_test_epoch_end(self):
        computed = self.metrics["test_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["test_metrics"].reset()


class ConformerCTCModule(pl.LightningModule):
    def __init__(
            self,
            in_features: int = 528,
            mlp_features: list = [384],
            d_model: int = 768,
            nhead: int = 8,
            num_layers: int = 4,
            kernel_size: int = 31,
            dropout: float = 0.1,
            optimizer: DictConfig = None,
            lr_scheduler: DictConfig = None,
            decoder: Any = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.debug = kwargs.get("debug", False)

        self.model = EMGConformerCTC(
            in_features=in_features,
            mlp_features=mlp_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=charset().num_classes,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        nn.init.normal_(self.model.ctc_head.weight, mean = 0, std=0.001)
        nn.init.constant_(self.model.ctc_head.bias, -math.log(charset().num_classes))

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        if decoder == "greedy":
            self.decoder = CTCGreedyDecoder()
        elif isinstance(decoder, (dict, DictConfig)):
            self.decoder = hydra.utils.instantiate(decoder)
        else:
            self.decoder = decoder

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, x):
        return self.model(x)  # (T, N, num_classes)

    def _step(self, phase, batch, batch_idx):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        # Filter zero-length targets
        valid = target_lengths > 0
        if valid.sum() == 0:
            return None

        inputs = inputs[:, valid]
        targets = targets[:, valid]
        input_lengths = input_lengths[valid]
        target_lengths = target_lengths[valid]
        N = valid.sum().item()

        emissions = self.forward(inputs)  # (T, N, num_classes)

        T = emissions.size(0)
        input_lengths_clamped = input_lengths.clamp(max=T).cpu().long()
        target_lengths_cpu = target_lengths.cpu().long()

        loss = self.ctc_loss(
            emissions,
            targets.transpose(0, 1),  # (N, S)
            input_lengths_clamped,
            target_lengths_cpu,
        )

        self.log(f"{phase}/loss", loss, batch_size=N, prog_bar=True, sync_dist=True)

        if self.debug and batch_idx % 250 == 0:
            preds = emissions.argmax(dim=-1)
            blank_ratio = (preds == charset().null_class).float().mean().item()
            print(f"[{phase}] batch {batch_idx}: loss={loss.item():.3f}  blank={blank_ratio:.3f}")

        if self.decoder and phase != "train" and target_lengths_cpu.sum() > 0:
            emissions_np = emissions.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            metrics = self.metrics[f"{phase}_metrics"]
            for i in range(N):
                tgt_len = int(target_lengths_cpu[i])
                elen = int(input_lengths_clamped[i])
                self.decoder.reset()
                pred = self.decoder.decode(
                    emissions=emissions_np[:elen, i, :],
                    timestamps=np.arange(elen),
                )
                target_data = LabelData.from_labels(targets_np[:tgt_len, i])
                metrics.update(prediction=pred, target=target_data)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]['lr']
            self.log('train/lr', lr, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        computed = self.metrics["train_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["train_metrics"].reset()

    def on_validation_epoch_end(self):
        computed = self.metrics["val_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["val_metrics"].reset()

    def on_test_epoch_end(self):
        computed = self.metrics["test_metrics"].compute()
        self.log_dict(computed, sync_dist=True)
        self.metrics["test_metrics"].reset()

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if self.hparams.lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.scheduler,
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.lr_scheduler.get("interval", "epoch"),
                    "monitor": self.hparams.lr_scheduler.get("monitor", None),
                },
            }
        return optimizer    