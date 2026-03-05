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
from emg2qwerty.vis_transformer import ConvVit

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

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

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


class ConvVitCTCModule(pl.LightningModule):
    def __init__(
        self,
        n_filters1: int,
        n_filters2: int = 128,
        n_head: int = 8,
        n_layers: int = 2,
        optimizer: DictConfig = None,
        lr_scheduler: DictConfig = None,
        decoder: Any = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.chars = charset()

        self.model = ConvVit(
            n_filters1=n_filters1,
            n_filters2=n_filters2,
            n_head=n_head,
            n_layers=n_layers,
            n_classes=self.chars.num_classes
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)
        
        if isinstance(decoder, (dict, DictConfig)):
            self.decoder = hydra.utils.instantiate(decoder)
        elif decoder == "greedy":
            self.decoder = CTCGreedyDecoder()
        else:
            self.decoder = decoder
            
        # Standard metrics setup
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if self.hparams.lr_scheduler:
            scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"] # [T, N, C, H, W]
        targets = batch["targets"] # [S, N]
        target_lengths = batch["target_lengths"]
        
        T_win, N, C, H, W = inputs.shape

        logits = self.forward(inputs.reshape(-1, C, H, W)) # [T*N, 36, Classes]
        num_tokens = logits.shape[1]
        
        # Reshape to CTC format: [Time_Total, Batch, Classes]
        emissions = logits.view(T_win, N, num_tokens, -1)
        emissions = emissions.permute(0, 2, 1, 3).reshape(-1, N, self.chars.num_classes)
        
        # keep these on cpu, since CTCLoss requirement
        T_total = emissions.shape[0]
        emission_lengths = torch.full((N,), T_total, device=inputs.device, dtype=torch.long)

        loss = self.ctc_loss(
            log_probs=emissions,            
            targets=targets.transpose(0, 1), 
            input_lengths=emission_lengths,
            target_lengths=target_lengths
        )


        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)
        
        if self.decoder is not None and phase != "train" and targets.shape[0] > 0:
            
            emissions_np = emissions.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            dummy_timestamps = np.arange(T_total) 
            metrics = self.metrics[f"{phase}_metrics"]

            for i in range(N):
                self.decoder.reset()
                item_emissions = emissions_np[:, i, :]
                pred_label_data = self.decoder.decode(
                    emissions=item_emissions,
                    timestamps=dummy_timestamps
                )

                current_target_len = int(target_lengths[i])
                current_target_labels = targets_np[:current_target_len, i]
                target_label_data = LabelData.from_labels(current_target_labels)

                metrics.update(prediction=pred_label_data, target=target_label_data)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metrics["val_metrics"].compute(), sync_dist=True)
        self.metrics["val_metrics"].reset()