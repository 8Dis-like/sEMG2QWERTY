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
from emg2qwerty.resnet import ResNet1DEncoder
from emg2qwerty.transforms import Transform
from emg2qwerty.decoder import CTCGreedyDecoder
from emg2qwerty.vis_transformer import ConvVit
import torch.nn.functional as F


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


class ResNetCTCModule(pl.LightningModule):
    """A PyTorch Lightning module pairing :class:`ResNet1DEncoder` with CTC
    loss for decoding raw sEMG signals into character sequences.

    Unlike :class:`TDSConvCTCModule` which operates on log-spectrogram
    features ``(T, N, 2, 16, freq)``, this module takes raw EMG input of
    shape ``(T, N, 2, 16)`` and learns temporal representations end-to-end
    through 1D residual convolutions with built-in temporal downsampling.

    The training loop is fully integrated with PyTorch Lightning: CTC loss
    computation, greedy/beam decoding, and CER metrics are handled by
    ``training_step`` / ``validation_step`` / ``test_step``.

    Run via Hydra::

        python -m emg2qwerty.train model=resnet_ctc transforms=raw_augmented

    Args:
        channels: Channel widths for each residual stage.
        layers_per_stage: Number of residual blocks per stage.
        kernel_size: Convolution kernel size in residual blocks.
        dropout: Dropout probability within residual blocks.
        optimizer: Hydra config for the optimizer.
        lr_scheduler: Hydra config for the learning-rate scheduler.
        decoder: Hydra config for the CTC decoder.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        channels: Sequence[int],
        layers_per_stage: Sequence[int],
        kernel_size: int,
        dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        in_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS

        self.encoder = ResNet1DEncoder(
            in_channels=in_channels,
            channels=channels,
            layers_per_stage=layers_per_stage,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.encoder.out_channels, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        self.decoder = instantiate(decoder)

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands=2, C=16)
        T, N, bands, C = inputs.shape
        x = inputs.reshape(T, N, bands * C)  # (T, N, 32)
        x = self.encoder(x)  # (T', N, encoder.out_channels)
        return self.output_proj(x)  # (T', N, num_classes)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        emissions = self.forward(inputs)

        # Strided convolutions change the temporal length non-linearly,
        # so we compute per-sample output lengths via the encoder formula
        # rather than using a constant T_diff offset.
        emission_lengths = self.encoder.compute_output_lengths(input_lengths)

        loss = self.ctc_loss(
            log_probs=emissions,  # (T', N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
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
    ):
        super().__init__()

        self.save_hyperparameters()

        self.debug = kwargs.get("debug", False)

        self.charset = charset()

        self.model = ConvVit(
            n_filters1=n_filters1,
            n_filters2=n_filters2,
            n_head=n_head,
            n_layers=n_layers,
        )

        self.temporal_downsample = nn.Conv1d(
            in_channels=n_filters2,
            out_channels=n_filters2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.temporal_encoder = TDSConvEncoder(
            num_features=n_filters2,
            block_channels=(n_filters2 // 4,) * 4,
            kernel_width=32,
        )

        self.classifier = nn.Linear(n_filters2, self.charset.num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        for m in self.temporal_encoder.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.ctc_loss = nn.CTCLoss(
            blank=self.charset.null_class,
            zero_infinity=True
        )

        if isinstance(decoder, (dict, DictConfig)):
            self.decoder = hydra.utils.instantiate(decoder)
        elif decoder == "greedy":
            self.decoder = CTCGreedyDecoder()
        else:
            self.decoder = decoder

        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            params=self.parameters()
        )
        if self.hparams.lr_scheduler:
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler,
                optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
        return optimizer

    def _step(self, phase: str, batch, batch_idx: int = 0):
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]

        T, N, C, H, W = inputs.shape

        frame_features = self.model(inputs.reshape(-1, C, H, W))
        frame_features = frame_features.view(T, N, -1)

        x = frame_features.permute(1, 2, 0)
        x = self.temporal_downsample(x)
        frame_features = x.permute(2, 0, 1)
        input_lengths_ds = (input_lengths - 1) // 2 + 1

        temporal_out = self.temporal_encoder(frame_features)

        emissions = F.log_softmax(
            self.classifier(temporal_out), dim=-1
        ).contiguous()

        T_diff = frame_features.size(0) - temporal_out.size(0)
        emission_lengths = (input_lengths_ds - T_diff).cpu().to(torch.long).clamp(min=0)

        target_lengths_cpu = target_lengths.cpu().to(torch.long)

        if target_lengths_cpu.sum() > 0:
            loss = self.ctc_loss(
                log_probs=emissions,
                targets=targets.transpose(0, 1),
                input_lengths=emission_lengths,
                target_lengths=target_lengths_cpu,
            )
        else:
            loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True, prog_bar=True)

        if self.decoder and phase != "train" and target_lengths_cpu.sum() > 0:
            emissions_np = emissions.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            metrics = self.metrics[f"{phase}_metrics"]
            emission_lengths_list = emission_lengths.tolist()

            for i in range(N):
                tgt_len = int(target_lengths_cpu[i])
                if tgt_len == 0:
                    continue

                elen = emission_lengths_list[i]
                self.decoder.reset()
                pred = self.decoder.decode(
                    emissions=emissions_np[:elen, i, :],
                    timestamps=np.arange(elen),
                )
                target_data = LabelData.from_labels(targets_np[:tgt_len, i])
                metrics.update(prediction=pred, target=target_data)

        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")
