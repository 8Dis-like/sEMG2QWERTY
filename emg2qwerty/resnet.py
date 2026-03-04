# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ResNet-based 1D convolutional encoder for raw sEMG signals.

This module provides a ResNet architecture adapted for temporal 1D signals,
designed to process raw EMG input of shape ``(T, N, bands, electrode_channels)``
— typically ``(T, N, 2, 16)`` — through strided residual blocks and produce
per-timestep feature representations suitable for CTC-based decoding.

Usage with the existing Hydra/Lightning training infrastructure::

    python -m emg2qwerty.train model=resnet_ctc transforms=raw_augmented
"""

from collections.abc import Sequence

import torch
from torch import nn


class ResidualBlock1D(nn.Module):
    """1D residual block with two convolutions and a skip connection.

    Standard ResNet BasicBlock pattern adapted for 1D temporal signals.
    A 1x1 projection shortcut is used when ``stride > 1`` or
    ``in_channels != out_channels``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for both convolutions.
        stride: Stride for the first convolution (controls temporal downsampling).
        dropout: Dropout probability applied between convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1DEncoder(nn.Module):
    """ResNet-based 1D convolutional encoder for temporal sEMG signals.

    Processes time-first input ``(T, N, C_in)`` through a stem and multiple
    stages of residual blocks, producing ``(T', N, out_channels)`` where
    ``T'`` reflects temporal downsampling from strided layers.

    Architecture:
      - **Stem**: 7-wide conv (stride 2) -> BN -> ReLU -> max-pool (stride 2)
        giving 4x temporal downsampling.
      - **Stages**: each stage is a stack of :class:`ResidualBlock1D` modules.
        Stage 0 preserves temporal resolution; subsequent stages downsample 2x
        via the first block's strided convolution.

    With the default 3 stages the total temporal downsampling factor is
    ``4 * 2^(num_stages - 1) = 16``, which mirrors the hop length of the
    spectrogram transform used by the TDS baseline.

    Args:
        in_channels: Input feature dimension (e.g. ``bands * electrodes = 32``).
        channels: Channel width for each residual stage.
        layers_per_stage: Number of residual blocks in each stage.
        kernel_size: Convolution kernel size inside residual blocks.
        dropout: Dropout rate within residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 32,
        channels: Sequence[int] = (64, 128, 256),
        layers_per_stage: Sequence[int] = (2, 2, 2),
        kernel_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(channels) == len(layers_per_stage) and len(channels) > 0

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        self.stages = nn.ModuleList()
        self._stage_strides: list[int] = []
        in_ch = channels[0]
        for i, (n_blocks, out_ch) in enumerate(zip(layers_per_stage, channels)):
            stride = 1 if i == 0 else 2
            self._stage_strides.append(stride)
            blocks: list[nn.Module] = [
                ResidualBlock1D(in_ch, out_ch, kernel_size, stride, dropout),
            ]
            for _ in range(1, n_blocks):
                blocks.append(
                    ResidualBlock1D(out_ch, out_ch, kernel_size, 1, dropout),
                )
            self.stages.append(nn.Sequential(*blocks))
            in_ch = out_ch

        self.out_channels = in_ch

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths given raw input lengths.

        Each strided layer with ``padding = (kernel - 1) // 2`` produces
        ``ceil(L_in / stride)`` output frames, computed here as
        ``(L_in - 1) // stride + 1`` for positive integer lengths.
        """
        lengths = input_lengths
        # Stem: conv (stride 2) + maxpool (stride 2)
        lengths = (lengths - 1) // 2 + 1
        lengths = (lengths - 1) // 2 + 1
        for stride in self._stage_strides:
            if stride > 1:
                lengths = (lengths - 1) // stride + 1
        return lengths

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: ``(T, N, C_in)`` time-first tensor.

        Returns:
            ``(T', N, out_channels)`` time-first tensor where
            ``T' = compute_output_lengths(T)``.
        """
        x = inputs.permute(1, 2, 0)  # (N, C, T)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x.permute(2, 0, 1)  # (T', N, C_out)
