# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# Spectrogram front-end
# ---------------------------------------------------------------------------


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


# ---------------------------------------------------------------------------
# TDS convolutional encoder (baseline)
# ---------------------------------------------------------------------------


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


# ---------------------------------------------------------------------------
# ResNet 1-D encoder (raw sEMG)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CNN/RNN hybrid encoders (GRU and LSTM variants)
# ---------------------------------------------------------------------------


class CNNRNNEncoder(nn.Module):
    """A hybrid encoder composing a `TDSConvEncoder` followed by a recurrent
    layer (GRU).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock` in the CNN encoder.
        kernel_width (int): The kernel size of the temporal convolutions.
        rnn_hidden_size (int): The hidden size of the GRU layer.
        rnn_num_layers (int): The number of layers in the GRU.
        rnn_bidirectional (bool): Whether the GRU is bidirectional.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        rnn_hidden_size: int = 384,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.cnn_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
        )

        rnn_out_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.out_projection = nn.Linear(rnn_out_size, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.cnn_encoder(inputs)  # (T_cnn, N, num_features)
        x, _ = self.rnn(x)  # (T_cnn, N, rnn_out_size)
        x = self.out_projection(x)  # (T_cnn, N, num_features)
        return self.layer_norm(x)


class CNNLSTMEncoder(nn.Module):
    """A hybrid encoder composing a `TDSConvEncoder` followed by a recurrent
    layer (LSTM).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock` in the CNN encoder.
        kernel_width (int): The kernel size of the temporal convolutions.
        rnn_hidden_size (int): The hidden size of the LSTM layer.
        rnn_num_layers (int): The number of layers in the LSTM.
        rnn_bidirectional (bool): Whether the LSTM is bidirectional.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        rnn_hidden_size: int = 384,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.cnn_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )

        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
        )

        rnn_out_size = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.out_projection = nn.Linear(rnn_out_size, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.cnn_encoder(inputs)  # (T_cnn, N, num_features)
        x, _ = self.rnn(x)  # (T_cnn, N, rnn_out_size)
        x = self.out_projection(x)  # (T_cnn, N, num_features)
        return self.layer_norm(x)


# ---------------------------------------------------------------------------
# Pure RNN encoder (no convolutions)
# ---------------------------------------------------------------------------


class ResidualBiGRUBlock(nn.Module):
    """A single bidirectional GRU layer with a residual connection, dropout,
    and LayerNorm.

    When ``input_size == hidden_size * 2``, a direct residual connection is
    applied (no extra parameters). Otherwise a bias-free linear projection
    aligns the residual to the output dimension.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): GRU hidden units per direction.
            The output size of this block is ``hidden_size * 2``.
        dropout (float): Dropout probability applied to the GRU output
            before the residual addition. (default: ``0.2``)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        output_size = hidden_size * 2

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.residual_proj: nn.Module = (
            nn.Identity()
            if input_size == output_size
            else nn.Linear(input_size, output_size, bias=False)
        )

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, input_size)
        x, _ = self.gru(inputs)         # (T, N, hidden_size * 2)
        x = self.dropout(x)
        x = x + self.residual_proj(inputs)
        return self.layer_norm(x)       # (T, N, hidden_size * 2)


class DeepBiGRUEncoder(nn.Module):
    """A purely recurrent encoder using stacked bidirectional GRU layers,
    each equipped with residual connections, inter-layer dropout, and LayerNorm.

    Unlike the ``TDSConvEncoder``, this encoder contains **no convolutional
    layers** -- all temporal feature extraction is performed recurrently.
    The time dimension ``T`` is preserved end-to-end (no shrinkage).

    Args:
        num_features (int): Input (and output) feature dimension.
        hidden_size (int): GRU hidden units per direction.
        num_layers (int): Number of stacked ``ResidualBiGRUBlock`` modules.
        dropout (float): Dropout probability applied between GRU blocks.
            The final block always uses ``dropout=0.0``. (default: ``0.2``)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 384,
        num_layers: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        assert num_layers >= 1, "num_layers must be at least 1"

        output_size = hidden_size * 2

        if num_features != output_size:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(num_features, output_size),
                nn.LayerNorm(output_size),
            )
            layer_input_size = output_size
        else:
            self.input_proj = nn.Identity()
            layer_input_size = num_features

        self.layers = nn.ModuleList(
            [
                ResidualBiGRUBlock(
                    input_size=layer_input_size if i == 0 else output_size,
                    hidden_size=hidden_size,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                )
                for i in range(num_layers)
            ]
        )

        self.out_projection = nn.Linear(output_size, num_features)
        self.out_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.input_proj(inputs)
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(self.out_projection(x))  # (T, N, num_features)


# ---------------------------------------------------------------------------
# Conv-ViT encoder (transformer)
# ---------------------------------------------------------------------------


class ConvVit(nn.Module):
    """Per-frame feature extractor using a convolutional stem followed by a
    Vision Transformer.  Processes each EMG spectrogram frame independently
    and outputs a feature vector per frame.  Temporal modelling is handled
    externally (e.g. by ``TDSConvEncoder`` in the Lightning module).

    Args:
        in_channels: Number of input channels (bands).
        n_filters1: Channels after the first conv layer.
        n_filters2: Channels after the second conv / transformer dimension.
        kernel_size: Kernel size for the conv stem.
        n_head: Number of attention heads in the transformer.
        n_layers: Number of transformer encoder layers.
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_filters1: int = 32,
        n_filters2: int = 128,
        kernel_size: int = 3,
        n_head: int = 8,
        n_layers: int = 2,
    ) -> None:
        super().__init__()

        self.n_filters2 = n_filters2

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_filters1, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(n_filters1),
            nn.GELU(),
            nn.Conv2d(n_filters1, n_filters2, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(n_filters2),
            nn.GELU(),
        )

        self.num_tokens: int | None = None
        self.pos_embed: nn.Parameter | None = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_filters2,
            nhead=n_head,
            dim_feedforward=512,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=0.1,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.norm = nn.LayerNorm(n_filters2)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="linear"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        for name, p in self.transformer.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1)

    def _init_pos_embed(self, H: int, W: int) -> None:
        self.num_tokens = H * W
        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                self.num_tokens,
                self.n_filters2,
                device=next(self.parameters()).device,
            )
            * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.stem(x)
        B, C, H, W = x.shape

        if self.pos_embed is None:
            self._init_pos_embed(H, W)

        if H * W != self.num_tokens:
            raise RuntimeError(
                f"Spatial size changed: got {H}x{W} ({H * W} tokens), "
                f"but pos_embed was built for {self.num_tokens} tokens."
            )

        tokens = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        if self.training:
            mask = (
                torch.rand(B, tokens.size(1), 1, device=x.device) > 0.05
            ).float()
            tokens = tokens * mask

        tokens = tokens + self.pos_embed

        vit_out = self.transformer(tokens)
        vit_out = self.norm(vit_out)
        vit_out = self.dropout(vit_out)

        return vit_out.mean(dim=1)  # [B, n_filters2]
