from collections.abc import Sequence
from typing import Any, ClassVar

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm


class ResidualBiGRUBlock(nn.Module):
    """A single bidirectional GRU layer with a residual connection, dropout,
    and LayerNorm.

    When ``input_size == hidden_size * 2``, a direct residual connection is
    applied (no extra parameters). Otherwise a bias-free linear projection
    aligns the residual to the output dimension — a convention borrowed from
    He et al. ResNet projection shortcuts.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): GRU hidden units per direction.
            The output size of this block is ``hidden_size * 2``.
        dropout (float): Dropout probability applied to the GRU output
            before the residual addition. Set to ``0.0`` on the last block
            to preserve the encoder's final representations. (default: ``0.2``)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        output_size = hidden_size * 2  # concatenated forward + backward states

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(p=dropout)

        # Projection for the residual shortcut when sizes differ.
        # bias=False is standard practice for residual projections.
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

    Unlike the ``TDSConvEncoder`` used in the CNN/RNN hybrid models, this
    encoder contains **no convolutional layers**. All temporal feature
    extraction is performed recurrently. As a result:

    - The time dimension ``T`` is preserved end-to-end (no shrinkage).
    - Each layer has access to the full sequence context from the first step.
    - The deeper stack (default 4 layers) compensates for the absent CNN
      inductive bias by building hierarchical recurrent representations.

    The first layer may include an input projection (``Linear + LayerNorm``)
    when ``num_features != hidden_size * 2``, ensuring residual connections
    are always well-defined. All subsequent layers operate at
    ``hidden_size * 2``.

    Args:
        num_features (int): Input (and output) feature dimension.
        hidden_size (int): GRU hidden units per direction.
            Setting ``hidden_size * 2 == num_features`` avoids the input
            projection and is therefore recommended (the default satisfies this
            for the standard ``mlp_features=[384]`` frontend).
        num_layers (int): Number of stacked ``ResidualBiGRUBlock`` modules.
            More layers compensate for the absence of a CNN stage.
            (default: ``4``)
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

        output_size = hidden_size * 2  # per-layer bidirectional output

        # Input projection aligns num_features → output_size so that
        # every ResidualBiGRUBlock can use a direct residual connection.
        if num_features != output_size:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(num_features, output_size),
                nn.LayerNorm(output_size),
            )
            layer_input_size = output_size
        else:
            self.input_proj = nn.Identity()
            layer_input_size = num_features

        # Stack of residual BiGRU blocks.
        # All layers (except the first) receive output_size inputs.
        # Dropout is disabled on the last block.
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

        # Final projection restores the output to ``num_features`` so the
        # downstream classifier head always receives a consistent dimension.
        self.out_projection = nn.Linear(output_size, num_features)
        self.out_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, num_features)
        x = self.input_proj(inputs)
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(self.out_projection(x))  # (T, N, num_features)


class PureRNNCTCModule(pl.LightningModule):
    """A purely recurrent CTC module for sEMG-to-keystroke decoding.

    Replaces the CNN backbone (TDS convolutions) of the hybrid models with a
    ``DeepBiGRUEncoder`` — a deep stack of bidirectional GRU layers with
    per-layer residual connections, inter-layer dropout, and LayerNorm.

    The input frontend (``SpectrogramNorm`` and
    ``MultiBandRotationInvariantMLP``) and training procedure (CTC loss,
    Adam + linear-warmup cosine-annealing, CER metrics) are **identical** to
    the CNN/RNN hybrid models.

    Key architectural differences from the hybrid models:

    - **No convolutions** — the ``TDSConvEncoder`` is removed entirely.
    - **No temporal shrinkage** — T is preserved end-to-end; CTC operates
      over all ~497 frames rather than ~373.
    - **Deeper RNN** — 4 layers (vs 2 in the hybrids) to compensate for the
      absent CNN's hierarchical feature extraction.
    - **Residual BiGRU blocks** — per-layer residual shortcuts enable stable
      gradient flow through the deeper stack.

    Args:
        in_features (int): Flattened feature size per band per time step
            (``electrode_channels * freq_bins = 16 * 33 = 528``).
        mlp_features (Sequence[int]): Hidden/output sizes for each MLP layer
            in ``MultiBandRotationInvariantMLP``.
        rnn_hidden_size (int): GRU hidden units per direction.
        rnn_num_layers (int): Number of stacked ``ResidualBiGRUBlock`` modules
            in the ``DeepBiGRUEncoder``.
        rnn_dropout (float): Inter-layer dropout probability in the encoder.
        optimizer (DictConfig): Hydra config for the optimizer.
        lr_scheduler (DictConfig): Hydra config for the LR scheduler.
        decoder (DictConfig): Hydra config for the CTC decoder.
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        rnn_hidden_size: int,
        rnn_num_layers: int,
        rnn_dropout: float,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model pipeline — no CNN stage.
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
            # (T, N, num_features) — purely recurrent encoder
            DeepBiGRUEncoder(
                num_features=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout,
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

        # The pure RNN encoder does not shrink the time dimension (no unpadded
        # convolutions). T_diff is computed dynamically so the code is correct
        # even if the frontend ever introduces minor length changes.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,              # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,   # (N,)
            target_lengths=target_lengths,    # (N,)
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
