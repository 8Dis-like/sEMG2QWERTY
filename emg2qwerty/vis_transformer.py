import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence


class ConvVit(nn.Module):
    """
    Per-frame feature extractor. Processes each EMG spectrogram frame
    independently via a Conv stem + ViT, outputting a feature vector
    per frame. Temporal modeling is handled externally by TDSConvEncoder
    in the Lightning module, mirroring the original TDSConvCTCModule design.
    """

    def __init__(
        self,
        in_channels=2,
        n_filters1=32,
        n_filters2=128,
        kernel_size=3,
        n_head=8,
        n_layers=2,
    ):
        super().__init__()

        self.n_filters2 = n_filters2

        # -------------------------
        # Convolutional stem
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_filters1, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(n_filters1),
            nn.GELU(),
            nn.Conv2d(n_filters1, n_filters2, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(n_filters2),
            nn.GELU(),
        )

        # pos_embed is created lazily on the first forward pass once we know
        # the actual H*W coming out of the stem.
        self.num_tokens = None
        self.pos_embed = None

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
            num_layers=n_layers
        )

        self.norm = nn.LayerNorm(n_filters2)
        self.dropout = nn.Dropout(0.1)

        # No classifier here — classification happens after TDSConvEncoder
        # in the full pipeline (ConvVitCTCModule)

        self._init_weights()

    # ------------------------------------------------

    def _init_weights(self):

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

        # Scale down transformer weights to prevent early softmax saturation
        for name, p in self.transformer.named_parameters():
            if 'weight' in name and p.dim() > 1:
                nn.init.xavier_normal_(p, gain=0.1)

    # ------------------------------------------------

    def _init_pos_embed(self, H, W):
        self.num_tokens = H * W
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, self.n_filters2,
                        device=next(self.parameters()).device) * 0.02
        )

    # ------------------------------------------------

    def forward(self, x):
        # x: [B, C, H, W]  — one frame per sample

        x = self.stem(x)

        B, C, H, W = x.shape

        if self.pos_embed is None:
            self._init_pos_embed(H, W)

        if H * W != self.num_tokens:
            raise RuntimeError(
                f"Spatial size changed: got {H}x{W} ({H*W} tokens), "
                f"but pos_embed was built for {self.num_tokens} tokens."
            )

        # flatten spatial → tokens: [B, HW, C]
        tokens = x.flatten(2).transpose(1, 2)

        # token dropout (out-of-place to protect autograd graph)
        if self.training:
            mask = (
                torch.rand(B, tokens.size(1), 1, device=x.device) > 0.05
            ).float()
            tokens = tokens * mask

        tokens = tokens + self.pos_embed

        vit_out = self.transformer(tokens)
        vit_out = self.norm(vit_out)
        vit_out = self.dropout(vit_out)

        # Pool spatial tokens → single feature vector per frame: [B, n_filters2]
        return vit_out.mean(dim=1)