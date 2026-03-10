import torch
import torch.nn as nn
from emg2qwerty.modules import SpectrogramNorm, MultiBandRotationInvariantMLP
import math


# SpectrogramNorm → MultiBandRotationInvariantMLP → Flatten


class EMGFeatureExtractor(nn.Module):
    """
    Replicates the pre-temporal-encoder pipeline from TDSConvCTCModule.
    Input:  (T, N, 2, 16, freq)
    Output: (T, N, num_features)  where num_features = 2 * mlp_features[-1]
    """
    NUM_BANDS: int = 2
    ELECTRODE_CHANNELS: int = 16

    def __init__(self, in_features: int  = 528, mlp_features: list = [384]):
        super().__init__()

        # BatchNorm2d on the input (T, N, B, E, F) -> signals are scaled between 0 and 1
        self.norm = SpectrogramNorm(channels= self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        # apply some shifts to the electrodes, run through an MLP and average results. 
        # Done for robustness
        self.mlp = MultiBandRotationInvariantMLP(
            in_features= in_features,
            mlp_features= mlp_features,
            num_bands= self.NUM_BANDS
        )
        self.num_features = self.NUM_BANDS * mlp_features[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, N, 2, 16, freq)
        x = self.norm(x)   # (T, N, 2, 16, freq)
        x = self.mlp(x)    # (T, N, 2, mlp_features[-1])
        x = x.flatten(start_dim=2)  # (T, N, num_features)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 200000):
        super().__init__()
        self.dropout = nn.Dropout(0.0)
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[ : x.size(0)]
        return self.dropout(x)



class TemporalTransformerEncoder(nn.Module):
    """
    Temporal transformer operating across T frames.
    Input:  (T, N, d_model)
    Output: (T, N, d_model)
    """
    def __init__(self, d_model: int = 768, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 2048, dropout: int = 0.1, max_len: int = 200000):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model= d_model,
            nhead= nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
            norm_first=False
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )


    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        x = self.pos_encoding(x)
        T = x.size(0)
        # Local attention window of 32 frames (like TDS kernel_width=32)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        # Zero out attention beyond 32 frames back
        for i in range(T):
            causal_mask[i, :max(0, i-32)] = float('-inf')
        out = self.transformer(x, mask=causal_mask, is_causal=False)
        return out
