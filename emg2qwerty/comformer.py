import torch
import torch.nn as nn
from vis_transformer import EMGFeatureExtractor, PositionalEncoding
import torch.nn.functional as F



class FeedForwardModule(nn.Module):
    """
    Conformer Feed-Forward Module:
      LayerNorm -> Linear(d_model, 4*d_model) -> Swish -> Dropout -> Linear(4*d_model, d_model) -> Dropout
    """
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.norm =nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return x + 0.5 * self.ff(self.norm(x))

class ConvolutionModule(nn.Module):
    """
    Conformer Convolution Module:
      LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BN -> Swish -> Pointwise Conv -> Dropout
    
    Input/Output: (T, N, d_model)
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)
        # Double the data for GLU
        self.pointwise_expand = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim = 1) # it halves the dimension of the data
        self.depthwise = nn.Conv1d(d_model, d_model, padding = padding, kernel_size=kernel_size, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pointwise_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (T, N, d_model)
        residual = x
        x = self.norm(x)
        # Conv1d wants (N, C, T)
        x = x.permute(1, 2, 0) # (N, d_model, T)
        x = self.pointwise_expand(x) #(N, 2*d_model, T)
        x = self.glu(x) # (N, d_model, T)
        x = self.depthwise(x) #(N, d_model, T)
        x = self.bn(x) # (N, d_model, T)
        x = self.swish(x) #(N, d_model, T)
        x = self.pointwise_proj(x) #(N, d_model, T)
        x = self.dropout(x) #(N, d_model, T)
        x = x.permute(2, 0, 1)
        return x + residual


class ConformerBlock(nn.Module):
    """
    Single Conformer block.
    Input/Output shape: (T, N, d_model)
    1) Feedforward
    2) Layernorm
    3) Multiheaded attention
    4) Dropout
    5) convolution
    6) Feedforward
    7) layernorm
    """
    def __init__(self, d_model:int,  nhead: int, dropout: float, kernel_size: int = 31):
        super().__init__()

        self.ff1 = FeedForwardModule(d_model=d_model, dropout=dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=False)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model=d_model, kernel_size= kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model=d_model, dropout=dropout)
        self.normout = nn.LayerNorm(d_model)
    
    def forward(self, x, src_key_padding_mask: torch.Tensor = None):
        x = self.ff1(x)
        residual = x

        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        x = residual + self.attn_drop(attn_out)
        x = self.conv(x)
        x = self.ff2(x)
        return self.normout(x)
    


class ConformerEncoder(nn.Module):
    """
    Stack of ConformerBlocks.
    Input/Output: (T, N, d_model)
    """

    def __init__(self, d_model: int, nhead: int, num_layers:int = 4, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([ConformerBlock(d_model=d_model, nhead=nhead, kernel_size=kernel_size, dropout=dropout)
                                     for _ in range(num_layers)])
    
    def forward(self, x, src_key_padding_mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x



class EMGConformerCTC(nn.Module):
    """
    Full pipeline:
      EMGFeatureExtractor -> ConformerEncoder -> Linear -> CTC loss
    
    Args:
        in_features:  frequency bins * 1  (33 for data → 528 = 2*16*33? confirm)
        mlp_features: e.g. [384]          → num_features = 2*384 = 768
        d_model:      conformer d_model, should equal num_features or projected
        nhead:        attention heads (d_model must be divisible)
        num_layers:   number of ConformerBlocks
        num_classes:  99 ( charset + blank)
        dropout:      dropout rate
    """

    def __init__(
            self,
            in_features: int = 528,
            mlp_features: list = [384],
            d_model: int = 768,
            nhead: int =  8,
            num_layers: int = 4,
            num_classes: int = 99,
            kernel_size: int = 31,
            dropout: float = 0.1):
        super().__init__()

        self.features_extractor = EMGFeatureExtractor(in_features=in_features, mlp_features=mlp_features)
        feat_dim = self.features_extractor.num_features # 2 * mlp_features[-1]

        self.input_proj = (nn.Linear(feat_dim, d_model) if feat_dim != d_model else nn.Identity())
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len=2000)
        self.encoder = ConformerEncoder(d_model, nhead, num_layers, kernel_size, dropout,)
        self.ctc_head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x = (T, N, 2, 16, freq)
        x = self.features_extractor(x) # (T, N, feat_dim)
        x = self.input_proj(x) # (T, N, d_model)
        x = self.pos_enc(x)# T, N, d_model
        x = self.encoder(x)# T, N, d_model
        x = self.ctc_head(x) ## T, N, num_classes
        return F.log_softmax(x, dim=-1)