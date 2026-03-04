import torch.nn as nn
import torch


# start with a few conv layers before the visual transformer
# shape of data: [Time, Bands, Electrodes, Freq]
# out: # 26 letters + space + blank (do we have numbers too?)

class ConvVit(nn.Module):
    def __init__(self, in_channels = 2, n_filters1 = 32, n_filters2 = 128, kernel_size = 3, n_head = 8, n_layers = 2, n_classes = 28):
        super().__init__()
        # stem convolutional layers
        # note that around half of the activations of the conv layers are 0, and the reason is that batchnorm centers
        # around 0 and relu zeros them out. So use Leaky ReLU instead
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, n_filters1, kernel_size, stride = 2, padding=1),
            nn.BatchNorm2d(n_filters1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(n_filters1, n_filters2, kernel_size, stride = 2, padding=1),
            nn.BatchNorm2d(n_filters2),
            nn.LeakyReLU(0.1)
        )
        
        # visual transformer
        self.num_tokens = 4 * 9
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, n_filters2))
        encoder_layer = nn.TransformerEncoderLayer(d_model = n_filters2, nhead = n_head, dim_feedforward = 512, batch_first = True, activation = 'gelu', norm_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(n_filters2, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_in', nonlinearity='relu') # fan in preserves mag(variace) in forward pass
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width] = [B, 2, 16, 33]
        x = self.stem(x) #[B, 128, 4, 9]
        tokens = x.flatten(2).transpose(1, 2) # [B, 128, 4*9] -> [B,  4*9, 128]
        tokens = tokens + self.pos_embed
        vit_out = self.transformer(tokens) # [B,  4*9, 128]
        
        logits = self.classifier(vit_out)
        return self.log_softmax(logits)