import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
from models.e_layer import ELayer

class EnhancedSN(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = smp.encoders.get_encoder("resnet50", weights="imagenet")

        

# 🔥 Modify encoder channels
        encoder_channels = list(self.encoder.out_channels)
        encoder_channels[-1] = 2304   # 2048 + 256 (fusion)

# 🔥 Use updated channels in decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256,128,64,32,16),
            n_blocks=5
        )

        self.e_layer = ELayer()
        self.head = nn.Conv2d(16,1,1)

    def forward(self, x, cam):
        feats = list(self.encoder(x))

        fused = self.e_layer(feats[-1], cam)

        feats[-1] = torch.cat([feats[-1], fused], dim=1)  # 🔥 FIX

        dec = self.decoder(feats)
        return self.head(dec)
