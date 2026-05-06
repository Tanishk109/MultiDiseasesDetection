import torch
import torch.nn as nn

from models.coarse_sn import AlignedXception, ASPP, DeepLabDecoder
from models.e_layer import ELayer


class EnhancedSN(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        self.encoder = AlignedXception(output_stride=output_stride, in_channels=3)
        self.aspp = ASPP(in_ch=2048, out_ch=256, output_stride=output_stride)
        self.e_layer = ELayer(encoder_channels=256, cam_channels=1, out_channels=256)
        self.decoder = DeepLabDecoder(low_level_ch=128, aspp_ch=256, num_classes=1)

    def forward(self, x, cam):
        input_size = x.shape[2:]
        low_level_feat, high_level_feat = self.encoder(x)
        aspp_feat = self.aspp(high_level_feat)
        fused_feat = self.e_layer(aspp_feat, cam)
        logits = self.decoder(fused_feat, low_level_feat, input_size)
        return logits

    def load_from_coarse_sn(self, coarse_state_dict, strict=False):
        own_state = self.state_dict()
        filtered = {
            k: v for k, v in coarse_state_dict.items()
            if k in own_state and not k.startswith("e_layer.")
        }
        self.load_state_dict(filtered, strict=strict)

