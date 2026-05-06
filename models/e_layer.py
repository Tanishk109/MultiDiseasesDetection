import torch
import torch.nn as nn
import torch.nn.functional as F


class ELayer(nn.Module):
    def __init__(self, encoder_channels=256, cam_channels=1, out_channels=256):
        super().__init__()
        in_channels = encoder_channels + cam_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, feat, cam):
        cam = F.interpolate(cam, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([feat, cam], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

