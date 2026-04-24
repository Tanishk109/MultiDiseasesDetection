import torch
import torch.nn as nn
import torch.nn.functional as F

class ELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2049, 256, 1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        
    def forward(self, feat, cam):
        cam = F.interpolate(cam, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([feat, cam], dim=1)
        return self.relu(self.bn(self.conv(x)))
