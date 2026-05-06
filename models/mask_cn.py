import torch
import torch.nn as nn
import torch.nn.functional as F

from models.coarse_sn import AlignedXception


class SeparableDilatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=2):
        super().__init__()
        padding = dilation if kernel_size == 3 else 0

        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MaskCN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = AlignedXception(output_stride=16, in_channels=4)
        self.dilated_conv1 = SeparableDilatedConv2d(2048, 2048, kernel_size=3, dilation=2)
        self.dilated_conv2 = SeparableDilatedConv2d(2048, 2048, kernel_size=3, dilation=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        _, feat = self.encoder(x)
        feat = self.dilated_conv1(feat)
        feat = self.dilated_conv2(feat)
        pooled = self.gap(feat).flatten(1)
        logits = self.fc(pooled)
        return logits, feat

    def get_cam(self, feat, class_idx):
        weights = self.fc.weight

        if isinstance(class_idx, int):
            w = weights[class_idx].unsqueeze(0).expand(feat.size(0), -1)
        else:
            w = weights[class_idx]

        cam = torch.einsum("bc,bchw->bhw", w, feat).unsqueeze(1)
        cam = F.relu(cam)

        b = cam.size(0)
        cam_flat = cam.view(b, -1)
        cam_min = cam_flat.min(dim=1)[0].view(b, 1, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(b, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

