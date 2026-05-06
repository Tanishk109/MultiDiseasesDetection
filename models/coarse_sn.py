import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()

        if out_ch != in_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = None

        layers = []
        ch = in_ch

        if grow_first:
            if start_with_relu:
                layers.append(nn.ReLU(inplace=False))
            layers.extend([
                SeparableConv2d(ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
            ])
            ch = out_ch

        for _ in range(reps - 1):
            layers.extend([
                nn.ReLU(inplace=False),
                SeparableConv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
            ])

        if not grow_first:
            layers.extend([
                nn.ReLU(inplace=False),
                SeparableConv2d(ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
            ])

        if stride != 1:
            layers.append(nn.MaxPool2d(kernel_size=stride, stride=stride))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        return self.rep(x) + identity


class AlignedXception(nn.Module):
    def __init__(self, output_stride=16, in_channels=3):
        super().__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_dilation = 1
            exit_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_dilation = 2
            exit_dilations = (2, 4)
        else:
            raise ValueError("output_stride must be 8 or 16")

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XceptionBlock(64, 128, reps=2, stride=2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=entry_block3_stride)

        self.middle_blocks = nn.Sequential(
            *[self._make_middle_block(728, middle_dilation) for _ in range(16)]
        )

        d1, d2 = exit_dilations
        self.block20 = XceptionBlock(728, 1024, reps=2, stride=1)
        self.conv3 = SeparableConv2d(1024, 1536, 3, padding=d1, dilation=d1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, padding=d1, dilation=d1)
        self.bn4 = nn.BatchNorm2d(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, padding=d2, dilation=d2)
        self.bn5 = nn.BatchNorm2d(2048)

    @staticmethod
    def _make_middle_block(ch, dilation):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
            SeparableConv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
            SeparableConv2d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_blocks(x)
        x = self.block20(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        high_level_feat = self.relu(self.bn5(self.conv5(x)))
        return low_level_feat, high_level_feat


class ASPPModule(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        if dilation == 1:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.conv = nn.Conv2d(
                in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation, bias=False
            )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256, output_stride=16):
        super().__init__()
        dilations = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]

        self.aspp1 = ASPPModule(in_ch, out_ch, dilations[0])
        self.aspp2 = ASPPModule(in_ch, out_ch, dilations[1])
        self.aspp3 = ASPPModule(in_ch, out_ch, dilations[2])
        self.aspp4 = ASPPModule(in_ch, out_ch, dilations[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(
            self.global_avg_pool(x),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)


class DeepLabDecoder(nn.Module):
    def __init__(self, low_level_ch=128, aspp_ch=256, num_classes=1):
        super().__init__()

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_ch, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
        )

        self.decode_conv = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, bridge_feat, low_level_feat, input_size):
        low = self.low_level_proj(low_level_feat)
        bridge_up = F.interpolate(
            bridge_feat, size=low.shape[2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([bridge_up, low], dim=1)
        x = self.decode_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


class CoarseSN(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        self.encoder = AlignedXception(output_stride=output_stride, in_channels=3)
        self.aspp = ASPP(in_ch=2048, out_ch=256, output_stride=output_stride)
        self.decoder = DeepLabDecoder(low_level_ch=128, aspp_ch=256, num_classes=1)

    def forward(self, x):
        input_size = x.shape[2:]
        low_level_feat, high_level_feat = self.encoder(x)
        aspp_feat = self.aspp(high_level_feat)
        logits = self.decoder(aspp_feat, low_level_feat, input_size)
        return logits

