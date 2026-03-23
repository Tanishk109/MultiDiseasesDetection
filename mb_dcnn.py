"""
Full implementation of:
  "A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation
   and Classification"  —  Xie et al., IEEE TMI 2020

Architecture
────────────
  Stage 1  ─  Coarse-SN   : DeepLabV3+ (ResNet-50 backbone) for coarse masks
  Stage 2  ─  Mask-CN     : Modified Xception classifier guided by coarse masks
  Stage 3  ─  Enhanced-SN : DeepLabV3+ + E-layer fusing encoder features with
                             the CAM produced by Mask-CN

Loss
────
  Hybrid = Dice  +  λ × Rank
  The rank loss selects the K hardest foreground & background pixels per image
  and enforces a margin between their predicted scores.

Python 3.11 / PyTorch 2.x compatible.
"""

# ── std-lib ──────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
import math
import copy
import random
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # paths — edit to match your environment
    "image_dir": (
        "/root/.cache/kagglehub/datasets"
        "/tschandl/isic2018-challenge-task1-data-segmentation"
        "/versions/1/ISIC2018_Task1-2_Training_Input"
    ),
    "mask_dir": (
        "/root/.cache/kagglehub/datasets"
        "/tschandl/isic2018-challenge-task1-data-segmentation"
        "/versions/1/ISIC2018_Task1_Training_GroundTruth"
    ),
    # image size used by the paper
    "img_size": 224,
    # training
    "batch_seg": 16,
    "batch_cls": 32,
    "lr": 1e-4,
    "epochs_coarse": 50,
    "epochs_mask": 50,
    "epochs_enhanced": 50,
    # hybrid loss
    "lambda_rank": 0.05,
    "K": 30,
    "margin": 0.3,
    # classification
    "num_classes": 3,          # melanoma / nevus / seborrheic keratosis
    # device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # random seed
    "seed": 42,
}

device = torch.device(CFG["device"])


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG["seed"])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASETS
# ─────────────────────────────────────────────────────────────────────────────
class AugmentedSegDataset(Dataset):
    """
    Segmentation dataset with the online augmentation described in the paper:
      • random crop 50-100 % of centre
      • random rotation ±10°
      • horizontal / vertical flip
      • random shear / shift / zoom
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: int = 224,
        augment: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.img_size  = img_size
        self.augment   = augment
        self.images    = sorted(p.name for p in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.images)

    def _load(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        name = self.images[idx]
        img  = cv2.imread(str(self.image_dir / name))
        if img is None:
            raise FileNotFoundError(self.image_dir / name)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask_name = name.replace(".jpg", "_segmentation.png")
        msk = cv2.imread(str(self.mask_dir / mask_name), cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(self.mask_dir / mask_name)
        return img, msk

    def _augment(
        self, img: np.ndarray, msk: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]

        # random crop (50-100 % of centre)
        scale = random.uniform(0.5, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0 = (h - ch) // 2 + random.randint(-(h - ch) // 4, (h - ch) // 4)
        x0 = (w - cw) // 2 + random.randint(-(w - cw) // 4, (w - cw) // 4)
        y0 = max(0, min(y0, h - ch))
        x0 = max(0, min(x0, w - cw))
        img = img[y0 : y0 + ch, x0 : x0 + cw]
        msk = msk[y0 : y0 + ch, x0 : x0 + cw]

        # random rotation ±10°
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cw, ch), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(msk, M, (cw, ch), flags=cv2.INTER_NEAREST)

        # horizontal / vertical flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            msk = cv2.flip(msk, 1)
        if random.random() < 0.5:
            img = cv2.flip(img, 0)
            msk = cv2.flip(msk, 0)

        return img, msk

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, msk = self._load(idx)
        if self.augment:
            img, msk = self._augment(img, msk)
        img = cv2.resize(img, (self.img_size, self.img_size))
        msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        msk_t = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0)
        return img_t, msk_t


class ClassificationDataset(Dataset):
    """
    Loads (image, class_label) pairs.
    Expects a flat directory of jpg files whose names encode the class
    via a suffix, e.g.  ISIC_0000001_melanoma.jpg  →  class 0.
    Provide a label_map dict  {substring: int}  to parse filenames.
    Falls back to label=0 when no substring matches.
    """

    LABEL_MAP: dict[str, int] = {
        "melanoma": 0,
        "nevus": 1,
        "seborrheic": 2,
    }

    def __init__(
        self,
        image_dir: str,
        img_size: int = 224,
        augment: bool = True,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.img_size  = img_size
        self.augment   = augment
        self.label_map = label_map or self.LABEL_MAP
        self.images    = sorted(p.name for p in self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.images)

    def _parse_label(self, name: str) -> int:
        name_lower = name.lower()
        for key, val in self.label_map.items():
            if key in name_lower:
                return val
        return 0  # default

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        name  = self.images[idx]
        img   = cv2.imread(str(self.image_dir / name))
        if img is None:
            raise FileNotFoundError(self.image_dir / name)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self._parse_label(name)

        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D(
                (img.shape[1] / 2, img.shape[0] / 2), angle, 1.0
            )
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        img = cv2.resize(img, (self.img_size, self.img_size))
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        return img_t, label


class MaskGuidedClassificationDataset(Dataset):
    """
    Returns (image‖coarse_mask, label).
    The coarse mask is generated on-the-fly by a pre-trained Coarse-SN.
    Input to Mask-CN has 4 channels: RGB + mask.
    """

    def __init__(
        self,
        image_dir: str,
        coarse_sn: nn.Module,
        img_size: int = 224,
        augment: bool = True,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.base       = ClassificationDataset(image_dir, img_size, augment, label_map)
        self.coarse_sn  = coarse_sn.eval()
        self.img_size   = img_size

    def __len__(self) -> int:
        return len(self.base)

    @torch.no_grad()
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_t, label = self.base[idx]
        # generate coarse mask  (1, H, W)
        x    = img_t.unsqueeze(0).to(next(self.coarse_sn.parameters()).device)
        pred = torch.sigmoid(self.coarse_sn(x))          # (1, 1, H, W)
        mask = pred.squeeze(0).cpu()                      # (1, H, W)
        # concatenate along channel dim → (4, H, W)
        combined = torch.cat([img_t, mask], dim=0)
        return combined, label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """Soft Dice loss (equation 2 in the paper)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds  = torch.sigmoid(logits)
        smooth = 1e-5
        inter  = (preds * targets).sum()
        dice   = (2.0 * inter + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1.0 - dice


class RankLoss(nn.Module):
    """
    Online rank loss (equation 3 in the paper).

    Selects the K pixels with the largest cross-entropy error in each of
    foreground and background, then enforces
        predicted_score(foreground) > predicted_score(background) + margin
    """

    def __init__(self, K: int = 30, margin: float = 0.3) -> None:
        super().__init__()
        self.K      = K
        self.margin = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)          # (B, 1, H, W)
        B     = preds.size(0)
        loss  = torch.tensor(0.0, device=logits.device)

        for b in range(B):
            p = preds[b, 0].view(-1)           # (H*W,)
            t = targets[b, 0].view(-1)         # (H*W,)

            fg_mask = t > 0.5
            bg_mask = ~fg_mask

            # pixel-wise BCE error  (higher = harder)
            err = F.binary_cross_entropy(p, t, reduction="none")

            # select K hardest pixels per region
            K_fg = min(self.K, int(fg_mask.sum().item()))
            K_bg = min(self.K, int(bg_mask.sum().item()))

            if K_fg == 0 or K_bg == 0:
                continue

            hard_fg_scores = p[fg_mask][err[fg_mask].topk(K_fg).indices]   # (K_fg,)
            hard_bg_scores = p[bg_mask][err[bg_mask].topk(K_bg).indices]   # (K_bg,)

            # pairwise hinge:  fg > bg + margin
            fg = hard_fg_scores.unsqueeze(1)   # (K_fg, 1)
            bg = hard_bg_scores.unsqueeze(0)   # (1, K_bg)
            pairwise = torch.clamp(self.margin + bg - fg, min=0.0)
            loss = loss + pairwise.mean()

        return loss / B


class HybridLoss(nn.Module):
    """
    L_hybrid = L_dice + λ × L_rank   (equation 1 in the paper)
    """

    def __init__(
        self,
        lambda_rank: float = 0.05,
        K: int = 30,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.rank = RankLoss(K, margin)
        self.lam  = lambda_rank

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, targets) + self.lam * self.rank(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COARSE-SN  — DeepLabV3+ with ResNet-50 backbone
# ─────────────────────────────────────────────────────────────────────────────
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_ch: int, out_ch: int = 256) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv_r6 = nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False)
        self.conv_r12= nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False)
        self.conv_r18= nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.gap_conv= nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        branches = [
            self.relu(self.bn(self.conv1(x))),
            self.relu(self.bn(self.conv_r6(x))),
            self.relu(self.bn(self.conv_r12(x))),
            self.relu(self.bn(self.conv_r18(x))),
            F.interpolate(
                self.relu(self.bn(self.gap_conv(self.gap(x)))),
                size=(h, w), mode="bilinear", align_corners=False,
            ),
        ]
        return self.project(torch.cat(branches, dim=1))


class DeepLabV3PlusSeg(nn.Module):
    """
    DeepLabV3+ for binary segmentation, as used for both Coarse-SN and as
    the encoder/decoder backbone of Enhanced-SN.

    Paper uses Xception, but ResNet-50 with OS=16 is the standard pytorch
    substitute and is equally described in the paper (section III-A).
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        backbone = tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # encoder - use OS=16: make layer3/layer4 dilated
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool   = backbone.maxpool
        self.layer1 = backbone.layer1    # stride 4,  ch=256
        self.layer2 = backbone.layer2    # stride 8,  ch=512
        # dilate layer3 & layer4 to keep OS=16
        self._make_dilated(backbone.layer3, dilation=2)
        self._make_dilated(backbone.layer4, dilation=4)
        self.layer3 = backbone.layer3    # ch=1024
        self.layer4 = backbone.layer4    # ch=2048

        self.aspp = ASPP(2048, 256)

        # low-level feature projection (from layer1, ch=256)
        self.low_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(256, 1, 1)

    @staticmethod
    def _make_dilated(layer: nn.Module, dilation: int) -> None:
        for m in layer.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
                m.dilation  = (dilation, dilation)
                m.padding   = (dilation, dilation)
                m.stride    = (1, 1)
            elif isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (1, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (low_level_feat, aspp_feat)."""
        x  = self.layer0(x)
        x  = self.pool(x)
        l1 = self.layer1(x)    # low-level features
        x  = self.layer2(l1)
        x  = self.layer3(x)
        x  = self.layer4(x)
        return l1, self.aspp(x)

    def decode(
        self,
        low: torch.Tensor,
        aspp_feat: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        low_proj = self.low_proj(low)
        aspp_up  = F.interpolate(
            aspp_feat, size=low_proj.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        fused    = torch.cat([aspp_up, low_proj], dim=1)
        decoded  = self.decoder(fused)
        decoded  = F.interpolate(
            decoded, size=target_size, mode="bilinear", align_corners=False
        )
        return self.head(decoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        low, aspp_feat = self.encode(x)
        return self.decode(low, aspp_feat, (h, w))


# Convenience aliases
def build_coarse_sn(pretrained: bool = True) -> DeepLabV3PlusSeg:
    return DeepLabV3PlusSeg(in_channels=3, pretrained=pretrained)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MASK-CN  — modified Xception classifier (4-channel input)
# ─────────────────────────────────────────────────────────────────────────────
class SeparableConv2d(nn.Module):
    """Depth-wise separable convolution."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size,
            stride=stride, padding=padding * dilation,
            dilation=dilation, groups=in_ch, bias=bias,
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class XceptionBlock(nn.Module):
    """Single Xception middle-flow block."""

    def __init__(self, ch: int, reps: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(reps):
            layers += [
                nn.ReLU(inplace=True),
                SeparableConv2d(ch, ch),
                nn.BatchNorm2d(ch),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MaskCN(nn.Module):
    """
    Mask-guided Classification Network (Section III-B).

    Input: 4-channel tensor (RGB image concatenated with coarse lesion mask).
    Architecture: simplified Xception-like backbone with two dilated separable
    conv layers replacing the last pooling layer, followed by GAP → FC → softmax.
    The last conv layer's weights are also used to generate CAMs.
    """

    def __init__(self, num_classes: int = 3, pretrained_rgb: bool = True) -> None:
        super().__init__()
        # Entry flow
        # Adapt first conv to 4-channel input
        self.entry_conv = nn.Conv2d(4, 32, 3, stride=2, padding=1, bias=False)
        self.entry_bn   = nn.BatchNorm2d(32)

        self.entry = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            SeparableConv2d(64, 128), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            SeparableConv2d(128, 256), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),

            SeparableConv2d(256, 728), nn.BatchNorm2d(728), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        # Middle flow (8 repeated blocks)
        self.middle = nn.Sequential(*[XceptionBlock(728) for _ in range(8)])

        # Exit flow — replace last pooling with two dilated separable convs
        # as described in Section III-B
        self.exit_conv1 = nn.Sequential(
            SeparableConv2d(728, 1536, dilation=2, padding=2),
            nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
        )
        # Last conv whose weights serve as CAM weights
        self.exit_conv2 = nn.Sequential(
            SeparableConv2d(1536, 2048, dilation=2, padding=2),
            nn.BatchNorm2d(2048), nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(2048, num_classes)

        self.num_classes = num_classes

        # initialise 4th channel weights by averaging RGB weights
        if pretrained_rgb:
            self._init_4th_channel()

    def _init_4th_channel(self) -> None:
        with torch.no_grad():
            w = self.entry_conv.weight        # (32, 4, 3, 3)
            # average of first 3 channels
            avg = w[:, :3, :, :].mean(dim=1, keepdim=True)
            w[:, 3:4, :, :] = avg

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (B, num_classes)
            feat   : last conv feature map (B, 2048, H', W') used for CAM
        """
        x    = F.relu(self.entry_bn(self.entry_conv(x)), inplace=True)
        x    = self.entry(x)
        x    = self.middle(x)
        x    = self.exit_conv1(x)
        feat = self.exit_conv2(x)              # (B, 2048, H', W')
        pooled = self.gap(feat).flatten(1)     # (B, 2048)
        logits = self.fc(pooled)               # (B, C)
        return logits, feat

    @torch.no_grad()
    def get_cam(
        self, feat: torch.Tensor, class_idx: int | None = None
    ) -> torch.Tensor:
        """
        Produce class activation maps via CAM (Zhou et al. 2016).
        feat      : (B, 2048, H', W')
        class_idx : which class to highlight. None → argmax of last forward pass.
        Returns   : (B, 1, H', W') – un-normalised CAM
        """
        W = self.fc.weight          # (C, 2048)
        if class_idx is None:
            class_idx = 0
        w = W[class_idx].view(1, -1, 1, 1)     # (1, 2048, 1, 1)
        cam = (feat * w).sum(dim=1, keepdim=True)   # (B, 1, H', W')
        return cam


# ─────────────────────────────────────────────────────────────────────────────
# 5.  ENHANCED-SN  — DeepLabV3+ + E-layer
# ─────────────────────────────────────────────────────────────────────────────
class ELayer(nn.Module):
    """
    Enhanced Layer (E-layer) described in Section III-C.

    Concatenates encoder feature maps with the fine CAM from Mask-CN,
    then applies Conv-BN-ReLU to fuse them.
    """

    def __init__(self, encoder_ch: int, cam_ch: int = 1, out_ch: int = 256) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(encoder_ch + cam_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, encoder_feat: torch.Tensor, cam: torch.Tensor
    ) -> torch.Tensor:
        cam_up = F.interpolate(
            cam,
            size=encoder_feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat([encoder_feat, cam_up], dim=1))


class EnhancedSN(nn.Module):
    """
    Enhanced Segmentation Network (Section III-C).

    Uses the same DeepLabV3+ encoder/decoder backbone as Coarse-SN,
    but inserts an E-layer between encoder and decoder that fuses the
    ASPP output with the fine CAM produced by Mask-CN.
    """

    def __init__(self, coarse_sn: DeepLabV3PlusSeg) -> None:
        super().__init__()
        # share encoder/decoder weights with coarse-SN as per the paper
        self.backbone = copy.deepcopy(coarse_sn)
        # E-layer: fuse ASPP output (256-ch) with CAM (1-ch) → 256-ch
        self.e_layer  = ELayer(encoder_ch=256, cam_ch=1, out_ch=256)

    def forward(
        self, x: torch.Tensor, cam: torch.Tensor
    ) -> torch.Tensor:
        """
        x   : input image (B, 3, H, W)
        cam : class activation map from Mask-CN (B, 1, H', W')
        """
        h, w = x.shape[-2:]
        low, aspp_feat = self.backbone.encode(x)
        # fuse ASPP features with CAM via E-layer
        fused = self.e_layer(aspp_feat, cam)
        return self.backbone.decode(low, fused, (h, w))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINER  — one class per stage
# ─────────────────────────────────────────────────────────────────────────────
class Trainer:
    """Generic training loop with early stopping based on validation loss."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        patience: int = 10,
    ) -> None:
        self.model     = model
        self.opt       = optimizer
        self.criterion = criterion
        self.device    = device
        self.patience  = patience

    # ── segmentation ─────────────────────────────────────────────────────────
    def train_epoch_seg(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            self.opt.zero_grad()
            preds = self.model(imgs)
            loss  = self.criterion(preds, masks)
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def val_epoch_seg(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            preds = self.model(imgs)
            total += self.criterion(preds, masks).item()
        return total / len(loader)

    def fit_seg(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        label: str = "Seg",
    ) -> list[float]:
        history, best_val, no_improve = [], float("inf"), 0
        for ep in range(1, epochs + 1):
            tr  = self.train_epoch_seg(train_loader)
            val = self.val_epoch_seg(val_loader)
            history.append(tr)
            print(f"[{label}] Epoch {ep:>3}/{epochs}  train={tr:.4f}  val={val:.4f}")
            if val < best_val:
                best_val   = val
                no_improve = 0
                torch.save(self.model.state_dict(), f"best_{label}.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {ep}")
                    break
        self.model.load_state_dict(torch.load(f"best_{label}.pt", map_location=self.device))
        return history

    # ── classification ────────────────────────────────────────────────────────
    def train_epoch_cls(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            self.opt.zero_grad()
            logits, _ = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def val_epoch_cls(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.model(imgs)
            total += self.criterion(logits, labels).item()
        return total / len(loader)

    def fit_cls(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        label: str = "Cls",
    ) -> list[float]:
        history, best_val, no_improve = [], float("inf"), 0
        for ep in range(1, epochs + 1):
            tr  = self.train_epoch_cls(train_loader)
            val = self.val_epoch_cls(val_loader)
            history.append(tr)
            print(f"[{label}] Epoch {ep:>3}/{epochs}  train={tr:.4f}  val={val:.4f}")
            if val < best_val:
                best_val   = val
                no_improve = 0
                torch.save(self.model.state_dict(), f"best_{label}.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {ep}")
                    break
        self.model.load_state_dict(torch.load(f"best_{label}.pt", map_location=self.device))
        return history

    # ── enhanced segmentation (needs CAM) ────────────────────────────────────
    def train_epoch_enhanced(
        self,
        loader: DataLoader,
        mask_cn: MaskCN,
    ) -> float:
        self.model.train()
        mask_cn.eval()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            # get CAM from mask-CN
            with torch.no_grad():
                logits, feat = mask_cn(
                    torch.cat([imgs, torch.zeros_like(imgs[:, :1])], dim=1)
                )
                pred_class = logits.argmax(dim=1)           # (B,)
                cam = torch.stack(
                    [mask_cn.get_cam(feat[i:i+1], pred_class[i].item())
                     for i in range(imgs.size(0))],
                    dim=0,
                ).squeeze(1)                                # (B, 1, H', W')
            self.opt.zero_grad()
            preds = self.model(imgs, cam)
            loss  = self.criterion(preds, masks)
            loss.backward()
            self.opt.step()
            total += loss.item()
        return total / len(loader)

    @torch.no_grad()
    def val_epoch_enhanced(
        self,
        loader: DataLoader,
        mask_cn: MaskCN,
    ) -> float:
        self.model.eval()
        mask_cn.eval()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(self.device), masks.to(self.device)
            logits, feat = mask_cn(
                torch.cat([imgs, torch.zeros_like(imgs[:, :1])], dim=1)
            )
            pred_class = logits.argmax(dim=1)
            cam = torch.stack(
                [mask_cn.get_cam(feat[i:i+1], pred_class[i].item())
                 for i in range(imgs.size(0))],
                dim=0,
            ).squeeze(1)
            preds = self.model(imgs, cam)
            total += self.criterion(preds, masks).item()
        return total / len(loader)

    def fit_enhanced(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        mask_cn: MaskCN,
        epochs: int,
        label: str = "Enhanced",
    ) -> list[float]:
        history, best_val, no_improve = [], float("inf"), 0
        for ep in range(1, epochs + 1):
            tr  = self.train_epoch_enhanced(train_loader, mask_cn)
            val = self.val_epoch_enhanced(val_loader, mask_cn)
            history.append(tr)
            print(f"[{label}] Epoch {ep:>3}/{epochs}  train={tr:.4f}  val={val:.4f}")
            if val < best_val:
                best_val   = val
                no_improve = 0
                torch.save(self.model.state_dict(), f"best_{label}.pt")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {ep}")
                    break
        self.model.load_state_dict(
            torch.load(f"best_{label}.pt", map_location=self.device)
        )
        return history


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────
def jaccard_index(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred   = (torch.sigmoid(pred) > thresh).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return (inter / (union + 1e-5)).item()


def dice_coeff(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred  = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-5)).item()


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred = (torch.sigmoid(pred) > thresh).float()
    return (pred == target).float().mean().item()


def pixel_sensitivity(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred  = (torch.sigmoid(pred) > thresh).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp / (tp + fn + 1e-5)).item()


def pixel_specificity(pred: torch.Tensor, target: torch.Tensor, thresh: float = 0.5) -> float:
    pred  = (torch.sigmoid(pred) > thresh).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn / (tn + fp + 1e-5)).item()


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cam_fn=None,          # optional callable(imgs) → cam tensor
) -> dict[str, float]:
    model.eval()
    ja = di = ac = se = sp = 0.0
    n  = 0
    for batch in loader:
        if len(batch) == 2:
            imgs, masks = batch
        else:
            imgs, masks, *_ = batch
        imgs, masks = imgs.to(device), masks.to(device)
        if cam_fn is not None:
            cam   = cam_fn(imgs)
            preds = model(imgs, cam)
        else:
            preds = model(imgs)
        ja += jaccard_index(preds, masks)
        di += dice_coeff(preds, masks)
        ac += pixel_accuracy(preds, masks)
        se += pixel_sensitivity(preds, masks)
        sp += pixel_specificity(preds, masks)
        n  += 1
    return {
        "JA": ja / n * 100,
        "DI": di / n * 100,
        "pixel-AC": ac / n * 100,
        "pixel-SE": se / n * 100,
        "pixel-SP": sp / n * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def visualise_pipeline(
    coarse_sn: DeepLabV3PlusSeg,
    mask_cn: MaskCN,
    enhanced_sn: EnhancedSN,
    dataset: AugmentedSegDataset,
    device: torch.device,
    n_samples: int = 4,
) -> None:
    coarse_sn.eval()
    mask_cn.eval()
    enhanced_sn.eval()

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    fig, axes = plt.subplots(n_samples, 5, figsize=(18, 4 * n_samples))

    col_titles = ["Input", "Coarse Mask", "Location Map (CAM)", "Enhanced Mask", "Ground Truth"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        img_t, mask_t = dataset[idx]
        img_b = img_t.unsqueeze(0).to(device)

        # Stage 1 — coarse mask
        coarse_logit = coarse_sn(img_b)
        coarse_mask  = torch.sigmoid(coarse_logit)

        # Stage 2 — CAM from mask-CN
        mask_input = torch.cat([img_b, coarse_mask], dim=1)   # (1, 4, H, W)
        logits, feat = mask_cn(mask_input)
        pred_cls     = logits.argmax(dim=1).item()
        cam          = mask_cn.get_cam(feat, pred_cls)          # (1, 1, H', W')
        cam_up       = F.interpolate(
            cam, size=img_b.shape[-2:], mode="bilinear", align_corners=False
        )

        # Stage 3 — enhanced segmentation
        enhanced_logit = enhanced_sn(img_b, cam)
        enhanced_mask  = torch.sigmoid(enhanced_logit)

        def t2np(t: torch.Tensor) -> np.ndarray:
            return t.squeeze().cpu().numpy()

        imgs_show = [
            img_t.permute(1, 2, 0).numpy(),
            t2np(coarse_mask),
            t2np(cam_up),
            t2np(enhanced_mask),
            t2np(mask_t),
        ]
        cmaps = [None, "gray", "jet", "gray", "gray"]
        for col, (im, cmap) in enumerate(zip(imgs_show, cmaps)):
            axes[row, col].imshow(im, cmap=cmap)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("mb_dcnn_predictions.png", dpi=120)
    plt.show()
    print("Saved: mb_dcnn_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_dataloaders(
    image_dir: str,
    mask_dir: str,
    img_size: int,
    batch_seg: int,
    val_split: float = 0.15,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    full_ds   = AugmentedSegDataset(image_dir, mask_dir, img_size, augment=True)
    val_size  = int(len(full_ds) * val_split)
    train_size= len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(CFG["seed"]),
    )
    # Turn off augmentation for validation subset
    val_ds.dataset = AugmentedSegDataset(image_dir, mask_dir, img_size, augment=False)

    train_dl = DataLoader(
        train_ds, batch_size=batch_seg, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_dl   = DataLoader(
        val_ds, batch_size=batch_seg, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_dl, val_dl


def run_pipeline(cfg: dict = CFG) -> dict[str, nn.Module]:
    """
    Execute the full three-stage MB-DCNN training pipeline.
    Returns a dict with the three trained models.
    """
    device = torch.device(cfg["device"])
    hybrid = HybridLoss(cfg["lambda_rank"], cfg["K"], cfg["margin"]).to(device)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Building dataloaders …")
    train_dl, val_dl = build_dataloaders(
        cfg["image_dir"], cfg["mask_dir"],
        cfg["img_size"], cfg["batch_seg"],
    )
    print(f"  Train: {len(train_dl.dataset)}  Val: {len(val_dl.dataset)}")

    # ── STAGE 1 : Coarse-SN ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1: Training Coarse-SN …")
    coarse_sn = build_coarse_sn(pretrained=True).to(device)
    opt1      = torch.optim.Adam(coarse_sn.parameters(), lr=cfg["lr"])
    trainer1  = Trainer(coarse_sn, opt1, hybrid, device, patience=10)
    trainer1.fit_seg(train_dl, val_dl, cfg["epochs_coarse"], label="CoarseSN")
    coarse_sn.eval()

    # ── STAGE 2 : Mask-CN ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2: Training Mask-CN …")
    mask_cn = MaskCN(num_classes=cfg["num_classes"]).to(device)
    ce_loss = nn.CrossEntropyLoss()

    # Build mask-guided classification datasets
    # For ISIC-2018 Task1 the segmentation dataset doesn't have class labels,
    # so we use the image IDs and assign dummy labels (replace with real ones
    # when you have image-level annotations).
    cls_train_ds = MaskGuidedClassificationDataset(
        cfg["image_dir"], coarse_sn, cfg["img_size"], augment=True
    )
    cls_val_ds   = MaskGuidedClassificationDataset(
        cfg["image_dir"], coarse_sn, cfg["img_size"], augment=False
    )
    val_size  = int(len(cls_train_ds) * 0.15)
    train_size= len(cls_train_ds) - val_size
    cls_train_ds, cls_val_ds = torch.utils.data.random_split(
        cls_train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    cls_train_dl = DataLoader(
        cls_train_ds, batch_size=cfg["batch_cls"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    cls_val_dl   = DataLoader(
        cls_val_ds, batch_size=cfg["batch_cls"],
        shuffle=False, num_workers=2, pin_memory=True,
    )
    opt2     = torch.optim.Adam(mask_cn.parameters(), lr=cfg["lr"])
    trainer2 = Trainer(mask_cn, opt2, ce_loss, device, patience=10)
    trainer2.fit_cls(cls_train_dl, cls_val_dl, cfg["epochs_mask"], label="MaskCN")
    mask_cn.eval()

    # ── STAGE 3 : Enhanced-SN ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3: Training Enhanced-SN …")
    enhanced_sn = EnhancedSN(coarse_sn).to(device)
    opt3        = torch.optim.Adam(enhanced_sn.parameters(), lr=cfg["lr"])
    trainer3    = Trainer(enhanced_sn, opt3, hybrid, device, patience=10)
    trainer3.fit_enhanced(
        train_dl, val_dl, mask_cn,
        cfg["epochs_enhanced"], label="EnhancedSN",
    )

    # ── Final Evaluation ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final evaluation on validation set …")

    metrics_coarse = evaluate_segmentation(coarse_sn, val_dl, device)
    print("\nCoarse-SN metrics:")
    for k, v in metrics_coarse.items():
        print(f"  {k}: {v:.2f} %")

    def cam_fn_for_enhanced(imgs: torch.Tensor) -> torch.Tensor:
        """Generate CAMs for a batch using mask-CN with zero mask channel."""
        mask_input = torch.cat(
            [imgs, torch.zeros_like(imgs[:, :1])], dim=1
        )
        with torch.no_grad():
            logits, feat = mask_cn(mask_input)
        pred_cls = logits.argmax(dim=1)
        cams = torch.stack(
            [mask_cn.get_cam(feat[i:i+1], pred_cls[i].item())
             for i in range(imgs.size(0))],
            dim=0,
        ).squeeze(1)
        return cams

    metrics_enhanced = evaluate_segmentation(
        enhanced_sn, val_dl, device, cam_fn=cam_fn_for_enhanced
    )
    print("\nEnhanced-SN metrics:")
    for k, v in metrics_enhanced.items():
        print(f"  {k}: {v:.2f} %")

    # ── Visualise ─────────────────────────────────────────────────────────────
    full_ds = AugmentedSegDataset(
        cfg["image_dir"], cfg["mask_dir"], cfg["img_size"], augment=False
    )
    visualise_pipeline(coarse_sn, mask_cn, enhanced_sn, full_ds, device)

    return {
        "coarse_sn":   coarse_sn,
        "mask_cn":     mask_cn,
        "enhanced_sn": enhanced_sn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_single(
    image_path: str,
    coarse_sn: DeepLabV3PlusSeg,
    mask_cn: MaskCN,
    enhanced_sn: EnhancedSN,
    device: torch.device,
    img_size: int = 224,
) -> dict[str, np.ndarray]:
    """
    Run the full three-stage pipeline on a single image.

    Returns a dict with:
        'image'        — original RGB image (H, W, 3) float in [0,1]
        'coarse_mask'  — coarse binary mask (H, W) float
        'cam'          — class activation map (H, W) float
        'enhanced_mask'— final binary mask (H, W) float
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    coarse_sn.eval(); mask_cn.eval(); enhanced_sn.eval()

    coarse_logit = coarse_sn(img_t)
    coarse_mask  = torch.sigmoid(coarse_logit)              # (1,1,H,W)

    mask_input   = torch.cat([img_t, coarse_mask], dim=1)  # (1,4,H,W)
    logits, feat = mask_cn(mask_input)
    pred_cls     = logits.argmax(dim=1).item()
    cam          = mask_cn.get_cam(feat, pred_cls)
    cam_up       = F.interpolate(cam, size=img_t.shape[-2:],
                                 mode="bilinear", align_corners=False)

    enhanced_logit = enhanced_sn(img_t, cam)
    enhanced_mask  = torch.sigmoid(enhanced_logit)

    def squeeze(t: torch.Tensor) -> np.ndarray:
        return t.squeeze().cpu().numpy()

    return {
        "image":         img / 255.0,
        "coarse_mask":   squeeze(coarse_mask),
        "cam":           squeeze(cam_up),
        "enhanced_mask": squeeze(enhanced_mask),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Verify dataset paths exist before starting
    for key in ("image_dir", "mask_dir"):
        p = Path(CFG[key])
        if not p.exists():
            print(f"[WARNING] {key} not found: {p}")
            print("  Update CFG paths before running the full pipeline.")

    # Run the three-stage pipeline
    models = run_pipeline(CFG)
    print("\nDone. Trained models available in `models` dict.")
