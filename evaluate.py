import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SkinDataset
from models.coarse_sn import CoarseSN
from models.mask_cn import MaskCN
from models.enhanced_sn import EnhancedSN
from utils.cam_utils import get_cam

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# METRICS
# =========================
def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter) / (pred.sum() + target.sum() + 1e-5)


def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return inter / (union + 1e-5)


# =========================
# DATA
# =========================
ds = SkinDataset("data/images", "data/masks")
loader = DataLoader(ds, batch_size=4, shuffle=False)


# =========================
# LOAD MODELS
# =========================
coarse = CoarseSN().to(device)
coarse.load_state_dict(torch.load("weights/coarse.pth"))
coarse.eval()

maskcn = MaskCN().to(device)
maskcn.load_state_dict(torch.load("weights/maskcn.pth"))
maskcn.eval()

enhanced = EnhancedSN().to(device)
enhanced.load_state_dict(torch.load("weights/enhanced.pth"))
enhanced.eval()


# =========================
# EVALUATION
# =========================
coarse_dice = 0
maskcn_dice = 0
enhanced_dice = 0

coarse_iou = 0
maskcn_iou = 0
enhanced_iou = 0


with torch.no_grad():
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)

        # -------------------------
        # COARSE
        # -------------------------
        coarse_pred = coarse(img)
        coarse_pred = F.interpolate(coarse_pred, size=mask.shape[-2:])

        coarse_dice += dice_score(coarse_pred, mask).item()
        coarse_iou += iou_score(coarse_pred, mask).item()


        # -------------------------
        # MASKCN (FINAL FIXED)
        # -------------------------
        x_mask = torch.cat([img, torch.sigmoid(coarse_pred)], dim=1)
        cam = get_cam(maskcn, x_mask)

        # Resize CAM to GT size
        cam = F.interpolate(cam, size=mask.shape[-2:])

        maskcn_dice += dice_score(cam, mask).item()
        maskcn_iou += iou_score(cam, mask).item()


        # -------------------------
        # ENHANCED
        # -------------------------
        cam_for_enhanced = F.interpolate(cam, size=img.shape[-2:])

        enhanced_pred = enhanced(img, cam_for_enhanced)
        enhanced_pred = F.interpolate(enhanced_pred, size=mask.shape[-2:])

        enhanced_dice += dice_score(enhanced_pred, mask).item()
        enhanced_iou += iou_score(enhanced_pred, mask).item()


# =========================
# FINAL RESULTS
# =========================
n = len(loader)

results = (
    "\n========== FINAL RESULTS ==========\n"
    f"Coarse   -> Dice: {coarse_dice/n:.4f} | IoU: {coarse_iou/n:.4f}\n"
    f"MaskCN   -> Dice: {maskcn_dice/n:.4f} | IoU: {maskcn_iou/n:.4f}\n"
    f"Enhanced -> Dice: {enhanced_dice/n:.4f} | IoU: {enhanced_iou/n:.4f}\n"
)

print(results)

# 🔥 Save to file
import os
os.makedirs("results", exist_ok=True)

with open("results/results.txt", "w") as f:
    f.write(results)
