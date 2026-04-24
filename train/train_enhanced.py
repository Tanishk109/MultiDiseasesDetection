import os
import torch
from torch.utils.data import DataLoader
from dataset import SkinDataset
from models.enhanced_sn import EnhancedSN
from models.mask_cn import MaskCN
from models.coarse_sn import CoarseSN
from utils.cam_utils import get_cam
from losses import CombinedLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 🔥 METRICS
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
# 🔥 DATA
# =========================
ds = SkinDataset("data/images", "data/masks")
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)


# =========================
# 🔥 LOAD MODELS
# =========================
coarse = CoarseSN().to(device)
coarse.load_state_dict(torch.load("weights/coarse.pth"))
coarse.eval()

maskcn = MaskCN().to(device)
maskcn.load_state_dict(torch.load("weights/maskcn.pth"))
maskcn.eval()

model = EnhancedSN().to(device)

loss_fn = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# =========================
# 🔥 TRAINING LOOP
# =========================
for epoch in range(20):

    total_loss = 0
    total_dice = 0
    total_iou = 0

    for img, mask in loader:

        img = img.to(device)
        mask = mask.to(device)

        # =========================
        # 🔥 GENERATE CAM
        # =========================
        with torch.no_grad():
            coarse_pred = torch.sigmoid(coarse(img))
            x_mask = torch.cat([img, coarse_pred], dim=1)

            cam = get_cam(maskcn, x_mask)

            # match image size
            cam = torch.nn.functional.interpolate(
                cam,
                size=img.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        # =========================
        # 🔥 FORWARD
        # =========================
        pred = model(img, cam)

        # match GT size
        pred = torch.nn.functional.interpolate(
            pred,
            size=mask.shape[-2:]
        )

        # =========================
        # 🔥 LOSS
        # =========================
        loss = loss_fn(pred, mask)

        # =========================
        # 🔥 METRICS
        # =========================
        dice = dice_score(pred, mask)
        iou = iou_score(pred, mask)

        # =========================
        # 🔥 BACKPROP
        # =========================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

    # =========================
    # 🔥 LOGGING
    # =========================
    print(
        f"Epoch {epoch} | "
        f"Loss: {total_loss/len(loader):.4f} | "
        f"Dice: {total_dice/len(loader):.4f} | "
        f"IoU: {total_iou/len(loader):.4f}"
    )


# =========================
# 🔥 SAVE MODEL
# =========================
os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/enhanced.pth")

print("✅ Enhanced model training complete and saved.")
model.eval()
os.makedirs("outputs", exist_ok=True)

with torch.no_grad():
    for i, (img, mask) in enumerate(loader):
        img = img.to(device)

        # generate CAM again
        coarse_pred = torch.sigmoid(coarse(img))
        x_mask = torch.cat([img, coarse_pred], dim=1)
        cam = get_cam(maskcn, x_mask)

        cam = torch.nn.functional.interpolate(
            cam,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        pred = model(img, cam)
        pred = torch.nn.functional.interpolate(pred, size=img.shape[-2:])

        pred_mask = torch.sigmoid(pred[0]).cpu().numpy()[0]
        pred_mask = (pred_mask > 0.5).astype("uint8") * 255

        import cv2
        cv2.imwrite(f"outputs/pred_{i}.png", pred_mask)

        if i == 5: break  # save few samples only
