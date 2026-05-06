import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import SkinSegDataset
from models.coarse_sn import CoarseSN
from losses import HybridLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LR = 1e-4
MAX_EPOCHS = 50
VAL_RATIO = 0.2
SEED = 42

LAMBDA = 0.05
K = 30
MARGIN = 0.3

IMG_DIR = "data/images"
MASK_DIR = "data/mask"

SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "coarse_best.pth")


def dice_score(prob, target, eps=1e-6):
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean().item()


def iou_score(prob, target, eps=1e-6):
    pred = (prob > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()


def build_dataloaders():
    full_aug_ds = SkinSegDataset(IMG_DIR, MASK_DIR, augment_data=True)
    full_eval_ds = SkinSegDataset(IMG_DIR, MASK_DIR, augment_data=False)

    n = len(full_aug_ds)
    if n < 2:
        raise ValueError("Need at least 2 samples to create a train/val split.")

    indices = list(range(n))
    random.Random(SEED).shuffle(indices)

    val_size = max(1, int(n * VAL_RATIO))
    train_size = n - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_ds = Subset(full_aug_ds, train_idx)
    val_ds = Subset(full_eval_ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Total samples: {n}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")

    return train_loader, val_loader


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for img, mask in loader:
        img = img.to(DEVICE, non_blocking=True)
        mask = mask.to(DEVICE, non_blocking=True).float()

        if training:
            optimizer.zero_grad()

        logits = model(img)

        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss_out = loss_fn(logits, mask)
        loss = loss_out[0] if isinstance(loss_out, (tuple, list)) else loss_out

        if training:
            loss.backward()
            optimizer.step()

        prob = torch.sigmoid(logits)

        total_loss += loss.item()
        total_dice += dice_score(prob.detach(), mask)
        total_iou += iou_score(prob.detach(), mask)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / num_batches,
        "iou": total_iou / num_batches,
    }


def main():
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Image folder not found: {IMG_DIR}")
    if not os.path.isdir(MASK_DIR):
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader, val_loader = build_dataloaders()

    model = CoarseSN().to(DEVICE)
    loss_fn = HybridLoss(lam=LAMBDA, K=K, margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_iou = -1.0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer=optimizer)

        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, loss_fn, optimizer=None)

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_iou": best_val_iou,
                },
                SAVE_PATH,
            )
            print(f"  Saved best model to {SAVE_PATH} (val_iou={best_val_iou:.4f})")

    print("Coarse-SN training complete.")


if __name__ == "__main__":
    main()

