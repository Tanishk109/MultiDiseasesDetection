import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import SkinClsDataset
from models.mask_cn import MaskCN
from models.coarse_sn import CoarseSN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 1e-4
MAX_EPOCHS = 50
VAL_RATIO = 0.2
SEED = 42

CLS_ROOT = "data/classification"
SAVE_DIR = "weights"
SAVE_PATH = os.path.join(SAVE_DIR, "maskcn_best.pth")
COARSE_CKPT = os.path.join(SAVE_DIR, "coarse_best.pth")

NUM_CLASSES = 7


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def build_dataloaders():
    full_aug_ds = SkinClsDataset(CLS_ROOT, split="train", augment_data=True)
    full_eval_ds = SkinClsDataset(CLS_ROOT, split="train", augment_data=False)

    n = len(full_aug_ds)
    if n < 2:
        raise ValueError("Need at least 2 classification samples to create a train/val split.")

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

    print(f"Total classification samples: {n}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples:   {len(val_idx)}")

    return train_loader, val_loader


def build_frozen_coarse():
    if not os.path.isfile(COARSE_CKPT):
        raise FileNotFoundError(f"Coarse-SN checkpoint not found: {COARSE_CKPT}")

    coarse = CoarseSN().to(DEVICE)
    ckpt = torch.load(COARSE_CKPT, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model" in ckpt:
        coarse.load_state_dict(ckpt["model"])
    else:
        coarse.load_state_dict(ckpt)

    coarse.eval()
    for p in coarse.parameters():
        p.requires_grad = False

    return coarse


def run_epoch(model, coarse, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for img, label in loader:
        img = img.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            coarse_logits = coarse(img)
            coarse_mask = torch.sigmoid(coarse_logits)

        x4 = torch.cat([img, coarse_mask], dim=1)

        if training:
            optimizer.zero_grad()

        logits, _ = model(x4)
        loss = loss_fn(logits, label)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits.detach(), label)
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "acc": total_acc / num_batches,
    }


def main():
    if not os.path.isdir(CLS_ROOT):
        raise FileNotFoundError(f"Classification root not found: {CLS_ROOT}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader, val_loader = build_dataloaders()
    coarse = build_frozen_coarse()

    model = MaskCN(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = -1.0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, coarse, train_loader, loss_fn, optimizer=optimizer)

        with torch.no_grad():
            val_metrics = run_epoch(model, coarse, val_loader, loss_fn, optimizer=None)

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                },
                SAVE_PATH,
            )
            print(f"  Saved best model to {SAVE_PATH} (val_acc={best_val_acc:.4f})")

    print("Mask-CN training complete.")


if __name__ == "__main__":
    main()

