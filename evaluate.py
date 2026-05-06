import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

from dataset import SkinSegDataset, SkinClsDataset
from models.coarse_sn import CoarseSN
from models.mask_cn import MaskCN
from models.enhanced_sn import EnhancedSN
from utils.cam_utils import get_cam

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.5
NUM_CLASSES = 7

SEG_IMG_DIR = "data/images"
SEG_MASK_DIR = "data/mask"
CLS_ROOT = "data/classification"

COARSE_CKPT = "weights/coarse_best.pth"
MASKCN_CKPT = "weights/maskcn_best.pth"
ENHANCED_CKPT = "weights/enhanced_best.pth"

RESULTS_DIR = "results"
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.txt")

CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def load_checkpoint_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return model


def segmentation_metrics(pred_prob, target, eps=1e-6):
    pred = (pred_prob > THRESH).float()
    target = target.float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1.0 - target)).sum(dim=(1, 2, 3))
    tn = ((1.0 - pred) * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred) * target).sum(dim=(1, 2, 3))

    ja = (tp + eps) / (tp + fp + fn + eps)
    di = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    pixel_ac = (tp + tn + eps) / (tp + fp + tn + fn + eps)
    pixel_se = (tp + eps) / (tp + fn + eps)
    pixel_sp = (tn + eps) / (tn + fp + eps)

    return {
        "JA": ja.mean().item(),
        "DI": di.mean().item(),
        "pixel_AC": pixel_ac.mean().item(),
        "pixel_SE": pixel_se.mean().item(),
        "pixel_SP": pixel_sp.mean().item(),
    }


def accumulate_metric_sums(accum, batch_metrics, batch_size):
    for k, v in batch_metrics.items():
        accum[k] = accum.get(k, 0.0) + v * batch_size


def finalize_metric_means(accum, n_samples):
    return {k: v / n_samples for k, v in accum.items()}


def classification_metrics(probs, labels):
    preds = probs.argmax(axis=1)
    overall_acc = float((preds == labels).mean())

    results = {
        "overall_AC": overall_acc,
        "per_class": {},
    }

    aucs = []
    for idx, name in enumerate(CLASS_NAMES):
        binary_gt = (labels == idx).astype(np.int64)
        binary_pred = (preds == idx).astype(np.int64)

        tp = np.sum((binary_pred == 1) & (binary_gt == 1))
        fp = np.sum((binary_pred == 1) & (binary_gt == 0))
        tn = np.sum((binary_pred == 0) & (binary_gt == 0))
        fn = np.sum((binary_pred == 0) & (binary_gt == 1))

        ac = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        se = tp / (tp + fn + 1e-8)
        sp = tn / (tn + fp + 1e-8)

        auc = float("nan")
        if binary_gt.sum() > 0 and (1 - binary_gt).sum() > 0:
            auc = roc_auc_score(binary_gt, probs[:, idx])
            aucs.append(auc)

        results["per_class"][name] = {
            "AUC": auc,
            "AC": ac,
            "SE": se,
            "SP": sp,
        }

    results["average_AUC"] = float(np.nanmean(aucs)) if len(aucs) > 0 else float("nan")
    return results


def fmt_seg(name, d):
    return (
        f"\n{'=' * 60}\n"
        f"{name}\n"
        f"{'=' * 60}\n"
        f"JA        : {d['JA']:.4f}\n"
        f"DI        : {d['DI']:.4f}\n"
        f"pixel-AC  : {d['pixel_AC']:.4f}\n"
        f"pixel-SE  : {d['pixel_SE']:.4f}\n"
        f"pixel-SP  : {d['pixel_SP']:.4f}\n"
    )


def fmt_cls(d, split_name):
    lines = [
        f"\n{'=' * 60}",
        f"Mask-CN Classification ({split_name})",
        f"{'=' * 60}",
        f"Average AUC            : {d['average_AUC']:.4f}",
        f"Overall Accuracy       : {d['overall_AC']:.4f}",
    ]

    for name in CLASS_NAMES:
        m = d["per_class"][name]
        lines.append(f"{name} AUC                : {m['AUC']:.4f}")
        lines.append(f"{name} AC                 : {m['AC']:.4f}")
        lines.append(f"{name} SE                 : {m['SE']:.4f}")
        lines.append(f"{name} SP                 : {m['SP']:.4f}")

    return "\n".join(lines) + "\n"


def choose_classification_split():
    val_csv = os.path.join(CLS_ROOT, "ISIC2018_Task3_Validation_GroundTruth.csv")
    if not os.path.isfile(val_csv):
        return "train"

    try:
        val_ds = SkinClsDataset(CLS_ROOT, split="val", augment_data=False)
        if len(val_ds) > 0:
            return "val"
    except Exception:
        pass

    return "train"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    seg_ds = SkinSegDataset(SEG_IMG_DIR, SEG_MASK_DIR, augment_data=False)
    seg_loader = DataLoader(seg_ds, batch_size=4, shuffle=False, num_workers=4)

    coarse = load_checkpoint_model(CoarseSN().to(DEVICE), COARSE_CKPT)
    coarse.eval()

    maskcn = load_checkpoint_model(MaskCN(num_classes=NUM_CLASSES).to(DEVICE), MASKCN_CKPT)
    maskcn.eval()

    enhanced = load_checkpoint_model(EnhancedSN().to(DEVICE), ENHANCED_CKPT)
    enhanced.eval()

    coarse_accum = {}
    enhanced_accum = {}
    n_seg_samples = 0

    with torch.no_grad():
        for img, mask in seg_loader:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE).float()
            bsz = img.size(0)

            coarse_logits = coarse(img)
            if coarse_logits.shape[-2:] != mask.shape[-2:]:
                coarse_logits = F.interpolate(
                    coarse_logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            coarse_prob = torch.sigmoid(coarse_logits)

            accumulate_metric_sums(
                coarse_accum,
                segmentation_metrics(coarse_prob, mask),
                bsz,
            )

            x4 = torch.cat([img, coarse_prob], dim=1)
            cam = get_cam(maskcn, x4)
            cam = F.interpolate(
                cam,
                size=img.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            enhanced_logits = enhanced(img, cam)
            if enhanced_logits.shape[-2:] != mask.shape[-2:]:
                enhanced_logits = F.interpolate(
                    enhanced_logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            enhanced_prob = torch.sigmoid(enhanced_logits)

            accumulate_metric_sums(
                enhanced_accum,
                segmentation_metrics(enhanced_prob, mask),
                bsz,
            )

            n_seg_samples += bsz

    coarse_avg = finalize_metric_means(coarse_accum, n_seg_samples)
    enhanced_avg = finalize_metric_means(enhanced_accum, n_seg_samples)

    cls_split = choose_classification_split()
    cls_ds = SkinClsDataset(CLS_ROOT, split=cls_split, augment_data=False)
    cls_loader = DataLoader(cls_ds, batch_size=16, shuffle=False, num_workers=4)

    all_cls_probs = []
    all_cls_labels = []

    with torch.no_grad():
        for img, label in cls_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)

            coarse_logits = coarse(img)
            coarse_prob = torch.sigmoid(coarse_logits)
            x4 = torch.cat([img, coarse_prob], dim=1)

            logits, _ = maskcn(x4)
            probs = torch.softmax(logits, dim=1)

            all_cls_probs.append(probs.cpu().numpy())
            all_cls_labels.append(label.cpu().numpy())

    all_cls_probs = np.concatenate(all_cls_probs, axis=0)
    all_cls_labels = np.concatenate(all_cls_labels, axis=0)
    cls_metrics = classification_metrics(all_cls_probs, all_cls_labels)

    results = (
        fmt_seg("Coarse-SN Segmentation", coarse_avg) +
        fmt_seg("Enhanced-SN Segmentation", enhanced_avg) +
        fmt_cls(cls_metrics, cls_split)
    )

    print(results)

    with open(RESULTS_PATH, "w") as f:
        f.write(results)

    print(f"\nEvaluation complete. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

