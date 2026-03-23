# MB-DCNN: Mutual Bootstrapping for Skin Lesion Segmentation and Classification

A PyTorch implementation of the three-stage mutual bootstrapping deep convolutional neural network (MB-DCNN) from the paper *"A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification"* by Xie et al., published in IEEE Transactions on Medical Imaging (2020).

---

## What this does

Diagnosing skin lesions from dermoscopy images requires two things to work well at the same time: knowing where the lesion is (segmentation) and knowing what kind of lesion it is (classification). The trouble is each task benefits from the other. A better segmentation helps classification, and a more accurate classification helps the model focus on the right region during segmentation.

This codebase implements a pipeline that lets both tasks improve each other across three training stages, using the ISIC 2018 dataset.

---

## Architecture overview

The pipeline runs in three sequential stages:

**Stage 1 — Coarse Segmentation Network (Coarse-SN)**
A DeepLabV3+ model with a ResNet-50 backbone generates an initial rough segmentation mask for each image. It runs at output stride 16 using dilated convolutions in the last two residual layers, with atrous spatial pyramid pooling (ASPP) to capture multi-scale context.

**Stage 2 — Mask-guided Classification Network (Mask-CN)**
A modified Xception-based classifier takes in the original image concatenated with the coarse mask as a 4-channel input. The last pooling layer is replaced with two dilated depthwise separable convolutions. After training, the fully connected layer weights are reused to generate class activation maps (CAMs) that highlight the discriminative regions per class.

**Stage 3 — Enhanced Segmentation Network (Enhanced-SN)**
The same DeepLabV3+ backbone from Stage 1 is extended with an E-layer that fuses the ASPP encoder features with the fine-grained CAM from Stage 2. This lets the segmentation model benefit from class-level spatial attention, producing sharper and more accurate final masks.

---

## Loss functions

Training uses a hybrid loss across segmentation stages:

```
L_hybrid = L_dice + λ × L_rank
```

The Dice component handles overall mask overlap. The rank loss mines the K hardest foreground and background pixels per image based on their cross-entropy error and applies a pairwise hinge margin between their predicted scores, which sharpens decision boundaries in ambiguous regions.

Default settings: `λ = 0.05`, `K = 30`, `margin = 0.3`.

---

## Setup

**Requirements**

```
Python >= 3.11
PyTorch >= 2.0
torchvision
opencv-python
numpy
matplotlib
```

Install dependencies:

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

**Dataset**

Download the ISIC 2018 Task 1 segmentation dataset from Kaggle:

```bash
kaggle datasets download tschandl/isic2018-challenge-task1-data-segmentation
```

Update the paths in `CFG` at the top of the script:

```python
CFG = {
    "image_dir": "/path/to/ISIC2018_Task1-2_Training_Input",
    "mask_dir":  "/path/to/ISIC2018_Task1_Training_GroundTruth",
    ...
}
```

---

## Running the pipeline

To train all three stages end to end:

```bash
python mb_dcnn.py
```

The script will print progress for each stage and save the best checkpoint per stage to disk (`best_CoarseSN.pt`, `best_MaskCN.pt`, `best_EnhancedSN.pt`). After training completes, validation metrics are printed and a visualization grid is saved to `mb_dcnn_predictions.png`.

**Single image inference**

```python
from mb_dcnn import infer_single, build_coarse_sn, MaskCN, EnhancedSN
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coarse_sn   = build_coarse_sn().to(device)
mask_cn     = MaskCN(num_classes=3).to(device)
enhanced_sn = EnhancedSN(coarse_sn).to(device)

# load saved weights
coarse_sn.load_state_dict(torch.load("best_CoarseSN.pt"))
mask_cn.load_state_dict(torch.load("best_MaskCN.pt"))
enhanced_sn.load_state_dict(torch.load("best_EnhancedSN.pt"))

result = infer_single("path/to/image.jpg", coarse_sn, mask_cn, enhanced_sn, device)
# result keys: 'image', 'coarse_mask', 'cam', 'enhanced_mask'
```

---

## Evaluation metrics

The evaluation computes the following on the validation set after each stage:

| Metric | Description |
|---|---|
| JA | Jaccard index (IoU) |
| DI | Dice coefficient |
| pixel-AC | Pixel accuracy |
| pixel-SE | Sensitivity (recall) |
| pixel-SP | Specificity |

---

## Configuration

All major settings live in the `CFG` dictionary:

| Key | Default | Description |
|---|---|---|
| `img_size` | 224 | Input resolution |
| `batch_seg` | 16 | Batch size for segmentation training |
| `batch_cls` | 32 | Batch size for classification training |
| `lr` | 1e-4 | Learning rate (Adam) |
| `epochs_coarse` | 50 | Max epochs for Stage 1 |
| `epochs_mask` | 50 | Max epochs for Stage 2 |
| `epochs_enhanced` | 50 | Max epochs for Stage 3 |
| `lambda_rank` | 0.05 | Rank loss weight |
| `K` | 30 | Hard pixel count per region |
| `margin` | 0.3 | Rank loss margin |
| `num_classes` | 3 | Melanoma / nevus / seborrheic keratosis |

Early stopping with patience 10 is applied at each stage.

---

## Data augmentation

The segmentation dataset applies the following augmentations during training:

- Random centre crop between 50% and 100% of the image
- Random rotation in the range of plus or minus 10 degrees
- Random horizontal and vertical flipping
- Affine-style shear and shift via the crop offset

The classification dataset applies random horizontal flip and rotation.

---

## Notes on classification labels

The ISIC 2018 Task 1 dataset provides only segmentation masks, not class labels. The `ClassificationDataset` class parses labels from filenames using keyword matching (melanoma, nevus, seborrheic). If your filenames follow a different convention, pass a custom `label_map` dictionary when instantiating the dataset. For real experiments you will want to link these images to the Task 3 diagnosis labels from the same ISIC release.

---

## Reference

```
@article{xie2020mutual,
  title   = {A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification},
  author  = {Xie, Yutong and Zhang, Jianpeng and Xia, Yong and Shen, Chunhua},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2020},
  volume  = {39},
  number  = {7},
  pages   = {2482--2493}
}
```
