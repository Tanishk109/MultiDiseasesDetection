# Multi-Stage Skin Lesion Segmentation using CAM

## 📌 Overview
This project implements a multi-stage deep learning pipeline for skin lesion segmentation.

Pipeline:
1. CoarseSN (DeepLabV3+) → initial segmentation
2. MaskCN → generates Class Activation Maps (CAM)
3. EnhancedSN → fuses features + CAM for refined segmentation

## 🚀 Results
| Model     | Dice  | IoU   |
|----------|-------|------|
| Coarse   | 0.934 | 0.878 |
| MaskCN   | 0.590 | 0.451 |
| Enhanced | 0.963 | 0.930 |

## 🧠 Key Idea
Combining segmentation with attention improves localization and boundary accuracy.

## ⚙️ Tech Stack
- PyTorch
- segmentation_models_pytorch
- OpenCV

## 📂 Structure
- models/
- train/
- utils/
- evaluate.py


## 📊 Metrics
- Dice Score
- IoU

## 🔥 Result
Achieved **0.96 Dice Score**, outperforming baseline segmentation.


