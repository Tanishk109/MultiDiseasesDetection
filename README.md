# MB-DCNN Skin Lesion Analysis — Final Results Report

## Project Title
Implementation and Evaluation of MB-DCNN for Automated Skin Lesion Segmentation and Classification

---

# 1. Introduction

This project implements the MB-DCNN (Mutual Bootstrapping Deep Convolutional Neural Network) framework proposed in the paper:

“A Mutual Bootstrapping Model for Automated Skin Lesion Segmentation and Classification”

The objective of the project is to jointly perform:
- Skin lesion segmentation
- Skin lesion classification

using a mutual learning strategy where:
- segmentation improves classification
- classification improves segmentation

---

# 2. Core Ideology of MB-DCNN

Traditional medical imaging approaches usually treat segmentation and classification as separate tasks.

The MB-DCNN framework proposes:

Segmentation → helps Classification
Classification → helps Segmentation

This creates a mutual bootstrapping mechanism.

The complete pipeline:

Input Image
      ↓
Coarse-SN
(Coarse Segmentation)
      ↓
Coarse Lesion Mask
      ↓
Mask-CN
(Mask-Guided Classification)
      ↓
Class Activation Maps (CAM)
      ↓
Enhanced-SN
(CAM-Guided Segmentation Refinement)
      ↓
Final Refined Segmentation

---

# 3. Implemented Networks

## 3.1 Coarse-SN

Purpose:
- Generate initial coarse lesion masks.

Architecture:
- DeepLabV3+
- Aligned Xception Encoder
- ASPP Module
- Decoder Network

Loss Function:
- Dice Loss
- Rank Loss
- Hybrid Dice + Rank Loss

Reason:
- Dice Loss handles class imbalance.
- Rank Loss focuses on difficult boundary pixels.

---

## 3.2 Mask-CN

Purpose:
- Perform skin lesion classification.
- Generate Class Activation Maps (CAM).

Input:
- RGB Image + Coarse Mask (4-channel input)

Architecture:
- Xception Backbone
- Dilated Convolutions
- Global Average Pooling
- Fully Connected Layer

Reason:
- Coarse mask removes irrelevant background.
- Dilated convolutions preserve spatial information.
- CAM helps localize lesion regions.

Dataset Used:
- ISIC 2018 (7-class classification)

Classes:
- MEL
- NV
- BCC
- AKIEC
- BKL
- DF
- VASC

---

## 3.3 Enhanced-SN

Purpose:
- Refine segmentation using CAM guidance.

Architecture:
- Encoder-Decoder Segmentation Network
- Enhancement Layer (E-layer)
- CAM + Feature Fusion

Reason:
- CAM transfers localization knowledge from classification to segmentation.
- Produces more refined lesion boundaries.

---

# 4. Dataset Information

## Segmentation Dataset
- ISIC 2018
- Total Samples: 2594

Split:
- Training Samples: 2076
- Validation Samples: 518

## Classification Dataset
- ISIC 2018 Task 3 Dataset
- Total Samples: 10015

Split:
- Training Samples: 8012
- Validation Samples: 2003

---

# 5. Training Results

## 5.1 Coarse-SN Training Results

Best Validation Results:
- Validation IoU (JA): 0.7709
- Validation Dice: 0.8514

Observation:
- Stable convergence achieved.
- Strong segmentation performance obtained.

---

## 5.2 Mask-CN Training Results

Best Validation Accuracy:
- 71.57%

Observation:
- Classification performance improved gradually.
- Achieved good performance for a 7-class medical image classification problem.

---

## 5.3 Enhanced-SN Training Results

Best Validation Results:
- Validation IoU (JA): 0.7882
- Validation Dice: 0.8671

Observation:
- Enhanced-SN improved segmentation compared to Coarse-SN.
- CAM-guided refinement successfully improved lesion boundaries.

---

# 6. Final Evaluation Results

# 6.1 Segmentation Performance

| Metric | Coarse-SN | Enhanced-SN |
|---|---|---|
| JA (IoU) | 0.7728 | 0.7869 |
| Dice | 0.8518 | 0.8653 |
| Pixel Accuracy | 0.9389 | 0.9466 |
| Sensitivity | 0.8818 | 0.8956 |
| Specificity | 0.9533 | 0.9667 |

Observation:
- Enhanced-SN outperformed Coarse-SN.
- Mutual bootstrapping improved segmentation quality.

---

# 6.2 Classification Performance

## Overall Metrics

| Metric | Result |
|---|---|
| Average AUC | 0.8736 |
| Overall Accuracy | 72.80% |

---

## Class-wise AUC

| Class | AUC |
|---|---|
| MEL | 0.8036 |
| NV | 0.8900 |
| BCC | 0.9221 |
| AKIEC | 0.9100 |
| BKL | 0.8398 |
| DF | 0.7806 |
| VASC | 0.9691 |

Observation:
- Strong performance on multiple lesion categories.
- Rare classes remain challenging due to class imbalance.

---

# 7. Comparison with Original Paper

| Component | Original Paper | Our Implementation |
|---|---|---|
| Dataset | ISIC 2017 | ISIC 2018 |
| Number of Classes | 3 | 7 |
| Segmentation JA | 0.804 | 0.7869 |
| Dice Score | ~0.87–0.89 | 0.8653 |
| Classification AUC | 0.938 | 0.8736 |

Important Note:
- The original paper used 3-class classification.
- This implementation used 7-class classification.
- ISIC 2018 is significantly more challenging because of:
  - higher inter-class similarity
  - severe class imbalance
  - increased task complexity

Therefore, the obtained performance is considered strong and competitive.

---

# 8. Key Achievements

## Successfully Implemented:

- Mutual Bootstrapping Pipeline
- Coarse-SN
- Mask-CN
- Enhanced-SN
- Dice + Rank Hybrid Loss
- CAM-guided Segmentation Refinement
- Multi-stage Medical Image Learning Pipeline

---

# 9. Challenges Faced During Implementation

- Dataset mismatch and validation image issues
- Empty validation loader handling
- 3-class vs 7-class model mismatch
- CAM generation debugging
- Circular import issues during Enhanced-SN implementation
- Medical image class imbalance

All issues were resolved successfully during implementation.

---

# 10. Conclusion

The MB-DCNN framework was successfully implemented and evaluated for automated skin lesion analysis.

Final achievements:
- 78.69% IoU in segmentation
- 86.53% Dice score
- 87.36% Average AUC in classification
- Successful adaptation from 3-class to 7-class classification
- Effective CAM-guided segmentation refinement

The project demonstrates that mutual learning between segmentation and classification can improve performance in medical image analysis tasks.

Overall, the implementation successfully reproduces the core ideology and architecture proposed in the original MB-DCNN research paper.
