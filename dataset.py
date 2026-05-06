import os
import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


ISIC_CLASS_MAP = {
    "melanoma": 0,
    "nevus": 1,
    "seborrheic_keratosis": 2,
    "seborrheic keratosis": 2,
    "keratosis": 2,
}


def _resize_image(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask, size):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def _warp_image(img, mat, out_size):
    return cv2.warpAffine(
        img,
        mat,
        out_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _warp_mask(mask, mat, out_size):
    return cv2.warpAffine(
        mask,
        mat,
        out_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def whiten_image(img):
    img = img.astype(np.float32)
    for c in range(3):
        ch = img[:, :, c]
        img[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-5)
    return img


def augment_image_and_mask(img, mask=None, size=224):
    h, w = img.shape[:2]

    scale = np.random.uniform(0.5, 1.0)
    crop_h = max(1, int(h * scale))
    crop_w = max(1, int(w * scale))
    top = np.random.randint(0, max(1, h - crop_h + 1))
    left = np.random.randint(0, max(1, w - crop_w + 1))

    img = img[top:top + crop_h, left:left + crop_w]
    if mask is not None:
        mask = mask[top:top + crop_h, left:left + crop_w]

    h, w = img.shape[:2]

    angle = np.random.uniform(-10.0, 10.0)
    shear = np.random.uniform(-0.1, 0.1)
    tx = np.random.uniform(-20.0, 20.0)
    ty = np.random.uniform(-20.0, 20.0)
    zoom = np.random.uniform(1.0, 1.1)

    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    rot_3 = np.vstack([rot, [0, 0, 1]]).astype(np.float32)

    shear_3 = np.array([
        [1.0, shear, 0.0],
        [0.0, 1.0,   0.0],
        [0.0, 0.0,   1.0],
    ], dtype=np.float32)

    trans_3 = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    affine = trans_3 @ shear_3 @ rot_3
    mat = affine[:2, :]

    img = _warp_image(img, mat, (w, h))
    if mask is not None:
        mask = _warp_mask(mask, mat, (w, h))

    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)

    if np.random.rand() > 0.5:
        img = cv2.flip(img, 0)
        if mask is not None:
            mask = cv2.flip(mask, 0)

    img = _resize_image(img, size)
    img = whiten_image(img)

    if mask is not None:
        mask = _resize_mask(mask, size)
        mask = (mask > 0.5).astype(np.float32)
        return img, mask

    return img, None


def preprocess_image(img, size=224):
    img = _resize_image(img, size)
    img = whiten_image(img)
    return img


def preprocess_mask(mask, size=224):
    mask = _resize_mask(mask, size)
    mask = (mask > 0.5).astype(np.float32)
    return mask


def load_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def load_gray_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read grayscale image: {path}")
    img = img.astype(np.float32) / 255.0
    return img


def image_to_tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).float()


def mask_to_tensor(mask):
    return torch.from_numpy(mask).unsqueeze(0).float()


class SkinSegDataset(Dataset):
    """
    Segmentation dataset.

    Expected:
      data/images/ISIC_xxx.jpg
      data/mask/ISIC_xxx_segmentation.png or .jpg or .jpeg
    """
    def __init__(self, img_dir, mask_dir, size=224, augment_data=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.augment_data = augment_data

        self.images = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        stem = os.path.splitext(name)[0]

        img_path = os.path.join(self.img_dir, name)

        mask_path_png = os.path.join(self.mask_dir, f"{stem}_segmentation.png")
        mask_path_jpg = os.path.join(self.mask_dir, f"{stem}_segmentation.jpg")
        mask_path_jpeg = os.path.join(self.mask_dir, f"{stem}_segmentation.jpeg")

        if os.path.isfile(mask_path_png):
            mask_path = mask_path_png
        elif os.path.isfile(mask_path_jpg):
            mask_path = mask_path_jpg
        elif os.path.isfile(mask_path_jpeg):
            mask_path = mask_path_jpeg
        else:
            raise FileNotFoundError(f"Mask not found for image: {stem}")

        img = load_rgb_image(img_path)
        mask = load_gray_image(mask_path)

        if self.augment_data:
            img, mask = augment_image_and_mask(img, mask, self.size)
        else:
            img = preprocess_image(img, self.size)
            mask = preprocess_mask(mask, self.size)

        return image_to_tensor(img), mask_to_tensor(mask)


class SkinClsDataset(Dataset):
    """
    ISIC 2018 Task 3 classification dataset.

    Expected:
      root/
        images/
          ISIC_xxx.jpg
        ISIC2018_Task3_Training_GroundTruth.csv
        ISIC2018_Task3_Validation_GroundTruth.csv

    CSV format:
      image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
    """
    CLASS_ORDER = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __init__(self, root, split="train", size=224, augment_data=True):
        self.root = root
        self.size = size
        self.augment_data = augment_data
        self.image_dir = os.path.join(root, "images")

        if split == "train":
            self.csv_path = os.path.join(root, "ISIC2018_Task3_Training_GroundTruth.csv")
        elif split == "val":
            self.csv_path = os.path.join(root, "ISIC2018_Task3_Validation_GroundTruth.csv")
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"Classification CSV not found: {self.csv_path}")

        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row["image"]

                label = None
                for idx, cls_name in enumerate(self.CLASS_ORDER):
                    if float(row[cls_name]) == 1.0:
                        label = idx
                        break

                if label is None:
                    continue

                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
                    p = os.path.join(self.image_dir, image_id + ext)
                    if os.path.isfile(p):
                        img_path = p
                        break

                if img_path is not None:
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = load_rgb_image(img_path)

        if self.augment_data:
            img, _ = augment_image_and_mask(img, None, self.size)
        else:
            img = preprocess_image(img, self.size)

        return image_to_tensor(img), torch.tensor(label, dtype=torch.long)


SkinDataset = SkinSegDataset

