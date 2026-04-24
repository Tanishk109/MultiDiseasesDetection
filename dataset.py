import os, cv2, torch
import numpy as np
from torch.utils.data import Dataset

class SkinDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=224):
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        mask = cv2.imread(os.path.join(
            self.mask_dir, name.split('.')[0] + "_segmentation.png"), 0)
        

        img = cv2.resize(img, (224,224)) / 255.0
        mask = cv2.resize(mask, (224,224)) / 255.0

        img = torch.tensor(img).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask
