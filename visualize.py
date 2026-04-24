import cv2
import os

os.makedirs("outputs/overlay", exist_ok=True)

for i in range(5):
    img = cv2.imread(f"data/images/ISIC_00000{i}.jpg")  # adjust name
    mask = cv2.imread(f"outputs/pred_{i}.png", 0)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    overlay = img.copy()
    overlay[mask > 0] = [0, 0, 255]

    result = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    cv2.imwrite(f"outputs/overlay/overlay_{i}.png", result)
