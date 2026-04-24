import os
import torch
from torch.utils.data import DataLoader
from dataset import SkinDataset
from models.mask_cn import MaskCN
from models.coarse_sn import CoarseSN

device="cuda"

ds = SkinDataset("data/images","data/masks")
loader = DataLoader(ds,4,shuffle=True)

coarse = CoarseSN().to(device)
coarse.load_state_dict(torch.load("weights/coarse.pth"))
coarse.eval()

model = MaskCN().to(device)
opt = torch.optim.Adam(model.parameters(),1e-4)
crit = torch.nn.BCEWithLogitsLoss()

for e in range(25):
    for img, mask in loader:

        img = img.to(device)
        mask = mask.to(device)

        # -------------------------
        # Coarse prediction
        # -------------------------
        with torch.no_grad():
            coarse_pred = torch.sigmoid(coarse(img))

        x = torch.cat([img, coarse_pred], dim=1)

        # -------------------------
        # Forward
        # -------------------------
        pred, feat = model(x)

        # -------------------------
        # Classification label
        # -------------------------
        label = (mask.sum(dim=[1,2,3]) > 0).float().to(device)

        cls_loss = crit(pred.squeeze(), label)

        # -------------------------
        # 🔥 Localization loss (NEW)
        # -------------------------
        cam = feat.mean(dim=1, keepdim=True)

        cam = torch.nn.functional.interpolate(
            cam,
            size=mask.shape[-2:]
        )

        loc_loss = torch.nn.functional.binary_cross_entropy_with_logits(cam, mask)

        # -------------------------
        # 🔥 Final loss
        # -------------------------
        loss = cls_loss + 0.5 * loc_loss

        # -------------------------
        # Backprop
        # -------------------------
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"MaskCN: {e} | Loss: {loss.item():.4f}")

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/maskcn.pth")
