import os
import torch
from torch.utils.data import DataLoader
from dataset import SkinDataset
from models.coarse_sn import CoarseSN
from losses import CombinedLoss

device="cuda"

ds = SkinDataset("data/images","data/masks")
loader = DataLoader(ds,4,shuffle=True,num_workers=4)

model = CoarseSN().to(device)
loss_fn = CombinedLoss()
opt = torch.optim.Adam(model.parameters(),1e-4)

for e in range(10):
    for img,mask in loader:
        img,mask = img.to(device),mask.to(device)

        pred = model(img)
        pred = torch.nn.functional.interpolate(pred,size=mask.shape[-2:])

        loss = loss_fn(pred,mask)

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Coarse:",e)

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/coarse.pth")
