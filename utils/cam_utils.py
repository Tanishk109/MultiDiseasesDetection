import torch
import torch.nn.functional as F

def get_cam(model, x):
    model.eval()

    pred, feat = model(x)   # feat: [B,2048,H,W]

    # FC weights
    weight = model.fc.weight.view(1, -1, 1, 1)  # [1,2048,1,1]

    # CAM
    cam = (feat * weight).sum(dim=1, keepdim=True)

    # ReLUß
    cam = F.relu(cam)

    # 🔥 Normalize PER SAMPLE (VERY IMPORTANT)
    B, _, H, W = cam.shape
    cam = cam.view(B, -1)

    cam = cam - cam.min(dim=1, keepdim=True)[0]
    cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-5)

    cam = cam.view(B, 1, H, W)

    return cam
