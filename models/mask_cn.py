import torch.nn as nn
import torchvision.models as models

class MaskCN(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet50(pretrained=True)
        base.conv1 = nn.Conv2d(4,64,7,2,3,bias=False)

        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048,1)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat).view(x.size(0), -1)
        out = self.fc(pooled)
        return out, feat
