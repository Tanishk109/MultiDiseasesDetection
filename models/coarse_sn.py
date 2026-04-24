import segmentation_models_pytorch as smp
import torch.nn as nn

class CoarseSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=1
        )

    def forward(self, x):
        return self.model(x)
