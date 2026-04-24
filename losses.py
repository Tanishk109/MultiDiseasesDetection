import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds).view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (preds * targets).sum(1)
        dice = (2 * inter) / (preds.sum(1) + targets.sum(1) + 1e-5)

        return 1 - dice.mean()

class RankLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        lesion = preds[targets > 0.5]
        bg = preds[targets <= 0.5]

        if len(lesion) == 0 or len(bg) == 0:
            return torch.tensor(0.0, device=preds.device)

        return torch.relu(self.margin - (lesion.mean() - bg.mean()))

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.rank = RankLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets) + self.dice(preds, targets) + self.rank(preds, targets)
