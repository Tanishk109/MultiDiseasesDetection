import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class RankLoss(nn.Module):
    def __init__(self, K=30, margin=0.3):
        super().__init__()
        self.K = K
        self.margin = margin

    def forward(self, logits, targets):
        targets = targets.float()
        probs = torch.sigmoid(logits)

        batch_size = probs.size(0)
        total_loss = probs.new_tensor(0.0)

        for b in range(batch_size):
            p = probs[b].view(-1)
            t = targets[b].view(-1)

            lesion_mask = t > 0.5
            bg_mask = t <= 0.5

            if lesion_mask.sum() == 0 or bg_mask.sum() == 0:
                continue

            lesion_probs = p[lesion_mask]
            bg_probs = p[bg_mask]

            lesion_err = torch.abs(lesion_probs - 1.0)
            bg_err = torch.abs(bg_probs - 0.0)

            k_lesion = min(self.K, lesion_probs.numel())
            k_bg = min(self.K, bg_probs.numel())

            lesion_idx = lesion_err.topk(k_lesion, largest=True).indices
            bg_idx = bg_err.topk(k_bg, largest=True).indices

            H1 = lesion_probs[lesion_idx]
            H0 = bg_probs[bg_idx]

            H0 = H0.unsqueeze(1)
            H1 = H1.unsqueeze(0)

            loss_b = F.relu(H0 - H1 + self.margin).mean()
            total_loss = total_loss + loss_b

        return total_loss / batch_size


class HybridLoss(nn.Module):
    def __init__(self, lam=0.05, K=30, margin=0.3):
        super().__init__()
        self.dice = DiceLoss()
        self.rank = RankLoss(K=K, margin=margin)
        self.lam = lam

    def forward(self, logits, targets):
        l_dice = self.dice(logits, targets)
        l_rank = self.rank(logits, targets)
        total = l_dice + self.lam * l_rank
        return total, l_dice, l_rank

