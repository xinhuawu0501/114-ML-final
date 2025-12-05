import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, alpha=None):
        # CE loss per element (no reduction)
        ce = F.cross_entropy(inputs, targets, reduction='none')

        # pt = probability of the true class
        pt = torch.exp(-ce)

        # focal term
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        # Apply alpha weighting AFTER CE
        if alpha is not None:
            # alpha[targets] broadcasts to match shape
            alpha_t = alpha.to(inputs.device)[targets]
            loss = alpha_t * loss

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss