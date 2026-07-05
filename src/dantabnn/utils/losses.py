"""Focal loss for binary and multiclass classification.

Focal Loss (Lin et al. 2017) down-weights easy examples and focuses
training on hard negatives — especially useful for imbalanced datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for binary classification with logits input.

    Parameters
    ----------
    alpha : float, default=0.25
        Weighting factor for the rare class. 0.25 balances the effect
        of gamma on positive vs negative examples.
    gamma : float, default=2.0
        Focusing parameter. Higher values down-weight easy examples more
        aggressively. Standard range: [0.5, 5.0].
    reduction : str, default='mean'
        Reduction mode: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits of shape (N, 1) or (N,).
        targets : torch.Tensor
            Binary targets of shape (N, 1) or (N,).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if inputs.ndim > 1 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.ndim > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class MultiClassFocalLoss(nn.Module):
    """Focal loss for multiclass classification.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing parameter.
    reduction : str, default='mean'
        Reduction mode.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute multiclass focal loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Raw logits of shape (N, C).
        targets : torch.Tensor
            Class indices of shape (N,).

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * nll_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss