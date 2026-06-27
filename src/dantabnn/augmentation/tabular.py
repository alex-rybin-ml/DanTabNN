"""Tabular data augmentation: CutMix + feature noise for regularization.

CutMix for tabular data mixes two samples with a convex combination:
    X_new = lambda * X_i + (1-lambda) * X_j
where lambda ~ Beta(alpha, alpha). This creates synthetic samples that
lie on the line between two real samples.

Feature noise adds small Gaussian noise to a random subset of features,
which acts as a form of input perturbation regularization.
"""

import random
from typing import Optional

import torch
import torch.nn as nn


class TabularAugmentation(nn.Module):
    """Apply CutMix + feature noise augmentation to tabular data during training.

    Parameters
    ----------
    p_cutmix : float, default=0.3
        Probability of applying CutMix to each sample in a batch.
    p_noise : float, default=0.1
        Probability of adding noise to each feature.
    noise_std : float, default=0.01
        Standard deviation of Gaussian noise (relative to feature scale).
        Since features are standardized (mu=0, sigma=1), 0.01 is small.
    beta_alpha : float, default=1.0
        Alpha parameter for Beta distribution (alpha=beta).
        alpha=1.0 gives uniform mixing ratios.
    """

    def __init__(
        self,
        p_cutmix: float = 0.3,
        p_noise: float = 0.1,
        noise_std: float = 0.01,
        beta_alpha: float = 1.0,
    ):
        super().__init__()
        self.p_cutmix = p_cutmix
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.beta_alpha = beta_alpha

    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        """Apply augmentation to a batch.

        Parameters
        ----------
        X : (B, D) tensor
            Input features.
        y : (B,) or (B, 1) tensor, optional
            Targets. Required if CutMix is enabled (for regression/binary).
            For multiclass classification targets, CutMix mixes the one-hot targets.

        Returns
        -------
        X_aug : (B, D) tensor
            Augmented features.
        y_aug : tensor or None
            Augmented targets (or original if no augmentation applied).
        """
        B = X.shape[0]
        X_aug = X.clone()

        # CutMix: mix pairs of samples
        if self.p_cutmix > 0 and B >= 2:
            for i in range(B):
                if random.random() < self.p_cutmix:
                    j = random.randint(0, B - 1)
                    lam = random.betavariate(self.beta_alpha, self.beta_alpha)
                    X_aug[i] = lam * X[i] + (1 - lam) * X[j]
                    if y is not None:
                        y[i] = lam * y[i] + (1 - lam) * y[j]

        # Feature noise: small Gaussian perturbation
        if self.p_noise > 0:
            mask = torch.rand_like(X_aug) < self.p_noise
            noise = torch.randn_like(X_aug) * self.noise_std
            X_aug = X_aug + mask.float() * noise

        return X_aug, y
