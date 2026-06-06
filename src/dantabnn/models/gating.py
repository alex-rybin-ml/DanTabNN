"""Feature selection gating via Gumbel-Softmax.

Implements differentiable feature selection using attention-based gating,
allowing the model to learn which features are most relevant for the task.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGating(nn.Module):
    """Differentiable feature selection via Gumbel-Softmax.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    temparature : float, default=1.0
        Temperature for Gumbel-Softmax relaxation. Lower values make the
        distribution more peaky (closer to one-hot).
    hard : bool, default=True
        If True, use straight-through estimator: during forward pass the
        sample are discretized (0 or 1) but gradients flow through the
        soft relaxation.
    droupout : flot, default=0.0
        Dropout applied to the gating logits (helps prevent over-reliance
        on a small subset of features).
    init_bias : float, default=0.0
        Initial bias for the gating logits. Positive values encourage
        features to be selected initially (useful for warm-up).
    """

    def __init__(
            self,
            input_dim: int,
            temperature: float = 1.0,
            hard: bool = True,
            dropout: float = 0.0,
            init_bias: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        self.hard = hard
        self.dropout = nn.Dropout(dropout)

        # Learnable per-featuregating logits (one logit per feature)
        self.gate_logits = nn.Parameter(torch.zeros(input_dim))
        # Initialize with a small bias to encourage exploration
        nn.init.constant_(self.gate_logits, init_bias)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            training: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature-wise gating mask and apply it to input.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        mask : torch.Tensor, optional
            Binary mask of shape (batch_size, input_dim) indicating missing
            features (1 = missing). If provided, the gating probability for
            missing features if forced to zero.
        training : bool, optional
            If True, apply dropout and use Gumbel noise for stochastic
            sampling. If None, uses `self.training`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Selected features (same shape as x) where unselected features
              are zeroed out (or scaled b gate probability).
            - Gate mask of shape (batch_size, input_dim) with values in [0,1]
              representing the probability of selected each feature.
        """
        if training is None:
            training = self.training

        # Expand per-feature logits to batch dimension
        logits = self.get_logits.expand(x.size(0), -1)  # (B, D)

        # Apply dropout to logits during training
        if training:
            logits = self.dropout(logits)

        # Zero out logits for missing features (if mask provided)
        if mask is not None:
            # mask is 1 where feature is missing -> we want logits = -inf
            logits = logits.masked_fill(mask.bool(), float('-inf'))

        # Gumbel-Softmax sampling (binary case: two-class)
        # We treat each feature independency, sampling a binary decision.
        # Equivalent to applying Gumbel-Softmax per feature with 2 classes:
        # class 0 = discard, class 1 = select.
        # We can compute probability of selection as sigmoid(logits).
        # The Gumbel-Softmax trick fo binary variables reduces to
        #   p = sigmoid((logits + gumbel_noise) / temperature)
        # where gumbel_noise = -log(-log(U)) with U ~ Uniform(0,1).
        # We then apply straight-through estimator if hard=True.
        if training:
            # Add Gumbel nose for stochasticity
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel_noise) / self.temperature
        else:
            noisy_logits = logits / self.temperature

        # Probability of selecting each feature
        gate_probs = torch.sigmoid(noisy_logits)  # (B, D)

        if self.hard:
            # Straight-through estimator: discretize during forward pass,
            # but gradients flow through the continuous probabilities.
            gate_hard = (gate_probs > 0.5).float()
            # Use get_probs for backward, gate_hard for forward
            gate = gate_hard - gate_probs.detach() + gate_probs
        else:
            gate = gate_probs

        # Apply to input (element-wise multiplication)
        x_gated = x * gate

        return x_gated, gate

    def get_selection_probabilities(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the probability of selecting each feature (without sampling).

        Parameters
        ----------
        mask : torch.Tensor, optional
             Missingness mask (1 = missing). If provided, missing features
             get probability

        Returns
        -------
        torch.Tensor of shape (input_dim,)
            Selection probability per feature, averaged over batch dimension
            if a batch mask id given.
        """
        logits = self.gate_logits  # (D,)
        if mask is not None:
            # mask shape (B, D) -> average over batch?
            # We'll compute per-feature probability as sigmoid(logits) where not missing
            # For simplicity, ignore missing by setting logits to -inf
            logits_expanded = logits.expand(mask.size(0), -1)
            logits_expanded = logits_expanded.masked_fill(mask.bool(), float('-inf'))
            probs = torch.sigmoid(logits_expanded).mean(dim=0)
        else:
            probs = torch.sigmoid(logits)
        return probs


class TopKFeatureGating(nn.Module):
    """Differentiable top-k feature selection via Gumbel-Softmax.

    Select exactly k features per sample, using a categorical
    distribution over features.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    k : int
        Number of features to select per sample.
    temperature : float, default=1.0
        Temperature for Gumbel-Softmax relaxation.
    hard : bool, default=True
        Use straight-through estimator.
    dropout : float, default=0.0
        Dropout on the selection logits.
    """

    def __init__(
            self,
            input_dim: int,
            k: int,
            temperature: float = 1.0,
            hard: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        assert 0 < k <= input_dim, f"k must be between 1 and input_dim, got {k}"
        self.input_dim = input_dim
        self.k = k
        self.temperature = temperature
        self.hard = hard
        self.dropout = nn.Dropout(dropout)

        # Learnable logits for ech feature (shared across samples)
        self.logits = nn.Parameter(torch.zeros(input_dim))

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            training: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k features per sample.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        mask : torch.Tensor, optional
            Binary mask of shape (batch_size, input_dim) where 1 indicates
            missing features (cannot be selected).
        training : bool, optional
            Whether to add Gumbel noise. Defaults to self.training.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Selected features (same shape as x) with unselected features zeroed.
            - Selected mask of shape (batch_size, input_dim) with values in [0,1].
        """
        if training is None:
            training = self.training

        # Expand logits to batch dimension
        logits = self.logits.expand(x.size(0), -1)  # (B, D)

        if training:
            logits = self.dropout(logits)

        # Mask out missing features
        if mask is not None:
            logits = logits.masked_fill(mask.bool(), float('-inf'))

        # Gumbel-Softmax to get a categorical distribution over features
        if training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
            noisy_logits = (logits + gumbel_noise) / self.temperature
        else:
            noisy_logits = logits / self.temperature

        # Compute softmax over features (each sample independency)
        scores = F.softmax(noisy_logits, dim=1)  # (B, D)

        # Top-k selection: we can use relaxed top-k via continuous relaxation
        # One simple approach: use the Gumbel-Softmax trick with a top-k
        # categorical distribution. Instead, we can treat each feature as
        # independent binary variable with probability proportional to scores.
        # For simplicity, we select the k features with the highest scores.
        # To keep differentiability, we use the straight-through estimator
        # on the top-k indicator.
        _, topk_indices = torch.topk(scores, self.k, dim=-1)  # (B, K)
        # Create a binary mask for top-k features
        mask_topk = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)

        if self.hard:
            # Straight-through estimator
            gate = mask_topk - scores.detach() + scores
        else:
            # Use soft scores (not exactly top-k)
            gate = scores

        x_gated = x * gate
        return x_gated, gate


def create_feature_gating(
        input_dim: int,
        gating_type: str = "soft",
        **kwargs,
) -> Optional[nn.Module]:
    """Factory function to create a feature gating module.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    gating_type : str
        Type of gating:
        - 'soft' : FeatureGating (independent Bernoulli)
        - 'topk' : TopKFeatureGating (select exactly k features)
        - 'none' or '' : returns None (no gating)
    **kwargs
        Additional keyword arguments passed to the gating constructor.

    Returns
    -------
    nn.Module or None
        The gating module, or None if gating_type is 'none' or empty.
    """
    import inspect

    if gating_type == 'none' or not gating_type:
        return None

    target_class = {'soft': FeatureGating, 'topk': TopKFeatureGating}.get(gating_type)
    if target_class is None:
        raise ValueError(f"Unknown gating_type: {gating_type}. "
                         f"Must be one of 'soft', 'topk', 'none'.")

    # Filter kwargs to only those accepted by the target class constructor
    valid_params = set(inspect.signature(target_class.__init__).parameters.keys()) - {'self'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return target_class(input_dim, **filtered_kwargs)
