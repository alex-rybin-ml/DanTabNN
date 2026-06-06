"""Cross Network module for sparce feature interaction.

Implements the cross layer from Deep @ Cross Network (DCN):
    x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
where * denotes element-wise multiplication.
"""

from typing import Optional

import torch
import torch.nn as nn


class CrossNetwork(nn.Module):
    """Cross Network for explicit feature crosses.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features (must match output dimension).
    num_layers : int, default=3
        Numbers of cross layers.
    low_rank : bool, default=False
        if True, use factorized cross layers (W = U * V^T) to reduce parameters.
    rank_ratio : float, default=0.5
        Ratio of rank to input_dim for low-rank factorization (ignored if low_rank=False).
    dropout : float, default=0.0
        Dropout rate applied after each cross layer (optional).
    """

    def __init__(
            self,
            input_dim: int,
            num_layers: int = 3,
            low_rank: bool = False,
            rank_ratio: float = 0.5,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.low_rank = low_rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Store layers
        self.cross_layers = nn.ModuleList()
        for i in range(num_layers):
            if low_rank:
                rank = max(1, int(input_dim * rank_ratio))
                layer = FactorizedCrossLayer(input_dim, rank)
            else:
                layer = CrossLayer(input_dim)
            self.cross_layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            output of shape (batch_size, input_dim).
        """
        x0 = x  # keep original input for residual connection
        for layer in self.cross_layers:
            x = layer(x0, x)
            x = self.dropout(x)
        return x


class CrossLayer(nn.Module):
    """Single cross layer: x_{out} = x_0 * (W * x + b) + x."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim, bias=True)
        # Initialize weights to small values as suggested in DCN paper
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x0 : torch.Tensor
            Original input (batch_size, input_dim).
        x : torch.Tensor
            Current representation (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        # W * x + b
        projected = self.linear(x)
        # element-wise multiplication with x0
        crossed = x0 * projected
        # residual connection
        return crossed + x


class FactorizedCrossLayer(nn.Module):
    """Factorized cross layer using low-rank decomposition W = U * V^T."""

    def __init__(self, input_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.U = nn.Linear(rank, input_dim, bias=False)  # shape (input_dim, rank)
        self.V = nn.Linear(input_dim, rank, bias=False)  # shape (rank, input_dim)
        self.bias = nn.Parameter(torch.zeros(input_dim))
        # Initialize weights
        nn.init.xavier_uniform_(self.U.weight, gain=0.1)
        nn.init.xavier_uniform_(self.V.weight, gain=0.1)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x0: torch.Tensor
            Original input (batch_size, input_dim).
        x: torch.Tensor
            Current representation (batch_size, input_dim).

        Return
        ------
        torch.Tensor
            Output tensor.
        """
        # V^T x: linear transform V from input_dim to rank
        v_out = self.V(x)  # (B, rank)
        # U (V^T x): linear transformation U from rank to input_dim
        u_out = self.U(v_out)  # (B, input_dim)
        projected = u_out + self.bias
        crossed = x0 * projected
        return crossed + x


# Convenience function to create a cross network based on configuration
def create_interaction_layer(
        input_dim: int,
        interaction_type: str = "legacy",
        num_cross_layers: int = 3,
        low_rank: bool = False,
        rank_ratio: float = 0.5,
        dropout: float = 0.0,
) -> Optional[nn.Module]:
    """Create an interaction layer (cross network, factorized, or none).

    Parameters
    ----------
    input_dim : int
        Dimensions of input features.
    interaction_type : str
        One of 'legacy' (no interaction), 'cross', 'factorized'.
    num_cross_layers : int
        Number of cross layers (used for 'cross' and 'factorized').
    low_rank : bool
        if True, use factorized cross layers (only for 'cross').
    rank_ratio : float
        Rank ratio for factorized layers.
    dropout : float
        Dropout rate.

    Returns
    -------
    nn.Module or None
        The interaction module, or None if interaction == 'legacy'.
    """
    if interaction_type == "legacy":
        return None
    elif interaction_type == "cross":
        return CrossNetwork(
            input_dim=input_dim,
            num_layers=num_cross_layers,
            low_rank=low_rank,
            rank_ratio=rank_ratio,
            dropout=dropout,
        )
    elif interaction_type == "factorized":
        return CrossNetwork(
            input_dim=input_dim,
            num_layers=num_cross_layers,
            low_rank=True,
            rank_ratio=rank_ratio,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown interaction_type: {interaction_type}")
