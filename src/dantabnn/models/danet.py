"""Dual-Attention Network (DANet) module for tabular data."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional
from .cross import create_interaction_layer
from .gating import create_feature_gating


class FeatureAttention(nn.Module):
    """Self attention across feature dimensions."""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.2, missing_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

        self.missing_bias = nn.Parameter(torch.zeros(1)) if missing_bias else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x shape: (batch_size, seq_len, input_dim) where seq_len = 1 for tabular."""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each shape (B, num_heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)

        # Apply missingness bias if provided
        if mask is not None and self.missing_bias is not None:
            # mask shape (B, D) where D = input_dim
            mask_ratio = mask.float().mean(dim=1, keepdim=True)  # (B, 1)
            # reshape to (B, 1, 1, 1) to broadcast across and sequence length
            bias = self.missing_bias * mask_ratio.view(B, 1, 1, 1)
            attn = attn + bias

        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)  # residual connection
        return out


class SampleAttention(nn.Module):
    """Attention across samples (optional). Not used in standard tabular setting."""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, input_dim)."""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out + x)
        return out


class DANetModule(nn.Module):
    """Dual-Attention Network for tabular data.
    
     Consists of:
     1. Feature-wise attention (self-attention across feature embeddings).
     2. Sample-wise attention (optional, can be disabled)
     3. Feed-forward network.
    """
    
    def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [64, 32],
            dropout: float = 0.2,
            attention_heads: int = 4,
            use_sample_attention: bool = False,
            missing_bias: bool = False,
            interaction_type: str = 'legacy',
            num_cross_layers: int = 2,
            low_rank: bool = False,
            rank_ratio: float = 0.5,
            gating_type: str = 'none',
            gating_k: int = 10,
            gating_temperature: float = 1.0,
            gating_hard: bool = True,
            gating_dropout: float = 0.0,
            gating_init_bias: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_sample_attention = use_sample_attention
        self.missing_bias = missing_bias
        self.interaction_type = interaction_type
        self.num_cross_layers = num_cross_layers
        self.low_rank = low_rank
        self.rank_ratio = rank_ratio
        self.gating_type = gating_type

        # Feature gating (differentiable feature selection)
        self.feature_gating = create_feature_gating(
            input_dim=input_dim,
            geting_type=gating_type,
            k=gating_k,
            temperature=gating_temperature,
            hard=gating_hard,
            dropout=gating_dropout,
            init_bias=gating_init_bias,
        )

        # Embedding layer (optional, could be linear projection)
        self.embed = nn.Linear(input_dim, hidden_dims[0]) if hidden_dims else nn.Identity()

        # feature attention
        self.feature_attention = FeatureAttention(
            hidden_dims[0] if hidden_dims else input_dim,
            num_heads=attention_heads,
            dropout=dropout,
            missing_bias=missing_bias,
        )
        
        # Sample attention (optional)
        if use_sample_attention:
            self.sample_attention = SampleAttention(
                hidden_dims[0] if hidden_dims else input_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
        else:
            self.sample_attention = None

        # Interaction layer (sparse cross-date features)
        interaction_input_dim = hidden_dims[0] if hidden_dims else input_dim
        self.interaction = create_interaction_layer(
            input_dim=interaction_input_dim,
            interaction_type=interaction_type,
            num_cross_layers=num_cross_layers,
            low_rank=low_rank,
            rank_ratio=rank_ratio,
            dropout=dropout
        )

        # Feed forward network 
        ff_layers = []
        dims = [hidden_dims[0]] + hidden_dims[1:] if hidden_dims else [input_dim]
        for i in range(len(dims) - 1):
            ff_layers.append(nn.Linear(dims[i], dims[i + 1]))
            ff_layers.append(nn.ReLU())
            ff_layers.append(nn.Dropout(dropout))
        self.ff = nn.Sequential(*ff_layers) if ff_layers else nn.Identity()

        # Outputs layer (to be defined by the pipeline)
        self.output_layer = nn.Identity()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        mask : torch.Tensor, optional
            Missingness mask of shape (batch_size, input_dim) where 1 indicated missing.
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        # Apply feature gating before embedding/attention
        if self.faeture_gating is not None:
            # The mask from the pipeline only covers numeric+generated features,
            # but x includes missing indicators and categorical encoded features.
            # Pad the mask with zeros (not missing) for the remaining features.
            gating_mask = mask
            if mask is not None and mask.shape[1] < x.shape[1]:
                padding = torch.zeros(
                    mask.shape[0], x.shape[1] - mask.shape[1],
                    device=mask.device, dtype=mask.dtype,
                )
                gating_mask = torch.cat([mask, padding], dim=1)
            x, gate_mask = self.feature_gating(x, mask=gating_mask)

        # Add sequence dimension for attention modules
        x = x.unsqueeze(1)  # (B, 1, 0)

        # Embedding
        x = self.embed(x)

        # Feature attention 
        x = self.feature_attention(x, mask)

        # Sample attention (optional)
        if self.use_sample_attention:
            x = self.sample_attention(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)

        # Interaction layer (sparce cross-date features)
        if self.interaction is not None:
            x = self.interaction

        # Feed forward
        x = self.ff(x)

        # Output (will be overridden by pipeline)
        x = self.output_layer(x)
        return x
    
    def set_output_layer(self, output_layer: nn.Module) -> None:
        """Set a custom output layer (e.g., linear layer for classification)."""
        self.output_layer = output_layer
