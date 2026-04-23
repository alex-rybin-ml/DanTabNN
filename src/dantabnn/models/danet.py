import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


class FeatureAttention(nn.Module):
    """Self attention across feature dimensions."""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.proj = nn.Linar(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch_size, seq_len, input_dim) where seq_len = 1 for tabular."""
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # each shape (B, num_heads, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * (self.num_heads ** -0.5)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = attn @ v # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.layer_norm(out + x) # resudual connection
        return out


class SampleAttention(nn.Module):
    """Attention across samples (optional). Not used in standard tabular setting."""

    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"

        self.qkv = nn.Linear(input_dim, 3 * input_dim)
        self.proj = nn.Linar(input_dim, input_dim)
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
            use_sample_attention: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_sample_attention = use_sample_attention

        # Embedding layer (optional, could be linear projection)
        self.embed = nn.Linear(input_dim, hidden_dims[0]) if hidden_dims else nn.Identity()

        # feature attention
        self.feature_attention = FeatureAttention(
            hidden_dims[0] if hidden_dims else input_dim,
            num_heads=attention_heads,
            dropout=dropout,
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

        # Feed forward network 
        ff_layers = []
        dims = [hidden_dims[0]] + hidden_dims[1:] if hidden_dims else [input_dim]
        for i in range(len(dims) - 1):
            ff_layers.append(nn.Linear(dims[i], dims[i + 1]))
            ff_layers.append(nn.ReLU())
            ff_layers.append(nn.Dropout(dropout))
        self.ff = nn.Sequential(*ff_layers) if ff_layers else nn.Identity()

        # Ouputs layer (to be defined by the pipeline)
        self.output_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        # Add sequence dimension for attention modules
        x = x.unsqueeze(1) # (B, 1, 0)

        # Embedding
        x = self.embed(x)

        # Feature attention 
        x = self.feature_attention(x)

        # Sample attention (optional)
        if self.use_sample_attention:
            x = self.sample_attention(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)

        # Feed forward
        x = self.ff(x)

        # Ouput (will be overridden by pipeline)
        x = self.output_layer(x)
        return x
    
    def set_output_layer(self, output_layer: nn.Module) -> None:
        """Set a custom output layer (e.g., linear layer for classification)."""
        self.output_layer = output_layer
