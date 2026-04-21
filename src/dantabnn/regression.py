"""Regression pipeline."""

import torch
import torch.nn as nn

from base import BaseNNPipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict
import pandas as pd 

class RegressionPipeline(BaseNNPipeline):
    """Pipeline for regression tasks."""

    def _build_model(self, input_dim: int, output_dim: int ) -> nn.Module:
        """Build a DANet module with a single-output linear layer."""
        from models.danet import DANetsModule

        model = DANetsModule(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            use_sample_attention=False
        )

        # Output layer: single continious value
        model.set_output_layer(nn.Linear(self.hidden_dims[-1] if self.hidden_dims else input_dim, 1))
        return model
    
    def _get_loss_fn(self) -> nn.Module:
        """Mean squared error loss."""
        return nn.MSELoss()
    
    def _get_metrics(self) -> Dict[str, callable]:
        """Default metris for regression."""
        return {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score
        }
    
    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert target column to float tensor."""
        target = super()._prepare_target(df)

        # Ensure target is float and shape (n_sample, 1)
        target = target.viwe(-1, 1)
        return target
