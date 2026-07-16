"""Regression pipeline."""

import numpy as np
from typing import Dict, Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .base import BaseNNPipeline


class RegressionPipeline(BaseNNPipeline):
    """Pipeline for regression tasks.

    Parameters
    ----------
    scale_target : bool, default=True
        Standardize target values to mean=0, std=1 during training.
        Predictions are automatically unscaled. Improves neural net
        convergence, especially for features with large numerical ranges.
    """

    def __init__(self, *args, scale_target: bool = True, **kwargs):
        self.scale_target = scale_target
        self._target_mean: Optional[torch.Tensor] = None
        self._target_std: Optional[torch.Tensor] = None
        super().__init__(*args, **kwargs)
        self._task_type = 'regression'
        self._n_classes = 2

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a DANet module with a single-output linear layer."""
        from .models.danet import DANetModule

        model = DANetModule(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            use_sample_attention=False,
            gating_type=self.gating_type,
            gating_k=self.gating_k,
            gating_temperature=self.gating_temperature,
            gating_hard=self.gating_hard,
            gating_dropout=self.gating_dropout,
            gating_init_bias=self.gating_init_bias,
            use_batch_norm=self.use_batch_norm,
            interaction_type=self.interaction_type,
            num_cross_layers=self.num_cross_layers,
        )

        # Output layer: single continious value
        model.set_output_layer(nn.Linear(self.hidden_dims[-1] if self.hidden_dims else input_dim, 1))
        return model
    
    def _get_loss_fn(self) -> nn.Module:
        """Huber loss (more robust to outliers than MSE)."""
        return nn.HuberLoss(delta=1.0)
    
    def _get_metrics(self) -> Dict[str, Callable]:
        """Default metris for regression."""
        return {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score
        }
    
    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert target column to float tensor, optionally standardize."""
        target = super()._prepare_target(df)
        target = target.view(-1, 1)

        if self.scale_target:
            self._target_mean = target.mean()
            self._target_std = target.std() + 1e-8
            target = (target - self._target_mean) / self._target_std

        return target
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate regression predictions (unscaled if target scaling is enabled)."""
        preds = super().predict(df)
        if self.scale_target and self._target_mean is not None:
            mean = self._target_mean.cpu().numpy()
            std = self._target_std.cpu().numpy()
            return preds * std + mean
        return preds

    def _val_metric_name(self) -> str:
        return "R²"

    def _compute_val_metric(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        yt = y_true.cpu().numpy().ravel()
        yp = y_pred.cpu().numpy().ravel()
        if self.scale_target and self._target_mean is not None:
            mean = self._target_mean.cpu().numpy()
            std = self._target_std.cpu().numpy()
            yt = yt * std + mean
            yp = yp * std + mean
        try:
            return float(r2_score(yt, yp))
        except ValueError:
            return 0.0
