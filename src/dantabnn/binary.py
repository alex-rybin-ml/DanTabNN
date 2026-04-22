"""Binary classification pipeline."""

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score

from .base import BaseNNPipeline


class BinaryClassificationPipeline(BaseNNPipeline):
    """Pipeline for binary classification tasks."""

    def _build_model(self, input_dim: int, output_dim: int):
        """Build a Danet module with a single-output linear layer."""

        from models.danet import DANetModule

        model = DANetModule(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            use_sample_attention=False,
        )

        # Output layer: single logit
        model.set_output_layer(nn.Linear(self.hidden_dims[-1] if self.hidden_dims else input_dim, 1))
        return model
    
    def _get_loss_fn(self) -> nn.Module:
        """Binary cross-entropy loss with logits"""
        return nn.BCEWithLogitsLoss()
    
    def _get_metrics(self) -> Dict[str, callable]:
        """Default metrics for binary classification."""
        return {
            "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred > 0.5),
            "roc_auc": roc_auc_score
        }
    
    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert target columns to float tensor."""
        target = super()._prepare_target(df)

        # Ensure target is float adn shape (n_samples, 1)
        target = target.view(-1, 1)
        return target
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities."""
        logits = super().predict(df)
        return torch.sigmoid(torch.FloatTensor(logits)).numpy()
    
    def predict_classes(self, df, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""
        probs = self.predict(df)
        return (probs > threshold).astype(int)
