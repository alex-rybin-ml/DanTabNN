"""Multiclass classification pipeline."""

from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .base import BaseNNPipeline


class MulticlassClassificationPipeline(BaseNNPipeline):
    """Pipline for multiclass classification tasks."""

    def __init__(
            self,
            numeric_features: List[str],
            categorical_features: List[str],
            target_column: str,
            n_classes: int,
            **kwargs,
    ):
        """
        Parameters
        ----------
        n_classes : int 
            Numbers of target classes.
        """
        self.n_classes = n_classes
        super().__init__(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_column=target_column,
            **kwargs,
        )

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a Danet module with a multi-output linear layer."""
        from .models.danet import DANetModule

        model = DANetModule(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            use_sample_attention=False
        )

        # Output layer: logits for each class
        model.set_output_layer(
            nn.Linear(self.hidden_dims[-1] if self.hidden_dims else input_dim, self.n_classes)
        )
        return model

    def _get_loss_fn(self) -> nn.Module:
        """Cross-engtropy loss."""
        return nn.CrossEntropyLoss()
    
    def _get_metrics(self) -> Dict[str, callable]:
        """Default metrics for multiclass classification."""
        return {
            "accuracy": accuracy_score,
            "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        }

    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert target column to log integer tensor (class indices)."""
        target = df[self.target_column].values

        # Assume target is integer-encoded (0, 1, ..., n_classes-1)
        return torch.LongTensor(target).to(self.device)

    def _get_output_dim(self, y: torch.Tensor) -> int:
        """Return number of classes."""
        return self.n_classes
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities."""
        logits = super().predict(df)  # shape (n_samples, n_classes)
        return torch.softmax(torch.FloatTensor(logits), dim=1).numpy()
        
    def predict_classes(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels."""
        probs = self.predict(df)
        return np.argmax(probs, axis=1)       
