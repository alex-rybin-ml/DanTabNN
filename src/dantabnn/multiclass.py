"""Multiclass classification pipeline."""

from typing import Dict, Callable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .base import BaseNNPipeline


class MulticlassClassificationPipeline(BaseNNPipeline):
    """Pipeline for multiclass classification tasks with automatic class weighting.

    Parameters
    ----------
    n_classes : int
        Number of target classes.
    class_weights : list of float, optional
        Per-class weights for CrossEntropyLoss. If None, auto-computes
        inverse frequency weights from training data (clamped to [0.1, 10]).
    """

    def __init__(
            self,
            numeric_features: List[str],
            categorical_features: List[str],
            target_column: str,
            n_classes: int,
            class_weights: Optional[List[float]] = None,
            **kwargs,
    ):
        self.n_classes = n_classes
        self.class_weights = class_weights
        super().__init__(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_column=target_column,
            **kwargs,
        )

    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build a DANet module with a multi-output linear layer."""
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

        # Output layer: logits for each class
        model.set_output_layer(
            nn.Linear(self.hidden_dims[-1] if self.hidden_dims else input_dim, self.n_classes)
        )
        return model

    def _get_loss_fn(self) -> nn.Module:
        """Multiclass loss function.
        
        Uses focal loss for imbalanced datasets (class_weights with
        max/min ratio > 3.0), weighted cross-entropy otherwise.
        """
        if self.class_weights is not None:
            w = np.asarray(self.class_weights, dtype=np.float64)
            if w.max() / max(w.min(), 1e-8) > 3.0:
                from .utils.losses import MultiClassFocalLoss
                return MultiClassFocalLoss(gamma=2.0)
            weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        return nn.CrossEntropyLoss()

    def _get_metrics(self) -> Dict[str, Callable]:
        """Default metrics for multiclass classification."""
        return {
            "accuracy": accuracy_score,
            "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        }

    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Convert target column to long integer tensor (class indices)."""
        target = df[self.target_column].values.astype(np.int64)
        return torch.LongTensor(target).to(self.device)

    def _get_output_dim(self, y: torch.Tensor) -> int:
        """Return number of classes."""
        return self.n_classes

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class probabilities."""
        logits = super().predict(df)
        return torch.softmax(torch.FloatTensor(logits), dim=1).numpy()

    def predict_classes(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels."""
        probs = self.predict(df)
        return np.argmax(probs, axis=1)

    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None,
            verbose: int = 1,
    ) -> "MulticlassClassificationPipeline":
        """Fit the pipeline. Auto-computes class_weights if not explicitly set.

        Inverse frequency weights are computed as:
            w_c = total_samples / (n_classes * count_c)
        and clamped to [0.1, 10.0] to prevent any single class from
        dominating the gradient.
        """
        if self.class_weights is None:
            targets = np.asarray(df_train[self.target_column].values, dtype=np.int64)
            class_counts = np.bincount(targets, minlength=self.n_classes)
            # Effective number of samples (Cui et al. 2019): naturally saturating
            # weights without arbitrary caps.
            beta = 1.0 - 1.0 / float(targets.size)
            effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
            weights = 1.0 / effective_num
            # Normalize to mean=1 so learning rate stays stable
            weights = weights / weights.mean()
            weights = np.clip(weights, 0.1, 100.0)  # safety clamp, much wider than old 10.0
            self.class_weights = weights.tolist()
            from .utils.logger import setup_logger
            logger = setup_logger(__name__)
            weights_str = {str(c): f"{w:.2f}" for c, w in enumerate(self.class_weights)}
            logger.info(
                f"Auto-computed class weights: {weights_str}"
            )
        return super().fit(df_train, df_val=df_val, verbose=verbose)

    def _val_metric_name(self) -> str:
        return "F1-macro"

    def _compute_val_metric(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        yt = y_true.cpu().numpy().ravel()
        yp = y_pred.argmax(dim=1).cpu().numpy().ravel()
        try:
            return float(f1_score(yt, yp, average="macro"))
        except ValueError:
            return 0.0
