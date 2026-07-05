"""Binary classification pipeline."""

from typing import Dict, Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, fbeta_score

from .base import BaseNNPipeline


class BinaryClassificationPipeline(BaseNNPipeline):
    """Pipeline for binary classification tasks.

    Parameters
    ----------
    pos_weight : float, optional
        Weight for positive class in BCEWithLogitsLoss. Use for
        imbalanced datasets (e.g., pos_weight=9.0 for 90/10 split).
    threshold_tuning : bool, default=True
        If True, sweep thresholds on validation set after fit and
        store the optimal threshold (by F2-score).
    """

    def __init__(self, *args, pos_weight: Optional[float] = None,
                 threshold_tuning: bool = True, **kwargs):
        self.pos_weight = pos_weight
        self.threshold_tuning = threshold_tuning
        self._optimal_threshold: float = 0.5
        super().__init__(*args, **kwargs)

    def _build_model(self, input_dim: int, output_dim: int):
        """Build a Danet module with a single-output linear layer."""

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

        # Output layer: single logit, init bias for class imbalance
        output_dim = self.hidden_dims[-1] if self.hidden_dims else input_dim
        output_layer = nn.Linear(output_dim, 1)
        # Initialize bias to log(prior) for imbalanced datasets to prevent BCE explosion
        if self.pos_weight is not None and self.pos_weight > 1.0:
            with torch.no_grad():
                output_layer.bias.data.fill_(float(np.log(1.0 / self.pos_weight)))
        model.set_output_layer(output_layer)
        return model
    
    def _get_loss_fn(self) -> nn.Module:
        """Binary cross-entropy loss with optional class weighting.

        pos_weight > 3.0 causes BCE gradient explosion on CPU due to
        unbounded attention logits multiplied by large class weight.
        Cap at 3.0 to ensure stability across all imbalance ratios.
        """
        if self.pos_weight is not None:
            capped = min(float(self.pos_weight), 3.0)
            if capped < self.pos_weight:
                from .utils.logger import setup_logger
                _lgr = setup_logger(__name__)
                _lgr.warning(f"Capping pos_weight {self.pos_weight:.1f} → {capped:.1f} to prevent BCE explosion")
            weight = torch.tensor([capped], device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        return nn.BCEWithLogitsLoss()
    
    def _get_metrics(self) -> Dict[str, Callable]:
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
    
    def predict_classes(self, df, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary class predictions using optimal threshold if available."""
        if threshold is None:
            threshold = getattr(self, '_optimal_threshold', 0.5)
        probs = self.predict(df)
        return (probs > threshold).astype(int)

    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None,
            verbose: int = 1,
    ) -> "BinaryClassificationPipeline":
        """Fit and optionally tune optimal classification threshold on validation data."""
        super().fit(df_train, df_val=df_val, verbose=verbose)
        if self.threshold_tuning and df_val is not None:
            self._tune_threshold(df_val)
        return self

    def _tune_threshold(self, df_val: pd.DataFrame) -> None:
        """Find optimal decision threshold optimizing F2-score on validation data."""
        y_val = df_val[self.target_column].values.astype(float)
        probs = self.predict(df_val).ravel()

        # Skip if only one class present (can't compute F-score)
        if len(np.unique(y_val)) < 2:
            return

        thresholds = np.linspace(0.05, 0.95, 91)
        best_t, best_score = 0.5, -1
        for t in thresholds:
            preds = (probs > t).astype(int)
            try:
                f2 = fbeta_score(y_val, preds, beta=2.0)
                if f2 > best_score:
                    best_score, best_t = f2, t
            except Exception:
                continue

        if best_score > -1:
            self._optimal_threshold = float(best_t)
            from .utils.logger import setup_logger
            logger = setup_logger(__name__)
            logger.info(
                f"Optimal threshold: {best_t:.3f} (F2={best_score:.4f}, "
                f"default=0.500)"
            )

    @property
    def optimal_threshold(self) -> float:
        """Return the fitted optimal decision threshold."""
        return getattr(self, '_optimal_threshold', 0.5)

    def _val_metric_name(self) -> str:
        return "ROC-AUC"

    def _compute_val_metric(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        yt = y_true.cpu().numpy().ravel()
        yp = torch.sigmoid(y_pred).cpu().numpy().ravel()
        try:
            return float(roc_auc_score(yt, yp))
        except ValueError:
            return 0.0
