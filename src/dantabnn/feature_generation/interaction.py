"""Selective interaction generator for pairwise feature products."""

from itertools import combinations
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from .base import BaseDANetFeatureGenerator


class SelectiveInteractionGenerator(BaseDANetFeatureGenerator):
    """Generative pairwise interaction features with mutual-information based selection.

    Parameters
    ----------
    numeric_columns : Optional[List[str]]
        Numeric columns to consider for interactions. If None, all numeric
        columns in the fitted dataframe are used.
    mi_threshold : float, default=1.2
        Minimum ratio of interaction MI to individual feature MI required to keep
        the interaction. If 'y' is not provided during fit, this threshold is ignored
        and all pairwise products are kept.
    correlation_threshold : float, default=0.98
        Maximum absolute Pearson correlation allowed between an interaction and any
        individual feature (or another interaction). Features with higher correlation
        are considered redundant and removed.
    max_interactions: int, default=100
        Maximum number of interaction features to keep (top by MI ratio).
    """

    def __init__(
            self,
            numeric_columns: Optional[List[str]] = None,
            mi_threshold: float = 1.2,
            correlation_threshold: float = 0.98,
            max_interactions: int = 100,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.numeric_columns = numeric_columns
        self.mi_threshold = mi_threshold
        self.correlation_threshold = correlation_threshold
        self.max_interactions = max_interactions

        # Internal state
        self._original_columns: List[str] = []
        self._interaction_pairs: List[Tuple] = []
        self._feature_names: List[str] = []
        self._mi_ratios: Dict[tuple, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseDANetFeatureGenerator":
        self._log_info("Fitting selective interaction features")
        # Determine numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.numeric_columns is not None:
            self._original_columns = [str(c) for c in self.numeric_columns if c in numeric_cols]
            missing = set(self.numeric_columns) - set(numeric_cols)
            if missing:
                self._log_warning(
                    f"Requested numeric columns {missing} are not numeric or missing: they will be ignored."
                )
        else:
            self._original_columns = numeric_cols

        if len(self._original_columns) < 2:
            self._log_warning("Fewer than two numeric columns; no interactions can be created.")
            self._feature_names = []
            self.is_fitted = True
            return self

        # Impute missing values in selected numeric columns
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=True)

        # Generate all unordered pairs
        pairs = list(combinations(self._original_columns, 2))
        self._log_info(f"Evaluating {len(pairs)} candidate interaction pairs")

        # Compute mutual information ratios if target is provided
        if y is not None:
            self._compute_mi_ratios(X_imputed, y, pairs)
            # Filter by MI threshold
            filtered_pairs = [
                p for p in pairs
                if self._mi_ratios.get(p, 0.0) >= self.mi_threshold
            ]
            self._log_info(f"{len(filtered_pairs)} pairs pass MI threshold {self.mi_threshold}")
        else:
            filtered_pairs = pairs
            self._log_info("No target provided; keep all pairs (MI threshold ignored)")

        # Further filter by correlation redundancy
        selected_pairs = self._filter_by_correlation(X_imputed, filtered_pairs)
        # Limit max_interactions
        if len(selected_pairs) > self.max_interactions:
            # Sort by MI ratio descending (or ability if MI not computed)
            if y is not None:
                selected_pairs.sort(key=lambda p: self._mi_ratios.get(p, 0.0), reverse=True)
            selected_pairs = selected_pairs[:self.max_interactions]
            self._log_info(f"Limited to top {self.max_interactions} interactions")

        self._interaction_pairs = selected_pairs
        self._feature_names = [f"{a}_x_{b}" for a, b in selected_pairs]
        self._log_info(f"Selected {len(self._feature_names)} interaction features")
        self.is_fitted = True
        return self

    def _compute_mi_ratios(self, X: pd.DataFrame, y: pd.Series, pairs: List[tuple]):
        """Compute mutual information of individual features and interactions with target."""
        y_vals = y.values if isinstance(y, pd.Series) else y
        is_classification = (len(np.unique(y_vals)) < 10)  # heuristic
        mi_func = mutual_info_classif if is_classification else mutual_info_regression

        # Compute MI for individual features
        individual_mi = {}
        for col in self._original_columns:
            mi = mi_func(X[[col]], y_vals).item()
            individual_mi[col] = mi

        # Compute MI for interaction products
        for a, b in pairs:
            interaction = X[a] * X[b]
            mi_iter = mi_func(interaction.values.reshape(-1, 1), y_vals).item()
            mi_individual = max(individual_mi[a], individual_mi[b])
            ratio = mi_iter / mi_individual if mi_individual > 0 else 0.0
            self._mi_ratios[(a, b)] = ratio

    def _filter_by_correlation(self, X: pd.DataFrame, pairs: List[tuple]) -> List[tuple]:
        """Remove pairs that are highly correlated with existing features."""
        if len(pairs) == 0:
            return []
        # Compute correlation matrix of all original columns plus candidates interactions
        corr_matrix = X[self._original_columns].corr().abs()
        selected = []
        for a, b in pairs:
            interaction = X[a] * X[b]
            # Correlation with each original column
            corr_with_original = max(
                abs(interaction.corr(X[col])) for col in self._original_columns
            )
            # Correlation with already selected interactions
            corr_with_selected = 0.0
            for sa, sb in selected:
                selected_interaction = X[sa] * X[sb]
                corr = abs(interaction.corr(selected_interaction))
                if corr > corr_with_selected:
                    corr_with_selected = corr
            max_corr = max(corr_with_original, corr_with_selected)
            if max_corr < self.correlation_threshold:
                selected.append((a, b))
            else:
                self._log_debug(f"Interaction {a}_x_{b} removed due to high correlation ({max_corr:.3f})")
        return selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform.")
        if len(self._interaction_pairs) == 0:
            return pd.DataFrame(index=X.index)

        missing = set(self._original_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns required for transformation: {missing}")

        # Impute missing values using stored means
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=False)

        data = {}
        for a, b in self._interaction_pairs:
            data[f"{a}_x_{b}"] = X_imputed[a] * X_imputed[b]
        return pd.DataFrame(data, index=X.index)

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """Interaction features are numeric; check count limit."""
        if len(self._feature_names) > 500:
            self._log_warning(
                f"Number of interaction features ({len(self._feature_names)}) exceeds DANet limit of 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "selective_interaction",
            "original_columns": self._original_columns.copy(),
            "interaction_paris": self._interaction_pairs.copy(),
            "mi_threshold": self.mi_threshold,
            "correlation_threshold": self.correlation_threshold,
            "max_interaction": self.max_interactions,
        })
        return metadata
