"""High-cardinality categorical embedder using categorical encoding."""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .base import BaseDANetFeatureGenerator


class HighCardinalityEmbedder(BaseDANetFeatureGenerator):
    """Convert high-cardinality categorical columns into numeric embeddings.

    This generator performs target encoding (mean of the target per category)
    with optional smoothing. Only columns whose number of unique values exceeds
    'cardinality_threshold' are processed.

    Parameters
    ----------
    categorical_columns : Optional[List[str]]
        Categorical columns to consider. If None, all non-numeric columns in the
        fitted dataframe are considered.
    cardinality_threshold : int, default=100
        Minimum number of unique values required to treat a column as high-cardinality.
    smoothing : float, default=10.0
        Smoothing factor for target encoding (higher = more smoothing toward global mean),
    unknown_value : float, default=0.0
        Value to use for categories not seen during fit.
    """

    def __init__(
            self,
            categorical_columns: Optional[List[str]] = None,
            cardinality_threshold: int = 100,
            smoothing: float = 10.0,
            unknown_value: float = 0.0,
            name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.categorical_columns = categorical_columns
        self.cardinality_threshold = cardinality_threshold
        self.smoothing = smoothing
        self.unknown_value = unknown_value

        # Internal state
        self._high_cardinality_columns: List[str] = []
        self._encoding_maps: Dict[str, Dict[str, float]] = {}
        self._global_means: Dict[str, float] = {}
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HighCardinalityEmbedder":
        self._log_info("Fitting high-cardinality embedder")
        if y is None:
            self._log_warning(
                "Target y not provided high-cardinality embedding requires target encoding. "
                "Generator will produce no features."
            )
            self._feature_names = []
            self.is_fitted = True
            return self

        # Determine categorical columns
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if self.categorical_columns is not None:
            self._high_cardinality_columns = [
                str(c) for c in self.categorical_columns if c in cat_cols
            ]
            missing = set(self.categorical_columns) - set(cat_cols)
            if missing:
                self._log_warning(
                    f"Requested categorical columns {missing} are numeric of missing; they will be ignored."
                )
        else:
            self._high_cardinality_columns = cat_cols

        # Filter by cardinality threshold
        high_card = []
        for col in self._high_cardinality_columns:
            n_unique = X[col].nunique()
            if n_unique >= self.cardinality_threshold:
                high_card.append(col)
                self._log_info(f"Column '{col}' has {n_unique} values (>{self.cardinality_threshold})")
            else:
                self._log_debug(f"Column '{col}' skipped (cardinality {n_unique} < {self.cardinality_threshold})")
        self._high_cardinality_columns = high_card

        if len(self._high_cardinality_columns) == 0:
            self._log_info("No high-cardinality columns meeting threshold.")
            self._feature_names = []
            self.is_fitted = True
            return self

        # Compute target encoding with smoothing
        y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
        for col in self._high_cardinality_columns:
            # Group means
            group_means = y_series.groupby(X[col]).mean()
            # Global means
            global_mean = y_series.mean()
            self._global_means[col] = global_mean
            # Smoothing
            group_counts = y_series.groupby(X[col]).count()
            smoothed = (group_means * group_counts + global_mean * self.smoothing) / (group_counts + self.smoothing)
            self._encoding_maps[col] = smoothed.to_dict()
            self._log_debug(f"Encoded '{col}' with {len(self._feature_names)} columns")

        self._feature_names = [f"embed_{col}" for col in self._high_cardinality_columns]
        self._log_info(f"Prepared embeddings for {len(self._feature_names)} columns")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform.")
        if len(self._high_cardinality_columns) == 0:
            return pd.DataFrame(index=X.index)

        missing = set(self._high_cardinality_columns) - set(X.columns)
        if missing:
            raise ValueError(f"missing columns required for transformation: {missing}")

        data = {}
        for col in self._high_cardinality_columns:
            encoding_map = self._encoding_maps[col]
            unknown = self.unknown_value
            # Map each category to its encoded value, default to unknown
            encoded = X[col].map(encoding_map).fillna(unknown).astype(np.float32)
            data[f"embed_{col}"] = encoded
        return pd.DataFrame(data, index=X.index)

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """Embeddings are numeric; check count limit."""
        if len(self._feature_names) > 500:
            self._log_warning(
                f"Number of embeddings features ({len(self._feature_names)}) exceeds DANet limit of 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "high_cardinality_embedding",
            "high_cardinality_columns": self._high_cardinality_columns.copy(),
            "cardinality_threshold": self.cardinality_threshold,
            "smoothing": self.smoothing,
            "unknown_value": self.unknown_value,
            "global_means": self._global_means.copy(),
        })
        return metadata
