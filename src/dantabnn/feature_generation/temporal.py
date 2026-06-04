"""Temporal aggregation generator for group-level time-series statistics."""
import numpy as np
import pandas as pd
from .base import BaseDANetFeatureGenerator
from typing import Optional, List, Dict, Any
from ..utils.hardware import get_optimal_backend


class TemporalAggregationGenerator(BaseDANetFeatureGenerator):
    """Generate temporal aggregations (rolling/expanding windows) within groups.

    Supports multiple backends for CPU/GPU acceleration: pandas (default),
    cudf (GPU DataFrame), conv (PyTorch 1D convolutions), numba (JIT-compiled loops).

    Parameters
    ----------
    date_column : str
        Name of the column containing timestamps (must be sortable).
    groupby_columns : List[str]
        Columns that define groups (e.g., ['store_id', 'product_id']).
    numeric_columns : Optional[List[str]]
        Numeric columns to aggregate. If None, all numeric columns except the
        date column and group-by columns are used.
    windows : List[int], default=[7, 30]
        Rolling window sizes (in days). If the date column is not daily, the
        window is interpreted in number of rows after sorting by date.
    aggregations : List[str], default=['mean', 'std', 'min', 'max']
        Functions to compute for each window. Supported: 'mean', 'std', 'min',
        'max', 'sum', 'median', 'count'.
    expanding : bool, default=True
        If True, also compute expanding-window statistics (from the first
        observation up to the current row).
    backend : str, default='auto'
        Computation backend. Allowed values: 'auto', 'pandas', 'cudf', 'conv', 'numba'.
        'auto' selects the optimal backend based on hardware detection (CUDA availability),
        cuDF installation, etc.). 'pandas' uses pandas rolling (CPU). 'cudf' uses cuDF
        rolling (GPU). 'conv' uses PyTorch 1D convolutions (GPU/CPU). 'numba' uses
        Numba-accelerated loops (CPU).
    """

    def __init__(
            self,
            date_column: str,
            groupby_columns: List[str],
            numeric_columns: Optional[List[str]] = None,
            windows: List[int] = (7, 30),
            aggregations: List[str] = ("mean", "std", "min", "max"),
            expanding: bool = True,
            backend: str = "auto",
            name: str = None,
    ):
        super().__init__(name=name)
        self.date_column = date_column
        self.groupby_columns = groupby_columns
        self.numeric_columns = numeric_columns
        self.windows = windows
        self.aggregations = list(aggregations)
        self.expanding = expanding
        self.backend = backend

        # Internal state
        self._feature_names: List[str] = []
        self._all_numeric_columns: List[str] = []
        self._supported_aggs = {"mean", "std", "min", "max", "sum", "median", "count"}

    def _get_effective_backend(self) -> str:
        """Resolve 'auto' backend to concrete backend based on hardware."""
        if self.backend != "auto":
            return self.backend
        # Use hardware detection to pick optimal backend
        return get_optimal_backend()

    @property
    def supports_jit(self) -> bool:
        """Whether this generator supports JIT generation tensors."""
        return self._get_effective_backend() == "conv"

    def jit_transform(self, *args, **kwargs):
        """Generate features from raw tensors (JIT mode)."""
        raise NotImplementedError("JIT transform for temporal aggregation in not yet implemented.")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TemporalAggregationGenerator":
        self._log_info("Fitting temporal aggregations")
        # Validate columns exist
        missing = set([self.date_column] + self.groupby_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Columns not found in dataframe: {missing}")

        # Determine numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude date column (if numeric) and group-by columns (if numeric)
        exclude = set(self.groupby_columns + [self.date_column])
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if self.numeric_columns is not None:
            # Keep only requested columns that are numeric
            self._all_numeric_columns = [c for c in self.numeric_columns if c in numeric_cols]
            missing = set(self.numeric_columns) - set(numeric_cols)
            if missing:
                self._log_warning(
                    f"Requested numeric columns {missing} are not numeric or missing; they will be ingored."
                )
        else:
            self._all_numeric_columns = numeric_cols

        if len(self._all_numeric_columns) == 0:
            self._log_warning("No numeric columns available for temporal aggregation.")

        # Build feature names (proactively)
        self._build_feature_names()
        self.is_fitted = True
        return self

    def _build_feature_names(self):
        """Construct descriptive feature names for all generated aggregations."""
        feature_names = []
        for col in self._all_numeric_columns:
            for agg in self.aggregations:
                for w in self.windows:
                    feature_names.append(f"temp_{col}_{agg}_w{w}")
            if self.expanding:
                for agg in self.aggregations:
                    feature_names.append(f"temp_{col}_{agg}_expanding")
        self._feature_names = feature_names

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform.")
        if len(self._all_numeric_columns) == 0:
            return pd.DataFrame

        # Ensure required columns are present
        required = set([self.date_column] + self.groupby_columns + self._all_numeric_columns)
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns required for transformation: {missing}")

        backend = self._get_effective_backend()
        if backend == "pandas":
            return self._transform_pandas(X)
        elif backend == "cudf":
            return self._transform_cudf(X)
        elif backend == "conv":
            return self._transform_conv(X)
        elif backend == "numba":
            return self._transform_numba(X)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _transform_pandas(self, X: pd.DataFrame) -> pd.DataFrame:
        # Sort by group and date (important for rolling windows)
        X_sorted = X.sort_values(by=self.groupby_columns + [self.date_column]).copy()
        results = []
        for _, group in X_sorted.groupby(self.groupby_columns + [self.date_column]).copy():
            for col in self._all_numeric_columns:
                series = group[col]
                # Rolling windows
                for w in self.windows:
                    rolling = series.rolling(window=w, min_periods=1)
                    for agg in self.aggregations:
                        if agg in self._supported_aggs:
                            agg_series = getattr(rolling, agg)()
                        else:
                            continue
                        group[f"temp_{col}_{agg}_w{w}"] = agg_series.values
                # Expanding window
                if self.expanding:
                    expanding = series.expanding(min_period=1)
                    for agg in self.aggregations:
                        if agg in self._supported_aggs:
                            agg_series = getattr(expanding, agg)()
                        else:
                            continue
                        group[f"temp_{col}_{agg}_expanding"] = agg_series
            results.append(group)

        # Combine groups back (preserving original order)
        transformed = pd.concat(results, axis=0).loc[X.index]  # re-index to original order
        # Keep only the generated columns
        generated_cols = [c for c in transformed.columns if c.startswith("temp_")]
        return transformed[generated_cols]

    def _transform_cudf(self, X: pd.DataFrame) -> pd.DataFrame:
        # Try to use cuDF for GPU acceleration; fall back to pandas with warning
        try:
            import cudf
        except ImportError:
            self._log_warning(
                "cuDF not installed. Falling back to pandas backend for temporal aggregation."
            )
            return self._transform_pandas(X)

        # if cuDF is available, convert DataFrame to cudf and compute rolling per group
        # For simplicity, we currently fall back to pandas; TODO implement cudf-native rolling
        self._log_warning(
            "cuDF backend is not yet fully implemented. Falling back to pandas."
        )
        return self._transform_pandas(X)

    def _transform_conv(self, X: pd.DataFrame) -> pd.DataFrame:
        # PyTorch 1D convolution backend (GPU/CPU)
        self._log_warning(
            "Conv backend (Pytorch) is not yet implemented. Falling back to pandas."
        )
        return self._transform_pandas(X)

    def _transform_numba(self, X: pd.DataFrame) -> pd.DataFrame:
        # Numba-accelerated loops backend
        self._log_warning(
            "Numba backend is not yet implemented. Falling back to pandas."
        )
        return self._transform_pandas(X)

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """Temporal aggregations produce numeric features; check count limit."""
        if len(self._feature_names) > 500:
            self._log_warning(
                f"Number of temporal features ({len(self._feature_names)}) exceeds DANet limit of 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "temporal_aggregation",
            "date_column": self.date_column,
            "groupby_columns": self.groupby_columns,
            "numeric_columns": self._all_numeric_columns.copy(),
            "windows": self.windows,
            "aggregations": self.aggregations,
            "expanding": self.expanding,
            "backend": self.backend,
            "effective_backend": self._get_effective_backend(),
        })
        return metadata

