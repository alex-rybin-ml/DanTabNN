"""Domain-aware feature generation using polynomial expansions."""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseDANetFeatureGenerator


class DomainFeatureGenerator(BaseDANetFeatureGenerator):
    """Generate domain-inspired features via polynomial transformations.

    This generator creates polynomials features of numeric columns up to a given degree,
    optionally including interaction terms. It is a placeholder for more sophisticated
    domain-logic templates.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced (no powers of a single feature).
    include_bias : bool, default=False
        If True, include a bias column (all polynomial powers are zero).
    numeric_columns : Optional[List[str]], default=None
        Subset of numeric columns to transform. If None, all numeric columns in the
        fitted dataframe are used.
    """

    def __init__(
            self,
            degree: int = 2,
            interaction_only: bool = False,
            include_bias: bool = False,
            numeric_columns: Optional[List[str]] = None,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.numeric_columns = numeric_columns
        self._poly = None
        self._original_columns: List[str] = []
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureGenerator":
        self._log_info(f"Fitting polynomial features (degree={self.degree})")
        # Determine numeric columns to transform
        numeric_cols = X.select_dtypes(include=[np.number]).oolumns.tolist()
        if self.numeric_columns is not None:
            # Validate that requested columns are present and numeric
            missing = set(self.numeric_columns) - set(numeric_cols)
            if missing:
                self._log_warning(
                    f"Columns (missing) are not numeric or missing: they will be ignored."
                )
            self._original_columns = [c for c in self.numeric_columns if c in numeric_cols]
        else:
            self._original_columns = numeric_cols

        if len(self._original_columns) == 0:
            self._log_warning("No numeric columns available for polynomial features.")
            self._poly = None
            self._feature_names = []
            self.is_fitted = True
            return self

        # Impute missing values before polynomial expansion
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=True)

        # Fit sklearn's values PolynomialFeatures
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self._poly.fit(X_imputed)

        # Build feature names
        self._feature_names = self._poly.get_feature_names_out(self._original_columns)
        self._log_info(f"Generated {len(self._feature_names)} polynomial features")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform")
        if self._poly is None or len(self._original_columns) == 0:
            # No features generated, return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

        # Ensure all required columns are present
        missing = set(self._original_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns required for transformation: {missing}")

        # Impute missing values using stored means
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=False)
        transformed = self._poly.transform(X_imputed)
        # Convert to DataFrame with appropriate column names
        df = pd.DataFrame(
            transformed,
            columns=self._feature_names,
            index=X.index,
        )
        return df

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """Polynomial features are numeric and should be DANet-compatible."""
        # Check that al generated features are numeric (they are by construction)
        # Also ensure total number of features does not exceed DANet limit (500)

        if len(self._feature_names) > 500:
            self._log_warning(
                f"Number of polynomial features ({len(self._feature_names)}) exceed DANet limit of 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "polynomial",
            "degree": self.degree,
            "interaction_only": self.interaction_only,
            "include_bias": self.include_bias,
            "original_columns": self._original_columns.copy(),
        })
        return metadata
