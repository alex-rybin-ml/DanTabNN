"""Missing value imputer for numeric and categorical features.

Numeric columns: median imputation.
Categorical columns: most-frequent (mode) imputation.
Operates on numpy arrays, fit on training data, transform on test data.
"""

import numpy as np
from typing import Optional, Dict
from collections import Counter


class NaNImputer:
    """Impute missing values using median (numeric) or mode (categorical).

    Parameters
    ----------
    categorical_mask : np.ndarray of bool, optional
        Boolean mask of shape (n_features,) indicating which columns
        are categorical (True) vs numeric (False). If None, all
        columns are treated as numeric.
    """

    def __init__(self, categorical_mask: Optional[np.ndarray] = None):
        self.categorical_mask = categorical_mask
        self._fill_values: Optional[Dict[int, float]] = None

    def fit(self, X: np.ndarray) -> "NaNImputer":
        """Compute per-feature imputation values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (may contain np.nan).

        Returns
        -------
        self
        """
        if X.size == 0:
            self._fill_values = {}
            return self

        n_features = X.shape[1] if X.ndim > 1 else 1
        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X

        if self.categorical_mask is None:
            cat_mask = np.zeros(n_features, dtype=bool)
        else:
            cat_mask = np.asarray(self.categorical_mask, dtype=bool)

        self._fill_values = {}
        for i in range(n_features):
            col = X_2d[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                fill = 0.0
            elif cat_mask[i] if i < len(cat_mask) else False:
                # Categorical: most frequent
                fill = float(Counter(valid.tolist()).most_common(1)[0][0])
            else:
                # Numeric: median
                fill = float(np.median(valid))
            self._fill_values[i] = fill

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN values with fitted imputation values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data with potential NaN.

        Returns
        -------
        np.ndarray
            Imputed data (same shape).
        """
        if self._fill_values is None:
            raise RuntimeError("NaNImputer must be fitted before transform.")

        if X.size == 0:
            return X.copy()

        n_features = X.shape[1] if X.ndim > 1 else 1
        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X
        X_imputed = X_2d.copy()

        for i, fill_val in self._fill_values.items():
            if i < X_imputed.shape[1]:
                col = X_imputed[:, i]
                nan_mask = np.isnan(col)
                if nan_mask.any():
                    col[nan_mask] = fill_val
                    X_imputed[:, i] = col

        return X_imputed.reshape(X.shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)