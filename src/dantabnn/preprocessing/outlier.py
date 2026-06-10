"""IQR-based outlier clipper for numeric features.

Winsorizes extreme values by capping to [Q1 - 1.5×IQR, Q3 + 1.5×IQR].
Operates on numpy arrays, fit on training data, transform on test data.
"""

import numpy as np
from typing import Optional, Dict


class OutlierClipper:
    """Clip numeric features using IQR-based bounds.

    For each feature column, computes lower and upper fences during
    ``fit()`` and caps values outside during ``transform()``. This
    is a *winsorization* approach — no rows are dropped.

    Parameters
    ----------
    iqr_multiplier : float, default=1.5
        Multiplier for IQR to define fence bounds.
        Standard 1.5 corresponds to Tukey's fences.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self._bounds: Optional[Dict[int, tuple]] = None

    def fit(self, X: np.ndarray) -> "OutlierClipper":
        """Compute per-feature IQR bounds.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (numeric only).

        Returns
        -------
        self
        """
        if X.size == 0:
            self._bounds = {}
            return self

        n_features = X.shape[1] if X.ndim > 1 else 1
        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X
        self._bounds = {}
        for i in range(n_features):
            col = X_2d[:, i]
            col = col[~np.isnan(col)]
            if len(col) < 4:
                # Too few values to compute meaningful IQR; skip
                continue
            q1 = np.percentile(col, 25)
            q3 = np.percentile(col, 75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            self._bounds[i] = (lower, upper)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip values to fitted IQR bounds.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to clip.

        Returns
        -------
        np.ndarray
            Clipped data (same shape).
        """
        if self._bounds is None:
            raise RuntimeError("OutlierClipper must be fitted before transform.")

        if X.size == 0:
            return X.copy()

        n_features = X.shape[1] if X.ndim > 1 else 1
        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X
        X_clipped = X_2d.copy()

        for i, (lower, upper) in self._bounds.items():
            if i < X_clipped.shape[1]:
                col = X_clipped[:, i]
            col = np.where(np.isnan(col), np.nanmedian(col), col)
            col[col < lower] = lower
            col[col > upper] = upper
            X_clipped[:, i] = col

        return X_clipped.reshape(X.shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)