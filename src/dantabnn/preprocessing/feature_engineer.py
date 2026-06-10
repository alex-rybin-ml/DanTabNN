"""Auto-feature engineering via generic polynomial/log transforms.

Generates x² for every numeric feature (captures non-linear relationships)
and log1p for right-skewed features (|skew| > 2.0).

Operates on numpy arrays, fit on training data, transform on test data.
No feature name dependencies — purely statistical.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


class AutoFeatureEngineer:
    """Generate polynomial and log-transformed features automatically.

    During ``fit()``, computes skewness for each feature column and:
    - Always generates x² for every feature (captures convex/concave effects).
    - Generates log1p for features where |skewness| > skew_threshold.

    The generated features are concatenated to the original features,
    producing (n_samples, n_orig_features + n_generated) arrays.

    Parameters
    ----------
    skew_threshold : float, default=2.0
        Absolute skewness threshold above which log1p is generated.
    generate_square : bool, default=True
        Whether to generate x² for each feature.
    max_generated : int, default=100
        Maximum number of generated features to append. If the
        auto-detection would exceed this, log1p candidates are
        trimmed by highest |skew| first.
    """

    def __init__(
        self,
        skew_threshold: float = 2.0,
        generate_square: bool = True,
        max_generated: int = 100,
    ):
        self.skew_threshold = skew_threshold
        self.generate_square = generate_square
        self.max_generated = max_generated

        # Fit state
        self._square_indices: List[int] = []
        self._log_indices: List[int] = []
        self._n_orig_features: int = 0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoFeatureEngineer":
        """Analyze feature distributions and decide which transforms to apply.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (numeric only, NaN-free).
        y : np.ndarray, optional
            Ignored (API compatibility).

        Returns
        -------
        self
        """
        if X.size == 0:
            self._n_orig_features = 0
            return self

        n_features = X.shape[1] if X.ndim > 1 else 1
        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X
        self._n_orig_features = n_features

        # Square: always for every feature (if enabled)
        if self.generate_square:
            self._square_indices = list(range(n_features))

        # Log1p: for skewed features
        log_candidates: List[Tuple[int, float]] = []
        for i in range(n_features):
            col = X_2d[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) < 3:
                continue
            # Clip to non-negative for skew computation
            col_clipped = valid[valid >= 0]
            if len(col_clipped) < 3:
                continue
            mean_val = col_clipped.mean()
            std_val = col_clipped.std()
            if std_val > 0:
                skew = abs(float(np.mean((col_clipped - mean_val) ** 3) / (std_val ** 3)))
            else:
                skew = 0.0
            if abs(skew) > self.skew_threshold:
                log_candidates.append((i, abs(skew)))

        # Sort by skew magnitude descending
        log_candidates.sort(key=lambda x: x[1], reverse=True)

        # Respect max_generated cap
        total_allowed = self.max_generated
        n_square = len(self._square_indices)
        if n_square + len(log_candidates) > total_allowed:
            # Trim log candidates, keep highest skew
            available_log_slots = max(0, total_allowed - n_square)
            log_candidates = log_candidates[:available_log_slots]

        self._log_indices = [idx for idx, _ in log_candidates]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Generate and append transformed features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        np.ndarray of shape (n_samples, n_orig_features + n_generated)
        """
        if self._n_orig_features == 0 or X.size == 0:
            return X.copy() if X.size > 0 else np.empty((0, 0))

        n_features = X.shape[1] if X.ndim > 1 else 1

        if n_features != self._n_orig_features:
            raise ValueError(
                f"Expected {self._n_orig_features} features, got {n_features}"
            )

        X_2d = X.reshape(-1, n_features) if X.ndim == 1 else X
        parts = [X_2d]

        # Append square features
        for i in self._square_indices:
            col = X_2d[:, i].astype(np.float64)
            squared = col ** 2
            parts.append(squared.reshape(-1, 1))

        # Append log1p features
        for i in self._log_indices:
            col = X_2d[:, i].astype(np.float64)
            # Clip to non-negative, add small epsilon for stability
            col_clipped = np.clip(col, 0, None)
            log_feat = np.log1p(col_clipped + 1e-10)
            parts.append(log_feat.reshape(-1, 1))

        if len(parts) == 1:
            return X_2d

        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_out(self) -> int:
        """Number of features after transformation."""
        return self._n_orig_features + len(self._square_indices) + len(self._log_indices)