"""Standard scaler for numeric features."""

import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler


class StandardScaler:
    """Wrapper around sklearn's StandardScaler with pandas-friendly interface."""

    def __init__(self):
        self.scaler = SklearnStandardScaler()
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit scaler to data.
        
        Parameters
        ----------
        X : np.ndarray
            Numeric data of shape (n_sample, n_features)

        Returns
        -------
        self
        """
        self.scaler.fit(X)
        self.mean_ = self.scaler.mean_
        self.scale_ = self.scaler.scale_
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.
        
        Parameters
        ----------
        X : np.ndarray
            Numeric data.

        Returns
        -------
        np.ndarray
            Scaled data.
        """
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.scaler.fit_transform(X)
