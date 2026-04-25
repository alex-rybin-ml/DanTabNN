"""Categorical encoder for one-hot encoding"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class CategoricalEncoder:
    """Wrapper around sklearn's OneHotEncoder with pandas-friendly interface."""

    def __init__(self, handle_unknown: str = "ignore"):
        self.encoder = SklearnOneHotEncoder(
            sparse_output=False, handle_unknown=handle_unknown
        )
        self.categories_ = None
        self.n_values_per_feature = None

    def fit(self, X: np.ndarray) -> "CategoricalEncoder":
        """Fit encoder to data
        
        Parameters
        ----------
        X : np.ndarray
            Categorical data of shape (n_sample, n_features).

        Returns
        -------
        self
        """
        self.encoder.fit(X)
        self.categories_ = self.encoder.categories_
        self.n_values_per_feature = [len(cats) for cats in self.categories_]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data.
        
        Parameters
        ----------
        X: np.ndarray
            Categorical data

        Returns
        -------
        np.ndarray
            One-hot encoded data.
        """
        return self.encoder.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.encoder.fit_transform(X)
