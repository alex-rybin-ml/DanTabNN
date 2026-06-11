"""Categorical encoder for one-hot encoding"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class CategoricalEncoder:
    """Wrapper around sklearn's OneHotEncoder with pandas-friendly interface."""

    def __init__(self, handle_unknown: str = "ignore", sparse_output: bool = True):
        self.encoder = SklearnOneHotEncoder(
            sparse_output=sparse_output, handle_unknown=handle_unknown
        )
        self.categories_ = None
        self.n_values_per_feature = None
        self.sparse_output = sparse_output

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
    
    def transform(self, X: np.ndarray):
        """Transform data. Returns sparse matrix if sparse_output=True, else dense.
        
        Parameters
        ----------
        X: np.ndarray
            Categorical data

        Returns
        -------
        scipy.sparse matrix or np.ndarray
            One-hot encoded data.
        """
        result = self.encoder.transform(X)
        if self.sparse_output:
            return result  # scipy sparse CSR
        return result.toarray() if hasattr(result, 'toarray') else result
    
    def fit_transform(self, X: np.ndarray):
        """Fit and transform in one step."""
        result = self.encoder.fit_transform(X)
        self.categories_ = self.encoder.categories_
        self.n_values_per_feature = [len(cats) for cats in self.categories_]
        if self.sparse_output:
            return result
        return result.toarray() if hasattr(result, 'toarray') else result
