"""Abstract base class for DANet feature generators."""


from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseDANetFeatureGenerator(ABC):
    """Abstract base class for feature generators that produce DANet-compatible features.

    Concrete subclasses must implement `fit`, `transform`, `get_feature_names`,
    and `validate_danet_compatibility`.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Parameters
        ----------
        name: str, optional
            Human-readable name of the generator (defaults to class name).
        """
        self.name = name or self.__class__.__name__
        self._feature_names: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self.is_fitted = False

    @property
    def supports_jit(self) -> bool:
        """Whether this generator supports JIT (just-in-time) generation in tensors."""
        return False

    def jit_transform(self):
        """Generate features from raw tensors (JIT mode). Must be implemented if supports_jit=True."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support JIT generation.")

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseDANetFeatureGenerator":
        """Learn characteristics from the data needed for feature generation.

        Parameters
        ----------
        X: pd.DataFrame
            Input dataframe with the original columns.
        y: pd.Series, optional
            Target columns (maybe used for supervised feature generation).

        Returns
        -------
        self
            Fitted generator.
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate new features from the input dataframe.

        Parameters
        ----------
        X: pd.DataFrame
            Input dataframe with the original columns.

        Returns
        -------
        pd.DataFrame
            Data Frame containing only the newly generated features (same index as X).
            The number of columns must match `len(self.get_features_names())`.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return names of the generated features.

        Returns
        -------
        List[str]
            List of feature names, in the same order as columns returned by `transform`.
        """
        pass

    @abstractmethod
    def validate_danet_compatibility(self) -> bool:
        """Check that generated features satisfy DANets constraints.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the generator and transform the data in one step.

        Parameters
        ----------
        X : pd.DataFame
            Input dataframe.
        y : pd.Series
            Target column.

        Returns
        -------
        pd.DataFrame
            Generated features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the generated features.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing at least:
            - 'generation_name': str
            - 'n_features': int
            - 'feature_names': List[str]
            - 'danet_compatible': bool
            Additional generator-specific metadata can be added.
        """
        return {
            "generator_name": self.name,
            "n_features": len(self._feature_names),
            "feature_names": self._feature_names.copy(),
            "danet_compatible": self.validate_danet_compatibility(),
            **self._metadata
        }

    def _log_info(self, msg: str) -> None:
        """Helper to log messages with generator name prefix."""
        logger.info(f"[{self.name}] {msg}")

    def _log_warning(self, msg: str) -> None:
        logger.warning(f"[{self.name}] {msg}")

    def _log_error(self, msg: str) -> None:
        logger.error(f"[{self.name}] {msg}")

    def _log_debug(self, msg: str) -> None:
        logger.debug(f"[{self.name}] {msg}")

    def _impute_numeric(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Impute missing values in numeric columns with columns means (or 0 if all NaN).

        Parameters
        ----------
        X: pd.DataFrame
            Input dataframe.
        fit: bool
            If True, compute and store imputation means from X.
            If False, use previously stored means (must have been fitted).

        Returns
        -------
        pd.DataFrame
            Data Frame with NaN imputed.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return X
        if fit:
            self._imputation_means = X[numeric_cols].mean()
            self._imputation_means = self._imputation_means.fillna(0.0)
        # Ensure imputation means exist
        if not hasattr(self, '_imputation_means'):
            raise RuntimeError("Imputation means not fitted. Call with fit=True first.")
        X_imputed = X.copy()
        X_imputed[numeric_cols] = X_imputed[numeric_cols].fillna(self._imputation_means)
        return X_imputed
