"""Orchestration pipeline or DANet feature generation."""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import pandas as pd
import yaml

from .base import BaseDANetFeatureGenerator
from .domain import DomainFeatureGenerator
from .embedding import HighCardinalityEmbedder
from .interaction import SelectiveInteractionGenerator
from .temporal import TemporalAggregationGenerator
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DANetFeatureGenerationPipeline:
    """Orchestrates multiple feature generators with redundancy removal and DANet constraints.

    Parameters
    ----------
    generators : List[BaseDANetFeatureGenerator]
        List of generator instances to apply in order.
    redundancy_threshold : float, default=0.98
        Maximum absolute Pearson correlation allowed between any two generated features.
        Features with correlation above this threshold are considered redundant; the later
        feature (in order of generators) is removed.
    max_features : int, default=500
        Maximum total number of generated features allowed. IF exceeded, features are
        filtered by importance (currently by order) until the limit is satisfied.
    """

    def __init__(
            self,
            generators: Optional[List[BaseDANetFeatureGenerator]],
            redundancy_threshold: float = 0.98,
            max_features: int = 500
    ):
        self.generators = generators or []
        self.redundancy_threshold = redundancy_threshold
        self.max_features = max_features

        # Internal state
        self._feature_names: List[str] = []
        self._generator_metadata: Dict[str, Any] = {}
        self._redundancy_mask: Optional[pd.Series] = None
        self.is_fitted = False

    def add_generator(self, generator: BaseDANetFeatureGenerator) -> None:
        """Append a generator to the pipeline."""
        self.generators.append(generator)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DANetFeatureGenerationPipeline":
        """Fit all generators on the training data."""
        logger.info("Fitting DANet feature generation pipeline")
        if len(self.generators) == 0:
            logger.warning("No generators configured: pipeline will produce no features.")
            self._feature_names = []
            self.is_fitted = True
            return self

        # Fit each generator
        for gen in self.generators:
            logger.info(f"Fitting generator: {gen.name}")
            gen.fit(X, y)

        # Collect all generated features
        all_features = []
        for gen in self.generators:
            try:
                features = gen.transform(X)
                if not features.empty:
                    all_features.append(features)
            except Exception as e:
                logger.error(f"Generator {gen.name} failed during transform {e}")
                raise

        if not all_features:
            self._feature_name = []
            self.is_fitted = True
            return self

        # Concatenate
        concatenated = pd.concat(all_features, axis=1)
        logger.info(f"Total generated redundancy removal: {concatenated.shape[1]}")

        # Apply redundancy removal
        selected_idx = self._removal_redundant(concatenated)
        concatenated = concatenated.iloc[:, selected_idx]
        logger.info(f"Features after redundancy removal: {concatenated.shape[1]}")

        # Enforce max_features limit (simple truncation by oder)
        if concatenated.shape[1] > self.max_features:
            logger.warning(
                f"Number of features ({concatenated.shape[1]} exceed limit {self.max_features}, "
                "Truncating to first {self.max_features} features."
            )
            concatenated = concatenated.iloc[:, : self.max_features]

        self._feature_names = concatenated.columns.tolist()
        self._generator_metadata = {
            gen.name: gen.get_metadata() for gen in self.generators
        }
        self.is_fitted = True
        logger.info(f"Pipeline fitted with with {len(self._feature_names)} features")
        return self

    def _remove_redundant(self, features_df: pd.DataFrame) -> List[int]:
        """Return indices of columns to keep after redundancy filtering."""
        n = features_df.shape[1]
        if n <= 1:
            return list(range(n))

        corr_matrix = features_df.corr().abs()
        to_keep = []
        for i in range(n):
            redundant = False
            for j in to_keep:
                if corr_matrix.iloc[i, j] > self.redundancy_threshold:
                    redundant = True
                    logger.debug(
                        f"Feature {features_df.columns[i]} redundant with {features_df.columns[j]} "
                        f"(corr={corr_matrix.iloc[i, j]:.3f})"
                    )
                    break
            if not redundant:
                to_keep.append(i)
        return to_keep

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted generators."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")
        if len(self._feature_names) == 0:
            return pd.DataFrame(index=X.index)

        all_features = []
        for gen in self.generators:
            try:
                features = gen.transform(X)
                if not features.empty:
                    all_features.append(features)
            except Exception as e:
                logger.error(f"Generator {gen.name} failed during transform: {e}")
                raise

        if not all_features:
            return pd.DataFrame(index=X.index)

        concatenated = pd.concat(all_features, axis=1)
        # Keep only the features that survived redundancy removal (by column names)
        available = concatenated.columns.intersection(self._feature_names)
        missing = set(self._feature_names) - set(available)
        if missing:
            logger.warning(f"Some fitted features are missing in transformed data: {missing}")
            # Fill missing columns with NaN
            for col in missing:
                concatenated[col] = np.nan
        # Reorder columns as in fitted feature list
        concatenated = concatenated[self._feature_names]
        return concatenated

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """FIt pipeline and transform training data."""
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self) -> List[str]:
        """Return names of the final generated features."""
        return self._feature_names.copy()

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata from all generators."""
        return self._generator_metadata.copy()

    def validate_danet_compatibility(self) -> bool:
        """Check that the final feature set satisfies DANet constraints."""
        if len(self._feature_names) > self.max_features:
            logger.error(f"Feature count {len(self._feature_names)} exceeds limit {self.max_features}")
            return False
        # Ensure all features are numeric (should be guaranteed by generators)
        return True

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DANetFeatureGenerationPipeline":
        """Create a pipeline from a YAML configuration file.

        The YAML file should have the following structure:
        ```yaml
        redundancy_threshold: 0.98
        max_features: 500
        generators:
          - type: DomainFeatureGenerator
            params:
              degree: 2
              interaction_only: false
              include_bias: False
          - type: TemporalAggregationGenerator
            params:
              date_column: "date"
              groupby_columns: ["store_id", "product_id"]
              windows: [7, 30]
              aggregations: ["mean", "std", "min", "max"]
          - type: SelectiveInteractionGenerator
            params:
              mi_threshold: 1.2
              correlation_threshold: 0.98
              max_interactions: 100
          - type: HighCardinalityEmbedder
            params:
              cardinality_threshold: 100
              smoothing: 10.0
              unknown_value: 0.0
        ```
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        redundancy_threshold = config.get("redundancy_threshold", 0.98)
        max_features = config.get("max_features", 500)

        generators = []
        for gen_config in config.get("generators", []):
            gen_type = gen_config["type"]
            params = gen_config.get("params", {})
            generator = cls._instantiate_generator(gen_type, params)
            generators.append(generator)

        return cls(
            generators=generators,
            redundancy_threshold=redundancy_threshold,
            max_features=max_features,
        )

    @staticmethod
    def _instantiate_generator(gen_type: str, params: Dict[str, Any]) -> BaseDANetFeatureGenerator:
        """Create a generator instance from type name and parameters."""
        # Backward compatibility: rename 'columns' to 'numeric_columns' for DomainFeatureGenerator
        if gen_type == "DomainFeatureGenerator" and "columns" in params:
            import warnings
            warnings.warn(
                "The parameter 'columns' for DomainFeatureGenerator is deprecated. "
                "Use 'numeric_columns' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            params["numeric_columns"] = params.get("columns")

        mapping = {
            "DomainFeatureGenerator": DomainFeatureGenerator,
            "TemporalAggregationGenerator": TemporalAggregationGenerator,
            "SelectiveInteractionGenerator": SelectiveInteractionGenerator,
            "HighCardinalityEmbedder": HighCardinalityEmbedder,
        }
        if gen_type not in mapping:
            raise ValueError(f"Unknown generator type: {gen_type}")
        return mapping[gen_type](**params)
