"""Tests for feature_generation module: base, domain, embedding, interaction, orchestrator, temporal."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dantabnn.feature_generation.base import BaseDANetFeatureGenerator
from dantabnn.feature_generation.domain import (
    DomainFeatureGenerator,
    DomainRatioGenerator,
    _PREDEFINED_TRANSFORMS,
    _safe_divide,
)
from dantabnn.feature_generation.embedding import HighCardinalityEmbedder
from dantabnn.feature_generation.interaction import SelectiveInteractionGenerator
from dantabnn.feature_generation.orchestrator import DANetFeatureGenerationPipeline
from dantabnn.feature_generation.temporal import TemporalAggregationGenerator


# ===========================================================================
# Helpers
# ===========================================================================

def _make_numeric_df(n_rows=100, n_cols=5, random_state=42):
    rng = np.random.RandomState(random_state)
    data = {f"col_{i}": rng.randn(n_rows) for i in range(n_cols)}
    return pd.DataFrame(data)


def _make_mixed_df(n_rows=100, random_state=42):
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame()
    df["num1"] = rng.randn(n_rows)
    df["num2"] = rng.randn(n_rows) + 5
    df["cat1"] = rng.choice(["a", "b", "c"], size=n_rows)
    df["cat2"] = rng.choice(["x", "y"], size=n_rows)
    df["num3"] = rng.exponential(2, size=n_rows)  # skewed
    df["cyclic"] = rng.randint(0, 7, size=n_rows)  # 0-6, like day of week
    df["target"] = (
        df["num1"] * 0.5 + df["num2"] * 0.3
        + (df["cat1"] == "a").astype(float) * 2.0
        + rng.randn(n_rows) * 0.2
    )
    return df


# ===========================================================================
# BaseDANetFeatureGenerator
# ===========================================================================

class _ConcreteGenerator(BaseDANetFeatureGenerator):
    """Minimal concrete implementation for testing abstract base."""

    def fit(self, X, y=None):
        self._feature_names = ["feat_1", "feat_2"]
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Not fitted")
        return pd.DataFrame({"feat_1": np.ones(len(X)), "feat_2": np.zeros(len(X))}, index=X.index)

    def get_feature_names(self):
        return self._feature_names.copy()

    def validate_danet_compatibility(self):
        return len(self._feature_names) <= 500


class TestBaseDANetFeatureGenerator:
    def test_init_sets_name(self):
        gen = _ConcreteGenerator(name="test_gen")
        assert gen.name == "test_gen"

    def test_init_default_name(self):
        gen = _ConcreteGenerator()
        assert gen.name == "_ConcreteGenerator"

    def test_is_fitted_initially_false(self):
        gen = _ConcreteGenerator()
        assert not gen.is_fitted

    def test_supports_jit_default_false(self):
        gen = _ConcreteGenerator()
        assert gen.supports_jit is False

    def test_jit_transform_raises(self):
        gen = _ConcreteGenerator()
        with pytest.raises(NotImplementedError):
            gen.jit_transform()

    def test_fit_transform(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        gen = _ConcreteGenerator()
        result = gen.fit_transform(df)
        assert gen.is_fitted
        assert result.shape == (3, 2)

    def test_get_metadata(self):
        gen = _ConcreteGenerator(name="meta_gen")
        gen.fit(pd.DataFrame({"a": [1, 2]}))
        meta = gen.get_metadata()
        assert meta["generator_name"] == "meta_gen"
        assert meta["n_features"] == 2
        assert "feature_names" in meta
        assert "danet_compatible" in meta

    def test_impute_numeric_fit_and_apply(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
        gen = _ConcreteGenerator()
        imputed = gen._impute_numeric(df, fit=True)
        # NaN should be filled with column means
        assert not imputed.isna().any().any()
        assert imputed["a"].iloc[1] == 2.0  # mean of (1, 3)
        assert imputed["b"].iloc[0] == 3.0  # mean of (2, 4)

    def test_impute_numeric_without_fit_raises(self):
        gen = _ConcreteGenerator()
        with pytest.raises(RuntimeError, match="not fitted"):
            gen._impute_numeric(pd.DataFrame({"a": [1.0]}), fit=False)

    def test_validate_danet_compatibility(self):
        gen = _ConcreteGenerator()
        gen.fit(pd.DataFrame({"a": [1]}))
        assert gen.validate_danet_compatibility() is True

    def test_validate_limits_over_500(self):
        gen = _ConcreteGenerator()
        gen._feature_names = [f"f{i}" for i in range(501)]
        gen.is_fitted = True
        assert gen.validate_danet_compatibility() is False


# ===========================================================================
# DomainFeatureGenerator
# ===========================================================================

class TestDomainFeatureGenerator:
    def test_fit_polynomial_features(self):
        df = _make_numeric_df(n_rows=10, n_cols=2)
        gen = DomainFeatureGenerator(degree=2)
        gen.fit(df)
        assert gen.is_fitted
        # degree=2 with 2 features → 1 (bias) + 2 (linear) + 1 (x1^2) + 2 (x1*x2) + 1 (x2^2)? No, include_bias=False
        # polynomial features without bias: x0, x1, x0^2, x0*x1, x1^2 = 5 features when interaction_only=False
        assert len(gen._feature_names) >= 3

    def test_fit_interaction_only(self):
        df = _make_numeric_df(n_rows=10, n_cols=2)
        gen = DomainFeatureGenerator(degree=2, interaction_only=True)
        gen.fit(df)
        # interaction_only=True: x0, x1, x0*x1 = 3
        assert len(gen._feature_names) == 3

    def test_transform_returns_dataframe(self):
        df = _make_numeric_df(n_rows=10, n_cols=3)
        gen = DomainFeatureGenerator(degree=2)
        gen.fit(df)
        result = gen.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == len(df)

    def test_transform_before_fit_raises(self):
        gen = DomainFeatureGenerator()
        with pytest.raises(RuntimeError):
            gen.transform(pd.DataFrame({"a": [1]}))

    def test_fit_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
        gen = DomainFeatureGenerator()
        gen.fit(df)
        assert gen.is_fitted
        assert gen._feature_names == []

    def test_numeric_columns_subset(self):
        df = _make_numeric_df(n_rows=10, n_cols=3)
        gen = DomainFeatureGenerator(numeric_columns=["col_0", "col_2"])
        gen.fit(df)
        assert gen.is_fitted
        names = gen._feature_names
        # Should only use col_0 and col_2
        assert any("col_0" in n for n in names)
        assert any("col_2" in n for n in names)

    def test_missing_numeric_columns_warning(self):
        df = _make_numeric_df(n_rows=10, n_cols=2)
        gen = DomainFeatureGenerator(numeric_columns=["col_0", "nonexistent"])
        gen.fit(df)
        assert gen.is_fitted

    def test_get_feature_names(self):
        df = _make_numeric_df(n_rows=10, n_cols=2)
        gen = DomainFeatureGenerator(degree=2)
        gen.fit(df)
        names = gen.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_metadata(self):
        df = _make_numeric_df(n_rows=10, n_cols=2)
        gen = DomainFeatureGenerator(degree=2, interaction_only=True)
        gen.fit(df)
        meta = gen.get_metadata()
        assert meta["generator_type"] == "polynomial"
        assert meta["degree"] == 2
        assert meta["interaction_only"] is True

    def test_transform_not_fitted(self):
        gen = DomainFeatureGenerator()
        with pytest.raises(RuntimeError):
            gen.transform(pd.DataFrame({"a": [1]}))


# ===========================================================================
# DomainRatioGenerator
# ===========================================================================

class TestDomainRatioGenerator:
    # --- Explicit mode ---
    def test_explicit_ratio(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 0, 40]})
        gen = DomainRatioGenerator(templates=[
            {"type": "ratio", "columns": ["a", "b"], "output_name": "r"}
        ])
        gen.fit(df, y=None)
        out = gen.transform(df)
        # row 2: b=0 → safe divide → 0.0
        assert out.shape == (4, 1)
        assert out["r"].tolist() == pytest.approx([0.1, 0.1, 0.0, 0.1])

    def test_explicit_log1p(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.shape == (3, 1)
        expected = np.log1p([0, 1, 2])
        assert out.iloc[:, 0].tolist() == pytest.approx(expected.tolist())

    def test_explicit_zscore(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "zscore", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.shape == (5, 1)
        # Output should have mean ~0 and std ~1 (ddof=0 matches population std)
        assert out.iloc[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
        assert out.iloc[:, 0].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    def test_explicit_clip(self):
        df = pd.DataFrame({"x": [0, 5, 10, 15, 20]})
        gen = DomainRatioGenerator(templates=[
            {"type": "clip", "columns": ["x"], "params": {"lower": 3, "upper": 12}}
        ])
        gen.fit(df)
        out = gen.transform(df)
        vals = out.iloc[:, 0].tolist()
        assert vals == [3, 5, 10, 12, 12]

    def test_explicit_cyclic_sin(self):
        df = pd.DataFrame({"dow": [0, 3, 6]})
        gen = DomainRatioGenerator(templates=[
            {"type": "cyclic_sin", "columns": ["dow"], "params": {"period": 7}}
        ])
        gen.fit(df)
        out = gen.transform(df)
        # sin(0) = 0, sin(2π*3/7) ≈ 0.4338, sin(2π*6/7) ≈ -0.4338
        assert out.iloc[0, 0] == pytest.approx(0.0, abs=1e-4)
        assert out.iloc[1, 0] == pytest.approx(np.sin(2 * np.pi * 3 / 7), abs=1e-4)

    def test_explicit_cyclic_cos(self):
        df = pd.DataFrame({"dow": [0, 3]})
        gen = DomainRatioGenerator(templates=[
            {"type": "cyclic_cos", "columns": ["dow"], "params": {"period": 7}}
        ])
        gen.fit(df)
        out = gen.transform(df)
        # cos(0) = 1, cos(2π*3/7) ≈ -0.9009
        assert out.iloc[0, 0] == pytest.approx(1.0, abs=1e-4)

    def test_explicit_log(self):
        df = pd.DataFrame({"x": [1.0, 10.0, 100.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.iloc[:, 0].tolist() == pytest.approx(np.log([1, 10, 100]).tolist())

    def test_explicit_sqrt(self):
        df = pd.DataFrame({"x": [0.0, 4.0, 9.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "sqrt", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.iloc[:, 0].tolist() == pytest.approx([0, 2, 3])

    def test_explicit_square(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "square", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.iloc[:, 0].tolist() == pytest.approx([1, 4, 9])

    def test_explicit_inverse(self):
        df = pd.DataFrame({"x": [0.0, 2.0, 4.0]})
        gen = DomainRatioGenerator(templates=[
            {"type": "inverse", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        # 1/0 → 0.0, 1/2 = 0.5, 1/4 = 0.25
        assert out.iloc[:, 0].tolist() == pytest.approx([0.0, 0.5, 0.25])

    def test_output_name_auto_generated(self):
        df = pd.DataFrame({"x": [1, 2]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["x"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.columns.tolist() == ["log1p_x"]

    def test_output_name_explicit(self):
        df = pd.DataFrame({"x": [1, 2]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["x"], "output_name": "my_log"}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.columns.tolist() == ["my_log"]

    def test_missing_columns_in_template_warns(self):
        df = pd.DataFrame({"a": [1, 2]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["nonexistent"]}
        ])
        gen.fit(df)
        # Template should be skipped gracefully
        out = gen.transform(df)
        assert out.empty

    # --- Auto-discovery mode ---
    def test_auto_discover_skewed(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"x": np.exp(rng.randn(500) * 3)})  # very skewed
        gen = DomainRatioGenerator(max_features=5)
        gen.fit(df, y=None)
        names = gen.get_feature_names()
        assert any("log1p" in n for n in names)

    def test_auto_discover_cyclic(self):
        df = pd.DataFrame({"dow": np.arange(200) % 7})
        gen = DomainRatioGenerator(max_features=5)
        gen.fit(df, y=None)
        names = gen.get_feature_names()
        assert any("cyclic_sin" in n for n in names)
        assert any("cyclic_cos" in n for n in names)

    def test_auto_discover_with_target(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "a": rng.randn(200),
            "b": rng.randn(200),
            "target": rng.randn(200),
        })
        gen = DomainRatioGenerator(max_features=10)
        gen.fit(df, y=df["target"])
        assert gen.is_fitted

    def test_auto_discover_caps_at_max_features(self):
        rng = np.random.RandomState(42)
        n_cols = 15
        data = {f"col_{i}": rng.randn(300) for i in range(n_cols)}
        df = pd.DataFrame(data)
        gen = DomainRatioGenerator(max_features=5)
        gen.fit(df, y=None)
        names = gen.get_feature_names()
        assert len(names) <= 5

    def test_auto_discover_no_numeric_columns(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
        gen = DomainRatioGenerator()
        gen.fit(df, y=None)
        assert gen.is_fitted
        assert gen.get_feature_names() == []

    def test_transform_empty_templates(self):
        df = pd.DataFrame({"a": [1, 2]})
        gen = DomainRatioGenerator(templates=[])
        gen.fit(df)
        out = gen.transform(df)
        assert out.empty

    # --- Helper functions ---
    def test_safe_divide(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 0.0, 30.0]})
        result = _safe_divide(df, "a", "b")
        # 2/0 → 0.0
        assert result.tolist() == pytest.approx([0.1, 0.0, 0.1])

    def test_predefined_transforms_has_all_types(self):
        expected_types = [
            "log", "log1p", "sqrt", "square", "inverse",
            "zscore", "ratio", "cyclic_sin", "cyclic_cos", "clip",
        ]
        for t in expected_types:
            assert t in _PREDEFINED_TRANSFORMS

    # --- Edge cases ---
    def test_explicit_missing_column_handled(self):
        df = pd.DataFrame({"a": [1, 2]})
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["missing_col"]}
        ])
        gen.fit(df)
        out = gen.transform(df)
        assert out.empty

    def test_get_metadata(self):
        gen = DomainRatioGenerator(templates=[
            {"type": "log1p", "columns": ["x"]}
        ])
        gen.fit(pd.DataFrame({"x": [1, 2]}))
        meta = gen.get_metadata()
        assert meta["generator_type"] == "domain_ratio"
        assert meta["max_features"] == 20

    def test_auto_discover_no_target(self):
        """y=None should work — only unsupervised rules triggered."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"x": np.exp(rng.randn(500) * 3)})
        gen = DomainRatioGenerator(max_features=5)
        gen.fit(df, y=None)
        assert gen.is_fitted


# ===========================================================================
# HighCardinalityEmbedder
# ===========================================================================

class TestHighCardinalityEmbedder:
    def test_fit_without_y_produces_no_features(self):
        df = pd.DataFrame({
            "cat": [f"val_{i}" for i in range(200)],
            "num": np.random.randn(200),
        })
        gen = HighCardinalityEmbedder(cardinality_threshold=100)
        gen.fit(df, y=None)
        assert gen.is_fitted
        assert gen.get_feature_names() == []

    def test_fit_with_y_target_encodes(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat": rng.choice([f"g{i}" for i in range(5)], size=200),
            "target": rng.randn(200),
        })
        gen = HighCardinalityEmbedder(cardinality_threshold=2)
        gen.fit(df, y=df["target"])
        assert gen.is_fitted
        names = gen.get_feature_names()
        # Cardinality 5 >= threshold 2 → should embed
        assert len(names) == 1
        assert "embed_cat" in names[0]

    def test_transform_produces_numeric_output(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat": rng.choice(["a", "b", "c", "d", "e"], size=100),
            "target": rng.randn(100),
        })
        gen = HighCardinalityEmbedder(cardinality_threshold=2)
        gen.fit(df, y=df["target"])
        result = gen.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (100, 1)
        assert pd.api.types.is_numeric_dtype(result.iloc[:, 0])

    def test_below_threshold_skipped(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat": rng.choice(["a", "b"], size=100),
            "target": rng.randn(100),
        })
        gen = HighCardinalityEmbedder(cardinality_threshold=100)
        gen.fit(df, y=df["target"])
        assert gen.get_feature_names() == []

    def test_custom_categorical_columns(self):
        df = pd.DataFrame({
            "cat_a": np.random.choice(["x", "y", "z", "w", "v"], size=100),
            "cat_b": np.random.choice(["p", "q"], size=100),
            "target": np.random.randn(100),
        })
        gen = HighCardinalityEmbedder(
            categorical_columns=["cat_a"], cardinality_threshold=2
        )
        gen.fit(df, y=df["target"])
        names = gen.get_feature_names()
        assert len(names) == 1
        assert "embed_cat_a" in names[0]

    def test_transform_before_fit_raises(self):
        gen = HighCardinalityEmbedder()
        with pytest.raises(RuntimeError):
            gen.transform(pd.DataFrame({"a": [1]}))

    def test_validate_danet_compatibility(self):
        gen = HighCardinalityEmbedder(cardinality_threshold=1)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat": rng.choice(["a", "b", "c"], size=50),
            "target": rng.randn(50),
        })
        gen.fit(df, y=df["target"])
        assert gen.validate_danet_compatibility() is True

    def test_get_metadata(self):
        gen = HighCardinalityEmbedder(cardinality_threshold=2)
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "cat": rng.choice(["a", "b", "c"], size=50),
            "target": rng.randn(50),
        })
        gen.fit(df, y=df["target"])
        meta = gen.get_metadata()
        assert meta["generator_type"] == "high_cardinality_embedding"
        assert "global_means" in meta


# ===========================================================================
# SelectiveInteractionGenerator
# ===========================================================================

class TestSelectiveInteractionGenerator:
    def test_fit_without_y_produces_all_pairs(self):
        df = _make_numeric_df(n_rows=50, n_cols=4)
        gen = SelectiveInteractionGenerator(max_interactions=50)
        gen.fit(df, y=None)
        assert gen.is_fitted
        names = gen.get_feature_names()
        # C(4,2) = 6 pairs
        assert len(names) == 6

    def test_fit_with_y_selective(self):
        rng = np.random.RandomState(42)
        df = _make_numeric_df(n_rows=100, n_cols=4)
        y = df["col_0"] * 2 + df["col_1"] * 0.1 + rng.randn(100) * 0.1
        gen = SelectiveInteractionGenerator(mi_threshold=1.0, max_interactions=50)
        gen.fit(df, y=y)
        assert gen.is_fitted

    def test_transform_returns_dataframe(self):
        df = _make_numeric_df(n_rows=30, n_cols=4)
        gen = SelectiveInteractionGenerator(max_interactions=10)
        gen.fit(df, y=None)
        result = gen.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 30

    def test_fewer_than_two_columns_no_interactions(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        gen = SelectiveInteractionGenerator()
        gen.fit(df, y=None)
        assert gen.get_feature_names() == []

    def test_max_interactions_cap(self):
        df = _make_numeric_df(n_rows=50, n_cols=10)
        gen = SelectiveInteractionGenerator(max_interactions=5)
        gen.fit(df, y=None)
        names = gen.get_feature_names()
        assert len(names) <= 5

    def test_correlation_filtering(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # perfectly correlated with a
            "c": [1, 0, 1, 0, 1],   # orthogonal
        })
        gen = SelectiveInteractionGenerator(correlation_threshold=0.95, max_interactions=10)
        gen.fit(df, y=None)
        names = gen.get_feature_names()
        # a*b should be removed (high correlation with a and b)
        assert "a_x_b" not in names

    def test_transform_before_fit_raises(self):
        gen = SelectiveInteractionGenerator()
        with pytest.raises(RuntimeError):
            gen.transform(pd.DataFrame({"a": [1]}))

    def test_get_metadata(self):
        df = _make_numeric_df(n_rows=30, n_cols=4)
        gen = SelectiveInteractionGenerator(max_interactions=5)
        gen.fit(df, y=None)
        meta = gen.get_metadata()
        assert meta["generator_type"] == "selective_interaction"
        assert "interaction_paris" in meta


# ===========================================================================
# DANetFeatureGenerationPipeline (Orchestrator)
# ===========================================================================

class TestOrchestrator:
    def test_empty_pipeline(self):
        pipe = DANetFeatureGenerationPipeline()
        df = pd.DataFrame({"a": [1, 2]})
        pipe.fit(df)
        assert pipe.is_fitted
        assert pipe.get_feature_names() == []

    def test_add_generator(self):
        pipe = DANetFeatureGenerationPipeline()
        gen = DomainFeatureGenerator(degree=2)
        pipe.add_generator(gen)
        assert len(pipe.generators) == 1

    def test_fit_with_generators(self):
        df = _make_numeric_df(n_rows=30, n_cols=3)
        pipe = DANetFeatureGenerationPipeline(
            generators=[
                DomainFeatureGenerator(degree=2, numeric_columns=["col_0", "col_1"]),
                DomainRatioGenerator(templates=[
                    {"type": "log1p", "columns": ["col_2"]}
                ]),
            ]
        )
        pipe.fit(df)
        assert pipe.is_fitted
        names = pipe.get_feature_names()
        assert len(names) > 0

    def test_transform_returns_dataframe(self):
        df = _make_numeric_df(n_rows=30, n_cols=3)
        pipe = DANetFeatureGenerationPipeline(
            generators=[DomainFeatureGenerator(degree=2)]
        )
        pipe.fit(df)
        result = pipe.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 30

    def test_fit_transform(self):
        df = _make_numeric_df(n_rows=20, n_cols=3)
        pipe = DANetFeatureGenerationPipeline(
            generators=[DomainFeatureGenerator(degree=2)]
        )
        result = pipe.fit_transform(df)
        assert pipe.is_fitted
        assert result.shape[0] == 20

    def test_redundancy_removal(self):
        # Generate two nearly identical features
        df = pd.DataFrame({
            "a": np.random.randn(50),
            "b": np.random.randn(50),
        })
        # Add a generator that produces col_a and col_a * 1.001 (nearly identical)
        class _RedundantGen(BaseDANetFeatureGenerator):
            def fit(self, X, y=None):
                self._feature_names = ["feat_1", "feat_2"]
                self.is_fitted = True
                return self
            def transform(self, X):
                return pd.DataFrame({
                    "feat_1": X["a"],
                    "feat_2": X["a"] * 1.001,
                }, index=X.index)
            def get_feature_names(self):
                return self._feature_names.copy()
            def validate_danet_compatibility(self):
                return True

        pipe = DANetFeatureGenerationPipeline(
            generators=[_RedundantGen()],
            redundancy_threshold=0.99,
        )
        pipe.fit(df)
        # feat_1 and feat_2 are nearly identical → one should be removed
        names = pipe.get_feature_names()
        assert len(names) <= 1  # at most one survives

    def test_max_features_cap(self):
        df = _make_numeric_df(n_rows=30, n_cols=5)
        pipe = DANetFeatureGenerationPipeline(
            generators=[DomainFeatureGenerator(degree=2)],
            max_features=5,
        )
        pipe.fit(df)
        names = pipe.get_feature_names()
        assert len(names) <= 5

    def test_transform_before_fit_raises(self):
        pipe = DANetFeatureGenerationPipeline()
        with pytest.raises(RuntimeError):
            pipe.transform(pd.DataFrame({"a": [1]}))

    def test_get_metadata(self):
        df = _make_numeric_df(n_rows=30, n_cols=3)
        gen = DomainFeatureGenerator(degree=2)
        pipe = DANetFeatureGenerationPipeline(generators=[gen])
        pipe.fit(df)
        meta = pipe.get_metadata()
        assert gen.name in meta

    def test_from_yaml(self):
        yaml_content = """\
redundancy_threshold: 0.95
max_features: 10
generators:
  - type: DomainFeatureGenerator
    params:
      degree: 2
      numeric_columns: ["col_0", "col_1"]
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            pipe = DANetFeatureGenerationPipeline.from_yaml(yaml_path)
            assert len(pipe.generators) == 1
            assert isinstance(pipe.generators[0], DomainFeatureGenerator)
            assert pipe.redundancy_threshold == 0.95
        finally:
            Path(yaml_path).unlink()

    def test_validate_danet_compatibility(self):
        df = _make_numeric_df(n_rows=30, n_cols=2)
        pipe = DANetFeatureGenerationPipeline(
            generators=[DomainFeatureGenerator(degree=2)]
        )
        pipe.fit(df)
        assert pipe.validate_danet_compatibility() is True


# ===========================================================================
# TemporalAggregationGenerator
# ===========================================================================

class TestTemporalAggregationGenerator:
    def test_fit_validates_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
            "store_id": ["S1"] * 50,
            "sales": np.random.randn(50),
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            windows=(7,),
            aggregations=("mean",),
            expanding=False,
        )
        gen.fit(df)
        assert gen.is_fitted
        names = gen.get_feature_names()
        assert len(names) > 0
        assert "temp_sales_mean_w7" in names

    def test_fit_missing_columns_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
        )
        with pytest.raises(ValueError):
            gen.fit(df)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "store_id": ["S1"] * 10,
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
        )
        gen.fit(df)
        assert gen.is_fitted
        assert gen.get_feature_names() == []

    def test_transform_pandas_returns_dataframe(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=30, freq="D"),
            "store_id": ["S1"] * 15 + ["S2"] * 15,
            "sales": np.random.randn(30) + 100,
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            windows=(7,),
            aggregations=("mean",),
            backend="pandas",
        )
        gen.fit(df)
        result = gen.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 30

    def test_expanding_window(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "store_id": ["S1"] * 20,
            "sales": np.random.randn(20) + 100,
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            windows=(),
            aggregations=("mean",),
            expanding=True,
            backend="pandas",
        )
        gen.fit(df)
        result = gen.transform(df)
        assert "temp_sales_mean_expanding" in result.columns

    def test_multiple_windows_and_aggs(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "store_id": ["S1"] * 20,
            "sales": np.random.randn(20) + 100,
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            windows=(7, 14),
            aggregations=("mean", "std"),
            backend="pandas",
        )
        gen.fit(df)
        result = gen.transform(df)
        assert "temp_sales_mean_w7" in result.columns
        assert "temp_sales_std_w7" in result.columns
        assert "temp_sales_mean_w14" in result.columns

    def test_transform_before_fit_raises(self):
        gen = TemporalAggregationGenerator(
            date_column="date", groupby_columns=["g"]
        )
        with pytest.raises(RuntimeError):
            gen.transform(pd.DataFrame({"date": [], "g": [], "v": []}))

    def test_get_metadata(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "store_id": ["S1"] * 10,
            "sales": np.random.randn(10),
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            windows=(7,),
            aggregations=("mean",),
            backend="pandas",
        )
        gen.fit(df)
        meta = gen.get_metadata()
        assert meta["generator_type"] == "temporal_aggregation"
        assert meta["date_column"] == "date"

    def test_numeric_columns_subset(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "store_id": ["S1"] * 20,
            "sales": np.random.randn(20) + 100,
            "discount": np.random.rand(20),
        })
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["store_id"],
            numeric_columns=["sales"],
            windows=(7,),
            aggregations=("mean",),
            backend="pandas",
        )
        gen.fit(df)
        names = gen.get_feature_names()
        # Only sales should be aggregated, not discount
        for n in names:
            assert "sales" in n
            assert "discount" not in n

    def test_supports_jit_pandas(self):
        gen = TemporalAggregationGenerator(
            date_column="date",
            groupby_columns=["g"],
            backend="pandas",
        )
        assert gen.supports_jit is False