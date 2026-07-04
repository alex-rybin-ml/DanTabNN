"""Tests for base.py, binary.py, regression.py, and multiclass.py pipelines."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression

from dantabnn.base import BaseNNPipeline
from dantabnn.binary import BinaryClassificationPipeline
from dantabnn.regression import RegressionPipeline
from dantabnn.multiclass import MulticlassClassificationPipeline


# ---------------------------------------------------------------------------
# Helper to generate synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_binary_df(n_samples=100, n_features=8, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y.astype(float)
    return df


def _make_regression_df(n_samples=100, n_features=8, random_state=42):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y
    return df


def _make_multiclass_df(n_samples=150, n_features=8, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(3, n_features // 2),
        n_classes=n_classes,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["target"] = y.astype(int)
    return df


def _make_mixed_df(n_samples=100, random_state=42):
    """DataFrame with both numeric and categorical features."""
    rng = np.random.RandomState(random_state)
    df = pd.DataFrame()
    df["num1"] = rng.randn(n_samples)
    df["num2"] = rng.randn(n_samples)
    df["cat1"] = rng.choice(["a", "b", "c"], size=n_samples)
    df["cat2"] = rng.choice(["x", "y"], size=n_samples)
    df["target"] = (
        df["num1"] * 0.5 + df["num2"] * 0.3 + (df["cat1"] == "a").astype(float) * 0.5
    )
    return df


# ---------------------------------------------------------------------------
# BaseNNPipeline — abstract method testing via concrete subclass
# ---------------------------------------------------------------------------

class _MinimalPipeline(BaseNNPipeline):
    """Concrete subclass for testing BaseNNPipeline."""

    def _build_model(self, input_dim, output_dim):
        return torch.nn.Linear(input_dim, output_dim)

    def _get_loss_fn(self):
        return torch.nn.MSELoss()

    def _get_metrics(self):
        from typing import Dict, Callable
        return {"mse": lambda t, p: float(np.mean((t - p) ** 2))}

    def _get_output_dim(self, y):
        return 1


class TestBaseNNPipeline:
    def test_init_sets_all_attributes(self):
        pipe = _MinimalPipeline(
            numeric_features=["a", "b"],
            categorical_features=["c"],
            target_column="y",
        )
        assert pipe.numeric_features == ["a", "b"]
        assert pipe.categorical_features == ["c"]
        assert pipe.target_column == "y"
        assert pipe.hidden_dims == [32, 16, 8]  # auto-computed from 2 numeric features
        assert pipe.dropout == 0.2
        assert pipe.batch_size == 32
        assert pipe.epochs == 100
        assert pipe.learning_rate == 1e-3
        assert not pipe.is_fitted

    def test_device_fallback_to_cpu(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            device="cpu",
        )
        assert pipe.device == "cpu"

    def test_set_seed_reproducibility(self):
        pipe1 = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            random_state=123,
        )
        pipe2 = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            random_state=123,
        )
        # Both should produce same numpy random state after init
        np1 = np.random.get_state()
        np2 = np.random.get_state()
        assert np.array_equal(np1[1], np2[1])

    def test_set_params_updates_attributes(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        pipe.set_params(dropout=0.5, epochs=200)
        assert pipe.dropout == 0.5
        assert pipe.epochs == 200

    def test_set_params_ignores_unknown(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        with pytest.warns(UserWarning, match="Ignoring unknown parameter"):
            pipe.set_params(nonexistent=999)

    def test_hyperparameters_property_excludes_internal(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
        )
        pipe.is_fitted = True
        params = pipe.hyperparameters
        assert "model" not in params
        assert "is_fitted" not in params
        assert "history" not in params
        assert "best_state" not in params
        assert "dropout" in params

    def test_fit_sets_is_fitted(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        assert pipe.is_fitted

    def test_fit_with_validation(self):
        df = _make_regression_df()
        train = df.iloc[:70]
        val = df.iloc[70:]
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(train, df_val=val)
        assert pipe.is_fitted
        assert "val_loss" in pipe.history

    def test_predict_before_fit_raises(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.predict(pd.DataFrame({"a": [1.0], "y": [0.0]}))

    def test_predict_returns_correct_shape(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        preds = pipe.predict(df)
        assert preds.shape[0] == len(df)

    def test_evaluate_returns_dict(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        result = pipe.evaluate(df)
        assert isinstance(result, dict)
        assert "mse" in result

    def test_evaluate_with_custom_metrics(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        result = pipe.evaluate(df, metrics=["mse"])
        assert "mse" in result

    def test_save_and_load(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline"
            pipe.save(path)
            assert (path / "model.pt").exists()
            assert (path / "hyperparameters.joblib").exists()

            # Load into new instance
            pipe2 = _MinimalPipeline(
                numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
                categorical_features=[],
                target_column="target",
            )
            pipe2.load(path)
            assert pipe2.is_fitted
            assert pipe2.model is not None

    def test_get_feature_importance_returns_dataframe(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        imp = pipe.get_feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) == len(pipe.feature_names)

    def test_get_feature_importance_before_fit_raises(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.get_feature_importance()

    def test_get_model_returns_nn_module(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        model = pipe.get_model()
        assert isinstance(model, torch.nn.Module)

    def test_get_model_before_build_raises(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
        )
        with pytest.raises(RuntimeError, match="not built"):
            pipe.get_model()

    def test_get_preprocessor_returns_dict(self):
        df = _make_regression_df()
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=5,
        )
        pipe.fit(df)
        prep = pipe.get_preprocessor()
        assert isinstance(prep, dict)
        assert "scaler" in prep
        assert "encoder" in prep

    def test_prepare_features_no_numeric_no_categorical(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            epochs=5,
        )
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        features, names = pipe._prepare_features(df, fit=True)
        assert features.shape == (3, 0)

    def test_prepare_features_with_numeric_only(self):
        pipe = _MinimalPipeline(
            numeric_features=["a", "b"], categorical_features=[], target_column="y",
            epochs=5,
            scale_numeric=True,
            engineer_features=False,  # v7 default adds x², test expects raw count
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0.0, 1.0]})
        features, names = pipe._prepare_features(df, fit=True)
        assert features.shape == (2, 2)
        assert "num_a" in names
        assert "num_b" in names

    def test_prepare_features_with_categorical_only(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=["cat"], target_column="y",
            epochs=5,
            encode_categorical=True,
        )
        df = pd.DataFrame({"cat": ["a", "b", "a"], "y": [0.0, 1.0, 1.0]})
        features, names = pipe._prepare_features(df, fit=True)
        assert features.shape == (3, 2)  # one-hot: 2 categories
        assert "cat_cat" in names[0]

    def test_prepare_features_no_scaling(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
            scale_numeric=False,
            engineer_features=False,  # v7 default adds x², test expects raw count
        )
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 1.0]})
        features, _ = pipe._prepare_features(df, fit=True)
        # Not scaled, but still numeric
        assert features.shape == (3, 1)

    def test_prepare_target_returns_tensor(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        df = pd.DataFrame({"y": [1.0, 2.0]})
        t = pipe._prepare_target(df)
        assert isinstance(t, torch.Tensor)

    def test_prepare_data_returns_triple(self):
        pipe = _MinimalPipeline(
            numeric_features=["a"], categorical_features=[], target_column="y",
        )
        df = pd.DataFrame({"a": [1.0, 2.0], "y": [0.0, 1.0]})
        features, target, names = pipe._prepare_data(df, fit=True)
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(names, list)

    def test_create_dataloader_with_target(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        features = torch.randn(10, 4)
        target = torch.randn(10, 1)
        loader = pipe._create_dataloader(features, target, shuffle=False)
        assert len(loader.dataset) == 10

    def test_create_dataloader_without_target(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        features = torch.randn(10, 4)
        loader = pipe._create_dataloader(features, target=None, shuffle=False)
        assert len(loader.dataset) == 10

    def test_repr(self):
        pipe = _MinimalPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        r = repr(pipe)
        assert "_MinimalPipeline" in r
        assert "target_column" in r

    def test_mixed_numeric_categorical(self):
        df = _make_mixed_df()
        pipe = _MinimalPipeline(
            numeric_features=["num1", "num2"],
            categorical_features=["cat1", "cat2"],
            target_column="target",
            epochs=5,
            scale_numeric=True,
            encode_categorical=True,
        )
        pipe.fit(df)
        assert pipe.is_fitted
        preds = pipe.predict(df)
        assert preds.shape[0] == len(df)

    def test_early_stopping_saves_best_state(self):
        df = _make_regression_df()
        train = df.iloc[:80]
        val = df.iloc[80:]
        pipe = _MinimalPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"],
            categorical_features=[],
            target_column="target",
            epochs=20,
            early_stopping_patience=5,
        )
        pipe.fit(train, df_val=val)
        assert pipe.best_state is not None
        assert pipe.best_epoch >= 0


# ---------------------------------------------------------------------------
# BinaryClassificationPipeline
# ---------------------------------------------------------------------------

class TestBinaryClassificationPipeline:
    def test_fit_predict_cycle(self):
        df = _make_binary_df()
        pipe = BinaryClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        probs = pipe.predict(df)
        assert probs.shape[0] == len(df)
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_classes(self):
        df = _make_binary_df()
        pipe = BinaryClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        cls = pipe.predict_classes(df)
        assert cls.shape[0] == len(df)
        assert set(np.unique(cls)).issubset({0, 1})

    def test_predict_classes_custom_threshold(self):
        df = _make_binary_df()
        pipe = BinaryClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        cls_high = pipe.predict_classes(df, threshold=0.9)
        cls_low = pipe.predict_classes(df, threshold=0.1)
        # Higher threshold should select fewer positives
        assert cls_high.sum() <= cls_low.sum()

    def test_evaluate_returns_metrics(self):
        df = _make_binary_df()
        pipe = BinaryClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        result = pipe.evaluate(df)
        assert "accuracy" in result
        assert "roc_auc" in result

    def test_get_metrics_contains_expected(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        metrics = pipe._get_metrics()
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        from typing import Callable
        for v in metrics.values():
            assert callable(v)

    def test_get_loss_fn_is_bce(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        loss = pipe._get_loss_fn()
        assert isinstance(loss, torch.nn.BCEWithLogitsLoss)


# ---------------------------------------------------------------------------
# RegressionPipeline
# ---------------------------------------------------------------------------

class TestRegressionPipeline:
    def test_fit_predict_cycle(self):
        df = _make_regression_df()
        pipe = RegressionPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        preds = pipe.predict(df)
        assert preds.shape[0] == len(df)
        assert preds.shape[1] == 1  # single-output regression

    def test_evaluate_returns_metrics(self):
        df = _make_regression_df()
        pipe = RegressionPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        result = pipe.evaluate(df)
        assert "mse" in result
        assert "mae" in result
        assert "r2" in result

    def test_get_loss_fn_is_huber(self):
        pipe = RegressionPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        loss = pipe._get_loss_fn()
        assert isinstance(loss, (torch.nn.MSELoss, torch.nn.HuberLoss))


# ---------------------------------------------------------------------------
# MulticlassClassificationPipeline
# ---------------------------------------------------------------------------

class TestMulticlassClassificationPipeline:
    def test_fit_predict_cycle(self):
        df = _make_multiclass_df(n_classes=3)
        pipe = MulticlassClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            n_classes=3,
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        probs = pipe.predict(df)
        assert probs.shape == (len(df), 3)
        # Each row should sum to 1 (softmax)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_classes_returns_int_labels(self):
        df = _make_multiclass_df(n_classes=3)
        pipe = MulticlassClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            n_classes=3,
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        cls = pipe.predict_classes(df)
        assert cls.shape[0] == len(df)
        assert set(np.unique(cls)).issubset({0, 1, 2})

    def test_evaluate_returns_metrics(self):
        df = _make_multiclass_df(n_classes=3)
        pipe = MulticlassClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            n_classes=3,
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        result = pipe.evaluate(df)
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "f1_weighted" in result

    def test_get_loss_fn_is_cross_entropy(self):
        pipe = MulticlassClassificationPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            n_classes=5,
        )
        loss = pipe._get_loss_fn()
        assert isinstance(loss, torch.nn.CrossEntropyLoss)

    def test_get_output_dim_returns_n_classes(self):
        pipe = MulticlassClassificationPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
            n_classes=7,
        )
        dim = pipe._get_output_dim(torch.zeros(0))
        assert dim == 7

    def test_more_classes(self):
        df = _make_multiclass_df(n_classes=5)
        pipe = MulticlassClassificationPipeline(
            numeric_features=[f"f{i}" for i in range(8)],
            categorical_features=[],
            target_column="target",
            n_classes=5,
            epochs=5,
            hidden_dims=[16, 8],
        )
        pipe.fit(df)
        probs = pipe.predict(df)
        assert probs.shape == (len(df), 5)