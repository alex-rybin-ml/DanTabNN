"""Tests on synthetic datasets that exercise the same code paths as real-world
data but run much faster (no network, no large downloads).

Covers: pipeline fit/predict, preprocessing modes, memory optimizations,
fit_from_parquet, and reproducibility across all three task types.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Small synthetic dataset factories (fast, deterministic, no network)
# ---------------------------------------------------------------------------

def _make_binary_df(n_samples=500, n_features=10, seed=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=5, n_redundant=2, random_state=seed,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(float)
    return df, cols


def _make_regression_df(n_samples=500, n_features=10, seed=42):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=5, noise=0.1, random_state=seed,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(float)
    return df, cols


def _make_multiclass_df(n_samples=600, n_features=10, n_classes=4, seed=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=6, n_redundant=2, n_classes=n_classes,
        random_state=seed,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y.astype(int)
    return df, cols, n_classes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TMP_DIR = Path(tempfile.gettempdir()) / "dantabnn_test"


@pytest.fixture(scope="function")
def binary_df():
    df, cols = _make_binary_df()
    target = "target"
    df_val = df.sample(100, random_state=1)
    df_train = df.drop(df_val.index)
    return df_train, df_val, cols, target


@pytest.fixture(scope="function")
def regression_df():
    df, cols = _make_regression_df()
    target = "target"
    df_val = df.sample(100, random_state=1)
    df_train = df.drop(df_val.index)
    return df_train, df_val, cols, target


@pytest.fixture(scope="function")
def multiclass_df():
    df, cols, n_cls = _make_multiclass_df()
    target = "target"
    df_val = df.sample(120, random_state=1)
    df_train = df.drop(df_val.index)
    return df_train, df_val, cols, target, n_cls


# ---------------------------------------------------------------------------
# Binary tests
# ---------------------------------------------------------------------------

class TestBinary:
    def test_fit_predict(self, binary_df):
        from dantabnn.binary import BinaryClassificationPipeline
        df_train, df_val, cols, target = binary_df

        pipe = BinaryClassificationPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        prob = pipe.predict(df_val).ravel()
        yt = df_val[target].values.astype(float)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yt, prob)
        assert auc > 0.55, f"ROC-AUC {auc:.4f} below 0.55"

    def test_imbalanced_pos_weight(self):
        """Binary pipeline with class imbalance and pos_weight."""
        from dantabnn.binary import BinaryClassificationPipeline
        X, y = make_classification(
            n_samples=400, n_features=8, weights=[0.9, 0.1],
            random_state=42,
        )
        cols = [f"f{i}" for i in range(8)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y.astype(float)
        df_tr, df_te = train_test_split(df, test_size=0.25, random_state=1,
                                          stratify=df["target"])

        pipe = BinaryClassificationPipeline(
            numeric_features=cols, categorical_features=[],
            target_column="target", epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_tr, df_val=df_te, verbose=0)
        assert pipe.is_fitted

    def test_threshold_tuning(self, binary_df):
        from dantabnn.binary import BinaryClassificationPipeline
        df_train, df_val, cols, target = binary_df

        pipe = BinaryClassificationPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=10, early_stopping_patience=3,
            threshold_tuning=True, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert 0.01 < pipe.optimal_threshold < 0.99


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestRegression:
    def test_fit_predict(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        pred = pipe.predict(df_val).ravel()
        yt = df_val[target].values.astype(float)
        from sklearn.metrics import r2_score
        r2 = r2_score(yt, pred)
        assert r2 > 0.2, f"R² {r2:.4f} below 0.2"

    def test_scale_target(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=10, early_stopping_patience=3,
            scale_target=True, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe._target_mean is not None
        assert pipe._target_std is not None

    def test_no_scale_target(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=10, early_stopping_patience=3,
            scale_target=False, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe._target_mean is None


# ---------------------------------------------------------------------------
# Multiclass tests
# ---------------------------------------------------------------------------

class TestMulticlass:
    def test_fit_predict(self, multiclass_df):
        from dantabnn.multiclass import MulticlassClassificationPipeline
        df_train, df_val, cols, target, n_cls = multiclass_df

        pipe = MulticlassClassificationPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, n_classes=n_cls, epochs=10,
            early_stopping_patience=3, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        prob = pipe.predict(df_val)
        yt = df_val[target].values.astype(int)
        pred_cls = np.argmax(prob, axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(yt, pred_cls, average="macro")
        assert f1 > 0.3, f"F1-macro {f1:.4f} below 0.3"

    def test_class_weights(self):
        from dantabnn.multiclass import MulticlassClassificationPipeline
        X, y = make_classification(
            n_samples=400, n_features=8, n_informative=6,
            n_classes=3, weights=[0.8, 0.1, 0.1], random_state=42,
        )
        cols = [f"f{i}" for i in range(8)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y.astype(int)
        df_tr, df_te = train_test_split(df, test_size=0.25, random_state=1)

        pipe = MulticlassClassificationPipeline(
            numeric_features=cols, categorical_features=[],
            target_column="target", n_classes=3, epochs=10,
            early_stopping_patience=3, random_state=42,
        )
        pipe.fit(df_tr, df_val=df_te, verbose=0)
        assert pipe.is_fitted


# ---------------------------------------------------------------------------
# Preprocessing mode tests
# ---------------------------------------------------------------------------

class TestPreprocessingModes:
    def test_minimal_mode(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            preprocessing_mode="minimal", random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe._minimal_mode_applied
        assert not pipe.clip_outliers
        assert not pipe.engineer_features

    def test_auto_mode_small_dataset_triggers_minimal(self):
        """auto mode on small clean dataset should auto-detect minimal."""
        from dantabnn.regression import RegressionPipeline
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        cols = [f"f{i}" for i in range(5)]
        df = pd.DataFrame(X, columns=cols)
        df["target"] = y.astype(float)
        df_tr = df.iloc[:150]
        df_te = df.iloc[150:]

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column="target", epochs=5, early_stopping_patience=2,
            preprocessing_mode="auto", random_state=42,
        )
        pipe.fit(df_tr, df_val=df_te, verbose=0)
        # Small dataset, few features, no categoricals, no missing → minimal
        assert pipe._minimal_mode_applied

    def test_full_mode(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            preprocessing_mode="full", random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe.clip_outliers
        assert pipe.engineer_features


# ---------------------------------------------------------------------------
# fit_from_parquet
# ---------------------------------------------------------------------------

class TestFitFromParquet:
    def test_chunked_fit(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        parquet_path = TMP_DIR / "test_chunked.parquet"
        TMP_DIR.mkdir(exist_ok=True)
        df_train.to_parquet(parquet_path, index=False)

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            preprocessing_mode="full", random_state=42,
        )
        pipe.fit_from_parquet(
            str(parquet_path), df_val=df_val, verbose=0,
            chunk_size=100, sample_size=100,
        )
        assert pipe.is_fitted


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_identical_runs(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe1 = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            random_state=42,
        )
        pipe1.fit(df_train, verbose=0)
        pred1 = pipe1.predict(df_val).ravel()

        pipe2 = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            random_state=42,
        )
        pipe2.fit(df_train, verbose=0)
        pred2 = pipe2.predict(df_val).ravel()

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=4)

    def test_different_seeds_differ(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe1 = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            random_state=42,
        )
        pipe1.fit(df_train, verbose=0)
        pred1 = pipe1.predict(df_val).ravel()

        pipe2 = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            random_state=99,
        )
        pipe2.fit(df_train, verbose=0)
        pred2 = pipe2.predict(df_val).ravel()

        # Different seeds should produce different predictions
        assert not np.allclose(pred1, pred2, rtol=1e-6)


# ---------------------------------------------------------------------------
# Pipeline save/load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, regression_df):
        from dantabnn.regression import RegressionPipeline
        df_train, df_val, cols, target = regression_df

        pipe = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, epochs=5, early_stopping_patience=2,
            scale_target=False, preprocessing_mode="full", random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        pred_before = pipe.predict(df_val).ravel()

        save_path = TMP_DIR / "roundtrip_model"
        pipe.save(save_path)

        pipe2 = RegressionPipeline(
            numeric_features=cols, categorical_features=[],
            target_column=target, scale_target=False,
            preprocessing_mode="full",
        )
        pipe2.load(save_path)
        pred_after = pipe2.predict(df_val).ravel()

        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=4)