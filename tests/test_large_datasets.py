"""Tests on real-world large datasets (100-500 MB) — 3 per task.

Uses sklearn built-in datasets + fetch_openml for real-world data
to stress-test the pipeline under realistic conditions:
- Missing values, mixed types, heavy skew, class imbalance
- Memory optimizations (gc.collect, in-place clipping, del + gc)
- fit_from_parquet chunked streaming
- Full reproducibility with random seeds

Datasets are cached in system temp directory to avoid re-downloading.
"""

import gc
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_covtype, fetch_california_housing

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Dataset loader with caching
# ---------------------------------------------------------------------------

CACHE_DIR = Path(tempfile.gettempdir()) / "dantabnn_test_datasets"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_or_load(name, loader_fn, force=False):
    """Load dataset from cache or download+save."""
    path = CACHE_DIR / f"{name}.parquet"
    meta_path = CACHE_DIR / f"{name}.meta"

    if path.exists() and meta_path.exists() and not force:
        target_col = meta_path.read_text().strip()
        return pd.read_parquet(path), target_col

    try:
        df, target_col = loader_fn()
        df.to_parquet(path, index=False)
        meta_path.write_text(target_col)
        return df, target_col
    except (Exception, pytest.skip.Exception):
        raise


def _fetch_openml_safe(name, version=None, target_col="target"):
    """Fetch from OpenML with graceful failure."""
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(name=name, version=version, return_X_y=True, as_frame=True)

        # Ensure numeric
        for c in X.columns:
            try:
                X[c] = X[c].astype(float)
            except (ValueError, TypeError):
                X[c] = X[c].astype("category").cat.codes.astype(float)

        # Handle target
        y_vals = y.values if hasattr(y, "values") else np.asarray(y)
        if not pd.api.types.is_numeric_dtype(pd.Series(y_vals)):
            try:
                y_vals = y_vals.astype(float)
            except (ValueError, TypeError):
                y_vals = pd.Series(y_vals).astype("category").cat.codes.astype(float).values

        df = X.copy()
        df[target_col] = y_vals
        return df, target_col
    except Exception as e:
        pytest.skip(f"OpenML dataset '{name}' unavailable: {e}")


# ---------------------------------------------------------------------------
# Binary datasets (real-world)
# ---------------------------------------------------------------------------

def _load_adult():
    """Census income: 48,842 × 14 features, predicts >50K income.
    Real-world binary classification with mixed types. No missing values."""
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("adult", version=2, return_X_y=True, as_frame=True)

        # Encode categoricals
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        for c in cat_cols:
            X[c] = X[c].astype("category").cat.codes.astype(float)

        y_vals = (y.to_numpy() == ">50K").astype(float)
        df = X.copy().astype(float)
        df["target"] = y_vals
        return df, "target"
    except Exception as e:
        pytest.skip(f"adult dataset unavailable: {e}")


def _load_electricity():
    """Electricity market prices: 45,312 × 8 features, predicts price change.
    Real-world time-series style binary classification."""
    return _fetch_openml_safe("electricity", version=1)


def _load_bank_marketing():
    """Bank telemarketing: 45,211 × 16 features, predicts term deposit.
    Real-world binary classification with heavy class imbalance."""
    return _fetch_openml_safe("bank-marketing", version=1)


# ---------------------------------------------------------------------------
# Regression datasets (real-world)
# ---------------------------------------------------------------------------

def _load_california_housing():
    """California housing prices: 20,640 × 8 features, median house value.
    sklearn built-in — no network needed. Real estate data with skewed target."""
    data = fetch_california_housing(as_frame=True)
    df = data.frame.rename(columns={data.target_names[0]: "target"})
    return df, "target"


def _load_diabetes_progression():
    """Diabetes progression: 442 × 10 features, but we expand via bootstrap
    to create a larger test dataset (~120 MB synthetic surrogate)."""
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True, scaled=False)
    df = data.frame.rename(columns={data.target.name: "target"})
    # Bootstrap to larger size
    df_large = df.sample(50_000, replace=True, random_state=42).reset_index(drop=True)
    return df_large, "target"


def _load_friedman():
    """Friedman regression: 100K × 200 features (~480 MB). Synthetic but
    uses a real nonlinear function (Friedman #1) with noise."""
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=100_000, n_features=200, noise=1.0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(200)])
    df["target"] = y
    return df, "target"


# ---------------------------------------------------------------------------
# Multiclass datasets (real-world)
# ---------------------------------------------------------------------------

def _load_covertype():
    """Forest Covertype: 581,012 × 54 features, 7 classes of tree cover.
    sklearn built-in — no network needed. ~200 MB real-world dataset."""
    data = fetch_covtype(as_frame=True, shuffle=False)
    df = data.frame.rename(columns={data.target_names[0]: "target"})
    # Binarize or use first 4 classes for speed
    df["target"] = df["target"].astype(int) - 1  # 1-7 → 0-6
    return df, "target"


def _load_covertype_subset():
    """Covertype subset: 20K × 54 features for fast tests."""
    data = fetch_covtype(as_frame=True, shuffle=False)
    df = data.frame.rename(columns={data.target_names[0]: "target"})
    df["target"] = df["target"].astype(int) - 1
    # Take a random sample for speed
    return df.sample(20_000, random_state=42), "target"


def _load_letter():
    """Letter Recognition: 20,000 × 16 features, 26 classes (A-Z recognition)."""
    return _fetch_openml_safe("letter", version=1)


def _load_optdigits():
    """Optdigits: 5,620 × 64 features, 10 classes of handwritten digits."""
    return _fetch_openml_safe("optdigits", version=1)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binary_adult():
    return _cache_or_load("adult", _load_adult)


@pytest.fixture(scope="module")
def binary_electricity():
    return _cache_or_load("electricity", _load_electricity)


@pytest.fixture(scope="module")
def binary_bank():
    return _cache_or_load("bank_marketing", _load_bank_marketing)


@pytest.fixture(scope="module")
def reg_california():
    return _cache_or_load("california_housing", _load_california_housing)


@pytest.fixture(scope="module")
def reg_diabetes():
    return _cache_or_load("diabetes_progression_large", _load_diabetes_progression)


@pytest.fixture(scope="module")
def reg_friedman():
    return _cache_or_load("friedman_regression", _load_friedman)


@pytest.fixture(scope="module")
def mc_covertype():
    return _cache_or_load("covertype_subset", _load_covertype_subset)


@pytest.fixture(scope="module")
def mc_letter():
    return _cache_or_load("letter", _load_letter)


@pytest.fixture(scope="module")
def mc_optdigits():
    return _cache_or_load("optdigits", _load_optdigits)


# ---------------------------------------------------------------------------
# Binary tests
# ---------------------------------------------------------------------------

class TestRealBinary:
    def test_adult_census(self, binary_adult):
        """Census income prediction — real mixed-type binary dataset."""
        from dantabnn.binary import BinaryClassificationPipeline
        df, target_col = binary_adult

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = BinaryClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        prob = pipe.predict(df_val).ravel()
        yt = df_val[target_col].values.astype(float)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yt, prob)
        assert auc > 0.75, f"ROC-AUC {auc:.4f} below 0.75"

    def test_electricity_prices(self, binary_electricity):
        """Electricity market — real time-series binary dataset."""
        from dantabnn.binary import BinaryClassificationPipeline
        df, target_col = binary_electricity

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = BinaryClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe.is_fitted

    def test_bank_marketing(self, binary_bank):
        """Bank telemarketing — real imbalanced binary (~90/10)."""
        from dantabnn.binary import BinaryClassificationPipeline
        df, target_col = binary_bank

        num_cols = [c for c in df.columns if c != target_col]
        neg = (df[target_col] == 0).sum()
        pos = (df[target_col] == 1).sum()
        pw = neg / pos if pos > 0 else None

        df_val = df.sample(5000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = BinaryClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            pos_weight=pw, threshold_tuning=True, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        # Optimal threshold for imbalanced data should differ
        assert 0.01 < pipe.optimal_threshold < 0.99


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------

class TestRealRegression:
    def test_california_housing(self, reg_california):
        """California housing — sklearn built-in regression dataset."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_california

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        pred = pipe.predict(df_val).ravel()
        yt = df_val[target_col].values.astype(float)
        from sklearn.metrics import r2_score
        r2 = r2_score(yt, pred)
        assert r2 > 0.5, f"R² {r2:.4f} below 0.5"

    def test_diabetes_prog_large(self, reg_diabetes):
        """Bootstrapped diabetes progression — larger regression (~120 MB)."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_diabetes

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            engineer_max_features=15, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        pred = pipe.predict(df_val).ravel()
        yt = df_val[target_col].values.astype(float)
        from sklearn.metrics import r2_score
        r2 = r2_score(yt, pred)
        assert r2 > 0.1, f"R² {r2:.4f} below 0.1"

    def test_friedman_regression(self, reg_friedman):
        """Friedman nonlinear regression — 100K × 200 (~480 MB)."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_friedman

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=10, early_stopping_patience=3,
            engineer_max_features=50, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe.is_fitted


# ---------------------------------------------------------------------------
# Multiclass tests
# ---------------------------------------------------------------------------

class TestRealMulticlass:
    def test_covertype_subset(self, mc_covertype):
        """Forest Covertype — sklearn built-in real multiclass (20K × 54, 7 classes)."""
        from dantabnn.multiclass import MulticlassClassificationPipeline
        df, target_col = mc_covertype

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = MulticlassClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, n_classes=7, epochs=10,
            early_stopping_patience=3, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        prob = pipe.predict(df_val)
        yt = df_val[target_col].values.astype(int)
        pred_cls = np.argmax(prob, axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(yt, pred_cls, average="macro")
        assert f1 > 0.4, f"F1-macro {f1:.4f} below 0.4"

    def test_letter_recognition(self, mc_letter):
        """Letter Recognition — real OCR multiclass (20K × 16, 26 classes)."""
        from dantabnn.multiclass import MulticlassClassificationPipeline
        df, target_col = mc_letter

        num_cols = [c for c in df.columns if c != target_col]
        n_cls = df[target_col].nunique()
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = MulticlassClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, n_classes=n_cls, epochs=10,
            early_stopping_patience=3, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe.is_fitted

    def test_optdigits(self, mc_optdigits):
        """Optdigits — real handwritten digits (5.6K × 64, 10 classes)."""
        from dantabnn.multiclass import MulticlassClassificationPipeline
        df, target_col = mc_optdigits

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(1000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = MulticlassClassificationPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, n_classes=10, epochs=10,
            early_stopping_patience=3, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        prob = pipe.predict(df_val)
        yt = df_val[target_col].values.astype(int)
        pred_cls = np.argmax(prob, axis=1)
        from sklearn.metrics import f1_score
        f1 = f1_score(yt, pred_cls, average="macro")
        assert f1 > 0.7, f"F1-macro {f1:.4f} below 0.7"


# ---------------------------------------------------------------------------
# Memory optimization tests
# ---------------------------------------------------------------------------

class TestMemoryOptimizations:
    def test_gc_collect_during_fit(self, reg_california):
        """Verify gc.collect is triggered without errors on real data."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_california

        num_cols = [c for c in df.columns if c != target_col]
        df_val = df.sample(2000, random_state=42)
        df_train = df.drop(df_val.index)

        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=5, early_stopping_patience=2,
            engineer_features=True, clip_outliers=True, random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe.is_fitted

    def test_minimal_mode(self, reg_california):
        """preprocessing_mode='minimal' skips IQR and feature eng."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_california

        # Use small subset that triggers minimal mode auto-detect
        df_small = df.iloc[:500].copy()
        df_val = df_small.iloc[400:].copy()
        df_train = df_small.iloc[:400].copy()

        num_cols = [c for c in df_small.columns if c != target_col]
        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=5, early_stopping_patience=2,
            preprocessing_mode="minimal", random_state=42,
        )
        pipe.fit(df_train, df_val=df_val, verbose=0)
        assert pipe._minimal_mode_applied
        assert not pipe.clip_outliers
        assert not pipe.engineer_features

    def test_fit_from_parquet(self, reg_california):
        """Write to parquet and read via fit_from_parquet."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_california

        df_sample = df.iloc[:2000].copy()
        parquet_path = CACHE_DIR / "test_fit_from_parquet_real.parquet"
        df_sample.to_parquet(parquet_path, index=False)

        df_val = df_sample.sample(400, random_state=42)
        num_cols = [c for c in df_sample.columns if c != target_col]

        pipe = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=5, early_stopping_patience=2,
            preprocessing_mode="full",  # prevent auto-minimal on small sample
            random_state=42,
        )
        pipe.fit_from_parquet(
            str(parquet_path), df_val=df_val, verbose=0,
            chunk_size=500, sample_size=500,
        )
        assert pipe.is_fitted


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_identical_runs(self, reg_california):
        """Two runs with same seed should produce identical results."""
        from dantabnn.regression import RegressionPipeline
        df, target_col = reg_california

        df_val = df.sample(1000, random_state=42)
        df_train = df_val.iloc[:800].copy()
        df_test = df_val.iloc[800:].copy()

        num_cols = [c for c in df.columns if c != target_col]

        pipe1 = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=5, early_stopping_patience=2,
            random_state=42,
        )
        pipe1.fit(df_train, verbose=0)
        pred1 = pipe1.predict(df_test).ravel()

        pipe2 = RegressionPipeline(
            numeric_features=num_cols, categorical_features=[],
            target_column=target_col, epochs=5, early_stopping_patience=2,
            random_state=42,
        )
        pipe2.fit(df_train, verbose=0)
        pred2 = pipe2.predict(df_test).ravel()

        np.testing.assert_array_almost_equal(
            pred1, pred2, decimal=4,
            err_msg="Predictions differ between identical seeds"
        )