# DanTabNN — Dual-Attention Neural Networks for Tabular Data

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-230%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-78%25-green)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

PyTorch pipeline for tabular classification and regression with **Dual-Attention Networks**, automated preprocessing, memory-efficient streaming, and Optuna hyperparameter tuning.

---

## Installation

```bash
git clone https://github.com/alex-rybin-ml/DanTabNN.git && cd DanTabNN
uv sync
```

---

## Quick Start

```python
from dantabnn.regression import RegressionPipeline

pipe = RegressionPipeline(
    numeric_features=["age", "income", "rooms"],
    categorical_features=["city"],
    target_column="price",
)
pipe.fit(df_train)                    # auto-runs: impute → clip outliers → x²+log1p → scale → encode
preds = pipe.predict(df_test)

# Hyperparameter tuning with CV
best = pipe.hyperparameters_tuning(df_train,
    n_iter=500, n_jobs=14, direction="maximize")
print(best.hyperparameters)
```

**Binary with imbalanced classes**:
```python
from dantabnn.binary import BinaryClassificationPipeline

pipe = BinaryClassificationPipeline(
    numeric_features=[...], categorical_features=[...],
    target_column="churn",
    pos_weight=9.0,              # for 90/10 imbalance
    threshold_tuning=True,        # auto-find optimal F2 threshold
)
pipe.fit(df_train, df_val=df_val)
print(f"Optimal threshold: {pipe.optimal_threshold:.3f}")
classes = pipe.predict_classes(df_test)  # uses optimal threshold
```

**Large datasets (10M+ rows) — chunked from Parquet**:
```python
pipe = RegressionPipeline(...)
pipe.fit_from_parquet(
    "/data/30gb_dataset.parquet",   # local or S3 path
    df_val=df_val,                  # optional validation data
    chunk_size=100_000,             # rows per chunk (peak memory ≈ 15 GB for 30 GB dataset)
    sample_size=100_000,            # rows for fitting preprocessor statistics
)
preds = pipe.predict(df_test)
```

**Other tasks**: `BinaryClassificationPipeline`, `MulticlassClassificationPipeline` — same API.

---

## Architecture

```
DataFrame → [NaNImputer → OutlierClipper → AutoFeatureEngineer → StandardScaler → OneHotEncoder]
         → Feature Gating → Feature Attention → Cross Network → Feed-Forward → Output
```

| Component | Purpose |
|-----------|---------|
| Preprocessing chain | IQR outlier clipping → median imputation → x²+log1p engineering → scale → encode |
| `preprocessing_mode="auto"` | Auto-skips IQR + feature eng on small/clean datasets (n<1000, d<20) |
| Feature Gating | Gumbel-Softmax differentiable feature selection |
| Feature Attention | 4-head self-attention across feature dimensions |
| Cross Network | Explicit DCN-style pairwise feature crosses |

---

## Benchmark Results (500-trial Optuna tuning, EarlyStoppingCallback + MedianPruner)

| Dataset | Task | Baseline (v2) | v7 (20tr) | v10 (500tr) |
|---------|------|---------------|-----------|-------------|
| Boston Housing | R² | 0.628 | 0.816 | **0.855** |
| Diabetes Progression | R² | 0.362 | 0.509 | **0.548** |
| Energy Efficiency | R² | 0.845 | 0.989 | **0.992** |
| Wine Quality | R² | -0.023 | 0.479 | **0.512** |
| Breast Cancer | AUC | 0.994 | 0.996 | **0.997** |
| German Credit | AUC | 0.703 | 0.801 | **0.815** |
| Pima Diabetes | AUC | 0.823 | 0.822 | **0.828** |
| Spambase | AUC | 0.972 | 0.986 | **0.989** |
| Iris | F1 | 0.898 | 0.967 | **1.000** |
| Digits | F1 | 0.964 | 0.980 | **0.983** |
| Vehicle | F1 | 0.940 | 0.985 | **0.992** |
| Segment | F1 | 0.978 | 1.000 | **1.000** |
| Wine | F1 | 1.000 | 1.000 | **1.000** |

> 500 trials with EarlyStoppingCallback (100-round patience) and MedianPruner consistently outperforms 20 trials (+0.01-0.04).

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[128,64,32]` | 3-layer MLP (adaptive per dataset) |
| `dropout` | 0.2 | Dropout rate |
| `gating_type` | `"soft"` | Feature gating: `"soft"` or `"none"` |
| `clip_outliers` | `True` | IQR winsorization [Q1-1.5×IQR, Q3+1.5×IQR] (in-place for memory) |
| `impute_missing` | `True` | Median imputation (handles pd.NA + np.nan) |
| `engineer_features` | `True` | x² for all features + log1p for |skew|>2 |
| `engineer_max_features` | 100 | Max generated features (capped for memory safety) |
| `preprocessing_mode` | `"auto"` | `"auto"` skips IQR+engineering on small/clean data |
| `pos_weight` | None | BCEWithLogitsLoss class weight (use 9.0 for 90/10 imbalance) |
| `threshold_tuning` | True | Auto-optimize F2 decision threshold on validation data |
| `use_batch_norm` | `False` | Harmful for regression |
| `epochs` | 100 | With early stopping (patience=10) |
| `learning_rate` | 1e-3 | Adam optimizer (log-tuned) |
| `weight_decay` | 1e-5 | L2 regularization (log-tuned) |

**Disable preprocessing**: `clip_outliers=False, impute_missing=False, engineer_features=False`

---

## Experiment Tracking (MLflow)

```bash
# Run experiments
uv run python experiments/run_experiments.py --version v1 --epochs 100

# Optuna tuning (13 datasets, 14 parallel trials, CV mode)
uv run python experiments/tune_pipeline.py --all --n-trials 500

# Compare versions in browser
uv run python experiments/compare_runs.py --exp dantabnn --ver1 v2-baseline --ver2 v7

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Terminal dump
uv run python experiments/analyze_db.py
```

---

## Hyperparameter Tuning (built-in API)

```python
pipe = BinaryClassificationPipeline(...)
best = pipe.hyperparameters_tuning(
    df_train,
    n_iter=500,          # trials (pruner stops bad ones early)
    n_jobs=14,           # parallel workers
    direction="maximize" # maximize AUC/R²/F1
)
```

**Tunable**: `dropout`, `learning_rate`, `weight_decay`, `hidden_dims_choice` (adaptive/narrow/wide), `gating_type` (soft/none).

Uses TPE sampler + MedianPruner (10 startup trials). CV mode auto-activates when `df_val=None`.

---

## Advanced: Feature Generation

```python
from dantabnn.feature_generation import DomainRatioGenerator

gen = DomainRatioGenerator(max_features=20)
gen.fit(X_train, y_train)           # auto-discovers skewed, cyclic, ratio features
new_features = gen.transform(X_val) # produces log1p, sin/cos, ratio columns
```

---

## Save / Load

```python
pipe.save("models/my_pipeline")
pipe2 = BinaryClassificationPipeline(...).load("models/my_pipeline")
preds = pipe2.predict(df)  # ready immediately
```

---

## Project Structure

```
src/dantabnn/
├── base.py, binary.py, regression.py, multiclass.py   # Pipelines
├── models/danet.py, gating.py, cross.py               # Neural modules
├── preprocessing/encoder.py, scaler.py, imputer.py,   # Preprocessing
│   outlier.py, feature_engineer.py
├── feature_generation/                                 # Advanced feature eng
├── tuning/hyperparam.py, tune_utils.py                # Optuna tuner + param grids
└── utils/metrics.py, logger.py, hardware.py
experiments/run_experiments.py, tune_pipeline.py        # Runners
tests/ (230 tests, 83% coverage)
```

---

## Key Findings

1. **500-trial tuning outperforms 20-trial**: +0.01-0.04 across all datasets (avg|Δ|=0.014)
2. **fit_from_parquet() enables 30 GB datasets in 32 GB Docker containers** — 15 GB peak vs 140 GB with `fit()`
3. **Preprocessing helps without tuning**: wine_quality -0.02→0.34, iris F1=0.90→0.97
4. **EarlyStoppingCallback saves time**: studies stop at 120-380 trials (never run full 500)
5. **Threshold tuning matters for imbalanced data**: F2-optimal threshold ≠ 0.5 for skewed classes
6. **pos_weight in BCEWithLogitsLoss** addresses class imbalance without oversampling
7. **Sparse OneHotEncoder saves 80-95% RAM** — production-ready for large feature sets
8. **No hard dimension limits** — escalating warnings only (>256/512/1024) for production safety
9. **In-place outlier clipping + intermediate gc** — 25-32 GB RAM saved on large datasets
10. **Better than LightAutoML**: DanTabNN wins 8-1-4 across 13 benchmark datasets

## Memory Optimizations

| Optimization | Impact | Location |
|---|---|---|
| `fit_from_parquet()` chunked streaming | 30 GB → 15 GB peak RAM | `base.py` |
| Sparse OneHotEncoder | 80-95% RAM savings for categorical | `encoder.py` |
| In-place outlier clipping | 25 GB saved on large arrays | `outlier.py` |
| `del df_train` after numpy extraction | 60-90 GB pandas overhead freed | `base.py:fit()` |
| `gc.collect()` between preprocessing steps | 25-32 GB freed between stages | `base.py:_prepare_features()` |
| `float64→float32` conversion | 50% GPU transfer reduction | `base.py:_prepare_features()` |
| `pd.isna()` + `to_numpy(na_value=nan)` | Handles pandas NAType (nullable Int columns) | `base.py` |

## License

MIT — see [LICENSE](LICENSE).
