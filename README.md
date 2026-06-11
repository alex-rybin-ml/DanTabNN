# DanTabNN — Dual-Attention Neural Networks for Tabular Data

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-236%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-84%25-green)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A PyTorch-based deep learning pipeline for tabular data classification and regression, featuring **Dual-Attention Networks (DANet)** with feature-wise self-attention, differentiable feature selection via Gumbel-Softmax gating, automated preprocessing pipeline, and built-in MLflow experiment tracking.

---

## Architecture

```
Pandas DataFrame
    │
    ▼
┌─────────────────────────┐
│  NaN Imputer (median)   │  ← Missing value handling
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Outlier Clipper (IQR)  │  ← [Q1-1.5×IQR, Q3+1.5×IQR] winsorization
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Feature Engineering    │  ← x² for all features + log1p for |skew|>2.0
│  (AutoFeatureEngineer)  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  StandardScaler         │  ← Mean=0, std=1
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  OneHotEncoder (sparse) │  ← Sparse output for memory efficiency
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  np.hstack → float32    │  ← Combined feature tensor
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Feature Gating (Soft)  │  ← Gumbel-Softmax differentiable selection
│  (per-feature logits)   │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Embedding (Linear)     │  ← Project to hidden_dims[0]
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Feature Attention      │  ← Multi-head self-attention across features
│  (LayerNorm + residual) │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Cross Network (opt)    │  ← Explicit feature crosses (DCN-style)
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Feed-Forward Network   │  ← 3-layer MLP: [128, 64, 32]
│  (ReLU + Dropout)       │     Optional BatchNorm between layers
└───────────┬─────────────┘
            ▼
       Output Layer
   (Linear → task-specific)
```

**Key components:**

| Component | Purpose |
|-----------|---------|
| **Preprocessing Pipeline** | IQR outlier clipping → median imputation → x²+log1p feature engineering → StandardScaler → OneHotEncoder |
| **Feature Gating** | Gumbel-Softmax learns to select/deselect features during training |
| **Feature Attention** | 4-head self-attention across feature dimensions (learns feature interactions) |
| **Cross Network** | Optional DCN-style explicit pairwise feature crosses |
| **Gradient Clipping** | `max_norm=1.0` prevents instability from gating gradients |
| **Huber Loss** | Robust regression loss (delta=1.0), resistant to outliers |

---

## Installation

```bash
git clone https://github.com/alex-rybin-ml/DanTabNN.git
cd DanTabNN
uv sync
```

**Requirements:** Python 3.9+, PyTorch 2.0+, scikit-learn 1.3+, pandas 2.0+, Optuna 3.5+, MLflow

---

## Quick Start

### Binary Classification

```python
from dantabnn.binary import BinaryClassificationPipeline
import pandas as pd

df = pd.read_csv("your_data.csv")
pipe = BinaryClassificationPipeline(
    numeric_features=["age", "income", "debt_ratio"],
    categorical_features=["job", "marital"],
    target_column="default",
    epochs=100,
    early_stopping_patience=10,
)
pipe.fit(df)

# Predict probabilities
probs = pipe.predict(df_test)
classes = pipe.predict_classes(df_test, threshold=0.5)

# Evaluate
metrics = pipe.evaluate(df_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

### Regression

```python
from dantabnn.regression import RegressionPipeline

pipe = RegressionPipeline(
    numeric_features=["CRIM", "ZN", "RM", "AGE", "DIS"],
    categorical_features=[],
    target_column="MEDV",
    epochs=100,
)
pipe.fit(df_train, df_val=df_val)
preds = pipe.predict(df_test)
```

### Multiclass Classification

```python
from dantabnn.multiclass import MulticlassClassificationPipeline

pipe = MulticlassClassificationPipeline(
    numeric_features=[f"f{i}" for i in range(64)],
    categorical_features=[],
    target_column="digit",
    n_classes=10,
)
pipe.fit(df_train, df_val=df_val)
probs = pipe.predict(df_test)
classes = np.argmax(probs, axis=1)
```

---

## Real-World Benchmark Results

Evaluated on 14 real-world datasets (5 binary, 4 regression, 5 multiclass). All metrics from 20-trial Optuna tuning with preprocessing enabled.

### Regression (R²)

| Dataset | v2 (baseline) | v6 (tuned, no preproc) | v7 (tuned + preproc) |
|---------|--------------|----------------------|---------------------|
| Boston Housing | 0.628 | **0.816** | **0.827** |
| Diabetes Progression | 0.362 | 0.494 | **0.510** |
| Energy Efficiency | 0.845 | **0.989** | 0.984 |
| Wine Quality | -0.023 | — | **0.479** |

### Binary Classification (ROC-AUC)

| Dataset | v2 (baseline) | v7 (tuned + preproc) |
|---------|--------------|---------------------|
| Breast Cancer | 0.994 | **0.996** |
| German Credit | 0.703 | **0.801** |
| Pima Diabetes | 0.823 | **0.824** |
| Spambase | 0.972 | **0.988** |

### Multiclass Classification (F1-Macro)

| Dataset | v2 (baseline) | v7 (tuned + preproc) |
|---------|--------------|---------------------|
| Iris | 0.898 | **0.967** |
| Wine | 1.000 | **1.000** |
| Digits | 0.964 | **0.980** |
| Vehicle | 0.940 | **0.992** |
| Segment | 0.978 | **1.000** |

> **v2** = fixed hyperparameters (100 epochs, no preprocessing). **v6** = 20-trial Optuna tuning without preprocessing. **v7** = 20-trial Optuna tuning with IQR clipping, imputation, and x²+log1p feature engineering. Regression R² improved by **+0.17** over v2 baseline with tuning, and preprocessing added another **+0.01** on top of that.

---

## Experiment Tracking with MLflow

All experiments are logged to a shared MLflow database with version tags for comparison.

### Run Experiments

```bash
# Run fixed-hyperparameter experiments (14 real-world datasets)
uv run python experiments/run_experiments.py --version v8-memory-optimized --epochs 100

# Run Optuna hyperparameter tuning (13 datasets, 14 parallel workers)
uv run python experiments/tune_pipeline.py --all --n-trials 20

# Run tuning without preprocessing for ablation studies
uv run python experiments/tune_pipeline.py --all --n-trials 20 --no-preprocessing
```

### Compare Results

**Method 1 — Browser-based comparison:**
```bash
uv run python experiments/compare_runs.py --exp dantabnn --ver1 v2-baseline --ver2 v7-preprocessing
# Opens experiments/compare_runs.html — side-by-side tables with delta columns
```

**Method 2 — MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
- Go to `dantabnn` experiment
- Click ⚙ → check `params.dataset` and `tags.model_version`
- Filter: `tags.model_version = "v7-preprocessing"`

**Method 3 — Terminal analysis:**
```bash
# SQLite-level full database dump
uv run python experiments/analyze_db.py
```

---

## Hyperparameter Configuration

### Default Architecture (v8)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[128, 64, 32]` | 3-layer MLP (adaptively sized per dataset) |
| `dropout` | 0.2 | Dropout after each layer |
| `attention_heads` | 4 | Multi-head self-attention heads |
| `gating_type` | `"soft"` | Gumbel-Softmax feature gating |
| `gating_k` | `n_features // 3` | Features to select |
| `use_batch_norm` | `False` | BatchNorm (harmful for regression!) |
| `device` | Auto (CUDA/CPU) | PyTorch device selection |
| `clip_outliers` | **`True`** | IQR-based winsorization (new in v7) |
| `impute_missing` | **`True`** | Median imputation (new in v7) |
| `engineer_features` | **`True`** | x² + log1p auto-detection (new in v7) |
| `batch_size` | 64 | Training batch size |
| `epochs` | 100 | Max epochs (early stopping usually cuts off earlier) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `early_stopping_patience` | 10 | Stop if no improvement for N epochs |

### Preprocessing Pipeline (enabled by default)

| Stage | Description | Memory Impact |
|-------|-------------|---------------|
| 1. `NaNImputer` | Median imputation for numeric, mode for categorical | — |
| 2. `OutlierClipper` | IQR-based winsorization [Q1-1.5×IQR, Q3+1.5×IQR] | — |
| 3. `AutoFeatureEngineer` | x² for every feature + log1p for skewed features | Doubles features |
| 4. `StandardScaler` | Mean=0, std=1 standardization | — |
| 5. `CategoricalEncoder` | Sparse OneHotEncoder (80-95% RAM savings) | Sparse until final hstack |
| 6. `float32 conversion` | Dtype downgrade before GPU transfer | 50% GPU memory reduction |

### Disabling Preprocessing

```python
pipe = RegressionPipeline(
    ..., clip_outliers=False, impute_missing=False, engineer_features=False
)
```

---

## Hyperparameter Tuning (Optuna)

```python
from dantabnn.binary import BinaryClassificationPipeline

pipe = BinaryClassificationPipeline(
    numeric_features=num_cols, categorical_features=cat_cols,
    target_column="target",
)
best_pipe = pipe.hyperparameters_tuning(
    df_train,
    df_val=df_val,
    n_iter=50,
    n_jobs=14,  # parallel trials
    direction="maximize",  # maximize AUC
    random_state=42,
)
print(f"Best params: {best_pipe.hyperparameters}")
```

Uses Bayesian optimization with TPE sampler, median pruner, and `param_mapper` to translate symbolic names (`hidden_dims_choice`) to concrete parameters.

**Tunable hyperparameters:**

| Parameter | Range | Type |
|-----------|-------|------|
| `dropout` | 0.0–0.5 | Float |
| `learning_rate` | 1e-4–1e-2 | Log-uniform |
| `weight_decay` | 1e-6–1e-3 | Log-uniform |
| `hidden_dims_choice` | adaptive/narrow/wide | Categorical |
| `gating_type` | soft/none | Categorical |

---

## Feature Generation

The `feature_generation` module provides domain-aware feature engineering to complement DANet attention:

| Generator | Purpose |
|-----------|---------|
| `DomainRatioGenerator` | Template-driven ratios, log1p, zscore, cyclic (sin/cos), clip |
| `DomainFeatureGenerator` | Polynomial expansions (degree-2 interactions) |
| `HighCardinalityEmbedder` | Target encoding for high-cardinality categoricals |
| `SelectiveInteractionGenerator` | MI-based pairwise interaction selection |
| `TemporalAggregationGenerator` | Rolling/expanding window aggregations within groups |
| `DANetFeatureGenerationPipeline` | Orchestrator with redundancy removal and feature count limits |

```python
from dantabnn.feature_generation import DomainRatioGenerator

gen = DomainRatioGenerator(max_features=20)
gen.fit(X_train, y_train)        # Auto-discovers skew, cyclic, ratio features
new_features = gen.transform(X_val)  # Produces log1p, sin/cos, ratio columns
```

---

## Saving and Loading

```python
# Save full pipeline (model + preprocessors + hyperparameters)
pipe.save("models/my_pipeline")

# Load
pipe2 = BinaryClassificationPipeline(
    numeric_features=[...], categorical_features=[...], target_column="target",
)
pipe2.load("models/my_pipeline")
preds = pipe2.predict(df)  # Ready immediately
```

---

## Project Structure

```
DanTabNN/
├── src/dantabnn/
│   ├── base.py                    # Abstract pipeline (fit, predict, evaluate, save/load)
│   ├── binary.py                  # Binary classification pipeline
│   ├── regression.py              # Regression pipeline (Huber loss, target scaling)
│   ├── multiclass.py              # Multiclass classification pipeline
│   ├── models/
│   │   ├── danet.py               # DANetModule (attention + FFN + gating + cross)
│   │   ├── gating.py              # FeatureGating, TopKFeatureGating
│   │   └── cross.py               # CrossNetwork, FactorizedCrossLayer
│   ├── preprocessing/
│   │   ├── scaler.py              # StandardScaler wrapper
│   │   ├── encoder.py             # OneHotEncoder wrapper (sparse by default)
│   │   ├── imputer.py             # NaNImputer (median/mode)
│   │   ├── outlier.py             # OutlierClipper (IQR winsorization)
│   │   └── feature_engineer.py    # AutoFeatureEngineer (x² + log1p auto-detection)
│   ├── feature_generation/        # Domain-aware feature engineering (advanced)
│   │   ├── base.py, domain.py, embedding.py, interaction.py,
│   │   ├── orchestrator.py, temporal.py
│   ├── tuning/
│   │   ├── hyperparam.py          # Optuna-based HyperparameterTuner
│   │   └── tune_utils.py          # Default param grids + param_mapper
│   └── utils/
│       ├── metrics.py             # Metric computation utilities
│       ├── logger.py              # Logging configuration
│       └── hardware.py            # GPU/CPU detection
├── experiments/
│   ├── run_experiments.py         # Fixed-hyperparameter experiment runner (14 datasets)
│   ├── tune_pipeline.py           # Optuna tuning runner (13 datasets, 14 parallel trials)
│   ├── compare_runs.py            # MLflow comparison → HTML output
│   ├── analyze_db.py              # SQLite database dump for analysis
│   ├── reset_db.py                # Wipe all MLflow experiments
│   └── cleanup_broken.py          # Remove corrupted MLflow runs
├── tests/                         # 236 tests, 84% coverage
│   ├── test_pipelines.py          # BaseNNPipeline + 3 concrete pipelines
│   ├── test_models.py             # Gating, Attention, DANetModule
│   ├── test_cross.py              # CrossNetwork, FactorizedCrossLayer
│   ├── test_feature_generation.py # All 6 feature generators
│   ├── test_preprocessing.py      # Scaler, Encoder
│   ├── test_utils.py              # Metrics, Logger
│   ├── test_tuning.py             # param_grid + param_mapper
│   ├── test_hardware.py           # Hardware detection
│   └── test_hyperparam.py         # HyperparameterTuner
├── mlruns/                        # MLflow database (auto-created)
└── pyproject.toml                 # Project metadata and dependencies
```

---

## Development

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=dantabnn --cov-report=term

# Run specific test file
uv run pytest tests/test_pipelines.py -v -k "binary"
```

236 tests, 84% code coverage, 0 failures.

---

## Key Experimental Findings

1. **Hyperparameter tuning is the #1 lever** — Optuna 20-trial tuning adds +0.15 avg|Δ| in regression R² over fixed hyperparameters
2. **Preprocessing helps most when tuning is NOT used** — wine_quality jumped from R²=-0.02 to 0.34, iris from F1=0.90 to 0.97
3. **With tuning, preprocessing adds marginal gains** — +0.01 avg|Δ| in regression over tuning-only, +0.002 in binary
4. **Sample attention provides no benefit** — tested with 20-trial tuning, avg|Δ|=0.006 vs no sample attention
5. **Sparse OneHotEncoder saves 80-95% RAM** — critical for datasets with high-cardinality categorical features
6. **Feature engineering doubles input dimension** — x² for every feature adds noise on well-behaved datasets like energy (−0.005 R²)
7. **BatchNorm is harmful for regression** — using `use_batch_norm=True` reduced R² from 0.63 to -1.86 on Boston Housing
8. **Huber loss helps outlier-heavy datasets** — wine_quality R² went from -0.02 to 0.33

---

## Memory Optimizations (v8)

| Optimization | Impact |
|---|---|
| `OneHotEncoder(sparse_output=True)` | 80-95% RAM savings for categorical features |
| `del` + `gc.collect()` between preprocessing stages | Frees intermediate numpy array copies |
| `float64` → `float32` conversion | 50% GPU transfer reduction |
| Input dimension validation (≤512 limit, >256 warning) | Prevents OOM from feature blowup |
| Sparse-to-dense only at final hstack | Avoids dense allocation of 500 GB+ matrices |

**Estimated capacity**: Pipeline handles 100 GB datasets in 150 GB RAM.

---

## License

MIT — see [LICENSE](LICENSE) for details.