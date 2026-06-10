# DanTabNN — Dual-Attention Neural Networks for Tabular Data

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-227%20passed-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-green)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A PyTorch-based deep learning pipeline for tabular data classification and regression, featuring **Dual-Attention Networks (DANet)** with feature-wise self-attention, differentiable feature selection via Gumbel-Softmax gating, and built-in MLflow experiment tracking.

---

## Architecture

```
Input Features (tabular)
    │
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

Evaluated on 14 real-world datasets (5 binary, 4 regression, 5 multiclass). All metrics are averages across 15+ experiments logged to MLflow.

### Regression (vs. Baselines)

| Dataset | Linear Regression | DanTabNN (v1) | DanTabNN (v2) |
|---------|-------------------|---------------|---------------|
| Boston Housing | R²=0.67 | R²=-0.24 | **R²=0.63** |
| Diabetes Progression | R²=0.45 | R²=-2.60 | **R²=0.36** |
| Energy Efficiency | R²=0.90 | R²=0.76 | **R²=0.85** |
| Wine Quality | R²=0.36 | R²=0.00 | **R²=0.33** |

### Binary Classification (ROC-AUC)

| Dataset | v1 | v2 |
|---------|----|----|
| Breast Cancer | 0.994 | **0.994** |
| German Credit | 0.698 | **0.703** |
| Pima Diabetes | 0.823 | **0.823** |
| Spambase | 0.972 | **0.972** |
| Bank Marketing | 0.500 | **0.500** |

### Multiclass Classification (F1-Macro)

| Dataset | v1 | v2 |
|---------|----|----|
| Iris | 0.833 | **0.898** |
| Wine | 1.000 | **1.000** |
| Digits | 0.953 | **0.964** |
| Vehicle | 0.945 | **0.940** |
| Segment | 0.987 | **0.978** |

> **v1** = original narrow architecture (50 epochs, MSE loss). **v2** = wider 3-layer architecture (100 epochs, Huber loss, gradient clipping, robust early stopping). Regression R² improved from **-0.52 to +0.45** (a complete turnaround from negative to meaningful predictions).

---

## Experiment Tracking with MLflow

All experiments are logged to a shared MLflow database with version tags for comparison.

### Run Experiments

```bash
# Run baseline experiments (14 real-world datasets)
uv run python experiments/run_experiments.py --version v2-baseline --epochs 100

# Run with a specific ablation (e.g., BatchNorm enabled)
uv run python experiments/run_experiments.py --version v2-batchnorm --epochs 100 --batchnorm
```

### Compare Results

**Method 1 — Browser-based comparison:**
```bash
uv run python experiments/compare_runs.py --exp dantabnn --ver1 v2-baseline --ver2 v2-batchnorm
# Opens experiments/compare_runs.html — side-by-side tables with delta columns
```

**Method 2 — MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
- Go to `dantabnn` experiment
- Click ⚙ → check `params.dataset` and `tags.model_version`
- Filter: `tags.model_version = "v2-baseline"`
- Check two runs for the same dataset → Compare

**Method 3 — Terminal analysis:**
```bash
# SQLite-level full database dump
uv run python experiments/analyze_db.py
```

**Method 4 — Reset database:**
```bash
uv run python experiments/reset_db.py  # Wipes all runs, fresh start
```

---

## Hyperparameter Configuration

### Default Architecture (v2-baseline)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[128, 64, 32]` | 3-layer MLP (adaptively sized per dataset) |
| `dropout` | 0.2 | Dropout after each layer |
| `attention_heads` | 4 | Multi-head self-attention heads |
| `gating_type` | `"soft"` | Gumbel-Softmax feature gating |
| `gating_k` | `n_features // 3` | Features to select |
| `use_batch_norm` | `False` | BatchNorm (harmful for regression!) |
| `batch_size` | 64 | Training batch size |
| `epochs` | 100 | Max epochs (early stopping usually cuts off earlier) |
| `learning_rate` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularization |
| `early_stopping_patience` | 10 | Stop if no improvement for N epochs |

### Robust Early Stopping

The training loop includes:
- **min_delta=1e-4**: Marginal improvements (< 0.0001) count toward patience
- **min_epochs=max(10, patience)**: Early stopping only activates after minimum epochs
- **Decoupled scheduler**: `ReduceLROnPlateau(patience=5, min_lr=1e-6)`
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` after each backward pass

---

## Hyperparameter Tuning (Optuna)

```python
pipe = BinaryClassificationPipeline(
    numeric_features=num_cols, categorical_features=cat_cols,
    target_column="target",
)
best_pipe = pipe.hyperparameters_tuning(
    df_train,
    cv=5,
    n_iter=50,
    direction="minimize",
    random_state=42,
)
print(f"Best params: {best_pipe.hyperparameters}")
```

Uses Bayesian optimization with TPE sampler, median pruner, and stratified K-fold CV for classification tasks.

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
# Save full pipeline (model + scaler + encoder + hyperparameters)
pipe.save("models/my_pipeline")
# Creates: models/my_pipeline/model.pt, scaler.joblib, encoder.joblib, hyperparameters.joblib

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
│   │   └── encoder.py             # OneHotEncoder wrapper
│   ├── feature_generation/        # Domain-aware feature engineering
│   │   ├── base.py, domain.py, embedding.py, interaction.py,
│   │   ├── orchestrator.py, temporal.py
│   ├── tuning/
│   │   ├── hyperparam.py          # Optuna-based hyperparameter tuner
│   │   └── tune_utils.py          # Default param grids for DANet
│   └── utils/
│       ├── metrics.py             # Metric computation utilities
│       ├── logger.py              # Logging configuration
│       └── hardware.py            # GPU/CPU detection
├── experiments/
│   ├── run_experiments.py         # Main experiment runner (15 real-world datasets)
│   ├── compare_runs.py            # MLflow comparison → HTML output
│   ├── analyze_db.py              # SQLite database dump for analysis
│   ├── reset_db.py                # Wipe all MLflow experiments
│   └── cleanup_broken.py          # Remove corrupted MLflow runs
├── tests/                         # 227 tests, 85% coverage
│   ├── test_pipelines.py          # BaseNNPipeline + 3 concrete pipelines
│   ├── test_models.py             # Gating, Attention, DANetModule
│   ├── test_cross.py              # CrossNetwork, FactorizedCrossLayer
│   ├── test_feature_generation.py # All 6 feature generators
│   ├── test_preprocessing.py      # Scaler, Encoder
│   ├── test_utils.py              # Metrics, Logger
│   ├── test_tuning.py             # Tune utilities
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

227 tests, 85% code coverage, 0 failures.

---

## Key Experimental Findings

1. **Wider architecture is critical** — going from `[64, 32]` to adaptive `[32-128, 16-64, 8-32]` improved regression R² by 0.97
2. **BatchNorm is harmful for regression** — using `use_batch_norm=True` reduced R² from 0.63 to -1.86 on Boston Housing
3. **Huber loss helps outlier-heavy datasets** — wine_quality R² went from -0.02 to 0.33
4. **Learning rate 1e-3 is optimal** — reducing to 1e-4 prevents convergence (model stops before plateau)
5. **Feature gating has no measurable effect** — identical performance with `gating_type="soft"` vs `"none"`
6. **Target standardization helps small datasets** — improved diabetes_prog R² but degraded energy

---

## License

MIT — see [LICENSE](LICENSE) for details.