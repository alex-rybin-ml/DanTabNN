# DanTabNN Pipeline Architecture

> Version history of the DanTabNN pipeline architecture.
> New versions are appended at the top when they beat the previous benchmark on large datasets.

---

## VERSION v0.2.5 (Current — 2025-06-27)

### High-Level Architecture

```
DataFrame
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  PREPROCESSING CHAIN                                             │
│                                                                  │
│  1. NaNImputer        — median (numeric) / mode (categorical)    │
│  2. OutlierClipper    — IQR winsorization (1.5×IQR fences)       │
│  3. AutoFeatureEngineer — x² (all) + log1p (|skew|>2.0)         │
│  4. StandardScaler    — zero mean, unit variance                 │
│  5. CategoricalEncoder — sparse OneHot (sklearn)                 │
│                                                                  │
│  Auto-mode: skips IQR + feature eng for n<1000, d<20, clean data │
└──────────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  DANetModule (Dual-Attention Network)                            │
│                                                                  │
│  Input: (B, D) tabular features                                  │
│    │                                                             │
│    ▼                                                             │
│  1. Feature Gating (Gumbel-Softmax)                              │
│     ─ Per-feature learnable logits                               │
│     ─ Bernoulli sampling, straight-through estimator             │
│     ─ Missingness-aware masking                                  │
│     ─ Types: 'soft' (independent Bernoulli), 'topk' (exact K)    │
│    │                                                             │
│    ▼                                                             │
│  2. Embedding (Linear → hidden_dims[0])                          │
│    │                                                             │
│    ▼                                                             │
│  3. Feature Attention (Multi-Head Self-Attention)                │
│     ─ 4 heads, dimension-preserving                              │
│     ─ QKV → scaled dot-product → residual + LayerNorm            │
│     ─ Constraint: D % num_heads == 0                             │
│    │                                                             │
│    ▼                                                             │
│  4. Cross Network (DCN-V1)                                       │
│     ─ 3 cross layers: x' = x₀ ⊙ (W·x + b) + x                  │
│     ─ Xavier init (gain=0.1), optional low-rank                  │
│    │                                                             │
│    ▼                                                             │
│  5. Feed-Forward Network                                         │
│     ─ 3-layer: [128, 64, 32] (default, adaptive)                 │
│     ─ ReLU, Dropout, optional BatchNorm                          │
│    │                                                             │
│    ▼                                                             │
│  6. Output Layer                                                 │
│     ─ Binary: Linear→1, BCEWithLogitsLoss                        │
│     ─ Regression: Linear→1, HuberLoss(delta=1.0)                 │
│     ─ Multiclass: Linear→n_classes, CrossEntropyLoss             │
└──────────────────────────────────────────────────────────────────┘
```


### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Optimizer | Adam | lr=1e-3, wd=1e-5 |
| Scheduler | ReduceLROnPlateau | mode=min, patience=5, factor=0.5 |
| Gradient Clip | max_norm=1.0 | Every step |
| Early Stop | patience=10 | min_delta=1e-4, min_epochs=10 |
| Batch Size | 32 | Default |
| Epochs | 100 | Maximum |
| Target Scale | True (regression) | Standardize to mu=0, sigma=1 |

### Hyperparameter Tuning (Optuna)

| Parameter | Space | Sampler |
|-----------|-------|---------|
| dropout | [0.0, 0.5] | Float |
| learning_rate | [1e-4, 1e-2] log | Float |
| weight_decay | [1e-6, 1e-3] log | Float |
| hidden_dims_choice | adaptive/narrow/wide | Categorical |
| gating_type | soft/none | Categorical |

- Sampler: TPE, Pruner: MedianPruner (10 startup), EarlyStoppingCallback (20% of trials)

### Task-Specific Details

| Task | Loss | Metrics | Output |
|------|------|---------|--------|
| Binary | BCEWithLogitsLoss (+ pos_weight) | accuracy, roc_auc | Sigmoid |
| Regression | HuberLoss(delta=1.0) | mse, mae, r2 | Linear (+ unscale) |
| Multiclass | CrossEntropyLoss (+ auto weights) | accuracy, f1_macro, f1_weighted | Softmax |

### Memory Optimizations

- fit_from_parquet(): chunked streaming (30GB -> 15GB peak)
- Sparse OneHotEncoder: 80-95% RAM savings
- In-place outlier clipping + intermediate gc.collect()
- float64 -> float32 before GPU transfer

### Known Limitations (v0.2.5)

1. Attention head divisibility: D % num_heads == 0 required
2. Limited tuning space: only 5 params tuned
3. No mixed precision training (AMP available but unused)
4. Fixed LR schedule: only ReduceLROnPlateau
5. No embedding for one-hot features before attention
6. DCN-V1 only: no matrix-based cross layers (DCN-V2)
7. No residual connections in FFN
8. Fixed ReLU activation throughout (no GELU/Swish)
9. No batch_size or attention_heads in hyperparameter search
10. Single model only: no built-in ensembling

---

## VERSION v0.3.0 — PROPOSED (Next)

See `plans/` directory for detailed improvement proposals.



---

## VERSION v0.3.0 — IN PROGRESS (2025-06-27)

### New Features vs v0.2.5

| Feature | Plan | Status |
|---------|------|--------|
| Automatic Mixed Precision (AMP) | Plan 01 | Implemented in base.py |
| CosineAnnealingWarmRestarts LR | Plan 01 | Implemented in base.py + tune_utils |
| TabularDataAugmentation module | Plan 04 | Module created (src/dantabnn/augmentation/) |
| CrossLayerV2 (DCN-V2) | Plan 03 | CrossLayerV2 class in cross.py |
| K-fold Ensemble API | Plan 05 | Full implementation in run_v03_experiments.py |
| Expanded tuning space | Plan 02 | Partially added (lr_scheduler) to tune_utils.py |

### How to Run

```bash
# Single model with AMP + Cosine LR:
uv run python experiments/run_v03_experiments.py --version v0.3-all

# K-fold Ensemble:
uv run python experiments/run_v03_experiments.py --version v0.3-ensemble --ensemble 5

# Quick test:
uv run python experiments/run_v03_experiments.py --version v0.3-test --epochs 20

# Compare:
uv run python experiments/db_dump.py
```
