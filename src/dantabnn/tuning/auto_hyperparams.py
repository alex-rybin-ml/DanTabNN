"""Automatic hyperparameter selection based on dataset complexity.

Hybrid approach:
  Stage 1 (always): Scaling laws from dataset statistics (n, d, PCA dim, variance, skewness)
  Stage 2 (budget > 2s): Zero-cost proxy (SNIP) to rank candidate architectures
"""
import time, numpy as np, torch, torch.nn as nn
from typing import Dict, List, Optional


def compute_dataset_complexity(X_in: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute dataset statistics: n, d, aspect_ratio, variance, skewness, PCA dim."""
    X = np.asarray(X_in, dtype=np.float64)
    if y is not None: y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    stats = {"n_samples": float(n), "n_features": float(d),
             "aspect_ratio": float(d) / max(float(n), 1.0),
             "log_n": np.log1p(n), "log_d": np.log1p(d)}
    finite_mask = np.isfinite(X)
    col_var = [np.var(X[:, i][finite_mask[:, i]]) if finite_mask[:, i].sum() > 1 else 0.0 for i in range(d)]
    stats["mean_variance"] = float(np.mean(col_var)) if col_var else 0.0
    stats["max_variance"] = float(np.max(col_var)) if col_var else 0.0
    if n * d > 50_000_000:
        stats["mean_skewness"] = 0.0  # skip: O(n*d) Python loop dominates PCA on large data
    else:
        col_skew = []
        for i in range(d):
            col = X[:, i][finite_mask[:, i]]
            if len(col) > 2:
                std = col.std()
                if std > 1e-10: col_skew.append(min(abs(float(np.mean((col - col.mean()) ** 3) / (std ** 3))), 10.0))
        stats["mean_skewness"] = float(np.mean(col_skew)) if col_skew else 0.0
    if n > d and d > 1:
        try:
            Xc = np.nan_to_num(X - X.mean(axis=0), 0.0)
            ev = np.linalg.eigvalsh(Xc.T @ Xc / (n - 1)); ev = np.maximum(ev, 0)
            total = ev.sum()
            pca_dim = int(np.searchsorted(np.cumsum(sorted(ev, reverse=True)) / total, 0.95) + 1) if total > 0 else d
        except: pca_dim = d
    else: pca_dim = d
    stats["pca_dim_95"] = float(min(pca_dim, d))
    if y is not None:
        yf = y[np.isfinite(y)]
        if len(yf) > 1: stats["target_std"] = float(np.std(yf)); stats["target_mean"] = float(np.mean(yf))
    return stats


def _param_count(hidden_dims, input_dim, output_dim, interaction_type='legacy',
                 num_cross_layers=2):
    """Honest parameter count including cross-layer and attention overhead."""
    if not hidden_dims:
        return input_dim * output_dim
    embed = input_dim * hidden_dims[0]
    ffn = sum(hidden_dims[i] * hidden_dims[i+1] for i in range(len(hidden_dims)-1)) if len(hidden_dims) > 1 else 0
    outp = hidden_dims[-1] * output_dim
    cross = num_cross_layers * hidden_dims[0] * hidden_dims[0] if interaction_type != 'legacy' else 0
    attn = hidden_dims[0] * hidden_dims[0]
    gating = input_dim
    total = embed + ffn + outp + cross + attn + gating
    return total


def scaling_law_hidden_dims(stats: Dict[str, float], task: str = "regression",
                            n_classes: int = 2) -> List[int]:
    """Scaling laws for hidden dimensions based on dataset complexity.
    
    Uses sqrt(n) scaling to actually utilize large datasets, with
    honest parameter counting including cross-layer and attention overhead.
    """
    n, d = stats["n_samples"], stats["n_features"]
    pca_dim = stats.get("pca_dim_95", d)
    aspect = stats.get("aspect_ratio", 0.1)
    # Wider scaling: sqrt(n) gives architectures that actually use available data
    base_width = max(32, int(np.sqrt(n) / 10))
    base_width = min(512, base_width)
    width_mult = min(2.0, max(0.5, pca_dim / max(d, 1.0)))
    max_params = int(n * min(20, max(5, n // 10000)))
    if pca_dim <= 2 and n > 20000: depth = 3
    elif pca_dim <= 2: depth = 2
    elif n < 300: depth = 2
    elif n < 800: depth = 2 if aspect > 0.05 else 3
    elif n < 3000: depth = 3
    elif n < 15000: depth = 3 if aspect < 0.02 else 4
    else: depth = 4
    if n_classes > 4:
        depth = max(depth, 3)
    dims = []; w = max(int(base_width * width_mult * 8), int(pca_dim))
    for _ in range(depth): w = max(8, w); dims.append(w); w = w // 2
    dims = [((d2 + 3) // 4) * 4 for d2 in dims]
    if n_classes > 4:
        min_width = ((n_classes * 4 + 3) // 4) * 4
        dims = [max(d2, min_width) for i, d2 in enumerate(dims)]
        min_width = min_width // 2
    output_dim = 1 if task == "regression" else 2
    while _param_count(dims, int(d), output_dim) > max_params and len(dims) > 1: dims = dims[:-1]
    while _param_count(dims, int(d), output_dim) > max_params and dims[0] > 16:
        dims = [max(8, d3 // 2) for d3 in dims]
        dims = [((d2 + 3) // 4) * 4 for d2 in dims]
    return dims


def zero_cost_score(model: nn.Module, X_batch: torch.Tensor) -> float:
    """SNIP proxy: sum of |grad * weight| at init. Higher = more trainable."""
    model.eval(); model.zero_grad()
    X_batch = X_batch.detach().requires_grad_(True)
    try:
        out = model(X_batch); loss = out.sum(dim=1).mean() if out.dim()>1 else out.mean()
        loss.backward()
    except: return 0.0
    score = sum((p.grad.abs() * p.abs()).sum().item() for p in model.parameters() if p.grad is not None)
    return score


def generate_candidates(stats, task="regression"):
    """Generate ~8 candidate hidden_dims for zero-cost filtering."""
    n = stats["n_samples"]; base = scaling_law_hidden_dims(stats, task)
    cand = [{"hidden_dims": base}, {"hidden_dims": [max(8, d//2) for d in base]},
            {"hidden_dims": [min(256, d*2) for d in base]}]
    if len(base) > 2: cand.append({"hidden_dims": base[:-1]})
    if n > 1000 and len(base) < 5: cand.append({"hidden_dims": base + [max(4, base[-1]//2)]})
    if n > 5000 and len(base) >= 3:
        nd = [max(4, int(d*0.7)) for d in base] + [max(4, int(base[-1]*0.35))]
        cand.append({"hidden_dims": nd})
    return cand[:8]


def auto_hyperparams(X: np.ndarray, y=None, task="regression", time_budget=10.0, device="cpu",
                     n_classes: int = 2) -> Dict:
    """Auto-select hyperparameters for tabular neural network training.

    Returns dict: hidden_dims, dropout, learning_rate, batch_size, augmentation dict,
                  use_feature_engineering, warmup_epochs, loss_type, gating_type,
                  interaction_type
    """
    t0 = time.time(); stats = compute_dataset_complexity(X, y)
    n, d = int(stats["n_samples"]), int(stats["n_features"])
    pca_dim = stats.get("pca_dim_95", d)

    # Stage 1: Scaling laws
    hidden_dims = scaling_law_hidden_dims(stats, task, n_classes)
    if pca_dim <= 2 and n < 5000: lr = 5e-4
    elif pca_dim <= 2: lr = 1e-3
    elif n < 500: lr = 5e-4
    elif n < 2000: lr = 1e-3 if d < 50 else 5e-4
    else: lr = 1e-3
    bs = max(8, min(256, n // 16)); bs = ((bs + 3) // 4) * 4
    if pca_dim <= 2 and n < 5000: dropout = 0.25
    elif pca_dim <= 2: dropout = 0.15
    elif n < 500: dropout = 0.3 if d < 30 else 0.4
    elif n < 2000: dropout = 0.2 if d < 50 else 0.3
    else: dropout = 0.1 if d < 100 else 0.2
    if n < 500: aug = {"cutmix": 0.0, "noise": 0.05, "noise_std": 0.005}
    elif n < 2000: aug = {"cutmix": 0.0, "noise": 0.1, "noise_std": 0.01}
    elif n < 10000: aug = {"cutmix": 0.15, "noise": 0.1, "noise_std": 0.01}
    else: aug = {"cutmix": 0.3, "noise": 0.1, "noise_std": 0.01}
    feat_eng = n >= 500; warmup = 5 if n < 2000 else 0
    loss_type = "mse" if task == "regression" else "default"

    # Stage 2: Zero-cost proxy refinement
    if time_budget > 2 and n > 200:
        try:
            from dantabnn.models.danet import DANetModule
            cand = generate_candidates(stats, task)
            Xb = torch.FloatTensor(X[:min(256, n)]).to(device)
            scores = []
            for c in cand:
                hd = c["hidden_dims"]
                try:
                    m = DANetModule(input_dim=d, hidden_dims=hd, dropout=0.1,
                                    attention_heads=min(4, hd[0]//2) if hd[0]>=4 else 2); m.to(device)
                    scores.append(zero_cost_score(m, Xb))
                except: scores.append(-1.0)
            bi = int(np.argmax(scores))
            if scores[bi] > 0: hidden_dims = cand[bi]["hidden_dims"]
        except ImportError: pass

    # Gating: enable for complex multiclass or high-dimensional data, not just PCA
    use_gating = (n_classes > 4) or (d > 30) or (pca_dim > 2)
    gating_type = "soft" if use_gating else "none"
    # Cross layer: enable DCN-V2 feature interactions for complex multiclass
    interaction_type = "cross" if (d > 30 and n_classes > 4) else "legacy"
    return {"hidden_dims": hidden_dims, "dropout": dropout, "learning_rate": lr,
            "batch_size": bs, "augmentation": aug, "use_feature_engineering": feat_eng,
            "warmup_epochs": warmup, "loss_type": loss_type,
            "gating_type": gating_type, "interaction_type": interaction_type,
            "dataset_stats": stats}