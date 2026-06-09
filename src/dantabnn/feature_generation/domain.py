"""Domain-aware feature generation using polynomial expansions and ratio/template-based transforms."""

import json
from itertools import combinations
from typing import Optional, List, Dict, Any, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .base import BaseDANetFeatureGenerator

# ---------------------------------------------------------------------------
# Predefined transforms registry (Section 3 of the spec)
# ---------------------------------------------------------------------------
_PREDEFINED_TRANSFORMS: Dict[str, Dict[str, Any]] = {
    "log": {
        "description": "ln(col), clipped to [1e-10, ∞)",
        "num_args": 1,
        "func": lambda X, col: np.log(np.clip(X[col].astype(float), 1e-10, None)),
    },
    "log1p": {
        "description": "ln(1 + col), clipped to [0, ∞)",
        "num_args": 1,
        "func": lambda X, col: np.log1p(np.clip(X[col].astype(float), 0, None)),
    },
    "sqrt": {
        "description": "√col, clipped to [0, ∞)",
        "num_args": 1,
        "func": lambda X, col: np.sqrt(np.clip(X[col].astype(float), 0, None)),
    },
    "square": {
        "description": "col²",
        "num_args": 1,
        "func": lambda X, col: X[col].astype(float) ** 2,
    },
    "inverse": {
        "description": "1 / col, 0 → 0.0",
        "num_args": 1,
        "func": lambda X, col: np.where(
            X[col].astype(float) == 0, 0.0, 1.0 / X[col].astype(float)
        ),
    },
    "zscore": {
        "description": "(col − μ) / σ, stats from fit",
        "num_args": 1,
        "func": None,  # handled specially (stateful)
    },
    "ratio": {
        "description": "a / b, safe divide",
        "num_args": 2,
        "func": None,  # handled specially (safe divide)
    },
    "cyclic_sin": {
        "description": "sin(2π · col / period)",
        "num_args": 1,
        "func": lambda X, col, period: np.sin(2 * np.pi * X[col].astype(float) / period),
    },
    "cyclic_cos": {
        "description": "cos(2π · col / period)",
        "num_args": 1,
        "func": lambda X, col, period: np.cos(2 * np.pi * X[col].astype(float) / period),
    },
    "clip": {
        "description": "clip(col, lower, upper)",
        "num_args": 1,
        "func": None,  # handled specially (params)
    },
}


# ---------------------------------------------------------------------------
# Helper: safe divide
# ---------------------------------------------------------------------------
def _safe_divide(
    X: pd.DataFrame, num_col: str, den_col: str, fill: float = 0.0
) -> pd.Series:
    """Compute num_col / den_col with protection against zero / inf."""
    num = X[num_col].astype(float)
    den = X[den_col].astype(float)
    result = num / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill)


# ---------------------------------------------------------------------------
# DomainFeatureGenerator (existing polynomial-based generator)
# ---------------------------------------------------------------------------
class DomainFeatureGenerator(BaseDANetFeatureGenerator):
    """Generate domain-inspired features via polynomial transformations.

    This generator creates polynomials features of numeric columns up to a given degree,
    optionally including interaction terms. It is a placeholder for more sophisticated
    domain-logic templates.

    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features.
    interaction_only : bool, default=False
        If True, only interaction features are produced (no powers of a single feature).
    include_bias : bool, default=False
        If True, include a bias column (all polynomial powers are zero).
    numeric_columns : Optional[List[str]], default=None
        Subset of numeric columns to transform. If None, all numeric columns in the
        fitted dataframe are used.
    """

    def __init__(
            self,
            degree: int = 2,
            interaction_only: bool = False,
            include_bias: bool = False,
            numeric_columns: Optional[List[str]] = None,
            name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.numeric_columns = numeric_columns
        self._poly = None
        self._original_columns: List[str] = []
        self._feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureGenerator":
        self._log_info(f"Fitting polynomial features (degree={self.degree})")
        # Determine numeric columns to transform
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.numeric_columns is not None:
            # Validate that requested columns are present and numeric
            missing = set(self.numeric_columns) - set(numeric_cols)
            if missing:
                self._log_warning(
                    f"Columns {missing} are not numeric or missing: they will be ignored."
                )
            self._original_columns = [c for c in self.numeric_columns if c in numeric_cols]
        else:
            self._original_columns = numeric_cols

        if len(self._original_columns) == 0:
            self._log_warning("No numeric columns available for polynomial features.")
            self._poly = None
            self._feature_names = []
            self.is_fitted = True
            return self

        # Impute missing values before polynomial expansion
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=True)

        # Fit sklearn's PolynomialFeatures
        self._poly = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias
        )
        self._poly.fit(X_imputed)

        # Build feature names
        self._feature_names = list(self._poly.get_feature_names_out(self._original_columns))
        self._log_info(f"Generated {len(self._feature_names)} polynomial features")
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform")
        if self._poly is None or len(self._original_columns) == 0:
            # No features generated, return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

        # Ensure all required columns are present
        missing = set(self._original_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns required for transformation: {missing}")

        # Impute missing values using stored means
        X_numeric = X[self._original_columns]
        X_imputed = self._impute_numeric(X_numeric, fit=False)
        transformed = self._poly.transform(X_imputed)
        # Convert to DataFrame with appropriate column names
        df = pd.DataFrame(
            transformed,
            columns=self._feature_names,
            index=X.index,
        )
        return df

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """Polynomial features are numeric and should be DANet-compatible."""
        if len(self._feature_names) > 500:
            self._log_warning(
                f"Number of polynomial features ({len(self._feature_names)}) exceeds DANet limit of 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "polynomial",
            "degree": self.degree,
            "interaction_only": self.interaction_only,
            "include_bias": self.include_bias,
            "original_columns": self._original_columns.copy(),
        })
        return metadata


# ---------------------------------------------------------------------------
# DomainRatioGenerator (template-driven ratio/transform generator)
# ---------------------------------------------------------------------------
class DomainRatioGenerator(BaseDANetFeatureGenerator):
    """Template-driven feature generator producing ratios, logs, cyclic, clip, etc.

    Designed to complement DANet attention by generating features the model
    cannot trivially learn internally (ratios, logarithms, cyclic encodings, zscore).

    Two modes:
    1. **Explicit mode** — user provides ``templates`` (list of dicts).
    2. **Auto-discovery mode** — ``templates=None`` or ``[]`` → the generator
       scans the data during ``fit()`` and automatically detects skewed columns
       (→ log1p), cyclic integer columns (→ sin/cos), informative ratios
       (→ MI-based selection), and outlier-prone columns (→ clip).

    Parameters
    ----------
    templates : Optional[List[Dict[str, Any]]], default=None
        Explicit list of transformation templates.
        ``None`` or ``[]`` triggers auto-discovery.
    max_features : int, default=20
        Maximum number of features in auto-discovery mode.
        Ignored in explicit mode.
    corr_threshold : float, default=0.95
        Maximum absolute Pearson correlation with any existing column for acceptance
        (auto-discovery only).
    mi_threshold : float, default=1.05
        Minimum MI_ratio / max(MI_a, MI_b) for ratio candidates (auto-discovery only).
    """

    def __init__(
        self,
        templates: Optional[List[Dict[str, Any]]] = None,
        max_features: int = 20,
        corr_threshold: float = 0.95,
        mi_threshold: float = 1.05,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "DomainRatioGenerator")
        self.templates = templates
        self.max_features = max_features
        self.corr_threshold = corr_threshold
        self.mi_threshold = mi_threshold

        # Internal state
        self._templates: List[Dict[str, Any]] = []
        self._feature_names: List[str] = []
        self._zscore_stats: Dict[str, Dict[str, float]] = {}
        self._imputation_means: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API (implementing BaseDANetFeatureGenerator)
    # ------------------------------------------------------------------
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DomainRatioGenerator":
        """Fit the generator on training data.

        In explicit mode, validates templates and precomputes zscore stats.
        In auto-discovery mode, scans data to build templates automatically.
        """
        if self.templates is not None and len(self.templates) > 0:
            self._log_info("Explicit mode: using provided templates")
            self._fit_explicit(X, y)
        else:
            self._log_info("Auto-discovery mode: scanning data for domain features")
            self._fit_auto_discover(X, y)
        self._feature_names = [self._make_output_name(t) for t in self._templates]
        self.is_fitted = True
        self._log_info(f"Fitted {len(self._feature_names)} domain features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate features for new data. Error-tolerant: skips failed templates."""
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted before transform.")
        if len(self._templates) == 0:
            return pd.DataFrame(index=X.index)

        results: Dict[str, pd.Series] = {}
        for tmpl, name in zip(self._templates, self._feature_names):
            try:
                series = self._apply_template(X, tmpl)
                if series is not None:
                    results[name] = series
            except Exception as e:
                self._log_warning(f"Skipping '{name}': {e}")

        if not results:
            return pd.DataFrame(index=X.index)
        return pd.DataFrame(results, index=X.index)

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def validate_danet_compatibility(self) -> bool:
        """All generated features are numeric → compatible by construction."""
        if len(self._feature_names) > 500:
            self._log_warning(
                f"Feature count {len(self._feature_names)} exceeds DANet limit 500."
            )
            return False
        return True

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata.update({
            "generator_type": "domain_ratio",
            "templates": self._templates,
            "max_features": self.max_features,
            "corr_threshold": self.corr_threshold,
            "mi_threshold": self.mi_threshold,
        })
        return metadata

    # ------------------------------------------------------------------
    # Explicit mode
    # ------------------------------------------------------------------
    def _fit_explicit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Validate templates and precompute zscore stats."""
        self._templates = []
        missing = set()
        for t in self.templates:
            # Validate referenced columns
            for c in t.get("columns", []):
                if c not in X.columns:
                    missing.add(c)
            self._templates.append(t)

        if missing:
            self._log_warning(f"Columns not in data: {missing}")

        # Precompute zscore stats for any zscore templates
        for t in self._templates:
            if t["type"] == "zscore":
                col = t["columns"][0]
                if col in X.columns:
                    vals = X[col].astype(float)
                    self._zscore_stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std(ddof=0)) or 1.0,
                    }

    # ------------------------------------------------------------------
    # Auto-discovery mode
    # ------------------------------------------------------------------
    def _fit_auto_discover(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> None:
        """Auto-discover domain features from data (Section 4 of spec)."""
        # 4.1 Preprocessing
        numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not numeric_cols:
            self._log_warning("No numeric columns; no features generated.")
            self._templates = []
            return

        X_imputed = self._impute_numeric(X[numeric_cols], fit=True)
        self._numeric_cols_original = numeric_cols  # store for transform

        # Determine task type from y
        mi_per_col: Dict[str, float] = {}
        if y is not None:
            try:
                from sklearn.feature_selection import (
                    mutual_info_classif,
                    mutual_info_regression,
                )
                n_unique = y.nunique() if hasattr(y, "nunique") else len(set(y))
                is_clf = 1 < n_unique < 20
                mi_func = mutual_info_classif if is_clf else mutual_info_regression
                y_vals = y.values if hasattr(y, "values") else np.asarray(y)
                for col in numeric_cols:
                    mi = mi_func(X_imputed[[col]], y_vals).item()
                    mi_per_col[col] = float(mi)
            except Exception as e:
                self._log_warning(f"MI computation failed: {e}; skipping MI-based detection.")

        # 4.2 Detection rules
        candidates: List[Dict[str, Any]] = []

        # Rule 1: Skewed → log1p
        skewed_cands = self._detect_skewed(X_imputed, numeric_cols)
        candidates.extend(skewed_cands)

        # Rule 2: Cyclic integer → sin + cos
        cyclic_cands = self._detect_cyclic(X_imputed, numeric_cols)
        candidates.extend(cyclic_cands)

        # Rule 3: Informative ratios → MI-based
        ratio_cands = self._detect_ratios(X_imputed, numeric_cols, y, mi_per_col)
        candidates.extend(ratio_cands)

        # Rule 4: Outlier-prone → clip
        outlier_cands = self._detect_outliers(X_imputed, numeric_cols)
        candidates.extend(outlier_cands)

        if not candidates:
            self._log_info("No candidates detected.")
            self._templates = []
            return

        # Sort by score descending
        candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        # 4.4 Greedy selection with correlation filtering
        self._templates = self._greedy_select(candidates, X_imputed, numeric_cols)

        # Precompute zscore stats if any
        for t in self._templates:
            if t["type"] == "zscore":
                col = t["columns"][0]
                vals = X_imputed[col].astype(float)
                self._zscore_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=0)) or 1.0,
                }

    def _detect_skewed(
        self, X: pd.DataFrame, numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Rule 1: Detect skewed columns → log1p candidates."""
        candidates = []
        for col in numeric_cols:
            try:
                vals = X[col].dropna()
                if len(vals) < 3:
                    continue
                skew = float(abs(vals.skew()))
                if skew > 2.0:
                    candidates.append({
                        "_tmpl_json": json.dumps({
                            "type": "log1p", "columns": [col],
                        }),
                        "score": skew,
                    })
            except Exception:
                continue
        return candidates

    def _detect_cyclic(
        self, X: pd.DataFrame, numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Rule 2: Detect cyclic integer columns → sin + cos candidates."""
        candidates = []
        for col in numeric_cols:
            try:
                vals = X[col].dropna()
                n_unique = vals.nunique()
                if not (2 <= n_unique <= 31):
                    continue
                # Check integer-like
                if not np.issubdtype(vals.dtype, np.integer):
                    # Heuristic: all values are integers within tolerance
                    if not np.allclose(vals, np.round(vals), rtol=0, atol=1e-8):
                        continue
                period = float(vals.max() - vals.min() + 1)
                if period <= 1:
                    continue
                # sin candidate
                candidates.append({
                    "_tmpl_json": json.dumps({
                        "type": "cyclic_sin",
                        "columns": [col],
                        "params": {"period": period},
                    }),
                    "score": 3.0,
                })
                # cos candidate
                candidates.append({
                    "_tmpl_json": json.dumps({
                        "type": "cyclic_cos",
                        "columns": [col],
                        "params": {"period": period},
                    }),
                    "score": 3.0,
                })
            except Exception:
                continue
        return candidates

    def _detect_ratios(
        self,
        X: pd.DataFrame,
        numeric_cols: List[str],
        y: Optional[pd.Series],
        mi_per_col: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Rule 3: Compute MI ratios for column pairs → ratio candidates."""
        if y is None or len(mi_per_col) == 0 or len(numeric_cols) < 2:
            return []

        # For large N, sample to avoid combinatorial explosion
        cols_to_eval = numeric_cols
        if len(numeric_cols) > 200:
            # Pre-filter: top-200 by MI
            cols_to_eval = sorted(numeric_cols, key=lambda c: mi_per_col.get(c, 0.0), reverse=True)[:200]
        pairs = list(combinations(cols_to_eval, 2))

        # Sample rows for MI computation if needed
        X_eval = X
        y_vals = y.values if hasattr(y, "values") else np.asarray(y)
        if len(X) > 100_000:
            idx = np.random.RandomState(42).choice(len(X), size=100_000, replace=False)
            X_eval = X.iloc[idx]
            y_vals = y_vals[idx]

        try:
            from sklearn.feature_selection import (
                mutual_info_classif,
                mutual_info_regression,
            )
            n_unique = y.nunique() if hasattr(y, "nunique") else len(set(y_vals))
            is_clf = 1 < n_unique < 20
            mi_func = mutual_info_classif if is_clf else mutual_info_regression
        except ImportError:
            return []

        candidates = []
        for a, b in pairs:
            try:
                ratio_val = _safe_divide(X_eval, a, b)
                if ratio_val.std() == 0:
                    continue
                mi_ratio = mi_func(ratio_val.values.reshape(-1, 1), y_vals).item()
                mi_max = max(mi_per_col.get(a, 0.0), mi_per_col.get(b, 0.0))
                ratio_score = mi_ratio / mi_max if mi_max > 0 else 0.0
                if ratio_score >= self.mi_threshold:
                    candidates.append({
                        "_tmpl_json": json.dumps({
                            "type": "ratio",
                            "columns": [a, b],
                        }),
                        "score": float(ratio_score),
                    })
            except Exception:
                continue
        return candidates

    def _detect_outliers(
        self, X: pd.DataFrame, numeric_cols: List[str]
    ) -> List[Dict[str, Any]]:
        """Rule 4: Detect outlier-prone columns → clip candidates."""
        candidates = []
        for col in numeric_cols:
            try:
                vals = X[col].dropna()
                if len(vals) < 10:
                    continue
                q1 = vals.quantile(0.01)
                q99 = vals.quantile(0.99)
                iqr = q99 - q1
                if iqr == 0:
                    continue
                lower = q1 - 1.5 * iqr
                upper = q99 + 1.5 * iqr
                outliers_ratio = (
                    ((vals < lower) | (vals > upper)).sum() / len(vals)
                )
                if outliers_ratio > 0.05:
                    candidates.append({
                        "_tmpl_json": json.dumps({
                            "type": "clip",
                            "columns": [col],
                            "params": {
                                "lower": float(lower),
                                "upper": float(upper),
                            },
                        }),
                        "score": float(outliers_ratio),
                    })
            except Exception:
                continue
        return candidates

    def _greedy_select(
        self,
        candidates: List[Dict[str, Any]],
        X: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Greedy selection: filter by correlation and cap at max_features."""
        selected: List[Dict[str, Any]] = []
        selected_series: List[pd.Series] = []

        for cand in candidates:
            if len(selected) >= self.max_features:
                break
            try:
                tmpl = json.loads(cand["_tmpl_json"])
                test_series = self._apply_template(X, tmpl)
                if test_series is None:
                    continue
                # Skip constant series
                if test_series.std() == 0 or test_series.isna().all():
                    continue

                # Check correlation with every original numeric column
                redundant = False
                for oc in numeric_cols:
                    try:
                        if abs(test_series.corr(X[oc])) > self.corr_threshold:
                            redundant = True
                            break
                    except Exception:
                        continue

                # Check correlation with already selected
                if not redundant:
                    for prev in selected_series:
                        try:
                            if abs(test_series.corr(prev)) > self.corr_threshold:
                                redundant = True
                                break
                        except Exception:
                            continue

                if not redundant:
                    selected.append(tmpl)
                    selected_series.append(test_series)
            except Exception:
                continue

        return selected

    # ------------------------------------------------------------------
    # Template application engine
    # ------------------------------------------------------------------
    def _apply_template(
        self, X: pd.DataFrame, tmpl: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """Apply a single template to produce a feature Series."""
        ttype = tmpl["type"]
        cols = tmpl["columns"]
        params = tmpl.get("params", {})

        # Stateful / special transforms
        if ttype == "zscore":
            return self._apply_zscore(X, cols)
        if ttype == "ratio":
            return self._apply_ratio(X, cols)
        if ttype == "clip":
            return self._apply_clip(X, cols, params)
        if ttype in ("cyclic_sin", "cyclic_cos"):
            return self._apply_cyclic(X, cols, params, ttype)

        # General lookup
        if ttype not in _PREDEFINED_TRANSFORMS:
            self._log_warning(f"Unknown transform type: {ttype}")
            return None

        spec = _PREDEFINED_TRANSFORMS[ttype]
        if spec["func"] is None:
            self._log_warning(f"Transform '{ttype}' has no func and no special handler.")
            return None
        if len(cols) != spec["num_args"]:
            self._log_warning(
                f"Transform '{ttype}' expects {spec['num_args']} columns, got {len(cols)}."
            )
            return None

        # Ensure columns exist
        missing = [c for c in cols if c not in X.columns]
        if missing:
            self._log_warning(f"Columns missing for {ttype}: {missing}")
            return None

        return spec["func"](X, *cols)

    def _apply_zscore(self, X: pd.DataFrame, cols: List[str]) -> Optional[pd.Series]:
        """Apply z-score using stored fit-time statistics."""
        col = cols[0]
        if col not in X.columns:
            return None
        if col not in self._zscore_stats:
            self._log_warning(f"No zscore stats stored for '{col}'; returning raw.")
            return X[col].astype(float)
        stats = self._zscore_stats[col]
        return (X[col].astype(float) - stats["mean"]) / stats["std"]

    def _apply_ratio(self, X: pd.DataFrame, cols: List[str]) -> Optional[pd.Series]:
        """Safe ratio a / b."""
        if len(cols) != 2:
            return None
        for c in cols:
            if c not in X.columns:
                return None
        return _safe_divide(X, cols[0], cols[1])

    def _apply_clip(
        self, X: pd.DataFrame, cols: List[str], params: Dict[str, Any]
    ) -> Optional[pd.Series]:
        """Clip values to [lower, upper]."""
        col = cols[0]
        if col not in X.columns:
            return None
        lower = params.get("lower", -np.inf)
        upper = params.get("upper", np.inf)
        return X[col].astype(float).clip(lower=lower, upper=upper)

    def _apply_cyclic(
        self,
        X: pd.DataFrame,
        cols: List[str],
        params: Dict[str, Any],
        ttype: str,
    ) -> Optional[pd.Series]:
        """Apply sin or cos with period."""
        col = cols[0]
        if col not in X.columns:
            return None
        period = params.get("period")
        if period is None:
            self._log_warning(f"Missing 'period' for {ttype} on '{col}'")
            return None
        spec = _PREDEFINED_TRANSFORMS[ttype]
        if spec["func"] is None:
            return None
        return spec["func"](X, col, period)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_output_name(tmpl: Dict[str, Any]) -> str:
        """Derive output column name from template."""
        if "output_name" in tmpl:
            return tmpl["output_name"]
        ttype = tmpl["type"]
        cols = tmpl["columns"]
        return f"{ttype}_{'_'.join(cols)}"