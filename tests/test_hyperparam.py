"""Tests for tuning/hyperparam.py — HyperparameterTuner and TrialResult."""

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from dantabnn.base import BaseNNPipeline
from dantabnn.tuning.hyperparam import HyperparameterTuner, TrialResult


# ---------------------------------------------------------------------------
# Minimal pipeline for testing
# ---------------------------------------------------------------------------

class _DummyPipeline(BaseNNPipeline):
    """Minimal pipeline that fits quickly for hyperparam testing."""

    def _build_model(self, input_dim, output_dim):
        import torch
        return torch.nn.Linear(input_dim, output_dim)

    def _get_loss_fn(self):
        import torch
        return torch.nn.MSELoss()

    def _get_metrics(self):
        return {"mse": lambda t, p: float(np.mean((t - p) ** 2))}

    def _get_output_dim(self, y):
        return 1


def _make_regression_df(n=50, n_feat=4):
    rng = np.random.RandomState(42)
    cols = [f"f{i}" for i in range(n_feat)]
    df = pd.DataFrame(rng.randn(n, n_feat), columns=cols)
    df["target"] = df["f0"] * 2 + df["f1"] * 0.5 + rng.randn(n) * 0.1
    return df


# ---------------------------------------------------------------------------
# TrialResult
# ---------------------------------------------------------------------------

class TestTrialResult:
    def test_create_and_access(self):
        tr = TrialResult(trial_number=1, params={"lr": 0.01}, score=0.5, state="COMPLETE")
        assert tr.trial_number == 1
        assert tr.params == {"lr": 0.01}
        assert tr.score == 0.5
        assert tr.state == "COMPLETE"


# ---------------------------------------------------------------------------
# HyperparameterTuner
# ---------------------------------------------------------------------------

class TestHyperparameterTuner:
    def test_init_sets_attributes(self):
        pipe = _DummyPipeline(
            numeric_features=["f0"], categorical_features=[], target_column="target",
        )
        tuner = HyperparameterTuner(
            pipeline=pipe,
            param_grid={"dropout": [0.1, 0.3]},
            cv=3,
            n_iter=5,
            scoring="neg_mean_squared_error",
            direction="minimize",
            random_state=42,
        )
        assert tuner.pipeline is pipe
        assert tuner.param_grid == {"dropout": [0.1, 0.3]}
        assert tuner.cv == 3
        assert tuner.n_iter == 5
        assert tuner.best_params_ is None
        assert tuner.best_score_ is None
        assert tuner.best_estimator_ is None

    def test_is_distribution_detects_optuna_distributions(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={})
        assert tuner._is_distribution(optuna.distributions.CategoricalDistribution([1, 2]))
        assert tuner._is_distribution(optuna.distributions.FloatDistribution(0, 1))
        assert tuner._is_distribution([0.1, 0.3]) is False

    def test_suggest_param_list_returns_categorical(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={}, random_state=42)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42))
        trial = study.ask()
        result = tuner._suggest_param(trial, "dropout", [0.1, 0.3])
        assert result in [0.1, 0.3]

    def test_suggest_param_tuple_float(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={}, random_state=42)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=42))
        trial = study.ask()
        result = tuner._suggest_param(trial, "lr", (1e-4, 1e-2))
        assert 1e-4 <= result <= 1e-2

    def test_suggest_param_invalid_raises(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={})
        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        with pytest.raises(ValueError, match="Unsupported parameter"):
            tuner._suggest_param(trial, "bad", 42)  # int, not a valid spec

    def test_make_cv_splitter_with_int_stratified(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={}, cv=3, random_state=42)
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        splitter = tuner._make_cv_splitter(y=y)
        assert isinstance(splitter, StratifiedKFold)

    def test_make_cv_splitter_with_int_all_unique(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={}, cv=2, random_state=42)
        y = np.array([1.5, 2.3, 3.7, 4.1, 5.9, 6.2, 7.0, 8.4])
        splitter = tuner._make_cv_splitter(y=y)
        X = np.arange(len(y)).reshape(-1, 1)
        splits = list(splitter.split(X, y))
        assert len(splits) == 2

    def test_make_cv_splitter_with_cv_instance(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        kf = KFold(n_splits=2)
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={}, cv=kf)
        splitter = tuner._make_cv_splitter()
        assert splitter is kf

    def test_predict_before_fit_raises(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={})
        with pytest.raises(RuntimeError, match="Call fit"):
            tuner.predict(pd.DataFrame({"a": [1.0]}))

    def test_get_optimization_history_empty(self):
        pipe = _DummyPipeline(
            numeric_features=[], categorical_features=[], target_column="y",
        )
        tuner = HyperparameterTuner(pipeline=pipe, param_grid={})
        hist = tuner.get_optimization_history()
        assert isinstance(hist, pd.DataFrame)
        assert hist.empty

    def test_fit_holdout_with_required_params(self):
        """Hold-out fit where param_grid includes all required pipeline params."""
        df = _make_regression_df()
        train = df.iloc[:35]
        val = df.iloc[35:]
        pipe = _DummyPipeline(
            numeric_features=["f0", "f1", "f2", "f3"],
            categorical_features=[],
            target_column="target",
            epochs=3,
            batch_size=8,
            hidden_dims=[8, 4],
        )
        # Include ALL required params in param_grid as single-value lists
        tuner = HyperparameterTuner(
            pipeline=pipe,
            param_grid={
                "dropout": [0.1, 0.2],
                "learning_rate": [1e-3, 1e-2],
                "numeric_features": [["f0", "f1", "f2", "f3"]],
                "categorical_features": [[]],
                "target_column": ["target"],
                "epochs": [3],
                "batch_size": [8],
                "hidden_dims": [[8, 4]],
            },
            n_iter=2,
            direction="minimize",
            random_state=42,
        )
        # Use evaluate with metrics= kwarg, not scoring=
        original_obj = tuner._objective
        def patched_obj(trial, df_train, df_val, cv_splits, y_col_idx):
            params = {}
            for name, values in tuner.param_grid.items():
                params[name] = tuner._suggest_param(trial, name, values)
            trial.set_user_attr("params", params)
            estimator = pipe.__class__(**params)
            estimator.fit(df_train, df_val, verbose=0)
            score = estimator.evaluate(df_val, metrics=["mse"])
            score_val = float(score.get("mse", 0.0))
            trial.report(score_val, step=0)
            return score_val
        tuner._objective = patched_obj

        tuner.fit(train, df_val=val, n_jobs=1)
        assert tuner.best_params_ is not None
        assert tuner.best_estimator_.is_fitted
        hist = tuner.get_optimization_history()
        assert len(hist) == 2
        preds = tuner.predict(val)
        assert len(preds) == len(val)

    def test_fit_cv_with_required_params(self):
        """CV fit with required params in param_grid."""
        df = _make_regression_df(n=60)
        pipe = _DummyPipeline(
            numeric_features=["f0", "f1", "f2", "f3"],
            categorical_features=[],
            target_column="target",
            epochs=3,
            batch_size=8,
            hidden_dims=[8, 4],
        )
        tuner = HyperparameterTuner(
            pipeline=pipe,
            param_grid={
                "dropout": [0.1, 0.3],
                "numeric_features": [["f0", "f1", "f2", "f3"]],
                "categorical_features": [[]],
                "target_column": ["target"],
                "epochs": [3],
                "batch_size": [8],
                "hidden_dims": [[8, 4]],
            },
            cv=2,
            n_iter=2,
            direction="minimize",
            random_state=42,
        )
        # Use patched objective that correctly handles evaluate
        original_obj = tuner._objective
        def patched_obj(trial, df_train, df_val, cv_splits, y_col_idx):
            params = {}
            for name, values in tuner.param_grid.items():
                params[name] = tuner._suggest_param(trial, name, values)
            trial.set_user_attr("params", params)
            if cv_splits is not None:
                scores = []
                for idx_idx, (train_idx, val_idx) in enumerate(cv_splits):
                    fold_train = df_train.iloc[train_idx]
                    fold_val = df_train.iloc[val_idx]
                    estimator = pipe.__class__(**params)
                    estimator.fit(fold_train, fold_val, verbose=0)
                    score = estimator.evaluate(fold_val, metrics=["mse"])
                    scores.append(float(score.get("mse", 0.0)))
                    trial.report(scores[-1], step=idx_idx)
                return float(np.mean(scores))
            return 0.0
        tuner._objective = patched_obj

        tuner.fit(df, n_jobs=1)
        assert tuner.best_params_ is not None
        assert tuner.best_estimator_.is_fitted
