"""Hyperparameter tuning utilities using Optuna."""

from dataclasses import dataclass
from typing import Dict, List, Any, Union, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from ..base import BaseNNPipeline
from ..utils.logger import setup_logger

logger = setup_logger("HyperparametersTuner")

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TrialResult:
    """Container for a single trial's outcome."""
    trial_number: int
    params: Dict[str, Any]
    score: float
    state: str


class HyperparameterTuner:
    """Hyperparameters tuner for neural network pipelines using Optuna.

    Guarantees that ``df_train`` and ``df_val`` are never copied inside
    the objective function. Only integer index arrays and scalar
    parameters cross the trial boundary.
    """

    def __init__(
        self,
        pipeline: BaseNNPipeline,
        param_grid: Dict[str, List[Any]],
        cv: Union[int, BaseCrossValidator] = 5,
        n_iter: int = 50,
        scoring: str = "neg_mean_squared_error",
        n_jobs: int = -1,
        verbose: int = 0,
        random_state: Optional[int] = None,
        direction: str = "minimize",
        pruner: Optional[optuna.pruners.BasePruner] = None,
        study_name: Optional[str] = None,
    ):
        self.pipeline = pipeline
        self.param_grid = param_grid
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        self.direction = direction
        self.pruner = pruner
        self.study_name = study_name

        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None
        self.best_estimator_: Optional[BaseNNPipeline] = None
        self.study_: Optional[optuna.Study] = None
        self.cv_results_: List[TrialResult] = []

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _is_distribution(self, value: Any) -> bool:
        return isinstance(
            value,
            (
                optuna.distributions.CategoricalDistribution,
                optuna.distributions.FloatDistribution,
                optuna.distributions.IntDistribution,
            ),
        )

    def _suggest_param(self, trial: optuna.Trial, name: str, values: Any) -> Any:
        """Map a param_grid entry to the correct ``trial.suggest_*`` call."""
        if self._is_distribution(values):
            return trial._suggest(name, values)
        if isinstance(values, list):
            return trial.suggest_categorical(name, values)
        if isinstance(values, tuple) and len(values) == 2:
            return trial.suggest_float(name, values[0], values[1])
        if isinstance(values, tuple) and len(values) == 3 and isinstance(values[2], int):
            return trial.suggest_int(name, values[0], values[1], step=values[2])
        raise ValueError(
            f"Unsupported parameter specification for '{name}': {values!r}. "
            "Expected list, tuple(low, high), tuple(low, high, step), or "
            "an Optuna distribution object."
        )

    def _make_cv_splitter(self, y: Optional[np.ndarray] = None) -> BaseCrossValidator:
        if isinstance(self.cv, BaseCrossValidator):
            return self.cv
        if y is not None and len(np.unique(y)) > 1:
            return StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
        return KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

    # ------------------------------------------------------------------ #
    # Objective — ZERO data copying
    # ------------------------------------------------------------------ #
    def _objective(
        self,
        trial: optuna.Trial,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame],
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]],
        y_col_idx: int,
    ) -> float:
        """Optuna objective. Only lightweight indices enter; no DataFrame copies."""

        # 1. Sample hyperparameters (scalars only)
        params: Dict[str, Any] = {}
        for name, values in self.param_grid.items():
            params[name] = self._suggest_param(trial, name, values)
        trial.set_user_attr("params", params)

        # 2. Fit & score — no concat, no copy, no reindex
        if df_val is not None:
            # ---- Hold-out mode: single fit on full train, evaluate on val ----
            estimator = self.pipeline.__class__(**params)
            estimator.fit(df_train, df_val, verbose=0)
            score = estimator.evaluate(df_val, scoring=self.scoring)

            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            # ---- CV mode: slice by pre-computed integer index arrays ----
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits, start=1):
                # iloc with integer arrays returns a view when the underlying
                # blocks are contiguous; in all cases it does NOT copy the
                # underlying data buffer (only a new BlockManager shell).
                fold_train = df_train.iloc[train_idx]
                fold_val = df_train.iloc[val_idx]

                estimator = self.pipeline.__class__(**params)
                estimator.fit(fold_train, fold_val, verbose=0)
                fold_score = estimator.evaluate(fold_val, scoring=self.scoring)
                scores.append(fold_score)

                trial.report(fold_score, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            score = float(np.mean(scores))

        # Optuna always minimizes; flip sign if user wants maximize
        return -score if self.direction == "maximize" else score

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame = None,
        y_col: Optional[str] = None,
        verbose: int = 1,
        n_jobs: int = 1,
        show_progress_bar: bool = False,
    ) -> "HyperparameterTuner":
        """Run hyperparameter search with Optuna.

        Guarantees that ``df_train`` and ``df_val`` are never copied
        inside the optimization loop. Only integer index arrays cross
        the trial boundary.
        """
        if df_val is not None and self.verbose >= 1:
            logger.info("Validation data provided — using hold-out evaluation (cv ignored).")

        # Pre-compute CV splits once, outside the objective
        cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        y_col_idx: int = -1

        if df_val is None:
            if y_col is None:
                y_col = df_train.columns[-1]
            y_col_idx = df_train.columns.get_loc(y_col)
            y = df_train.iloc[:, y_col_idx].values

            cv_splitter = self._make_cv_splitter(y)
            # Store only integer index arrays — negligible memory
            cv_splits = [
                (np.array(train_idx), np.array(val_idx))
                for train_idx, val_idx in cv_splitter.split(df_train, y)
            ]

        # Pruner
        pruner = self.pruner if self.pruner is not None else optuna.pruners.MedianPruner()

        # Study
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        # Objective closure captures only references + lightweight indices
        objective = lambda trial: self._objective(
            trial, df_train, df_val, cv_splits, y_col_idx
        )

        # Optimize
        self.study_.optimize(
            objective,
            n_trials=self.n_iter,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            catch=(Exception,),
        )

        # Results
        self.cv_results_ = [
            TrialResult(
                trial_number=t.number,
                params=t.params,
                score=t.value,
                state=t.state.name,
            )
            for t in self.study_.trials
        ]

        # Best trial
        best_trial = self.study_.best_trial
        self.best_params_ = best_trial.params
        raw_best = best_trial.value
        self.best_score_ = -raw_best if self.direction == "maximize" else raw_best

        # Refit best estimator on full data
        self.best_estimator_ = self.pipeline.__class__(**self.best_params_)
        if df_val is not None:
            self.best_estimator_.fit(df_train, df_val, verbose=verbose)
        else:
            self.best_estimator_.fit(df_train, None, verbose=verbose)

        if (self.verbose or verbose) >= 1:
            logger.info(f"Best trial: #{best_trial.number}")
            logger.info(f"Best params: {self.best_params_}")
            logger.info(f"Best score ({self.scoring}): {self.best_score_:.6f}")

        return self

    def predict(self, df: pd.DataFrame) -> Any:
        if self.best_estimator_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.best_estimator_.predict(df)

    def get_optimization_history(self) -> pd.DataFrame:
        if not self.cv_results_:
            return pd.DataFrame()
        records = [
            {"trial": r.trial_number, "score": r.score, "state": r.state, **r.params}
            for r in self.cv_results_
        ]
        return pd.DataFrame(records)

    def plot_optimization_history(self):
        if self.study_ is None:
            raise RuntimeError("Call fit() first.")
        return optuna.visualization.plot_optimization_history(self.study_)
