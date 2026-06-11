"""Base abstract class for neural network pipelines."""

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Any, Callable

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import DataLoader, TensorDataset

from .preprocessing.encoder import CategoricalEncoder
from .preprocessing.scaler import StandardScaler
from .preprocessing.outlier import OutlierClipper
from .preprocessing.imputer import NaNImputer
from .preprocessing.feature_engineer import AutoFeatureEngineer
from .utils.logger import setup_logger
from .utils.metrics import compute_metrics

logger = setup_logger(__name__)


class BaseNNPipeline(ABC):
    def __init__(
            self,
            numeric_features: List[str],
            categorical_features: List[str],
            target_column: str,

            # Model architecture — wider 3-layer default from v2-baseline experiments
            hidden_dims: List[int] = [128, 64, 32],
            dropout: float = 0.2,
            attention_heads: int = 4,

            # Feature gating (differentiable feature selection)
            gating_type: str = 'soft',
            gating_k: int = 10,
            gating_temperature: float = 1.0,
            gating_hard: bool = True,
            gating_dropout: float = 0.0,
            gating_init_bias: float = 0.0,

            # Batch normalization
            use_batch_norm: bool = False,

            # Training
            batch_size: int = 32,
            epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            early_stopping_patience: int = 10,

            # Preprocessing
            scale_numeric: bool = True,
            encode_categorical: bool = True,
            clip_outliers: bool = True,
            impute_missing: bool = True,
            engineer_features: bool = True,
            engineer_max_features: int = 100,

            # Device
            device: Optional[str] = None,

            # Random seed
            random_state: int = 42

    ):
        # Feature columns
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column

        # Architecture
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.attention_heads = attention_heads

        # Feature gating
        self.gating_type = gating_type
        self.gating_k = gating_k
        self.gating_temperature = gating_temperature
        self.gating_hard = gating_hard
        self.gating_dropout = gating_dropout
        self.gating_init_bias = gating_init_bias

        # Batch normalization
        self.use_batch_norm = use_batch_norm

        # Training hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        # Preprocessing flags
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical
        self.clip_outliers = clip_outliers
        self.impute_missing = impute_missing
        self.engineer_features = engineer_features
        self.engineer_max_features = engineer_max_features

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        self._set_seed()

        # Internal state 
        self.model: Optional[nn.Module] = None 
        self.scaler: Optional[StandardScaler] = None 
        self.encoder: Optional[CategoricalEncoder] = None
        self.outlier_clipper: Optional[OutlierClipper] = None
        self.imputer: Optional[NaNImputer] = None
        self.feature_engineer: Optional[AutoFeatureEngineer] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False 
        self.history: Dict[str, List[float]] = {}
        self.best_epoch = 0
        self.best_state: Optional[OrderedDict[str, torch.Tensor]] = None

    def _set_seed(self):
        """set random seeds for reproducibility"""
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

    @abstractmethod
    def _build_model(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build and return the Pytorch model. 
        
        Parameters
        ----------

        input_dim : int 
            Dimension of the input features after preprocessing.
        output_dim : int 
            Dimension of the output (e.g., 1 for regression/binary, num_classes for multiclass).

        Returns
        -------
        nn.Module
            The neural network model.
        """
        pass

    @abstractmethod
    def _get_loss_fn(self) -> nn.Module:
        """Return the loss funciton for the task."""
        pass

    @abstractmethod
    def _get_metrics(self) -> Dict[str, Callable]:
        """Return a dictionary of metric functions (name -> callable)."""
        pass

    def _prepare_features(
            self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """Preprocess numeric and categorical features.

        Pipeline:
        1. Impute missing values (if enabled)
        2. Clip outliers via IQR (if enabled)
        3. Engineer features — x² + log1p auto-detection (if enabled)
        4. Scale numeric
        5. Encode categorical
        6. Combine into single tensor

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the preprocessors (True) or use already fitted ones (False).

        Returns
        -------
        torch.Tensor
            Preprocessed tensor of shape (n_samples, n_features).
        List[str]
            Names of the features after preprocessing.
        """
        numeric_data = df[self.numeric_features].values.astype(np.float64) if self.numeric_features else np.empty((len(df), 0))
        categorical_data = df[self.categorical_features].values if self.categorical_features else np.empty((len(df), 0))

        # Step 0: Impute missing values
        if self.impute_missing and numeric_data.size > 0:
            if fit:
                self.imputer = NaNImputer()
                numeric_data = self.imputer.fit_transform(numeric_data)
            else:
                if self.imputer is None:
                    raise RuntimeError("Imputer not fitted. Call fit first.")
                numeric_data = self.imputer.transform(numeric_data)

        # Step 1: Clip outliers via IQR
        if self.clip_outliers and numeric_data.size > 0:
            if fit:
                self.outlier_clipper = OutlierClipper()
                numeric_data = self.outlier_clipper.fit_transform(numeric_data)
            else:
                if self.outlier_clipper is None:
                    raise RuntimeError("OutlierClipper not fitted. Call fit first.")
                numeric_data = self.outlier_clipper.transform(numeric_data)

        # Step 2: Engineer features (x² + log1p auto-detection)
        if self.engineer_features and numeric_data.size > 0:
            if fit:
                self.feature_engineer = AutoFeatureEngineer(
                    max_generated=self.engineer_max_features,
                )
                numeric_data = self.feature_engineer.fit_transform(numeric_data)
            else:
                if self.feature_engineer is None:
                    raise RuntimeError("FeatureEngineer not fitted. Call fit first.")
                numeric_data = self.feature_engineer.transform(numeric_data)

        # Step 3: Scale numeric features
        if self.scale_numeric and numeric_data.size > 0:
            if fit:
                self.scaler = StandardScaler()
                numeric_scaled = self.scaler.fit_transform(numeric_data)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler not fitted. Call fit first.")
                numeric_scaled = self.scaler.transform(numeric_data)
        else:
            numeric_scaled = numeric_data

        # Step 4: Encode categorical features
        if self.encode_categorical and categorical_data.size > 0:
            if fit:
                self.encoder = CategoricalEncoder()
                categorical_encoded = self.encoder.fit_transform(categorical_data)
            else:
                if self.encoder is None:
                    raise RuntimeError("Encoder not fitted. Call fit first.")
                categorical_encoded = self.encoder.transform(categorical_data)
        else:
            categorical_encoded = categorical_data

        # Step 5: Combine features (handle sparse categorical encoding)
        cat_dense = categorical_encoded.toarray() if issparse(categorical_encoded) else categorical_encoded
        features = np.hstack([numeric_scaled, cat_dense]) if (
                numeric_scaled.size > 0 or cat_dense.size > 0) else np.empty((len(df), 0))

        # Build feature names (before memory cleanup — needs numeric_scaled.shape)
        if self.engineer_features and self.feature_engineer is not None:
            n_orig = len(self.numeric_features)
            n_gen = numeric_scaled.shape[1] - n_orig if hasattr(numeric_scaled, 'ndim') and numeric_scaled.ndim > 1 else 0
            gen_names = [f"gen_{i}" for i in range(n_gen)] if n_gen > 0 else []
            num_names = [f"num_{f}" for f in self.numeric_features] + gen_names
        else:
            num_names = [f"num_{f}" for f in self.numeric_features]

        if self.encode_categorical and self.encoder is not None:
            cat_names = [
                f"cat_{self.categorical_features[i]}"
                for i in range(len(self.categorical_features))
                for _ in range(
                    self.encoder.n_values_per_feature[i]
                )
            ]
        else:
            cat_names = []

        features_names = num_names + cat_names

        # Memory cleanup: free intermediate numpy arrays
        del numeric_data, categorical_data, numeric_scaled, cat_dense
        gc.collect()

        # Convert to float32 for GPU transfer
        features = features.astype(np.float32) if features.dtype == np.float64 else features
        return torch.FloatTensor(features).to(self.device), features_names

    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Ectract target column and convert to tensor."""
        target = df[self.target_column].values
        # Subclasses may override this to reshape or encode target differently
        return torch.FloatTensor(target).to(self.device)

    def _prepare_data(self, data, fit=False) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Preprocess and convert to tensors features and target"""
        features, feature_names = self._prepare_features(data, fit=fit)
        target = self._prepare_target(data)
        return features, target, feature_names
    
    def _create_dataloader(
            self, features: torch.Tensor, target: Optional[torch.Tensor] = None, shuffle: bool = False
    ) -> DataLoader:
        """Create a Pytorch DataLoader from tensors."""
        if target is not None:
            dataset = TensorDataset(features, target)
        else:
            dataset = TensorDataset(features)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None,
            verbose: int = 1,
    ) -> "BaseNNPipeline":
        """Fit the pipeline on training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data
        df_val : pd.DataFrame, optional
            Validation data for early stopping
        verbose : int
            Verbosity level (0 = silent, 1 = progress, 2 = detailed).

        Returns
        -------
        self
        """
        logger.info("Starting pipline fitting...")
        self.feature_names = None

        # Prepare features and target for train, validation data 
        train_features, train_target, self.feature_names = self._prepare_data(df_train, fit=True)
        val_features, val_target, _ = self._prepare_data(df_val, fit=False) if df_val is not None else (None, None, None)

        # Determine input/output dimensions
        input_dim = train_features.shape[1]
        output_dim = self._get_output_dim(train_target)
        logger.debug(f"Input dim: {input_dim}, output dim: {output_dim}")

        # Validate input dimension (P1: prevent OOM on large feature sets)
        if input_dim > 512:
            raise ValueError(
                f"Input dimension {input_dim} exceeds DANet limit of 512. "
                f"Reduce feature set: disable engineer_features, "
                f"reduce engineer_max_features, or use OrdinalEncoder instead of OneHot."
            )
        if input_dim > 256:
            logger.warning(f"Large input dimension ({input_dim}). "
                          f"Consider reducing features for speed and memory.")

        # Build model 
        self.model = self._build_model(input_dim, output_dim).to(self.device)
        loss_fn = self._get_loss_fn()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(5, self.early_stopping_patience // 2),
            factor=0.5, min_lr=1e-6,
        )

        # Training loop
        train_loader = self._create_dataloader(train_features, train_target, shuffle=True)
        val_loader = self._create_dataloader(val_features, val_target, shuffle=False) \
            if val_features is not None else None

        self.history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        min_delta = 1e-4
        min_epochs = max(10, self.early_stopping_patience)

        for epoch in range(self.epochs):
            # Training 
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item() * batch_X.size(0)

            epoch_train_loss /= len(train_loader.dataset)
            self.history["train_loss"].append(epoch_train_loss)

            # Validation
            if val_loader is not None:
                self.model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        pred = self.model(batch_X)
                        loss = loss_fn(pred, batch_y)
                        epoch_val_loss += loss.item() * batch_X.size(0)
                epoch_val_loss /= len(val_loader.dataset)
                self.history["val_loss"].append(epoch_val_loss)
                scheduler.step(epoch_val_loss)
            else:
                epoch_val_loss = None
                scheduler.step(epoch_train_loss)

            # Early stopping — requires minimum epochs + meaningful improvement
            if val_loader is not None and epoch >= min_epochs:
                if epoch_val_loss < best_val_loss - min_delta:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    self.best_epoch = epoch
                    self.best_state = self.model.state_dict()
                elif epoch_val_loss < best_val_loss:
                    # Marginal improvement (< min_delta) — update best but also count patience
                    best_val_loss = epoch_val_loss
                    self.best_state = self.model.state_dict()
                    patience_counter += 1
                    if patience_counter > self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                else:
                    patience_counter += 1
                    if patience_counter > self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            else:
                self.best_state = self.model.state_dict()
            
            if verbose >= 1 and epoch % 10 == 0:
                msg = f"Epoch {epoch}: train loss = {epoch_train_loss:.4f}"
                if epoch_val_loss is not None:
                    msg += f", val loss = {epoch_val_loss:.4f}"
                logger.info(msg)

        # Restore best model
        if hasattr(self, "best_state"):
            self.model.load_state_dict(self.best_state)
        self.is_fitted = True
        logger.info("Fitting completed.")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for the input data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        
        Returns
        -------
        np.ndarray
            Predictions
        
        """
        if not self.is_fitted:
            raise RuntimeError("Pipline not fitted. Call fit first.")
        self.model.eval()
        features, _ = self._prepare_features(df, fit=False)
        loader = self._create_dataloader(features, target=None, shuffle=False)
        predictions = []
        with torch.no_grad():
            for (batch_X,) in loader:
                pred = self.model(batch_X)
                predictions.append(pred.cpu().numpy())
        return np.vstack(predictions)   

    def evaluate(
            self, df: pd.DataFrame, metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate the model on the given data.
        
        Parameters
        ----------
        df : pd.DataFrame 
             DataFrame containing both feature and target.
        metrics : List[str], optional
            List of metric names to compute. If None, uses the default metrics
            defined by "_get_metrics".

        Returns
        -------
        Dict[str, float]
            Dictionary of metric scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipline not fitted. Call fit first.")
        y_true = self._prepare_target(df).cpu().numpy()
        y_pred = self.predict(df)
        metric_funcs = self._get_metrics()

        if metrics:
            # Allow custom metric selection
            metric_funcs = {m: metric_funcs[m] for m in metrics if m in metric_funcs}

        return compute_metrics(y_true, y_pred, metric_funcs)

    def hyperparameters_tuning(
            self,
            df_train: pd.DataFrame,
            param_grid: Optional[Dict[str, List[Any]]] = None,
            df_val: Optional[pd.DataFrame] = None,
            cv: Union[int, BaseCrossValidator] = 5,
            n_iter: Optional[int] = 50,
            scoring: str = "neg_mean_squared_error",
            direction: str = "minimize",
            random_state: Optional[int] = None,
            n_jobs: int = 1,
            verbose: int = 1,
            show_progress_bar: bool = False,
            small_search: bool = False,
            **tuner_kwargs,
    ) -> "BaseNNPipeline":
        """Perform hyperparameter tuning using Optuna Bayesian optimization.

        If ``param_grid`` is not provided, automatically generates one
        suitable for DANetModule based on the pipeline's architecture.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training data.
        param_grid : Dict[str, List[Any]], optional
            Dictionary with parameter names and lists of values to try,
            or Optuna distribution objects. If None, auto-generated for DANet.
        df_val : pd.DataFrame, optional
            Validation data for hold-out evaluation. If provided, ``cv``
            is ignored.
        cv : int or BaseCrossValidator, default=5
            Cross-validation strategy when ``df_val`` is None.
        n_iter : int, optional
            Number of Optuna trials. Default is 50.
        scoring : str, default="neg_mean_squared_error"
            Scoring metric (sklearn scorer string).
        direction : {"minimize", "maximize"}, default="minimize"
            Whether the metric should be minimized or maximized.
        random_state : int, optional
            Seed for reproducibility. Defaults to ``self.random_state``.
        n_jobs : int, default=1
            Number of parallel trials. Use -1 for all cores.
        verbose : int, default=1
            Verbosity level.
        show_progress_bar : bool, default=False
            Show tqdm progress bar during optimization.
        small_search : bool, default=False
            If True and auto-generating param_grid, use a minimal grid
            for quick prototyping (~10 trials).
        **tuner_kwargs
            Additional arguments passed to HyperparameterTuner
            (e.g., ``pruner``, ``study_name``).

        Returns
        -------
        BaseNNPipeline
            A *new* pipeline instance fitted with the best hyperparameters.
            The original instance remains unmodified.
        """
        from .tuning.hyperparam import HyperparameterTuner
        from .tuning.tune_utils import get_danet_param_grid, get_danet_param_mapper

        # Auto-generate param_grid if not provided
        if param_grid is None:
            # Infer input_dim from training data after preprocessing
            train_features, _, _ = self._prepare_data(df_train, fit=True)
            input_dim = train_features.shape[1]

            param_grid = get_danet_param_grid(
                input_dim=input_dim,
                small_search=small_search,
            )

            if verbose >= 1:
                logger.info(
                    f"Auto-generated param_grid for DANet "
                    f"(input_dim={input_dim}, small_search={small_search})"
                )

        # Use instance random_state if none provided
        if random_state is None:
            random_state = self.random_state

        tuner = HyperparameterTuner(
            pipeline=self,
            param_grid=param_grid,
            cv=cv,
            n_iter=n_iter if n_iter is not None else 50,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            direction=direction,
            param_mapper=get_danet_param_mapper,
            **tuner_kwargs,
        )

        tuner.fit(
            df_train=df_train,
            df_val=df_val,
            verbose=verbose,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        return tuner.best_estimator_

    def save(self, path: Union[str, Path]) -> None:
        """Save the entire pipeline (model, scaler, encoder, hyperparameters) to disk.
        
        Parameters
        ----------
        path : str or Path
            Directory where the pipeline will be saved.
        """
        import joblib
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path / "model.pt")

            # Save processing objects
            if self.scaler is not None:
                joblib.dump(self.scaler, path / "scaler.joblib")
            if self.encoder is not None:
                joblib.dump(self.encoder, path / "encoder.joblib")
            if self.outlier_clipper is not None:
                joblib.dump(self.outlier_clipper, path / "outlier_clipper.joblib")
            if self.imputer is not None:
                joblib.dump(self.imputer, path / "imputer.joblib")
            if self.feature_engineer is not None:
                joblib.dump(self.feature_engineer, path / "feature_engineer.joblib")

            # Save hyperparameters
            joblib.dump(self.hyperparameters, path / "hyperparameters.joblib")
            logger.info(f"Pipeline saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save Pipline to {path}, Error: {e}")

    def load(self, path: Union[str, Path]) -> "BaseNNPipeline":
        """Load previously saved pipeline.
        
        Parameters
        ----------
        path : str or Path
            Directory where pipeline was saved.

        Returns 
        -------
        self
        """
        import joblib
        path = Path(path)

        # Load hyperparameters
        hyperparameters = joblib.load(path / "hyperparameters.joblib")
        self.set_params(**hyperparameters)

        # Load Processing objects
        scaler_path = path / "scaler.joblib"
        encoder_path = path / "encoder.joblib"
        outlier_path = path / "outlier_clipper.joblib"
        imputer_path = path / "imputer.joblib"
        engineer_path = path / "feature_engineer.joblib"

        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        if encoder_path.exists():
            self.encoder = joblib.load(encoder_path)
        if outlier_path.exists():
            self.outlier_clipper = joblib.load(outlier_path)
        if imputer_path.exists():
            self.imputer = joblib.load(imputer_path)
        if engineer_path.exists():
            self.feature_engineer = joblib.load(engineer_path)

        # Rebuild Model — compute input_dim from feature counts
        num_features = len(self.numeric_features)
        if self.engineer_features and self.feature_engineer is not None:
            num_features = self.feature_engineer.n_features_out
        if self.encode_categorical and self.encoder is not None:
            cat_features = sum(
                len(self.encoder.categories_[i])
                for i in range(len(self.categorical_features))
            )
        elif self.categorical_features:
            cat_features = len(self.categorical_features)  # fallback if no encoder fitted
        else:
            cat_features = 0
        input_dim = num_features + cat_features if (num_features + cat_features) > 0 else 1
        output_dim = self._get_output_dim(torch.zeros(0))

        self.model = self._build_model(input_dim, output_dim).to(self.device)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))
        self.is_fitted = True

        logger.info(f"Pipeline loaded from {path}")
        return self

    def get_feature_importance(self, method: str = "attention") -> pd.DataFrame:
        """Compute feature importance scores.
        
        Parameters
        ----------
        method : str 
            Method to compute importance ('attention', 'gradient', 'permutation').
        
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipline not fitted. Call fit first.")
        
        # Placeholder implementation - should be overridden by subclasses
        # or implemented in a separated module.
        importance = np.ones(len(self.feature_names)) if self.feature_names else np.array([])
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

    def get_model(self):
        """Return the enderlying PyTorch model."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        return self.model

    def get_preprocessor(self):
        """Return the fitted preprocessing objects."""
        return {
            "scaler": self.scaler,
            "encoder": self.encoder,
            "outlier_clipper": self.outlier_clipper,
            "imputer": self.imputer,
            "feature_engineer": self.feature_engineer,
        }

    def _get_output_dim(self, y: torch.Tensor) -> int:
        """Determine ouput dimension base on target tensor."""
        # Default: regression/binary -> 1 (current usage only for this 2 types of tasks)
        # Override in multiclass classification
        return 1

    def set_params(self, **params):
        """Set pipeline parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Ignoring unknown parameter '{key}'.")
        return self

    @property
    def hyperparameters(self):
        """Get pipline parameters."""
        exclude = {
            "model", "scaler", "encoder",
            "outlier_clipper", "imputer", "feature_engineer",
            "is_fitted", "history", "best_state",
        }
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in exclude}

    def __repr__(self):
        return f"{self.__class__.__name__}(target_column={self.target_column})"
