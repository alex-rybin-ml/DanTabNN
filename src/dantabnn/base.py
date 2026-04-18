from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocessing.scaler import StandardScaler
from preprocessing.encoder import CategoricalEncoder

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseNNPipline(ABC):
    def __init__(
            self,
            numeric_features: List[str],
            categorical_features: List[str],
            target_column: List[str],

            # Training
            batch_size: int = 32,
            epochs: int = 100,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            early_stopping_patience: int = 10,

            # Preprocessing
            scale_numeric: bool = True,
            encode_categorical: bool = True,

            # Device
            device: Optional[str] = None,

            # Random seed
            random_state: int = 42

    ):
        # Feature columns
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column

        # Training hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        # Preprocessing flags
        self.scale_numeric = scale_numeric
        self.encode_categorical = encode_categorical

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        self._set_seed()

        # Internal state 
        self.model: Optional[nn.Model] = None 
        self.scaler: Optional[StandardScaler] = None 
        self.encoder: Optional[CategoricalEncoder] = None 
        self.is_fitted = False 
        self.history: Dict[str, List[float]] = {}
        self.best_epoch = 0
        self.best_state: Optional[OrderedDict[str, torch.Tensor]] = None

    def _set_seed(self):
        """set random seeds for reproducibility"""
        torch.cuda.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

    @abstractmethod
    def _build_model(self, input_dim: int, output_dim: int):
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
    def _get_metrics(self) -> Dict[str, callable]:
        """Return a dictionary of metric funcitons (name -> callable)."""
        pass

    def _prepare_features(
            self, df: pd.DataFrame, fit: bool = False
    ) -> Tuple[torch.Tensor, List[str]]:
        """Preprocess numeric and categorical features
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        fit : bool 
            Whether to fit the scaler/encoder (True) or use already fitted ones (False).

        Returns
        -------
        torch.Tensor
            Preprocessed tensor of shape (n_samples, n_features).
        Listp[str]
            Names of the features after preprocessing.
        """
        numeric_data = df[self.numeric_features].values if self.numeric_features else np.empty((len(df), 0))
        categorical_data = df[self.categorical_data].values if self.categorical_features else np.empty((len(df), 0))

        # Scale numeric features
        if self.scale_numeric and numeric_data.size > 0:
            if fit:
                self.scaler = StandardScaler()
                numeric_scaled = self.scaler.fit_transform(numeric_data)
            else:
                if self.scaler is None:
                    raise RuntimeError("Scaler not fitted. Call fit first.")
        else:
            numeric_scaled = numeric_data

        # Encode categorical features
        if self.encode_categorical and categorical_data.size > 0:
            if fit:
                self.encoder = CategoricalEncoder()
                categorical_scaled = self.encoder.fit_transform(categorical_data)
            else:
                if self.encoder is None:
                    raise RuntimeError("Encoder not fitted. Call fit first.")
                categorical_encoded = self.encoder.transform(categorical_data)
        else:
            categorical_encoded = categorical_data

        # Combine features
        features = np.hstack([numeric_scaled, categorical_encoded]) if numeric_scaled.size > 0 or categorical_encoded.size > 0 else np.empty((len(df), 0))
        features_names = (
            [f"num_{f}" for f in self.numeric_features] + 
            [f"cat_{f}" for f in self.categorical_features for _ in range(self.encoder.n_values_per_feature if self.encoder else 1)]
        )
        return torch.FloatTensor(features).to(self.device), features_names


    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Ectract target column and convert to tensor."""
        target = df[self.target_column].values
        # Subclasses may override this to reshape or encode target differently
        return torch.FloatTensor(target).to(self.device)

    def _prepare_data(self, data, fit=False):
        """Preprocess and convert to tensors features and target"""
        X, feature_names = self.prepare_features(data, fit=True)
        y = self.prepare_target(data)
        return X, y, feature_names
    
    def _create_dataloader(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, shuffle: bool = False
    ) -> DataLoader:
        """Create a Pytorch DataLoader from tensors."""
        if y is not None:
            dataset = TensorDataset(X, y)
        else:
            dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None,
            verbose: int = 1,
    ) -> "BaseNNPipline":
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
        X_train, y_train, self.feature_names = self._prepare_data(df_train, fit=True)
        X_val, y_val, _ = self._prepare_data(df_train, fit=False) if df_val else None, None, None

        # Determine input/output dimensions
        input_dim = X_train.shape[1]
        output_dim = self.get_output_dim(y_train)
        logger.debug("Input dim: {input_dim}, ouput dim: {output_dim}")

        # Build model 
        self.model = self._build_model(input_dim, output_dim).to(self.device)
        loss_fn = self._get_loss_fn()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.early_stopping_patience // 2, verbose = verbose > 0
        )

        # Training loop
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False) if X_val is not None else None 

        self.history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training 
            self.model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.sero_grad()
                pred = self.model(batch_X)
                loss = loss_fn(pred, batch_y)
                loss.backward()
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

            # Early stopping 
            if val_loader is not None:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    self.best_epoch = epoch
                    # Save best model state
                    self.best_state = self.model.state_dict()
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

    def predict(self, df: pd.DateFrame) -> np.ndarray:
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
        X, _ = self._prepare_features(df, fit=False)
        loader = self._create_dataloader(X, shuffle=True)
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
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
        X, _ = self._prepare_features(df, fit=False)
        y = self._prepare_target(df).cpu().numpy()
        y_pred = self.predict(df)
        metric_funcs = self._get_metrics()

        if metrics:
            # Allow custom metric selection
            metric_funcs = {m : metric_funcs[m] for m in metrics if m in metric_funcs}

        return self.compute_metrics(y, y_pred, metric_funcs)

    def hyperparameter_tuning(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_feature_importance(self):
        pass

    def get_model(self):
        """Return the enderlying PyTorch model."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        return self.model

    def get_preprocessor(self):
        """Return the fitted preprocessing objects."""
        return {"scaller": self.scaler, "encoder": self.encoder}

    def _get_output_dim(self):
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

    def get_params(self):
        """Get pipline parameters."""
        exclude = {"model", "scaler", "encoder", "feature_names", "is_fitted", "history", "best_state"}
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k not in exclude}

    def __repr__(self):
        return f"{self.__class__.__name__}(target_column={self.target_column})"
