from abc import ABC, abstractmethod
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataSet

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseNNPipline(ABC):
    def __init__(
            self,
            numeric_features: List[str],
            target_column: List[str],

            # Training
            batch_size: int = 32,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            early_stopping_patience: int = 10,

            # Device
            device: Optional[str] = None,

            # Random seed
            random_state: int = 42

    ):
        # Feature columns
        self.numeric_features = numeric_features
        self.target_column = target_column

        # Training hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience

        # Device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.random_state = random_state
        self._set_seed()

        # Internal state 
        self.model = Optional[nn.Model]

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

    def _prepare_features(self):
        pass

    def _prepare_target(self, df: pd.DataFrame) -> torch.Tensor:
        """Ectract target column and convert to tensor."""
        target = df[self.target_column].values
        # Subclasses may override this to reshape or encode target differently
        return torch.FloatTensor(target).to(self.device)

    def _prepare_data(self, data, fit=False):
        X, feature_names = self.prepare_features(data, fit=True)
        y = self.prepare_target(data)
        return X, y, feature_names
    
    def _create_dataloader(
            self, X: torch.Tensor, y: Optional[torch.Tensor] = None, shuffle: bool = False
    ) -> DataLoader:
        """Create a Pytorch DataLoader from tensors."""
        if y is not None:
            dataset = TensorDataSet(X, y)
        else:
            dataset = TensorDataSet(X)
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
        X_train, y_train, self.feature_names = self.prepare_features(df_train, fit=True)
        X_val, y_val, _ = self.prepare_features(df_train, fit=False) if df_val else None, None, None

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

    def predict(self):
        pass

    def evaluate(self):
        pass

    def hyperparameter_tuning(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_feature_importance(self):
        pass

    def get_model(self):
        pass

    def get_preprocessor(self):
        pass

    def _get_output_dim(self):
        pass

    def set_params(self):
        pass

    def get_params(self):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(target_column={self.target_column})"
