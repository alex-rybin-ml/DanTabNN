from abc import ABC, abstractmethod

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseNNPipline(ABC):
    def __init__(self):
        self.target_column = None

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _get_loss_fn(self):
        pass

    @abstractmethod
    def _get_metrics(self):
        pass

    def _prepare_features(self):
        pass

    def _prepare_target(self):
        pass

    def _create_dataloader(self):
        pass

    def fit(self):
        pass

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
