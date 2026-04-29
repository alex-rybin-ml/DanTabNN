"""DANet Pipeline — Dual-Attention Neural Networks for tabular data."""

__version__ = "0.1.0"

from .base import BaseNNPipeline
from .binary import BinaryClassificationPipeline
from .regression import RegressionPipeline
from .multiclass import MulticlassClassificationPipeline


__all__ = [
    "BaseNNPipeline",
    "BinaryClassificationPipeline",
    "RegressionPipeline",
    "MulticlassClassificationPipeline"
]
