"""Utility functions for computing evaluation metrics."""

import numpy as np 
from typing import Dict, Callable


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> Dict[str, float]:
    """Compute multiple metrics given true labels and predictions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    metrics : Dict[str, Callable]
        Dictionary mapping metric names to callable functions
        that accept (y_true, y_pred) and return a float.

    Returns 
    -------
    Dict[str, float]
        Dictionary of metrics scores.
    """
    results = {}
    for name, func in metrics.items():
        try:
            results[name] = float(func(y_true, y_pred))
        except Exception as e:
            results[name] = np.nan
            import warnings
            warnings.warn(f"Metric '{name}' failed: {e}")
    return results
