"""Tests for metrics utility."""
import numpy as np
import pytest
from dantabnn.utils.metrics import compute_metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


class TestComputeMetrics:
    def test_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])
        result = compute_metrics(y_true, y_pred, {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
        })
        assert "mse" in result
        assert "mae" in result
        assert result["mse"] > 0
        assert result["mae"] > 0

    def test_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = compute_metrics(y_true, y_pred, {
            "mse": mean_squared_error,
        })
        assert result["mse"] == 0.0

    def test_failing_metric_returns_nan(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])

        def raises_error(a, b):
            raise ValueError("broken")

        result = compute_metrics(y_true, y_pred, {"bad": raises_error})
        assert np.isnan(result["bad"])

    def test_empty_metrics(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])
        result = compute_metrics(y_true, y_pred, {})
        assert result == {}