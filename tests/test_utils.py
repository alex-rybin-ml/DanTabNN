"""Tests for utils/metrics.py and utils/logger.py."""

import logging
import warnings
import numpy as np
import pytest
from dantabnn.utils.metrics import compute_metrics
from dantabnn.utils.logger import setup_logger


class TestComputeMetrics:
    def test_computes_multiple_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = compute_metrics(
            y_true,
            y_pred,
            {
                "mse": lambda t, p: np.mean((t - p) ** 2),
                "mae": lambda t, p: np.mean(np.abs(t - p)),
            },
        )
        assert result["mse"] == pytest.approx(0.0)
        assert result["mae"] == pytest.approx(0.0)

    def test_metric_failure_returns_nan(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([1.0, 2.0])

        def failing_metric(t, p):
            raise ValueError("metric error")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute_metrics(y_true, y_pred, {"bad": failing_metric})
        assert np.isnan(result["bad"])
        assert len(w) == 1

    def test_empty_metrics_returns_empty(self):
        result = compute_metrics(np.array([1.0]), np.array([1.0]), {})
        assert result == {}

    def test_single_metric(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_metrics(
            y_true, y_pred, {"acc": lambda t, p: np.mean(t == p)}
        )
        assert result["acc"] == 1.0

    def test_different_shapes_handling(self):
        # metrics should handle shape mismatch through their own validation
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.9, 0.1, 0.8])
        result = compute_metrics(
            y_true,
            y_pred,
            {"mean_pred": lambda t, p: float(np.mean(p))},
        )
        assert result["mean_pred"] == pytest.approx(0.6)


class TestSetupLogger:
    def test_creates_logger_with_given_name(self):
        logger = setup_logger("test_module")
        assert logger.name == "test_module"
        assert logger.level == logging.INFO

    def test_returns_existing_logger_without_duplicating_handlers(self):
        logger1 = setup_logger("test_dup")
        initial_handler_count = len(logger1.handlers)
        logger2 = setup_logger("test_dup")
        assert logger2 is logger1
        assert len(logger2.handlers) == initial_handler_count

    def test_custom_log_level(self):
        logger = setup_logger("test_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_propagate_is_false(self):
        logger = setup_logger("test_no_propagate")
        assert logger.propagate is False