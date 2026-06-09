"""Tests for utils/hardware.py — detect_hardware, get_optimal_backend."""

import pytest
from dantabnn.utils.hardware import detect_hardware, get_optimal_backend


class TestDetectHardware:
    def test_returns_dict_with_expected_keys(self):
        hw = detect_hardware()
        expected_keys = {
            "device_type", "device", "cuda_available", "cuda_device_count",
            "recommended_batch_size", "mixed_precision", "cudf_available",
            "num_workers",
        }
        assert set(hw.keys()) == expected_keys

    def test_device_type_is_str(self):
        hw = detect_hardware()
        assert hw["device_type"] in ("cuda", "cpu")

    def test_cuda_available_is_bool(self):
        hw = detect_hardware()
        assert isinstance(hw["cuda_available"], bool)

    def test_cuda_device_count_is_int(self):
        hw = detect_hardware()
        assert isinstance(hw["cuda_device_count"], int)

    def test_recommended_batch_size_positive(self):
        hw = detect_hardware()
        assert hw["recommended_batch_size"] > 0

    def test_mixed_precision_is_bool(self):
        hw = detect_hardware()
        assert isinstance(hw["mixed_precision"], bool)

    def test_cudf_available_is_bool(self):
        hw = detect_hardware()
        assert isinstance(hw["cudf_available"], bool)

    def test_num_workers_positive(self):
        hw = detect_hardware()
        assert hw["num_workers"] > 0


class TestGetOptimalBackend:
    def test_returns_string(self):
        backend = get_optimal_backend()
        assert isinstance(backend, str)

    def test_returns_known_backend(self):
        backend = get_optimal_backend()
        assert backend in ("cudf", "conv", "numba", "pandas")