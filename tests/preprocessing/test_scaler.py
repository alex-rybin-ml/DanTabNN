"""Tests for StandardScaler wrapper."""
import numpy as np
import pytest
from dantabnn.preprocessing.scaler import StandardScaler


class TestStandardScaler:
    def test_fit(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        s = StandardScaler()
        s.fit(X)
        assert s.mean_ is not None
        assert s.scale_ is not None
        np.testing.assert_array_almost_equal(s.mean_, [2.0, 20.0])
        np.testing.assert_array_almost_equal(s.scale_, [0.81649658, 8.16496581], decimal=4)

    def test_transform(self):
        X_train = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        X_test = np.array([[2.0, 15.0]])
        s = StandardScaler()
        s.fit(X_train)
        transformed = s.transform(X_test)
        assert transformed.shape == (1, 2)
        # Should be centered: (2-2)/0.816 ~ 0, (15-20)/8.165 ~ -0.612
        assert abs(transformed[0, 0]) < 0.01
        assert transformed[0, 1] < 0

    def test_fit_transform(self):
        X = np.array([[1.0], [2.0], [3.0]])
        s = StandardScaler()
        result = s.fit_transform(X)
        assert result.shape == (3, 1)
        np.testing.assert_array_almost_equal(result.mean(), 0.0, decimal=6)
        np.testing.assert_array_almost_equal(result.std(ddof=0), 1.0, decimal=6)

    def test_zero_variance(self):
        X = np.array([[5.0], [5.0], [5.0]])
        s = StandardScaler()
        _ = s.fit_transform(X)  # should not raise
        assert s.scale_ is not None