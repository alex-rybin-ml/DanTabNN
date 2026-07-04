"""Tests for OutlierClipper."""
import numpy as np
import pytest
from dantabnn.preprocessing.outlier import OutlierClipper


class TestOutlierClipper:
    def test_iqr_clipping(self):
        X = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [100.0, -50.0],  # outliers
        ])
        clipper = OutlierClipper(iqr_multiplier=1.5)
        result = clipper.fit_transform(X)
        # Column 0: Q1=2, Q3=4, IQR=2 → lower=-1, upper=7 → 100 clipped to 7
        # Column 1: Q1=20, Q3=40, IQR=20 → lower=-10, upper=70 → -50 clipped to -10
        assert result[4, 0] < 100
        assert result[4, 1] > -50
        assert not np.isnan(result).any()

    def test_insufficient_data_skips(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        clipper = OutlierClipper()
        # 3 samples → < 4 → skips IQR computation → no clipping
        clipper.fit(X)
        assert clipper._bounds == {}

    def test_empty_array(self):
        clipper = OutlierClipper()
        clipper.fit(np.array([[]]))
        result = clipper.transform(np.array([[]]))
        assert result.shape == (1, 0)

    def test_transform_before_fit_raises(self):
        clipper = OutlierClipper()
        with pytest.raises(RuntimeError, match="fitted"):
            clipper.transform(np.array([[1.0]]))

    def test_fit_returns_self(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        clipper = OutlierClipper()
        assert clipper.fit(X) is clipper

    def test_zero_iqr_skips(self):
        """Column with zero IQR should be skipped."""
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]])
        clipper = OutlierClipper()
        clipper.fit(X)
        # First column has IQR=0 → skipped; second has IQR>0 → bounds created
        assert 0 not in clipper._bounds  # col 0 skipped (IQR=0)
        assert 1 in clipper._bounds       # col 1 has bounds

    def test_with_nan_values(self):
        X = np.array([
            [1.0, 10.0],
            [np.nan, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])
        clipper = OutlierClipper()
        result = clipper.fit_transform(X)
        assert result.shape == X.shape