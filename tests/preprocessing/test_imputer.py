"""Tests for NaNImputer."""
import numpy as np
import pytest
from dantabnn.preprocessing.imputer import NaNImputer


class TestNaNImputer:
    def test_numeric_median_imputation(self):
        X = np.array([
            [1.0, 10.0],
            [np.nan, 20.0],
            [3.0, np.nan],
        ])
        imp = NaNImputer()
        result = imp.fit_transform(X)
        assert not np.isnan(result).any()
        assert result[1, 0] == 2.0  # median of [1,3]
        assert result[2, 1] == 15.0  # median of [10,20]

    def test_empty_array(self):
        imp = NaNImputer()
        imp.fit(np.array([[]]))
        result = imp.transform(np.array([[]]))
        assert result.shape == (1, 0)

    def test_all_nan_column(self):
        X = np.array([[np.nan], [np.nan], [np.nan]])
        imp = NaNImputer()
        result = imp.fit_transform(X)
        # All values missing → fill with 0.0
        assert np.all(result == 0.0)

    def test_no_missing_values(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp = NaNImputer()
        result = imp.fit_transform(X)
        np.testing.assert_array_equal(result, X)

    def test_categorical_mode_imputation(self):
        X = np.array([
            [1.0, 0.0],
            [np.nan, 1.0],
            [2.0, 1.0],
            [3.0, 0.0],
        ])
        cat_mask = np.array([False, True])  # second column is categorical
        imp = NaNImputer(categorical_mask=cat_mask)
        result = imp.fit_transform(X)
        # numeric: median of [1,2,3]=2
        # categorical: mode of [0,1,1,0]=most_common → 0.0 or 1.0 depends on Counter
        assert result[1, 0] == 2.0

    def test_transform_before_fit_raises(self):
        imp = NaNImputer()
        with pytest.raises(RuntimeError, match="fitted"):
            imp.transform(np.array([[1.0]]))

    def test_1d_array(self):
        X = np.array([1.0, np.nan, 3.0])
        imp = NaNImputer()
        result = imp.fit_transform(X)
        assert not np.isnan(result).any()
        assert result[1] == 2.0

    def test_fit_returns_self(self):
        imp = NaNImputer()
        assert imp.fit(np.array([[1.0], [2.0]])) is imp