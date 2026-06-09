"""Tests for preprocessing/scaler.py and preprocessing/encoder.py."""

import numpy as np
import pytest
from dantabnn.preprocessing.scaler import StandardScaler
from dantabnn.preprocessing.encoder import CategoricalEncoder


class TestStandardScaler:
    def test_fit_sets_mean_and_scale(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        assert scaler.mean_ is not None
        assert scaler.scale_ is not None
        assert scaler.mean_.shape == (2,)
        assert scaler.scale_.shape == (2,)

    def test_transform_centers_and_scales(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        scaler.fit(X)
        transformed = scaler.transform(X)
        assert transformed.shape == X.shape
        # After standard scaling, each column should have mean ~0 and std ~1
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-7)
        assert np.allclose(transformed.std(axis=0, ddof=0), 1.0, atol=1e-7)

    def test_fit_transform_combines_both(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == X.shape
        assert scaler.mean_ is not None
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-7)

    def test_transform_before_fit_raises(self):
        scaler = StandardScaler()
        with pytest.raises(Exception):
            scaler.transform(np.array([[1.0]]))

    def test_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == (3, 1)
        assert np.allclose(transformed.mean(), 0.0, atol=1e-7)

    def test_constant_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)
        assert transformed.shape == (3, 2)
        # Constant column becomes all zeros
        assert np.allclose(transformed[:, 0], 0.0, atol=1e-7)

    def test_transform_new_data(self):
        X_train = np.array([[1.0], [2.0], [3.0]])
        X_test = np.array([[4.0], [5.0]])
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        transformed = scaler.transform(X_test)
        assert transformed.shape == (2, 1)


class TestCategoricalEncoder:
    def test_fit_sets_categories(self):
        X = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
        encoder = CategoricalEncoder()
        encoder.fit(X)
        assert encoder.categories_ is not None
        assert len(encoder.categories_) == 2
        assert encoder.n_values_per_feature is not None
        assert len(encoder.n_values_per_feature) == 2

    def test_fit_transform_one_hot_encodes(self):
        X = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
        encoder = CategoricalEncoder()
        encoded = encoder.fit_transform(X)
        assert encoded.shape[0] == 3
        # First col: 2 categories (a,b), second: 3 categories (x,y,z) → 5 features
        assert encoded.shape[1] == 5
        # Row 0: a=1, x=1
        assert encoded[0, 0] == 1.0  # category "a"
        assert encoded[0, 1] == 0.0  # category "b"

    def test_transform_uses_fitted_categories(self):
        X_train = np.array([["a"], ["b"], ["c"]])
        X_test = np.array([["a"], ["c"]])
        encoder = CategoricalEncoder()
        encoder.fit(X_train)
        encoded = encoder.transform(X_test)
        assert encoded.shape == (2, 3)

    def test_transform_before_fit_raises(self):
        encoder = CategoricalEncoder()
        with pytest.raises(Exception):
            encoder.transform(np.array([["a"]]))

    def test_handle_unknown_ignore(self):
        X_train = np.array([["a"], ["b"]])
        X_test = np.array([["a"], ["z"]])  # "z" unknown
        encoder = CategoricalEncoder(handle_unknown="ignore")
        encoder.fit(X_train)
        encoded = encoder.transform(X_test)
        assert encoded.shape == (2, 2)
        # "a" should be (1, 0), "z" should be (0, 0)
        assert encoded[1, 0] == 0.0
        assert encoded[1, 1] == 0.0

    def test_single_feature(self):
        X = np.array([["red"], ["blue"], ["red"]])
        encoder = CategoricalEncoder()
        encoded = encoder.fit_transform(X)
        assert encoded.shape == (3, 2)

    def test_numeric_as_categorical(self):
        # encoder should work with numeric strings or ints
        X = np.array([[1, 100], [2, 200], [1, 300]])
        encoder = CategoricalEncoder()
        encoded = encoder.fit_transform(X)
        assert encoded.shape[0] == 3

    def test_fit_transform_preserves_state(self):
        X = np.array([["a", "x"], ["b", "y"]])
        encoder = CategoricalEncoder()
        encoder.fit_transform(X)
        assert encoder.categories_ is not None
        assert encoder.n_values_per_feature is not None