"""Tests for tuning/tune_utils.py and tune_utils mapper."""
import optuna
from dantabnn.tuning.tune_utils import get_danet_param_grid, get_danet_param_mapper, _compute_hidden_dims
from dantabnn.binary import BinaryClassificationPipeline


class TestGetDanetParamGrid:
    def test_small_search_returns_compact_grid(self):
        grid = get_danet_param_grid(input_dim=16, small_search=True)
        assert "hidden_dims_choice" in grid
        assert len(grid["hidden_dims_choice"]) == 1  # ["adaptive"]
        assert "dropout" in grid
        assert isinstance(grid["dropout"], list)

    def test_full_search_returns_optuna_distributions(self):
        grid = get_danet_param_grid(input_dim=32, small_search=False)
        assert "hidden_dims_choice" in grid
        assert isinstance(grid["dropout"], optuna.distributions.FloatDistribution)

    def test_low_input_dim_limits_heads(self):
        grid = get_danet_param_grid(input_dim=8, small_search=False)
        # gating_type always has both options
        assert "soft" in grid["gating_type"]

    def test_high_input_dim_includes_more_heads(self):
        grid = get_danet_param_grid(input_dim=32, small_search=False)
        assert "hidden_dims_choice" in grid
        assert len(grid["hidden_dims_choice"]) == 3  # ["adaptive", "narrow", "wide"]

    def test_hidden_dims_in_full_grid(self):
        grid = get_danet_param_grid(input_dim=64, small_search=False)
        assert len(grid["hidden_dims_choice"]) == 3

    def test_hidden_dims_in_small_grid(self):
        grid = get_danet_param_grid(input_dim=16, small_search=True)
        assert len(grid["hidden_dims_choice"]) == 1


class TestParamMapper:
    def test_adaptive_choice(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"],
            categorical_features=[], target_column="y",
        )
        params = {"hidden_dims_choice": "adaptive", "dropout": 0.3, "gating_type": "soft"}
        result = get_danet_param_mapper(params, pipe)
        assert "hidden_dims" in result
        assert result["hidden_dims"] == _compute_hidden_dims(10)
        assert result["dropout"] == 0.3

    def test_narrow_choice(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0"], categorical_features=[], target_column="y",
        )
        params = {"hidden_dims_choice": "narrow"}
        result = get_danet_param_mapper(params, pipe)
        assert result["hidden_dims"] == [32, 16, 8]

    def test_wide_choice(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1", "f2", "f3", "f4"],
            categorical_features=[], target_column="y",
        )
        params = {"hidden_dims_choice": "wide"}
        result = get_danet_param_mapper(params, pipe)
        assert result["hidden_dims"] == [256, 128, 64]