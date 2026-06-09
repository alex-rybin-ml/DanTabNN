"""Tests for tuning/tune_utils.py."""

import optuna
from dantabnn.tuning.tune_utils import get_danet_param_grid


class TestGetDanetParamGrid:
    def test_small_search_returns_compact_grid(self):
        grid = get_danet_param_grid(input_dim=16, small_search=True)
        assert "hidden_dims" in grid
        assert "dropout" in grid
        assert "attention_heads" in grid
        assert "use_sample_attention" in grid
        # Small search has discrete dropout values
        assert isinstance(grid["dropout"], list)

    def test_full_search_returns_optuna_distributions(self):
        grid = get_danet_param_grid(input_dim=32, small_search=False)
        assert "hidden_dims" in grid
        assert "attention_heads" in grid
        assert "use_sample_attention" in grid
        # Full search uses Optuna FloatDistribution for dropout
        assert isinstance(grid["dropout"], optuna.distributions.FloatDistribution)

    def test_low_input_dim_limits_heads(self):
        grid = get_danet_param_grid(input_dim=8, small_search=False)
        # input_dim=8 < 16, so safe_heads = [2, 4] only
        assert 8 not in grid["attention_heads"]

    def test_high_input_dim_includes_more_heads(self):
        grid = get_danet_param_grid(input_dim=32, small_search=False)
        assert 8 in grid["attention_heads"]

    def test_hidden_dims_in_full_grid(self):
        grid = get_danet_param_grid(input_dim=64, small_search=False)
        assert len(grid["hidden_dims"]) == 6  # 6 architecture variants

    def test_hidden_dims_in_small_grid(self):
        grid = get_danet_param_grid(input_dim=16, small_search=True)
        assert len(grid["hidden_dims"]) == 2