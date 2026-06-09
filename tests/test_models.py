"""Tests for models/gating.py and models/danet.py."""

import numpy as np
import pytest
import torch
from dantabnn.models.gating import (
    FeatureGating,
    TopKFeatureGating,
    create_feature_gating,
)
from dantabnn.models.danet import FeatureAttention, SampleAttention, DANetModule


# ---------------------------------------------------------------------------
# FeatureGating
# ---------------------------------------------------------------------------

class TestFeatureGating:
    def test_soft_gating_output_shape(self):
        gating = FeatureGating(input_dim=8)
        x = torch.randn(4, 8)
        out, gate = gating(x)
        assert out.shape == (4, 8)
        assert gate.shape == (4, 8)
        # Gate values should be in [0, 1]
        assert (gate >= 0).all() and (gate <= 1).all()

    def test_hard_gating_binary_mask(self):
        gating = FeatureGating(input_dim=8, hard=True)
        x = torch.randn(4, 8)
        out, gate = gating(x)
        # With hard=True, gate should be exactly 0 or 1
        unique_vals = gate.unique().tolist()
        for v in unique_vals:
            assert v == pytest.approx(0.0) or v == pytest.approx(1.0)

    def test_soft_gating_probabilistic(self):
        gating = FeatureGating(input_dim=8, hard=False)
        x = torch.randn(4, 8)
        out, gate = gating(x)
        # With hard=False, gate values should be continuous in (0, 1)
        assert ((gate > 0) & (gate < 1)).any()

    def test_mask_zeros_missing_features(self):
        gating = FeatureGating(input_dim=4)
        x = torch.randn(2, 4)
        mask = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float)
        out, gate = gating(x, mask=mask)
        # Missing features (mask=1) should get gate=0
        assert gate[0, 1].item() == pytest.approx(0.0)
        assert gate[1, 2].item() == pytest.approx(0.0)

    def test_get_selection_probabilities(self):
        gating = FeatureGating(input_dim=4)
        probs = gating.get_selection_probabilities()
        assert probs.shape == (4,)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_get_selection_probabilities_with_mask(self):
        gating = FeatureGating(input_dim=4)
        # Feature 1 is missing in ALL batch samples → prob must be 0
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
        probs = gating.get_selection_probabilities(mask=mask)
        assert probs.shape == (4,)
        # Feature 0 and 1 are always missing → prob should be 0
        assert probs[0].item() == pytest.approx(0.0)
        assert probs[1].item() == pytest.approx(0.0)

    def test_eval_mode_no_gumbel_noise(self):
        gating = FeatureGating(input_dim=8)
        gating.eval()
        x = torch.randn(4, 8)
        out1, _ = gating(x)
        out2, _ = gating(x)
        # In eval mode, output should be deterministic
        assert torch.allclose(out1, out2)

    def test_temperature_affects_sharpness(self):
        # High temperature → softer decisions
        gating_hot = FeatureGating(input_dim=8, temperature=10.0, hard=False)
        gating_cold = FeatureGating(input_dim=8, temperature=0.1, hard=False)
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        # Set identical logits
        gating_hot.gate_logits.data = torch.zeros(8)
        gating_cold.gate_logits.data = torch.zeros(8)
        _, gate_hot = gating_hot(x)
        _, gate_cold = gating_cold(x)
        # Hot should have more variance (softer sigmoid)
        assert gate_hot.std() < gate_cold.std()

    def test_dropout_applied_during_training(self):
        gating = FeatureGating(input_dim=8, dropout=0.9)
        gating.train()
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        out1, _ = gating(x)
        out2, _ = gating(x)
        # With high dropout, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_init_bias_positive(self):
        gating = FeatureGating(input_dim=4, init_bias=2.0)
        x = torch.randn(2, 4)
        gating.eval()
        _, gate = gating(x)
        # With positive init bias, most features should have prob > 0.5
        assert (gate > 0.5).float().mean() > 0.5


# ---------------------------------------------------------------------------
# TopKFeatureGating
# ---------------------------------------------------------------------------

class TestTopKFeatureGating:
    def test_selects_exactly_k_features(self):
        gating = TopKFeatureGating(input_dim=10, k=3, hard=True)
        x = torch.randn(4, 10)
        out, gate = gating(x)
        assert out.shape == (4, 10)
        # Each row should have exactly k ones
        assert gate.sum(dim=1).tolist() == [3.0, 3.0, 3.0, 3.0]

    def test_hard_gating_binary(self):
        gating = TopKFeatureGating(input_dim=8, k=2, hard=True)
        x = torch.randn(4, 8)
        _, gate = gating(x)
        unique_vals = gate.unique().tolist()
        for v in unique_vals:
            assert v == pytest.approx(0.0) or v == pytest.approx(1.0)

    def test_raises_on_invalid_k(self):
        with pytest.raises(AssertionError):
            TopKFeatureGating(input_dim=5, k=0)
        with pytest.raises(AssertionError):
            TopKFeatureGating(input_dim=5, k=6)

    def test_mask_prevents_selection_of_missing(self):
        gating = TopKFeatureGating(input_dim=4, k=2, hard=True)
        x = torch.randn(1, 4)
        # Force feature 0 and 1 to be "missing"
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        _, gate = gating(x, mask=mask)
        # Features 0,1 must not be selected
        assert gate[0, 0].item() == 0.0
        assert gate[0, 1].item() == 0.0


# ---------------------------------------------------------------------------
# create_feature_gating
# ---------------------------------------------------------------------------

class TestCreateFeatureGating:
    def test_returns_none_for_none_type(self):
        gating = create_feature_gating(8, gating_type="none")
        assert gating is None
        gating = create_feature_gating(8, gating_type="")
        assert gating is None

    def test_returns_soft_gating(self):
        gating = create_feature_gating(8, gating_type="soft")
        assert isinstance(gating, FeatureGating)

    def test_returns_topk_gating(self):
        gating = create_feature_gating(8, gating_type="topk", k=4)
        assert isinstance(gating, TopKFeatureGating)

    def test_raises_for_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown gating_type"):
            create_feature_gating(8, gating_type="invalid")

    def test_passes_kwargs_to_constructor(self):
        gating = create_feature_gating(
            8, gating_type="soft", temperature=2.0, hard=False
        )
        assert gating.temperature == 2.0
        assert gating.hard is False

    def test_filters_invalid_kwargs(self):
        # "unknown_param" should be silently ignored
        gating = create_feature_gating(8, gating_type="soft", unknown_param=999)
        assert isinstance(gating, FeatureGating)


# ---------------------------------------------------------------------------
# FeatureAttention
# ---------------------------------------------------------------------------

class TestFeatureAttention:
    def test_forward_output_shape(self):
        attn = FeatureAttention(input_dim=16, num_heads=4)
        x = torch.randn(2, 1, 16)  # (B, L=1, D)
        out = attn(x)
        assert out.shape == (2, 1, 16)

    def test_asserts_divisible_dim(self):
        with pytest.raises(AssertionError, match="input_dim must be divisible"):
            FeatureAttention(input_dim=10, num_heads=3)

    def test_forward_with_mask(self):
        attn = FeatureAttention(input_dim=16, num_heads=4, missing_bias=True)
        x = torch.randn(2, 1, 16)
        mask = torch.zeros(2, 16)  # no missing
        out = attn(x, mask=mask)
        assert out.shape == (2, 1, 16)

    def test_forward_without_mask_bias(self):
        attn = FeatureAttention(input_dim=16, num_heads=4, missing_bias=False)
        x = torch.randn(2, 1, 16)
        out = attn(x)
        assert out.shape == (2, 1, 16)


# ---------------------------------------------------------------------------
# SampleAttention
# ---------------------------------------------------------------------------

class TestSampleAttention:
    def test_forward_output_shape(self):
        attn = SampleAttention(input_dim=16, num_heads=4)
        x = torch.randn(4, 2, 16)  # (B=4, L=2, D=16)
        out = attn(x)
        assert out.shape == (4, 2, 16)

    def test_asserts_divisible_dim(self):
        with pytest.raises(AssertionError, match="input_dim must be divisible"):
            SampleAttention(input_dim=10, num_heads=3)


# ---------------------------------------------------------------------------
# DANetModule
# ---------------------------------------------------------------------------

class TestDANetModule:
    def test_build_minimal_model(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8])
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)

    def test_build_with_sample_attention(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8], use_sample_attention=True)
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)

    def test_build_no_hidden_dims(self):
        model = DANetModule(input_dim=8, hidden_dims=[])
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)

    def test_build_with_gating(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8], gating_type="soft")
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)

    def test_build_with_topk_gating(self):
        model = DANetModule(
            input_dim=8, hidden_dims=[16, 8], gating_type="topk", gating_k=4
        )
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 1)

    def test_forward_with_gating_mask(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8], gating_type="soft")
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        mask = torch.zeros(2, 8)
        out = model(x, mask=mask)
        assert out.shape == (2, 1)

    def test_forward_with_partial_gating_mask(self):
        """Mask smaller than input → should be padded."""
        model = DANetModule(input_dim=8, hidden_dims=[16, 8], gating_type="soft")
        model.set_output_layer(torch.nn.Linear(8, 1))
        x = torch.randn(2, 8)
        mask = torch.zeros(2, 4)  # only first 4 features
        out = model(x, mask=mask)
        assert out.shape == (2, 1)

    def test_set_output_layer(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8])
        assert isinstance(model.output_layer, torch.nn.Identity)
        model.set_output_layer(torch.nn.Linear(8, 3))
        assert isinstance(model.output_layer, torch.nn.Linear)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 3)

    def test_output_without_output_layer(self):
        model = DANetModule(input_dim=8, hidden_dims=[16, 8])
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 8)  # Identity passes through last FF dim