"""Tests for models/cross.py — CrossNetwork, CrossLayer, FactorizedCrossLayer."""

import pytest
import torch
from dantabnn.models.cross import (
    CrossLayer,
    FactorizedCrossLayer,
    CrossNetwork,
    create_interaction_layer,
)


class TestCrossLayer:
    def test_forward_shape(self):
        layer = CrossLayer(input_dim=16)
        x0 = torch.randn(4, 16)
        x = torch.randn(4, 16)
        out = layer(x0, x)
        assert out.shape == (4, 16)

    def test_forward_not_identity(self):
        layer = CrossLayer(input_dim=16)
        x0 = torch.randn(4, 16)
        x = torch.randn(4, 16)
        out = layer(x0, x)
        assert not torch.allclose(out, x)  # cross should modify


class TestFactorizedCrossLayer:
    def test_forward_shape(self):
        layer = FactorizedCrossLayer(input_dim=16, rank=4)
        x0 = torch.randn(4, 16)
        x = torch.randn(4, 16)
        out = layer(x0, x)
        assert out.shape == (4, 16)

    def test_forward_with_minimal_rank(self):
        layer = FactorizedCrossLayer(input_dim=8, rank=1)
        x0 = torch.randn(4, 8)
        x = torch.randn(4, 8)
        out = layer(x0, x)
        assert out.shape == (4, 8)


class TestCrossNetwork:
    def test_forward_shape_full_rank(self):
        net = CrossNetwork(input_dim=16, num_layers=3, low_rank=False)
        x = torch.randn(4, 16)
        out = net(x)
        assert out.shape == (4, 16)

    def test_forward_shape_low_rank(self):
        net = CrossNetwork(input_dim=16, num_layers=2, low_rank=True, rank_ratio=0.25)
        x = torch.randn(4, 16)
        out = net(x)
        assert out.shape == (4, 16)

    def test_single_layer(self):
        net = CrossNetwork(input_dim=8, num_layers=1)
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 8)

    def test_with_dropout(self):
        net = CrossNetwork(input_dim=8, num_layers=2, dropout=0.5)
        net.train()
        x = torch.randn(4, 8)
        out1 = net(x)
        out2 = net(x)
        # With dropout in train mode, outputs should differ
        assert not torch.allclose(out1, out2)

    def test_no_dropout_deterministic(self):
        net = CrossNetwork(input_dim=8, num_layers=2, dropout=0.0)
        net.eval()
        x = torch.randn(4, 8)
        out1 = net(x)
        out2 = net(x)
        assert torch.allclose(out1, out2)


class TestCreateInteractionLayer:
    def test_legacy_returns_none(self):
        layer = create_interaction_layer(16, interaction_type="legacy")
        assert layer is None

    def test_cross_returns_crossnetwork(self):
        layer = create_interaction_layer(16, interaction_type="cross", num_cross_layers=2)
        assert isinstance(layer, CrossNetwork)
        assert layer.num_layers == 2
        assert not layer.low_rank

    def test_factorized_returns_crossnetwork_low_rank(self):
        layer = create_interaction_layer(
            16, interaction_type="factorized", num_cross_layers=3, rank_ratio=0.5
        )
        assert isinstance(layer, CrossNetwork)
        assert layer.low_rank is True

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown interaction_type"):
            create_interaction_layer(16, interaction_type="invalid")