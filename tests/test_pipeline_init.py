"""Tests for pipeline auto-computation of hidden_dims and gating_k."""
import pytest
from dantabnn.binary import BinaryClassificationPipeline
from dantabnn.regression import RegressionPipeline
from dantabnn.multiclass import MulticlassClassificationPipeline


# ---------------------------------------------------------------------------
# BaseNNPipeline._default_hidden_dims (via any subclass)
# ---------------------------------------------------------------------------

class TestDefaultHiddenDims:
    def test_typical_feature_count(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(10)],
            categorical_features=[], target_column="y",
        )
        # 10 features -> h0=max(32,min(20,128))=32, h1=max(16,min(10,64))=16, h2=max(8,min(5,32))=8
        assert pipe.hidden_dims == [32, 16, 8]

    def test_large_feature_count(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(100)],
            categorical_features=[], target_column="y",
        )
        # 100 features -> h0=max(32,min(200,128))=128, h1=max(16,min(100,64))=64, h2=max(8,min(50,32))=32
        assert pipe.hidden_dims == [128, 64, 32]

    def test_single_feature(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0"],
            categorical_features=[], target_column="y",
        )
        # 1 feature -> h0=max(32,min(2,128))=32, h1=max(16,min(1,64))=16, h2=max(8,min(0,32))=8
        assert pipe.hidden_dims == [32, 16, 8]

    def test_alignment_to_attention_heads(self):
        """All dimensions must be divisible by 4 (aligned to attention_heads=4)."""
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(7)],
            categorical_features=[], target_column="y",
        )
        for dim in pipe.hidden_dims:
            assert dim % 4 == 0

    def test_regression_pipeline(self):
        pipe = RegressionPipeline(
            numeric_features=["f" + str(i) for i in range(15)],
            categorical_features=[], target_column="y",
        )
        # 15 features -> h0=max(32,min(30,128))=32, h1=max(16,min(15,64))=16, h2=max(8,min(7,32))=8
        assert pipe.hidden_dims == [32, 16, 8]

    def test_multiclass_pipeline(self):
        pipe = MulticlassClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(20)],
            categorical_features=[], target_column="y", n_classes=5,
        )
        # 20 features -> h0=max(32,min(40,128))=40->40/4*4=40, h1=max(16,min(20,64))=20->20, h2=max(8,min(10,32))=12->12
        assert pipe.hidden_dims == [40, 20, 12]


# ---------------------------------------------------------------------------
# BaseNNPipeline._default_gating_k
# ---------------------------------------------------------------------------

class TestDefaultGatingK:
    def test_typical(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(15)],
            categorical_features=[], target_column="y",
        )
        assert pipe.gating_k == max(1, 15 // 3)  # 5

    def test_small_feature_count(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1"],
            categorical_features=[], target_column="y",
        )
        assert pipe.gating_k == max(1, 2 // 3)  # 1

    def test_regression_pipeline(self):
        pipe = RegressionPipeline(
            numeric_features=["f" + str(i) for i in range(30)],
            categorical_features=[], target_column="y",
        )
        assert pipe.gating_k == 10

    def test_multiclass_pipeline(self):
        pipe = MulticlassClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(9)],
            categorical_features=[], target_column="y", n_classes=3,
        )
        assert pipe.gating_k == 3


# ---------------------------------------------------------------------------
# Auto-computation: None defaults trigger auto-compute
# ---------------------------------------------------------------------------

class TestAutoComputeTriggered:
    def test_hidden_dims_none_triggers_auto_compute(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(10)],
            categorical_features=[], target_column="y",
        )
        assert pipe.hidden_dims is not None
        assert len(pipe.hidden_dims) == 3

    def test_gating_k_none_triggers_auto_compute(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f" + str(i) for i in range(10)],
            categorical_features=[], target_column="y",
        )
        assert pipe.gating_k == 3  # max(1, 10//3)


# ---------------------------------------------------------------------------
# Explicit values override auto-compute
# ---------------------------------------------------------------------------

class TestExplicitOverride:
    def test_explicit_hidden_dims_overrides(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1", "f2"],
            categorical_features=[], target_column="y",
            hidden_dims=[64, 32, 16],
        )
        assert pipe.hidden_dims == [64, 32, 16]

    def test_explicit_gating_k_overrides(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1", "f2"],
            categorical_features=[], target_column="y",
            gating_k=7,
        )
        assert pipe.gating_k == 7

    def test_partial_override(self):
        pipe = RegressionPipeline(
            numeric_features=["f" + str(i) for i in range(8)],
            categorical_features=[], target_column="y",
            hidden_dims=[96, 48, 24],  # explicit
            # gating_k left as None -> auto-compute
        )
        assert pipe.hidden_dims == [96, 48, 24]
        assert pipe.gating_k == max(1, 8 // 3)


# ---------------------------------------------------------------------------
# Subclass-specific init still works
# ---------------------------------------------------------------------------

class TestSubclassSpecificParams:
    def test_binary_pipeline_params(self):
        pipe = BinaryClassificationPipeline(
            numeric_features=["f0", "f1"],
            categorical_features=[], target_column="y",
            pos_weight=3.0, threshold_tuning=False,
        )
        assert pipe.pos_weight == 3.0
        assert pipe.threshold_tuning is False
        assert pipe.hidden_dims is not None  # auto-computed
        assert pipe.gating_k is not None

    def test_regression_pipeline_params(self):
        pipe = RegressionPipeline(
            numeric_features=["f0", "f1"],
            categorical_features=[], target_column="y",
            scale_target=False,
        )
        assert pipe.scale_target is False
        assert pipe.hidden_dims is not None
        assert pipe.gating_k is not None

    def test_multiclass_pipeline_params(self):
        pipe = MulticlassClassificationPipeline(
            numeric_features=["f0", "f1", "f2"],
            categorical_features=[], target_column="y",
            n_classes=7, class_weights=[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        assert pipe.n_classes == 7
        assert pipe.class_weights is not None
        assert pipe.hidden_dims is not None
        assert pipe.gating_k is not None


# ---------------------------------------------------------------------------
# Zero numeric_features edge case
# ---------------------------------------------------------------------------

class TestZeroNumericFeatures:
    def test_no_numeric_features(self):
        """Pipeline with only categorical features still auto-computes."""
        pipe = BinaryClassificationPipeline(
            numeric_features=[],
            categorical_features=["cat1", "cat2"],
            target_column="y",
        )
        assert pipe.hidden_dims == [32, 16, 8]  # h0=max(32,min(0,128))=32, etc.
        assert pipe.gating_k == 1  # max(1, 0//3)