"""Test registries — no duplicate keys, every entry imports correctly."""

from __future__ import annotations

import pytest


def test_model_registry_has_baselines():
    """MODEL_REGISTRY must contain both baseline models."""
    from cdlib.models.baselines import _import_all_baselines
    from cdlib.models._model_registry import MODEL_REGISTRY

    _import_all_baselines()

    assert "rgb_ssim" in MODEL_REGISTRY, "rgb_ssim not registered"
    assert "fc_siam_diff" in MODEL_REGISTRY, "fc_siam_diff not registered"


def test_model_registry_no_duplicates():
    """Re-importing baselines must not raise duplicate key errors."""
    from cdlib.models.baselines import _import_all_baselines
    from cdlib.models._model_registry import MODEL_REGISTRY

    _import_all_baselines()
    keys_before = MODEL_REGISTRY.keys()

    # Re-import should be idempotent
    _import_all_baselines()
    keys_after = MODEL_REGISTRY.keys()

    assert keys_before == keys_after


def test_loss_registry_exists():
    """LOSS_REGISTRY must be importable and be a Registry instance."""
    from cdlib.losses.registry import LOSS_REGISTRY
    from cdlib.utils.registry import Registry

    assert isinstance(LOSS_REGISTRY, Registry)


def test_dataset_registry_exists():
    """DATASET_REGISTRY must be importable and be a Registry instance."""
    from cdlib.data.registry import DATASET_REGISTRY
    from cdlib.utils.registry import Registry

    assert isinstance(DATASET_REGISTRY, Registry)


def test_encoder_registry_exists():
    """ENCODER_REGISTRY must be importable and be a Registry instance."""
    from cdlib.models.encoders.registry import ENCODER_REGISTRY
    from cdlib.utils.registry import Registry

    assert isinstance(ENCODER_REGISTRY, Registry)


def test_fusion_registry_exists():
    """FUSION_REGISTRY must be importable and be a Registry instance."""
    from cdlib.models.fusion.registry import FUSION_REGISTRY
    from cdlib.utils.registry import Registry

    assert isinstance(FUSION_REGISTRY, Registry)


def test_alignment_registry_exists():
    """ALIGNMENT_REGISTRY must be importable and be a Registry instance."""
    from cdlib.models.alignment.registry import ALIGNMENT_REGISTRY
    from cdlib.utils.registry import Registry

    assert isinstance(ALIGNMENT_REGISTRY, Registry)


def test_registry_build_unknown_key_raises():
    """Building with an unregistered key must raise KeyError."""
    from cdlib.utils.registry import Registry

    reg = Registry("TEST")
    with pytest.raises(KeyError, match="not found"):
        reg.build("nonexistent_key")


def test_registry_duplicate_key_raises():
    """Registering the same key twice must raise KeyError."""
    from cdlib.utils.registry import Registry

    reg = Registry("TEST")

    @reg.register("dup_key")
    class A:
        pass

    with pytest.raises(KeyError, match="Duplicate key"):

        @reg.register("dup_key")
        class B:
            pass
