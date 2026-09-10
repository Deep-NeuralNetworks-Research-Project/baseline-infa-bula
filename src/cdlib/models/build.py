"""Central builder functions — the ONLY entrypoints the CLI calls.

These four functions are frozen contracts. The CLI and trainer never import
a model/dataset/loss/optimizer class directly; they go through these builders.
This makes every component swappable via config without code changes.

Builders:
    build_model(cfg) -> nn.Module
    build_dataset(cfg, split) -> Dataset
    build_loss(cfg) -> Loss
    build_optimizer(cfg, params) -> Optimizer
"""

from __future__ import annotations

from typing import Any, Iterator

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import Dataset

# Import registries — this triggers registration of all components
# that have been imported elsewhere (e.g., via __init__.py imports)
from cdlib.data.registry import DATASET_REGISTRY
from cdlib.losses.registry import LOSS_REGISTRY
from cdlib.models.encoders.registry import ENCODER_REGISTRY  # noqa: F401


def build_model(cfg: DictConfig) -> nn.Module:
    """Build a model from config.

    The config must have a `model.name` key that matches a key in the
    appropriate registry. For baselines, this looks up directly; for the
    proposed model, it composes encoder+fusion+alignment+decoder+heads.

    Args:
        cfg: Full Hydra config. Must contain `model.name` and model-specific params.

    Returns:
        nn.Module whose forward signature matches the frozen contract:
        forward(img1, img2) -> {"logits": Tensor[B,1,H,W], "confidence": ..., "aux": {...}}
    """
    model_cfg = cfg.model
    model_name = model_cfg.name

    # Import model registries lazily to allow all components to register
    from cdlib.models.baselines import _import_all_baselines

    _import_all_baselines()

    # For baselines, we use a flat MODEL_REGISTRY approach
    # For the proposed model, we compose from sub-registries
    from cdlib.models._model_registry import MODEL_REGISTRY

    model_params = {k: v for k, v in model_cfg.items() if k != "name"}
    return MODEL_REGISTRY.build(model_name, **model_params)


def build_dataset(cfg: DictConfig, split: str) -> Dataset:
    """Build a dataset for the given split.

    Args:
        cfg: Full Hydra config. Must contain `data.name` and data-specific params.
        split: One of "train", "val", "test".

    Returns:
        Dataset whose __getitem__ returns the frozen contract dict.
    """
    data_cfg = cfg.data
    dataset_name = data_cfg.name

    dataset_params = {k: v for k, v in data_cfg.items() if k != "name"}
    return DATASET_REGISTRY.build(dataset_name, split=split, **dataset_params)


def build_loss(cfg: DictConfig) -> Any:
    """Build a loss module from config.

    Args:
        cfg: Full Hydra config. Must contain `loss.name` and loss-specific params.

    Returns:
        Loss module whose compute(outputs, batch) returns:
        {"loss": Tensor[], "loss/bce": ..., "loss/dice": ..., ...}
    """
    loss_cfg = cfg.loss
    loss_name = loss_cfg.name

    loss_params = {k: v for k, v in loss_cfg.items() if k != "name"}
    
    # Dynamic Loss Weighting for Imbalance
    if "data" in cfg and "name" in cfg.data:
        from cdlib.data.registry import DATASET_REGISTRY
        dataset_cls = DATASET_REGISTRY.get(cfg.data.name)
        ratio = getattr(dataset_cls, "published_changed_pixel_ratio", None)
        if ratio is not None and ratio > 0 and ratio < 1:
            pos_weight = (1.0 - ratio) / ratio
            loss_params["pos_weight"] = pos_weight

    return LOSS_REGISTRY.build(loss_name, **loss_params)


def build_optimizer(
    cfg: DictConfig, params: Iterator[nn.Parameter]
) -> torch.optim.Optimizer:
    """Build an optimizer from config.

    Args:
        cfg: Full Hydra config. Must contain `train.optimizer` with `name` and params.
        params: Model parameters to optimise.

    Returns:
        torch.optim.Optimizer instance.
    """
    optim_cfg = cfg.train.optimizer
    optim_name = optim_cfg.name.lower()

    # Standard PyTorch optimizers — no registry needed, these are stable
    optim_params = {k: v for k, v in optim_cfg.items() if k != "name"}

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    if optim_name not in optimizers:
        raise ValueError(
            f"Unknown optimizer '{optim_name}'. Available: {list(optimizers.keys())}"
        )

    return optimizers[optim_name](params, **optim_params)
