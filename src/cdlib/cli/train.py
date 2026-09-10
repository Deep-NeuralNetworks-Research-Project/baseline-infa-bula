"""CLI entrypoint for training.

Usage:
    python -m cdlib.cli.train +experiment=baseline_fcsiamdiff_sysu
    python -m cdlib.cli.train data=levir_cd model=siamese_resnet18
    python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff
    python -m cdlib.cli.train resume_from=auto
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra-powered training entrypoint with multirun support.

    Hydra's ``-m`` flag enables sweeps automatically:
        python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4
    """
    from torch.utils.data import DataLoader

    from cdlib.engine.trainer import Trainer
    from cdlib.models.build import build_loss, build_model, build_optimizer

    logger.info(f"Starting training: model={cfg.model.name}, data={cfg.data.name}")

    # Build components through the frozen builder contracts
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model.parameters())

    # Build dataset + dataloader
    # NOTE: dataset builders depend on P1's data loaders (not yet available).
    # For now, if DATASET_REGISTRY has the requested dataset, use it.
    # Otherwise, fall back to a synthetic dataset for testing.
    train_loader, val_loader = _build_dataloaders(cfg)

    # Run directory from Hydra's output dir
    run_dir = hydra.utils.get_original_cwd() + "/results/" + cfg.get(
        "experiment_name", f"{cfg.model.name}_{cfg.data.name}"
    )

    # Create and run trainer
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        run_dir=run_dir,
    )

    metrics = trainer.train()
    logger.info(f"Training complete. Final metrics: {metrics}")
    return metrics


def _build_dataloaders(cfg: DictConfig) -> tuple:
    """Build train and val DataLoaders.

    Falls back to synthetic data if the requested dataset isn't registered yet
    (P1's data loaders are expected by end of week 2).
    """
    from torch.utils.data import DataLoader

    from cdlib.data.registry import DATASET_REGISTRY

    dataset_name = cfg.data.name
    batch_size = cfg.train.get("batch_size", 8)
    num_workers = cfg.data.get("num_workers", 4)

    if dataset_name in DATASET_REGISTRY:
        from cdlib.models.build import build_dataset

        train_ds = build_dataset(cfg, split="train")
        val_ds = build_dataset(cfg, split="val")
    else:
        logger.warning(
            f"Dataset '{dataset_name}' not registered yet. "
            f"Using synthetic data for testing. "
            f"P1's data loaders should be available by end of week 2."
        )
        train_ds = _SyntheticDataset(
            num_samples=100,
            img_size=cfg.data.get("img_size", 256),
            in_channels=cfg.model.get("in_channels", 3),
        )
        val_ds = _SyntheticDataset(
            num_samples=20,
            img_size=cfg.data.get("img_size", 256),
            in_channels=cfg.model.get("in_channels", 3),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class _SyntheticDataset:
    """Minimal synthetic dataset for testing the training pipeline.

    Generates random image pairs and binary change masks.
    Matches the frozen dataset __getitem__ contract.
    """

    def __init__(
        self, num_samples: int = 100, img_size: int = 256, in_channels: int = 3
    ) -> None:
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_channels = in_channels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        import torch

        H = W = self.img_size
        C = self.in_channels
        return {
            "img1": torch.rand(C, H, W),
            "img2": torch.rand(C, H, W),
            "mask": (torch.rand(1, H, W) > 0.5).float(),
            "nuisance_label": torch.tensor(0, dtype=torch.int64),
            "meta": {
                "source_video": "synthetic",
                "scene_id": f"synth_{idx}",
                "frame_idx": (0, 1),
                "pair_id": f"synth_pair_{idx}",
                "dataset": "synthetic",
            },
        }


if __name__ == "__main__":
    main()
