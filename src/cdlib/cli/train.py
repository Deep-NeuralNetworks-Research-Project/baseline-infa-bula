"""CLI entrypoint for training.

Usage:
    python -m cdlib.cli.train +experiment=baseline_fcsiamdiff_sysu
    python -m cdlib.cli.train data=levir_cd model=siamese_resnet18
    python -m cdlib.cli.train -m train.lr=1e-3,3e-4,1e-4 model=fc_siam_diff
    python -m cdlib.cli.train resume_from=auto
"""

from __future__ import annotations


def main() -> None:
    """Hydra-powered training entrypoint — to be implemented in Phase 2."""
    raise NotImplementedError(
        "Training CLI will be implemented in Phase 2 (Week 2). "
        "See task 2.1 in the implementation plan."
    )


if __name__ == "__main__":
    main()
