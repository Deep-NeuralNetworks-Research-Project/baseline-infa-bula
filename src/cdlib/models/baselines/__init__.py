"""Baseline model imports — triggers registration in MODEL_REGISTRY.

Call _import_all_baselines() to ensure all baseline models are registered
before building from config. This pattern keeps imports lazy.
"""

_imported = False


def _import_all_baselines() -> None:
    """Import all baseline modules to trigger their @register decorators."""
    global _imported
    if _imported:
        return

    # These imports trigger the @MODEL_REGISTRY.register(...) decorators
    # in each baseline file. Add new baselines here as they are created.
    # fmt: off
    from cdlib.models.baselines import rgb_ssim as _rgb_ssim       # noqa: F401
    from cdlib.models.baselines import fc_siam_diff as _fc_siam     # noqa: F401
    # fmt: on

    _imported = True
