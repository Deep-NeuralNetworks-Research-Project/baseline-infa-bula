"""FUSION_REGISTRY — all feature fusion strategies register here.

Register your fusion module with:
    @FUSION_REGISTRY.register("abs_diff")
    class AbsDiffFusion(nn.Module):
        def forward(self, f1: Tensor, f2: Tensor) -> Tensor:
            ...
"""

from cdlib.utils.registry import Registry

FUSION_REGISTRY = Registry("FUSION")
