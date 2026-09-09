"""ENCODER_REGISTRY — all encoder backbones register here.

Register your encoder with:
    @ENCODER_REGISTRY.register("my_encoder")
    class MyEncoder(nn.Module):
        def forward(self, x: Tensor) -> list[Tensor]:
            '''Return multi-scale feature list [C2, C3, C4, ...]'''
            ...
"""

from cdlib.utils.registry import Registry

ENCODER_REGISTRY = Registry("ENCODER")
