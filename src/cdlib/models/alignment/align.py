import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .registry import ALIGNMENT_REGISTRY


@ALIGNMENT_REGISTRY.register("bounded")
class BoundedAlignment(nn.Module):
    """
    A bounded, change-aware feature alignment module.
    
    Computes a small optical-flow-like offset to warp f2 to align with f1,
    while explicitly bounding the maximum pixel shift to prevent warping away
    genuine semantic changes. Includes a confidence gate so regions with
    high change probability are not warped.
    """
    def __init__(
        self,
        in_channels: int,
        max_offset: float = 4.0,
        **kwargs
    ):
        super().__init__()
        self.max_offset = max_offset
        
        # Predicts 2 channels for X, Y offsets
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        )
        
        # Predicts 1 channel for the change-confidence mask [0, 1]
        self.gate_predictor = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize predictions to zero (identity mapping)
        nn.init.zeros_(self.offset_predictor[-1].weight)
        nn.init.zeros_(self.offset_predictor[-1].bias)
        nn.init.zeros_(self.gate_predictor[-2].weight)
        nn.init.zeros_(self.gate_predictor[-2].bias)
        
    def _create_mesh_grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create a normalized mesh grid [-1, 1]."""
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing='ij'
        )
        # [1, H, W, 2]
        return torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """
        Warp f2 to align with f1.
        """
        b, c, h, w = f1.shape
        
        # Concatenate features
        cat_feat = torch.cat([f1, f2], dim=1)
        
        # 1. Predict raw unbounded offset
        raw_offset = self.offset_predictor(cat_feat) # [B, 2, H, W]
        
        # 2. Bound the offset using tanh and max_offset constraint
        # Convert pixel offset to normalized grid offset [-1, 1]
        # X offset is normalized by (W - 1) / 2
        # Y offset is normalized by (H - 1) / 2
        bound_x = self.max_offset / max((w - 1) / 2.0, 1.0)
        bound_y = self.max_offset / max((h - 1) / 2.0, 1.0)
        
        offset_x = torch.tanh(raw_offset[:, 0:1, :, :]) * bound_x
        offset_y = torch.tanh(raw_offset[:, 1:2, :, :]) * bound_y
        bounded_offset = torch.cat([offset_x, offset_y], dim=1) # [B, 2, H, W]
        
        # 3. Change-confidence gate
        # 1 = genuine change (don't warp), 0 = pure viewpoint shift (warp freely)
        gate = self.gate_predictor(cat_feat) # [B, 1, H, W]
        gated_offset = bounded_offset * (1.0 - gate)
        
        # 4. Warp using F.grid_sample
        base_grid = self._create_mesh_grid(h, w, f1.device) # [1, H, W, 2]
        base_grid = base_grid.expand(b, -1, -1, -1) # [B, H, W, 2]
        
        # Note: grid_sample expects coordinates in shape [B, H, W, 2]
        # Permute gated_offset from [B, 2, H, W] to [B, H, W, 2]
        gated_offset_permuted = gated_offset.permute(0, 2, 3, 1)
        
        sampling_grid = base_grid + gated_offset_permuted
        
        # Warp f2
        f2_aligned = F.grid_sample(f2, sampling_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return f2_aligned
