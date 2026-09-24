from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from cdlib.models._model_registry import MODEL_REGISTRY
from cdlib.models.alignment.registry import ALIGNMENT_REGISTRY
import segmentation_models_pytorch as smp

@MODEL_REGISTRY.register("proposed_effnet")
class ProposedModel(nn.Module):
    """The proposed composed model architecture for Change Detection."""
    
    def __init__(
        self,
        encoder: dict[str, Any],
        fusion: dict[str, Any],
        alignment: dict[str, Any],
        decoder: dict[str, Any],
        heads: dict[str, Any],
        in_channels: int = 3,
        **kwargs
    ) -> None:
        super().__init__()
        
        encoder_name = encoder.get("name", "efficientnet_b0").replace("_", "-")
        encoder_weights = encoder.get("weights", "imagenet")
        
        self.encoder = smp.encoders.get_encoder(
            encoder_name, 
            in_channels=in_channels, 
            depth=5, 
            weights=encoder_weights
        )
        
        if encoder.get("freeze", False):
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        
        self.fusion_name = fusion.get("name", "abs_diff")
        
        encoder_channels = self.encoder.out_channels

        # 3. Alignment Module (Optional)
        self.alignments = nn.ModuleList()
        alignment_name = alignment.get("name") if alignment else None
        
        if alignment_name:
            alignment_params = {k: v for k, v in alignment.items() if k != "name"}
            for in_ch in encoder_channels:
                align_mod = ALIGNMENT_REGISTRY.build(
                    alignment_name, 
                    in_channels=in_ch, 
                    **alignment_params
                )
                self.alignments.append(align_mod)
                
        # 4. Decoder (using SMP's Unet decoder)
        
        # For abs_diff and signed, channels are the same as encoder output.
        decoder_channels = (256, 128, 64, 32, 16)
        
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_norm='batchnorm',
            add_center_block=False,
            attention_type=None
        )
        
        self.segmentation_head = smp.base.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=1,
            activation=None,
            kernel_size=3
        )
        
        self.has_even_head = heads.get("confidence", False)
        if self.has_even_head:
            # Even head costs "one extra decoder pass" according to paper
            self.even_decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=5,
                use_norm='batchnorm',
                add_center_block=False,
                attention_type=None
            )
            self.even_head = smp.base.SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=1,
                activation=None,
                kernel_size=3
            )

    def _fuse(self, f1: list[torch.Tensor], f2: list[torch.Tensor]) -> list[torch.Tensor]:
        fused = []
        for i, (x, y) in enumerate(zip(f1, f2)):
            # Optional Alignment
            if len(self.alignments) > 0:
                y = self.alignments[i](x, y)
                
            # Fusion
            if self.fusion_name == "abs_diff":
                fused.append(torch.abs(x - y))
            elif self.fusion_name == "signed_fusion":
                fused.append(x - y)
            elif self.fusion_name == "concat":
                fused.append(torch.cat([x, y], dim=1))
            else:
                fused.append(torch.abs(x - y))
        return fused

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict[str, Any]:
        f1 = self.encoder(img1)
        f2 = self.encoder(img2)
        
        fused = self._fuse(f1, f2)
        
        # Unet Decoder expects a list of features [x, f1, f2, f3, f4, f5]
        dec_out = self.decoder(*fused) if tuple(map(int, torch.__version__.split('.')[:2])) < (2, 0) else self.decoder(fused)
        logits = self.segmentation_head(dec_out)
        
        confidence = None
        if self.has_even_head:
            even_fused = []
            for x, y in zip(f1, f2):
                even_fused.append(x + y)
            even_dec_out = self.even_decoder(*even_fused) if tuple(map(int, torch.__version__.split('.')[:2])) < (2, 0) else self.even_decoder(even_fused)
            confidence = self.even_head(even_dec_out)
            
        return {
            "logits": logits,
            "confidence": confidence,
            "aux": {}
        }
