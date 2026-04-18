"""Linear segmentation head.

Takes the last patch-token feature map (B, N, C) from a ViT backbone, applies
a 1x1 conv (= linear over channels), reshapes to (B, num_classes, H_p, W_p),
and bilinearly upsamples back to the input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_perception.models.backbones.dinov2 import DINOv2Output


class LinearHead(nn.Module):
    """Frozen-backbone linear probe. Multi-GPU users should call
    `nn.SyncBatchNorm.convert_sync_batchnorm(model)` (Lightning does this
    automatically when `sync_batchnorm=True`)."""

    def __init__(self, embed_dim: int, num_classes: int, use_bn: bool = True) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(embed_dim) if use_bn else nn.Identity()
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, backbone_out: DINOv2Output, out_hw: tuple[int, int]) -> torch.Tensor:
        feats = backbone_out.features[-1]            # (B, N, C)
        b, n, c = feats.shape
        gh, gw = backbone_out.grid_hw
        if n != gh * gw:
            raise ValueError(f"feature length {n} != gh*gw={gh*gw}")
        x = feats.transpose(1, 2).reshape(b, c, gh, gw).contiguous()
        x = self.bn(x)
        logits = self.classifier(x)                  # (B, K, gh, gw)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits
