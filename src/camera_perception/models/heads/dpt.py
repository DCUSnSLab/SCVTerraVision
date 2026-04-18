"""DPT-lite segmentation head.

A simplified DPT (Dense Prediction Transformer) decoder that fuses 4 ViT
intermediate layers with progressive upsampling. Faithful to the original
intent (multi-scale fusion) but trimmed for clarity and Jetson-friendliness.

Reference: Ranftl et al. "Vision Transformers for Dense Prediction" (2021).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from camera_perception.models.backbones.dinov2 import DINOv2Output


class _ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _FusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.res1 = _ResidualConv(channels)
        self.res2 = _ResidualConv(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.res1(skip)
        x = self.res2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.proj(x)


class DPTHead(nn.Module):
    """Fuses 4 token feature maps from DINOv2 into a dense prediction.

    Reassembles each (B, N, C) into (B, C, gh, gw), projects to a common
    dim, applies multiscale resampling (rates 4, 2, 1, 0.5) so the four
    feature maps roughly cover an FPN-like pyramid, then fuses top-down.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        rescale_rates: tuple[float, float, float, float] = (4.0, 2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        self.rescale_rates = rescale_rates
        self.proj = nn.ModuleList(
            [nn.Conv2d(embed_dim, hidden_dim, kernel_size=1) for _ in range(4)]
        )
        self.fusion = nn.ModuleList([_FusionBlock(hidden_dim) for _ in range(4)])
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def _tokens_to_grid(self, tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        b, n, c = tokens.shape
        if n != gh * gw:
            raise ValueError(f"tokens length {n} != gh*gw={gh*gw}")
        return tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()

    @staticmethod
    def _resample(x: torch.Tensor, rate: float) -> torch.Tensor:
        if rate == 1.0:
            return x
        return F.interpolate(x, scale_factor=rate, mode="bilinear", align_corners=False)

    def forward(self, backbone_out: DINOv2Output, out_hw: tuple[int, int]) -> torch.Tensor:
        if len(backbone_out.features) != 4:
            raise ValueError(
                f"DPTHead expects 4 feature layers, got {len(backbone_out.features)}"
            )
        gh, gw = backbone_out.grid_hw

        # Step 1: tokens → grid → 1x1 proj → multiscale resample
        feats = []
        for i, tok in enumerate(backbone_out.features):
            grid = self._tokens_to_grid(tok, gh, gw)
            proj = self.proj[i](grid)
            feats.append(self._resample(proj, self.rescale_rates[i]))
        # feats[0]: highest-res (rate=4), ..., feats[3]: lowest-res (rate=0.5)

        # Step 2: top-down fusion (start from lowest-res, fuse upward)
        x = self.fusion[3](feats[3])
        # Match resolution before adding skip via bilinear resize.
        for i in [2, 1, 0]:
            skip = F.interpolate(feats[i], size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = self.fusion[i](x, skip)

        logits = self.classifier(x)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits
