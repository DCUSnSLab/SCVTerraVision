"""DINOv2 backbone wrapper.

Loads DINOv2 from HuggingFace transformers (`facebook/dinov2-{small,base,large}`),
returns multi-layer patch token features suitable for both linear and DPT heads.

Output is a `DINOv2Output` with:
    features: list of (B, N_patches, C) per requested layer  (CLS already removed)
    grid_hw:  (H_p, W_p) - patch grid size for the input
    patch_size: int (14 for ViT-S/14)
    embed_dim: int

The wrapper assumes input images already padded to a multiple of patch_size
(see camera_perception.data.transforms).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

VARIANT_TO_HF = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
}

VARIANT_TO_DEPTH = {
    "small": 12,
    "base": 12,
    "large": 24,
}


@dataclass
class DINOv2Output:
    features: list[torch.Tensor]   # each (B, N, C); CLS removed
    grid_hw: tuple[int, int]
    patch_size: int
    embed_dim: int


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        variant: str = "small",
        out_layers: tuple[int, ...] | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_TO_HF:
            raise ValueError(f"Unknown DINOv2 variant: {variant!r}")
        from transformers import Dinov2Model  # lazy import

        self.variant = variant
        self.model = Dinov2Model.from_pretrained(VARIANT_TO_HF[variant])
        self.patch_size = self.model.config.patch_size
        self.embed_dim = self.model.config.hidden_size
        depth = VARIANT_TO_DEPTH[variant]

        if out_layers is None:
            # Default: last 4 layers, useful for DPT-style fusion.
            out_layers = (depth - 4, depth - 3, depth - 2, depth - 1)
        self.out_layers = tuple(int(i) for i in out_layers)
        for i in self.out_layers:
            if i < 0 or i >= depth:
                raise ValueError(f"out_layer index {i} out of range [0, {depth})")

        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> DINOv2Backbone:  # type: ignore[override]
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor) -> DINOv2Output:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(x.shape)}")
        b, _, h, w = x.shape
        ps = self.patch_size
        if h % ps != 0 or w % ps != 0:
            raise ValueError(
                f"Input H={h}, W={w} must be multiples of patch_size={ps}"
            )
        gh, gw = h // ps, w // ps

        ctx = torch.no_grad() if self.freeze else _NullCtx()
        with ctx:
            out = self.model(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
            )
        # hidden_states: tuple of length depth+1 (embedding + each block).
        # Index i+1 corresponds to block i's output.
        hs = out.hidden_states
        feats: list[torch.Tensor] = []
        for i in self.out_layers:
            h_state = hs[i + 1]                 # (B, 1+N, C)
            feats.append(h_state[:, 1:, :])     # drop CLS

        return DINOv2Output(
            features=feats,
            grid_hw=(gh, gw),
            patch_size=ps,
            embed_dim=self.embed_dim,
        )


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:
        return None
