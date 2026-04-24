"""DINOv3 + HF Deformable DETR adapter (Phase 1-2b baseline).

The approved head is HF `DeformableDetrForObjectDetection` (ADR
20260424_detr-head-library), wired on top of our `DinoV3Backbone` instead
of HF's default ResNet-50. Single feature level (num_feature_levels=1) —
DINOv3 ViT-B/16 is a stride-16 single-scale encoder. A multi-scale
pyramid (stride 8/16/32) is deferred to Phase 1-2c if the baseline is
weak.

Design:
  - torch and transformers are imported lazily inside `load()` / forward
    paths. The module imports cleanly in the torch-less dev env (same
    pattern as `DinoV3Backbone`).
  - We compose, not subclass: `DinoV3DeformableDetr` owns a
    `DinoV3Backbone` and a lazily-built `DeformableDetrForObjectDetection`
    whose native convolutional backbone is replaced with a shim that
    forwards to DINOv3. The shim also rebuilds the input-projection
    layer so 768 → d_model works on a single level.
  - Forward returns the HF `DeformableDetrObjectDetectionOutput` verbatim
    (has `loss`, `loss_dict`, `logits`, `pred_boxes`, `auxiliary_outputs`).
    Callers consume it like any HF model output.

Gated smoke: see `tests/test_detr_head.py` behind RUN_DINO_SMOKE=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.backbone.dinov3_backbone import (
    DEFAULT_HIDDEN_DIM,
    DinoV3Backbone,
    DinoV3BackboneConfig,
)

# 16 classes come from configs/dataset/coda_taxonomy.yaml (Phase 1-1b).
DEFAULT_NUM_LABELS = 16
DEFAULT_NUM_QUERIES = 300
DEFAULT_D_MODEL = 256
DEFAULT_NUM_ENCODER_LAYERS = 6
DEFAULT_NUM_DECODER_LAYERS = 6


@dataclass(frozen=True)
class DetrHeadConfig:
    """Architecture knobs for the DINOv3 + Deformable DETR head.

    Defaults match the Phase 1-2b baseline (CODa, 16 ops classes, single
    feature level). Override via Hydra:
        python -m training.train_detection detection.num_queries=500
    """

    num_labels: int = DEFAULT_NUM_LABELS
    num_queries: int = DEFAULT_NUM_QUERIES
    d_model: int = DEFAULT_D_MODEL
    encoder_layers: int = DEFAULT_NUM_ENCODER_LAYERS
    decoder_layers: int = DEFAULT_NUM_DECODER_LAYERS
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    encoder_ffn_dim: int = 1024
    decoder_ffn_dim: int = 1024
    dropout: float = 0.1
    activation_function: str = "relu"
    # Single-scale until Phase 1-2c pyramid experiment.
    num_feature_levels: int = 1
    # DINOv3 ViT-B hidden dim. If you swap to ViT-L (1024) or DINOv2 ViT-B/14
    # (768), adjust here; shim rebuilds the input projection from this value.
    backbone_channels: int = DEFAULT_HIDDEN_DIM
    # DETR hyperparameters — baseline values from HF defaults + plan.
    aux_loss: bool = True
    two_stage: bool = False
    with_box_refine: bool = False
    # `disable_custom_kernels=True` forces the PyTorch fallback for deformable
    # attention — no ninja/CUDA build needed. Performance hit is acceptable
    # for the baseline; flip to False once the CUDA ext compiles on target HW.
    disable_custom_kernels: bool = True
    # Matcher / set-loss cost coefficients (HF defaults — left exposed for
    # sweep experiments).
    class_cost: float = 2.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    focal_alpha: float = 0.25

    def __post_init__(self) -> None:
        if self.num_labels <= 0:
            raise ValueError(f"num_labels must be positive, got {self.num_labels}")
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {self.num_queries}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.encoder_layers <= 0 or self.decoder_layers <= 0:
            raise ValueError(
                "encoder_layers and decoder_layers must be positive; got "
                f"{self.encoder_layers}, {self.decoder_layers}"
            )
        if self.num_feature_levels != 1:
            # Phase 1-2b is explicitly single-scale (see ADR). A pyramid
            # variant would need a separate shim; reject early to avoid
            # silent config mismatches.
            raise ValueError(
                "num_feature_levels must be 1 in Phase 1-2b (single-scale "
                f"DINOv3 ViT); got {self.num_feature_levels}. A multi-scale "
                "variant is deferred to Phase 1-2c."
            )
        if self.backbone_channels <= 0:
            raise ValueError(
                f"backbone_channels must be positive, got {self.backbone_channels}"
            )
        if self.encoder_attention_heads <= 0 or self.decoder_attention_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.d_model % self.encoder_attention_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"encoder_attention_heads={self.encoder_attention_heads}"
            )
        if self.d_model % self.decoder_attention_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by "
                f"decoder_attention_heads={self.decoder_attention_heads}"
            )


class DinoV3DeformableDetr:
    """DINOv3 ViT-B/16 backbone wired into HF Deformable DETR.

    Usage:
        head_cfg = DetrHeadConfig()
        model = DinoV3DeformableDetr(head_cfg)
        model.load()               # materializes torch modules
        out = model.forward(pixel_values, pixel_mask=None, labels=labels)
        # out.loss, out.loss_dict, out.logits, out.pred_boxes
    """

    def __init__(
        self,
        head_config: DetrHeadConfig | None = None,
        backbone_config: DinoV3BackboneConfig | None = None,
    ) -> None:
        self.head_config = head_config or DetrHeadConfig()
        self.backbone = DinoV3Backbone(backbone_config)
        self._model: Any = None  # DeformableDetrForObjectDetection, lazy.

    # -- Build / load ------------------------------------------------------

    def _build_detr_config(self) -> Any:
        """Translate DetrHeadConfig → HF DeformableDetrConfig."""
        from transformers import DeformableDetrConfig

        return DeformableDetrConfig(
            num_labels=self.head_config.num_labels,
            num_queries=self.head_config.num_queries,
            d_model=self.head_config.d_model,
            encoder_layers=self.head_config.encoder_layers,
            decoder_layers=self.head_config.decoder_layers,
            encoder_attention_heads=self.head_config.encoder_attention_heads,
            decoder_attention_heads=self.head_config.decoder_attention_heads,
            encoder_ffn_dim=self.head_config.encoder_ffn_dim,
            decoder_ffn_dim=self.head_config.decoder_ffn_dim,
            dropout=self.head_config.dropout,
            activation_function=self.head_config.activation_function,
            num_feature_levels=self.head_config.num_feature_levels,
            auxiliary_loss=self.head_config.aux_loss,
            two_stage=self.head_config.two_stage,
            with_box_refine=self.head_config.with_box_refine,
            disable_custom_kernels=self.head_config.disable_custom_kernels,
            class_cost=self.head_config.class_cost,
            bbox_cost=self.head_config.bbox_cost,
            giou_cost=self.head_config.giou_cost,
            bbox_loss_coefficient=self.head_config.bbox_loss_coefficient,
            giou_loss_coefficient=self.head_config.giou_loss_coefficient,
            focal_alpha=self.head_config.focal_alpha,
            # HF tries to instantiate a ResNet backbone when this is True; we
            # replace it post-construction, so set False and let HF skip the
            # timm path.
            use_timm_backbone=False,
        )

    def load(self) -> Any:
        """Materialize the DINOv3 backbone + HF DeformableDetr head.

        Raises ImportError with guidance if torch / transformers are not
        installed — Phase 1-2a / 1-2b dev environments frequently omit the
        ML stack; see DinoV3Backbone.load() for the same pattern.
        """
        if self._model is not None:
            return self._model

        try:
            import torch
            import torch.nn as nn
            from transformers import DeformableDetrForObjectDetection
        except ImportError as e:
            raise ImportError(
                "DinoV3DeformableDetr.load() requires torch and transformers. "
                "Install via `pip install -e .[dev]` or the training extras."
            ) from e

        self.backbone.load()
        detr_config = self._build_detr_config()

        model = DeformableDetrForObjectDetection(detr_config)

        # Replace HF's conv-encoder backbone with a shim that forwards to
        # DINOv3 and emits the single feature level HF expects (list of
        # (features, mask) pairs).
        shim = _build_conv_encoder_shim(
            backbone=self.backbone,
            backbone_channels=self.head_config.backbone_channels,
        )
        model.model.backbone.conv_encoder = shim

        # HF DeformableDetr builds `input_proj` from the backbone's declared
        # channel sizes at init time (ResNet's [512, 1024, 2048] by default).
        # With num_feature_levels=1 and DINOv3 ViT, we need one 1x1 conv
        # mapping 768 → d_model. Rebuild it to match.
        model.model.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.head_config.backbone_channels,
                        self.head_config.d_model,
                        kernel_size=1,
                    ),
                    nn.GroupNorm(32, self.head_config.d_model),
                )
            ]
        )
        # Standard HF-style init on the newly built projection.
        for proj in model.model.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1.0)
            nn.init.constant_(proj[0].bias, 0.0)

        self._model = model
        return model

    # -- Forward -----------------------------------------------------------

    def forward(
        self,
        pixel_values: Any,
        pixel_mask: Any = None,
        labels: Any = None,
    ) -> Any:
        """Forward through DINOv3 + DeformableDetr head.

        pixel_values: (B, 3, H, W) tensor, H and W multiples of patch_size.
        pixel_mask:   optional (B, H, W) bool tensor; True = valid pixel.
                      If None, all pixels are treated as valid.
        labels:       optional list[dict] with HF-standard keys
                      `class_labels`, `boxes` (cxcywh normalized). Required
                      for loss computation.

        Returns a `DeformableDetrObjectDetectionOutput` (HF) whose main
        fields are `.loss`, `.loss_dict`, `.logits`, `.pred_boxes`,
        `.auxiliary_outputs`.
        """
        import torch

        model = self.load()
        if pixel_mask is None:
            # All-valid mask; HF expects (B, H, W) of 1s.
            B, _, H, W = pixel_values.shape
            pixel_mask = torch.ones(
                (B, H, W), dtype=torch.long, device=pixel_values.device
            )
        return model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )

    __call__ = forward

    @property
    def model(self) -> Any:
        """The underlying HF `DeformableDetrForObjectDetection` (None until load())."""
        return self._model


def _build_conv_encoder_shim(
    backbone: DinoV3Backbone, backbone_channels: int
) -> Any:
    """Return a torch nn.Module that mimics HF `DeformableDetrConvEncoder`.

    HF's `DeformableDetrModel` calls
        self.backbone.conv_encoder(pixel_values, pixel_mask)
    and expects a list of `(features, mask)` tuples, one per feature level.
    With num_feature_levels=1 we return a single-element list. The mask is
    downsampled to patch resolution via nearest interpolation.

    `intermediate_channel_sizes` is read by HF during its own model init;
    we rebuild `input_proj` explicitly in `DinoV3DeformableDetr.load()`,
    but expose the attribute anyway for parity.

    Torch is imported inside this factory so the module stays import-safe
    without torch (same pattern as `DinoV3Backbone`).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _DinoV3ConvEncoderShim(nn.Module):
        intermediate_channel_sizes = [backbone_channels]

        def __init__(self) -> None:
            super().__init__()
            # Register the backbone so its parameters show up in
            # model.parameters() / state_dict.
            self.model = backbone.load()

        def forward(self, pixel_values: Any, pixel_mask: Any) -> Any:
            features = backbone.forward(pixel_values)  # (B, C, Hp, Wp)
            if pixel_mask is None:
                B, _, H, W = pixel_values.shape
                pixel_mask = torch.ones(
                    (B, H, W), dtype=torch.long, device=pixel_values.device
                )
            mask_f = pixel_mask[:, None].float()
            mask_ds = F.interpolate(
                mask_f, size=features.shape[-2:], mode="nearest"
            )
            mask_out = mask_ds[:, 0].to(torch.bool)
            return [(features, mask_out)]

    return _DinoV3ConvEncoderShim()
