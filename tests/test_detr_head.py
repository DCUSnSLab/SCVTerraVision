"""Tests for models.detection.detr_head.

Two tiers (same pattern as test_backbone.py):

1. **Always-on**: config defaults & validation, Hydra YAML round-trip for
   configs/detection/dinov3_detr_base.yaml, ImportError path when
   transformers is missing. These pin the interface the training loop
   (training/train_detection.py) relies on.

2. **Gated smoke**: actual model instantiation (downloads DINOv3 weights,
   builds HF DeformableDetr). Opt-in via RUN_DINO_SMOKE=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from models.detection.detr_head import (
    DEFAULT_D_MODEL,
    DEFAULT_NUM_DECODER_LAYERS,
    DEFAULT_NUM_ENCODER_LAYERS,
    DEFAULT_NUM_LABELS,
    DEFAULT_NUM_QUERIES,
    DetrHeadConfig,
    DinoV3DeformableDetr,
)

REPO_ROOT = Path(__file__).parent.parent
DETR_YAML = REPO_ROOT / "configs" / "detection" / "dinov3_detr_base.yaml"


# --- Config defaults --------------------------------------------------------


def test_config_defaults_match_plan():
    cfg = DetrHeadConfig()
    assert cfg.num_labels == DEFAULT_NUM_LABELS == 16  # CODa taxonomy
    assert cfg.num_queries == DEFAULT_NUM_QUERIES == 300
    assert cfg.d_model == DEFAULT_D_MODEL == 256
    assert cfg.encoder_layers == DEFAULT_NUM_ENCODER_LAYERS == 6
    assert cfg.decoder_layers == DEFAULT_NUM_DECODER_LAYERS == 6
    assert cfg.num_feature_levels == 1  # single-scale per ADR
    assert cfg.backbone_channels == 768  # ViT-B hidden dim
    assert cfg.aux_loss is True
    assert cfg.disable_custom_kernels is True


# --- Config validation ------------------------------------------------------


def test_config_rejects_nonpositive_num_labels():
    with pytest.raises(ValueError, match="num_labels"):
        DetrHeadConfig(num_labels=0)


def test_config_rejects_nonpositive_num_queries():
    with pytest.raises(ValueError, match="num_queries"):
        DetrHeadConfig(num_queries=-1)


def test_config_rejects_nonpositive_d_model():
    with pytest.raises(ValueError, match="d_model"):
        DetrHeadConfig(d_model=0)


def test_config_rejects_nonpositive_layers():
    with pytest.raises(ValueError, match="encoder_layers and decoder_layers"):
        DetrHeadConfig(encoder_layers=0)
    with pytest.raises(ValueError, match="encoder_layers and decoder_layers"):
        DetrHeadConfig(decoder_layers=0)


def test_config_rejects_multi_scale():
    """Phase 1-2b is explicitly single-scale; guard against silent drift."""
    with pytest.raises(ValueError, match="single-scale"):
        DetrHeadConfig(num_feature_levels=4)


def test_config_rejects_nonpositive_backbone_channels():
    with pytest.raises(ValueError, match="backbone_channels"):
        DetrHeadConfig(backbone_channels=0)


def test_config_rejects_d_model_not_divisible_by_heads():
    # 256 / 10 = 25.6 — an attention head count that won't divide d_model
    # is a classic misconfiguration; reject at config construction time.
    with pytest.raises(ValueError, match="divisible"):
        DetrHeadConfig(d_model=256, encoder_attention_heads=10)
    with pytest.raises(ValueError, match="divisible"):
        DetrHeadConfig(d_model=256, decoder_attention_heads=10)


# --- Hydra YAML round-trip --------------------------------------------------


def test_detection_yaml_loads_into_config():
    """The shipped Hydra config must construct a valid DetrHeadConfig.

    Hydra merges defaults at runtime, so we read the raw YAML and feed only
    the head-level keys into the dataclass (dropping `defaults`, `name`,
    `image_size`, `training:` block, which are loop-level, not arch-level).
    """
    with DETR_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    head_keys = {
        "num_labels", "num_queries", "d_model",
        "encoder_layers", "decoder_layers",
        "encoder_attention_heads", "decoder_attention_heads",
        "encoder_ffn_dim", "decoder_ffn_dim",
        "dropout", "activation_function",
        "num_feature_levels", "backbone_channels",
        "aux_loss", "two_stage", "with_box_refine",
        "disable_custom_kernels",
        "class_cost", "bbox_cost", "giou_cost",
        "bbox_loss_coefficient", "giou_loss_coefficient",
        "focal_alpha",
    }
    head_data = {k: v for k, v in data.items() if k in head_keys}
    cfg = DetrHeadConfig(**head_data)

    assert cfg.num_labels == 16
    assert cfg.num_queries == 300
    assert cfg.d_model == 256
    assert cfg.num_feature_levels == 1
    assert cfg.backbone_channels == 768


def test_detection_yaml_has_coda_defaults_group():
    """Sanity: the YAML must pull in backbone=dinov3_vitb16 and dataset=coda."""
    with DETR_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    defaults = data.get("defaults", [])
    flat = []
    for entry in defaults:
        if isinstance(entry, dict):
            flat.extend(entry.items())
    names = {k.lstrip("/"): v for k, v in flat}
    assert names.get("backbone") == "dinov3_vitb16"
    assert names.get("dataset") == "coda"


# --- Lazy import path -------------------------------------------------------


def test_load_raises_importerror_when_transformers_missing():
    """transformers unavailable → load() should fail fast with guidance."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("transformers is installed; import-error path not exercised")

    wrapper = DinoV3DeformableDetr()
    with pytest.raises(ImportError, match="torch and transformers"):
        wrapper.load()


def test_wrapper_construction_is_torch_free():
    """Constructing the wrapper must not import torch/transformers."""
    # Simply constructing should not raise anything — even in the torch-less
    # dev env. This mirrors DinoV3Backbone's contract so Hydra configs can be
    # validated before the ML stack is installed.
    wrapper = DinoV3DeformableDetr()
    assert wrapper.head_config.num_labels == 16
    assert wrapper.backbone.config.patch_size == 16
    assert wrapper.model is None  # not loaded yet


# --- Gated smoke ------------------------------------------------------------

_RUN_SMOKE = os.environ.get("RUN_DINO_SMOKE") == "1"


@pytest.mark.skipif(
    not _RUN_SMOKE,
    reason="Set RUN_DINO_SMOKE=1 to exercise real DINOv3 + DeformableDetr load",
)
def test_load_builds_hf_deformabledetr():
    """Full load: DINOv3 backbone + HF DeformableDetr wiring."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import DeformableDetrForObjectDetection

    wrapper = DinoV3DeformableDetr()
    model = wrapper.load()

    assert isinstance(model, DeformableDetrForObjectDetection)
    assert model.config.num_labels == 16
    assert model.config.num_queries == 300
    assert model.config.num_feature_levels == 1
    # The shim has replaced HF's native backbone.
    assert hasattr(model.model.backbone.conv_encoder, "intermediate_channel_sizes")
    assert model.model.backbone.conv_encoder.intermediate_channel_sizes == [768]


@pytest.mark.skipif(
    not _RUN_SMOKE,
    reason="Set RUN_DINO_SMOKE=1 to exercise real forward pass",
)
def test_forward_returns_detr_output_with_loss():
    """Forward with synthetic labels returns a loss + logits + pred_boxes."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    wrapper = DinoV3DeformableDetr()
    wrapper.load()

    pixel_values = torch.zeros(1, 3, 224, 224)
    labels = [
        {
            "class_labels": torch.tensor([0, 3], dtype=torch.long),
            "boxes": torch.tensor(
                [[0.5, 0.5, 0.1, 0.1], [0.2, 0.3, 0.05, 0.05]],
                dtype=torch.float32,
            ),
        }
    ]
    out = wrapper.forward(pixel_values=pixel_values, labels=labels)

    assert out.loss is not None
    assert "loss_ce" in out.loss_dict
    assert "loss_bbox" in out.loss_dict
    assert out.logits.shape[0] == 1
    assert out.logits.shape[1] == 300  # num_queries
    assert out.logits.shape[2] == 16  # num_labels
    assert out.pred_boxes.shape == (1, 300, 4)
