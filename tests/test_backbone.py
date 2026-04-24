"""Tests for models.backbone.dinov3_backbone.

Two tiers:

1. **Always-on** (no torch / no HF_TOKEN required): config defaults,
   patch-grid math, Hydra YAML round-trip. These pin the shape contract
   the DETR head (Phase 1-2b) relies on — a ViT-B/16 on a 1024×1024 input
   must deliver a 64×64 token grid, period.

2. **Gated smoke** (actually downloads the 300MB DINOv3 weights):
   opt-in via `RUN_DINO_SMOKE=1` env var. Skipped otherwise. The plan
   specifies HF_TOKEN must also be set — we let the HF library raise
   naturally if it isn't.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from models.backbone.dinov3_backbone import (
    DEFAULT_HIDDEN_DIM,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    DinoV3Backbone,
    DinoV3BackboneConfig,
)

REPO_ROOT = Path(__file__).parent.parent
BACKBONE_YAML = REPO_ROOT / "configs" / "backbone" / "dinov3_vitb16.yaml"


# --- Config -----------------------------------------------------------------

def test_config_defaults_match_plan():
    cfg = DinoV3BackboneConfig()
    assert cfg.model_id == DEFAULT_MODEL_ID
    assert cfg.patch_size == DEFAULT_PATCH_SIZE == 16
    assert cfg.hidden_dim == DEFAULT_HIDDEN_DIM == 768
    assert cfg.freeze is True  # plan: freeze for first epochs
    assert cfg.dtype == "float32"
    assert cfg.output_layer == -1


def test_config_rejects_invalid_patch_size():
    with pytest.raises(ValueError, match="patch_size"):
        DinoV3BackboneConfig(patch_size=0)


def test_config_rejects_invalid_hidden_dim():
    with pytest.raises(ValueError, match="hidden_dim"):
        DinoV3BackboneConfig(hidden_dim=-1)


def test_config_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="dtype"):
        DinoV3BackboneConfig(dtype="fp32")  # common mistake; only long names


# --- Hydra YAML round-trip --------------------------------------------------

def test_backbone_yaml_loads_into_config():
    with BACKBONE_YAML.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # YAML has an extra `name:` key for Hydra subgroup identification; we
    # drop it and feed the rest into the dataclass.
    data.pop("name", None)
    cfg = DinoV3BackboneConfig(**data)

    assert cfg.model_id == DEFAULT_MODEL_ID
    assert cfg.patch_size == 16
    assert cfg.hidden_dim == 768
    assert cfg.freeze is True
    assert cfg.dtype == "float32"
    assert cfg.output_layer == -1


# --- Patch-grid math --------------------------------------------------------

@pytest.mark.parametrize(
    "h,w,expected",
    [
        (224, 224, (14, 14)),      # canonical ViT input
        (1024, 1024, (64, 64)),    # plan's high-res target
        (480, 640, (30, 40)),      # non-square
        (720, 1280, (45, 80)),     # BDD100K native
        (1024, 1216, (64, 76)),    # CODa 1224-wide cropped to a patch-multiple
        (16, 16, (1, 1)),          # degenerate but valid
    ],
)
def test_patch_grid_shape_for_valid_inputs(h: int, w: int, expected: tuple[int, int]):
    bb = DinoV3Backbone()
    assert bb.patch_grid_shape(h, w) == expected
    assert bb.num_patches(h, w) == expected[0] * expected[1]


@pytest.mark.parametrize("h,w", [(225, 224), (224, 225), (1000, 1024), (100, 100)])
def test_patch_grid_shape_rejects_non_multiples(h: int, w: int):
    bb = DinoV3Backbone()
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        bb.patch_grid_shape(h, w)


def test_patch_grid_shape_rejects_nonpositive_dims():
    bb = DinoV3Backbone()
    with pytest.raises(ValueError, match="positive"):
        bb.patch_grid_shape(0, 224)
    with pytest.raises(ValueError, match="positive"):
        bb.patch_grid_shape(224, -16)


def test_patch_grid_shape_respects_custom_patch_size():
    """DINOv2 fallback uses patch=14; config override must change the math."""
    bb = DinoV3Backbone(DinoV3BackboneConfig(patch_size=14))
    assert bb.patch_grid_shape(224, 224) == (16, 16)
    with pytest.raises(ValueError):
        bb.patch_grid_shape(224, 225)


# --- Load/forward without torch: import-error path --------------------------

def test_load_raises_importerror_when_torch_missing():
    """When torch isn't installed, .load() must fail fast with guidance."""
    torch_available = True
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_available = False

    if torch_available:
        pytest.skip("torch is installed; import-error path not exercised")

    bb = DinoV3Backbone()
    with pytest.raises(ImportError, match="torch and transformers"):
        bb.load()


# --- Gated smoke: real model download + forward -----------------------------

_RUN_SMOKE = os.environ.get("RUN_DINO_SMOKE") == "1"


@pytest.mark.skipif(not _RUN_SMOKE, reason="Set RUN_DINO_SMOKE=1 to exercise real DINOv3 load + forward")
def test_forward_shape_224_square():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    bb = DinoV3Backbone()
    x = torch.zeros(1, 3, 224, 224)
    features = bb.forward(x)
    assert features.shape == (1, 768, 14, 14)


@pytest.mark.skipif(not _RUN_SMOKE, reason="Set RUN_DINO_SMOKE=1 to exercise real DINOv3 load + forward")
def test_forward_shape_1024_square():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    bb = DinoV3Backbone()
    x = torch.zeros(1, 3, 1024, 1024)
    features = bb.forward(x)
    assert features.shape == (1, 768, 64, 64)


@pytest.mark.skipif(not _RUN_SMOKE, reason="Set RUN_DINO_SMOKE=1 to exercise real DINOv3 load + forward")
def test_forward_rejects_non_multiple_input():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    bb = DinoV3Backbone()
    x = torch.zeros(1, 3, 225, 224)
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        bb.forward(x)
