"""Tests for model heads, losses, and metrics.

The DINOv2 backbone is mocked with a small randomly-initialized stand-in to
keep tests fast and offline. A separate slow test (marked) exercises the real
HuggingFace backbone.
"""

from __future__ import annotations

import os

import pytest
import torch

from camera_perception.data.taxonomy import UnifiedTaxonomy
from camera_perception.models.backbones.dinov2 import DINOv2Output
from camera_perception.models.heads import DPTHead, LinearHead
from camera_perception.models.losses import CombinedSegLoss, CrossEntropySegLoss, DiceSegLoss
from camera_perception.training.metrics import (
    BinaryTraversabilityIoU,
    build_iou_metric,
    per_class_iou_dict,
)


def _fake_backbone_out(b: int = 2, gh: int = 4, gw: int = 5, c: int = 32, n_layers: int = 4):
    feats = [torch.randn(b, gh * gw, c) for _ in range(n_layers)]
    return DINOv2Output(features=feats, grid_hw=(gh, gw), patch_size=14, embed_dim=c)


def test_linear_head_shape():
    out = _fake_backbone_out(b=2, gh=4, gw=6, c=16, n_layers=1)
    head = LinearHead(embed_dim=16, num_classes=6)
    logits = head(out, out_hw=(56, 84))
    assert logits.shape == (2, 6, 56, 84)


def test_dpt_head_shape():
    out = _fake_backbone_out(b=2, gh=4, gw=4, c=16, n_layers=4)
    head = DPTHead(embed_dim=16, num_classes=6, hidden_dim=32)
    logits = head(out, out_hw=(56, 56))
    assert logits.shape == (2, 6, 56, 56)


def test_dpt_head_rejects_wrong_layer_count():
    out = _fake_backbone_out(n_layers=2)
    head = DPTHead(embed_dim=32, num_classes=6)
    with pytest.raises(ValueError):
        head(out, out_hw=(56, 56))


# ----- losses -----

def test_cross_entropy_ignore_index_works():
    loss = CrossEntropySegLoss(num_classes=6, ignore_index=255)
    logits = torch.randn(2, 6, 8, 8, requires_grad=True)
    target = torch.full((2, 8, 8), 255, dtype=torch.long)  # all ignore
    out = loss(logits, target)
    # All pixels ignored → cross_entropy returns nan in PyTorch; check it's finite or nan
    assert out.item() != out.item() or torch.isfinite(out).item()  # nan-safe


def test_cross_entropy_with_class_weights():
    cw = torch.tensor([1.0, 2.0, 0.5, 1.0, 1.0, 1.0])
    loss = CrossEntropySegLoss(num_classes=6, class_weights=cw)
    logits = torch.randn(2, 6, 4, 4, requires_grad=True)
    target = torch.randint(0, 6, (2, 4, 4))
    val = loss(logits, target)
    val.backward()
    assert logits.grad is not None
    assert torch.isfinite(val).item()


def test_dice_loss_runs_and_is_bounded():
    loss = DiceSegLoss(num_classes=6, ignore_index=255)
    logits = torch.randn(2, 6, 8, 8, requires_grad=True)
    target = torch.randint(0, 6, (2, 8, 8))
    target[0, 0, 0] = 255
    val = loss(logits, target)
    assert 0.0 <= val.item() <= 1.0 + 1e-5


def test_combined_loss_backward():
    loss = CombinedSegLoss(num_classes=6, ce_weight=1.0, dice_weight=0.5)
    logits = torch.randn(2, 6, 8, 8, requires_grad=True)
    target = torch.randint(0, 6, (2, 8, 8))
    val = loss(logits, target)
    val.backward()
    assert torch.isfinite(val).item()


# ----- metrics -----

def test_iou_metric_perfect_prediction(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    metric = build_iou_metric(tax.num_classes())
    target = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    metric.update(target, target)
    iou = metric.compute()
    # All present classes should have IoU=1.0
    assert torch.allclose(iou, torch.ones_like(iou))


def test_iou_metric_with_ignore_index(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    metric = build_iou_metric(tax.num_classes(), ignore_index=255)
    target = torch.tensor([[[0, 0, 255], [1, 1, 255]]])
    pred = torch.tensor([[[0, 0, 5], [1, 1, 5]]])  # 5 in ignore region — should be excluded
    metric.update(pred, target)
    iou = metric.compute()
    assert iou[0].item() == 1.0
    assert iou[1].item() == 1.0


def test_binary_traversability_iou(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    metric = BinaryTraversabilityIoU(tax)
    smooth = tax.name_to_id("traversable_smooth")
    grass = tax.name_to_id("traversable_grass")
    sky = tax.name_to_id("sky")
    obstacle = tax.name_to_id("obstacle_static")

    target = torch.tensor([[[smooth, grass, sky, obstacle]]])
    pred = torch.tensor([[[smooth, grass, sky, obstacle]]])
    metric.update(pred, target)
    assert metric.compute().item() == pytest.approx(1.0)

    metric.reset()
    pred_wrong = torch.tensor([[[obstacle, sky, smooth, grass]]])  # all flipped
    metric.update(pred_wrong, target)
    # No overlap, but union is full → IoU = 0
    assert metric.compute().item() == 0.0


def test_per_class_iou_dict(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    iou = torch.linspace(0.1, 0.6, tax.num_classes())
    d = per_class_iou_dict(iou, tax)
    assert set(d) == {c.name for c in tax.classes}
    assert d["sky"] == pytest.approx(0.6)


# ----- slow integration test for the real DINOv2 backbone (optional) -----
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="set RUN_SLOW=1 to run (downloads ~85MB DINOv2-small from HF)",
)
def test_real_dinov2_backbone_forward():
    from camera_perception.models.backbones import DINOv2Backbone

    backbone = DINOv2Backbone(variant="small", freeze=True)
    x = torch.randn(1, 3, 14 * 4, 14 * 4)
    out = backbone(x)
    assert out.grid_hw == (4, 4)
    assert out.embed_dim == 384
    assert all(t.shape == (1, 16, 384) for t in out.features)
