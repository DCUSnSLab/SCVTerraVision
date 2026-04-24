"""Tests for evaluation.det_metrics.

Always-on: schema validation of `compute_coco_map` inputs, lazy-import
ImportError when pycocotools is missing, FileNotFoundError on bad paths.

Gated smoke (pycocotools-installed): synthetic GT + perfect/empty
predictions against known expected mAP values. Gated via
`RUN_COCO_SMOKE=1` OR automatic when pycocotools + numpy are importable.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from evaluation.det_metrics import compute_coco_map

# Auto-enable the smoke tests when both libs are present so CI with the
# full stack exercises them without needing an extra env flag. Explicit
# opt-out via RUN_COCO_SMOKE=0 for debugging.
_LIBS_AVAILABLE = (
    importlib.util.find_spec("pycocotools") is not None
    and importlib.util.find_spec("numpy") is not None
)
_OPT_OUT = os.environ.get("RUN_COCO_SMOKE") == "0"
_RUN_SMOKE = _LIBS_AVAILABLE and not _OPT_OUT


# --- Schema validation ------------------------------------------------------


def test_rejects_missing_keys():
    preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}]  # no score
    with pytest.raises(ValueError, match="score"):
        compute_coco_map(preds, "/nonexistent.json")


def test_rejects_malformed_bbox():
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10], "score": 0.9}
    ]
    with pytest.raises(ValueError, match="bbox must be"):
        compute_coco_map(preds, "/nonexistent.json")


def test_raises_when_gt_file_missing(tmp_path: Path):
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    with pytest.raises(FileNotFoundError):
        compute_coco_map(preds, tmp_path / "missing.json")


# --- Lazy import path -------------------------------------------------------


def test_raises_importerror_when_pycocotools_missing(tmp_path: Path):
    if _LIBS_AVAILABLE:
        pytest.skip("pycocotools is installed; import-error path not exercised")

    # Need a valid GT file path so we don't hit FileNotFoundError first.
    gt_path = tmp_path / "gt.json"
    gt_path.write_text("{}", encoding="utf-8")

    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}
    ]
    with pytest.raises(ImportError, match="pycocotools"):
        compute_coco_map(preds, gt_path)


# --- Smoke: synthetic mAP ---------------------------------------------------


def _make_synthetic_gt(tmp_path: Path) -> Path:
    """Write a minimal COCO GT with one image, one category, one ann."""
    gt = {
        "images": [
            {"id": 1, "file_name": "dummy.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 50, 50],
                "area": 2500.0,
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "pedestrian", "supercategory": "object"}
        ],
    }
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt), encoding="utf-8")
    return path


@pytest.mark.skipif(not _RUN_SMOKE, reason="pycocotools/numpy not installed")
def test_perfect_prediction_yields_unit_map(tmp_path: Path):
    """Exact box match at score=1.0 → mAP == 1.0 across all IoU thresholds."""
    gt_path = _make_synthetic_gt(tmp_path)
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 50, 50],
            "score": 1.0,
        }
    ]
    result = compute_coco_map(preds, gt_path)
    assert result["mAP"] == pytest.approx(1.0)
    assert result["AP50"] == pytest.approx(1.0)
    assert result["AP75"] == pytest.approx(1.0)
    # Per-class view reports the same number for the single category.
    assert result["per_class"]["pedestrian"] == pytest.approx(1.0)


@pytest.mark.skipif(not _RUN_SMOKE, reason="pycocotools/numpy not installed")
def test_empty_predictions_yield_zero_map(tmp_path: Path):
    gt_path = _make_synthetic_gt(tmp_path)
    result = compute_coco_map([], gt_path)
    assert result["mAP"] == 0.0
    assert result["AP50"] == 0.0
    assert result["per_class"] == {}


@pytest.mark.skipif(not _RUN_SMOKE, reason="pycocotools/numpy not installed")
def test_offset_prediction_drops_ap_at_high_iou(tmp_path: Path):
    """Shifted box — passes AP50 but fails AP75."""
    gt_path = _make_synthetic_gt(tmp_path)
    # GT bbox is [100,100,50,50]; predict [115,115,50,50] → IoU ≈ 0.52, above
    # 0.5 threshold but below 0.75. AP50 stays 1.0, AP75 drops to 0.
    preds = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [115, 115, 50, 50],
            "score": 0.9,
        }
    ]
    result = compute_coco_map(preds, gt_path)
    assert result["AP50"] == pytest.approx(1.0)
    assert result["AP75"] == pytest.approx(0.0)
