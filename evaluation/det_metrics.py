"""COCO mAP evaluation for DETR-style detection outputs (Phase 1-2b).

Wraps `pycocotools.cocoeval.COCOeval` and returns a plain-dict summary
that's easy to log (W&B, stdout) and compare across runs. Designed to run
on the CPU after inference — no torch dependency inside this module.

pycocotools is imported lazily inside `compute_coco_map()` so the module
imports cleanly when the dev env hasn't installed it (currently true in
this repo). Tests that construct prediction fixtures but don't actually
call `compute_coco_map()` stay green.

Prediction format accepted by `compute_coco_map(predictions, coco_gt_path)`:

    predictions = [
        {
            "image_id": int,         # COCO image id
            "category_id": int,      # matches coco_gt categories
            "bbox": [x, y, w, h],    # COCO xywh, pixel coords, float
            "score": float,          # detection confidence in [0, 1]
        },
        ...
    ]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 12-row COCOeval summary indices (COCOeval.stats layout — stable across
# pycocotools versions, verified against upstream source).
_COCOEVAL_STAT_NAMES = (
    "mAP",        # 0: AP @[ IoU=0.50:0.95 | area=all  | maxDets=100 ]
    "AP50",       # 1: AP @[ IoU=0.50      | area=all  | maxDets=100 ]
    "AP75",       # 2: AP @[ IoU=0.75      | area=all  | maxDets=100 ]
    "AP_small",   # 3: AP @[ IoU=0.50:0.95 | area=small]
    "AP_medium",  # 4: AP @[ IoU=0.50:0.95 | area=medium]
    "AP_large",   # 5: AP @[ IoU=0.50:0.95 | area=large]
    "AR_1",       # 6: AR @[ maxDets=1 ]
    "AR_10",      # 7: AR @[ maxDets=10 ]
    "AR_100",     # 8: AR @[ maxDets=100 ]
    "AR_small",   # 9
    "AR_medium",  # 10
    "AR_large",   # 11
)


def _validate_predictions(predictions: list[dict[str, Any]]) -> None:
    required = {"image_id", "category_id", "bbox", "score"}
    for i, p in enumerate(predictions):
        missing = required - set(p.keys())
        if missing:
            raise ValueError(
                f"prediction #{i}: missing keys {sorted(missing)}; "
                f"got {sorted(p.keys())}"
            )
        if not isinstance(p["bbox"], (list, tuple)) or len(p["bbox"]) != 4:
            raise ValueError(
                f"prediction #{i}: bbox must be [x, y, w, h]; got {p['bbox']!r}"
            )


def compute_coco_map(
    predictions: list[dict[str, Any]],
    coco_gt_path: str | Path,
    *,
    iou_type: str = "bbox",
    per_class: bool = True,
) -> dict[str, Any]:
    """Evaluate `predictions` against COCO ground truth at `coco_gt_path`.

    Returns a dict with:
        - all 12 COCOeval stats (mAP, AP50, AP75, AP_small, ..., AR_large)
        - `per_class` (optional): {category_name: AP@[.50:.95]} when
          `per_class=True` and the GT JSON has per-category data.

    Raises:
        ImportError  — pycocotools / numpy not installed (with install hint).
        ValueError   — predictions malformed.
        FileNotFoundError — coco_gt_path missing.
    """
    _validate_predictions(predictions)

    coco_gt_path = Path(coco_gt_path)
    if not coco_gt_path.exists():
        raise FileNotFoundError(f"COCO GT not found: {coco_gt_path}")

    try:
        import numpy as np
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError(
            "compute_coco_map() requires pycocotools and numpy. "
            "Install via `pip install pycocotools numpy` or the training extras."
        ) from e

    coco_gt = COCO(str(coco_gt_path))
    # Empty predictions → COCO.loadRes refuses to build a result; surface a
    # zero-AP summary rather than a crash.
    if not predictions:
        return {name: 0.0 for name in _COCOEVAL_STAT_NAMES} | (
            {"per_class": {}} if per_class else {}
        )

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    result: dict[str, Any] = {
        name: float(stats[i]) for i, name in enumerate(_COCOEVAL_STAT_NAMES)
    }

    if per_class:
        # precision has shape (T, R, K, A, M): IoU thresholds × recall ×
        # categories × area ranges × max detections. Average over T, R at
        # area=all (index 0), maxDets=100 (last index) for @[0.5:0.95]·all.
        precision = coco_eval.eval["precision"]  # type: ignore[index]
        cat_ids = coco_gt.getCatIds()
        categories = coco_gt.loadCats(cat_ids)
        per_class_ap = {}
        for k, cat in enumerate(categories):
            ap_k = precision[:, :, k, 0, -1]
            ap_k = ap_k[ap_k > -1]
            per_class_ap[cat["name"]] = (
                float(np.mean(ap_k)) if ap_k.size else 0.0
            )
        result["per_class"] = per_class_ap

    return result
