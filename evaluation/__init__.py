"""Evaluation metrics — COCO mAP (Phase 1-2b), MOT metrics (Phase 1-3)."""

from .det_metrics import compute_coco_map

__all__ = ["compute_coco_map"]
