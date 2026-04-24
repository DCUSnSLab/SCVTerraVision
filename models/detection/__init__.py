"""Detection heads — Phase 1-2b onwards.

Currently exports the DINOv3 + HF Deformable DETR adapter used for the
CODa baseline. mmdetection DINO-DETR was considered and rejected in
docs/decisions/20260424_detr-head-library.md.
"""

from .detr_head import DetrHeadConfig, DinoV3DeformableDetr

__all__ = ["DetrHeadConfig", "DinoV3DeformableDetr"]
