"""Segmentation metrics built on torchmetrics.

Provides per-class IoU, mIoU, and a binary "traversability" IoU that groups
unified classes into traversable vs non-traversable.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassJaccardIndex

from camera_perception.data.taxonomy import UnifiedTaxonomy

# Unified class ids that are considered traversable for the binary metric.
# Keep in sync with configs/taxonomy/traversability_v1.yaml.
TRAVERSABLE_CLASS_NAMES = ("traversable_smooth", "traversable_grass")


def build_iou_metric(num_classes: int, ignore_index: int = 255) -> MulticlassJaccardIndex:
    """Per-class IoU (no reduction). Use `.compute().mean()` for mIoU."""
    return MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=None,
    )


class BinaryTraversabilityIoU(Metric):
    """Binary IoU after collapsing unified classes into {traversable, other}.

    Pixels equal to `ignore_index` are excluded from both numerator and denominator.
    """

    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self, taxonomy: UnifiedTaxonomy, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        trav_ids = [taxonomy.name_to_id(n) for n in TRAVERSABLE_CLASS_NAMES]
        self.register_buffer(
            "trav_ids",
            torch.tensor(trav_ids, dtype=torch.long),
            persistent=False,
        )
        self.add_state("inter", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """preds: (B, H, W) integer class ids, target: (B, H, W)."""
        valid = target != self.ignore_index
        pred_trav = torch.isin(preds, self.trav_ids) & valid
        tgt_trav = torch.isin(target, self.trav_ids) & valid
        self.inter += (pred_trav & tgt_trav).sum()
        self.union += (pred_trav | tgt_trav).sum()

    def compute(self) -> torch.Tensor:
        if self.union == 0:
            return torch.tensor(float("nan"))
        return self.inter.float() / self.union.float()


def per_class_iou_dict(
    iou_per_class: torch.Tensor, taxonomy: UnifiedTaxonomy
) -> dict[str, float]:
    """Convert (K,) IoU tensor into {class_name: float}."""
    if iou_per_class.numel() != taxonomy.num_classes():
        raise ValueError(
            f"IoU tensor has {iou_per_class.numel()} entries but taxonomy has "
            f"{taxonomy.num_classes()} classes"
        )
    return {c.name: float(iou_per_class[c.id].item()) for c in taxonomy.classes}
