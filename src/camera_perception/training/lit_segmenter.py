"""Lightning module wrapping DINOv2 backbone + segmentation head + loss + metrics."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn

from camera_perception.data.taxonomy import UnifiedTaxonomy
from camera_perception.models.backbones import DINOv2Backbone
from camera_perception.models.heads import DPTHead, LinearHead
from camera_perception.models.losses import CombinedSegLoss
from camera_perception.training.metrics import (
    BinaryTraversabilityIoU,
    build_iou_metric,
    per_class_iou_dict,
)

HEAD_REGISTRY = {
    "linear": LinearHead,
    "dpt": DPTHead,
}


def build_head(head_name: str, embed_dim: int, num_classes: int, **kwargs: Any) -> nn.Module:
    if head_name not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head_name!r}")
    return HEAD_REGISTRY[head_name](embed_dim=embed_dim, num_classes=num_classes, **kwargs)


class LitSegmenter(L.LightningModule):
    def __init__(
        self,
        taxonomy_config: str,
        backbone_variant: str = "small",
        backbone_freeze: bool = True,
        head: str = "linear",
        head_kwargs: dict[str, Any] | None = None,
        ignore_index: int = 255,
        class_weights: list[float] | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,
        label_smoothing: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.taxonomy = UnifiedTaxonomy.load(taxonomy_config)
        num_classes = self.taxonomy.num_classes()

        # head=linear uses only the last layer; dpt uses last 4
        out_layers = None if head == "dpt" else (-1,)  # type: ignore[assignment]
        # backbone wrapper handles tuple of int indices
        if out_layers == (-1,):
            from camera_perception.models.backbones.dinov2 import VARIANT_TO_DEPTH

            depth = VARIANT_TO_DEPTH[backbone_variant]
            out_layers = (depth - 1,)

        self.backbone = DINOv2Backbone(
            variant=backbone_variant,
            out_layers=out_layers,
            freeze=backbone_freeze,
        )
        self.head = build_head(
            head,
            embed_dim=self.backbone.embed_dim,
            num_classes=num_classes,
            **(head_kwargs or {}),
        )
        cw = (
            torch.tensor(class_weights, dtype=torch.float32)
            if class_weights is not None
            else None
        )
        self.criterion = CombinedSegLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            class_weights=cw,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            label_smoothing=label_smoothing,
        )

        # Metrics — separate train/val to keep state independent
        self.val_iou = build_iou_metric(num_classes, ignore_index)
        self.val_trav_iou = BinaryTraversabilityIoU(self.taxonomy, ignore_index)
        self.test_iou = build_iou_metric(num_classes, ignore_index)
        self.test_trav_iou = BinaryTraversabilityIoU(self.taxonomy, ignore_index)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        backbone_out = self.backbone(image)
        return self.head(backbone_out, out_hw=(h, w))

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        image = batch["image"]
        target = batch["mask"]
        logits = self(image)
        loss = self.criterion(logits, target)
        return loss, logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        loss, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        loss, logits = self._step(batch)
        preds = logits.argmax(dim=1)
        self.val_iou.update(preds, batch["mask"])
        self.val_trav_iou.update(preds, batch["mask"])
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        iou = self.val_iou.compute()
        miou = float(torch.nanmean(iou).item())
        self.log("val/mIoU", miou, prog_bar=True)
        for name, v in per_class_iou_dict(iou, self.taxonomy).items():
            self.log(f"val/iou/{name}", v)
        self.log("val/trav_IoU", self.val_trav_iou.compute(), prog_bar=True)
        self.val_iou.reset()
        self.val_trav_iou.reset()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        _, logits = self._step(batch)
        preds = logits.argmax(dim=1)
        self.test_iou.update(preds, batch["mask"])
        self.test_trav_iou.update(preds, batch["mask"])

    def on_test_epoch_end(self) -> None:
        iou = self.test_iou.compute()
        miou = float(torch.nanmean(iou).item())
        self.log("test/mIoU", miou)
        for name, v in per_class_iou_dict(iou, self.taxonomy).items():
            self.log(f"test/iou/{name}", v)
        self.log("test/trav_IoU", self.test_trav_iou.compute())
        self.test_iou.reset()
        self.test_trav_iou.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        # Train only parameters that require grad (head + optional unfrozen backbone).
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.max_steps:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return {"optimizer": optimizer}
