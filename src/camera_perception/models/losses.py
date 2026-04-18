"""Loss functions for semantic segmentation.

All losses respect a single `ignore_index` (default 255) and accept logits
of shape (B, K, H, W) and integer targets of shape (B, H, W).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySegLoss(nn.Module):
    """Standard cross-entropy with optional class weights and ignore_index."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if class_weights is not None:
            if class_weights.shape != (num_classes,):
                raise ValueError(
                    f"class_weights must be shape ({num_classes},), got {tuple(class_weights.shape)}"
                )
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        else:
            self.class_weights = None  # type: ignore[assignment]
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class DiceSegLoss(nn.Module):
    """Multi-class soft Dice loss. ignore_index pixels are masked out before
    computing per-class numerator/denominator."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)                       # (B, K, H, W)
        valid = (target != self.ignore_index)                   # (B, H, W)
        target_clamped = torch.where(valid, target, torch.zeros_like(target))
        target_oh = F.one_hot(target_clamped.long(), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()       # (B, K, H, W)
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        target_oh = target_oh * valid_f

        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dims)
        cardinality = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """Weighted sum of CE + Dice."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.ce = CrossEntropySegLoss(
            num_classes, ignore_index, class_weights, label_smoothing
        )
        self.dice = DiceSegLoss(num_classes, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce_weight * self.ce(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target)
        return loss
