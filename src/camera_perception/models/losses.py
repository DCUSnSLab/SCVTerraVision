"""Loss functions for semantic segmentation.

All losses respect a single `ignore_index` (default 255) and accept logits
of shape (B, K, H, W) and integer targets of shape (B, H, W).

When training on mixed datasets with different class coverage, a per-sample
`present_classes` mask of shape (B, K) can be passed. Absent classes get
their logits suppressed to -inf before softmax, so (a) gradients for those
classes on this sample vanish and (b) the model cannot pick an absent class
as an argmax prediction. Targets are expected to never reference an absent
class — dataset remapping drops such labels to `ignore_index`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mask_logits(logits: torch.Tensor, present_classes: torch.Tensor | None) -> torch.Tensor:
    """Set logits of absent classes to -inf.

    Args:
        logits: (B, K, H, W)
        present_classes: (B, K) bool, True where class is labeled for that sample.

    Returns:
        Masked logits, same shape. If present_classes is None, returns logits
        unchanged.
    """
    if present_classes is None:
        return logits
    if present_classes.dtype != torch.bool:
        present_classes = present_classes.bool()
    # (B, K) -> (B, K, 1, 1) broadcastable over (H, W)
    mask = present_classes.unsqueeze(-1).unsqueeze(-1)
    return logits.masked_fill(~mask, float("-inf"))


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

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        present_classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = _mask_logits(logits, present_classes)
        return F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class DiceSegLoss(nn.Module):
    """Multi-class soft Dice loss. ignore_index pixels are masked out before
    computing per-class numerator/denominator. Absent classes (via
    `present_classes`) are excluded from the final mean so their (spurious)
    zeros don't dominate the loss."""

    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        present_classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = _mask_logits(logits, present_classes)
        probs = F.softmax(logits, dim=1)                       # (B, K, H, W)
        # After -inf masking, softmax for absent classes is exactly 0; no NaN.
        valid = (target != self.ignore_index)                   # (B, H, W)
        target_clamped = torch.where(valid, target, torch.zeros_like(target))
        target_oh = F.one_hot(target_clamped.long(), num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()       # (B, K, H, W)
        valid_f = valid.unsqueeze(1).float()
        probs = probs * valid_f
        target_oh = target_oh * valid_f

        # Per-class dice across the whole batch.
        dims = (0, 2, 3)
        intersection = (probs * target_oh).sum(dims)
        cardinality = probs.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)   # (K,)

        if present_classes is not None:
            # A class is "present in this batch" if any sample has it present.
            batch_present = present_classes.bool().any(dim=0)   # (K,)
            if batch_present.any():
                dice_mean = dice[batch_present].mean()
            else:
                dice_mean = dice.mean()  # degenerate batch; fall back
        else:
            dice_mean = dice.mean()
        return 1.0 - dice_mean


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

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        present_classes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.ce_weight * self.ce(logits, target, present_classes)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, target, present_classes)
        return loss
