"""Tests for PredictionVizCallback.

Uses a stub LightningModule + stub trainer/logger so the test runs in a few
hundred milliseconds without touching DINOv2 weights or real datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from camera_perception.data.taxonomy import UnifiedTaxonomy
from camera_perception.training.callbacks import PredictionVizCallback


REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_CFG = REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"


# --- Stubs -----------------------------------------------------------------

class _StubExperiment:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add_image(self, tag: str, img, step: int, dataformats: str = "CHW") -> None:
        # Mirror TensorBoard SummaryWriter signature
        self.calls.append({"tag": tag, "shape": tuple(img.shape), "step": step, "fmt": dataformats})


class _StubLogger:
    def __init__(self) -> None:
        self.experiment = _StubExperiment()


class _StubLoggerNoAddImage:
    def __init__(self) -> None:
        self.experiment = object()  # has no add_image attr


@dataclass
class _StubDataModule:
    val: list

    @property
    def _val(self):
        return self.val


@dataclass
class _StubTrainer:
    logger: object
    datamodule: _StubDataModule
    current_epoch: int = 0


class _StubModule(nn.Module):
    """Pretends to be a LightningModule. Returns class-id `mod num_classes` per pixel."""

    def __init__(self, num_classes: int, taxonomy: UnifiedTaxonomy) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.taxonomy = taxonomy
        # registered param so .device works
        self._w = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self._w.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        # Construct logits where argmax is class 0 everywhere — deterministic
        logits = torch.zeros(b, self.num_classes, h, w, device=x.device)
        logits[:, 0] = 1.0
        return logits

    def eval(self):
        return super().eval()

    def train(self, mode: bool = True):
        return super().train(mode)


def _make_val_samples(n: int, h: int = 14, w: int = 14):
    return [
        {
            "image": torch.randn(3, h, w),
            "mask": torch.zeros(h, w, dtype=torch.long),
            "meta": {"index": i},
        }
        for i in range(n)
    ]


# --- Tests -----------------------------------------------------------------

def test_callback_logs_correct_number_of_images():
    tax = UnifiedTaxonomy.load(TAXONOMY_CFG)
    cb = PredictionVizCallback(num_samples=3, every_n_epochs=1, tag_prefix="viz")
    logger = _StubLogger()
    dm = _StubDataModule(val=_make_val_samples(5))
    trainer = _StubTrainer(logger=logger, datamodule=dm, current_epoch=0)
    module = _StubModule(num_classes=tax.num_classes(), taxonomy=tax)

    cb.on_validation_epoch_end(trainer, module)

    assert len(logger.experiment.calls) == 3
    for i, call in enumerate(logger.experiment.calls):
        assert call["tag"] == f"viz/sample_{i}"
        assert call["fmt"] == "HWC"
        assert call["step"] == 0
        # Tile is image|gt|pred concatenated horizontally → width tripled
        h, w, c = call["shape"]
        assert c == 3
        assert w == 14 * 3


def test_callback_respects_every_n_epochs():
    tax = UnifiedTaxonomy.load(TAXONOMY_CFG)
    cb = PredictionVizCallback(num_samples=2, every_n_epochs=2)
    logger = _StubLogger()
    dm = _StubDataModule(val=_make_val_samples(2))
    module = _StubModule(num_classes=tax.num_classes(), taxonomy=tax)

    # Epoch 0 → (0+1) % 2 = 1, skip
    cb.on_validation_epoch_end(_StubTrainer(logger, dm, current_epoch=0), module)
    assert len(logger.experiment.calls) == 0

    # Epoch 1 → (1+1) % 2 = 0, fire
    cb.on_validation_epoch_end(_StubTrainer(logger, dm, current_epoch=1), module)
    assert len(logger.experiment.calls) == 2


def test_callback_caps_at_dataset_size():
    tax = UnifiedTaxonomy.load(TAXONOMY_CFG)
    cb = PredictionVizCallback(num_samples=10, every_n_epochs=1)
    logger = _StubLogger()
    dm = _StubDataModule(val=_make_val_samples(3))
    module = _StubModule(num_classes=tax.num_classes(), taxonomy=tax)

    cb.on_validation_epoch_end(_StubTrainer(logger, dm), module)
    assert len(logger.experiment.calls) == 3


def test_callback_silent_for_non_image_logger():
    tax = UnifiedTaxonomy.load(TAXONOMY_CFG)
    cb = PredictionVizCallback(num_samples=2)
    logger = _StubLoggerNoAddImage()
    dm = _StubDataModule(val=_make_val_samples(2))
    module = _StubModule(num_classes=tax.num_classes(), taxonomy=tax)
    # Should not raise
    cb.on_validation_epoch_end(_StubTrainer(logger, dm), module)


def test_callback_handles_empty_val_dataset():
    tax = UnifiedTaxonomy.load(TAXONOMY_CFG)
    cb = PredictionVizCallback(num_samples=2)
    logger = _StubLogger()
    dm = _StubDataModule(val=[])
    module = _StubModule(num_classes=tax.num_classes(), taxonomy=tax)
    cb.on_validation_epoch_end(_StubTrainer(logger, dm), module)
    assert logger.experiment.calls == []


def test_callback_validates_constructor_args():
    with pytest.raises(ValueError):
        PredictionVizCallback(num_samples=0)
    with pytest.raises(ValueError):
        PredictionVizCallback(every_n_epochs=0)
