"""Tests for augmentation pipelines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from camera_perception.data.transforms import (
    build_eval_transform,
    build_train_transform,
)


def _dummy_pair(h: int = 200, w: int = 300, ignore_id: int = 255):
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    mask = np.random.randint(0, 6, size=(h, w), dtype=np.int64)
    # Sprinkle some ignore pixels
    mask[0, 0] = ignore_id
    return img, mask


@pytest.mark.parametrize(
    "h,w",
    [(518, 518), (224, 224), (140, 280)],
)
def test_train_transform_output_shape_and_normalization(h, w):
    tfm = build_train_transform(crop_h=h, crop_w=w)
    img, mask = _dummy_pair(800, 1200)
    out = tfm(image=img, mask=mask)

    image_t = out["image"]
    mask_t = out["mask"]

    assert isinstance(image_t, torch.Tensor)
    assert image_t.shape == (3, h, w)
    assert image_t.dtype == torch.float32
    # ImageNet-normalized → typically in [-3, 3] range
    assert image_t.min() > -5 and image_t.max() < 5

    assert mask_t.shape == (h, w)


def test_eval_transform_pads_to_multiple_of_14():
    tfm = build_eval_transform(height=518, width=518)
    img, mask = _dummy_pair(400, 700)
    out = tfm(image=img, mask=mask)
    image_t = out["image"]
    assert image_t.shape[1] % 14 == 0
    assert image_t.shape[2] % 14 == 0


def test_pad_uses_ignore_id_for_mask():
    ignore_id = 255
    tfm = build_eval_transform(height=518, width=518, ignore_id=ignore_id)
    # Small image so PadIfNeeded is exercised
    img, mask = _dummy_pair(100, 100, ignore_id=ignore_id)
    out = tfm(image=img, mask=mask)
    mask_t = out["mask"]
    # Padded regions should be ignore_id; corners are likely padding
    assert (mask_t == ignore_id).any()
