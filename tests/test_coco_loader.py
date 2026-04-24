"""Smoke tests for training.datasets.coco_loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.datasets.coco_loader import (
    CocoDetectionDataset,
    CocoIndex,
    load_coco_index,
)


def _write_synthetic_dataset(tmp_path: Path) -> tuple[Path, Path]:
    images_root = tmp_path / "images"
    images_root.mkdir()

    # Two 16x16 solid-color RGB PNGs so the loader has real bytes to read.
    for name, color in [("a.png", (255, 0, 0)), ("b.png", (0, 128, 255))]:
        arr = np.full((16, 16, 3), color, dtype=np.uint8)
        Image.fromarray(arr).save(images_root / name)

    coco = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "a.png", "width": 16, "height": 16},
            {"id": 2, "file_name": "b.png", "width": 16, "height": 16},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": [1.0, 2.0, 5.0, 6.0],
                "area": 30.0,
                "iscrowd": 0,
                "segmentation": [],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 7,
                "bbox": [0.0, 0.0, 4.0, 4.0],
                "area": 16.0,
                "iscrowd": 0,
                "segmentation": [],
            },
            # image_id=2 has no annotations on purpose.
        ],
        "categories": [
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 7, "name": "motorcycle", "supercategory": "vehicle"},
        ],
    }
    ann_path = tmp_path / "instances.json"
    ann_path.write_text(json.dumps(coco))
    return ann_path, images_root


def test_load_coco_index_groups_annotations_by_image(tmp_path: Path):
    ann_path, _ = _write_synthetic_dataset(tmp_path)
    idx = load_coco_index(ann_path)

    assert isinstance(idx, CocoIndex)
    assert idx.num_images == 2
    assert idx.num_categories == 2
    assert len(idx.annotations_by_image[1]) == 2
    assert idx.annotations_by_image.get(2, []) == []


def test_load_coco_index_rejects_missing_keys(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"images": [], "annotations": []}))  # no 'categories'

    import pytest

    with pytest.raises(ValueError):
        load_coco_index(bad)


def test_dataset_getitem_shapes_and_dtypes(tmp_path: Path):
    ann_path, images_root = _write_synthetic_dataset(tmp_path)
    ds = CocoDetectionDataset(ann_path, images_root)

    assert len(ds) == 2

    item0 = ds[0]
    assert item0["image_id"] == 1
    assert item0["file_name"] == "a.png"
    assert item0["image"].shape == (16, 16, 3)
    assert item0["image"].dtype == np.uint8
    assert item0["orig_size"] == (16, 16)
    assert item0["boxes"].shape == (2, 4)
    assert item0["boxes"].dtype == np.float32
    assert item0["labels"].tolist() == [3, 7]
    assert item0["labels"].dtype == np.int64
    assert item0["area"].tolist() == [30.0, 16.0]
    assert item0["iscrowd"].tolist() == [0, 0]


def test_dataset_item_with_no_annotations_returns_empty_arrays(tmp_path: Path):
    ann_path, images_root = _write_synthetic_dataset(tmp_path)
    ds = CocoDetectionDataset(ann_path, images_root)

    item1 = ds[1]  # image_id=2, no annotations
    assert item1["boxes"].shape == (0, 4)
    assert item1["labels"].shape == (0,)
    assert item1["area"].shape == (0,)
    assert item1["iscrowd"].shape == (0,)
