"""Smoke tests for training.datasets.bdd100k_to_coco."""

from __future__ import annotations

import json
from pathlib import Path

from training.datasets.bdd100k_to_coco import (
    BDD100K_DET_CATEGORIES,
    convert_bdd100k_to_coco,
    convert_file,
)


def _sample_bdd_records() -> list[dict]:
    return [
        {
            "name": "img0.jpg",
            "attributes": {"weather": "clear"},
            "timestamp": 0,
            "labels": [
                {
                    "id": "0",
                    "category": "car",
                    "box2d": {"x1": 10, "y1": 20, "x2": 110, "y2": 220},
                },
                {
                    "id": "1",
                    "category": "pedestrian",
                    "box2d": {"x1": 200, "y1": 50, "x2": 240, "y2": 180},
                },
                # Should be dropped: no box2d.
                {"id": "2", "category": "car", "poly2d": []},
                # Should be dropped: category not in detection taxonomy.
                {
                    "id": "3",
                    "category": "lane",
                    "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                },
                # Should be dropped: zero-size box.
                {
                    "id": "4",
                    "category": "car",
                    "box2d": {"x1": 5, "y1": 5, "x2": 5, "y2": 5},
                },
            ],
        },
        {
            "name": "img1.jpg",
            "labels": [
                {
                    "id": "0",
                    "category": "traffic sign",
                    "box2d": {"x1": 100, "y1": 100, "x2": 150, "y2": 140},
                },
            ],
        },
        # Image with no labels at all.
        {"name": "img2.jpg", "labels": []},
    ]


def test_convert_in_memory_shapes_categories_and_bbox():
    records = _sample_bdd_records()
    coco = convert_bdd100k_to_coco(records, image_width=1280, image_height=720)

    assert set(coco.keys()) >= {"images", "annotations", "categories", "info"}
    assert [c["name"] for c in coco["categories"]] == [n for _, n in BDD100K_DET_CATEGORIES]
    assert [c["id"] for c in coco["categories"]] == [i for i, _ in BDD100K_DET_CATEGORIES]

    assert len(coco["images"]) == 3
    for img in coco["images"]:
        assert img["width"] == 1280
        assert img["height"] == 720
        assert {"id", "file_name", "width", "height"} <= set(img.keys())

    # 2 valid boxes from img0, 1 from img1, 0 from img2 → 3 total.
    assert len(coco["annotations"]) == 3

    # Annotation IDs are 1-indexed and unique.
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert ann_ids == [1, 2, 3]

    # First box: car (cid=3), xywh=[10,20,100,200], area=20000.
    first = coco["annotations"][0]
    assert first["image_id"] == 1
    assert first["category_id"] == 3
    assert first["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert first["area"] == 20000.0
    assert first["iscrowd"] == 0

    # Last box is from image 2, traffic sign (cid=10).
    last = coco["annotations"][-1]
    assert last["image_id"] == 2
    assert last["category_id"] == 10


def test_convert_file_roundtrips_to_disk(tmp_path: Path):
    records = _sample_bdd_records()
    bdd_path = tmp_path / "bdd.json"
    out_path = tmp_path / "out" / "coco.json"
    bdd_path.write_text(json.dumps(records))

    coco = convert_file(bdd_path, out_path)

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["images"] == coco["images"]
    assert on_disk["annotations"] == coco["annotations"]


def test_convert_rejects_non_list_top_level(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))

    import pytest

    with pytest.raises(ValueError):
        convert_file(bad, tmp_path / "out.json")
