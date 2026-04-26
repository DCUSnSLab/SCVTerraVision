"""Convert BDD100K detection annotations to COCO JSON format.

BDD100K input shape (one element per image, list at top level):
    {
        "name": "b1c81faa-3df17267.jpg",
        "attributes": {...},
        "timestamp": 10000,
        "labels": [
            {
                "id": "0",
                "category": "traffic sign",
                "box2d": {"x1": 1000, "y1": 281, "x2": 1040, "y2": 326},
                ...
            },
            ...
        ]
    }

COCO output (single dict with images / annotations / categories).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

# BDD100K detection 10-class taxonomy, 1-indexed to match COCO convention.
BDD100K_DET_CATEGORIES: list[tuple[int, str]] = [
    (1, "pedestrian"),
    (2, "rider"),
    (3, "car"),
    (4, "truck"),
    (5, "bus"),
    (6, "train"),
    (7, "motorcycle"),
    (8, "bicycle"),
    (9, "traffic light"),
    (10, "traffic sign"),
]


def _build_categories() -> list[dict[str, Any]]:
    return [
        {"id": cid, "name": name, "supercategory": "bdd100k"}
        for cid, name in BDD100K_DET_CATEGORIES
    ]


def _category_lookup() -> dict[str, int]:
    return {name: cid for cid, name in BDD100K_DET_CATEGORIES}


def convert_bdd100k_to_coco(
    bdd_records: Iterable[dict[str, Any]],
    *,
    image_width: int = 1280,
    image_height: int = 720,
) -> dict[str, Any]:
    """Pure-in-memory conversion. `bdd_records` is an iterable of BDD100K image dicts.

    Labels without a `box2d` field (e.g. drivable-area polygons) are skipped.
    Labels whose `category` is not in the 10-class detection taxonomy are also skipped.
    """
    name_to_cid = _category_lookup()

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_id, record in enumerate(bdd_records, start=1):
        file_name = record["name"]
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_width,
                "height": image_height,
            }
        )

        for label in record.get("labels", []) or []:
            box2d = label.get("box2d")
            if box2d is None:
                continue
            category_name = label.get("category")
            category_id = name_to_cid.get(category_name)
            if category_id is None:
                continue

            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w == 0.0 or h == 0.0:
                continue

            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

    return {
        "info": {"description": "BDD100K detection converted to COCO format"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": _build_categories(),
    }


def convert_file(
    bdd_json_path: Path | str,
    output_path: Path | str,
    *,
    image_width: int = 1280,
    image_height: int = 720,
) -> dict[str, Any]:
    bdd_json_path = Path(bdd_json_path)
    output_path = Path(output_path)

    with bdd_json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(
            f"BDD100K JSON must be a list at the top level; got {type(records).__name__}"
        )

    coco = convert_bdd100k_to_coco(
        records, image_width=image_width, image_height=image_height
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f)
    return coco


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert BDD100K detection JSON to COCO JSON."
    )
    p.add_argument("--bdd-json", type=Path, required=True, help="Input BDD100K JSON")
    p.add_argument("--output", type=Path, required=True, help="Output COCO JSON")
    p.add_argument("--image-width", type=int, default=1280)
    p.add_argument("--image-height", type=int, default=720)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    coco = convert_file(
        args.bdd_json,
        args.output,
        image_width=args.image_width,
        image_height=args.image_height,
    )
    print(
        f"wrote {args.output} "
        f"(images={len(coco['images'])}, annotations={len(coco['annotations'])})"
    )


if __name__ == "__main__":
    main()
