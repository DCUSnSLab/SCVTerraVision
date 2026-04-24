"""COCO-format detection dataset loader.

Phase 1-1 scope: a dependency-light index + item accessor that matches the
PyTorch `Dataset` protocol (`__len__` + `__getitem__`) without importing torch.
This keeps collection/tests green before the training environment is installed.

Per-item dict returned by `CocoDetectionDataset.__getitem__`:
    {
        "image_id": int,
        "file_name": str,
        "image": np.ndarray,       # H x W x 3, RGB, uint8
        "orig_size": (H, W),
        "boxes": np.ndarray,       # (N, 4), COCO xywh in pixels, float32
        "labels": np.ndarray,      # (N,) int64 category ids
        "area": np.ndarray,        # (N,) float32
        "iscrowd": np.ndarray,     # (N,) uint8
    }
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CocoIndex:
    """Immutable, torch-free view of a COCO annotation file."""

    images: list[dict[str, Any]]
    annotations_by_image: dict[int, list[dict[str, Any]]]
    categories: list[dict[str, Any]]

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_categories(self) -> int:
        return len(self.categories)


def load_coco_index(annotation_path: Path | str) -> CocoIndex:
    """Parse a COCO JSON into a `CocoIndex`. No torch/pycocotools required."""
    annotation_path = Path(annotation_path)
    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(
                f"{annotation_path}: missing required COCO key '{key}'"
            )

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    return CocoIndex(
        images=list(data["images"]),
        annotations_by_image=dict(anns_by_image),
        categories=list(data["categories"]),
    )


class CocoDetectionDataset:
    """Detection dataset over a COCO annotation file.

    Implements the PyTorch `Dataset` protocol (`__len__`, `__getitem__`) by
    duck-typing, so it plugs into `torch.utils.data.DataLoader` without forcing
    a torch import at module load time. Training code adds torch tensor
    conversion in its collate function.
    """

    def __init__(
        self,
        annotation_path: Path | str,
        images_root: Path | str,
        *,
        index: CocoIndex | None = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.images_root = Path(images_root)
        self.index = index if index is not None else load_coco_index(annotation_path)

    def __len__(self) -> int:
        return self.index.num_images

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_info = self.index.images[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        image_path = self.images_root / file_name
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            image = np.asarray(im, dtype=np.uint8)
        h, w = image.shape[:2]

        raw_anns = self.index.annotations_by_image.get(image_id, [])
        if raw_anns:
            boxes = np.asarray([a["bbox"] for a in raw_anns], dtype=np.float32)
            labels = np.asarray([a["category_id"] for a in raw_anns], dtype=np.int64)
            area = np.asarray(
                [a.get("area", a["bbox"][2] * a["bbox"][3]) for a in raw_anns],
                dtype=np.float32,
            )
            iscrowd = np.asarray(
                [a.get("iscrowd", 0) for a in raw_anns], dtype=np.uint8
            )
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            area = np.zeros((0,), dtype=np.float32)
            iscrowd = np.zeros((0,), dtype=np.uint8)

        return {
            "image_id": int(image_id),
            "file_name": file_name,
            "image": image,
            "orig_size": (h, w),
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
        }
