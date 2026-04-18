"""Tests for dataset loaders using synthetic on-disk data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from camera_perception.data.datasets import RUGDDataset, Rellis3DDataset


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _make_rugd_tree(root: Path, n: int = 3) -> None:
    """Create a tiny RUGD-shaped directory with RGB-encoded labels."""
    images_dir = root / "images" / "creek"
    labels_dir = root / "labels" / "creek"
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    stems = []
    for i in range(n):
        stem = f"creek/creek_{i:05d}"
        img = (np.random.rand(64, 96, 3) * 255).astype(np.uint8)
        # Build a label with a few known RUGD colors
        lab = np.zeros((64, 96, 3), dtype=np.uint8)
        lab[:32, :48] = (0, 0, 255)     # sky
        lab[:32, 48:] = (0, 102, 0)     # grass
        lab[32:, :48] = (64, 64, 64)    # asphalt
        lab[32:, 48:] = (255, 255, 0)   # vehicle
        _write_png(images_dir / f"creek_{i:05d}.png", img)
        _write_png(labels_dir / f"creek_{i:05d}.png", lab)
        stems.append(stem)

    (splits_dir / "train.txt").write_text("\n".join(stems) + "\n")


def _make_rellis_tree(root: Path, n: int = 3) -> None:
    """Create a tiny RELLIS-3D-shaped directory with id-encoded labels."""
    seq = root / "00000"
    img_dir = seq / "pylon_camera_node"
    lab_dir = seq / "pylon_camera_node_label_id"
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(n):
        img = (np.random.rand(64, 96, 3) * 255).astype(np.uint8)
        # Native RELLIS ids: 3=grass, 7=sky, 17=person, 10=asphalt
        lab = np.zeros((64, 96), dtype=np.uint8)
        lab[:32, :48] = 7
        lab[:32, 48:] = 3
        lab[32:, :48] = 10
        lab[32:, 48:] = 17
        img_path = img_dir / f"frame_{i:05d}.jpg"
        lab_path = lab_dir / f"frame_{i:05d}.png"
        img_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(img_path, format="JPEG")
        _write_png(lab_path, lab)
        lines.append(
            f"{img_path.relative_to(root)} {lab_path.relative_to(root)}"
        )
    (splits_dir / "train.lst").write_text("\n".join(lines) + "\n")


@pytest.fixture
def rugd_root(tmp_path):
    root = tmp_path / "rugd"
    _make_rugd_tree(root)
    return root


@pytest.fixture
def rellis_root(tmp_path):
    root = tmp_path / "rellis3d"
    _make_rellis_tree(root)
    return root


def test_rugd_loads_and_returns_unified_mask(rugd_root, rugd_config, taxonomy_config):
    ds = RUGDDataset(
        dataset_config=rugd_config,
        taxonomy_config=taxonomy_config,
        split="train",
        transform=None,
        root_override=rugd_root,
    )
    assert len(ds) == 3

    sample = ds[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].shape == (3, 64, 96)
    assert sample["mask"].shape == (64, 96)
    assert sample["mask"].dtype == torch.long

    mask_np = sample["mask"].numpy()
    sky = ds.taxonomy.name_to_id("sky")
    grass = ds.taxonomy.name_to_id("traversable_grass")
    smooth = ds.taxonomy.name_to_id("traversable_smooth")
    dyn = ds.taxonomy.name_to_id("obstacle_dynamic")
    assert (mask_np[:32, :48] == sky).all()
    assert (mask_np[:32, 48:] == grass).all()
    assert (mask_np[32:, :48] == smooth).all()
    assert (mask_np[32:, 48:] == dyn).all()

    # image normalized to [0, 1]
    assert sample["image"].min() >= 0.0
    assert sample["image"].max() <= 1.0
    assert sample["meta"]["index"] == 0


def test_rellis_loads_and_returns_unified_mask(rellis_root, rellis3d_config, taxonomy_config):
    ds = Rellis3DDataset(
        dataset_config=rellis3d_config,
        taxonomy_config=taxonomy_config,
        split="train",
        transform=None,
        root_override=rellis_root,
    )
    assert len(ds) == 3

    sample = ds[1]
    mask_np = sample["mask"].numpy()
    sky = ds.taxonomy.name_to_id("sky")
    grass = ds.taxonomy.name_to_id("traversable_grass")
    smooth = ds.taxonomy.name_to_id("traversable_smooth")
    dyn = ds.taxonomy.name_to_id("obstacle_dynamic")
    assert (mask_np[:32, :48] == sky).all()
    assert (mask_np[:32, 48:] == grass).all()
    assert (mask_np[32:, :48] == smooth).all()
    assert (mask_np[32:, 48:] == dyn).all()


def test_rugd_with_train_transform(rugd_root, rugd_config, taxonomy_config):
    from camera_perception.data.transforms import build_train_transform

    tfm = build_train_transform(crop_h=56, crop_w=56)  # multiples of 14
    ds = RUGDDataset(
        dataset_config=rugd_config,
        taxonomy_config=taxonomy_config,
        split="train",
        transform=tfm,
        root_override=rugd_root,
    )
    sample = ds[0]
    assert sample["image"].shape == (3, 56, 56)
    assert sample["mask"].shape == (56, 56)
    assert sample["mask"].dtype == torch.long
