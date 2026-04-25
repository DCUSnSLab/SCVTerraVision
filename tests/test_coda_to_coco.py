"""Unit tests for training.datasets.coda_to_coco.

Runs without any real CODa data — all fixtures are synthetic calibration
YAMLs + in-memory 3D bbox dicts + fake metadata JSONs in tmp_path. cv2 is
required (listed as opencv-python in pyproject); numpy + PyYAML are stdlib-
tier for this repo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from training.datasets.coda_to_coco import (
    CameraCalibration,
    ConversionStats,
    Taxonomy,
    annotations_from_frame,
    convert_coda_split,
    corners_to_xywh,
    get_3dbbox_corners,
    load_camera_calibration,
    load_taxonomy,
    project_corners_to_image,
)


REAL_TAXONOMY = Path(__file__).parent.parent / "configs" / "dataset" / "coda_taxonomy.yaml"


# --- Fixtures ---------------------------------------------------------------

def _identity_calibration(image_w: int = 1280, image_h: int = 720) -> CameraCalibration:
    """Camera frame == LiDAR frame. K simple, no distortion."""
    K = np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        image_width=image_w,
        image_height=image_h,
        K=K,
        dist=np.zeros(5, dtype=np.float64),
        T_lidar_to_cam=np.eye(4, dtype=np.float64),
    )


def _bbox(
    cx: float, cy: float, cz: float,
    h: float = 1.0, l: float = 1.0, w: float = 1.0,
    r: float = 0.0, p: float = 0.0, y: float = 0.0,
    classId: str = "Pedestrian", isOccluded: str = "None",
) -> dict[str, Any]:
    return {
        "classId": classId,
        "instanceId": f"{classId}:0",
        "cX": cx, "cY": cy, "cZ": cz,
        "h": h, "l": l, "w": w,
        "r": r, "p": p, "y": y,
        "isOccluded": isOccluded,
    }


def _write_calibration(
    coda_root: Path,
    seq: str,
    calib: CameraCalibration,
) -> None:
    calib_dir = coda_root / "calibrations" / seq
    calib_dir.mkdir(parents=True)
    intr = {
        "image_width": int(calib.image_width),
        "image_height": int(calib.image_height),
        "camera_name": "cam0",
        "camera_matrix": {
            "rows": 3, "cols": 3,
            "data": [float(x) for x in calib.K.flatten()],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1, "cols": int(calib.dist.size),
            "data": [float(x) for x in calib.dist.flatten()],
        },
    }
    extr = {
        "extrinsic_matrix": {
            "rows": 4, "cols": 4,
            "data": [float(x) for x in calib.T_lidar_to_cam.flatten()],
        }
    }
    (calib_dir / "calib_cam0_intrinsics.yaml").write_text(yaml.safe_dump(intr))
    (calib_dir / "calib_os1_to_cam0.yaml").write_text(yaml.safe_dump(extr))


def _write_metadata(
    coda_root: Path,
    seq: str,
    *,
    training: list[int] | None = None,
    validation: list[int] | None = None,
    testing: list[int] | None = None,
) -> None:
    meta_dir = coda_root / "metadata"
    meta_dir.mkdir(exist_ok=True)
    meta = {
        "trajectory": seq,
        "ObjectTracking": {
            "training": list(training or []),
            "validation": list(validation or []),
            "testing": list(testing or []),
        },
    }
    (meta_dir / f"{seq}.json").write_text(json.dumps(meta))


def _write_frame(
    coda_root: Path,
    seq: str,
    frame_idx: int,
    annotations: list[dict[str, Any]],
) -> None:
    bbox_dir = coda_root / "3d_bbox" / "os1" / seq
    bbox_dir.mkdir(parents=True, exist_ok=True)
    path = bbox_dir / f"3d_bbox_os1_{seq}_{frame_idx}.json"
    path.write_text(json.dumps({"3dannotations": annotations}))


# --- Tests ------------------------------------------------------------------

def test_get_corners_at_origin_no_rotation():
    corners = get_3dbbox_corners(_bbox(0.0, 0.0, 0.0, h=2.0, l=4.0, w=3.0))
    assert corners.shape == (8, 3)
    # Axis-aligned half-extents: l/2=2, w/2=1.5, h/2=1.
    assert np.isclose(corners[:, 0].min(), -2.0)
    assert np.isclose(corners[:, 0].max(), +2.0)
    assert np.isclose(corners[:, 1].min(), -1.5)
    assert np.isclose(corners[:, 1].max(), +1.5)
    assert np.isclose(corners[:, 2].min(), -1.0)
    assert np.isclose(corners[:, 2].max(), +1.0)


def test_projection_identity_calibration_matches_pinhole():
    """Box at (0,0,5) with unit dimensions projects to a predictable pixel square."""
    calib = _identity_calibration()
    corners = get_3dbbox_corners(_bbox(0.0, 0.0, 5.0, h=1.0, l=1.0, w=1.0))
    img_pts, in_front = project_corners_to_image(corners, calib)

    assert in_front.all()
    # All 8 corners should project. Expected axis-aligned bbox via pinhole:
    #   u = 500 * X/Z + 640, v = 500 * Y/Z + 360
    # Z spans [4.5, 5.5], X,Y span ±0.5.
    # Widest u extent is at Z=4.5 (nearest): ±55.555...
    # Same for v.
    xywh = corners_to_xywh(
        img_pts, in_front, calib.image_size,
        min_visible_corners=2, min_area=64.0,
    )
    assert xywh is not None
    x, y, w, h = xywh
    assert x == pytest.approx(584.444, abs=0.05)
    assert y == pytest.approx(304.444, abs=0.05)
    assert w == pytest.approx(111.111, abs=0.05)
    assert h == pytest.approx(111.111, abs=0.05)


def test_corners_to_xywh_drops_box_behind_camera():
    calib = _identity_calibration()
    corners = get_3dbbox_corners(_bbox(0.0, 0.0, -5.0, h=1.0, l=1.0, w=1.0))
    img_pts, in_front = project_corners_to_image(corners, calib)
    assert not in_front.any()
    assert corners_to_xywh(
        img_pts, in_front, calib.image_size,
        min_visible_corners=2, min_area=64.0,
    ) is None


def test_corners_to_xywh_drops_box_outside_fov():
    """Box 5m in front but 15m to the right — projects well off-screen."""
    calib = _identity_calibration()
    corners = get_3dbbox_corners(_bbox(15.0, 0.0, 5.0, h=1.0, l=1.0, w=1.0))
    img_pts, in_front = project_corners_to_image(corners, calib)
    # u ~= 500 * 15 / 5 + 640 = 2140, far outside image width=1280.
    assert in_front.all()
    result = corners_to_xywh(
        img_pts, in_front, calib.image_size,
        min_visible_corners=2, min_area=64.0,
    )
    assert result is None


def test_annotations_from_frame_applies_taxonomy_and_occlusion_filters():
    calib = _identity_calibration()
    taxonomy = Taxonomy(
        operational_classes=[(1, "pedestrian"), (2, "bicycle")],
        coda_to_operational={"Pedestrian": 1, "Bike": 2},
        coda_dropped=frozenset({"Chair"}),
    )
    stats = ConversionStats()

    frame = [
        # Valid pedestrian (kept).
        _bbox(0.0, 0.0, 5.0, h=1.7, l=0.5, w=0.5, classId="Pedestrian"),
        # Valid class but Heavy occlusion (dropped by occlusion filter).
        _bbox(0.0, 0.0, 6.0, classId="Pedestrian", isOccluded="Heavy"),
        # Explicitly dropped class (Chair → None, counted under taxonomy).
        _bbox(1.0, 0.0, 5.0, classId="Chair"),
        # Valid bike (kept).
        _bbox(0.0, 0.0, 8.0, h=1.2, l=1.8, w=0.5, classId="Bike"),
    ]

    out = annotations_from_frame(
        frame, calib, taxonomy,
        allow_occlusion=frozenset({"None", "Light", "Medium"}),
        min_visible_corners=2, min_area=64.0,
        stats=stats,
    )

    assert [cat for cat, _ in out] == [1, 2]
    assert stats.dropped_by_taxonomy == 1
    assert stats.dropped_by_occlusion == 1
    assert stats.dropped_by_projection == 0


def test_annotations_from_frame_raises_on_unknown_class():
    calib = _identity_calibration()
    taxonomy = Taxonomy(
        operational_classes=[(1, "pedestrian")],
        coda_to_operational={"Pedestrian": 1},
        coda_dropped=frozenset(),
    )
    with pytest.raises(ValueError, match="unknown CODa class"):
        annotations_from_frame(
            [_bbox(0.0, 0.0, 5.0, classId="Alien")],
            calib, taxonomy,
            allow_occlusion=frozenset({"None"}),
            min_visible_corners=2, min_area=64.0,
            stats=ConversionStats(),
        )


def test_load_taxonomy_real_yaml():
    """The shipped taxonomy YAML loads and has the expected shape."""
    tax = load_taxonomy(REAL_TAXONOMY)
    assert len(tax.operational_classes) == 16
    assert tax.operational_id_for("Pedestrian") == 1
    assert tax.operational_id_for("Bike") == 2
    assert tax.operational_id_for("Chair") is None  # dropped
    # Every entry in coda_to_operational maps to a declared id.
    declared_ids = {cid for cid, _ in tax.operational_classes}
    for coda_name, op_id in tax.coda_to_operational.items():
        assert op_id in declared_ids, f"{coda_name} → unknown id {op_id}"


def test_load_camera_calibration_roundtrip(tmp_path: Path):
    calib = _identity_calibration(image_w=1224, image_h=1024)
    coda_root = tmp_path
    _write_calibration(coda_root, "seq0", calib)

    loaded = load_camera_calibration(
        coda_root / "calibrations" / "seq0" / "calib_cam0_intrinsics.yaml",
        coda_root / "calibrations" / "seq0" / "calib_os1_to_cam0.yaml",
    )
    assert loaded.image_width == 1224
    assert loaded.image_height == 1024
    np.testing.assert_allclose(loaded.K, calib.K)
    np.testing.assert_allclose(loaded.dist, calib.dist)
    np.testing.assert_allclose(loaded.T_lidar_to_cam, calib.T_lidar_to_cam)


def test_convert_coda_split_empty_training(tmp_path: Path):
    calib = _identity_calibration()
    coda_root = tmp_path
    _write_calibration(coda_root, "seq0", calib)
    _write_metadata(coda_root, "seq0", training=[])

    taxonomy = load_taxonomy(REAL_TAXONOMY)
    coco, stats = convert_coda_split(coda_root, "training", taxonomy)

    assert coco["images"] == []
    assert coco["annotations"] == []
    assert len(coco["categories"]) == 16
    assert stats.images == 0
    assert stats.annotations == 0


def test_convert_coda_split_end_to_end(tmp_path: Path):
    calib = _identity_calibration()
    coda_root = tmp_path
    _write_calibration(coda_root, "seq0", calib)
    _write_metadata(coda_root, "seq0", training=[100, 101])

    # Frame 100: 1 valid Pedestrian + 1 dropped Chair + 1 Heavy-occluded Pedestrian.
    _write_frame(coda_root, "seq0", 100, [
        _bbox(0.0, 0.0, 5.0, h=1.7, l=0.5, w=0.5, classId="Pedestrian"),
        _bbox(1.0, 0.0, 5.0, classId="Chair"),
        _bbox(0.0, 0.0, 6.0, classId="Pedestrian", isOccluded="Heavy"),
    ])
    # Frame 101: 1 valid Bike.
    _write_frame(coda_root, "seq0", 101, [
        _bbox(0.0, 0.0, 8.0, h=1.2, l=1.8, w=0.5, classId="Bike",
              isOccluded="Light"),
    ])

    taxonomy = load_taxonomy(REAL_TAXONOMY)
    coco, stats = convert_coda_split(coda_root, "training", taxonomy)

    assert stats.images == 2
    assert stats.annotations == 2
    assert stats.dropped_by_taxonomy == 1
    assert stats.dropped_by_occlusion == 1

    # Two images with seq-prefixed file_name. CODa 2023 release uses .png
    # (older drafts assumed .jpg — kept this assertion as the canary so a
    # future release-format flip is caught here, not at training time).
    assert [img["file_name"] for img in coco["images"]] == [
        "seq0/2d_rect_cam0_seq0_100.png",
        "seq0/2d_rect_cam0_seq0_101.png",
    ]
    assert all(img["width"] == 1280 and img["height"] == 720 for img in coco["images"])

    # Category ids present: 1 (pedestrian) and 2 (bicycle).
    ann_cats = sorted(a["category_id"] for a in coco["annotations"])
    assert ann_cats == [1, 2]

    # Output COCO is compatible with CocoDetectionDataset (keys + id uniqueness).
    assert {"images", "annotations", "categories", "info", "licenses"} <= set(coco.keys())
    ann_ids = [a["id"] for a in coco["annotations"]]
    assert ann_ids == sorted(set(ann_ids)) and min(ann_ids) == 1


def test_convert_coda_split_respects_allow_occlusion(tmp_path: Path):
    """When allow_occlusion is restricted to {None}, Light annotations drop."""
    calib = _identity_calibration()
    coda_root = tmp_path
    _write_calibration(coda_root, "seq0", calib)
    _write_metadata(coda_root, "seq0", training=[0])
    _write_frame(coda_root, "seq0", 0, [
        _bbox(0.0, 0.0, 5.0, classId="Pedestrian", isOccluded="None"),
        _bbox(0.5, 0.0, 5.0, classId="Pedestrian", isOccluded="Light"),
    ])

    taxonomy = load_taxonomy(REAL_TAXONOMY)
    coco, stats = convert_coda_split(
        coda_root, "training", taxonomy,
        allow_occlusion=("None",),
    )

    assert stats.annotations == 1
    assert stats.dropped_by_occlusion == 1
