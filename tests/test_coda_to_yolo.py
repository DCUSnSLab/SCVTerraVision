"""Unit tests for training.datasets.coda_to_yolo.

Goals (Phase 1-3a gate dependency):
  - taxonomy YAML round-trip (raw name → YOLO id), including vehicle dispatch
  - xywh → YOLO normalized cxcywh (with edge clipping)
  - per-frame conversion drops malformed / out-of-frame boxes the same way
    the COCO converter does (sanity that we share the same pipeline)
  - dataset YAML auto-emit lists only present splits

No real CODa data needed — all fixtures synthetic, mirroring
test_coda_to_coco.py's style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from training.datasets.coda_to_coco import (
    CameraCalibration,
    ConversionStats,
)
from training.datasets.coda_to_yolo import (
    YoloTaxonomy,
    load_yolo_taxonomy,
    write_ultralytics_yaml,
    xywh_to_yolo_norm,
    yolo_annotations_from_frame,
)


REAL_YOLO_TAXONOMY = (
    Path(__file__).parent.parent / "configs" / "dataset" / "coda_yolo_taxonomy.yaml"
)


# --- Fixtures ---------------------------------------------------------------

def _identity_calibration(image_w: int = 1280, image_h: int = 720) -> CameraCalibration:
    K = np.array([[500.0, 0.0, 640.0], [0.0, 500.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraCalibration(
        image_width=image_w,
        image_height=image_h,
        K=K,
        dist=np.zeros(5, dtype=np.float64),
        T_lidar_to_cam=np.eye(4, dtype=np.float64),
    )


def _bbox(classId: str, *, occluded: str = "None") -> dict[str, Any]:
    """Place a small box at z=10m, slightly offset right of optical axis."""
    return {
        "classId": classId,
        "instanceId": f"{classId}:0",
        "cX": 0.5, "cY": 0.0, "cZ": 10.0,
        "h": 1.0, "l": 1.0, "w": 1.0,
        "r": 0.0, "p": 0.0, "y": 0.0,
        "isOccluded": occluded,
    }


# --- Taxonomy ---------------------------------------------------------------


class TestYoloTaxonomyLoad:
    def test_real_taxonomy_loads(self) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        assert tax.num_classes == 91
        assert len(tax.class_names) == 91
        # Spot-check anchor classes
        assert tax.class_names[0] == "person"
        assert tax.class_names[2] == "car"
        assert tax.class_names[80] == "scooter"
        assert tax.class_names[89] == "service_vehicle"
        assert tax.class_names[90] == "golf_cart"

    def test_vehicle_subtypes_dropped(self) -> None:
        """D12 — vehicle 8 raw 클래스 모두 학습 라벨에서 제외 (pseudo 가 대체)."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        for raw in ("Car", "Bus", "Truck", "Pickup Truck", "Delivery Truck",
                    "Service Vehicle", "Utility Vehicle", "Golf Cart"):
            assert raw not in tax.coda_raw_to_yolo, raw
            assert tax.yolo_id_for(raw) is None, raw  # dropped

    def test_coco_overlap_classes_dropped(self) -> None:
        """D13 — COCO80 overlap 6 raw 클래스 학습 라벨에서 제외 (pseudo 대체)."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        for raw in ("Pedestrian", "Bike", "Motorcycle",
                    "Traffic Light", "Fire Hydrant", "Bench"):
            assert raw not in tax.coda_raw_to_yolo, raw
            assert tax.yolo_id_for(raw) is None, raw  # dropped

    def test_dropped_class_returns_none(self) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        assert tax.yolo_id_for("Canopy") is None
        assert tax.yolo_id_for("Door") is None

    def test_unknown_raw_name_raises(self) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        with pytest.raises(ValueError, match="unknown CODa raw class"):
            tax.yolo_id_for("RoboFido")

    def test_yolo_to_operational_consistency(self) -> None:
        """Every CoDA-mapped YOLO id should round-trip to an operational id."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        for raw_name, yolo_id in tax.coda_raw_to_yolo.items():
            assert yolo_id in tax.yolo_to_operational, (
                f"YOLO id {yolo_id} (from raw {raw_name!r}) is not in "
                f"yolo_to_operational"
            )

    def test_overlap_detection(self, tmp_path: Path) -> None:
        bad = {
            "version": 1,
            "num_classes": 2,
            "yolo_classes": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}],
            "coda_raw_to_yolo": {"X": 0},
            "coda_dropped": ["X"],  # overlaps coda_raw_to_yolo
            "yolo_to_operational": {},
        }
        path = tmp_path / "bad.yaml"
        with path.open("w") as f:
            yaml.safe_dump(bad, f)
        with pytest.raises(ValueError, match="appear in both"):
            load_yolo_taxonomy(path)

    def test_out_of_range_id_rejected(self, tmp_path: Path) -> None:
        bad = {
            "version": 1,
            "num_classes": 2,
            "yolo_classes": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}],
            "coda_raw_to_yolo": {"X": 5},  # 5 ∉ [0, 1]
            "coda_dropped": [],
            "yolo_to_operational": {},
        }
        path = tmp_path / "bad.yaml"
        with path.open("w") as f:
            yaml.safe_dump(bad, f)
        with pytest.raises(ValueError, match="outside"):
            load_yolo_taxonomy(path)


# --- Coordinate conversion --------------------------------------------------


class TestXywhToYoloNorm:
    def test_centered_full_box(self) -> None:
        cx, cy, w, h = xywh_to_yolo_norm((100.0, 100.0, 200.0, 100.0), 1000, 500)
        # box top-left (100, 100), w=200, h=100
        # → center (200, 150), normalize by (1000, 500)
        assert cx == pytest.approx(0.2)
        assert cy == pytest.approx(0.3)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)

    def test_full_frame(self) -> None:
        cx, cy, w, h = xywh_to_yolo_norm((0.0, 0.0, 1280.0, 720.0), 1280, 720)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    def test_clip_handles_float_drift(self) -> None:
        # Box that should mathematically be 1.0 but lands at 1.0+1e-9.
        cx, cy, w, h = xywh_to_yolo_norm((0.0, 0.0, 1280.001, 720.0), 1280, 720)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0


# --- Per-frame conversion ---------------------------------------------------


class TestYoloAnnotationsFromFrame:
    def test_scooter_emits_class_80(self) -> None:
        """CoDA-only 9 (id 80..88) 만 학습 라벨에 들어감. Scooter → 80."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        calib = _identity_calibration()
        stats = ConversionStats()
        rows = yolo_annotations_from_frame(
            [_bbox("Scooter")],
            calib,
            tax,
            allow_occlusion=frozenset({"None", "Light", "Medium"}),
            min_visible_corners=2,
            min_area=64.0,
            stats=stats,
        )
        assert len(rows) == 1
        cls_id, (cx, cy, w, h) = rows[0]
        assert cls_id == 80
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert w > 0.0 and h > 0.0

    def test_vehicle_and_overlap_classes_increment_taxonomy_drop(self) -> None:
        """D12+D13 — vehicle / COCO80 overlap 라벨은 taxonomy drop 처리."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        calib = _identity_calibration()
        stats = ConversionStats()
        anns = [
            _bbox("Car"),
            _bbox("Bus"),
            _bbox("Truck"),
            _bbox("Service Vehicle"),
            _bbox("Golf Cart"),
            _bbox("Pedestrian"),
            _bbox("Bench"),
        ]
        rows = yolo_annotations_from_frame(
            anns,
            calib,
            tax,
            allow_occlusion=frozenset({"None"}),
            min_visible_corners=2,
            min_area=64.0,
            stats=stats,
        )
        assert rows == []
        assert stats.dropped_by_taxonomy == 7
        assert stats.dropped_by_occlusion == 0

    def test_dropped_class_increments_taxonomy_stat(self) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        calib = _identity_calibration()
        stats = ConversionStats()
        rows = yolo_annotations_from_frame(
            [_bbox("Door")],  # in coda_dropped
            calib,
            tax,
            allow_occlusion=frozenset({"None"}),
            min_visible_corners=2,
            min_area=64.0,
            stats=stats,
        )
        assert rows == []
        assert stats.dropped_by_taxonomy == 1
        assert stats.dropped_by_occlusion == 0

    def test_occlusion_filter(self) -> None:
        """occlusion 필터 — CoDA-only 클래스(Tree)에 적용해 검증."""
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        calib = _identity_calibration()
        stats = ConversionStats()
        rows = yolo_annotations_from_frame(
            [_bbox("Tree", occluded="Heavy")],
            calib,
            tax,
            allow_occlusion=frozenset({"None"}),
            min_visible_corners=2,
            min_area=64.0,
            stats=stats,
        )
        assert rows == []
        assert stats.dropped_by_occlusion == 1


# --- Dataset YAML emission --------------------------------------------------


class TestUltralyticsYamlEmission:
    def test_only_present_splits_listed(self, tmp_path: Path) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        out = tmp_path / "coda_yolo"
        out.mkdir()

        # Only training present
        path = write_ultralytics_yaml(out, tax, splits_present=["training"])
        with path.open("r") as f:
            data = yaml.safe_load(f)
        assert "train" in data
        assert "val" not in data
        assert "test" not in data
        assert data["nc"] == 91
        assert data["names"][0] == "person"

    def test_train_and_val_listed(self, tmp_path: Path) -> None:
        tax = load_yolo_taxonomy(REAL_YOLO_TAXONOMY)
        out = tmp_path / "coda_yolo"
        out.mkdir()
        path = write_ultralytics_yaml(
            out, tax, splits_present=["training", "validation"]
        )
        with path.open("r") as f:
            data = yaml.safe_load(f)
        assert data["train"] == "images/train"
        assert data["val"] == "images/val"
