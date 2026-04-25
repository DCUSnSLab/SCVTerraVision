"""Convert CODa (UT Campus Object Dataset) 3D annotations to COCO 2D JSON.

CODa ships 3D cuboids in the os1 LiDAR frame with no native 2D boxes. This
module projects each cuboid's 8 corners onto the rectified cam0 image and
returns an axis-aligned COCO xywh bbox. The output is compatible with
Phase 1-1's `CocoDetectionDataset`.

Pipeline per 3D annotation:
    classId          → taxonomy (drop if on coda_dropped list)
    isOccluded       → kept if in allow_occlusion set
    (cX,cY,cZ,h,l,w,r,p,y) → 8 corners in os1 frame
    T_lidar_to_cam   → 8 corners in cam frame
    cv2.projectPoints + K, dist → 8 pixel coords (+ in-front mask)
    axis-aligned clip → (x, y, w, h), dropped if too small / too few corners

CODa filesystem layout assumed (UT AMRL, 2023 release):
    {coda_root}/2d_rect/cam0/{SEQ}/2d_rect_cam0_{SEQ}_{FRAME}.jpg
    {coda_root}/3d_bbox/os1/{SEQ}/3d_bbox_os1_{SEQ}_{FRAME}.json
    {coda_root}/calibrations/{SEQ}/calib_cam0_intrinsics.yaml
    {coda_root}/calibrations/{SEQ}/calib_os1_to_cam0.yaml
    {coda_root}/metadata/{SEQ}.json   # contains ObjectTracking.{training,validation,testing}

License notes for data produced by this converter: CC BY-NC-SA 4.0 + UT
Non-Commercial Agreement carry over. See
docs/decisions/20260422_coda-primary-dataset.md.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# CODa filename templates. Hardcoded to the UT AMRL 2023 release layout. If a
# future release renames files, update both templates AND the file_name we
# write into the COCO JSON so `images_root / file_name` stays resolvable.
_IMAGE_TEMPLATE = "2d_rect_cam0_{sequence}_{frame}.png"
_BBOX_TEMPLATE = "3d_bbox_os1_{sequence}_{frame}.json"

DEFAULT_ALLOW_OCCLUSION: tuple[str, ...] = ("None", "Light", "Medium")
DEFAULT_MIN_VISIBLE_CORNERS = 2
DEFAULT_MIN_AREA = 64.0


# === Taxonomy =================================================================

@dataclass(frozen=True)
class Taxonomy:
    operational_classes: list[tuple[int, str]]
    coda_to_operational: dict[str, int]
    coda_dropped: frozenset[str]
    bdd100k_to_operational: dict[str, int] = field(default_factory=dict)

    def operational_id_for(self, coda_name: str) -> int | None:
        """Return operational id, or None if the CODa class is on the drop list.

        Raises ValueError if `coda_name` is neither mapped nor explicitly
        dropped — forces the caller to update the taxonomy YAML rather than
        silently losing classes when CODa adds new labels.
        """
        if coda_name in self.coda_to_operational:
            return self.coda_to_operational[coda_name]
        if coda_name in self.coda_dropped:
            return None
        raise ValueError(
            f"unknown CODa class {coda_name!r}: neither in coda_to_operational "
            f"nor coda_dropped. Update configs/dataset/coda_taxonomy.yaml."
        )

    def categories_coco(self) -> list[dict[str, Any]]:
        return [
            {"id": cid, "name": name, "supercategory": "coda"}
            for cid, name in self.operational_classes
        ]


def load_taxonomy(path: Path | str) -> Taxonomy:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    classes = [(int(c["id"]), str(c["name"])) for c in data["operational_classes"]]
    classes.sort(key=lambda t: t[0])
    mapping = {str(k): int(v) for k, v in data["coda_to_operational"].items()}
    dropped = frozenset(str(x) for x in data["coda_dropped"])
    bdd = {
        str(k): int(v)
        for k, v in (data.get("bdd100k_to_operational") or {}).items()
    }

    overlap = dropped & set(mapping.keys())
    if overlap:
        raise ValueError(
            f"coda taxonomy: class(es) appear in both coda_to_operational and "
            f"coda_dropped: {sorted(overlap)}"
        )
    mapped_ids = {v for v in mapping.values()}
    declared_ids = {cid for cid, _ in classes}
    unknown_ids = mapped_ids - declared_ids
    if unknown_ids:
        raise ValueError(
            f"coda taxonomy: mapping points to undeclared operational id(s) "
            f"{sorted(unknown_ids)}; declared ids are {sorted(declared_ids)}"
        )

    return Taxonomy(
        operational_classes=classes,
        coda_to_operational=mapping,
        coda_dropped=dropped,
        bdd100k_to_operational=bdd,
    )


# === Calibration ==============================================================

@dataclass(frozen=True)
class CameraCalibration:
    image_width: int
    image_height: int
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # plumb_bob: (k1, k2, p1, p2, k3) or subset
    T_lidar_to_cam: np.ndarray  # (4, 4) rigid transform

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)


def _yaml_matrix(d: dict[str, Any], expected_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = int(d["rows"]), int(d["cols"])
    if (rows, cols) != expected_shape:
        raise ValueError(
            f"matrix shape {(rows, cols)} ≠ expected {expected_shape}"
        )
    flat = np.asarray(d["data"], dtype=np.float64)
    if flat.size != rows * cols:
        raise ValueError(
            f"matrix data length {flat.size} ≠ rows*cols {rows * cols}"
        )
    return flat.reshape(rows, cols)


def load_camera_calibration(
    intrinsics_yaml: Path | str,
    extrinsics_yaml: Path | str,
) -> CameraCalibration:
    with Path(intrinsics_yaml).open("r", encoding="utf-8") as f:
        intr = yaml.safe_load(f)
    with Path(extrinsics_yaml).open("r", encoding="utf-8") as f:
        extr = yaml.safe_load(f)

    K = _yaml_matrix(intr["camera_matrix"], (3, 3))
    dist = np.asarray(intr["distortion_coefficients"]["data"], dtype=np.float64)
    T = _yaml_matrix(extr["extrinsic_matrix"], (4, 4))

    return CameraCalibration(
        image_width=int(intr["image_width"]),
        image_height=int(intr["image_height"]),
        K=K,
        dist=dist,
        T_lidar_to_cam=T,
    )


# === Geometry =================================================================

# Box-local corner order: 4 bottom (z=-h/2) then 4 top (z=+h/2), CCW viewed
# from +z. Matches the CODa devkit convention in helpers/geometry.py.
_LOCAL_CORNER_SIGNS: np.ndarray = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float64,
)


def get_3dbbox_corners(bbox: dict[str, Any]) -> np.ndarray:
    """Return the 8 corners of a CODa 3D bbox in the os1 LiDAR frame.

    CODa fields per cuboid:
        cX, cY, cZ → center (m)
        h          → extent along z (height)
        l          → extent along x (length)
        w          → extent along y (width)
        r, p, y    → roll, pitch, yaw (rad); rotation = Rz(y) @ Ry(p) @ Rx(r)
    """
    cx, cy, cz = float(bbox["cX"]), float(bbox["cY"]), float(bbox["cZ"])
    h, length, width = float(bbox["h"]), float(bbox["l"]), float(bbox["w"])
    roll, pitch, yaw = float(bbox["r"]), float(bbox["p"]), float(bbox["y"])

    half = np.array([length / 2.0, width / 2.0, h / 2.0], dtype=np.float64)
    local = _LOCAL_CORNER_SIGNS * half  # (8, 3)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cyw, syw = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cyw, -syw, 0], [syw, cyw, 0], [0, 0, 1]], dtype=np.float64)
    R = Rz @ Ry @ Rx

    return (R @ local.T).T + np.array([cx, cy, cz])


def project_corners_to_image(
    corners_lidar: np.ndarray,
    calib: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Project os1-frame corners onto the cam0 image plane.

    Returns:
        image_points : (N, 2) float64. Values for points behind the camera
                       are not meaningful — always filter with `in_front`.
        in_front     : (N,) bool. True where cam-frame Z > 0.
    """
    import cv2

    n = corners_lidar.shape[0]
    hom = np.concatenate([corners_lidar, np.ones((n, 1))], axis=1)  # (N, 4)
    cam = (calib.T_lidar_to_cam @ hom.T).T[:, :3]
    in_front = cam[:, 2] > 1e-6

    img_pts, _ = cv2.projectPoints(
        cam.reshape(-1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        calib.K,
        calib.dist,
    )
    return img_pts.reshape(-1, 2).astype(np.float64), in_front


def corners_to_xywh(
    image_points: np.ndarray,
    in_front: np.ndarray,
    image_size: tuple[int, int],
    *,
    min_visible_corners: int,
    min_area: float,
) -> tuple[float, float, float, float] | None:
    """Reduce projected corners to an axis-aligned COCO xywh bbox.

    Returns None if:
      - zero corners are in front of the camera, or
      - fewer than `min_visible_corners` corners (front-facing AND inside the
        image rectangle) land in frame, or
      - the clipped bbox area is below `min_area`.
    """
    W, H = image_size
    front = image_points[in_front]
    if front.shape[0] == 0:
        return None

    xs, ys = front[:, 0], front[:, 1]
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if int(inside.sum()) < min_visible_corners:
        return None

    x_min = max(0.0, float(xs.min()))
    y_min = max(0.0, float(ys.min()))
    x_max = min(float(W - 1), float(xs.max()))
    y_max = min(float(H - 1), float(ys.max()))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0.0 or h <= 0.0 or w * h < min_area:
        return None

    return (x_min, y_min, w, h)


# === Conversion driver ========================================================

@dataclass
class ConversionStats:
    images: int = 0
    annotations: int = 0
    dropped_by_taxonomy: int = 0
    dropped_by_occlusion: int = 0
    dropped_by_projection: int = 0


def annotations_from_frame(
    frame_annotations: list[dict[str, Any]],
    calib: CameraCalibration,
    taxonomy: Taxonomy,
    *,
    allow_occlusion: frozenset[str],
    min_visible_corners: int,
    min_area: float,
    stats: ConversionStats,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Turn one frame's 3D annotations into (category_id, xywh) tuples."""
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for entry in frame_annotations:
        classname = str(entry.get("classId", entry.get("className", "")))
        cat_id = taxonomy.operational_id_for(classname)
        if cat_id is None:
            stats.dropped_by_taxonomy += 1
            continue

        # CODa 2023 nests occlusion inside labelAttributes.isOccluded; early
        # dumps had it flat at the top level. Check the nested path first and
        # fall back so both revisions work.
        label_attrs = entry.get("labelAttributes") or {}
        occl = str(label_attrs.get("isOccluded", entry.get("isOccluded", "Unknown")))
        if occl not in allow_occlusion:
            stats.dropped_by_occlusion += 1
            continue

        corners = get_3dbbox_corners(entry)
        img_pts, in_front = project_corners_to_image(corners, calib)
        xywh = corners_to_xywh(
            img_pts,
            in_front,
            image_size=calib.image_size,
            min_visible_corners=min_visible_corners,
            min_area=min_area,
        )
        if xywh is None:
            stats.dropped_by_projection += 1
            continue

        out.append((cat_id, xywh))
    return out


def iter_sequences(coda_root: Path | str) -> list[str]:
    """List sequence names from `{coda_root}/metadata/*.json`."""
    meta_dir = Path(coda_root) / "metadata"
    if not meta_dir.exists():
        raise FileNotFoundError(f"CODa metadata dir not found: {meta_dir}")
    return sorted(p.stem for p in meta_dir.glob("*.json"))


def frames_for_split(metadata_path: Path | str, split: str) -> list[int]:
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    ot = meta.get("ObjectTracking", {})
    if split not in ot:
        raise ValueError(
            f"{metadata_path}: ObjectTracking.{split} not found. "
            f"Available: {sorted(ot.keys())}"
        )
    entries = ot[split]
    frames: list[int] = []
    for entry in entries:
        # CODa 2023 ships these as 3d_bbox JSON paths
        # ("3d_bbox/os1/{seq}/3d_bbox_os1_{seq}_{frame}.json"), not raw ints.
        # Accept both forms so the converter keeps working if the release
        # format switches to plain ints.
        if isinstance(entry, int):
            frames.append(entry)
        else:
            stem = Path(str(entry)).stem  # 3d_bbox_os1_{seq}_{frame}
            frames.append(int(stem.rsplit("_", 1)[-1]))
    # CODa metadata lists are unordered — sort so dataset order is deterministic
    # across runs (matters for eval reproducibility).
    return sorted(set(frames))


def convert_coda_split(
    coda_root: Path | str,
    split: str,
    taxonomy: Taxonomy,
    *,
    sequences: list[str] | None = None,
    allow_occlusion: Iterable[str] = DEFAULT_ALLOW_OCCLUSION,
    min_visible_corners: int = DEFAULT_MIN_VISIBLE_CORNERS,
    min_area: float = DEFAULT_MIN_AREA,
) -> tuple[dict[str, Any], ConversionStats]:
    """Convert one CODa split into a COCO dict + per-conversion stats."""
    coda_root = Path(coda_root)
    allow_set = frozenset(allow_occlusion)
    stats = ConversionStats()

    sequence_names = (
        list(sequences) if sequences is not None else iter_sequences(coda_root)
    )

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 0
    annotation_id = 0

    for seq in sequence_names:
        calib = load_camera_calibration(
            coda_root / "calibrations" / seq / "calib_cam0_intrinsics.yaml",
            coda_root / "calibrations" / seq / "calib_os1_to_cam0.yaml",
        )
        metadata_path = coda_root / "metadata" / f"{seq}.json"
        frames = frames_for_split(metadata_path, split)

        for frame_idx in frames:
            image_id += 1
            file_name = f"{seq}/{_IMAGE_TEMPLATE.format(sequence=seq, frame=frame_idx)}"
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": calib.image_width,
                    "height": calib.image_height,
                    "sequence": seq,
                    "frame_index": int(frame_idx),
                }
            )
            stats.images += 1

            bbox_path = (
                coda_root
                / "3d_bbox"
                / "os1"
                / seq
                / _BBOX_TEMPLATE.format(sequence=seq, frame=frame_idx)
            )
            if not bbox_path.exists():
                continue
            with bbox_path.open("r", encoding="utf-8") as f:
                # CODa 2023 releases use key "3dbbox"; earlier revisions used
                # "3dannotations". Accept either so a future key rename (or a
                # backported dump) is not silently empty.
                _frame_data = json.load(f)
                frame_anns = _frame_data.get("3dbbox") or _frame_data.get(
                    "3dannotations", []
                )

            for cat_id, xywh in annotations_from_frame(
                frame_anns,
                calib,
                taxonomy,
                allow_occlusion=allow_set,
                min_visible_corners=min_visible_corners,
                min_area=min_area,
                stats=stats,
            ):
                annotation_id += 1
                x, y, w, h = xywh
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                stats.annotations += 1

    coco = {
        "info": {
            "description": (
                f"CODa ObjectTracking.{split} projected to COCO 2D via "
                "training.datasets.coda_to_coco"
            ),
            "split": split,
        },
        "licenses": [
            {
                "name": "CC BY-NC-SA 4.0 + UT Non-Commercial Agreement (Oct 2023)",
                "url": "https://amrl.cs.utexas.edu/coda/",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": taxonomy.categories_coco(),
    }
    return coco, stats


# === CLI ======================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a CODa split (training|validation|testing) into COCO "
            "JSON by projecting 3D cuboids onto cam0 rectified images."
        )
    )
    p.add_argument("--coda-root", type=Path, required=True)
    p.add_argument(
        "--split",
        choices=("training", "validation", "testing"),
        required=True,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/dataset/coda_taxonomy.yaml"),
    )
    p.add_argument(
        "--sequences",
        nargs="*",
        default=None,
        help="Restrict to these sequence names; default = all under metadata/",
    )
    p.add_argument(
        "--cameras",
        nargs="*",
        default=["cam0"],
        help="Currently only cam0 is supported.",
    )
    p.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    p.add_argument(
        "--min-visible-corners",
        type=int,
        default=DEFAULT_MIN_VISIBLE_CORNERS,
    )
    p.add_argument(
        "--allow-occlusion",
        nargs="*",
        default=list(DEFAULT_ALLOW_OCCLUSION),
        help="CODa isOccluded values to keep.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.cameras != ["cam0"]:
        raise NotImplementedError(
            f"Phase 1-1b exports cam0 only; requested {args.cameras}"
        )

    taxonomy = load_taxonomy(args.taxonomy)
    coco, stats = convert_coda_split(
        args.coda_root,
        args.split,
        taxonomy,
        sequences=args.sequences,
        allow_occlusion=args.allow_occlusion,
        min_visible_corners=args.min_visible_corners,
        min_area=args.min_area,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(
        f"wrote {args.output} | images={stats.images} "
        f"annotations={stats.annotations} "
        f"dropped(taxonomy={stats.dropped_by_taxonomy}, "
        f"occlusion={stats.dropped_by_occlusion}, "
        f"projection={stats.dropped_by_projection})"
    )


if __name__ == "__main__":
    main()
