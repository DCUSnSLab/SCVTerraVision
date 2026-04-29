"""Convert CODa (UT Campus Object Dataset) raw 3D annotations to YOLO format.

Phase 1-3 — sister script to `coda_to_coco.py`. Same projection / filtering
pipeline (3D cuboid → 2D xywh on cam0), but emits Ultralytics YOLO format
(per-image `.txt` with `cls cx cy w h` normalized to [0, 1]) and a 91-class
taxonomy = COCO80 + CoDA campus-specific (80..90). See
`configs/dataset/coda_yolo_taxonomy.yaml` and
`docs/decisions/20260428_pivot-to-yolo26.md` for the rationale.

Why a direct raw-CODa converter (not COCO-JSON-then-YOLO):
    The 16-class operational COCO export collapses CoDA's vehicle subtypes
    (Car / Truck / Bus / Service Vehicle / Golf Cart) into a single id 5.
    YOLO uses subtype-aware ids — Car→2, Truck→7, Bus→5, plus campus
    additions service_vehicle=89 / golf_cart=90 — so the dispatch must
    happen at the raw-name level. Going via the COCO JSON would lose the
    subtype information.

Output layout:
    {output_dir}/
      images/
        {split}/{seq}_{frame}.png   # symlink to raw CODa png
      labels/
        {split}/{seq}_{frame}.txt   # YOLO labels, one box per line
      coda.yaml                      # auto-generated, ultralytics dataset YAML

The geometry / calibration / filtering helpers are imported from
`coda_to_coco` to avoid duplication — if the projection ever evolves, both
COCO and YOLO exports stay consistent.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from training.datasets.coda_to_coco import (
    DEFAULT_ALLOW_OCCLUSION,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_VISIBLE_CORNERS,
    CameraCalibration,
    ConversionStats,
    _BBOX_TEMPLATE,
    _IMAGE_TEMPLATE,
    corners_to_xywh,
    frames_for_split,
    get_3dbbox_corners,
    iter_sequences,
    load_camera_calibration,
    project_corners_to_image,
)


# === YOLO Taxonomy ============================================================


@dataclass(frozen=True)
class YoloTaxonomy:
    num_classes: int
    class_names: list[str]                  # index = YOLO id
    coda_raw_to_yolo: dict[str, int]
    coda_dropped: frozenset[str]
    yolo_to_operational: dict[int, int] = field(default_factory=dict)

    def yolo_id_for(self, raw_name: str) -> int | None:
        """Return YOLO id, or None if the raw CODa name is dropped.

        Raises ValueError if `raw_name` is unknown — same strict policy as
        coda_to_coco.Taxonomy.operational_id_for, so a future CODa class
        addition cannot silently leak into "background".
        """
        if raw_name in self.coda_raw_to_yolo:
            return self.coda_raw_to_yolo[raw_name]
        if raw_name in self.coda_dropped:
            return None
        raise ValueError(
            f"unknown CODa raw class {raw_name!r}: not in coda_raw_to_yolo "
            f"and not in coda_dropped. Update "
            f"configs/dataset/coda_yolo_taxonomy.yaml."
        )

    def names_dict(self) -> dict[int, str]:
        return {i: n for i, n in enumerate(self.class_names)}


def load_yolo_taxonomy(path: Path | str) -> YoloTaxonomy:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    nc = int(data["num_classes"])
    yolo_classes = sorted(
        ((int(c["id"]), str(c["name"])) for c in data["yolo_classes"]),
        key=lambda t: t[0],
    )
    declared_ids = [cid for cid, _ in yolo_classes]
    if declared_ids != list(range(nc)):
        raise ValueError(
            f"yolo_classes ids must be a contiguous 0..{nc - 1} block; got "
            f"{declared_ids}"
        )
    class_names = [name for _, name in yolo_classes]

    raw_to_yolo: dict[str, int] = {
        str(k): int(v) for k, v in data["coda_raw_to_yolo"].items()
    }
    dropped = frozenset(str(x) for x in data["coda_dropped"])

    overlap = dropped & set(raw_to_yolo.keys())
    if overlap:
        raise ValueError(
            f"coda_yolo_taxonomy: name(s) appear in both coda_raw_to_yolo "
            f"and coda_dropped: {sorted(overlap)}"
        )
    out_of_range = {n: i for n, i in raw_to_yolo.items() if not (0 <= i < nc)}
    if out_of_range:
        raise ValueError(
            f"coda_raw_to_yolo points to ids outside [0,{nc - 1}]: "
            f"{out_of_range}"
        )

    yolo_to_op_raw = data.get("yolo_to_operational") or {}
    yolo_to_op = {int(k): int(v) for k, v in yolo_to_op_raw.items()}

    return YoloTaxonomy(
        num_classes=nc,
        class_names=class_names,
        coda_raw_to_yolo=raw_to_yolo,
        coda_dropped=dropped,
        yolo_to_operational=yolo_to_op,
    )


# === Conversion ===============================================================


def xywh_to_yolo_norm(
    xywh: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> tuple[float, float, float, float]:
    """COCO xywh (pixels, top-left origin) → YOLO cxcywh normalized to [0,1].

    Coordinates are clipped to [0, 1] on output: `corners_to_xywh` already
    clips to image bounds, so this clip only guards against float drift on
    boxes that exactly touch a border.
    """
    x, y, w, h = xywh
    cx = (x + w / 2.0) / float(image_w)
    cy = (y + h / 2.0) / float(image_h)
    nw = w / float(image_w)
    nh = h / float(image_h)
    # Tiny clip — full-frame boxes occasionally land at 1.0+epsilon.
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return cx, cy, nw, nh


def yolo_annotations_from_frame(
    frame_annotations: list[dict[str, Any]],
    calib: CameraCalibration,
    taxonomy: YoloTaxonomy,
    *,
    allow_occlusion: frozenset[str],
    min_visible_corners: int,
    min_area: float,
    stats: ConversionStats,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Project + filter one frame's 3D boxes into (yolo_cls, normalized cxcywh)."""
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for entry in frame_annotations:
        classname = str(entry.get("classId", entry.get("className", "")))
        cls_id = taxonomy.yolo_id_for(classname)
        if cls_id is None:
            stats.dropped_by_taxonomy += 1
            continue

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

        cxcywh = xywh_to_yolo_norm(xywh, calib.image_width, calib.image_height)
        out.append((cls_id, cxcywh))
    return out


# === Filesystem layout ========================================================


def _split_dirname(split: str) -> str:
    """Map CODa split name to the YOLO directory name (`val` not `validation`)."""
    return {"training": "train", "validation": "val", "testing": "test"}[split]


def _stem_from_image(image_src: Path) -> str:
    """Filename stem unique across (sequence, frame).

    Uses the raw CODa image filename as-is (e.g. `2d_rect_cam0_0_400`) so the
    image symlink and the YOLO label txt always share the same stem —
    ultralytics pairs them by stem at load time. Synthesizing a different
    stem (e.g. `f"{seq}_{frame:06d}"`) breaks that pairing because the
    symlink would still carry the original filename.
    """
    return image_src.stem


def _link_image(src: Path, dst: Path, *, copy: bool) -> None:
    """Symlink (default) or copy an image into the YOLO tree."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil

        shutil.copy2(src, dst)
    else:
        # Use absolute target so the symlink keeps working when accessed from
        # any cwd (ultralytics resolves images relative to the dataset root).
        dst.symlink_to(src.resolve())


def _write_label_file(
    path: Path,
    rows: list[tuple[int, tuple[float, float, float, float]]],
) -> None:
    """Write a YOLO label txt — one box per line, empty file allowed (0 boxes)."""
    lines = [
        f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cls, (cx, cy, w, h) in rows
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def write_ultralytics_yaml(
    output_dir: Path,
    taxonomy: YoloTaxonomy,
    *,
    splits_present: Iterable[str],
) -> Path:
    """Emit `coda.yaml` consumed by ultralytics' YOLO trainer.

    Only the splits actually populated are referenced — running the
    converter with `--split training` only doesn't dangle a `val:` entry
    pointing at an empty dir.
    """
    yaml_path = output_dir / "coda.yaml"
    body: dict[str, Any] = {
        "path": str(output_dir.resolve()),
        "nc": taxonomy.num_classes,
        "names": taxonomy.names_dict(),
    }
    for split in splits_present:
        body[_split_dirname(split)] = f"images/{_split_dirname(split)}"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    return yaml_path


# === Driver ===================================================================


def convert_coda_split_to_yolo(
    coda_root: Path | str,
    split: str,
    taxonomy: YoloTaxonomy,
    output_dir: Path | str,
    *,
    sequences: list[str] | None = None,
    allow_occlusion: Iterable[str] = DEFAULT_ALLOW_OCCLUSION,
    min_visible_corners: int = DEFAULT_MIN_VISIBLE_CORNERS,
    min_area: float = DEFAULT_MIN_AREA,
    copy_images: bool = False,
) -> ConversionStats:
    """Convert one CODa split into the YOLO directory layout.

    Returns:
        ConversionStats with image / annotation counts and drop reasons.
    """
    coda_root = Path(coda_root)
    output_dir = Path(output_dir)
    allow_set = frozenset(allow_occlusion)
    stats = ConversionStats()

    split_name = _split_dirname(split)
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    sequence_names = (
        list(sequences) if sequences is not None else iter_sequences(coda_root)
    )

    for seq in sequence_names:
        calib = load_camera_calibration(
            coda_root / "calibrations" / seq / "calib_cam0_intrinsics.yaml",
            coda_root / "calibrations" / seq / "calib_os1_to_cam0.yaml",
        )
        metadata_path = coda_root / "metadata" / f"{seq}.json"
        frames = frames_for_split(metadata_path, split)

        for frame_idx in frames:
            image_src = (
                coda_root
                / "2d_rect"
                / "cam0"
                / seq
                / _IMAGE_TEMPLATE.format(sequence=seq, frame=frame_idx)
            )
            if not image_src.exists():
                # Frames listed in metadata but missing from disk are skipped
                # (matches coda_to_coco's behaviour of recording the frame in
                # `images` regardless — for YOLO we drop the pair entirely
                # since there is no image for the trainer to load).
                continue
            stem = _stem_from_image(image_src)
            _link_image(image_src, images_out / image_src.name, copy=copy_images)
            stats.images += 1

            bbox_path = (
                coda_root
                / "3d_bbox"
                / "os1"
                / seq
                / _BBOX_TEMPLATE.format(sequence=seq, frame=frame_idx)
            )
            label_rows: list[tuple[int, tuple[float, float, float, float]]] = []
            if bbox_path.exists():
                with bbox_path.open("r", encoding="utf-8") as f:
                    _frame_data = json.load(f)
                    frame_anns = _frame_data.get("3dbbox") or _frame_data.get(
                        "3dannotations", []
                    )
                label_rows = yolo_annotations_from_frame(
                    frame_anns,
                    calib,
                    taxonomy,
                    allow_occlusion=allow_set,
                    min_visible_corners=min_visible_corners,
                    min_area=min_area,
                    stats=stats,
                )
                stats.annotations += len(label_rows)

            # Match the .png stem 1:1 — ultralytics pairs `images/x.png` with
            # `labels/x.txt`. Empty txt = "image with no labels" (kept so the
            # frame still contributes to background statistics).
            _write_label_file(labels_out / f"{stem}.txt", label_rows)

    return stats


# === CLI ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert a CODa split (training|validation|testing) into "
            "YOLO format under data/processed/coda_yolo/. Reuses the "
            "coda_to_coco projection pipeline, emits 91-class labels."
        )
    )
    p.add_argument(
        "--coda-root",
        type=Path,
        default=Path(
            os.environ.get(
                "CODA_ROOT", "/home/marsberry/dataset/coda-devkit/data/CODa_full"
            )
        ),
    )
    p.add_argument(
        "--split",
        choices=("training", "validation", "testing"),
        required=True,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/coda_yolo"),
    )
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/dataset/coda_yolo_taxonomy.yaml"),
    )
    p.add_argument("--sequences", nargs="*", default=None)
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
    )
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy raw images instead of symlinking (default: symlink, ~45GB saved).",
    )
    p.add_argument(
        "--no-yaml",
        action="store_true",
        help="Skip writing the ultralytics coda.yaml (use when running for one "
        "split at a time and the other split's path would be dangling).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    taxonomy = load_yolo_taxonomy(args.taxonomy)
    stats = convert_coda_split_to_yolo(
        args.coda_root,
        args.split,
        taxonomy,
        args.output_dir,
        sequences=args.sequences,
        allow_occlusion=args.allow_occlusion,
        min_visible_corners=args.min_visible_corners,
        min_area=args.min_area,
        copy_images=args.copy_images,
    )

    splits_present: list[str] = []
    for s in ("training", "validation", "testing"):
        d = args.output_dir / "images" / _split_dirname(s)
        if d.exists() and any(d.iterdir()):
            splits_present.append(s)

    if not args.no_yaml and splits_present:
        yaml_path = write_ultralytics_yaml(
            args.output_dir, taxonomy, splits_present=splits_present
        )
        print(f"wrote {yaml_path}")

    print(
        f"split={args.split} images={stats.images} "
        f"annotations={stats.annotations} "
        f"dropped(taxonomy={stats.dropped_by_taxonomy}, "
        f"occlusion={stats.dropped_by_occlusion}, "
        f"projection={stats.dropped_by_projection})"
    )


if __name__ == "__main__":
    main()
