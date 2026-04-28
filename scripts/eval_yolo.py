"""Evaluate a trained YOLO26 checkpoint with multiple comparison lenses.

Three evaluation modes (Phase 1-3 sub-phase gates use --mode coda16):

  --mode coda16
      Run inference, project the 91-class predictions back to CoDA's
      16-class operational space via yolo_to_operational, and run the
      same `evaluation.det_metrics.compute_coco_map` as Phase 1-2b.
      Produces the apples-to-apples comparison with the DINOv3+DETR
      baseline (mAP=0.623, AP50=0.925, AP_small=0.35).

  --mode all91
      Evaluate as 91-class on the same CODa val set. Predictions stay in
      YOLO ids; ground truth is the same CODa val converted to 91-class.
      For tail-class (CoDA-only ids 80..90) sanity checking.

  --mode coco80_regression
      Run inference on a held-out COCO80 mini-val (external annotations
      JSON) to monitor catastrophic forgetting. Uses the YOLO ids 0..79
      directly — no remapping. Compared run-over-run to the previous
      sub-phase's regression number.

Output JSON schema mirrors `evaluation/det_metrics.compute_coco_map`'s
return dict so Phase 1-2b and Phase 1-3 results can be diffed line-by-line.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

LOG = logging.getLogger("eval_yolo")


# === Taxonomy helpers =========================================================


def _load_yolo_to_operational(taxonomy_path: Path) -> dict[int, int]:
    """Read yolo_to_operational from coda_yolo_taxonomy.yaml."""
    with taxonomy_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data.get("yolo_to_operational") or {}
    return {int(k): int(v) for k, v in raw.items()}


# === Inference ================================================================


def _run_predictions_on_coco(
    model: Any,
    coco_gt: dict[str, Any],
    images_root: Path,
    *,
    keep_classes: set[int] | None,
    remap: dict[int, int] | None,
    device: Any,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[dict[str, Any]]:
    """Run YOLO on every image listed in `coco_gt` and collect predictions.

    Args:
        keep_classes: if set, drop predictions whose YOLO class id is not
            in this set (e.g. for --mode coda16, keep only YOLO ids that
            map back to a CoDA op id).
        remap: optional {YOLO id → COCO category_id} map applied after
            keep_classes filtering.

    Returns COCO-format predictions (image_id, category_id, bbox xywh, score).
    """
    predictions: list[dict[str, Any]] = []
    images = coco_gt.get("images", [])
    for img_meta in images:
        img_path = images_root / img_meta["file_name"]
        if not img_path.exists():
            LOG.warning("missing image: %s", img_path)
            continue
        results = model.predict(
            source=str(img_path),
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        if not results:
            continue
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        # `xyxy` and `cls` / `conf` are torch tensors on whatever device the
        # model ran on — `.cpu().tolist()` makes them JSON-serialisable
        # without pulling torch into this module's import space.
        xyxy = boxes.xyxy.cpu().tolist()
        cls_ids = [int(c) for c in boxes.cls.cpu().tolist()]
        scores = [float(s) for s in boxes.conf.cpu().tolist()]

        for (x1, y1, x2, y2), cls, score in zip(xyxy, cls_ids, scores):
            if keep_classes is not None and cls not in keep_classes:
                continue
            cat_id = remap[cls] if remap is not None else cls
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            predictions.append(
                {
                    "image_id": int(img_meta["id"]),
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
    return predictions


# === Modes ====================================================================


def _eval_coda16(
    model: Any,
    coda_val_json: Path,
    images_root: Path,
    yolo_to_operational: dict[int, int],
    *,
    device: Any,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    """Project YOLO predictions to CoDA 16 ops and score via compute_coco_map."""
    from evaluation.det_metrics import compute_coco_map

    with coda_val_json.open("r", encoding="utf-8") as f:
        coco_gt = json.load(f)

    predictions = _run_predictions_on_coco(
        model,
        coco_gt,
        images_root,
        keep_classes=set(yolo_to_operational.keys()),
        remap=yolo_to_operational,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    LOG.info("coda16 predictions: %d (after taxonomy filter)", len(predictions))
    return compute_coco_map(predictions, coda_val_json, per_class=True)


def _eval_all91(
    model: Any,
    coco_gt_json_91: Path,
    images_root: Path,
    *,
    device: Any,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    """Evaluate predictions in their native 91-class space.

    The ground truth JSON must be a 91-class export (use coda_to_coco-style
    converter that emits YOLO ids — not provided in 1-3a; placeholder until
    needed by the final-baseline writeup).
    """
    from evaluation.det_metrics import compute_coco_map

    with coco_gt_json_91.open("r", encoding="utf-8") as f:
        coco_gt = json.load(f)
    predictions = _run_predictions_on_coco(
        model,
        coco_gt,
        images_root,
        keep_classes=None,
        remap=None,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    LOG.info("all91 predictions: %d", len(predictions))
    return compute_coco_map(predictions, coco_gt_json_91, per_class=True)


def _eval_coco80_regression(
    model: Any,
    coco_gt_json: Path,
    images_root: Path,
    *,
    device: Any,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict[str, Any]:
    """COCO80 mini-val regression — keep only YOLO ids 0..79."""
    from evaluation.det_metrics import compute_coco_map

    with coco_gt_json.open("r", encoding="utf-8") as f:
        coco_gt = json.load(f)
    predictions = _run_predictions_on_coco(
        model,
        coco_gt,
        images_root,
        keep_classes=set(range(80)),
        remap=None,
        device=device,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
    )
    LOG.info("coco80_regression predictions: %d", len(predictions))
    return compute_coco_map(predictions, coco_gt_json, per_class=True)


# === CLI ======================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("coda16", "all91", "coco80_regression"),
        default="coda16",
    )
    p.add_argument(
        "--val-annotations",
        type=Path,
        required=True,
        help="COCO-format ground truth JSON for the chosen --mode.",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Directory the COCO 'file_name' field is resolved against.",
    )
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/dataset/coda_yolo_taxonomy.yaml"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default=None,
                   help="Pass through to ultralytics, e.g. 0 or '0,3' or 'cpu'.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001,
                   help="Score floor — 0.001 matches COCO eval convention.")
    p.add_argument("--iou", type=float, default=0.7,
                   help="NMS IoU; ultralytics default. YOLO26 is NMS-free at "
                        "inference but the predict path still applies a soft "
                        "filter that this knob controls.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise SystemExit(f"ultralytics not installed: {e}") from e

    LOG.info("loading checkpoint: %s", args.checkpoint)
    model = YOLO(str(args.checkpoint))

    if args.mode == "coda16":
        yolo_to_op = _load_yolo_to_operational(args.taxonomy)
        result = _eval_coda16(
            model,
            args.val_annotations,
            args.images_root,
            yolo_to_op,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
    elif args.mode == "all91":
        result = _eval_all91(
            model,
            args.val_annotations,
            args.images_root,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
    else:
        result = _eval_coco80_regression(
            model,
            args.val_annotations,
            args.images_root,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    LOG.info("wrote %s", args.output)

    # Echo headline metrics so the run log shows the gate-relevant numbers
    # without opening the JSON.
    headline = {k: result[k] for k in ("mAP", "AP50", "AP75", "AP_small",
                                        "AP_medium", "AP_large") if k in result}
    LOG.info("headline: %s", headline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
