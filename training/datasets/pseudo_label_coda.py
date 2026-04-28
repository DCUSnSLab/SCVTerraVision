"""Generate pseudo-labels on CoDA images using a pretrained YOLO checkpoint.

Phase 1-3a 후속 — Catastrophic forgetting + bbox 정밀도 해결 (D11~D17).
사전학습 yolo26n.pt 의 시각적 박스를 학습 라벨로 사용해서:
  1. CoDA 이미지에 라벨 안 달린 COCO80 객체에 학습 신호 부여 (forgetting 방지)
  2. CoDA 의 LiDAR 3D→2D 투영 박스 (시각적으로 약간 부정확) 를 yolo 의
     시각적 박스로 대체 — bbox 정밀도 회복

본 스크립트는 inference만 수행. CoDA 라벨과의 merge 는
`merge_pseudo_with_coda.py` 가 담당.

출력 형식 (ultralytics YOLO format, cls 0..79):
    {output_dir}/labels/{train,val}/<image_stem>.txt
    각 줄: cls cx_n cy_n w_n h_n

CLI 예:
    python -m training.datasets.pseudo_label_coda \\
        --weights yolo26n.pt \\
        --images data/processed/coda_yolo/images \\
        --output data/processed/coda_yolo_pseudo \\
        --conf 0.25 --imgsz 1024 --device 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOG = logging.getLogger("pseudo_label_coda")


def _write_label_file(
    out_path: Path,
    cls_ids: list[int],
    xywhn: list[tuple[float, float, float, float]],
) -> None:
    """Write a YOLO label txt — one box per line, empty file allowed (0 boxes)."""
    lines = [
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for c, (cx, cy, w, h) in zip(cls_ids, xywhn)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def generate_pseudo_labels(
    weights: str,
    images_dir: Path,
    output_dir: Path,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    device: str | int | None,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, int]:
    """Run inference on each split's images and emit YOLO-format pseudo labels.

    Args:
        images_dir: parent dir, expects `images_dir/{train,val}/*.png`.
        output_dir: parent dir, writes `output_dir/labels/{train,val}/*.txt`.

    Returns: {split: image_count} for the report.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as e:
        raise SystemExit(f"ultralytics not installed: {e}") from e

    LOG.info("loading weights: %s", weights)
    model = YOLO(weights)

    counts: dict[str, int] = {}
    for split in splits:
        split_imgs = images_dir / split
        if not split_imgs.exists():
            LOG.warning("split dir missing, skipping: %s", split_imgs)
            counts[split] = 0
            continue
        labels_out = output_dir / "labels" / split
        labels_out.mkdir(parents=True, exist_ok=True)

        LOG.info(
            "inference split=%s imgsz=%d conf=%.2f iou=%.2f device=%s",
            split, imgsz, conf, iou, device,
        )
        # `stream=True` yields one Results per image — keeps memory bounded
        # for the 19,511-image train split.
        n = 0
        for result in model.predict(
            source=str(split_imgs),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
        ):
            img_path = Path(result.path)
            stem = img_path.stem
            cls_ids: list[int] = []
            xywhn: list[tuple[float, float, float, float]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                cls_tensor = result.boxes.cls.cpu()
                xywhn_tensor = result.boxes.xywhn.cpu()
                for i in range(len(cls_tensor)):
                    cls_ids.append(int(cls_tensor[i].item()))
                    cx, cy, w, h = xywhn_tensor[i].tolist()
                    xywhn.append((float(cx), float(cy), float(w), float(h)))
            _write_label_file(labels_out / f"{stem}.txt", cls_ids, xywhn)
            n += 1
            if n % 2000 == 0:
                LOG.info("  split=%s progress=%d", split, n)
        counts[split] = n
        LOG.info("split=%s done: %d images", split, n)
    return counts


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--weights",
        default="yolo26n.pt",
        help="Pretrained YOLO weights (auto-downloaded by ultralytics on first use).",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("data/processed/coda_yolo/images"),
        help="Parent dir holding {train,val}/ image dirs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/coda_yolo_pseudo"),
        help="Output dir. Pseudo labels go to {output}/labels/{train,val}/.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument(
        "--imgsz", type=int, default=1024,
        help="Inference image size. 1024 catches small objects better than 640.",
    )
    p.add_argument(
        "--device", default=0,
        help="Pass-through to ultralytics: int, '0' / '0,1', or 'cpu'.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        choices=["train", "val", "test"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    counts = generate_pseudo_labels(
        weights=args.weights,
        images_dir=args.images,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        splits=tuple(args.splits),
    )
    LOG.info("pseudo-label generation complete: %s", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
