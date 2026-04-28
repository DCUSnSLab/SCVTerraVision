"""Sanity-check + visualize a YOLO-format dataset (Phase 1-3 conversion).

Run after `training/datasets/coda_to_yolo.py` to confirm the export is
ultralytics-loadable before kicking off a long training run. Three checks:

  1. Cross-check: every image in `images/{split}/` has a matching
     `labels/{split}/` txt (and vice versa). Orphans abort.
  2. Per-line label sanity: cls ∈ [0, nc), all coords ∈ [0, 1], width &
     height > 0. Any violation prints the offending file:line and aborts.
  3. Visualization: sample N images, render GT bboxes + class names, write
     to `outputs/verify_yolo/{split}/`. Eyeball this directory before
     trusting the dataset.

Also prints class-frequency stats (raw count + percentage) per split — used
during 1-3a to confirm the long tail (CoDA-only ids 80..90) actually has
non-trivial sample counts.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import yaml


def _load_names(yaml_path: Path) -> dict[int, str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    raw = data["names"]
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    return {i: str(n) for i, n in enumerate(raw)}


def _list_split(yolo_root: Path, split_dir: str) -> tuple[list[Path], list[Path]]:
    images = sorted((yolo_root / "images" / split_dir).glob("*"))
    labels = sorted((yolo_root / "labels" / split_dir).glob("*.txt"))
    return images, labels


def cross_check(images: list[Path], labels: list[Path]) -> list[str]:
    """Return list of error messages — empty list = pass."""
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    errors: list[str] = []
    for stem in sorted(img_stems - lbl_stems):
        errors.append(f"image without label: {stem}")
    for stem in sorted(lbl_stems - img_stems):
        errors.append(f"label without image: {stem}")
    return errors


def validate_labels(labels: list[Path], num_classes: int) -> list[str]:
    """Return list of malformed-line errors — empty list = pass."""
    errors: list[str] = []
    for path in labels:
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{path}:{lineno} expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    errors.append(f"{path}:{lineno} non-numeric fields: {line!r}")
                    continue
                if not (0 <= cls < num_classes):
                    errors.append(
                        f"{path}:{lineno} cls={cls} outside [0,{num_classes - 1}]"
                    )
                for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                    if not (0.0 <= val <= 1.0):
                        errors.append(
                            f"{path}:{lineno} {name}={val} outside [0,1]"
                        )
                if w <= 0 or h <= 0:
                    errors.append(f"{path}:{lineno} non-positive box w={w} h={h}")
    return errors


def class_distribution(labels: list[Path]) -> Counter:
    counter: Counter = Counter()
    for path in labels:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    counter[int(parts[0])] += 1
                except ValueError:
                    pass
    return counter


def render_samples(
    images: list[Path],
    labels_dir: Path,
    names: dict[int, str],
    out_dir: Path,
    *,
    n: int,
    seed: int = 0,
) -> int:
    """Draw GT bboxes on `n` sampled images, save under `out_dir`. Returns count."""
    try:
        import cv2  # type: ignore
    except ImportError:
        print("[warn] cv2 not available — skipping sample rendering")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = list(images)
    rng.shuffle(pool)
    pool = pool[:n]

    rendered = 0
    for img_path in pool:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            with lbl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(x) for x in parts[1:])
                    x1 = int((cx - bw / 2.0) * w)
                    y1 = int((cy - bh / 2.0) * h)
                    x2 = int((cx + bw / 2.0) * w)
                    y2 = int((cy + bh / 2.0) * h)
                    # Color-by-class via simple hash (BGR). Distinguishes
                    # adjacent classes well enough for eyeball check.
                    color = ((cls * 53) % 255, (cls * 97) % 255, (cls * 31) % 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = names.get(cls, str(cls))
                    cv2.putText(
                        img,
                        label,
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        cv2.imwrite(str(out_dir / img_path.name), img)
        rendered += 1
    return rendered


def verify_split(
    yolo_root: Path,
    split_dir: str,
    names: dict[int, str],
    *,
    sample_n: int,
    samples_root: Path,
) -> bool:
    """Run all checks on one split. Returns True if it passes, False otherwise."""
    print(f"\n=== {split_dir} ===")
    images, labels = _list_split(yolo_root, split_dir)
    print(f"images={len(images)} labels={len(labels)}")
    if not images:
        print(f"[skip] no images under images/{split_dir}/")
        return True

    errors = cross_check(images, labels)
    if errors:
        for e in errors[:10]:
            print(f"[err] {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    errors = validate_labels(labels, num_classes=len(names))
    if errors:
        for e in errors[:10]:
            print(f"[err] {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False

    dist = class_distribution(labels)
    total = sum(dist.values())
    print(f"total annotations={total}")
    if total:
        for cls in sorted(dist.keys()):
            n = dist[cls]
            pct = 100.0 * n / total
            print(f"  cls {cls:3d} {names.get(cls, '?'):>20s} {n:>7d} ({pct:5.2f}%)")

    if sample_n > 0:
        rendered = render_samples(
            images,
            yolo_root / "labels" / split_dir,
            names,
            samples_root / split_dir,
            n=sample_n,
        )
        print(f"rendered {rendered} sample(s) under {samples_root / split_dir}")

    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--yolo-root",
        type=Path,
        default=Path("data/processed/coda_yolo"),
    )
    p.add_argument(
        "--samples-root",
        type=Path,
        default=Path("outputs/verify_yolo"),
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=100,
        help="Sample this many images per split for visual verification.",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    yaml_path = args.yolo_root / "coda.yaml"
    if not yaml_path.exists():
        print(f"[err] {yaml_path} not found — run training/datasets/coda_to_yolo.py first")
        return 2
    names = _load_names(yaml_path)
    print(f"loaded {len(names)} class names from {yaml_path}")

    ok = True
    for split in args.splits:
        ok = verify_split(
            args.yolo_root,
            split,
            names,
            sample_n=args.sample_n,
            samples_root=args.samples_root,
        ) and ok

    print("\n" + ("[pass] all checks ok" if ok else "[FAIL] one or more checks failed"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
