"""Merge pseudo labels (yolo26n inference, cls 0..79) with CoDA-only 9 labels
(scooter/tree/pole/sign/bollard/cone/barrier/bike_rack/trash_can, cls 80..88).

Phase 1-3a 후속 (D11~D17, 2026-04-29). 이전 단계에서 만들어진 두 라벨 source
를 한 디렉토리(`output_dir`)에 합쳐서 ultralytics 학습 데이터셋으로 만든다.

Per image 라벨 합치기 규칙:
  - Pseudo (yolo26n 추론) 의 모든 박스 (cls 0..79) → keep
  - CoDA v1 라벨에서 cls ∈ {80..88} (CoDA-only 9) → keep
  - CoDA v1 의 다른 cls ({0, 1, 3, 5, 7, 9, 10, 13, 89, 90}) → 폐기
    (D12 vehicle dispatch 폐기, D13 COCO80 overlap 폐기)
  - dedup 불필요 — pseudo (0..79) 와 CoDA-only (80..88) 는 ID 공간이 분리됨

이미지는 새로 복사하지 않고 v1 디렉토리를 `output_dir/images` 로 심볼릭 링크
(45GB 데이터 두 번 들고 다닐 이유 없음).

`coda.yaml` (ultralytics dataset YAML) 도 자동 생성. taxonomy YAML 의 names
91개를 그대로 keep — 89/90 (service_vehicle/golf_cart) 은 학습 신호 0 으로
deprecated 상태가 되지만 head shape 는 유지.

CLI 예:
    python -m training.datasets.merge_pseudo_with_coda \\
        --pseudo data/processed/coda_yolo_pseudo \\
        --coda data/processed/coda_yolo \\
        --output data/processed/coda_yolo_v2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

LOG = logging.getLogger("merge_pseudo_with_coda")

# CoDA-only 9 클래스가 차지하는 YOLO id (D14 적용 후 keep 대상).
CODA_ONLY_KEEP: frozenset[int] = frozenset(range(80, 89))


def _read_label_file(path: Path) -> list[tuple[int, str]]:
    """Return [(cls_int, full_line_str), ...]. Empty / missing → []."""
    if not path.exists():
        return []
    out: list[tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                cls = int(stripped.split()[0])
            except (ValueError, IndexError):
                continue
            out.append((cls, stripped))
    return out


def _write_label_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def merge_split(
    pseudo_labels_dir: Path,
    coda_labels_dir: Path,
    output_labels_dir: Path,
) -> dict[str, int]:
    """Merge one split. Returns counters for the report."""
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    counters = {
        "images": 0,
        "pseudo_boxes": 0,
        "coda_only_boxes_kept": 0,
        "coda_boxes_dropped": 0,
        "labels_written": 0,
    }

    # Iterate over the union of stems present in either source.
    stems: set[str] = set()
    if pseudo_labels_dir.exists():
        stems.update(p.stem for p in pseudo_labels_dir.glob("*.txt"))
    if coda_labels_dir.exists():
        stems.update(p.stem for p in coda_labels_dir.glob("*.txt"))

    for stem in sorted(stems):
        pseudo_lines = _read_label_file(pseudo_labels_dir / f"{stem}.txt")
        coda_lines = _read_label_file(coda_labels_dir / f"{stem}.txt")

        merged: list[str] = []
        for cls, line in pseudo_lines:
            merged.append(line)
            counters["pseudo_boxes"] += 1
        for cls, line in coda_lines:
            if cls in CODA_ONLY_KEEP:
                merged.append(line)
                counters["coda_only_boxes_kept"] += 1
            else:
                counters["coda_boxes_dropped"] += 1

        _write_label_file(output_labels_dir / f"{stem}.txt", merged)
        counters["images"] += 1
        if merged:
            counters["labels_written"] += 1

    return counters


def link_images(coda_root: Path, output_root: Path) -> None:
    """Make `output_root/images` a symlink → `coda_root/images`.

    Cheaper than re-symlinking each image; keeps stem alignment automatic.
    Idempotent — replaces an existing symlink in place.
    """
    src = (coda_root / "images").resolve()
    dst = output_root / "images"
    if not src.exists():
        raise FileNotFoundError(f"CoDA images dir missing: {src}")
    output_root.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            # 디렉토리가 이미 있으면 재생성 안 함 — 사용자가 의도적으로 만든 것일 수 있음
            LOG.warning("output images dir already exists (not a symlink), skipping link: %s", dst)
            return
    dst.symlink_to(src)
    LOG.info("symlinked images: %s -> %s", dst, src)


def write_coda_yaml(
    output_root: Path,
    taxonomy_path: Path,
    splits_present: list[str],
) -> Path:
    """Auto-generate ultralytics dataset YAML for v2."""
    with taxonomy_path.open("r", encoding="utf-8") as f:
        tax = yaml.safe_load(f)
    nc = int(tax["num_classes"])
    yolo_classes = sorted(
        ((int(c["id"]), str(c["name"])) for c in tax["yolo_classes"]),
        key=lambda t: t[0],
    )
    names_dict = {i: name for i, (idx, name) in enumerate(yolo_classes) if idx == i}

    body: dict = {
        "path": str(output_root.resolve()),
        "nc": nc,
        "names": names_dict,
    }
    for split in splits_present:
        body[split] = f"images/{split}"

    yaml_path = output_root / "coda.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(body, f, sort_keys=False)
    return yaml_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--pseudo",
        type=Path,
        default=Path("data/processed/coda_yolo_pseudo"),
        help="Pseudo-label root, expects {pseudo}/labels/{train,val}/.",
    )
    p.add_argument(
        "--coda",
        type=Path,
        default=Path("data/processed/coda_yolo"),
        help="CoDA v1 root (output of coda_to_yolo.py).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/coda_yolo_v2"),
    )
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("configs/dataset/coda_yolo_taxonomy.yaml"),
    )
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    splits_present: list[str] = []
    for split in args.splits:
        pseudo_dir = args.pseudo / "labels" / split
        coda_dir = args.coda / "labels" / split
        out_dir = args.output / "labels" / split
        if not pseudo_dir.exists() and not coda_dir.exists():
            LOG.warning("no labels for split=%s, skipping", split)
            continue
        counters = merge_split(pseudo_dir, coda_dir, out_dir)
        LOG.info(
            "split=%s images=%d pseudo_boxes=%d coda_only_kept=%d "
            "coda_dropped=%d labels_written=%d",
            split, counters["images"], counters["pseudo_boxes"],
            counters["coda_only_boxes_kept"], counters["coda_boxes_dropped"],
            counters["labels_written"],
        )
        splits_present.append(split)

    link_images(args.coda, args.output)
    yaml_path = write_coda_yaml(args.output, args.taxonomy, splits_present)
    LOG.info("wrote %s (splits=%s)", yaml_path, splits_present)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
