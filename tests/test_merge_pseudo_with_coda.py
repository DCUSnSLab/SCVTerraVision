"""Unit tests for training.datasets.merge_pseudo_with_coda.

핵심 검증:
  - pseudo (cls 0..79) 모두 keep
  - CoDA 의 cls 80..88 만 keep (CoDA-only 9)
  - CoDA 의 cls ∈ {0,1,3,5,7,9,10,13,89,90} → 폐기
  - dedup 불필요 (서로 다른 ID 공간)
  - 빈 라벨 파일 / 한쪽만 있는 stem 처리
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.datasets.merge_pseudo_with_coda import (
    CODA_ONLY_KEEP,
    merge_split,
)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def _read_cls(path: Path) -> list[int]:
    if not path.exists():
        return []
    return [int(l.split()[0]) for l in path.read_text().splitlines() if l.strip()]


def test_keep_set_matches_decision() -> None:
    """D14 — CoDA-only 9 이 80..88 정확히."""
    assert CODA_ONLY_KEEP == frozenset({80, 81, 82, 83, 84, 85, 86, 87, 88})


def test_pseudo_only_image(tmp_path: Path) -> None:
    """Pseudo 가 모든 박스, CoDA 라벨 빈 경우."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "img1.txt", ["0 0.5 0.5 0.1 0.1", "2 0.3 0.3 0.2 0.2"])
    _write(coda / "img1.txt", [])

    counters = merge_split(pseudo, coda, out)
    assert counters["pseudo_boxes"] == 2
    assert counters["coda_only_boxes_kept"] == 0
    assert counters["coda_boxes_dropped"] == 0
    assert _read_cls(out / "img1.txt") == [0, 2]


def test_coda_only_kept_classes_pass_through(tmp_path: Path) -> None:
    """CoDA 80..88 keep + dedup 없음."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "img2.txt", ["0 0.1 0.1 0.05 0.05"])
    _write(coda / "img2.txt", [
        "80 0.5 0.5 0.1 0.1",   # scooter — keep
        "85 0.6 0.6 0.1 0.1",   # cone   — keep
        "88 0.7 0.7 0.1 0.1",   # trash_can — keep
    ])
    counters = merge_split(pseudo, coda, out)
    assert counters["pseudo_boxes"] == 1
    assert counters["coda_only_boxes_kept"] == 3
    assert counters["coda_boxes_dropped"] == 0
    assert sorted(_read_cls(out / "img2.txt")) == [0, 80, 85, 88]


def test_coda_overlap_classes_dropped(tmp_path: Path) -> None:
    """D12+D13 — vehicle dispatch + COCO80 overlap 모두 폐기."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "img3.txt", ["0 0.1 0.1 0.05 0.05", "2 0.4 0.4 0.1 0.1"])
    _write(coda / "img3.txt", [
        "0 0.5 0.5 0.1 0.1",    # CoDA Pedestrian (D13) — drop
        "2 0.6 0.6 0.1 0.1",    # CoDA Car (D12)        — drop
        "5 0.7 0.7 0.1 0.1",    # CoDA Bus (D12)        — drop
        "7 0.8 0.8 0.1 0.1",    # CoDA Truck (D12)      — drop
        "13 0.9 0.9 0.1 0.1",   # CoDA Bench (D13)      — drop
        "89 0.5 0.4 0.1 0.1",   # service_vehicle (D12) — drop (deprecated id)
        "90 0.6 0.4 0.1 0.1",   # golf_cart (D12)       — drop (deprecated id)
        "82 0.3 0.3 0.1 0.1",   # CoDA-only pole — keep
    ])
    counters = merge_split(pseudo, coda, out)
    assert counters["pseudo_boxes"] == 2
    assert counters["coda_only_boxes_kept"] == 1   # only pole
    assert counters["coda_boxes_dropped"] == 7
    assert sorted(_read_cls(out / "img3.txt")) == [0, 2, 82]


def test_stems_unique_to_each_source(tmp_path: Path) -> None:
    """한쪽에만 stem 있는 경우 — 다른 쪽 빈 파일로 처리."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "only_pseudo.txt", ["0 0.5 0.5 0.1 0.1"])
    _write(coda / "only_coda.txt", ["82 0.5 0.5 0.1 0.1"])

    counters = merge_split(pseudo, coda, out)
    assert counters["images"] == 2
    assert _read_cls(out / "only_pseudo.txt") == [0]
    assert _read_cls(out / "only_coda.txt") == [82]


def test_empty_inputs(tmp_path: Path) -> None:
    """양쪽 비어있는 stem — 빈 라벨 파일 keep (background image 의도)."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "empty.txt", [])
    _write(coda / "empty.txt", [])

    counters = merge_split(pseudo, coda, out)
    assert counters["images"] == 1
    assert counters["pseudo_boxes"] == 0
    assert counters["coda_only_boxes_kept"] == 0
    assert (out / "empty.txt").exists()
    assert _read_cls(out / "empty.txt") == []


def test_malformed_lines_skipped(tmp_path: Path) -> None:
    """non-int cls / 빈 줄 무시."""
    pseudo = tmp_path / "p"
    coda = tmp_path / "c"
    out = tmp_path / "o"
    _write(pseudo / "img.txt", [
        "0 0.5 0.5 0.1 0.1",
        "",
        "garbage",
        "82 0.3 0.3 0.1 0.1",
    ])
    _write(coda / "img.txt", [])
    counters = merge_split(pseudo, coda, out)
    assert sorted(_read_cls(out / "img.txt")) == [0, 82]
