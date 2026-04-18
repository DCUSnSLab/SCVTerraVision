"""Verify a dataset: count samples, check file integrity, print class distribution."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from camera_perception.data.datasets import RUGDDataset, Rellis3DDataset  # noqa: E402

DATASET_CLASSES = {
    "rugd": RUGDDataset,
    "rellis3d": Rellis3DDataset,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CLASSES))
    parser.add_argument("--root", default=None, help="Override dataset root path")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=200, help="Cap for class stats")
    args = parser.parse_args()

    cfg = REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml"
    tax = REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"
    ds_cls = DATASET_CLASSES[args.dataset]
    ds = ds_cls(
        dataset_config=cfg,
        taxonomy_config=tax,
        split=args.split,
        transform=None,
        root_override=args.root,
    )

    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Samples: {len(ds)}")

    n = min(args.max_samples, len(ds))
    print(f"Computing class distribution over first {n} samples...")
    counts: Counter = Counter()
    bad = 0
    for i in tqdm(range(n)):
        try:
            sample = ds[i]
            mask = sample["mask"].numpy()
            for cid, c in zip(*np.unique(mask, return_counts=True)):
                counts[int(cid)] += int(c)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"  [error] index {i}: {e}")

    print("\nUnified class pixel counts:")
    name_by_id = {c.id: c.name for c in ds.taxonomy.classes}
    name_by_id[ds.taxonomy.ignore_id] = "ignore"
    total = sum(counts.values()) or 1
    for cid in sorted(counts):
        name = name_by_id.get(cid, f"unknown_{cid}")
        pct = 100.0 * counts[cid] / total
        print(f"  [{cid:>3}] {name:<25} {counts[cid]:>14}  ({pct:5.2f}%)")

    if bad:
        print(f"\nWARNING: {bad} sample(s) failed to load")
        sys.exit(1)


if __name__ == "__main__":
    main()
