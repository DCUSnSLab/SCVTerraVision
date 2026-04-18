"""Save N triptychs (image | mask | overlay) to a directory."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from camera_perception.data.datasets import RUGDDataset, Rellis3DDataset  # noqa: E402
from camera_perception.data.viz import save_triptych  # noqa: E402

DATASET_CLASSES = {
    "rugd": RUGDDataset,
    "rellis3d": Rellis3DDataset,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CLASSES))
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", default="outputs/samples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml"
    tax = REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"
    ds = DATASET_CLASSES[args.dataset](
        dataset_config=cfg,
        taxonomy_config=tax,
        split=args.split,
        transform=None,
        root_override=args.root,
    )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), k=min(args.n, len(ds)))
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        sample = ds[i]
        # When transform is None, image is float [0,1] CHW; convert back for display.
        img_t = sample["image"]
        img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = sample["mask"].numpy()
        out_path = out_dir / f"{i:06d}_{Path(sample['meta']['stem']).name}.png"
        save_triptych(img, mask, ds.taxonomy, out_path, title=sample["meta"]["stem"])
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
