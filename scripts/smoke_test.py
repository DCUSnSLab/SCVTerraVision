"""End-to-end smoke test: synthetic data → DINOv2 + linear head → 1 epoch CPU.

Verifies the full Phase 2 pipeline (datamodule, lit module, trainer) before
running real experiments on the workstation. Downloads DINOv2-small weights
(~85 MB) on first run.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import lightning as L  # noqa: E402
from lightning.pytorch.loggers import TensorBoardLogger  # noqa: E402

from camera_perception.training.callbacks import PredictionVizCallback  # noqa: E402
from camera_perception.training.data_module import SegDataModule  # noqa: E402
from camera_perception.training.lit_segmenter import LitSegmenter  # noqa: E402

# Reuse the test fixture builder
from test_datasets import _make_rugd_tree  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", default="cpu", help="cpu | mps | gpu")
    parser.add_argument("--n-train", type=int, default=4)
    parser.add_argument("--n-val", type=int, default=2)
    parser.add_argument("--crop", type=int, default=56, help="must be multiple of 14")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--with-viz", action="store_true", help="enable TB logger + viz callback")
    parser.add_argument("--logdir", default="outputs/smoke_test", help="TB log dir when --with-viz")
    args = parser.parse_args()

    L.seed_everything(0, workers=True)
    taxonomy_cfg = REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"
    dataset_cfg_dir = REPO_ROOT / "configs" / "datasets"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "rugd"
        # Use 'creek' scene; create train/val/test by reusing the fixture builder
        # then duplicating with split-specific filenames for val/test.
        _make_rugd_tree(root, n=args.n_train)
        # Build val split file by sampling first n_val stems
        train_stems = (root / "splits" / "train.txt").read_text().strip().splitlines()
        val_stems = train_stems[: args.n_val]
        (root / "splits" / "val.txt").write_text("\n".join(val_stems) + "\n")

        dm = SegDataModule(
            datasets=["rugd"],
            dataset_config_dir=dataset_cfg_dir,
            taxonomy_config=taxonomy_cfg,
            batch_size=2,
            num_workers=0,
            crop_h=args.crop,
            crop_w=args.crop,
            eval_h=args.crop,
            eval_w=args.crop,
            ignore_index=255,
            root_overrides={"rugd": str(root)},
            pin_memory=False,
        )

        model = LitSegmenter(
            taxonomy_config=str(taxonomy_cfg),
            backbone_variant="small",
            backbone_freeze=True,
            head="linear",
            head_kwargs={"use_bn": True},
            ignore_index=255,
            lr=1e-3,
            weight_decay=1e-4,
        )

        callbacks: list = []
        if args.with_viz:
            logger = TensorBoardLogger(save_dir=args.logdir, name="smoke", version="")
            callbacks.append(
                PredictionVizCallback(num_samples=2, every_n_epochs=1, tag_prefix="predictions")
            )
        else:
            logger = False  # type: ignore[assignment]

        trainer = L.Trainer(
            max_epochs=args.epochs,
            accelerator=args.accelerator,
            devices=1,
            precision="32-true",
            log_every_n_steps=1,
            enable_checkpointing=False,
            enable_progress_bar=True,
            logger=logger,
            callbacks=callbacks,
        )

        print(f"[smoke] training {args.epochs} epoch(s) on {args.accelerator} ...")
        trainer.fit(model, datamodule=dm)
        print("[smoke] OK — training pipeline runs end-to-end")
        if args.with_viz:
            print(f"[smoke] TensorBoard logs at: {args.logdir}/smoke")


if __name__ == "__main__":
    main()
