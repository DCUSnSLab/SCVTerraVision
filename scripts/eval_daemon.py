"""Polling validation-eval daemon for an in-flight training run.

Watches a checkpoint directory, runs `compute_coco_map` on every new
`epoch_NNN.pt` it sees, and logs per-epoch metrics (mAP, AP50, AP75, etc.)
to W&B as a sister run. Exits when `--max-epoch` is processed.

Designed to live alongside `training.train_detection` on a different GPU
so it does not interrupt training. Picks up already-saved checkpoints on
startup, so re-running is idempotent — useful when restarting the daemon.

Usage (on GPU0 while training runs on GPU3):

    CUDA_VISIBLE_DEVICES=0 python -m scripts.eval_daemon \\
        --checkpoint-dir outputs/checkpoints/dinov3_detr_base_full \\
        --val-annotations data/annotations/coda_validation_coco.json \\
        --images-root /home/marsberry/dataset/coda-devkit/data/CODa_full/2d_rect/cam0 \\
        --wandb-project terravision \\
        --wandb-run-name feasible-dream-4-eval \\
        --max-epoch 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

LOG = logging.getLogger("eval_daemon")

_CKPT_RE = re.compile(r"epoch_(\d+)\.pt$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--wandb-project", default="terravision")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Defaults to '{checkpoint_dir.name}-eval'.",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=50,
        help="Daemon exits after evaluating this epoch's checkpoint.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Sleep between checkpoint-dir scans.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Where to write per-epoch JSON. Default: <checkpoint-dir>/eval/.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from torch.utils.data import DataLoader
    import wandb

    from evaluation import compute_coco_map
    from scripts.eval_detection import build_model, load_checkpoint, run_inference
    from training.datasets.coco_loader import CocoDetectionDataset
    from training.train_detection import make_collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device: %s", device)

    results_dir = args.results_dir or args.checkpoint_dir / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb_name = args.wandb_run_name or f"{args.checkpoint_dir.name}-eval"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_name,
        config={
            "checkpoint_dir": str(args.checkpoint_dir),
            "val_annotations": str(args.val_annotations),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "score_threshold": args.score_threshold,
            "top_k": args.top_k,
        },
    )
    LOG.info("wandb run: %s", run.url)

    # Use `epoch` as the x-axis for every val/* chart. Without this, wandb 0.26
    # treats `step=` arguments to log() in confusing ways when the first step
    # jumps from 0 → 5 (it queues the metrics under an autostep counter that
    # the dashboard never plots) and the chart panels stay empty.
    run.define_metric("epoch")
    run.define_metric("val/*", step_metric="epoch")

    # Build dataset/loader/model once and reuse across checkpoints.
    dataset = CocoDetectionDataset(
        annotation_path=args.val_annotations, images_root=args.images_root
    )
    meta_by_image_id = {img["id"]: img for img in dataset.index.images}
    LOG.info("val set: %d images", len(dataset))

    def build_loader() -> DataLoader:
        return DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=make_collate_fn(args.image_size),
            pin_memory=True,
        )

    model = build_model(device)

    processed: set[int] = set()

    # Backfill existing per-epoch eval JSONs into W&B before going into the
    # polling loop. Lets a daemon restart re-emit metrics without re-running
    # inference (idempotent recovery from earlier failed log calls).
    for json_path in sorted(results_dir.glob("eval_epoch_*.json")):
        m = re.search(r"eval_epoch_(\d+)\.json$", json_path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        try:
            cached = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            LOG.warning("could not read %s: %s", json_path, e)
            continue
        log_dict = {
            "epoch": epoch,
            "val/mAP": cached["mAP"],
            "val/AP50": cached["AP50"],
            "val/AP75": cached["AP75"],
            "val/AP_small": cached["AP_small"],
            "val/AP_medium": cached["AP_medium"],
            "val/AP_large": cached["AP_large"],
            "val/AR_1": cached["AR_1"],
            "val/AR_10": cached["AR_10"],
            "val/AR_100": cached["AR_100"],
            "val/num_predictions": cached.get("_num_predictions", 0),
        }
        for cls, ap in cached.get("per_class", {}).items():
            log_dict[f"val/AP_per_class/{cls}"] = ap
        run.log(log_dict)
        processed.add(epoch)
        LOG.info("backfilled epoch=%d from %s (mAP=%.4f)", epoch, json_path, cached["mAP"])

    while True:
        candidates = []
        for f in args.checkpoint_dir.glob("epoch_*.pt"):
            m = _CKPT_RE.search(f.name)
            if not m:
                continue
            epoch = int(m.group(1))
            if epoch in processed:
                continue
            # Skip files still being written: require last-modified > 30s ago.
            if time.time() - f.stat().st_mtime < 30:
                continue
            candidates.append((epoch, f))
        candidates.sort()

        for epoch, ckpt in candidates:
            LOG.info("evaluating %s (epoch=%d)", ckpt, epoch)
            load_checkpoint(model, ckpt, device)
            model.eval()
            loader = build_loader()

            preds = run_inference(
                model,
                loader,
                meta_by_image_id,
                image_size=args.image_size,
                score_threshold=args.score_threshold,
                top_k=args.top_k,
                device=device,
            )
            result = compute_coco_map(preds, args.val_annotations)
            result["_checkpoint"] = str(ckpt)
            result["_checkpoint_epoch"] = epoch
            result["_num_predictions"] = len(preds)

            out_path = results_dir / f"eval_epoch_{epoch:03d}.json"
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            LOG.info(
                "  epoch=%d mAP=%.4f AP50=%.4f AP75=%.4f wrote %s",
                epoch,
                result["mAP"],
                result["AP50"],
                result["AP75"],
                out_path,
            )

            log_dict = {
                "epoch": epoch,
                "val/mAP": result["mAP"],
                "val/AP50": result["AP50"],
                "val/AP75": result["AP75"],
                "val/AP_small": result["AP_small"],
                "val/AP_medium": result["AP_medium"],
                "val/AP_large": result["AP_large"],
                "val/AR_1": result["AR_1"],
                "val/AR_10": result["AR_10"],
                "val/AR_100": result["AR_100"],
                "val/num_predictions": len(preds),
            }
            for cls, ap in result.get("per_class", {}).items():
                log_dict[f"val/AP_per_class/{cls}"] = ap
            # No explicit step= — let wandb autoincrement, and the
            # define_metric("val/*", step_metric="epoch") above re-binds the
            # x-axis to the `epoch` field embedded in the dict.
            run.log(log_dict)

            processed.add(epoch)
            if epoch >= args.max_epoch:
                LOG.info("reached max_epoch=%d; finishing", args.max_epoch)
                run.finish()
                return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
