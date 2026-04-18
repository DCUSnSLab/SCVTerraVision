"""Evaluate a checkpoint (or untrained model for B0) on a dataset split.

Outputs per-class IoU + mIoU + binary traversability IoU + a few qualitative
visualizations. Writes a JSON report under outputs/eval/<run_name>/.

Examples:
    # B0 sanity check (no checkpoint, untrained linear head)
    python scripts/eval.py --model dinov2_linear --dataset rugd --split val \
        --baseline_b0 --out outputs/eval/b0_rugd_val

    # Evaluate trained checkpoint
    python scripts/eval.py --ckpt outputs/runs/<run>/checkpoints/best.ckpt \
        --dataset rugd --split val --out outputs/eval/run_rugd_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from camera_perception.data.datasets import RUGDDataset, Rellis3DDataset  # noqa: E402
from camera_perception.data.transforms import build_eval_transform  # noqa: E402
from camera_perception.data.viz import save_triptych  # noqa: E402
from camera_perception.training.lit_segmenter import LitSegmenter  # noqa: E402
from camera_perception.training.metrics import (  # noqa: E402
    BinaryTraversabilityIoU,
    build_iou_metric,
    per_class_iou_dict,
)

DATASET_REGISTRY = {"rugd": RUGDDataset, "rellis3d": Rellis3DDataset}


def build_model_for_b0(model_cfg_name: str, taxonomy_config: Path) -> LitSegmenter:
    cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / f"{model_cfg_name}.yaml")
    return LitSegmenter(
        taxonomy_config=str(taxonomy_config),
        backbone_variant=cfg.backbone_variant,
        backbone_freeze=cfg.backbone_freeze,
        head=cfg.head,
        head_kwargs=OmegaConf.to_container(cfg.get("head_kwargs", {}), resolve=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=None, help="Lightning checkpoint path")
    parser.add_argument("--model", default="dinov2_linear", help="Model config (B0 mode)")
    parser.add_argument("--baseline_b0", action="store_true", help="Use untrained model (B0)")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REGISTRY))
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-h", type=int, default=518)
    parser.add_argument("--eval-w", type=int, default=518)
    parser.add_argument("--out", default="outputs/eval/run", help="output directory")
    parser.add_argument("--n-viz", type=int, default=8)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    args = parser.parse_args()

    taxonomy_cfg = REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"
    dataset_cfg = REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ckpt and args.baseline_b0:
        raise SystemExit("--ckpt and --baseline_b0 are mutually exclusive")
    if not args.ckpt and not args.baseline_b0:
        raise SystemExit("Must provide either --ckpt or --baseline_b0")

    model = (
        LitSegmenter.load_from_checkpoint(args.ckpt)
        if args.ckpt
        else build_model_for_b0(args.model, taxonomy_cfg)
    )
    model.eval()
    device = (
        torch.device(args.device)
        if args.device != "auto"
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    model.to(device)

    eval_tfm = build_eval_transform(height=args.eval_h, width=args.eval_w)
    ds_cls = DATASET_REGISTRY[args.dataset]
    ds = ds_cls(
        dataset_config=dataset_cfg,
        taxonomy_config=taxonomy_cfg,
        split=args.split,
        transform=eval_tfm,
        root_override=args.root,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    num_classes = model.taxonomy.num_classes()
    iou_metric = build_iou_metric(num_classes).to(device)
    trav_metric = BinaryTraversabilityIoU(model.taxonomy).to(device)

    n_seen = 0
    viz_indices = set(range(min(args.n_viz, len(ds))))
    L.seed_everything(0, workers=True)

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["mask"].to(device)
            logits = model(image)
            preds = logits.argmax(dim=1)
            iou_metric.update(preds, target)
            trav_metric.update(preds, target)

            for k in range(image.shape[0]):
                if n_seen + k in viz_indices:
                    img_np = (
                        (image[k].cpu().permute(1, 2, 0).numpy() * 0.0)  # ignore norm; viz needs raw
                    )
                    # Re-load original image without normalization for clean viz
                    raw_index = n_seen + k
                    raw = ds._read_image(ds.samples[raw_index].image)
                    save_triptych(
                        raw,
                        preds[k].cpu().numpy(),
                        model.taxonomy,
                        out_dir / f"pred_{raw_index:06d}.png",
                        title=f"pred {raw_index}",
                    )
                    save_triptych(
                        raw,
                        target[k].cpu().numpy(),
                        model.taxonomy,
                        out_dir / f"gt_{raw_index:06d}.png",
                        title=f"gt {raw_index}",
                    )
                    del img_np
            n_seen += image.shape[0]

    iou = iou_metric.compute().cpu()
    miou = float(torch.nanmean(iou).item())
    trav = float(trav_metric.compute().cpu().item())
    per_class = per_class_iou_dict(iou, model.taxonomy)

    report = {
        "ckpt": args.ckpt,
        "baseline_b0": args.baseline_b0,
        "model_cfg": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": n_seen,
        "mIoU": miou,
        "trav_IoU": trav,
        "per_class_iou": per_class,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
