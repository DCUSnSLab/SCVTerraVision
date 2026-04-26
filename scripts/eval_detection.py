"""COCO mAP evaluation for DINOv3 + DeformableDetr checkpoints (Phase 1-2b+).

Loads a training checkpoint, runs inference over a COCO-format validation
set, and writes a JSON summary with mAP / AP50 / AP75 / per-class AP.

Designed to be reusable across Phase 1-2b baseline, Phase 1-2c fine-tune,
and Phase 1-5 optimization — hence a standalone script rather than inline
snippets.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m scripts.eval_detection \\
        --checkpoint outputs/2026-04-25/01-14-11/checkpoints/dinov3_detr_base/epoch_003.pt \\
        --val-annotations data/annotations/coda_validation_subset.json \\
        --images-root /home/marsberry/dataset/coda-devkit/data/CODa_full/2d_rect/cam0 \\
        --output outputs/eval_phase1-2b_subset.json \\
        --image-size 1024 --batch-size 2 --score-threshold 0.0 --top-k 100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger("eval_detection")


def letterbox_scale(orig_h: int, orig_w: int, target_size: int) -> float:
    """Match training.train_detection.letterbox_resize scale formula."""
    return target_size / float(max(orig_h, orig_w))


def build_model(device: str) -> Any:
    from models.detection.detr_head import DinoV3DeformableDetr

    wrapper = DinoV3DeformableDetr()
    model = wrapper.load()
    model.to(device).eval()
    return model


def load_checkpoint(model: Any, checkpoint_path: Path, device: str) -> int:
    import torch

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        LOG.warning("missing keys when loading ckpt: %d (first 5: %s)", len(missing), missing[:5])
    if unexpected:
        LOG.warning("unexpected keys: %d (first 5: %s)", len(unexpected), unexpected[:5])
    return int(ckpt.get("epoch", -1))


def run_inference(
    model: Any,
    loader: Any,
    meta_by_image_id: dict[int, dict[str, Any]],
    *,
    image_size: int,
    score_threshold: float,
    top_k: int,
    device: str,
) -> list[dict[str, Any]]:
    """Forward each batch, convert to COCO predictions in original pixel coords."""
    import torch

    predictions: list[dict[str, Any]] = []
    # HF letterbox-frame (image_size, image_size) for post_process target.
    target_sizes_lb = torch.tensor(
        [[image_size, image_size]], dtype=torch.long, device=device
    )
    from transformers import DeformableDetrImageProcessor

    processor = DeformableDetrImageProcessor()

    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            image_ids = batch["image_ids"].tolist()

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            bs = pixel_values.shape[0]
            target_sizes = target_sizes_lb.expand(bs, 2)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=score_threshold,
                top_k=top_k,
            )

            for image_id, per_image in zip(image_ids, results):
                meta = meta_by_image_id[image_id]
                scale = letterbox_scale(meta["height"], meta["width"], image_size)

                scores = per_image["scores"].cpu().tolist()
                labels = per_image["labels"].cpu().tolist()
                boxes = per_image["boxes"].cpu().tolist()  # xyxy in letterbox frame

                for score, label, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = box
                    # Undo letterbox: divide by scale to get original pixel coords.
                    x1o, y1o = x1 / scale, y1 / scale
                    x2o, y2o = x2 / scale, y2 / scale
                    # Clamp to image bounds (letterbox pad may yield coords
                    # slightly outside, especially for boxes touching the padded
                    # right/bottom edges).
                    x1o = max(0.0, min(float(meta["width"]), x1o))
                    x2o = max(0.0, min(float(meta["width"]), x2o))
                    y1o = max(0.0, min(float(meta["height"]), y1o))
                    y2o = max(0.0, min(float(meta["height"]), y2o))
                    w, h = x2o - x1o, y2o - y1o
                    if w <= 0 or h <= 0:
                        continue
                    # HF returns 0-indexed labels (the same index space the
                    # loss was trained on); convert back to 1-indexed COCO
                    # category_id so pycocotools can match GT.
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label) + 1,
                            "bbox": [round(x1o, 2), round(y1o, 2), round(w, 2), round(h, 2)],
                            "score": float(score),
                        }
                    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import torch
    from torch.utils.data import DataLoader

    from evaluation import compute_coco_map
    from training.datasets.coco_loader import CocoDetectionDataset
    from training.train_detection import make_collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info("device: %s", device)

    dataset = CocoDetectionDataset(
        annotation_path=args.val_annotations, images_root=args.images_root
    )
    meta_by_image_id = {img["id"]: img for img in dataset.index.images}
    LOG.info("val set: %d images", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(args.image_size),
        pin_memory=True,
    )

    model = build_model(device)
    epoch = load_checkpoint(model, args.checkpoint, device)
    LOG.info("loaded checkpoint: %s (epoch=%s)", args.checkpoint, epoch)

    predictions = run_inference(
        model,
        loader,
        meta_by_image_id,
        image_size=args.image_size,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        device=device,
    )
    LOG.info("predictions: %d", len(predictions))

    result = compute_coco_map(predictions, args.val_annotations)
    result["_checkpoint"] = str(args.checkpoint)
    result["_checkpoint_epoch"] = epoch
    result["_val_annotations"] = str(args.val_annotations)
    result["_num_predictions"] = len(predictions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOG.info("wrote %s", args.output)
    for key in ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"):
        LOG.info("  %s = %.4f", key, result[key])
    if "per_class" in result:
        for name, ap in sorted(result["per_class"].items()):
            LOG.info("  per-class %s = %.4f", name, ap)


if __name__ == "__main__":
    main()
