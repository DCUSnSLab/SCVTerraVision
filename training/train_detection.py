"""Hydra entrypoint — DINOv3 + DeformableDetr detection training.

Scope (Phase 1-2b): plain PyTorch loop on top of `DinoV3DeformableDetr`,
without HF Trainer (opacity + distributed/logging coupling we don't want
during education/debugging — see ADR 20260424_detr-head-library, §편차).

torch / transformers / pycocotools are imported lazily inside `main()` so
this module collects cleanly in the torch-less dev env (e.g. `pytest`
collects `tests/test_detr_head.py` without attempting to import this file
end-to-end).

Typical invocation (in the GPU env):
    HF_TOKEN=hf_... python -m training.train_detection \
        detection=dinov3_detr_base \
        dataset.split=training \
        detection.training.epochs=50

Override knobs on the CLI:
    detection.training.lr=1e-4
    detection.training.freeze_backbone_epochs=0
    backbone.dtype=bfloat16
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger("train_detection")


# --------------------------------------------------------------------------
# Image / label preprocessing
# --------------------------------------------------------------------------

# ImageNet normalization stats — DINOv3 inherits these from its pretraining
# pipeline. Diverging means the backbone sees out-of-distribution color
# statistics and AP collapses silently, so pin them here.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_resize(
    image: Any, target_size: int, pad_value: int = 114
) -> tuple[Any, float, tuple[int, int]]:
    """Aspect-preserving resize + bottom-right pad to (target_size, target_size).

    Returns:
        padded image HxWx3 uint8,
        scale factor (new_short / old_short equivalent),
        (pad_h, pad_w) pixels of padding applied.

    Pad on the right/bottom only, so top-left origin is preserved and the
    (x, y) bbox arithmetic below is just a multiply by `scale`.
    """
    import numpy as np

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # Simple nearest/bilinear via PIL to avoid adding cv2 as a hard dep
    # here; training env has both, but tests might not.
    from PIL import Image

    resized = np.asarray(
        Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR),
        dtype=np.uint8,
    )

    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    return padded, scale, (pad_h, pad_w)


def coco_xywh_to_cxcywh_norm(
    boxes_xywh: Any, image_h: int, image_w: int
) -> Any:
    """Convert (N, 4) COCO xywh pixel boxes → (N, 4) cxcywh normalized [0,1].

    HF DeformableDetr expects normalized cxcywh; do the conversion here so
    the backward pass stays aligned with the matcher's assumptions.
    """
    import numpy as np

    boxes = np.asarray(boxes_xywh, dtype=np.float32).reshape(-1, 4)
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x + w / 2.0) / float(image_w)
    cy = (y + h / 2.0) / float(image_h)
    nw = w / float(image_w)
    nh = h / float(image_h)
    return np.stack([cx, cy, nw, nh], axis=1)


def preprocess_sample(
    sample: dict[str, Any], target_size: int
) -> dict[str, Any]:
    """Turn a `CocoDetectionDataset` item into a model-ready numpy sample.

    Output keys:
        pixel_values: (3, H, W) float32, ImageNet-normalized
        pixel_mask:   (H, W) uint8, 1 inside the real image area
        class_labels: (N,) int64
        boxes:        (N, 4) float32 cxcywh in [0,1] — already adjusted for
                      letterbox padding (divides by target_size, not original).
        image_id:     int
    """
    import numpy as np

    image = sample["image"]
    h, w = image.shape[:2]
    padded, scale, (pad_h, pad_w) = letterbox_resize(image, target_size)

    # Normalize and transpose to CHW.
    arr = padded.astype(np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32
    )
    pixel_values = arr.transpose(2, 0, 1)

    pixel_mask = np.zeros((target_size, target_size), dtype=np.uint8)
    pixel_mask[: target_size - pad_h, : target_size - pad_w] = 1

    boxes_xywh = sample["boxes"] * scale  # scale pixel coords into padded frame
    # Normalize by the **padded** size, since HF DETR normalizes against the
    # full input it sees. Boxes that lie fully in the padded border would be
    # bogus — the coco converter already filters those, so we don't re-clip.
    boxes = coco_xywh_to_cxcywh_norm(boxes_xywh, target_size, target_size)

    # COCO category_id is 1-indexed (1..num_classes); HF DETR / DeformableDetr
    # expects 0-indexed class labels in [0, num_labels). Shift here so the
    # matcher's `logits[:, target_ids]` indexing stays in-bounds for every
    # class — subset training happened to not trigger this because class
    # id=16 (fire_hydrant) was absent from sequences 0·1·2.
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": sample["labels"].astype(np.int64) - 1,
        "boxes": boxes.astype(np.float32),
        "image_id": int(sample["image_id"]),
    }


# --------------------------------------------------------------------------
# Collate + DataLoader wiring (torch-local)
# --------------------------------------------------------------------------


def make_collate_fn(target_size: int):
    """Return a collate fn that assembles a batch for HF DeformableDetr."""
    import numpy as np
    import torch

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        processed = [preprocess_sample(s, target_size) for s in samples]
        pixel_values = torch.from_numpy(
            np.stack([p["pixel_values"] for p in processed])
        )
        pixel_mask = torch.from_numpy(
            np.stack([p["pixel_mask"] for p in processed])
        ).long()
        labels = [
            {
                "class_labels": torch.from_numpy(p["class_labels"]),
                "boxes": torch.from_numpy(p["boxes"]),
            }
            for p in processed
        ]
        image_ids = torch.tensor(
            [p["image_id"] for p in processed], dtype=torch.long
        )
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
            "image_ids": image_ids,
        }

    return collate


# --------------------------------------------------------------------------
# Optimizer / schedule
# --------------------------------------------------------------------------


def build_optimizer(model: Any, cfg: Any) -> Any:
    """AdamW with a lower LR for backbone params.

    Params whose name contains "backbone" belong to the DINOv3 subtree
    (inside `model.model.backbone.conv_encoder.model`) and get
    `cfg.lr_backbone`. Everything else (encoder, decoder, heads,
    input_proj) gets `cfg.lr`.

    Backbone params are added to the optimizer **even when initially
    frozen** (`requires_grad=False`). The freeze schedule flips
    `requires_grad=True` mid-training (`set_backbone_frozen(False)` at
    `freeze_backbone_epochs`), and the optimizer must already own those
    params for the unfrozen backbone gradients to propagate. AdamW skips
    params with `grad is None` per step, so including frozen params here
    is safe — they simply receive no updates while frozen.
    """
    import torch

    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        # Note: do NOT filter by requires_grad here. See docstring.
        if "backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"params": head_params, "lr": cfg.lr},
        {"params": backbone_params, "lr": cfg.lr_backbone},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: Any, cfg: Any, steps_per_epoch: int) -> Any:
    """Warmup → cosine decay.

    Implemented as a lambda LR so we can chain warmup with cosine without
    pulling an extra dep (HF's `get_cosine_schedule_with_warmup` would work
    too — left as future simplification).
    """
    import torch

    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Freeze schedule
# --------------------------------------------------------------------------


def set_backbone_frozen(model: Any, frozen: bool) -> None:
    """Toggle requires_grad on the DINOv3 backbone inside the HF model."""
    # The shim exposes the DINOv3 module as `shim.model` (see
    # models/detection/detr_head.py:_build_conv_encoder_shim).
    conv_encoder = model.model.backbone.conv_encoder
    for p in conv_encoder.model.parameters():
        p.requires_grad_(not frozen)


# --------------------------------------------------------------------------
# Main entrypoint
# --------------------------------------------------------------------------


def _ensure_ml_deps() -> None:
    """Fail fast with a helpful message if the training deps are missing."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "training.train_detection requires torch + transformers. "
            "Install the training extras (`pip install -e .[dev]`) and ensure "
            "HF_TOKEN is exported for gated DINOv3 access."
        ) from e


def _wandb_init(cfg: Any) -> Any:
    """Return a wandb run or None if disabled / unavailable."""
    if not cfg.logging.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        LOG.warning("logging.wandb.enabled=True but wandb not installed; skipping.")
        return None
    return wandb.init(
        project=cfg.logging.wandb.project,
        entity=cfg.logging.wandb.entity,
        mode=cfg.logging.wandb.mode,
        config=dict(cfg),
    )


def main(cfg: Any) -> None:
    """Entry point — invoked by Hydra."""
    _ensure_ml_deps()

    import torch
    from torch.utils.data import DataLoader

    from models.backbone.dinov3_backbone import DinoV3BackboneConfig
    from models.detection.detr_head import DetrHeadConfig, DinoV3DeformableDetr
    from training.datasets.coco_loader import CocoDetectionDataset

    # ---- Build dataset ----
    dataset = CocoDetectionDataset(
        annotation_path=cfg.dataset.annotations_path,
        images_root=cfg.dataset.images_root,
    )
    LOG.info(
        "dataset: %d images, %d annotations",
        len(dataset),
        sum(len(v) for v in dataset.index.annotations_by_image.values()),
    )

    loader = DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=cfg.detection.training.batch_size,
        shuffle=True,
        num_workers=cfg.detection.training.num_workers,
        collate_fn=make_collate_fn(cfg.detection.image_size),
        pin_memory=True,
    )

    # ---- Build model ----
    backbone_cfg = DinoV3BackboneConfig(
        model_id=cfg.backbone.model_id,
        patch_size=cfg.backbone.patch_size,
        hidden_dim=cfg.backbone.hidden_dim,
        # Freeze starts True regardless of backbone YAML — the schedule below
        # handles the unfreeze; passing freeze=True to DinoV3Backbone sets the
        # initial requires_grad state.
        freeze=True,
        dtype=cfg.backbone.dtype,
        output_layer=cfg.backbone.output_layer,
    )
    head_cfg = DetrHeadConfig(
        num_labels=cfg.detection.num_labels,
        num_queries=cfg.detection.num_queries,
        d_model=cfg.detection.d_model,
        encoder_layers=cfg.detection.encoder_layers,
        decoder_layers=cfg.detection.decoder_layers,
        encoder_attention_heads=cfg.detection.encoder_attention_heads,
        decoder_attention_heads=cfg.detection.decoder_attention_heads,
        encoder_ffn_dim=cfg.detection.encoder_ffn_dim,
        decoder_ffn_dim=cfg.detection.decoder_ffn_dim,
        dropout=cfg.detection.dropout,
        activation_function=cfg.detection.activation_function,
        num_feature_levels=cfg.detection.num_feature_levels,
        backbone_channels=cfg.detection.backbone_channels,
        aux_loss=cfg.detection.aux_loss,
        two_stage=cfg.detection.two_stage,
        with_box_refine=cfg.detection.with_box_refine,
        disable_custom_kernels=cfg.detection.disable_custom_kernels,
        class_cost=cfg.detection.class_cost,
        bbox_cost=cfg.detection.bbox_cost,
        giou_cost=cfg.detection.giou_cost,
        bbox_loss_coefficient=cfg.detection.bbox_loss_coefficient,
        giou_loss_coefficient=cfg.detection.giou_loss_coefficient,
        focal_alpha=cfg.detection.focal_alpha,
    )
    wrapper = DinoV3DeformableDetr(head_config=head_cfg, backbone_config=backbone_cfg)
    model = wrapper.load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    LOG.info("device: %s", device)

    # ---- Optimizer / schedule ----
    opt = build_optimizer(model, cfg.detection.training)
    sched = build_scheduler(
        opt, cfg.detection.training, steps_per_epoch=max(1, len(loader))
    )

    # ---- Start frozen ----
    if cfg.detection.training.freeze_backbone_epochs > 0:
        set_backbone_frozen(model, True)

    # ---- Logging ----
    run = _wandb_init(cfg)

    # ---- Checkpoints ----
    ckpt_dir = Path(cfg.detection.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train ----
    global_step = 0
    for epoch in range(cfg.detection.training.epochs):
        if epoch == cfg.detection.training.freeze_backbone_epochs:
            LOG.info("epoch %d: unfreezing backbone", epoch)
            set_backbone_frozen(model, False)

        model.train()
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [
                {k: v.to(device) for k, v in t.items()} for t in batch["labels"]
            ]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.detection.training.grad_clip
            )
            opt.step()
            sched.step()

            if global_step % 50 == 0:
                LOG.info(
                    "epoch %d step %d loss %.4f", epoch, global_step, float(loss)
                )
                if run is not None:
                    run.log(
                        {
                            "train/loss": float(loss),
                            "train/lr": opt.param_groups[0]["lr"],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
            global_step += 1

        if (epoch + 1) % cfg.detection.training.save_every_n_epochs == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                },
                ckpt_path,
            )
            LOG.info("saved checkpoint: %s", ckpt_path)

    if run is not None:
        run.finish()


def _cli() -> None:
    """Hydra-decorated CLI wrapper.

    Hydra's decorator is applied lazily inside this function so importing
    `training.train_detection` doesn't require Hydra/OmegaConf to be
    installed (tests collect this module without them).
    """
    import hydra

    @hydra.main(version_base=None, config_path="../configs", config_name="base")
    def _run(cfg: Any) -> None:
        logging.basicConfig(level=logging.INFO)
        main(cfg)

    _run()


if __name__ == "__main__":
    _cli()
