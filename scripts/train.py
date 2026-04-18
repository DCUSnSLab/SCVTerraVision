"""Train a segmentation model.

Examples:
    # Local CPU smoke test (1 epoch on tiny synthetic data via root_override)
    python scripts/train.py model=dinov2_linear trainer.max_epochs=1 \
        trainer.accelerator=cpu data.num_workers=0

    # Real workstation run on RUGD (B1 baseline)
    python scripts/train.py model=dinov2_linear data.datasets=[rugd] \
        trainer.max_epochs=30

    # B2 baseline (DPT head)
    python scripts/train.py model=dinov2_dpt data.datasets=[rugd,rellis3d]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from camera_perception.training.callbacks import PredictionVizCallback  # noqa: E402
from camera_perception.training.data_module import SegDataModule  # noqa: E402
from camera_perception.training.lit_segmenter import LitSegmenter  # noqa: E402


def load_config(args: argparse.Namespace) -> DictConfig:
    train_cfg = OmegaConf.load(REPO_ROOT / "configs" / "train" / f"{args.train}.yaml")
    model_cfg = OmegaConf.load(REPO_ROOT / "configs" / "model" / f"{args.model}.yaml")
    cfg = OmegaConf.merge(train_cfg, OmegaConf.create({"model": model_cfg}))
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    return cfg  # type: ignore[return-value]


def build_logger(cfg: DictConfig, run_name: str) -> Logger:
    save_dir = cfg.logger.save_dir
    if cfg.logger.type == "tensorboard":
        return TensorBoardLogger(save_dir=save_dir, name=run_name, version="")
    if cfg.logger.type == "csv":
        return CSVLogger(save_dir=save_dir, name=run_name, version="")
    if cfg.logger.type == "wandb":
        from lightning.pytorch.loggers import WandbLogger

        return WandbLogger(project="camera-perception", name=run_name, save_dir=save_dir)
    raise ValueError(f"Unknown logger type: {cfg.logger.type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dinov2_linear", help="configs/model/<name>.yaml")
    parser.add_argument("--train", default="default", help="configs/train/<name>.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. trainer.max_epochs=5 data.batch_size=4",
    )
    args = parser.parse_args()
    cfg = load_config(args)

    L.seed_everything(cfg.seed, workers=True)

    run_name = cfg.logger.name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[train] run_name = {run_name}")
    print(OmegaConf.to_yaml(cfg))

    dm = SegDataModule(
        datasets=list(cfg.data.datasets),
        dataset_config_dir=REPO_ROOT / "configs" / "datasets",
        taxonomy_config=REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml",
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        crop_h=cfg.data.crop_h,
        crop_w=cfg.data.crop_w,
        eval_h=cfg.data.eval_h,
        eval_w=cfg.data.eval_w,
        ignore_index=cfg.data.ignore_index,
        root_overrides=OmegaConf.to_container(cfg.data.get("root_overrides", {}), resolve=True)
        or None,
    )

    model = LitSegmenter(
        taxonomy_config=str(REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"),
        backbone_variant=cfg.model.backbone_variant,
        backbone_freeze=cfg.model.backbone_freeze,
        head=cfg.model.head,
        head_kwargs=OmegaConf.to_container(cfg.model.get("head_kwargs", {}), resolve=True),
        ignore_index=cfg.data.ignore_index,
        class_weights=OmegaConf.to_container(cfg.model.get("class_weights"), resolve=True),
        ce_weight=cfg.model.ce_weight,
        dice_weight=cfg.model.dice_weight,
        label_smoothing=cfg.model.label_smoothing,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )

    logger = build_logger(cfg, run_name)
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(cfg.logger.save_dir) / run_name / "checkpoints",
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
            save_top_k=cfg.checkpoint.save_top_k,
            save_last=cfg.checkpoint.save_last,
            filename="{epoch:03d}-{val/mIoU:.4f}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    viz_cfg = cfg.get("viz", None)
    if viz_cfg is not None and viz_cfg.get("enabled", False):
        callbacks.append(
            PredictionVizCallback(
                num_samples=int(viz_cfg.get("n_samples", 4)),
                every_n_epochs=int(viz_cfg.get("every_n_epochs", 1)),
                tag_prefix=str(viz_cfg.get("tag_prefix", "predictions")),
            )
        )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        deterministic=cfg.trainer.deterministic,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=cfg.logger.save_dir,
    )

    # Snapshot resolved config alongside the run for reproducibility
    out_dir = Path(cfg.logger.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)

    trainer.fit(model, datamodule=dm)
    if dm._test is not None or trainer.datamodule.test_dataloader() is not None:  # type: ignore[union-attr]
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
