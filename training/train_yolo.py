"""Hydra entrypoint — YOLO26 fine-tune training (Phase 1-3).

Wraps Ultralytics' Python API in our project's Hydra/W&B conventions:

    python -m training.train_yolo +detection=yolo26_n
    python -m training.train_yolo +detection=yolo26_n detection.training.imgsz=1024
    python -m training.train_yolo +detection=yolo26_l detection.training.device='0,3'

Why we wrap rather than calling `yolo train ...` directly:
    - reuse the existing W&B init flow (cfg.logging.wandb), avoiding two
      parallel run streams (ultralytics has its own wandb integration that
      we disable below — see D9 in docs/decisions/20260428_pivot-to-yolo26.md)
    - keep the same hyperparam-sweep ergonomics as train_detection.py
      (Hydra CLI override syntax)
    - record the resolved cfg into the W&B run so YOLO + DETR runs are
      directly comparable on the same dashboard

Ultralytics is imported lazily inside `main()` so this module collects
under pytest in the dev env (where `ultralytics` may not be installed).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

LOG = logging.getLogger("train_yolo")


def _ensure_ml_deps() -> None:
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "training.train_yolo requires ultralytics. Install via "
            "`pip install -e .` (which now pulls ultralytics>=8.3.50) or "
            "`pip install ultralytics`."
        ) from e


def _wandb_init(cfg: Any) -> Any:
    """Mirror of train_detection._wandb_init — single source of W&B truth."""
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
        name=cfg.detection.name,
    )


def _make_wandb_callback(run: Any):
    """Return an `on_fit_epoch_end` callback that ships metrics to W&B.

    Ultralytics calls each registered callback with the trainer instance.
    Trainer state we care about:
      - trainer.epoch                          : current epoch (int)
      - trainer.metrics                        : dict, e.g. {"metrics/mAP50(B)": ...,
                                                              "metrics/mAP50-95(B)": ...,
                                                              "train/box_loss": ...,
                                                              "lr/pg0": ...}
    Returning a no-op when run is None keeps the rest of the wiring uniform.
    """
    if run is None:
        return lambda _trainer: None

    def _cb(trainer: Any) -> None:
        try:
            payload: dict[str, Any] = {"epoch": int(getattr(trainer, "epoch", 0))}
            metrics = getattr(trainer, "metrics", None) or {}
            for k, v in metrics.items():
                # Skip non-scalar entries; W&B's structured logging handles
                # numpy scalars but not nested dicts.
                try:
                    payload[k] = float(v)
                except (TypeError, ValueError):
                    continue
            run.log(payload)
        except Exception as exc:  # pragma: no cover — never fail training
            LOG.warning("wandb logging failed: %s", exc)

    return _cb


def _disable_ultralytics_wandb() -> None:
    """Turn off ultralytics' built-in W&B integration (D9).

    Two channels writing to the same project would split the run history
    and double-count steps. We keep our own callback (_make_wandb_callback)
    as the single writer.
    """
    try:
        from ultralytics.utils import SETTINGS  # type: ignore

        SETTINGS.update({"wandb": False})
    except Exception as exc:
        LOG.warning("could not disable ultralytics wandb integration: %s", exc)


def _resolve_data_yaml(cfg: Any) -> str:
    """Compute the absolute path to the ultralytics dataset YAML.

    The `data:` key in the detection config interpolates to
    `${dataset.ultralytics_yaml}`. Ultralytics resolves dataset YAMLs
    relative to its own `datasets_dir` setting unless the path is
    absolute, so we materialise it here to remove the ambiguity.
    """
    data_path = Path(cfg.detection.data)
    if not data_path.is_absolute():
        data_path = (Path.cwd() / data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"dataset YAML not found: {data_path}. Run "
            f"`python -m training.datasets.coda_to_yolo --split training` "
            f"and `--split validation` first."
        )
    return str(data_path)


def _resolve_project_dir(cfg: Any) -> str:
    """Make `detection.training.project` absolute.

    Ultralytics prepends its own `SETTINGS.runs_dir` (default `runs/detect`)
    to any *relative* `project` value, which lands checkpoints at e.g.
    `runs/detect/outputs/checkpoints/<name>/` instead of the
    `outputs/checkpoints/<name>/` we configured in `paths.output_root`.
    Forcing an absolute path bypasses that prepending and keeps
    everything under our project's standard `outputs/` tree.
    """
    project = Path(cfg.detection.training.project)
    if not project.is_absolute():
        project = (Path.cwd() / project).resolve()
    project.mkdir(parents=True, exist_ok=True)
    return str(project)


def main(cfg: Any) -> None:
    """Entry point — invoked by Hydra. DDP is delegated to ultralytics."""
    _ensure_ml_deps()
    _disable_ultralytics_wandb()

    from ultralytics import YOLO  # type: ignore

    run = _wandb_init(cfg)
    train_cfg = cfg.detection.training
    aug_cfg = cfg.detection.augmentation

    data_yaml = _resolve_data_yaml(cfg)
    LOG.info("dataset yaml: %s", data_yaml)
    LOG.info("weights: %s", cfg.detection.weights)

    model = YOLO(cfg.detection.weights)
    cb = _make_wandb_callback(run)
    model.add_callback("on_fit_epoch_end", cb)

    train_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "epochs": int(train_cfg.epochs),
        "batch": int(train_cfg.batch),
        "imgsz": int(train_cfg.imgsz),
        "optimizer": str(train_cfg.optimizer),
        "lr0": float(train_cfg.lr0),
        "lrf": float(train_cfg.lrf),
        "momentum": float(train_cfg.momentum),
        "weight_decay": float(train_cfg.weight_decay),
        "warmup_epochs": float(train_cfg.warmup_epochs),
        "warmup_momentum": float(train_cfg.warmup_momentum),
        "warmup_bias_lr": float(train_cfg.warmup_bias_lr),
        "cos_lr": bool(train_cfg.cos_lr),
        "freeze": int(train_cfg.freeze),
        "patience": int(train_cfg.patience),
        "cache": bool(train_cfg.cache),
        "amp": bool(train_cfg.amp),
        "workers": int(train_cfg.workers),
        "close_mosaic": int(train_cfg.close_mosaic),
        "project": _resolve_project_dir(cfg),
        "name": str(train_cfg.exp_name),
        "exist_ok": bool(train_cfg.exist_ok),
        "seed": int(train_cfg.seed),
        "val": bool(train_cfg.val),
        # Augmentation
        "mosaic": float(aug_cfg.mosaic),
        "mixup": float(aug_cfg.mixup),
        "copy_paste": float(aug_cfg.copy_paste),
        "hsv_h": float(aug_cfg.hsv_h),
        "hsv_s": float(aug_cfg.hsv_s),
        "hsv_v": float(aug_cfg.hsv_v),
        "degrees": float(aug_cfg.degrees),
        "translate": float(aug_cfg.translate),
        "scale": float(aug_cfg.scale),
        "shear": float(aug_cfg.shear),
        "perspective": float(aug_cfg.perspective),
        "flipud": float(aug_cfg.flipud),
        "fliplr": float(aug_cfg.fliplr),
    }
    if train_cfg.device is not None:
        # Ultralytics accepts either an int, a string like '0,3', or a list.
        train_kwargs["device"] = train_cfg.device

    LOG.info("launching ultralytics trainer (epochs=%d, imgsz=%d, batch=%d)",
             train_kwargs["epochs"], train_kwargs["imgsz"], train_kwargs["batch"])
    results = model.train(**train_kwargs)
    LOG.info("training finished — best.pt at %s",
             Path(train_kwargs["project"]) / train_kwargs["name"] / "weights" / "best.pt")

    if run is not None:
        # `results.results_dict` is a flat dict of final metrics. Push as
        # the run's summary so the W&B run card shows final mAP without
        # scrolling through epoch history.
        try:
            summary = getattr(results, "results_dict", None) or {}
            for k, v in summary.items():
                try:
                    run.summary[k] = float(v)
                except (TypeError, ValueError):
                    continue
        except Exception as exc:
            LOG.warning("wandb summary push failed: %s", exc)
        run.finish()


def _cli() -> None:
    """Hydra-decorated CLI wrapper, lazy-imported (mirrors train_detection)."""
    import hydra

    @hydra.main(version_base=None, config_path="../configs", config_name="base")
    def _run(cfg: Any) -> None:
        logging.basicConfig(level=logging.INFO)
        main(cfg)

    _run()


if __name__ == "__main__":
    _cli()
