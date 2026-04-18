"""Lightning callbacks for training-time visualization.

`PredictionVizCallback` runs the model on a fixed slice of validation samples
every N epochs, builds an `image | gt | pred` tile, and pushes it to
TensorBoard via `trainer.logger.experiment.add_image`. Other loggers without
`add_image` are silently skipped (the callback becomes a no-op).
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch

from camera_perception.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from camera_perception.data.viz import colorize_mask


class PredictionVizCallback(L.Callback):
    def __init__(
        self,
        num_samples: int = 4,
        every_n_epochs: int = 1,
        tag_prefix: str = "predictions",
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if every_n_epochs <= 0:
            raise ValueError(f"every_n_epochs must be positive, got {every_n_epochs}")
        self.num_samples = num_samples
        self.every_n_epochs = every_n_epochs
        self.tag_prefix = tag_prefix
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        if not hasattr(experiment, "add_image"):
            return  # not a TensorBoard-style logger; silently skip

        dm = trainer.datamodule
        val_ds = getattr(dm, "_val", None)
        if val_ds is None or len(val_ds) == 0:
            return

        taxonomy = getattr(pl_module, "taxonomy", None)
        if taxonomy is None:
            return

        device = pl_module.device
        was_training = pl_module.training
        pl_module.eval()

        n = min(self.num_samples, len(val_ds))
        with torch.no_grad():
            for i in range(n):
                sample = val_ds[i]
                image = sample["image"]                          # (3, H, W) normalized
                target = sample["mask"].numpy()
                logits = pl_module(image.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1)[0].cpu().numpy()

                img_disp = (image.cpu() * self._std + self._mean).clamp(0.0, 1.0)
                img_disp = (img_disp.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                gt_color = colorize_mask(target, taxonomy)
                pred_color = colorize_mask(pred, taxonomy)
                tile = np.concatenate([img_disp, gt_color, pred_color], axis=1)

                experiment.add_image(
                    f"{self.tag_prefix}/sample_{i}",
                    tile,
                    trainer.current_epoch,
                    dataformats="HWC",
                )

        if was_training:
            pl_module.train()
