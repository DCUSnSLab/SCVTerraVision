"""LightningDataModule wrapping our segmentation datasets.

Supports a single dataset (rugd / rellis3d) with separate train/val/test splits,
or a "combined" mode that concatenates multiple datasets for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from camera_perception.data.datasets import RUGDDataset, Rellis3DDataset
from camera_perception.data.transforms import (
    build_eval_transform,
    build_train_transform,
)

REGISTRY: dict[str, type[Dataset]] = {
    "rugd": RUGDDataset,
    "rellis3d": Rellis3DDataset,
}


class SegDataModule(L.LightningDataModule):
    def __init__(
        self,
        datasets: list[str],
        dataset_config_dir: str | Path,
        taxonomy_config: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        crop_h: int = 518,
        crop_w: int = 518,
        eval_h: int = 518,
        eval_w: int = 518,
        ignore_index: int = 255,
        root_overrides: dict[str, str] | None = None,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        # taxonomy_config is also stored on LitSegmenter; skip here to avoid
        # Lightning's hparams merge collision (str vs Path counts as different).
        self.save_hyperparameters(ignore=["root_overrides", "taxonomy_config", "dataset_config_dir"])
        self.dataset_names = list(datasets)
        self.dataset_config_dir = Path(dataset_config_dir)
        self.taxonomy_config = Path(taxonomy_config)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ignore_index = ignore_index
        self.root_overrides = root_overrides or {}

        self.train_tfm = build_train_transform(
            crop_h=crop_h, crop_w=crop_w, ignore_id=ignore_index
        )
        self.eval_tfm = build_eval_transform(
            height=eval_h, width=eval_w, ignore_id=ignore_index
        )
        self.pin_memory = pin_memory

        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None

    def _build(self, split: str, transform: Any) -> Dataset:
        parts: list[Dataset] = []
        for name in self.dataset_names:
            if name not in REGISTRY:
                raise ValueError(f"Unknown dataset: {name!r}")
            cfg_path = self.dataset_config_dir / f"{name}.yaml"
            ds = REGISTRY[name](
                dataset_config=cfg_path,
                taxonomy_config=self.taxonomy_config,
                split=split,
                transform=transform,
                root_override=self.root_overrides.get(name),
            )
            parts.append(ds)
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        self._train = self._build("train", self.train_tfm)
        # val/test may not exist for all datasets; allow failure to skip.
        try:
            self._val = self._build("val", self.eval_tfm)
        except RuntimeError:
            self._val = None
        try:
            self._test = self._build("test", self.eval_tfm)
        except RuntimeError:
            self._test = None

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        return self._loader(self._val, shuffle=False) if self._val else None

    def test_dataloader(self) -> DataLoader | None:
        return self._loader(self._test, shuffle=False) if self._test else None
