"""Smoke tests for training.train_yolo.

Module collection always succeeds (lazy imports). The actual training
smoke is `skipif ultralytics not installed`, so the dev env without GPU
keeps the rest of the test suite green.
"""

from __future__ import annotations

import importlib.util

import pytest

ULTRALYTICS_AVAILABLE = importlib.util.find_spec("ultralytics") is not None


def test_module_imports() -> None:
    """Module collection must work even without ultralytics installed."""
    import training.train_yolo as mod

    assert hasattr(mod, "main")
    assert hasattr(mod, "_make_wandb_callback")


def test_wandb_callback_no_run_returns_noop() -> None:
    """Callback factory must tolerate run=None for the wandb-disabled path."""
    from training.train_yolo import _make_wandb_callback

    cb = _make_wandb_callback(None)
    # Should accept any trainer-like object and return None
    assert cb(object()) is None


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
def test_disable_ultralytics_wandb() -> None:
    """SETTINGS update happens without exception when ultralytics is present."""
    from training.train_yolo import _disable_ultralytics_wandb

    _disable_ultralytics_wandb()
    from ultralytics.utils import SETTINGS  # type: ignore

    assert SETTINGS.get("wandb") is False
