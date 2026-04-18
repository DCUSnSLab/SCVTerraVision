"""Shared fixtures: paths to real config files."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def taxonomy_config() -> Path:
    return REPO_ROOT / "configs" / "taxonomy" / "traversability_v1.yaml"


@pytest.fixture
def rugd_config() -> Path:
    return REPO_ROOT / "configs" / "datasets" / "rugd.yaml"


@pytest.fixture
def rellis3d_config() -> Path:
    return REPO_ROOT / "configs" / "datasets" / "rellis3d.yaml"
