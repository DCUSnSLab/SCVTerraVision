"""Tests for taxonomy loading and dataset label remapping."""

from __future__ import annotations

import numpy as np
import pytest

from camera_perception.data.taxonomy import (
    DatasetMapping,
    UnifiedTaxonomy,
    load_dataset_config,
)


def test_unified_taxonomy_loads(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    assert tax.name == "traversability_v1"
    assert tax.num_classes() == 6
    assert tax.ignore_id == 255
    # core class names exist
    for n in [
        "traversable_smooth",
        "traversable_grass",
        "non_traversable_terrain",
        "obstacle_static",
        "obstacle_dynamic",
        "sky",
    ]:
        assert tax.name_to_id(n) >= 0


def test_palette_shape(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    pal = tax.palette()
    assert pal.shape[1] == 3
    assert pal.dtype == np.uint8


def test_unknown_unified_name_raises(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    with pytest.raises(KeyError):
        tax.name_to_id("not_a_class")


@pytest.mark.parametrize("dataset_fixture", ["rugd_config", "rellis3d_config"])
def test_dataset_mapping_loads_and_covers_all_native_classes(request, taxonomy_config, dataset_fixture):
    cfg_path = request.getfixturevalue(dataset_fixture)
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(cfg_path, tax)

    raw = load_dataset_config(cfg_path)
    assert mapping.name == raw["name"]
    # Every native class must have a unified mapping (covered at load time, but assert anyway)
    for nc in raw["native_classes"]:
        assert nc["name"] in raw["to_unified"]


def test_remap_id_format(taxonomy_config, rellis3d_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rellis3d_config, tax)
    assert mapping.label_format == "id"

    # Synthetic 4x4 label using a few known native ids: 0(void), 3(grass), 17(person), 12(building)
    label = np.array(
        [
            [0, 0, 3, 3],
            [0, 3, 3, 17],
            [12, 12, 3, 17],
            [12, 12, 99, 0],  # 99 is out-of-range → ignore
        ],
        dtype=np.uint8,
    )
    out = mapping.remap(label)
    assert out.shape == (4, 4)
    assert out.dtype == np.int32

    grass_id = tax.name_to_id("traversable_grass")
    person_id = tax.name_to_id("obstacle_dynamic")
    building_id = tax.name_to_id("obstacle_static")

    assert out[0, 0] == tax.ignore_id  # void → ignore
    assert out[0, 2] == grass_id
    assert out[1, 3] == person_id
    assert out[2, 0] == building_id
    assert out[3, 2] == tax.ignore_id  # 99 out-of-range → ignore


def test_remap_rgb_format(taxonomy_config, rugd_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rugd_config, tax)
    assert mapping.label_format == "rgb"

    # Build a small RGB label using known RUGD colors:
    #   sky=(0,0,255), grass=(0,102,0), tree=(0,255,0), asphalt=(64,64,64)
    label = np.zeros((2, 4, 3), dtype=np.uint8)
    label[0, 0] = (0, 0, 255)     # sky
    label[0, 1] = (0, 102, 0)     # grass
    label[0, 2] = (0, 255, 0)     # tree
    label[0, 3] = (64, 64, 64)    # asphalt
    label[1, 0] = (255, 255, 0)   # vehicle
    label[1, 1] = (1, 2, 3)       # unknown color → ignore
    label[1, 2] = (0, 0, 0)       # void → ignore
    label[1, 3] = (108, 64, 20)   # dirt

    out = mapping.remap(label)
    assert out.shape == (2, 4)

    sky_id = tax.name_to_id("sky")
    grass_id = tax.name_to_id("traversable_grass")
    tree_id = tax.name_to_id("obstacle_static")
    smooth_id = tax.name_to_id("traversable_smooth")
    dyn_id = tax.name_to_id("obstacle_dynamic")

    assert out[0, 0] == sky_id
    assert out[0, 1] == grass_id
    assert out[0, 2] == tree_id
    assert out[0, 3] == smooth_id
    assert out[1, 0] == dyn_id
    assert out[1, 1] == tax.ignore_id
    assert out[1, 2] == tax.ignore_id
    assert out[1, 3] == smooth_id  # dirt → traversable_smooth
