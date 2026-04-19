"""Tests for taxonomy loading and dataset label remapping (v2)."""

from __future__ import annotations

import numpy as np
import pytest

from camera_perception.data.taxonomy import (
    DatasetMapping,
    UnifiedTaxonomy,
    load_dataset_config,
)


def test_unified_taxonomy_loads_v2(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    assert tax.name == "traversability_v2"
    assert tax.version == "v2"
    assert tax.num_classes() == 18
    assert tax.ignore_id == 255

    # Representative class names exist across all groups.
    for n in [
        "road_paved",
        "path_unpaved",
        "grass_low",
        "furrow",
        "water",
        "mud",
        "rough_terrain",
        "building_wall",
        "vertical_pole",
        "tree",
        "barrier_fence",
        "polytunnel",
        "person",
        "rider",
        "bicycle",
        "vehicle",
        "animal",
        "sky",
    ]:
        assert tax.name_to_id(n) >= 0


def test_v2_groups(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    assert set(tax.groups) == {
        "TRAVERSABLE",
        "NON_TRAVERSABLE_TERRAIN",
        "OBSTACLE_STATIC",
        "OBSTACLE_DYNAMIC",
        "CONTEXT",
    }
    trav = set(tax.class_ids_in_group("TRAVERSABLE"))
    assert trav == {
        tax.name_to_id("road_paved"),
        tax.name_to_id("path_unpaved"),
        tax.name_to_id("grass_low"),
        tax.name_to_id("furrow"),
    }
    dyn = set(tax.class_ids_in_group("OBSTACLE_DYNAMIC"))
    assert dyn == {
        tax.name_to_id(n) for n in ("person", "rider", "bicycle", "vehicle", "animal")
    }


def test_v2_rollup_to_v1(taxonomy_config, taxonomy_v1_config):
    v2 = UnifiedTaxonomy.load(taxonomy_config)
    v1 = UnifiedTaxonomy.load(taxonomy_v1_config)
    lut = v2.rollup_lut(v1)

    # All traversable v2 classes roll up to v1 traversable_smooth or grass
    assert lut[v2.name_to_id("road_paved")] == v1.name_to_id("traversable_smooth")
    assert lut[v2.name_to_id("path_unpaved")] == v1.name_to_id("traversable_smooth")
    assert lut[v2.name_to_id("grass_low")] == v1.name_to_id("traversable_grass")
    # All non-traversable terrain v2 classes -> v1 non_traversable_terrain
    assert lut[v2.name_to_id("water")] == v1.name_to_id("non_traversable_terrain")
    assert lut[v2.name_to_id("mud")] == v1.name_to_id("non_traversable_terrain")
    # All obstacle_dynamic v2 classes -> v1 obstacle_dynamic
    for name in ("person", "rider", "bicycle", "vehicle", "animal"):
        assert lut[v2.name_to_id(name)] == v1.name_to_id("obstacle_dynamic")
    # ignore preserved
    assert lut[v2.ignore_id] == v1.ignore_id


def test_palette_shape(taxonomy_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    pal = tax.palette()
    assert pal.shape[1] == 3
    assert pal.dtype == np.uint8
    # palette must include all 19 classes + ignore slot
    assert pal.shape[0] >= tax.ignore_id + 1


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
    for nc in raw["native_classes"]:
        assert nc["name"] in raw["to_unified"]


def test_rellis_ignore_if_absent_propagates(taxonomy_config, rellis3d_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rellis3d_config, tax)
    # RELLIS-3D declares these as absent.
    for name in ("rider", "bicycle", "animal", "furrow", "polytunnel"):
        assert tax.name_to_id(name) in mapping.absent_class_ids
    # RELLIS-3D labels these explicitly.
    for name in ("person", "vehicle", "tree", "grass_low", "mud", "water"):
        assert tax.name_to_id(name) not in mapping.absent_class_ids

    mask = mapping.present_class_mask()
    assert mask.shape == (tax.num_classes(),)
    assert mask.dtype == bool
    assert not mask[tax.name_to_id("bicycle")]
    assert mask[tax.name_to_id("person")]


def test_rugd_ignore_if_absent_includes_rider(taxonomy_config, rugd_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rugd_config, tax)
    # RUGD labels bicycle object but not rider (mounted person).
    assert tax.name_to_id("rider") in mapping.absent_class_ids
    assert tax.name_to_id("bicycle") not in mapping.absent_class_ids


def test_remap_id_format(taxonomy_config, rellis3d_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rellis3d_config, tax)
    assert mapping.label_format == "id"

    # Synthetic 4x4 label using known native ids:
    # 0=void->ignore, 3=grass->grass_low, 17=person->person, 12=building->building_wall
    label = np.array(
        [
            [0, 0, 3, 3],
            [0, 3, 3, 17],
            [12, 12, 3, 17],
            [12, 12, 99, 0],  # 99 out-of-range -> ignore
        ],
        dtype=np.uint8,
    )
    out = mapping.remap(label)
    assert out.shape == (4, 4)
    assert out.dtype == np.int32

    grass_id = tax.name_to_id("grass_low")
    person_id = tax.name_to_id("person")
    building_id = tax.name_to_id("building_wall")

    assert out[0, 0] == tax.ignore_id
    assert out[0, 2] == grass_id
    assert out[1, 3] == person_id
    assert out[2, 0] == building_id
    assert out[3, 2] == tax.ignore_id  # 99 -> ignore


def test_remap_rgb_format(taxonomy_config, rugd_config):
    tax = UnifiedTaxonomy.load(taxonomy_config)
    mapping = DatasetMapping.load(rugd_config, tax)
    assert mapping.label_format == "rgb"

    # Build a small RGB label using known RUGD colors.
    #   sky=(0,0,255)->sky, grass=(0,102,0)->grass_low, tree=(0,255,0)->tree
    #   asphalt=(64,64,64)->road_paved, vehicle=(255,255,0)->vehicle
    #   bicycle=(0,255,128)->bicycle, dirt=(108,64,20)->path_unpaved
    label = np.zeros((2, 4, 3), dtype=np.uint8)
    label[0, 0] = (0, 0, 255)     # sky
    label[0, 1] = (0, 102, 0)     # grass -> grass_low
    label[0, 2] = (0, 255, 0)     # tree
    label[0, 3] = (64, 64, 64)    # asphalt -> road_paved
    label[1, 0] = (255, 255, 0)   # vehicle
    label[1, 1] = (1, 2, 3)       # unknown color -> ignore
    label[1, 2] = (0, 0, 0)       # void -> ignore
    label[1, 3] = (108, 64, 20)   # dirt -> path_unpaved

    out = mapping.remap(label)
    assert out.shape == (2, 4)

    assert out[0, 0] == tax.name_to_id("sky")
    assert out[0, 1] == tax.name_to_id("grass_low")
    assert out[0, 2] == tax.name_to_id("tree")
    assert out[0, 3] == tax.name_to_id("road_paved")
    assert out[1, 0] == tax.name_to_id("vehicle")
    assert out[1, 1] == tax.ignore_id
    assert out[1, 2] == tax.ignore_id
    assert out[1, 3] == tax.name_to_id("path_unpaved")


def test_rollup_applies_elementwise(taxonomy_config, taxonomy_v1_config):
    v2 = UnifiedTaxonomy.load(taxonomy_config)
    v1 = UnifiedTaxonomy.load(taxonomy_v1_config)
    lut = v2.rollup_lut(v1)

    mask_v2 = np.array(
        [
            [v2.name_to_id("road_paved"), v2.name_to_id("grass_low"), v2.ignore_id],
            [v2.name_to_id("person"), v2.name_to_id("sky"), v2.name_to_id("water")],
        ],
        dtype=np.int64,
    )
    mask_v1 = lut[mask_v2]

    assert mask_v1[0, 0] == v1.name_to_id("traversable_smooth")
    assert mask_v1[0, 1] == v1.name_to_id("traversable_grass")
    assert mask_v1[0, 2] == v1.ignore_id
    assert mask_v1[1, 0] == v1.name_to_id("obstacle_dynamic")
    assert mask_v1[1, 1] == v1.name_to_id("sky")
    assert mask_v1[1, 2] == v1.name_to_id("non_traversable_terrain")
