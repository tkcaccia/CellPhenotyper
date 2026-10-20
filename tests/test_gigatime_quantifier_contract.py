"""CPU-only checks for virtual-score provenance and physical-grid cell mapping."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "bin/quantify_gigatime_intensity.py"
spec = importlib.util.spec_from_file_location("quantify_gigatime_contract", SOURCE)
quantify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantify)


class Image:
    height = 4
    width = 4

    def read_block(self, y0, y1, x0, x1):
        return np.full((1, y1 - y0, x1 - x0), .5, dtype=np.float32)


class Mask:
    def __init__(self, labels):
        self.labels = labels

    def read_block(self, y0, y1, x0, x1, shape):
        assert shape == (4, 4)
        return self.labels[y0:y1, x0:x1]


class OrthogonalArray:
    def __init__(self, values):
        self.values = values
        self.oindex = self

    def __getitem__(self, key):
        if np.isscalar(key[0]):
            return self.values[int(key[0])][np.ix_(key[1], key[2])]
        if len(key) == 3:
            return self.values[:, :, int(key[2])][np.ix_(key[0], key[1])]
        return self.values[np.ix_(key[0], key[1])]


def test_pixel_center_mapping_is_independent_of_block_partition():
    for values, axis in ((np.arange(6).reshape(1, 2, 3), 0),
                         (np.arange(6).reshape(2, 3, 1), 2)):
        reader = quantify.LazyMaskReader.__new__(quantify.LazyMaskReader)
        reader.arr = OrthogonalArray(values)
        reader.singleton_axis = axis
        reader.height, reader.width = 2, 3
        full = reader.read_block(0, 4, 0, 6, (4, 6))
        parts = np.block([[reader.read_block(y, y + 1, x, x + 1, (4, 6))
                           for x in range(6)] for y in range(4)])
        expected = np.repeat(np.repeat(np.arange(6).reshape(2, 3), 2, 0), 2, 1)
        np.testing.assert_array_equal(full, expected)
        np.testing.assert_array_equal(parts, expected)


def test_sparse_ids_and_virtual_score_metadata():
    labels = np.zeros((4, 4), dtype=np.uint32)
    labels[0, 0] = 7
    labels[3, 3] = 100_000_001
    a = quantify.quantify_blockwise(Image(), Mask(labels), ["CD68"],
                                    mask_name="nuclei", block_size=2)
    b = quantify.quantify_blockwise(Image(), Mask(labels), ["CD68"],
                                    mask_name="nuclei", block_size=4)
    assert a == b
    assert [row["label_id"] for row in a[1]] == [7, 100_000_001]
    assert [row["area_px"] for row in a[1]] == [1, 1]
    assert all(row["value_semantics"] == "uncalibrated_virtual_marker_score"
               and row["calibrated_probability"] is False
               and row["measured_protein_abundance"] is False for row in a[1])


def test_optional_nuclear_exclusion_keeps_cytoplasm_disjoint():
    cells = np.zeros((4, 4), dtype=np.uint32)
    cells[0, 0:2] = 7
    nuclei = np.zeros_like(cells)
    nuclei[0, 0] = 7
    result = quantify.quantify_blockwise(
        Image(), Mask(cells), ["CD68"], mask_name="cyto", block_size=2,
        exclude_reader=Mask(nuclei))
    assert result[1][0]["label_id"] == 7
    assert result[1][0]["area_px"] == 1
    nuclei[0, 0] = 8
    with pytest.raises(ValueError, match="labels disagree"):
        quantify.quantify_blockwise(
            Image(), Mask(cells), ["CD68"], mask_name="cyto", block_size=2,
            exclude_reader=Mask(nuclei))
