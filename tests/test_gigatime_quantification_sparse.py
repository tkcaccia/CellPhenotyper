"""CPU-only contracts for native-coordinate per-cell virtual-marker tables."""
from __future__ import annotations

import importlib.util
import ast
from pathlib import Path
import sys
import types

import numpy as np
import pytest


try:
    import zarr  # noqa: F401
except ImportError:
    sys.modules["zarr"] = types.ModuleType("zarr")

spec = importlib.util.spec_from_file_location(
    "quantify_gigatime_intensity_sparse_test",
    Path(__file__).resolve().parents[1] / "bin" / "quantify_gigatime_intensity.py",
)
quantify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantify)


class Image:
    height = 4
    width = 4

    def read_block(self, y0, y1, x0, x1):
        values = np.full((1, y1 - y0, x1 - x0), .1, dtype=np.float32)
        if y0 == 0 and x0 == 0:
            values[0, 0, 0] = .9
        return values


class Mask:
    def __init__(self):
        self.labels = np.zeros((4, 4), dtype=np.uint32)
        self.labels[0, 0] = 7
        self.labels[3, 3] = 100_000_001

    def read_block(self, y0, y1, x0, x1, shape):
        assert shape == (4, 4)
        return self.labels[y0:y1, x0:x1]


def test_sparse_label_ids_and_block_partition_are_stable():
    a = quantify.quantify_blockwise(Image(), Mask(), ["CD68"], mask_name="nuclei", block_size=2)
    b = quantify.quantify_blockwise(Image(), Mask(), ["CD68"], mask_name="nuclei", block_size=4)
    assert [row["label_id"] for row in a[1]] == [7, 100_000_001]
    assert [row["area_px"] for row in a[1]] == [1, 1]
    assert [row["centroid_x_px"] for row in a[1]] == [0, 3]
    assert a == b


class OrthogonalArray:
    def __init__(self, values):
        self.values = values
        self.oindex = self

    def __getitem__(self, key):
        if len(key) == 2:
            return self.values[np.ix_(key[0], key[1])]
        if np.isscalar(key[0]):
            return self.values[int(key[0])][np.ix_(key[1], key[2])]
        return self.values[:, :, int(key[2])][np.ix_(key[0], key[1])]


def test_pixel_center_mapping_is_block_independent_and_keeps_singleton_axes():
    for values, axis in ((np.arange(6).reshape(1, 2, 3), 0),
                         (np.arange(6).reshape(2, 3, 1), 2)):
        reader = quantify.LazyMaskReader.__new__(quantify.LazyMaskReader)
        reader.arr = OrthogonalArray(values)
        reader.singleton_axis = axis
        reader.height, reader.width = 2, 3
        full = reader.read_block(0, 4, 0, 6, (4, 6))
        parts = np.block([[reader.read_block(y, y + 1, x, x + 1, (4, 6))
                           for x in range(6)] for y in range(4)])
        np.testing.assert_array_equal(full, parts)
        np.testing.assert_array_equal(full, np.repeat(np.repeat(np.arange(6).reshape(2, 3), 2, 0), 2, 1))


def test_real_tiff_zarr_singleton_plane_is_read_in_bounded_blocks(tmp_path: Path):
    pytest.importorskip("zarr")
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "mask.tif"
    tifffile.imwrite(path, np.arange(6, dtype=np.uint16).reshape(1, 2, 3),
                     metadata={"axes": "CYX"})
    with quantify.LazyMaskReader(str(path)) as reader:
        assert reader.arr.ndim == 3
        assert (reader.height, reader.width) == (2, 3)
        actual = reader.read_block(1, 2, 2, 4, (4, 6))
        np.testing.assert_array_equal(actual, np.asarray([[1, 1]], dtype=np.uint16))


def test_integrated_gigatime_quantifier_uses_compact_cell_ids(tmp_path: Path):
    zarr = pytest.importorskip("zarr")
    tifffile = pytest.importorskip("tifffile")
    source = Path(__file__).resolve().parents[1] / "bin" / "run_gigatime_on_crop.py"
    tree = ast.parse(source.read_text())
    wanted = {"LazyMaskReader", "_scan_labels", "TileQuantifier"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = dict(np=np, zarr=zarr, tifffile=tifffile, Path=Path,
                     csv=__import__("csv"), json=__import__("json"),
                     GIGATIME_VALUE_SEMANTICS={"value_type": "uncalibrated_virtual_marker_score"})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    mask = np.zeros((4, 4), dtype=np.uint32)
    mask[0, 0] = 7
    mask[3, 3] = 100_000_001
    mask_path = tmp_path / "integrated_mask.tif"
    tifffile.imwrite(mask_path, mask)
    quantifier = namespace["TileQuantifier"](mask_name="nuclei", mask_path=str(mask_path),
                                               target_shape=(4, 4), channel_names=["CD68"],
                                               block_size=2)
    try:
        assert len(quantifier.counts) == 2
        quantifier.accumulate_tile(0, 4, 0, 4, np.full((1, 4, 4), .5, dtype=np.float32))
        quantifier.write_outputs(tmp_path / "quantification", "specimen")
        with (tmp_path / "quantification" / "specimen_nuclei_gigatime_mean_intensity.csv").open() as stream:
            records = list(__import__("csv").DictReader(stream))
        assert [int(row["label_id"]) for row in records] == [7, 100_000_001]
        assert [int(row["area_px"]) for row in records] == [1, 1]
    finally:
        quantifier.close()
