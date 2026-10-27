import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("grandqc_mask", ROOT / "bin" / "grandqc_mask.py")
grandqc_mask = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(grandqc_mask)


def write_shift(path: Path, width: int, height: int) -> None:
    path.write_text(json.dumps({"crop_size": {"width": width, "height": height}}))


def write_rectangle_roi(path: Path, width: int, height: int) -> None:
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [width, 0], [width, height], [0, height], [0, 0]]],
            },
        }],
    }))


def test_crop_mask_uses_crop_coordinate_scale(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.tif"
    shift_path = tmp_path / "shift.json"
    tifffile.imwrite(mask_path, np.array([[255, 0], [0, 255]], dtype=np.uint8))
    write_shift(shift_path, 100, 100)
    mask = grandqc_mask.CropCleanTissueMask(mask_path, shift_path)
    assert mask.contains_xy(25, 25)
    assert not mask.contains_xy(75, 25)
    assert not mask.contains_xy(-1, 25)


def test_cell_payload_and_label_map_apply_the_same_filter(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.tif"
    shift_path = tmp_path / "shift.json"
    tifffile.imwrite(mask_path, np.array([[255, 0], [0, 255]], dtype=np.uint8))
    write_shift(shift_path, 100, 100)
    payload = {"cells": [{"id": "1", "centroid": [20, 20]}, {"id": "2", "centroid": [80, 20]}]}
    filtered, summary = grandqc_mask.filter_cell_payload(payload, mask_path, shift_path)
    assert [cell["id"] for cell in filtered["cells"]] == ["1"]
    assert summary["removed_outside_grandqc_clean_tissue"] == 1

    labels = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32)
    grandqc_mask.filter_labels_in_place(labels, {1}, labels.shape, block_size=2)
    np.testing.assert_array_equal(labels, np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint32))


def test_full_grandqc_mask_is_cropped_without_loading_a_full_resolution_mask(tmp_path: Path) -> None:
    source = tmp_path / "full_mask.tif"
    shift = tmp_path / "shift.json"
    roi = tmp_path / "roi.geojson"
    output = tmp_path / "crop_mask.tif"
    summary = tmp_path / "summary.json"
    tifffile.imwrite(source, np.arange(100, dtype=np.uint8).reshape(10, 10) > 20)
    shift.write_text(json.dumps({
        "crop_bbox_xyxy": {"x0": 2, "y0": 3, "x1": 8, "y1": 9},
        "full_size": {"width": 10, "height": 10},
        "crop_size": {"width": 6, "height": 6},
    }))
    write_rectangle_roi(roi, 6, 6)
    subprocess.run([
        sys.executable, str(ROOT / "bin" / "crop_grandqc_clean_mask.py"),
        "--mask", str(source), "--shift", str(shift), "--roi", str(roi), "--output", str(output),
        "--summary", str(summary),
    ], check=True)
    assert tifffile.imread(output).shape == (6, 6)
    payload = json.loads(summary.read_text())
    assert payload["analysis_crop_shape_yx"] == [6, 6]
    assert payload["mask_contract"] == "grandqc_clean_tissue_intersection_roi"


def test_crop_bbox_uses_grandqc_clean_tissue_intersected_with_roi() -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_analysis_crop", ROOT / "bin" / "prepare_analysis_crop.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    clean = np.zeros((10, 10), dtype=np.uint8)
    clean[2:8, 1:9] = 1
    roi = {
        "type": "Polygon",
        "coordinates": [[[50, 0], [100, 0], [100, 100], [50, 100], [50, 0]]],
    }
    bbox, metadata = module.tissue_roi_bbox(clean, roi, (100, 100), pad=0)
    assert bbox == (50, 20, 90, 80)
    assert metadata["tissue_roi_pixels_at_grandqc_scale"] == 24


def test_crop_mask_excludes_pixels_outside_shifted_roi(tmp_path: Path) -> None:
    source = tmp_path / "full_mask.tif"
    shift = tmp_path / "shift.json"
    roi = tmp_path / "roi.geojson"
    output = tmp_path / "crop_mask.tif"
    summary = tmp_path / "summary.json"
    tifffile.imwrite(source, np.ones((4, 4), dtype=np.uint8))
    shift.write_text(json.dumps({
        "crop_bbox_xyxy": {"x0": 0, "y0": 0, "x1": 4, "y1": 4},
        "full_size": {"width": 4, "height": 4},
        "crop_size": {"width": 4, "height": 4},
    }))
    roi.write_text(json.dumps({
        "type": "Polygon",
        "coordinates": [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]],
    }))
    subprocess.run([
        sys.executable, str(ROOT / "bin" / "crop_grandqc_clean_mask.py"),
        "--mask", str(source), "--shift", str(shift), "--roi", str(roi),
        "--output", str(output), "--summary", str(summary),
    ], check=True)
    result = tifffile.imread(output)
    assert np.all(result[:, :2] != 0)
    assert np.all(result[:, 3:] == 0)


def test_crop_writer_resolves_pyramidal_zarr_group_without_full_crop_ram(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("zarr")
    spec = importlib.util.spec_from_file_location(
        "prepare_analysis_crop_writer", ROOT / "bin" / "prepare_analysis_crop.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    source = tmp_path / "pyramid.ome.tif"
    output = tmp_path / "crop.tif"
    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    with tifffile.TiffWriter(source, bigtiff=True) as tif:
        tif.write(image, photometric="rgb", tile=(16, 16), subifds=1)
        tif.write(image[::2, ::2], photometric="rgb", tile=(16, 16), subfiletype=1)

    with tifffile.TiffFile(source) as tif:
        store = tif.series[0].aszarr()
        try:
            import zarr

            resolved = module.resolve_zarr_array(zarr.open(store, mode="r"))
            assert resolved.shape == image.shape
        finally:
            if hasattr(store, "close"):
                store.close()

    module.write_crop_with_tifffile_zarr(source, output, (7, 9, 45, 51), block_rows=11)
    np.testing.assert_array_equal(tifffile.imread(output), image[9:51, 7:45])
    assert not list(tmp_path.glob("*.crop-buffer-*.dat"))
