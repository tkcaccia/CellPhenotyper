"""Original-to-crop support registration, without image inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
spec = importlib.util.spec_from_file_location("crop_support", ROOT / "bin/crop_grandqc_clean_mask.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def shift(full=(100, 80), box=(13, 7, 57, 46)):
    x0, y0, x1, y1 = box
    return {"full_size": dict(zip(("width", "height"), full)),
            "crop_bbox_xyxy": dict(zip(("x0", "y0", "x1", "y1"), box)),
            "crop_size": {"width": x1-x0, "height": y1-y0},
            "offset_crop_to_original": {"dx": x0, "dy": y0}}


@pytest.mark.parametrize("box", [(13, 7, 57, 46), (3, 1, 47, 77), (0, 0, 100, 80),
                                  (99, 79, 100, 80), (20, 20, 50, 40)])
def test_output_centres_sample_original_mask_without_stretching_source_slice(box):
    mask = (np.arange(8 * 10).reshape(8, 10) % 3 == 0).astype(np.uint8)
    declaration = shift(box=box)
    actual, metadata = module.crop_aligned_support(mask, declaration)
    x0, y0, x1, y1 = box
    h, w = actual.shape
    for y in range(h):
        for x in range(w):
            ox = x0 + (x+.5)*(x1-x0)/w
            oy = y0 + (y+.5)*(y1-y0)/h
            assert actual[y, x] == bool(mask[int(oy/10), int(ox/10)])
    assert metadata["output_origin_original_pixels_xy"] == [x0, y0]
    assert metadata["output_pixel_size_original_pixels_xy"] == [(x1-x0)/w, (y1-y0)/h]
    if box == (13, 7, 57, 46):
        # The previous ceil/floor slice had 5x5 pixels spanning [10,60)x[0,50),
        # while the declared crop spans [13,57)x[7,46). It must not be stretched.
        assert actual.shape == (4, 5)


@pytest.mark.parametrize("case", ["size", "outside", "offset", "fractional", "negative", "nan"])
def test_invalid_geometry_and_mask_fail_instead_of_becoming_tissue(case):
    declaration = shift()
    mask = np.ones((8, 10), np.float32)
    if case == "size": declaration["crop_size"]["width"] += 1
    if case == "outside": declaration["crop_bbox_xyxy"]["x0"] = -1
    if case == "offset": declaration["offset_crop_to_original"]["dx"] += 1
    if case == "fractional": declaration["full_size"]["width"] = 100.5
    if case == "negative": mask[0, 0] = -1
    if case == "nan": mask[0, 0] = np.nan
    with pytest.raises(ValueError):
        module.crop_aligned_support(mask, declaration)


def test_actual_cli_records_crop_aligned_transform_and_preserves_input(tmp_path):
    declaration = shift()
    mask = np.ones((8, 10), np.uint8)
    mask[:, 2] = 0
    source, destination = tmp_path / "source.tif", tmp_path / "crop.tif"
    tifffile.imwrite(source, mask, compression="deflate")
    before = source.read_bytes()
    (tmp_path / "shift.json").write_text(json.dumps(declaration))
    (tmp_path / "roi.json").write_text(json.dumps({"type": "Polygon", "coordinates": [
        [[0, 0], [44, 0], [44, 39], [0, 39], [0, 0]]]}))
    result = subprocess.run([sys.executable, str(ROOT / "bin/crop_grandqc_clean_mask.py"),
        "--mask", str(source), "--shift", str(tmp_path / "shift.json"), "--roi", str(tmp_path / "roi.json"),
        "--output", str(destination), "--summary", str(tmp_path / "summary.json")],
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    expected, geometry = module.crop_aligned_support(mask, declaration)
    np.testing.assert_array_equal(tifffile.imread(destination) != 0, expected)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert all(summary[key] == value for key, value in geometry.items())
    assert "finer gaps are not reconstructed" in summary["gap_precision"]
    assert source.read_bytes() == before


def polygon(outer, holes=()):
    return {"type": "Polygon", "coordinates": [outer, *holes]}


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_half_roi_excludes_column_whose_centre_is_outside():
    actual = module.rasterize_crop_roi_centres(polygon(rectangle(0, 0, 2, 4)), (4, 4), (4, 4))
    expected = np.zeros((4, 4), bool)
    expected[:, :2] = True
    np.testing.assert_array_equal(actual, expected)
    assert not actual[:, 2].any()  # x=2.5 must not be included by a rounded vertex.


@pytest.mark.parametrize("reverse", [False, True])
def test_roi_holes_and_boundary_centres_have_explicit_semantics(reverse):
    outer = rectangle(.5, .5, 5.5, 5.5)
    hole = rectangle(1.5, 1.5, 4.5, 4.5)
    rings = [outer[::-1], hole[::-1]] if reverse else [outer, hole]
    actual = module.rasterize_crop_roi_centres(polygon(rings[0], [rings[1]]), (6, 6), (6, 6))
    expected = np.ones((6, 6), bool)
    expected[1:5, 1:5] = False
    np.testing.assert_array_equal(actual, expected)


def test_diagonal_and_concave_roi_use_centre_inclusion():
    triangle = polygon([[0, 0], [4, 4], [0, 4], [0, 0]])
    actual = module.rasterize_crop_roi_centres(triangle, (4, 4), (4, 4))
    np.testing.assert_array_equal(actual, np.tri(4, dtype=bool))
    concave = polygon([[0, 0], [2, 0], [2, 4], [6, 4], [6, 0], [8, 0], [8, 6], [0, 6], [0, 0]])
    actual = module.rasterize_crop_roi_centres(concave, (6, 8), (8, 6))
    expected = np.ones((6, 8), bool)
    expected[:4, 2:6] = False
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_multipolygon_union_does_not_erase_other_polygons_inside_a_hole(reverse):
    parts = [
        [rectangle(0, 0, 6, 6), rectangle(1, 1, 5, 5)],
        [rectangle(2, 2, 4, 4)],
    ]
    if reverse:
        parts.reverse()
    roi = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
        "geometry": {"type": "MultiPolygon", "coordinates": parts}}]}
    actual = module.rasterize_crop_roi_centres(roi, (6, 6), (6, 6))
    expected = np.ones((6, 6), bool)
    expected[1:5, 1:5] = False
    expected[2:4, 2:4] = True
    np.testing.assert_array_equal(actual, expected)


def test_outside_roi_vertices_are_not_clamped_into_phantom_edge_tissue():
    outside = module.rasterize_crop_roi_centres(polygon(rectangle(-20, 0, -10, 4)), (4, 4), (4, 4))
    assert not outside.any()
    partial = module.rasterize_crop_roi_centres(polygon(rectangle(-20, 0, 2, 4)), (4, 4), (4, 4))
    expected = np.zeros((4, 4), bool)
    expected[:, :2] = True
    np.testing.assert_array_equal(partial, expected)


def test_roi_rasterization_allocates_only_low_resolution_mask_and_rows(monkeypatch):
    allocated_shapes = []
    original_zeros = np.zeros

    def tracked_zeros(shape, *args, **kwargs):
        allocated_shapes.append(shape)
        assert shape == (4, 5) or shape == 5
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(module.np, "zeros", tracked_zeros)
    actual = module.rasterize_crop_roi_centres(
        polygon(rectangle(0, 0, 22_000_000, 39_000_000)), (4, 5), (44_000_000, 39_000_000)
    )
    assert actual.shape == (4, 5)
    assert (4, 5) in allocated_shapes
    # Centre x=22M lies on the exterior boundary and is deliberately included.
    assert actual[:, :3].all() and not actual[:, 3:].any()


@pytest.mark.parametrize(
    "roi",
    [
        {"type": "FeatureCollection", "features": []},
        polygon([[0, 0], [2, 0], [2, 2], [0, 2]]),
        polygon([[0, 0], [1, 0], [2, 0], [0, 0]]),
        polygon([[0, 0], [float("nan"), 0], [2, 2], [0, 0]]),
        polygon([[0, 0], [True, 0], [2, 2], [0, 0]]),
    ],
)
def test_malformed_roi_rings_fail(roi):
    with pytest.raises(ValueError):
        module.rasterize_crop_roi_centres(roi, (4, 4), (4, 4))


def test_cli_roi_centres_use_crop_coordinates_for_an_offset_anisotropic_grid(tmp_path):
    declaration = shift()  # Original crop offset is (13,7), extent is 44x39.
    source, destination = tmp_path / "source.tif", tmp_path / "crop.tif"
    tifffile.imwrite(source, np.ones((8, 10), np.uint8), compression="deflate")
    (tmp_path / "shift.json").write_text(json.dumps(declaration))
    # Original-slide rectangle [20,38]x[13,32] after shifting into this crop.
    (tmp_path / "roi.json").write_text(json.dumps(polygon(rectangle(7, 6, 25, 25))))
    result = subprocess.run([
        sys.executable, str(ROOT / "bin/crop_grandqc_clean_mask.py"),
        "--mask", str(source), "--shift", str(tmp_path / "shift.json"), "--roi", str(tmp_path / "roi.json"),
        "--output", str(destination), "--summary", str(tmp_path / "summary.json"),
    ], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    # x centres: 4.4,13.2,22,30.8,39.6; y: 4.875,14.625,24.375,34.125.
    expected = np.zeros((4, 5), bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(tifffile.imread(destination) != 0, expected)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "crop_pixel_centres" in summary["roi_sampling"]
