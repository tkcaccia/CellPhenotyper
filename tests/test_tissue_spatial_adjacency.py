"""Synthetic image-only grid adjacency checks; no models or expert references."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import build_tissue_spatial_adjacency as module


def observation(rows=None):
    if rows is None:
        rows = [("001", 8.5, 10.5, 0, 0), ("NA", 22.5, 10.5, 0, 1)]
    return pd.DataFrame(rows, columns=module.OBSERVATION_COLUMNS)


def sources(tmp_path, image=None, support=None, suffix=""):
    if image is None:
        image = np.full((32, 48, 3), 128, dtype=np.uint8)
    if support is None:
        support = np.ones(image.shape[:2], dtype=np.uint8)
    image_path, support_path = tmp_path / f"image{suffix}.tif", tmp_path / f"support{suffix}.tif"
    tifffile.imwrite(image_path, image, photometric="rgb" if image.ndim == 3 else "minisblack",
                     tile=(16, 16), compression="deflate")
    tifffile.imwrite(support_path, support, photometric="minisblack", tile=(16, 16), compression="deflate")
    return image_path, support_path


def run(tmp_path, frame=None, image=None, support=None, **kwargs):
    paths = sources(tmp_path, image, support)
    return module.build_adjacency(observation() if frame is None else frame, *paths,
                                  **dict({"radius_um": 50}, **kwargs))


def test_literal_source_order_and_cardinal_only_graph_preserve_sources(tmp_path):
    frame = observation([("NA", 15.5, 15.5, 1, 1), ("001", 5.5, 5.5, 0, 0),
                         ("7", 5.5, 15.5, 1, 0), ("0", 15.5, 5.5, 0, 1)])
    original = frame.copy(deep=True)
    paths = sources(tmp_path)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    edges, vertices, metadata = module.build_adjacency(frame, *paths, image_mpp=(1., 2.), radius_um=20)
    assert edges[["source_index", "target_index"]].values.tolist() == [[1, 3], [1, 4], [2, 3], [2, 4]]
    assert edges.distance_um.tolist() == [10., 20., 20., 10.]
    np.testing.assert_allclose(edges.boundary_strength, 0, atol=1e-14)
    np.testing.assert_array_equal(edges.boundary_weight, np.ones(4))
    assert vertices.label.tolist() == ["NA", "001", "7", "0"]
    assert vertices.source_index.tolist() == [1, 2, 3, 4]
    pd.testing.assert_frame_equal(vertices[module.OBSERVATION_COLUMNS], original)
    pd.testing.assert_frame_equal(frame, original)
    assert vertices.in_tissue_support.all() and vertices.status.eq("supported").all()
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == hashes
    assert metadata["cardinal_candidates"] == 4
    assert not metadata["expert_annotations_used"] and not metadata["model_inference_used"]
    json.dumps(metadata, allow_nan=False)


def test_anisotropic_same_extent_support_and_radius_cap(tmp_path):
    image = np.full((32, 48, 3), 128, dtype=np.uint8)
    support = np.ones((8, 16), dtype=np.uint8)
    support[2, 7] = 0  # x=22.5,y=10.5 maps to support pixel (7,2).
    frame = observation([("a", 8.5, 10.5, 0, 0), ("b", 22.5, 10.5, 0, 1),
                         ("c", 8.5, 20.5, 1, 0)])
    edges, vertices, metadata = run(tmp_path, frame, image, support, image_mpp=(2., 3.), radius_um=29.9)
    assert edges.empty and list(edges) == module.EDGE_COLUMNS
    assert vertices.in_tissue_support.tolist() == [True, False, True]
    assert vertices.status.tolist() == ["supported", "unsupported", "supported"]
    assert metadata["support_mpp_xy"] == [6., 12.]
    assert metadata["unsupported_endpoint_rejections"] == 1
    assert metadata["radius_rejections"] == 1
    assert metadata["observation_count"] == 3


@pytest.mark.parametrize("gap_kind", ["thin_gap", "corner", "grid_line"])
def test_exact_segment_rejects_gaps_independent_of_appearance_sampling(tmp_path, gap_kind):
    support = np.ones((32, 48), dtype=np.uint8)
    frame = observation()
    if gap_kind == "thin_gap":
        support[:, 16] = 0
    elif gap_kind == "corner":
        support[:] = 0
        support[1, 1] = support[2, 2] = 1
        frame = observation([("a", 1.5, 1.5, 0, 0), ("b", 2.5, 2.5, 0, 1)])
    else:
        support[9, :] = 0
        frame = observation([("a", 8.5, 10., 0, 0), ("b", 22.5, 10., 0, 1)])
    edges, vertices, metadata = run(tmp_path, frame, support=support, path_step_um=100.)
    assert edges.empty and vertices.in_tissue_support.all()
    assert metadata["support_segment_rejections"] == 1
    assert metadata["memory"]["descriptors_computed"] == 0


@pytest.mark.parametrize("grid", [[(0, 0), (0, 2)], [(0, 0), (1, 1)]])
def test_missing_tiles_and_grid_diagonals_never_become_candidates(tmp_path, grid):
    frame = observation()
    frame[["grid_row", "grid_col"]] = grid
    edges, vertices, metadata = run(tmp_path, frame)
    assert edges.empty and len(vertices) == 2 and metadata["cardinal_candidates"] == 0


def test_boundary_strength_matches_declared_od_and_weight(tmp_path):
    image = np.full((32, 48, 3), 200, dtype=np.uint8)
    image[:, 16:, :] = 160
    edges, _, metadata = run(tmp_path, image=image, descriptor_radius_um=.25, path_step_um=1.)
    epsilon = 1/255
    expected = np.sqrt(3) * abs(math.log((200/255+epsilon)/(1+epsilon)) - math.log((160/255+epsilon)/(1+epsilon)))
    assert edges.boundary_strength.iloc[0] == pytest.approx(expected, abs=1e-14)
    assert edges.boundary_weight.iloc[0] == pytest.approx(math.exp(-.5*(expected/.15)**2), abs=1e-14)
    assert metadata["descriptor"]["white_level"] == 255


def test_identical_endpoints_do_not_hide_sampled_dark_stripe(tmp_path):
    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    image[:, 16, :] = 0
    edges, _, _ = run(tmp_path, image=image, descriptor_radius_um=.25, path_step_um=1.)
    assert edges.boundary_strength.iloc[0] == pytest.approx(np.sqrt(3)*math.log(256))
    assert edges.boundary_weight.iloc[0] == 0


@pytest.mark.parametrize("encoding", ["uint16", "float64", "float32"])
def test_equivalent_declared_image_encodings(tmp_path, encoding):
    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    expected = module.build_adjacency(observation(), *sources(tmp_path, image, suffix="base"), radius_um=30,
                                      image_mpp=(.5, .8), path_step_um=.75)
    converted = image.astype(np.uint16)*257 if encoding == "uint16" else image.astype(encoding)/255
    actual = module.build_adjacency(observation(), *sources(tmp_path, converted, suffix=encoding), radius_um=30,
                                    image_mpp=(.5, .8), path_step_um=.75)
    pd.testing.assert_frame_equal(actual[1], expected[1])
    pd.testing.assert_frame_equal(actual[0], expected[0], check_exact=False,
        rtol=1e-6 if encoding == "float32" else 1e-13, atol=1e-9 if encoding == "float32" else 1e-14)
    assert actual[2]["descriptor"]["image_dtype"] == encoding


def test_fractional_pixel_area_square_mean_with_anisotropic_calibration(tmp_path):
    image = np.arange(3*4*3, dtype=np.uint8).reshape(3, 4, 3)*6
    path, _ = sources(tmp_path, image)
    with module.RasterReader(path) as reader:
        appearance = module._Appearance(reader, (2., 1.), 1., 1)
        actual = appearance.mean_od([1.25, 1.75])
        weights = np.array([.25, 1., .75])[:, None] * np.array([.25, .75])[None, :]
        od = -np.log((image[:, :2].astype(float)/255+1/255)/(1+1/255))
        expected = (weights[:, :, None]*od).sum(axis=(0, 1))/weights.sum()
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        assert appearance.max_window_pixels == 1


def test_path_spacing_includes_endpoints_and_is_physical(tmp_path, monkeypatch):
    seen, original = [], module._Appearance.mean_od
    def recording(self, xy):
        seen.append(np.asarray(xy).copy())
        return original(self, xy)
    monkeypatch.setattr(module._Appearance, "mean_od", recording)
    frame = observation([("a", 8.5, 5.5, 0, 0), ("b", 22.5, 10.5, 0, 1)])
    _, _, metadata = run(tmp_path, frame, image_mpp=(.7, 1.3), path_step_um=2.3)
    np.testing.assert_array_equal(seen[0], [8.5, 5.5])
    np.testing.assert_array_equal(seen[-1], [22.5, 10.5])
    distances = np.linalg.norm(np.diff(seen, axis=0)*[.7, 1.3], axis=1)
    assert distances.max() <= 2.3
    assert len(seen) == metadata["path_samples"]
    assert metadata["max_actual_path_step_um"] == pytest.approx(distances.max())


def test_large_descriptor_stays_tile_bounded_without_whole_image_read(tmp_path, monkeypatch):
    image = np.full((256, 256, 3), 128, dtype=np.uint8)
    frame = observation([("a", 20.5, 20.5, 0, 0), ("b", 230.5, 20.5, 0, 1)])
    calls, original = [], module.RasterReader.window
    def bounded(self, x0, y0, x1, y1):
        calls.append((x1-x0, y1-y0))
        assert x1-x0 <= 17 and y1-y0 <= 17
        return original(self, x0, y0, x1, y1)
    monkeypatch.setattr(module.RasterReader, "window", bounded)
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: pytest.fail("Full raster decoder used"))
    edges, _, metadata = run(tmp_path, frame, image, radius_um=250, descriptor_radius_um=100,
                             path_step_um=150, tile_size=17)
    assert len(edges) == 1 and calls
    assert metadata["memory"]["max_rgb_window_pixels"] <= 17*17
    assert metadata["memory"]["max_support_window_pixels"] <= 17*17


@pytest.mark.parametrize("field,value", [("radius_um", 0), ("radius_um", np.inf),
    ("descriptor_radius_um", -1), ("path_step_um", np.nan), ("boundary_sigma", 0),
    ("boundary_sigma", True), ("tile_size", 1.5), ("tile_size", True), ("tile_size", 0),
    ("image_mpp", [1]), ("image_mpp", [1, 0]), ("image_mpp", [1, np.inf]), ("image_mpp", [True, 1])])
def test_malformed_parameters_fail_before_reading_sources(field, value):
    with pytest.raises(ValueError):
        module.build_adjacency(observation(), "absent-image", "absent-support", **dict({"radius_um": 30}, **{field: value}))


@pytest.mark.parametrize("field,value", [("label", "001"), ("label", ""), ("label", " a"),
    ("label", "a\n"), ("label", "a\x00b"), ("label", None), ("label", 5),
    ("grid_row", -1), ("grid_col", 1.5), ("grid_col", True), ("grid_col", 0),
    ("x", np.nan), ("x", np.inf), ("x", "9"), ("y", True)])
def test_malformed_observations_fail_before_reading_sources(field, value):
    frame = observation().astype({field: object})
    frame.loc[1, field] = value
    with pytest.raises(ValueError):
        module.build_adjacency(frame, "absent-image", "absent-support", radius_um=30)


@pytest.mark.parametrize("fault", ["empty", "missing", "duplicated_column"])
def test_invalid_table_schema(fault):
    frame = observation()
    if fault == "empty":
        frame = frame.iloc[:0]
    elif fault == "missing":
        frame = frame.drop(columns="label")
    else:
        frame.columns = ["label", "x", "x", "grid_row", "grid_col"]
    with pytest.raises(ValueError):
        module.build_adjacency(frame, "absent-image", "absent-support", radius_um=30)


@pytest.mark.parametrize("value", [-.1, 48.])
def test_out_of_bounds_coordinate_rejected_not_silently_excluded(tmp_path, value):
    frame = observation()
    frame.loc[0, "x"] = value
    with pytest.raises(ValueError, match="bounds"):
        run(tmp_path, frame)


@pytest.mark.parametrize("image", [np.ones((32, 48), dtype=np.uint8), np.ones((32, 48, 4), dtype=np.uint8),
    np.ones((32, 48, 3), dtype=np.int16), np.ones((32, 48, 3), dtype=np.uint32)])
def test_ambiguous_image_shape_or_integer_encoding_is_rejected(tmp_path, image):
    with pytest.raises(ValueError, match="RGB|dtype"):
        run(tmp_path, image=image)


@pytest.mark.parametrize("value", [np.nan, np.inf, -.001, 1.001])
def test_sampled_float_rgb_must_be_finite_and_normalized(tmp_path, value):
    image = np.full((32, 48, 3), .5, dtype=np.float32)
    image[10, 8, 0] = value
    with pytest.raises(ValueError, match="Sampled RGB"):
        run(tmp_path, image=image)


def test_unsampled_values_are_not_claimed_validated(tmp_path):
    image = np.full((32, 48, 3), .5, dtype=np.float32)
    image[31, 47, 0] = np.nan
    edges, _, metadata = run(tmp_path, image=image)
    assert len(edges) == 1
    assert "unsampled image pixels are not scanned" in metadata["descriptor"]["value_validation_scope"]


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_invalid_sampled_support_rejected(tmp_path, value):
    support = np.ones((32, 48), dtype=np.float32)
    support[10, 8] = value
    with pytest.raises(ValueError, match="Support values"):
        run(tmp_path, support=support)


def test_duplicate_grid_centres_are_a_coordinate_conflict(tmp_path):
    frame = observation()
    frame.loc[1, ["x", "y"]] = frame.loc[0, ["x", "y"]].values
    with pytest.raises(ValueError, match="duplicate x/y"):
        run(tmp_path, frame)
