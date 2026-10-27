"""Physical geometry and label handoff checks for the Virchow2/Bora route."""
from pathlib import Path
import csv
import gzip
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
cv2 = pytest.importorskip("cv2")
pytest.importorskip("pyvips")

from run_virchow2_bora_wsi import (CalibratedSlide, load_mask_reader, mask_window,
                                   remap_unobserved_partial_edges, working_coarse)
import virchow2_runtime
import extract_virchow2_grid


def test_calibrated_wsi_window_preserves_physical_extent_and_coarse_ids(tmp_path: Path):
    rgb = np.full((500, 517, 3), (170, 100, 140), dtype=np.uint8)
    image = tmp_path / "crop.tif"
    tifffile.imwrite(image, rgb, photometric="rgb")
    sidecar = tmp_path / "resolution.json"
    sidecar.write_text(json.dumps({"source_mpp_x": .25, "source_mpp_y": .25}))
    grid = {"source_mpp_x": .25, "source_mpp_y": .25}
    slide = CalibratedSlide(image, sidecar, grid)
    windows = list(slide.windows(core_pixels=256, halo_pixels=128))
    assert len(windows) == 6
    coverage = np.zeros(rgb.shape[:2], dtype=np.uint8)
    coarse = np.zeros(rgb.shape[:2], dtype=np.uint16)
    coarse[:, :256] = 1
    coarse[:, 256:] = 2
    coarse_path = tmp_path / "coarse.tif"
    tifffile.imwrite(coarse_path, coarse)
    reader = load_mask_reader(coarse_path)
    for window in windows:
        native, working, mpp = slide.read(window)
        assert min(working.shape[:2]) >= 224
        assert native.shape[:2] == (window.height, window.width)
        np.testing.assert_allclose(
            np.asarray(working.shape[:2][::-1]) * mpp,
            np.asarray([window.width, window.height]) * slide.native_mpp_xy,
        )
        sampled = mask_window(reader, window, slide.shape_xy, binary=False)
        np.testing.assert_array_equal(sampled, coarse[window.y:window.y + window.height,
                                                      window.x:window.x + window.width])
        resized = working_coarse(sampled, working.shape[:2])
        assert set(np.unique(resized)).issubset({0, 1, 2})
        coverage[window.core_y:window.core_y + window.core_height,
                 window.core_x:window.core_x + window.core_width] += 1
    assert np.all(coverage == 1)


def test_calibration_mismatch_is_rejected(tmp_path: Path):
    image = tmp_path / "crop.tif"
    tifffile.imwrite(image, np.zeros((4, 4, 3), dtype=np.uint8), photometric="rgb")
    sidecar = tmp_path / "resolution.json"
    sidecar.write_text(json.dumps({"source_mpp": .25}))
    with pytest.raises(ValueError, match="disagrees"):
        CalibratedSlide(image, sidecar, {"source_mpp_x": .5, "source_mpp_y": .5})


def test_spatial_encoder_uses_disjoint_tiles_including_edge(monkeypatch):
    calls = []

    def fake_forward(_model, patches, _device):
        calls.extend(patches.copy())
        return np.zeros((len(patches), 261, 1280), dtype=np.float32)

    monkeypatch.setattr(virchow2_runtime, "_forward", fake_forward)
    rgb = np.arange(230 * 226 * 3, dtype=np.uint32).reshape(230, 226, 3).astype(np.uint8)
    field, report = virchow2_runtime.spatial_tokens_nonoverlap(
        object(), rgb, device="cpu", batch=2)
    assert len(calls) == 4
    np.testing.assert_array_equal(calls[0], rgb[:224, :224])
    np.testing.assert_array_equal(calls[1][:224, :2], rgb[:224, 224:])
    assert np.all(calls[1][:, 2:] == 255)
    assert field.shape == (17, 17, 1280)
    assert report["tile_overlap_px"] == 0


def test_grid_extractor_hands_one_embedding_per_disjoint_tile(tmp_path: Path, monkeypatch):
    image = tmp_path / "image.tif"
    rgb = np.full((450, 450, 3), 200, dtype=np.uint8)
    tifffile.imwrite(image, rgb, photometric="rgb")
    objects = tmp_path / "grid.csv"
    fields = ["label", "grid_row", "grid_col", "tile_x0", "tile_y0", "tile_x1",
              "tile_y1", "core_x0", "core_y0", "x", "y", "area_px"]
    with objects.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in range(3):
            for col in range(3):
                x, y = col * 224, row * 224
                writer.writerow(dict(label=row * 3 + col + 1, grid_row=row, grid_col=col,
                                     tile_x0=x, tile_y0=y, tile_x1=x + 224, tile_y1=y + 224,
                                     core_x0=x, core_y0=y, x=min(x + 112, 449),
                                     y=min(y + 112, 449), area_px=4))
    metadata = tmp_path / "grid.json"
    metadata.write_text(json.dumps(dict(tile_layout="nonoverlapping", tile_origin="top_left",
                                        context_overlap_source_px=0,
                                        extraction_tile_size_source_px=224,
                                        grid_stride_source_px=224, model_tile_size_px=224,
                                        target_mpp=.5, source_mpp_x=.5, source_mpp_y=.5,
                                        image_width_px=450, image_height_px=450,
                                        retained_units=9)))
    monkeypatch.setattr(extract_virchow2_grid, "load_model", lambda *_args, **_kw: object())
    monkeypatch.setattr(extract_virchow2_grid, "pooled_tiles",
                        lambda _model, batch, **_kw: np.ones((len(batch), 2560), np.float32))
    report = extract_virchow2_grid.run(image, objects, metadata, tmp_path,
                                       tmp_path / "features", batch=4, device="cpu")
    assert report["observations"] == 9
    assert report["tile_overlap_source_px"] == 0
    with gzip.open(tmp_path / "features" / "virchow2_embeddings_shard0000.csv.gz", "rt") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 10
    assert len(rows[0]) == 2565
    assert [int(row[0]) for row in rows[1:]] == list(range(1, 10))


def test_unsupported_partial_outer_tile_is_filled_from_observed_neighbor():
    coarse = np.ones((100, 100), dtype=np.uint8)
    coarse[:, 50:96] = 2
    coarse[:, 96:] = 3
    coarse[96:, :] = 3
    window = SimpleNamespace(x=0, y=0, width=100, height=100)
    fixed, audit = remap_unobserved_partial_edges(
        coarse, np.array([1, 2]), window, (100, 100), 32)
    assert audit["merged_partial_edge_groups"] == [3]
    assert 3 not in np.unique(fixed)
    assert np.all(fixed[:, 96:] > 0)
    assert np.array_equal(fixed[:96, :96], coarse[:96, :96])


def test_unsupported_interior_group_fails_closed():
    coarse = np.ones((100, 100), dtype=np.uint8)
    coarse[25:35, 25:35] = 2
    window = SimpleNamespace(x=0, y=0, width=100, height=100)
    with pytest.raises(ValueError, match="away from partial outer tiles"):
        remap_unobserved_partial_edges(coarse, np.array([1]), window, (100, 100), 32)
