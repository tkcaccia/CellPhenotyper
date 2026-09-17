import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from assemble_spatial_cell_profiles import assemble, sample_domain_grid
from build_cell_profiles import build_profiles, parser
from cell_profile_io import sha256_file


def fixture(tmp_path, *, full_size=None, origin=(100, 200)):
    objects = tmp_path / "objects.csv"
    pd.DataFrame({"label": [2, 1, 3], "x": [7, 2, 5], "y": [3, 3, 3], "xmin": [6, 1, 4], "ymin": [2]*3, "xmax": [8, 3, 6], "ymax": [4]*3}).to_csv(objects, index=False)
    shift = tmp_path / "shift.json"
    geometry = {"crop_size": {"width": 10, "height": 8}, "offset_crop_to_original": {"dx": origin[0], "dy": origin[1]}}
    if full_size:
        geometry["full_size"] = full_size
    shift.write_text(json.dumps(geometry))
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    profile = tmp_path / "base"
    build_profiles(parser().parse_args(["--objects", str(objects), "--sample-id", "s", "--shift", str(shift), "--resolution-json", str(resolution), "--outdir", str(profile)]))
    support = np.ones((8, 10), dtype=np.uint8)
    support[:, 4:6] = 0
    mask = tmp_path / "support.tif"
    tifffile.imwrite(mask, support)
    domain = tmp_path / "domain.tif"
    tifffile.imwrite(domain, np.where(support, np.where(np.indices(support.shape)[1] < 5, 1, 2), 0).astype(np.uint8))
    return profile, mask, shift, resolution, domain


def test_complete_assembly_preserves_order_masks_and_missing_cells(tmp_path):
    profile, mask, shift, resolution, domain = fixture(tmp_path)
    out = tmp_path / "spatial"
    cells, manifest = assemble(profile, mask, shift, resolution, out, domain_mask=domain, radii_um=(10.,))
    assert cells.cell_id.tolist() == ["2", "1", "3"]
    assert cells.x_um.tolist() == [53.5, 51., 52.5]
    assert cells.in_tissue_support.tolist() == [True, True, False]
    assert cells.niche_id.isna().all()
    assert cells.niche_status.iloc[-1] == "outside_tissue_support"
    graph = sparse.load_npz(out / "neighborhood_graph_10um.npz")
    assert graph.nnz == 0  # Nearby cells on opposite sides are not neighbours.
    assert manifest["neighborhoods"]["tissue_domain_count_is_independent"]
    summary = json.loads((out / "neighborhood_summary.json").read_text())
    assert "feature_rows.csv" in summary["graph_contract"]
    assert "graph_cells.csv" not in summary["graph_contract"]
    assert summary["graph_axes"] == {"path": "feature_rows.csv", "sha256": sha256_file(out / "feature_rows.csv"),
        "count": 3, "keys": ["sample_id", "cell_id", "cell_uid"], "same_order_on_both_axes": True}
    assert pd.read_csv(out / summary["graph_axes"]["path"], dtype=str).cell_id.tolist() == cells.cell_id.tolist()
    assert (profile / "cell_profiles_manifest.json").exists()
    with pytest.raises(FileExistsError):
        assemble(profile, mask, shift, resolution, out)


def test_assembly_rejects_coordinate_and_content_mismatch(tmp_path):
    profile, mask, shift, resolution, domain = fixture(tmp_path)
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .6, "mpp_y": .6}))
    with pytest.raises(ValueError, match="coordinate frames"):
        assemble(profile, mask, shift, resolution, tmp_path / "bad")


@pytest.mark.parametrize("axis,delta", [("dx", 1), ("dy", -1)])
def test_large_origin_rejects_one_native_pixel_frame_shift(tmp_path, axis, delta):
    profile, mask, shift, resolution, domain = fixture(tmp_path, origin=(200000, 200000))
    geometry = json.loads(shift.read_text())
    geometry["offset_crop_to_original"][axis] += delta
    shift.write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="coordinate frames"):
        assemble(profile, mask, shift, resolution, tmp_path / "bad",
            native_support_mask=mask, native_support_coordinates="crop_pixels")
    assert not (tmp_path / "bad").exists()


def test_label_sampling_uses_nearest_centres(tmp_path):
    path = tmp_path / "domains.tif"
    image = np.arange(32, dtype=np.uint8).reshape(4, 8)
    tifffile.imwrite(path, image, compression="deflate")
    np.testing.assert_array_equal(sample_domain_grid(path, (2, 4)), image[1::2, 1::2])


def native_fixture(tmp_path):
    profile, coarse_path, shift, resolution, domain = fixture(tmp_path, full_size={"width": 120, "height": 220})
    native = np.ones((8, 10), np.uint8)
    native[:, 4] = 0
    native_path = tmp_path / "native_crop.tif"
    tifffile.imwrite(native_path, native, tile=(16, 16), compression="deflate")
    # The one-pixel gap is genuinely absent from the sampled support source.
    coarse = native[1::2, 1::2]
    assert coarse.all()
    tifffile.imwrite(coarse_path, coarse, compression="deflate")
    full = np.ones((220, 120), np.uint8)
    full[200:208, 100:110] = native
    full_path = tmp_path / "native_original.tif"
    tifffile.imwrite(full_path, full, tile=(16, 16), compression="deflate")
    return profile, coarse_path, native_path, full_path, shift, resolution, domain


def test_native_support_override_preserves_valid_cells_and_blocks_erased_thin_gap(tmp_path):
    profile, coarse, native, full, shift, resolution, domain = native_fixture(tmp_path)
    before = {path: sha256_file(path) for path in profile.rglob("*") if path.is_file()}
    old, _ = assemble(profile, coarse, shift, resolution, tmp_path / "coarse_only", radii_um=(10.,))
    old_graph = sparse.load_npz(tmp_path / "coarse_only/neighborhood_graph_10um.npz")
    assert old_graph.nnz == 6  # Demonstrates the missing-input information, not a native guarantee.
    exact, manifest = assemble(profile, coarse, shift, resolution, tmp_path / "native",
        native_support_mask=native, native_support_coordinates="crop_pixels", domain_mask=domain,
        max_support_pixels=2, support_tile_size=2, support_cache_tiles=2, radii_um=(10.,))
    graph = sparse.load_npz(tmp_path / "native/neighborhood_graph_10um.npz")
    assert exact.cell_uid.tolist() == old.cell_uid.tolist()
    assert exact.cell_id.tolist() == ["2", "1", "3"] and exact.in_tissue_support.all()
    assert graph.nnz == 2 and set(graph[0].indices) == {2}
    assert graph[0, 2] == 1.0 and not graph[1].nnz
    assert exact.r10um_neighbor_count.tolist() == [1, 0, 1]
    assert exact.r10um_tissue_area_um2.tolist() == [18., 18., 18.]
    record = manifest["neighborhoods"]["graph_support"]
    assert record["explicit_native_support"] and record["source_grid_matches_native_crop"]
    assert record["graph_and_density_resampling"] == "none" and record["sha256"] == sha256_file(native)
    assert record["mpp_xy"] == [.5, .5] and record["origin_um_xy"] == [50., 100.]
    summary = json.loads((tmp_path / "native/neighborhood_summary.json").read_text())
    assert np.prod(summary["domain_analysis_geometry"]["s"]["shape_yx"]) <= 2
    assert before == {path: sha256_file(path) for path in before}


def test_existing_native_support_argument_exceeds_simulated_guard_without_sampling(tmp_path):
    profile, coarse, native, full, shift, resolution, domain = native_fixture(tmp_path)
    large, _ = assemble(profile, native, shift, resolution, tmp_path / "reference", radii_um=(10.,))
    small, manifest = assemble(profile, native, shift, resolution, tmp_path / "bounded",
        radii_um=(10.,), max_support_pixels=1, support_tile_size=2, support_cache_tiles=1)
    pd.testing.assert_frame_equal(large, small)
    first = sparse.load_npz(tmp_path / "reference/neighborhood_graph_10um.npz")
    second = sparse.load_npz(tmp_path / "bounded/neighborhood_graph_10um.npz")
    for attribute in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(first, attribute), getattr(second, attribute))
    record = manifest["neighborhoods"]["graph_support"]
    assert record["shape_yx"] == [8, 10] and record["source_grid_matches_native_crop"]
    assert not record["explicit_native_support"]  # No new argument is required for a native crop source.


@pytest.mark.parametrize("frame", ["crop_pixels", "original_pixels", "existing_support"])
def test_native_support_cli_matches_crop_view_and_clips_density(tmp_path, frame):
    profile, coarse, native, full, shift, resolution, domain = native_fixture(tmp_path)
    expected, _ = assemble(profile, coarse, shift, resolution, tmp_path / "crop",
        native_support_mask=native, native_support_coordinates="crop_pixels", radii_um=(10.,))
    output = tmp_path / "cli"
    selected = full if frame == "original_pixels" else native
    options = [] if frame == "existing_support" else ["--native-support-mask", str(selected), "--native-support-coordinates", frame]
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "bin/assemble_spatial_cell_profiles.py"),
        "--profile-dir", str(profile), "--support-mask", str(native if frame == "existing_support" else coarse), "--shift", str(shift),
        "--resolution-json", str(resolution), *options, "--radii-um", "10",
        "--support-tile-size", "2", "--support-cache-tiles", "1", "--max-support-pixels", "1",
        "--outdir", str(output)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    pd.testing.assert_frame_equal(expected, pd.read_parquet(output / "cell_profiles.parquet"))
    manifest = json.loads((output / "cell_profiles_manifest.json").read_text())
    support = manifest["neighborhoods"]["graph_support"]
    assert support["view_window_xyxy"] == ([100, 200, 110, 208] if frame == "original_pixels" else [0, 0, 10, 8])
    assert support["source_coordinates"] == ("crop_extent_scaled_support_grid" if frame == "existing_support" else frame)
    assert support["shape_yx"] == [8, 10] and support["sha256"] == sha256_file(selected)


def test_native_support_never_guesses_frame_or_rescales_non_native_input(tmp_path):
    profile, coarse, native, full, shift, resolution, domain = native_fixture(tmp_path)
    for kwargs, match in [
        ({"native_support_mask": native}, "both a mask and explicit"),
        ({"native_support_coordinates": "crop_pixels"}, "both a mask and explicit"),
        ({"native_support_mask": coarse, "native_support_coordinates": "crop_pixels"}, "exact calibrated crop"),
        ({"native_support_mask": native, "native_support_coordinates": "original_pixels"}, "full_size"),
    ]:
        with pytest.raises(ValueError, match=match):
            assemble(profile, coarse, shift, resolution, tmp_path / "bad", **kwargs)
        assert not (tmp_path / "bad").exists()
