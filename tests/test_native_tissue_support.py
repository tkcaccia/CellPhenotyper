import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from build_native_tissue_support import build_support, classify_rgb, verify_support_bundle
from cell_profile_io import sha256_file
from build_cell_profiles import build_profiles, parser
from assemble_spatial_cell_profiles import assemble


def fixture(tmp_path, *, shape=(49, 67), gap=True):
    h, w = shape
    rgb = np.full((h, w, 3), [236, 214, 225], np.uint8)
    if gap:
        rgb[:, w//2-3:w//2+4] = 255
    image = tmp_path / "image.tif"
    tifffile.imwrite(image, rgb, tile=(16, 16), compression="deflate", photometric="rgb")
    support = tmp_path / "coarse.tif"
    tifffile.imwrite(support, np.ones((3, 5), np.uint8), compression="deflate")
    shift = tmp_path / "shift.json"
    shift.write_text(json.dumps({"crop_size": {"width": w, "height": h},
        "offset_crop_to_original": {"dx": 1000, "dy": 2000}}))
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    return image, support, shift, resolution


def test_strict_rule_protects_pale_colour_texture_and_unknown_edges():
    rgb = np.full((20, 30, 3), 255, np.uint8)
    rgb[5:8, 5:8] = [253, 239, 245]  # Pale stained tissue remains supported.
    rgb[12, 12] = [241, 241, 241]  # Tiny structure protects its physical neighbourhood.
    candidate, excluded = classify_rgb(rgb, radius_px=2)
    assert excluded[10, 20] and not excluded[6, 6]
    assert not excluded[10:15, 10:15].any()
    assert candidate[11, 12] and not excluded[11, 12]
    assert not excluded[:2].any() and not excluded[:, :2].any()


def test_streaming_is_tile_invariant_and_never_restores_upstream_support(tmp_path):
    image, support, shift, resolution = fixture(tmp_path)
    upstream = np.ones((3, 5), np.uint8)
    upstream[1, 0] = 0
    tifffile.imwrite(support, upstream)
    before = {p: sha256_file(p) for p in (image, support, shift, resolution)}
    reports = [build_support(image, support, shift, resolution, tmp_path / f"t{size}", tile_size=size)
               for size in (16, 32, 64)]
    for name in ("support", "reasons"):
        reference = tifffile.imread(tmp_path / "t16" / f"{name}.tif")
        for size in (32, 64):
            np.testing.assert_array_equal(reference, tifffile.imread(tmp_path / f"t{size}" / f"{name}.tif"))
    reasons = tifffile.imread(tmp_path / "t16/reasons.tif")
    result = tifffile.imread(tmp_path / "t16/support.tif")
    assert set(np.unique(reasons)) == {0, 1, 2, 3}
    assert not result[reasons == 0].any() and not result[reasons == 2].any()
    assert result[reasons == 3].all()
    assert reports[0]["pixel_counts"] == reports[1]["pixel_counts"]
    assert sum(reports[0]["pixel_counts"].values()) == 49 * 67
    assert reports[0]["max_image_window_pixels"] <= 512**2
    assert before == {p: sha256_file(p) for p in before}
    with pytest.raises(FileExistsError):
        build_support(image, support, shift, resolution, tmp_path / "t16")


@pytest.mark.parametrize("problem", ["uint16", "shape", "anisotropic", "tile", "radius", "range"])
def test_invalid_inputs_fail_before_creating_outputs(tmp_path, problem):
    image, support, shift, resolution = fixture(tmp_path)
    kwargs = {}
    if problem == "uint16":
        tifffile.imwrite(image, np.ones((49, 67, 3), np.uint16), photometric="rgb")
    if problem == "shape":
        tifffile.imwrite(image, np.ones((48, 67, 3), np.uint8), photometric="rgb")
    if problem == "anisotropic":
        resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .7}))
    if problem == "tile":
        kwargs["tile_size"] = 17
    if problem == "radius":
        kwargs["guard_radius_um"] = float("nan")
    if problem == "range":
        kwargs["minimum_channel"] = 256
    with pytest.raises(ValueError):
        build_support(image, support, shift, resolution, tmp_path / "out", **kwargs)
    assert not (tmp_path / "out").exists()


def test_receipt_rejects_corruption_changed_sources_and_missing_image_binding(tmp_path):
    image, support, shift, resolution = fixture(tmp_path)
    build_support(image, support, shift, resolution, tmp_path / "native")
    receipt = tmp_path / "native/support_manifest.json"
    args = dict(image_sha256=sha256_file(image), support_mask=support, shift=shift, resolution_json=resolution)
    assert verify_support_bundle(receipt, **args)["complete"]
    with pytest.raises(ValueError, match="source identities"):
        verify_support_bundle(receipt, **{**args, "image_sha256": None})
    with (tmp_path / "native/reasons.tif").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="identity failure"):
        verify_support_bundle(receipt, **args)


def test_image_gap_changes_graph_not_canonical_cells_and_bundle_is_portable(tmp_path):
    image, support, shift, resolution = fixture(tmp_path)
    objects = tmp_path / "objects.csv"
    pd.DataFrame({"label": [2, 1, 3], "x": [20, 45, 33], "y": [24]*3,
        "xmin": [19, 44, 32], "xmax": [21, 46, 34], "ymin": [23]*3, "ymax": [25]*3}).to_csv(objects, index=False)
    base = tmp_path / "base"
    build_profiles(parser().parse_args(["--objects", str(objects), "--sample-id", "s", "--shift", str(shift),
        "--resolution-json", str(resolution), "--image", str(image), "--outdir", str(base)]))
    before = {p: sha256_file(p) for p in base.rglob("*") if p.is_file()}
    original, _ = assemble(base, support, shift, resolution, tmp_path / "provided", radii_um=(20.,))
    assert sparse.load_npz(tmp_path / "provided/neighborhood_graph_20um.npz").nnz == 6
    build_support(image, support, shift, resolution, tmp_path / "native")
    cells, manifest = assemble(base, support, shift, resolution, tmp_path / "profiles", radii_um=(20.,),
        native_support_mask=tmp_path / "native/support.tif", native_support_coordinates="crop_pixels",
        native_support_manifest=tmp_path / "native/support_manifest.json")
    assert cells.cell_uid.tolist() == original.cell_uid.tolist()
    assert cells.in_tissue_support.tolist() == [True, True, False]
    assert sparse.load_npz(tmp_path / "profiles/neighborhood_graph_20um.npz").nnz == 0
    assert before == {p: sha256_file(p) for p in before}
    assert (tmp_path / "profiles/neighborhood_support/reasons.tif").exists()
    assert manifest["neighborhoods"]["graph_support"]["canonical_measurement_support_unchanged"]
    assert manifest["neighborhoods"]["graph_support"]["biological_validation"] == "not_established"
    for name, checksum in manifest["files"].items():
        assert sha256_file(tmp_path / "profiles" / name) == checksum
    with pytest.raises(ValueError, match="original base profiles"):
        assemble(tmp_path / "profiles", support, shift, resolution, tmp_path / "bad_repeat", radii_um=(20.,))


def test_scanner_tinted_background_is_normalized_without_using_annotations(tmp_path):
    image, support, shift, resolution = fixture(tmp_path, shape=(80, 96))
    rgb = np.full((80, 96, 3), [210, 190, 211], np.uint8)
    rgb[:, :32] = [233, 233, 249]  # Automatically excluded scanner background.
    rgb[:, 46:52] = [233, 233, 249]  # Same optical gap missed by coarse support.
    rgb[30:40, 60:70] = [233, 222, 235]  # Pale stained structure must remain.
    tifffile.imwrite(image, rgb, photometric="rgb")
    coarse = np.ones((5, 6), np.uint8)
    coarse[:, :2] = 0
    tifffile.imwrite(support, coarse)
    nominal = build_support(image, support, shift, resolution, tmp_path / "nominal", white_reference="nominal")
    assert nominal["pixel_counts"]["2"] == 0
    auto = build_support(image, support, shift, resolution, tmp_path / "auto", tile_size=16)
    assert auto["white_reference"]["white_rgb"] == [233., 233., 249.]
    assert auto["white_reference"]["status"] == "estimated_from_upstream_exclusions"
    result = tifffile.imread(tmp_path / "auto/support.tif")
    assert not result[2:-2, 48:50].any() and result[30:40, 60:70].all()
    second = build_support(image, support, shift, resolution, tmp_path / "other", tile_size=64)
    np.testing.assert_array_equal(result, tifffile.imread(tmp_path / "other/support.tif"))
    assert auto["white_reference"] == second["white_reference"]


def test_inconsistent_or_dark_excluded_pixels_cannot_define_white_reference(tmp_path):
    image, support, shift, resolution = fixture(tmp_path, shape=(80, 96))
    rgb = np.full((80, 96, 3), 110, np.uint8)
    tifffile.imwrite(image, rgb, photometric="rgb")
    tifffile.imwrite(support, np.zeros((4, 6), np.uint8))
    result = build_support(image, support, shift, resolution, tmp_path / "dark")
    assert result["white_reference"]["status"] == "nominal_white_fallback"
    assert result["white_reference"]["reason"] == "background_samples_not_consistently_bright"
    assert result["pixel_counts"]["0"] == 80 * 96


def test_diagonal_gap_and_structured_white_pixels_cross_tile_edges_without_seams(tmp_path):
    image, support, shift, resolution = fixture(tmp_path, shape=(64, 64))
    rgb = np.full((64, 64, 3), [230, 210, 225], np.uint8)
    yy, xx = np.indices((64, 64))
    rgb[np.abs(xx-yy) <= 4] = 255
    tifffile.imwrite(image, rgb, photometric="rgb")
    for size in (16, 64):
        build_support(image, support, shift, resolution, tmp_path / f"d{size}", tile_size=size)
    a, b = (tifffile.imread(tmp_path / f"d{size}/support.tif") for size in (16, 64))
    np.testing.assert_array_equal(a, b)
    assert not a[np.arange(2, 62), np.arange(2, 62)].any()
    assert a[32, 45] == 1


def test_rehashed_semantic_corruption_is_rejected(tmp_path):
    image, support, shift, resolution = fixture(tmp_path)
    build_support(image, support, shift, resolution, tmp_path / "native")
    path = tmp_path / "native/support.tif"
    tifffile.imwrite(path, np.ones((49, 67), np.uint8))
    receipt = tmp_path / "native/support_manifest.json"
    record = json.loads(receipt.read_text())
    record["outputs"]["support"].update(sha256=sha256_file(path), bytes=path.stat().st_size)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="rasters disagree"):
        verify_support_bundle(receipt, image_sha256=sha256_file(image), support_mask=support, shift=shift, resolution_json=resolution)


def test_assembly_rejects_support_mutation_during_graph_analysis(tmp_path, monkeypatch):
    import assemble_spatial_cell_profiles as assembly
    image, support, shift, resolution = fixture(tmp_path)
    objects = tmp_path / "objects.csv"
    pd.DataFrame({"label": [1], "x": [20], "y": [24], "xmin": [19], "xmax": [21], "ymin": [23], "ymax": [25]}).to_csv(objects, index=False)
    base = tmp_path / "base"
    build_profiles(parser().parse_args(["--objects", str(objects), "--sample-id", "s", "--shift", str(shift),
        "--resolution-json", str(resolution), "--image", str(image), "--outdir", str(base)]))
    build_support(image, support, shift, resolution, tmp_path / "native")
    original = assembly.analyze_cells
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with (tmp_path / "native/support.tif").open("ab") as handle:
            handle.write(b"changed during graph analysis")
        return result
    monkeypatch.setattr(assembly, "analyze_cells", mutate)
    with pytest.raises(ValueError, match="changed during neighbourhood"):
        assemble(base, support, shift, resolution, tmp_path / "bad", radii_um=(20.,),
            native_support_mask=tmp_path / "native/support.tif", native_support_coordinates="crop_pixels",
            native_support_manifest=tmp_path / "native/support_manifest.json")
    assert not (tmp_path / "bad").exists()


def test_storage_accounts_for_native_bundle_and_work_copy(tmp_path, monkeypatch):
    import storage_preflight as storage
    image = tmp_path / "image.tif"
    image.write_bytes(b"fixture")
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 500, "test"))
    arguments = ["--input", str(image), "--outdir", str(tmp_path / "results"), "--workdir", str(tmp_path / "work"),
        "--output-json", str(tmp_path / "report.json"), "--cell-profiles-enabled", "true",
        "--start-point", "convert", "--end-point", "cluster_geojson"]
    first = storage.build_report(storage.parser().parse_args(arguments))
    second = storage.build_report(storage.parser().parse_args(arguments + ["--cell-neighborhood-support-mode", "brightfield_native"]))
    # Inspect the authoritative stage model, not the aggregate capacity verdict.
    before, after = (r["stage_storage_models"]["cell_profiles"] for r in (first, second))
    assert after["native_support_bundle_bytes"] == 2 * 1000 * 500
    assert after["published_bytes"] - before["published_bytes"] == 2 * 1000 * 500
    assert after["work_bytes"] - before["work_bytes"] == 4 * 1000 * 500
    assert after["retained_work_bytes"] - before["retained_work_bytes"] == 4 * 1000 * 500
