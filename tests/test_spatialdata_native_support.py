"""Actual SpatialData export/readback of image-rule categorical support bundles."""
import json
import pytest

pytest.importorskip("spatialdata")
import numpy as np
import spatialdata as sd
import tifffile

from test_spatialdata_export import specimen, affine
import export_spatialdata as exporter
from build_native_tissue_support import build_support
from cell_profile_io import sha256_file


@pytest.fixture
def native_specimen(specimen, tmp_path):
    inputs = dict(specimen[0])
    root = inputs["profile_dir"]
    image = np.full((64, 96, 3), [236, 214, 225], np.uint8)
    image[:, 42:51] = 255
    tifffile.imwrite(inputs["image"], image, photometric="rgb", tile=(16, 16), compression="deflate")
    coarse = tmp_path / "coarse.tif"
    support = np.ones((4, 6), np.uint8)
    support[0, 0] = 0
    tifffile.imwrite(coarse, support)
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5}))
    inputs["resolution_json"] = resolution
    bundle = root / "neighborhood_support"
    receipt = build_support(inputs["image"], coarse, inputs["shift"], resolution, bundle, tile_size=16)
    manifest_path = root / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = {name + "_sha256": sha256_file(inputs[name])
                          for name in ("image", "labels", "shift", "resolution_json")}
    manifest["inputs"].update({"native_support_mask_sha256": sha256_file(bundle / "support.tif"),
                                "support_mask_sha256": sha256_file(coarse)})
    manifest["neighborhoods"] = {"graph_support": {
        "path": "neighborhood_support/support.tif", "path_basis": "profile_directory",
        "producer_manifest": "neighborhood_support/support_manifest.json",
        "producer_manifest_sha256": sha256_file(bundle / "support_manifest.json"),
        "sha256": sha256_file(bundle / "support.tif"), "explicit_native_support": True,
        "source_coordinates": "crop_pixels", "shape_yx": [64, 96], "mpp_xy": [.5, .5],
        "origin_um_xy": [50., 100.], "view_window_xyxy": [0, 0, 96, 64],
        "graph_and_density_resampling": "none"}}
    for name in ("support.tif", "reasons.tif", "support_manifest.json"):
        manifest["files"]["neighborhood_support/" + name] = sha256_file(bundle / name)
    manifest_path.write_text(json.dumps(manifest))
    assert set(receipt["pixel_counts"]) == {"0", "1", "2", "3"}
    return inputs, bundle


def rebind_bundle(inputs, bundle, receipt):
    """Update every declared hash so semantic-mutation tests exceed checksum checks."""
    for key in ("support", "reasons"):
        path = bundle / (key + ".tif")
        receipt["outputs"][key].update({"sha256": sha256_file(path), "bytes": path.stat().st_size})
    (bundle / "support_manifest.json").write_text(json.dumps(receipt))
    root = inputs["profile_dir"]
    path = root / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    for name in ("support.tif", "reasons.tif", "support_manifest.json"):
        manifest["files"]["neighborhood_support/" + name] = sha256_file(bundle / name)
    manifest["neighborhoods"]["graph_support"].update({
        "sha256": sha256_file(bundle / "support.tif"),
        "producer_manifest_sha256": sha256_file(bundle / "support_manifest.json")})
    manifest["inputs"]["native_support_mask_sha256"] = sha256_file(bundle / "support.tif")
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("pyramid_levels", [0, 1])
def test_native_support_actual_roundtrip_pixels_geometry_semantics_and_no_instance_links(native_specimen, tmp_path, pyramid_levels):
    inputs, bundle = native_specimen
    before = {p: sha256_file(p) for p in inputs["profile_dir"].rglob("*") if p.is_file()}
    output = tmp_path / "native.zarr"
    summary = exporter.export_spatialdata(**inputs, outdir=output, pyramid_levels=pyramid_levels)
    loaded = sd.SpatialData.read(output)
    assert set(loaded.labels) == {"canonical_cells", *exporter.NATIVE_SUPPORT_LABELS.values()}
    expected_transform = np.array([[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    for key, name in exporter.NATIVE_SUPPORT_LABELS.items():
        actual = loaded.labels[name]
        assert actual.shape == (64, 96) and actual.dtype == np.uint8
        np.testing.assert_array_equal(actual.data.compute(), tifffile.imread(bundle / (key + ".tif")))
        np.testing.assert_array_equal(affine(actual), expected_transform)
    table = loaded.tables["cells"]
    assert table.obs.spatial_region.astype(str).eq("canonical_cells").all()
    assert table.obs.instance_id.tolist() == [1, 7]
    assert np.atleast_1d(table.uns["spatialdata_attrs"]["region"]).tolist() == ["canonical_cells"]
    record = json.loads(table.uns["cellphenotyper"]["neighborhood_support_json"])
    assert record == summary["neighborhood_support"] == loaded.attrs["cellphenotyper"]["neighborhood_support"]
    assert record["reason_codes"] == exporter.NATIVE_SUPPORT_CODES
    assert "NOT cell instance masks" in record["semantics"]
    assert record["biological_validation"] == "not_established"
    assert record["source_profile_paths_to_label_elements"] == {
        "neighborhood_support/support.tif": "neighborhood_tissue_support",
        "neighborhood_support/reasons.tif": "neighborhood_tissue_support_reasons"}
    assert json.loads(record["producer_manifest_json"]) == json.loads((bundle / "support_manifest.json").read_text())
    assert before == {p: sha256_file(p) for p in before}


def test_native_support_never_follows_upstream_receipt_paths_and_is_window_bounded(native_specimen, tmp_path, monkeypatch):
    inputs, bundle = native_specimen
    receipt = json.loads((bundle / "support_manifest.json").read_text())
    for record in receipt["inputs"].values():
        record["path"] = "/not-an-export-input/must-not-be-opened"
    rebind_bundle(inputs, bundle, receipt)
    original = exporter.RasterReader.window
    requests = []
    def bounded(self, x0, y0, x1, y1):
        requests.append((x1-x0, y1-y0))
        assert x1-x0 <= 16 and y1-y0 <= 16
        return original(self, x0, y0, x1, y1)
    monkeypatch.setattr(exporter.RasterReader, "window", bounded)
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: pytest.fail("Unbounded decoder used"))
    summary = exporter.export_spatialdata(**inputs, outdir=tmp_path / "portable.zarr")
    assert requests and "upstream_pixels_not_reopened" in summary["neighborhood_support"]["upstream_support_verification"]


@pytest.mark.parametrize("kind", ["support_bytes", "receipt_bytes", "graph_hash", "profile_hash", "manifest_path", "missing_manifest_link",
    "support_path", "parent_symlink", "file_symlink", "image_binding", "shift_binding", "resolution_binding",
    "missing_resolution", "upstream_binding", "shape", "dtype", "reason_codes", "nonbinary", "binary_consistency",
    "counts", "geometry", "graph_geometry", "semantic_names", "biological_claim"])
def test_native_support_rejects_corruption_and_rehashed_semantic_changes_before_writing(native_specimen, tmp_path, kind):
    inputs, bundle = native_specimen
    receipt_path = bundle / "support_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    root = inputs["profile_dir"]
    manifest_path = root / "cell_profiles_manifest.json"
    if kind in {"support_bytes", "receipt_bytes"}:
        target = bundle / ("support.tif" if kind == "support_bytes" else "support_manifest.json")
        with target.open("ab") as stream:
            stream.write(b"corrupt")
    elif kind in {"graph_hash", "profile_hash", "manifest_path", "missing_manifest_link", "support_path", "graph_geometry"}:
        manifest = json.loads(manifest_path.read_text())
        graph = manifest["neighborhoods"]["graph_support"]
        if kind == "graph_hash":
            graph["sha256"] = "0" * 64
        elif kind == "profile_hash":
            manifest["files"]["neighborhood_support/reasons.tif"] = "0" * 64
        elif kind == "manifest_path":
            graph["producer_manifest"] = "../outside/support_manifest.json"
        elif kind == "missing_manifest_link":
            del graph["producer_manifest"]
        elif kind == "support_path":
            graph["path"] = str((bundle / "support.tif").resolve())
        else:
            graph["origin_um_xy"][0] += 1
        manifest_path.write_text(json.dumps(manifest))
    elif kind == "parent_symlink":
        moved = root / "moved_bundle"
        bundle.rename(moved)
        bundle.symlink_to(moved, target_is_directory=True)
    elif kind == "file_symlink":
        moved = bundle / "moved_support.tif"
        (bundle / "support.tif").rename(moved)
        (bundle / "support.tif").symlink_to(moved)
    elif kind == "missing_resolution":
        inputs.pop("resolution_json")
    else:
        if kind.endswith("_binding"):
            source = {"image_binding": "image", "shift_binding": "shift", "resolution_binding": "resolution_json",
                      "upstream_binding": "support_mask"}[kind]
            receipt["inputs"][source]["sha256"] = "0" * 64
        elif kind in {"shape", "dtype", "reason_codes", "nonbinary", "binary_consistency"}:
            key = "reasons" if kind == "reason_codes" else "support"
            target = bundle / (key + ".tif")
            values = tifffile.imread(target)
            if kind == "shape":
                values = values[:-1]
            elif kind == "dtype":
                values = values.astype(np.uint16)
            elif kind == "reason_codes":
                values[20, 20] = 4
            elif kind == "nonbinary":
                values[20, 20] = 2
            else:
                values[20, 20] = 1 - values[20, 20]
            tifffile.imwrite(target, values, tile=(16, 16), compression="deflate")
        elif kind == "counts":
            receipt["pixel_counts"]["1"] += 1
        elif kind == "geometry":
            receipt["origin_original_pixels_xy"][0] += 1
        elif kind == "semantic_names":
            receipt["reason_codes"]["2"] = "validated_background"
        elif kind == "biological_claim":
            receipt["biological_validation"] = "passed"
        rebind_bundle(inputs, bundle, receipt)
    output = tmp_path / "invalid.zarr"
    with pytest.raises(ValueError, match="Native neighborhood support|native neighborhood support"):
        exporter.export_spatialdata(**inputs, outdir=output)
    assert not output.exists()


def test_native_support_checks_initial_bundle_identity_after_export(native_specimen, tmp_path, monkeypatch):
    inputs, bundle = native_specimen
    original = exporter._window
    changed = False
    def mutate_after_validation(*args):
        nonlocal changed
        if not changed:
            changed = True
            receipt = json.loads((bundle / "support_manifest.json").read_text())
            receipt["runtime_seconds"] += 1
            rebind_bundle(inputs, bundle, receipt)
        return original(*args)
    monkeypatch.setattr(exporter, "_window", mutate_after_validation)
    with pytest.raises(ValueError, match="changed during SpatialData export"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "changed.zarr")
    assert changed
