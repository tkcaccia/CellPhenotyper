"""Real optional SpatialData API tests, run in the isolated Python 3.12 runtime."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sd = pytest.importorskip("spatialdata")
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
from scipy import sparse
from spatialdata.transformations import get_transformation

BIN = Path(__file__).parents[1] / "bin"
sys.path.insert(0, str(BIN))
import export_spatialdata as exporter
from cell_profile_io import sha256_file


@pytest.fixture
def specimen(tmp_path):
    h, w = 64, 96
    image = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    labels = np.zeros((h, w), dtype=np.uint32)
    labels[8:20, 10:22] = 1
    labels[35:49, 60:74] = 7
    image_path, labels_path = tmp_path / "he.tif", tmp_path / "labels.tif"
    tifffile.imwrite(image_path, image, photometric="rgb", tile=(16, 16), compression="deflate")
    tifffile.imwrite(labels_path, labels, tile=(16, 16), compression="deflate")
    shift = {"source_mpp": 0.5, "offset_crop_to_original": {"dx": 100, "dy": 200},
             "crop_size": {"width": w, "height": h}}
    shift_path = tmp_path / "shift.json"
    shift_path.write_text(json.dumps(shift))
    profile = tmp_path / "profiles"
    profile.mkdir()
    cells = pd.DataFrame({"sample_id": ["sample A"] * 2, "cell_id": ["001", "7"],
        "cell_uid": ["sampleA:seg:001", "sampleA:seg:7"], "x_crop_px": [15., 65.],
        "y_crop_px": [12., 40.], "x_um": [57.5, 82.5], "y_um": [106., 120.],
        "tissue_domain": [1, 2], "predicted__nucleus__CD3__mean": [0.1, np.nan],
        "status": ["usable", "missing_marker"], "detector_agreement": [True, False]})
    cells.to_csv(profile / "cell_profiles.csv", index=False)
    cells.to_parquet(profile / "cell_profiles.parquet", index=False)
    cells[["sample_id", "cell_id", "cell_uid"]].to_csv(profile / "feature_rows.csv", index=False)
    context = np.array([[1, 2, 3, 4], [np.nan] * 4], dtype=np.float32)
    local = np.array([[0.1, 0.5], [0.7, 0.3]], dtype=np.float32)
    np.save(profile / "context.npy", context)
    np.save(profile / "local.npy", local)
    graph = sparse.csr_matrix([[0., 1.], [1., 0.]])
    sparse.save_npz(profile / "neighbors.npz", graph)
    manifest = {"schema_version": "1.0.0", "observation_unit": "cell", "sample_id": "sample A",
        "coordinate_system": "original_slide_micrometres",
        "cell_count": 2, "source_mpp": 0.5, "crop_origin_um": [50., 100.], "crop_size_px": [w, h],
        "feature_blocks": {name: {"path": name + ".npy", "shape": list(values.shape),
            "sha256": sha256_file(profile / (name + ".npy")),
            "feature_definition": {"model": "test", "context": name}} for name, values in (("context", context), ("local", local))},
        "spatial_graphs": {"25": {"path": "neighbors.npz", "sha256": sha256_file(profile / "neighbors.npz")}},
        "files": {name: sha256_file(profile / name) for name in ("cell_profiles.csv", "cell_profiles.parquet", "feature_rows.csv")}}
    (profile / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    polygons = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": "domain_one", "properties": {"value": 1, "classification": "cluster_1"},
         "geometry": {"type": "Polygon", "coordinates": [[[1, 1], [40, 1], [40, 50], [1, 50], [1, 1]],
                                                                [[5, 5], [5, 8], [8, 8], [8, 5], [5, 5]]]}},
        {"type": "Feature", "properties": {"value": 2},
         "geometry": {"type": "Polygon", "coordinates": [[[45, 10], [90, 10], [90, 60], [45, 60], [45, 10]]]}}
    ]}
    geojson = tmp_path / "domains.geojson"
    geojson.write_text(json.dumps(polygons))
    return {"profile_dir": profile, "image": image_path, "labels": labels_path, "shift": shift_path,
            "tissue_geojson": geojson, "tissue_coordinates": "crop_pixels", "tile_size": 16, "workers": 1}, image, labels, context


def affine(element):
    return get_transformation(element, to_coordinate_system="original_um").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))


def test_csv_fallback_retains_detector_identifiers(specimen, tmp_path):
    inputs = specimen[0]
    csv_only = tmp_path / "csv_only"
    csv_only.mkdir()
    root = inputs["profile_dir"]
    cells = pd.read_csv(root / "cell_profiles.csv", dtype={"cell_id": str})
    cells["cellvitpp_id"] = ["0004", "008"]
    cells["stardist_id"] = ["0009", "003"]
    cells["hovernet_id"] = ["0001", "007"]
    cells.to_csv(csv_only / "cell_profiles.csv", index=False)
    (csv_only / "feature_rows.csv").write_bytes((root / "feature_rows.csv").read_bytes())
    manifest = json.loads((root / "cell_profiles_manifest.json").read_text())
    manifest["files"]["cell_profiles.csv"] = sha256_file(csv_only / "cell_profiles.csv")
    (csv_only / "cell_profiles_manifest.json").write_text(json.dumps(manifest))
    _, loaded, _, ids = exporter.load_cell_profile(csv_only, exporter.calibration(inputs["shift"]))
    assert loaded.cellvitpp_id.tolist() == ["0004", "008"]
    assert loaded.stardist_id.tolist() == ["0009", "003"]
    assert loaded.hovernet_id.tolist() == ["0001", "007"]
    assert ids.tolist() == [1, 7]


def test_neighborhood_names_roundtrip_with_reversible_mapping_and_compartment_flags(specimen, tmp_path):
    inputs = specimen[0]
    root = inputs["profile_dir"]
    table_path = root / "cell_profiles.parquet"
    cells = pd.read_parquet(table_path)
    cells["r25um_phenotype_fraction:epithelial"] = [.25, .75]
    cells["r25um_phenotype_fraction_epithelial"] = [.5, .6]
    cells["compartment__status"] = ["whole_cell_proxy", "nucleus"]
    cells.to_parquet(table_path, index=False)
    manifest_path = root / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][table_path.name] = sha256_file(table_path)
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "neighborhood.zarr"
    exporter.export_spatialdata(**inputs, outdir=output)
    loaded = sd.SpatialData.read(output).tables["cells"]
    names = json.loads(loaded.uns["cellphenotyper"]["field_name_mapping_json"])["obs"]
    a, b = names["r25um_phenotype_fraction:epithelial"], names["r25um_phenotype_fraction_epithelial"]
    assert a != b and ":" not in a
    np.testing.assert_array_equal(loaded.obs[a], [.25, .75])
    np.testing.assert_array_equal(loaded.obs[b], [.5, .6])
    assert loaded.obs[names["compartment__status"]].astype(str).tolist() == ["whole_cell_proxy", "nucleus"]
    assert loaded.obs_names.tolist() == cells.cell_uid.tolist()


def test_real_spatialdata_round_trip_preserves_full_resolution_and_identity(specimen, tmp_path):
    inputs, image, labels, context = specimen
    output = tmp_path / "specimen.zarr"
    summary = exporter.export_spatialdata(**inputs, outdir=output)
    loaded = sd.SpatialData.read(output)
    assert summary["versions"]["spatialdata"] == "0.8.0"
    assert set(loaded.images) == {"he_image"}
    assert set(loaded.labels) == {"canonical_cells"}
    assert set(loaded.shapes) == {"tissue_domains"}
    assert set(loaded.tables) == {"cells", "tissue_domain_annotations"}
    assert isinstance(loaded.images["he_image"].data, da.Array)
    assert loaded.images["he_image"].shape == (3, 64, 96)
    np.testing.assert_array_equal(loaded.images["he_image"].data.compute(), np.moveaxis(image, -1, 0))
    np.testing.assert_array_equal(loaded.labels["canonical_cells"].data.compute(), labels)
    table = loaded.tables["cells"]
    assert table.obs_names.tolist() == ["sampleA:seg:001", "sampleA:seg:7"]
    assert table.obs["cell_id"].astype(str).tolist() == ["001", "7"]
    assert table.obs["instance_id"].tolist() == [1, 7]
    assert table.uns["spatialdata_attrs"]["region_key"] == "spatial_region"
    assert table.uns["spatialdata_attrs"]["instance_key"] == "instance_id"
    assert table.obs.spatial_region.astype(str).eq("canonical_cells").all()
    np.testing.assert_allclose(table.obsm["context"], context, equal_nan=True)
    assert table.obsm["local"].shape == (2, 2)
    assert table.X.shape == (2, 0)
    np.testing.assert_array_equal(table.obsp["neighbors_25um"].toarray(), [[0, 1], [1, 0]])
    assert "measured" not in loaded.tables
    assert loaded.attrs["cellphenotyper"]["measured_modalities_status"].startswith("not_provided")
    lineage = loaded.attrs["cellphenotyper"]["canonical_raster_lineage"]
    assert lineage["status"] == "legacy_or_partial_raster_lineage"
    assert set(lineage["missing_profile_raster_hashes"]) == {"image", "labels"}


def test_nonzero_crop_offset_and_shape_table_transform_round_trip(specimen, tmp_path):
    inputs, _, _, _ = specimen
    output = tmp_path / "physical.zarr"
    exporter.export_spatialdata(**inputs, outdir=output)
    loaded = sd.SpatialData.read(output)
    expected = np.array([[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    for element in (loaded.images["he_image"], loaded.labels["canonical_cells"], loaded.shapes["tissue_domains"]):
        np.testing.assert_allclose(affine(element), expected)
    domains = loaded.shapes["tissue_domains"]
    assert len(domains.geometry.iloc[0].interiors) == 1
    table = loaded.tables["tissue_domain_annotations"]
    assert table.obs.domain_instance_id.tolist() == list(domains.index)
    assert table.uns["spatialdata_attrs"]["instance_key"] == "domain_instance_id"
    np.testing.assert_allclose(loaded.tables["cells"].obsm["spatial"], [[57.5, 106], [82.5, 120]])


def test_no_whole_image_decoder_and_all_requests_are_bounded(specimen, tmp_path, monkeypatch):
    inputs, _, _, _ = specimen
    original_window = exporter.RasterReader.window
    requests = []
    def window(self, x0, y0, x1, y1):
        requests.append((x1 - x0, y1 - y0))
        assert x1 - x0 <= 16 and y1 - y0 <= 16
        return original_window(self, x0, y0, x1, y1)
    monkeypatch.setattr(exporter.RasterReader, "window", window)
    monkeypatch.setattr(tifffile, "imread", lambda *a, **kw: pytest.fail("Whole-image decoder used"))
    exporter.export_spatialdata(**inputs, outdir=tmp_path / "bounded.zarr")
    assert len(requests) >= 3 * 24  # labels verified, image written, labels written


def test_lazy_raster_reads_only_when_computed(specimen, monkeypatch):
    inputs, image, _, _ = specimen
    original = exporter._window
    calls = []
    def block(*args):
        calls.append(args[1:5])
        return original(*args)
    monkeypatch.setattr(exporter, "_window", block)
    lazy = exporter.lazy_raster(inputs["image"], tile_size=16, channels=True)
    assert calls == []
    np.testing.assert_array_equal(lazy[:, :16, :16].compute(scheduler="single-threaded"), np.moveaxis(image[:16, :16], -1, 0))
    assert len(calls) == 1


def test_optional_pyramid_preserves_native_level_and_categorical_labels(specimen, tmp_path):
    inputs, image, labels, _ = specimen
    output = tmp_path / "multiscale.zarr"
    exporter.export_spatialdata(**inputs, outdir=output, pyramid_levels=1)
    loaded = sd.SpatialData.read(output)
    native_image = next(iter(loaded.images["he_image"]["scale0"].data_vars.values()))
    native_labels = next(iter(loaded.labels["canonical_cells"]["scale0"].data_vars.values()))
    coarse_labels = next(iter(loaded.labels["canonical_cells"]["scale1"].data_vars.values()))
    np.testing.assert_array_equal(native_image.data.compute(), np.moveaxis(image, -1, 0))
    np.testing.assert_array_equal(native_labels.data.compute(), labels)
    assert coarse_labels.shape == (32, 48)
    assert set(np.unique(coarse_labels.data.compute())) <= {0, 1, 7}


@pytest.mark.parametrize("mode,expected", [("crop_pixels", [[.5, 0, 50], [0, .5, 100], [0, 0, 1]]),
    ("original_pixels", [[.5, 0, 0], [0, .5, 0], [0, 0, 1]]),
    ("original_um", np.eye(3))])
def test_explicit_geometry_coordinate_modes(mode, expected):
    cal = {"mpp": .5, "origin_px": np.array([100, 200])}
    transformation = exporter.physical_transform(cal, mode)
    np.testing.assert_allclose(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), expected)


def test_missing_or_foreign_labels_are_rejected_before_writing(specimen, tmp_path):
    inputs, _, labels, _ = specimen
    labels[labels == 7] = 99
    tifffile.imwrite(inputs["labels"], labels, tile=(16, 16), compression="deflate")
    output = tmp_path / "invalid.zarr"
    with pytest.raises(ValueError, match="foreign cell IDs"):
        exporter.export_spatialdata(**inputs, outdir=output)
    assert not output.exists()


def test_mismatched_profile_coordinates_are_rejected(specimen, tmp_path):
    inputs, _, _, _ = specimen
    shift = json.loads(inputs["shift"].read_text())
    shift["offset_crop_to_original"]["dx"] = 900
    inputs["shift"].write_text(json.dumps(shift))
    with pytest.raises(ValueError, match="offsets disagree"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "wrong.zarr")


def test_cli_real_export_and_immutable_output(specimen, tmp_path):
    inputs, _, _, _ = specimen
    output = tmp_path / "cli.zarr"
    command = [sys.executable, str(BIN / "export_spatialdata.py"), "--outdir", str(output)]
    for name, value in inputs.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "written_and_read_back" in result.stdout
    assert len(sd.SpatialData.read(output).tables["cells"]) == 2
    with pytest.raises(FileExistsError):
        exporter.export_spatialdata(**inputs, outdir=output)


def measured_package(inputs, directory, assay_id="registered-mIF"):
    """Build a real importer package; fixture values are not experimental data."""
    from integrate_measured_assay import INPUT_SCHEMA, integrate
    directory.mkdir()
    root = inputs["profile_dir"]
    cells = pd.read_parquet(root / "cell_profiles.parquet")
    sample_id = str(cells.sample_id.iloc[0])
    measured = directory / "measurements.csv"
    pd.DataFrame({"sample_id": [sample_id], "cell_uid": [cells.cell_uid[0]],
        "assay_observation_id": ["assay:001"], "assay_x": [15.], "assay_y": [12.],
        "registered_x_um": [57.5], "registered_y_um": [106.],
        "CD3_signal": [16777217.25], "Ki67_signal": [np.nan]}).to_csv(measured, index=False)
    transform = directory / "transform.json"
    transform.write_text(json.dumps({"matrix": [[.5, 0, 50], [0, .5, 100], [0, 0, 1]]}))
    raw = directory / "raw-assay.txt"
    raw.write_text("Independent assay fixture; not experimental evidence")
    contract = {"schema_version": INPUT_SCHEMA, "assay_id": assay_id, "assay_type": "protein_imaging",
        "assay_platform": "mIF", "assay_protocol": "locked-independent-assay-v1", "panel_version": "panel-v1",
        "sample_id": sample_id, "observation_unit": "cell", "reference_design": "same_section_registered",
        "coordinates": {"source_frame": "assay_level0", "source_units": "pixel",
            "target_frame": "original_slide_micrometres", "target_units": "um"},
        "registration": {"status": "passed", "locked_before_prediction_review": True,
            "method": "independent_landmarks", "independent_landmarks": 3,
            "median_error_um": .2, "p95_error_um": .5, "acceptance_p95_um": 1.,
            "target_observations_sha256": sha256_file(root / "cell_profiles.parquet"),
            "source_observations_sha256": sha256_file(measured),
            "transform_artifact": {"path": str(transform), "sha256": sha256_file(transform)}},
        "matching": {"method": "provided_one_to_one_ids", "independent_of_predicted_markers": True,
            "protocol": "prespecified cell ID mapping", "eligible_observation_count": 2,
            "minimum_matched_fraction": .5, "matched_fraction": .5, "maximum_match_distance_um": 1.},
        "source_artifacts": [{"path": str(raw), "sha256": sha256_file(raw), "role": "raw_assay"}],
        "markers": [{"name": marker, "column": f"{marker}_signal", "units": "arbitrary_units",
            "measurement_type": "measured_protein_intensity", "normalization": "none"} for marker in ("CD3", "Ki67")]}
    assay_manifest = directory / "assay_manifest.json"
    assay_manifest.write_text(json.dumps(contract))
    package = directory / "package"
    integrate(measured=measured, assay_manifest=assay_manifest, cell_profiles=root, outdir=package)
    return package


def test_measured_tables_are_separate_linked_exact_float64_modalities(specimen, tmp_path):
    inputs, _, _, context = specimen
    first = measured_package(inputs, tmp_path / "first")
    second = measured_package(inputs, tmp_path / "second", assay_id="another assay / independent")
    before = sha256_file(inputs["profile_dir"] / "cell_profiles.parquet")
    output = tmp_path / "measured.zarr"
    summary = exporter.export_spatialdata(**inputs, outdir=output, measured_assay=[first, second])
    loaded = sd.SpatialData.read(output)
    assert summary["measured_modalities_status"] == "verified_separate_cell_tables"
    assert len(summary["measured_modalities"]) == 2
    assert len(loaded.tables) == 4
    for record in summary["measured_modalities"]:
        table = loaded.tables[record["table"]]
        assert table.X.dtype == np.float64
        np.testing.assert_array_equal(table.X, [[16777217.25, np.nan], [np.nan, np.nan]])
        assert table.obs_names.tolist() == ["sampleA:seg:001", "sampleA:seg:7"]
        assert table.obs.instance_id.tolist() == [1, 7]
        assert table.obs.spatial_region.astype(str).eq("canonical_cells").all()
        assert table.uns["spatialdata_attrs"]["instance_key"] == "instance_id"
        assert table.var.marker_name.astype(str).tolist() == ["CD3", "Ki67"]
        assert table.var.units.astype(str).tolist() == ["arbitrary_units", "arbitrary_units"]
        assert table.obs.measured_status.astype(str).tolist() == ["matched", "unmatched"]
        assert not any(name.startswith("predicted__") for name in table.obs)
        assert table.uns["cellphenotyper"]["modality"] == "measured_assay"
    assert loaded.tables["cells"].X.shape == (2, 0)
    np.testing.assert_array_equal(loaded.tables["cells"].obsm["context"], context)
    assert sha256_file(inputs["profile_dir"] / "cell_profiles.parquet") == before


@pytest.mark.parametrize("kind,match", [("region", "region-shape linkage"), ("serial", "same-section"),
    ("manifest_lineage", "manifest lineage"), ("table_lineage", "table lineage"),
    ("registration", "registration gate"), ("units", "declared units"),
    ("shape", "shape/marker schema"), ("order", "marker order/identity")])
def test_measured_manifest_contract_failures_precede_writes(specimen, tmp_path, kind, match):
    inputs = specimen[0]
    package = measured_package(inputs, tmp_path / "assay")
    path = package / "measured_assay_manifest.json"
    manifest = json.loads(path.read_text())
    if kind == "region":
        manifest["observation_unit"] = "spatial_bin"
    elif kind == "serial":
        manifest["reference_design"] = "serial_section_region_level"
    elif kind == "manifest_lineage":
        manifest["canonical_registry"]["manifest_sha256"] = "0" * 64
    elif kind == "table_lineage":
        manifest["canonical_registry"]["table_sha256"] = "0" * 64
    elif kind == "registration":
        manifest["registration"]["p95_error_um"] = 99.
    elif kind == "units":
        del manifest["markers"][0]["units"]
    elif kind == "shape":
        manifest["matrix"]["shape"] = [2, 3]
    else:
        manifest["markers"] = manifest["markers"][::-1]
    path.write_text(json.dumps(manifest))
    output = tmp_path / "rejected.zarr"
    with pytest.raises(ValueError, match=match):
        exporter.export_spatialdata(**inputs, outdir=output, measured_assay=[package])
    assert not output.exists()


@pytest.mark.parametrize("kind,match", [("hash", "artifact hash"), ("rows", "row order/identities"),
    ("foreign", "row order/identities"), ("nan", "values/NaNs differ"),
    ("precision", "shape/marker schema"), ("duplicate", "row order/identities"),
    ("match_distance", "physical match-distance")])
def test_measured_content_identity_and_nan_failures(specimen, tmp_path, kind, match):
    inputs = specimen[0]
    package = measured_package(inputs, tmp_path / "assay")
    path = package / "measured_assay_manifest.json"
    manifest = json.loads(path.read_text())
    if kind in {"hash", "nan", "precision"}:
        target = package / "measured_values.npy"
        matrix = np.load(target)
        if kind == "precision":
            matrix = matrix.astype(np.float32)
        else:
            matrix[1, 0] = 0.  # replacing missingness is not permitted
        np.save(target, matrix)
        if kind != "hash":
            manifest["files"][target.name] = sha256_file(target)
            manifest["matrix"]["sha256"] = sha256_file(target)
    elif kind == "rows":
        target = package / "measured_rows.csv"
        frame = pd.read_csv(target, dtype=str)
        frame.iloc[::-1].to_csv(target, index=False)
        manifest["files"][target.name] = sha256_file(target)
    else:
        target = package / "measured_observations.parquet"
        frame = pd.read_parquet(target)
        if kind == "match_distance":
            frame.loc[0, "registered_x_um"] += 100.
        else:
            frame.loc[1, "cell_uid"] = "unknown" if kind == "foreign" else frame.loc[0, "cell_uid"]
        frame.to_parquet(target, index=False)
        manifest["files"][target.name] = sha256_file(target)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "rejected.zarr", measured_assay=[package])


def test_duplicate_measured_packages_are_rejected(specimen, tmp_path):
    inputs = specimen[0]
    package = measured_package(inputs, tmp_path / "assay")
    with pytest.raises(ValueError, match="Duplicate measured assay"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "duplicate.zarr", measured_assay=[package, package])


def test_cli_accepts_repeatable_measured_assay_option(specimen, tmp_path):
    inputs = specimen[0]
    package = measured_package(inputs, tmp_path / "assay")
    output = tmp_path / "cli-measured.zarr"
    command = [sys.executable, str(BIN / "export_spatialdata.py"), "--outdir", str(output),
               "--measured-assay", str(package)]
    for name, value in inputs.items():
        command.extend(["--" + name.replace("_", "-"), str(value)])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(sd.SpatialData.read(output).tables) == 3


@pytest.mark.parametrize("raster", ["labels", "image"])
def test_profile_bound_raster_hashes_reject_same_shape_and_label_ids(specimen, tmp_path, raster):
    inputs, image, labels, _ = specimen
    manifest_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = {f"{name}_sha256": sha256_file(inputs[name]) for name in ("image", "labels")}
    manifest_path.write_text(json.dumps(manifest))
    if raster == "labels":
        changed = labels.copy()
        changed[20, 20] = 1  # same IDs/shape, changed canonical geometry
        tifffile.imwrite(inputs["labels"], changed, tile=(16, 16), compression="deflate")
    else:
        changed = image.copy()
        changed[0, 0, 0] = 255
        tifffile.imwrite(inputs["image"], changed, photometric="rgb", tile=(16, 16), compression="deflate")
    output = tmp_path / "changed-raster.zarr"
    with pytest.raises(ValueError, match=f"{raster} raster hash mismatch"):
        exporter.export_spatialdata(**inputs, outdir=output)
    assert not output.exists()


def test_profile_bound_images_and_labels_report_verified_lineage(specimen, tmp_path):
    inputs = specimen[0]
    manifest_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inputs"] = {f"{name}_sha256": sha256_file(inputs[name]) for name in ("image", "labels")}
    manifest_path.write_text(json.dumps(manifest))
    summary = exporter.export_spatialdata(**inputs, outdir=tmp_path / "bound.zarr")
    assert summary["canonical_raster_lineage"]["status"] == "verified_image_and_labels"
    assert summary["canonical_raster_lineage"]["missing_profile_raster_hashes"] == []


def measured_region_package(inputs, directory, source_coordinates="crop_pixels"):
    from integrate_measured_assay import INPUT_SCHEMA, REGION_SCHEMA, integrate
    directory.mkdir()
    sample_id = pd.read_parquet(inputs["profile_dir"] / "cell_profiles.parquet").sample_id.iloc[0]
    xy = [(12., 12.), (50., 30.), (80., 52.)]
    region_ids = ["bin:001", "bin:002", "bin:003"]
    boxes = [(4., 4., 20., 20.), (40., 20., 60., 40.), (72., 44., 88., 60.)]
    features = []
    for uid, (x0, y0, x1, y1) in zip(region_ids, boxes):
        coords = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        if source_coordinates == "original_pixels":
            coords += [100., 200.]
        elif source_coordinates == "original_um":
            coords = coords * .5 + [50., 100.]
        features.append({"type": "Feature", "properties": {"region_uid": uid},
            "geometry": {"type": "Polygon", "coordinates": [coords.tolist()]}})
    geometry = directory / "registered_regions.geojson"
    geometry.write_text(json.dumps({"type": "FeatureCollection", "features": features[::-1]}))
    regions = pd.DataFrame({"sample_id": [sample_id] * 3, "region_uid": region_ids,
        "region_id": ["001", "002", "003"], "x_um": [56., 75., 90.], "y_um": [106., 115., 126.],
        "area_um2": [64., 100., 64.]})
    table = directory / "regions.parquet"
    regions.to_parquet(table, index=False)
    registry_manifest = directory / "regions_manifest.json"
    registry_manifest.write_text(json.dumps({"schema_version": REGION_SCHEMA, "sample_id": sample_id,
        "observation_unit": "spatial_bin", "coordinate_system": "original_slide_micrometres", "region_count": 3,
        "files": {table.name: sha256_file(table)}, "region_definition": {"kind": "fixed_spatial_bins",
            "description": "Synthetic registered polygons for technical testing, not experimental measurements",
            "artifact": {"path": geometry.name, "sha256": sha256_file(geometry)}}}))
    measured = directory / "measured.csv"
    pd.DataFrame({"sample_id": [sample_id] * 2, "region_uid": [region_ids[2], region_ids[0]],
        "assay_observation_id": ["spot003", "spot001"], "assay_x": [80., 12.], "assay_y": [52., 12.],
        "registered_x_um": [90., 56.], "registered_y_um": [126., 106.],
        "gene_counts": [16777217.25, np.nan], "protein_signal": [np.nan, 2.5]}).to_csv(measured, index=False)
    transform = directory / "registration.json"
    transform.write_text(json.dumps({"matrix": [[.5, 0, 50], [0, .5, 100], [0, 0, 1]]}))
    raw = directory / "assay_raw.txt"
    raw.write_text("Synthetic assay export fixture only")
    contract = {"schema_version": INPUT_SCHEMA, "assay_id": "registered-regions", "assay_type": "spatial_assay",
        "assay_platform": "Visium", "assay_protocol": "locked independent region protocol", "panel_version": "test-v1",
        "sample_id": sample_id, "observation_unit": "spatial_bin", "reference_design": "serial_section_region_level",
        "coordinates": {"source_frame": "independent_assay_pixels", "source_units": "pixel",
            "target_frame": "original_slide_micrometres", "target_units": "um"},
        "registration": {"status": "passed", "locked_before_prediction_review": True,
            "method": "independent_registration", "independent_landmarks": 3, "median_error_um": .2,
            "p95_error_um": .5, "acceptance_p95_um": 1.,
            "target_observations_sha256": sha256_file(table), "source_observations_sha256": sha256_file(measured),
            "transform_artifact": {"path": transform.name, "sha256": sha256_file(transform)}},
        "matching": {"method": "provided_region_ids", "independent_of_predicted_markers": True,
            "protocol": "locked explicit region mapping", "eligible_observation_count": 3,
            "minimum_matched_fraction": .5, "matched_fraction": 2/3, "maximum_match_distance_um": 1.},
        "source_artifacts": [{"path": raw.name, "sha256": sha256_file(raw), "role": "raw_assay"}],
        "markers": [{"name": name, "column": col, "units": units, "measurement_type": kind, "normalization": "none"}
            for name, col, units, kind in [("CD3D", "gene_counts", "counts", "transcript_count"),
                                           ("CD3", "protein_signal", "arbitrary_units", "protein_intensity")]]}
    assay_manifest = directory / "assay.json"
    assay_manifest.write_text(json.dumps(contract))
    package = directory / "package"
    integrate(regions=table, regions_manifest=registry_manifest, measured=measured, assay_manifest=assay_manifest, outdir=package)
    link = directory / "shape_link.json"
    link.write_text(json.dumps({"schema_version": "cellphenotyper.measured_region_shapes.v1",
        "assay_id": contract["assay_id"], "sample_id": sample_id, "source_coordinates": source_coordinates,
        "target_coordinates": "original_slide_micrometres", "coordinate_anchor": "geometry_centroid",
        "target_image_sha256": sha256_file(inputs["image"]), "target_shift_sha256": sha256_file(inputs["shift"]),
        "registration_transform_sha256": sha256_file(transform),
        "shapes": {"path": geometry.name, "sha256": sha256_file(geometry)},
        "registry_table": {"path": table.name, "sha256": sha256_file(table)},
        "registry_manifest": {"path": registry_manifest.name, "sha256": sha256_file(registry_manifest)}}))
    return package, link, regions


@pytest.mark.parametrize("frame", ["crop_pixels", "original_pixels", "original_um"])
def test_region_assay_links_its_own_shapes_with_exact_ids_transform_values(specimen, tmp_path, frame):
    inputs = specimen[0]
    package, link, regions = measured_region_package(inputs, tmp_path / "regions", frame)
    output = tmp_path / "spatial-regions.zarr"
    summary = exporter.export_spatialdata(**inputs, outdir=output, measured_assay=[package], measured_region_shapes=[link])
    loaded = sd.SpatialData.read(output)
    record = summary["measured_modalities"][0]
    assert record["observation_unit"] == "spatial_bin"
    assert record["identity_key"] == "region_uid"
    assert record["spatial_element"].startswith("measured_regions_")
    shapes = loaded.shapes[record["spatial_element"]]
    table = loaded.tables[record["table"]]
    assert table.obs_names.tolist() == regions.region_uid.tolist()
    assert shapes.region_uid.tolist() == regions.region_uid.tolist()
    assert table.obs.instance_id.tolist() == list(shapes.index) == [1, 2, 3]
    assert table.obs.spatial_region.astype(str).eq(record["spatial_element"]).all()
    assert "cell_uid" not in table.obs and "cell_id" not in table.obs
    np.testing.assert_array_equal(table.X, [[np.nan, 2.5], [np.nan, np.nan], [16777217.25, np.nan]])
    assert table.obs.measured_status.astype(str).tolist() == ["matched", "unmatched", "matched"]
    expected = {"crop_pixels": [[.5, 0, 50], [0, .5, 100], [0, 0, 1]],
                "original_pixels": [[.5, 0, 0], [0, .5, 0], [0, 0, 1]], "original_um": np.eye(3)}[frame]
    np.testing.assert_array_equal(affine(shapes), expected)
    assert loaded.tables["cells"].shape == (2, 0)
    assert loaded.tables["cells"].obs_names.tolist() == ["sampleA:seg:001", "sampleA:seg:7"]


@pytest.mark.parametrize("kind,match", [("sample", "identity mismatch"), ("image", "image/shift hash"),
    ("transform", "transform lineage"), ("coordinates", "centroid/physical area"),
    ("registry", "registry lineage"), ("anchor", "geometry_centroid")])
def test_region_shape_links_fail_closed_before_export(specimen, tmp_path, kind, match):
    inputs = specimen[0]
    package, link, _ = measured_region_package(inputs, tmp_path / "regions")
    payload = json.loads(link.read_text())
    if kind == "sample":
        payload["sample_id"] = "different"
    elif kind == "image":
        payload["target_image_sha256"] = "0" * 64
    elif kind == "transform":
        payload["registration_transform_sha256"] = "0" * 64
    elif kind == "coordinates":
        payload["source_coordinates"] = "original_um"
    elif kind == "registry":
        # A different hash-bound registry manifest cannot be substituted.
        registry_path = link.parent / payload["registry_manifest"]["path"]
        registry_path.write_text(registry_path.read_text() + "\n")
        payload["registry_manifest"]["sha256"] = sha256_file(registry_path)
    else:
        payload["coordinate_anchor"] = "nearest_cell"
    link.write_text(json.dumps(payload))
    output = tmp_path / "bad.zarr"
    with pytest.raises(ValueError, match=match):
        exporter.export_spatialdata(**inputs, outdir=output, measured_assay=[package], measured_region_shapes=[link])
    assert not output.exists()


def rewrite_region_geometry(package, link, edit):
    payload = json.loads(link.read_text())
    geometry = link.parent / payload["shapes"]["path"]
    geojson = json.loads(geometry.read_text())
    edit(geojson)
    geometry.write_text(json.dumps(geojson))
    digest = sha256_file(geometry)
    payload["shapes"]["sha256"] = digest
    registry = link.parent / payload["registry_manifest"]["path"]
    registry_data = json.loads(registry.read_text())
    registry_data["region_definition"]["artifact"]["sha256"] = digest
    registry.write_text(json.dumps(registry_data))
    payload["registry_manifest"]["sha256"] = sha256_file(registry)
    link.write_text(json.dumps(payload))
    manifest_path = package / "measured_assay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["canonical_registry"]["manifest_sha256"] = sha256_file(registry)
    manifest["region_definition"]["artifact"]["sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("kind,match", [("missing", "region identities"), ("foreign", "region identities"),
    ("duplicate", "unique explicit region_uid"), ("geometry", "centroid/physical area")])
def test_explicit_region_polygon_identity_and_geometry_are_checked(specimen, tmp_path, kind, match):
    inputs = specimen[0]
    package, link, _ = measured_region_package(inputs, tmp_path / "regions")
    def edit(data):
        if kind == "missing":
            data["features"].pop()
        elif kind == "foreign":
            data["features"][0]["properties"]["region_uid"] = "foreign"
        elif kind == "duplicate":
            data["features"][0]["properties"]["region_uid"] = data["features"][1]["properties"]["region_uid"]
        else:
            data["features"][0]["geometry"]["coordinates"][0][1][0] += 1.
    rewrite_region_geometry(package, link, edit)
    with pytest.raises(ValueError, match=match):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "bad.zarr", measured_assay=[package], measured_region_shapes=[link])


def test_region_link_must_be_unique_and_used(specimen, tmp_path):
    inputs = specimen[0]
    package, link, _ = measured_region_package(inputs, tmp_path / "regions")
    with pytest.raises(ValueError, match="Duplicate measured region shape-link"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "dup.zarr", measured_assay=[package], measured_region_shapes=[link, link])
    with pytest.raises(ValueError, match="Unused measured region shape-link"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "unused.zarr", measured_region_shapes=[link])
    with pytest.raises(ValueError, match="region-shape linkage"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "missing.zarr", measured_assay=[package])


def test_cli_region_export_and_expected_sample_id(specimen, tmp_path):
    inputs = specimen[0]
    package, link, _ = measured_region_package(inputs, tmp_path / "regions")
    command = [sys.executable, str(BIN / "export_spatialdata.py"), "--outdir", str(tmp_path / "cli-region.zarr"),
        "--measured-assay", str(package), "--measured-region-shapes", str(link), "--sample-id", "sample A"]
    for name, value in inputs.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["measured_assays"] == 1
    with pytest.raises(ValueError, match="requested sample-id"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "wrong-sample.zarr", expected_sample_id="sample_B")
