"""Real SpatialData hierarchy/overlap round trips; synthetic engineering QA."""
import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

sd = pytest.importorskip("spatialdata")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import export_spatialdata as exporter
from cell_profile_io import sha256_file
from test_spatialdata_export import affine, measured_package, specimen


def refresh_hierarchy(directory):
    manifest_path = directory / "region_profiles" / "region_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in manifest["files"]:
        manifest["files"][name] = sha256_file(manifest_path.parent / name)
    manifest_path.write_text(json.dumps(manifest))
    path = directory / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary["outputs"] = {str(file.relative_to(directory)): sha256_file(file)
                          for file in directory.rglob("*") if file.is_file() and file != path}
    path.write_text(json.dumps(summary))


@pytest.fixture
def hierarchy_specimen(specimen, tmp_path):
    inputs, image, labels, _ = specimen
    inputs = dict(inputs)
    directory = tmp_path / "hierarchy"
    region_root = directory / "region_profiles"
    region_root.mkdir(parents=True)
    parent = np.ones(labels.shape, np.uint16)
    parent[:, 48:] = 2
    parent[:4] = 0
    parent[8:10, 10:12] = 0
    region = np.ones(labels.shape, np.uint32)
    region[:, 16:48] = 2
    region[:, 48:] = 3
    region[parent == 0] = 0
    uncertainty = np.zeros(labels.shape, np.uint8)
    uncertainty[35:38, 60:64] = 3
    region[uncertainty > 0] = 0
    region[40:44, 65:69] = 0
    status = np.where(region > 0, 1, np.where(parent > 0, 2, 0)).astype(np.uint8)
    status[uncertainty > 0] = 10
    maps = {"parent_domains.ome.tif": parent, "subdomain_mask.ome.tif": region,
            "region_mask.ome.tif": region, "hierarchy_status.ome.tif": status,
            "parent_uncertainty.ome.tif": uncertainty}
    for name, values in maps.items():
        tifffile.imwrite(directory / name, values, tile=(16, 16), compression="deflate")
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"status": "pass", "mpp_x": .5, "mpp_y": .5,
                                      "width_px": 1000, "height_px": 1000}))
    inputs.update(resolution_json=resolution, hierarchy_dir=directory)
    profile_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(profile_path.read_text())
    manifest["inputs"] = {"image_sha256": sha256_file(inputs["image"]),
        "labels_sha256": sha256_file(inputs["labels"]), "shift_sha256": sha256_file(inputs["shift"]),
        "resolution_json_sha256": sha256_file(resolution),
        "domain_mask_sha256": sha256_file(directory / "parent_domains.ome.tif"),
        "domain_uncertainty_sha256": sha256_file(directory / "parent_uncertainty.ome.tif")}
    profile_path.write_text(json.dumps(manifest))
    records = []
    for value, repx in ((1, 8), (2, 30), (3, 80)):
        yy, xx = np.nonzero(region == value)
        records.append({"region_uid": f"sample%20A::hierarchy-test::region_{value}",
            "region_id": str(value), "sample_id": "sample A", "parent_domain_id": 1 if value < 3 else 2,
            "subdomain_id": value, "local_subdomain_id": value if value < 3 else 1,
            "area_um2": len(xx) * .25, "x_um": (xx.mean() + .5 + 100) * .5,
            "y_um": (yy.mean() + .5 + 200) * .5, "representative_grid_id": value,
            "representative_x_um": (repx + 100) * .5, "representative_y_um": 115.,
            "grid_observations": 1, "label_status": "unsupervised_discovery"})
    regions = pd.DataFrame(records)
    regions.to_csv(region_root / "region_profiles.csv", index=False)
    regions[["region_uid", "region_id", "sample_id"]].to_csv(region_root / "feature_rows.csv", index=False)
    blocks = {"local": np.array([[1., 2.], [3., 4.], [5., 6.]], np.float32),
              "context": np.array([[.1, .2, .3], [.4, .5, .6], [np.nan] * 3], np.float32)}
    region_manifest = {"schema_version": "1.0.0", "observation_unit": "tissue_region",
        "region_count": 3, "hierarchy_id": "hierarchy-test", "coordinate_space": "original_slide_micrometres",
        "feature_blocks": {}, "files": {"region_profiles.csv": "", "feature_rows.csv": ""}}
    for name, block in blocks.items():
        np.save(region_root / f"{name}.npy", block)
        region_manifest["feature_blocks"][name] = {"path": f"{name}.npy", "shape": list(block.shape),
            "feature_names": [f"{name}_{i}" for i in range(block.shape[1])],
            "feature_definition": {"model": "synthetic-test", "observation_unit": "tissue_region",
                                   "aggregation": "assigned_grid_core_tissue_area_weighted_mean"}}
        region_manifest["files"][f"{name}.npy"] = ""
    (region_root / "region_profiles_manifest.json").write_text(json.dumps(region_manifest))
    grid = pd.DataFrame({"label": [1, 2, 3], "sample_id": ["sample A"] * 3,
        "parent_domain_id": [1, 1, 2], "subdomain_id": [1, 2, 3], "status_code": [1, 1, 1],
        "x_um": regions.representative_x_um, "y_um": regions.representative_y_um})
    grid.to_csv(directory / "grid_subdomains.csv", index=False)
    summary = {"schema_version": "1.0.0", "sample_id": "sample A", "hierarchy_id": "hierarchy-test",
        "parent_labels_immutable": True, "regions": 3,
        "geometry": {"shape_yx": list(labels.shape), "mpp_xy": [.5, .5], "origin_px_xy": [100, 200],
                     "origin_um_xy": [50., 100.], "coordinate_space": "original_slide_micrometres"},
        "status_codes": {"0": "background", "1": "accepted_subdomain", "2": "parent_only_no_observation",
                         "10": "parent_assignment_uncertain"},
        "parent_uncertainty_semantics": "0 accepted; 3 preserved upstream categorical uncertainty",
        "inputs": {key: {"sha256": sha256_file(path)} for key, path in {
            "image": inputs["image"], "shift_json": inputs["shift"], "resolution_json": resolution,
            "parent_mask": directory / "parent_domains.ome.tif",
            "parent_uncertainty": directory / "parent_uncertainty.ome.tif"}.items()}}
    (directory / "hierarchy_summary.json").write_text(json.dumps(summary))
    refresh_hierarchy(directory)
    return inputs, maps, regions, blocks


def linked_inputs(hierarchy_specimen, tmp_path):
    from link_cell_tissue_hierarchy import link_profiles
    inputs = dict(hierarchy_specimen[0])
    target = tmp_path / "linked_profiles"
    link_profiles(inputs["profile_dir"], inputs["labels"], inputs["hierarchy_dir"], target, tile_size=16)
    inputs["profile_dir"] = target
    return inputs


def test_hierarchy_native_arrays_region_features_and_overlap_round_trip(hierarchy_specimen, tmp_path):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    maps, regions, blocks = hierarchy_specimen[1:]
    originals = {str(path): sha256_file(path) for directory in (inputs["profile_dir"], inputs["hierarchy_dir"])
                 for path in directory.rglob("*") if path.is_file()}
    output = tmp_path / "hierarchy.zarr"
    result = exporter.export_spatialdata(**inputs, outdir=output)
    loaded = sd.SpatialData.read(output)
    assert set(loaded.labels) == {"canonical_cells", *exporter.HIERARCHY_LABELS.values()}
    mapping = {"parent": "parent_domains.ome.tif", "subdomain": "subdomain_mask.ome.tif",
        "region": "region_mask.ome.tif", "status": "hierarchy_status.ome.tif",
        "parent_uncertainty": "parent_uncertainty.ome.tif"}
    for key, name in exporter.HIERARCHY_LABELS.items():
        np.testing.assert_array_equal(loaded.labels[name].data.compute(), maps[mapping[key]])
        np.testing.assert_array_equal(affine(loaded.labels[name]), [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    table = loaded.tables["hierarchy_region_profiles"]
    assert table.obs_names.tolist() == regions.region_uid.tolist()
    assert table.obs.instance_id.tolist() == [1, 2, 3]
    assert table.obs.spatial_region.astype(str).eq("hierarchy_regions").all()
    assert table.uns["spatialdata_attrs"]["instance_key"] == "instance_id"
    assert not {"cell_uid", "cell_id"} & set(table.obs)
    for name, block in blocks.items():
        np.testing.assert_array_equal(table.obsm[name], block)
        assert table.obsm[name].dtype == block.dtype
    np.testing.assert_array_equal(table.obsm["spatial"], regions[["x_um", "y_um"]].to_numpy())
    relation = loaded.tables["cell_hierarchy_overlaps"]
    expected = pd.read_parquet(inputs["profile_dir"] / "cell_hierarchy_overlaps.parquet")
    assert "spatialdata_attrs" not in relation.uns
    assert relation.obs.cell_uid.duplicated().any()
    for column in expected:
        np.testing.assert_array_equal(relation.obs[column].to_numpy(), expected[column].to_numpy())
    assert relation.obs.loc[relation.obs.region_id == 0, "region_uid"].astype(str).eq("").all()
    by_cell = relation.obs.groupby(["cell_uid", "compartment"], observed=True).overlap_fraction.sum()
    np.testing.assert_allclose(by_cell, 1)
    assert set(loaded.tables) == {"cells", "tissue_domain_annotations", "hierarchy_region_profiles", "cell_hierarchy_overlaps"}
    cells = loaded.tables["cells"]
    assert len(cells) == 2 and cells.obs.cell_id.astype(str).tolist() == ["001", "7"]
    np.testing.assert_array_equal(cells.obs.predicted__nucleus__CD3__mean, [.1, np.nan])
    assert result["tissue_hierarchy"]["region_count"] == 3
    assert {path: sha256_file(path) for path in originals} == originals


def test_hierarchy_requires_linked_profile_and_linked_profile_requires_hierarchy(hierarchy_specimen, tmp_path):
    inputs = hierarchy_specimen[0]
    with pytest.raises(ValueError, match="link|Link"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "unlinked.zarr")
    linked = linked_inputs(hierarchy_specimen, tmp_path)
    linked.pop("hierarchy_dir")
    with pytest.raises(ValueError, match="require --hierarchy-dir"):
        exporter.export_spatialdata(**linked, outdir=tmp_path / "dropped.zarr")
    assert not (tmp_path / "unlinked.zarr").exists()
    assert not (tmp_path / "dropped.zarr").exists()


@pytest.mark.parametrize("source", ["image", "shift", "resolution_json"])
def test_hierarchy_binds_actual_export_sources(hierarchy_specimen, tmp_path, source):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    if source == "image":
        image = tifffile.imread(inputs[source])
        image[0, 0, 0] ^= 1
        path = tmp_path / "other_image.tif"
        tifffile.imwrite(path, image, tile=(16, 16), compression="deflate")
    else:
        payload = json.loads(inputs[source].read_text())
        payload["unrelated_metadata"] = "same geometry but different provenance"
        path = tmp_path / f"other_{source}.json"
        path.write_text(json.dumps(payload))
    inputs[source] = path
    with pytest.raises(ValueError, match="SHA256|hash mismatch"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "bad.zarr")
    assert not (tmp_path / "bad.zarr").exists()


def test_hierarchy_map_corruption_fails_before_export(hierarchy_specimen, tmp_path):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    path = inputs["hierarchy_dir"] / "region_mask.ome.tif"
    values = tifffile.imread(path)
    values[8, 12] = 100
    tifffile.imwrite(path, values, tile=(16, 16), compression="deflate")
    with pytest.raises(ValueError, match="SHA256|hash"):
        exporter.export_spatialdata(**inputs, outdir=tmp_path / "bad.zarr")
    assert not (tmp_path / "bad.zarr").exists()


@pytest.mark.parametrize("assay_registry", ["original", "linked"])
def test_hierarchy_and_measured_modalities_preserve_exact_independent_lineage(hierarchy_specimen, tmp_path, assay_registry):
    original_inputs = hierarchy_specimen[0]
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    package = measured_package(original_inputs if assay_registry == "original" else inputs, tmp_path / "assay")
    hashes = {str(path): sha256_file(path) for path in package.rglob("*") if path.is_file()}
    output = tmp_path / "hierarchy_measured.zarr"
    result = exporter.export_spatialdata(**inputs, measured_assay=[package], outdir=output)
    record = result["measured_modalities"][0]
    expected_status = "verified_additive_hierarchy_source_registry" if assay_registry == "original" else "current_profile_registry"
    assert record["profile_registry_binding"]["status"] == expected_status
    source_root = original_inputs["profile_dir"] if assay_registry == "original" else inputs["profile_dir"]
    assert record["canonical_profile_manifest_sha256"] == sha256_file(source_root / "cell_profiles_manifest.json")
    assert record["profile_registry_binding"]["exported_profile_manifest_sha256"] == sha256_file(inputs["profile_dir"] / "cell_profiles_manifest.json")
    loaded = sd.SpatialData.read(output)
    measured = loaded.tables[record["table"]]
    assert measured.X.dtype == np.float64
    np.testing.assert_array_equal(measured.X, [[16777217.25, np.nan], [np.nan, np.nan]])
    assert measured.obs_names.tolist() == loaded.tables["cells"].obs_names.tolist()
    assert measured.obs.instance_id.tolist() == [1, 7]
    assert len(loaded.tables["hierarchy_region_profiles"]) == 3
    assert len(loaded.tables["cells"]) == 2
    np.testing.assert_array_equal(loaded.tables["cells"].obs.predicted__nucleus__CD3__mean, [.1, np.nan])
    assert {path: sha256_file(path) for path in hashes} == hashes


def test_additive_measured_registry_cannot_hide_changed_cell_value(hierarchy_specimen, tmp_path):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    package = measured_package(hierarchy_specimen[0], tmp_path / "assay")
    root = inputs["profile_dir"]
    manifest = json.loads((root / "cell_profiles_manifest.json").read_text())
    cells = pd.read_parquet(root / "cell_profiles.parquet")
    cells.loc[0, "predicted__nucleus__CD3__mean"] = .99
    header = json.loads((package / "measured_assay_manifest.json").read_text())
    with pytest.raises(ValueError, match="changed an original canonical cell"):
        exporter.measured_profile_registry(header, root, cells, manifest)


def test_hierarchy_full_roundtrip_reading_is_window_bounded(hierarchy_specimen, tmp_path, monkeypatch):
    inputs = linked_inputs(hierarchy_specimen, tmp_path)
    reader_window = exporter.RasterReader.window
    windows = []

    def bounded(self, x, y, x1, y1):
        windows.append((x1 - x, y1 - y))
        assert x1 - x <= 16 and y1 - y <= 16
        return reader_window(self, x, y, x1, y1)

    monkeypatch.setattr(exporter.RasterReader, "window", bounded)
    exporter.export_spatialdata(**inputs, outdir=tmp_path / "bounded.zarr")
    assert len(windows) > 100


def test_no_accepted_regions_keeps_parent_tissue_and_unresolved_cell_links(hierarchy_specimen, tmp_path):
    inputs, maps, regions, blocks = hierarchy_specimen
    root = inputs["hierarchy_dir"]
    for name in ("subdomain_mask.ome.tif", "region_mask.ome.tif"):
        tifffile.imwrite(root / name, np.zeros_like(maps[name]), tile=(16, 16), compression="deflate")
    status = np.where(maps["parent_domains.ome.tif"] > 0, 2, 0).astype(np.uint8)
    status[maps["parent_uncertainty.ome.tif"] > 0] = 10
    tifffile.imwrite(root / "hierarchy_status.ome.tif", status, tile=(16, 16), compression="deflate")
    region_root = root / "region_profiles"
    regions.iloc[:0].to_csv(region_root / "region_profiles.csv", index=False)
    regions.iloc[:0][["region_uid", "region_id", "sample_id"]].to_csv(region_root / "feature_rows.csv", index=False)
    manifest_path = region_root / "region_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["region_count"] = 0
    for name, block in blocks.items():
        np.save(region_root / f"{name}.npy", block[:0])
        manifest["feature_blocks"][name]["shape"][0] = 0
    manifest_path.write_text(json.dumps(manifest))
    summary_path = root / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["regions"] = 0
    summary_path.write_text(json.dumps(summary))
    refresh_hierarchy(root)
    linked = linked_inputs(hierarchy_specimen, tmp_path)
    output = tmp_path / "unresolved.zarr"
    exporter.export_spatialdata(**linked, outdir=output)
    loaded = sd.SpatialData.read(output)
    assert len(loaded.tables["hierarchy_region_profiles"]) == 0
    assert loaded.tables["hierarchy_region_profiles"].obsm["local"].shape == (0, 2)
    assert loaded.tables["cell_hierarchy_overlaps"].obs.region_id.eq(0).all()
    assert len(loaded.tables["cells"]) == 2
    np.testing.assert_array_equal(loaded.labels["hierarchy_parent_domains"].data.compute(), maps["parent_domains.ome.tif"])


def test_legacy_discovery_cli_to_linker_to_spatialdata(hierarchy_specimen, tmp_path):
    """Exercise explicit legacy comparison; native fitting has its own CLI suite."""
    inputs = dict(hierarchy_specimen[0])
    source_hierarchy = inputs["hierarchy_dir"]
    producer_python = sys.executable if importlib.util.find_spec("sklearn") else ROOT / ".venv-spatial/bin/python"
    if not Path(producer_python).is_file():
        pytest.skip("Real discovery integration needs the core scikit-learn runtime")
    yy, xx = np.indices((16, 24))
    grid = pd.DataFrame({"label": np.arange(1, 385), "x": (xx.ravel() * 4 + 2), "y": yy.ravel() * 4 + 2,
        "grid_row": yy.ravel(), "grid_col": xx.ravel(), "core_x0": xx.ravel() * 4, "core_y0": yy.ravel() * 4,
        "core_x1": (xx.ravel() + 1) * 4, "core_y1": (yy.ravel() + 1) * 4})
    truth = (grid.y.to_numpy() >= 32).astype(int)
    rng = np.random.default_rng(7)
    blocks = {"local": rng.normal(0, .08, (len(grid), 6)) + truth[:, None] * 5,
              "context": rng.normal(0, .08, (len(grid), 10)) + truth[:, None] * 3}
    base = {"model_id": "synthetic-test/uni2", "model_revision": "a" * 40,
            "preprocessing": "rgb,scale-preserving,model-normalization", "pooling": "cls"}
    definitions = {"local": {**base, "field_width_source_px": 4, "input_context_width_source_px": 4},
                   "context": {**base, "field_width_source_px": 12, "input_context_width_source_px": 12}}
    grid_path = tmp_path / "source_grid.csv"
    grid.to_csv(grid_path, index=False)
    metadata = {"observation_type": "spatial_grid", "coordinate_space": "crop_roi_level0_pixels",
        "image_height_px": 64, "image_width_px": 96,
        "source_mpp_x": .5, "source_mpp_y": .5}
    metadata_path = tmp_path / "grid_metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    definition_path = tmp_path / "source_features.json"
    definition_path.write_text(json.dumps(definitions))
    source_inputs = {key: {"sha256": sha256_file(path)} for key, path in {
        "image": inputs["image"], "grid_objects": grid_path, "grid_metadata": metadata_path,
        "shift_json": inputs["shift"], "resolution_json": inputs["resolution_json"]}.items()}
    for name, values in blocks.items():
        bundle = tmp_path / f"source_{name}"
        bundle.mkdir()
        np.save(bundle / "embeddings.npy", values.astype(np.float32))
        frame = pd.DataFrame({"cell_id": grid.label, "cx": grid.x, "cy": grid.y,
            "observation_type": "grid", "source_mpp": .5,
            "extraction_tile_size": definitions[name]["input_context_width_source_px"]})
        frame.to_csv(bundle / "rows.csv", index=False)
        resolved_definition = {**definitions[name], "source_mpp_xy": [.5, .5],
            "field_width_um_xy": [definitions[name]["field_width_source_px"] * .5] * 2,
            "input_context_width_um_xy": [definitions[name]["input_context_width_source_px"] * .5] * 2}
        (bundle / "embedding_manifest.json").write_text(json.dumps({
            "format": "cellphenotyper_grid_embeddings_npy", "sample_id": "sample A",
            "source_inputs": source_inputs, "feature_definition": resolved_definition,
            "feature_names": [f"feat_{i + 1}" for i in range(values.shape[1])],
            "matrix": {"path": "embeddings.npy", "sha256": sha256_file(bundle / "embeddings.npy"), "shape": list(values.shape)},
            "rows": {"path": "rows.csv", "sha256": sha256_file(bundle / "rows.csv"), "count": len(grid)}}))
    discovered = tmp_path / "actual_discovery"
    command = [str(producer_python), str(ROOT / "bin" / "discover_tissue_hierarchy.py"),
        "--discovery-method", "legacy_kmeans",
        "--parent-mask", str(source_hierarchy / "parent_domains.ome.tif"),
        "--parent-uncertainty", str(source_hierarchy / "parent_uncertainty.ome.tif"),
        "--image", str(inputs["image"]), "--grid-objects", str(grid_path),
        "--grid-metadata", str(metadata_path), "--shift-json", str(inputs["shift"]),
        "--resolution-json", str(inputs["resolution_json"]),
        "--local-embeddings", str(tmp_path / "source_local"),
        "--context-embeddings", str(tmp_path / "source_context"),
        "--embedding-metadata", str(definition_path), "--sample-id", "sample A",
        "--min-observations", "8", "--fixed-k", "2", "--max-k", "2", "--repeats", "2",
        "--tile-size", "16", "--outdir", str(discovered)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((discovered / "hierarchy_summary.json").read_text())
    assert summary["regions"] == 4 and summary["subdomains"] == 4
    assert sha256_file(discovered / "parent_domains.ome.tif") == sha256_file(source_hierarchy / "parent_domains.ome.tif")
    assert sha256_file(discovered / "parent_uncertainty.ome.tif") == sha256_file(source_hierarchy / "parent_uncertainty.ome.tif")
    inputs["hierarchy_dir"] = discovered
    linked = linked_inputs((inputs, *hierarchy_specimen[1:]), tmp_path)
    output = tmp_path / "actual_discovery.zarr"
    exporter.export_spatialdata(**linked, outdir=output)
    loaded = sd.SpatialData.read(output)
    regions = loaded.tables["hierarchy_region_profiles"]
    mask = loaded.labels["hierarchy_regions"].data.compute()
    assert len(regions) == 4 and len(loaded.tables["cells"]) == 2
    for row in regions.obs.itertuples():
        yy, xx = np.nonzero(mask == row.instance_id)
        np.testing.assert_allclose([row.x_um, row.y_um], [(xx.mean() + .5 + 100) * .5,
                                                       (yy.mean() + .5 + 200) * .5], rtol=0, atol=1e-12)
        assert row.area_um2 == len(xx) * .25
    for name in ("local", "context"):
        np.testing.assert_array_equal(regions.obsm[name], np.load(discovered / "region_profiles" / f"{name}.npy"))
    relation = loaded.tables["cell_hierarchy_overlaps"].obs
    assert set(relation.cell_uid.astype(str)) == set(loaded.tables["cells"].obs_names)
    assert relation.overlap_pixels.groupby(relation.cell_uid, observed=True).sum().tolist() == [144, 196]
    assert not np.any(mask[tifffile.imread(discovered / "parent_uncertainty.ome.tif") > 0])
    from cell_inspector import CellInspector
    inspector = CellInspector(profile_dir=linked["profile_dir"], image=linked["image"],
        labels=linked["labels"], shift=linked["shift"], resolution_json=linked["resolution_json"],
        hierarchy_dir=discovered)
    assert inspector.metadata()["cell_count"] == 2
    region_metadata = inspector.regions.metadata()
    assert region_metadata["region_count"] == 4
    detail = inspector.regions.detail(region_metadata["first_region_uid"], 64)
    assert detail["region_uid"] == region_metadata["first_region_uid"]


def test_preserved_ring_geometry_exports_without_original_compartment_directory(hierarchy_specimen, tmp_path):
    from link_cell_tissue_hierarchy import link_profiles

    inputs = dict(hierarchy_specimen[0])
    compartments = tmp_path / "original_compartments"
    compartments.mkdir()
    ring = np.zeros((64, 96), np.uint32)
    ring[20:22, 10:22] = 1
    ring_path = compartments / "labels_perinuclear_ring.tif"
    tifffile.imwrite(ring_path, ring, tile=(16, 16), compression="deflate")
    summary_path = compartments / "compartment_summary.json"
    summary_path.write_text(json.dumps({"inputs": {
        "labels": {"sha256": sha256_file(inputs["labels"])},
        "shift": {"sha256": sha256_file(inputs["shift"])},
        "resolution": {"sha256": sha256_file(inputs["resolution_json"])}},
        "output_artifacts": {"perinuclear_ring": {"sha256": sha256_file(ring_path)}}}))
    manifest_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("tables", {})["compartment_qc"] = {"mask_lineage_verified": True,
        "summary_sha256": sha256_file(summary_path)}
    manifest_path.write_text(json.dumps(manifest))
    target = tmp_path / "linked_profiles"
    link_profiles(inputs["profile_dir"], inputs["labels"], inputs["hierarchy_dir"], target,
                  ring_labels=ring_path, tile_size=16)
    inputs["profile_dir"] = target
    compartments.rename(tmp_path / "original_compartments_unavailable")
    output = tmp_path / "ring.zarr"
    exporter.export_spatialdata(**inputs, outdir=output)
    loaded = sd.SpatialData.read(output)
    np.testing.assert_array_equal(loaded.labels[exporter.CELL_RING_LABELS].data.compute(), ring)
    np.testing.assert_array_equal(affine(loaded.labels[exporter.CELL_RING_LABELS]), [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    relation = loaded.tables["cell_hierarchy_overlaps"]
    assert set(relation.obs.compartment.astype(str)) == {"nucleus", "perinuclear_ring"}
    ring_rows = relation.obs.loc[relation.obs.compartment == "perinuclear_ring"]
    assert set(ring_rows.cell_id.astype(str)) == {"001"}
    assert ring_rows.overlap_pixels.sum() == 24
    assert json.loads(relation.uns["cellphenotyper"]["foreign_keys_json"])["compartment"]["geometry_labels"]["perinuclear_ring"] == exporter.CELL_RING_LABELS
    cells = loaded.tables["cells"]
    assert cells.obs.hierarchy_perinuclear_ring_pixels.tolist() == [24, 0]
    assert np.isnan(cells.obs.hierarchy_perinuclear_ring_accepted_fraction.iloc[1])


def test_absent_parent_uncertainty_remains_unavailable_not_accepted(hierarchy_specimen, tmp_path):
    inputs, maps, _, _ = hierarchy_specimen
    root = inputs["hierarchy_dir"]
    status = maps["hierarchy_status.ome.tif"].copy()
    status[status == 10] = 2
    tifffile.imwrite(root / "hierarchy_status.ome.tif", status, tile=(16, 16), compression="deflate")
    (root / "parent_uncertainty.ome.tif").unlink()
    summary_path = root / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["inputs"].pop("parent_uncertainty")
    summary.pop("parent_uncertainty_semantics")
    summary_path.write_text(json.dumps(summary))
    refresh_hierarchy(root)
    profile_path = inputs["profile_dir"] / "cell_profiles_manifest.json"
    manifest = json.loads(profile_path.read_text())
    manifest["inputs"].pop("domain_uncertainty_sha256")
    profile_path.write_text(json.dumps(manifest))
    linked = linked_inputs(hierarchy_specimen, tmp_path)
    output = tmp_path / "unavailable_uncertainty.zarr"
    exporter.export_spatialdata(**linked, outdir=output)
    loaded = sd.SpatialData.read(output)
    assert "hierarchy_parent_uncertainty" not in loaded.labels
    table = loaded.tables["cell_hierarchy_overlaps"]
    assert table.obs.parent_uncertainty_code.eq(255).all()
    assert not table.uns["cellphenotyper"]["parent_uncertainty_available"]
    assert "not supplied" in table.uns["cellphenotyper"]["parent_uncertainty_code_semantics"]
    assert loaded.tables["cells"].obs.hierarchy_nucleus_parent_uncertain_fraction.isna().all()
