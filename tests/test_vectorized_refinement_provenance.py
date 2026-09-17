"""Native-raster provenance survives vector smoothing without confidence claims."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
spec = importlib.util.spec_from_file_location("vector_provenance", ROOT / "bin/mask_to_geojson.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path, compressed=False):
    labels = np.zeros((17, 19), np.uint16)
    labels[1:12, 1:12] = 1
    labels[4:6, 4:6] = 0
    labels[9:16, 13:18] = 2
    labels[14, 1] = 3
    uncertain = np.zeros(labels.shape, np.uint8)
    provenance = (labels > 0).astype(np.uint8)
    for x, p, u in ((1, 2, 250), (2, 3, 2), (3, 5, 251), (4, 8, 254), (5, 7, 0)):
        provenance[1, x], uncertain[1, x] = p, u
    provenance[4:6, 4:6], uncertain[4:6, 4:6] = 4, 253
    provenance[0, 1], uncertain[0, 1] = 6, 252
    paths = {}
    for name, array in (("labels", labels), ("uncertainty", uncertain), ("provenance", provenance)):
        paths[name] = tmp_path / f"{name}.tif"
        options = {"tile": (16, 16), "compression": "deflate"} if compressed else {}
        tifffile.imwrite(paths[name], array, **options)
    metadata = {"schema_version": "1.1.0", "coordinate_frame": "analysis_crop", "shape_yx": list(labels.shape),
        "provenance_codes": {str(k): v for k, v in module.REFINEMENT_CODES.items()},
        "original_uncertainty_codes_preserved": True, "input_uncertainty_provided": True,
        "pixel_counts": {module.REFINEMENT_CODES[int(code)]: int(count) for code, count in zip(*np.unique(provenance, return_counts=True))}}
    paths["metadata"] = tmp_path / "metadata.json"
    paths["metadata"].write_text(json.dumps(metadata))
    return paths, labels, uncertain, provenance


def summarize(paths, **kwargs):
    return module.raster_provenance_summary(paths["labels"], paths["uncertainty"], paths["provenance"], paths["metadata"], **kwargs)


def run_cli(tmp_path, paths, extra=(), sidecars=True):
    out = tmp_path / "result.geojson"
    args = [sys.executable, str(ROOT / "bin/mask_to_geojson.py"), "--mask", str(paths["labels"]),
        "--out", str(out), "--polygon-backend", "rasterio", "--sample-key", "specimen::standard",
        "--sample-id", "specimen", "--cluster-variant", "standard"]
    if sidecars:
        args += ["--uncertainty-mask", str(paths["uncertainty"]), "--provenance-mask", str(paths["provenance"]), "--provenance-metadata", str(paths["metadata"])]
    result = subprocess.run([*args, *extra], text=True, capture_output=True, timeout=30)
    return result, out


@pytest.mark.parametrize("compressed", [False, True])
def test_exact_native_multiclass_counts_are_bounded_and_include_holes(tmp_path, monkeypatch, compressed):
    paths, labels, uncertain, provenance = fixture(tmp_path, compressed)
    from profile_cell_morphology import WindowReader
    original = WindowReader.read
    windows = []
    def read(self, x0, y0, x1, y1):
        windows.append((x1-x0, y1-y0))
        return original(self, x0, y0, x1, y1)
    monkeypatch.setattr(WindowReader, "read", read)
    before = {key: module.file_sha256(path) for key, path in paths.items()}
    result = summarize(paths, tile_size=3)
    assert max(w for w, h in windows) <= 3 and max(h for w, h in windows) <= 3
    assert result["io"]["maximum_scan_window_pixels"] <= 9
    for label in np.unique(labels):
        selected = labels == label
        record = result["label_summaries"][str(label)]
        assert record["source_pixel_count"] == int(selected.sum())
        assert record["uncertainty_counts"] == {str(c): int(n) for c, n in zip(*np.unique(uncertain[selected], return_counts=True))}
        assert record["provenance_counts"] == {str(c): int(n) for c, n in zip(*np.unique(provenance[selected], return_counts=True))}
        assert record["accepted_biological_confidence"] is None
    assert result["label_summaries"]["0"]["provenance_counts"]["4"] == 4
    assert result["label_summaries"]["0"]["uncertainty_counts"]["253"] == 4
    assert before == {key: module.file_sha256(path) for key, path in paths.items()}
    assert result["inputs"]["labels"]["sha256"] == before["labels"]


def test_smoothed_hole_filled_vectors_link_native_not_geometry_fractions(tmp_path):
    paths, labels, _, _ = fixture(tmp_path)
    result, out = run_cli(tmp_path, paths, ["--fill-holes", "--smooth-buffer", "0.2", "--simplify", "0.1", "--min-area", "2", "--dissolve-by-value"])
    assert result.returncode == 0, result.stdout + result.stderr
    geojson = json.loads(out.read_text())
    summary = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert {feature["properties"]["value"] for feature in geojson["features"]} == {1, 2}
    assert "3" in summary["label_summaries"]  # Tiny vector-filtered label remains in source inventory.
    assert summary["label_summaries"]["0"]["provenance_counts"]["4"] == 4
    for feature in geojson["features"]:
        label = feature["properties"]["value"]
        evidence = feature["properties"]["raster_provenance"]
        assert evidence["source_pixel_count"] == int((labels == label).sum())
        assert evidence["summary"]["sha256"] == module.file_sha256(Path(str(out) + ".provenance.json"))
        assert evidence["summary_key"] == f"label_summaries/{label}"
        assert "not_this_vector_geometry" in evidence["scope"]
        assert evidence["calibrated_confidence"] is None
    assert "NOT polygon-specific" in summary["vectorization"]["summary_scope"]
    assert summary["lineage"] == {"sample_key": "specimen::standard", "sample_id": "specimen", "cluster_variant": "standard"}


def test_multiclass_vectors_preserve_nested_class_holes_without_overlap(tmp_path):
    paths, labels, uncertain, provenance = fixture(tmp_path)
    labels[:] = 0
    labels[1:16, 1:18] = 1
    labels[4:13, 5:15] = 2
    uncertain[:] = 0
    provenance[:] = (labels > 0).astype(np.uint8)
    tifffile.imwrite(paths["labels"], labels)
    tifffile.imwrite(paths["uncertainty"], uncertain)
    tifffile.imwrite(paths["provenance"], provenance)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["pixel_counts"] = {
        module.REFINEMENT_CODES[int(code)]: int(count)
        for code, count in zip(*np.unique(provenance, return_counts=True))
    }
    paths["metadata"].write_text(json.dumps(metadata))

    result, out = run_cli(tmp_path, paths, ["--dissolve-by-value"])
    assert result.returncode == 0, result.stdout + result.stderr
    collection = json.loads(out.read_text())
    geometries = {
        feature["properties"]["value"]: module.shp_shape(feature["geometry"])
        for feature in collection["features"]
    }
    assert set(geometries) == {1, 2}
    assert geometries[1].intersection(geometries[2]).area == 0
    assert geometries[1].geom_type == "Polygon" and len(geometries[1].interiors) == 1
    summary = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert summary["vectorization"]["multiclass_geometry_mutually_exclusive"] is True
    assert summary["vectorization"]["multiclass_overlap_area_px2"] == 0


def test_pipeline_multiclass_geojson_defaults_preserve_topology():
    config = (ROOT / "nextflow.config").read_text()
    assert "cluster_geojson_polygon_backend = 'rasterio'" in config
    assert "cluster_geojson_connectivity   = 4" in config
    assert "cluster_geojson_page           = 0" in config
    assert "cluster_geojson_smooth_buffer  = 0.0" in config
    assert "cluster_geojson_min_area       = 0" in config
    assert "cluster_geojson_simplify       = 16.0" in config
    assert "cluster_geojson_shared_boundary_simplify = true" in config
    assert "cluster_geojson_simplify_max_categorical_difference_fraction = 0.01" in config
    assert "cluster_geojson_shared_boundary_smooth = true" in config
    assert "cluster_geojson_shared_boundary_smoothing_coefficient = 0.05" in config
    assert "cluster_geojson_shared_boundary_minimum_turn_degrees = 45.0" in config
    assert "cluster_geojson_shared_boundary_minimum_adjacent_length = 16.0" in config
    assert "cluster_geojson_fill_holes     = false" in config


def test_shared_boundary_coverage_simplification_keeps_clusters_exclusive():
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "coverage_simplify"):
        pytest.skip("Shapely 2.1 coverage simplification unavailable")
    left = module.Polygon([(0, 0), (5, 0), (5, 2), (4, 3), (5, 4), (5, 6), (0, 6)])
    right = module.Polygon([(5, 0), (10, 0), (10, 6), (5, 6), (5, 4), (4, 3), (5, 2)])
    simplified = dict(module.simplify_multiclass_coverage([(1, left), (2, right)], 1.5))
    assert set(simplified) == {1, 2}
    assert simplified[1].intersection(simplified[2]).area == 0
    assert bool(shapely.coverage_is_valid(list(simplified.values())))


def test_shared_boundary_coverage_nodes_mismatched_collinear_vertices():
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "coverage_simplify"):
        pytest.skip("Shapely 2.1 coverage simplification unavailable")
    left = module.Polygon([(0, 0), (5, 0), (5, 3), (5, 6), (0, 6)])
    right = module.Polygon([(5, 0), (10, 0), (10, 6), (5, 6)])
    assert not bool(shapely.coverage_is_valid([left, right]))
    rows, diagnostics = module.simplify_multiclass_coverage(
        [(1, left), (2, right)],
        1.0,
        return_diagnostics=True,
    )
    simplified = dict(rows)
    assert diagnostics["edge_match_repair_applied"] is True
    assert diagnostics["repaired_coverage_valid"] is True
    assert simplified[1].intersection(simplified[2]).area == 0


def test_shared_boundary_simplification_obeys_fidelity_limit():
    shapely = pytest.importorskip("shapely")
    if not hasattr(shapely, "coverage_simplify"):
        pytest.skip("Shapely 2.1 coverage simplification unavailable")
    left = module.Polygon([(0, 0), (5, 0), (5, 1), (4, 2), (5, 3), (5, 4), (0, 4)])
    right = module.Polygon([(5, 0), (10, 0), (10, 4), (5, 4), (5, 3), (4, 2), (5, 1)])
    rows, diagnostics = module.simplify_multiclass_coverage(
        [(1, left), (2, right)], 4.0,
        max_categorical_difference_fraction=0.01,
        search_steps=10,
        return_diagnostics=True,
    )
    assert diagnostics["selected_fidelity"]["categorical_difference_fraction"] <= 0.01
    assert diagnostics["selected_simplify_px"] < 4.0
    assert diagnostics["selected_segment_count"] <= diagnostics["source_segment_count"]
    assert dict(rows)[1].intersection(dict(rows)[2]).area == 0


def test_cli_selective_shared_boundary_smoothing_is_auditable_and_exclusive(tmp_path):
    paths, labels, uncertain, provenance = fixture(tmp_path)
    labels[:] = 0
    seam = [8, 8, 7, 9, 7, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    for row, split in enumerate(seam, start=1):
        labels[row, 1:split] = 1
        labels[row, split:18] = 2
    uncertain[:] = 0
    provenance[:] = (labels > 0).astype(np.uint8)
    tifffile.imwrite(paths["labels"], labels)
    tifffile.imwrite(paths["uncertainty"], uncertain)
    tifffile.imwrite(paths["provenance"], provenance)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["pixel_counts"] = {
        module.REFINEMENT_CODES[int(code)]: int(count)
        for code, count in zip(*np.unique(provenance, return_counts=True))
    }
    paths["metadata"].write_text(json.dumps(metadata))
    result, out = run_cli(tmp_path, paths, [
        "--dissolve-by-value", "--shared-boundary-simplify", "--simplify", "0.01",
        "--simplify-max-categorical-difference-fraction", "0.01",
        "--shared-boundary-smooth", "--shared-boundary-smoothing-coefficient", "0.05",
        "--shared-boundary-minimum-turn-degrees", "45",
        "--shared-boundary-minimum-adjacent-length", "1",
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(Path(str(out) + ".provenance.json").read_text())
    smoothing = summary["vectorization"]["shared_boundary_smoothing"]
    assert smoothing["applied"] is True
    assert smoothing["outer_tissue_boundary_unchanged"] is True
    assert smoothing["multiclass_overlap_area_px2"] == 0
    assert summary["vectorization"]["multiclass_geometry_mutually_exclusive"] is True


def test_disconnected_same_label_features_explicitly_share_label_level_summary(tmp_path):
    paths, labels, uncertain, provenance = fixture(tmp_path)
    labels[15, 10:12] = 1
    provenance[15, 10:12] = 1
    tifffile.imwrite(paths["labels"], labels)
    tifffile.imwrite(paths["provenance"], provenance)
    metadata = json.loads(paths["metadata"].read_text())
    metadata["pixel_counts"] = {module.REFINEMENT_CODES[int(c)]: int(n) for c, n in zip(*np.unique(provenance, return_counts=True))}
    paths["metadata"].write_text(json.dumps(metadata))
    result, out = run_cli(tmp_path, paths)
    assert result.returncode == 0, result.stderr
    records = [f["properties"]["raster_provenance"] for f in json.loads(out.read_text())["features"] if f["properties"]["value"] == 1]
    assert len(records) == 2 and records[0] == records[1]
    assert records[0]["source_pixel_count"] == int((labels == 1).sum())


def test_binary_summary_preserves_native_label_inventory(tmp_path):
    paths, labels, _, _ = fixture(tmp_path)
    result, out = run_cli(tmp_path, paths, ["--binary", "--dissolve"])
    assert result.returncode == 0, result.stderr
    geojson = json.loads(out.read_text())
    evidence = geojson["features"][0]["properties"]["raster_provenance"]
    assert evidence["summary_key"] == "foreground_summary"
    assert evidence["source_pixel_count"] == int((labels > 0).sum())


def test_legacy_invocation_explicitly_reports_unavailable(tmp_path):
    paths, _, _, _ = fixture(tmp_path)
    result, out = run_cli(tmp_path, paths, sidecars=False)
    assert result.returncode == 0, result.stderr
    for feature in json.loads(out.read_text())["features"]:
        evidence = feature["properties"]["raster_provenance"]
        assert evidence["status"] == "unavailable"
        assert evidence["nonzero_uncertainty_fraction"] is None
        assert evidence["uncertainty_fractions"] == {}


def test_source_uncertainty_alone_does_not_certify_growth(tmp_path):
    paths, _, _, _ = fixture(tmp_path)
    result = module.raster_provenance_summary(paths["labels"], paths["uncertainty"])
    assert result["status"] == "source_uncertainty_only_growth_provenance_unavailable"
    assert result["label_summaries"]["1"]["nonzero_uncertainty_fraction"] > 0
    assert result["label_summaries"]["2"]["accepted_biological_confidence"] is None
    assert result["original_clustering_uncertainty_available"] is None


@pytest.mark.parametrize("mutation,match", [
    ("shape", "alignment"), ("dtype", "integer"), ("unknown_code", "legend"),
    ("counts", "pixel counts"), ("legend", "legend"), ("frame", "coordinate frame"),
    ("metadata_shape", "shape conflicts"), ("uncertainty255", "0..254"),
    ("presence", "label-presence"), ("lost_uncertainty", "retain its source code"),
])
def test_refinement_sidecar_conflicts_fail(tmp_path, mutation, match):
    paths, labels, uncertain, provenance = fixture(tmp_path)
    metadata = json.loads(paths["metadata"].read_text())
    if mutation == "shape":
        tifffile.imwrite(paths["uncertainty"], uncertain[:-1])
    elif mutation == "dtype":
        tifffile.imwrite(paths["uncertainty"], uncertain.astype(float))
    elif mutation == "unknown_code":
        provenance[1, 1] = 99
        tifffile.imwrite(paths["provenance"], provenance)
    elif mutation == "counts":
        metadata["pixel_counts"]["original_label_unchanged"] += 1
    elif mutation == "legend":
        metadata["provenance_codes"]["3"] = "confident"
    elif mutation == "frame":
        metadata["coordinate_frame"] = "original_image"
    elif mutation == "metadata_shape":
        metadata["shape_yx"] = [19, 17]
    elif mutation == "uncertainty255":
        uncertain[1, 1] = 255
        tifffile.imwrite(paths["uncertainty"], uncertain)
    elif mutation == "presence":
        labels[1, 1] = 0
        tifffile.imwrite(paths["labels"], labels)
    elif mutation == "lost_uncertainty":
        uncertain[1, 2] = 0
        tifffile.imwrite(paths["uncertainty"], uncertain)
    paths["metadata"].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        summarize(paths, tile_size=3)


def test_missing_refinement_sidecars_and_label_limit_fail(tmp_path):
    paths, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="together"):
        module.raster_provenance_summary(paths["labels"], paths["uncertainty"], paths["provenance"])
    with pytest.raises(ValueError, match="max-labels"):
        summarize(paths, max_labels=2)


def test_changed_source_between_scan_and_vector_export_fails(tmp_path):
    paths, labels, _, _ = fixture(tmp_path)
    summary = summarize(paths)
    labels[2, 2] = 2
    tifffile.imwrite(paths["labels"], labels)
    args = SimpleNamespace(out=str(tmp_path / "result.geojson"), provenance_summary_out=None,
        mask=paths["labels"], uncertainty_mask=paths["uncertainty"], provenance_mask=paths["provenance"], provenance_metadata=paths["metadata"])
    with pytest.raises(ValueError, match="changed during vectorization"):
        module.write_geojson_with_provenance(args, [], summary, 0, 1., 1., "rasterio")
    assert not Path(args.out).exists()


def test_vector_output_cannot_overwrite_source(tmp_path):
    paths, _, _, _ = fixture(tmp_path)
    before = module.file_sha256(paths["labels"])
    result, _ = run_cli(tmp_path, paths, ["--out", str(paths["labels"])])
    assert result.returncode != 0 and "must not overwrite source" in result.stderr
    assert module.file_sha256(paths["labels"]) == before


def test_conflicting_sample_key_cannot_be_exported(tmp_path):
    paths, _, _, _ = fixture(tmp_path)
    result, out = run_cli(tmp_path, paths, ["--sample-key", "wrong::standard"])
    assert result.returncode != 0 and "matching sample-key" in result.stderr
    assert not out.exists()


def bind_fixture(paths):
    metadata = json.loads(paths["metadata"].read_text())
    metadata.update(output_binding_status="complete", output_artifacts={
        name: {"filename": paths[name].name, "size_bytes": paths[name].stat().st_size, "sha256": module.file_sha256(paths[name])}
        for name in ("labels", "uncertainty", "provenance")})
    paths["metadata"].write_text(json.dumps(metadata))


@pytest.mark.parametrize("changed", ["labels", "uncertainty", "provenance"])
def test_bound_same_histogram_spatial_changes_are_rejected(tmp_path, changed):
    paths, _, _, _ = fixture(tmp_path)
    bind_fixture(paths)
    before = tifffile.imread(paths[changed])
    changed_map = np.roll(before, shift=1, axis=1)
    np.testing.assert_array_equal(np.sort(before.ravel()), np.sort(changed_map.ravel()))
    tifffile.imwrite(paths[changed], changed_map)
    with pytest.raises(ValueError, match=f"SHA256 mismatch for {changed}"):
        summarize(paths)


@pytest.mark.parametrize("problem", ["pending", "missing_artifact", "empty_artifacts", "size"])
def test_incomplete_or_malformed_producer_binding_fails(tmp_path, problem):
    paths, _, _, _ = fixture(tmp_path)
    bind_fixture(paths)
    metadata = json.loads(paths["metadata"].read_text())
    if problem == "pending":
        metadata["output_binding_status"] = "pending_final_label_export"
    elif problem == "missing_artifact":
        del metadata["output_artifacts"]["provenance"]
    elif problem == "empty_artifacts":
        metadata["output_artifacts"] = None
    else:
        metadata["output_artifacts"]["labels"]["size_bytes"] += 1
    paths["metadata"].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="binding|size mismatch"):
        summarize(paths)


def test_real_producer_finalization_binds_compressed_outputs_to_vectors(tmp_path):
    pytest.importorskip("skimage")
    import refine_grown_tissue_medsam as refine
    paths, labels, _, _ = fixture(tmp_path)
    grown = labels.copy()
    incoming = np.zeros(labels.shape, np.uint8)
    incoming[2:4, 2:4] = 7
    tissue = (labels > 0).astype(np.uint8)
    grown_path, incoming_path, tissue_path = (tmp_path / name for name in ("grown.tif", "incoming.tif", "support.tif"))
    for path, array in ((grown_path, grown), (incoming_path, incoming), (tissue_path, tissue)):
        tifffile.imwrite(path, array)
    output = tmp_path / "final.ome.tif"
    args = SimpleNamespace(out=str(output), provenance_out="", uncertainty_out="", clustering_uncertainty=str(incoming_path),
        grown_mask=str(grown_path), tissue_mask=str(tissue_path), source_mpp_x=.5, source_mpp_y=.5, stream_block_rows=3, overwrite=False)
    original = refine.TiffWindowReader(str(grown_path), "grown")
    support = refine.ScaledBinaryMaskReader(str(tissue_path), labels.shape)
    uncertainty = refine.TiffWindowReader(str(incoming_path), "uncertainty")
    try:
        pending = refine.write_refinement_provenance_outputs(args, labels, original, support, uncertainty)
    finally:
        original.close(); support.close(); uncertainty.close()
    # Simulate the final lossless pyramidization with real compressed TIFF I/O.
    with tifffile.TiffWriter(output) as writer:
        writer.write(labels, subifds=1, compression="deflate", tile=(16, 16), metadata={"axes": "YX"})
        writer.write(labels[::2, ::2], subfiletype=1, compression="deflate", tile=(16, 16))
    finalized = refine.finalize_refinement_provenance_outputs(args, pending)
    actual = {"labels": output, "uncertainty": Path(pending["uncertainty_path"]), "provenance": Path(pending["provenance_path"]),
              "metadata": Path(str(output) + ".provenance.json")}
    assert finalized["output_artifacts"]["labels"]["sha256"] == module.file_sha256(output)
    result, geojson_path = run_cli(tmp_path, actual, ["--page", "1"])
    assert result.returncode == 0, result.stderr
    summary = json.loads(Path(str(geojson_path) + ".provenance.json").read_text())
    assert summary["producer_output_binding"] == "exact_output_sha256_verified"
    assert summary["source_hashes_verified_after_vectorization"] is True
    collection = json.loads(geojson_path.read_text())
    assert collection["cellphenotyper_provenance"]["producer_output_binding"] == "exact_output_sha256_verified"
    for feature in collection["features"]:
        assert feature["properties"]["raster_provenance"]["producer_output_binding"] == "exact_output_sha256_verified"
    assert summary["label_summaries"]["1"]["uncertainty_counts"]["7"] == 4


def test_reduced_pyramid_geometry_still_reports_exact_native_counts(tmp_path):
    paths, labels, _, _ = fixture(tmp_path)
    with tifffile.TiffWriter(paths["labels"]) as writer:
        writer.write(labels, subifds=1, metadata={"axes": "YX"})
        writer.write(labels[::2, ::2], subfiletype=1)
    result, out = run_cli(tmp_path, paths, ["--page", "1"])
    assert result.returncode == 0, result.stderr
    summary = json.loads(Path(str(out) + ".provenance.json").read_text())
    assert summary["vectorization"]["page"] == 1
    assert summary["label_summaries"]["1"]["source_pixel_count"] == int((labels == 1).sum())


def nextflow_fixture(tmp_path):
    for folder in ("modules", "lib"):
        (tmp_path / folder).mkdir()
    shutil.copyfile(ROOT / "modules/mask_to_geojson.nf", tmp_path / "modules/mask_to_geojson.nf")
    shutil.copyfile(ROOT / "lib/PipelineHelpers.groovy", tmp_path / "lib/PipelineHelpers.groovy")
    shutil.copyfile(ROOT / "lib/ProcessCode.groovy", tmp_path / "lib/ProcessCode.groovy")
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    paths, _, _, _ = fixture(tmp_path)
    params = {"outdir_base": str(tmp_path / "output"), "publish_dir_mode": "copy", "_executor_max_cpus": 1,
        "_executor_max_memory_gb": 2, "cluster_geojson_cpus": 1, "cluster_geojson_memory_gb": 1,
        "cluster_geojson_time": "5m", "cluster_geojson_script": "bin/mask_to_geojson.py",
        "cluster_geojson_polygon_backend": "rasterio",
        "cluster_geojson_connectivity": 4,
        "cluster_geojson_dissolve_by_value": True, "cluster_geojson_fill_holes": False,
        "cluster_geojson_preserve_topology": True, "cluster_geojson_group_map": None,
        "cluster_geojson_page": 0, "cluster_geojson_max_page_side": 8192, "cluster_geojson_min_area": 0,
        "cluster_geojson_smooth_buffer": 0, "cluster_geojson_smooth_passes": 1, "cluster_geojson_simplify": 0,
        "cluster_geojson_shared_boundary_simplify": False,
        "cluster_geojson_simplify_max_categorical_difference_fraction": 0.001,
        "cluster_geojson_simplify_search_steps": 10,
        "cluster_geojson_shared_boundary_smooth": False,
        "cluster_geojson_shared_boundary_smoothing_coefficient": 0.05,
        "cluster_geojson_shared_boundary_smoothing_passes": 1,
        "cluster_geojson_shared_boundary_minimum_turn_degrees": 45.0,
        "cluster_geojson_shared_boundary_minimum_adjacent_length": 16.0,
        "cluster_geojson_group_prefix": "domain_"}
    (tmp_path / "params.json").write_text(json.dumps(params))
    entries = ", ".join(f"file('{paths[name]}')" for name in ("labels", "uncertainty", "provenance", "metadata"))
    (tmp_path / "workflow.nf").write_text(f"""nextflow.enable.dsl=2
include {{ MASK_TO_GEOJSON }} from './modules/mask_to_geojson'
workflow {{
  input = Channel.of(tuple('s1::tile', 's1', 'tile', {entries}, [uncertainty:true,refinement:true]),
                     tuple('s2::tile', 's2', 'tile', {entries}, [uncertainty:true,refinement:true]))
  MASK_TO_GEOJSON(input)
  MASK_TO_GEOJSON.out.provenance_summary.view {{ key, id, variant, summary -> "SUMMARY ${{key}} ${{id}} ${{variant}} ${{summary.name}}" }}
}}
""")


@pytest.mark.parametrize("stub", [False, True])
def test_real_nextflow_module_keeps_canonical_lineage_and_summaries(tmp_path, stub):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    nextflow_fixture(tmp_path)
    command = [nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
               "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-work-dir", str(tmp_path / "work")]
    if stub:
        command += ["-stub-run"]
    result = subprocess.run(command, cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true", "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"}, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    for sample in ("s1", "s2"):
        out = tmp_path / "output/15_cluster_geojson" / sample / f"{sample}_tile_grown_mask_smooth_class.geojson"
        assert out.exists()
        summary = json.loads(Path(str(out) + ".provenance.json").read_text())
        assert f"SUMMARY {sample}::tile {sample} tile" in result.stdout
        if not stub:
            assert summary["lineage"]["sample_key"] == f"{sample}::tile"
            assert summary["status"] == "refinement_provenance_available"


def test_primary_and_auxiliary_sidecars_join_on_entire_identity():
    for filename in ("post_grow_spatial_outputs.nf", "run_auxiliary_cell_spatial.nf"):
        text = (ROOT / "subworkflows" / filename).read_text()
        assert text.count("by: [0, 1, 2], failOnMismatch: true, failOnDuplicate: true") >= 3
        assert "vector_provenance_summary =" in text
        assert "uncertainty != null, refinement: false" in text


@pytest.mark.parametrize("uncertainty_case", ["partial", "foreign", "duplicate"])
def test_actual_post_grow_workflow_handles_optional_uncertainty_strictly(tmp_path, uncertainty_case):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    nextflow_fixture(tmp_path)
    (tmp_path / "subworkflows").mkdir()
    shutil.copyfile(ROOT / "subworkflows/post_grow_spatial_outputs.nf", tmp_path / "subworkflows/post_grow_spatial_outputs.nf")
    shutil.copyfile(ROOT / "modules/refine_grown_tissue_medsam.nf", tmp_path / "modules/refine_grown_tissue_medsam.nf")
    shutil.copytree(ROOT / "lib", tmp_path / "lib", dirs_exist_ok=True)
    params = json.loads((tmp_path / "params.json").read_text())
    params.update(grown_tissue_refine_method="none", cluster_primary_variant="tile", cluster_secondary_variant=None)
    (tmp_path / "params.json").write_text(json.dumps(params))
    uncertainty = "tuple('s1::tile', 's1', 'tile', uncertainty)"
    if uncertainty_case == "foreign":
        uncertainty += ", tuple('s3::tile', 's3', 'tile', uncertainty)"
    elif uncertainty_case == "duplicate":
        uncertainty += ", tuple('s1::tile', 's1', 'tile', uncertainty)"
    (tmp_path / "workflow.nf").write_text(f"""nextflow.enable.dsl=2
include {{ POST_GROW_SPATIAL_OUTPUTS }} from './subworkflows/post_grow_spatial_outputs'
workflow {{
  labels = file('{tmp_path}/labels.tif')
  uncertainty = file('{tmp_path}/uncertainty.tif')
  masks = Channel.of(tuple('s1::tile','s1','tile',labels), tuple('s2::tile','s2','tile',labels))
  uncertain = Channel.of({uncertainty})
  POST_GROW_SPATIAL_OUTPUTS(Channel.empty(), masks, Channel.empty(), Channel.empty(), uncertain,
                           Channel.empty(), Channel.empty(), Channel.empty(), false, true,
                           [schema_version:1, compute_device:'cpu', profile:'conservative',
                            cpu_budget:1, memory_budget_gb:2, stages:[medsam_refine:[cpus:1,memory_gb:2]], settings:[:]])
  POST_GROW_SPATIAL_OUTPUTS.out.vector_provenance_summary.view {{ key,id,variant,summary -> "SUMMARY ${{key}}" }}
}}
""")
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-work-dir", str(tmp_path / "work")],
        cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true", "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}"},
        text=True, capture_output=True, timeout=60)
    if uncertainty_case != "partial":
        assert result.returncode != 0
        assert ("Unmatched uncertainty sidecar" if uncertainty_case == "foreign" else "duplicate") in (result.stdout + result.stderr)
        return
    assert result.returncode == 0, result.stdout + result.stderr
    for sample, expected in (("s1", "source_uncertainty_only_growth_provenance_unavailable"), ("s2", "unavailable")):
        summary_path = tmp_path / "output/15_cluster_geojson" / sample / f"{sample}_tile_grown_mask_smooth_class.geojson.provenance.json"
        assert json.loads(summary_path.read_text())["status"] == expected
