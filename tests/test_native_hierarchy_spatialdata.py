"""Actual native hierarchy -> canonical linkage -> SpatialData, without encoders.

The retained 63-observation native products are real CPU KODAMA outputs from
synthetic numeric fields. Cell masks/feature canaries here are synthetic, too;
these tests establish data lineage and geometry, never biological accuracy.
"""
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

sd = pytest.importorskip("spatialdata")
from spatialdata.transformations import get_transformation

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
ARCHIVE = ROOT / "audits/spatial_atlas_20260904/hierarchy_kodama_20260905/relative_output_fixed/synthetic_artifacts/cli"
sys.path.insert(0, str(BIN))
from cell_profile_io import sha256_file
from link_cell_tissue_hierarchy import load_verified_hierarchy
from export_spatialdata import HIERARCHY_LABELS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inventory(root):
    return {str(path): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def verify_hierarchy(directory):
    summary = json.loads((directory / "hierarchy_summary.json").read_text())
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and path != directory / "hierarchy_summary.json"}
    assert actual == set(summary["outputs"])
    assert all(sha256_file(directory / name) == sha for name, sha in summary["outputs"].items())
    assert summary["discovery"]["method"] == "within_parent_native_kodama_graph"
    assert summary["grid_observations"] == 63
    assert len(summary["realized_identity"]["native_graph_sha256"]) == 6
    return summary


def command(root, label, arguments, *, timeout=90):
    cache = root / (label + "_fresh_pycache")
    cache.mkdir()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(cache),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    env.pop("CELLPHENOTYPER_HIERARCHY_RSCRIPT", None)
    result = subprocess.run([str(arg) for arg in arguments], capture_output=True, text=True, env=env, timeout=timeout)
    (root / (label + ".stdout.txt")).write_text(result.stdout)
    (root / (label + ".stderr.txt")).write_text(result.stderr)
    write_json(root / (label + ".command.json"), [str(arg) for arg in arguments])
    assert not list(cache.iterdir())
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def canonical_sources(root, source, hierarchy, *, probe_status=None):
    """Nine literal-ID cell instances include gaps and cross-parent boundaries."""
    root.mkdir()
    labels = np.zeros((32, 32), np.uint32)
    rectangles = {"009": (1, 3, 1, 7), "2": (1, 3, 9, 11), "11": (29, 31, 29, 31),
                  "7": (1, 3, 29, 31), "101": (14, 18, 14, 18), "5": (6, 9, 5, 8),
                  "42": (23, 25, 21, 24), "17": (0, 1, 0, 2), "003": (4, 7, 28, 31)}
    for cid, (y0, y1, x0, x1) in rectangles.items():
        assert not labels[y0:y1, x0:x1].any()
        labels[y0:y1, x0:x1] = int(cid)
    order = ["101", "009", "11", "003", "2", "7", "17", "42", "5"]
    if probe_status is not None:
        status = tifffile.imread(hierarchy / "hierarchy_status.ome.tif")
        candidates = np.argwhere((status == probe_status) & (labels == 0))
        assert len(candidates), "Actual native producer must emit the requested unresolved status"
        y, x = map(int, candidates[0])
        labels[y, x] = 404
        order.append("404")
    objects = []
    for cid in order:
        yy, xx = np.where(labels == int(cid))
        objects.append({"label": cid, "x": (xx + .5).mean(), "y": (yy + .5).mean(),
                        "xmin": int(xx.min()), "ymin": int(yy.min()),
                        "xmax": int(xx.max()) + 1, "ymax": int(yy.max()) + 1})
    pd.DataFrame(objects).to_csv(root / "objects.csv", index=False)
    tifffile.imwrite(root / "labels.tif", labels)
    # Feature canaries, not a newly run encoder or independently measured assay.
    marker_dir = root / "synthetic_markers"
    marker_dir.mkdir()
    pd.DataFrame({"label_id": order[:-1], "DAPI__mean": np.arange(len(order) - 1) / 16,
                  "Cy5__mean": .25}).iloc[::-1].to_csv(marker_dir / "fixture_nuclei_gigatime_quantification.csv", index=False)
    for name, dimension in (("context", 4), ("local", 3)):
        values = np.arange(len(order) * dimension, dtype=np.float32).reshape(len(order), dimension) / 8
        frame = pd.DataFrame(values, columns=[f"feat_{i + 1}" for i in range(dimension)])
        frame.insert(0, "cell_id", order)
        frame.insert(1, "observation_type", "cell")
        if name == "local":
            frame = frame.iloc[:-1]  # absent cell stays in canonical table with NaNs.
        frame.iloc[::-1].to_csv(root / f"synthetic_{name}.csv", index=False)
    # Derive polygon geometry exactly from the immutable parent pixels. This is
    # not an expert annotation and never participates in discovery/fitting.
    from shapely.geometry import box, mapping
    from shapely.ops import unary_union
    parent = tifffile.imread(hierarchy / "parent_domains.ome.tif")
    features = []
    for value in np.unique(parent[parent > 0]):
        yy, xx = np.where(parent == value)
        polygon = unary_union([box(int(x), int(y), int(x) + 1, int(y) + 1) for y, x in zip(yy, xx)])
        features.append({"type": "Feature", "properties": {"value": int(value), "source": "immutable_parent_raster_pixel_union"},
                         "geometry": mapping(polygon)})
    write_json(root / "parent_domains.geojson", {"type": "FeatureCollection", "features": features})
    sample = json.loads((hierarchy / "hierarchy_summary.json").read_text())["sample_id"]
    profiles = root / "profiles"
    arguments = [sys.executable, "-B", BIN / "build_cell_profiles.py", "--objects", root / "objects.csv",
        "--sample-id", sample, "--image", source / "image.tif", "--labels", root / "labels.tif",
        "--shift", source / "shift.json", "--resolution-json", source / "resolution.json",
        "--domain-mask", hierarchy / "parent_domains.ome.tif", "--domain-uncertainty", hierarchy / "parent_uncertainty.ome.tif",
        "--marker-quant-dir", marker_dir, "--uni2-context", root / "synthetic_context.csv",
        "--uni2-local", root / "synthetic_local.csv", "--outdir", profiles]
    command(root, "build", arguments)
    return {"root": root, "source": source, "hierarchy": hierarchy, "profiles": profiles,
            "labels": root / "labels.tif", "labels_array": labels, "cell_ids": order, "sample_id": sample}


def link_and_export(fixture):
    root, source = fixture["root"], fixture["source"]
    before = {**inventory(fixture["hierarchy"]), **inventory(fixture["profiles"]), **inventory(source)}
    linked, output = root / "linked_profiles", root / "linked_native.zarr"
    command(root, "link", [sys.executable, "-B", BIN / "link_cell_tissue_hierarchy.py",
        "--profile-dir", fixture["profiles"], "--labels", fixture["labels"],
        "--hierarchy-dir", fixture["hierarchy"], "--tile-size", "7", "--outdir", linked])
    linked_before = inventory(linked)
    command(root, "export", [sys.executable, "-B", BIN / "export_spatialdata.py",
        "--profile-dir", linked, "--image", source / "image.tif", "--labels", fixture["labels"],
        "--shift", source / "shift.json", "--resolution-json", source / "resolution.json",
        "--hierarchy-dir", fixture["hierarchy"], "--tissue-geojson", root / "parent_domains.geojson",
        "--tissue-coordinates", "crop_pixels", "--sample-id", fixture["sample_id"],
        "--tile-size", "16", "--workers", "1", "--outdir", output])
    assert {path: sha256_file(path) for path in before} == before
    assert inventory(linked) == linked_before
    return {**fixture, "linked": linked, "output": output, "source_hashes": before}


@pytest.fixture(scope="module", params=["native_a0", "native_b0"])
def native_linked(request, tmp_path_factory):
    retained = ARCHIVE / request.param
    assert retained.is_dir(), "Corrected actual-native engineering producer artifacts are required"
    verify_hierarchy(retained / "hierarchy")
    root = tmp_path_factory.mktemp(request.param + "_consumer")
    fixture = canonical_sources(root / "consumer", retained / "sources", retained / "hierarchy")
    return link_and_export(fixture)


def check_exact_links(fixture):
    original = pd.read_parquet(fixture["profiles"] / "cell_profiles.parquet")
    linked = pd.read_parquet(fixture["linked"] / "cell_profiles.parquet")
    pd.testing.assert_frame_equal(linked[list(original)], original)
    assert linked.cell_id.tolist() == fixture["cell_ids"]
    assert linked.cell_uid.is_unique
    manifest = json.loads((fixture["profiles"] / "cell_profiles_manifest.json").read_text())
    for record in manifest["feature_blocks"].values():
        assert sha256_file(fixture["profiles"] / record["path"]) == sha256_file(fixture["linked"] / record["path"])
    assert sha256_file(fixture["profiles"] / "feature_rows.csv") == sha256_file(fixture["linked"] / "feature_rows.csv")
    maps = {key: tifffile.imread(fixture["hierarchy"] / name) for key, name in {
        "parent": "parent_domains.ome.tif", "subdomain": "subdomain_mask.ome.tif", "region": "region_mask.ome.tif",
        "status": "hierarchy_status.ome.tif", "uncertainty": "parent_uncertainty.ome.tif"}.items()}
    overlaps = pd.read_parquet(fixture["linked"] / "cell_hierarchy_overlaps.parquet")
    assert set(overlaps.cell_uid) == set(original.cell_uid)
    assert overlaps.compartment.eq("nucleus").all()
    for row in original.itertuples():
        mask = fixture["labels_array"] == int(row.cell_id)
        expected = Counter(zip(*(maps[key][mask].tolist() for key in ("parent", "subdomain", "region", "status", "uncertainty"))))
        actual_rows = overlaps[overlaps.cell_uid == row.cell_uid]
        actual = {tuple(getattr(item, column) for column in ("parent_domain_id", "subdomain_id", "region_id", "hierarchy_status_code", "parent_uncertainty_code")): item.overlap_pixels
                  for item in actual_rows.itertuples()}
        assert actual == expected
        assert actual_rows.overlap_pixels.sum() == mask.sum()
        np.testing.assert_array_equal(actual_rows.area_um2, actual_rows.overlap_pixels * .25)
        np.testing.assert_array_equal(actual_rows.overlap_fraction, actual_rows.overlap_pixels / int(mask.sum()))
        assert actual_rows.compartment_pixels.eq(mask.sum()).all()
    assert overlaps.loc[overlaps.region_id == 0, "region_uid"].eq("").all()
    assert linked.set_index("cell_id").loc["101", "hierarchy_nucleus_crosses_parent_boundary"]
    assert linked.set_index("cell_id").loc["7", "hierarchy_nucleus_background_fraction"] == 1
    assert linked.set_index("cell_id").loc["17", "hierarchy_nucleus_parent_uncertain_fraction"] == 1
    assert linked.set_index("cell_id").loc["2", "hierarchy_nucleus_unresolved_fraction"] == 1
    assert linked.set_index("cell_id").loc["11", "hierarchy_nucleus_unresolved_fraction"] == 1
    return original, linked, overlaps, maps


def check_spatialdata(fixture):
    original, linked, overlaps, maps = check_exact_links(fixture)
    data = sd.SpatialData.read(fixture["output"])
    summary = verify_hierarchy(fixture["hierarchy"])
    cells = data.tables["cells"]
    assert cells.obs_names.tolist() == linked.cell_uid.tolist()
    assert cells.obs.cell_id.astype(str).tolist() == fixture["cell_ids"]
    assert cells.obs.instance_id.tolist() == [int(value) for value in fixture["cell_ids"]]
    assert cells.obs.spatial_region.astype(str).eq("canonical_cells").all()
    fields = json.loads(cells.uns["cellphenotyper"]["field_name_mapping_json"])["obs"]
    for column in linked:
        a, b = cells.obs[fields[column]].to_numpy(), linked[column].to_numpy()
        if b.dtype.kind in "fiu":
            np.testing.assert_array_equal(a, b)
        else:
            assert a.tolist() == b.tolist()
    manifest = json.loads((fixture["profiles"] / "cell_profiles_manifest.json").read_text())
    for name, record in manifest["feature_blocks"].items():
        source = np.load(fixture["profiles"] / record["path"])
        np.testing.assert_array_equal(cells.obsm[name], source)
        assert cells.obsm[name].dtype == source.dtype
    np.testing.assert_array_equal(data.labels["canonical_cells"].data.compute(), fixture["labels_array"])
    for key, name in HIERARCHY_LABELS.items():
        expected = maps["uncertainty" if key == "parent_uncertainty" else key]
        np.testing.assert_array_equal(data.labels[name].data.compute(), expected)
        affine = get_transformation(data.labels[name], to_coordinate_system="original_um").to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        np.testing.assert_array_equal(affine, [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    region_table = data.tables["hierarchy_region_profiles"]
    region_source = fixture["hierarchy"] / "region_profiles"
    regions = pd.read_csv(region_source / "region_profiles.csv", dtype={"region_id": str}, float_precision="round_trip")
    assert region_table.obs_names.tolist() == regions.region_uid.tolist()
    assert region_table.obs.instance_id.tolist() == regions.region_id.astype(int).tolist()
    assert region_table.obs.spatial_region.astype(str).eq("hierarchy_regions").all()
    assert not {"cell_id", "cell_uid"} & set(region_table.obs)
    for name in ("local", "context"):
        raw = np.load(region_source / (name + ".npy"))
        np.testing.assert_array_equal(region_table.obsm[name], raw)
        assert region_table.obsm[name].dtype == raw.dtype
    np.testing.assert_array_equal(region_table.obsm["spatial"], regions[["x_um", "y_um"]])
    relation = data.tables["cell_hierarchy_overlaps"]
    assert "spatialdata_attrs" not in relation.uns  # many-to-many relation, not duplicated cells
    for column in overlaps:
        np.testing.assert_array_equal(relation.obs[column], overlaps[column])
    foreign = json.loads(relation.uns["cellphenotyper"]["foreign_keys_json"])
    assert foreign["region_uid"] == {"table": "hierarchy_region_profiles", "key": "region_uid"}
    assert foreign["cell_uid"] == {"table": "cells", "key": "cell_uid"}
    provenance = data.attrs["cellphenotyper"]
    record = provenance["tissue_hierarchy"]
    assert record["hierarchy_id"] == summary["hierarchy_id"]
    assert record["hierarchy_summary_sha256"] == sha256_file(fixture["hierarchy"] / "hierarchy_summary.json")
    assert record["status_codes"] == summary["status_codes"]
    assert record["source_output_hashes"] == summary["outputs"]
    assert provenance["reference_mappings"] == {}
    assert not any(column.startswith("reference_") for column in cells.obs)
    assert not any(column.startswith("reference_") for column in region_table.obs)
    assert provenance["measured_modalities"] == []
    assert "predicted" in provenance["measured_modalities_status"]
    assert {path: sha256_file(path) for path in fixture["source_hashes"]} == fixture["source_hashes"]
    report = {"status": "pass", "sample_id": fixture["sample_id"], "canonical_cells": len(cells),
        "native_grid_observations": 63, "regions": len(regions), "overlap_rows": len(overlaps),
        "observed_overlap_statuses": sorted(map(int, overlaps.hierarchy_status_code.unique())),
        "hierarchy_id": summary["hierarchy_id"], "hierarchy_summary_sha256": record["hierarchy_summary_sha256"],
        "native_graph_sha256": summary["realized_identity"]["native_graph_sha256"],
        "versions": provenance["versions"], "linked_manifest_sha256": sha256_file(fixture["linked"] / "cell_profiles_manifest.json"),
        "source_hashes": fixture["source_hashes"], "interpretation": "Engineering linkage/roundtrip only; synthetic cells/features, actual native graph discovery; no reference fitting or biological accuracy claim"}
    write_json(fixture["root"] / "consumer_acceptance.json", report)
    return data, overlaps


def test_actual_native_hierarchy_links_all_canonical_cells_and_roundtrips_spatialdata(native_linked):
    check_spatialdata(native_linked)


@pytest.fixture(scope="module")
def ambiguous_native(tmp_path_factory):
    python = ROOT / ".venv-spatial/bin/python"
    library = Path("/Users/stefano/Documents/KODAMA-cpp 2/tmp/Rlib-kodama-latest")
    if not python.is_file() or not shutil.which("Rscript") or not (library / "KODAMA/DESCRIPTION").is_file():
        pytest.skip("Explicit local native fitting runtime unavailable; no automatic install")
    root = tmp_path_factory.mktemp("actual_ambiguous_native")
    source = ARCHIVE / "native_a0/sources"
    out = root / "hierarchy"
    script = """import pathlib, subprocess, sys
sys.path.insert(0, sys.argv[1])
from test_kodama_hierarchy_cli import cli_arguments
fixture={'root':pathlib.Path(sys.argv[2]), 'sample':'native_a'}
cmd=cli_arguments(fixture,pathlib.Path(sys.argv[3]))+['--fixed-k','3','--min-affinity-margin','0.9']
raise SystemExit(subprocess.call(cmd))
"""
    command(root, "native_fit", [python, "-B", "-c", script, ROOT / "tests", source, out], timeout=180)
    summary = verify_hierarchy(out)
    assert summary["discovery"]["fixed_k"] == 3
    assert summary["discovery"]["thresholds"]["affinity_margin"] == .9
    assert summary["status_pixel_counts"].get("ambiguous_graph_assignment", 0) > 0
    fixture = canonical_sources(root / "consumer", source, out, probe_status=12)
    return link_and_export(fixture)


def test_actual_native_affinity_ambiguity_stays_unresolved_through_link_and_export(ambiguous_native):
    data, overlaps = check_spatialdata(ambiguous_native)
    ambiguous = overlaps[overlaps.hierarchy_status_code == 12]
    assert len(ambiguous) > 0
    assert ambiguous.parent_domain_id.gt(0).all()
    assert ambiguous.subdomain_id.eq(0).all() and ambiguous.region_id.eq(0).all()
    assert ambiguous.region_uid.eq("").all()
    probe = data.tables["cells"].obs.set_index("cell_id").loc["404"]
    assert probe.hierarchy_nucleus_unresolved_fraction == 1
    assert probe.hierarchy_nucleus_accepted_fraction == 0
    assert data.attrs["cellphenotyper"]["tissue_hierarchy"]["status_codes"]["12"] == "ambiguous_graph_assignment"


@pytest.mark.parametrize("code, declared", [(11, False), (12, False), (13, True)])
def test_undeclared_or_unknown_status_is_rejected_on_explicit_synthetic_copy(native_linked, tmp_path, code, declared):
    # Negative contract only: this status edit was NOT produced by native fit.
    copied = tmp_path / "synthetic_status_negative"
    shutil.copytree(native_linked["hierarchy"], copied)
    path = copied / "hierarchy_status.ome.tif"
    status = tifffile.imread(path)
    status[1:3, 9:11] = code  # already unresolved/missing; region raster remains zero.
    tifffile.imwrite(path, status)
    summary_path = copied / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["test_only_derivative"] = "Injected status-code negative contract; not an actual graph-isolate observation"
    summary["outputs"][path.name] = sha256_file(path)
    if declared:
        summary["status_codes"][str(code)] = "synthetic_unknown_code_must_fail"
    else:
        summary["status_codes"].pop(str(code), None)
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="status code|categorical|status_codes"):
        load_verified_hierarchy(native_linked["profiles"], native_linked["labels"], copied, tile_size=7)


def test_changed_native_region_features_rejected_before_linking(native_linked, tmp_path):
    copied = tmp_path / "corrupt_native_hierarchy"
    shutil.copytree(native_linked["hierarchy"], copied)
    path = copied / "region_profiles/local.npy"
    values = np.load(path)
    values[0, 0] += np.float32(1)
    np.save(path, values)
    with pytest.raises(ValueError, match="SHA256"):
        load_verified_hierarchy(native_linked["profiles"], native_linked["labels"], copied, tile_size=7)
