"""Exact synthetic engineering checks; these are not biological validation."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from build_cell_profiles import build_profiles, parser
from cell_profile_io import sha256_file
from link_cell_tissue_hierarchy import link_profiles, load_verified_hierarchy, load_verified_links, _summaries


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def hierarchy_fixture(tmp_path, sparse_id=9000000001):
    """Self-contained bound real-format synthetic cell+hierarchy fixture."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    shape = (8, 10)
    image = tmp_path / "image.tif"
    tifffile.imwrite(image, np.arange(240, dtype=np.uint8).reshape(8, 10, 3), photometric="rgb")
    labels_array = np.zeros(shape, np.uint64)
    labels_array[:2, :2] = sparse_id
    labels_array[2:5, 4:6] = 2
    labels_array[2:4, 2:4] = 7
    labels = tmp_path / "labels.tif"
    tifffile.imwrite(labels, labels_array)
    shift, resolution = tmp_path / "shift.json", tmp_path / "resolution.json"
    write_json(shift, {"crop_size": {"width": 10, "height": 8}, "offset_crop_to_original": {"dx": 100, "dy": 200}})
    write_json(resolution, {"status": "pass", "mpp_x": .5, "mpp_y": .5})
    objects = tmp_path / "objects.csv"
    pd.DataFrame({"label": [2, sparse_id, 7], "x": [4.5, .5, 2.5], "y": [3, .5, 2.5],
        "xmin": [4, 0, 2], "ymin": [2, 0, 2], "xmax": [6, 2, 4], "ymax": [5, 2, 4]}).to_csv(objects, index=False)
    hierarchy = tmp_path / "hierarchy"
    hierarchy.mkdir()
    parent = np.where(np.indices(shape)[1] < 5, 1, 2).astype(np.uint16)
    parent[2:4, 2:4] = 0  # True empty gap, still counted in a crossing compartment.
    status = np.where(parent > 0, 1, 0).astype(np.uint8)
    status[-1] = 2  # Sparse unobserved tissue must remain present and unresolved.
    status[0, 0] = 10
    uncertainty = np.zeros(shape, np.uint8)
    uncertainty[0, 0] = 6
    subdomain = np.where(status == 1, parent + 10, 0).astype(np.uint32)
    region = np.where(status == 1, np.where(parent == 1, 5, 9), 0).astype(np.uint32)
    masks = {"parent_domains.ome.tif": parent, "subdomain_mask.ome.tif": subdomain,
        "region_mask.ome.tif": region, "hierarchy_status.ome.tif": status, "parent_uncertainty.ome.tif": uncertainty}
    for name, values in masks.items():
        tifffile.imwrite(hierarchy / name, values)
    profile = tmp_path / "profile"
    build_profiles(parser().parse_args(["--objects", str(objects), "--sample-id", "specimen A", "--shift", str(shift),
        "--resolution-json", str(resolution), "--image", str(image), "--labels", str(labels),
        "--domain-mask", str(hierarchy / "parent_domains.ome.tif"), "--domain-uncertainty", str(hierarchy / "parent_uncertainty.ome.tif"), "--outdir", str(profile)]))
    manifest_path = profile / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    feature = profile / "feature_blocks/test.npy"
    np.save(feature, np.array([[1, np.nan], [2, 3], [4, 5]], np.float32))
    manifest["feature_blocks"]["test"] = {"path": "feature_blocks/test.npy", "sha256": sha256_file(feature), "shape": [3, 2]}
    graph = profile / "graph.npz"
    sparse.save_npz(graph, sparse.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    manifest["spatial_graphs"] = {"25": {"path": "graph.npz", "sha256": sha256_file(graph)}}
    # Bound optional ring with an empty compartment for the last cell.
    compartment_dir = tmp_path / "compartments"
    compartment_dir.mkdir()
    ring = np.zeros(shape, np.uint64)
    ring[6, 1] = 2
    ring[6, 2] = sparse_id
    ring_path = compartment_dir / "labels_perinuclear_ring.tif"
    tifffile.imwrite(ring_path, ring)
    comp_summary = compartment_dir / "compartment_summary.json"
    write_json(comp_summary, {"inputs": {"labels": {"sha256": sha256_file(labels)}, "shift": {"sha256": sha256_file(shift)},
        "resolution": {"sha256": sha256_file(resolution)}}, "output_artifacts": {"perinuclear_ring": {"sha256": sha256_file(ring_path)}}})
    manifest["tables"]["compartment_qc"] = {"mask_lineage_verified": True, "summary_sha256": sha256_file(comp_summary)}
    write_json(manifest_path, manifest)
    region_root = hierarchy / "region_profiles"
    region_root.mkdir()
    rows = []
    for rid in [5, 9]:
        yy, xx = np.nonzero(region == rid)
        p = int(parent[yy[0], xx[0]])
        rows.append({"region_uid": f"specimen%20A::testhierarchy::region_{rid}", "region_id": str(rid), "sample_id": "specimen A",
            "parent_domain_id": p, "subdomain_id": p + 10, "area_um2": len(xx) * .25,
            "x_um": (xx + .5).mean() * .5 + 50, "y_um": (yy + .5).mean() * .5 + 100})
    regions = pd.DataFrame(rows)
    regions.to_csv(region_root / "region_profiles.csv", index=False)
    regions[["region_uid", "region_id", "sample_id"]].to_csv(region_root / "feature_rows.csv", index=False)
    region_manifest = {"observation_unit": "tissue_region", "hierarchy_id": "testhierarchy", "coordinate_space": "original_slide_micrometres",
        "region_count": 2, "files": {name: sha256_file(region_root / name) for name in ("region_profiles.csv", "feature_rows.csv")}, "feature_blocks": {}}
    write_json(region_root / "region_profiles_manifest.json", region_manifest)
    summary = {"hierarchy_id": "testhierarchy", "sample_id": "specimen A", "parent_labels_immutable": True, "regions": 2,
        "status_codes": {"0": "background", "1": "accepted_subdomain", "2": "parent_only_no_observation", "10": "parent_assignment_uncertain"},
        "geometry": {"shape_yx": [8, 10], "mpp_xy": [.5, .5], "origin_px_xy": [100, 200], "origin_um_xy": [50, 100], "coordinate_space": "original_slide_micrometres"},
        "inputs": {key: {"sha256": sha256_file(path)} for key, path in {"image": image, "shift_json": shift, "resolution_json": resolution,
            "parent_mask": hierarchy / "parent_domains.ome.tif", "parent_uncertainty": hierarchy / "parent_uncertainty.ome.tif"}.items()}}
    summary["outputs"] = {str(path.relative_to(hierarchy)): sha256_file(path) for path in hierarchy.rglob("*") if path.is_file()}
    write_json(hierarchy / "hierarchy_summary.json", summary)
    return {"profile": profile, "labels": labels, "hierarchy": hierarchy, "image": image, "shift": shift, "resolution": resolution,
        "ring": ring_path, "parent": parent, "region": region, "labels_array": labels_array, "manifest": manifest}


def freeze(root):
    return {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def refresh_hierarchy(f):
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary["outputs"] = {str(p.relative_to(f["hierarchy"])): sha256_file(p) for p in f["hierarchy"].rglob("*") if p.is_file() and p != path}
    write_json(path, summary)


@pytest.mark.parametrize("code, name", [(11, "native_graph_isolate"), (12, "ambiguous_graph_assignment")])
def test_declared_graph_abstentions_retain_cells_without_region_assignment(tmp_path, code, name):
    # An explicit synthetic contract fixture, not a naturally fitted isolate.
    f = hierarchy_fixture(tmp_path / "inputs")
    root = f["hierarchy"]
    status_path = root / "hierarchy_status.ome.tif"
    status = tifffile.imread(status_path)
    status[-1] = code  # already positive-parent, zero-region tissue
    tifffile.imwrite(status_path, status)
    summary_path = root / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status_codes"][str(code)] = name
    write_json(summary_path, summary)
    refresh_hierarchy(f)
    verified = load_verified_hierarchy(f["profile"], f["labels"], root, tile_size=2)
    assert verified
    # Place an existing cell's ring pixel in this unresolved strip.
    ring = tifffile.imread(f["ring"])
    ring[-1, 1] = 2
    tifffile.imwrite(f["ring"], ring)
    comp_path = f["ring"].parent / "compartment_summary.json"
    comp = json.loads(comp_path.read_text())
    comp["output_artifacts"]["perinuclear_ring"]["sha256"] = sha256_file(f["ring"])
    write_json(comp_path, comp)
    profile_path = f["profile"] / "cell_profiles_manifest.json"
    manifest = json.loads(profile_path.read_text())
    manifest["tables"]["compartment_qc"]["summary_sha256"] = sha256_file(comp_path)
    write_json(profile_path, manifest)
    original = pd.read_parquet(f["profile"] / "cell_profiles.parquet")
    cells, _ = link_profiles(f["profile"], f["labels"], root, tmp_path / "linked",
                             ring_labels=f["ring"], tile_size=2)
    pd.testing.assert_frame_equal(cells[list(original)], original)
    overlaps = pd.read_parquet(tmp_path / "linked/cell_hierarchy_overlaps.parquet")
    abstained = overlaps[overlaps.hierarchy_status_code == code]
    assert len(abstained) == 1 and abstained.overlap_pixels.iloc[0] == 1
    assert abstained.parent_domain_id.gt(0).all()
    assert abstained.region_id.eq(0).all() and abstained.subdomain_id.eq(0).all()
    assert abstained.region_uid.eq("").all()


@pytest.mark.parametrize("legend", [None, {}, {"01": "accepted"}, {"13": "unknown"}, {"1": ""}])
def test_invalid_or_missing_status_legend_fails_closed(tmp_path, legend):
    f = hierarchy_fixture(tmp_path / "inputs")
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    if legend is None:
        summary.pop("status_codes")
    else:
        summary["status_codes"] = legend
    write_json(path, summary)
    with pytest.raises(ValueError, match="categorical hierarchy status legend"):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"], tile_size=2)


def test_exact_overlap_linkage_preserves_every_cell_feature_graph_and_input(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    before = freeze(tmp_path / "inputs")
    original = pd.read_parquet(f["profile"] / "cell_profiles.parquet")
    out = tmp_path / "linked"
    cells, manifest = link_profiles(f["profile"], f["labels"], f["hierarchy"], out, tile_size=2)
    pd.testing.assert_frame_equal(cells[list(original)], original)
    assert cells.cell_id.tolist() == ["2", "9000000001", "7"]
    for name in ("feature_rows.csv", "feature_blocks/test.npy", "graph.npz"):
        assert sha256_file(out / name) == sha256_file(f["profile"] / name)
    assert freeze(tmp_path / "inputs") == before
    assert sha256_file(out / "hierarchy_source/cell_profiles_manifest.json") == sha256_file(f["profile"] / "cell_profiles_manifest.json")
    assert manifest["cell_hierarchy"]["source_profile_manifest_sha256"] == sha256_file(f["profile"] / "cell_profiles_manifest.json")
    info = load_verified_hierarchy(out, f["labels"], f["hierarchy"], tile_size=3)
    overlaps = load_verified_links(out, info)
    crossing = overlaps[overlaps.cell_id == "2"]
    assert crossing.parent_domain_id.tolist() == [1, 2]
    assert crossing.overlap_pixels.tolist() == [3, 3]
    assert crossing.overlap_fraction.tolist() == [.5, .5]
    assert crossing.area_um2.tolist() == [.75, .75]
    assert cells.hierarchy_nucleus_crosses_parent_boundary.tolist() == [True, False, False]
    assert cells.hierarchy_nucleus_parent_uncertain_fraction.tolist() == [0., .25, 0.]
    assert cells.hierarchy_nucleus_background_fraction.tolist() == [0., 0., 1.]
    assert cells.hierarchy_nucleus_dominant_parent_id.iloc[0] == 1  # deterministic tie
    with pytest.raises(FileExistsError):
        link_profiles(f["profile"], f["labels"], f["hierarchy"], out)


def test_ring_empty_compartment_is_nan_and_native_counts_retained(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    cells, _ = link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked", ring_labels=f["ring"], tile_size=3)
    assert cells.hierarchy_perinuclear_ring_pixels.tolist() == [1, 1, 0]
    assert cells.hierarchy_perinuclear_ring_status.iloc[-1] == "empty_compartment"
    assert np.isnan(cells.hierarchy_perinuclear_ring_background_fraction.iloc[-1])


def test_array_backed_neighborhoods_survive_immutable_hierarchy_copy(tmp_path):
    from neighborhood_feature_io import finalize_store, FeatureColumns
    f = hierarchy_fixture(tmp_path / "inputs")
    root = f["profile"]
    manifest = json.loads((root / "cell_profiles_manifest.json").read_text())
    block = manifest["feature_blocks"]["test"]
    columns = ["own_test:feature_0", "own_test:feature_1"]
    manifest["neighborhood_feature_store"] = finalize_store(root, {"own:test": {
        "columns": columns, "segments": [{**block, "dtype": "float32", "columns": columns}]}})
    write_json(root / "cell_profiles_manifest.json", manifest)
    source = FeatureColumns(root)
    out = tmp_path / "linked"
    link_profiles(root, f["labels"], f["hierarchy"], out, tile_size=3)
    linked = FeatureColumns(out)
    np.testing.assert_array_equal(source.read(slice(None), columns), linked.read(slice(None), columns))
    assert linked.store_record == source.store_record
    for name in ("neighborhood_features/feature_store.json", "feature_blocks/test.npy", "feature_rows.csv"):
        assert sha256_file(out / name) == sha256_file(root / name)


def test_sparse_acellular_parent_tissue_is_not_removed(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    before = sha256_file(f["hierarchy"] / "parent_domains.ome.tif")
    info = load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"], tile_size=2)
    assert info["nuclear_pixels"] == {2: 6, 9000000001: 4, 7: 4}
    assert sha256_file(f["hierarchy"] / "parent_domains.ome.tif") == before
    assert np.all(tifffile.imread(info["raster_paths"]["parent"])[-1] > 0)
    assert np.all(tifffile.imread(info["raster_paths"]["region"])[-1] == 0)


@pytest.mark.parametrize("field", ["image", "shift_json", "resolution_json", "parent_mask", "parent_uncertainty"])
def test_source_fingerprint_mismatch_rejected(tmp_path, field):
    f = hierarchy_fixture(tmp_path / "inputs")
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary["inputs"][field]["sha256"] = "a" * 64
    write_json(path, summary)
    with pytest.raises(ValueError, match="source SHA256"):
        link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked")
    assert not (tmp_path / "linked").exists()


@pytest.mark.parametrize("key,value", [("sample_id", "wrong"), ("parent_labels_immutable", False), ("regions", 4)])
def test_hierarchy_identity_mismatch_rejected(tmp_path, key, value):
    f = hierarchy_fixture(tmp_path / "inputs")
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary[key] = value
    write_json(path, summary)
    with pytest.raises(ValueError):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"])


@pytest.mark.parametrize("key,value", [("shape_yx", [9, 10]), ("mpp_xy", [.25, .25]), ("origin_um_xy", [0, 0]), ("origin_px_xy", [0, 0])])
def test_calibration_geometry_mismatch_rejected(tmp_path, key, value):
    f = hierarchy_fixture(tmp_path / "inputs")
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    summary["geometry"][key] = value
    write_json(path, summary)
    with pytest.raises(ValueError, match="geometry differs"):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"])


def test_native_label_identity_and_artifact_hash_corruption_rejected(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    labels = f["labels_array"].copy()
    labels[0, 0] = 2
    tifffile.imwrite(f["labels"], labels)
    with pytest.raises(ValueError, match="labels SHA256"):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"])
    tifffile.imwrite(f["labels"], f["labels_array"])
    with (f["hierarchy"] / "region_mask.ome.tif").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"])


@pytest.mark.parametrize("raster,coordinate,value", [("region_mask.ome.tif", (2, 4), 9),
    ("subdomain_mask.ome.tif", (2, 4), 12), ("hierarchy_status.ome.tif", (0, 0), 1),
    ("region_mask.ome.tif", (7, 0), 5)])
def test_raster_geometry_inconsistency_rejected_even_with_refreshed_hash(tmp_path, raster, coordinate, value):
    f = hierarchy_fixture(tmp_path / "inputs")
    path = f["hierarchy"] / raster
    values = tifffile.imread(path)
    values[coordinate] = value
    tifffile.imwrite(path, values)
    refresh_hierarchy(f)
    with pytest.raises(ValueError):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"], tile_size=2)


def test_region_area_centroid_mismatch_rejected(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    root = f["hierarchy"] / "region_profiles"
    table = pd.read_csv(root / "region_profiles.csv")
    table.loc[0, "area_um2"] += .25
    table.to_csv(root / "region_profiles.csv", index=False)
    manifest = json.loads((root / "region_profiles_manifest.json").read_text())
    manifest["files"]["region_profiles.csv"] = sha256_file(root / "region_profiles.csv")
    write_json(root / "region_profiles_manifest.json", manifest)
    refresh_hierarchy(f)
    with pytest.raises(ValueError, match="area/centroid"):
        load_verified_hierarchy(f["profile"], f["labels"], f["hierarchy"])


def test_moved_tiled_workflow_is_self_contained_and_cli_works(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    out = tmp_path / "linked"
    result = subprocess.run([sys.executable, str(ROOT / "bin/link_cell_tissue_hierarchy.py"), "--profile-dir", str(f["profile"]),
        "--labels", str(f["labels"]), "--hierarchy-dir", str(f["hierarchy"]), "--outdir", str(out), "--tile-size", "2"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["cells"] == 3
    f["profile"].rename(tmp_path / "old_profile_unavailable")
    info = load_verified_hierarchy(out, f["labels"], f["hierarchy"], tile_size=2)
    assert len(load_verified_links(out, info)) == 5


def test_additive_canonical_lineage_rejects_changed_original_value(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    out = tmp_path / "linked"
    link_profiles(f["profile"], f["labels"], f["hierarchy"], out)
    cells = pd.read_parquet(out / "cell_profiles.parquet")
    cells.loc[0, "x_um"] += 1
    cells.to_parquet(out / "cell_profiles.parquet", index=False)
    manifest = json.loads((out / "cell_profiles_manifest.json").read_text())
    manifest["files"]["cell_profiles.parquet"] = sha256_file(out / "cell_profiles.parquet")
    write_json(out / "cell_profiles_manifest.json", manifest)
    info = load_verified_hierarchy(out, f["labels"], f["hierarchy"])
    with pytest.raises(ValueError, match="changed the original"):
        load_verified_links(out, info)


def test_overlap_memory_guard_is_explicit(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    with pytest.raises(ValueError, match="memory guard"):
        link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked", max_overlap_records=1)
    assert not (tmp_path / "linked").exists()


def test_rehashed_self_consistent_wrong_overlap_is_rejected_by_actual_rasters(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    out = tmp_path / "linked"
    _, manifest = link_profiles(f["profile"], f["labels"], f["hierarchy"], out)
    table = pd.read_parquet(out / "cell_hierarchy_overlaps.parquet")
    selected = table.cell_id == "2"
    table.loc[selected, "overlap_pixels"] = [4, 2]  # Still sums to the true nucleus area.
    table["overlap_fraction"] = table.overlap_pixels / table.compartment_pixels
    table["area_um2"] = table.overlap_pixels * .25
    table.to_parquet(out / "cell_hierarchy_overlaps.parquet", index=False)
    table.to_csv(out / "cell_hierarchy_overlaps.csv", index=False)
    original = pd.read_parquet(out / "hierarchy_source/cell_profiles.parquet")
    changed = _summaries(original, table, ["nucleus"])
    changed.to_parquet(out / "cell_profiles.parquet", index=False)
    changed.to_csv(out / "cell_profiles.csv", index=False)
    for name in ("cell_profiles.parquet", "cell_profiles.csv", "cell_hierarchy_overlaps.parquet", "cell_hierarchy_overlaps.csv"):
        manifest["files"][name] = sha256_file(out / name)
    manifest["hierarchy_links"]["table_sha256"] = manifest["files"]["cell_hierarchy_overlaps.parquet"]
    write_json(out / "cell_profiles_manifest.json", manifest)
    info = load_verified_hierarchy(out, f["labels"], f["hierarchy"])
    with pytest.raises(ValueError, match="exact native rasters"):
        load_verified_links(out, info)


def test_preserved_ring_can_be_verified_when_original_bundle_is_unavailable(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    out = tmp_path / "linked"
    link_profiles(f["profile"], f["labels"], f["hierarchy"], out, ring_labels=f["ring"])
    f["ring"].parent.rename(tmp_path / "original_ring_unavailable")
    info = load_verified_hierarchy(out, f["labels"], f["hierarchy"])
    table = load_verified_links(out, info)
    assert set(table.compartment) == {"nucleus", "perinuclear_ring"}
    ring = out / "hierarchy_source/labels_perinuclear_ring.tif"
    values = tifffile.imread(ring)
    values[6, 1] = 0
    tifffile.imwrite(ring, values)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_verified_links(out, info)


def test_tiles_are_bounded_on_both_axes(tmp_path, monkeypatch):
    import link_cell_tissue_hierarchy as module
    f = hierarchy_fixture(tmp_path / "inputs")
    original = module.RasterReader.window
    shapes = []
    def bounded(reader, x, y, x1, y1):
        shapes.append((y1 - y, x1 - x))
        assert x1 - x <= 3 and y1 - y <= 3
        return original(reader, x, y, x1, y1)
    monkeypatch.setattr(module.RasterReader, "window", bounded)
    link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked", tile_size=3)
    assert shapes and max(h for h, w in shapes) == 3 and max(w for h, w in shapes) == 3


def test_ring_cannot_overlap_any_nuclear_pixel_even_with_refreshed_provenance(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    ring = tifffile.imread(f["ring"])
    ring[0, 0] = 2  # The wrong neighbouring nucleus is still prohibited.
    tifffile.imwrite(f["ring"], ring)
    path = f["ring"].parent / "compartment_summary.json"
    summary = json.loads(path.read_text())
    summary["output_artifacts"]["perinuclear_ring"]["sha256"] = sha256_file(f["ring"])
    write_json(path, summary)
    path = f["profile"] / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tables"]["compartment_qc"]["summary_sha256"] = sha256_file(f["ring"].parent / "compartment_summary.json")
    write_json(path, manifest)
    with pytest.raises(ValueError, match="overlaps nuclear"):
        link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked", ring_labels=f["ring"])


def test_uint64_cell_ids_above_float_precision_remain_exact_with_signed_hierarchy(tmp_path):
    large = 9007199254741001
    f = hierarchy_fixture(tmp_path / "inputs", sparse_id=large)
    path = f["hierarchy"] / "subdomain_mask.ome.tif"
    tifffile.imwrite(path, tifffile.imread(path).astype(np.int32))
    refresh_hierarchy(f)
    cells, _ = link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked", tile_size=2)
    assert cells.cell_id.tolist() == ["2", str(large), "7"]
    table = pd.read_parquet(tmp_path / "linked/cell_hierarchy_overlaps.parquet")
    assert table.loc[table.cell_id == str(large), "overlap_pixels"].sum() == 4


def test_absent_upstream_uncertainty_is_unavailable_not_false_zero(tmp_path):
    f = hierarchy_fixture(tmp_path / "inputs")
    status_path = f["hierarchy"] / "hierarchy_status.ome.tif"
    status = tifffile.imread(status_path)
    status[status == 10] = 2
    tifffile.imwrite(status_path, status)
    path = f["hierarchy"] / "hierarchy_summary.json"
    summary = json.loads(path.read_text())
    del summary["inputs"]["parent_uncertainty"]
    write_json(path, summary)
    refresh_hierarchy(f)
    path = f["profile"] / "cell_profiles_manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["inputs"]["domain_uncertainty_sha256"]
    write_json(path, manifest)
    cells, manifest = link_profiles(f["profile"], f["labels"], f["hierarchy"], tmp_path / "linked")
    assert not manifest["hierarchy_links"]["parent_uncertainty_available"]
    assert cells.hierarchy_nucleus_parent_uncertain_fraction.isna().all()
    no_parent = cells[cells.cell_id == "7"].iloc[0]
    assert no_parent.hierarchy_nucleus_status == "no_parent_assignment_uncertainty_unavailable"
    assert np.isnan(no_parent.hierarchy_nucleus_background_fraction)
    assert np.isnan(no_parent.hierarchy_nucleus_unassigned_parent_uncertain_fraction)
    assert no_parent.hierarchy_nucleus_parent_zero_fraction == 1.
    assert no_parent.hierarchy_nucleus_unassigned_parent_uncertainty_unavailable_fraction == 1.
    table = pd.read_parquet(tmp_path / "linked/cell_hierarchy_overlaps.parquet")
    assert table.parent_uncertainty_code.eq(255).all()


def test_actual_uncertain_grid_core_preserves_literal_pixel_uncertainty(tmp_path):
    from cell_profile_io import RasterReader
    from discover_tissue_hierarchy import assign_parent_domains, rasterize
    f = hierarchy_fixture(tmp_path / "inputs")
    root = f["hierarchy"]
    summary = json.loads((root / "hierarchy_summary.json").read_text())
    # Native parent label 0 + code 253 means unassigned tissue, not background.
    uncertainty_path = root / "parent_uncertainty.ome.tif"
    raw = tifffile.imread(uncertainty_path)
    raw[2:4, 2:4] = 253
    tifffile.imwrite(uncertainty_path, raw)
    summary["inputs"]["parent_uncertainty"]["sha256"] = sha256_file(uncertainty_path)
    profile_manifest = json.loads((f["profile"] / "cell_profiles_manifest.json").read_text())
    profile_manifest["inputs"]["domain_uncertainty_sha256"] = sha256_file(uncertainty_path)
    write_json(f["profile"] / "cell_profiles_manifest.json", profile_manifest)
    # One uncertain source pixel makes this whole observed 2x2 core abstain.
    grid = pd.DataFrame([{"label": 1, "x": 0, "y": 0, "grid_row": 0, "grid_col": 0,
        "core_x0": 0, "core_y0": 0, "core_x1": 2, "core_y1": 2}])
    with RasterReader(root / "parent_domains.ome.tif") as parent, RasterReader(root / "parent_uncertainty.ome.tif") as uncertainty:
        assigned = assign_parent_domains(grid, parent, .8, uncertainty)
        assert assigned.status_code.tolist() == [10]
        subdomains, status, _ = rasterize(assigned, parent, root, summary["geometry"], 2, uncertainty)
        assert np.all(status[:2, :2] == 10)
    tifffile.imwrite(root / "region_mask.ome.tif", np.zeros((8, 10), np.uint32))
    region_root = root / "region_profiles"
    regions = pd.read_csv(region_root / "region_profiles.csv").iloc[:0]
    regions.to_csv(region_root / "region_profiles.csv", index=False)
    regions[["region_uid", "region_id", "sample_id"]].to_csv(region_root / "feature_rows.csv", index=False)
    manifest = json.loads((region_root / "region_profiles_manifest.json").read_text())
    manifest["region_count"] = 0
    for name in ("region_profiles.csv", "feature_rows.csv"):
        manifest["files"][name] = sha256_file(region_root / name)
    write_json(region_root / "region_profiles_manifest.json", manifest)
    summary["regions"] = 0
    write_json(root / "hierarchy_summary.json", summary)
    refresh_hierarchy(f)
    cells, _ = link_profiles(f["profile"], f["labels"], root, tmp_path / "linked", tile_size=2)
    cell = cells[cells.cell_id == "9000000001"].iloc[0]
    assert cell.hierarchy_nucleus_unresolved_fraction == 1.
    assert cell.hierarchy_nucleus_parent_uncertain_fraction == .25
    overlaps = pd.read_parquet(tmp_path / "linked/cell_hierarchy_overlaps.parquet")
    rows = overlaps[overlaps.cell_id == "9000000001"]
    assert rows.hierarchy_status_code.eq(10).all()
    assert rows.parent_uncertainty_code.tolist() == [0, 6]
    assert rows.overlap_pixels.tolist() == [3, 1]
    no_parent = cells[cells.cell_id == "7"].iloc[0]
    assert no_parent.hierarchy_nucleus_background_fraction == 0.
    assert no_parent.hierarchy_nucleus_parent_zero_fraction == 1.
    assert no_parent.hierarchy_nucleus_unassigned_parent_uncertain_fraction == 1.
    assert no_parent.hierarchy_nucleus_unresolved_fraction == 1.
    assert no_parent.hierarchy_nucleus_status != "background_only"
    assert np.allclose(cells.hierarchy_nucleus_background_fraction + cells.hierarchy_nucleus_unresolved_fraction + cells.hierarchy_nucleus_accepted_fraction, 1.)
