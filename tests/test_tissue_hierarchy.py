import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from cell_profile_io import RasterReader, sha256_file
from discover_tissue_hierarchy import (
    assign_parent_domains,
    connected_regions,
    discover_subdomains,
    export_region_profiles,
    load_embedding_block,
    main,
    rasterize,
    validate_feature_definitions,
    validate_geometry,
    validate_grid,
    validate_parent_support,
)


def grid_fixture(rows=8, cols=8, core=4):
    records = []
    for y in range(rows):
        for x in range(cols):
            records.append({"label": 1 + y * cols + x, "x": x * core + core // 2, "y": y * core + core // 2, "grid_row": y, "grid_col": x, "core_x0": x * core, "core_y0": y * core, "core_x1": (x + 1) * core, "core_y1": (y + 1) * core})
    grid = pd.DataFrame(records)
    parent = np.ones((rows * core, cols * core), np.uint16)
    parent[:, cols * core // 2:] = 2
    rng = np.random.default_rng(7)
    truth = (grid.y.to_numpy() >= rows * core // 2).astype(int)
    local = rng.normal(0, .08, (len(grid), 6)) + truth[:, None] * 5
    context = rng.normal(0, .08, (len(grid), 10)) + truth[:, None] * 3
    return grid, parent, {"local": local, "context": context}, truth


def geometry(shape):
    return {"mpp_xy": [.5, .5], "origin_px_xy": [100, 200], "origin_um_xy": [50, 100], "shape_yx": list(shape)}


def definitions():
    base = {"model_id": "test/uni2", "model_revision": "a" * 40, "preprocessing": "rgb,scale-preserving,model-normalization", "pooling": "cls"}
    return {"local": {**base, "field_width_source_px": 4, "input_context_width_source_px": 4}, "context": {**base, "field_width_source_px": 12, "input_context_width_source_px": 12}}


def test_parent_domains_immutable_and_subdomains_recover_known_partition(tmp_path):
    grid, parent, blocks, truth = grid_fixture()
    path = tmp_path / "parent.tif"
    tifffile.imwrite(path, parent)
    original_hash = sha256_file(path)
    with RasterReader(path) as reader:
        assigned = assign_parent_domains(validate_grid(grid, parent.shape), reader)
        mapped, report = discover_subdomains(assigned, blocks, min_observations=8, max_k=3, repeats=3)
        for parent_id in [1, 2]:
            selected = mapped.parent_domain_id == parent_id
            assert report["parent_domains"][str(parent_id)]["selected_k"] == 2
            assert adjusted_rand_score(truth[selected], mapped.loc[selected, "local_subdomain_id"]) == 1
        mask, status, counts = rasterize(mapped, reader, tmp_path, geometry(parent.shape), tile_size=7)
        assert counts == {1: parent.size // 2, 2: parent.size // 2}
        for subdomain in np.unique(mask):
            if subdomain:
                assert len(np.unique(parent[mask == subdomain])) == 1
        assert set(np.unique(status)) == {1}
    assert sha256_file(path) == original_hash
    np.testing.assert_array_equal(tifffile.imread(path), parent)


def test_background_gaps_diagonal_components_and_unobserved_tissue_retained(tmp_path):
    labels = np.array([[1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 1, 0], [0, 0, 1, 0, 0, 0], [2, 2, 0, 1, 1, 0]], np.uint32)
    components, stats = connected_regions(labels, tmp_path / "regions.tif", geometry(labels.shape), tile_size=2)
    assert len(stats) == 5
    assert components[0, 0] == components[1, 1]
    assert components[0, 0] != components[0, 3]
    assert components[2, 2] not in {components[1, 1], components[1, 3], components[3, 3]}
    assert np.all(components[labels == 0] == 0)
    grid, parent, blocks, _ = grid_fixture(rows=4, cols=4)
    # Drop some grid observations, not their parent tissue.
    grid = grid.iloc[:8].copy()
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    with RasterReader(tmp_path / "parent.tif") as reader:
        assigned = assign_parent_domains(grid, reader)
        mapped, _ = discover_subdomains(assigned, {name: block[:8] for name, block in blocks.items()}, min_observations=8, max_k=2, repeats=2)
        mask, status, counts = rasterize(mapped, reader, tmp_path, geometry(parent.shape))
        assert np.all(mask[8:] == 0)
        assert np.all(status[8:] == 2)
        assert sum(counts.values()) == parent.size


def test_mixed_parent_cores_and_missing_features_are_explicit(tmp_path):
    grid, parent, blocks, _ = grid_fixture()
    parent[:4, 2:4] = 2
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    with RasterReader(tmp_path / "parent.tif") as reader:
        assigned = assign_parent_domains(grid, reader)
    assert assigned.iloc[0].status_code == 3
    blocks["local"][1] = np.nan
    mapped, _ = discover_subdomains(assigned, blocks, min_observations=8, max_k=2, repeats=2)
    assert mapped.iloc[0].status_code == 3
    assert mapped.iloc[1].status_code == 4
    assert mapped.iloc[1].subdomain_id == 0


def test_discordant_scales_abstain_but_keep_raw_discovery(tmp_path):
    grid, parent, blocks, _ = grid_fixture()
    # Context has no discriminatory information, so independent scale evidence
    # must not be invented even when local features make a very stable split.
    blocks["context"][:] = 1
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    with RasterReader(tmp_path / "parent.tif") as reader:
        assigned = assign_parent_domains(grid, reader)
    mapped, report = discover_subdomains(assigned, blocks, fixed_k=2, max_k=2, min_observations=8, repeats=2)
    assert mapped.local_subdomain_id.gt(0).all()
    assert mapped.subdomain_id.eq(0).all()
    assert mapped.status_code.eq(8).all()
    assert mapped.scale_agreement.eq(.5).all()
    assert report["confidence_is_calibrated_probability"] is False


def test_deterministic_under_input_row_permutation(tmp_path):
    grid, parent, blocks, _ = grid_fixture()
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    with RasterReader(tmp_path / "parent.tif") as reader:
        assigned = assign_parent_domains(validate_grid(grid, parent.shape), reader)
        shuffled = assign_parent_domains(validate_grid(grid.sample(frac=1, random_state=17), parent.shape), reader)
    one, summary1 = discover_subdomains(assigned, blocks, min_observations=8, max_k=2, repeats=2)
    two, summary2 = discover_subdomains(shuffled, blocks, min_observations=8, max_k=2, repeats=2)
    pd.testing.assert_frame_equal(one, two)
    assert summary1 == summary2


def test_geometry_and_feature_scale_checks():
    meta = {"observation_type": "spatial_grid", "coordinate_space": "crop_roi_level0_pixels", "image_height_px": 10, "image_width_px": 20, "source_mpp_x": .5, "source_mpp_y": .5}
    shift = {"crop_size": {"height": 10, "width": 20}, "offset_crop_to_original": {"dx": 100, "dy": 200}}
    resolution = {"status": "pass", "mpp_x": .5, "mpp_y": .5, "width_px": 200, "height_px": 300}
    resolved = validate_geometry(meta, shift, resolution, (10, 20))
    assert resolved["origin_um_xy"] == [50, 100]
    checked = validate_feature_definitions(definitions(), resolved)
    assert checked["local"]["field_width_um_xy"] == [2, 2]
    assert checked["context"]["field_width_um_xy"] == [6, 6]
    with pytest.raises(ValueError, match="differs from verified"):
        validate_geometry({**meta, "source_mpp_x": .25}, shift, resolution, (10, 20))
    bad = definitions()
    bad["context"]["model_revision"] = "main"
    with pytest.raises(ValueError, match="immutable"):
        validate_feature_definitions(bad, resolved)
    bad = definitions()
    bad["context"]["field_width_source_px"] = 3
    with pytest.raises(ValueError, match="wider field"):
        validate_feature_definitions(bad, resolved)


def test_shard_loader_keys_metadata_missing_rows_and_rejects_duplicates(tmp_path):
    grid, _, _, _ = grid_fixture(rows=2, cols=2)
    source = pd.DataFrame({"cell_id": grid.label.iloc[:3].to_numpy(), "cx": grid.x.iloc[:3].to_numpy(), "cy": grid.y.iloc[:3].to_numpy(), "observation_type": "grid", "extraction_tile_size": 12, "feat_1": [1, 2, 3], "feat_2": [4, 5, 6]})
    path = tmp_path / "source.csv"
    source.sample(frac=1, random_state=22).to_csv(path, index=False)
    matrix, names, provenance = load_embedding_block(path, grid, tmp_path / "data.npy", definitions()["context"], chunk_rows=1)
    np.testing.assert_array_equal(matrix[:3, 0], [1, 2, 3])
    assert np.isnan(matrix[3]).all()
    assert names == ["feat_1", "feat_2"] and provenance["observations_missing"] == 1
    pd.concat([source, source.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicated"):
        load_embedding_block(path, grid, tmp_path / "duplicate.npy", definitions()["context"], chunk_rows=2)
    source["extraction_tile_size"] = 224
    source.to_csv(path, index=False)
    with pytest.raises(ValueError, match="context size"):
        load_embedding_block(path, grid, tmp_path / "bad.npy", definitions()["context"])


def test_cli_end_to_end_atlas_region_schema_and_parent_copy(tmp_path):
    grid, parent, blocks, _ = grid_fixture()
    grid.to_csv(tmp_path / "grid.csv", index=False)
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    meta = {"observation_type": "spatial_grid", "coordinate_space": "crop_roi_level0_pixels", "image_height_px": parent.shape[0], "image_width_px": parent.shape[1], "source_mpp_x": .5, "source_mpp_y": .5}
    shift = {"crop_size": {"height": parent.shape[0], "width": parent.shape[1]}, "offset_crop_to_original": {"dx": 100, "dy": 200}}
    resolution = {"status": "pass", "mpp_x": .5, "mpp_y": .5, "width_px": 500, "height_px": 500}
    for name, payload in [("metadata", meta), ("shift", shift), ("resolution", resolution), ("features", definitions())]:
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    for name, values in blocks.items():
        frame = pd.DataFrame(values, columns=[f"feat_{i + 1}" for i in range(values.shape[1])])
        frame["cell_id"], frame["cx"], frame["cy"] = grid.label, grid.x, grid.y
        frame["observation_type"] = "grid"
        frame["extraction_tile_size"] = definitions()[name]["input_context_width_source_px"]
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    out = tmp_path / "hierarchy"
    command = [sys.executable, str(ROOT / "bin/discover_tissue_hierarchy.py"), "--parent-mask", str(tmp_path / "parent.tif"), "--grid-objects", str(tmp_path / "grid.csv"), "--grid-metadata", str(tmp_path / "metadata.json"), "--shift-json", str(tmp_path / "shift.json"), "--resolution-json", str(tmp_path / "resolution.json"), "--local-embeddings", str(tmp_path / "local.csv"), "--context-embeddings", str(tmp_path / "context.csv"), "--embedding-metadata", str(tmp_path / "features.json"), "--sample-id", "s1", "--min-observations", "8", "--max-k", "2", "--repeats", "2", "--tile-size", "7", "--outdir", str(out)]
    # Preserve explicit coverage of the historical prototype; native KODAMA
    # is now the production/default route and has separate actual-R tests.
    command += ["--discovery-method", "legacy_kmeans"]
    subprocess.run(command, check=True)
    assert sha256_file(out / "parent_domains.ome.tif") == sha256_file(tmp_path / "parent.tif")
    summary = json.loads((out / "hierarchy_summary.json").read_text())
    assert summary["parent_labels_immutable"] is True
    assert summary["regions"] == 4 and summary["subdomains"] == 4
    for relative, digest in summary["outputs"].items():
        assert sha256_file(out / relative) == digest
    assert {"region_mask.ome.tif", "hierarchy_status.ome.tif", "parent_domains.ome.tif", "grid_subdomains.csv", "region_profiles/region_profiles_manifest.json"} <= set(summary["outputs"])
    directory = out / "region_profiles"
    manifest = json.loads((directory / "region_profiles_manifest.json").read_text())
    profiles = pd.read_csv(directory / "region_profiles.csv")
    assert profiles.region_uid.is_unique and profiles.subdomain_id.is_unique
    assert profiles.x_um.ge(50).all() and profiles.y_um.ge(100).all()
    assert manifest["observation_unit"] == "tissue_region"
    for name in ["local", "context"]:
        entry = manifest["feature_blocks"][name]
        matrix = np.load(directory / entry["path"])
        assert list(matrix.shape) == entry["shape"]
        assert matrix.shape[0] == 4
        assert entry["feature_definition"]["model_revision"] == "a" * 40
    output = subprocess.run(command, capture_output=True, text=True)
    assert output.returncode != 0 and "new or empty" in output.stderr


def test_grid_overlap_and_component_limit_rejected(tmp_path):
    grid, parent, _, _ = grid_fixture()
    grid.loc[1, "core_x0"] = 0
    with pytest.raises(ValueError, match="overlap"):
        validate_grid(grid, parent.shape)
    fragmented = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint32)
    with pytest.raises(ValueError, match="component bound"):
        connected_regions(fragmented, tmp_path / "regions.tif", geometry(fragmented.shape), tile_size=2, max_components=4)


def test_scaled_support_holes_fail_closed_without_parent_changes(tmp_path):
    parent = np.ones((12, 15), np.uint16)
    support = np.ones((4, 5), np.uint8)
    support[1, 2] = 0
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    tifffile.imwrite(tmp_path / "support.tif", support)
    digest = sha256_file(tmp_path / "parent.tif")
    with RasterReader(tmp_path / "parent.tif") as p, RasterReader(tmp_path / "support.tif") as s:
        with pytest.raises(ValueError, match="9 pixels"):
            validate_parent_support(p, s, tile_size=2)
    assert sha256_file(tmp_path / "parent.tif") == digest
    parent[3:6, 6:9] = 0
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    with RasterReader(tmp_path / "parent.tif") as p, RasterReader(tmp_path / "support.tif") as s:
        checked = validate_parent_support(p, s, tile_size=5)
    assert checked["support_checked"] and checked["parent_pixels_outside_support"] == 0


def test_uncertain_parent_pixels_never_become_accepted_subdomains(tmp_path):
    grid, parent, blocks, _ = grid_fixture()
    uncertainty = np.zeros(parent.shape, np.uint8)
    uncertainty[:4, :4] = 254  # whole growth-inferred core
    uncertainty[:, 9] = 7  # categorical upstream strip, not a probability
    tifffile.imwrite(tmp_path / "parent.tif", parent)
    tifffile.imwrite(tmp_path / "uncertainty.tif", uncertainty)
    with RasterReader(tmp_path / "parent.tif") as p, RasterReader(tmp_path / "uncertainty.tif") as u:
        checked = validate_parent_support(p, parent_uncertainty=u, tile_size=7)
        assert checked["parent_uncertainty_code_counts"][254] == 16
        assigned = assign_parent_domains(grid, p, parent_uncertainty=u)
        assert assigned.iloc[0].status_code == 10
        mapped, _ = discover_subdomains(assigned, blocks, min_observations=8, max_k=2, repeats=2)
        mask, status, _ = rasterize(mapped, p, tmp_path, geometry(parent.shape), tile_size=7, parent_uncertainty=u)
        assert np.all(mask[uncertainty > 0] == 0)
        assert np.all(status[uncertainty > 0] == 10)
        assert np.any(status[uncertainty == 0] == 1)
    np.testing.assert_array_equal(tifffile.imread(tmp_path / "uncertainty.tif"), uncertainty)
