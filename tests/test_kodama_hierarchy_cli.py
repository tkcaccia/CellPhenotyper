"""Small actual-native hierarchy acceptance; synthetic features, no image encoder.

The explicit locally installed native package is never installed or downloaded by
these tests. Two specimen runs exercise the default CLI, full raster/profile
export and portable corrected graphs; they are not tissue-accuracy benchmarks.
"""
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
NATIVE_LIBRARY = Path("/Users/stefano/Documents/KODAMA-cpp 2/tmp/Rlib-kodama-latest")
sys.path.insert(0, str(ROOT / "bin"))
from discover_tissue_hierarchy import validate_feature_definitions, validate_geometry
from test_tissue_hierarchy import definitions, grid_fixture
from test_tissue_hierarchy_workflow import workflow_fixture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def verify_output_inventory(out, summary):
    """Validate the durable package after the CLI/Nextflow process has exited."""
    actual = {str(path.relative_to(out)) for path in out.rglob("*")
              if path.is_file() and path != out / "hierarchy_summary.json"}
    expected = summary["outputs"]
    assert set(expected) == actual, ("Durable hierarchy output inventory differs", set(expected) - actual, actual - set(expected))
    for relative, sha256 in expected.items():
        assert (out / relative).is_file(), f"Missing declared durable output: {relative}"
        assert digest(out / relative) == sha256, f"Declared output checksum differs: {relative}"


def test_postprocess_inventory_rejects_ephemeral_feature_claim(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"durable output")
    summary = {"outputs": {"artifact.bin": digest(tmp_path / "artifact.bin"),
                           "hierarchy_features_removed/context.npy": "a" * 64}}
    with pytest.raises(AssertionError, match="Durable hierarchy output inventory differs"):
        verify_output_inventory(tmp_path, summary)


def source_fixture(root, sample="native_a", seed=7):
    """Source-bound synthetic NPY features, deliberately non-canonical file order."""
    root.mkdir(parents=True)
    grid, parent, blocks, _ = grid_fixture()
    # Exclusions are explicit, not silently removed from observation lineage.
    uncertainty = np.zeros(parent.shape, np.uint8)
    uncertainty[:4, :4] = 254  # ID1: uncertain upstream parent assignment.
    parent[:4, 6:8] = 2      # ID2: a 50:50 mixed-parent core.
    blocks["local"][2] = np.nan  # ID3: missing local feature vector.
    parent[:4, 28:32] = 0    # ID8: real declared background, not a third parent.
    grid = grid.iloc[:-1].copy()  # ID64: unobserved positive-parent tissue remains.
    blocks = {name: np.asarray(values[:-1], dtype=np.float32) for name, values in blocks.items()}
    image = np.random.default_rng(seed).integers(20, 245, (*parent.shape, 3), dtype=np.uint8)
    tifffile.imwrite(root / "image.tif", image, photometric="rgb")
    tifffile.imwrite(root / "parent.tif", parent)
    tifffile.imwrite(root / "uncertainty.tif", uncertainty)
    tifffile.imwrite(root / "support.tif", (parent > 0).astype(np.uint8))
    grid.sample(frac=1, random_state=seed).to_csv(root / "grid.csv", index=False)
    meta = {"observation_type": "spatial_grid", "coordinate_space": "crop_roi_level0_pixels",
            "image_height_px": 32, "image_width_px": 32, "source_mpp_x": .5, "source_mpp_y": .5}
    shift = {"crop_size": {"height": 32, "width": 32},
             "offset_crop_to_original": {"dx": 100, "dy": 200}}
    resolution = {"status": "pass", "mpp_x": .5, "mpp_y": .5, "width_px": 500, "height_px": 500}
    # Explicit fixture identity: never suggest that these arrays came from UNI2.
    feature_definitions = definitions()
    for definition in feature_definitions.values():
        definition.update(model_id="synthetic/engineering-fixture-no-encoder",
                          preprocessing="deterministic synthetic numeric features; no learned inference")
    features = root / "features"
    features.mkdir()
    for name, payload in (("metadata", meta), ("shift", shift), ("resolution", resolution)):
        write_json(root / f"{name}.json", payload)
    write_json(features / "embedding_metadata.json", feature_definitions)
    geometry = validate_geometry(meta, shift, resolution, parent.shape)
    normalized = validate_feature_definitions(feature_definitions, geometry)
    sources = {"image": root / "image.tif", "grid_objects": root / "grid.csv",
               "grid_metadata": root / "metadata.json", "shift_json": root / "shift.json",
               "resolution_json": root / "resolution.json"}
    for index, (name, values) in enumerate(blocks.items()):
        folder = features / name
        folder.mkdir()
        order = np.random.default_rng(seed + index + 1).permutation(len(grid))
        np.save(folder / "features.npy", values[order])
        rows = pd.DataFrame({"cell_id": grid.label.astype(str), "cx": grid.x, "cy": grid.y,
                             "observation_type": "grid", "source_mpp": .5,
                             "extraction_tile_size": normalized[name]["input_context_width_source_px"]})
        rows.iloc[order].to_csv(folder / "rows.csv", index=False)
        write_json(folder / "embedding_manifest.json", {
            "format": "cellphenotyper_grid_embeddings_npy", "sample_id": sample,
            "feature_definition": normalized[name],
            "matrix": {"path": "features.npy", "sha256": digest(folder / "features.npy"), "shape": list(values.shape)},
            "rows": {"path": "rows.csv", "sha256": digest(folder / "rows.csv"), "count": len(grid)},
            "feature_names": [f"feat_{i + 1}" for i in range(values.shape[1])],
            "source_inputs": {key: {"sha256": digest(path)} for key, path in sources.items()}})
    hashes = {str(path): digest(path) for path in sorted(root.rglob("*")) if path.is_file()}
    return {"root": root, "sample": sample, "grid": grid, "parent": parent,
            "uncertainty": uncertainty, "blocks": blocks, "hashes": hashes, "geometry": geometry}


def cli_arguments(fixture, out):
    root = fixture["root"]
    command = [sys.executable, "-B", str(ROOT / "bin/discover_tissue_hierarchy.py")]
    for option, path in {
        "image": root / "image.tif", "parent-mask": root / "parent.tif",
        "parent-uncertainty": root / "uncertainty.tif", "support-mask": root / "support.tif",
        "grid-objects": root / "grid.csv", "grid-metadata": root / "metadata.json",
        "shift-json": root / "shift.json", "resolution-json": root / "resolution.json",
        "local-embeddings": root / "features/local", "context-embeddings": root / "features/context",
        "embedding-metadata": root / "features/embedding_metadata.json"}.items():
        command += ["--" + option, str(path)]
    # Intentionally omit --discovery-method to test the default native path.
    command += ["--sample-id", fixture["sample"], "--outdir", str(out),
                "--kodama-r-library", str(NATIVE_LIBRARY), "--kodama-ncomp", "50",
                "--kodama-m", "2", "--kodama-tcycle", "2", "--kodama-neighbors", "4",
                "--kodama-cpus", "1", "--components-per-block", "3", "--fit-limit", "20",
                "--min-observations", "8", "--max-k", "5", "--repeats", "2",
                "--seed", "17", "--min-seed-stability", "0", "--min-scale-agreement", "0",
                "--min-affinity-margin", "0", "--tile-size", "7"]
    return command


def run_cli(fixture, out, *, extra=()):
    cache = out.parent / (out.name + "_fresh_pycache")
    cache.mkdir()
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(cache), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "LOKY_MAX_CPU_COUNT": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    # Test the explicitly requested library and discovered system Rscript, not
    # a caller's unrelated optional runtime override.
    env.pop("CELLPHENOTYPER_HIERARCHY_RSCRIPT", None)
    command = [*cli_arguments(fixture, out), *extra]
    result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=180)
    (out.parent / (out.name + "_stdout.txt")).write_text(result.stdout)
    (out.parent / (out.name + "_stderr.txt")).write_text(result.stderr)
    write_json(out.parent / (out.name + "_command.json"), command)
    assert not list(cache.iterdir()), "Fresh source isolation must not populate shared/library bytecode caches"
    return result


@pytest.fixture(scope="module")
def native_runtime():
    executable = shutil.which("Rscript")
    if not executable or not (NATIVE_LIBRARY / "KODAMA/DESCRIPTION").is_file():
        pytest.skip("Explicit local native KODAMA R runtime is not installed; no automatic installation")
    code = """args <- commandArgs(TRUE); .libPaths(c(args[1], .libPaths()))
stopifnot(all(c('KODAMA.matrix','KODAMA.graph.materialize') %in% getNamespaceExports('KODAMA')))
stopifnot(all(c('ncomp','M','Tcycle','graph.neighbors','return.graph') %in% names(formals(KODAMA::KODAMA.matrix))))
cat(as.character(utils::packageVersion('KODAMA')))"""
    result = subprocess.run([executable, "-e", code, str(NATIVE_LIBRARY)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return {"executable": executable, "version": result.stdout.strip()}


@pytest.fixture(scope="module", params=[("native_a", 7), ("native_b", 19)])
def completed(request, tmp_path_factory, native_runtime):
    root = tmp_path_factory.mktemp(request.param[0])
    fixture = source_fixture(root / "sources", *request.param)
    out = root / "hierarchy"
    result = run_cli(fixture, out)
    logs = "\n".join(path.read_text() for path in out.rglob("native.log")) if out.exists() else ""
    assert result.returncode == 0, result.stdout + result.stderr + logs
    return fixture, out, native_runtime


def test_default_native_full_cli_preserves_parent_population_geometry_and_sources(completed):
    fixture, out, runtime = completed
    summary = json.loads((out / "hierarchy_summary.json").read_text())
    mapped = pd.read_csv(out / "grid_subdomains.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(mapped[fixture["grid"].columns], fixture["grid"].reset_index(drop=True))
    assert mapped.label.tolist() == list(range(1, 64))
    assert mapped.sample_id.eq(fixture["sample"]).all()
    np.testing.assert_array_equal(mapped[["x_um", "y_um"]], mapped[["x", "y"]].to_numpy() * .5 + [50, 100])
    assert summary["geometry"] == fixture["geometry"]
    assert summary["parent_labels_immutable"] is True
    assert (out / "parent_domains.ome.tif").read_bytes() == (fixture["root"] / "parent.tif").read_bytes()
    assert (out / "parent_uncertainty.ome.tif").read_bytes() == (fixture["root"] / "uncertainty.tif").read_bytes()
    assert {path: digest(path) for path in fixture["hashes"]} == fixture["hashes"]
    assert summary["sources_unchanged_after_export"] is True
    assert summary["native_artifacts_unchanged_after_export"] is True
    verify_output_inventory(out, summary)
    assert summary["realized_identity"]["realized_grid_assignment_sha256"] == digest(out / "grid_subdomains.csv")
    assert len(summary["realized_identity"]["native_graph_sha256"]) == 6
    for label, expected in ((1, 10), (2, 3), (3, 4), (8, 0)):
        row = mapped.set_index("label").loc[label]
        assert row.status_code == expected
        assert row.subdomain_id == row.local_subdomain_id == 0
    status = tifffile.imread(out / "hierarchy_status.ome.tif")
    subdomains = tifffile.imread(out / "subdomain_mask.ome.tif")
    regions = tifffile.imread(out / "region_mask.ome.tif")
    assert np.all(status[28:32, 28:32] == 2), "Absent observation must not remove positive parent tissue"
    assert np.all(status[:4, :4] == 10)
    assert np.all(subdomains[status != 1] == 0)
    assert np.all(regions[status != 1] == 0)
    assert np.all(status[fixture["parent"] == 0] == 0)
    for subdomain in np.unique(subdomains[subdomains > 0]):
        assert len(np.unique(fixture["parent"][subdomains == subdomain])) == 1
    assert summary["grid_observations"] == 63
    assert sum(summary["status_pixel_counts"].values()) == 32 * 32
    assert sum(summary["parent_pixel_counts"].values()) == np.count_nonzero(fixture["parent"])
    discovery = summary["discovery"]
    assert discovery["method"] == "within_parent_native_kodama_graph"
    assert discovery["requested_kodama_ncomp"] == 50
    assert discovery["pca_components_per_block"] == 3
    assert discovery["confidence_is_calibrated_probability"] is False
    assert "centroid_margin" not in mapped.columns
    assert {"affinity_margin", "own_affinity_fraction", "graph_degree"} <= set(mapped)
    assert not mapped.status_code.eq(9).any()
    for name, sha in discovery["code_sha256"].items():
        assert digest(out / "kodama_graphs/code" / name) == sha == digest(ROOT / "bin" / name)
    evidence = {"sample_id": fixture["sample"], "status": "pass", "native_version": runtime["version"],
                "observations": len(mapped), "parent_domains": 2, "regions": summary["regions"],
                "status_counts": mapped.status_code.value_counts().sort_index().to_dict(),
                "hierarchy_summary_sha256": digest(out / "hierarchy_summary.json"),
                "native_graph_sha256": summary["realized_identity"]["native_graph_sha256"],
                "source_sha256": fixture["hashes"], "code_sha256": discovery["code_sha256"],
                "interpretation": "Synthetic engineering acceptance; real native CPU fit, no image encoder or biological accuracy claim"}
    write_json(out.parent / "acceptance_summary.json", evidence)


def read_graphs(out, runtime):
    """Read actual plain graph slots; calculate evidence independently in Python."""
    code = """args <- commandArgs(TRUE); .libPaths(c(args[2], .libPaths())); requireNamespace('Matrix')
paths <- list.files(args[1], pattern='^kodama_graph.rds$', recursive=TRUE, full.names=TRUE)
exact <- function(x) matrix(sprintf('%.17g', x), nrow=nrow(x), ncol=ncol(x))
result <- lapply(paths, function(path) {
 p <- readRDS(path); g <- p$distance_graph
 e <- new.env(parent=emptyenv()); load(file.path(dirname(path),'input_features.RData'), envir=e)
 list(path=path, ids=p$observation_ids, metadata=p$metadata,
      i=g@i, p=g@p, x=sprintf('%.17g', g@x), pca=exact(e$pca), xy=exact(e$xy), pca_ids=rownames(e$pca), xy_ids=rownames(e$xy))
})
cat(jsonlite::toJSON(result, auto_unbox=TRUE, digits=NA, na='null', null='null'))"""
    result = subprocess.run([runtime["executable"], "-e", code, str(out / "kodama_graphs"), str(NATIVE_LIBRARY)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    # requireNamespace prints TRUE when called at top level; consume last JSON line.
    return json.loads(result.stdout.splitlines()[-1])


def test_saved_native_graphs_bind_all_eligible_rows_and_actual_affinity_not_centroids(completed):
    fixture, out, runtime = completed
    summary = json.loads((out / "hierarchy_summary.json").read_text())
    mapped = pd.read_csv(out / "grid_subdomains.csv", float_precision="round_trip")
    graph_records = read_graphs(out, runtime)
    assert len(graph_records) == 6
    for graph in graph_records:
        folder = Path(graph["path"]).parent
        representation, parent_dir = folder.name, folder.parent.parent
        parent_id = int(parent_dir.name.removeprefix("parent_"))
        report = summary["discovery"]["parent_domains"][str(parent_id)]
        execution = report["native_execution"]
        runner = execution["runner"]
        generated = json.loads((parent_dir / "input_manifest.json").read_text())
        assert execution["input_manifest"] == generated, "The exact original manifest, not R's descriptive float echo, is authoritative"
        params = runner["requested_parameters"]
        assert params["ncomp"] == 50 and params["M"] == params["Tcycle"] == 2
        assert params["cores"] == 1 and params["neighbors"] == 4
        assert params["cluster_seeds"] == [17, 18, 19]
        assert params["landmarks"] == report["pca_training_observations"] == 20
        assert report["native_fit_observations"] > 20, "fit-limit must not project or omit native graph observations"
        assert runner["sources_before"] == runner["sources_after"] == execution["input_sha256"]
        assert runner["producer_before"] == runner["producer_after"] == execution["producer_sha256"]
        assert runner["native_package_before_load"] == runner["native_package_after"]
        assert runner["runtime_before"] == runner["runtime_after"]
        for hashes in (execution["input_sha256"], execution["artifact_sha256"], execution["producer_sha256"]):
            assert all(digest(path) == expected for path, expected in hashes.items())
        assert Path(execution["command"][-1]) == NATIVE_LIBRARY.resolve()
        assert execution["command"][1] == "--vanilla", "Unbound user/site R startup scripts must not influence native execution"
        assert runner["runtime"]["packages"]["KODAMA"]["version"] == runtime["version"]
        native = runner["representations"][representation]
        assert native["status"] == "fit"
        assert native["effective_native_parameters"]["classifier"] == "knn"
        assert native["effective_native_parameters"]["backend"] == "cpu"
        assert native["effective_native_parameters"]["ncomp"] == 50
        assert native["dimensions"] == (6 if representation == "combined" else 3)
        assert native["ncomp_applicability"].startswith("not_PLS_rank_for_raw_data_knn_classifier")
        assert graph["metadata"]["all_observations"] is True
        assert graph["metadata"]["projected"] is False
        assert graph["metadata"]["native_corrected"] is True
        assert graph["metadata"]["directed"] is True
        assert graph["ids"] == graph["pca_ids"] == graph["xy_ids"]
        indices = generated["scope"]["source_row_indices"]
        expected = mapped.iloc[indices]
        assert graph["ids"] == expected.label.astype(str).tolist()
        assert not set(graph["ids"]) & {"1", "2", "3", "8", "64"}
        assert generated["scope"]["coordinates_used_for_fitting"] is False
        np.testing.assert_array_equal(np.asarray(graph["xy"], float), expected[["x_um", "y_um"]])
        shape = generated["representations"][representation]["shape"]
        prepared = np.fromfile(parent_dir / f"{representation}.f64", dtype="<f8").reshape(shape)
        np.testing.assert_array_equal(np.asarray(graph["pca"], float), prepared)
        n = len(graph["ids"])
        affinity = np.zeros((n, n), dtype=np.float64)  # bounded to <32 vertices in this fixture only
        columns = np.repeat(np.arange(n), np.diff(graph["p"]))
        affinity[np.asarray(graph["i"], int), columns] = 1 / (1 + np.asarray(graph["x"], float))
        affinity = np.maximum(affinity, affinity.T)
        assert not np.diag(affinity).any()
        partitions = pd.read_csv(folder.parent / "partitions.csv", dtype={"label": str}, float_precision="round_trip")
        partitions = partitions[partitions.representation == representation]
        assert len(partitions) == n * 8 * 3
        for (_, _), rows in partitions.groupby(["resolution", "seed"], sort=False):
            assert rows.label.tolist() == graph["ids"]
            clusters = rows.cluster.to_numpy()
            strength = affinity.sum(axis=1)
            degree = np.count_nonzero(affinity, axis=1)
            own = np.array([affinity[i, clusters == clusters[i]].sum() for i in range(n)])
            other = np.array([max((affinity[i, clusters == c].sum() for c in np.unique(clusters)
                                  if c != clusters[i]), default=0.) for i in range(n)])
            np.testing.assert_array_equal(rows.degree, degree)
            np.testing.assert_allclose(rows.own_affinity_fraction, np.divide(own, strength, out=np.full(n, np.nan), where=strength > 0), atol=2e-15, rtol=2e-14)
            np.testing.assert_allclose(rows.affinity_margin, np.divide(own - other, strength, out=np.full(n, np.nan), where=strength > 0), atol=2e-15, rtol=2e-14)
        if representation == "combined" and report["selected_resolution"] is not None:
            selected = partitions[(partitions.resolution == report["selected_resolution"]) & (partitions.seed == 17)]
            np.testing.assert_array_equal(expected.local_subdomain_id, selected.cluster)
            np.testing.assert_array_equal(expected.graph_degree, selected.degree)
            np.testing.assert_array_equal(expected.affinity_margin, selected.affinity_margin)


@pytest.mark.parametrize("field, basename", [
    ("input_sha256", "combined.f64"),
    ("artifact_sha256", "kodama_graph.rds"),
    ("producer_sha256", "run_hierarchy_kodama.R"),
])
def test_native_receipt_rejects_modified_retained_inputs_graph_or_producer(completed, tmp_path, field, basename):
    _, out, _ = completed
    summary = json.loads((out / "hierarchy_summary.json").read_text())
    execution = summary["discovery"]["parent_domains"]["1"]["native_execution"]
    original = next(Path(path) for path in execution[field] if Path(path).name == basename)
    expected = execution[field].pop(str(original))
    copy = tmp_path / basename
    shutil.copyfile(original, copy)
    execution[field][str(copy)] = expected
    receipt = tmp_path / "execution.json"
    write_json(receipt, execution)
    cache = tmp_path / "fresh_pycache"
    cache.mkdir()
    env = {**os.environ, "PYTHONPYCACHEPREFIX": str(cache), "PYTHONDONTWRITEBYTECODE": "1"}
    code = "import json,sys; sys.path.insert(0,sys.argv[1]); from hierarchy_kodama import _verify_native_execution; _verify_native_execution(json.load(open(sys.argv[2])))"
    command = [sys.executable, "-B", "-c", code, str(ROOT / "bin"), str(receipt)]
    clean = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    payload = copy.read_bytes()
    copy.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    changed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
    assert changed.returncode != 0
    assert field + " identity differs" in changed.stderr
    assert digest(original) == expected, "Only an isolated test copy may be corrupted"
    assert not list(cache.iterdir())


def test_native_region_profiles_are_actual_area_weighted_source_features(completed):
    fixture, out, _ = completed
    mapped = pd.read_csv(out / "grid_subdomains.csv", float_precision="round_trip")
    raster = tifffile.imread(out / "region_mask.ome.tif")
    profiles = pd.read_csv(out / "region_profiles/region_profiles.csv", dtype={"region_id": str}, float_precision="round_trip")
    manifest = json.loads((out / "region_profiles/region_profiles_manifest.json").read_text())
    assert len(profiles) > 0, "Synthetic fixture must exercise nonempty region profiles, not only abstention"
    assert manifest["region_count"] == len(profiles)
    assert profiles.region_uid.is_unique and profiles.sample_id.eq(fixture["sample"]).all()
    assert profiles.label_status.eq("unsupervised_discovery").all()
    for ri, region in profiles.iterrows():
        region_id = int(region.region_id)
        yy, xx = np.where(raster == region_id)
        assert region.area_um2 == len(xx) * .25
        np.testing.assert_allclose([region.x_um, region.y_um], [(xx + .5).mean() * .5 + 50, (yy + .5).mean() * .5 + 100], rtol=0, atol=2e-14)
        weights = np.array([np.count_nonzero(raster[row.core_y0:row.core_y1, row.core_x0:row.core_x1] == region_id)
                            if row.status_code == 1 else 0 for row in mapped.itertuples()])
        assert weights.sum() == len(xx)
        assert int(region.representative_grid_id) in set(mapped.loc[weights > 0, "label"])
        for name, values in fixture["blocks"].items():
            actual = np.load(out / "region_profiles" / f"{name}.npy")
            expected = (values[weights > 0].astype(np.float64) * weights[weights > 0, None]).sum(axis=0) / weights.sum()
            np.testing.assert_array_equal(actual[ri], expected.astype(np.float32))
            assert manifest["feature_blocks"][name]["feature_definition"]["aggregation"] == "assigned_grid_core_tissue_area_weighted_mean"


@pytest.mark.parametrize("fault, expected", [
    ("foreign_feature_id", "IDs must exactly match"),
    ("changed_image", "image checksum differs"),
    ("feature_coordinate", "coordinates disagree"),
    ("bad_geometry", "dimensions differ"),
    ("support_leak", "outside"),
    ("existing_output", "preserve previous analyses"),
])
def test_invalid_sources_fail_before_native_fit_and_preserve_inputs(tmp_path, fault, expected):
    fixture = source_fixture(tmp_path / "sources")
    root, out = fixture["root"], tmp_path / "hierarchy"
    if fault in {"foreign_feature_id", "feature_coordinate"}:
        path = root / "features/local/rows.csv"
        rows = pd.read_csv(path, dtype={"cell_id": str})
        rows.loc[0, "cell_id" if fault == "foreign_feature_id" else "cx"] = "999" if fault == "foreign_feature_id" else 999
        rows.to_csv(path, index=False)
        receipt_path = root / "features/local/embedding_manifest.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["rows"]["sha256"] = digest(path)
        write_json(receipt_path, receipt)
    elif fault == "changed_image":
        pixels = tifffile.imread(root / "image.tif")
        pixels[0, 0, 0] += 1
        tifffile.imwrite(root / "image.tif", pixels, photometric="rgb")
    elif fault == "bad_geometry":
        path = root / "shift.json"
        shift = json.loads(path.read_text())
        shift["crop_size"]["width"] = 31
        write_json(path, shift)
    elif fault == "support_leak":
        support = tifffile.imread(root / "support.tif")
        support[12, 12] = 0
        tifffile.imwrite(root / "support.tif", support)
    else:
        out.mkdir()
        (out / "previous.txt").write_text("Keep previous successful hierarchy")
    before = {str(path): digest(path) for path in root.rglob("*") if path.is_file()}
    result = run_cli(fixture, out)
    assert result.returncode != 0
    assert expected in result.stdout + result.stderr
    assert not (out / "hierarchy_summary.json").exists()
    assert not (out / "kodama_graphs").exists(), "Invalid inputs must fail before any native fit"
    assert {path: digest(path) for path in before} == before
    if fault == "existing_output":
        assert (out / "previous.txt").read_text() == "Keep previous successful hierarchy"


def test_two_specimen_actual_nextflow_module_uses_native_runtime_without_encoder(tmp_path, native_runtime):
    executable = shutil.which("nextflow")
    if not executable:
        pytest.skip("Nextflow executable is not available")
    workflow_fixture(tmp_path)
    fixtures = [source_fixture(tmp_path / name, name, seed) for name, seed in (("workflow_a", 31), ("workflow_b", 43))]
    params_path = tmp_path / "params.json"
    params = json.loads(params_path.read_text())
    params.update(tissue_hierarchy_python=sys.executable, tissue_hierarchy_kodama_r_library=str(NATIVE_LIBRARY),
                  tissue_hierarchy_kodama_m=2, tissue_hierarchy_kodama_tcycle=2, tissue_hierarchy_kodama_neighbors=4,
                  tissue_hierarchy_components_per_block=3, tissue_hierarchy_fit_limit=20,
                  tissue_hierarchy_min_observations=8, tissue_hierarchy_repeats=2,
                  tissue_hierarchy_min_seed_stability=0, tissue_hierarchy_min_scale_agreement=0,
                  tissue_hierarchy_min_affinity_margin=0, tissue_hierarchy_tile_size=7)
    write_json(params_path, params)
    write_json(tmp_path / "specimens.json", [[str(fixture["root"]), fixture["sample"]] for fixture in fixtures])
    (tmp_path / "nextflow.config").write_text("process.executor = 'local'\nexecutor.queueSize = 1\ntrace.fields = 'task_id,name,status,cpus,memory,hash'\n")
    (tmp_path / "workflow.nf").write_text('''nextflow.enable.dsl=2
include { DISCOVER_TISSUE_HIERARCHY } from './modules/discover_tissue_hierarchy'
workflow {
    rows = new groovy.json.JsonSlurper().parse(file("${projectDir}/specimens.json").toFile())
    specimens = Channel.fromList(rows).map { root, id ->
        tuple("${id}::tile", id, 'tile',
            file("${root}/image.tif"), file("${root}/grid.csv"), file("${root}/metadata.json"),
            file("${root}/shift.json"), file("${root}/resolution.json"), file("${root}/support.tif"),
            file("${root}/parent.tif"), file("${root}/uncertainty.tif"), file("${root}/features"))
    }
    runtime_plan = TaskRuntime.create(HardwarePolicy.resolve(params, 1, 4, false, 0d), 'cpu')
    DISCOVER_TISSUE_HIERARCHY(specimens, runtime_plan)
    DISCOVER_TISSUE_HIERARCHY.out.summaries.view { key, id, variant, path -> "NATIVE_RESULT ${key} ${path.name}" }
}
''')
    env = {**os.environ, "NXF_OFFLINE": "true", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    env.pop("CELLPHENOTYPER_HIERARCHY_RSCRIPT", None)
    command = [executable, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
               "-params-file", str(params_path), "-ansi-log", "false", "-work-dir", str(tmp_path / "work"),
               "-with-trace", str(tmp_path / "trace.tsv")]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240)
    (tmp_path / "workflow_stdout.txt").write_text(result.stdout)
    (tmp_path / "workflow_stderr.txt").write_text(result.stderr)
    logs = "\n".join(path.read_text() for path in (tmp_path / "work").rglob("native.log"))
    assert result.returncode == 0, result.stdout + result.stderr + logs
    trace = pd.read_csv(tmp_path / "trace.tsv", sep="\t")
    assert len(trace) == 2
    assert trace.status.eq("COMPLETED").all()
    assert trace.name.str.startswith("DISCOVER_TISSUE_HIERARCHY (").all()
    assert trace.cpus.eq(1).all()
    summaries = []
    for fixture in fixtures:
        out = tmp_path / "output/22_tissue_hierarchy" / fixture["sample"] / "tile"
        summary = json.loads((out / "hierarchy_summary.json").read_text())
        # The module passes relative --outdir 'tile'. Temporary embedding
        # alignment arrays must not survive as dangling output receipt entries.
        verify_output_inventory(out, summary)
        assert summary["sample_id"] == fixture["sample"]
        assert summary["grid_observations"] == 63
        assert summary["discovery"]["method"] == "within_parent_native_kodama_graph"
        assert (out / "parent_domains.ome.tif").read_bytes() == (fixture["root"] / "parent.tif").read_bytes()
        assert (out / "parent_uncertainty.ome.tif").read_bytes() == (fixture["root"] / "uncertainty.tif").read_bytes()
        assert {path: digest(path) for path in fixture["hashes"]} == fixture["hashes"]
        for parent in summary["discovery"]["parent_domains"].values():
            params = parent["native_execution"]["runner"]["requested_parameters"]
            assert params["ncomp"] == 50 and params["cores"] == 1
        assert (out / "region_profiles/region_profiles_manifest.json").is_file()
        summaries.append({"sample_id": fixture["sample"], "hierarchy_summary_sha256": digest(out / "hierarchy_summary.json"),
                          "native_graph_sha256": summary["realized_identity"]["native_graph_sha256"],
                          "postprocess_output_inventory_verified": True, "durable_output_count": len(summary["outputs"])})
    write_json(tmp_path / "workflow_acceptance_summary.json", {"status": "pass", "specimens": summaries,
        "native_version": native_runtime["version"], "trace_sha256": digest(tmp_path / "trace.tsv"),
        "scope": "Actual two-specimen discovery module only; synthetic prepared features; no encoder or graph-cache reuse claim"})
