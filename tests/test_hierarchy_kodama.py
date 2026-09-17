"""Pure native-hierarchy selection/receipt checks; no R fitting or learned data."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import hierarchy_kodama as module
from cell_profile_io import sha256_file
from discover_tissue_hierarchy import STATUS, _transform_blocks


COLUMNS = ["representation", "resolution", "seed", "label", "cluster", "degree",
           "affinity_margin", "own_affinity_fraction"]
LABELS = ["001", "NA", "003", "004", "005", "006", "007", "008"]
BASE = np.array([1, 1, 1, 1, 2, 2, 2, 2])


def test_frozen_code_covers_recursive_static_local_python_imports():
    code_root = ROOT / "bin"
    declared = {(code_root / name).resolve() for name in module.CODE}
    assert len(declared) == len(module.CODE) and all(path.is_file() for path in declared)
    pending = [code_root / "hierarchy_kodama.py", code_root / "discover_tissue_hierarchy.py"]
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        assert path in declared, f"Executed local dependency is not frozen: {path.relative_to(code_root)}"
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else [entry.name for entry in node.names]
            else:
                continue
            for name in names:
                relative = Path(*name.split("."))
                candidates = [path.parent / relative.with_suffix(".py"),
                              code_root / relative.with_suffix(".py"),
                              path.parent / relative / "__init__.py",
                              code_root / relative / "__init__.py"]
                dependency = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
                if dependency is not None:
                    pending.append(dependency)
    assert {code_root / name for name in ("run_hierarchy_kodama.R", "kodama_graph_export.R",
                                         "kodama_graph_clustering.R")} <= declared


def partition_frame(labels=LABELS, clusters=BASE, *, representations=("combined", "local", "context"),
                    resolutions=(.3,), seeds=(17, 18, 19), margin=.6):
    clusters = np.asarray(clusters)
    records = []
    for representation in representations:
        for resolution in resolutions:
            for seed in seeds:
                for label, cluster in zip(labels, clusters):
                    records.append(dict(representation=representation, resolution=resolution,
                        seed=seed, label=label, cluster=int(cluster), degree=3 if cluster else 0,
                        affinity_margin=margin if cluster else np.nan,
                        own_affinity_fraction=(1 + margin) / 2 if cluster else np.nan))
    return pd.DataFrame(records, columns=COLUMNS)


def select(frame, labels=LABELS, **updates):
    options = dict(seed=17, repeats=2, max_k=3, fixed_k=None, min_observations=4,
                   min_seed_stability=1., min_scale_agreement=1., min_margin=.1)
    options.update(updates)
    return module.select_partition(frame, list(labels), **options)


def test_label_permutation_is_exact_evidence_not_label_number_identity():
    frame = partition_frame()
    mask = ~((frame.representation == "combined") & (frame.seed == 17))
    frame.loc[mask, "cluster"] = frame.loc[mask, "cluster"].map({1: 91, 2: 37})
    selected, report = select(frame)
    np.testing.assert_array_equal(selected["labels"], BASE)
    np.testing.assert_array_equal(selected["status"], np.ones(8))
    np.testing.assert_array_equal(selected["seed_stability"], np.ones(8))
    np.testing.assert_array_equal(selected["scale_agreement"], np.ones(8))
    assert report["selected_k"] == 2 and report["selected_resolution"] == .3
    assert report["candidates"][0]["mean_seed_ari"] == 1.


def test_unequal_k_uses_one_to_one_matching_without_merging_unmatched_clusters():
    votes, ari = module._align_votes([5, 5, 6, 6, 6, 7, 0, 7], [1, 1, 2, 2, 2, 2, 2, 0])
    np.testing.assert_array_equal(votes, [True, True, True, True, True, False, False, False])
    assert ari is not None and 0 < ari < 1
    reverse, _ = module._align_votes([1, 1, 1, 1, 2, 2], [7, 7, 8, 8, 9, 9])
    assert reverse.sum() == 4  # One predicted group cannot vote for two reference groups.


@pytest.mark.parametrize("prediction,reference", [([0, 0], [0, 0]), ([0, 0], [1, 1]), ([1, 1], [0, 0])])
def test_isolate_zero_never_supplies_alignment_evidence(prediction, reference):
    votes, ari = module._align_votes(prediction, reference)
    assert not votes.any() and ari is None


def test_single_nonisolated_overlap_has_vote_but_no_invented_ari():
    votes, ari = module._align_votes([42, 0], [7, 0])
    np.testing.assert_array_equal(votes, [True, False])
    assert ari is None


def test_native_isolates_remain_zero_with_missing_graph_evidence():
    clusters = np.r_[BASE[:-1], 0]
    selected, report = select(partition_frame(clusters=clusters))
    assert report["selected_k"] == 2 and len(selected["labels"]) == len(LABELS)
    assert selected["labels"][-1] == 0 and selected["status"][-1] == 11
    assert selected["graph_degree"][-1] == 0
    assert np.isnan(selected["affinity_margin"][-1]) and np.isnan(selected["own_affinity_fraction"][-1])
    assert selected["seed_stability"][-1] == selected["scale_agreement"][-1] == 0


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign", "reordered"])
def test_every_partition_preserves_exact_parent_population_and_order(mutation):
    frame = partition_frame()
    group = (frame.representation == "local") & (frame.seed == 18)
    positions = frame.index[group].to_numpy()
    if mutation == "missing":
        frame = frame.drop(positions[0])
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame.loc[[positions[0]]]], ignore_index=True)
    elif mutation == "foreign":
        frame.loc[positions[0], "label"] = "another_parent_same_position"
    else:
        frame.loc[positions[:2], "label"] = list(reversed(frame.loc[positions[:2], "label"].tolist()))
    with pytest.raises(ValueError, match="observations|population|order"):
        select(frame)


def test_seed_disagreement_is_unresolved_even_when_fixed_k_requested():
    frame = partition_frame()
    crossed = [1, 1, 2, 2, 1, 1, 2, 2]
    frame.loc[(frame.representation == "combined") & (frame.seed == 18), "cluster"] = crossed
    selected, _ = select(frame, fixed_k=2)
    unstable = selected["seed_stability"] < 1
    assert unstable.any() and (~unstable).any()
    assert np.all(selected["status"][unstable] == 7)
    np.testing.assert_array_equal(selected["labels"], BASE)


def test_field_disagreement_preserves_raw_discovery_but_not_acceptance():
    frame = partition_frame()
    for seed in (17, 18, 19):
        frame.loc[(frame.representation == "context") & (frame.seed == seed), "cluster"] = [1, 1, 2, 2, 1, 1, 2, 2]
    selected, _ = select(frame)
    disagreed = selected["scale_agreement"] < 1
    assert disagreed.any() and np.all(selected["status"][disagreed] == 8)
    np.testing.assert_array_equal(selected["labels"], BASE)


@pytest.mark.parametrize("field", ["local", "context"])
def test_constant_or_absent_field_contributes_zero_not_fabricated_agreement(field):
    frame = partition_frame(representations=tuple(name for name in ("combined", "local", "context") if name != field))
    selected, report = select(frame)
    np.testing.assert_array_equal(selected["status"], np.full(8, 8))
    np.testing.assert_array_equal(selected["scale_agreement"], np.full(8, .5))
    assert report["candidates"][0]["field_ari_by_seed"][field] == [None, None, None]


@pytest.mark.parametrize("clusters", [np.ones(8, int), np.zeros(8, int)])
def test_constant_or_all_isolated_partitions_never_invent_subdivision(clusters):
    selected, report = select(partition_frame(clusters=clusters), fixed_k=2)
    assert selected is None and report["selected_k"] == 0


def test_no_partitions_is_explicitly_unresolved():
    selected, report = select(pd.DataFrame(columns=COLUMNS))
    assert selected is None and report == {"selected_k": 0, "selected_resolution": None, "candidates": []}


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_native_partition_field_fails_closed(column):
    with pytest.raises(ValueError, match="required|evidence"):
        select(partition_frame().drop(columns=column))


@pytest.mark.parametrize("column,value", [
    ("cluster", -.1), ("cluster", 1.5), ("cluster", np.nan), ("cluster", np.inf),
    ("degree", -1), ("degree", .5), ("degree", np.nan), ("degree", np.inf),
    ("affinity_margin", np.nan), ("affinity_margin", np.inf), ("affinity_margin", -1.1), ("affinity_margin", 1.1),
    ("own_affinity_fraction", np.nan), ("own_affinity_fraction", np.inf),
    ("own_affinity_fraction", -.1), ("own_affinity_fraction", 1.1),
])
def test_invalid_supported_memberships_degrees_and_margins_fail(column, value):
    frame = partition_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="graph|membership|degree|evidence"):
        select(frame)


@pytest.mark.parametrize("cluster,degree,margin,own", [(0, 3, .6, .8), (1, 0, .6, .8), (0, 0, 0., np.nan), (0, 0, np.nan, .5)])
def test_isolates_cannot_have_fabricated_assignment_or_evidence(cluster, degree, margin, own):
    frame = partition_frame()
    frame.loc[0, ["cluster", "degree", "affinity_margin", "own_affinity_fraction"]] = [cluster, degree, margin, own]
    with pytest.raises(ValueError, match="graph|isolate|degree|evidence"):
        select(frame)


def test_simple_graph_degree_cannot_exceed_parent_population():
    frame = partition_frame()
    frame.loc[0, "degree"] = len(LABELS)
    with pytest.raises(ValueError, match="degree|graph"):
        select(frame)


def test_same_frozen_representation_graph_has_fixed_degrees_across_seeds():
    frame = partition_frame()
    frame.loc[(frame.representation == "combined") & (frame.seed == 18) & (frame.label == LABELS[0]), "degree"] = 4
    with pytest.raises(ValueError, match="degree|graph"):
        select(frame)


def test_same_frozen_representation_graph_has_fixed_degrees_across_resolutions():
    frame = partition_frame(resolutions=(.3, .5))
    frame.loc[(frame.representation == "context") & (frame.resolution == .5) & (frame.label == LABELS[0]), "degree"] = 4
    with pytest.raises(ValueError, match="degree|graph"):
        select(frame)


def test_missing_requested_combined_seed_fails_instead_of_reducing_denominator():
    frame = partition_frame()
    frame = frame[~((frame.representation == "combined") & (frame.seed == 19))]
    with pytest.raises(ValueError, match="seed|partition"):
        select(frame)


@pytest.mark.parametrize("clusters,fixed,max_k", [(np.ones(8, int), 2, 3), ([1, 1, 1, 2, 2, 2, 3, 3], 2, 3),
                                                  ([1, 1, 1, 2, 2, 2, 3, 3], None, 2)])
def test_k_bounds_filter_observed_partitions_without_splitting_or_merging(clusters, fixed, max_k):
    frame = partition_frame(clusters=clusters)
    original = frame.copy(deep=True)
    selected, report = select(frame, fixed_k=fixed, max_k=max_k)
    assert selected is None and report["selected_k"] == 0
    assert report["candidates"][0]["reason"] == "outside_requested_cluster_count_bounds"
    pd.testing.assert_frame_equal(frame, original)


def test_native_margin_is_not_replaced_by_centroid_evidence():
    frame = partition_frame(margin=.05)
    frame["centroid_margin"] = 1.
    selected, report = select(frame, fixed_k=2)
    assert selected is not None and np.all(selected["status"] == 12)
    assert "centroid_margin" not in selected
    np.testing.assert_array_equal(selected["affinity_margin"], np.full(8, .05))
    assert report["candidates"][0]["mean_affinity_margin"] == .05


def test_automatic_selection_rejects_tiny_supported_groups_even_when_stable():
    selected, report = select(partition_frame(clusters=[1, 1, 1, 1, 1, 1, 1, 2]))
    assert selected is None and not report["candidates"][0]["eligible"]


def test_all_constant_feature_consumer_preserves_full_grid_unresolved(tmp_path, monkeypatch):
    grid = pd.DataFrame({"label": LABELS, "x": np.arange(8), "y": np.zeros(8),
                         "parent_domain_id": 1, "status_code": 2})
    seen = []
    def native(manifest, output, code, library, cores):
        record = json.loads(manifest.read_text())
        assert all(entry["constant"] for entry in record["representations"].values())
        seen.append(record)
        _, source_hashes = module._manifest_sources(manifest)
        receipt = manifest.parent / "synthetic_no_fit.json"
        receipt.write_text('{"synthetic_no_fit":true}')
        execution = {"synthetic_no_fit": True, "input_sha256": source_hashes,
            "producer_sha256": {str(code / "run_hierarchy_kodama.R"): sha256_file(code / "run_hierarchy_kodama.R")},
            "artifact_sha256": {str(receipt): sha256_file(receipt)}}
        return pd.DataFrame(columns=COLUMNS), execution
    monkeypatch.setattr(module, "_run_native", native)
    result, report = module.discover_kodama_subdomains(grid,
        {"local": np.ones((8, 3)), "context": np.ones((8, 4))},
        outdir=tmp_path / "discovery", transform_blocks=_transform_blocks, status_names=STATUS,
        source_hashes={}, geometry={"mpp_xy": [.5, .5], "origin_um_xy": [10., 20.]},
        min_observations=4, repeats=2, fit_limit=8, components=3)
    assert len(seen) == 1 and report["parent_domains"]["1"]["selected_k"] == 0
    pd.testing.assert_frame_equal(result[grid.columns[:-1]], grid[grid.columns[:-1]])
    assert result.status_code.eq(6).all() and result.subdomain_id.eq(0).all()
    assert result.affinity_margin.isna().all() and "centroid_margin" not in result
    assert report["seed_scope"] == "Clustering seeds on frozen native graphs; graph-building variability is separate"


def synthetic_partition_runner(monkeypatch, clusters, *, omit_context=False):
    """Test the Python handoff only; these are not native KODAMA results."""
    seen = []
    def native(manifest, output, code, library, cores):
        record, source_hashes = module._manifest_sources(manifest)
        rows = pd.read_csv(manifest.parent / record["rows"]["path"],
                           dtype={"label": str}, keep_default_na=False)
        representations = [name for name, entry in record["representations"].items()
                           if not entry["constant"] and not (omit_context and name == "context")]
        frame = partition_frame(rows.label.tolist(), clusters, representations=representations,
            resolutions=record["parameters"]["resolutions"],
            seeds=record["parameters"]["cluster_seeds"])
        frame.loc[frame.cluster > 0, "degree"] = min(3, np.count_nonzero(clusters) - 1)
        output.mkdir()
        receipt = output / "synthetic_partition_fixture.json"
        receipt.write_text('{"synthetic_partitions_only":true,"native_fit_executed":false}')
        execution = {"synthetic_partitions_only": True, "input_sha256": source_hashes,
            "producer_sha256": {str(code / "run_hierarchy_kodama.R"): sha256_file(code / "run_hierarchy_kodama.R")},
            "artifact_sha256": {str(receipt): sha256_file(receipt)}}
        seen.append((record, rows))
        return frame, execution
    monkeypatch.setattr(module, "_run_native", native)
    return seen


def test_outer_discovery_preserves_all_excluded_and_missing_rows_and_physical_audit(tmp_path, monkeypatch):
    labels = LABELS + ["009"]
    grid = pd.DataFrame({"label": labels, "x": np.arange(9) * 10, "y": np.arange(9) * 2,
        "parent_domain_id": [1, 1, 1, 1, 1, 0, 1, 1, 2],
        "status_code": [2, 2, 2, 2, 2, 0, 3, 10, 2]})
    original = grid.copy(deep=True)
    local = np.arange(27, dtype=float).reshape(9, 3)
    local[4, 1] = np.nan
    context = np.square(np.arange(27, dtype=float).reshape(9, 3))
    seen = synthetic_partition_runner(monkeypatch, [1, 1, 2, 2])
    result, report = module.discover_kodama_subdomains(grid, {"local": local, "context": context},
        outdir=tmp_path / "discovery", transform_blocks=_transform_blocks, status_names=STATUS,
        source_hashes={}, geometry={"mpp_xy": [.5, .25], "origin_um_xy": [10., 20.]},
        min_observations=4, repeats=2, fit_limit=4, components=3, ncomp=50)
    pd.testing.assert_frame_equal(grid, original)
    assert result.label.tolist() == labels
    np.testing.assert_array_equal(result.status_code, [1, 1, 1, 1, 4, 0, 3, 10, 5])
    np.testing.assert_array_equal(result.subdomain_id, [1, 1, 2, 2, 0, 0, 0, 0, 0])
    assert result.loc[4:, "graph_degree"].isna().all()
    assert result.loc[4:, "affinity_margin"].isna().all()
    assert "centroid_margin" not in result
    assert report["requested_kodama_ncomp"] == 50 and report["pca_components_per_block"] == 3
    assert report["parent_domains"]["1"]["native_fit_observations"] == 4
    assert report["parent_domains"]["2"]["observations"] == 1
    assert len(seen) == 1
    manifest, fitted_rows = seen[0]
    assert fitted_rows.label.tolist() == labels[:4]
    np.testing.assert_array_equal(fitted_rows[["x", "y"]],
        grid.loc[:3, ["x", "y"]].to_numpy() * [.5, .25] + [10., 20.])
    assert manifest["scope"]["source_row_indices"] == [0, 1, 2, 3]
    assert manifest["scope"]["coordinates_used_for_fitting"] is False
    assert manifest["parameters"]["ncomp"] == 50


def test_outer_no_candidate_keeps_isolate_status_and_actual_degree(tmp_path, monkeypatch):
    grid = pd.DataFrame({"label": LABELS[:4], "x": np.arange(4), "y": np.zeros(4),
                         "parent_domain_id": 1, "status_code": 2})
    synthetic_partition_runner(monkeypatch, [1, 1, 1, 0])
    result, report = module.discover_kodama_subdomains(grid,
        {"local": np.arange(12.).reshape(4, 3), "context": np.square(np.arange(12.).reshape(4, 3))},
        outdir=tmp_path / "discovery", transform_blocks=_transform_blocks, status_names=STATUS,
        source_hashes={}, geometry={"mpp_xy": [.5, .5], "origin_um_xy": [0., 0.]},
        min_observations=4, repeats=2, fit_limit=4, components=3, fixed_k=2)
    assert result.label.tolist() == LABELS[:4]
    np.testing.assert_array_equal(result.status_code, [6, 6, 6, 11])
    np.testing.assert_array_equal(result.graph_degree, [2, 2, 2, 0])
    assert result.subdomain_id.eq(0).all() and result.local_subdomain_id.eq(0).all()
    assert result.affinity_margin.isna().all()
    assert report["parent_domains"]["1"]["selected_k"] == 0


def test_outer_nonconstant_representation_cannot_silently_disappear(tmp_path, monkeypatch):
    grid = pd.DataFrame({"label": LABELS[:4], "x": np.arange(4), "y": np.zeros(4),
                         "parent_domain_id": 1, "status_code": 2})
    synthetic_partition_runner(monkeypatch, [1, 1, 2, 2], omit_context=True)
    with pytest.raises(ValueError, match="omitted or added.*population"):
        module.discover_kodama_subdomains(grid,
            {"local": np.arange(12.).reshape(4, 3), "context": np.square(np.arange(12.).reshape(4, 3))},
            outdir=tmp_path / "discovery", transform_blocks=_transform_blocks, status_names=STATUS,
            source_hashes={}, geometry={"mpp_xy": [.5, .5], "origin_um_xy": [0., 0.]},
            min_observations=4, repeats=2, fit_limit=4, components=3)
    receipt = json.loads((tmp_path / "discovery" / "discovery_status.json").read_text())
    assert receipt["status"] == "failed"


def runner_fixture(tmp_path, monkeypatch, corruption=None, *, calls=None):
    monkeypatch.delenv("CELLPHENOTYPER_HIERARCHY_RSCRIPT", raising=False)
    code = tmp_path / "code"
    code.mkdir()
    for name in ("run_hierarchy_kodama.R", "kodama_graph_export.R", "kodama_graph_clustering.R"):
        (code / name).write_text("# synthetic command-recorder fixture, never executed\n")
    rows = tmp_path / "rows.csv"
    pd.DataFrame({"label": LABELS, "x": np.arange(8), "y": np.zeros(8)}).to_csv(rows, index=False)
    representations = {}
    for name in ("local", "context", "combined"):
        path = tmp_path / f"{name}.f64"
        np.arange(16, dtype="<f8").tofile(path)
        representations[name] = {"path": path.name, "sha256": sha256_file(path), "shape": [8, 2], "constant": False}
    manifest = tmp_path / "input_manifest.json"
    record = {"format": "cellphenotyper_hierarchy_kodama_input", "schema_version": "1.0.0",
        "rows": {"path": rows.name, "sha256": sha256_file(rows)}, "representations": representations,
        "parameters": {"ncomp": 50, "M": 8, "Tcycle": 6, "landmarks": 8, "cores": 1,
            "neighbors": 3, "seed": 17, "cluster_seeds": [17, 18, 19], "resolutions": [.3]},
        "scope": {"parent_id": 1, "source_row_indices": list(range(8)),
            "source_inputs_sha256": {str(rows.resolve()): sha256_file(rows)},
            "pca_training_row_indices": [0, 2, 4, 6],
            "normalization": {"context": {"mean": [1.6618342202156782, 0., 1.e-100],
                                          "scale": [1., 2., 3.]}},
            "coordinate_space": "original_slide_micrometres", "coordinates_used_for_fitting": False}}
    manifest.write_text(json.dumps(record))
    output = tmp_path / "native"
    runtime = tmp_path / "synthetic_runtime.txt"
    runtime.write_text("Synthetic receipt fixture only; no native runtime executed.\n")
    runtime_identity = {str(runtime.resolve()): sha256_file(runtime)}
    monkeypatch.setattr(module.shutil, "which", lambda name: "/synthetic/Rscript")
    def run(command, **kwargs):
        if calls is not None:
            calls.append((list(command), dict(kwargs["env"])))
        assert command[1] == "--vanilla"
        assert command[2] == str(code / "run_hierarchy_kodama.R")
        assert command[3] == str(manifest) and command[4] == str(output)
        assert command[5] == sha256_file(manifest)
        output.mkdir()
        frame = partition_frame(clusters=np.r_[BASE[:-1], 0])
        frame.to_csv(output / "partitions.csv", index=False, na_rep="NA")
        paths = [manifest, rows] + [tmp_path / entry["path"] for entry in representations.values()]
        sources = {str(path.resolve()): sha256_file(path) for path in paths}
        producer = {str(path.resolve()): sha256_file(path) for path in code.iterdir()}
        summary = {"format": "cellphenotyper_hierarchy_kodama_runner", "schema_version": "1.0.0",
            "status": "pass", "sources_before": sources, "sources_after": dict(sources),
            "producer_before": producer, "producer_after": dict(producer),
            "observation_count": 8, "partition_run_count": 9, "partition_rows": len(frame),
            "requested_parameters": record["parameters"], "feature_protocol": "raw_data_native_retained_graph",
            "scope": json.loads(json.dumps(record["scope"])),
            "runtime_before": runtime_identity, "runtime_after": dict(runtime_identity),
            "native_package_before_load": runtime_identity, "native_package_after": dict(runtime_identity),
            "output_sha256": {"partitions.csv": sha256_file(output / "partitions.csv")}}
        if corruption == "foreign_manifest":
            summary["sources_before"][str(manifest.resolve())] = "f" * 64
            summary["sources_after"] = dict(summary["sources_before"])
        elif corruption == "missing_source_receipt":
            del summary["sources_before"]
            del summary["sources_after"]
        elif corruption == "changed_source_payload":
            (tmp_path / "local.f64").write_bytes(b"changed after native consumption")
        elif corruption == "changed_source_after_map":
            summary["sources_after"][str(rows.resolve())] = "f" * 64
        elif corruption == "wrong_protocol":
            summary["feature_protocol"] = "graph_only_invented_features"
        elif corruption == "scope_float_roundoff":
            summary["scope"]["normalization"]["context"]["mean"][0] = 1.66183422021568
        elif corruption == "scope_integer":
            summary["scope"]["parent_id"] = 2
        elif corruption == "scope_integer_type":
            summary["scope"]["parent_id"] = 1.0
        elif corruption == "scope_integer_boolean":
            summary["scope"]["parent_id"] = True
        elif corruption == "scope_order":
            summary["scope"]["source_row_indices"] = list(reversed(record["scope"]["source_row_indices"]))
        elif corruption == "scope_hash":
            summary["scope"]["source_inputs_sha256"][str(rows.resolve())] = "f" * 64
        elif corruption == "scope_significant_float":
            summary["scope"]["normalization"]["context"]["mean"][0] += 1.e-12
        elif corruption == "scope_zero_absolute_difference":
            summary["scope"]["normalization"]["context"]["mean"][1] = 1.e-100
        elif corruption == "scope_tiny_relative_difference":
            summary["scope"]["normalization"]["context"]["mean"][2] = 1.000000001e-100
        elif corruption == "scope_missing_key":
            del summary["scope"]["coordinate_space"]
        elif corruption == "scope_extra_key":
            summary["scope"]["foreign_field"] = "unrequested"
        (output / "runner_summary.json").write_text(json.dumps(summary))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    return manifest, output, code


def test_native_receipt_consumer_keeps_literal_na_id_and_nullable_isolate_evidence(tmp_path, monkeypatch):
    manifest, output, code = runner_fixture(tmp_path, monkeypatch)
    frame, execution = module._run_native(manifest, output, code, None, 1)
    assert frame.label.iloc[:8].tolist() == LABELS
    assert frame.loc[frame.cluster == 0, "affinity_margin"].isna().all()
    assert frame.loc[frame.cluster > 0, "affinity_margin"].notna().all()
    assert execution["runner"]["feature_protocol"] == "raw_data_native_retained_graph"


def test_descriptive_scope_float_echo_roundoff_preserves_exact_original_manifest(tmp_path, monkeypatch):
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, "scope_float_roundoff")
    original_bytes, original_hash = manifest.read_bytes(), sha256_file(manifest)
    original = json.loads(original_bytes)
    _, execution = module._run_native(manifest, output, code, None, 1)
    assert execution["input_manifest"] == original
    assert execution["input_manifest"]["scope"]["normalization"]["context"]["mean"][0] == 1.6618342202156782
    assert execution["runner"]["scope"]["normalization"]["context"]["mean"][0] == 1.66183422021568
    assert execution["runner"]["scope"] != execution["input_manifest"]["scope"]
    assert manifest.read_bytes() == original_bytes
    assert execution["input_sha256"][str(manifest.resolve())] == original_hash
    assert execution["runner"]["sources_before"][str(manifest.resolve())] == original_hash
    assert execution["runner"]["sources_after"][str(manifest.resolve())] == original_hash
    assert "relative tolerance 1e-14, zero absolute tolerance" in execution["scope_echo_precision"]
    assert "IDs/integers/hashes remain exact" in execution["scope_echo_precision"]


@pytest.mark.parametrize("corruption", ["scope_integer", "scope_integer_type", "scope_integer_boolean",
    "scope_order", "scope_hash", "scope_significant_float", "scope_zero_absolute_difference",
    "scope_tiny_relative_difference", "scope_missing_key", "scope_extra_key"])
def test_descriptive_scope_tolerance_never_relaxes_ids_order_hashes_or_structure(tmp_path, monkeypatch, corruption):
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, corruption)
    with pytest.raises(ValueError, match="bind.*input/producer/protocol"):
        module._run_native(manifest, output, code, None, 1)


@pytest.mark.parametrize("actual,expected", [(float("nan"), 1.), (float("inf"), 1.), (1., float("nan")),
    (1., float("inf")), (True, 1.), ("1.0", 1.), ("001", "1"), ([2, 1], [1, 2]),
    ((1, 2), [1, 2]), (-1.e-100, 1.e-100)])
def test_scope_echo_rejects_nonfinite_coercions_and_changed_literal_identity(actual, expected):
    assert not module._scope_echo_matches({"nested": actual}, {"nested": expected})


def test_scope_echo_accepts_native_integer_encoding_of_an_expected_float():
    assert module._scope_echo_matches({"mean": [1, 0]}, {"mean": [1., 0.]})


@pytest.mark.parametrize("corruption", ["foreign_manifest", "missing_source_receipt", "changed_source_payload",
                                      "changed_source_after_map", "wrong_protocol"])
def test_native_receipt_consumer_rejects_unbound_or_changed_result(tmp_path, monkeypatch, corruption):
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, corruption)
    with pytest.raises(ValueError, match="source|input|manifest|protocol|receipt|identity"):
        module._run_native(manifest, output, code, None, 1)


@pytest.mark.parametrize("relative", ["input_manifest.json", "rows.csv", "local.f64", "native/partitions.csv",
                                     "native/runner_summary.json", "code/run_hierarchy_kodama.R"])
def test_consumed_native_inputs_outputs_and_summary_remain_pinned_after_readback(tmp_path, monkeypatch, relative):
    manifest, output, code = runner_fixture(tmp_path, monkeypatch)
    _, execution = module._run_native(manifest, output, code, None, 1)
    (tmp_path / relative).write_bytes(b"changed after successful consumer readback")
    with pytest.raises(ValueError, match="identity"):
        module._verify_native_execution(execution)


def synthetic_executable(prefix):
    executable = prefix / "bin" / "Rscript"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n# Synthetic selection fixture: never executed.\nexit 99\n")
    executable.chmod(0o700)
    return executable


@pytest.mark.parametrize("invalid", ["empty", "relative", "missing", "directory", "not_executable",
                                     "newline", "tab", "delete_character"])
def test_invalid_explicit_native_runtime_fails_without_path_fallback(tmp_path, monkeypatch, invalid):
    calls = []
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, calls=calls)
    executable = synthetic_executable(tmp_path / "explicit_runtime")
    values = {"empty": "", "relative": "bin/Rscript", "missing": str(tmp_path / "missing"),
              "directory": str(executable.parent), "not_executable": str(executable)}
    if invalid in ("newline", "tab", "delete_character"):
        control = {"newline": "\n", "tab": "\t", "delete_character": "\x7f"}[invalid]
        executable = executable.with_name("Rscript" + control)
        executable.write_text("# Existing executable with a disallowed control character in its name.\n")
        executable.chmod(0o700)
        value = str(executable)
    else:
        value = values[invalid]
    if invalid == "not_executable":
        executable.chmod(0o600)
    monkeypatch.setenv("CELLPHENOTYPER_HIERARCHY_RSCRIPT", value)
    def unexpected_fallback(name):
        pytest.fail("An invalid explicit native runtime must not consult PATH")
    monkeypatch.setattr(module.shutil, "which", unexpected_fallback)
    with pytest.raises(ValueError, match="absolute executable; no fallback"):
        module._run_native(manifest, output, code, None, 1)
    assert not calls and not output.exists()


@pytest.mark.parametrize("with_native_lib", [False, True])
@pytest.mark.parametrize("via_symlink", [False, True])
def test_explicit_runtime_and_native_library_environment_are_isolated(tmp_path, monkeypatch,
                                                                    with_native_lib, via_symlink):
    calls = []
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, calls=calls)
    prefix = tmp_path / "selected native runtime"
    executable = synthetic_executable(prefix)
    native_lib = prefix / "lib"
    if with_native_lib:
        native_lib.mkdir()
    selected = executable
    if via_symlink:
        selected = tmp_path / "shortcut" / "bin" / "Rscript"
        selected.parent.mkdir(parents=True)
        selected.symlink_to(executable)
        # A symlink's unrelated prefix must not supply the selected runtime's libraries.
        (tmp_path / "shortcut" / "lib").mkdir()
    monkeypatch.setenv("CELLPHENOTYPER_HIERARCHY_RSCRIPT", str(selected))
    inherited = {name: f"/unrelated_python_stack/{name}" for name in (
        "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES")}
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("SYNTHETIC_ORDINARY_SETTING", "preserved")
    def unexpected_fallback(name):
        pytest.fail("An explicit executable must take precedence over PATH")
    monkeypatch.setattr(module.shutil, "which", unexpected_fallback)
    _, execution = module._run_native(manifest, output, code, None, 1)
    assert len(calls) == 1
    command, environment = calls[0]
    assert command[0] == str(executable.resolve())
    assert execution["command"] == command
    assert environment.get("LD_LIBRARY_PATH") == (str(native_lib) if with_native_lib else None)
    assert not set(("LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES")) & set(environment)
    assert environment["SYNTHETIC_ORDINARY_SETTING"] == "preserved"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert environment[name] == "1"
    # Isolation is confined to the child environment, not the caller's process.
    assert {name: module.os.environ[name] for name in inherited} == inherited


def test_unconfigured_native_runtime_uses_path_once(tmp_path, monkeypatch):
    calls, lookups = [], []
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, calls=calls)
    executable = synthetic_executable(tmp_path / "path_runtime")
    def lookup(name):
        lookups.append(name)
        return str(executable)
    monkeypatch.setattr(module.shutil, "which", lookup)
    module._run_native(manifest, output, code, None, 1)
    assert lookups == ["Rscript"] and calls[0][0][0] == str(executable)


def test_unconfigured_missing_runtime_does_not_install_or_start_process(tmp_path, monkeypatch):
    calls = []
    manifest, output, code = runner_fixture(tmp_path, monkeypatch, calls=calls)
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="installed Rscript; no automatic installation"):
        module._run_native(manifest, output, code, None, 1)
    assert not calls and not output.exists()
