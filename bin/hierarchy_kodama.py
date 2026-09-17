"""Source-bound, within-parent KODAMA discovery and graph-specific evidence.

The parent raster is not an input to a learned fitting operation. Its already
verified grid assignments select each parent population; no parent is changed.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from cell_profile_io import sha256_file


RESOLUTIONS = (.1, .2, .3, .5, .75, 1., 1.5, 2.)
CODE = ("hierarchy_kodama.py", "discover_tissue_hierarchy.py", "run_hierarchy_kodama.R",
        "kodama_graph_export.R", "kodama_graph_clustering.R", "build_cell_profiles.py",
        "cell_morphology_io.py", "cell_phenotype_io.py", "cell_profile_io.py",
        "cellvit_embedding_io.py", "ome_tiff_metadata.py", "profile_cell_morphology.py",
        "uni2_embedding_io.py")


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _align_votes(prediction, reference):
    """Match arbitrary, possibly unequal cluster sets; 0 is never evidence."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    valid = (prediction > 0) & (reference > 0)
    votes = np.zeros(len(reference), bool)
    if not valid.any():
        return votes, None
    pvalues, pi = np.unique(prediction[valid], return_inverse=True)
    rvalues, ri = np.unique(reference[valid], return_inverse=True)
    counts = np.zeros((len(pvalues), len(rvalues)), dtype=np.int64)
    np.add.at(counts, (pi, ri), 1)
    matched_p, matched_r = linear_sum_assignment(-counts)
    for p, r in zip(pvalues[matched_p], rvalues[matched_r]):
        votes |= valid & (prediction == p) & (reference == r)
    ari = float(adjusted_rand_score(reference[valid], prediction[valid])) if valid.sum() >= 2 else None
    return votes, ari


def select_partition(partitions, labels, *, seed, repeats, max_k, fixed_k,
                     min_observations, min_seed_stability, min_scale_agreement, min_margin):
    """Select graph resolutions with explicit seed/field uncertainty, never KMeans."""
    required = {"representation", "resolution", "seed", "label", "cluster", "degree", "affinity_margin", "own_affinity_fraction"}
    if not required <= set(partitions):
        raise ValueError("Native hierarchy partitions lack required graph evidence")
    groups, graph_degrees = {}, {}
    for (representation, resolution, run_seed), frame in partitions.groupby(["representation", "resolution", "seed"], sort=False):
        if frame.label.tolist() != labels:
            raise ValueError("Native graph partitions lost or reordered parent observations")
        cluster = frame.cluster.to_numpy()
        degree = frame.degree.to_numpy()
        if (not np.isfinite(cluster).all() or not np.equal(cluster, np.floor(cluster)).all()
                or (cluster < 0).any() or not np.isfinite(degree).all()
                or not np.equal(degree, np.floor(degree)).all() or (degree < 0).any()
                or (degree >= len(labels)).any()
                or not np.array_equal(cluster == 0, degree == 0)):
            raise ValueError("Invalid graph memberships/degrees or fabricated isolate assignment")
        if representation in graph_degrees and not np.array_equal(degree, graph_degrees[representation]):
            raise ValueError("Native graph degrees changed between partitions of the same representation")
        graph_degrees[representation] = degree
        for column, lower, upper in (("affinity_margin", -1., 1.), ("own_affinity_fraction", 0., 1.)):
            value = frame[column].to_numpy(float)
            if not np.isnan(value[degree == 0]).all() or not np.isfinite(value[degree > 0]).all() or (value[degree > 0] < lower - 1e-12).any() or (value[degree > 0] > upper + 1e-12).any():
                raise ValueError("Native graph evidence contradicts degree/support")
        groups[representation, float(resolution), int(run_seed)] = frame
    report = {"selected_k": 0, "selected_resolution": None, "candidates": []}
    selected, best = None, -math.inf
    for resolution in RESOLUTIONS:
        base = groups.get(("combined", resolution, seed))
        if base is None:
            continue
        reference = base.cluster.to_numpy(dtype=int)
        positive = reference > 0
        values, counts = np.unique(reference[positive], return_counts=True)
        k = len(values)
        if k < 2 or k > max_k or (fixed_k is not None and k != fixed_k):
            report["candidates"].append({"resolution": resolution, "k": k, "eligible": False, "reason": "outside_requested_cluster_count_bounds"})
            continue
        votes = np.zeros(len(labels), float)
        aris = []
        for repeat_seed in range(seed + 1, seed + repeats + 1):
            frame = groups.get(("combined", resolution, repeat_seed))
            if frame is None:
                raise ValueError("Native graph omitted a requested repeated-seed partition")
            agreement, ari = _align_votes(frame.cluster.to_numpy(), reference)
            votes += agreement
            aris.append(ari)
        field_votes = np.zeros(len(labels), float)
        field_aris = {}
        for name in ("local", "context"):
            scores = []
            for run_seed in range(seed, seed + repeats + 1):
                frame = groups.get((name, resolution, run_seed))
                if frame is None:
                    scores.append(None)  # A constant field supplies no evidence.
                    continue
                agreement, ari = _align_votes(frame.cluster.to_numpy(), reference)
                field_votes += agreement
                scores.append(ari)
            field_aris[name] = scores
        finite_aris = [value for value in aris if value is not None]
        score_ari = float(np.mean(finite_aris)) if finite_aris else 0.
        margin = base.affinity_margin.to_numpy(float)
        separation = float(np.mean(margin[positive]))
        smallest = int(counts.min())
        eligible = fixed_k is not None or (score_ari >= .75 and separation >= .1 and smallest >= max(2, min_observations // 2))
        score = score_ari * max(0., separation) - .01 * k
        report["candidates"].append({"resolution": resolution, "k": k, "eligible": eligible,
            "mean_seed_ari": score_ari, "mean_affinity_margin": separation,
            "smallest_supported_cluster": smallest, "field_ari_by_seed": field_aris, "score": score})
        if eligible and score > best:
            seed_stability = votes / repeats
            field_agreement = field_votes / (2 * (repeats + 1))
            status = np.ones(len(labels), dtype=int)
            status[~np.isfinite(margin) | (margin < min_margin)] = 12
            status[field_agreement < min_scale_agreement] = 8
            status[seed_stability < min_seed_stability] = 7
            status[~positive] = 11
            selected = {"labels": reference, "status": status, "affinity_margin": margin,
                "seed_stability": seed_stability, "scale_agreement": field_agreement,
                "graph_degree": base.degree.to_numpy(int),
                "own_affinity_fraction": base.own_affinity_fraction.to_numpy(float)}
            report.update(selected_k=k, selected_resolution=resolution)
            best = score
    return selected, report


def _verify_hashes(expected, kind):
    if not isinstance(expected, dict) or not expected:
        raise ValueError(f"Native hierarchy lacks {kind} identity")
    for path, digest in expected.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
            raise ValueError(f"Native hierarchy {kind} identity differs")


def _manifest_sources(manifest_path):
    """Bind generated native inputs before dispatch, not just the result labels."""
    manifest_path = Path(manifest_path).resolve()
    digest = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected = {str(manifest_path): digest}
    for record in [manifest["rows"], *manifest["representations"].values()]:
        relative = Path(record["path"])
        path = (manifest_path.parent / relative).resolve()
        if (relative.is_absolute() or len(relative.parts) != 1 or relative.name in (".", "..")
                or not path.is_relative_to(manifest_path.parent) or str(path) in expected):
            raise ValueError("Native hierarchy input path is not a distinct contained artifact")
        expected[str(path)] = record["sha256"]
    _verify_hashes(expected, "input source")
    return manifest, expected


def _verify_native_execution(execution):
    """Recheck retained producer inputs and outputs after Python consumption."""
    for field in ("input_sha256", "producer_sha256", "artifact_sha256"):
        _verify_hashes(execution[field], field)


def verify_discovery_artifacts(report, outdir):
    """Bind the Python dependency closure as well as each native producer."""
    outdir = Path(outdir)
    code = Path(__file__).resolve().parent
    for root in (code, outdir / "code"):
        _verify_hashes({str(root / name): digest for name, digest in report["code_sha256"].items()}, "discovery code")
    for parent in report["parent_domains"].values():
        if "native_execution" in parent:
            _verify_native_execution(parent["native_execution"])


def _scope_echo_matches(actual, expected):
    """Check jsonlite's descriptive scope echo without using it as input identity.

    Native jsonlite rounds decimal floats; the exact original manifest bytes
    remain bound by sources_before/after and are retained by Python separately.
    No tolerance applies to IDs, hashes, integer fields, keys or list ordering.
    """
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            _scope_echo_matches(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _scope_echo_matches(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected) and math.isclose(
            actual, expected, rel_tol=1e-14, abs_tol=0.)
    return type(actual) is type(expected) and actual == expected


def _run_native(manifest_path, output, code_directory, library, cores):
    manifest_path, output, code_directory = Path(manifest_path).resolve(), Path(output).resolve(), Path(code_directory).resolve()
    manifest, source_hashes = _manifest_sources(manifest_path)
    producer_hashes = {str(code_directory / name): sha256_file(code_directory / name)
                       for name in ("run_hierarchy_kodama.R", "kodama_graph_export.R", "kodama_graph_clustering.R")}
    configured_executable = os.environ.get("CELLPHENOTYPER_HIERARCHY_RSCRIPT")
    if configured_executable is not None:
        selected_executable = Path(configured_executable)
        if (not configured_executable or not selected_executable.is_absolute()
                or any(ord(char) < 32 or ord(char) == 127 for char in configured_executable)
                or not selected_executable.is_file() or not os.access(selected_executable, os.X_OK)):
            raise ValueError("Configured native hierarchy Rscript must be an existing absolute executable; no fallback")
        executable = str(selected_executable.resolve())
    else:
        executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Native hierarchy requires an installed Rscript; no automatic installation")
    command = [executable, "--vanilla", str(code_directory / "run_hierarchy_kodama.R"), str(manifest_path),
               str(output), source_hashes[str(manifest_path)]]
    if library is not None:
        library = Path(library).resolve()
        if not library.is_dir():
            raise ValueError("Explicit native KODAMA R library does not exist")
        command.append(str(library))
    env = dict(os.environ, OMP_NUM_THREADS=str(cores), OPENBLAS_NUM_THREADS=str(cores),
               MKL_NUM_THREADS=str(cores), VECLIB_MAXIMUM_THREADS=str(cores))
    # The isolated atlas Python launcher exposes its own lib directory. It
    # must not inject that Python/BLAS stack into a different native R runtime.
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"):
        env.pop(variable, None)
    native_lib = Path(executable).resolve().parent.parent / "lib"
    if native_lib.is_dir():
        env["LD_LIBRARY_PATH"] = str(native_lib)
    started = time.monotonic()
    with (manifest_path.parent / "native.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Native KODAMA hierarchy failed; inspect {manifest_path.parent / 'native.log'}")
    summary_path = output / "runner_summary.json"
    digest = sha256_file(summary_path)
    summary = json.loads(summary_path.read_text())
    if sha256_file(summary_path) != digest or summary.get("status") != "pass":
        raise ValueError("Native hierarchy runner lacks an unchanged pass receipt")
    if (summary.get("format") != "cellphenotyper_hierarchy_kodama_runner"
            or summary.get("schema_version") != "1.0.0"
            or summary.get("sources_before") != source_hashes
            or summary.get("sources_after") != source_hashes
            or summary.get("producer_before") != producer_hashes
            or summary.get("producer_after") != producer_hashes
            or summary.get("requested_parameters") != manifest["parameters"]
            or not _scope_echo_matches(summary.get("scope"), manifest.get("scope"))
            or summary.get("feature_protocol") != "raw_data_native_retained_graph"):
        raise ValueError("Native hierarchy receipt does not bind the requested input/producer/protocol")
    for before, after in (("runtime_before", "runtime_after"), ("native_package_before_load", "native_package_after")):
        if not summary.get(before) or summary[before] != summary.get(after):
            raise ValueError("Native hierarchy receipt reports a changed or absent runtime identity")
    output_hashes = summary.get("output_sha256", {})
    if "partitions.csv" not in output_hashes:
        raise ValueError("Native hierarchy runner omitted partition identity")
    artifact_hashes = {str(summary_path): digest}
    for relative, expected in output_hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not (output / path).resolve().is_relative_to(output.resolve()) or sha256_file(output / path) != expected:
            raise ValueError("Native hierarchy output identity differs")
        artifact_hashes[str(output / path)] = expected
    execution = {"command": command, "summary_sha256": digest, "runner": summary,
                 "input_sha256": source_hashes, "producer_sha256": producer_hashes,
                 "artifact_sha256": artifact_hashes, "input_manifest": manifest,
                 "scope_echo_precision": "Native JSON float echo is descriptive (relative tolerance 1e-14, zero absolute tolerance); exact input identity is manifest SHA256 before/after and original Python-parsed input_manifest; IDs/integers/hashes remain exact"}
    _verify_native_execution(execution)
    frame = pd.read_csv(output / "partitions.csv", dtype={"label": str}, keep_default_na=False, float_precision="round_trip")
    for column in ("affinity_margin", "own_affinity_fraction"):
        frame[column] = pd.to_numeric(frame[column].replace({"NA": np.nan, "": np.nan}), errors="raise")
    _verify_native_execution(execution)
    execution["wall_seconds"] = time.monotonic() - started
    return frame, execution


def discover_kodama_subdomains(grid, blocks, *, outdir, transform_blocks, status_names, source_hashes,
        geometry, weights=None, max_k=5, fixed_k=None, seed=17, repeats=5, fit_limit=5000,
        components=64, min_observations=20, min_seed_stability=.8, min_scale_agreement=1.,
        min_margin=.1, ncomp=50, m=100, tcycle=20, neighbors=100, cores=1, library=None):
    if set(blocks) != {"local", "context"}:
        raise ValueError("KODAMA hierarchy requires local and context blocks")
    weights = weights or {"local": 1., "context": 1.}
    if set(weights) != set(blocks) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Positive local/context weights required")
    if min(repeats, max_k, min_observations) < 2 or fit_limit < min_observations or components < 3 or min(ncomp, m, tcycle, neighbors, cores) < 1 or seed < 0 or seed + repeats > 2147483647:
        raise ValueError("Invalid KODAMA hierarchy bounds or seeds")
    if fixed_k is not None and not 2 <= fixed_k <= max_k:
        raise ValueError("fixed_k must be between 2 and max_k")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (min_seed_stability, min_scale_agreement, min_margin)):
        raise ValueError("Hierarchy evidence thresholds must be finite in [0,1]")
    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise ValueError("Native hierarchy graph output must be new")
    for path, expected in source_hashes.items():
        if sha256_file(path) != expected:
            raise ValueError("Hierarchy source changed before native discovery")
    code = Path(__file__).resolve().parent
    code_hashes = {name: sha256_file(code / name) for name in CODE}
    frozen = outdir / "code"
    frozen.mkdir(parents=True)
    for name in CODE:
        shutil.copyfile(code / name, frozen / name)
        if sha256_file(frozen / name) != code_hashes[name]:
            raise ValueError("Hierarchy code changed while freezing sources")
    rows = grid.copy().reset_index(drop=True)
    for name in ("local_subdomain_id", "subdomain_id"):
        rows[name] = 0
    for name in ("seed_stability", "scale_agreement", "affinity_margin", "own_affinity_fraction", "graph_degree"):
        rows[name] = np.nan
    complete = np.ones(len(rows), bool)
    for block in blocks.values():
        if block.ndim != 2 or block.shape[0] != len(rows):
            raise ValueError("Feature rows must match the complete sorted grid")
        for start in range(0, len(rows), 1024):
            complete[start:start+1024] &= np.isfinite(block[start:start+1024]).all(axis=1)
    rows.loc[(rows.status_code == 2) & ~complete, "status_code"] = 4
    report = {"method": "within_parent_native_kodama_graph", "parent_domains": {},
        "seed": seed, "repeats": repeats, "weights": weights, "fixed_k": fixed_k,
        "requested_kodama_ncomp": ncomp, "pca_components_per_block": components,
        "source_sha256": source_hashes, "code_sha256": code_hashes,
        "thresholds": {"seed_stability": min_seed_stability, "scale_agreement": min_scale_agreement, "affinity_margin": min_margin},
        "selection_rule": "Within-parent KODAMA corrected graph, Leiden resolution candidates; automatic selection requires mean seed ARI>=.75, mean graph affinity margin>=.1 and minimum group size, then maximizes ARI*max(margin,0)-.01*K; fixed K filters actual raw counts and never fabricates a split/merge",
        "confidence_is_calibrated_probability": False,
        "seed_scope": "Clustering seeds on frozen native graphs; graph-building variability is separate",
        "scale_evidence": "Hungarian-aligned local/context graph partitions over all recorded clustering seeds; not independent biological validation",
        "graph_scope": "Feature similarity graphs within immutable parents, not physical-neighbour graphs; background is excluded by unchanged parent raster and four-connected region export"}
    _write(outdir / "discovery_status.json", {"status": "running"})
    rng, next_subdomain = np.random.default_rng(seed), 1
    try:
        for parent_id in sorted(rows.parent_domain_id.unique()):
            if parent_id == 0:
                continue
            indices = np.flatnonzero((rows.parent_domain_id == parent_id) & (rows.status_code == 2))
            parent_report = {"observations": len(indices), "selected_k": 0, "candidates": []}
            report["parent_domains"][str(parent_id)] = parent_report
            if len(indices) < min_observations:
                rows.loc[indices, "status_code"] = 5
                continue
            training = np.sort(rng.choice(len(indices), min(fit_limit, len(indices)), replace=False))
            transformed, normalization = transform_blocks(blocks, indices, training, weights, components, seed)
            transformed["combined"] = np.column_stack([transformed[name] for name in ("local", "context")])
            parent_dir = outdir / f"parent_{int(parent_id)}"
            parent_dir.mkdir()
            parent_rows = rows.iloc[indices][["label", "x", "y"]].copy()
            parent_rows[["x", "y"]] = parent_rows[["x", "y"]].to_numpy() * np.array(geometry["mpp_xy"]) + np.array(geometry["origin_um_xy"])
            parent_rows.to_csv(parent_dir / "rows.csv", index=False, float_format="%.17g")
            representations = {}
            for name, values in transformed.items():
                path = parent_dir / f"{name}.f64"
                np.asarray(values, dtype="<f8", order="C").tofile(path)
                representations[name] = {"path": path.name, "sha256": sha256_file(path), "shape": list(values.shape),
                    "constant": bool(np.equal(values, values[0]).all())}
            manifest = {"format": "cellphenotyper_hierarchy_kodama_input", "schema_version": "1.0.0",
                "rows": {"path": "rows.csv", "sha256": sha256_file(parent_dir / "rows.csv")},
                "representations": representations,
                "parameters": {"ncomp": ncomp, "M": m, "Tcycle": tcycle, "landmarks": min(fit_limit, len(indices)),
                    "cores": cores, "neighbors": min(neighbors, len(indices)-1), "seed": seed,
                    "cluster_seeds": list(range(seed, seed+repeats+1)), "resolutions": list(RESOLUTIONS)},
                "scope": {"parent_id": int(parent_id), "population": "all complete eligible grid observations within this immutable parent; excluded grid rows retained unresolved in final table",
                    "source_row_indices": indices.tolist(), "source_inputs_sha256": source_hashes,
                    "pca_training_row_indices": indices[training].tolist(), "normalization": normalization,
                    "coordinate_space": "original_slide_micrometres", "coordinates_used_for_fitting": False}}
            manifest_path = parent_dir / "input_manifest.json"
            _write(manifest_path, manifest)
            partitions, execution = _run_native(manifest_path, parent_dir / "native", frozen, library, cores)
            expected = {(name, resolution, run_seed) for name, entry in representations.items() if not entry["constant"]
                        for resolution in RESOLUTIONS for run_seed in range(seed, seed+repeats+1)}
            actual = set(partitions[["representation", "resolution", "seed"]].itertuples(index=False, name=None))
            if expected != actual or len(partitions) != len(expected) * len(indices):
                raise ValueError("Native hierarchy omitted or added a declared representation/resolution/seed population")
            selected, selection = select_partition(partitions, parent_rows.label.astype(str).tolist(), seed=seed, repeats=repeats,
                max_k=max_k, fixed_k=fixed_k, min_observations=min_observations,
                min_seed_stability=min_seed_stability, min_scale_agreement=min_scale_agreement, min_margin=min_margin)
            parent_report.update(selection, normalization=normalization, pca_training_observations=len(training),
                                 native_fit_observations=len(indices), native_execution=execution)
            if selected is None:
                rows.loc[indices, "status_code"] = 6
                base = partitions[(partitions.representation == "combined")
                                  & (partitions.resolution == RESOLUTIONS[0]) & (partitions.seed == seed)]
                if len(base):
                    degree = base.degree.to_numpy(int)
                    rows.loc[indices, "graph_degree"] = degree
                    rows.loc[indices[degree == 0], "status_code"] = 11
                continue
            rows.loc[indices, "local_subdomain_id"] = selected["labels"]
            rows.loc[indices, "status_code"] = selected["status"]
            for name in ("seed_stability", "scale_agreement", "affinity_margin", "own_affinity_fraction", "graph_degree"):
                rows.loc[indices, name] = selected[name]
            for label in sorted(set(selected["labels"]) - {0}):
                accepted = indices[(selected["labels"] == label) & (selected["status"] == 1)]
                if len(accepted):
                    rows.loc[accepted, "subdomain_id"] = next_subdomain
                    next_subdomain += 1
            parent_report["accepted_observations"] = int((selected["status"] == 1).sum())
        verify_discovery_artifacts(report, outdir)
        if any(sha256_file(path) != expected for path, expected in source_hashes.items()) or any(
                sha256_file(code / name) != expected or sha256_file(frozen / name) != expected for name, expected in code_hashes.items()):
            raise ValueError("Hierarchy sources or executed code changed during native discovery")
        rows["hierarchy_status"] = rows.status_code.map(status_names)
        report["sources_and_frozen_code_unchanged"] = True
        report["python_runtime"] = {"version": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")}}
        _write(outdir / "discovery_status.json", {"status": "pass", "report": report})
        return rows, report
    except Exception as error:
        _write(outdir / "discovery_status.json", {"status": "failed", "error": str(error)})
        raise
