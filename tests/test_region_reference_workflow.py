"""Real Nextflow reference-mapping smoke test with independent synthetic rows.

This tests the integration and abstention contract, not biological validity.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def checksum_tree(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def region_profiles(directory, sample, values, labels=None):
    directory.mkdir(parents=True)
    values = np.asarray(values, np.float32)
    rows = pd.DataFrame({"region_uid": [f"{sample}::synthetic::region_{i}" for i in range(len(values))],
                         "region_id": [f"{i:03d}" for i in range(len(values))], "sample_id": sample,
                         "parent_domain_id": 1, "subdomain_id": labels or ["1"] * len(values),
                         "label_status": "unsupervised_discovery", "x_um": np.arange(len(values)) * 20.,
                         "y_um": np.arange(len(values)) * 30., "area_um2": 200.})
    rows.to_csv(directory / "region_profiles.csv", index=False)
    rows[["region_uid", "region_id", "sample_id"]].to_csv(directory / "feature_rows.csv", index=False)
    manifest = {"schema_version": "1.0.0", "observation_unit": "tissue_region", "region_count": len(rows),
                "coordinate_space": "original_slide_micrometres", "feature_blocks": {}, "files": {}}
    for name, width in (("local", 56.), ("context", 224.)):
        array = np.column_stack((values, values * .5 + 2)).astype(np.float32)
        np.save(directory / f"{name}.npy", array)
        manifest["feature_blocks"][name] = {"path": f"{name}.npy", "shape": list(array.shape),
            "feature_names": ["feat_1", "feat_2"], "feature_definition": {
                "model_id": "engineering-test/synthetic", "model_revision": "a" * 64,
                "field_width_um_xy": [width, width], "pooling": "synthetic_two_features",
                "preprocessing": "fixture_only_not_model_inference", "independent_transformer_context": True,
                "aggregation": "assigned_grid_core_tissue_area_weighted_mean", "observation_unit": "tissue_region"}}
    manifest["files"] = checksum_tree(directory)
    (directory / "region_profiles_manifest.json").write_text(json.dumps(manifest))
    return rows


def test_actual_region_reference_nextflow_mapping_abstains_and_preserves_reference(tmp_path):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    for folder in ("modules", "lib"):
        (tmp_path / folder).mkdir()
    shutil.copyfile(ROOT / "modules/map_region_reference_atlas.nf", tmp_path / "modules/map_region_reference_atlas.nf")
    for source in (ROOT / "lib").glob("*.groovy"):
        shutil.copyfile(source, tmp_path / "lib" / source.name)
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    reference_a, reference_b = tmp_path / "reference_A/region_profiles", tmp_path / "reference_B/region_profiles"
    region_profiles(reference_a, "reference_A", [-.2, 0, .2])
    region_profiles(reference_b, "reference_B", [4.8, 5, 5.2])
    # Match basenames deliberately: module staging must distinguish query and
    # reference directories instead of colliding or guessing their contents.
    frozen = tmp_path / "frozen/region_profiles"
    build = subprocess.run([sys.executable, str(ROOT / "bin/cell_reference_atlas.py"), "build",
                            "--profiles", str(reference_a), str(reference_b), "--outdir", str(frozen),
                            "--version", "synthetic-engineering-v1", "--feature-groups", "local", "context",
                            "--label-column", "subdomain_id", "--label-scope", "sample"],
                           text=True, capture_output=True, timeout=30)
    assert build.returncode == 0, build.stdout + build.stderr
    frozen_before = checksum_tree(frozen)
    query = tmp_path / "query/region_profiles"
    original = region_profiles(query, "query_specimen", [0, 5, 100, np.nan], labels=["9", "8", "7", "6"])
    query_before = checksum_tree(query)
    params = {"outdir_base": str(tmp_path / "output"), "publish_dir_mode": "copy", "_executor_max_cpus": 2,
              "_executor_max_memory_gb": 4, "cell_profiles_cpus": 1, "cell_profiles_memory_gb": 2,
              "cell_atlas_python": sys.executable}
    (tmp_path / "params.json").write_text(json.dumps(params))
    (tmp_path / "workflow.nf").write_text(f"""nextflow.enable.dsl=2
include {{ MAP_REGION_REFERENCE_ATLAS }} from './modules/map_region_reference_atlas'
workflow {{
    query = file('{query}', checkIfExists: true)
    reference = file('{frozen}', checkIfExists: true)
    inputs = Channel.of(tuple('query_specimen::tile', 'query_specimen', 'tile', query, reference,
        PipelineHelpers.atlasTaskFingerprint('region_reference_mapping', projectDir, [query, reference])))
    runtime_plan = TaskRuntime.create(HardwarePolicy.resolve(params, 2, 4, false, 0d), 'cpu')
    MAP_REGION_REFERENCE_ATLAS(inputs, runtime_plan)
    MAP_REGION_REFERENCE_ATLAS.out.assignments.view {{ key, id, variant, output -> "ASSIGNMENTS ${{key}} ${{id}} ${{variant}} ${{output.name}}" }}
}}
""")
    run = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
                          "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-with-trace", str(tmp_path / "trace.tsv"),
                          "-work-dir", str(tmp_path / "work")], cwd=tmp_path,
                         env={**os.environ, "NXF_OFFLINE": "true", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "ASSIGNMENTS query_specimen::tile query_specimen tile reference_assignments.csv" in run.stdout
    output = tmp_path / "output/23_region_reference_mapping/query_specimen/tile/reference_assignments.csv"
    assignments = pd.read_csv(output, dtype={"region_id": str, "subdomain_id": str})
    assert assignments.region_uid.tolist() == original.region_uid.tolist()
    assert assignments.region_id.tolist() == original.region_id.tolist()
    assert assignments.subdomain_id.tolist() == ["9", "8", "7", "6"]
    assert assignments.reference_assignment.tolist() == ["reference_A:subdomain_id:1", "reference_B:subdomain_id:1", "unknown", "unknown"]
    assert assignments.reference_status.tolist() == ["assigned", "assigned", "outside_reference", "missing_features"]
    assert checksum_tree(frozen) == frozen_before
    assert checksum_tree(query) == query_before
    trace = pd.read_csv(tmp_path / "trace.tsv", sep="\t")
    assert len(trace) == 1 and trace.status.tolist() == ["COMPLETED"] and trace.exit.tolist() == [0]
