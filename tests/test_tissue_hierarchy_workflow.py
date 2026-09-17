"""Executable Nextflow stub-contract tests; not model/scientific validation."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def workflow_fixture(tmp_path, *, enabled=True, uncertainty=True, wrong_key=False):
    for folder in ("modules", "subworkflows", "lib", "snapshot"):
        (tmp_path / folder).mkdir()
    for name in ("prepare_hierarchy_features", "discover_tissue_hierarchy"):
        shutil.copyfile(ROOT / f"modules/{name}.nf", tmp_path / f"modules/{name}.nf")
    shutil.copyfile(ROOT / "subworkflows/run_tissue_hierarchy.nf", tmp_path / "subworkflows/run_tissue_hierarchy.nf")
    for source in (ROOT / "lib").glob("*.groovy"):
        shutil.copyfile(source, tmp_path / "lib" / source.name)
    (tmp_path / "bin").symlink_to(ROOT / "bin", target_is_directory=True)
    # Deliberate stub-only placeholders. No real model or TIFF is required.
    (tmp_path / "snapshot/config.json").write_text('{"architecture":"vit_giant_patch14_224"}')
    (tmp_path / "snapshot/model.safetensors").write_bytes(b"stub checkpoint")
    (tmp_path / "input.dat").write_bytes(b"immutable input placeholder")
    params = {
        "outdir_base": str(tmp_path / "output"), "publish_dir_mode": "copy",
        "_executor_max_cpus": 2, "_executor_max_memory_gb": 4,
        "cluster_primary_variant": "tile", "tissue_hierarchy_parent_variant": None,
        "tissue_hierarchy_model_snapshot": str(tmp_path / "snapshot") if enabled else None,
        "tissue_hierarchy_weights_filename": None, "tissue_hierarchy_python": "python3",
        "tissue_hierarchy_cpus": 1, "tissue_hierarchy_memory": "1 GB",
        "tissue_hierarchy_local_field_um": 56.0, "tissue_hierarchy_context_field_um": 224.0,
        "tissue_hierarchy_minimum_field_coverage": 1., "tissue_hierarchy_max_window_pixels": 16777216,
        "tissue_hierarchy_feature_batch": 2, "tissue_hierarchy_device": "cpu", "tissue_hierarchy_seed": 17,
        "kodama_ncomp": 50, "tissue_hierarchy_kodama_m": 100,
        "tissue_hierarchy_kodama_tcycle": 20, "tissue_hierarchy_kodama_neighbors": 100,
        "tissue_hierarchy_kodama_r_library": None, "tissue_hierarchy_min_affinity_margin": .1,
        "tissue_hierarchy_local_weight": 1., "tissue_hierarchy_context_weight": 1.,
        "tissue_hierarchy_max_k": 5, "tissue_hierarchy_fixed_k": 0, "tissue_hierarchy_repeats": 3,
        "tissue_hierarchy_fit_limit": 500, "tissue_hierarchy_components_per_block": 32,
        "tissue_hierarchy_min_observations": 20, "tissue_hierarchy_parent_purity": .8,
        "tissue_hierarchy_min_seed_stability": .8, "tissue_hierarchy_min_scale_agreement": 1.,
        "tissue_hierarchy_min_centroid_margin": .1, "tissue_hierarchy_tile_size": 512,
        "tissue_hierarchy_max_components": 1000000,
    }
    (tmp_path / "params.json").write_text(json.dumps(params))
    uncertain = "Channel.empty()" if not uncertainty else f"Channel.of(tuple('{'wrong' if wrong_key else 's1::tile'}', 's1', 'tile', source), tuple('s2::tile', 's2', 'tile', source))"
    script = f"""nextflow.enable.dsl=2
include {{ RUN_TISSUE_HIERARCHY }} from './subworkflows/run_tissue_hierarchy'
workflow {{
    source = file('{tmp_path}/input.dat', checkIfExists: true)
    samples = Channel.of(tuple('s1', source), tuple('s2', source))
    parents = Channel.of(tuple('s1::tile', 's1', 'tile', source), tuple('s2::tile', 's2', 'tile', source), tuple('s1::unused', 's1', 'unused', source))
    uncertain = {uncertain}
    runtime_plan = TaskRuntime.create(HardwarePolicy.resolve(params, 2, 4, false, 0d), 'cpu')
    RUN_TISSUE_HIERARCHY(samples, samples, samples, samples, samples, samples, parents, uncertain, {str(enabled).lower()}, runtime_plan)
    RUN_TISSUE_HIERARCHY.out.summaries.view {{ key, id, variant, file -> "SUMMARY ${{key}} ${{id}} ${{variant}} ${{file.name}}" }}
    RUN_TISSUE_HIERARCHY.out.features.view {{ id, directory -> "FEATURES ${{id}} ${{directory.name}}" }}
    RUN_TISSUE_HIERARCHY.out.parent_uncertainty.view {{ key, id, variant, file -> "UNCERTAINTY ${{key}} ${{id}} ${{variant}} ${{file.name}}" }}
}}
"""
    (tmp_path / "workflow.nf").write_text(script)


def run_stub(tmp_path):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow not available")
    return subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
                           "-params-file", str(tmp_path / "params.json"), "-stub-run", "-ansi-log", "false",
                           "-work-dir", str(tmp_path / "work")], cwd=tmp_path,
                          env={**os.environ, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=60)


def test_hierarchy_modules_run_stub_and_keep_canonical_outputs(tmp_path):
    workflow_fixture(tmp_path)
    result = run_stub(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    for sample in ("s1", "s2"):
        assert f"SUMMARY {sample}::tile {sample} tile hierarchy_summary.json" in result.stdout
        assert f"FEATURES {sample} features" in result.stdout
        assert f"UNCERTAINTY {sample}::tile {sample} tile parent_uncertainty.ome.tif" in result.stdout
        root = tmp_path / "output/22_tissue_hierarchy" / sample
        assert (root / "features/local/embedding_manifest.json").exists()
        assert (root / "features/context/embedding_manifest.json").exists()
        assert (root / "tile/parent_domains.ome.tif").read_bytes() == b"immutable input placeholder"
        assert (root / "tile/parent_uncertainty.ome.tif").read_bytes() == b"immutable input placeholder"
        assert (root / "tile/region_profiles/region_profiles_manifest.json").exists()
        assert not (root / "unused").exists()


def test_disabled_hierarchy_requires_no_model_and_emits_no_products(tmp_path):
    workflow_fixture(tmp_path, enabled=False, uncertainty=False)
    result = run_stub(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output").exists()
    assert "Submitted process" not in result.stdout


@pytest.mark.parametrize("wrong_key", [False, True])
def test_hierarchy_requires_matching_parent_uncertainty(tmp_path, wrong_key):
    workflow_fixture(tmp_path, uncertainty=wrong_key, wrong_key=wrong_key)
    result = run_stub(tmp_path)
    assert result.returncode != 0
    assert ("keys differ" if wrong_key else "requires canonical parent uncertainty") in result.stdout + result.stderr


def test_hierarchy_code_fingerprints_and_scientific_parameter_separation():
    feature_module = (ROOT / "modules/prepare_hierarchy_features.nf").read_text()
    hierarchy_module = (ROOT / "modules/discover_tissue_hierarchy.nf").read_text()
    for module in (feature_module, hierarchy_module):
        assert "PipelineHelpers.codeFingerprint" in module
        assert "--geojson" not in module
    assert "cache 'deep'" in feature_module
    # Installed native R/package bytes are bound inside discovery, but are not
    # yet a pre-cache scheduler input. Do not reuse a graph after runtime edits.
    assert "cache false" in hierarchy_module
    assert "kodama_ncomp" not in feature_module
    assert "--discovery-method kodama_graph" in hierarchy_module
    assert "--kodama-ncomp ${params.kodama_ncomp}" in hierarchy_module
    assert "--min-centroid-margin" not in hierarchy_module
    assert "path(model_weights, stageAs: 'uni2_snapshot/*')" in feature_module
    assert "--context-field-um" in feature_module and "--local-field-um" in feature_module
    assert "--parent-uncertainty" in hierarchy_module and "--support-mask" in hierarchy_module
