"""Real main.nf graph/staging/resource regression, with every task in stub mode.

The bundled engineering specimens and invalid local checkpoint/reference fixtures
exercise wiring only. No weights are downloaded, models run, or accuracy claimed.
"""
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ("ROI_A", "ROI_B")
ATLAS_STAGES = {
    "cell_profiles", "cell_tissue_links", "reference_mapping", "region_reference_mapping",
    "spatialdata", "hierarchy_features", "hierarchy_discovery",
}


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def migrated_process_stages():
    """Resolve process aliases from their modules, not their workflow nesting."""
    processes = {}
    modules = {}

    def register(name, stage):
        previous = processes.get(name)
        assert previous is None or previous == stage, f"Ambiguous process alias {name}: {previous} vs {stage}"
        processes[name] = stage

    for path in (ROOT / "modules").glob("*.nf"):
        source = path.read_text()
        cpu = re.findall(r"TaskRuntime\.cpus\(runtime_plan,\s*'([^']+)'\)", source)
        if not cpu:
            continue
        memory = re.findall(r"TaskRuntime\.memory\(runtime_plan,\s*'([^']+)'\)", source)
        assert len(set(cpu)) == 1 and set(cpu) == set(memory), path.name
        name = re.search(r"\bprocess\s+(\w+)\s*\{", source).group(1)
        modules[path.stem] = cpu[0]
        register(name, cpu[0])
    for path in [ROOT / "main.nf", *(ROOT / "subworkflows").glob("*.nf")]:
        for body, target in re.findall(r"include\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", path.read_text()):
            stage = modules.get(Path(target).name.removesuffix(".nf"))
            if stage is None:
                continue
            for declaration in body.split(";"):
                names = declaration.strip().split()
                if names:
                    register(names[-1], stage)  # canonical name or explicit alias
    assert ATLAS_STAGES <= set(processes.values())
    return processes


def build_fixture(tmp_path, mode, full_model_routes=False):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    immutable = {}
    for sample in SAMPLES:
        source = ROOT / "Data" / f"{sample}.ome.tif"
        assert source.is_file(), f"Missing bundled engineering specimen: {source}"
        immutable[source] = checksum(source)
        (inputs / source.name).symlink_to(source)
    snapshot = tmp_path / "stub_uni2_snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"stub":true,"architecture":"vit_giant_patch14_224"}')
    (snapshot / "model.safetensors").write_bytes(b"INVALID CHECKPOINT: stub workflow fixture, never model weights\n")
    reference = tmp_path / "stub_reference"
    reference.mkdir()
    (reference / "atlas_manifest.json").write_text('{"stub":true,"scientific_claim":"No fitted reference atlas"}')
    pathsegmentor_repo = tmp_path / "stub_pathsegmentor_repo"
    pathsegmentor_repo.mkdir()
    (pathsegmentor_repo / "PINNED_REVISION").write_text("stub-only; not executable model source\n")
    pathsegmentor_config = tmp_path / "stub_pathsegmentor_config.yaml"
    pathsegmentor_config.write_text("stub: true\n")
    pathsegmentor_checkpoint = tmp_path / "stub_pathsegmentor_checkpoint.pt"
    pathsegmentor_checkpoint.write_bytes(b"INVALID CHECKPOINT: stub workflow fixture\n")
    for path in [*snapshot.iterdir(), *reference.iterdir()]:
        immutable[path] = checksum(path)
    output = tmp_path / "results"
    params = {
        "folder_input": str(inputs), "image_input": None, "roi_geojson": None,
        "outdir_base": str(output), "publish_dir_mode": "copy", "compute_device": "cpu",
        "start_point": "convert", "end_point": "cluster_geojson", "run_full_pipeline": True,
        "cell_detection_mode": "stardist", "uni2_sampling_mode": mode,
        "uni2_embedding_storage": "binary" if mode == "both" else "csv",
        "cluster_target_clusters": 2, "cluster_forced_count_sensitivity_acknowledged": True,
        "kodama_ncomp": 50, "gigatime_kodama_enable": True,
        "cell_profiles_enable": True, "cell_profiles_uni2_enable": True,
        "cell_profiles_markers_enable": True, "cell_profiles_domain_source": "refined",
        "cell_profiles_spatialdata": True, "cell_reference_atlas": str(reference),
        "region_reference_atlas": str(reference), "tissue_hierarchy_enable": True,
        "tissue_hierarchy_model_snapshot": str(snapshot), "tissue_hierarchy_device": "cpu",
        "tissue_hierarchy_memory": "1536 MB", "tissue_hierarchy_cpus": 16,
        "cell_profiles_cpus": 16, "cell_profiles_memory_gb": 64,
        # Keep the low-memory UNI-2 special case identical to its declared plan
        # even on a small CI worker; all other atlas stages exercise auto-capping.
        "uni2_cpus": 1, "max_cpus": 2, "max_memory_gb": 4, "max_parallel_tasks": 2,
        "storage_preflight_mode": "off", "neoplastic_section_enable": False,
        "titan_enable": False, "pathofmpred_enable": False,
    }
    if full_model_routes:
        # These are explicitly stub-only routes: no accelerator, model weights,
        # protected R package or container engine is required or exercised.
        params.update(compute_device="gpu", host_arch="amd64", gpu_scheduler_enable=False,
                      cell_detection_mode="consensus", cellvit_export_embeddings=True,
                      end_point="pathofmpred", neoplastic_section_enable=True,
                      titan_enable=True, pathofmpred_enable=True, pathofmpred_cancer="BRCA",
                      pathsegmentor_enable=True, pathsegmentor_guided_refine_enable=True,
                      pathsegmentor_repo=str(pathsegmentor_repo),
                      pathsegmentor_config=str(pathsegmentor_config),
                      pathsegmentor_checkpoint=str(pathsegmentor_checkpoint))
    (tmp_path / "params.json").write_text(json.dumps(params, indent=2))
    # Retain production process directives/hooks; configure only execution surface
    # and trace serialization. No withName resource overrides or fake processes.
    (tmp_path / "probe.config").write_text("""
docker.enabled = false
singularity.enabled = false
apptainer.enabled = false
process.executor = 'local'
trace {
    fields = 'task_id,hash,native_id,process,name,status,exit,cpus,memory,submit,duration,realtime,%cpu,peak_rss,peak_vmem,rchar,wchar'
    raw = true
}
""")
    return output, immutable


@pytest.mark.parametrize("mode,full_model_routes", [("grid", False), ("both", False), ("grid", True)])
@pytest.mark.skipif(shutil.which("nextflow") is None, reason="Nextflow executable is unavailable")
def test_main_stub_runtime_plan_matches_every_migrated_task_and_both_specimens(tmp_path, mode, full_model_routes):
    output, immutable = build_fixture(tmp_path, mode, full_model_routes)
    completed = subprocess.run([
        "nextflow", "-log", str(tmp_path / "nextflow.log"), "run", str(ROOT / "main.nf"),
        "-c", str(tmp_path / "probe.config"), "-params-file", str(tmp_path / "params.json"),
        "-stub-run", "-ansi-log", "false", "-work-dir", str(tmp_path / "work"),
    ], cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true", "NXF_DISABLE_CHECK_LATEST": "true"},
        capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    execution = output / "00_execution"
    document = json.loads((execution / "hardware_plan.json").read_text())
    plan = document["task_runtime_plan"]
    assert plan["schema_version"] == 1
    assert plan["compute_device"] == ("gpu" if full_model_routes else "cpu")
    assert 0 < plan["cpu_budget"] <= 2 and 0 < plan["memory_budget_gb"] <= 4
    assert plan["stages"] == document["policy"]["stages"]
    with (execution / "trace.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows and all(row["status"] == "COMPLETED" and row["exit"] == "0" for row in rows)
    stages_by_process = migrated_process_stages()
    observed = {}
    for row in rows:
        stage = stages_by_process.get(row["process"].split(":")[-1])
        if stage is None:
            continue  # stages not yet migrated have separate legacy-policy tests
        allocation = plan["stages"][stage]
        assert int(row["cpus"]) == allocation["cpus"], row
        assert int(row["memory"]) == int(Decimal(str(allocation["memory_gb"])) * 1024**3), row
        assert int(row["cpus"]) <= plan["cpu_budget"]
        assert Decimal(row["memory"]) <= Decimal(str(plan["memory_budget_gb"])) * 1024**3
        observed.setdefault(stage, []).append(row)
    assert ATLAS_STAGES | {"uni2", "kodama", "gigatime", "grandqc", "stardist", "medsam_refine"} <= observed.keys()
    if full_model_routes:
        assert {"hovernet", "cellvit", "titan"} <= observed.keys()
        for stage in ("hovernet", "cellvit", "titan"):
            assert len(observed[stage]) == 2
    for stage in ATLAS_STAGES:
        assert len(observed[stage]) == 2, (stage, observed[stage])
        assert all(any(sample in row["name"] for row in observed[stage]) for sample in SAMPLES)
    assert plan["stages"]["hierarchy_features"]["memory_gb"] == 1.5
    assert plan["stages"]["hierarchy_discovery"]["memory_gb"] == 1.5
    contract = json.loads((execution / "analysis_contract.json").read_text())
    assert contract["scientific_route"]["uni2_sampling_mode"] == mode
    assert contract["scientific_route"]["auxiliary_cell_comparison"] is (mode == "both")
    for sample in SAMPLES:
        linked = output / "24_cell_tissue_links" / sample / "cell_profiles/cell_profiles_manifest.json"
        hierarchy = output / "22_tissue_hierarchy" / sample / "features/hierarchy_features_summary.json"
        spatial = output / "20_spatialdata" / sample / "spatialdata.zarr/stub.json"
        assert all(json.loads(path.read_text())["stub"] is True for path in (linked, hierarchy, spatial))
        assert (output / "21_reference_mapping" / sample / "reference_assignments.csv").is_file()
        assert (output / "23_region_reference_mapping" / sample / "standard/reference_assignments.csv").is_file()
        assert (output / "09_embeddings" / f"{sample}__cells" / f"embeddings_{sample}__cells_tile").is_dir()
        auxiliary_kodama = output / "10_kodama" / f"{sample}__cells" / "kodama_output/kodama_stub.txt"
        assert auxiliary_kodama.exists() is (mode == "both")
        comparison = output / "10_kodama" / sample / f"uni2_route_comparison_{sample}/{sample}_uni2_route_comparison.csv"
        assert comparison.exists() is (mode == "both")
        auxiliary_refinement = output / "14_medsam_refine_tissue" / f"{sample}__cells" / f"{sample}__cells_standard_grown_mask_refined.ome.tif"
        assert auxiliary_refinement.exists() is (mode == "both")
        pathsegmentor = output / "09b_pathsegmentor" / sample / f"pathsegmentor_{sample}/pathsegmentor_manifest.json"
        semantic = output / "09c_pathsegmentor_annotations" / sample / "grid/standard" / f"pathsegmentor_{sample}_grid_standard/pathsegmentor_annotation_manifest.json"
        semantic_refine = output / "14b_pathsegmentor_refine" / sample / f"{sample}_standard_pathsegmentor_refined.tif"
        assert pathsegmentor.exists() is full_model_routes
        assert semantic.exists() is full_model_routes
        assert semantic_refine.exists() is full_model_routes
        assert (output / "10_kodama" / sample / "kodama_output/kodama_stub.txt").is_file()
    assert {path: checksum(path) for path in immutable} == immutable
