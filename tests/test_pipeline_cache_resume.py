"""Real full-main stub cache acceptance in a small, isolated project copy.

No detector/encoder/model executes. Only copied Python/R comment text changes;
the bundled engineering images and the actual working tree remain untouched.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

import test_runtime_plan_workflow as runtime_fixture


ROOT = Path(__file__).resolve().parents[1]
NEXTFLOW = shutil.which("nextflow")
LIMIT_BYTES = 100 * 1024**2
RARE_MODULES = {"prepare_stardist_auto_roi", "extract_uni2_embeddings", "quantify_gigatime_intensity", "roi_geojson_to_mask"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bounded_size(directory):
    size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file() and not path.is_symlink())
    assert size < LIMIT_BYTES, f"Stub fixture exceeded 100 MiB: {size} bytes"
    return size


def copy_project(directory):
    project = directory / "project"
    project.mkdir()
    for name in ("bin", "lib", "modules", "subworkflows", "resources"):
        shutil.copytree(ROOT / name, project / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("main.nf", "nextflow.config"):
        shutil.copy2(ROOT / name, project / name)
    (project / "Data").mkdir()
    originals = {}
    for sample in runtime_fixture.SAMPLES:
        source = ROOT / "Data" / f"{sample}.ome.tif"
        originals[source] = digest(source)
        shutil.copy2(source, project / "Data" / source.name)
    bounded_size(directory)
    return project, originals


def module_aliases(project):
    """Exact include declarations, not process-name substring guesses."""
    aliases, modules = {}, {}
    for path in (project / "modules").glob("*.nf"):
        name = re.search(r"^process\s+(\w+)\s*\{", path.read_text(), re.M).group(1)
        modules[path.stem] = name
        aliases[name] = path.stem
    for path in [project / "main.nf", *(project / "subworkflows").glob("*.nf")]:
        for declarations, target in re.findall(r"include\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", path.read_text()):
            stem = Path(target).name.removesuffix(".nf")
            if stem not in modules:
                continue
            for declaration in declarations.split(";"):
                fields = declaration.split()
                if not fields:
                    continue
                name = fields[-1]
                assert name not in aliases or aliases[name] == stem
                aliases[name] = stem
    return aliases, set(modules)


def run_fixture(project, launch, script, label, *, resume=False):
    trace = launch / f"{label}.tsv"
    command = [NEXTFLOW, "-log", str(launch / f"{label}.nextflow.log"), "run", str(project / script),
        "-c", str(launch / "probe.config"), "-params-file", str(launch / "params.json"),
        "-stub-run", "-ansi-log", "false", "-with-trace", str(trace), "-work-dir", str(launch / "work")]
    if resume:
        command.append("-resume")
    environment = dict(os.environ, NXF_OFFLINE="true", NXF_DISABLE_CHECK_LATEST="true",
        CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    completed = subprocess.run(command, cwd=launch, env=environment, text=True, capture_output=True, timeout=240)
    (launch / f"{label}.stdout.txt").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if script == "dependency_inventory.nf":
        return {}
    with trace.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows and all(row["status"] in {"COMPLETED", "CACHED"} and row["exit"] == "0" for row in rows)
    assert len({row["name"] for row in rows}) == len(rows), "Task names must identify complete specimen/variant instances"
    for row in rows:
        stdout = (Path(row["workdir"]) / ".command.out").read_text()
        for field in ("code", "directory"):
            match = re.search(rf"Process {field} cache fingerprint: ([0-9a-f]{{64}})", stdout)
            assert match, f"Executed stub omitted its literal task.ext {field} hook: {row['name']}"
            row[f"{field}_fingerprint"] = match.group(1)
    bounded_size(launch.parent)
    return {row["name"]: row for row in rows}


def assert_all_cached(previous, current):
    assert previous.keys() == current.keys()
    # Native hierarchy discovery cannot safely resume before its external R
    # runtime identity is part of the scheduler key. Its derivatives may rerun;
    # expensive upstream feature extraction and unrelated stages must still cache.
    hierarchy_derivatives = {"LINK_CELL_TISSUE_HIERARCHY", "MAP_REGION_REFERENCE_ATLAS",
                             "MAP_CELL_REFERENCE_ATLAS", "FIT_COHORT_NICHES", "EXPORT_SPATIALDATA"}
    for name, row in current.items():
        process = row["process"].split(":")[-1]
        if process == "DISCOVER_TISSUE_HIERARCHY":
            assert row["status"] == "COMPLETED", row
            assert row["code_fingerprint"] == previous[name]["code_fingerprint"]
            continue
        if process in hierarchy_derivatives and row["status"] == "COMPLETED":
            assert row["code_fingerprint"] == previous[name]["code_fingerprint"]
            continue
        assert row["status"] == "CACHED", row
        for field in ("hash", "workdir", "code_fingerprint", "directory_fingerprint"):
            assert row[field] == previous[name][field], (name, field)


def mutate_same_stat(path, old, new, project):
    assert path.is_relative_to(project) and not path.is_symlink(), "Never mutate original or external files"
    assert len(old) == len(new) and old != new
    original = path.read_bytes()
    assert old in original
    stat = path.stat()
    changed = original.replace(old, new, 1)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = path.stat()
    assert after.st_size == stat.st_size and after.st_mtime_ns == stat.st_mtime_ns
    return {"path": str(path.relative_to(project)), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "before_sha256": hashlib.sha256(original).hexdigest(), "after_sha256": digest(path)}


def assert_selective_rerun(previous, current, inventory, changed_path, aliases):
    assert previous.keys() == current.keys()
    affected = {module for module, paths in inventory.items() if str(changed_path.resolve()) in paths}
    assert affected, f"Mutation was absent from the actual dependency closure: {changed_path}"
    direct, independent = [], []
    independent_modules = {"prepare_input_ometiff", "run_grandqc_artifact_analysis"}
    assert not (affected & independent_modules)
    for name, row in current.items():
        module = aliases[row["process"].split(":")[-1]]
        if module in affected:
            direct.append(name)
            assert row["status"] == "COMPLETED" and row["hash"] != previous[name]["hash"], row
            assert row["code_fingerprint"] != previous[name]["code_fingerprint"], row
        elif module in independent_modules:
            independent.append(name)
            assert row["status"] == "CACHED" and row["hash"] == previous[name]["hash"], row
            assert row["code_fingerprint"] == previous[name]["code_fingerprint"], row
        # A changed upstream task may legitimately invalidate its consumers,
        # even if those consumers do not directly import the changed helper.
    assert direct and len(independent) == 2 * len(runtime_fixture.SAMPLES)
    return {"directly_affected_modules": sorted(affected), "directly_rerun_tasks": direct,
            "independent_cached_tasks": independent}


def write_rare_probe(project, missing):
    assert missing <= RARE_MODULES, f"New full-main coverage gaps need explicit probes: {sorted(missing)}"
    inputs = project / "rare_inputs"
    inputs.mkdir()
    for name in ("image.tif", "labels.tif", "resolution.json", "roi.geojson", "gigatime/stub.json"):
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub-only engineering input; never image/model inference")
    statements = {
        "prepare_stardist_auto_roi": "PREPARE_STARDIST_AUTO_ROI(Channel.of(tuple('rare', file('" + str(inputs / "image.tif") + "'))))",
        "quantify_gigatime_intensity": "QUANTIFY_GIGATIME_INTENSITY(Channel.of(tuple('rare', file('" + str(inputs / "gigatime") + "'), file('" + str(inputs / "labels.tif") + "'), 'nuclei')))",
        "extract_uni2_embeddings": "EXTRACT_UNI2_EMBEDDINGS(Channel.of(tuple('rare', file('" + str(inputs / "image.tif") + "'), file('" + str(inputs / "labels.tif") + "'), file('" + str(inputs / "resolution.json") + "'), 'tile', false, 'none', 255)), runtime_plan)",
        "roi_geojson_to_mask": "ROI_GEOJSON_TO_MASK(Channel.of(tuple('rare', file('" + str(inputs / "roi.geojson") + "'), file('" + str(inputs / "image.tif") + "'))))",
    }
    source = "nextflow.enable.dsl=2\n"
    for module in sorted(missing):
        source += f"include {{ {module.upper()} }} from './modules/{module}'\n"
    source += "workflow {\n    runtime_plan = TaskRuntime.create(HardwarePolicy.resolve(params, 2, 4, false, 0d), 'cpu')\n"
    source += "\n".join("    " + statements[module] for module in sorted(missing)) + "\n}\n"
    (project / "rare_probe.nf").write_text(source)


@pytest.mark.skipif(NEXTFLOW is None, reason="Nextflow executable is unavailable")
def test_full_main_stub_selective_cache_resume_and_all_module_hook_coverage(tmp_path, monkeypatch):
    project, originals = copy_project(tmp_path)
    launch = tmp_path / "execution"
    launch.mkdir()
    # Reuse the established real main.nf engineering fixture, but point its
    # image inventory exclusively at our copied project.
    with monkeypatch.context() as patch:
        patch.setattr(runtime_fixture, "ROOT", project)
        runtime_fixture.build_fixture(launch, "both", full_model_routes=True)
    parameters_path = launch / "params.json"
    parameters = json.loads(parameters_path.read_text())
    parameters.update(_executor_max_cpus=2, _executor_max_memory_gb=4, hf_token_env_file=None,
                      cohort_niches_enable=True)
    parameters_path.write_text(json.dumps(parameters, indent=2))
    configuration = launch / "probe.config"
    configuration.write_text(configuration.read_text() + "\nreport.enabled=false\ntimeline.enabled=false\ndag.enabled=false\n"
        + "trace.fields='task_id,hash,process,name,status,exit,workdir'\n")
    aliases, all_modules = module_aliases(project)
    assert len(all_modules) == 41
    (project / "dependency_inventory.nf").write_text('''nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
    def inventory = [:]
    file("${projectDir}/modules").listFiles().findAll { it.name.endsWith('.nf') }.each { source ->
        def module = source.name[0..-4]
        inventory[module] = ProcessCode.dependencies(projectDir, module, params).collect { it.toString() }
    }
    file("${launchDir}/dependencies.json").text = JsonOutput.toJson(inventory)
}
''')
    run_fixture(project, launch, "dependency_inventory.nf", "dependencies")
    inventory = json.loads((launch / "dependencies.json").read_text())
    baseline = run_fixture(project, launch, "main.nf", "baseline")
    assert all(row["status"] == "COMPLETED" for row in baseline.values())
    unchanged = run_fixture(project, launch, "main.nf", "unchanged", resume=True)
    assert_all_cached(baseline, unchanged)
    report = {"scientific_claim": "Stub-only cache/graph engineering acceptance; no model inference", "mutations": []}
    previous = unchanged
    for label, relative, old, new in (
        ("python_same_stat", "bin/cell_profile_io.py", b"Shared identity", b"shared identity"),
        ("r_same_stat", "bin/uni2_embedding_io.R", b"# Strict reader", b"# STRICT reader"),
    ):
        changed = project / relative
        receipt = mutate_same_stat(changed, old, new, project)
        current = run_fixture(project, launch, "main.nf", label, resume=True)
        receipt.update(assert_selective_rerun(previous, current, inventory, changed, aliases))
        report["mutations"].append(receipt)
        previous = current
    observed = {aliases[row["process"].split(":")[-1]] for row in baseline.values()}
    missing = all_modules - observed
    report["main_modules"] = sorted(observed)
    report["direct_probe_modules"] = sorted(missing)
    if missing:
        write_rare_probe(project, missing)
        rare = run_fixture(project, launch, "rare_probe.nf", "rare_baseline")
        assert all(row["status"] == "COMPLETED" for row in rare.values())
        rare_cached = run_fixture(project, launch, "rare_probe.nf", "rare_unchanged", resume=True)
        assert_all_cached(rare, rare_cached)
        observed |= {aliases[row["process"].split(":")[-1]] for row in rare.values()}
    assert observed == all_modules
    assert {path: digest(path) for path in originals} == originals
    report["covered_modules"] = sorted(observed)
    report["fixture_bytes"] = bounded_size(tmp_path)
    (launch / "cache_acceptance.json").write_text(json.dumps(report, indent=2))
