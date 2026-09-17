"""Offline packaging, interpreter-isolation and real compatibility probes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER = ROOT / "docker"
ATLAS = DOCKER / "cellphenotyper-atlas-python"
HIERARCHY = DOCKER / "cellphenotyper-hierarchy-python"
SOURCE_CACHE = ROOT / "bin/activate_source_python.sh"


def source_cache_env():
    """A caller must deliberately opt in to the complete task cache contract."""
    return {key: value for key, value in os.environ.items() if key not in {
        "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE",
        "CELLPHENOTYPER_SOURCE_PYCACHEPREFIX"}}


def selected_prefix():
    candidate = ROOT / ".venv-spatialdata"
    return candidate if (candidate / "bin/python").is_file() else Path(sys.prefix)


def fake_native_rscript(tmp_path):
    """Routing tests do not execute R; discovery now requires an explicit path."""
    path = tmp_path / "fake Rscript"
    path.write_text("#!/bin/bash\nexit 99\n")
    path.chmod(0o755)
    return str(path)


def run(command, **kwargs):
    return subprocess.run(list(map(str, command)), capture_output=True, text=True, timeout=120, **kwargs)


def selected_cpu_supplement(tmp_path):
    """Expose only already installed CPU dependencies, never the ambient ML site."""
    cpu_python = ROOT / ".venv-spatial/bin/python"
    if not cpu_python.is_file():
        pytest.skip("Offline compatibility probe needs the existing CPU test runtime")
    code = "import importlib.metadata as m,json; names=['scikit-learn','scikit-image','joblib','threadpoolctl']; print(json.dumps({str(d.locate_file(p.parts[0])):p.parts[0] for n in names for d in [m.distribution(n)] for p in d.files if p.parts and p.parts[0] not in ('.','..')}))"
    result = run([cpu_python, "-E", "-s", "-c", code])
    assert result.returncode == 0, result.stderr
    supplement = tmp_path / "selected_cpu_packages"
    supplement.mkdir()
    for source, name in json.loads(result.stdout).items():
        target = supplement / name
        if target.exists():
            continue
        target.symlink_to(source, target_is_directory=Path(source).is_dir())
    return supplement


@pytest.mark.parametrize("script", ["cellphenotyper-atlas-python", "cellphenotyper-hierarchy-python", "install_atlas_runtime.sh"])
def test_shell_syntax(script):
    result = run(["bash", "-n", DOCKER / script])
    assert result.returncode == 0, result.stderr


def test_atlas_interpreter_clears_host_python_and_ml_libraries(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "foreign_host_module.py").write_text("polluted=True\n")
    prefix = selected_prefix()
    env = {**os.environ, "CELLPHENOTYPER_ATLAS_PREFIX": str(prefix),
        "PYTHONHOME": "/nonexistent-host-python", "PYTHONPATH": str(poison), "PYTHONUSERBASE": str(poison),
        "VIRTUAL_ENV": "/another-venv", "CONDA_PREFIX": "/ml-prefix", "LD_LIBRARY_PATH": "/ml-libraries"}
    code = "import importlib.util,json,os,sys; print(json.dumps({'exe':sys.executable,'site':sys.flags.no_user_site,'ignore':sys.flags.ignore_environment,'foreign':importlib.util.find_spec('foreign_host_module') is not None,'ld':os.environ.get('LD_LIBRARY_PATH'),'pyhome':os.environ.get('PYTHONHOME'),'argv':sys.argv[1:]}))"
    literal = "spaces ; $(touch should_not_exist) ' quote"
    result = run([ATLAS, "-c", code, literal], env=env, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["site"] == 1 and payload["ignore"] == 1 and payload["foreign"] is False
    assert payload["pyhome"] is None and payload["ld"] == str(prefix / "lib")
    assert payload["argv"] == [literal]
    assert not (tmp_path / "should_not_exist").exists()


def test_atlas_missing_runtime_fails_without_host_fallback(tmp_path):
    result = run([ATLAS, "-c", "raise SystemExit(99)"], env={**os.environ, "CELLPHENOTYPER_ATLAS_PREFIX": str(tmp_path / "absent")})
    assert result.returncode == 78 and "unavailable" in result.stderr


@pytest.mark.parametrize("target", ["prepare_hierarchy_features.py", "discover_tissue_hierarchy.py"])
def test_hierarchy_dispatches_to_explicit_existing_runtime(tmp_path, target):
    folder = tmp_path / "source folder"
    folder.mkdir()
    script = folder / target
    script.write_text("import json,sys; print(json.dumps({'executable':sys.executable,'args':sys.argv[1:]}))\n")
    prefix = selected_prefix()
    env = {**os.environ, "CELLPHENOTYPER_ATLAS_PREFIX": str(prefix), "CELLPHENOTYPER_ML_PYTHON": sys.executable,
           "CELLPHENOTYPER_HIERARCHY_RSCRIPT": fake_native_rscript(tmp_path)}
    result = run([HIERARCHY, script, "unchanged argument"], env=env)
    assert result.returncode == 0, result.stderr
    expected = sys.executable if target.startswith("prepare") else str(prefix / "bin/python")
    assert json.loads(result.stdout) == {"executable": expected, "args": ["unchanged argument"]}


def test_hierarchy_never_silently_routes_unknown_scripts_or_missing_model_runtime(tmp_path):
    unknown = run([HIERARCHY, "other.py"])
    assert unknown.returncode == 64
    missing = run([HIERARCHY, "prepare_hierarchy_features.py"], env={**os.environ,
        "CELLPHENOTYPER_ML_PYTHON": str(tmp_path / "missing"), "CELLPHENOTYPER_ATLAS_PREFIX": str(selected_prefix())})
    assert missing.returncode == 78 and "no atlas fallback" in missing.stderr


@pytest.mark.parametrize("wrapper,target", [
    (ATLAS, "probe.py"),
    (HIERARCHY, "prepare_hierarchy_features.py"),
    (HIERARCHY, "discover_tissue_hierarchy.py"),
])
def test_isolated_wrappers_execute_same_stat_changed_helper_without_bytecode_writes(tmp_path, wrapper, target):
    """Exercise actual interpreters, not a string-only wrapper flag check."""
    task = tmp_path / "task ' ; $(touch should_not_exist)"
    task.mkdir()
    source = tmp_path / "source ' ; folder"
    source.mkdir()
    module = source / "cached_helper.py"
    module.write_text('value = "FIRST"\n')
    prefix = selected_prefix()
    python = prefix / "bin/python"
    compile_code = "import py_compile,sys; print(py_compile.compile(sys.argv[1],doraise=True,invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))"
    compiled = run([python, "-E", "-s", "-c", compile_code, module], env=source_cache_env())
    assert compiled.returncode == 0, compiled.stderr
    bytecode = Path(compiled.stdout.strip())
    old_bytecode = bytecode.read_bytes()
    stat = module.stat()
    module.write_text('value = "AFTER"\n')
    os.utime(module, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert module.stat().st_size == stat.st_size
    entry = source / target
    entry.write_text("import cached_helper,json,sys; print(json.dumps({"
        "'value':cached_helper.value,'ignore':sys.flags.ignore_environment,"
        "'site':sys.flags.no_user_site,'no_write':sys.dont_write_bytecode,"
        "'prefix':sys.pycache_prefix,'argv':sys.argv[1:]}))\n")
    literal = "literal ' ; $(touch argument_was_executed)"
    env = {**source_cache_env(), "CELLPHENOTYPER_ATLAS_PREFIX": str(prefix),
           "CELLPHENOTYPER_ML_PYTHON": str(python),
           "CELLPHENOTYPER_HIERARCHY_RSCRIPT": fake_native_rscript(tmp_path)}
    result = run(["bash", "-c", 'set -euo pipefail; source "$1"; "$2" "$3" "$4"',
        "source-cache-wrapper-test", SOURCE_CACHE, wrapper, entry, literal], env=env, cwd=task)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["value"] == "AFTER"
    assert payload["ignore"] == 1 and payload["site"] == 1 and payload["no_write"] is True
    assert payload["argv"] == [literal]
    cache_prefix = Path(payload["prefix"])
    assert cache_prefix.parent == task and cache_prefix.name.startswith(".cellphenotyper-pycache.")
    assert list(cache_prefix.iterdir()) == []
    assert bytecode.read_bytes() == old_bytecode
    assert b"FIRST" in old_bytecode and b"AFTER" not in old_bytecode
    assert list(task.iterdir()) == [cache_prefix]


@pytest.mark.parametrize("wrapper,target", [(ATLAS, "probe.py"), (HIERARCHY, "prepare_hierarchy_features.py")])
@pytest.mark.parametrize("failure", ["missing_marker", "missing_python_prefix", "mismatch",
    "missing_no_write", "write_enabled", "absent", "relative", "symlink", "nonempty"])
def test_isolated_wrappers_reject_incomplete_or_stale_task_cache_contract(tmp_path, wrapper, target, failure):
    prefix = selected_prefix()
    cache = tmp_path / ".cellphenotyper-pycache.fixture"
    cache.mkdir()
    entry = tmp_path / target
    entry.write_text("raise RuntimeError('entry point must not run')\n")
    env = {**source_cache_env(), "CELLPHENOTYPER_ATLAS_PREFIX": str(prefix),
        "CELLPHENOTYPER_ML_PYTHON": str(prefix / "bin/python"),
        "CELLPHENOTYPER_SOURCE_PYCACHEPREFIX": str(cache),
        "PYTHONPYCACHEPREFIX": str(cache), "PYTHONDONTWRITEBYTECODE": "1"}
    if failure == "missing_marker":
        del env["CELLPHENOTYPER_SOURCE_PYCACHEPREFIX"]
    elif failure == "missing_python_prefix":
        del env["PYTHONPYCACHEPREFIX"]
    elif failure == "mismatch":
        env["PYTHONPYCACHEPREFIX"] = str(tmp_path / ".cellphenotyper-pycache.other")
    elif failure == "missing_no_write":
        del env["PYTHONDONTWRITEBYTECODE"]
    elif failure == "write_enabled":
        env["PYTHONDONTWRITEBYTECODE"] = "0"
    elif failure in {"absent", "relative", "symlink"}:
        target_cache = tmp_path / ".cellphenotyper-pycache.absent"
        if failure == "relative":
            target_cache = Path(cache.name)
        elif failure == "symlink":
            target_cache.symlink_to(cache, target_is_directory=True)
        env["CELLPHENOTYPER_SOURCE_PYCACHEPREFIX"] = str(target_cache)
        env["PYTHONPYCACHEPREFIX"] = str(target_cache)
    else:
        (cache / "possibly_stale.pyc").write_bytes(b"not trusted")
    result = run([wrapper, entry], env=env, cwd=tmp_path)
    assert result.returncode == 78, result.stdout + result.stderr
    assert "cache" in result.stderr and "entry point must not run" not in result.stderr
    assert result.stdout == ""


def test_both_overlay_recipes_ship_actual_current_runtime_and_source():
    docker = (DOCKER / "Dockerfile.atlas").read_text()
    singularity = (ROOT / "singularity/cellphenotyper_atlas.def").read_text()
    for artifact in ("requirements-atlas.txt", "requirements-spatialdata.txt", "cell_profiles.nf", "nextflow_schema.json"):
        assert artifact in docker and artifact in singularity and (ROOT / artifact).is_file()
    for directory in ("bin", "lib", "resources", "modules", "subworkflows", "docker"):
        assert f"COPY {directory} /opt/cellphenotyper/{directory}" in docker
        assert f"  {directory} /opt/cellphenotyper/{directory}" in singularity
    assert "install_atlas_runtime.sh" in docker and "install_atlas_runtime.sh" in singularity
    assert "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}" in docker and "From: {{ BASE_IMAGE }}" in singularity
    installer = (DOCKER / "install_atlas_runtime.sh").read_text()
    assert 'micromamba create -y --prefix "$atlas_prefix"' in installer
    assert "--only-binary=:all:" in installer and "--strict-isolation" in installer
    assert "--component model" in installer
    config = (DOCKER / "atlas-runtime.config").read_text()
    for name in ("cell_atlas_python", "spatialdata_python", "tissue_hierarchy_python"):
        assert f"params.{name}" in config


def test_candidate_cpu_dependency_combination_executes_real_pipeline_and_spatialdata(tmp_path):
    """Reuse installed packages without installing or claiming a Linux build.

    The local SpatialData environment lacks sklearn/skimage. Append the existing
    CPU test runtime only to provide those dependencies, then verify actual
    package origins and version pins plus native algorithms and real export.
    """
    prefix = ROOT / ".venv-spatialdata"
    if not (prefix / "bin/python").is_file():
        pytest.skip("Offline combined-runtime probe requires the two existing project test environments")
    core = selected_cpu_supplement(tmp_path)
    report_path = tmp_path / "compatibility.json"
    verifier = DOCKER / "verify_atlas_runtime.py"
    code = "import runpy,sys; sys.path.append(sys.argv.pop(1)); sys.argv[0]=sys.argv.pop(1); runpy.run_path(sys.argv[0],run_name='__main__')"
    result = run([prefix / "bin/python", "-E", "-s", "-c", code, core, verifier,
        "--source-root", ROOT, "--requirements", ROOT / "requirements-atlas.txt", "--report", report_path])
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text())
    assert report["result"]["canonical_profile_spatialdata_roundtrip"] == "passed"
    assert report["result"]["native_pca_kmeans"] == "passed"
    assert report["result"]["native_region_morphology"] == "passed"
    assert report["requirements_checked"]["scikit-learn"] == "1.8.0"
    assert report["requirements_files_sha256"] == {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("requirements-atlas.txt", "requirements-spatialdata.txt")}
    assert report["imported_package_origins"]["numpy"]["version"] == "2.4.6"
    assert report["imported_package_origins"]["sklearn"]["version"] == "1.8.0"
    assert not report["strict_isolation_checked"] and not report["container_build_verified_by_this_probe"]


def test_strict_isolation_rejects_the_mixed_development_probe(tmp_path):
    prefix = ROOT / ".venv-spatialdata"
    if not (prefix / "bin/python").is_file():
        pytest.skip("Requires existing project CPU and SpatialData test runtimes")
    core = selected_cpu_supplement(tmp_path)
    verifier = DOCKER / "verify_atlas_runtime.py"
    code = "import runpy,sys; sys.path.append(sys.argv.pop(1)); sys.argv[0]=sys.argv.pop(1); runpy.run_path(sys.argv[0],run_name='__main__')"
    result = run([prefix / "bin/python", "-E", "-s", "-c", code, core, verifier,
        "--source-root", ROOT, "--strict-isolation", "--report", tmp_path / "must_not_exist.json"])
    assert result.returncode != 0 and "outside isolated Python prefix" in result.stderr
    assert not (tmp_path / "must_not_exist.json").exists()
