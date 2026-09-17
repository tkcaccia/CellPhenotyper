"""Offline native-runtime routing and real KODAMA graph capability checks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_kodama_graph_export import native_library, rscript  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = ROOT / "docker/cellphenotyper-hierarchy-python"
INSTALLER = ROOT / "docker/install_atlas_runtime.sh"
VERIFIER = ROOT / "docker/verify_atlas_runtime.py"


def clean_env():
    return {key: value for key, value in os.environ.items() if key not in {
        "PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE", "CELLPHENOTYPER_SOURCE_PYCACHEPREFIX",
        "CELLPHENOTYPER_HIERARCHY_RSCRIPT"}}


def executable(path, body="#!/bin/bash\nexit 99\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def verifier_module():
    spec = importlib.util.spec_from_file_location("native_runtime_probe", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discovery_carries_absolute_native_path_through_atlas_path_reset(tmp_path):
    prefix = tmp_path / "atlas prefix"
    executable(prefix / "bin/python", '#!/bin/bash\nprintf "%s\\n" "$CELLPHENOTYPER_HIERARCHY_RSCRIPT" "$PATH"\nprintf "<%s>\\n" "$@"\n')
    native = executable(tmp_path / "native ' ; $(touch forbidden)" / "bin/Rscript")
    entry = tmp_path / "source folder" / "discover_tissue_hierarchy.py"
    entry.parent.mkdir()
    entry.write_text("# Fake routed input, never executed.\n")
    literal = "literal ; $(touch forbidden) ' argument"
    result = subprocess.run([str(DISPATCHER), str(entry), literal], cwd=tmp_path,
        env={**clean_env(), "CELLPHENOTYPER_ATLAS_PREFIX": str(prefix), "CELLPHENOTYPER_HIERARCHY_RSCRIPT": str(native)},
        text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(native)
    assert lines[1].startswith(str(prefix / "bin") + ":")
    assert str(native.parent) not in lines[1]
    assert lines[2:] == ["<-E>", "<-s>", f"<{entry}>", f"<{literal}>"]
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("kind", ["empty", "relative", "missing", "directory", "not_executable", "padding", "control"])
def test_discovery_rejects_invalid_native_override_without_fallback(tmp_path, kind):
    native = executable(tmp_path / "Rscript")
    value = str(native)
    if kind == "empty":
        value = ""
    elif kind == "relative":
        value = "Rscript"
    elif kind == "missing":
        value = str(tmp_path / "absent")
    elif kind == "directory":
        value = str(tmp_path)
    elif kind == "not_executable":
        native.chmod(0o644)
    elif kind == "padding":
        value = str(executable(tmp_path / "Rscript "))
    elif kind == "control":
        value = str(executable(tmp_path / "Rscript\nother"))
    result = subprocess.run([str(DISPATCHER), "discover_tissue_hierarchy.py"],
        env={**clean_env(), "CELLPHENOTYPER_HIERARCHY_RSCRIPT": value}, cwd=tmp_path,
        text=True, capture_output=True, timeout=20)
    assert result.returncode == 78
    assert "no fallback" in result.stderr


def test_feature_extraction_route_does_not_require_or_select_native_r(tmp_path):
    fake_python = executable(tmp_path / "ml-python", '#!/bin/bash\nprintf "<%s>\\n" "$@"\n')
    result = subprocess.run([str(DISPATCHER), "prepare_hierarchy_features.py", "input"],
        env={**clean_env(), "CELLPHENOTYPER_ML_PYTHON": str(fake_python),
             "CELLPHENOTYPER_HIERARCHY_RSCRIPT": "/does/not/exist"},
        text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["<-E>", "<-s>", "<prepare_hierarchy_features.py>", "<input>"]


def test_installer_is_build_only_adds_missing_dependencies_and_pins_existing_kodama():
    result = subprocess.run(["bash", "-n", str(INSTALLER)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    source = INSTALLER.read_text()
    assert 'native_prefix=/opt/micromamba/envs/kodama-r' in source
    assert 'required <- c("Matrix","igraph","digest","jsonlite")' in source
    assert 'missing <- setdiff(required, installed)' in source
    assert 'install.packages(missing, lib=.Library, dependencies=NA' in source
    assert 'before <- identity()' in source and 'identical(before, identity())' in source
    assert 'system2("sha256sum", shQuote(files)' in source
    assert 'install_github' not in source
    assert '--component hierarchy-native' in source and '--native-rscript "$native_rscript"' in source
    assert 'atlas_native_hierarchy_runtime_build.json' in source
    assert 'R_ENVIRON_USER=/dev/null R_PROFILE_USER=/dev/null' in source


def test_installer_embedded_r_parses_without_executing_installation(tmp_path, rscript):
    source = INSTALLER.read_text().split("<<'RSCRIPT'\n", 1)[1].split("\nRSCRIPT\n", 1)[0]
    snippet = tmp_path / "install_syntax_only.R"
    snippet.write_text(source)
    result = subprocess.run([rscript, "--vanilla", "-e", 'invisible(parse(file=commandArgs(TRUE)[1]))', str(snippet)],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_actual_local_native_runtime_functional_probe_cli(tmp_path, rscript, native_library):
    report = tmp_path / "native_runtime.json"
    code_paths = [ROOT / "bin" / name for name in ("run_hierarchy_kodama.R", "kodama_graph_export.R", "kodama_graph_clustering.R")]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}
    result = subprocess.run([sys.executable, str(VERIFIER), "--component", "hierarchy-native", "--source-root", str(ROOT),
        "--native-rscript", rscript, "--native-r-library", native_library, "--report", str(report)],
        env={**os.environ, "LD_LIBRARY_PATH": "/poison-atlas-libraries", "R_HOME": "/poison-R-home",
             "R_LIBS_USER": "/poison-user-library", "R_LIBS_SITE": "/poison-site-library"},
        text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(report.read_text())
    assert record["component"] == "hierarchy-native"
    assert record["imported_package_origins"] == {}
    assert not record["biological_validation"] and not record["container_build_verified_by_this_probe"]
    observed = record["result"]
    for field in ("native_raw_data_kodama_handle", "portable_graph_exact_readback", "native_graph_leiden", "literal_ids_all_rows"):
        assert observed[field] == "passed"
    assert observed["models_loaded"] is False
    receipt = observed["runner_receipt"]
    assert receipt["partition_run_count"] == 12 and receipt["partition_rows"] == 360
    assert receipt["runtime"]["packages"]["KODAMA"]["version"] == "0.99.7"
    assert receipt["producer_before"] == receipt["producer_after"] == before
    assert receipt["sources_before"] == receipt["sources_after"]
    assert receipt["runtime_before"] == receipt["runtime_after"]
    assert receipt["representations"]["local"]["dimensions"] == 4
    assert receipt["representations"]["local"]["effective_native_parameters"]["ncomp"] == 50
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths} == before


def test_native_probe_rejects_mixed_external_library_under_strict_isolation(rscript, native_library):
    prefix = Path(rscript).resolve().parent.parent
    if Path(native_library).resolve().is_relative_to(prefix):
        pytest.skip("This fixture's native package is already inside the selected R prefix")
    with pytest.raises(RuntimeError, match="outside isolated native prefix"):
        verifier_module().hierarchy_native_probe(ROOT, rscript, native_library, strict_isolation=True)


@pytest.mark.parametrize("value", ["", "relative/Rscript", "/not/a/real/Rscript", "/tmp"])
def test_native_probe_rejects_invalid_executable_without_host_fallback(value):
    with pytest.raises(ValueError, match="absolute executable"):
        verifier_module().hierarchy_native_probe(ROOT, value)


def test_native_probe_failure_cannot_write_pass_report(tmp_path):
    bad_r = executable(tmp_path / "Rscript", '#!/bin/bash\necho "native runtime deliberately unavailable" >&2\nexit 71\n')
    report = tmp_path / "must_not_exist.json"
    result = subprocess.run([sys.executable, str(VERIFIER), "--component", "hierarchy-native", "--source-root", str(ROOT),
        "--native-rscript", str(bad_r), "--report", str(report)], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and "native runtime deliberately unavailable" in result.stderr
    assert not report.exists()


def test_native_probe_preexisting_report_is_preserved_before_execution(tmp_path):
    report = tmp_path / "keep.json"
    report.write_text("existing report")
    with pytest.raises(FileExistsError, match="new path"):
        verifier_module().main(["--component", "hierarchy-native", "--report", str(report), "--native-rscript", "/not/used"])
    assert report.read_text() == "existing report"
