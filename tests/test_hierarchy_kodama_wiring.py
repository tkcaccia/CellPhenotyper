"""Native hierarchy configuration and real Nextflow argv; no model or R fit."""
import json
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import jsonschema
import pytest

from test_tissue_hierarchy_workflow import workflow_fixture
from test_process_code_dependencies import inventory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from validate_pipeline_params import cross_field_errors, validate


def schema():
    return json.loads((ROOT / "nextflow_schema.json").read_text())


def enabled_options(**updates):
    return {"tissue_hierarchy_enable": True, "uni2_sampling_mode": "grid",
            "uni2_encoder": "uni2-h", "uni2_include_inner_square": True,
            "uni2_fuse_tile_inner_square": True, "uni2_use_roi_crop": True,
            "tissue_hierarchy_model_snapshot": "/explicit/task-visible/snapshot", **updates}


def test_defaults_keep_hierarchy_opt_in_and_native_controls_explicit():
    properties = schema()["definitions"]["cell_atlas_options"]["properties"]
    config = (ROOT / "nextflow.config").read_text()
    expected = {"tissue_hierarchy_enable": False, "tissue_hierarchy_kodama_m": 100,
        "tissue_hierarchy_kodama_tcycle": 20, "tissue_hierarchy_kodama_neighbors": 100,
        "tissue_hierarchy_kodama_r_library": None, "tissue_hierarchy_min_affinity_margin": .1}
    for key, value in expected.items():
        assert properties[key]["default"] == value
        literal = "null" if value is None else str(value).lower()
        assert re.search(rf"^\s*{key}\s*=\s*{re.escape(literal)}(?:\s|$)", config, re.M)
    assert "tissue_hierarchy_discovery_method" not in properties
    assert "Deprecated" in properties["tissue_hierarchy_min_centroid_margin"]["description"]
    assert re.search(r"^\s*kodama_ncomp\s*=\s*50\b", config, re.M)


@pytest.mark.parametrize("key", ["tissue_hierarchy_kodama_m", "tissue_hierarchy_kodama_tcycle",
    "tissue_hierarchy_kodama_neighbors"])
@pytest.mark.parametrize("value", [0, -1, 1.25, "100", True])
def test_native_integer_controls_reject_nonpositive_or_noninteger_values(key, value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(schema()).validate({key: value})


@pytest.mark.parametrize("value", [-.01, 1.01, "0.1", True])
def test_affinity_margin_schema_is_bounded_numeric(value):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(schema()).validate({"tissue_hierarchy_min_affinity_margin": value})


@pytest.mark.parametrize("value", [0., .1, 1.])
def test_affinity_margin_endpoints_and_default_validate(value):
    options = enabled_options(tissue_hierarchy_min_affinity_margin=value)
    jsonschema.Draft7Validator(schema()).validate(options)
    assert cross_field_errors(options) == []


@pytest.mark.parametrize("value", ["", " ", " path", "path ", "a\nb", "a\x00b", 42])
def test_explicit_r_library_rejects_empty_or_control_character_declarations(tmp_path, value):
    options = enabled_options(tissue_hierarchy_kodama_r_library=value)
    path = tmp_path / "params.json"
    path.write_text(json.dumps(options))
    with pytest.raises(ValueError, match="tissue_hierarchy_kodama_r_library"):
        validate(path, ROOT / "nextflow_schema.json")


def test_library_is_only_an_explicit_runtime_path_and_legacy_selector_is_not_silently_used():
    literal = "/task visible/library ' ; $(touch must_not_execute)"
    options = enabled_options(tissue_hierarchy_kodama_r_library=literal)
    jsonschema.Draft7Validator(schema()).validate(options)
    assert cross_field_errors(options) == []
    for method in ("legacy_kmeans", "kodama_graph"):
        assert any("not a pipeline parameter" in error for error in cross_field_errors(
            enabled_options(tissue_hierarchy_discovery_method=method)))
    assert any("finite" in error for error in cross_field_errors(
        enabled_options(tissue_hierarchy_min_affinity_margin=float("nan"))))


def test_native_source_dependency_closure_is_in_actual_process_cache_key(inventory):
    records, _ = inventory
    files = {Path(path).name for path in records["discover_tissue_hierarchy"]["files"]}
    assert {"discover_tissue_hierarchy.py", "hierarchy_kodama.py", "run_hierarchy_kodama.R",
            "kodama_graph_export.R", "kodama_graph_clustering.R", "activate_source_python.sh"} <= files


def argv_fixture(tmp_path, updates):
    workflow_fixture(tmp_path)
    interpreter = tmp_path / "capture hierarchy argv"
    interpreter.write_text("#!" + sys.executable + "\n" + '''import json, pathlib, shutil, sys
args = sys.argv[1:]
def option(name): return args[args.index(name) + 1]
out = pathlib.Path(option('--outdir'))
(out / 'region_profiles').mkdir(parents=True)
(out / 'argv.json').write_text(json.dumps(args))
shutil.copyfile(option('--parent-mask'), out / 'parent_domains.ome.tif')
shutil.copyfile(option('--parent-uncertainty'), out / 'parent_uncertainty.ome.tif')
for name in ('subdomain_mask.ome.tif', 'hierarchy_status.ome.tif', 'region_mask.ome.tif', 'grid_subdomains.csv'):
    (out / name).write_bytes(b'argv probe only; not scientific data')
(out / 'hierarchy_summary.json').write_text('{"argv_probe":true,"model_inference":false}')
''')
    interpreter.chmod(0o755)
    params_path = tmp_path / "params.json"
    params = json.loads(params_path.read_text())
    params.update(tissue_hierarchy_python=str(interpreter), tissue_hierarchy_cpus=2, **updates)
    params_path.write_text(json.dumps(params))
    (tmp_path / "nextflow.config").write_text("trace.fields = 'task_id,name,status,cpus,memory'\n")
    (tmp_path / "workflow.nf").write_text('''nextflow.enable.dsl=2
include { DISCOVER_TISSUE_HIERARCHY } from './modules/discover_tissue_hierarchy'
workflow {
    source = file('input.dat', checkIfExists:true)
    features = file('snapshot', checkIfExists:true)
    runtime_plan = TaskRuntime.create(HardwarePolicy.resolve(params, 2, 4, false, 0d), 'cpu')
    DISCOVER_TISSUE_HIERARCHY(Channel.of(tuple('s1::tile', 's1', 'tile', source, source, source, source, source,
        source, source, source, features)), runtime_plan)
}
''')
    return params


def run_argv(tmp_path):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    return subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "workflow.nf"),
        "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false", "-work-dir", str(tmp_path / "work"),
        "-with-trace", str(tmp_path / "trace.tsv")],
        cwd=tmp_path, env={**os.environ, "NXF_OFFLINE": "true"}, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("library,ncomp", [(None, 50), ("/explicit/library ' ; $(touch injected_marker)", 37)])
def test_real_nextflow_command_forwards_native_method_resources_and_literal_library(tmp_path, library, ncomp):
    params = argv_fixture(tmp_path, {"tissue_hierarchy_kodama_r_library": library, "kodama_ncomp": ncomp})
    result = run_argv(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    out = tmp_path / "output/22_tissue_hierarchy/s1/tile"
    args = json.loads((out / "argv.json").read_text())
    option = lambda name: args[args.index(name) + 1]
    assert option("--discovery-method") == "kodama_graph"
    assert option("--kodama-ncomp") == str(ncomp)
    assert option("--components-per-block") == str(params["tissue_hierarchy_components_per_block"])
    assert option("--kodama-m") == "100" and option("--kodama-tcycle") == "20"
    with (tmp_path / "trace.tsv").open() as stream:
        trace = list(csv.DictReader(stream, delimiter="\t"))
    assert len(trace) == 1 and trace[0]["status"] == "COMPLETED"
    allocated_cpus = int(trace[0]["cpus"])
    assert 1 <= allocated_cpus <= params["tissue_hierarchy_cpus"]
    assert option("--kodama-neighbors") == "100" and option("--kodama-cpus") == str(allocated_cpus)
    assert option("--min-affinity-margin") == "0.1"
    assert "--min-centroid-margin" not in args and "legacy_kmeans" not in args
    if library is None:
        assert "--kodama-r-library" not in args
    else:
        assert option("--kodama-r-library") == library
    assert not list(tmp_path.rglob("injected_marker"))
    assert (out / "parent_domains.ome.tif").read_bytes() == (tmp_path / "input.dat").read_bytes()
    assert (out / "parent_uncertainty.ome.tif").read_bytes() == (tmp_path / "input.dat").read_bytes()


@pytest.mark.parametrize("options,error", [
    ({"tissue_hierarchy_kodama_m": "1; touch injected_marker"}, "must be a positive integer"),
    ({"tissue_hierarchy_min_affinity_margin": "NaN"}, "must be finite in [0,1]"),
    ({"tissue_hierarchy_kodama_r_library": " "}, "explicit nonempty path"),
])
def test_actual_task_rejects_malformed_new_options_before_executing(tmp_path, options, error):
    argv_fixture(tmp_path, options)
    result = run_argv(tmp_path)
    assert result.returncode != 0 and error in result.stdout + result.stderr
    assert not list(tmp_path.rglob("argv.json"))
    assert not list(tmp_path.rglob("injected_marker"))
