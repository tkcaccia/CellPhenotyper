"""Static coverage of the task.ext cache contract proven by real Nextflow runs.

These checks prevent an added path input, module, or stub branch from silently
escaping the shared byte-content key. They do not replace resume experiments.
"""
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted((ROOT / "modules").glob("*.nf"))


def input_section(text):
    match = re.search(r"^\s*input:\s*\n(.*?)^\s*output:", text, re.M | re.S)
    assert match is not None, "Module has no input/output section"
    return match.group(1)


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.stem)
def test_every_module_has_safe_cache_policy_and_the_correct_code_dependency_key(module):
    text = module.read_text()
    expected = "false" if module.stem == "discover_tissue_hierarchy" else "'deep'"
    assert re.findall(r"^\s*cache\s+([^\n]+)", text, re.M) == [expected]
    code_hooks = re.findall(
        r"\bext\s+code_fingerprint:\s*\{\s*ProcessCode\.fingerprint\(projectDir,\s*'([^']+)',\s*params\)\s*\}",
        text,
    )
    assert code_hooks == [module.stem], "Fingerprint must bind this module's actual dependency closure"
    assert not re.search(r"^\s*ext\.[a-z_]+\s*=", text, re.M), "Dotted process-body ext assignment is invalid DSL"


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.stem)
def test_directory_fingerprint_covers_every_declared_path_input_exactly(module):
    text = module.read_text()
    # Restrict parsing to declarations; path(...) in output blocks or scripts
    # must not accidentally become a source dependency.
    declared = re.findall(r"\bpath\s*\(\s*([A-Za-z_]\w*)", input_section(text))
    assert declared and len(declared) == len(set(declared))
    calls = re.findall(
        r"\bsource_fingerprint:\s*\{\s*ProcessCode\.directoryFingerprint\(\[([^\]]*)\]\)\s*\}",
        text,
    )
    assert len(calls) == 1, "Exactly one dynamic source-directory key is required"
    covered = [name.strip() for name in calls[0].split(",") if name.strip()]
    assert covered == declared, "A new, missing, duplicated, or reordered path input needs explicit cache coverage"


@pytest.mark.parametrize("module", MODULES, ids=lambda path: path.stem)
def test_real_and_stub_scripts_literally_reference_both_cache_properties(module):
    text = module.read_text()
    sections = re.search(r"^    script:\s*\n(.*?)^    stub:\s*\n(.*)", text, re.M | re.S)
    assert sections is not None
    for section in sections.groups():
        for name, label in (("code", "code"), ("source", "directory")):
            command = 'echo "[INFO] Process ' + label + ' cache fingerprint: ${task.ext.' + name + '_fingerprint}"'
            assert section.count(command) == 1, "Nextflow only hashes explicitly referenced named task.ext properties"


def test_inventory_includes_all_current_processes():
    assert len(MODULES) == 44
    for module in MODULES:
        assert len(re.findall(r"^process\s+\w+\s*\{", module.read_text(), re.M)) == 1


def test_cellvit_resolution_input_and_cli_binding_survive_cache_migration():
    text = (ROOT / "modules/run_cellvitpp.nf").read_text()
    assert re.findall(r"\bpath\s*\(\s*(\w+)", input_section(text)) == [
        "crop_tif", "shift_json", "clean_tissue_mask", "resolution_json",
    ]
    assert '--resolution-json "${resolution_json}"' in text
    assert "val(runtime_plan)" in input_section(text)


@pytest.mark.parametrize("stem", ["map_cell_reference_atlas", "map_region_reference_atlas", "export_spatialdata"])
def test_existing_explicit_atlas_fingerprint_and_runtime_inputs_remain(stem):
    section = input_section((ROOT / "modules" / f"{stem}.nf").read_text())
    assert section.count("val(source_fingerprint)") == 1
    assert section.count("val(runtime_plan)") == 1
