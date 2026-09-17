"""Task-private, no-write Python imports preserve traps and shared caches."""
import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

from test_process_code_dependencies import inventory

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bin/activate_source_python.sh"
STAGES = ["build_spatial_cell_profiles", "fit_cohort_niches", "link_cell_tissue_hierarchy",
          "map_cell_reference_atlas", "map_region_reference_atlas", "export_spatialdata",
          "prepare_hierarchy_features", "discover_tissue_hierarchy"]


def shell(script, directory, *args):
    return subprocess.run(["bash", "-c", script, "source-cache-test", str(HELPER), *map(str,args)],
        cwd=directory, capture_output=True, text=True, timeout=30,
        env={name:value for name,value in os.environ.items()
             if name not in {"PYTHONPYCACHEPREFIX", "PYTHONDONTWRITEBYTECODE",
                             "CELLPHENOTYPER_SOURCE_PYCACHEPREFIX"}})


def test_private_prefix_ignores_same_stat_stale_pyc_without_writes_or_trap_changes(tmp_path):
    directory=tmp_path/"task with spaces"; directory.mkdir()
    source=tmp_path/"source"; source.mkdir()
    module=source/"cached_helper.py"; module.write_text('value = "FIRST"\n')
    bytecode=Path(py_compile.compile(str(module),doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
    old_bytecode=bytecode.read_bytes(); stat=module.stat()
    module.write_text('value = "AFTER"\n'); os.utime(module,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    result=shell(r'''set -euo pipefail
trap 'printf "%s\n" ORIGINAL_EXIT_TRAP' EXIT
original_trap="$(trap -p EXIT)"
source "$1"
[[ "$(trap -p EXIT)" == "$original_trap" ]]
first_prefix="$PYTHONPYCACHEPREFIX"
[[ "$PYTHONDONTWRITEBYTECODE" == 1 ]]
[[ "$CELLPHENOTYPER_SOURCE_PYCACHEPREFIX" == "$PYTHONPYCACHEPREFIX" ]]
"$2" -c 'import sys; sys.path.insert(0,sys.argv[1]); import cached_helper; print(cached_helper.value)' "$3"
source "$1"
[[ "$first_prefix" != "$PYTHONPYCACHEPREFIX" ]]
[[ "$(trap -p EXIT)" == "$original_trap" ]]
printf '%s\n' "$first_prefix" "$PYTHONPYCACHEPREFIX"
''',directory,sys.executable,source)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.splitlines()[0]=="AFTER"
    assert result.stdout.splitlines()[-1]=="ORIGINAL_EXIT_TRAP"
    assert bytecode.read_bytes()==old_bytecode
    prefixes=list(directory.glob(".cellphenotyper-pycache.*"))
    assert len(prefixes)==2 and all(not list(path.iterdir()) for path in prefixes)


@pytest.mark.parametrize("failure",["return 73", "printf '%s' /nonexistent/private-prefix"])
def test_mktemp_failure_is_explicit_and_preserves_existing_environment_and_traps(tmp_path,failure):
    result=shell(r'''set -euo pipefail
export PYTHONPYCACHEPREFIX=original-prefix PYTHONDONTWRITEBYTECODE=original-setting
export CELLPHENOTYPER_SOURCE_PYCACHEPREFIX=original-marker
trap 'printf "%s\n" ORIGINAL_EXIT_TRAP' EXIT
original_trap="$(trap -p EXIT)"
mktemp() { '''+failure+r'''; }
if source "$1"; then exit 99; fi
[[ "$PYTHONPYCACHEPREFIX" == original-prefix && "$PYTHONDONTWRITEBYTECODE" == original-setting ]]
[[ "$CELLPHENOTYPER_SOURCE_PYCACHEPREFIX" == original-marker ]]
[[ "$(trap -p EXIT)" == "$original_trap" ]]
printf '%s\n' FAILURE_CAUGHT
''',tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "[ERROR]" in result.stderr and "FAILURE_CAUGHT" in result.stdout
    assert result.stdout.splitlines()[-1]=="ORIGINAL_EXIT_TRAP"
    assert not list(tmp_path.iterdir())


def test_each_scoped_real_stage_sources_helper_before_python_and_hashes_it(inventory):
    records,_=inventory
    for stage in STAGES:
        module=(ROOT/"modules"/(stage+".nf")).read_text()
        real=module.split("    script:\n",1)[1].split("    stub:\n",1)[0]
        command='source "${projectDir}/bin/activate_source_python.sh"'
        assert real.count(command)==1
        assert real.index("set -euo pipefail")<real.index(command)<real.index('export OMP_NUM_THREADS')
        assert HELPER in set(map(Path,records[stage]["files"]))
    assert HELPER not in set(map(Path,records["prepare_input_ometiff"]["files"]))
