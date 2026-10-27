"""Execute the real Nextflow task with tiny CPU fakes; no model/GPU is used."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _groovy(value):
    return "'" + str(value).replace('\\', '\\\\').replace("'", "\\'") + "'"


def _fake_project(tmp_path, *, explicit=False, executable_name=False, missing_python=False, fail_preflight=False, prepare_only=False):
    project = tmp_path / "project with spaces and 'quotes'"
    project.mkdir()
    (project / 'bin').mkdir()
    (project / 'resources').mkdir()
    (project / 'lib').mkdir()
    shutil.copyfile(ROOT / 'lib/PipelineHelpers.groovy', project / 'lib/PipelineHelpers.groovy')
    for name in ('grandqc_mask.py', 'cellvit_models.py', 'hardware_runtime.py', 'model_provenance.py'):
        (project / 'bin' / name).symlink_to(ROOT / 'bin' / name)
    (project / 'resources/cellvit_model_catalog.json').symlink_to(ROOT / 'resources/cellvit_model_catalog.json')
    runtime = project / "isolated cellvit env's bin"
    runtime.mkdir()
    executable = runtime / 'cellvit-inference'
    executable.write_text('#!/bin/sh\necho "This fake detector must never run" >&2\nexit 98\n')
    executable.chmod(0o755)
    python = runtime / ('custom-python' if explicit else 'python')
    if not missing_python:
        # A shell trampoline forwards to this test's real CPU interpreter.
        import shlex
        python.write_text('#!/bin/sh\nexport CELLVIT_BOUND_INTERPRETER=' + shlex.quote(str(python)) + '\nexec ' + shlex.quote(sys.executable) + ' "$@"\n')
        python.chmod(0o755)
    log = project / 'binding-calls.jsonl'
    wrapper = project / 'bin/fake_wrapper.py'
    wrapper.write_text('''import gzip,json,os,sys
from pathlib import Path
args=sys.argv[1:]
preflight='--preflight-only' in args
with Path(os.environ['CELLVIT_BINDING_LOG']).open('a') as f:
 f.write(json.dumps({'args':args,'phase':'preflight' if preflight else 'processing','bound_interpreter':os.environ.get('CELLVIT_BOUND_INTERPRETER'),'path':os.environ['PATH'],'ld_library_path':os.environ.get('LD_LIBRARY_PATH','')})+'\\n')
if preflight:
 if os.environ.get('CELLVIT_FAIL_PREFLIGHT')=='1':
  print('SYNTHETIC dependency failure: pyvips unavailable',file=sys.stderr)
  sys.exit(19)
 print('{"status":"ready","binding_test_only":true}')
 sys.exit(0)
out=Path(args[args.index('--outdir')+1]);out.mkdir()
for name in ['cellvit_cells.json','cellvit_metadata.json','cellvit_cell_features.manifest.json','cellvit_feature_cells.json','cellvit_cell_classification.json','cellvit_model_preflight.json']:
 (out/name).write_text('{"binding_test_only":true}')
with gzip.open(out/'cellvit_cell_classification.csv.gz','wt') as f:f.write('cell_id,model_id,status\\n')
with gzip.open(out/'cellvit_cell_features.csv.gz','wt') as f:f.write('cell_id,source_id,x,y\\n')
''')
    for name in ('crop.tif', 'shift.json', 'clean.tif'):
        (project / name).write_text('SYNTHETIC command binding fixture, not image data')
    settings = {
        'outdir_base': str(project / 'published'), 'publish_dir_mode': 'copy',
        '_executor_max_cpus': 1, 'cellvit_cpus': 1, '_executor_max_memory_gb': 1,
        'cellvit_memory_gb': 1, 'cellvit_time': '1m', 'cellvit_script': 'bin/fake_wrapper.py',
        '_resolved_compute_device': 'gpu', 'compute_device': 'gpu', # declarative guard only; the CPU fake never starts the detector
        'cellvit_executable': executable.name if executable_name else str(executable),
        'cellvit_python': str(python) if explicit else '', 'cellvit_amp': False,
        '_resolved_cell_identification_enable': False, 'cellvit_classification_device': 'cpu',
        'cellvit_default_mpp': .5, 'cellvit_cache_dir': str(project / 'cache that must follow preflight'),
        'cellvit_model': 'HIPT', 'cellvit_taxonomy': 'pannuke', 'cellvit_gpu': 0,
        'cellvit_batch_size': 2, 'cellvit_ray_workers': 1, 'cellvit_ray_worker_cpus': 1,
        'cellvit_classifier_cache': '',
    }
    declarations = []
    for key, value in settings.items():
        encoded = str(value).lower() if isinstance(value, bool) else str(value) if isinstance(value, (int, float)) else _groovy(value)
        declarations.append(f'params.{key} = {encoded}')
    (project / 'main.nf').write_text("nextflow.enable.dsl=2\n" + '\n'.join(declarations) + f'''
include {{ RUN_CELLVITPP }} from {_groovy(ROOT / 'modules/run_cellvitpp')}
workflow {{
  RUN_CELLVITPP(Channel.of(tuple('test', file("${{projectDir}}/crop.tif"), file("${{projectDir}}/shift.json"), file("${{projectDir}}/clean.tif"))),
    [enabled:false, taxonomies:['pannuke'], classification_device:'cpu', default_mpp:0.5, compute_device:'gpu'])
}}
''')
    env = dict(os.environ, NXF_OFFLINE='true', NXF_ANSI_LOG='false',
               NXF_OPTS='-Xms64m -Xmx512m', CELLVIT_BINDING_LOG=str(log),
               CELLVIT_FAIL_PREFLIGHT='1' if fail_preflight else '0')
    env['PATH'] = str(runtime) + os.pathsep + env['PATH']
    result = None if prepare_only else subprocess.run(['nextflow', 'run', str(project / 'main.nf')], cwd=project,
                            env=env, capture_output=True, text=True, timeout=60)
    return project, runtime, executable, python, log, result


@pytest.mark.skipif(shutil.which('nextflow') is None, reason='Nextflow unavailable')
@pytest.mark.parametrize('explicit,executable_name', [(False, False), (True, True)])
def test_real_task_uses_bound_interpreter_for_preflight_and_processing(tmp_path, explicit, executable_name):
    project, runtime, executable, python, log, result = _fake_project(tmp_path, explicit=explicit, executable_name=executable_name)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [c['phase'] for c in calls] == ['preflight', 'processing']
    assert all(c['args'][c['args'].index('--executable') + 1] == str(executable) for c in calls)
    assert all(c['path'].split(os.pathsep)[0] == str(runtime) for c in calls)
    assert all(c['ld_library_path'].split(os.pathsep)[0] == str(runtime / '../lib') for c in calls)
    command = next((project / 'work').glob('*/*/.command.sh')).read_text()
    assert all(c['bound_interpreter'] == str(python) for c in calls)
    assert command.count('"$cellvit_python"') >= 3
    assert (project / 'cache that must follow preflight').is_dir()
    assert (project / 'published/03c_cellvitpp/test/cellvit_test/cellvit_cells.json').is_file()


@pytest.mark.skipif(shutil.which('nextflow') is None, reason='Nextflow unavailable')
def test_dependency_failure_stops_before_cache_or_detector(tmp_path):
    project, _, _, _, log, result = _fake_project(tmp_path, fail_preflight=True)
    assert result.returncode != 0
    assert 'SYNTHETIC dependency failure' in result.stdout + result.stderr
    assert [json.loads(line)['phase'] for line in log.read_text().splitlines()] == ['preflight']
    assert not (project / 'cache that must follow preflight').exists()
    assert not (project / 'published').exists()


@pytest.mark.skipif(shutil.which('nextflow') is None, reason='Nextflow unavailable')
def test_missing_sibling_python_has_actionable_error_and_no_generic_fallback(tmp_path):
    project, _, _, _, log, result = _fake_project(tmp_path, missing_python=True)
    assert result.returncode != 0
    assert 'Set --cellvit_python' in result.stdout + result.stderr
    assert not log.exists()
    assert not (project / 'cache that must follow preflight').exists()
