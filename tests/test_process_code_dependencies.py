"""Actual Groovy source inventory checked against independent Python AST reads."""
import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def inventory(tmp_path_factory):
    if not shutil.which('nextflow'):
        pytest.skip('Nextflow unavailable')
    project = tmp_path_factory.mktemp('process_code_inventory')
    (project / 'lib').mkdir()
    for name in ('ProcessCode.groovy', 'PipelineHelpers.groovy'):
        shutil.copyfile(ROOT / 'lib' / name, project / 'lib' / name)
    parameters = dict(re.findall(r"^\s*(\w+_script)\s*=\s*'([^']*)'", (ROOT / 'nextflow.config').read_text(), re.M))
    modules = sorted(path.stem for path in (ROOT / 'modules').glob('*.nf'))
    output = project / 'inventory.json'
    parameters.update(source_root=str(ROOT), modules=modules, inventory=str(output))
    (project / 'params.json').write_text(json.dumps(parameters))
    (project / 'probe.nf').write_text('''nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
    def result = [:]
    params.modules.each { module ->
        result[module] = [files: ProcessCode.dependencies(file(params.source_root), module, params).collect { it.toString() },
                          fingerprint: ProcessCode.fingerprint(file(params.source_root), module, params)]
    }
    file(params.inventory).text = JsonOutput.toJson(result)
}
''')
    run = subprocess.run(['nextflow', '-log', str(project / 'nextflow.log'), 'run', str(project / 'probe.nf'),
        '-params-file', str(project / 'params.json'), '-ansi-log', 'false'], cwd=project,
        env=dict(os.environ, NXF_OFFLINE='true'), capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads(output.read_text()), parameters


def test_all_module_sources_and_configured_entrypoints_are_bound(inventory):
    records, parameters = inventory
    assert len(records) == len(list((ROOT / 'modules').glob('*.nf')))
    for module, record in records.items():
        files = set(map(Path, record['files']))
        assert re.fullmatch('[0-9a-f]{64}', record['fingerprint'])
        assert ROOT / 'modules' / f'{module}.nf' in files
        assert ROOT / 'lib/ProcessCode.groovy' in files
        assert ROOT / 'lib/PipelineHelpers.groovy' in files
        source = (ROOT / 'modules' / f'{module}.nf').read_text()
        keys = re.findall(r'\$\{projectDir\}/\$\{params\.(\w+_script)\}', source)
        for key in keys:
            assert (ROOT / parameters[key]).resolve() in files, (module, key)
        assert any(path.parent == ROOT / 'bin' for path in files), module
        assert not any(path.name.endswith('.env') for path in files)


def local_python_imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            modules = [node.module] if node.module else [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            continue
        for name in modules:
            relative = name.replace('.', '/')
            candidates = [path.parent / (relative + '.py'), ROOT / 'bin' / (relative + '.py'),
                          path.parent / relative / '__init__.py', ROOT / 'bin' / relative / '__init__.py']
            existing = next((value.resolve() for value in candidates if value.is_file()), None)
            if existing:
                yield existing


def test_python_local_dependency_closure_matches_ast(inventory):
    records, _ = inventory
    for module, record in records.items():
        files = set(map(Path, record['files']))
        for path in files:
            if path.suffix == '.py':
                for dependency in local_python_imports(path):
                    assert dependency in files, (module, path.name, dependency.name)


def test_r_helpers_and_static_resource_schemas_are_bound(inventory):
    records, _ = inventory
    for module, record in records.items():
        files = set(map(Path, record['files']))
        for path in files:
            if path.suffix != '.R':
                continue
            for name in re.findall(r'''["']([^"'\n]+\.R)["']''', path.read_text()):
                candidate = ROOT / 'bin' / name
                if candidate.is_file():
                    assert candidate.resolve() in files, (module, name)
    assert ROOT / 'bin/uni2_embedding_io.R' in set(map(Path, records['run_kodama_analysis']['files']))
    assert ROOT / 'bin/kodama_graph_clustering.R' in set(map(Path, records['run_rcode_clustering']['files']))


def test_unrelated_atlas_sources_do_not_invalidate_image_conversion(inventory):
    records, _ = inventory
    names = {Path(path).name for path in records['prepare_input_ometiff']['files']}
    assert 'cell_reference_atlas.py' not in names
    assert 'export_spatialdata.py' not in names


def test_cohort_export_binds_consumer_without_fitting_runtime(inventory):
    records, _ = inventory
    names = {Path(path).name for path in records['export_spatialdata']['files']}
    assert 'cohort_niche_io.py' in names
    assert 'fit_cohort_niches.py' not in names
    assert 'analyze_cell_neighborhoods.py' not in names


def test_dependency_identity_and_directory_safety(tmp_path):
    """Exercise production Groovy with source edits and HF-style blob links."""
    if not shutil.which('nextflow'):
        pytest.skip('Nextflow unavailable')
    original = tmp_path / 'original'
    for name in ('modules', 'lib', 'bin', 'resources'):
        (original / name).mkdir(parents=True)
    for name in ('ProcessCode.groovy', 'PipelineHelpers.groovy'):
        shutil.copyfile(ROOT / 'lib' / name, original / 'lib' / name)
    (original / 'modules/probe.nf').write_text('''process PROBE {
    script:
    """python "${projectDir}/${params.probe_script}" --token-file "${projectDir}/${params.hf_token_env_file}""""
}
''')
    (original / 'bin/entry.py').write_text('from helper import value\n')
    (original / 'bin/helper.py').write_text('import leaf\nvalue = 1\n')
    (original / 'bin/leaf.py').write_text('value = 1\n')
    (original / 'credentials.json').write_text('{"fixture": "not-a-real-token-A"}\n')
    changed_code, changed_secret, unchanged = (tmp_path / name for name in ('code', 'secret', 'unchanged'))
    for path in (changed_code, changed_secret, unchanged):
        shutil.copytree(original, path, copy_function=shutil.copy2)
    for path, old, new in ((changed_code / 'bin/leaf.py', '1', '2'),
                           (changed_secret / 'credentials.json', 'token-A', 'token-B')):
        metadata = path.stat()
        path.write_text(path.read_text().replace(old, new))
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        assert path.stat().st_size == metadata.st_size
        assert path.stat().st_mtime_ns == metadata.st_mtime_ns

    snapshot_a, snapshot_b = tmp_path / 'snapshot_a', tmp_path / 'snapshot_b'
    snapshot_a.mkdir()
    snapshot_b.mkdir()
    blob_a, blob_b = tmp_path / 'blob_a', tmp_path / 'blob_b'
    blob_a.write_bytes(b'weight-A')
    blob_b.write_bytes(b'weight-B')
    stat = blob_a.stat()
    os.utime(blob_b, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    (snapshot_a / 'weights.bin').symlink_to(blob_a)
    (snapshot_b / 'weights.bin').symlink_to(blob_b)
    alias = tmp_path / 'snapshot_alias'
    alias.symlink_to(snapshot_a, target_is_directory=True)
    escape, cycle = tmp_path / 'directory_escape', tmp_path / 'directory_cycle'
    escape.mkdir()
    cycle.mkdir()
    (escape / 'foreign').symlink_to(snapshot_a, target_is_directory=True)
    (cycle / 'back').symlink_to(cycle, target_is_directory=True)
    placeholder = tmp_path / 'placeholder.txt'
    placeholder.write_text('No optional directory provided\n')
    project = tmp_path / 'probe'
    (project / 'lib').mkdir(parents=True)
    for name in ('ProcessCode.groovy', 'PipelineHelpers.groovy'):
        shutil.copyfile(ROOT / 'lib' / name, project / 'lib' / name)
    paths = {name: str(value) for name, value in locals().copy().items() if name in {
        'original', 'changed_code', 'changed_secret', 'unchanged', 'snapshot_a',
        'snapshot_b', 'alias', 'escape', 'cycle', 'placeholder'}}
    (project / 'params.json').write_text(json.dumps(paths))
    (project / 'probe.nf').write_text('''nextflow.enable.dsl=2
workflow {
    def options = [probe_script: 'bin/entry.py', hf_token_env_file: 'credentials.json']
    def original = ProcessCode.fingerprint(file(params.original), 'probe', options)
    assert original == ProcessCode.fingerprint(file(params.unchanged), 'probe', options)
    assert original != ProcessCode.fingerprint(file(params.changed_code), 'probe', options)
    assert original == ProcessCode.fingerprint(file(params.changed_secret), 'probe', options)
    def names = ProcessCode.dependencies(file(params.original), 'probe', options).collect { it.fileName.toString() }
    assert names.containsAll(['entry.py', 'helper.py', 'leaf.py'])
    assert !names.contains('credentials.json')
    [[:], [probe_script: 'missing.py'], [probe_script: 42]].each { invalid ->
        def rejected = false
        try { ProcessCode.fingerprint(file(params.original), 'probe', invalid) }
        catch (IllegalArgumentException expected) { rejected = true }
        assert rejected
    }
    def weights = ProcessCode.directoryFingerprint([file(params.snapshot_a)])
    assert weights == ProcessCode.directoryFingerprint([file(params.alias)])
    assert weights == ProcessCode.directoryFingerprint([new nextflow.processor.TaskPath(file(params.alias), 'model_snapshot')])
    assert weights != ProcessCode.directoryFingerprint([file(params.snapshot_b)])
    assert ProcessCode.directoryFingerprint([file(params.placeholder)]) == ProcessCode.directoryFingerprint([])
    // Portable profile receipts still reject external file links by default.
    def strictRejected = false
    try { PipelineHelpers.contentFingerprint([file(params.snapshot_a)]) }
    catch (IllegalArgumentException expected) { strictRejected = true }
    assert strictRejected
    [params.escape, params.cycle].each { path ->
        def rejected = false
        try { ProcessCode.directoryFingerprint([file(path)]) }
        catch (IllegalArgumentException expected) { rejected = true }
        assert rejected
    }
    println 'DEPENDENCY_AND_DIRECTORY_CHECKS_PASSED'
}
''')
    run = subprocess.run(['nextflow', '-log', str(project / 'nextflow.log'), 'run', str(project / 'probe.nf'),
        '-params-file', str(project / 'params.json'), '-ansi-log', 'false'], cwd=project,
        env=dict(os.environ, NXF_OFFLINE='true'), capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'DEPENDENCY_AND_DIRECTORY_CHECKS_PASSED' in run.stdout
