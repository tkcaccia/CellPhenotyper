import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("sample_id", [".", ".."])
def test_standalone_nextflow_rejects_path_segment_sample_ids(tmp_path, sample_id):
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps([{"sample_id": sample_id}]))
    environment = dict(os.environ, NXF_OFFLINE="true")
    result = subprocess.run([nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(ROOT / "cell_profiles.nf"),
                             "-ansi-log", "false", "-work-dir", str(tmp_path / "work"),
                             "--cell_profile_samples", str(manifest), "--outdir_base", str(tmp_path / "output")],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "must not be dot path segments" in result.stdout + result.stderr
    assert not (tmp_path / "output/19_cell_profiles").exists()


def test_profile_process_binds_canonical_raster_image_and_support():
    module = (ROOT / "modules/build_spatial_cell_profiles.nf").read_text()
    assert "--image '${image_tif}' --labels '${labels_tif}' --tissue-mask '${tissue_mask}'" in module
    assert "--compartment-qc" in module and "--domain-uncertainty" in module


def test_reference_mapping_and_spatial_export_remain_separate_products():
    workflow = (ROOT / "subworkflows/run_cell_profile_atlas.nf").read_text()
    assert "failOnDuplicate: true, failOnMismatch: true" in workflow
    assert "MAP_CELL_REFERENCE_ATLAS(" in workflow
    assert "EXPORT_SPATIALDATA(exportInputs, runtime_plan)" in workflow
    mapping = (ROOT / "modules/map_cell_reference_atlas.nf").read_text()
    assert "cell_reference_atlas.py\" map" in mapping
    assert "cell_reference_atlas.py\" build" not in mapping


def test_main_rejects_silently_ignored_cell_atlas_before_scheduling(tmp_path):
    nextflow = shutil.which('nextflow')
    if not nextflow:
        pytest.skip('Nextflow unavailable')
    result = subprocess.run([nextflow, '-log', str(tmp_path / 'nextflow.log'),
        'run', str(ROOT / 'main.nf'), '-ansi-log', 'false',
        '-work-dir', str(tmp_path / 'work'), '--cell_reference_atlas', str(tmp_path / 'not_read'),
        '--cell_profiles_enable', 'false', '--outdir_base', str(tmp_path / 'output')],
        cwd=tmp_path, env=dict(os.environ, NXF_OFFLINE='true'), capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert 'cell_reference_atlas requires cell_profiles_enable=true' in result.stdout + result.stderr
    # The existing onComplete hook may still write failure/execution reports.
    assert not (tmp_path / 'output/21_reference_mapping').exists()
    assert not list((tmp_path / 'work').rglob('.command.sh'))
