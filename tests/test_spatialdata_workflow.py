"""Actual export-only Nextflow execution; no model inference or fabricated accuracy."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sd = pytest.importorskip('spatialdata')
import numpy as np
import pandas as pd
from test_spatialdata_export import specimen, measured_package, measured_region_package, affine
from test_spatialdata_hierarchy import hierarchy_specimen, refresh_hierarchy
from cell_profile_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def safe_specimen(specimen, tmp_path):
    inputs = specimen[0]
    profile = inputs['profile_dir']
    cells = pd.read_parquet(profile / 'cell_profiles.parquet')
    cells['sample_id'] = 'sample_A'
    cells.to_parquet(profile / 'cell_profiles.parquet', index=False)
    cells.to_csv(profile / 'cell_profiles.csv', index=False)
    cells[['sample_id', 'cell_id', 'cell_uid']].to_csv(profile / 'feature_rows.csv', index=False)
    manifest_path = profile / 'cell_profiles_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sample_id'] = 'sample_A'
    manifest['files'] = {name: sha256_file(profile / name) for name in manifest['files']}
    manifest_path.write_text(json.dumps(manifest))
    resolution = tmp_path / 'resolution.json'
    resolution.write_text(json.dumps({'status': 'pass', 'mpp_x': .5, 'mpp_y': .5, 'width_px': 256, 'height_px': 512}))
    row = {'sample_id': 'sample_A', 'existing_profiles': str(profile),
           'image': str(inputs['image']), 'labels': str(inputs['labels']), 'shift': str(inputs['shift']),
           'resolution': str(resolution), 'tissue_geojson': str(inputs['tissue_geojson']), 'tissue_coordinates': 'crop_pixels'}
    return inputs, row


def run_workflow(row, tmp_path, *, export=True):
    nextflow = shutil.which('nextflow')
    if not nextflow:
        pytest.skip('Nextflow unavailable')
    manifest = tmp_path / 'samples.json'
    manifest.write_text(json.dumps([row]))
    return subprocess.run([nextflow, '-log', str(tmp_path / 'nextflow.log'), 'run', str(ROOT / 'cell_profiles.nf'),
        '-ansi-log', 'false', '-work-dir', str(tmp_path / 'work'), '--cell_profile_samples', str(manifest),
        '--outdir_base', str(tmp_path / 'output'), '--cell_profiles_spatialdata', str(export).lower(),
        '--spatialdata_python', sys.executable, '--spatialdata_tile_size', '16',
        '--cell_atlas_python', sys.executable,
        '--cell_profiles_cpus', '1', '--cell_profiles_memory_gb', '2'],
        cwd=tmp_path, env=dict(os.environ, NXF_OFFLINE='true'), capture_output=True, text=True, timeout=120)


def test_existing_profiles_and_two_measured_packages_roundtrip_through_nextflow(specimen, tmp_path):
    inputs, row = safe_specimen(specimen, tmp_path)
    packages = [measured_package(inputs, tmp_path / f'assay_{i}', f'mIF-{i}') for i in range(2)]
    row['measured_assays'] = [str(path) for path in packages]
    before = {str(path): sha256_file(path) for path in inputs['profile_dir'].rglob('*') if path.is_file()}
    result = run_workflow(row, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / 'output/20_spatialdata/sample_A/spatialdata.zarr'
    loaded = sd.SpatialData.read(output)
    measured = [table for name, table in loaded.tables.items() if name.startswith('measured_assay_')]
    assert len(measured) == 2
    for table in measured:
        assert table.X.dtype == np.dtype('float64')
        np.testing.assert_array_equal(table.X, [[16777217.25, np.nan], [np.nan, np.nan]])
        assert table.obs_names.tolist() == loaded.tables['cells'].obs_names.tolist()
    assert not (tmp_path / 'output/19_cell_profiles').exists()  # no reconstruction/model work
    assert before == {str(path): sha256_file(Path(path)) for path in before}


def test_measured_attachments_are_never_silently_ignored(specimen, tmp_path):
    inputs, row = safe_specimen(specimen, tmp_path)
    row['measured_assays'] = [str(measured_package(inputs, tmp_path / 'assay'))]
    result = run_workflow(row, tmp_path, export=False)
    assert result.returncode != 0
    assert 'must not be silently ignored' in result.stdout + result.stderr


def test_mixed_cell_and_region_assays_survive_relative_bundle_staging(specimen, tmp_path):
    """Exercise actual file staging/export, not biological assay validation."""
    from spatialdata.models import ShapesModel

    inputs, row = safe_specimen(specimen, tmp_path)
    cell_package = measured_package(inputs, tmp_path / 'cell_assay', 'measured-cells')
    region_source = tmp_path / 'original_region_sources'
    source_package, source_link, regions = measured_region_package(inputs, region_source)
    region_package = tmp_path / 'portable_region_package'
    shutil.copytree(source_package, region_package)
    bundle = tmp_path / 'portable_region_shapes'
    resources = bundle / 'resources'
    resources.mkdir(parents=True)
    link = json.loads(source_link.read_text())
    for name in ('shapes', 'registry_table', 'registry_manifest'):
        source = source_link.parent / link[name]['path']
        shutil.copyfile(source, resources / source.name)
        link[name]['path'] = f'resources/{source.name}'
    (bundle / 'link.json').write_text(json.dumps(link))
    # Provenance retains the old paths, which are now genuinely unavailable.
    # Only the explicitly declared portable files may serve as export inputs.
    region_source.rename(tmp_path / 'detached_original_region_sources')
    assert not region_source.exists()
    row['measured_assays'] = [str(cell_package.relative_to(tmp_path)), str(region_package.relative_to(tmp_path))]
    row['measured_region_shapes'] = [str(bundle.relative_to(tmp_path))]
    frozen = {path for folder in (inputs['profile_dir'], cell_package, region_package, bundle)
              for path in folder.rglob('*') if path.is_file()}
    frozen.update(Path(row[name]) for name in ('image', 'labels', 'shift', 'resolution', 'tissue_geojson'))
    before = {path: sha256_file(path) for path in frozen}

    result = run_workflow(row, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / 'output/20_spatialdata/sample_A/spatialdata.zarr'
    loaded = sd.SpatialData.read(output)
    records = {record['assay_id']: record for record in loaded.attrs['cellphenotyper']['measured_modalities']}
    assert set(records) == {'measured-cells', 'registered-regions'}
    cell_record, region_record = records['measured-cells'], records['registered-regions']
    cell_table = loaded.tables[cell_record['table']]
    region_table = loaded.tables[region_record['table']]
    shapes = loaded.shapes[region_record['spatial_element']]
    ShapesModel.validate(shapes)

    assert cell_record['observation_unit'] == 'cell'
    assert cell_record['spatial_element'] == 'canonical_cells'
    assert cell_table.obs_names.tolist() == loaded.tables['cells'].obs_names.tolist()
    assert cell_table.obs.instance_id.tolist() == [1, 7]
    assert cell_table.obs.spatial_region.astype(str).eq('canonical_cells').all()
    assert cell_table.X.dtype == np.dtype('float64')
    np.testing.assert_array_equal(cell_table.X, [[16777217.25, np.nan], [np.nan, np.nan]])

    assert region_record['observation_unit'] == 'spatial_bin'
    assert region_record['identity_key'] == 'region_uid'
    assert region_record['spatial_element'].startswith('measured_regions_')
    assert region_record['spatial_element'] != cell_record['spatial_element']
    assert region_table.obs_names.tolist() == shapes.region_uid.tolist() == regions.region_uid.tolist()
    assert region_table.obs.instance_id.tolist() == list(shapes.index) == [1, 2, 3]
    assert region_table.obs.spatial_region.astype(str).eq(region_record['spatial_element']).all()
    assert 'cell_uid' not in region_table.obs and 'cell_id' not in region_table.obs
    assert region_table.X.dtype == np.dtype('float64')
    np.testing.assert_array_equal(region_table.X, [[np.nan, 2.5], [np.nan, np.nan], [16777217.25, np.nan]])
    assert region_table.obs.measured_status.astype(str).tolist() == ['matched', 'unmatched', 'matched']
    np.testing.assert_array_equal(region_table.obsm['spatial'], regions[['x_um', 'y_um']].to_numpy())
    np.testing.assert_array_equal(affine(shapes), [[.5, 0, 50], [0, .5, 100], [0, 0, 1]])
    transformed_centres = np.column_stack((shapes.geometry.centroid.x, shapes.geometry.centroid.y)) * .5 + [50., 100.]
    np.testing.assert_array_equal(transformed_centres, regions[['x_um', 'y_um']].to_numpy())
    np.testing.assert_array_equal(shapes.geometry.area.to_numpy() * .25, regions.area_um2.to_numpy())
    # Canonical profiles retain predicted values and the original two cells;
    # neither spatial bins nor independently measured values are folded in.
    assert loaded.tables['cells'].shape == (2, 0)
    assert loaded.tables['cells'].obs_names.tolist() == ['sampleA:seg:001', 'sampleA:seg:7']
    np.testing.assert_array_equal(loaded.tables['cells'].obs.predicted__nucleus__CD3__mean, [.1, np.nan])
    assert not any(column.startswith('measured__') for column in loaded.tables['cells'].obs)
    assert not (tmp_path / 'output/19_cell_profiles').exists()
    assert before == {path: sha256_file(path) for path in before}

    # Inspect actual staged task inputs, not only the original portable bundle.
    commands = list((tmp_path / 'work').rglob('.command.sh'))
    export_commands = [path for path in commands if 'export_spatialdata.py' in path.read_text()]
    assert len(export_commands) == 1
    task_dir = export_commands[0].parent
    assert '--measured-region-shapes' in export_commands[0].read_text()
    staged_links = list((task_dir / 'measured_regions').glob('bundle*/link.json'))
    assert len(staged_links) == 1
    staged_link = json.loads(staged_links[0].read_text())
    for name in ('shapes', 'registry_table', 'registry_manifest'):
        relative = Path(staged_link[name]['path'])
        assert not relative.is_absolute() and relative.parts[0] == 'resources'
        assert sha256_file(staged_links[0].parent / relative) == staged_link[name]['sha256']


def hierarchy_workflow_record(hierarchy_specimen):
    inputs = hierarchy_specimen[0]
    profile, hierarchy = inputs['profile_dir'], inputs['hierarchy_dir']
    cells = pd.read_parquet(profile / 'cell_profiles.parquet')
    cells['sample_id'] = 'sample_A'
    cells.to_parquet(profile / 'cell_profiles.parquet', index=False)
    cells.to_csv(profile / 'cell_profiles.csv', index=False)
    cells[['sample_id', 'cell_id', 'cell_uid']].to_csv(profile / 'feature_rows.csv', index=False)
    manifest_path = profile / 'cell_profiles_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['sample_id'] = 'sample_A'
    manifest['files'] = {name: sha256_file(profile / name) for name in manifest['files']}
    manifest_path.write_text(json.dumps(manifest))
    for relative in ('region_profiles/region_profiles.csv', 'region_profiles/feature_rows.csv', 'grid_subdomains.csv'):
        path = hierarchy / relative
        table = pd.read_csv(path, dtype={'region_id': str})
        table['sample_id'] = 'sample_A'
        table.to_csv(path, index=False)
    summary_path = hierarchy / 'hierarchy_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['sample_id'] = 'sample_A'
    summary_path.write_text(json.dumps(summary))
    refresh_hierarchy(hierarchy)
    row = {'sample_id': 'sample_A', 'existing_profiles': str(profile), 'hierarchy': str(hierarchy),
           'image': str(inputs['image']), 'labels': str(inputs['labels']), 'shift': str(inputs['shift']),
           'resolution': str(inputs['resolution_json']), 'tissue_geojson': str(inputs['tissue_geojson']),
           'tissue_coordinates': 'crop_pixels'}
    return inputs, row


def test_actual_nextflow_links_hierarchy_and_preserves_base_measured_registry(hierarchy_specimen, tmp_path):
    inputs, row = hierarchy_workflow_record(hierarchy_specimen)
    package = measured_package(inputs, tmp_path / 'measured_base_registry', 'base-registry-mIF')
    row['measured_assays'] = [str(package)]
    frozen = {path: sha256_file(path) for root in (inputs['profile_dir'], inputs['hierarchy_dir'], package)
              for path in root.rglob('*') if path.is_file()}
    result = run_workflow(row, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    linked = tmp_path / 'output/24_cell_tissue_links/sample_A/cell_profiles'
    derived = pd.read_parquet(linked / 'cell_profiles.parquet')
    original = pd.read_parquet(inputs['profile_dir'] / 'cell_profiles.parquet')
    pd.testing.assert_frame_equal(derived[original.columns], original)
    assert (linked / 'cell_hierarchy_overlaps.parquet').is_file()
    loaded = sd.SpatialData.read(tmp_path / 'output/20_spatialdata/sample_A/spatialdata.zarr')
    assert len(loaded.tables['cells']) == len(original) == 2
    assert len(loaded.tables['hierarchy_region_profiles']) == 3
    assert loaded.tables['cell_hierarchy_overlaps'].obs.cell_uid.duplicated().any()
    assay = [table for name, table in loaded.tables.items() if name.startswith('measured_assay_')]
    assert len(assay) == 1
    np.testing.assert_array_equal(assay[0].X, [[16777217.25, np.nan], [np.nan, np.nan]])
    assert frozen == {path: sha256_file(path) for path in frozen}
    assert not (tmp_path / 'output/19_cell_profiles').exists()
    commands = [path.read_text() for path in (tmp_path / 'work').rglob('.command.sh')]
    assert sum('link_cell_tissue_hierarchy.py' in command for command in commands) == 1
    assert sum('--hierarchy-dir' in command and 'export_spatialdata.py' in command for command in commands) == 1


def test_actual_nextflow_hierarchy_source_mismatch_fails_without_export(hierarchy_specimen, tmp_path):
    inputs, row = hierarchy_workflow_record(hierarchy_specimen)
    path = inputs['hierarchy_dir'] / 'hierarchy_summary.json'
    summary = json.loads(path.read_text())
    summary['inputs']['image']['sha256'] = 'a' * 64
    path.write_text(json.dumps(summary))
    result = run_workflow(row, tmp_path)
    assert result.returncode != 0
    assert 'image' in result.stdout + result.stderr
    assert not (tmp_path / 'output/20_spatialdata').exists()


def test_already_linked_profile_reexports_without_rebuilding_or_relinking(hierarchy_specimen, tmp_path):
    from link_cell_tissue_hierarchy import link_profiles
    inputs, row = hierarchy_workflow_record(hierarchy_specimen)
    linked = tmp_path / 'previous_linked_profiles'
    link_profiles(inputs['profile_dir'], inputs['labels'], inputs['hierarchy_dir'], linked, tile_size=16)
    row['existing_profiles'] = str(linked)
    before = {path: sha256_file(path) for path in linked.rglob('*') if path.is_file()}
    result = run_workflow(row, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / 'output/24_cell_tissue_links').exists()
    assert not (tmp_path / 'output/19_cell_profiles').exists()
    assert (tmp_path / 'output/20_spatialdata/sample_A/spatialdata.zarr').is_dir()
    assert before == {path: sha256_file(path) for path in before}


def test_existing_profile_specimen_mismatch_fails_before_export(specimen, tmp_path):
    inputs, row = safe_specimen(specimen, tmp_path)
    row['sample_id'] = 'different_specimen'
    before = {path: sha256_file(path) for path in inputs['profile_dir'].rglob('*') if path.is_file()}
    result = run_workflow(row, tmp_path)
    assert result.returncode != 0
    assert 'existing_profiles has a different specimen or observation unit' in result.stdout + result.stderr
    assert not (tmp_path / 'output/20_spatialdata').exists()
    assert not list((tmp_path / 'work').rglob('.command.sh'))
    assert before == {path: sha256_file(path) for path in before}
