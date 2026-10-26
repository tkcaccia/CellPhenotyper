"""CPU-only archive, index and project reopen contract on disposable data."""
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from cellvit_models import digest_file
from cellvit_shards import (ARCHIVE_NAME, ARCHIVE_RECEIPT, SHARD_FILES, atomic_json,
                            compact_shard, complete_shard, finalize_shards,
                            materialize_shard_records, plan_shards, verify_complete_shard)
from export_cellvit_partition_experiment import export_partition_experiment
from export_experiment import validate_experiment
from index_cellvit_partitions import build_partition_index, validate_partition_index
from cellvit_shards import CLASSIFICATION_HEADER


def _write(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + '\n')


def _bundle(tmp_path, table_name, feature_value=.4):
    root = tmp_path / 'bundle'
    root.mkdir()
    image = root / 'image.png'
    Image.new('RGB', (64, 64), (177, 132, 167)).save(image)
    shift = root / 'shift.json'
    _write(shift, {'source_mpp': .25})
    plan = plan_shards(image, shift, core_size=1024, halo=960)
    atomic_json(root / 'cellvit_shard_plan.json', plan)
    shard = plan['shards'][0]
    directory = root / 'shards' / shard['id']
    directory.mkdir(parents=True)
    frame = plan['image']
    cell = {'cell_id': 'cell-1', 'source_id': 'native-1', 'x': 10.0, 'y': 12.0,
            'contour': [[9, 11], [11, 11], [10, 13]],
            'predictions': {'head': {'status': 'predicted', 'class_id': 1,
                                     'label': 'Tumor', 'probability': .8,
                                     'probabilities': [.2, .8]}}}
    feature_table = directory / 'cellvit_cell_features.csv.gz'
    with feature_table.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['cell_id', 'source_id', 'x', 'y', 'feat_1'])
                writer.writerow(['cell-1', 'native-1', 10, 12, feature_value])
    feature = {'kind': 'cellvit-cell', 'observation_unit': 'cell',
               'coordinate_frame': 'image', 'frame_id': frame['frame_id'],
               'feature_set_id': 'synthetic-cell-tokens',
               'id_column': 'cell_id', 'x_column': 'x', 'y_column': 'y',
               'path': feature_table.name, 'sha256': digest_file(feature_table),
               'format': 'csv.gz', 'feature_columns': ['feat_1'], 'row_count': 1,
               'descriptor': {'backbone': 'SAM', 'checkpoint_sha256': 'a'*64,
                   'runtime_version': 'synthetic', 'feature_definition': 'fixture',
                   'feature_count': 1, 'coordinate_frame': 'image',
                   'observation_unit': 'cell',
                   'sampling': {'model_input_mpp': .25, 'patch_size_px': 1024,
                                'patch_overlap_px': 64, 'token_size_px': 16},
                   'preprocessing': {'runtime_wheel_sha256': 'f'*64,
                                     'normalization': 'fixture', 'resampling': 'none',
                                     'token_provenance': 'analytic fixture'}}}
    metadata_cells = directory / 'cellvit_feature_cells.json'
    _write(metadata_cells, {'image': frame, 'cells': [
        {key: value for key, value in cell.items() if key != 'predictions'}]})
    feature_manifest = directory / 'cellvit_cell_features.manifest.json'
    _write(feature_manifest, {'schema': 'cellphenotyper-cell-features',
        'schema_version': 1, 'image': frame, 'feature_set': feature,
        'cell_metadata': {'path': metadata_cells.name, 'sha256': digest_file(metadata_cells)}})
    model = {'model_id': 'head', 'kind': 'mlp', 'status': 'available',
             'taxonomy': 'synthetic', 'backbone': 'SAM', 'checkpoint_sha256': 'b'*64,
             'class_names': ['Other', 'Tumor'],
             'score_semantics': 'uncalibrated_model_probability'}
    _write(directory / 'cellvit_cell_classification.json', {
        'schema': 'cellphenotyper-cell-classification', 'schema_version': 1,
        'image': frame, 'feature_set': feature, 'models': [model], 'cells': [cell],
        'provenance': {'feature_manifest_sha256': digest_file(feature_manifest)}})
    _write(directory / 'cellvit_cells.json', {'cells': [
        {'id': 'native-1', 'cell_id': 'cell-1', 'centroid': [10.0, 12.0],
         'contour': cell['contour']}]})
    _write(directory / 'cellvit_metadata.json', {
        'model': 'SAM', 'requested_taxonomies': ['synthetic'],
        'model_provenance': {'checkpoints': [{'sha256': 'a'*64}]}})
    for name in SHARD_FILES:
        if not (directory / name).exists():
            _write(directory / name, {})
    path = directory / table_name
    opener = gzip.open if table_name.endswith('.gz') else open
    with opener(path, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(CLASSIFICATION_HEADER)
        writer.writerow(['cell-1', 'native-1', 10, 12, 'head', 'mlp', 'SAM',
                         'b'*64, 'predicted', 1, 'Tumor', .8, '[0.2, 0.8]',
                         'uncalibrated_model_probability', ''])
    complete_shard(plan, shard, directory)
    return root, image, plan, shard, directory


@pytest.mark.parametrize('table_name', ('cellvit_cell_classification.csv.gz',
                                         'cellvit_cell_classification.csv'))
@pytest.mark.parametrize('archive', (False, True))
def test_complete_to_index_and_portable_project_reopen(tmp_path, table_name, archive):
    root, image, plan, shard, directory = _bundle(tmp_path, table_name)
    if archive:
        compact_shard(plan, shard, directory)
        assert not (directory / 'core_cells.jsonl').exists()
    manifest = finalize_shards(plan, root)
    entry = manifest['partitions'][0]
    for field in ('classification', 'features', 'segmentation', 'global_cell_index'):
        if archive:
            assert entry[field]['storage'] == 'archive-member'
            assert entry[field]['archive'] == f'shards/{shard["id"]}/{ARCHIVE_NAME}'
            assert not (root / 'shards' / shard['id'] / entry[field]['member']).exists()
        else:
            assert (root / entry[field]).is_file()
    table_ref = entry['classification_table']
    assert table_ref['format'] == ('csv.gz' if table_name.endswith('.gz') else 'csv')
    if archive:
        assert table_ref['path']['member'] == table_name
        with materialize_shard_records(directory) as materialized:
            assert digest_file(materialized / table_name) == table_ref['sha256']
    else:
        assert (root / table_ref['path']).is_file()
    index = root / 'cells.sqlite'
    metadata = build_partition_index(root / 'cellvit_partition_manifest.json', image, index)
    assert metadata['cell_count'] == metadata['prediction_count'] == 1
    assert validate_partition_index(index, root / 'cellvit_partition_manifest.json',
                                    index.with_suffix('.features.json'), image) == metadata
    project_path = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(root / 'cellvit_partition_manifest.json',
        image, index, project_path, name='Archive fixture', tissue_id='fixture')
    assert validate_experiment(project_path) == project
    moved = tmp_path / 'moved'
    shutil.copytree(root, moved)
    assert validate_experiment(moved / project_path.name)['experiment_id'] == project['experiment_id']


def test_corrupt_archive_and_receipt_are_rejected(tmp_path):
    root, image, plan, shard, directory = _bundle(tmp_path, 'cellvit_cell_classification.csv.gz')
    compact_shard(plan, shard, directory)
    archive = directory / ARCHIVE_NAME
    original = archive.read_bytes()
    archive.write_bytes(original[:-8])
    with pytest.raises(ValueError, match='archive receipt or bytes'):
        verify_complete_shard(plan, shard, directory)
    archive.write_bytes(original)
    receipt_path = directory / ARCHIVE_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    receipt['members']['core_cells.jsonl']['sha256'] = '0'*64
    _write(receipt_path, receipt)
    with pytest.raises(ValueError, match='does not preserve checked record'):
        verify_complete_shard(plan, shard, directory)


def test_interrupted_archive_recovery_and_idempotence(tmp_path):
    root, _, plan, shard, directory = _bundle(tmp_path, 'cellvit_cell_classification.csv.gz')
    temporary = directory / ('.' + ARCHIVE_NAME + '.tmp')
    temporary.write_bytes(b'incomplete')
    with pytest.raises(FileExistsError, match='temporary'):
        compact_shard(plan, shard, directory)
    assert (directory / 'core_cells.jsonl').exists()
    temporary.unlink()
    files = sorted(path for path in directory.iterdir() if path.is_file() and
                   path.name not in ('complete.json', ARCHIVE_NAME, ARCHIVE_RECEIPT))
    with tarfile.open(directory / ARCHIVE_NAME, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=path.name, recursive=False)
    first = compact_shard(plan, shard, directory)
    second = compact_shard(plan, shard, directory)
    assert first == second
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 1


def test_published_manifest_rejects_stale_archived_record_path(tmp_path):
    root, image, plan, shard, directory = _bundle(tmp_path, 'cellvit_cell_classification.csv.gz')
    compact_shard(plan, shard, directory)
    manifest = finalize_shards(plan, root)
    manifest['partitions'][0]['classification'] = f'shards/{shard["id"]}/cellvit_cell_classification.json'
    atomic_json(root / 'cellvit_partition_manifest.json', manifest)
    with pytest.raises(ValueError, match='does not resolve to checked record'):
        build_partition_index(root / 'cellvit_partition_manifest.json', image, root / 'cells.sqlite')
