"""CPU-only end-to-end checks for the versioned feature storage alternative."""
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from test_cellvit_archive_consumer import _bundle
from cellvit_models import digest_file
from cellvit_shards import (
    ARCHIVE_NAME, FEATURE_CSV, FEATURE_MATRIX_DIR, FEATURE_MATRIX_RECEIPT,
    compact_shard, finalize_shards, materialize_shard_records,
    replace_feature_csv, verify_complete_shard,
)
from export_cellvit_partition_experiment import export_partition_experiment
from export_experiment import validate_experiment
from index_cellvit_partitions import build_partition_index, validate_partition_index
from index_cell_classification import query_index


@pytest.mark.parametrize('archive', (False, True))
@pytest.mark.parametrize('table_name', ('cellvit_cell_classification.csv.gz',
                                        'cellvit_cell_classification.csv'))
def test_matrix_checkpoint_index_project_reopen(tmp_path, archive, table_name):
    root, image, plan, shard, directory = _bundle(
        tmp_path, table_name, feature_value=.5)
    original_sha = digest_file(directory / FEATURE_CSV)
    checkpoint_sha = digest_file(directory / 'complete.json')
    receipt = replace_feature_csv(plan, shard, directory)
    assert receipt['source_csv_sha256'] == original_sha
    assert not (directory / FEATURE_CSV).exists()
    assert digest_file(directory / 'complete.json') == checkpoint_sha
    assert replace_feature_csv(plan, shard, directory) == receipt
    with materialize_shard_records(directory) as materialized:
        assert digest_file(materialized / FEATURE_CSV) == original_sha
    if archive:
        compact_shard(plan, shard, directory)
        assert (directory / ARCHIVE_NAME).is_file()
        assert not (directory / FEATURE_MATRIX_DIR).exists() or not any((directory / FEATURE_MATRIX_DIR).iterdir())
        assert not (directory / FEATURE_MATRIX_RECEIPT).exists()
        with materialize_shard_records(directory) as materialized:
            assert digest_file(materialized / FEATURE_CSV) == original_sha
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 1
    manifest = finalize_shards(plan, root)
    storage = manifest['partitions'][0]['feature_storage']
    assert storage['schema'] == 'cellphenotyper-cellvit-feature-storage-v1'
    assert storage['source_csv_sha256'] == original_sha
    assert all(ref['storage'] == ('archive-member' if archive else 'file')
               for ref in storage['references'].values())
    index = root / 'cells.sqlite'
    metadata = build_partition_index(root / 'cellvit_partition_manifest.json', image, index)
    assert metadata['cell_count'] == 1
    assert validate_partition_index(index, root / 'cellvit_partition_manifest.json',
                                    index.with_suffix('.features.json'), image) == metadata
    assert metadata['feature_storage'] == 'partition-row-reference-v1'
    assert query_index(index, mode='review_subset', cell_ids=['cell-1'])['features'] == {'cell-1': [.5]}
    project_path = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(root / 'cellvit_partition_manifest.json',
        image, index, project_path, name='Matrix fixture', tissue_id='fixture')
    assert validate_experiment(project_path) == project
    saved = json.loads(project_path.read_text())
    assert saved['tissues'][0]['cell_classification']['cell_count'] == 1
    assert saved['tissues'][0]['cell_classification']['index']['format'] == 'sqlite'
    assert json.loads(index.with_suffix('.features.json').read_text())['schema'] == 'cellphenotyper-partitioned-cell-features-v2'
    moved = tmp_path / 'moved'
    shutil.copytree(root, moved)
    assert validate_experiment(moved / project_path.name)['experiment_id'] == project['experiment_id']
    assert query_index(moved / index.name, mode='review_subset', cell_ids=['cell-1'])['features'] == {'cell-1': [.5]}


def test_non_float32_input_keeps_original_checkpoint(tmp_path):
    root, _, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz')
    source = directory / FEATURE_CSV
    sha = digest_file(source)
    with pytest.raises(ValueError, match='loses precision'):
        replace_feature_csv(plan, shard, directory)
    assert digest_file(source) == sha
    assert not (directory / FEATURE_MATRIX_RECEIPT).exists()
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 1


def test_matrix_corruption_fails_closed(tmp_path):
    root, image, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    replace_feature_csv(plan, shard, directory)
    matrix = directory / FEATURE_MATRIX_DIR / 'features.f32.gz'
    original = matrix.read_bytes()
    matrix.write_bytes(original[:-4])
    with pytest.raises(ValueError, match='Feature matrix member differs'):
        verify_complete_shard(plan, shard, directory)
    with pytest.raises(ValueError, match='Feature matrix member differs'):
        finalize_shards(plan, root)
    assert not (root / 'cellvit_partition_manifest.json').exists()
    matrix.write_bytes(original)
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 1


def test_interrupted_bundle_without_receipt_is_recoverable(tmp_path):
    root, _, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    from feature_matrix import pack
    source = directory / FEATURE_CSV
    sha = digest_file(source)
    pack(source, directory / FEATURE_MATRIX_DIR,
         directory / 'cellvit_cell_features.manifest.json')
    assert (directory / FEATURE_MATRIX_DIR).is_dir()
    assert not (directory / FEATURE_MATRIX_RECEIPT).exists()
    replace_feature_csv(plan, shard, directory)
    assert not source.exists()
    with materialize_shard_records(directory) as materialized:
        assert digest_file(materialized / FEATURE_CSV) == sha


def test_corrupt_archived_matrix_rejected_before_index(tmp_path):
    root, image, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    replace_feature_csv(plan, shard, directory)
    compact_shard(plan, shard, directory)
    finalize_shards(plan, root)
    archive = directory / ARCHIVE_NAME
    original = archive.read_bytes()
    archive.write_bytes(original[:-9])
    with pytest.raises(ValueError, match='archive receipt or bytes'):
        build_partition_index(root / 'cellvit_partition_manifest.json', image, root / 'cells.sqlite')
    assert not (root / 'cells.sqlite').exists()
    archive.write_bytes(original)
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 1


def test_feature_storage_pointer_must_match_checked_members(tmp_path):
    root, image, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    replace_feature_csv(plan, shard, directory)
    manifest = finalize_shards(plan, root)
    manifest['partitions'][0]['feature_storage']['references'][
        'cellvit_feature_matrix/features.f32.gz']['sha256'] = '0' * 64
    (root / 'cellvit_partition_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='feature storage does not resolve'):
        build_partition_index(root / 'cellvit_partition_manifest.json', image, root / 'cells.sqlite')
    assert not (root / 'cells.sqlite').exists()


def test_reference_row_and_id_mismatch_fail_training_retrieval(tmp_path):
    root, image, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    replace_feature_csv(plan, shard, directory)
    finalize_shards(plan, root)
    index = root / 'cells.sqlite'
    build_partition_index(root / 'cellvit_partition_manifest.json', image, index)
    with sqlite3.connect(index) as connection:
        connection.execute('UPDATE features SET values_json=?',
                           (json.dumps({'partition_id': shard['id'], 'row': 1}),))
    with pytest.raises(ValueError, match='outside its partition'):
        query_index(index, mode='review_subset', cell_ids=['cell-1'])
    with sqlite3.connect(index) as connection:
        connection.execute('UPDATE features SET values_json=?',
                           (json.dumps({'partition_id': shard['id'], 'row': 0}),))
        connection.execute("UPDATE cells SET source_id='wrong-source'")
    with pytest.raises(ValueError, match='ID or coordinates differ'):
        query_index(index, mode='review_subset', cell_ids=['cell-1'])


def test_legacy_version2_json_vector_query_remains_readable(tmp_path):
    root, image, plan, shard, directory = _bundle(
        tmp_path, 'cellvit_cell_classification.csv.gz', feature_value=.5)
    finalize_shards(plan, root)
    index = root / 'cells.sqlite'
    build_partition_index(root / 'cellvit_partition_manifest.json', image, index)
    with sqlite3.connect(index) as connection:
        connection.execute('UPDATE features SET values_json=?', (json.dumps([.5]),))
        connection.execute('UPDATE metadata SET value=? WHERE key=?', (json.dumps(2), 'schema_version'))
        connection.execute('UPDATE metadata SET value=? WHERE key=?',
                           (json.dumps('ordered-server-only-JSON-vector-per-cell'), 'feature_storage'))
    assert query_index(index, mode='review_subset', cell_ids=['cell-1'])['features'] == {'cell-1': [.5]}
