#!/usr/bin/env python3
"""Stream completed CellViT partitions into one global-frame SQLite review index.

No global cells JSON, feature CSV, GPU, model import or whole-population Python
container is created. The source partition manifest and each checkpoint remain
immutable evidence; SQLite stores bounded individual rows for viewport queries.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from cellvit_shards import (ARCHIVE_NAME, ARCHIVE_RECEIPT, SCHEMA as PARTITION_SCHEMA,
                            _identity, materialize_shard_records, plan_shards_geometry,
                            validate_plan, verify_complete_shard,
                            verify_partition_record_locations)
from index_cell_classification import (SQL, SCHEMA, _cell, _descriptor, _header,
    _prediction, digest_file, read_classification_header, stream_classification)

VERSION = 3
POINTER_SCHEMA = 'cellphenotyper-partitioned-cell-features-v2'
LEGACY_POINTER_SCHEMA = 'cellphenotyper-partitioned-cell-features-v1'
ALL_SAM_TAXONOMIES = frozenset(('pannuke', 'consep', 'lizard', 'nucls_main',
                                'nucls_super', 'ocelot', 'midog', 'panoptils'))


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _ref(path, root):
    path = Path(path).resolve()
    return {'path': Path(os.path.relpath(path, root)).as_posix(), 'sha256': digest_file(path)}


def _load_source(partition_manifest, image_path, *, strict_location=True):
    partition_manifest = Path(partition_manifest).resolve()
    root = partition_manifest.parent
    manifest = json.loads(partition_manifest.read_text())
    plan = json.loads((root / 'cellvit_shard_plan.json').read_text())
    if strict_location:
        validate_plan(plan)
    elif (plan.get('schema') != PARTITION_SCHEMA or plan.get('identity_sha256') != _identity(plan) or
          plan.get('shards') != plan_shards_geometry(plan['image']['width'], plan['image']['height'],
                                                    plan['core_size'], plan['halo'])):
        raise ValueError('Relocated partition plan identity or geometry differs from source')
    if (manifest.get('schema') != PARTITION_SCHEMA or manifest.get('plan_sha256') != plan['identity_sha256'] or
            manifest.get('image') != plan['image'] or manifest.get('partition_count') != len(plan['shards']) or
            len(manifest.get('partitions', [])) != len(plan['shards'])):
        raise ValueError('Partition manifest does not match the immutable complete plan')
    image_path = Path(image_path).resolve()
    if (strict_location and image_path != Path(plan['image_path'])) or digest_file(image_path) != plan['image']['sha256']:
        raise ValueError('Partition image bytes do not match the authoritative global frame')
    return manifest, plan


def _offset_cell(cell, shard, global_image):
    """Translate all native XY geometry without changing any class evidence."""
    cell = copy.deepcopy(cell)
    ox, oy = shard['input'][:2]
    x, y = float(cell['x']) + ox, float(cell['y']) + oy
    x0, y0, x1, y1 = shard['core']
    if not (x0 <= x < x1 and y0 <= y < y1):
        raise ValueError('Partition classification contains a halo or misregistered core cell')
    cell['x'], cell['y'] = x, y
    cell['source_id'] = f"{shard['id']}:{cell['source_id']}"
    if cell.get('contour') is not None:
        cell['contour'] = [[float(px)+ox, float(py)+oy] for px, py in cell['contour']]
    if cell.get('bbox') is not None:
        bbox = cell['bbox']
        cell['bbox'] = [[float(bbox[0][0])+oy, float(bbox[0][1])+ox],
                        [float(bbox[1][0])+oy, float(bbox[1][1])+ox]]
    _cell(cell, global_image)
    return cell


def _validate_requested_models(plan, models):
    requested = plan['run_contract'].get('requested_taxonomies')
    if plan['run_contract'].get('model') == 'SAM':
        explicit_eight = (isinstance(requested, list) and
                          len(requested) == len(ALL_SAM_TAXONOMIES) and
                          set(requested) == ALL_SAM_TAXONOMIES)
        if (requested != ['all'] and not explicit_eight) or (
                {model['taxonomy'] for model in models} != ALL_SAM_TAXONOMIES or
                len(models) != len(ALL_SAM_TAXONOMIES) or
                any(model['status'] != 'available' for model in models)):
            raise ValueError('Full-slide SAM-H project requires all eight available CellViT taxonomies')


def _stream_partition(connection, root, directory, entry, shard, global_image, expected_models, expected_feature):
    classification_path = directory / 'cellvit_cell_classification.json'
    feature_manifest_path = directory / 'cellvit_cell_features.manifest.json'
    header = read_classification_header(classification_path)
    local_image, feature, models = _header(header, feature_manifest_path)
    _descriptor(feature)
    staged = root / 'staging' / shard['id'] / 'input.tif'
    if staged.exists() and local_image['sha256'] != digest_file(staged):
        raise ValueError('Staged shard image differs from classification frame')
    if ((local_image['width'], local_image['height']) !=
        (shard['input'][2]-shard['input'][0], shard['input'][3]-shard['input'][1]) or
        local_image['mpp_x'] != global_image['mpp_x'] or local_image['mpp_y'] != global_image['mpp_y']):
        raise ValueError('Partition local image dimensions/MPP do not register to the global frame')
    if expected_models is not None and header['models'] != expected_models:
        raise ValueError('CellViT taxonomies or checkpoints differ between partitions')
    signature = {key: feature[key] for key in ('kind','observation_unit','coordinate_frame','feature_set_id','feature_columns','descriptor')}
    if expected_feature is not None and signature != expected_feature:
        raise ValueError('CellViT feature representation differs between partitions')
    index_path = directory / 'core_cells.jsonl'
    index_stream = index_path.open(encoding='utf-8')
    count = 0
    try:
        for kind, _, raw in stream_classification(classification_path):
            if kind != 'cell':
                continue
            local = json.loads(raw)
            indexed_line = index_stream.readline()
            if not indexed_line:
                raise ValueError('Core index has fewer cells than its classification')
            indexed = json.loads(indexed_line)
            cell = _offset_cell(local, shard, global_image)
            if (indexed.get('cell_id') != cell['cell_id'] or indexed.get('source_id') != cell['source_id'] or
                    indexed.get('x') != cell['x'] or indexed.get('y') != cell['y'] or
                    indexed.get('partition_id') != shard['id'] or
                    indexed.get('predictions') != {mid: {key: cell['predictions'][mid].get(key)
                        for key in ('status', 'class_id', 'label', 'probability')}
                        for mid in cell['predictions']}):
                raise ValueError('Core index and translated CellViT classification differ')
            identity = _cell(cell, global_image)
            predicted = _prediction(cell, models)
            count += 1
            raw_global = _json(cell)
            cursor = connection.execute('INSERT INTO cells(cell_id,source_id,x,y,xmin,ymin,xmax,ymax,raw_json) '
                                        'VALUES(?,?,?,?,?,?,?,?,?)', (*identity, raw_global))
            row_id = cursor.lastrowid
            connection.execute('INSERT INTO bbox_rtree VALUES(?,?,?,?,?)',
                               (row_id, identity[4], identity[6], identity[5], identity[7]))
            connection.executemany('INSERT INTO predictions VALUES(?,?,?,?,?,?)',
                                   [(row_id,*prediction) for prediction in predicted])
        if index_stream.readline():
            raise ValueError('Core index has more cells than its classification')
    finally:
        index_stream.close()
    if count != entry['cell_count']:
        raise ValueError('Partition declared/actual cell counts differ')
    # The original CSV is read one row at a time, joined to the checked global
    # cell row through its immutable canonical ID; no global feature CSV is made.
    path = directory / feature['path']
    opener = gzip.open if path.name.endswith('.gz') else open
    rows = 0
    with opener(path, 'rt', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        columns = feature['feature_columns']
        if reader.fieldnames != ['cell_id','source_id','x','y',*columns]:
            raise ValueError('Partition feature columns are not the exact CellViT allowlist')
        for row in reader:
            found = connection.execute('SELECT row_id,source_id,x,y FROM cells WHERE cell_id=?',
                                       (row['cell_id'],)).fetchone()
            if found is None or found[1] != f"{shard['id']}:{row['source_id']}" or (
                    float(row['x'])+shard['input'][0], float(row['y'])+shard['input'][1]) != (found[2],found[3]):
                raise ValueError('Partition feature row differs from global cell identity/coordinates')
            values = [float(row[name]) for name in columns]
            if not all(math.isfinite(value) for value in values):
                raise ValueError('Partition features contain nonfinite values')
            # Store only an immutable partition/row locator. The checked source
            # matrix (or historical CSV) remains the sole retained vector copy.
            connection.execute('INSERT INTO features VALUES(?,?)',
                               (found[0], _json({'partition_id': shard['id'], 'row': rows})))
            rows += 1
    if rows != count or feature['row_count'] != count or digest_file(path) != feature['sha256']:
        raise ValueError('Partition feature population or hash differs from classification')
    return header, signature, count


def _verify_archive_entry(root, entry, directory):
    declared = entry.get('lossless_archive')
    receipt = directory / ARCHIVE_RECEIPT
    if bool(declared) != receipt.is_file():
        raise ValueError('Partition archive state differs from published manifest')
    if declared and (declared.get('path') != str((directory / ARCHIVE_NAME).relative_to(root)) or
                     declared.get('receipt') != str(receipt.relative_to(root)) or
                     declared.get('sha256') != digest_file(directory / ARCHIVE_NAME) or
                     declared.get('receipt_sha256') != digest_file(receipt)):
        raise ValueError('Partition archive or receipt differs from published manifest')


def build_partition_index(partition_manifest, image_path, output):
    partition_manifest, output = Path(partition_manifest).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(output.suffix + '.json').exists():
        raise FileExistsError('Choose a NEW partition index destination')
    manifest, plan = _load_source(partition_manifest, image_path)
    root, global_image = partition_manifest.parent, manifest['image']
    if output.parent != root:
        raise ValueError('Place the partition SQLite index beside its manifest for relocatable references')
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.partition-index-', suffix='.sqlite', dir=output.parent)
    os.close(handle)
    connection = None
    pointer = output.with_suffix('.features.json')
    try:
        connection = sqlite3.connect(temporary)
        connection.execute('PRAGMA cache_size=-8192')
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.executescript(SQL)
        connection.execute('CREATE VIRTUAL TABLE bbox_rtree USING rtree(row_id,xmin,xmax,ymin,ymax)')
        model_list = feature_signature = first_feature = None
        total = 0
        for entry, shard in zip(manifest['partitions'], plan['shards']):
            if (entry['id'] != shard['id'] or entry['core_xyxy'] != shard['core'] or
                    entry['input_xyxy'] != shard['input']):
                raise ValueError('Partition order/core geometry differs from plan')
            directory = root / 'shards' / shard['id']
            marker = verify_complete_shard(plan, shard, directory)
            if digest_file(directory / 'complete.json') != entry['checkpoint_sha256']:
                raise ValueError('Partition checkpoint changed after manifest publication')
            _verify_archive_entry(root, entry, directory)
            verify_partition_record_locations(root, entry, directory, marker)
            with materialize_shard_records(directory) as materialized:
                header, signature, count = _stream_partition(connection, root, materialized,
                    entry, shard, global_image, model_list, feature_signature)
            if model_list is None:
                model_list = header['models']
                feature_signature = signature
                first_feature = header['feature_set']
            total += count
            connection.commit()
        if total != manifest['cell_count']:
            raise ValueError('Global partition cell count differs from manifest')
        _validate_requested_models(plan, model_list)
        if plan['run_contract'].get('model') == 'SAM':
            descriptor = first_feature.get('descriptor', {})
            if (descriptor.get('feature_count') != 1280 or
                    len(first_feature.get('feature_columns', [])) != 1280):
                raise ValueError('Eight-head SAM-H CellViT requires every 1280-D feature vector')
        feature = copy.deepcopy(first_feature)
        feature.update(frame_id=global_image['frame_id'], row_count=total,
                       format='partitioned-sqlite', path=pointer.name)
        feature.pop('sha256', None)
        pointer_payload = {'schema': POINTER_SCHEMA, 'feature_storage': 'partition-row-reference-v1',
                           'image': global_image,
                           'partition_manifest_sha256': digest_file(partition_manifest),
                           'feature_set_id': feature['feature_set_id'],
                           'feature_columns': feature['feature_columns'],
                           'descriptor': feature['descriptor'], 'row_count': total}
        if pointer.exists():
            raise FileExistsError(f'Existing feature pointer: {pointer}')
        with pointer.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(pointer_payload, indent=2, sort_keys=True, allow_nan=False)+'\n')
        feature['sha256'] = digest_file(pointer)
        header = {'schema': 'cellphenotyper-cell-classification', 'schema_version': 1,
                  'image': global_image, 'feature_set': feature, 'models': model_list,
                  'cells': [], 'provenance': {'source': PARTITION_SCHEMA,
                                             'partition_manifest_sha256': digest_file(partition_manifest)}}
        metadata = {'schema': SCHEMA, 'schema_version': VERSION,
                    'source_format': PARTITION_SCHEMA,
                    'classification_header': header,
                    'classification': _ref(partition_manifest, output.parent),
                    'classification_sha256': digest_file(partition_manifest),
                    'feature_manifest': _ref(partition_manifest, output.parent),
                    'feature_store': _ref(pointer, output.parent),
                    'image': global_image, 'feature_set': feature, 'models': model_list,
                    'cell_count': total, 'prediction_count': total * len(model_list),
                    'spatial_index': 'rtree', 'feature_storage': 'partition-row-reference-v1',
                    'sidecars': [], 'review_state': 'separate-user-project-state',
                    'conversion_provenance': {'source': PARTITION_SCHEMA,
                        'cells_streamed': True, 'full_population_loaded': False,
                        'source_labels_changed': False, 'lineage_verified': True,
                        'scientific_accuracy_qualified': False}}
        connection.executemany('INSERT INTO metadata VALUES(?,?)',
                               [(key,_json(value)) for key,value in metadata.items()])
        connection.commit()
        connection.execute('PRAGMA optimize')
        connection.close(); connection = None
        os.link(temporary, output)
        sidecar = dict(metadata, index=_ref(output, output.parent), format='sqlite')
        with output.with_suffix(output.suffix+'.json').open('x') as stream:
            json.dump(sidecar, stream, indent=2, allow_nan=False)
            stream.write('\n')
        return metadata
    except sqlite3.IntegrityError as exc:
        raise ValueError('Duplicate partition cell or contradictory feature identity') from exc
    finally:
        if connection is not None:
            connection.close()
        Path(temporary).unlink(missing_ok=True)


def validate_partition_index(index_path, partition_manifest, feature_pointer, image_path):
    index_path = Path(index_path).resolve()
    manifest, plan = _load_source(partition_manifest, image_path, strict_location=False)
    pointer = json.loads(Path(feature_pointer).read_text())
    if (pointer.get('schema') not in (POINTER_SCHEMA, LEGACY_POINTER_SCHEMA) or
            pointer.get('partition_manifest_sha256') != digest_file(partition_manifest) or
            pointer.get('image') != manifest['image']):
        raise ValueError('Partition feature pointer/image/manifest identity mismatch')
    connection = sqlite3.connect(f'file:{index_path}?mode=ro', uri=True)
    try:
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger') LIMIT 1").fetchone():
            raise ValueError('Partition index cannot contain executable views or triggers')
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Partition SQLite index failed integrity checking')
        data = {key:json.loads(value) for key,value in connection.execute('SELECT key,value FROM metadata')}
        legacy = data.get('schema_version') == 2 and data.get('feature_storage') == 'ordered-server-only-JSON-vector-per-cell'
        current = data.get('schema_version') == VERSION and data.get('feature_storage') == 'partition-row-reference-v1'
        if ((legacy and pointer.get('schema') != LEGACY_POINTER_SCHEMA) or
                (current and pointer.get('schema') != POINTER_SCHEMA) or
                data.get('schema') != SCHEMA or not (legacy or current) or
                data.get('source_format') != PARTITION_SCHEMA or data.get('image') != manifest['image'] or
                data.get('classification_sha256') != digest_file(partition_manifest) or
                data.get('feature_store',{}).get('sha256') != digest_file(feature_pointer) or
                (index_path.parent / data.get('feature_store',{}).get('path','')).resolve() != Path(feature_pointer).resolve() or
                data.get('feature_set',{}).get('sha256') != digest_file(feature_pointer) or
                data.get('classification_header',{}).get('image') != manifest['image'] or
                data.get('cell_count') != manifest['cell_count']):
            raise ValueError('Partition index metadata differs from source manifest')
        counts = [connection.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]
                  for table in ('cells','features','bbox_rtree','predictions')]
        if counts != [manifest['cell_count']]*3+[manifest['cell_count']*len(data['models'])]:
            raise ValueError('Partition index data-table counts differ from source')
        if connection.execute('SELECT 1 FROM features f LEFT JOIN cells c ON c.row_id=f.row_id WHERE c.row_id IS NULL LIMIT 1').fetchone():
            raise ValueError('Partition feature store contains an unknown cell')
        if current:
            expected_partitions = {entry['id']: entry['cell_count'] for entry in manifest['partitions']}
            observed = {pid: set() for pid in expected_partitions}
            for (raw,) in connection.execute('SELECT values_json FROM features'):
                locator = json.loads(raw)
                if not isinstance(locator, dict):
                    raise ValueError('Partition feature row locator is invalid')
                pid, ordinal = locator.get('partition_id'), locator.get('row')
                if (set(locator) != {'partition_id', 'row'} or pid not in observed or
                        type(ordinal) is not int or ordinal < 0 or ordinal >= expected_partitions[pid] or
                        ordinal in observed[pid]):
                    raise ValueError('Partition feature row locator is invalid or duplicated')
                observed[pid].add(ordinal)
            if any(len(observed[pid]) != count for pid, count in expected_partitions.items()):
                raise ValueError('Partition feature row locators are incomplete')
        if connection.execute('SELECT 1 FROM predictions p LEFT JOIN cells c ON c.row_id=p.row_id WHERE c.row_id IS NULL LIMIT 1').fetchone():
            raise ValueError('Partition predictions contain an unknown cell')
        known_models = {model['model_id'] for model in data['models']}
        actual_models = {row[0] for row in connection.execute('SELECT DISTINCT model_id FROM predictions')}
        if actual_models != (known_models if manifest['cell_count'] else set()):
            raise ValueError('Partition predictions differ from the eight-model catalog')
        if connection.execute('SELECT 1 FROM predictions GROUP BY row_id HAVING COUNT(*) != ? LIMIT 1',
                              (len(known_models),)).fetchone():
            raise ValueError('Partition cell is missing a classifier view')
        for entry, shard in zip(manifest['partitions'], plan['shards']):
            directory = Path(partition_manifest).parent/'shards'/shard['id']
            marker = verify_complete_shard(plan, shard, directory)
            if digest_file(directory/'complete.json') != entry['checkpoint_sha256']:
                raise ValueError('Partition checkpoint differs from registered source')
            _verify_archive_entry(Path(partition_manifest).parent, entry, directory)
            verify_partition_record_locations(Path(partition_manifest).parent, entry, directory, marker)
        return data
    finally:
        connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition-manifest', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = build_partition_index(args.partition_manifest,args.image,args.output)
    print(json.dumps({'index':str(args.output),'cells':metadata['cell_count'],
                      'models':[model['taxonomy'] for model in metadata['models']]}))


if __name__ == '__main__':
    main()
