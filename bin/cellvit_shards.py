"""Bounded, independently executable CellViT++ WSI partitions.

This module never starts a GPU process. A supervisor must run one shard at a
time and verify its owned workers/GPU context before admitting the next one.
The partition manifest is deliberately distinct from the legacy monolithic
cell JSON contract; consumers must apply each partition's image origin.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from contextlib import contextmanager
import shutil
import tarfile
import tempfile
from feature_matrix import MANIFEST as MATRIX_MANIFEST, MATRIX, INDEX, pack as pack_feature_matrix, restore_csv

from cellvit_models import CLASSIFICATION_HEADER, digest_file, image_frame

SCHEMA = 'cellphenotyper-cellvit-shards-v1'
# Pinned CellViT 1.0.9 uses 1024-pixel patches with 64-pixel overlap at
# 0.25 um/px. Keep every shard's patch origin on its full-image grid.
NATIVE_PATCH_STRIDE = 1024 - 64
# PM016 lymph-node H&E has >40k hematoxylin nuclear peaks within a 4096-core
# inference input before core pruning. A native-stride-aligned 1920-pixel core
# with one 960-pixel stride of halo on every side has an interior input of
# 3840² = 14.746 MP. In a denser PM016 input, the 27.648-MP/2048-core variant
# had ~36k nuclear peaks. This is a bounded admission size, not proof of
# CellViT cell counts or native border parity.
MAX_INFERENCE_INPUT_PIXELS = 16_000_000
SHARD_FILES = (
    'cellvit_cells.json', 'cellvit_metadata.json',
    'cellvit_cell_features.manifest.json', 'cellvit_cell_features.csv.gz',
    'cellvit_feature_cells.json', 'cellvit_cell_classification.json',
    'cellvit_model_preflight.json',
)
CLASSIFICATION_TABLES = ('cellvit_cell_classification.csv.gz',
                         'cellvit_cell_classification.csv')
ARCHIVE_NAME = 'cellvit_records.tar.gz'
ARCHIVE_RECEIPT = 'cellvit_records.compact.json'
ARCHIVE_SCHEMA = 'cellphenotyper-cellvit-lossless-archive-v1'
FEATURE_MATRIX_DIR = 'cellvit_feature_matrix'
FEATURE_MATRIX_RECEIPT = 'cellvit_feature_matrix.storage.json'
FEATURE_MATRIX_SCHEMA = 'cellphenotyper-cellvit-feature-storage-v1'
FEATURE_CSV = 'cellvit_cell_features.csv.gz'


def _matrix_files():
    return tuple(f'{FEATURE_MATRIX_DIR}/{name}' for name in (MATRIX_MANIFEST, MATRIX, INDEX))


def _matrix_receipt(directory, marker, archive_receipt=None):
    """Validate a versioned replacement without changing the original checkpoint."""
    directory = Path(directory)
    if archive_receipt is None:
        path = directory / FEATURE_MATRIX_RECEIPT
        if not path.is_file():
            return None
        receipt = json.loads(path.read_text())
    else:
        if FEATURE_MATRIX_RECEIPT not in archive_receipt['members']:
            return None
        with tarfile.open(directory / ARCHIVE_NAME, 'r:gz') as archive:
            stream = archive.extractfile(FEATURE_MATRIX_RECEIPT)
            if stream is None:
                raise ValueError('Feature matrix receipt is missing from archive')
            with stream:
                receipt = json.load(stream)
    expected = marker.get('files_sha256', {}).get(FEATURE_CSV)
    if (receipt.get('schema') != FEATURE_MATRIX_SCHEMA or
            receipt.get('checkpoint_sha256') != digest_file(directory / 'complete.json') or
            receipt.get('source_csv_sha256') != expected or
            receipt.get('feature_manifest_sha256') != marker['files_sha256'].get('cellvit_cell_features.manifest.json') or
            set(receipt.get('members', {})) != set(_matrix_files())):
        raise ValueError('Feature matrix replacement differs from immutable checkpoint')
    for name, identity in receipt['members'].items():
        if (not isinstance(identity.get('bytes'), int) or identity['bytes'] < 0 or
                not isinstance(identity.get('sha256'), str) or len(identity['sha256']) != 64):
            raise ValueError('Feature matrix member identity is invalid')
        if archive_receipt is None:
            path = directory / name
            if not path.is_file() or path.stat().st_size != identity['bytes'] or digest_file(path) != identity['sha256']:
                raise ValueError('Feature matrix member differs from receipt')
        elif archive_receipt['members'].get(name) != identity:
            raise ValueError('Archived feature matrix member differs from receipt')
    return receipt


def replace_feature_csv(plan, shard, directory):
    """CPU-only, resumable checkpoint-preserving conversion of one completed shard.

    The source CSV is removed only after its exact byte restoration, a durable
    matrix receipt, and a second complete-shard verification. Old CSV-only
    shards remain valid and are never converted implicitly by a reader.
    """
    directory = Path(directory)
    if (directory.parent.parent / 'cellvit_partition_manifest.json').exists():
        raise ValueError('Pack features before publishing the immutable partition manifest')
    if (directory / ARCHIVE_RECEIPT).exists():
        raise ValueError('Pack features before archiving shard records')
    marker = verify_complete_shard(plan, shard, directory)
    source, bundle = directory / FEATURE_CSV, directory / FEATURE_MATRIX_DIR
    receipt_path = directory / FEATURE_MATRIX_RECEIPT
    if receipt_path.exists():
        _matrix_receipt(directory, marker)
        if source.exists():
            if digest_file(source) != marker['files_sha256'][FEATURE_CSV]:
                raise ValueError('Feature CSV changed before checked-copy release')
            source.unlink()
            _fsync_directory(directory)
        verify_complete_shard(plan, shard, directory)
        return json.loads(receipt_path.read_text())
    if not source.is_file() or digest_file(source) != marker['files_sha256'][FEATURE_CSV]:
        raise ValueError('Original checked feature CSV required before conversion')
    if not bundle.exists():
        pack_feature_matrix(source, bundle, directory / 'cellvit_cell_features.manifest.json')
    # An interrupted prior pack is accepted only if it can reproduce the CSV.
    with tempfile.TemporaryDirectory(prefix='.cellvit-feature-proof-', dir=directory) as temporary:
        restored = Path(temporary) / FEATURE_CSV
        restore_csv(bundle, restored)
        if digest_file(restored) != digest_file(source):
            raise ValueError('Feature matrix cannot restore original CSV bytes')
    members = {name: {'sha256': digest_file(directory / name),
                      'bytes': (directory / name).stat().st_size} for name in _matrix_files()}
    receipt = {'schema': FEATURE_MATRIX_SCHEMA,
               'checkpoint_sha256': digest_file(directory / 'complete.json'),
               'source_csv_sha256': marker['files_sha256'][FEATURE_CSV],
               'feature_manifest_sha256': marker['files_sha256']['cellvit_cell_features.manifest.json'],
               'members': members}
    atomic_json(receipt_path, receipt)
    _matrix_receipt(directory, marker)
    source.unlink()
    _fsync_directory(directory)
    verify_complete_shard(plan, shard, directory)
    return receipt


def rounded_mpp_matches(actual, nominal, pixels):
    """Admit only the scale error caused by rounding an output axis to pixels.

    A converted field of view rounded to the nearest output pixel changes its
    physical pixel size by at most nominal/(2*pixels). Preserve the actual
    calibrated X/Y MPP in every frame and checkpoint.
    """
    return (isinstance(pixels, int) and pixels > 0 and math.isfinite(float(actual)) and
            math.isfinite(float(nominal)) and float(nominal) > 0 and
            abs(float(actual)-float(nominal)) <= float(nominal)/(2*pixels) + 1e-9)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix='.' + path.name + '.',
                                     suffix='.tmp', encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def file_identity(path):
    stat = Path(path).stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'device': stat.st_dev, 'inode': stat.st_ino}


def _identity(plan):
    return hashlib.sha256(json.dumps({key: plan[key] for key in
        ('schema', 'image', 'image_stat', 'shift_sha256', 'core_size', 'halo', 'run_contract', 'shards')},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def plan_shards(image, shift, core_size=1920, halo=960, run_contract=None):
    """Cover the full level-0 frame with nonoverlapping half-open cores."""
    core_size, halo = int(core_size), int(halo)
    if core_size < NATIVE_PATCH_STRIDE or core_size % 64 or halo < NATIVE_PATCH_STRIDE or halo > core_size:
        raise ValueError('CellViT shards require 64-pixel-aligned cores and at least one 960-pixel native patch stride of halo')
    image, shift = Path(image).resolve(), Path(shift).resolve()
    frame = image_frame(image, shift)
    if (not rounded_mpp_matches(frame['mpp_x'], 0.25, frame['width']) or
            not rounded_mpp_matches(frame['mpp_y'], 0.25, frame['height'])):
        raise ValueError('CellViT shard grid requires the retained 0.25 um/px RGB image')
    width, height = frame['width'], frame['height']
    shards = plan_shards_geometry(width, height, core_size, halo)
    validate_input_budget(shards)
    plan = {'schema': SCHEMA, 'image': frame, 'image_path': str(image),
            'image_stat': file_identity(image), 'shift_path': str(shift),
            'shift_sha256': digest_file(shift), 'core_size': core_size,
            'halo': halo, 'run_contract': run_contract or {}, 'shards': shards}
    plan['identity_sha256'] = _identity(plan)
    return plan


def validate_plan(plan, image=None, shift=None):
    if plan.get('schema') != SCHEMA or plan.get('identity_sha256') != _identity(plan):
        raise ValueError('CellViT shard plan identity is invalid')
    image = Path(image or plan['image_path']).resolve()
    shift = Path(shift or plan['shift_path']).resolve()
    if str(image) != plan['image_path'] or file_identity(image) != plan['image_stat']:
        raise ValueError('CellViT source image changed after shard planning')
    if str(shift) != plan['shift_path'] or digest_file(shift) != plan['shift_sha256']:
        raise ValueError('CellViT shift metadata changed after shard planning')
    mask = plan['run_contract'].get('clean_tissue_mask')
    if mask and digest_file(mask) != plan['run_contract'].get('clean_tissue_mask_sha256'):
        raise ValueError('CellViT clean-tissue mask changed after shard planning')
    expected = plan_shards_geometry(plan['image']['width'], plan['image']['height'],
                                    plan['core_size'], plan['halo'])
    if plan['shards'] != expected:
        raise ValueError('CellViT shard coverage or coordinates changed')
    validate_input_budget(expected)
    return plan


def validate_input_budget(shards):
    """Fail before GPU work if any raw native input exceeds the CPU-audited size."""
    for shard in shards:
        x0, y0, x1, y1 = shard['input']
        pixels = (x1 - x0) * (y1 - y0)
        if pixels > MAX_INFERENCE_INPUT_PIXELS:
            raise ValueError(f'CellViT shard {shard["id"]} has {pixels} raw input pixels '
                             f'(>{MAX_INFERENCE_INPUT_PIXELS}); use 1920-pixel cores with 960-pixel halo '
                             'or a separately validated density-aware plan')


def plan_shards_geometry(width, height, core_size, halo):
    shards = []
    for row, y0 in enumerate(range(0, height, core_size)):
        for col, x0 in enumerate(range(0, width, core_size)):
            x1, y1 = min(width, x0 + core_size), min(height, y0 + core_size)
            ix0 = max(0, ((x0-halo)//NATIVE_PATCH_STRIDE)*NATIVE_PATCH_STRIDE)
            iy0 = max(0, ((y0-halo)//NATIVE_PATCH_STRIDE)*NATIVE_PATCH_STRIDE)
            ix1 = min(width, ((x1+halo+NATIVE_PATCH_STRIDE-1)//NATIVE_PATCH_STRIDE)*NATIVE_PATCH_STRIDE)
            iy1 = min(height, ((y1+halo+NATIVE_PATCH_STRIDE-1)//NATIVE_PATCH_STRIDE)*NATIVE_PATCH_STRIDE)
            shards.append({'id': f'r{row:04d}c{col:04d}', 'core': [x0,y0,x1,y1],
                           'input': [ix0,iy0,ix1,iy1],
                           'local_core': [x0-ix0,y0-iy0,x1-ix0,y1-iy0]})
    return shards


def get_shard(plan, shard_id):
    matches = [item for item in plan['shards'] if item['id'] == shard_id]
    if len(matches) != 1:
        raise ValueError(f'Unknown CellViT shard: {shard_id}')
    return matches[0]


def stage_shard(plan, shard, directory):
    """Make one lossless bounded crop; never materialize the whole slide."""
    import pyvips
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / 'input.tif'
    shift_path = directory / 'shift.json'
    if image_path.exists() or shift_path.exists():
        raise FileExistsError(f'Shard staging already exists: {directory}')
    x0, y0, x1, y1 = shard['input']
    source = pyvips.Image.new_from_file(plan['image_path'], access='random')
    if source.bands != 3 or source.format != 'uchar' or source.interpretation not in ('srgb', 'rgb'):
        raise ValueError('CellViT shard source must be RGB8')
    if (source.width, source.height) != (plan['image']['width'], plan['image']['height']):
        raise ValueError('CellViT source dimensions changed after planning')
    source.crop(x0, y0, x1-x0, y1-y0).tiffsave(
        str(image_path), tile=True, tile_width=512, tile_height=512,
        pyramid=True, compression='lzw', bigtiff=True)
    shift = json.loads(Path(plan['shift_path']).read_text())
    shift.update(source_mpp=plan['image']['mpp_x'], mpp_x=plan['image']['mpp_x'],
                 mpp_y=plan['image']['mpp_y'],
                 crop_size={'width': x1-x0, 'height': y1-y0},
                 cellvit_shard={'id': shard['id'], 'input_origin_xy': [x0, y0],
                                'core_xyxy': shard['core'], 'plan_sha256': plan['identity_sha256']})
    atomic_json(shift_path, shift)
    return image_path, shift_path


def core_cells(payload, shard, full_width, full_height):
    """Translate selected local centroids into exactly one global core."""
    cells = payload.get('cells', [])
    if not isinstance(cells, list):
        raise ValueError('Normalized CellViT shard requires a cell list')
    ix0, iy0, ix1, iy1 = shard['input']
    x0, y0, x1, y1 = shard['core']
    ids, canonical_ids = set(), set()
    output = []
    for cell in cells:
        centroid = cell.get('centroid')
        if not isinstance(centroid, list) or len(centroid) != 2:
            raise ValueError('CellViT shard cell has no XY centroid')
        lx, ly = map(float, centroid)
        if not all(math.isfinite(v) for v in (lx, ly)) or not (0 <= lx < ix1-ix0 and 0 <= ly < iy1-iy0):
            raise ValueError('CellViT shard centroid outside its input crop')
        gx, gy = ix0 + lx, iy0 + ly
        if not (x0 <= gx < x1 and y0 <= gy < y1):
            raise ValueError('CellViT shard output retains a halo cell outside its core')
        if not (0 <= gx < full_width and 0 <= gy < full_height):
            raise ValueError('CellViT global centroid outside the source image')
        local_id = str(cell.get('id', ''))
        canonical = str(cell.get('cell_id', ''))
        if not local_id or not canonical or local_id in ids or canonical in canonical_ids:
            raise ValueError('CellViT shard source or canonical identity is missing/duplicated')
        ids.add(local_id)
        canonical_ids.add(canonical)
        output.append({'cell_id': canonical, 'source_id': f"{shard['id']}:{local_id}",
                       'partition_id': shard['id'], 'x': gx, 'y': gy})
    return output


def validate_classification_table(directory, name, classification):
    """Reopen every CSV row before publishing a shard checkpoint.

    This checks gzip CRC, the exact output schema, and the JSON cell/model join.
    Existing plain CSV shards use the same scientific row contract.
    """
    path = Path(directory) / name
    opener = gzip.open if name.endswith('.gz') else Path.open
    try:
        with opener(path, 'rt', newline='', encoding='utf-8') as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader, None) != CLASSIFICATION_HEADER:
                raise ValueError('CellViT classification CSV header differs from the checked schema')
            for cell in classification['cells']:
                for model in classification['models']:
                    row = next(reader, None)
                    if (row is None or len(row) != len(CLASSIFICATION_HEADER) or
                            row[0] != cell['cell_id'] or row[1] != cell['source_id'] or
                            row[4] != model['model_id'] or
                            float(row[2]) != cell['x'] or float(row[3]) != cell['y']):
                        raise ValueError('CellViT classification CSV/JSON row join differs')
            if next(reader, None) is not None:
                raise ValueError('CellViT classification CSV has extra rows')
    except (OSError, EOFError, csv.Error) as error:
        raise ValueError('CellViT classification CSV cannot be reopened') from error


def complete_shard(plan, shard, directory):
    """Validate and atomically record a completed, core-only partition."""
    directory = Path(directory)
    marker = directory / 'complete.json'
    if marker.exists():
        raise FileExistsError(f'Existing shard checkpoint must be verified, not overwritten: {marker}')
    present_tables = [name for name in CLASSIFICATION_TABLES if (directory / name).is_file()]
    if len(present_tables) != 1:
        raise ValueError('CellViT shard requires exactly one plain or gzip classification CSV')
    shard_files = (*SHARD_FILES, present_tables[0])
    missing = [name for name in shard_files if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f'CellViT shard outputs are missing: {missing}')
    normalized = json.loads((directory / 'cellvit_cells.json').read_text())
    classification = json.loads((directory / 'cellvit_cell_classification.json').read_text())
    validate_classification_table(directory, present_tables[0], classification)
    metadata = json.loads((directory / 'cellvit_metadata.json').read_text())
    if (not metadata.get('model') or not metadata.get('requested_taxonomies') or
            not metadata.get('model_provenance', {}).get('checkpoints') or
            not classification.get('models')):
        raise ValueError('CellViT shard model provenance is incomplete')
    selected = core_cells(normalized, shard, plan['image']['width'], plan['image']['height'])
    if {cell['cell_id'] for cell in selected} != {cell['cell_id'] for cell in classification['cells']}:
        raise ValueError('CellViT shard segmentation/classification core IDs differ')
    if len(selected) != len(classification['cells']):
        raise ValueError('CellViT shard classification has duplicate IDs')
    classified = {cell['cell_id']: cell for cell in classification['cells']}
    model_ids = [model['model_id'] for model in classification['models']]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError('CellViT shard classification has duplicate model IDs')
    for cell in selected:
        result = classified[cell['cell_id']]
        local_x = cell['x'] - shard['input'][0]
        local_y = cell['y'] - shard['input'][1]
        if result.get('x') != local_x or result.get('y') != local_y:
            raise ValueError('CellViT shard classification coordinates differ from core cell geometry')
        predictions = result.get('predictions', {})
        if set(predictions) != set(model_ids):
            raise ValueError('CellViT shard classification lacks a requested model prediction')
        cell['predictions'] = {model_id: {
            key: predictions[model_id].get(key) for key in
            ('status', 'class_id', 'label', 'probability')}
            for model_id in model_ids}
    index_path = directory / 'core_cells.jsonl'
    with tempfile.NamedTemporaryFile('w', dir=directory, prefix='.core_cells.', suffix='.tmp',
                                     encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            for cell in selected:
                stream.write(json.dumps(cell, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, index_path)
    files = {name: digest_file(directory / name) for name in (*shard_files, index_path.name)}
    model_signature = shard_model_signature(metadata, classification)
    marker_data = {'schema': SCHEMA, 'plan_sha256': plan['identity_sha256'],
                   'partition_id': shard['id'], 'core': shard['core'], 'input': shard['input'],
                   'cell_count': len(selected), 'files_sha256': files,
                   'model_signature_sha256': model_signature,
                   'image_stat_after': file_identity(plan['image_path'])}
    if marker_data['image_stat_after'] != plan['image_stat']:
        raise ValueError('Source image changed while processing CellViT shard')
    atomic_json(marker, marker_data)
    return marker_data


def shard_model_signature(metadata, classification):
    model_contract = {'model': metadata.get('model'),
                      'requested_taxonomies': metadata.get('requested_taxonomies'),
                      'segmentation_checkpoint': metadata.get('model_provenance', {}).get('checkpoints'),
                      'classification_models': classification.get('models')}
    return hashlib.sha256(json.dumps(model_contract, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_complete_shard(plan, shard, directory):
    directory = Path(directory)
    marker = json.loads((directory / 'complete.json').read_text())
    if (marker.get('schema') != SCHEMA or marker.get('plan_sha256') != plan['identity_sha256'] or
            marker.get('partition_id') != shard['id'] or marker.get('core') != shard['core'] or
            marker.get('input') != shard['input'] or marker.get('image_stat_after') != plan['image_stat']):
        raise ValueError(f'CellViT shard checkpoint identity mismatch: {shard["id"]}')
    compact = _verify_archive_receipt(directory, marker) if (directory / ARCHIVE_RECEIPT).exists() else None
    replacement = _matrix_receipt(directory, marker, compact)
    for name, expected in marker.get('files_sha256', {}).items():
        if name not in (*SHARD_FILES, *CLASSIFICATION_TABLES, 'core_cells.jsonl'):
            raise ValueError(f'CellViT shard output name is unexpected: {shard["id"]}/{name}')
        path = directory / name
        if path.exists():
            if digest_file(path) != expected:
                raise ValueError(f'CellViT shard output hash mismatch: {shard["id"]}/{name}')
        elif name == FEATURE_CSV and replacement is not None:
            continue
        elif compact is None or compact['members'].get(name, {}).get('sha256') != expected:
            raise ValueError(f'CellViT shard output hash mismatch: {shard["id"]}/{name}')
    selected_tables = set(marker.get('files_sha256', {})).intersection(CLASSIFICATION_TABLES)
    if (len(selected_tables) != 1 or
            set(marker.get('files_sha256', {})) != set((*SHARD_FILES, *selected_tables, 'core_cells.jsonl'))):
        raise ValueError('CellViT shard checkpoint file set is incomplete')
    metadata = _read_record_json(directory, 'cellvit_metadata.json')
    classification = _read_record_json(directory, 'cellvit_cell_classification.json')
    if marker.get('model_signature_sha256') != shard_model_signature(metadata, classification):
        raise ValueError('CellViT shard model signature differs from checkpoint')
    return marker


def _read_record_json(directory, name):
    directory = Path(directory)
    path = directory / name
    if path.is_file():
        return json.loads(path.read_text())
    with tarfile.open(directory / ARCHIVE_NAME, 'r:gz') as archive:
        member = archive.extractfile(name)
        if member is None:
            raise ValueError(f'Missing CellViT archive member: {name}')
        with member:
            return json.load(member)


def _verify_archive_receipt(directory, marker):
    directory = Path(directory)
    receipt = json.loads((directory / ARCHIVE_RECEIPT).read_text())
    if (receipt.get('schema') != ARCHIVE_SCHEMA or
            receipt.get('checkpoint_sha256') != digest_file(directory / 'complete.json') or
            receipt.get('archive_sha256') != digest_file(directory / ARCHIVE_NAME) or
            not isinstance(receipt.get('members'), dict)):
        raise ValueError('CellViT lossless archive receipt or bytes differ from checkpoint')
    has_matrix = FEATURE_MATRIX_RECEIPT in receipt['members']
    for name, expected in marker.get('files_sha256', {}).items():
        if name == FEATURE_CSV and has_matrix and name not in receipt['members']:
            continue
        if receipt['members'].get(name, {}).get('sha256') != expected:
            raise ValueError(f'CellViT archive does not preserve checked record: {name}')
    return receipt


def _verify_archive_members(path, files):
    with tarfile.open(path, 'r:gz') as archive:
        names = archive.getnames()
        if len(names) != len(files) or set(names) != set(files):
            raise ValueError('CellViT archive member set differs from source records')
        for name, expected in files.items():
            member = archive.getmember(name)
            if not member.isfile() or member.size != expected['bytes']:
                raise ValueError(f'CellViT archive member size differs: {name}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'CellViT archive member is unreadable: {name}')
            digest = hashlib.sha256()
            with stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError(f'CellViT archive member hash differs: {name}')


def compact_shard(plan, shard, directory):
    """Archive every retained record byte-for-byte, then remove checked copies.

    No scientific file is discarded: the archive can restore the exact source
    tree. An interrupted operation leaves either verified originals or the
    verified archive. Large model/cell objects are streamed through tarfile.
    """
    directory = Path(directory)
    marker = verify_complete_shard(plan, shard, directory)
    if (directory.parent.parent / 'cellvit_partition_manifest.json').exists():
        raise ValueError('Compact before publishing the immutable partition manifest')
    if (directory / 'cellvit_input.tif').exists() or (directory.parent.parent / 'staging' / shard['id'] / 'input.tif').exists():
        raise ValueError('Release verified redundant input crops before archiving shard records')
    if (directory / ARCHIVE_RECEIPT).exists():
        receipt = _verify_archive_receipt(directory, marker)
        for name, expected in receipt['members'].items():
            path = directory / name
            if path.exists() and digest_file(path) != expected['sha256']:
                raise ValueError(f'CellViT source changed before archived-copy release: {name}')
        for name in receipt['members']:
            (directory / name).unlink(missing_ok=True)
        return receipt
    excluded = {'complete.json', 'input_crops_released.json', ARCHIVE_NAME, ARCHIVE_RECEIPT}
    paths = sorted(path for path in directory.rglob('*') if path.is_file() or path.is_symlink())
    files = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError('CellViT archive refuses symlinked scientific records')
        name = path.relative_to(directory).as_posix()
        if name in excluded:
            continue
        files[name] = {'sha256': digest_file(path), 'bytes': path.stat().st_size}
    replacement = _matrix_receipt(directory, marker)
    if any(files.get(name, {}).get('sha256') != expected for name, expected in marker['files_sha256'].items()
           if name != FEATURE_CSV or replacement is None):
        raise ValueError('CellViT checked record changed before archiving')
    if replacement is not None and (FEATURE_MATRIX_RECEIPT not in files or
            any(files.get(name) != identity for name, identity in replacement['members'].items())):
        raise ValueError('CellViT replacement feature files changed before archiving')
    temporary = directory / ('.' + ARCHIVE_NAME + '.tmp')
    if temporary.exists():
        raise FileExistsError('Incomplete archive temporary file is preserved')
    archive_path = directory / ARCHIVE_NAME
    if not archive_path.exists():
        with tarfile.open(temporary, 'w:gz', dereference=True) as archive:
            for name in files:
                archive.add(directory / name, arcname=name, recursive=False)
        _verify_archive_members(temporary, files)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, archive_path)
        _fsync_directory(directory)
    else:
        # A crash after the archive rename but before its receipt is resumable
        # only while every original source byte is still present and checked.
        _verify_archive_members(archive_path, files)
    receipt = {'schema': ARCHIVE_SCHEMA,
               'checkpoint_sha256': digest_file(directory / 'complete.json'),
               'archive_sha256': digest_file(archive_path),
               'members': files,
               'recovery': 'extract checked members from cellvit_records.tar.gz; original bytes retained'}
    atomic_json(directory / ARCHIVE_RECEIPT, receipt)
    _verify_archive_receipt(directory, marker)
    for name, expected in files.items():
        if digest_file(directory / name) != expected['sha256']:
            raise ValueError(f'CellViT source changed before archived-copy release: {name}')
    for name in files:
        (directory / name).unlink()
    verify_complete_shard(plan, shard, directory)
    return receipt


@contextmanager
def materialize_shard_records(directory):
    """Temporarily expose checked records for one partition index operation."""
    directory = Path(directory)
    archived = (directory / ARCHIVE_RECEIPT).exists()
    marker = json.loads((directory / 'complete.json').read_text())
    receipt = _verify_archive_receipt(directory, marker) if archived else None
    matrix = _matrix_receipt(directory, marker, receipt)
    if not archived and (matrix is None or (directory / FEATURE_CSV).is_file()):
        yield directory
        return
    with tempfile.TemporaryDirectory(prefix='.cellvit-index-', dir=directory.parent.parent) as temporary:
        target = Path(temporary)
        needed = {'cellvit_cell_classification.json',
                  'cellvit_cell_features.manifest.json',
                  'cellvit_feature_cells.json',
                  'cellvit_cells.json', 'core_cells.jsonl'}
        if matrix is None:
            needed.add(FEATURE_CSV)
        else:
            needed.update((*_matrix_files(), FEATURE_MATRIX_RECEIPT))
        selected_tables = set(marker['files_sha256']).intersection(CLASSIFICATION_TABLES)
        if len(selected_tables) != 1:
            raise ValueError('CellViT archive checkpoint has no unique classification table')
        needed.update(selected_tables)
        if not archived:
            for name in needed:
                output = target / name
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / name, output)
                expected = matrix['members'][name]['sha256'] if name in _matrix_files() else (
                    digest_file(directory / name))
                if digest_file(output) != expected:
                    raise ValueError(f'CellViT direct index record changed during copy: {name}')
        else:
            with tarfile.open(directory / ARCHIVE_NAME, 'r:gz') as archive:
                for name in needed:
                    expected = receipt['members'].get(name)
                    if not expected:
                        raise ValueError(f'CellViT archive lacks index record: {name}')
                    member = archive.getmember(name)
                    if not member.isfile() or member.size != expected['bytes']:
                        raise ValueError(f'CellViT archive index record has wrong size: {name}')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f'CellViT archive index record is unreadable: {name}')
                    output = target / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with stream, output.open('wb') as destination:
                        shutil.copyfileobj(stream, destination, length=1024 * 1024)
                    if digest_file(output) != expected['sha256']:
                        raise ValueError(f'CellViT archive index record has wrong hash: {name}')
        if matrix is not None:
            restore_csv(target / FEATURE_MATRIX_DIR, target / FEATURE_CSV)
            if digest_file(target / FEATURE_CSV) != marker['files_sha256'][FEATURE_CSV]:
                raise ValueError('Restored feature CSV differs from checked source')
        yield target


@contextmanager
def materialize_feature_matrix(directory):
    """Expose only the checked matrix bundle, without expanding the old CSV."""
    directory = Path(directory)
    marker = json.loads((directory / 'complete.json').read_text())
    archive_receipt = (_verify_archive_receipt(directory, marker)
                       if (directory / ARCHIVE_RECEIPT).exists() else None)
    matrix = _matrix_receipt(directory, marker, archive_receipt)
    if matrix is None:
        raise ValueError('Partition has no matrix feature storage')
    if archive_receipt is None:
        yield directory / FEATURE_MATRIX_DIR
        return
    with tempfile.TemporaryDirectory(prefix='.cellvit-feature-read-', dir=directory.parent.parent) as temporary:
        target = Path(temporary)
        with tarfile.open(directory / ARCHIVE_NAME, 'r:gz') as archive:
            for name in _matrix_files():
                expected = matrix['members'][name]
                member = archive.getmember(name)
                if not member.isfile() or member.size != expected['bytes']:
                    raise ValueError(f'Archived matrix member size differs: {name}')
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f'Archived matrix member unreadable: {name}')
                output = target / name
                output.parent.mkdir(parents=True, exist_ok=True)
                with source, output.open('wb') as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
                if digest_file(output) != expected['sha256']:
                    raise ValueError(f'Archived matrix member hash differs: {name}')
        yield target / FEATURE_MATRIX_DIR


def _manifest_record_location(root, directory, name, marker, receipt):
    """Give archived records a resolvable member, never a missing file path."""
    if receipt is None:
        return str((directory / name).relative_to(root))
    member = receipt['members'].get(name)
    if (member is None or member.get('sha256') != marker['files_sha256'].get(name) or
            not isinstance(member.get('bytes'), int) or member['bytes'] < 0):
        raise ValueError(f'CellViT archive lacks checked manifest record: {name}')
    return {'storage': 'archive-member',
            'archive': str((directory / ARCHIVE_NAME).relative_to(root)),
            'member': name, 'sha256': member['sha256'], 'bytes': member['bytes']}


def _feature_storage_location(root, directory, marker, archive_receipt):
    matrix = _matrix_receipt(directory, marker, archive_receipt)
    if matrix is None:
        return None
    names = (*_matrix_files(), FEATURE_MATRIX_RECEIPT)
    references = {}
    for name in names:
        if archive_receipt is None:
            references[name] = {'storage': 'file',
                'path': str((directory / name).relative_to(root)),
                'sha256': digest_file(directory / name), 'bytes': (directory / name).stat().st_size}
        else:
            identity = archive_receipt['members'][name]
            references[name] = {'storage': 'archive-member',
                'archive': str((directory / ARCHIVE_NAME).relative_to(root)),
                'member': name, **identity}
    return {'schema': FEATURE_MATRIX_SCHEMA, 'source_csv_sha256': matrix['source_csv_sha256'],
            'references': references}


def verify_partition_record_locations(root, entry, directory, marker):
    """Reject stale or altered record pointers before downstream indexing."""
    root, directory = Path(root), Path(directory)
    receipt = (_verify_archive_receipt(directory, marker)
               if (directory / ARCHIVE_RECEIPT).exists() else None)
    for field, name in (('classification', 'cellvit_cell_classification.json'),
                        ('features', 'cellvit_cell_features.manifest.json'),
                        ('segmentation', 'cellvit_cells.json'),
                        ('global_cell_index', 'core_cells.jsonl')):
        if entry.get(field) != _manifest_record_location(root, directory, name, marker, receipt):
            raise ValueError(f'CellViT partition {field} does not resolve to checked record')
    tables = set(marker['files_sha256']).intersection(CLASSIFICATION_TABLES)
    if len(tables) != 1:
        raise ValueError('CellViT partition has no unique classification table')
    name = next(iter(tables))
    expected = {'path': _manifest_record_location(root, directory, name, marker, receipt),
                'sha256': marker['files_sha256'][name],
                'format': 'csv.gz' if name.endswith('.gz') else 'csv'}
    if entry.get('classification_table') != expected:
        raise ValueError('CellViT partition classification table does not resolve to checked record')
    if entry.get('feature_storage') != _feature_storage_location(root, directory, marker, receipt):
        raise ValueError('CellViT feature storage does not resolve to checked matrix')


def release_verified_input_crops(plan, shard, directory, staging_dir):
    """Release reproducible TIFF copies only after a durable checked checkpoint.

    The source slide, immutable plan and shift remain. This routine is safe to
    call after an interrupted cleanup; it never deletes an unverified crop or
    any scientific cell, token, classification or geometry record.
    """
    directory, staging_dir = Path(directory), Path(staging_dir)
    marker = verify_complete_shard(plan, shard, directory)
    metadata = json.loads((directory / 'cellvit_metadata.json').read_text())
    preparation = metadata.get('input_preparation', {})
    expected = {
        staging_dir / 'input.tif': preparation.get('source_sha256'),
        directory / 'cellvit_input.tif': preparation.get('prepared_sha256'),
    }
    if (not all(isinstance(value, str) and len(value) == 64 for value in expected.values()) or
            preparation.get('width') != shard['input'][2] - shard['input'][0] or
            preparation.get('height') != shard['input'][3] - shard['input'][1]):
        raise ValueError('CellViT crop release lacks exact input preparation provenance')
    receipt_path = directory / 'input_crops_released.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('checkpoint_sha256') != digest_file(directory / 'complete.json') or
                receipt.get('plan_sha256') != plan['identity_sha256'] or
                receipt.get('input_sha256') != expected[staging_dir / 'input.tif'] or
                receipt.get('prepared_sha256') != expected[directory / 'cellvit_input.tif']):
            raise ValueError('Existing CellViT crop release receipt differs from checkpoint')
    else:
        if any(not path.is_file() for path in expected):
            raise FileNotFoundError('Both exact input crops are required before first verified release')
        receipt = {'schema': 'cellphenotyper-cellvit-input-release-v1',
                   'plan_sha256': plan['identity_sha256'],
                   'checkpoint_sha256': digest_file(directory / 'complete.json'),
                   'input_sha256': expected[staging_dir / 'input.tif'],
                   'prepared_sha256': expected[directory / 'cellvit_input.tif'],
                   'regeneration': 'source image + immutable shard plan; no cell or feature records removed'}
    for path, expected_hash in expected.items():
        if path.exists() and digest_file(path) != expected_hash:
            raise ValueError(f'CellViT redundant crop changed before release: {path}')
    if not receipt_path.exists():
        atomic_json(receipt_path, receipt)
    for path in expected:
        path.unlink(missing_ok=True)
    return receipt


def finalize_shards(plan, root, *, publish=True):
    """Publish partition references only; never load whole-slide cell rows."""
    root = Path(root)
    partitions = []
    model_signature = None
    for shard in plan['shards']:
        directory = root / 'shards' / shard['id']
        marker = verify_complete_shard(plan, shard, directory)
        if model_signature is None:
            model_signature = marker['model_signature_sha256']
        elif marker['model_signature_sha256'] != model_signature:
            raise ValueError('CellViT partitions used different segmentation/classifier models')
        table_name = next(name for name in CLASSIFICATION_TABLES
                          if name in marker['files_sha256'])
        receipt = (_verify_archive_receipt(directory, marker)
                   if (directory / ARCHIVE_RECEIPT).exists() else None)
        partitions.append({'id': shard['id'], 'core_xyxy': shard['core'],
                           'input_xyxy': shard['input'], 'input_origin_xy': shard['input'][:2],
                           'cell_count': marker['cell_count'],
                           'classification': _manifest_record_location(root, directory, 'cellvit_cell_classification.json', marker, receipt),
                           'classification_table': {
                               'path': _manifest_record_location(root, directory, table_name, marker, receipt),
                               'sha256': marker['files_sha256'][table_name],
                               'format': 'csv.gz' if table_name.endswith('.gz') else 'csv'},
                           'features': _manifest_record_location(root, directory, 'cellvit_cell_features.manifest.json', marker, receipt),
                           'feature_storage': _feature_storage_location(root, directory, marker, receipt),
                           'segmentation': _manifest_record_location(root, directory, 'cellvit_cells.json', marker, receipt),
                           'global_cell_index': _manifest_record_location(root, directory, 'core_cells.jsonl', marker, receipt),
                           'checkpoint': str((directory / 'complete.json').relative_to(root)),
                           'checkpoint_sha256': digest_file(directory / 'complete.json'),
                           'lossless_archive': ({'path': str((directory / ARCHIVE_NAME).relative_to(root)),
                                                 'sha256': digest_file(directory / ARCHIVE_NAME),
                                                 'receipt': str((directory / ARCHIVE_RECEIPT).relative_to(root)),
                                                 'receipt_sha256': digest_file(directory / ARCHIVE_RECEIPT)}
                                                if receipt is not None else None)})
    manifest = {'schema': SCHEMA, 'image': plan['image'],
                'plan_sha256': plan['identity_sha256'], 'partition_count': len(partitions),
                'cell_count': sum(item['cell_count'] for item in partitions),
                'model_signature_sha256': model_signature,
                'coordinate_policy': 'half-open-disjoint-global-cores; partition local XY plus input_origin_xy',
                'gpu_lifecycle_policy': 'one externally guarded shard per admission; no automatic next shard',
                'partitions': partitions}
    if publish:
        atomic_json(root / 'cellvit_partition_manifest.json', manifest)
    return manifest
