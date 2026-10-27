#!/usr/bin/env python3
"""Adapt verified CellViT instances to the shared cell-mask/table contract.

This is a CPU-only format/coordinate conversion. It never loads cell feature
vectors, runs a detector, fuses detectors, or changes a classifier's evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

import numpy as np
import rasterio
from shapely.geometry import Polygon, box, mapping
import tifffile

from build_cell_consensus import write_mask, validate_mask_label_coverage
from cellvit_models import image_frame

SCHEMA = 'cellphenotyper-cellvit-primary-segmentation-v1'
CONVENTION = 'pixel-edge-origin_centres-in-native-pixels'
HEADS = {'pannuke', 'consep', 'lizard', 'nucls_main', 'nucls_super', 'ocelot', 'midog', 'panoptils'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def dimensions(path):
    with tifffile.TiffFile(path) as opened:
        return int(opened.pages[0].imagewidth), int(opened.pages[0].imagelength)


def frame_valid(frame):
    if not isinstance(frame, dict):
        raise ValueError('Source image frame is missing')
    required = ('sha256', 'width', 'height', 'mpp_x', 'mpp_y', 'origin_x', 'origin_y',
                'coordinate_space', 'coordinate_convention', 'frame_id')
    if any(key not in frame for key in required):
        raise ValueError('Source image frame is incomplete')
    if (not re.fullmatch('[a-f0-9]{64}', str(frame['sha256'])) or
        frame['coordinate_convention'] != CONVENTION or
        frame['coordinate_space'] != 'crop-native-pixels' or
        frame['origin_x'] != 0 or frame['origin_y'] != 0 or
        any(type(frame[k]) is not int or frame[k] <= 0 for k in ('width', 'height')) or
        any(not math.isfinite(float(frame[k])) or float(frame[k]) <= 0 for k in ('mpp_x', 'mpp_y'))):
        raise ValueError('Source image frame is invalid')
    serialized = {key: frame[key] for key in required if key != 'frame_id'}
    expected = 'cellvit-frame-'+hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()[:24]
    if frame['frame_id'] != expected:
        raise ValueError('Source image frame identity does not match its fields')


def asset(root, reference, label):
    if not isinstance(reference, dict) or not reference.get('path') or not reference.get('sha256'):
        raise ValueError(label+' requires a path and SHA-256')
    path = (root / reference['path']).resolve()
    if digest(path) != reference['sha256']:
        raise ValueError(label+' hash mismatch')
    return path


def pannuke_model(models):
    """Use the declared taxonomy, never a model-ID substring or head ordering."""
    matches = [model for model in models if isinstance(model, dict) and model.get('taxonomy') == 'pannuke']
    if len(matches) != 1:
        raise ValueError('Exactly one declared PanNuke model is required for the primary type view')
    model = matches[0]
    if (model.get('status') != 'available' or not isinstance(model.get('model_id'), str) or
        not model['model_id'] or not isinstance(model.get('class_names'), list) or not model['class_names']):
        raise ValueError('The primary PanNuke head must have an available, explicit ontology')
    return model


def pannuke_view(item, lineage):
    """Expose one named ontology for legacy downstream consumers; keep all heads."""
    if lineage['input_scope'] == 'full_slide':
        model = lineage['primary_type_model']
        prediction = item.get('predictions', {}).get(model['model_id'])
        if not isinstance(prediction, dict):
            raise ValueError('Cell is missing the exact declared PanNuke prediction')
        status = prediction.get('status')
        if status != 'predicted':
            if status not in ('abstained', 'unavailable') or any(prediction.get(k) is not None for k in ('class_id', 'label', 'probability')):
                raise ValueError('Nonpredicted PanNuke evidence cannot contain an assignment')
            return dict(type=None, type_id=None, probability=None, status=status)
        class_id, label, probability = prediction.get('class_id'), prediction.get('label'), prediction.get('probability')
        if (type(class_id) is not int or not 0 <= class_id < len(model['class_names']) or
            label != model['class_names'][class_id]):
            raise ValueError('PanNuke prediction differs from its declared ontology')
    else:
        class_id = item.get('type_id')
        label = lineage['primary_type_map'].get(str(class_id))
        if label is None or str(item.get('type', '')).lower() != str(label).lower():
            raise ValueError('Native PanNuke type differs from its explicit type map')
        probability, status = item.get('type_prob'), 'predicted'
    if probability is not None and (not math.isfinite(float(probability)) or not 0 <= float(probability) <= 1):
        raise ValueError('PanNuke probability must be finite and in [0,1]')
    return dict(type=label, type_id=class_id, probability=probability, status=status)


@contextmanager
def fullslide_cells(pointer_path, receipt_path, shift):
    pointer = read_json(pointer_path)
    if pointer.get('complete') is not True:
        raise ValueError('A complete full-slide project pointer is required')
    receipt_path = Path(receipt_path).resolve()
    # Nextflow stages a copy named cellvit_fullslide_complete.json; byte
    # identity, rather than that task-local filename, binds the receipt.
    if (not pointer.get('completion_receipt_path') or
        digest(receipt_path) != pointer.get('completion_receipt_sha256')):
        raise ValueError('Completion receipt is not bound to the full-slide pointer')
    receipt = read_json(receipt_path)
    planned = receipt.get('planned_shards')
    if (receipt.get('schema') != 'cellphenotyper-cellvit-fullslide-complete-v2' or
        type(planned) is not int or planned < 1 or receipt.get('complete_shards') != planned or
        receipt.get('shard_count') != planned or receipt.get('every_core_verified') is not True or
        set(receipt.get('requested_heads', [])) != HEADS):
        raise ValueError('Full-slide CellViT is incomplete or does not contain all eight heads')
    index_path = Path(pointer.get('index_path', '')).resolve()
    index_hash = digest(index_path)
    if index_hash != receipt.get('index_sha256'):
        raise ValueError('Full-slide index hash differs from completion receipt')
    project_path = Path(pointer.get('project_path', '')).resolve()
    project = read_json(project_path)
    tissues = project.get('tissues', [])
    if len(tissues) != 1:
        raise ValueError('Primary CellViT conversion requires one exact source tissue')
    tissue = tissues[0]
    source_image = asset(project_path.parent, tissue.get('image'), 'Project image')
    frame = {key: tissue['image'][key] for key in ('sha256', 'width', 'height', 'mpp_x', 'mpp_y',
             'origin_x', 'origin_y', 'coordinate_space', 'coordinate_convention', 'frame_id')}
    frame_valid(frame)
    if dimensions(source_image) != (frame['width'], frame['height']) or frame['sha256'] != receipt.get('source_image_sha256'):
        raise ValueError('Full-slide source image differs from its registered frame')
    classification = tissue.get('cell_classification', {})
    bound_index = asset(project_path.parent, classification.get('index'), 'Project cell index')
    partition_path = asset(project_path.parent, classification, 'Project cell partition manifest')
    partition = read_json(partition_path)
    if (bound_index != index_path or digest(partition_path) != receipt.get('partition_manifest_sha256') or
        partition.get('image') != frame or partition.get('cell_count') != receipt.get('cell_count') or
        partition.get('partition_count') != planned or len(partition.get('partitions', [])) != planned):
        raise ValueError('Project, partition manifest and complete index disagree')
    if shift.get('full_size') != {'width': frame['width'], 'height': frame['height']}:
        raise ValueError('Analysis crop full-image dimensions differ from CellViT frame')
    crop_source = Path(shift.get('input_image', '')).resolve()
    if digest(crop_source) != frame['sha256']:
        raise ValueError('Analysis crop was prepared from a different full image')
    mpp = float(shift.get('source_mpp', shift.get('microns_per_pixel', 0)))
    if mpp <= 0 or any(abs(float(frame[k])-mpp)/mpp > .02 for k in ('mpp_x', 'mpp_y')):
        raise ValueError('Analysis crop MPP differs from CellViT frame')
    connection = sqlite3.connect('file:'+quote(str(index_path), safe='/')+'?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type IN ('view','trigger') LIMIT 1").fetchone():
            raise ValueError('Cell index cannot contain executable views or triggers')
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Cell index failed SQLite integrity checking')
        metadata = {key: json.loads(value) for key, value in connection.execute('SELECT key,value FROM metadata')}
        count = connection.execute('SELECT COUNT(*) FROM cells').fetchone()[0]
        if (metadata.get('schema') != 'cellphenotyper-cell-index' or metadata.get('image') != frame or
            metadata.get('classification_header', {}).get('image') != frame or
            metadata.get('classification_sha256') != receipt.get('partition_manifest_sha256') or
            metadata.get('cell_count') != receipt.get('cell_count') or count != receipt.get('cell_count')):
            raise ValueError('Cell index metadata/frame/count differs from complete source')
        primary_model = pannuke_model(metadata.get('classification_header', {}).get('models', []))
        rows = (json.loads(row[0]) for row in connection.execute('SELECT raw_json FROM cells ORDER BY row_id'))
        yield rows, frame, {'input_scope': 'full_slide', 'source_cell_count': count,
              'complete_shards': planned, 'planned_shards': planned,
              'project_pointer_sha256': digest(pointer_path), 'index_sha256': index_hash,
              'completion_receipt_sha256': digest(receipt_path),
              'primary_type_taxonomy': 'pannuke', 'primary_type_model': primary_model}
        if digest(index_path) != index_hash:
            raise ValueError('Source index changed while exporting segmentation')
    finally:
        connection.close()


@contextmanager
def roi_cells(cells_path, metadata_path, image_path, shift_path):
    payload = read_json(cells_path)
    metadata = read_json(metadata_path)
    expected = image_frame(image_path, shift_path)
    frame_valid(expected)
    if payload.get('model') != 'CellViT++' or payload.get('metadata') != metadata:
        raise ValueError('Native CellViT cells/metadata disagree')
    if any(metadata.get(key) != value for key, value in expected.items()) or metadata.get('source_image_sha256') != expected['sha256']:
        raise ValueError('Native CellViT cells do not match the exact crop image/frame')
    if payload.get('taxonomy') != 'pannuke' or metadata.get('taxonomy') != 'pannuke' or not isinstance(payload.get('type_map'), dict) or not payload['type_map']:
        raise ValueError('Native primary type view requires an explicitly declared PanNuke head/type map')
    cells = payload.get('cells')
    if not isinstance(cells, list):
        raise ValueError('Native CellViT cells must be an array')
    yield iter(cells), expected, {'input_scope': 'roi', 'source_cell_count': len(cells),
          'native_cells_sha256': digest(cells_path), 'native_metadata_sha256': digest(metadata_path),
          'primary_type_taxonomy': 'pannuke', 'primary_type_map': payload['type_map']}


def export_primary(image, shift_path, clean_mask, outdir, *, project_pointer=None,
                   cellvit_cells=None, cellvit_metadata=None, source_metadata=None,
                   tile_size=512, compression='deflate'):
    image, shift_path, clean_mask, outdir = map(Path, (image, shift_path, clean_mask, outdir))
    if bool(project_pointer) == bool(cellvit_cells):
        raise ValueError('Choose exactly one full-slide pointer or native CellViT cells input')
    if project_pointer and not source_metadata:
        raise ValueError('Full-slide conversion requires its bound completion receipt')
    if cellvit_cells and not cellvit_metadata:
        raise ValueError('Native cells conversion requires cellvit_metadata')
    if tile_size < 16 or tile_size > 4096 or tile_size % 16:
        raise ValueError('Raster tile size must be a multiple of 16 between 16 and 4096')
    width, height = dimensions(image)
    shift = read_json(shift_path)
    if shift.get('crop_size') != {'width': width, 'height': height}:
        raise ValueError('Shared crop dimensions differ from shift metadata')
    bbox = shift.get('crop_bbox_xyxy', {})
    if (any(type(bbox.get(k)) is not int for k in ('x0', 'y0', 'x1', 'y1')) or
        bbox['x1']-bbox['x0'] != width or bbox['y1']-bbox['y0'] != height or
        min(bbox['x0'], bbox['y0']) < 0):
        raise ValueError('Shared crop offset/dimensions are invalid')
    output_frame = image_frame(image, shift_path)
    frame_valid(output_frame)
    offset_x, offset_y = (bbox['x0'], bbox['y0']) if project_pointer else (0, 0)
    reader = fullslide_cells(project_pointer, source_metadata, shift) if project_pointer else roi_cells(cellvit_cells, cellvit_metadata, image, shift_path)
    records, identifiers, excluded = [], set(), 0
    crop_box = box(0, 0, width, height)
    with rasterio.open(clean_mask) as mask, reader as (cells, frame, lineage):
        if mask.count != 1 or (mask.width, mask.height) != (width, height):
            raise ValueError('Clean-tissue mask differs from the shared crop')
        for item in cells:
            identifier = item.get('cell_id')
            source_id = item.get('source_id', item.get('id'))
            if not isinstance(identifier, str) or not identifier or identifier in identifiers or not source_id:
                raise ValueError('CellViT cells require unique canonical IDs and retained source IDs')
            identifiers.add(identifier)
            centroid = [item.get('x'), item.get('y')] if project_pointer else item.get('centroid', [item.get('x'), item.get('y')])
            x, y = float(centroid[0])-offset_x, float(centroid[1])-offset_y
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError('CellViT centroid must be finite')
            if not (0 <= x < width and 0 <= y < height):
                excluded += 1
                continue
            from rasterio.windows import Window
            if mask.read(1, window=Window(int(x), int(y), 1, 1))[0, 0] == 0:
                excluded += 1
                continue
            contour = item.get('contour')
            if not isinstance(contour, list) or len(contour) < 3:
                raise ValueError('CellViT cells require original native contours')
            xy = [(float(p[0])-offset_x, float(p[1])-offset_y) for p in contour]
            if any(not math.isfinite(v) for p in xy for v in p):
                raise ValueError('CellViT contour must be finite')
            polygon = Polygon(xy)
            if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
                raise ValueError('CellViT contour is not a valid positive-area polygon')
            polygon = polygon.intersection(crop_box)
            if polygon.is_empty or polygon.area <= 0:
                raise ValueError('Retained CellViT cell has no geometry inside crop')
            records.append({'label': len(records)+1, 'x': x, 'y': y, 'polygon': polygon, 'support': 1,
                'cell_id': identifier, 'source_id': str(source_id), 'predictions': item.get('predictions', {}),
                'type_id': item.get('type_id'), 'type_name': item.get('type_name', item.get('type')),
                'type_probability': item.get('type_prob'), 'pannuke': pannuke_view(item, lineage)})
    outdir.mkdir(parents=True, exist_ok=False)
    seeds = write_mask(records, width, height, outdir/'labels.tif', tile_size, compression)
    coverage = validate_mask_label_coverage(outdir/'labels.tif', len(records))
    columns = ['label', 'area', 'x', 'y', 'xmin', 'ymin', 'xmax', 'ymax', 'mask_seed_x', 'mask_seed_y',
               'cell_id', 'source_id', 'geometry_source', 'cell_type', 'cell_type_id', 'cell_type_probability',
               'cellvitpp_type', 'cellvitpp_type_id', 'cellvitpp_probability', 'cellvitpp_type_status']
    with (outdir/'objects.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            xmin, ymin, xmax, ymax = record['polygon'].bounds
            writer.writerow(dict(label=record['label'], area=record['polygon'].area,
                x=record['x'], y=record['y'], xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                mask_seed_x=record['mask_seed_x'], mask_seed_y=record['mask_seed_y'],
                cell_id=record['cell_id'], source_id=record['source_id'], geometry_source='cellvitpp',
                cell_type=record['type_name'], cell_type_id=record['type_id'],
                cell_type_probability=record['type_probability'], cellvitpp_type=record['pannuke']['type'],
                cellvitpp_type_id=record['pannuke']['type_id'], cellvitpp_probability=record['pannuke']['probability'],
                cellvitpp_type_status=record['pannuke']['status']))
    with (outdir/'cell_instances.geojson').open('w') as stream:
        header = dict(type='FeatureCollection', coordinate_space='crop-native-pixels',
            coordinate_convention=CONVENTION, source_frame_id=output_frame['frame_id'],
            source_image_sha256=output_frame['sha256'])
        stream.write(json.dumps(header, separators=(',', ':'))[:-1]+',"features":[')
        for i, record in enumerate(records):
            if i: stream.write(',')
            feature = dict(type='Feature', id=record['cell_id'], geometry=mapping(record['polygon']),
                properties=dict(label=record['label'], cell_id=record['cell_id'], source_id=record['source_id'],
                    x=record['x'], y=record['y'], geometry_source='cellvitpp', predictions=record['predictions'],
                    primary_type_taxonomy='pannuke',
                    cell_type=record['type_name'], cell_type_id=record['type_id'], cell_type_probability=record['type_probability'],
                    cellvitpp_type=record['pannuke']['type'], cellvitpp_type_id=record['pannuke']['type_id'],
                    cellvitpp_probability=record['pannuke']['probability'], cellvitpp_type_status=record['pannuke']['status']))
            json.dump(feature, stream, separators=(',', ':'), allow_nan=False)
        stream.write(']}\n')
    metadata = dict(schema=SCHEMA, cell_detection_mode='cellvitpp', detector='CellViT++',
        geometry_source='cellvitpp', image_sha256=output_frame['sha256'], image=output_frame,
        frame_id=output_frame['frame_id'], shift_sha256=digest(shift_path),
        clean_tissue_mask_sha256=digest(clean_mask), width=width, height=height,
        coordinate_space='crop-native-pixels', coordinate_convention=CONVENTION,
        source_frame_id=frame['frame_id'], source_image_sha256=frame['sha256'],
        crop_to_source_offset_xy=[bbox['x0'], bbox['y0']], cell_count=len(records),
        excluded_centroids=excluded, support_policy='centroid-inside-shared-GrandQC-and-ROI-support',
        conversion_only=True, classifier_evidence_changed=False, primary_type_taxonomy='pannuke',
        primary_type_model_id=(lineage.get('primary_type_model') or {}).get('model_id'),
        other_classifier_heads='retained separately; no cross-taxonomy vote', seed_preservation=seeds,
        label_coverage=coverage, source=lineage,
        outputs={name: {'path': name, 'sha256': digest(outdir/name)} for name in
                 ('labels.tif', 'objects.csv', 'cell_instances.geojson')})
    (outdir/'segmentation_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('image', 'shift', 'clean-tissue-mask', 'outdir'):
        parser.add_argument('--'+option, type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--project-pointer', type=Path)
    sources.add_argument('--cellvit-cells', type=Path)
    parser.add_argument('--cellvit-metadata', type=Path)
    parser.add_argument('--source-metadata', type=Path)
    parser.add_argument('--tile-size', type=int, default=512)
    parser.add_argument('--compression', default='deflate')
    args = parser.parse_args(argv)
    result = export_primary(args.image, args.shift, args.clean_tissue_mask, args.outdir,
        project_pointer=args.project_pointer, cellvit_cells=args.cellvit_cells,
        cellvit_metadata=args.cellvit_metadata, source_metadata=args.source_metadata,
        tile_size=args.tile_size, compression=args.compression)
    print(json.dumps({'cell_count': result['cell_count'], 'geometry_source': result['geometry_source']}))


if __name__ == '__main__':
    main()
