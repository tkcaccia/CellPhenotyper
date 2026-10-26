#!/usr/bin/env python3
"""Stream immutable cell classifications into a bounded-query SQLite derivative.

No model, GPU, external database or optional Python package is required. Feature
vectors stay in server-side storage; viewport/cell/queue queries never send them.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import closing
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

SCHEMA = 'cellphenotyper-cell-index'
VERSION = 1
MAX_VALUE_BYTES = 8 * 1024 * 1024
MAX_PAGE = 2000
FORBIDDEN_FEATURES = {'cell_id', 'source_id', 'id', 'x', 'y', 'cx', 'cy', 'label', 'class_id',
                      'probability', 'prediction', 'review_label', 'review_status', 'split', 'subject_id', 'tissue_id'}


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _loads(value):
    def invalid(raw):
        raise ValueError('Nonfinite JSON value: ' + raw)
    return json.loads(value, object_pairs_hook=_pairs, parse_constant=invalid)


def _json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False)


class _Stream:
    """Incremental stdlib JSON decoder; retain at most one bounded cell value."""
    def __init__(self, path):
        self.file = Path(path).open(encoding='utf-8')
        self.buffer, self.pos, self.eof = '', 0, False
        self.decoder = json.JSONDecoder(object_pairs_hook=_pairs,
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError('Nonfinite JSON value: ' + raw)))

    def close(self):
        self.file.close()

    def more(self):
        self.buffer = self.buffer[self.pos:]
        self.pos = 0
        chunk = self.file.read(64 * 1024)
        self.eof = not chunk
        self.buffer += chunk
        if len(self.buffer) > MAX_VALUE_BYTES:
            raise ValueError('One JSON header/cell exceeds the explicit 8 MiB streaming value limit')

    def peek(self):
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                return ''
            self.more()

    def expect(self, token):
        if self.peek() != token:
            raise ValueError('Malformed classification JSON: expected ' + repr(token))
        self.pos += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                # A scalar number can end exactly at a chunk boundary. Read its
                # delimiter before accepting it (e.g. 1e+12 split after "1").
                if end == len(self.buffer) and not self.eof:
                    self.more()
                    continue
                if end < len(self.buffer) and self.buffer[end] not in ',]}: \r\n\t' and not self.eof:
                    self.more()
                    continue
                raw = self.buffer[self.pos:end]
                self.pos = end
                return value, raw
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError('Malformed or truncated classification JSON') from exc
                self.more()


def stream_classification(path):
    """Yield ('header', key, value) and ('cell', None, original_JSON_text).

    Top-level key order is unrestricted. Nested duplicate keys, malformed array
    separators and trailing content fail. Unicode is decoded incrementally by
    Python's text reader. No complete cell population is retained.
    """
    source, seen = _Stream(path), set()
    try:
        source.expect('{')
        if source.peek() == '}':
            raise ValueError('Classification object is empty')
        while True:
            key, _ = source.value()
            if not isinstance(key, str) or key in seen:
                raise ValueError('Unique top-level classification keys required')
            seen.add(key)
            source.expect(':')
            if key == 'cells':
                source.expect('[')
                if source.peek() != ']':
                    while True:
                        value, raw = source.value()
                        if not isinstance(value, dict):
                            raise ValueError('Classification cells must be JSON objects')
                        yield 'cell', None, raw
                        delimiter = source.peek()
                        if delimiter == ']':
                            break
                        source.expect(',')
                        if source.peek() == ']':
                            raise ValueError('Trailing cell-array comma is invalid')
                source.expect(']')
            else:
                value, _ = source.value()
                yield 'header', key, value
            if source.peek() == '}':
                break
            source.expect(',')
            if source.peek() == '}':
                raise ValueError('Trailing object comma is invalid')
        source.expect('}')
        if source.peek():
            raise ValueError('Trailing content after classification JSON')
        if 'cells' not in seen:
            raise ValueError('Classification cells array is required')
    finally:
        source.close()


def read_classification_header(path):
    """Read the complete top-level header with an empty cell list, bounded RAM."""
    result = {key: value for kind, key, value in stream_classification(path) if kind == 'header'}
    result['cells'] = []
    if result.get('schema') != 'cellphenotyper-cell-classification' or type(result.get('schema_version')) is not int or result['schema_version'] != 1:
        raise ValueError('Unsupported classification schema/version')
    return result


def _reference(path, root):
    path = Path(path).resolve()
    return dict(path=Path(os.path.relpath(path, root)).as_posix(), sha256=digest_file(path))


def _asset(reference, root):
    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
        raise ValueError('Explicit source path and SHA-256 reference required')
    path = (Path(root) / reference['path']).resolve()
    if reference.get('sha256') != digest_file(path):
        raise ValueError('Source hash mismatch: ' + str(path))
    return path


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric ' + name + ' required')
    return value


def _header(header, manifest_path):
    import re
    image, feature, models = header.get('image'), header.get('feature_set'), header.get('models')
    if not isinstance(image, dict) or not isinstance(feature, dict) or not isinstance(models, list) or not models or len(models) > 64:
        raise ValueError('Complete image, typed feature set and bounded model catalog required')
    for axis in ('width', 'height'):
        if type(image.get(axis)) is not int or image[axis] < 1:
            raise ValueError('Positive image dimensions required')
    for axis in ('mpp_x', 'mpp_y'):
        if _number(image.get(axis), axis) <= 0:
            raise ValueError('Calibrated positive MPP required')
    if (not re.fullmatch('[a-f0-9]{64}',str(image.get('sha256',''))) or
            not isinstance(image.get('frame_id'), str) or not image['frame_id'] or
            image.get('coordinate_convention') != 'pixel-edge-origin_centres-in-native-pixels' or
            image.get('coordinate_space') != 'crop-native-pixels' or image.get('origin_x') != 0 or image.get('origin_y') != 0):
        raise ValueError('Authoritative source-image hash and complete native frame required')
    columns = feature.get('feature_columns')
    if (feature.get('kind') not in ('cellvit-cell', 'cell-morphology') or feature.get('frame_id') != image['frame_id'] or
            feature.get('observation_unit') != 'cell' or feature.get('coordinate_frame') != 'image' or
            not isinstance(columns, list) or not 0 < len(columns) <= 4096 or
            any(not isinstance(key, str) or not key or key.lower() in FORBIDDEN_FEATURES for key in columns) or len(set(columns)) != len(columns)):
        raise ValueError('An explicit bounded typed-cell predictor allowlist is required')
    model_map = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get('model_id'), str) or not model['model_id'] or model['model_id'] in model_map:
            raise ValueError('Unique explicit model IDs required')
        classes = model.get('class_names')
        if (not isinstance(classes, list) or any(not isinstance(label, str) or not label for label in classes) or len(set(classes)) != len(classes) or
                model.get('status') not in ('available', 'unavailable') or (model['status'] == 'available' and not classes)):
            raise ValueError('Explicit model availability and ordered ontology required')
        if model.get('backbone') != feature.get('descriptor',{}).get('backbone') or not isinstance(model.get('taxonomy'),str) or not model['taxonomy']:
            raise ValueError('Model taxonomy/backbone differs from typed feature descriptor')
        if model['status']=='available':
            semantics={'mlp':'uncalibrated_model_probability','native_segmentation_head':'uncalibrated_pixel_type_vote_fraction',
                       'native_instance_head':'uncalibrated_softmax_cell_class_vote_counts'}
            if (model.get('kind') not in semantics or model.get('score_semantics')!=semantics[model['kind']] or
                    not isinstance(model.get('checkpoint_sha256'),str) or len(model['checkpoint_sha256'])!=64):
                raise ValueError('Available model requires exact checkpoint and score semantics')
            if model['kind']=='native_segmentation_head' and (model['taxonomy']!='pannuke' or model['checkpoint_sha256']!=feature['descriptor'].get('checkpoint_sha256')):
                raise ValueError('Native PanNuke model must match its detector checkpoint')
            if feature['kind']=='cell-morphology':
                if model['kind']!='native_instance_head' or model['taxonomy'] not in ('lizard','pannuke') or model['checkpoint_sha256']!=feature['descriptor'].get('checkpoint_sha256'):
                    raise ValueError('Morphology features require their native detector evidence')
            elif model['kind']=='native_instance_head':
                raise ValueError('HoVer-NeXt evidence cannot label CellViT token features')
        elif not isinstance(model.get('reason'),str) or not model['reason']:
            raise ValueError('Unavailable models require an explicit reason')
        model_map[model['model_id']] = model
    if Path(manifest_path).stat().st_size > MAX_VALUE_BYTES:
        raise ValueError('Feature manifest exceeds the bounded header limit')
    manifest = _loads(Path(manifest_path).read_text())
    if (header.get('provenance', {}).get('feature_manifest_sha256') != digest_file(manifest_path) or
            manifest.get('schema') != 'cellphenotyper-cell-features' or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1 or
            manifest.get('image') != image or {k:v for k,v in manifest.get('feature_set',{}).items() if k!='path'} != {k:v for k,v in feature.items() if k!='path'}):
        raise ValueError('Classification/feature manifest hash, image and feature definition must agree exactly')
    _asset(manifest.get('cell_metadata'), Path(manifest_path).parent)
    normalized = manifest.get('source', {}).get('normalized_segmentation')
    if normalized is not None:
        _asset(normalized, Path(manifest_path).parent)
        if header.get('provenance', {}).get('normalized_segmentation_sha256') != normalized['sha256']:
            raise ValueError('Normalized segmentation lineage mismatch')
    _asset(manifest['feature_set'],Path(manifest_path).parent)
    return image, feature, model_map


def _descriptor(feature):
    """Match the producer's typed-cell descriptor checks without a CSV ID set."""
    import re
    from export_experiment import NON_PREDICTORS
    d=feature.get('descriptor',{})
    morphology=feature['kind']=='cell-morphology'
    for key in ('backbone','runtime_version','feature_definition'):
        if not isinstance(d.get(key),str) or not d[key]:
            raise ValueError('Complete typed cell feature descriptor required')
    if type(d.get('feature_count')) is not int or d['feature_count']!=len(feature['feature_columns']) or d.get('coordinate_frame')!='image' or d.get('observation_unit')!='cell':
        raise ValueError('Typed descriptor width/unit/frame differs from feature allowlist')
    if not re.fullmatch('[a-fA-F0-9]{64}',str(d.get('checkpoint_sha256',''))):
        raise ValueError('Typed descriptor checkpoint SHA-256 required')
    if NON_PREDICTORS.intersection(feature['feature_columns']):
        raise ValueError('Cell identity, labels and measurement metadata cannot become feature predictors')
    if morphology:
        from run_hovernext import FEATURE_COLUMNS
        if (d.get('detector')!='HoVer-NeXt' or d['backbone'] not in ('HoVer-NeXt-ConvNeXtV2-Tiny','HoVer-NeXt-ConvNeXtV2-Base','HoVer-NeXt-ConvNeXtV2-Large') or
                d['feature_definition']!='hovernext-native-instance-morphology-rgb-v1' or
                not re.fullmatch('[a-f0-9]{40}',str(d.get('upstream_source_commit',''))) or d['runtime_version']!=d['upstream_source_commit'] or
                not re.fullmatch('[a-f0-9]{32}',str(d.get('checkpoint_archive_md5',''))) or feature['feature_columns']!=FEATURE_COLUMNS):
            raise ValueError('Unsupported morphology representation signature')
    elif d['backbone'] not in ('HIPT','SAM','Virchow2'):
        raise ValueError('Unsupported cell token backbone')
    sampling,pre=d.get('sampling'),d.get('preprocessing')
    if not isinstance(sampling,dict) or not sampling or not isinstance(pre,dict) or not pre:
        raise ValueError('Actual sampling and preprocessing metadata required')
    if _number(sampling.get('model_input_mpp'),'model MPP')<=0:
        raise ValueError('Positive model MPP required')
    for key in (() if morphology else ('patch_size_px','token_size_px')):
        if type(sampling.get(key)) is not int or sampling[key]<=0:
            raise ValueError('Positive integral model patch/token sizes required')
    if morphology:
        if sampling.get('resampling')!='none':
            raise ValueError('Native morphology requires no resampling')
    elif type(sampling.get('patch_overlap_px')) is not int or not 0<=sampling['patch_overlap_px']<sampling['patch_size_px']:
        raise ValueError('Invalid model patch overlap')
    if not morphology and not re.fullmatch('[a-fA-F0-9]{64}',str(pre.get('runtime_wheel_sha256',''))):
        raise ValueError('Immutable runtime wheel hash required')
    for key in (('mask_geometry','colour') if morphology else ('normalization','resampling','token_provenance')):
        if not isinstance(pre.get(key),str) or not pre[key]:
            raise ValueError('Complete typed preprocessing metadata required')
    if feature.get('encoder') is not None:
        encoder=feature['encoder']
        if (encoder.get('name')!=('HoVer-NeXt' if morphology else 'CellViT++') or encoder.get('revision')!=d['runtime_version'] or
                encoder.get('checkpoint_sha256')!=d['checkpoint_sha256'] or ('backbone' in encoder and encoder['backbone']!=d['backbone'])):
            raise ValueError('Declared encoder contradicts typed descriptor')
    if feature.get('preprocessing',{}).get('feature_definition',d['feature_definition'])!=d['feature_definition']:
        raise ValueError('Preprocessing feature definition contradicts descriptor')
    _json(d)


def _verify_lineage(connection, manifest_path, image):
    """Disk-backed exact metadata/normalized source joins, without full arrays."""
    manifest=_loads(Path(manifest_path).read_text())
    connection.execute('CREATE TEMP TABLE lineage_seen(kind TEXT,row_id INTEGER,PRIMARY KEY(kind,row_id))')
    entries=[('cell_metadata',manifest['cell_metadata'])]
    if manifest.get('source',{}).get('normalized_segmentation') is not None:
        entries.append(('normalized',manifest['source']['normalized_segmentation']))
    for kind,reference in entries:
        path=_asset(reference,Path(manifest_path).parent)
        frame={};count=0
        for category,key,value in stream_classification(path):
            if category=='header':
                frame[key]=value
                continue
            cell=_loads(value)
            row=connection.execute('SELECT row_id,raw_json FROM cells WHERE cell_id=?',(cell.get('cell_id'),)).fetchone()
            if row is None:
                raise ValueError('Source lineage has an unknown canonical cell ID')
            expected=_loads(row[1])
            connection.execute('INSERT INTO lineage_seen VALUES(?,?)',(kind,row[0]))
            if kind=='cell_metadata':
                if any(expected.get(k)!=v for k,v in cell.items()):
                    raise ValueError('Feature cell metadata contradicts source classification')
            elif (cell.get('source_id')!=expected['source_id'] or str(cell.get('id'))!=expected['source_id'] or
                    cell.get('centroid')!=[expected['x'],expected['y']] or
                    any(cell.get(k)!=expected[k] for k in ('bbox','contour') if k in expected)):
                raise ValueError('Normalized segmentation cell identity/geometry differs from classification')
            count+=1
        if count!=connection.execute('SELECT COUNT(*) FROM cells').fetchone()[0]:
            raise ValueError('Source lineage and indexed cells must be an exact population')
        if kind=='cell_metadata' and frame.get('image')!=image:
            raise ValueError('Feature cell metadata image/frame differs from classification')
        if kind=='normalized':
            metadata=frame.get('metadata',{})
            if metadata.get('source_image_sha256')!=image['sha256'] or any(metadata.get(k)!=v for k,v in image.items()):
                raise ValueError('Normalized segmentation image/frame differs from classification')
    connection.execute('DROP TABLE lineage_seen')


def _prediction(cell, models):
    values = cell.get('predictions')
    if not isinstance(values, dict) or set(values) != set(models):
        raise ValueError('Every indexed cell must explicitly cover every declared model')
    result = []
    for mid, predicted in values.items():
        if not isinstance(predicted, dict) or predicted.get('status') not in ('predicted', 'unavailable', 'abstained'):
            raise ValueError('Explicit prediction status required')
        status, model = predicted['status'], models[mid]
        class_id, label, score = predicted.get('class_id'), predicted.get('label'), predicted.get('probability')
        vector = predicted.get('probabilities')
        if status == 'predicted':
            if (model['status'] != 'available' or type(class_id) is not int or not 0 <= class_id < len(model['class_names']) or
                    label != model['class_names'][class_id]):
                raise ValueError('Prediction differs from its available ordered model ontology')
            no_score = (model.get('kind') == 'native_instance_head' and score is None and vector is None and
                        predicted.get('probability_vector_status') == 'not_exported_by_upstream')
            if not no_score and not 0 <= _number(score, 'probability') <= 1:
                raise ValueError('Probability must be in [0,1]')
            if vector is not None:
                if (not isinstance(vector, list) or len(vector) != len(model['class_names']) or
                        any(not 0 <= _number(v, 'probability vector') <= 1 for v in vector) or
                        abs(sum(vector)-1) > 1e-5 or abs(vector[class_id]-score) > 1e-7):
                    raise ValueError('Prediction vector differs from its ontology or selected score')
            elif not (no_score or (model.get('kind') == 'native_segmentation_head' and predicted.get('probability_vector_status') == 'not_exported_by_upstream')):
                raise ValueError('Missing prediction vector must be explicit upstream native output')
        elif any(value is not None for value in (class_id, label, score, vector)):
            raise ValueError('Unavailable or abstained evidence cannot contain a biological assignment')
        result.append((mid, status, class_id, label, score))
    return result


def _cell(cell, image):
    _json(cell)  # Reject nonfinite/unsupported metadata anywhere in this row.
    cid, sid = cell.get('cell_id'), cell.get('source_id')
    if not isinstance(cid, str) or not cid.strip() or not isinstance(sid, str) or not sid.strip():
        raise ValueError('Immutable canonical and source cell IDs required')
    x, y = _number(cell.get('x'), 'cell x'), _number(cell.get('y'), 'cell y')
    if not 0 <= x < image['width'] or not 0 <= y < image['height']:
        raise ValueError('Cell centroid lies outside the native image')
    contour = cell.get('contour')
    if contour:
        if not isinstance(contour, list) or len(contour) < 3:
            raise ValueError('Explicit original contour coordinates required')
        points = []
        for p in contour:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError('Contour points must be native XY pairs')
            points.append((_number(p[0], 'contour x'), _number(p[1], 'contour y')))
        xs, ys = zip(*points)
        bounds = min(xs), min(ys), max(xs), max(ys)
    elif cell.get('bbox') is not None:
        bbox = cell['bbox']
        if not isinstance(bbox, list) or len(bbox) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in bbox):
            raise ValueError('Original bbox must be [[ymin,xmin],[ymax,xmax]]')
        bounds = tuple(_number(v, 'bbox') for v in (bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0]))
    else:
        bounds = x, y, x, y
    if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
        raise ValueError('Original cell bbox is reversed')
    return cid, sid, x, y, *bounds


SQL = '''
CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE cells(row_id INTEGER PRIMARY KEY,cell_id TEXT NOT NULL UNIQUE,source_id TEXT NOT NULL UNIQUE,
 x REAL NOT NULL,y REAL NOT NULL,xmin REAL NOT NULL,ymin REAL NOT NULL,xmax REAL NOT NULL,ymax REAL NOT NULL,raw_json TEXT NOT NULL);
CREATE TABLE predictions(row_id INTEGER NOT NULL,model_id TEXT NOT NULL,status TEXT NOT NULL,class_id INTEGER,label TEXT,probability REAL,
 PRIMARY KEY(row_id,model_id),FOREIGN KEY(row_id) REFERENCES cells(row_id));
CREATE TABLE features(row_id INTEGER PRIMARY KEY,values_json TEXT NOT NULL,FOREIGN KEY(row_id) REFERENCES cells(row_id));
CREATE INDEX cells_x ON cells(xmin,xmax);
CREATE INDEX cells_y ON cells(ymin,ymax);
CREATE INDEX predictions_queue ON predictions(model_id,probability,row_id);
CREATE INDEX predictions_class ON predictions(model_id,class_id,status,row_id);
'''


def _features(connection, feature, classification_path, verify_only=False):
    path = _asset(feature, Path(classification_path).parent)
    columns = feature['feature_columns']
    expected_format='csv.gz' if path.name.lower().endswith('.csv.gz') else 'csv' if path.suffix.lower()=='.csv' else None
    if expected_format is None or feature.get('format')!=expected_format:
        raise ValueError('Feature CSV format must match the explicit source filename')
    keys = [feature.get(k) for k in ('id_column', 'x_column', 'y_column')]
    if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != 3 or set(keys) & set(columns):
        raise ValueError('Explicit distinct feature ID and native coordinate columns required')
    count = 0
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or any(key not in names for key in [*keys,'source_id',*columns]):
            raise ValueError('Feature CSV is missing declared columns or has duplicates')
        for row in reader:
            if None in row or any(row.get(key) is None for key in [*keys,'source_id',*columns]):
                raise ValueError('Malformed indexed feature row')
            found = connection.execute('SELECT row_id,source_id,x,y FROM cells WHERE cell_id=?', (row[keys[0]],)).fetchone()
            if found is None or row['source_id'] != found[1]:
                raise ValueError('Feature row has an unknown canonical/source cell ID')
            values = [float(row[key]) for key in columns]
            if not all(math.isfinite(v) for v in [float(row[keys[1]]),float(row[keys[2]]),*values]) or float(row[keys[1]]) != found[2] or float(row[keys[2]]) != found[3]:
                raise ValueError('Feature values or native coordinates differ from indexed cells')
            if verify_only:
                existing=connection.execute('SELECT values_json FROM features WHERE row_id=?',(found[0],)).fetchone()
                if existing is None or _loads(existing[0])!=values:
                    raise ValueError('Indexed training vector differs from immutable source features')
            else:
                connection.execute('INSERT INTO features VALUES(?,?)', (found[0], _json(values)))
            count += 1
    if count != connection.execute('SELECT COUNT(*) FROM cells').fetchone()[0] or feature.get('row_count') != count:
        raise ValueError('Indexed feature/cell population must be an exact bijection')
    if digest_file(path) != feature['sha256']:
        raise ValueError('Feature CSV changed during indexing')
    return path


def _stream_review_identity_rows(connection, path):
    # A temporary disk database detects duplicate aliases even while verifying
    # an immutable read-only index. No complete review/cell population is kept.
    with closing(sqlite3.connect('')) as seen:
        seen.execute('PRAGMA cache_size=-1024')
        seen.execute('CREATE TABLE seen(row_id INTEGER PRIMARY KEY)')
        for kind, _, raw in stream_classification(path):
            if kind != 'cell':
                continue
            row = _loads(raw)
            row_id = row.get('cell_id')
            if not isinstance(row_id, str) or not row_id.strip():
                raise ValueError('Segmentation review requires a canonical/source cell ID')
            candidates = connection.execute('SELECT row_id,cell_id,source_id,raw_json FROM cells WHERE cell_id=? OR source_id=?',
                (row_id, row_id)).fetchall()
            if len(candidates) != 1:
                raise ValueError('Segmentation review row has an unknown or ambiguous canonical/source cell ID')
            try:
                seen.execute('INSERT INTO seen VALUES(?)', (candidates[0][0],))
            except sqlite3.IntegrityError as error:
                raise ValueError('Segmentation review has duplicate immutable cell IDs') from error
            yield raw, row, candidates[0]


def _index_segmentation_review(connection, path, image, manifest_path, destination_root, *, verify_only=False):
    """Validate/join one QC row at a time; original classification bytes stay intact."""
    from segmentation_review_validation import (SCHEMA as REVIEW_SCHEMA, GEOMETRY_COUNTS, explicit_geometry_status,
        validate_header, validate_row, validate_counts)
    path = Path(path).resolve()
    source_hash = digest_file(path)
    header, row_has_status = {}, False
    for kind, key, raw in stream_classification(path):
        if kind == 'header':
            header[key] = raw
        elif 'segmentation_review_status' in _loads(raw):
            row_has_status = True
    explicit = explicit_geometry_status(header) or row_has_status
    manifest = _loads(Path(manifest_path).read_text())
    reference = dict(_reference(path, destination_root), role='segmentation_review',
        source_frame_id=image['frame_id'], coordinate_convention=image['coordinate_convention'])
    input_hash = validate_header(reference, header, image, manifest, explicit=explicit)
    if not verify_only:
        connection.execute('CREATE TABLE segmentation_review(row_id INTEGER PRIMARY KEY,cell_id TEXT NOT NULL UNIQUE,'
            'source_id TEXT NOT NULL UNIQUE,qc_json TEXT NOT NULL,FOREIGN KEY(row_id) REFERENCES cells(row_id))')
    count, statuses = 0, dict(available=0, unavailable=0)
    geometry_counts = dict.fromkeys(GEOMETRY_COUNTS, 0)
    with closing(_stream_review_identity_rows(connection, path)) as rows:
        for raw, row, candidate in rows:
            ordinal, cid, sid, original_raw = candidate
            original = _loads(original_raw)
            status = validate_row(row, original, explicit=explicit, quarantine=header.get('quarantine_invalid_geometry'),
                quarantine_overlapping=header.get('quarantine_overlapping_geometry', False))
            statuses[status] += 1
            if status == 'unavailable':
                geometry_counts[next(reason for reason in row['segmentation_review_reason'] if reason in GEOMETRY_COUNTS)] += 1
            count += 1
            if verify_only:
                stored = connection.execute('SELECT cell_id,source_id,qc_json FROM segmentation_review WHERE row_id=?', (ordinal,)).fetchone()
                if stored is None or tuple(stored) != (cid, sid, raw):
                    raise ValueError('Indexed segmentation review differs from its immutable source row')
            else:
                connection.execute('INSERT INTO segmentation_review VALUES(?,?,?,?)', (ordinal, cid, sid, raw))
    total = connection.execute('SELECT COUNT(*) FROM cells').fetchone()[0]
    indexed_count = connection.execute('SELECT COUNT(*) FROM segmentation_review').fetchone()[0]
    if count != total or indexed_count != total or connection.execute(
            'SELECT 1 FROM segmentation_review q LEFT JOIN cells c USING(row_id) '
            'WHERE c.row_id IS NULL OR c.cell_id!=q.cell_id OR c.source_id!=q.source_id LIMIT 1').fetchone():
        raise ValueError('Segmentation review must cover every original cell exactly once')
    validate_counts(header, total, statuses, explicit=explicit, geometry_counts=geometry_counts)
    if digest_file(path) != source_hash:
        raise ValueError('Segmentation review source changed during indexing')
    return dict(reference, schema=REVIEW_SCHEMA, indexed_table='segmentation_review',
        frame=header['frame'], feature_manifest_sha256=digest_file(manifest_path), source_cells_sha256=input_hash,
        cell_count=count, original_cell_count=count, candidate_cell_count=count,
        rasterized_cell_count=statuses['available'], geometry_unavailable_cell_count=statuses['unavailable'],
        explicit_geometry_status=explicit, quarantine_invalid_geometry=header.get('quarantine_invalid_geometry', False),
        quarantine_overlapping_geometry=header.get('quarantine_overlapping_geometry', False),
        **{key: geometry_counts[reason] for reason,key in GEOMETRY_COUNTS.items()},
        reference_quality='model_agreement_only', cell_type_evidence_changed=False)


def build_index(classification_path, manifest_path, output_path, *, segmentation_review_path=None):
    """Build an immutable index using bounded JSON/CSV streaming and disk joins."""
    classification_path, manifest_path, output_path = map(lambda p: Path(p).resolve(), (classification_path,manifest_path,output_path))
    if output_path.exists() or output_path.with_suffix(output_path.suffix+'.json').exists():
        raise FileExistsError('Choose a NEW SQLite index destination')
    source_hash = digest_file(classification_path)
    header = read_classification_header(classification_path)
    image, feature, models = _header(header, manifest_path)
    _descriptor(feature)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.cell-index-',suffix='.sqlite',dir=output_path.parent)
    os.close(handle)
    connection = None
    try:
        connection = sqlite3.connect(temporary)
        connection.execute('PRAGMA cache_size=-8192')
        connection.execute('PRAGMA temp_store=FILE')
        connection.execute('PRAGMA journal_mode=DELETE')
        connection.executescript(SQL)
        spatial_index = 'rtree'
        try:
            connection.execute('CREATE VIRTUAL TABLE bbox_rtree USING rtree(row_id,xmin,xmax,ymin,ymax)')
        except sqlite3.OperationalError:
            spatial_index = 'btree'
        count = 0
        for kind, _, raw in stream_classification(classification_path):
            if kind != 'cell':
                continue
            cell = _loads(raw)
            identity = _cell(cell, image)
            predictions = _prediction(cell, models)
            count += 1
            connection.execute('INSERT INTO cells VALUES(?,?,?,?,?,?,?,?,?,?)', (count,*identity,raw))
            if spatial_index == 'rtree':
                connection.execute('INSERT INTO bbox_rtree VALUES(?,?,?,?,?)', (count,identity[4],identity[6],identity[5],identity[7]))
            connection.executemany('INSERT INTO predictions VALUES(?,?,?,?,?,?)', [(count,*p) for p in predictions])
        feature_path = _features(connection, feature, classification_path)
        _verify_lineage(connection,manifest_path,image)
        sidecars = []
        if segmentation_review_path is not None:
            sidecars.append(_index_segmentation_review(connection, segmentation_review_path, image, manifest_path, output_path.parent))
        if digest_file(classification_path) != source_hash:
            raise ValueError('Classification source changed during streaming; index publication refused')
        rebased_header=copy.deepcopy(header)
        rebased_header['feature_set']['path']=_reference(feature_path,output_path.parent)['path']
        metadata = dict(schema=SCHEMA,schema_version=VERSION,classification_header=rebased_header,source_classification_header=header,
            classification=_reference(classification_path,output_path.parent),classification_sha256=source_hash,
            feature_manifest=_reference(manifest_path,output_path.parent),feature_manifest_sha256=digest_file(manifest_path),
            feature_csv=_reference(feature_path,output_path.parent),feature_csv_sha256=feature['sha256'],
            image=image,feature_set=rebased_header['feature_set'],models=header['models'],cell_count=count,prediction_count=count*len(models),
            spatial_index=spatial_index,feature_storage='ordered-server-only-JSON-vector-per-cell',
            sidecars=sidecars,review_state='separate-user-project-state',
            conversion_provenance=dict(source='immutable-classification-and-feature-manifest',
                cells_streamed=True,full_population_loaded=False,source_labels_changed=False,lineage_verified=True,
                scientific_accuracy_qualified=False,
                segmentation_review_indexed=bool(sidecars), segmentation_review_population_loaded=False))
        connection.executemany('INSERT INTO metadata VALUES(?,?)', [(k,_json(v)) for k,v in metadata.items()])
        connection.commit()
        connection.execute('PRAGMA optimize')
        connection.close();connection = None
        # A hard-link publication is atomic and refuses a destination race.
        os.link(temporary, output_path)
        sidecar = dict(metadata,index=_reference(output_path,output_path.parent),format='sqlite')
        with output_path.with_suffix(output_path.suffix+'.json').open('x') as stream:
            json.dump(sidecar,stream,indent=2,allow_nan=False);stream.write('\n')
        return metadata
    except sqlite3.IntegrityError as exc:
        raise ValueError('Duplicate or contradictory indexed cell/feature identity') from exc
    finally:
        if connection is not None:
            connection.close()
        Path(temporary).unlink(missing_ok=True)


def _connect(path,timeout_seconds=30):
    uri = Path(path).resolve().as_uri()+'?mode=ro'
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA trusted_schema=OFF')
    deadline = time.monotonic()+timeout_seconds
    connection.set_progress_handler(lambda: int(time.monotonic()>deadline),10000)
    return connection


def read_index_metadata(index_path, connection=None):
    """Inspect only fixed SQLite data tables. Arbitrary views/triggers are refused."""
    owned = connection is None
    connection = connection or _connect(index_path)
    try:
        objects = connection.execute("SELECT type,name FROM sqlite_master WHERE type IN ('view','trigger')").fetchall()
        if objects:
            raise ValueError('Cell indexes cannot contain executable views or triggers')
        expected = {'metadata':{'key','value'},'cells':{'row_id','cell_id','source_id','x','y','xmin','ymin','xmax','ymax','raw_json'},
                    'predictions':{'row_id','model_id','status','class_id','label','probability'},'features':{'row_id','values_json'}}
        for table, columns in expected.items():
            info = connection.execute('PRAGMA table_info('+table+')').fetchall()
            if {row['name'] for row in info} != columns:
                raise ValueError('Unsupported cell-index data table: '+table)
        result = {row['key']:_loads(row['value']) for row in connection.execute('SELECT key,value FROM metadata')}
        partitioned = (result.get('source_format') == 'cellphenotyper-cellvit-shards-v1' and
                       ((result.get('schema_version') == 2 and
                         result.get('feature_storage') == 'ordered-server-only-JSON-vector-per-cell') or
                        (result.get('schema_version') == 3 and
                         result.get('feature_storage') == 'partition-row-reference-v1')))
        if (result.get('schema') != SCHEMA or type(result.get('schema_version')) is not int or
                not (result['schema_version'] == VERSION or partitioned)):
            raise ValueError('Unsupported cell-index schema/version')
        sidecars = result.get('sidecars', [])
        if (not isinstance(sidecars, list) or len(sidecars) > 1 or
                any(not isinstance(ref, dict) or ref.get('role') != 'segmentation_review' or
                    ref.get('indexed_table') != 'segmentation_review' for ref in sidecars)):
            raise ValueError('Unsupported indexed cell review/context sidecar metadata')
        if sidecars:
            info = connection.execute('PRAGMA table_info(segmentation_review)').fetchall()
            if {row['name'] for row in info} != {'row_id', 'cell_id', 'source_id', 'qc_json'}:
                raise ValueError('Indexed segmentation review requires its checked data table')
        elif connection.execute("SELECT 1 FROM sqlite_master WHERE name='segmentation_review'").fetchone():
            raise ValueError('Indexed segmentation review table lacks its immutable sidecar binding')
        return result
    finally:
        if owned:
            connection.close()


def validate_index(index_path, classification_path, manifest_path, *, segmentation_review_path=None):
    """Rehash authoritative sources and validate index/header/population invariants.

    The experiment reference must independently bind the index SHA-256. This
    validation establishes derivative consistency, not source authenticity or
    learned-model accuracy. No full classification population is loaded.
    """
    index_path = Path(index_path).resolve()
    with closing(_connect(index_path,180)) as connection:
        metadata = read_index_metadata(index_path,connection)
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite cell index failed integrity checking')
        for key, path in (('classification',classification_path),('feature_manifest',manifest_path)):
            if _asset(metadata.get(key),index_path.parent) != Path(path).resolve() or metadata.get(key+'_sha256') != digest_file(path):
                raise ValueError('Cell index belongs to a different '+key)
        header = read_classification_header(classification_path)
        image,feature,models=_header(header,manifest_path)
        _descriptor(feature)
        rebased=copy.deepcopy(header)
        rebased['feature_set']['path']=metadata['feature_csv']['path']
        if header != metadata.get('source_classification_header') or rebased!=metadata.get('classification_header') or any(rebased.get(k) != metadata.get(k) for k in ('image','feature_set','models')):
            raise ValueError('Indexed source image/frame/models/feature header differs from current classification')
        _asset(metadata.get('feature_csv'),index_path.parent)
        if metadata.get('feature_csv_sha256') != header['feature_set']['sha256']:
            raise ValueError('Indexed feature CSV signature differs from classification')
        counts = {table:connection.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ('cells','predictions','features')}
        if (counts['cells'] != metadata.get('cell_count') or counts['features'] != counts['cells'] or
                counts['predictions'] != metadata.get('prediction_count') or counts['predictions'] != counts['cells']*len(metadata['models']) or
                header['feature_set'].get('row_count') != counts['cells']):
            raise ValueError('Cell-index declared and actual populations differ')
        if connection.execute('SELECT 1 FROM features f LEFT JOIN cells c USING(row_id) WHERE c.row_id IS NULL LIMIT 1').fetchone():
            raise ValueError('Cell index contains orphaned feature rows')
        if metadata.get('spatial_index') == 'rtree' and connection.execute('SELECT COUNT(*) FROM bbox_rtree').fetchone()[0] != counts['cells']:
            raise ValueError('Cell-index spatial population differs')
        if metadata.get('spatial_index') == 'rtree' and connection.execute('SELECT 1 FROM cells c LEFT JOIN bbox_rtree r USING(row_id) WHERE r.row_id IS NULL OR r.xmin>c.xmin OR r.xmax<c.xmax OR r.ymin>c.ymin OR r.ymax<c.ymax LIMIT 1').fetchone():
            raise ValueError('Cell-index spatial bounds omit original cell geometry')
        count=0
        for kind,_,raw in stream_classification(classification_path):
            if kind!='cell':
                continue
            count+=1
            original=_loads(raw)
            identity=_cell(original,image)
            stored=connection.execute('SELECT cell_id,source_id,x,y,xmin,ymin,xmax,ymax,raw_json FROM cells WHERE row_id=?',(count,)).fetchone()
            if stored is None or tuple(stored)!=(*identity,raw):
                raise ValueError('Indexed cell IDs/geometry/raw evidence differs from immutable classification')
            predictions=_prediction(original,models)
            stored_predictions=[tuple(row) for row in connection.execute('SELECT model_id,status,class_id,label,probability FROM predictions WHERE row_id=? ORDER BY model_id',(count,))]
            if sorted(predictions)!=stored_predictions:
                raise ValueError('Indexed prediction differs from immutable classification')
        if count!=counts['cells']:
            raise ValueError('Indexed and immutable classification populations differ')
        _features(connection,feature,classification_path,verify_only=True)
        sidecars = metadata.get('sidecars', [])
        if sidecars:
            review_path = _asset(sidecars[0], index_path.parent)
            if segmentation_review_path is not None and review_path != Path(segmentation_review_path).resolve():
                raise ValueError('Cell index belongs to a different segmentation review sidecar')
            actual = _index_segmentation_review(connection, review_path, image, manifest_path, index_path.parent, verify_only=True)
            if actual != sidecars[0]:
                raise ValueError('Indexed segmentation review metadata differs from its immutable source')
        elif segmentation_review_path is not None:
            raise ValueError('Cell index has no registered segmentation review sidecar')
        if metadata.get('conversion_provenance',{}).get('lineage_verified') is not True:
            raise ValueError('Cell index lacks explicit verified source lineage')
        return metadata


def _partition_review_features(index_path, metadata, rows):
    """Resolve a bounded reviewed subset from checked shard storage on demand."""
    from cellvit_shards import (materialize_feature_matrix, materialize_shard_records,
                                verify_complete_shard)
    from feature_matrix import MANIFEST as MATRIX_MANIFEST, iter_rows as iter_matrix_rows
    partition_manifest = _asset(metadata['classification'], Path(index_path).parent)
    source = _loads(partition_manifest.read_text())
    root = partition_manifest.parent
    plan = _loads((root / 'cellvit_shard_plan.json').read_text())
    shards = {shard['id']: shard for shard in plan['shards']}
    entries = {entry['id']: entry for entry in source['partitions']}
    expected = {}
    for row in rows:
        locator = _loads(row['values_json'])
        if (not isinstance(locator, dict) or set(locator) != {'partition_id', 'row'} or
                locator['partition_id'] not in shards or type(locator['row']) is not int or locator['row'] < 0):
            raise ValueError('Reviewed feature locator is malformed')
        expected.setdefault(locator['partition_id'], {})[locator['row']] = row
    if len(expected) > 16:
        raise ValueError('Reviewed subset spans more than 16 partitions; request smaller batches')
    results = {}
    columns = metadata['feature_set']['feature_columns']
    for pid, wanted in expected.items():
        entry, shard = entries[pid], shards[pid]
        if max(wanted) >= entry['cell_count']:
            raise ValueError('Reviewed feature row is outside its partition')
        directory = root / 'shards' / pid
        marker = verify_complete_shard(plan, shard, directory)
        if digest_file(directory / 'complete.json') != entry['checkpoint_sha256'] or marker['cell_count'] != entry['cell_count']:
            raise ValueError('Reviewed feature shard differs from checked project')
        def accept(ordinal, meta, values):
            if ordinal not in wanted:
                return
            indexed = wanted[ordinal]
            if (meta[0] != indexed['cell_id'] or
                    pid + ':' + meta[1] != indexed['source_id'] or
                    (float(meta[2]) + shard['input'][0], float(meta[3]) + shard['input'][1]) !=
                    (indexed['x'], indexed['y'])):
                raise ValueError('Reviewed feature ID or coordinates differ from indexed cell')
            values = [float(value) for value in values]
            if len(values) != len(columns) or not all(math.isfinite(value) for value in values):
                raise ValueError('Reviewed feature row has wrong width or nonfinite values')
            results[indexed['cell_id']] = values
        if entry.get('feature_storage') is not None:
            with materialize_feature_matrix(directory) as bundle:
                matrix_manifest = _loads((bundle / MATRIX_MANIFEST).read_text())
                if matrix_manifest.get('columns') != ['cell_id', 'source_id', 'x', 'y', *columns]:
                    raise ValueError('Reviewed matrix columns differ from project')
                for ordinal, (meta, values) in enumerate(iter_matrix_rows(bundle)):
                    accept(ordinal, meta, values)
        else:
            with materialize_shard_records(directory) as materialized:
                feature_manifest = _loads((materialized / 'cellvit_cell_features.manifest.json').read_text())
                feature = feature_manifest['feature_set']
                if feature['feature_columns'] != columns:
                    raise ValueError('Reviewed feature columns differ from project')
                csv_path = materialized / feature['path']
                opener = gzip.open if csv_path.name.endswith('.gz') else open
                with opener(csv_path, 'rt', newline='', encoding='utf-8-sig') as stream:
                    reader = csv.DictReader(stream)
                    if reader.fieldnames != ['cell_id', 'source_id', 'x', 'y', *columns]:
                        raise ValueError('Reviewed feature table has unexpected columns')
                    for ordinal, feature_row in enumerate(reader):
                        accept(ordinal, [feature_row[key] for key in ('cell_id', 'source_id', 'x', 'y')],
                               [feature_row[name] for name in columns])
            if len(results.keys() & {row['cell_id'] for row in wanted.values()}) != len(wanted):
                raise ValueError('Reviewed feature locator has no matching source row')
    return results


def query_index(index_path, mode='metadata', *, bbox=None, cell_id=None, model_id=None, class_id=None,
                status=None, max_probability=None, limit=500, after_row_id=0, cell_ids=None, include_geometry=True):
    """Bounded deterministic keyset queries. Features are returned only by explicit
    server-only review_subset, never ordinary browser viewport/cell/queue modes.
    """
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE or type(after_row_id) is not int or after_row_id < 0:
        raise ValueError('Explicit bounded limit and nonnegative keyset cursor required')
    with closing(_connect(index_path)) as connection:
        metadata = read_index_metadata(index_path,connection)
        base = dict(source_sha256=metadata['classification_sha256'],cell_count=metadata['cell_count'])
        if mode == 'metadata':
            return dict(base,metadata=metadata,cells=[],total=metadata['cell_count'],truncated=False,next_cursor=None)
        if model_id is not None and model_id not in {model['model_id'] for model in metadata['models']}:
            raise ValueError('Unknown indexed model ID')
        clauses, parameters, join = [], [], ''
        if model_id is not None:
            join = ' JOIN predictions p ON p.row_id=c.row_id'
            clauses.append('p.model_id=?');parameters.append(model_id)
        elif any(value is not None for value in (class_id,status,max_probability)):
            raise ValueError('Prediction filters require an explicit model ID')
        if class_id is not None:
            if type(class_id) is not int or class_id < 0:
                raise ValueError('Nonnegative class ID required')
            clauses.append('p.class_id=?');parameters.append(class_id)
        if status is not None:
            if status not in ('predicted','unavailable','abstained'):
                raise ValueError('Unknown prediction status filter')
            clauses.append('p.status=?');parameters.append(status)
        if max_probability is not None:
            if not 0 <= _number(max_probability,'probability threshold') <= 1:
                raise ValueError('Probability threshold must be in [0,1]')
            clauses.append('(p.probability IS NULL OR p.probability<=?)');parameters.append(max_probability)
        if bbox is not None:
            if isinstance(bbox,dict):
                bbox = [bbox[key] for key in ('xmin','ymin','xmax','ymax')]
            if len(bbox) != 4:
                raise ValueError('Viewport requires xmin,ymin,xmax,ymax')
            x0,y0,x1,y1 = [_number(v,'viewport bound') for v in bbox]
            if x0 > x1 or y0 > y1:
                raise ValueError('Viewport bounds are reversed')
            if metadata['spatial_index'] == 'rtree':
                clauses.append('c.row_id IN (SELECT row_id FROM bbox_rtree WHERE xmax>=? AND xmin<=? AND ymax>=? AND ymin<=?)')
                parameters.extend([x0,x1,y0,y1])
            clauses.append('c.xmax>=? AND c.xmin<=? AND c.ymax>=? AND c.ymin<=?')
            parameters.extend([x0,x1,y0,y1])
        if mode == 'cell':
            if not isinstance(cell_id,str) or not cell_id:
                raise ValueError('Canonical cell ID required')
            clauses.append('c.cell_id=?');parameters.append(cell_id)
        if mode == 'review_subset':
            if not isinstance(cell_ids,list) or not cell_ids or len(cell_ids)>MAX_PAGE or len(set(cell_ids))!=len(cell_ids):
                raise ValueError('Explicit unique bounded reviewed cell-ID subset required')
            clauses.append('c.cell_id IN ('+','.join('?' for _ in cell_ids)+')');parameters.extend(cell_ids)
        if mode not in ('summary','viewport','cell','queue','review_subset'):
            raise ValueError('Unknown cell-index query mode')
        where = (' WHERE '+' AND '.join(clauses)) if clauses else ''
        source = ' FROM cells c'+join+where
        total = connection.execute('SELECT COUNT(*)'+source,parameters).fetchone()[0]
        if mode == 'summary':
            # Aggregates describe the full requested population, never a sampled page.
            summary = [dict(row) for row in connection.execute('SELECT p.model_id,p.status,p.class_id,p.label,COUNT(*) AS count FROM predictions p JOIN cells c ON c.row_id=p.row_id'+where+' GROUP BY p.model_id,p.status,p.class_id,p.label ORDER BY p.model_id,p.class_id',parameters)]
            return dict(base,cells=[],summary=summary,total=total,truncated=False,next_cursor=None)
        page_clause = (' AND ' if where else ' WHERE ')+'c.row_id>?'
        rows = connection.execute('SELECT c.row_id,c.raw_json'+source+page_clause+' ORDER BY c.row_id LIMIT ?', [*parameters,after_row_id,limit+1]).fetchall()
        more = len(rows)>limit;rows=rows[:limit]
        cells = []
        for row in rows:
            cell = _loads(row['raw_json'])
            if metadata.get('sidecars'):
                qc = connection.execute('SELECT qc_json FROM segmentation_review WHERE row_id=?', (row['row_id'],)).fetchone()
                if qc is None:
                    raise ValueError('Indexed cell has no checked segmentation review row')
                review = _loads(qc['qc_json'])
                cell.update({key: value for key, value in review.items() if key.startswith('segmentation_') or key == 'stardist_label'})
                cell['segmentation_review_status'] = review.get('segmentation_review_status', 'available')
                cell['segmentation_review_source_sha256'] = metadata['sidecars'][0]['sha256']
            if not include_geometry:
                cell.pop('contour',None);cell.pop('bbox',None)
            cells.append(cell)
        result = dict(base,cells=cells,total=total,truncated=more,next_cursor=rows[-1]['row_id'] if more and rows else None)
        if mode == 'review_subset':
            # This mode is a server-to-server training API, never a browser payload.
            if len(rows) != len(cell_ids):
                raise ValueError('Reviewed feature subset includes unknown cell IDs or exceeds the requested page')
            feature_rows = connection.execute(
                'SELECT c.cell_id,c.source_id,c.x,c.y,f.values_json FROM features f JOIN cells c USING(row_id) WHERE c.cell_id IN ('+','.join('?' for _ in cell_ids)+')',cell_ids).fetchall()
            if metadata.get('feature_storage') == 'partition-row-reference-v1':
                result['features'] = _partition_review_features(index_path, metadata, feature_rows)
            else:
                result['features'] = {row['cell_id']:_loads(row['values_json']) for row in feature_rows}
            result['feature_columns'] = metadata['feature_set']['feature_columns']
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build')
    for name in ('classification','feature-manifest','output'):
        build.add_argument('--'+name,required=True,type=Path)
    build.add_argument('--segmentation-review',type=Path)
    verify=sub.add_parser('verify')
    for name in ('index','classification','feature-manifest'):
        verify.add_argument('--'+name,required=True,type=Path)
    verify.add_argument('--segmentation-review',type=Path)
    query=sub.add_parser('query')
    query.add_argument('--index',required=True,type=Path)
    query.add_argument('--mode',choices=('metadata','summary','viewport','cell','queue','review_subset'),default='metadata')
    query.add_argument('--bbox',nargs=4,type=float)
    query.add_argument('--cell-id');query.add_argument('--model-id');query.add_argument('--class-id',type=int)
    query.add_argument('--status');query.add_argument('--max-probability',type=float)
    query.add_argument('--limit',type=int,default=500);query.add_argument('--after-row-id',type=int,default=0)
    query.add_argument('--cell-ids',nargs='+');query.add_argument('--omit-geometry',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='build':
            result=build_index(args.classification,args.feature_manifest,args.output,segmentation_review_path=args.segmentation_review)
        elif args.command=='verify':
            result=validate_index(args.index,args.classification,args.feature_manifest,segmentation_review_path=args.segmentation_review)
        else:
            result=query_index(args.index,args.mode,bbox=args.bbox,cell_id=args.cell_id,model_id=args.model_id,
                class_id=args.class_id,status=args.status,max_probability=args.max_probability,limit=args.limit,
                after_row_id=args.after_row_id,cell_ids=args.cell_ids,include_geometry=not args.omit_geometry)
        print(json.dumps(result,allow_nan=False));return 0
    except (ValueError,OSError,sqlite3.DatabaseError) as exc:
        print('Cannot use cell index: '+str(exc),file=__import__('sys').stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
