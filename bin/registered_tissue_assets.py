"""Checked dual-MPP attachment contract for a full-slide CellViT project.

Registration is pixel-edge and uses the exact converted image dimensions.
0.5-µm tissue assets retain their native frame; no coordinates are relabelled
as 0.25-µm cell pixels in the experiment manifest.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile

from index_cell_classification import digest_file

FORMAT = 'cellphenotyper-dual-mpp-registration-v1'
CONVENTION = 'pixel-edge-origin_centres-in-native-pixels'
ROLES = {'virchow2_tiles': ('csv', 'csv.gz'), 'kodama_map': ('csv', 'csv.gz'),
         'bora_geometry': ('geojson',)}


def _geojson_xy(coordinates):
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError('Bora GeoJSON geometry has empty/non-array coordinates')
    if len(coordinates) == 2 and all(type(value) in (int, float) for value in coordinates):
        x, y = coordinates
        if not all(math.isfinite(value) for value in (x, y)):
            raise ValueError('Bora GeoJSON has nonfinite coordinates')
        yield x, y
        return
    for child in coordinates:
        yield from _geojson_xy(child)


def _asset(reference, root):
    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
        raise ValueError('Registered tissue requires explicit local asset path and SHA-256')
    path = (Path(root) / reference['path']).resolve()
    if not path.is_file() or not re.fullmatch('[a-f0-9]{64}', str(reference.get('sha256', ''))) or digest_file(path) != reference['sha256']:
        raise ValueError(f'Registered tissue asset hash differs: {path}')
    return path


def _shape(path):
    if path.suffix.lower() in ('.tif', '.tiff', '.btf'):
        import tifffile
        with tifffile.TiffFile(path) as source:
            page = source.pages[0]
            return int(page.imagewidth), int(page.imagelength)
    from PIL import Image
    with Image.open(path) as source:
        return source.size


def _frame(frame, path):
    if (not isinstance(frame, dict) or frame.get('sha256') != digest_file(path) or
            (frame.get('width'), frame.get('height')) != _shape(path) or
            frame.get('coordinate_space') != 'crop-native-pixels' or
            frame.get('coordinate_convention') != CONVENTION or
            frame.get('origin_x') != 0 or frame.get('origin_y') != 0 or
            not isinstance(frame.get('frame_id'), str) or not frame['frame_id']):
        raise ValueError('Dual-MPP frame differs from exact image bytes, shape or native coordinates')
    for axis in ('mpp_x', 'mpp_y'):
        value = frame.get(axis)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Dual-MPP frame requires calibrated positive X/Y MPP')
    identity = {key: frame[key] for key in ('sha256','width','height','mpp_x','mpp_y',
        'origin_x','origin_y','coordinate_space','coordinate_convention')}
    expected = 'cellvit-frame-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    if frame['frame_id'] != expected:
        raise ValueError('Dual-MPP frame ID does not bind its exact calibrated image identity')


def _affine(value, expected, label):
    if (not isinstance(value, list) or len(value) != 3 or
            any(not isinstance(row, list) or len(row) != 3 for row in value)):
        raise ValueError(f'{label} requires an exact 3x3 pixel-edge affine')
    for row, target in zip(value, expected):
        for number, wanted in zip(row, target):
            if (isinstance(number, bool) or not isinstance(number, (int, float)) or
                    not math.isfinite(number) or not math.isclose(number, wanted, rel_tol=1e-12, abs_tol=1e-9)):
                raise ValueError(f'{label} differs from image dimensions and crop translation')


def validate_registered_tissue(item, root, cell_frame, *, verify_rows=True):
    """Validate the sidecar and optional assets without reading a whole population."""
    registration_path = _asset(item.get('registration'), root)
    if registration_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Dual-MPP registration sidecar exceeds bounded metadata size')
    sidecar = json.loads(registration_path.read_text())
    if (sidecar.get('format') != FORMAT or sidecar.get('verified') is not True or
            sidecar.get('coordinate_convention') != CONVENTION or
            not re.fullmatch('[a-f0-9]{64}', str(sidecar.get('source_scene_sha256', '')))):
        raise ValueError('Verified dual-MPP source-scene registration is required')
    image_path = _asset(item.get('image'), root)
    _frame(sidecar.get('image'), image_path)
    if sidecar['cell_full_image'] != {key: cell_frame[key] for key in sidecar['cell_full_image']}:
        raise ValueError('Dual-MPP full cell frame differs from the CellViT project image')
    tissue = sidecar['image']
    if item['image'].get('frame_id') != tissue['frame_id'] or item['image'].get('mpp_x') != tissue['mpp_x'] or item['image'].get('mpp_y') != tissue['mpp_y']:
        raise ValueError('Registered tissue image declaration differs from verified sidecar')
    cw, ch, tw, th = (sidecar['cell_full_image']['width'], sidecar['cell_full_image']['height'],
                      tissue['width'], tissue['height'])
    if any(not math.isclose(c * cm, t * tm, rel_tol=1e-6, abs_tol=1e-6) for c, cm, t, tm in (
            (cw, cell_frame['mpp_x'], tw, tissue['mpp_x']),
            (ch, cell_frame['mpp_y'], th, tissue['mpp_y']))):
        raise ValueError('Dual-MPP converted images do not share the same physical field of view')
    crop = sidecar.get('cell_crop_bbox_in_cell_full_xyxy')
    if (not isinstance(crop, list) or len(crop) != 4 or
            any(type(value) is not int for value in crop) or
            not (0 <= crop[0] < crop[2] <= cw and 0 <= crop[1] < crop[3] <= ch)):
        raise ValueError('Dual-MPP prepared-cell crop bounds are invalid')
    project = sidecar.get('cell_project_image')
    if not isinstance(project, dict) or (project.get('width'), project.get('height')) != (
            crop[2]-crop[0], crop[3]-crop[1]):
        raise ValueError('Dual-MPP prepared-cell crop frame differs from its bounds')
    sx, sy = cw / tw, ch / th
    _affine(sidecar.get('tissue_to_cell_full_affine'), [[sx,0,0],[0,sy,0],[0,0,1]], 'tissue_to_cell_full_affine')
    _affine(sidecar.get('tissue_to_cell_project_affine'),
            [[sx,0,-crop[0]],[0,sy,-crop[1]],[0,0,1]], 'tissue_to_cell_project_affine')
    _affine(sidecar.get('cell_project_to_tissue_affine'),
            [[1/sx,0,crop[0]/sx],[0,1/sy,crop[1]/sy],[0,0,1]], 'cell_project_to_tissue_affine')
    assets = item.get('assets', [])
    if not isinstance(assets, list) or len(assets) > 32:
        raise ValueError('Registered tissue assets require a bounded explicit list')
    ids = set()
    for asset in assets:
        if not isinstance(asset, dict) or asset.get('role') not in ROLES or asset.get('format') not in ROLES[asset['role']]:
            raise ValueError('Unknown registered tissue asset role/format')
        aid = asset.get('asset_id')
        if not isinstance(aid, str) or not aid or aid in ids:
            raise ValueError('Registered tissue asset IDs must be unique')
        ids.add(aid)
        if asset.get('frame_id') != tissue['frame_id'] or asset.get('image_sha256') != tissue['sha256']:
            raise ValueError('Registered tissue asset frame/image differs from 0.5-µm source')
        path = _asset(asset, root)
        if asset['role'] in ('virchow2_tiles','kodama_map'):
            keys = [asset.get(key) for key in ('id_column','x_column','y_column')]
            if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != 3:
                raise ValueError('Registered tissue CSV requires exact ID/native XY columns')
            if asset['role'] == 'kodama_map':
                dims = asset.get('dimension_columns')
                if not isinstance(dims, list) or len(dims) != 2 or any(not isinstance(d, str) or not d for d in dims) or len(set(keys+dims)) != 5:
                    raise ValueError('KODAMA map requires two distinct reduction columns')
            if verify_rows:
                opener = gzip.open if asset['format'] == 'csv.gz' else open
                count = 0
                with opener(path, 'rt', newline='', encoding='utf-8-sig') as stream:
                    reader = csv.DictReader(stream)
                    columns = reader.fieldnames or []
                    required = [*keys, *(asset.get('dimension_columns') or [])]
                    if len(columns) != len(set(columns)) or not set(required).issubset(columns):
                        raise ValueError('Registered tissue CSV is missing declared columns')
                    with tempfile.TemporaryDirectory(prefix='tissue-ids-') as temporary:
                        db = sqlite3.connect(Path(temporary)/'ids.sqlite')
                        db.execute('CREATE TABLE ids(id TEXT PRIMARY KEY)')
                        for row in reader:
                            if None in row or any(row.get(k) is None for k in required):
                                raise ValueError('Malformed registered tissue CSV row')
                            values = [float(row[k]) for k in keys[1:]]
                            if not all(math.isfinite(v) for v in values) or not (0 <= values[0] < tw and 0 <= values[1] < th):
                                raise ValueError('Registered tissue CSV coordinate lies outside its 0.5-µm frame')
                            if asset['role'] == 'kodama_map' and not all(math.isfinite(float(row[k])) for k in asset['dimension_columns']):
                                raise ValueError('KODAMA reduction has nonfinite coordinates')
                            try:
                                db.execute('INSERT INTO ids VALUES(?)', (row[keys[0]],))
                            except sqlite3.IntegrityError as exc:
                                raise ValueError('Registered tissue CSV contains duplicate tile IDs') from exc
                            count += 1
                        db.close()
                if type(asset.get('row_count')) is not int or asset['row_count'] != count:
                    raise ValueError('Registered tissue CSV row count differs from its declaration')
        elif verify_rows:
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError('Bora GeoJSON requires a bounded or indexed full-slide derivative')
            geo = json.loads(path.read_text())
            if geo.get('type') != 'FeatureCollection' or not isinstance(geo.get('features'), list):
                raise ValueError('Bora geometry must be a GeoJSON FeatureCollection')
            for key, wanted in (('source_frame_id', tissue['frame_id']),
                                ('image_sha256', tissue['sha256']),
                                ('native_shape_yx', [th, tw]),
                                ('coordinate_space', 'native_image_pixel_edges'),
                                ('origin', 'top_left')):
                if key in geo and geo[key] != wanted:
                    raise ValueError(f'Bora GeoJSON {key} differs from registered tissue frame')
            for feature in geo['features']:
                geometry = feature.get('geometry') if isinstance(feature, dict) else None
                if (not isinstance(feature, dict) or feature.get('type') != 'Feature' or not isinstance(geometry, dict) or
                        geometry.get('type') not in ('Polygon', 'MultiPolygon')):
                    raise ValueError('Bora GeoJSON requires polygon Features')
                points = 0
                for x, y in _geojson_xy(geometry.get('coordinates')):
                    if not (0 <= x <= tw and 0 <= y <= th):
                        raise ValueError('Bora GeoJSON geometry leaves its 0.5-µm frame')
                    points += 1
                if points < 4:
                    raise ValueError('Bora GeoJSON polygon has fewer than four points')
    return sidecar
