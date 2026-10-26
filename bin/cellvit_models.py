#!/usr/bin/env python3
"""Offline, provenance-checked CellViT++ graph export and pretrained token heads.

No import-time torch, CUDA probing, network access, fitting, or taxonomy merging.
The official 1.0.9 graph and its cells.json must be supplied together.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

CATALOG_PATH = Path(__file__).resolve().parents[1] / 'resources' / 'cellvit_model_catalog.json'
FEATURE_DEFINITION = 'mean_nucleus_bounding_box_patch_tokens'
RUNTIME_VERSION = '1.0.9'
RUNTIME_WHEEL_SHA256 = '0dc07a8cb5dd4318882f5200d60170d3326de6de8514cae6f6dafe0a8230b699'
MAX_BOUNDED_CELLS = 40_000
MAX_NATIVE_GRAPH_BYTES = 256 * 1024 * 1024
MAX_NATIVE_CELLS_JSON_BYTES = 96 * 1024 * 1024
CLASSIFICATION_HEADER = [
    'cell_id', 'source_id', 'x', 'y', 'model_id', 'taxonomy', 'backbone',
    'checkpoint_sha256', 'status', 'class_id', 'label', 'probability',
    'probabilities', 'score_semantics', 'reason',
]


def require_bounded_native_outputs(cells_json, graph_path):
    """Fail before torch/JSON materialize an unsafe upstream WSI export.

    The legacy exporter keeps Python feature rows, geometry and head results
    simultaneously. These are conservative per-partition limits, not a claim
    that an entire slide is supported in RAM.
    """
    for path, limit in ((cells_json, MAX_NATIVE_CELLS_JSON_BYTES),
                        (graph_path, MAX_NATIVE_GRAPH_BYTES)):
        size = Path(path).stat().st_size
        if size > limit:
            raise ValueError(f'CellViT native artifact exceeds bounded partition limit: '
                             f'{path} ({size} > {limit} bytes); use smaller shard cores')


def digest_file(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def checkpoint_identity(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    return {'checkpoint_sha256': sha.hexdigest(), 'checkpoint_md5': md5.hexdigest()}


def segmentation_descriptor(checkpoint, backbone, runtime_version=RUNTIME_VERSION, catalog=None):
    catalog = catalog or load_catalog()
    if backbone not in catalog['backbones']:
        raise ValueError('Unsupported backbone')
    descriptor = dict(checkpoint_identity(checkpoint), backbone=backbone,
                      runtime_version=runtime_version, feature_definition=FEATURE_DEFINITION,
                      feature_count=catalog['backbones'][backbone]['feature_count'], observation_unit='cell')
    validate_descriptor(descriptor, catalog, require_sampling=False)
    return descriptor


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_catalog(path=None):
    data = json.loads(Path(path or CATALOG_PATH).read_text())
    if data.get('schema') != 'cellvit-model-catalog' or data.get('schema_version') != 1:
        raise ValueError('Unsupported CellViT model catalog')
    return data


def requested_models(backbone, taxonomies, catalog=None):
    catalog = catalog or load_catalog()
    if backbone not in catalog['backbones']:
        raise ValueError(f'Unsupported CellViT backbone: {backbone}; no implicit substitution')
    names = list(dict.fromkeys(taxonomies))
    available = [m for m in catalog['models'] if m['backbone'] == backbone]
    if names == ['all']:
        return available
    known = {m['taxonomy']: m for m in available}
    unknown = set(names) - set(known)
    if unknown:
        raise ValueError(f'Unknown taxonomies for {backbone}: {sorted(unknown)}')
    return [known[name] for name in names]


def preflight_models(backbone, taxonomies, cache_dir, catalog=None):
    """Return explicit availability; do not fetch or replace a checkpoint."""
    models = []
    for original in requested_models(backbone, taxonomies, catalog):
        item = dict(original)
        if item['status'] == 'available' and item['kind'] == 'mlp':
            path = Path(cache_dir) / item['relative_checkpoint']
            if not path.is_file():
                item.update(status='unavailable', reason=f'Checkpoint not cached: {path}')
            elif digest_file(path) != item['checkpoint_sha256']:
                item.update(status='unavailable', reason=f'Checkpoint SHA-256 mismatch: {path}')
            else:
                item['checkpoint_path'] = str(path.resolve())
        models.append(item)
    return models


def _torch():
    import torch
    return torch


# A data-only allowlist for the official graph dataclass. No arbitrary pickle globals.
class CellGraphDataWSI:
    pass

CellGraphDataWSI.__module__ = 'cellvit.data.dataclass.cell_graph'


def read_graph(path):
    torch = _torch()
    # torch 2.4 exposes add/get/clear; its scoped safe_globals helper is newer.
    previous = torch.serialization.get_safe_globals()
    torch.serialization.clear_safe_globals()
    torch.serialization.add_safe_globals([CellGraphDataWSI])
    try:
        graph = torch.load(path, map_location='cpu', weights_only=True)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    if isinstance(graph, dict):
        x, positions = graph.get('x'), graph.get('positions')
    elif isinstance(graph, CellGraphDataWSI):
        x, positions = graph.x, graph.positions
    else:
        raise ValueError('Expected the CellViT 1.0.9 graph dataclass or a tensor-only graph dict')
    if not isinstance(x, torch.Tensor) or not isinstance(positions, torch.Tensor):
        raise ValueError('Cell graph requires tensor x and positions')
    if x.ndim != 2 or positions.shape != (x.shape[0], 2):
        raise ValueError('Invalid cell graph feature/position shape')
    if not torch.isfinite(x).all() or not torch.isfinite(positions).all():
        raise ValueError('Cell graph contains non-finite features or coordinates')
    return x.to(dtype=torch.float32), positions.to(dtype=torch.float64)


def validate_descriptor(descriptor, catalog=None, require_sampling=True):
    catalog = catalog or load_catalog()
    backbone = descriptor.get('backbone')
    if backbone not in catalog['backbones']:
        raise ValueError('Feature backbone is unsupported; dimension alone is not a model signature')
    expected = catalog['backbones'][backbone]
    if descriptor.get('runtime_version') != RUNTIME_VERSION:
        raise ValueError('Feature extraction requires the verified CellViT runtime 1.0.9')
    if descriptor.get('feature_definition') != FEATURE_DEFINITION:
        raise ValueError('Wrong CellViT feature definition; pooled tissue embeddings are incompatible')
    if descriptor.get('feature_count') != expected['feature_count']:
        raise ValueError('CellViT feature dimension does not match its declared backbone')
    if descriptor.get('checkpoint_md5') != expected['checkpoint_md5']:
        raise ValueError('Feature segmentation checkpoint does not match the official backbone signature')
    if not re.fullmatch('[0-9a-f]{64}', descriptor.get('checkpoint_sha256', '')):
        raise ValueError('Feature descriptor requires a segmentation checkpoint SHA-256')
    known_sha = expected.get('checkpoint_sha256')
    if known_sha and descriptor['checkpoint_sha256'] != known_sha:
        raise ValueError('Feature segmentation checkpoint SHA-256 does not match the catalog')
    if not require_sampling:
        return backbone
    sampling = descriptor.get('sampling', {})
    if not all(key in sampling for key in ('model_input_mpp', 'patch_size_px', 'patch_overlap_px', 'token_size_px')):
        raise ValueError('Feature descriptor requires actual producer model sampling/patch geometry')
    if not isinstance(sampling['model_input_mpp'], (float, int)) or not math.isfinite(sampling['model_input_mpp']) or sampling['model_input_mpp'] <= 0:
        raise ValueError('Invalid model input MPP')
    if sampling['token_size_px'] != 16 or not isinstance(sampling['patch_size_px'], int) or sampling['patch_size_px'] < 16 or sampling['patch_size_px'] % 16:
        raise ValueError('Invalid CellViT patch/token geometry')
    if not isinstance(sampling['patch_overlap_px'], int) or not 0 <= sampling['patch_overlap_px'] < sampling['patch_size_px']:
        raise ValueError('Invalid CellViT patch overlap')
    if descriptor.get('preprocessing', {}).get('runtime_wheel_sha256') != RUNTIME_WHEEL_SHA256:
        raise ValueError('Missing immutable CellViT extraction/preprocessing signature')
    return backbone


def producer_sampling(payload):
    """Use actual pathopatch output, never equate source MPP with model MPP."""
    metadata = payload.get('wsi_metadata', {}) if isinstance(payload, dict) else {}
    missing = [key for key in ('target_patch_mpp', 'patch_size', 'patch_overlap') if key not in metadata]
    if missing:
        raise ValueError(f'Missing actual producer sampling metadata: {missing}')
    patch, overlap = metadata['patch_size'], metadata['patch_overlap']
    if float(patch) != int(patch) or float(overlap) != int(overlap):
        raise ValueError('Producer patch geometry must be in integral model-input pixels')
    return {'model_input_mpp': float(metadata['target_patch_mpp']), 'patch_size_px': int(patch),
            'patch_overlap_px': int(overlap), 'token_size_px': 16}


def preprocessing_signature(extraction_contract=None):
    signature = {'runtime_wheel_sha256': RUNTIME_WHEEL_SHA256,
            'normalization': 'torchvision.ToTensor RGB uint8-to-[0,1]; Normalize from immutable segmentation checkpoint transformations.normalize, default mean/std=[0.5,0.5,0.5]',
            'resampling': 'CellViT-1.0.9 load_wsi_meta + pathopatch target_patch_mpp with 0.035 MPP tolerance; native coordinates realigned after interpolation/padding',
            'token_provenance': 'CellViT-1.0.9 forward predictions.tokens; postprocessing bbox floor/ceil divided by token size; mean over spatial token box'}
    if extraction_contract is not None:
        if (isinstance(extraction_contract, dict) and
                extraction_contract.get('contract_id') == 'cellphenotyper-native-wsi-lossless-v1' and
                extraction_contract.get('prepared_pixel_policy') == 'lossless-RGB8-white-right-bottom-minimum1024-native-px' and
                extraction_contract.get('runtime_geometry_sources_verified') is True):
            signature.update(execution_adapter=extraction_contract,
                             resampling=signature['resampling'] + '; source RGB8 preserved with LZW level0; white right/bottom native padding; exact same-row source-centroid extent selection')
            return signature
        # A custom sampler is a different representation even when the encoder,
        # token width and output MPP match the native WSI runtime.
        if (not isinstance(extraction_contract, dict) or
                extraction_contract.get('contract_id') != 'cellphenotyper-custom-cpu-448-v1' or
                extraction_contract.get('native_wsi_parity_qualified') is not False or
                not re.fullmatch('[0-9a-f]{64}', extraction_contract.get('adapter_sha256', ''))):
            raise ValueError('Unsupported or ambiguous custom CellViT extraction contract')
        signature.update(execution_adapter=extraction_contract,
                         normalization='Custom CPU adapter: RGB uint8 /255 then exact recorded checkpoint mean/std',
                         resampling='Custom CPU adapter: white pad top/left64 native pixels, Pillow LANCZOS512to1024; native_xy=(model_xy-128)*0.5; native WSI parity NOT qualified')
    return signature


def representation_feature_id(descriptor, columns):
    """Share representations across tissues; source rows/frame retain own hashes.

    A CSV hash identifies an artifact, not its feature definition. Putting that
    hash in feature_set_id prevents the same encoder/sampler from being joined
    across images in wsiTools. Geometry/image IDs must remain outside this key.
    """
    contract = dict(kind='cellvit-cell', observation_unit='cell',
                    descriptor=descriptor, feature_columns=list(columns))
    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'cellvit-' + hashlib.sha256(encoded).hexdigest()[:24]


def image_frame(image, shift):
    image = Path(image)
    if image.suffix.lower() in ('.tif', '.tiff', '.btf'):
        # TIFF/BigTIFF identity needs only level-0 dimensions. Pillow TIFF
        # support varies with its libtiff build; never require a pixel decoder
        # merely to derive the same native-frame serialization.
        import tifffile
        with tifffile.TiffFile(image) as opened:
            if not opened.pages:
                raise ValueError('Cell feature export requires a nonempty TIFF')
            width = int(opened.pages[0].imagewidth)
            height = int(opened.pages[0].imagelength)
        if width < 1 or height < 1:
            raise ValueError('Cell feature export requires positive image dimensions')
    else:
        from PIL import Image
        with Image.open(image) as opened:
            width, height = opened.size
    data = json.loads(Path(shift).read_text()) if shift else {}
    mpp = data.get('source_mpp', data.get('microns_per_pixel'))
    mpp_x, mpp_y = data.get('mpp_x', mpp), data.get('mpp_y', mpp)
    if any(v is None or not math.isfinite(float(v)) or float(v) <= 0 for v in (mpp_x, mpp_y)):
        raise ValueError('Cell feature export requires positive, finite source MPP')
    # Coordinates are local to the supplied image. Global ROI shift remains separate evidence.
    frame = {'sha256': digest_file(image), 'width': width, 'height': height,
             'mpp_x': float(mpp_x), 'mpp_y': float(mpp_y),
             'origin_x': 0, 'origin_y': 0, 'coordinate_space': 'crop-native-pixels',
             'coordinate_convention': 'pixel-edge-origin_centres-in-native-pixels'}
    frame['frame_id'] = 'cellvit-frame-' + hashlib.sha256(json.dumps(frame, sort_keys=True).encode()).hexdigest()[:24]
    return frame


def _source_cells(payload):
    cells = payload.get('cells', payload) if isinstance(payload, dict) else payload
    if isinstance(cells, dict):
        items = cells.items()
    elif isinstance(cells, list):
        items = enumerate(cells)
    else:
        raise ValueError('CellViT cells must be an object or list')
    result, ids = [], set()
    for fallback, item in items:
        if not isinstance(item, dict) or 'centroid' not in item:
            raise ValueError('Every CellViT cell must have a centroid; cannot skip rows')
        source_id = str(item.get('source_id', item.get('id', fallback)))
        if not source_id or source_id in ids:
            raise ValueError('CellViT source IDs must be nonempty and unique')
        ids.add(source_id)
        centroid = item['centroid']
        if len(centroid) != 2 or any(not math.isfinite(float(v)) for v in centroid):
            raise ValueError('Invalid CellViT centroid')
        result.append((source_id, item))
    return result


def export_graph_features(cells_json, graph_path, outdir, image, shift, checkpoint,
                          backbone, runtime_version=RUNTIME_VERSION, catalog=None,
                          extraction_contract=None, source_evidence=None):
    """Export exact graph rows after validating the official graph/cell order contract."""
    catalog = catalog or load_catalog()
    if backbone not in catalog['backbones']:
        raise ValueError('Unsupported backbone')
    frame = image_frame(image, shift)
    descriptor = segmentation_descriptor(checkpoint, backbone, runtime_version, catalog)
    descriptor['coordinate_frame'] = 'image'
    require_bounded_native_outputs(cells_json, graph_path)
    x, positions = read_graph(graph_path)
    if x.shape[0] > MAX_BOUNDED_CELLS:
        raise ValueError('CellViT graph has too many rows for one bounded partition; use smaller shard cores')
    cell_payload = json.loads(Path(cells_json).read_text())
    geometry = cell_payload.get('geometry', {}) if isinstance(cell_payload, dict) else {}
    if geometry.get('status', '').startswith('custom_') and extraction_contract is None:
        raise ValueError('Custom CellViT samples require their distinct extraction contract; cannot label native WSI parity')
    descriptor['sampling'] = producer_sampling(cell_payload)
    descriptor['preprocessing'] = preprocessing_signature(extraction_contract)
    validate_descriptor(descriptor, catalog)
    raw_map = cell_payload.get('type_map', {}) if isinstance(cell_payload, dict) else {}
    expected_foreground = {str(i): name.lower() for i, name in enumerate(
        ['Neoplastic', 'Inflammatory', 'Connective', 'Dead', 'Epithelial'], start=1)}
    # Pinned CellViT 1.0.9 exports foreground 1..5; some exports also name background 0.
    expected_with_background = {'0': 'background', **expected_foreground}
    native_verified = bool(raw_map) and isinstance(raw_map, dict) and (
        {str(k): str(v).lower() for k, v in raw_map.items()} in
        (expected_foreground, expected_with_background))
    if raw_map and not native_verified:
        raise ValueError('Native taxonomy is not PanNuke; export a PanNuke segmentation graph once before evaluating token heads')
    cells = _source_cells(cell_payload)
    if x.shape != (len(cells), descriptor['feature_count']):
        raise ValueError('Graph row count or width differs from the cells/declared backbone')
    rows, metadata = [], []
    namespace = hashlib.sha256((frame['sha256'] + ':' + frame['frame_id']).encode()).hexdigest()[:24]
    for index, (source_id, cell) in enumerate(cells):
        px, py = map(float, cell['centroid'])
        if not (0 <= px < frame['width'] and 0 <= py < frame['height']):
            raise ValueError('Cell centroid is outside the supplied source image')
        # Upstream realigns float32 graph tensors and Python JSON floats separately.
        # Bound their rounding difference without accepting whole-pixel misjoins.
        if any(abs(float(positions[index, j]) - value) > max(1e-4, abs(value) * 2 ** -22)
               for j, value in enumerate((px, py))):
            raise ValueError('Graph/cells coordinate order mismatch; refusing a positional misjoin')
        cell_id = f'cellvit:{namespace}:{quote(source_id, safe="")}'
        rows.append([cell_id, source_id, px, py, *x[index].tolist()])
        record = {'cell_id': cell_id, 'source_id': source_id, 'source': 'cellvitpp', 'x': px, 'y': py}
        for field in ('contour', 'bbox', 'label'):
            if field in cell:
                record[field] = cell[field]
        native_type = cell.get('type_id', cell.get('type'))
        if native_verified and (isinstance(native_type, int) or (isinstance(native_type, str) and native_type.isdigit())):
            record['native_type_id'] = int(native_type)
        if native_verified and 'type_prob' in cell:
            probability = float(cell['type_prob'])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Native PanNuke score must be a finite fraction')
            record['native_type_probability'] = probability
        metadata.append(record)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    columns = [f'feat_{i + 1}' for i in range(descriptor['feature_count'])]
    csv_path = outdir / 'cellvit_cell_features.csv.gz'
    with csv_path.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['cell_id', 'source_id', 'x', 'y', *columns])
                writer.writerows(rows)
    cells_path = outdir / 'cellvit_feature_cells.json'
    write_json(cells_path, {'image': frame, 'cells': metadata})
    feature_set = {'kind': 'cellvit-cell', 'observation_unit': 'cell',
                   'coordinate_frame': 'image', 'frame_id': frame['frame_id'],
                   'feature_set_id': representation_feature_id(descriptor, columns),
                   'path': csv_path.name, 'sha256': digest_file(csv_path), 'format': 'csv.gz',
                   'id_column': 'cell_id', 'x_column': 'x', 'y_column': 'y',
                   'feature_columns': columns, 'row_count': len(rows), 'descriptor': descriptor}
    manifest = {'schema': 'cellphenotyper-cell-features', 'schema_version': 1,
                'image': frame, 'feature_set': feature_set,
                'cell_metadata': {'path': cells_path.name, 'sha256': digest_file(cells_path)},
                'source': {'cells_sha256': digest_file(cells_json), 'graph_sha256': digest_file(graph_path),
                           'graph_alignment': 'CellViT-1.0.9-identical-postprocessing-order-checked-centroid-by-row',
                           'native_taxonomy': 'pannuke' if native_verified else 'unverified'}}
    if source_evidence is not None:
        manifest['source']['extraction_evidence'] = source_evidence
    manifest_path = outdir / 'cellvit_cell_features.manifest.json'
    write_json(manifest_path, manifest)
    return manifest_path


def read_feature_manifest(manifest_path, catalog=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'cellphenotyper-cell-features' or manifest.get('schema_version') != 1:
        raise ValueError('Unsupported cell feature manifest')
    fs = manifest['feature_set']
    if fs.get('row_count', 0) > MAX_BOUNDED_CELLS:
        raise ValueError('CellViT feature partition exceeds bounded-memory classifier limit')
    validate_descriptor(fs['descriptor'], catalog)
    if fs.get('kind') != 'cellvit-cell' or fs.get('observation_unit') != 'cell':
        raise ValueError('Cell classifier only accepts cellvit-cell observations')
    columns = fs['feature_columns']
    if columns != [f'feat_{i + 1}' for i in range(fs['descriptor']['feature_count'])]:
        raise ValueError('Feature columns must be the exact explicit ordered CellViT allowlist')
    if fs.get('coordinate_frame') != 'image' or fs['descriptor'].get('coordinate_frame') != 'image':
        raise ValueError('CellViT coordinates must be in the source image frame')
    if fs.get('frame_id') != manifest['image'].get('frame_id'):
        raise ValueError('Feature coordinate frame mismatch')
    path = manifest_path.parent / fs['path']
    if digest_file(path) != fs['sha256']:
        raise ValueError('Feature CSV SHA-256 mismatch')
    reference = manifest['cell_metadata']
    cells_path = manifest_path.parent / reference['path']
    if digest_file(cells_path) != reference['sha256']:
        raise ValueError('Cell metadata SHA-256 mismatch')
    cell_payload = json.loads(cells_path.read_text())
    if cell_payload['image'] != manifest['image']:
        raise ValueError('Cell metadata image/frame mismatch')
    cells = cell_payload['cells']
    rows, ids = [], set()
    with gzip.open(path, 'rt', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['cell_id', 'source_id', 'x', 'y', *columns]:
            raise ValueError('Unexpected CellViT CSV header/ordered columns')
        for index, row in enumerate(reader):
            if index >= len(cells):
                raise ValueError('Feature CSV has more rows than cell metadata')
            cell = cells[index]
            if row['cell_id'] in ids or row['cell_id'] != cell['cell_id'] or row['source_id'] != cell['source_id']:
                raise ValueError('Feature/cell ID join mismatch or duplicate ID')
            ids.add(row['cell_id'])
            if any(float(row[axis]) != cell[axis] for axis in ('x', 'y')):
                raise ValueError('Feature/cell coordinates differ')
            values = [float(row[column]) for column in columns]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Non-finite CellViT features')
            rows.append(values)
    if len(rows) != fs['row_count'] or len(rows) != len(cells):
        raise ValueError('Feature manifest row count mismatch')
    return manifest, cells, rows


def load_head(model, device='cpu'):
    """Load only the known official tensor/config checkpoint, with safe unpickling."""
    if device != 'cpu' and not re.fullmatch(r'cuda:\d+', device):
        raise ValueError('Device must be cpu or an explicitly indexed cuda:N')
    torch = _torch()
    path = Path(model['checkpoint_path'])
    if digest_file(path) != model['checkpoint_sha256']:
        raise ValueError('Classifier checkpoint SHA-256 mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state, config = checkpoint['model_state_dict'], checkpoint['config']
    if set(state) != {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}:
        raise ValueError('Unexpected classifier architecture/state keys')
    dim, hidden, count = model['feature_count'], model['hidden_dim'], len(model['class_names'])
    shapes = {'fc1.weight': (hidden, dim), 'fc1.bias': (hidden,), 'fc2.weight': (count, hidden), 'fc2.bias': (count,)}
    if any(tuple(state[k].shape) != shape or not torch.isfinite(state[k]).all() for k, shape in shapes.items()):
        raise ValueError('Classifier tensor dimensions or values are invalid')
    if config.get('data.num_classes') != count or [config.get(f'data.label_map.{i}') for i in range(count)] != model['class_names']:
        raise ValueError('Classifier ordered ontology differs from its catalog signature')
    return {key: tensor.to(device=device, dtype=torch.float32) for key, tensor in state.items()}


def run_head(model, rows, device='cpu', batch_size=1024):
    torch = _torch()
    if batch_size <= 0:
        raise ValueError('Classifier batch size must be positive')
    state = load_head(model, device)
    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            x = torch.tensor(rows[start:start + batch_size], dtype=torch.float32, device=device)
            if x.ndim != 2 or x.shape[1] != model['feature_count'] or not torch.isfinite(x).all():
                raise ValueError('Classifier feature width or values are invalid')
            hidden = torch.relu(torch.nn.functional.linear(x, state['fc1.weight'], state['fc1.bias']))
            logits = torch.nn.functional.linear(hidden, state['fc2.weight'], state['fc2.bias'])
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            if any(not all(math.isfinite(v) for v in row) for row in probabilities):
                raise ValueError('Classifier returned non-finite probabilities')
            results.extend(probabilities)
    return results


def classify_feature_manifest(manifest_path, cache_dir, taxonomies, outdir=None, device='cpu', catalog=None):
    catalog = catalog or load_catalog()
    manifest_path = Path(manifest_path).resolve()
    manifest, cells, rows = read_feature_manifest(manifest_path, catalog)
    backbone = manifest['feature_set']['descriptor']['backbone']
    models = preflight_models(backbone, taxonomies, cache_dir, catalog)
    for cell in cells:
        cell['predictions'] = {}
    for model in models:
        if model['kind'] == 'native_segmentation_head':
            model['checkpoint_sha256'] = manifest['feature_set']['descriptor']['checkpoint_sha256']
            model['probability_vector_status'] = 'not_exported_by_upstream'
            predictions = None
        elif model['status'] == 'available':
            predictions = run_head(model, rows, device)
        else:
            predictions = None
        for index, cell in enumerate(cells):
            prediction = {'status': 'unavailable', 'reason': model.get('reason', 'Native PanNuke class/score not present'),
                          'label': None, 'class_id': None, 'probability': None, 'probabilities': None}
            if model['kind'] == 'native_segmentation_head' and 'native_type_id' in cell and 'native_type_probability' in cell:
                class_id = cell['native_type_id']
                if class_id < 0 or class_id >= len(model['class_names']):
                    raise ValueError('Native PanNuke class outside the ordered ontology')
                prediction = {'status': 'predicted', 'class_id': class_id, 'label': model['class_names'][class_id],
                              'probability': cell['native_type_probability'], 'probabilities': None,
                              'probability_vector_status': 'not_exported_by_upstream'}
            elif predictions is not None:
                vector = predictions[index]
                class_id = max(range(len(vector)), key=vector.__getitem__)
                prediction = {'status': 'predicted', 'class_id': class_id, 'label': model['class_names'][class_id],
                              'probability': vector[class_id], 'probabilities': vector}
            cell['predictions'][model['model_id']] = prediction
    outdir = Path(outdir or manifest_path.parent).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    import os
    feature_set = dict(manifest['feature_set'])
    feature_set['path'] = os.path.relpath(manifest_path.parent / feature_set['path'], outdir)
    result = {'schema': 'cellphenotyper-cell-classification', 'schema_version': 1,
              'image': manifest['image'], 'feature_set': feature_set, 'models': models, 'cells': cells,
              'provenance': {'feature_manifest_sha256': digest_file(manifest_path), 'runtime_version': RUNTIME_VERSION,
                             'execution_device': device, 'fitted': False, 'segmentation_repeated': False,
                             'interpretation': 'Separate pretrained ontologies; uncalibrated scores; no biological equivalence or clinical validation implied.'}}
    json_path = outdir / 'cellvit_cell_classification.json'
    write_json(json_path, result)
    csv_path = outdir / 'cellvit_cell_classification.csv.gz'
    temporary = outdir / ('.' + csv_path.name + '.tmp')
    # Gzip's embedded timestamp and filename would otherwise change the file
    # hash across equivalent runs. Publish only after the compressed stream closes.
    with temporary.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(CLASSIFICATION_HEADER)
                for cell in cells:
                    for model in models:
                        p = cell['predictions'][model['model_id']]
                        writer.writerow([cell['cell_id'], cell['source_id'], cell['x'], cell['y'], model['model_id'], model['taxonomy'],
                                         model['backbone'], model.get('checkpoint_sha256', ''), p['status'], p['class_id'], p['label'],
                                         p['probability'], json.dumps(p['probabilities']), model.get('score_semantics', ''), p.get('reason', '')])
        raw.flush()
        import os
        os.fsync(raw.fileno())
    os.replace(temporary, csv_path)
    return json_path
