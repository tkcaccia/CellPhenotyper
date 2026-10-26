"""Synthetic multi-partition project bridge; no model or GPU execution."""
import copy
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from cellvit_shards import (ARCHIVE_NAME, CLASSIFICATION_HEADER, SHARD_FILES,
                            atomic_json, compact_shard,
                            complete_shard, finalize_shards, plan_shards)
from cellvit_models import digest_file
from export_cellvit_partition_experiment import export_partition_experiment
from export_experiment import (COORDINATE_CONVENTION, image_shape, sha256,
                               validate_experiment)
from registered_tissue_assets import validate_registered_tissue
from index_cell_classification import query_index
from index_cellvit_partitions import (ALL_SAM_TAXONOMIES, _offset_cell,
    _validate_requested_models, build_partition_index, validate_partition_index)
from cellvit_models import image_frame


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def classification_fixture(root, tissue, *, empty=False, width=3):
    """Dynamic-width analytic HIPT-layout fixture; no encoder or classifier was run."""
    folder = root / 'classifier'
    folder.mkdir(parents=True)
    image_width,image_height=image_shape(Path(tissue['image']['path']))
    image = dict(sha256=sha256(Path(tissue['image']['path'])), width=image_width, height=image_height,
                 mpp_x=.5, mpp_y=.5, frame_id='producer-native-image',
                 coordinate_convention=COORDINATE_CONVENTION, coordinate_space='crop-native-pixels', origin_x=0, origin_y=0)
    cells = [] if empty else [dict(cell_id=f'cellvit:synthetic:{i}', source_id=str(i), source='cellvitpp',
                                  x=x, y=y, contour=[[x-1,y-1],[x+1,y-1],[x,y+1]])
                             for i, (x,y) in enumerate([(16.,20.),(40.,32.)])]
    table = root / 'tokens' / 'features.csv.gz'
    table.parent.mkdir(parents=True)
    columns = [f'feat_{i+1}' for i in range(width)]
    with gzip.open(table, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cell_id','source_id','x','y',*columns])
        for cell in cells:
            writer.writerow([cell['cell_id'],cell['source_id'],cell['x'],cell['y'],*[(i+1)/10 for i in range(width)]])
    feature = dict(kind='cellvit-cell', observation_unit='cell', coordinate_frame='image', frame_id=image['frame_id'],
                   feature_set_id='synthetic-cell-tokens', path='../tokens/features.csv.gz', sha256=sha256(table),
                   format='csv.gz', id_column='cell_id',x_column='x',y_column='y',feature_columns=columns,row_count=len(cells),
                   descriptor=dict(backbone='HIPT',checkpoint_sha256='a'*64,runtime_version='synthetic-fixture-not-model-runtime',
                       feature_definition='synthetic_analytic_cell_features',feature_count=width,coordinate_frame='image',observation_unit='cell',
                       sampling=dict(model_input_mpp=.25,patch_size_px=1024,patch_overlap_px=64,token_size_px=16),
                       preprocessing=dict(runtime_wheel_sha256='f'*64,normalization='synthetic analytic fixture; no model run',
                                          resampling='synthetic analytic fixture; no resampling',token_provenance='synthetic analytic values; not encoder inference')))
    manifest = dict(schema='cellphenotyper-cell-features',schema_version=1,image=image,feature_set=feature)
    cell_metadata = folder/'cellvit_feature_cells.json'
    write_json(cell_metadata,dict(image=image,cells=cells))
    manifest['cell_metadata']=dict(path=cell_metadata.name,sha256=sha256(cell_metadata))
    manifest_path=folder/'cellvit_cell_features.manifest.json'
    write_json(manifest_path,manifest)
    model=dict(model_id='synthetic-model',backbone='HIPT',taxonomy='synthetic-two-class',
               status='available',kind='mlp',checkpoint_sha256='b'*64,class_names=['Other','Tumor'],
               score_semantics='uncalibrated_model_probability')
    predicted=copy.deepcopy(cells)
    for cell in predicted:
        cell['predictions']={model['model_id']:dict(status='predicted',class_id=1,label='Tumor',probability=.75,probabilities=[.25,.75])}
    classification=dict(schema='cellphenotyper-cell-classification',schema_version=1,image=image,feature_set=feature,
                        models=[model],cells=predicted,provenance=dict(synthetic=True,fitted=False,feature_manifest_sha256=sha256(manifest_path)))
    path=folder/'cellvit_cell_classification.json'
    write_json(path,classification)
    return path,manifest_path



def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def _partition(root, image, plan, shard, *, duplicate=False, empty=False):
    folder = root / 'shards' / shard['id']
    folder.mkdir(parents=True)
    classification, manifest = classification_fixture(folder, {'image': {'path': str(image)}}, empty=empty)
    header, feature_manifest = json.loads(classification.read_text()), json.loads(manifest.read_text())
    first = shard['core'][0] == 0
    cells = header['cells'] if first else header['cells'][:1]
    if not first:
        cells[0]['cell_id'] = 'cellvit:synthetic:0' if duplicate else 'cellvit:synthetic:right'
        cells[0]['x'] = shard['core'][0] + 0.5
        cells[0]['y'] = 20.0
        cells[0]['contour'] = [[cells[0]['x']-1,19],[cells[0]['x']+1,19],[cells[0]['x'],21]]
    header['cells'] = cells
    frame = plan['image']
    header['image'] = frame
    feature_manifest['image'] = frame
    for value in (header['feature_set'], feature_manifest['feature_set']):
        value['frame_id'] = frame['frame_id']
        value['path'] = 'cellvit_cell_features.csv.gz'
        value['row_count'] = len(cells)
    features = folder / 'cellvit_cell_features.csv.gz'
    with gzip.open(features, 'wt', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cell_id','source_id','x','y',*header['feature_set']['feature_columns']])
        for cell in cells:
            writer.writerow([cell['cell_id'],cell['source_id'],cell['x'],cell['y'],.1,.2,.3])
    for value in (header['feature_set'], feature_manifest['feature_set']):
        value['sha256'] = digest_file(features)
    metadata_cells = [{key:value for key,value in cell.items() if key != 'predictions'} for cell in cells]
    metadata = folder / 'cellvit_feature_cells.json'
    _write(metadata, {'image': frame, 'cells': metadata_cells})
    feature_manifest['cell_metadata'] = {'path': metadata.name, 'sha256': digest_file(metadata)}
    _write(folder / 'cellvit_cell_features.manifest.json', feature_manifest)
    header['provenance']['feature_manifest_sha256'] = digest_file(folder / 'cellvit_cell_features.manifest.json')
    _write(folder / 'cellvit_cell_classification.json', header)
    with (folder / 'cellvit_cell_classification.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(CLASSIFICATION_HEADER)
        for cell in cells:
            for model in header['models']:
                prediction = cell['predictions'][model['model_id']]
                writer.writerow([cell['cell_id'], cell['source_id'], cell['x'],
                                 cell['y'], model['model_id'], model['taxonomy'],
                                 model['backbone'], model.get('checkpoint_sha256', ''),
                                 prediction['status'], prediction.get('class_id', ''),
                                 prediction.get('label', ''), prediction.get('probability', ''),
                                 json.dumps(prediction.get('probabilities', [])),
                                 model.get('score_semantics', ''), prediction.get('reason', '')])
    normalized = [{**{key:value for key,value in cell.items() if key != 'predictions'},
                   'id': cell['source_id'], 'centroid': [cell['x'], cell['y']]} for cell in cells]
    _write(folder / 'cellvit_cells.json', {'cells': normalized})
    _write(folder / 'cellvit_metadata.json', {'model': 'HIPT', 'requested_taxonomies': ['synthetic'],
        'model_provenance': {'checkpoints': [{'sha256': 'a'*64}]}})
    for name in SHARD_FILES:
        path = folder / name
        if not path.exists():
            path.write_text('{}')
    complete_shard(plan, shard, folder)


def _source(tmp_path, *, duplicate=False, compact=False):
    root = tmp_path / 'bundle'
    root.mkdir()
    image = root / 'slide.png'
    Image.new('RGB', (1025, 48), (180,145,175)).save(image)
    shift = root / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    plan = plan_shards(image, shift, 1024, 1024)
    assert len(plan['shards']) == 2
    atomic_json(root / 'cellvit_shard_plan.json', plan)
    for shard in plan['shards']:
        _partition(root, image, plan, shard, duplicate=duplicate)
    if compact:
        compact_shard(plan, plan['shards'][0], root / 'shards' / plan['shards'][0]['id'])
    manifest = root / 'cellvit_partition_manifest.json'
    finalize_shards(plan, root)
    return root, image, manifest


def test_streamed_two_shards_and_portable_project(tmp_path):
    root, image, manifest = _source(tmp_path)
    index = root / 'cells.sqlite'
    metadata = build_partition_index(manifest, image, index)
    assert metadata['cell_count'] == 3
    assert metadata['prediction_count'] == 3
    pointer = index.with_suffix('.features.json')
    assert validate_partition_index(index, manifest, pointer, image) == metadata
    outlined = query_index(index, mode='viewport', bbox=[0, 0, 1025, 48],
                           include_geometry=True)
    assert outlined['total'] == 3
    assert all(len(cell['contour']) >= 3 for cell in outlined['cells'])
    centroids = query_index(index, mode='viewport', bbox=[0, 0, 1025, 48],
                            include_geometry=False)
    assert all('contour' not in cell for cell in centroids['cells'])
    output = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(manifest, image, index, output,
        name='Synthetic two-shard image', tissue_id='sample-a')
    assert validate_experiment(output) == project
    saved = json.loads(output.read_text())
    assert saved['tissues'][0]['cell_classification']['cell_count'] == 3
    assert saved['tissues'][0]['cell_classification']['index']['format'] == 'sqlite'
    import sqlite3
    with sqlite3.connect(index) as connection:
        rows = connection.execute('SELECT cell_id,source_id,x FROM cells ORDER BY x').fetchall()
    assert len(rows) == 3 and rows[-1] == ('cellvit:synthetic:right','r0000c0001:0',1024.5)
    assert project['tissues'][0]['feature_sets'][0]['format'] == 'partitioned-sqlite'
    moved = tmp_path / 'moved'
    shutil.copytree(root, moved)
    assert validate_experiment(moved / output.name)['experiment_id'] == project['experiment_id']


def test_empty_core_and_adjacent_nonempty_core_form_one_full_image_project(tmp_path):
    root, image, manifest = _source(tmp_path)
    manifest.unlink()
    plan = json.loads((root / 'cellvit_shard_plan.json').read_text())
    first = plan['shards'][0]
    shutil.rmtree(root / 'shards' / first['id'])
    _partition(root, image, plan, first, empty=True)
    refreshed = finalize_shards(plan, root)
    assert refreshed['partition_count'] == 2
    assert refreshed['cell_count'] == 1
    assert [part['cell_count'] for part in refreshed['partitions']] == [0, 1]
    index = root / 'cells.sqlite'
    result = build_partition_index(manifest, image, index)
    assert result['cell_count'] == 1
    cells = query_index(index, mode='viewport', bbox=[0, 0, 1025, 48],
                        include_geometry=True)['cells']
    assert len(cells) == 1 and cells[0]['x'] == 1024.5
    project_path = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(manifest, image, index, project_path,
                                          name='Empty plus tissue', tissue_id='sample-a')
    assert validate_experiment(project_path) == project


def test_lossless_compacted_partition_still_indexes_and_reopens(tmp_path):
    root, image, manifest = _source(tmp_path, compact=True)
    first = root / 'shards' / 'r0000c0000'
    assert (first / ARCHIVE_NAME).is_file()
    assert not (first / 'cellvit_cell_classification.json').exists()
    index = root / 'cells.sqlite'
    metadata = build_partition_index(manifest, image, index)
    assert metadata['cell_count'] == 3
    pointer = index.with_suffix('.features.json')
    assert validate_partition_index(index, manifest, pointer, image) == metadata
    output = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(manifest, image, index, output,
        name='One compacted shard', tissue_id='sample-a')
    moved = tmp_path / 'compact-moved'
    shutil.copytree(root, moved)
    assert validate_experiment(moved / output.name)['experiment_id'] == project['experiment_id']


def test_duplicate_canonical_cell_across_cores_is_rejected(tmp_path):
    root, image, manifest = _source(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='Duplicate partition cell'):
        build_partition_index(manifest, image, root / 'cells.sqlite')


def test_nonzero_input_origin_translates_centroid_contour_and_bbox():
    shard = {'id': 'r0001c0002', 'input': [960, 960, 3000, 3000],
             'core': [2048, 2048, 3000, 3000]}
    global_image = {'width': 4000, 'height': 4000}
    local = {'cell_id': 'c', 'source_id': '7', 'x': 1088.5, 'y': 1088.5,
             'contour': [[1087,1087],[1090,1087],[1090,1090]],
             'bbox': [[1087,1087],[1090,1090]]}
    shifted = _offset_cell(local, shard, global_image)
    assert (shifted['x'],shifted['y']) == (2048.5,2048.5)
    assert shifted['source_id'] == 'r0001c0002:7'
    assert shifted['contour'][0] == [2047,2047]
    assert shifted['bbox'] == [[2047,2047],[2050,2050]]
    assert local['x'] == 1088.5


def test_all_eight_sam_taxonomies_are_required_without_silent_partial_export():
    plan = {'run_contract': {'model': 'SAM', 'requested_taxonomies': ['all']}}
    models = [{'taxonomy': taxonomy, 'status': 'available'} for taxonomy in sorted(ALL_SAM_TAXONOMIES)]
    _validate_requested_models(plan, models)
    plan['run_contract']['requested_taxonomies'] = sorted(ALL_SAM_TAXONOMIES)
    _validate_requested_models(plan, models)
    plan['run_contract']['requested_taxonomies'] = ['pannuke']
    with pytest.raises(ValueError, match='all eight'):
        _validate_requested_models(plan, models)
    plan['run_contract']['requested_taxonomies'] = ['all']
    with pytest.raises(ValueError, match='all eight'):
        _validate_requested_models(plan, models[:-1])
    models[0]['status'] = 'unavailable'
    with pytest.raises(ValueError, match='all eight'):
        _validate_requested_models(plan, models)


def test_registered_half_mpp_kodama_assets_keep_exact_full_cell_base(tmp_path):
    root, image, manifest = _source(tmp_path)
    index = root / 'cells.sqlite'
    build_partition_index(manifest, image, index)
    tissue_image = root / 'tissue.png'
    Image.new('RGB', (513,24), (180,145,175)).save(tissue_image)
    tissue_shift = root / 'tissue_shift.json'
    tissue_shift.write_text(json.dumps({'mpp_x': 1025*.25/513, 'mpp_y': .5}))
    tissue_frame = image_frame(tissue_image, tissue_shift)
    full = json.loads((root/'cellvit_shard_plan.json').read_text())['image']
    sx, sy = full['width']/tissue_frame['width'], full['height']/tissue_frame['height']
    registration = root / 'slide.tissue_registration.json'
    _write(registration, {'format': 'cellphenotyper-dual-mpp-registration-v1',
        'verified': True, 'coordinate_convention': full['coordinate_convention'],
        'source_scene_sha256': 'a'*64, 'image': tissue_frame,
        'cell_full_image': full, 'cell_project_image': full,
        'cell_crop_bbox_in_cell_full_xyxy': [0,0,full['width'],full['height']],
        'tissue_to_cell_full_affine': [[sx,0,0],[0,sy,0],[0,0,1]],
        'tissue_to_cell_project_affine': [[sx,0,0],[0,sy,0],[0,0,1]],
        'cell_project_to_tissue_affine': [[1/sx,0,0],[0,1/sy,0],[0,0,1]]})
    points = root / 'kodama.csv'
    points.write_text('cell_id,native_x,native_y,kodama_1,kodama_2\n'
                      'tile-a,16,10,-1,0.5\ntile-b,40,20,2,1\n')
    embeddings = root / 'virchow2.csv'
    embeddings.write_text('tile_id,native_x,native_y,feature_1\n'
                          'tile-a,16,10,0.1\ntile-b,40,20,0.9\n')
    bora = root / 'bora.geojson'
    _write(bora, {'type':'FeatureCollection','features':[{'type':'Feature',
        'geometry':{'type':'Polygon','coordinates':[[[5,5],[20,5],[20,18],[5,18],[5,5]]]},
        'properties':{'class':'tumor'}}]})
    assets = root / 'assets.json'
    _write(assets, [{'asset_id':'virchow2-main','role':'virchow2_tiles','path':'virchow2.csv',
        'format':'csv','frame_id':tissue_frame['frame_id'],
        'image_sha256':tissue_frame['sha256'],'id_column':'tile_id',
        'x_column':'native_x','y_column':'native_y','row_count':2},
        {'asset_id':'kodama-main','role':'kodama_map','path':'kodama.csv',
        'format':'csv','frame_id':tissue_frame['frame_id'],
        'image_sha256':tissue_frame['sha256'],'id_column':'cell_id',
        'x_column':'native_x','y_column':'native_y',
        'dimension_columns':['kodama_1','kodama_2'],'row_count':2},
        {'asset_id':'bora-main','role':'bora_geometry','path':'bora.geojson',
        'format':'geojson','frame_id':tissue_frame['frame_id'],
        'image_sha256':tissue_frame['sha256']}])
    output = root / 'experiment.cellphenotyper.json'
    project = export_partition_experiment(manifest, image, index, output,
        name='Registered tissue', tissue_id='sample-a',
        tissue_registration=registration, tissue_image=tissue_image, tissue_assets=assets)
    assert validate_experiment(output) == project
    linked = project['tissues'][0]['registered_tissue']
    assert linked['image']['frame_id'] == tissue_frame['frame_id']
    assert project['tissues'][0]['image']['frame_id'] == full['frame_id']
    assert [asset['role'] for asset in linked['assets']] == [
        'virchow2_tiles', 'kodama_map', 'bora_geometry']
    moved = tmp_path / 'registered-moved'
    shutil.copytree(root, moved)
    assert validate_experiment(moved / output.name)['experiment_id'] == project['experiment_id']
    invalid_bora = json.loads(bora.read_text())
    invalid_bora['features'][0]['geometry']['coordinates'][0][0][0] = 999
    _write(bora, invalid_bora)
    linked['assets'][2]['sha256'] = digest_file(bora)
    with pytest.raises(ValueError, match='leaves its 0.5'):
        validate_registered_tissue(linked, root, full)
    altered = json.loads(registration.read_text())
    altered['tissue_to_cell_full_affine'][0][0] += .01
    _write(registration, altered)
    with pytest.raises(Exception, match='registration|affine|hash'):
        validate_experiment(output)
