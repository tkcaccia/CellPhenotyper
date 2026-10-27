from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest
import tifffile

pytest.importorskip('shapely')
pytest.importorskip('rasterio')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'bin'))
import export_cellvit_primary_segmentation as primary
from cellvit_models import image_frame


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True)+'\n')
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(cid, x, y, radius=2):
    return {'cell_id': cid, 'source_id': 'native-'+cid, 'x': x, 'y': y,
            'centroid': [x, y], 'contour': [[x-radius,y-radius],[x+radius,y-radius],
              [x+radius,y+radius],[x-radius,y+radius]], 'type_id': 1, 'type': 'neoplastic',
            'type_prob': .8, 'predictions': {'pannuke': {'label': 'neoplastic', 'probability': .8}}}


@pytest.fixture
def crop_fixture(tmp_path):
    full = tmp_path/'full.tif'
    crop = tmp_path/'crop.tif'
    mask = tmp_path/'mask.tif'
    tifffile.imwrite(full, np.full((40,32,3), 200, dtype=np.uint8), photometric='rgb')
    tifffile.imwrite(crop, np.full((12,12,3), 200, dtype=np.uint8), photometric='rgb')
    support = np.ones((12,12), dtype=np.uint8); support[8:,8:] = 0
    tifffile.imwrite(mask, support)
    shift = write_json(tmp_path/'shift.json', {'input_image': str(full), 'source_mpp': .25,
        'crop_size': {'width':12,'height':12}, 'full_size': {'width':32,'height':40},
        'crop_bbox_xyxy': {'x0':10,'y0':20,'x1':22,'y1':32}})
    return dict(root=tmp_path, full=full, image=crop, mask=mask, shift=shift)


def native_fixture(f, cells=None):
    metadata = {**image_frame(f['image'], f['shift'])}
    metadata['source_image_sha256'] = metadata['sha256']
    metadata['taxonomy'] = 'pannuke'
    cells = cells if cells is not None else [cell('a',4,4),cell('b',4,4),cell('outside-support',9,9)]
    payload = {'model':'CellViT++','taxonomy':'pannuke','type_map':{'1':'neoplastic'},'metadata':metadata,'cells':cells}
    return write_json(f['root']/'cells.json',payload), write_json(f['root']/'metadata.json',metadata)


def fullslide_fixture(f):
    frame = image_frame(f['full'], f['shift'])
    cells = [cell('a',14,24),cell('b',14,24),cell('outside-support',19,29),cell('outside-crop',2,3)]
    # Only indexed x/y and contours are authoritative global coordinates.
    # An unused source centroid field must not override the translated x/y.
    cells[0]['centroid'] = [999,999]
    models = [
      {'model_id':'misleading-pannuke-text-but-consep','taxonomy':'consep','status':'available','class_names':['Other','Inflammatory']},
      {'model_id':'exact-declared-primary','taxonomy':'pannuke','status':'available','class_names':['Background','Neoplastic']} ]
    for c in cells:
        c['predictions'] = {
          models[0]['model_id']:{'status':'predicted','class_id':1,'label':'Inflammatory','probability':.99},
          models[1]['model_id']:{'status':'predicted','class_id':1,'label':'Neoplastic','probability':.8}}
    index = f['root']/'index.sqlite'
    partition = write_json(f['root']/'partition.json',dict(image=frame,partition_count=2,
        partitions=[{'id':'first'},{'id':'second'}],cell_count=len(cells)))
    metadata = dict(schema='cellphenotyper-cell-index',image=frame,
        classification_header={'image':frame,'models':models},classification_sha256=sha(partition),cell_count=len(cells))
    with sqlite3.connect(index) as db:
        db.execute('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)')
        db.execute('CREATE TABLE cells(row_id INTEGER PRIMARY KEY,raw_json TEXT)')
        db.executemany('INSERT INTO metadata VALUES(?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        db.executemany('INSERT INTO cells(raw_json) VALUES(?)',[(json.dumps(c),) for c in cells])
    ref = lambda path: dict(path=path.name,sha256=sha(path))
    project = write_json(f['root']/'experiment.cellphenotyper.json',{'tissues':[
        {'image':{**ref(f['full']), **frame},'cell_classification':{**ref(partition),'index':ref(index)}}]})
    receipt = write_json(f['root']/'complete.json',dict(schema='cellphenotyper-cellvit-fullslide-complete-v2',
        planned_shards=2,complete_shards=2,shard_count=2,every_core_verified=True,cell_count=len(cells),
        source_image_sha256=frame['sha256'],partition_manifest_sha256=sha(partition),
        index_sha256=sha(index),project_sha256=sha(project),requested_heads=sorted(primary.HEADS)))
    pointer = write_json(f['root']/'pointer.json',dict(complete=True,project_path=str(project),index_path=str(index),
        completion_receipt_path=str(receipt),completion_receipt_sha256=sha(receipt)))
    return pointer, receipt, index


def export(f, **kwargs):
    return primary.export_primary(f['image'],f['shift'],f['mask'],f['root']/'out',tile_size=16,**kwargs)


def test_native_cells_overlap_seed_ids_evidence_and_support(crop_fixture):
    f=crop_fixture; cells, metadata = native_fixture(f)
    result = export(f,cellvit_cells=cells,cellvit_metadata=metadata)
    assert result['cell_count']==2 and result['excluded_centroids']==1
    labels=tifffile.imread(f['root']/'out/labels.tif')
    assert set(np.unique(labels))=={0,1,2}, 'Overlapping nuclei must each retain their own raster label'
    with (f['root']/'out/objects.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert [r['cell_id'] for r in rows]==['a','b']
    assert all(r['geometry_source']=='cellvitpp' for r in rows)
    assert all(not any('consensus' in k or 'agreement' in k for k in r) for r in rows)
    features=json.loads((f['root']/'out/cell_instances.geojson').read_text())['features']
    assert features[0]['properties']['predictions']['pannuke']['label']=='neoplastic'
    assert result['classifier_evidence_changed'] is False and result['conversion_only'] is True
    assert result['label_coverage']['missing_label_count']==0
    assert result['primary_type_taxonomy']=='pannuke'
    assert all(r['cellvitpp_type']=='neoplastic' and r['cellvitpp_type_id']=='1' for r in rows)


def test_fullslide_rows_translate_to_shared_crop_and_keep_complete_coverage(crop_fixture):
    f=crop_fixture; pointer,receipt,index=fullslide_fixture(f)
    staged_receipt=f['root']/'cellvit_fullslide_complete.json'
    staged_receipt.write_bytes(receipt.read_bytes())
    result=export(f,project_pointer=pointer,source_metadata=staged_receipt)
    assert result['source']['complete_shards']==result['source']['planned_shards']==2
    assert result['cell_count']==2 and result['excluded_centroids']==2
    with (f['root']/'out/objects.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    assert [(float(r['x']),float(r['y'])) for r in rows]==[(4,4),(4,4)]
    assert result['crop_to_source_offset_xy']==[10,20]
    assert result['frame_id']==image_frame(f['image'],f['shift'])['frame_id']
    assert result['source_frame_id']!=result['frame_id'], 'Native crop and original full-slide frames remain distinct'
    geo=json.loads((f['root']/'out/cell_instances.geojson').read_text())
    assert geo['source_frame_id']==result['frame_id']
    assert geo['source_image_sha256']==sha(f['image'])
    assert sha(index)==result['source']['index_sha256']
    assert result['primary_type_taxonomy']=='pannuke'
    assert result['primary_type_model_id']=='exact-declared-primary'
    assert all(r['cellvitpp_type']=='Neoplastic' and r['cellvitpp_type_id']=='1' and float(r['cellvitpp_probability'])==.8 for r in rows)
    predictions=geo['features'][0]['properties']['predictions']
    assert predictions['misleading-pannuke-text-but-consep']['label']=='Inflammatory'
    assert predictions['exact-declared-primary']['label']=='Neoplastic'
    assert len(predictions)==2, 'Both incompatible classifier heads remain separately retained'


@pytest.mark.parametrize('change', ['partial','unverified','index','wrong_image','wrong_frame','wrong_partition_count'])
def test_fullslide_incomplete_or_unregistered_outputs_are_rejected(crop_fixture,change):
    f=crop_fixture; pointer,receipt,index=fullslide_fixture(f)
    p=json.loads(pointer.read_text()); r=json.loads(receipt.read_text())
    if change=='partial': r['complete_shards']=1
    elif change=='unverified': r['every_core_verified']=False
    elif change=='index': index.write_bytes(index.read_bytes()+b'changed')
    elif change=='wrong_image':
        shifted=json.loads(f['shift'].read_text()); shifted['input_image']=str(f['image']); write_json(f['shift'],shifted)
    elif change in ('wrong_frame','wrong_partition_count'):
        project=json.loads(Path(p['project_path']).read_text()); partition_path=f['root']/'partition.json'
        partition=json.loads(partition_path.read_text())
        if change=='wrong_frame': project['tissues'][0]['image']['frame_id']='cellvit-frame-wrong'
        else:
            partition['partition_count']=1; write_json(partition_path,partition)
            project['tissues'][0]['cell_classification']['sha256']=sha(partition_path)
            r['partition_manifest_sha256']=sha(partition_path)
        write_json(Path(p['project_path']),project)
    write_json(receipt,r);p['completion_receipt_sha256']=sha(receipt);write_json(pointer,p)
    with pytest.raises(ValueError): export(f,project_pointer=pointer,source_metadata=receipt)
    assert not (f['root']/'out').exists(), 'Rejected source must not produce accepted segmentation artifacts'


def test_native_frame_mismatch_duplicate_ids_and_invalid_contour_fail(crop_fixture):
    f=crop_fixture; cells,metadata=native_fixture(f)
    original=json.loads(cells.read_text()); m=json.loads(metadata.read_text())
    m['frame_id']='wrong';write_json(metadata,m)
    with pytest.raises(ValueError,match='metadata disagree'): export(f,cellvit_cells=cells,cellvit_metadata=metadata)
    write_json(metadata,original['metadata'])
    original['cells'][1]['cell_id']='a';write_json(cells,original)
    with pytest.raises(ValueError,match='unique canonical'): export(f,cellvit_cells=cells,cellvit_metadata=metadata)
    original['cells'][1]['cell_id']='b';original['cells'][0]['contour']=[[2,2],[6,6],[6,2],[2,6]];write_json(cells,original)
    with pytest.raises(ValueError,match='valid positive-area polygon'): export(f,cellvit_cells=cells,cellvit_metadata=metadata)


def test_empty_valid_support_exports_zero_cells_without_fabricated_classification(crop_fixture):
    f=crop_fixture; cells,metadata=native_fixture(f,[])
    result=export(f,cellvit_cells=cells,cellvit_metadata=metadata)
    assert result['cell_count']==0
    assert np.count_nonzero(tifffile.imread(f['root']/'out/labels.tif'))==0
    assert json.loads((f['root']/'out/cell_instances.geojson').read_text())['features']==[]


@pytest.mark.parametrize('taxonomy', ['consep', None])
def test_roi_native_type_requires_explicit_pannuke_head(crop_fixture, taxonomy):
    f=crop_fixture;cells,metadata=native_fixture(f)
    payload=json.loads(cells.read_text());m=json.loads(metadata.read_text())
    payload['taxonomy']=taxonomy;m['taxonomy']=taxonomy;payload['metadata']=m
    write_json(cells,payload);write_json(metadata,m)
    with pytest.raises(ValueError,match='explicitly declared PanNuke'):
        export(f,cellvit_cells=cells,cellvit_metadata=metadata)
    assert not (f['root']/'out').exists()


def test_primary_view_rejects_ambiguous_pannuke_models_and_ontology_mismatch():
    m={'model_id':'one','taxonomy':'pannuke','status':'available','class_names':['Background','Neoplastic']}
    with pytest.raises(ValueError,match='Exactly one'):
        primary.pannuke_model([m,{**m,'model_id':'two'}])
    with pytest.raises(ValueError,match='declared ontology'):
        primary.pannuke_view({'predictions':{'one':{'status':'predicted','class_id':1,'label':'Inflammatory','probability':.9}}},
            {'input_scope':'full_slide','primary_type_model':m})
