"""Actual binary shard round trips through canonical profile/hierarchy readers."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'bin'))
from build_cell_profiles import canonical_cells, export_uni2_block, build_profiles
from discover_tissue_hierarchy import load_embedding_block
from cell_profile_io import sha256_file
from uni2_embedding_io import write_binary_shard, binary_shard_paths
from test_cell_profiles import inputs, receipt_fixture
from test_tissue_hierarchy import grid_fixture, definitions


def cell_metadata(ids=('1',)):
    return pd.DataFrame({'cell_id': list(ids), 'observation_type': 'cell',
        'cx': [2 if str(i) == '1' else 8 for i in ids], 'cy': [2 if str(i) == '1' else 4 for i in ids],
        'source_mpp': .5, 'model_tile_size': 224, 'target_mpp': .25,
        'effective_mpp': .25, 'mask_context_mode': 'full'})


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_profile_binary_matches_legacy_exact_float32_arrays_and_missingness(tmp_path, dtype):
    objects, shift = inputs(tmp_path)
    cells, _ = canonical_cells(objects, 'sample', shift)
    rows = cell_metadata()
    values = np.array([[-0., np.nextafter(np.float32(0), np.float32(1)), .123456789123]], dtype=dtype)
    names = ['feat_1', 'feat_2', 'feat_3']
    legacy = tmp_path / 'original.csv'
    pd.concat([rows, pd.DataFrame(values, columns=names)], axis=1).to_csv(legacy, index=False)
    shard = write_binary_shard(tmp_path / 'binary' / 'a_embeddings_shard0000', rows, values, names)
    seen_csv, csv_record = export_uni2_block(cells, legacy, 'csv', tmp_path)
    seen_bin, record = export_uni2_block(cells, shard.parent, 'binary', tmp_path)
    np.testing.assert_array_equal(seen_bin, seen_csv)
    actual, expected = np.load(tmp_path/'binary.npy'), np.load(tmp_path/'csv.npy')
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert np.isnan(actual[0]).all() and seen_bin.tolist() == [False, True]
    assert record['feature_definition'] == csv_record['feature_definition']
    assert record['reference_compatible'] is False
    assert record['provenance']['status'] == 'legacy_unverified_inputs'
    assert record['storage']['binary_input_dtypes'] == [np.dtype(dtype).name]
    assert {Path(s['path']) for s in record['sources']} == {p.resolve() for p in binary_shard_paths(shard)}


def binary_receipt_fixture(tmp_path, *, missing_target=False):
    opts, legacy = receipt_fixture(tmp_path)
    frame = pd.read_csv(legacy/'a_embeddings_shard0.csv', dtype={'cell_id':str})
    if missing_target:
        frame.target_mpp=np.nan
        frame.to_csv(legacy/'a_embeddings_shard0.csv',index=False)
        grid_path=legacy/'.a_grid_complete.json'
        grid=json.loads(grid_path.read_text())
        grid['shard_files'][0]['sha256']=sha256_file(legacy/'a_embeddings_shard0.csv')
        grid_path.write_text(json.dumps(grid))
    rows = frame.drop(columns=['feat_1'])
    values = frame[['feat_1']].to_numpy(np.float32)
    source = tmp_path/'binary'
    shard = write_binary_shard(source/'a_embeddings_shard0000', rows, values, embedding_mode='tile')
    complete = json.loads((legacy/'.a_embedding_complete.json').read_text())
    complete['cache_contract']['parameters']['embedding_storage'] = 'binary'
    complete['embedding_storage'] = 'binary'
    digest = hashlib.sha256(json.dumps(complete['cache_contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    complete['cache_contract_sha256'] = digest
    payloads = binary_shard_paths(shard)
    marker_path = source/'.a_grid_complete.json'
    marker_path.write_text(json.dumps({'cache_contract_sha256': digest, 'rows_written':2,
        'index_start':0, 'index_end':2, 'shards':1, 'embedding_storage':'binary', 'embedding_mode':'tile', 'shard_files':[
            {'name':shard.name, 'sha256':sha256_file(shard), 'size_bytes':shard.stat().st_size, 'storage':'binary',
             'payload_files':[{'name':p.name,'sha256':sha256_file(p),'size_bytes':p.stat().st_size} for p in payloads[1:]]}]}))
    complete['payload_inventory'] = [{'path':p.relative_to(source).as_posix(),'sha256':sha256_file(p),'size_bytes':p.stat().st_size}
        for p in (*payloads, marker_path)]
    complete_path = source/'.a_embedding_complete.json'
    complete_path.write_text(json.dumps(complete))
    return opts, legacy, source, shard, marker_path, complete_path


def refresh_receipts(source, shard, marker, complete):
    """Rebind a deliberately changed fixture so nested invariants are tested."""
    grid = json.loads(marker.read_text())
    grid['shard_files'][0].update(sha256=sha256_file(shard), size_bytes=shard.stat().st_size)
    marker.write_text(json.dumps(grid))
    root = json.loads(complete.read_text())
    root['payload_inventory'] = [{'path':p.relative_to(source).as_posix(), 'sha256':sha256_file(p), 'size_bytes':p.stat().st_size}
        for p in (*binary_shard_paths(shard), marker)]
    complete.write_text(json.dumps(root))


def test_profile_receipts_bind_all_payloads_without_storage_in_biological_identity(tmp_path):
    opts, legacy, source, shard, marker, complete = binary_receipt_fixture(tmp_path)
    _, old = build_profiles(opts)
    opts.outdir = str(tmp_path/'binary_profiles'); opts.uni2_context = str(source)
    cells, new = build_profiles(opts)
    block = new['feature_blocks']['uni2_context']; old_block = old['feature_blocks']['uni2_context']
    np.testing.assert_array_equal(np.load(tmp_path/'profiles'/old_block['path']), np.load(Path(opts.outdir)/block['path']))
    assert block['reference_compatible'] is True
    assert block['provenance']['status'] == 'verified_exact_inputs_and_shards'
    assert block['provenance']['storage']['root_payload_inventory'] == 'verified_exact_files'
    assert block['feature_definition'] == old_block['feature_definition']
    assert 'embedding_storage' not in block['feature_definition']['verified_execution']['inference_parameters']
    expected = {p.resolve() for p in (*binary_shard_paths(shard), marker, complete)}
    assert {Path(s['path']) for s in block['sources']} == expected
    assert cells.cell_id.tolist() == ['2','1']


@pytest.mark.parametrize('corrupt', ['feature_bytes','row_bytes','root_hash','root_omission','grid_payload','missing_root_inventory','receipt_storage'])
def test_profile_rejects_binary_or_receipt_corruption(tmp_path, corrupt):
    opts, _, source, shard, marker, complete = binary_receipt_fixture(tmp_path)
    if corrupt in {'feature_bytes','row_bytes'}:
        path = binary_shard_paths(shard)[2 if corrupt == 'feature_bytes' else 1]
        value = bytearray(path.read_bytes()); value[-1] ^= 1; path.write_bytes(value)
    elif corrupt.startswith('root_'):
        record = json.loads(complete.read_text())
        if corrupt == 'root_hash': record['payload_inventory'][0]['sha256'] = '0'*64
        else: record['payload_inventory'].pop()
        complete.write_text(json.dumps(record))
    elif corrupt == 'grid_payload':
        value = json.loads(marker.read_text()); value['shard_files'][0]['payload_files'][0]['sha256'] = '0'*64
        marker.write_text(json.dumps(value))
        value = json.loads(complete.read_text())
        for record in value['payload_inventory']:
            if record['path'] == marker.name:
                record.update(sha256=sha256_file(marker), size_bytes=marker.stat().st_size)
        complete.write_text(json.dumps(value))
    else:
        value = json.loads(complete.read_text())
        if corrupt == 'missing_root_inventory': value.pop('payload_inventory')
        else:
            value['cache_contract']['parameters']['embedding_storage'] = 'csv'
            value['cache_contract_sha256'] = hashlib.sha256(json.dumps(value['cache_contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        complete.write_text(json.dumps(value))
    opts.uni2_context = str(source)
    with pytest.raises(ValueError, match='(SHA256|checksum|inventory|storage)'):
        build_profiles(opts)


def test_hierarchy_binary_and_legacy_array_order_missingness_and_bounded_reads(tmp_path, monkeypatch):
    grid, _, _, _ = grid_fixture(rows=2, cols=2)
    selected = grid.iloc[[2,0,1]]
    rows = pd.DataFrame({'cell_id':selected.label.astype(str).to_numpy(), 'cx':selected.x.to_numpy(),
        'cy':selected.y.to_numpy(), 'observation_type':'grid', 'extraction_tile_size':12, 'source_mpp':.5})
    values = np.array([[3.3,1.1],[6.6,4.4],[9.9,7.7]],np.float32)
    names = ['feat_2','feat_1']
    legacy = tmp_path/'grid.csv'
    pd.concat([rows,pd.DataFrame(values,columns=names)],axis=1).to_csv(legacy,index=False)
    shard = write_binary_shard(tmp_path/'grid_binary'/'g_embeddings_shard0000',rows,values,names)
    expected, old_names, _ = load_embedding_block(legacy,grid,tmp_path/'legacy.npy',definitions()['context'],chunk_rows=1)
    calls=[]; original=pd.read_csv
    def bounded(path,*args,**kwargs):
        if str(path).endswith('.rows.csv'):
            calls.append(kwargs.get('chunksize')); assert kwargs.get('chunksize') == 1
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',bounded)
    actual, actual_names, provenance = load_embedding_block(shard.parent,grid,tmp_path/'binary.npy',definitions()['context'],chunk_rows=1)
    np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
    assert actual_names == old_names == ['feat_1','feat_2']
    assert provenance['observations_present'] == 3 and provenance['observations_missing'] == 1
    assert provenance['extraction_provenance']['status'] == 'legacy_unverified_inputs'
    assert {Path(p) for p in provenance['files_sha256']} == {p.resolve() for p in binary_shard_paths(shard)}
    assert calls == [1]


@pytest.mark.parametrize('problem', ['duplicate','foreign','coordinate','observation_type','mpp','mixed','orphan'])
def test_profile_binary_strict_identity_metadata_and_format_selection(tmp_path, problem):
    objects, shift = inputs(tmp_path); cells,_=canonical_cells(objects,'sample',shift)
    rows=cell_metadata(); source=tmp_path/'binary'
    if problem=='foreign': rows.cell_id='not-a-canonical-cell'
    if problem=='coordinate': rows.cx=100
    if problem=='observation_type': rows.observation_type='grid'
    if problem=='mpp': rows.source_mpp=.25
    write_binary_shard(source/'a_embeddings_shard0000',rows,np.ones((1,1),np.float32))
    if problem=='duplicate': write_binary_shard(source/'a_embeddings_shard0001',rows,np.ones((1,1),np.float32))
    if problem=='mixed': pd.DataFrame({'cell_id':['1'],'feat_1':[1]}).to_csv(source/'a_embeddings_shard0002.csv',index=False)
    if problem=='orphan': (source/'a_embeddings_shard9999.features.bin').write_bytes(b'1234')
    with pytest.raises(ValueError): export_uni2_block(cells,source,'bad',tmp_path)


def test_hierarchy_npy_bundle_cannot_silently_hide_new_shards(tmp_path):
    source=tmp_path/'ambiguous'; source.mkdir()
    (source/'embedding_manifest.json').write_text(json.dumps({'format':'cellphenotyper_grid_embeddings_npy'}))
    write_binary_shard(source/'a_embeddings_shard0000',cell_metadata(),np.ones((1,1),np.float32))
    grid,_,_,_=grid_fixture(rows=1,cols=1)
    with pytest.raises(ValueError,match='Ambiguous'):
        load_embedding_block(source,grid,tmp_path/'data.npy',definitions()['context'])


def test_profile_checks_actual_rows_not_only_self_consistent_receipt_counts(tmp_path):
    opts, _, source, shard, marker, complete = binary_receipt_fixture(tmp_path)
    grid = json.loads(marker.read_text()); grid.update(rows_written=3, index_end=3)
    marker.write_text(json.dumps(grid))
    root = json.loads(complete.read_text()); root.update(rows_written=3, expected_observations=3)
    complete.write_text(json.dumps(root)); refresh_receipts(source, shard, marker, complete)
    opts.uni2_context = str(source)
    with pytest.raises(ValueError, match='actual shard rows'):
        build_profiles(opts)


@pytest.mark.parametrize('location', ['shard', 'grid', 'completion', 'all_swapped', 'missing_shard_mode'])
def test_profile_rejects_mode_receipt_conflicts_or_swapped_context(tmp_path, location):
    opts, _, source, shard, marker, complete = binary_receipt_fixture(tmp_path)
    targets = {'shard':shard, 'grid':marker, 'completion':complete}
    selected = list(targets.values()) if location == 'all_swapped' else [targets.get(location, shard)]
    for path in selected:
        record = json.loads(path.read_text())
        if location == 'missing_shard_mode': record.pop('embedding_mode')
        else: record['embedding_mode'] = 'inner_square'
        path.write_text(json.dumps(record))
    refresh_receipts(source, shard, marker, complete)
    opts.uni2_context = str(source)
    with pytest.raises(ValueError, match='embedding_mode'):
        build_profiles(opts)


def test_paired_local_mode_uses_own_receipts_not_shared_primary_contract(tmp_path):
    opts, _, source, shard, marker, complete = binary_receipt_fixture(tmp_path)
    for path in (shard, marker, complete):
        record=json.loads(path.read_text()); record['embedding_mode']='inner_square'
        if path == complete:
            record['cache_contract']['parameters']['embedding_mode']='tile'
            digest=hashlib.sha256(json.dumps(record['cache_contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            record['cache_contract_sha256']=digest
        path.write_text(json.dumps(record))
    record=json.loads(marker.read_text()); record['cache_contract_sha256']=digest
    marker.write_text(json.dumps(record)); refresh_receipts(source, shard, marker, complete)
    opts.uni2_context=None; opts.uni2_local=str(source)
    _, manifest=build_profiles(opts)
    block=manifest['feature_blocks']['uni2_local']
    assert block['reference_compatible']
    assert block['provenance']['representation_binding']['embedding_mode']=='inner_square'
    assert block['feature_definition']['verified_execution']['inference_parameters']['embedding_mode']=='tile'


def test_legacy_declared_mode_rejects_wrong_role_without_claiming_encoder_provenance(tmp_path):
    objects,shift=inputs(tmp_path); cells,_=canonical_cells(objects,'sample',shift)
    shard=write_binary_shard(tmp_path/'a_embeddings_shard0',cell_metadata(),np.ones((1,1),np.float32),embedding_mode='inner_square')
    with pytest.raises(ValueError,match='expected tile profile role'):
        export_uni2_block(cells,shard,'uni2_context',tmp_path)
    _,record=export_uni2_block(cells,shard,'uni2_local',tmp_path)
    assert record['provenance']['status']=='legacy_unverified_inputs'
    assert not record['reference_compatible']


def test_binary_root_inventory_survives_legitimate_staged_directory_symlink(tmp_path):
    opts,_,source,_,_,_=binary_receipt_fixture(tmp_path)
    staging=tmp_path/'staged_binary'; staging.symlink_to(source,target_is_directory=True)
    opts.uni2_context=str(staging)
    _,manifest=build_profiles(opts)
    assert manifest['feature_blocks']['uni2_context']['provenance']['storage']['root_payload_inventory']=='verified_exact_files'


def grid_binary_receipt_fixture(tmp_path, problem=None):
    _, legacy = receipt_fixture(tmp_path)
    grid,_,_,_=grid_fixture(rows=2,cols=2)
    selected=grid.iloc[[2,0,1]]
    rows=pd.DataFrame({'cell_id':selected.label.astype(str).to_numpy(), 'cx':selected.x.to_numpy(),
        'cy':selected.y.to_numpy(), 'observation_type':'grid', 'extraction_tile_size':12, 'source_mpp':.5})
    if problem=='foreign': rows.loc[0,'cell_id']='999'
    if problem=='coordinate': rows.loc[0,'cx']+=1
    if problem=='observation_type': rows.observation_type='cell'
    if problem=='mpp': rows.source_mpp=.25
    source=tmp_path/'grid_binary'
    values=np.array([[.3,.7],[.1,.5],[.2,.6]],np.float32)
    shard=write_binary_shard(source/'grid_embeddings_shard0000',rows,values,embedding_mode='tile')
    root=json.loads((legacy/'.a_embedding_complete.json').read_text())
    root.update(observation_type='grid',embedding_storage='binary',rows_written=3,expected_observations=3)
    root['cache_contract']['parameters'].update(observation_type='grid',embedding_storage='binary')
    digest=hashlib.sha256(json.dumps(root['cache_contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    root['cache_contract_sha256']=digest
    marker=source/'.grid_grid_complete.json'
    marker.write_text(json.dumps({'cache_contract_sha256':digest, 'rows_written':3, 'index_start':0,
        'index_end':3, 'shards':1, 'embedding_storage':'binary', 'embedding_mode':'tile',
        'shard_files':[{'name':shard.name,'sha256':sha256_file(shard),'size_bytes':shard.stat().st_size,'storage':'binary',
            'payload_files':[{'name':p.name,'sha256':sha256_file(p),'size_bytes':p.stat().st_size} for p in binary_shard_paths(shard)[1:]]}]}))
    complete=source/'.grid_embedding_complete.json'; complete.write_text(json.dumps(root))
    refresh_receipts(source,shard,marker,complete)
    definition={**definitions()['context'],'source_mpp_xy':[.5,.5],'embedding_mode':'tile'}
    return grid,source,shard,marker,complete,definition,values


def test_hierarchy_grid_receipts_bind_all_payloads_without_inventing_source_lineage(tmp_path):
    grid,source,shard,marker,complete,definition,values=grid_binary_receipt_fixture(tmp_path)
    actual,names,provenance=load_embedding_block(source,grid,tmp_path/'grid.npy',definition,chunk_rows=1)
    np.testing.assert_array_equal(actual[:3].view(np.uint32),values[[1,2,0]].view(np.uint32))
    assert np.isnan(actual[3]).all() and names==['feat_1','feat_2']
    assert provenance['extraction_provenance']['status']=='verified_shards_partial_input_binding'
    assert provenance['extraction_provenance']['verified_inputs']==[]
    assert provenance['extraction_provenance']['rows_verified']==3
    assert provenance['extraction_provenance']['storage']['root_payload_inventory']=='verified_exact_files'
    assert 'source_inputs' not in provenance
    assert {Path(p) for p in provenance['files_sha256']}=={p.resolve() for p in (*binary_shard_paths(shard),marker,complete)}


@pytest.mark.parametrize('problem',['foreign','coordinate','observation_type','mpp','duplicate','count','mode'])
def test_hierarchy_binary_identity_receipt_and_mode_negatives(tmp_path,problem):
    grid,source,shard,marker,complete,definition,_=grid_binary_receipt_fixture(tmp_path,problem)
    if problem=='duplicate':
        # A second independently valid shard repeats an existing observation.
        rows=pd.DataFrame({'cell_id':['1'],'cx':[2],'cy':[2],'observation_type':['grid']})
        write_binary_shard(source/'grid_embeddings_shard0001',rows,np.ones((1,2),np.float32),embedding_mode='tile')
    elif problem=='count':
        record=json.loads(marker.read_text()); record.update(rows_written=4,index_end=4)
        marker.write_text(json.dumps(record))
        record=json.loads(complete.read_text()); record.update(rows_written=4,expected_observations=4)
        complete.write_text(json.dumps(record)); refresh_receipts(source,shard,marker,complete)
    elif problem=='mode':
        definition['embedding_mode']='inner_square'
    with pytest.raises(ValueError):
        load_embedding_block(source,grid,tmp_path/'bad.npy',definition,chunk_rows=1)


@pytest.mark.parametrize('location',['completion','grid','shard_record'])
def test_binary_storage_declarations_cannot_disagree(tmp_path,location):
    opts,_,source,shard,marker,complete=binary_receipt_fixture(tmp_path)
    path=complete if location=='completion' else marker
    record=json.loads(path.read_text())
    if location=='shard_record': record['shard_files'][0]['storage']='csv'
    else: record['embedding_storage']='csv'
    path.write_text(json.dumps(record)); refresh_receipts(source,shard,marker,complete)
    opts.uni2_context=str(source)
    with pytest.raises(ValueError,match='storage declaration'):
        build_profiles(opts)


@pytest.mark.parametrize('missing_target',[False,True])
def test_physical_definition_is_identical_for_non_dyadic_scale_and_missing_metadata(tmp_path,missing_target):
    objects,shift=inputs(tmp_path); cells,_=canonical_cells(objects,'sample',shift)
    rows=cell_metadata(); effective_mpp=.273774374855905*224/256
    rows.effective_mpp=effective_mpp
    if missing_target: rows.target_mpp=np.nan
    values=np.array([[.123456789]],np.float32)
    legacy=tmp_path/'legacy.csv'
    pd.concat([rows,pd.DataFrame(values,columns=['feat_1'])],axis=1).to_csv(legacy,index=False)
    shard=write_binary_shard(tmp_path/'a_embeddings_shard0',rows,values)
    _,csv_record=export_uni2_block(cells,legacy,'csv',tmp_path)
    _,binary_record=export_uni2_block(cells,shard,'binary',tmp_path)
    assert csv_record['feature_definition']==binary_record['feature_definition']
    assert binary_record['feature_definition']['preprocessing']['effective_mpp']==str(effective_mpp)
    if missing_target:
        assert binary_record['feature_definition']['preprocessing']['target_mpp']=='unavailable'
        assert not csv_record['reference_compatible'] and not binary_record['reference_compatible']


def test_hash_verified_profiles_with_missing_physical_setting_are_not_reference_compatible(tmp_path):
    opts,_,source,_,_,_=binary_receipt_fixture(tmp_path,missing_target=True)
    _,legacy=build_profiles(opts)
    opts.outdir=str(tmp_path/'binary_profiles'); opts.uni2_context=str(source)
    _,binary=build_profiles(opts)
    old,new=legacy['feature_blocks']['uni2_context'],binary['feature_blocks']['uni2_context']
    assert old['feature_definition']==new['feature_definition']
    for block in (old,new):
        assert block['provenance']['status']=='verified_exact_inputs_and_shards'
        assert block['feature_definition']['preprocessing']['target_mpp']=='unavailable'
        assert not block['reference_compatible']


def test_profile_csv_binary_preserve_literal_and_large_ids_with_missing_numeric_metadata(tmp_path):
    _,shift=inputs(tmp_path)
    ids=['NA','NaN','0001','9007199254740993','absent']
    objects=tmp_path/'literal_objects.csv'
    pd.DataFrame({'label':ids,'x':[1,3,5,7,9],'y':2,'xmin':[0,2,4,6,8],
        'xmax':[2,4,6,8,10],'ymin':1,'ymax':3}).to_csv(objects,index=False)
    cells,_=canonical_cells(objects,'sample',shift)
    assert cells.cell_id.tolist()==ids
    selected=cells.iloc[[3,0,2,1]]
    rows=pd.DataFrame({'cell_id':selected.cell_id.to_numpy(),'observation_type':'cell',
        'cx':selected.x_crop_px.to_numpy(),'cy':selected.y_crop_px.to_numpy(),
        'source_mpp':.5,'model_tile_size':224,'target_mpp':np.nan,'effective_mpp':.25,'mask_context_mode':'full'})
    values=np.array([[4,40],[1,10],[3,30],[2,20]],np.float32)
    legacy=tmp_path/'literal.csv'
    pd.concat([rows,pd.DataFrame(values,columns=['feat_1','feat_2'])],axis=1).to_csv(legacy,index=False)
    binary=write_binary_shard(tmp_path/'literal_embeddings_shard0',rows,values)
    for source,name in ((legacy,'csv'),(binary,'binary')):
        seen,record=export_uni2_block(cells,source,name,tmp_path)
        actual=np.load(tmp_path/f'{name}.npy')
        assert seen.tolist()==[True,True,True,True,False]
        np.testing.assert_array_equal(actual[:4],[[1,10],[2,20],[3,30],[4,40]])
        assert np.isnan(actual[4]).all()
        assert record['feature_definition']['preprocessing']['target_mpp']=='unavailable'
    np.testing.assert_array_equal(np.load(tmp_path/'csv.npy').view(np.uint32),np.load(tmp_path/'binary.npy').view(np.uint32))


def test_hierarchy_csv_binary_preserve_literal_and_large_ids(tmp_path):
    grid,_,_,_=grid_fixture(rows=1,cols=5)
    grid['label']=['NA','NaN','0001','9007199254740993','absent']
    selected=grid.iloc[[3,0,2,1]]
    rows=pd.DataFrame({'cell_id':selected.label.to_numpy(),'observation_type':'grid',
        'cx':selected.x.to_numpy(),'cy':selected.y.to_numpy(),'source_mpp':.5,'extraction_tile_size':12})
    values=np.array([[4],[1],[3],[2]],np.float32)
    legacy=tmp_path/'literal_grid.csv'
    pd.concat([rows,pd.DataFrame(values,columns=['feat_1'])],axis=1).to_csv(legacy,index=False)
    binary=write_binary_shard(tmp_path/'literal_grid_embeddings_shard0',rows,values)
    definition={**definitions()['context'],'source_mpp_xy':[.5,.5]}
    bundle=tmp_path/'existing_npy_bundle'; bundle.mkdir()
    missing=grid.iloc[-1]
    full_rows=pd.concat([rows,pd.DataFrame({'cell_id':[missing.label],'observation_type':['grid'],
        'cx':[missing.x],'cy':[missing.y],'source_mpp':[.5],'extraction_tile_size':[12]})],ignore_index=True)
    full_rows.to_csv(bundle/'rows.csv',index=False)
    np.save(bundle/'features.npy',np.concatenate([values,np.full((1,1),np.nan,np.float32)]))
    (bundle/'embedding_manifest.json').write_text(json.dumps({'format':'cellphenotyper_grid_embeddings_npy',
        'feature_definition':definition,'feature_names':['feat_1'],
        'rows':{'path':'rows.csv','sha256':sha256_file(bundle/'rows.csv'),'count':5},
        'matrix':{'path':'features.npy','sha256':sha256_file(bundle/'features.npy'),'shape':[5,1]}}))
    results=[]
    for source,name in ((legacy,'csv'),(binary,'binary'),(bundle,'existing_npy')):
        matrix,_,record=load_embedding_block(source,grid,tmp_path/f'{name}.npy',definition,chunk_rows=1)
        np.testing.assert_array_equal(matrix[:4,0],[1,2,3,4])
        assert np.isnan(matrix[4]).all() and record['observations_missing']==1
        results.append(matrix)
    for actual in results[1:]:
        np.testing.assert_array_equal(results[0].view(np.uint32),actual.view(np.uint32))
