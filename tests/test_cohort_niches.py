"""Actual tiny profile/assembly/cohort round trips; no learned models or labels."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from assemble_spatial_cell_profiles import assemble
from build_cell_profiles import build_profiles, parser
from cell_profile_io import sha256_file
from fit_cohort_niches import fit_cohort, source_record, compatibility, pool_profiles, normalize_missing_composition


def make_spatial(tmp_path, sample, *, with_features=True, phenotype='A'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    xy = np.array([[4,4],[5,4],[4,5],[32,4],[33,4],[32,5],[21,4]])
    objects = tmp_path / 'objects.csv'
    pd.DataFrame({'label':['007','NA','3','4','5','6','7'], 'x':xy[:,0], 'y':xy[:,1],
        'xmin':xy[:,0]-1,'ymin':xy[:,1]-1,'xmax':xy[:,0]+1,'ymax':xy[:,1]+1,
        'cellvitpp_type':[phenotype]*7}).to_csv(objects,index=False)
    shift = tmp_path / 'shift.json'
    shift.write_text(json.dumps({'crop_size':{'width':40,'height':12},
        'offset_crop_to_original':{'dx':0,'dy':0}}))
    resolution = tmp_path / 'resolution.json'
    resolution.write_text(json.dumps({'status':'pass','mpp_x':.5,'mpp_y':.5}))
    base = tmp_path / 'base'
    _, manifest = build_profiles(parser().parse_args(['--objects',str(objects),'--sample-id',sample,
        '--shift',str(shift),'--resolution-json',str(resolution),'--outdir',str(base)]))
    manifest['phenotype_definition'] = {'verified':True,
        'definition':{'method':'synthetic_test_fixture_labels_not_a_detector','vocabulary':['A','B']}}
    groups = ()
    if with_features:
        values = np.array([[0.],[1.],[0.],[100.],[101.],[100.],[np.nan]])
        block = base / 'feature_blocks/synthetic.npy'
        np.save(block,values,allow_pickle=False)
        manifest['feature_blocks']['synthetic'] = {'path':'feature_blocks/synthetic.npy',
            'sha256':sha256_file(block),'shape':list(values.shape),'dtype':'float64','feature_names':['test_feature'],
            'reference_compatible':True,'feature_definition':{'method':'synthetic_test_fixture_not_learned'}}
        (base / 'cell_profiles_manifest.json').write_text(json.dumps(manifest))
        groups = ('synthetic',)
    (base / 'cell_profiles_manifest.json').write_text(json.dumps(manifest))
    support = np.ones((12,40),np.uint8)
    support[:,20:24] = 0
    mask = tmp_path / 'support.tif'
    tifffile.imwrite(mask,support)
    out = tmp_path / 'spatial'
    assemble(base,mask,shift,resolution,out,radii_um=(3.,),feature_groups=groups,repeats=2)
    return out


def rewrite_manifest(root, updater):
    path = root / 'cell_profiles_manifest.json'
    data = json.loads(path.read_text())
    updater(data)
    path.write_text(json.dumps(data))


def test_actual_two_specimen_shared_niches_preserve_sources_and_literal_ids(tmp_path):
    a,b = [make_spatial(tmp_path / name,name) for name in ('A','B')]
    before = {p:sha256_file(p) for root in (a,b) for p in root.rglob('*') if p.is_file()}
    assignments,model,summary = fit_cohort([b,a],tmp_path/'cohort',fixed_k=2,repeats=2)
    assert len(assignments)==14 and summary['selected_k']==2
    assert assignments.sample_id.tolist()==['A']*7+['B']*7
    assert assignments.cell_id.tolist()==['007','NA','3','4','5','6','7']*2
    for _,rows in assignments.groupby('sample_id',sort=False):
        assert rows.cohort_niche_id.iloc[:3].nunique()==1
        assert rows.cohort_niche_id.iloc[3:6].nunique()==1
        assert rows.cohort_niche_id.iloc[0]!=rows.cohort_niche_id.iloc[3]
        assert pd.isna(rows.cohort_niche_id.iloc[-1])
        assert rows.cohort_niche_status.iloc[-1]=='outside_tissue_support'
    assert assignments.cohort_niche_id.iloc[:7].reset_index(drop=True).equals(assignments.cohort_niche_id.iloc[7:].reset_index(drop=True))
    assert 'own:synthetic' in model['feature_groups']
    assert model['reference_assignment'] is False
    assert before=={p:sha256_file(p) for p in before}
    for source in model['sources']:
        assert len(source['graphs'])==1
    completion=json.loads((tmp_path/'cohort/cohort_niches_completion.json').read_text())
    assert all(sha256_file(tmp_path/'cohort'/name)==value for name,value in completion['files'].items())
    readback=pd.read_parquet(tmp_path/'cohort/cohort_niche_assignments.parquet')
    pd.testing.assert_frame_equal(assignments,readback)
    again,second,_=fit_cohort([a,b],tmp_path/'again',fixed_k=2,repeats=2)
    assert second['cohort_niche_model_id']==model['cohort_niche_model_id']
    pd.testing.assert_frame_equal(assignments,again)


def test_missing_modality_is_nan_and_absent_phenotype_is_zero_only_with_neighbors(tmp_path):
    a=make_spatial(tmp_path/'a','A',phenotype='A')
    b=make_spatial(tmp_path/'b','B',with_features=False,phenotype='B')
    records=[source_record(path) for path in (a,b)]
    groups,_,_=compatibility(records)
    pooled,missing=pool_profiles(records,groups)
    assert pooled.loc[pooled.sample_id.eq('B'),'own_synthetic:test_feature'].isna().all()
    assert pooled.loc[pooled.sample_id.eq('A'),'r3um_phenotype_fraction:B'].iloc[:6].eq(0).all()
    assert pd.isna(pooled.loc[pooled.sample_id.eq('A'),'r3um_phenotype_fraction:B'].iloc[-1])
    assert 'own:synthetic' in missing['B']['absent_feature_groups']
    result,_,_=fit_cohort([a,b],tmp_path/'out',repeats=2)
    assert len(result)==14


@pytest.mark.parametrize('change', ['definition','unverified','count','escaping','graph','table','rows','legacy'])
def test_corruption_and_incompatible_inputs_fail_before_outputs(tmp_path,change):
    a,b=[make_spatial(tmp_path/name,name) for name in ('A','B')]
    if change=='definition':
        rewrite_manifest(b,lambda m:m['feature_blocks']['synthetic']['feature_definition'].update(method='different'))
    elif change=='unverified':
        rewrite_manifest(b,lambda m:m['feature_blocks']['synthetic'].update(reference_compatible=False))
    elif change=='count':
        rewrite_manifest(b,lambda m:m.update(cell_count=8))
    elif change=='escaping':
        rewrite_manifest(b,lambda m:m['spatial_graphs']['3'].update(path='../support.tif'))
    elif change=='graph':
        path=b/'neighborhood_graph_3um.npz'
        graph=sparse.load_npz(path).tolil(); graph[0,0]=1
        sparse.save_npz(path,graph.tocsr())
        rewrite_manifest(b,lambda m:m['spatial_graphs']['3'].update(sha256=sha256_file(path)))
    elif change=='table':
        with (b/'cell_profiles.parquet').open('ab') as stream: stream.write(b'corruption')
    elif change=='rows':
        path=b/'feature_rows.csv'
        rows=pd.read_csv(path,dtype=str,keep_default_na=False).iloc[::-1]
        rows.to_csv(path,index=False)
        rewrite_manifest(b,lambda m:m['files'].update({'feature_rows.csv':sha256_file(path)}))
    elif change=='legacy':
        path=b/'neighborhood_summary.json'; data=json.loads(path.read_text())
        data.pop('feature_representation_version'); path.write_text(json.dumps(data))
        rewrite_manifest(b,lambda m:m['files'].update({'neighborhood_summary.json':sha256_file(path)}))
    with pytest.raises(ValueError): fit_cohort([a,b],tmp_path/'out',repeats=2)
    assert not (tmp_path/'out').exists()


def test_duplicate_single_and_memory_budget_fail_closed(tmp_path):
    a,b=[make_spatial(tmp_path/name,name) for name in ('A','B')]
    for inputs in ([a],[a,a]):
        with pytest.raises(ValueError,match='two distinct specimens'):
            fit_cohort(inputs,tmp_path/'out')
    with pytest.raises(ValueError,match='working estimate'):
        fit_cohort([a,b],tmp_path/'out',max_working_mb=.000001)
    assert not (tmp_path/'out').exists()


def test_assembly_available_default_separate_weights_and_dense_memory_guard(tmp_path):
    make_spatial(tmp_path/'A','A')
    source=tmp_path/'A'
    cells,_=assemble(source/'base',source/'support.tif',source/'shift.json',source/'resolution.json',
        source/'auto',radii_um=(3.,),feature_weights={'own:synthetic':4},repeats=2)
    assert 'own_synthetic:test_feature' in cells
    summary=json.loads((source/'auto/neighborhood_summary.json').read_text())
    assert summary['selected_cell_feature_groups']==['synthetic']
    assert summary['niche_discovery']['scaling']['own:synthetic']['weight']==4
    assert summary['niche_discovery']['scaling']['synthetic']['weight']==1
    with pytest.raises(ValueError,match='working estimate'):
        assemble(source/'base',source/'support.tif',source/'shift.json',source/'resolution.json',
            source/'small',max_working_mb=.000001)
    assert not (source/'small').exists()


@pytest.mark.parametrize('policy', ['unverified','incompatible'])
def test_shared_fit_never_equates_unverified_or_different_phenotype_taxonomies(tmp_path,policy):
    a,b=[make_spatial(tmp_path/name,name) for name in ('A','B')]
    rewrite_manifest(b,lambda m:m['phenotype_definition'].update(
        {'verified':False} if policy=='unverified' else {'definition':{'method':'other_taxonomy'}}))
    result,model,summary=fit_cohort([a,b],tmp_path/'out',repeats=2)
    assert len(result)==14
    assert not summary['phenotype_composition_policy']['included_in_shared_fit']
    assert 'composition' not in model['feature_groups']
    assert 'composition' in model['available_feature_groups']
    with pytest.raises(ValueError,match='composition weight'):
        fit_cohort([a,b],tmp_path/'explicit',feature_weights={'composition':1},repeats=2)


def test_explicit_missing_phenotype_aliases_coalesce_only_in_temporary_representation():
    columns=['r3um_phenotype_fraction:'+value for value in ('unknown','__unknown__','unknown_99','lymphocyte')]
    frame=pd.DataFrame([[.2,.1,.3,.4],[np.nan]*4],columns=columns)
    policy={'verified_definition':{'unknown_label_semantics':
        'unknown, __unknown__, and producer unknown_<unmapped_type_id> are missing categorical assignments'}}
    groups,mapping=normalize_missing_composition(frame,{'composition':columns},policy)
    assert groups['composition']==['r3um_phenotype_fraction:__unknown__','r3um_phenotype_fraction:lymphocyte']
    assert frame['r3um_phenotype_fraction:__unknown__'].iloc[0]==pytest.approx(.6)
    assert pd.isna(frame['r3um_phenotype_fraction:__unknown__'].iloc[1])
    assert len(mapping['r3um_phenotype_fraction:__unknown__'])==3
