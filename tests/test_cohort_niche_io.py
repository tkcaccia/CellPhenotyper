"""Portable attachment contracts using actual tiny producer outputs, no inference."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_cohort_niches import make_spatial, rewrite_manifest
from fit_cohort_niches import fit_cohort, digest
from cell_profile_io import sha256_file
from cohort_niche_io import (load_cohort_bundle, verify_cohort_sources, COHORT_COLUMNS,
                             KEYS, MODEL, SUMMARY, ASSIGNMENTS, COMPLETION)


@pytest.fixture
def bundle(tmp_path):
    profiles = [make_spatial(tmp_path/name, name) for name in ('A', 'B')]
    output = tmp_path/'cohort'
    assignments, model, summary = fit_cohort(profiles, output, fixed_k=2, repeats=2)
    return profiles, output, assignments, model, summary


def reseal(output, *, model=None, summary=None, assignments=None):
    """Intentionally rebuild engineering checksums to test semantic rejection."""
    if model is not None:
        model = dict(model)
        model.pop('cohort_niche_model_id', None)
        model['cohort_niche_model_id'] = digest(model)
        (output/MODEL).write_text(json.dumps(model, allow_nan=False))
    else:
        model = json.loads((output/MODEL).read_text())
    if summary is None:
        summary = json.loads((output/SUMMARY).read_text())
    summary['cohort_niche_model_id'] = model['cohort_niche_model_id']
    (output/SUMMARY).write_text(json.dumps(summary, allow_nan=False))
    if assignments is not None:
        assignments['cohort_niche_model_id'] = model['cohort_niche_model_id']
        assignments.to_parquet(output/ASSIGNMENTS, index=False)
    completion = json.loads((output/COMPLETION).read_text())
    completion['cohort_niche_model_id'] = model['cohort_niche_model_id']
    completion['files'] = {name:sha256_file(output/name) for name in (MODEL,SUMMARY,ASSIGNMENTS)}
    (output/COMPLETION).write_text(json.dumps(completion))


def test_actual_fit_source_order_missingness_and_portable_record(bundle, tmp_path):
    profiles, output, expected, model, _ = bundle
    before = {p:sha256_file(p) for root in profiles for p in root.rglob('*') if p.is_file()}
    for source in profiles:
        frame, record = load_cohort_bundle(output, profile_dir=source)
        sample = record['sample_id']
        pd.testing.assert_frame_equal(frame, expected.loc[expected.sample_id.eq(sample),KEYS+COHORT_COLUMNS].reset_index(drop=True))
        assert frame.cell_id.tolist() == ['007','NA','3','4','5','6','7']
        assert pd.isna(frame.cohort_niche_id.iloc[-1])
        assert record['model'] == model
        assert record['model']['reference_assignment'] is False
        assert json.loads(json.dumps(record, allow_nan=False)) == record
        verify_cohort_sources(output, source, record)
    # A copied single-source profile+shared bundle needs no other source dirs.
    copy = tmp_path/'portable'; copy.mkdir()
    shutil.copytree(profiles[0], copy/'profile')
    shutil.copytree(output, copy/'cohort')
    first, _ = load_cohort_bundle(copy/'cohort', profile_dir=copy/'profile', sample_id='A')
    pd.testing.assert_frame_equal(first, expected.iloc[:7][KEYS+COHORT_COLUMNS])
    assert before == {p:sha256_file(p) for p in before}


@pytest.mark.parametrize('artifact', [MODEL, SUMMARY, ASSIGNMENTS, COMPLETION])
def test_corrupt_or_incomplete_bundle_fails(bundle, artifact):
    profiles, output, *_ = bundle
    (output/artifact).unlink()
    with pytest.raises(ValueError, match='Missing'):
        load_cohort_bundle(output, profile_dir=profiles[0])


@pytest.mark.parametrize('artifact', ['cell_profiles_manifest.json','cell_profiles.parquet',
    'feature_rows.csv','neighborhood_summary.json','neighborhood_graph_3um.npz','feature_blocks/synthetic.npy'])
def test_all_selected_source_bytes_are_bound_and_rechecked(bundle, artifact):
    profiles, output, *_ = bundle
    _, record = load_cohort_bundle(output, profile_dir=profiles[0])
    with (profiles[0]/artifact).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        verify_cohort_sources(output, profiles[0], record)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_cohort_bundle(output, profile_dir=profiles[0])


@pytest.mark.parametrize('fault', ['reorder','foreign_uid','foreign_sample','numeric_cell_id',
    'duplicate_uid','source_row','coordinate','support','status','missing_id','invalid_id',
    'score','unassigned_score','source_label','summary_count','summary_status','eligible_count',
    'reference_claim','model_id','centroid_shape','feature_axes','fixed_k','feature_definition',
    'aggregation_definition'])
def test_rehashed_semantic_contradictions_fail(bundle, fault):
    profiles, output, assignments, model, summary = bundle
    if fault == 'reorder': assignments = assignments.iloc[::-1].reset_index(drop=True)
    elif fault == 'foreign_uid': assignments.loc[0,'cell_uid'] = 'unrelated'
    elif fault == 'foreign_sample': assignments.loc[0,'sample_id'] = 'C'
    elif fault == 'numeric_cell_id': assignments['cell_id'] = np.arange(len(assignments))
    elif fault == 'duplicate_uid': assignments.loc[7,'cell_uid'] = assignments.loc[0,'cell_uid']
    elif fault == 'source_row': assignments.loc[0,'source_row_index'] = 2
    elif fault == 'coordinate': assignments.loc[0,'x_um'] += .01
    elif fault == 'support': assignments.loc[0,'in_tissue_support'] = False
    elif fault == 'status': assignments.loc[0,'cohort_niche_status'] = 'unknown_unversioned'
    elif fault == 'missing_id': assignments.loc[0,'cohort_niche_id'] = pd.NA
    elif fault == 'invalid_id': assignments.loc[0,'cohort_niche_id'] = 3
    elif fault == 'score': assignments.loc[0,'cohort_niche_stability'] = 2.
    elif fault == 'unassigned_score': assignments.loc[6,'cohort_niche_centroid_margin'] = .8
    elif fault == 'source_label': assignments.loc[0,'source_niche_id'] = 999
    elif fault == 'summary_count': summary['cell_count'] = 15
    elif fault == 'summary_status': summary['status_counts'] = {'assigned_fixed_k':14}
    elif fault == 'eligible_count': model['discovery']['eligible_cells'] = 14
    elif fault == 'reference_claim': model['reference_assignment'] = True
    elif fault == 'centroid_shape': model['discovery']['centroids_scaled'].pop()
    elif fault == 'feature_axes':
        next(iter(model['discovery']['scaling'].values()))['matrix_columns'][0]['kind'] = 'unversioned_transform'
    elif fault == 'fixed_k': model['discovery']['fixed_k'] = 3
    elif fault == 'feature_definition': model['source_feature_definitions']['synthetic']['dimensions'] = 2
    elif fault == 'aggregation_definition': model['feature_group_definitions_by_specimen']['A']['density']['role'] = 'different'
    reseal(output, model=model, summary=summary, assignments=assignments)
    if fault == 'model_id':
        data = json.loads((output/MODEL).read_text())
        data['cohort_niche_model_id'] = '0'*64
        (output/MODEL).write_text(json.dumps(data))
        receipt = json.loads((output/COMPLETION).read_text())
        receipt['files'][MODEL] = sha256_file(output/MODEL)
        (output/COMPLETION).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_cohort_bundle(output, profile_dir=profiles[0])


@pytest.mark.parametrize('fault', ['duplicate_json','overflow','escape','symlink'])
def test_strict_json_and_containment(bundle, tmp_path, fault):
    profiles, output, *_ = bundle
    if fault == 'duplicate_json':
        text = (output/COMPLETION).read_text()
        (output/COMPLETION).write_text(text[:-1] + ',"format":"cellphenotyper_cohort_niches"}')
    elif fault == 'overflow':
        text = (output/COMPLETION).read_text()
        (output/COMPLETION).write_text(text[:-1] + ',"overflow":1e999}')
    elif fault == 'escape':
        data = json.loads((output/COMPLETION).read_text())
        data['files']['../cohort_niche_model.json'] = data['files'].pop(MODEL)
        (output/COMPLETION).write_text(json.dumps(data))
    else:
        target = tmp_path/'outside_model.json'
        shutil.copyfile(output/MODEL, target)
        (output/MODEL).unlink(); (output/MODEL).symlink_to(target)
    with pytest.raises(ValueError):
        load_cohort_bundle(output, profile_dir=profiles[0])


@pytest.mark.parametrize('fixed_k', [None, 1])
def test_single_niche_scores_remain_missing(tmp_path, fixed_k):
    profiles = [make_spatial(tmp_path/name,name,with_features=False) for name in ('A','B')]
    # Explicit empty feature axes model a supported population without any
    # selected informative variables. Even synthetic density is not constant
    # near tissue/image edges, so omitting embeddings alone does not force K=1.
    for source in profiles:
        path = source/'neighborhood_summary.json'
        metadata = json.loads(path.read_text())
        metadata['feature_groups'] = {'density': []}
        metadata['feature_group_definitions'] = {'density': {
            **metadata['feature_group_definitions']['density'], 'columns': []}}
        path.write_text(json.dumps(metadata))
        rewrite_manifest(source, lambda m: m['files'].update({path.name:sha256_file(path)}))
    out = tmp_path/'out'
    fit_cohort(profiles,out,fixed_k=fixed_k,repeats=2)
    frame,_ = load_cohort_bundle(out, profile_dir=profiles[0])
    assert frame.cohort_niche_id.iloc[:6].eq(1).all()
    assert frame.cohort_niche_stability.isna().all()
    assert frame.cohort_niche_centroid_margin.isna().all()


def test_batched_assignment_validation_and_cross_batch_duplicate(bundle, monkeypatch):
    import cohort_niche_io
    profiles, output, assignments, _, _ = bundle
    monkeypatch.setattr(cohort_niche_io, 'ASSIGNMENT_BATCH_ROWS', 3)
    frame,_ = load_cohort_bundle(output, profile_dir=profiles[1])
    pd.testing.assert_frame_equal(frame, assignments.iloc[7:][KEYS+COHORT_COLUMNS].reset_index(drop=True))
    assignments.loc[7, 'cell_uid'] = assignments.loc[0, 'cell_uid']
    reseal(output, assignments=assignments)
    with pytest.raises(ValueError, match='across batches'):
        load_cohort_bundle(output, profile_dir=profiles[1])


def test_no_supported_neighbourhoods_retains_all_unassigned(tmp_path):
    import tifffile
    from assemble_spatial_cell_profiles import assemble
    profiles = []
    for name in ('A','B'):
        parent = tmp_path/name
        make_spatial(parent,name,with_features=False)
        tifffile.imwrite(parent/'zero.tif',np.zeros((12,40),np.uint8))
        source = parent/'outside'
        assemble(parent/'base',parent/'zero.tif',parent/'shift.json',parent/'resolution.json',
                 source,radii_um=(3.,),feature_groups=(),repeats=2)
        profiles.append(source)
    output = tmp_path/'out'
    fit_cohort(profiles,output,repeats=2)
    frame, record = load_cohort_bundle(output,profile_dir=profiles[0])
    assert record['summary']['selected_k'] == 0
    assert len(frame) == 7 and frame.cohort_niche_id.isna().all()
    assert frame.cohort_niche_status.eq('outside_tissue_support').all()


def test_wrong_requested_sample_rejected(bundle):
    profiles, output, *_ = bundle
    with pytest.raises(ValueError, match='specimen'):
        load_cohort_bundle(output, profile_dir=profiles[0], sample_id='B')


def test_reading_has_no_fitting_torch_or_sklearn_import(bundle):
    profiles, output, *_ = bundle
    code = """
import sys
sys.path.insert(0, sys.argv[1])
class RejectML:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'sklearn','torch','fit_cohort_niches','analyze_cell_neighborhoods'}:
            raise RuntimeError('Consumer imported a model/fitting runtime: '+name)
sys.meta_path.insert(0, RejectML())
from cohort_niche_io import load_cohort_bundle
frame, record = load_cohort_bundle(sys.argv[2], profile_dir=sys.argv[3])
assert len(frame) == 7
"""
    result = subprocess.run([sys.executable,'-c',code,str(Path(__file__).resolve().parents[1]/'bin'),str(output),str(profiles[0])],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout+result.stderr
