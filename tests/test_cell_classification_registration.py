"""Synthetic interchange fixtures only; no segmentation/classifier inference."""
import builtins
import copy
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from export_experiment import (COORDINATE_CONVENTION, ExperimentError, FILENAME,
                               discover_run, export_experiment, export_runs, image_shape,
                               sha256, try_export_run, validate_experiment)
from register_cell_classification import register_cell_classification
from test_experiment_export import cached_tissue, run_tree
from test_gigatime_experiment import gigatime_tissue
from test_spatial_experiment import spatial_tissue


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


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


def project_fixture(root, *, empty=False):
    target,_,_=gigatime_tissue(root/'assets')
    spatial,_,_=spatial_tissue(root/'assets','rna-section')
    tile=cached_tissue(root/'assets','tile-section')
    classification,manifest=classification_fixture(root/'cell-cache',target,empty=empty)
    source=root/'source'/FILENAME
    export_experiment(dict(experiment_id='spatial-study',name='Synthetic integrated project',settings=dict(retain=True),
        provenance=dict(synthetic=True),extension_metadata=dict(keep='unchanged'),tissues=[target,spatial,tile]),source)
    return source,classification,manifest


def test_new_registration_roundtrip_alias_relative_paths_preserves_all_existing_assets(tmp_path,monkeypatch):
    source,classification,manifest=project_fixture(tmp_path)
    before=source.read_bytes()
    assets={str(p):sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    old=validate_experiment(source)
    original_import=builtins.__import__
    def no_models(name,*args,**kwargs):
        assert name.split('.')[0] not in {'torch','timm','nextflow','rpy2','pyreadr','pyvips'}
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_models)
    output=tmp_path/'project'/'nested'/FILENAME
    payload=register_cell_classification(source,output,tissue_id='section-a',classification=classification)
    assert validate_experiment(output)==payload
    assert source.read_bytes()==before
    assert payload['settings']==old['settings'] and payload['extension_metadata']==old['extension_metadata']
    tissue=payload['tissues'][0]
    assert tissue['image']['frame_id']=='section-a:image'
    reference=tissue['cell_classification']
    assert reference['registration']['source_frame_id']=='producer-native-image'
    assert reference['registration']['target_frame_id']=='section-a:image'
    assert reference['registration']['source_to_image_affine']==[[1,0,0],[0,1,0],[0,0,1]]
    assert reference['cell_count']==2 and reference['review_status']=='available'
    assert (output.parent/reference['path']).resolve()==classification.resolve()
    assert (output.parent/reference['feature_manifest']['path']).resolve()==manifest.resolve()
    feature=tissue['feature_sets'][0]
    raw=json.loads(classification.read_text())
    assert (output.parent/feature['path']).resolve()==(classification.parent/raw['feature_set']['path']).resolve()
    assert feature['descriptor']['feature_count']==3
    assert feature['source_frame_id']=='producer-native-image' and feature['frame_id']=='section-a:image'
    assert feature['registration']==reference['registration']
    assert tissue['artifacts']==old['tissues'][0]['artifacts'] or [a['sha256'] for a in tissue['artifacts']]==[a['sha256'] for a in old['tissues'][0]['artifacts']]
    assert payload['tissues'][1]['artifacts'][0]['sha256']==old['tissues'][1]['artifacts'][0]['sha256']
    assert payload['tissues'][2]['feature_sets'][0]['sha256']==old['tissues'][2]['feature_sets'][0]['sha256']
    assert all(sha256(Path(path))==value for path,value in assets.items())
    # Re-registration and repeated index export retain exactly one typed feature.
    repeated=tmp_path/'again'/FILENAME
    again=register_cell_classification(output,repeated,tissue_id='section-a',classification=classification)
    assert len(again['tissues'][0]['feature_sets'])==1
    assert validate_experiment(repeated)==again
    relocated=tmp_path.parent/(tmp_path.name+'-relocated')
    shutil.copytree(tmp_path,relocated)
    assert validate_experiment(relocated/'project'/'nested'/FILENAME)==payload


def test_new_spec_adopts_classification_frame_and_empty_review_is_explicit(tmp_path):
    target,_,_=gigatime_tissue(tmp_path/'assets')
    target['image'].pop('frame_id')
    # Marker references already have a different frame, so use an image-only new spec here.
    target['artifacts']=[]
    classification,manifest=classification_fixture(tmp_path/'cache',target,empty=True,width=5)
    target['cell_classification']=dict(path=str(classification),feature_manifest=dict(path=str(manifest)))
    output=tmp_path/'project'/FILENAME
    payload=export_experiment(dict(tissues=[target]),output)
    assert validate_experiment(output)==payload
    tissue=payload['tissues'][0]
    assert tissue['image']['frame_id']=='producer-native-image'
    assert tissue['cell_classification']['review_status']=='empty'
    assert tissue['cell_classification']['cell_count']==0
    assert tissue['feature_sets'][0]['row_count']==0


@pytest.mark.parametrize('failure',['image_hash','mpp','origin','space','convention','missing_origin','nested_frame','row_count','descriptor_width','label_metadata','duplicate_cell','coordinate_join','missing_model','wrong_label','probability','vector_sum','manifest_metadata','nonfinite_geometry','score_semantics','unsupported_backbone','missing_wheel','missing_normalization','missing_resampling','missing_token_provenance','missing_model_sampling'])
def test_incompatible_classification_never_creates_or_changes_projects(tmp_path,failure):
    source,classification,manifest=project_fixture(tmp_path)
    before=source.read_bytes()
    payload=json.loads(classification.read_text())
    if failure=='image_hash':payload['image']['sha256']='0'*64
    elif failure=='mpp':payload['image']['mpp_x']=.25
    elif failure=='origin':payload['image']['origin_x']=1
    elif failure=='space':payload['image']['coordinate_space']='global-pixels'
    elif failure=='convention':payload['image']['coordinate_convention']='zero-based-index'
    elif failure=='missing_origin':del payload['image']['origin_x']
    elif failure=='nested_frame':payload['feature_set']['frame_id']='another-frame'
    elif failure=='row_count':payload['feature_set']['row_count']=1
    elif failure=='descriptor_width':payload['feature_set']['descriptor']['feature_count']=4
    elif failure=='label_metadata':payload['feature_set']['feature_columns'][0]='label'
    elif failure=='duplicate_cell':payload['cells'][1]['cell_id']=payload['cells'][0]['cell_id']
    elif failure=='coordinate_join':payload['cells'][0]['x']+=1
    elif failure=='missing_model':payload['cells'][0]['predictions']={}
    elif failure=='wrong_label':payload['cells'][0]['predictions']['synthetic-model']['label']='Other'
    elif failure=='probability':payload['cells'][0]['predictions']['synthetic-model']['probability']=1.2
    elif failure=='vector_sum':payload['cells'][0]['predictions']['synthetic-model']['probabilities']=[.25,.25]
    elif failure=='nonfinite_geometry':payload['cells'][0]['contour'][0][0]=float('nan')
    elif failure=='score_semantics':payload['models'][0]['score_semantics']='uncalibrated_virtual_marker_score'
    elif failure=='unsupported_backbone':payload['feature_set']['descriptor']['backbone']='unsupported-model';payload['models'][0]['backbone']='unsupported-model'
    elif failure=='missing_wheel':del payload['feature_set']['descriptor']['preprocessing']['runtime_wheel_sha256']
    elif failure=='missing_normalization':del payload['feature_set']['descriptor']['preprocessing']['normalization']
    elif failure=='missing_resampling':del payload['feature_set']['descriptor']['preprocessing']['resampling']
    elif failure=='missing_token_provenance':del payload['feature_set']['descriptor']['preprocessing']['token_provenance']
    elif failure=='missing_model_sampling':del payload['feature_set']['descriptor']['sampling']['model_input_mpp']
    elif failure=='manifest_metadata':
        data=json.loads(manifest.read_text());data['feature_set']['descriptor']['sampling']['model_input_mpp']=.5;write_json(manifest,data)
    write_json(classification,payload)
    output=tmp_path/'new'/FILENAME
    with pytest.raises(ExperimentError):
        register_cell_classification(source,output,tissue_id='section-a',classification=classification)
    assert source.read_bytes()==before and not output.exists()


@pytest.mark.parametrize('failure',['classification_hash','feature_hash','mapping','missing_mapping','registered_signature','registered_duplicate'])
def test_reopen_rejects_tampered_reference_or_mapping(tmp_path,failure):
    source,classification,_=project_fixture(tmp_path)
    output=tmp_path/'project'/FILENAME
    payload=register_cell_classification(source,output,tissue_id='section-a',classification=classification)
    tissue=payload['tissues'][0]
    if failure=='classification_hash':tissue['cell_classification']['sha256']='0'*64
    elif failure=='feature_hash':tissue['feature_sets'][0]['sha256']='0'*64
    elif failure=='mapping':tissue['cell_classification']['registration']['source_to_image_affine'][0][2]=1
    elif failure=='missing_mapping':del tissue['cell_classification']['registration']
    elif failure=='registered_signature':tissue['feature_sets'][0]['descriptor']['sampling']['model_input_mpp']=.5
    elif failure=='registered_duplicate':tissue['feature_sets'].append(copy.deepcopy(tissue['feature_sets'][0]))
    write_json(output,payload)
    with pytest.raises(ExperimentError):validate_experiment(output)


def test_cli_defaults_manifest_and_refuses_existing_destination(tmp_path):
    source,classification,_=project_fixture(tmp_path)
    output=tmp_path/'ready'/FILENAME
    command=[sys.executable,str(ROOT/'bin/register_cell_classification.py'),'--experiment',str(source),'--output',str(output),
             '--tissue-id','section-a','--classification',str(classification)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert 'Ready to open in wsiTools' in result.stdout and '(2 cells)' in result.stdout
    before=output.read_bytes()
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and 'NEW filename' in result.stderr and 'Traceback' not in result.stderr
    assert output.read_bytes()==before


def test_odd_width_white_padded_grid_uses_clipped_logical_representatives(tmp_path):
    run,embedding=run_tree(tmp_path/'results')
    objects=next((run/'09_grid_tiles').glob('**/*objects.csv'))
    objects.write_text('label,x,y\n1,112,112\n2,336,112\n3,449,449\n')
    with gzip.open(embedding,'rt') as stream:rows=list(csv.reader(stream))
    rows[3][1:3]=['449','449']
    with gzip.open(embedding,'wt',newline='') as stream:csv.writer(stream).writerows(rows)
    marker=embedding.parent/'VIRCHOW2_COMPLETE.json';data=json.loads(marker.read_text())
    data.update(grid_sha256=sha256(objects),embeddings_sha256=sha256(embedding));write_json(marker,data)
    tissue=discover_run(run)[0]
    grid=tissue['feature_sets'][0]['grid']
    assert grid['edge_policy']=='white-padded'
    assert grid['representative_policy']=='logical-centre-clipped-to-image'
    output=tmp_path/'edge-project'/FILENAME
    payload=export_experiment(dict(tissues=[tissue]),output)
    assert validate_experiment(output)==payload
    rows[3][1]='448'  # in bounds, but not the producer's declared logical representative.
    with gzip.open(embedding,'wt',newline='') as stream:csv.writer(stream).writerows(rows)
    tissue['feature_sets'][0].pop('sha256',None)
    with pytest.raises(ExperimentError,match='logical clipped grid'):
        export_experiment(dict(tissues=[tissue]),tmp_path/'wrong-edge.json')


def cell_run_fixture(root, sample='section', *, combined=False, folder='03c_cellvitpp'):
    if combined:
        run_tree(root,sample)
        image=next((root/'03_stardist'/sample).glob('**/crop_roi.tif'))
    else:
        from PIL import Image
        image=root/'03_stardist'/sample/'prepared_crop'/'crop_roi.tif'
        image.parent.mkdir(parents=True)
        Image.new('RGB',(64,48),(185,145,175)).save(image)
    shift_path=image.with_name('shift.json')
    shift=json.loads(shift_path.read_text()) if shift_path.exists() else {}
    shift.update(source_mpp=.5,microns_per_pixel=.5)
    write_json(shift_path,shift)
    classification,manifest=classification_fixture(root/folder/sample,dict(image=dict(path=str(image))))
    return image,classification,manifest


@pytest.mark.parametrize('folder',['03c_cellvitpp','03_cellvitpp'])
def test_cell_only_discovery_exports_automatic_open_project_without_tissue_embeddings(tmp_path,folder):
    run=tmp_path/'results';image,classification,manifest=cell_run_fixture(run,folder=folder)
    before={str(path):sha256(path) for path in run.rglob('*') if path.is_file()}
    status=try_export_run(run,name='Cell-only review')
    assert status['status']=='available' and status['tissue_count']==1
    project=run/'00_execution'/FILENAME
    payload=validate_experiment(project)
    tissue=payload['tissues'][0]
    assert tissue['name']=='section' and tissue['cell_classification']['cell_count']==2
    assert [feature['kind'] for feature in tissue['feature_sets']]==['cellvit-cell']
    assert (project.parent/tissue['image']['path']).resolve()==image.resolve()
    assert (project.parent/tissue['cell_classification']['path']).resolve()==classification.resolve()
    assert (project.parent/tissue['cell_classification']['feature_manifest']['path']).resolve()==manifest.resolve()
    assert all(sha256(Path(path))==value for path,value in before.items())


def test_combined_samples_union_and_same_sample_features_register_once(tmp_path):
    run=tmp_path/'results'
    cell_run_fixture(run,'paired',combined=True)
    cell_run_fixture(run,'cell-only')
    run_tree(run,'tile-only')
    output=run/'00_execution'/FILENAME
    payload=export_runs([run],output)
    assert validate_experiment(output)==payload
    tissues={tissue['name']:tissue for tissue in payload['tissues']}
    assert set(tissues)=={'paired','cell-only','tile-only'}
    assert [f['kind'] for f in tissues['paired']['feature_sets']]==['virchow2-pooled','cellvit-cell']
    assert [f['kind'] for f in tissues['cell-only']['feature_sets']]==['cellvit-cell']
    assert [f['kind'] for f in tissues['tile-only']['feature_sets']]==['virchow2-pooled']
    assert len({t['tissue_id'] for t in payload['tissues']})==3


@pytest.mark.parametrize('failure',['missing_crop','duplicate_crop','changed_image','dimensions','missing_calibration','wrong_calibration','conflicting_calibration','missing_manifest','duplicate_classification'])
def test_cell_discovery_fails_closed_on_missing_or_mismatched_durable_identity(tmp_path,failure):
    run=tmp_path/'results';image,classification,manifest=cell_run_fixture(run)
    if failure=='missing_crop':image.unlink()
    elif failure=='duplicate_crop':
        other=image.parent.parent/'ambiguous'/'crop_roi.tif';other.parent.mkdir();shutil.copy2(image,other)
    elif failure=='changed_image':
        from PIL import Image
        Image.new('RGB',(64,48),(1,2,3)).save(image)
    elif failure=='dimensions':
        from PIL import Image
        Image.new('RGB',(63,48),(185,145,175)).save(image)
    elif failure=='missing_calibration':write_json(image.with_name('shift.json'),{})
    elif failure=='wrong_calibration':write_json(image.with_name('shift.json'),dict(source_mpp=.25,microns_per_pixel=.25))
    elif failure=='conflicting_calibration':write_json(image.with_name('shift.json'),dict(source_mpp=.5,microns_per_pixel=.25))
    elif failure=='missing_manifest':manifest.unlink()
    elif failure=='duplicate_classification':
        target=run/'03_cellvitpp'/'section'/'another'/classification.name;target.parent.mkdir(parents=True);shutil.copy2(classification,target)
    with pytest.raises(ExperimentError):discover_run(run)
    status=try_export_run(run)
    assert status['status']=='error' and status['path'] is None
    assert not (run/'00_execution'/FILENAME).exists()
