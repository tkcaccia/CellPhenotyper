"""CPU handoff checks: semantic names preserve geometry, cell index and lineage."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from bora_contextual_fullslide import _asset_references, _validate_project, sha256
from combine_cellvit_bora_project import combine
from export_bora_region_identity_project import export_identity_project, _geometry
from test_bora_contextual_fullslide import _complete_cellvit_merge_case


def _write(path, value):
    path.write_text(json.dumps(value) + '\n')


def _case(root):
    base, cells, geo, smoothing = _complete_cellvit_merge_case(root / 'inputs')
    source = json.loads(geo.read_text())
    source['features'][0]['properties']['value'] = 1
    _write(geo, source)
    receipt = json.loads(smoothing.read_text())
    receipt['review_geojson_sha256'] = sha256(geo)
    _write(smoothing, receipt)
    accepted = root / 'accepted/experiment.cellphenotyper.json'
    combine(base, cells, geo, smoothing, accepted,
            expected_bora_sha256=sha256(geo), expected_shards=104)
    project = json.loads(accepted.read_text())
    image = project['tissues'][0]['image']
    identity = root / 'identity'
    identity.mkdir()
    predicted = copy.deepcopy(source)
    predicted['features'][0]['properties'].update(value=7, name='stroma',
        classification={'name': 'stroma'}, source_group_id=1,
        source_group_name='KODAMA Group 1')
    _write(identity / 'tissue-annotations.geojson', predicted)
    np.savez(identity / 'descriptors.npz', x=np.ones((1, 2560)))
    _write(identity / 'report.json', {
        'format': 'bora-region-identity-v1', 'complete': True,
        'geometry_unchanged': True, 'biological_accuracy_qualified': False,
        'responses_are_calibrated_probabilities': False,
        'source_frame_id': image['frame_id'], 'source_image_sha256': image['sha256'],
        'native_shape_yx': [image['height'], image['width']],
        'native_mpp_xy': [image['mpp_x'], image['mpp_y']],
        'input_sha256': {'contours': sha256(geo), 'tokens': 'a'*64,
                         'model': 'b'*64, 'registration': 'c'*64},
        'geometry_sha256': _geometry(source['features']),
        'output_sha256': {name: sha256(identity / name) for name in
                         ('tissue-annotations.geojson', 'descriptors.npz')}})
    return accepted, identity


def test_names_replace_groups_without_changing_contours_or_index(tmp_path):
    accepted, identity = _case(tmp_path)
    original = _validate_project(accepted)
    destination = tmp_path / 'named/experiment.cellphenotyper.json'
    result = export_identity_project(accepted, identity, destination)
    assert result['geometry_unchanged'] is True
    updated = _validate_project(destination)
    before, after = original['tissues'][0], updated['tissues'][0]
    assert before['cell_classification']['index']['sha256'] == after['cell_classification']['index']['sha256']
    assert after['cell_classification']['cell_count'] == before['cell_classification']['cell_count']
    artifact = after['artifacts'][0]
    assert artifact['role'] == 'tissue_annotations' and artifact['review_only'] is False
    geo = json.loads((destination.parent / artifact['path']).read_text())
    assert geo['features'][0]['properties']['classification']['name'] == 'stroma'
    assert geo['features'][0]['properties']['source_group_id'] == 1
    assert artifact['provenance']['accepted_geometry']['sha256'] == before['artifacts'][0]['sha256']
    for ref in _asset_references(updated):
        assert (destination.parent / ref['path']).is_file()
    assert _validate_project(accepted) == original
    with pytest.raises(FileExistsError):
        export_identity_project(accepted, identity, destination)


@pytest.mark.parametrize('failure', ['geometry', 'frame', 'source', 'hash', 'group'])
def test_invalid_identity_never_publishes_project(tmp_path, failure):
    accepted, identity = _case(tmp_path)
    report_path = identity / 'report.json'
    report = json.loads(report_path.read_text())
    if failure in ('geometry', 'group'):
        geo_path = identity / 'tissue-annotations.geojson'
        geo = json.loads(geo_path.read_text())
        if failure == 'geometry':
            geo['features'][0]['geometry']['coordinates'][0][0][0] += .5
        else:
            geo['features'][0]['properties']['source_group_id'] = 2
        _write(geo_path, geo)
        report['output_sha256'][geo_path.name] = sha256(geo_path)
    elif failure == 'frame':
        report['source_frame_id'] = 'other-slide'
    elif failure == 'source':
        report['input_sha256']['contours'] = 'd'*64
    else:
        (identity / 'descriptors.npz').write_bytes(b'corrupt')
    _write(report_path, report)
    output = tmp_path / 'named/experiment.cellphenotyper.json'
    with pytest.raises(ValueError):
        export_identity_project(accepted, identity, output)
    assert not output.parent.exists()
