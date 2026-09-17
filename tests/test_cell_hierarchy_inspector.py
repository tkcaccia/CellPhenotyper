"""The inspector consumes actual linker outputs, never hand-authored membership."""
import json
import os
from pathlib import Path
import subprocess
import threading
import numpy as np
import pandas as pd
import pytest
import tifffile

pytest.importorskip('spatialdata')
from test_spatialdata_hierarchy import hierarchy_specimen, refresh_hierarchy
from test_spatialdata_export import specimen
from link_cell_tissue_hierarchy import link_profiles
from cell_profile_io import sha256_file
from cell_inspector import CellInspector
import cell_inspector as module


def linked_inspector(hierarchy_specimen, tmp_path):
    inputs = hierarchy_specimen[0]
    manifest_path = inputs['profile_dir'] / 'cell_profiles_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['biological_marker_features'] = ['predicted__nucleus__CD3__mean']
    manifest_path.write_text(json.dumps(manifest))
    linked = tmp_path / 'linked'
    link_profiles(inputs['profile_dir'], inputs['labels'], inputs['hierarchy_dir'], linked, tile_size=16)
    kwargs = {name: inputs[name] for name in ('image', 'labels', 'shift', 'resolution_json', 'hierarchy_dir')}
    return CellInspector(profile_dir=linked, **kwargs), linked


def test_inspector_displays_fractional_membership_and_cross_boundary_members(hierarchy_specimen, tmp_path):
    inspector, linked = linked_inspector(hierarchy_specimen, tmp_path)
    relations = pd.read_parquet(linked / 'cell_hierarchy_overlaps.parquet')
    uid = inspector.cells.cell_uid.iloc[0]
    detail = inspector.detail(uid)
    expected = relations[relations.cell_uid == uid]
    assert detail['tissue_membership_overlap_count'] == len(expected)
    assert not detail['tissue_membership_overlaps_truncated']
    assert len(detail['tissue_membership_overlaps']) == len(expected)
    assert any(name.startswith('hierarchy_nucleus_') for name in detail['spatial'])
    assert np.isclose(sum(row['overlap_fraction'] for row in detail['tissue_membership_overlaps']), 1)
    crossing = expected[(expected.region_id > 0) & (expected.overlap_fraction < 1)]
    assert len(crossing) >= 2
    for row in crossing.itertuples():
        region = inspector.regions.detail(row.region_uid)
        assert 'exact nuclear pixel overlap' in region['cell_membership']
        member = next(cell for cell in region['representative_cells'] if cell['cell_uid'] == uid)
        assert member['nuclear_overlap_fraction'] == row.overlap_fraction
        assert region['fractional_nuclear_cell_count'] > 0
        marker = region['predicted_marker_distributions']['predicted__nucleus__CD3__mean']
        assert marker['mean_weighting'] == 'whole-nucleus overlap fraction'
        assert marker['mean'] == pytest.approx(.1)


@pytest.mark.parametrize('fractional', [True, False])
def test_region_similarity_reports_the_actual_marker_weighting_and_missingness(hierarchy_specimen, tmp_path, fractional):
    inputs = hierarchy_specimen[0]
    root = inputs['profile_dir']
    primary, partial = 'predicted__nucleus__CD3__mean', 'predicted__nucleus__CD8__mean'
    # Retain the first fixture nucleus crossing regions 1/2. Move the second
    # nucleus to cross regions 2/3: its overlap with region 2 is 0.6, whereas the
    # first nucleus contributes 0.5. Both links are derived by the real linker.
    labels = tifffile.imread(inputs['labels'])
    labels[labels == 7] = 0
    labels[24:32, 36:56] = 7
    tifffile.imwrite(inputs['labels'], labels, tile=(16, 16), compression='deflate')
    cells = pd.read_parquet(root / 'cell_profiles.parquet')
    cells.loc[1, ['x_crop_px', 'y_crop_px', 'x_um', 'y_um']] = [46., 28., 73., 114.]
    cells[primary], cells[partial] = [.1, .9], [.2, np.nan]
    cells.to_parquet(root / 'cell_profiles.parquet', index=False)
    cells.to_csv(root / 'cell_profiles.csv', index=False)
    manifest_path = root / 'cell_profiles_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['inputs']['labels_sha256'] = sha256_file(inputs['labels'])
    manifest['biological_marker_features'] = [primary, partial]
    for name in ('cell_profiles.parquet', 'cell_profiles.csv'):
        manifest['files'][name] = sha256_file(root / name)
    manifest_path.write_text(json.dumps(manifest))
    if fractional:
        linked = tmp_path / 'similarity_linked'
        link_profiles(root, inputs['labels'], inputs['hierarchy_dir'], linked, tile_size=16)
        root = linked
    kwargs = {name: inputs[name] for name in ('image', 'labels', 'shift', 'resolution_json', 'hierarchy_dir')}
    inspector = CellInspector(profile_dir=root, **kwargs)
    regions = inspector.regions
    uid = {int(row.region_id): row.region_uid for row in regions.cells.itertuples()}
    expected_mean = (.1 * .5 + .9 * .6) / (.5 + .6) if fractional else .9
    expected_weighting = 'whole-nucleus overlap fraction' if fractional else 'equal centroid-member cells'
    assert regions.cells.loc[regions.cells.region_id.astype(int) == 2, primary].iloc[0] == pytest.approx(expected_mean)
    if fractional:
        assert sorted(regions.member_weights[2]) == pytest.approx([.5, .6])
        assert expected_mean != pytest.approx(.5)  # An unweighted overlap mean is different.
        assert regions.cells.loc[regions.cells.region_id.astype(int) == 2, partial].iloc[0] == pytest.approx(.2)
    result = regions.similar(uid[1], ['local'], marker=primary, min_difference=.01, k=10)
    hit = next(row for row in result['results'] if row['region_uid'] == uid[2])
    assert hit['predicted_marker_difference'] == pytest.approx(expected_mean - .1)
    assert result['marker_mean_weighting'] == expected_weighting
    assert result['cell_membership'] == regions.membership_method
    assert 'missing values remain unknown' in result['marker_semantics']
    assert 'never zero-filled' in result['marker_semantics']
    assert 'not measured markers' in result['marker_semantics']
    if fractional:
        assert 'overlap-fraction-weighted' in result['marker_semantics']
        assert 'centroid-member' not in result['marker_semantics']
    else:
        assert 'for centroid-member cells' in result['marker_semantics']
    assert regions.detail(uid[2])['predicted_marker_distributions'][primary]['mean_weighting'] == expected_weighting
    # Region 3 has no finite CD8 values (and in centroid mode no member at all).
    assert np.isnan(regions.cells.loc[regions.cells.region_id.astype(int) == 3, partial].iloc[0])
    missing = regions.similar(uid[3], ['local'], marker=partial, min_difference=0)
    assert missing['status'] == 'missing_marker' and missing['results'] == []
    assert missing['marker_mean_weighting'] == expected_weighting
    assert 'missing values remain unknown' in missing['marker_semantics']


def test_linked_inspector_cannot_silently_omit_hierarchy(hierarchy_specimen, tmp_path):
    inspector, linked = linked_inspector(hierarchy_specimen, tmp_path)
    inputs = hierarchy_specimen[0]
    kwargs = {name: inputs[name] for name in ('image', 'labels', 'shift', 'resolution_json')}
    with pytest.raises(ValueError, match='require their verified hierarchy'):
        CellInspector(profile_dir=linked, **kwargs)


def test_parent_zero_with_raw_253_remains_unresolved_when_picked(hierarchy_specimen, tmp_path):
    inputs, maps, _, _ = hierarchy_specimen
    root = inputs['hierarchy_dir']
    raw_path = root / 'parent_uncertainty.ome.tif'
    raw = maps['parent_uncertainty.ome.tif'].copy()
    assert maps['parent_domains.ome.tif'][8, 10] == 0
    assert maps['hierarchy_status.ome.tif'][8, 10] == 0
    raw[8, 10] = 253
    tifffile.imwrite(raw_path, raw, tile=(16, 16), compression='deflate')
    profile_path = inputs['profile_dir'] / 'cell_profiles_manifest.json'
    profile = json.loads(profile_path.read_text())
    profile['inputs']['domain_uncertainty_sha256'] = sha256_file(raw_path)
    profile_path.write_text(json.dumps(profile))
    summary_path = root / 'hierarchy_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['inputs']['parent_uncertainty']['sha256'] = sha256_file(raw_path)
    summary_path.write_text(json.dumps(summary))
    refresh_hierarchy(root)
    before = {path.name: sha256_file(path) for path in root.glob('*.tif')}
    inspector, _ = linked_inspector(hierarchy_specimen, tmp_path)
    picked = inspector.regions.pick(10, 8)
    assert picked['status'] == 'unassigned_parent_uncertain_tissue'
    assert picked['parent_domain_id'] == 0
    assert picked['hierarchy_status_code'] == 0
    assert picked['parent_uncertainty_code'] == 253
    assert picked['parent_uncertainty_available']
    assert 'not background' in picked['reason']
    background = inspector.regions.pick(11, 8)
    assert background['status'] == 'parent_map_background'
    assert background['parent_uncertainty_code'] == 0
    assert 'not independent proof' in background['reason']
    assert {path.name: sha256_file(path) for path in root.glob('*.tif')} == before


def test_parent_zero_with_absent_uncertainty_is_not_confident_background(hierarchy_specimen, tmp_path):
    inputs, maps, _, _ = hierarchy_specimen
    root = inputs['hierarchy_dir']
    status = maps['hierarchy_status.ome.tif'].copy()
    status[status == 10] = 2
    tifffile.imwrite(root / 'hierarchy_status.ome.tif', status, tile=(16, 16), compression='deflate')
    summary_path = root / 'hierarchy_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['inputs'].pop('parent_uncertainty')
    summary.pop('parent_uncertainty_semantics')
    summary_path.write_text(json.dumps(summary))
    refresh_hierarchy(root)
    profile_path = inputs['profile_dir'] / 'cell_profiles_manifest.json'
    profile = json.loads(profile_path.read_text())
    profile['inputs'].pop('domain_uncertainty_sha256')
    profile_path.write_text(json.dumps(profile))
    inspector, _ = linked_inspector(hierarchy_specimen, tmp_path)
    picked = inspector.regions.pick(10, 8)
    assert picked['status'] == 'no_parent_assignment_uncertainty_unavailable'
    assert picked['parent_domain_id'] == 0 and picked['hierarchy_status_code'] == 0
    assert picked['parent_uncertainty_code'] is None
    assert not picked['parent_uncertainty_available']
    assert 'background is not established' in picked['reason']


def test_membership_wording_does_not_claim_centroid_only_assignment():
    html = module.HTML_PATH.read_text()
    assert 'No canonical cell is assigned by the reported region-membership method' in html
    assert 'including crossing nuclei when exact overlaps are available' in html
    assert 'membership and mean/quantile weighting are reported' in html
    assert 'No canonical cell centroid lies inside this accepted region' not in html
    assert 'Centroid membership; finite values only' not in html


@pytest.mark.skipif(os.environ.get('RUN_INSPECTOR_BROWSER_TESTS') != '1', reason='Opt-in headless browser QA')
def test_fractional_membership_is_visible_in_actual_browser(hierarchy_specimen, tmp_path, monkeypatch):
    chrome = os.environ.get('CHROME_BIN', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    if not Path(chrome).is_file():
        pytest.skip('Chrome unavailable')
    inspector, _ = linked_inspector(hierarchy_specimen, tmp_path)
    page = tmp_path / 'test.html'
    probe = '''<script>
    async function checkMembership(){
      try {
        for(let i=0;i<500&&!window.current;i++)await new Promise(r=>setTimeout(r,20));
        const data=await api('/api/metadata'); await selectCell(data.first_cell_uid);
        const displayed=document.getElementById('tissue-overlaps').textContent;
        if(!displayed.includes('overlap_fraction')||!displayed.includes('nucleus'))throw Error('Missing exact overlaps');
        if(!document.getElementById('spatial').textContent.includes('hierarchy_nucleus_'))throw Error('Missing summary');
        document.body.dataset.qa='passed';
      }catch(error){document.body.dataset.qa='failed';document.body.dataset.error=String(error);}
    }window.addEventListener('load',checkMembership);
    </script>'''
    page.write_text(module.HTML_PATH.read_text().replace('</body>', probe + '</body>'))
    monkeypatch.setattr(module, 'HTML_PATH', page)
    server = module.make_server(inspector, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [chrome, '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
                   '--user-data-dir=' + str(tmp_path / 'chrome'), '--virtual-time-budget=18000', '--dump-dom',
                   f'http://127.0.0.1:{server.server_port}']
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            rendered = result.stdout
            assert result.returncode == 0, result.stderr[-1500:]
        except subprocess.TimeoutExpired as exc:
            rendered = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        assert 'data-qa="passed"' in rendered, rendered[-3500:]
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)
