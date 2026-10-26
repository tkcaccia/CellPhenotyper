"""CPU-only checks for CellViT full-slide partition planning and checkpoints."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from cellvit_shards import (ARCHIVE_NAME, SCHEMA, SHARD_FILES, compact_shard,
                            CLASSIFICATION_HEADER,
                            complete_shard, core_cells, materialize_shard_records,
                            NATIVE_PATCH_STRIDE, MAX_INFERENCE_INPUT_PIXELS,
                            finalize_shards, get_shard, plan_shards, release_verified_input_crops,
                            plan_shards_geometry, rounded_mpp_matches, validate_plan,
                            validate_input_budget, verify_complete_shard)
from cellvit_models import digest_file, require_bounded_native_outputs


def test_independently_rounded_converted_axes_are_accepted_only_within_half_pixel():
    # PM016 0.25 um/px conversion dimensions round independently; the actual
    # calibrated X/Y scales must be preserved, not replaced by nominal 0.25.
    assert rounded_mpp_matches(0.250001518, 0.25, 23904)
    assert rounded_mpp_matches(0.250000866, 0.25, 41800)
    assert not rounded_mpp_matches(0.250010, 0.25, 23904)
    from run_cellvitpp import source_mpp
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        shift = Path(directory) / 'shift.json'
        shift.write_text(json.dumps({'source_mpp': 0.25, 'mpp_x': 0.250001518,
            'mpp_y': 0.250000866, 'crop_size': {'width': 23904, 'height': 41800}}))
        assert source_mpp(shift, 0) == 0.25


def test_half_open_cores_cover_each_coordinate_once():
    width, height, side, halo = 2301, 1703, 1024, 1024
    shards = plan_shards_geometry(width, height, side, halo)
    assert len(shards) == 6
    assert all(s['input'][0] % NATIVE_PATCH_STRIDE == 0 and
               s['input'][1] % NATIVE_PATCH_STRIDE == 0 for s in shards)
    for y in (0, 1023, 1024, height - 1):
        for x in (0, 1023, 1024, 2047, 2048, width - 1):
            owners = [s for s in shards if s['core'][0] <= x < s['core'][2]
                      and s['core'][1] <= y < s['core'][3]]
            assert len(owners) == 1
            owner = owners[0]
            assert owner['input'][0] <= x < owner['input'][2]
            assert owner['input'][1] <= y < owner['input'][3]
    left = shards[0]
    assert core_cells({'cells': [{'id': '7', 'cell_id': 'local-7',
                                  'centroid': [1023.9, 4.0]}]}, left, width, height)[0]['x'] == 1023.9
    with pytest.raises(ValueError, match='halo cell'):
        core_cells({'cells': [{'id': '8', 'cell_id': 'local-8',
                              'centroid': [1024.0, 4.0]}]}, left, width, height)
    with pytest.raises(ValueError, match='duplicated'):
        core_cells({'cells': [
            {'id': '1', 'cell_id': 'same', 'centroid': [2.0, 2.0]},
            {'id': '2', 'cell_id': 'same', 'centroid': [3.0, 3.0]}],
        }, left, width, height)


def test_dense_full_slide_raw_input_budget_precedes_gpu_admission():
    # Real PM016 converted dimensions; its old r0001c0002 input held an
    # estimated 41k-55k H&E nuclear peaks versus the 40k raw CellViT cap.
    width, height = 23904, 14762
    safe = plan_shards_geometry(width, height, 1920, 960)
    assert len(safe) == 104
    assert max((s['input'][2]-s['input'][0]) * (s['input'][3]-s['input'][1])
               for s in safe) == 14_745_600
    assert MAX_INFERENCE_INPUT_PIXELS == 16_000_000
    validate_input_budget(safe)
    for side in (2880, 4096):
        with pytest.raises(ValueError, match='raw input pixels'):
            validate_input_budget(plan_shards_geometry(width, height, side, 960))


def test_four_slide_shard_sizes_stay_below_dense_tissue_input_cap():
    # Pinned converted dimensions (the latter three are pre-conversion
    # projections). Counts exclude occupancy skipping: all cores are planned.
    slides = ((23904, 14762, 104), (30362, 30996, 272),
              (99642, 86229, 2340), (109223, 86229, 2565))
    for width, height, expected_count in slides:
        shards = plan_shards_geometry(width, height, 1920, 960)
        assert len(shards) == expected_count
        assert max((s['input'][2]-s['input'][0]) *
                   (s['input'][3]-s['input'][1]) for s in shards) == 14_745_600
        validate_input_budget(shards)


def test_stride_aligned_halo_contains_every_native_patch_touching_its_core():
    # CellViT 1.0.9 uses 1024-pixel patches at 960-pixel stride. Every full
    # native patch that can contribute a core pixel is fully inside the input.
    width, height = 23904, 14762
    for shard in plan_shards_geometry(width, height, 1920, 960):
        x0, y0, x1, y1 = shard['core']
        ix0, iy0, ix1, iy1 = shard['input']
        assert ix0 % NATIVE_PATCH_STRIDE == iy0 % NATIVE_PATCH_STRIDE == 0
        for axis_core, axis_input, edge in (((x0,x1),(ix0,ix1),width),
                                            ((y0,y1),(iy0,iy1),height)):
            for patch_start in range(0, edge, NATIVE_PATCH_STRIDE):
                patch_end = min(edge, patch_start + 1024)
                if patch_start < axis_core[1] and patch_end > axis_core[0]:
                    assert axis_input[0] <= patch_start
                    assert patch_end <= axis_input[1]


def test_plan_checkpoint_resume_and_tamper(tmp_path):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((6, 8, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text(json.dumps({'source_mpp': 0.25, 'crop_size': {'width': 8, 'height': 6}}))
    plan = plan_shards(image, shift, 1024, 1024)
    assert plan['schema'] == SCHEMA
    validate_plan(plan)
    shard = get_shard(plan, 'r0000c0000')
    directory = tmp_path / 'shards' / shard['id']
    directory.mkdir(parents=True)
    for name in SHARD_FILES:
        (directory / name).write_text('{}')
    (directory / 'cellvit_cells.json').write_text(json.dumps({'cells': [
        {'id': 'upstream-1', 'cell_id': 'canonical-1', 'centroid': [2.5, 3.5]}]}))
    (directory / 'cellvit_cell_classification.json').write_text(json.dumps({
        'models': [{'model_id': 'sam-pannuke', 'status': 'available'}],
        'cells': [{'cell_id': 'canonical-1', 'source_id': 'upstream-1', 'x': 2.5, 'y': 3.5,
                   'predictions': {'sam-pannuke': {'status': 'predicted',
                       'class_id': 1, 'label': 'neoplastic', 'probability': 0.9}}}]}))
    with (directory / 'cellvit_cell_classification.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(CLASSIFICATION_HEADER)
        writer.writerow(['canonical-1', 'upstream-1', 2.5, 3.5, 'sam-pannuke',
                         'pannuke', 'SAM', '', 'predicted', 1, 'neoplastic', .9,
                         '', '', ''])
    (directory / 'cellvit_metadata.json').write_text(json.dumps({
        'model': 'SAM', 'requested_taxonomies': ['all'],
        'model_provenance': {'checkpoints': [{'sha256': 'synthetic'}]}}))
    marker = complete_shard(plan, shard, directory)
    assert marker['cell_count'] == 1
    assert verify_complete_shard(plan, shard, directory) == marker
    preview = finalize_shards(plan, tmp_path, publish=False)
    assert preview['cell_count'] == 1
    assert not (tmp_path / 'cellvit_partition_manifest.json').exists()
    manifest = finalize_shards(plan, tmp_path)
    assert manifest == preview
    assert manifest['cell_count'] == 1
    assert manifest['partitions'][0]['input_origin_xy'] == [0, 0]
    indexed = json.loads((directory / 'core_cells.jsonl').read_text())
    assert indexed['source_id'] == 'r0000c0000:upstream-1'
    assert indexed['predictions']['sam-pannuke']['label'] == 'neoplastic'
    (directory / 'cellvit_metadata.json').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_complete_shard(plan, shard, directory)


def test_redundant_input_crops_release_only_after_exact_checkpoint(tmp_path, monkeypatch):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((6, 8, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    plan = plan_shards(image, shift, 1024, 1024)
    shard = plan['shards'][0]
    directory = tmp_path / 'shards' / shard['id']
    staging = tmp_path / 'staging' / shard['id']
    directory.mkdir(parents=True)
    staging.mkdir(parents=True)
    staged_crop = staging / 'input.tif'
    prepared_crop = directory / 'cellvit_input.tif'
    staged_crop.write_bytes(b'exact-stage-crop')
    prepared_crop.write_bytes(b'exact-prepared-crop')
    for name in SHARD_FILES:
        (directory / name).write_text('{}')
    (directory / 'cellvit_cells.json').write_text('{"cells":[]}')
    (directory / 'cellvit_cell_classification.json').write_text(
        '{"models":[{"model_id":"m"}],"cells":[]}')
    (directory / 'cellvit_cell_classification.csv').write_text(
        ','.join(CLASSIFICATION_HEADER) + '\n')
    (directory / 'raw').mkdir()
    (directory / 'raw' / 'native_cells.geojson').write_text(
        '{"type":"FeatureCollection","features":[' +
        ','.join('{"type":"Feature","properties":{"class":"lymphocyte"},"geometry":{"type":"Point","coordinates":[1,2]}}'
                 for _ in range(10000)) + ']}')
    (directory / 'cellvit_metadata.json').write_text(json.dumps({
        'model': 'SAM', 'requested_taxonomies': ['all'],
        'model_provenance': {'checkpoints': [{'sha256': 'synthetic'}]},
        'input_preparation': {'source_sha256': digest_file(staged_crop),
                              'prepared_sha256': digest_file(prepared_crop),
                              'width': 8, 'height': 6}}))
    with pytest.raises(FileNotFoundError, match='complete.json'):
        release_verified_input_crops(plan, shard, directory, staging)
    assert staged_crop.exists() and prepared_crop.exists()
    complete_shard(plan, shard, directory)
    prepared_crop.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed before release'):
        release_verified_input_crops(plan, shard, directory, staging)
    assert staged_crop.exists() and prepared_crop.exists()
    prepared_crop.write_bytes(b'exact-prepared-crop')
    receipt = release_verified_input_crops(plan, shard, directory, staging)
    assert receipt['checkpoint_sha256'] == digest_file(directory / 'complete.json')
    assert not staged_crop.exists() and not prepared_crop.exists()
    assert release_verified_input_crops(plan, shard, directory, staging) == receipt
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 0
    original_bytes = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()
                         and p.name not in ('complete.json', 'input_crops_released.json'))
    import cellvit_shards
    write_receipt = cellvit_shards.atomic_json
    monkeypatch.setattr(cellvit_shards, 'atomic_json',
                        lambda *_: (_ for _ in ()).throw(RuntimeError('simulated receipt crash')))
    with pytest.raises(RuntimeError, match='simulated receipt crash'):
        compact_shard(plan, shard, directory)
    assert (directory / ARCHIVE_NAME).is_file()
    assert (directory / 'cellvit_cells.json').is_file()
    monkeypatch.setattr(cellvit_shards, 'atomic_json', write_receipt)
    compact = compact_shard(plan, shard, directory)
    archive_bytes = (directory / ARCHIVE_NAME).stat().st_size
    print(f'synthetic lossless CellViT archive: {original_bytes} -> {archive_bytes} bytes')
    assert archive_bytes < original_bytes // 3
    assert compact['members']['raw/native_cells.geojson']['bytes'] > 1_000_000
    assert not (directory / 'cellvit_cell_classification.csv').exists()
    assert compact_shard(plan, shard, directory) == compact
    assert verify_complete_shard(plan, shard, directory)['cell_count'] == 0
    with materialize_shard_records(directory) as materialized:
        assert json.loads((materialized / 'cellvit_cell_classification.json').read_text())['models'][0]['model_id'] == 'm'
        assert (materialized / 'cellvit_cell_classification.csv').read_text().splitlines()[0] == ','.join(CLASSIFICATION_HEADER)
    assert finalize_shards(plan, tmp_path)['partitions'][0]['lossless_archive']['sha256'] == compact['archive_sha256']
    with (directory / ARCHIVE_NAME).open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='archive receipt or bytes'):
        verify_complete_shard(plan, shard, directory)


def test_plan_rejects_identity_changes(tmp_path):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((4, 4, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    plan = plan_shards(image, shift, 1024, 1024)
    corrupted = json.loads(json.dumps(plan))
    corrupted['shards'][0]['core'][2] = 3
    with pytest.raises(ValueError, match='identity'):
        validate_plan(corrupted)
    shift.write_text('{"source_mpp":0.5}')
    with pytest.raises(ValueError, match='shift metadata'):
        validate_plan(plan)
    with pytest.raises(ValueError, match='0.25 um/px'):
        plan_shards(image, shift, 1024, 1024)


def test_plan_rejects_changed_clean_mask(tmp_path):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((4, 4, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    mask = tmp_path / 'mask.tif'
    mask.write_bytes(b'initial-mask')
    contract = {'clean_tissue_mask': str(mask), 'clean_tissue_mask_sha256': digest_file(mask)}
    plan = plan_shards(image, shift, 1024, 1024, contract)
    validate_plan(plan)
    mask.write_bytes(b'different-mask')
    with pytest.raises(ValueError, match='clean-tissue mask'):
        validate_plan(plan)


def test_native_artifact_admission_precedes_deserialization(tmp_path, monkeypatch):
    import cellvit_models
    raw = tmp_path / 'cells.json'
    graph = tmp_path / 'cells.pt'
    raw.write_bytes(b'abcdef')
    graph.write_bytes(b'12345')
    monkeypatch.setattr(cellvit_models, 'MAX_NATIVE_CELLS_JSON_BYTES', 5)
    with pytest.raises(ValueError, match='bounded partition limit'):
        require_bounded_native_outputs(raw, graph)


def test_runner_cli_plans_without_model_or_gpu(tmp_path):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((5, 7, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    output = tmp_path / 'result'
    completed = subprocess.run([sys.executable, str(ROOT / 'bin/run_cellvitpp.py'),
        '--image', str(image), '--shift', str(shift), '--outdir', str(output),
        '--model', 'SAM', '--taxonomies', 'all', '--shard-plan'],
        text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)['shards'] == 1
    plan = json.loads((output / 'cellvit_shard_plan.json').read_text())
    assert plan['run_contract']['model'] == 'SAM'
    assert plan['core_size'] == 1920
    assert plan['halo'] == 960
    assert not (output / 'shards').exists()


def test_finalize_rejects_mixed_models(tmp_path):
    image = tmp_path / 'slide.tif'
    tifffile.imwrite(image, np.full((2, 1025, 3), 255, dtype=np.uint8), photometric='rgb')
    shift = tmp_path / 'shift.json'
    shift.write_text('{"source_mpp":0.25}')
    plan = plan_shards(image, shift, 1024, 1024)
    for shard, model in zip(plan['shards'], ('SAM', 'HIPT')):
        directory = tmp_path / 'shards' / shard['id']
        directory.mkdir(parents=True)
        for name in SHARD_FILES:
            (directory / name).write_text('{}')
        (directory / 'cellvit_cells.json').write_text('{"cells":[]}')
        (directory / 'cellvit_cell_classification.json').write_text(json.dumps({
            'models': [{'model_id': model}], 'cells': []}))
        (directory / 'cellvit_cell_classification.csv').write_text(
            ','.join(CLASSIFICATION_HEADER) + '\n')
        (directory / 'cellvit_metadata.json').write_text(json.dumps({
            'model': model, 'requested_taxonomies': ['all'],
            'model_provenance': {'checkpoints': [{'sha256': model}]}}))
        complete_shard(plan, shard, directory)
    with pytest.raises(ValueError, match='different segmentation'):
        finalize_shards(plan, tmp_path)
    assert not (tmp_path / 'cellvit_partition_manifest.json').exists()


def test_native_graph_and_cells_select_identical_core_rows(tmp_path):
    torch = pytest.importorskip('torch')
    from run_cellvitpp import select_native_source_outputs
    raw = tmp_path / 'cells.json'
    graph = tmp_path / 'cells.pt'
    raw.write_text(json.dumps({'cells': [
        {'id': 'core', 'centroid': [1023.5, 3.0]},
        {'id': 'halo', 'centroid': [1024.0, 3.0]}]}))
    torch.save({'x': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                'positions': torch.tensor([[1023.5, 3.0], [1024.0, 3.0]])}, graph)
    selected_json, selected_graph, evidence = select_native_source_outputs(
        raw, graph, tmp_path / 'selected', 2048, 1024, (0, 0, 1024, 1024))
    assert [cell['id'] for cell in json.loads(selected_json.read_text())['cells']] == ['core']
    loaded = torch.load(selected_graph, weights_only=True)
    assert loaded['x'].tolist() == [[1.0, 2.0]]
    assert evidence['retained_cells'] == 1
