import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from build_cell_profiles import build_profiles, canonical_cells, export_uni2_block, join_columns
from cell_profile_io import RasterReader, sha256_file


def inputs(tmp_path):
    objects = tmp_path / "objects.csv"
    pd.DataFrame({"label": [2, 1], "x": [8., 2.], "y": [4., 2.],
                  "xmin": [7, 1], "ymin": [3, 1], "xmax": [10, 4], "ymax": [6, 4],
                  "cellvitpp_type": ["connective", "inflammatory"]}).to_csv(objects, index=False)
    shift = tmp_path / "shift.json"
    shift.write_text(json.dumps({"source_mpp": 0.5, "offset_crop_to_original": {"dx": 100, "dy": 200}, "crop_size": {"width": 12, "height": 8}}))
    labels = np.zeros((8, 12), dtype=np.uint32)
    labels[1:4, 1:4] = 1
    labels[3:6, 7:10] = 2
    tifffile.imwrite(tmp_path / "labels.tif", labels)
    return objects, shift


def options(tmp_path, **kwargs):
    objects, shift = inputs(tmp_path)
    defaults = dict(objects=str(objects), shift=str(shift), labels=str(tmp_path / "labels.tif"), sample_id="sample:A", outdir=str(tmp_path / "profiles"), segmentation_id=None, morphology=None, marker_quant_dir=None, uni2_context=None, uni2_local=None, cellvit_features=None, domain_mask=None, domain_uncertainty=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_profiles_keep_all_cells_coordinates_and_missing_modalities(tmp_path):
    cells, manifest = build_profiles(options(tmp_path))
    assert cells.cell_id.tolist() == ["2", "1"]
    np.testing.assert_allclose(cells[["x_um", "y_um"]], [[54, 102], [51, 101]])
    assert cells.cell_uid.is_unique
    assert not cells.markers_available.any()
    assert not cells.uni2_context_available.any()
    assert manifest["join_policy"] == "canonical_left_join_no_imputation"
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "profiles/cell_profiles.parquet"), cells)
    with pytest.raises(FileExistsError):
        build_profiles(options(tmp_path))


def test_marker_join_by_id_not_marker_pixel_coordinates(tmp_path):
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    pd.DataFrame({"label_id": [1], "centroid_x_px": [9999], "DAPI__mean": [0.125], "Cy5__mean": [0.9]}).to_csv(marker_dir / "s_nuclei_gigatime_quantification.csv", index=False)
    cells, manifest = build_profiles(options(tmp_path, marker_quant_dir=str(marker_dir)))
    assert cells.cell_id.tolist() == ["2", "1"]
    assert cells.markers_available.tolist() == [False, True]
    assert np.isnan(cells.loc[0, "predicted__nucleus__DAPI__mean"])
    assert cells.loc[1, "x_um"] == 51
    assert manifest["biological_marker_features"] == ["predicted__nucleus__DAPI__mean"]
    assert "predicted__nucleus__Cy5__mean" in cells  # retained for QC, excluded as biological feature


@pytest.mark.parametrize("ids", [["1", "1"], ["3"]])
def test_join_rejects_duplicates_and_foreign_ids(tmp_path, ids):
    obj, shift = inputs(tmp_path)
    cells, _ = canonical_cells(obj, "s", shift)
    path = tmp_path / "other.csv"
    pd.DataFrame({"label_id": ids, "a": list(range(len(ids)))}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        join_columns(cells, path, "label_id", "marker__")


def test_embedding_alignment_missing_rows_and_reject_grid(tmp_path):
    obj, shift = inputs(tmp_path)
    cells, _ = canonical_cells(obj, "s", shift)
    shard = tmp_path / "shard.csv"
    frame = pd.DataFrame({"cell_id": [1], "observation_type": ["cell"], "cx": [2], "cy": [2], "feat_1": [.2], "feat_2": [.8]})
    frame.to_csv(shard, index=False)
    found, record = export_uni2_block(cells, shard, "uni2", tmp_path)
    assert found.tolist() == [False, True]
    matrix = np.load(tmp_path / "uni2.npy", allow_pickle=False)
    assert np.isnan(matrix[0]).all()
    np.testing.assert_allclose(matrix[1], [.2, .8])
    assert record["missing_cells"] == 1
    frame.observation_type = "grid"
    frame.to_csv(shard, index=False)
    with pytest.raises(ValueError, match="grid observations"):
        export_uni2_block(cells, shard, "bad", tmp_path)


def test_domain_and_uncertainty_grids_have_explicit_crop_transform(tmp_path):
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, :3] = 1
    mask[:, 3:] = 2
    uncertainty = np.zeros((8, 12), dtype=np.uint8)
    uncertainty[4, 8] = 2
    tifffile.imwrite(tmp_path / "domains.tif", mask)
    tifffile.imwrite(tmp_path / "uncertain.tif", uncertainty)
    cells, _ = build_profiles(options(tmp_path, domain_mask=str(tmp_path / "domains.tif"), domain_uncertainty=str(tmp_path / "uncertain.tif")))
    assert cells.tissue_domain.tolist() == [2, 1]
    assert cells.tissue_domain_status.tolist() == ["uncertain", "assigned"]


def test_tiled_compressed_raster_reader_does_not_use_whole_image_read(tmp_path, monkeypatch):
    image = np.arange(40 * 50, dtype=np.uint16).reshape(40, 50)
    path = tmp_path / "compressed.tif"
    tifffile.imwrite(path, image, tile=(16, 16), compression="deflate", metadata={"axes": "YX"})
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: pytest.fail("must not decode full image"))
    with RasterReader(path) as reader:
        np.testing.assert_array_equal(reader.window(13, 17, 22, 23), image[17:23, 13:22])
        np.testing.assert_array_equal(reader.sample([49, 0, 20], [39, 0, 18]), image[[39, 0, 18], [49, 0, 20]])


def test_unverified_coordinate_frame_fails(tmp_path):
    obj, shift = inputs(tmp_path)
    payload = json.loads(shift.read_text())
    del payload["offset_crop_to_original"]
    shift.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="offset"):
        canonical_cells(obj, "s", shift)


def test_authoritative_resolution_overrides_legacy_mpp_and_preserves_source_ids(tmp_path):
    opts = options(tmp_path)
    path = Path(opts.objects)
    table = pd.read_csv(path)
    table["cellvitpp_id"] = ["007", "09"]
    table.to_csv(path, index=False)
    shift = Path(opts.shift)
    payload = json.loads(shift.read_text())
    payload["source_mpp"] = 1000
    shift.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="verified source_mpp"):
        canonical_cells(path, "s", shift)
    report = tmp_path / "resolution.json"
    report.write_text(json.dumps({"status": "pass", "mpp_x": .273774, "mpp_y": .273774}))
    opts.resolution_json = str(report)
    cells, manifest = build_profiles(opts)
    assert cells.cellvitpp_id.tolist() == ["007", "09"]
    assert cells.source_mpp.tolist() == [.273774, .273774]
    assert manifest["calibration_source"] == "passed_resolution_report"
    assert "resolution_json_sha256" in manifest["inputs"]


def test_requested_missing_markers_fail(tmp_path):
    with pytest.raises(ValueError, match="no canonical quantification"):
        build_profiles(options(tmp_path, marker_quant_dir=str(tmp_path / "missing")))


def test_nextflow_symlinked_marker_directory_is_discovered(tmp_path):
    source = tmp_path / "published_quantification"
    source.mkdir()
    pd.DataFrame({"label_id": [1, 2], "DAPI__mean": [.3, .7]}).to_csv(source / "s_nuclei_gigatime_quantification.csv", index=False)
    stage = tmp_path / "staged_markers"
    stage.mkdir()
    (stage / "quantification_s").symlink_to(source, target_is_directory=True)
    (source / "cycle").symlink_to(stage, target_is_directory=True)
    cells, _ = build_profiles(options(tmp_path, marker_quant_dir=str(stage)))
    assert cells.markers_available.all()
    assert cells["predicted__nucleus__DAPI__mean"].tolist() == [.7, .3]


def test_legacy_uni2_model_identity_is_portable_but_inputs_are_unverified(tmp_path):
    opts = options(tmp_path)
    source = tmp_path / "embeddings"
    source.mkdir()
    table = pd.DataFrame({"cell_id": [1], "observation_type": ["cell"], "feat_1": [.4], "model_tile_size": [224], "target_mpp": [.25], "effective_mpp": [.25], "source_mpp": [.5], "mask_context_mode": ["full"]})
    table.to_csv(source / "a_embeddings_shard_0.csv", index=False)
    completion = {"encoder": "uni2-h", "embedding_mode": "tile", "paired_inner_square_mode": "token_pool", "model_provenance": {"resolved_revision": "a" * 40, "source_repository": "UNI2", "cache_path": "/private/cache", "checkpoints": [{"sha256": "b" * 64, "logical_name": "model"}]}}
    (source / ".a_embedding_complete.json").write_text(json.dumps(completion))
    opts.uni2_context = str(source)
    _, manifest = build_profiles(opts)
    block = manifest["feature_blocks"]["uni2_context"]
    assert not block["reference_compatible"]
    assert block['provenance']['status'] == 'legacy_unverified_inputs'
    assert "/private/cache" not in json.dumps(block["feature_definition"])
    assert block["sha256"]


def receipt_fixture(tmp_path):
    opts = options(tmp_path)
    image_path = tmp_path / 'he.tif'
    tifffile.imwrite(image_path, np.zeros((8, 12, 3), dtype=np.uint8), photometric='rgb')
    resolution = tmp_path / 'resolution.json'
    resolution.write_text(json.dumps({'status': 'pass', 'mpp_x': .5, 'mpp_y': .5}))
    opts.image, opts.resolution_json = str(image_path), str(resolution)
    source = tmp_path / 'embeddings'
    source.mkdir()
    shard = source / 'a_embeddings_shard0.csv'
    pd.DataFrame({'cell_id': [1, 2], 'observation_type': ['cell'] * 2, 'feat_1': [.4, .6],
        'model_tile_size': [224] * 2, 'target_mpp': [.25] * 2, 'effective_mpp': [.25] * 2,
        'source_mpp': [.5] * 2, 'mask_context_mode': ['full'] * 2}).to_csv(shard, index=False)
    contract = {'schema_version': '2.0.0', 'encoder_state_sha256': 'c' * 64,
        'parameters': {'observation_type': 'cell'}, 'inputs': {'objects_csv': None,
        **{key: {'sha256': sha256_file(path)} for key, path in
           [('image', image_path), ('mask', opts.labels), ('resolution_json', resolution)]}}}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    complete = {'encoder': 'uni2-h', 'embedding_mode': 'tile', 'observation_type': 'cell',
        'paired_inner_square_mode': 'token_subset', 'expected_observations': 2, 'rows_written': 2, 'completed_grids': 1,
        'model_provenance': {'source_repository': 'UNI2', 'resolved_revision': 'a' * 40,
                            'checkpoints': [{'logical_name': 'model', 'sha256': 'b' * 64}], 'runtime_state_sha256': 'c' * 64},
        'cache_contract': contract, 'cache_contract_sha256': digest}
    (source / '.a_embedding_complete.json').write_text(json.dumps(complete))
    (source / '.a_grid_complete.json').write_text(json.dumps({'cache_contract_sha256': digest,
        'rows_written': 2, 'index_start': 0, 'index_end': 2, 'shards': 1,
        'shard_files': [{'name': shard.name, 'sha256': sha256_file(shard)}]}))
    opts.uni2_context = str(source)
    return opts, source


def test_uni2_exact_producer_receipt_is_bound_to_canonical_inputs(tmp_path):
    opts, source = receipt_fixture(tmp_path)
    _, manifest = build_profiles(opts)
    block = manifest['feature_blocks']['uni2_context']
    assert block['reference_compatible']
    assert block['provenance']['status'] == 'verified_exact_inputs_and_shards'
    assert block['provenance']['rows_verified'] == 2
    assert block['feature_definition']['verified_execution']['encoder_state_sha256'] == 'c' * 64


@pytest.mark.parametrize('corrupt', ['image', 'labels', 'shard', 'receipt', 'extra_shard'])
def test_uni2_receipt_rejects_wrong_inputs_or_changed_features(tmp_path, corrupt):
    opts, source = receipt_fixture(tmp_path)
    if corrupt in {'image', 'labels'}:
        with Path(getattr(opts, corrupt)).open('ab') as handle:
            handle.write(b'different source content')
    elif corrupt == 'shard':
        path = source / 'a_embeddings_shard0.csv'
        path.write_text(path.read_text().replace('0.4', '0.9'))
    elif corrupt == 'receipt':
        path = source / '.a_embedding_complete.json'
        record = json.loads(path.read_text())
        record['cache_contract']['encoder_state_sha256'] = 'd' * 64
        path.write_text(json.dumps(record))
    else:
        pd.DataFrame({'cell_id': [], 'observation_type': [], 'feat_1': []}).to_csv(source / 'a_embeddings_shard1.csv', index=False)
    with pytest.raises(ValueError, match='source differs|checksum|fingerprint|exact supplied shards'):
        build_profiles(opts)


def test_mixed_embedding_context_rejected(tmp_path):
    obj, shift = inputs(tmp_path)
    cells, _ = canonical_cells(obj, "s", shift)
    shard = tmp_path / "shard.csv"
    pd.DataFrame({"cell_id": [1, 2], "observation_type": ["cell"] * 2, "feat_1": [.1, .2], "target_mpp": [.25, .5]}).to_csv(shard, index=False)
    with pytest.raises(ValueError, match="incompatible physical"):
        export_uni2_block(cells, shard, "uni2", tmp_path)


def test_embedding_calibration_conflict_rejected(tmp_path):
    obj, shift = inputs(tmp_path)
    cells, _ = canonical_cells(obj, "s", shift)
    shard = tmp_path / "shard.csv"
    pd.DataFrame({"cell_id": [1], "observation_type": ["cell"], "feat_1": [.1], "source_mpp": [.25]}).to_csv(shard, index=False)
    with pytest.raises(ValueError, match="source MPP conflicts"):
        export_uni2_block(cells, shard, "uni2", tmp_path)


def versioned_markers(tmp_path):
    inputs(tmp_path)
    directory = tmp_path / "markers"
    directory.mkdir()
    names = ["DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin"]
    table = {"label_id": [1, 2], "quantification_status": ["authoritative_full_precision_integrated"] * 2}
    table.update({f"{name}__mean": [.2, .4] for name in names})
    pd.DataFrame(table).to_csv(directory / "s_nuclei_gigatime_quantification.csv", index=False)
    schema = {"schema_version": "cellphenotyper.gigatime.v1", "marker_names": names,
              "default_phenotype_excluded_channels": ["TRITC", "Cy5"], "checkpoint_sha256": "b"*64,
              "prediction_precision": "float32", "reduction_precision": "float64", "model_arithmetic": "float32",
              "prediction_settings": {"patch_size": 256, "stride": 128},
              "coordinate_contract": {"source_mpp": .5, "effective_mpp": .25, "original_shape_yx": [8, 12]},
              "compartments": {"nuclei": {"semantics": "canonical_nuclear_mask", "mask_sha256": sha256_file(tmp_path / "labels.tif")}}}
    return directory, schema


def write_marker_summary(directory, schema, status="authoritative_full_precision_integrated"):
    schema["schema_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in schema.items() if k != "schema_sha256"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (directory / "s_nuclei_gigatime_intensity_summary.json").write_text(json.dumps({"marker_schema": schema, "authority_status": status}))


def test_marker_reference_definition_preserves_precision_not_sample_identity(tmp_path):
    directory, schema = versioned_markers(tmp_path)
    write_marker_summary(directory, schema)
    cells, manifest = build_profiles(options(tmp_path, marker_quant_dir=directory))
    block = manifest["feature_blocks"]["markers_nucleus"]
    assert block["reference_compatible"]
    assert len(block["feature_names"]) == 21
    assert "mask_sha256" not in json.dumps(block["feature_definition"])
    assert "schema_sha256" not in block["feature_definition"]
    assert block["feature_definition"]["checkpoint_sha256"] == "b"*64
    assert cells["predicted__nucleus__quantification_status"].eq("authoritative_full_precision_integrated").all()
    assert manifest["tables"]["nucleus"]["provenance"]["compartment_mask_sha256"] == sha256_file(tmp_path / "labels.tif")


@pytest.mark.parametrize("fault", ["calibration", "fingerprint", "order", "mask", "row_status"])
def test_marker_schema_conflicts_fail(tmp_path, fault):
    directory, schema = versioned_markers(tmp_path)
    if fault == "calibration":
        schema["coordinate_contract"]["source_mpp"] = .25
    if fault == "order":
        schema["marker_names"] = list(reversed(schema["marker_names"]))
    if fault == "mask":
        schema["compartments"]["nuclei"]["mask_sha256"] = "a" * 64
    if fault == "row_status":
        path = directory / "s_nuclei_gigatime_quantification.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "quantification_status"] = "legacy_research_not_equivalent"
        frame.to_csv(path, index=False)
    write_marker_summary(directory, schema)
    if fault == "fingerprint":
        path = directory / "s_nuclei_gigatime_intensity_summary.json"
        payload = json.loads(path.read_text())
        payload["marker_schema"]["schema_sha256"] = "0"*64
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="conflicts|mismatch|differs"):
        build_profiles(options(tmp_path, marker_quant_dir=directory))


def test_non_equivalent_marker_restart_never_reference_compatible(tmp_path):
    directory, schema = versioned_markers(tmp_path)
    write_marker_summary(directory, schema, "legacy_research_not_equivalent")
    path = directory / "s_nuclei_gigatime_quantification.csv"
    frame = pd.read_csv(path)
    frame["quantification_status"] = "legacy_research_not_equivalent"
    frame.to_csv(path, index=False)
    _, manifest = build_profiles(options(tmp_path, marker_quant_dir=directory))
    assert not manifest["feature_blocks"]["markers_nucleus"]["reference_compatible"]


def test_blank_numeric_markers_stay_missing_in_table_and_array(tmp_path):
    directory = tmp_path / "markers"
    directory.mkdir()
    pd.DataFrame({"label_id": [1, 2], "DAPI__mean": ["", .4]}).to_csv(directory / "s_nuclei_gigatime_quantification.csv", index=False)
    cells, _ = build_profiles(options(tmp_path, marker_quant_dir=directory))
    assert cells.markers_available.tolist() == [True, False]
    assert pd.isna(cells.loc[1, "predicted__nucleus__DAPI__mean"])
    assert np.isnan(np.load(tmp_path / "profiles/feature_blocks/markers_nucleus.npy")[1, 0])


def test_versioned_markers_without_canonical_mask_are_not_reference_compatible(tmp_path):
    directory, schema = versioned_markers(tmp_path)
    write_marker_summary(directory, schema)
    _, manifest = build_profiles(options(tmp_path, marker_quant_dir=directory, labels=None))
    assert not manifest["feature_blocks"]["markers_nucleus"]["reference_compatible"]


def test_segmentation_identity_includes_label_raster(tmp_path):
    opts = options(tmp_path)
    a, _ = build_profiles(opts)
    raster = tifffile.imread(opts.labels)
    raster[0, 0] = 1
    tifffile.imwrite(opts.labels, raster)
    opts.outdir = str(tmp_path / "profiles_changed")
    b, manifest = build_profiles(opts)
    assert a.cell_id.tolist() == b.cell_id.tolist()
    assert not a.cell_uid.equals(b.cell_uid)
    assert manifest["segmentation_identity_basis"] == "objects_and_label_raster_sha256"


def test_compartment_qc_join_and_construction_retained(tmp_path):
    qc = tmp_path / "compartment_qc.csv"
    pd.DataFrame({"label": [1, 2], "crowding_flag": [1, 0], "ring_pixels": [12, 8]}).to_csv(qc, index=False)
    (tmp_path / "compartment_summary.json").write_text(json.dumps({"schema_version": "1.0", "expand_um": 3., "mpp_x": .5, "mpp_y": .5,
        "label_frame": "crop", "crop_offset_xy": [100, 200], "distance_metric": "tissue-constrained", "semantics": {"ring": "not membrane"}}))
    cells, manifest = build_profiles(options(tmp_path, compartment_qc=qc))
    assert cells["compartment__crowding_flag"].tolist() == [0, 1]
    assert cells["compartment__available"].all()
    assert manifest["tables"]["compartment_qc"]["definition"]["expand_um"] == 3.
