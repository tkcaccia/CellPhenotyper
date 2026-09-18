import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import storage_preflight as storage  # noqa: E402


def test_reference_mapping_storage_counts_query_copy_and_export_interpretation(tmp_path):
    atlas = tmp_path / 'atlas'
    atlas.mkdir()
    manifest = atlas / 'atlas_manifest.json'
    manifest.write_text('{"atlas_id":"test"}')
    model = storage.reference_mapping_storage_model(observations=10, source_table_bytes=12000, atlas=str(atlas))
    assert model['source_table_copy_bytes'] == 12000
    assert model['atlas_manifest_bytes'] == manifest.stat().st_size
    assert model['published_bytes'] == 12000 + model['interpretation_bytes'] + model['receipt_allowance_bytes'] + model['atlas_manifest_bytes']
    assert model['spatialdata_attachment_bytes'] == model['interpretation_bytes'] + 3 * (model['receipt_allowance_bytes'] + model['atlas_manifest_bytes'])
    assert model['work_bytes'] == model['published_bytes']
    assert model['additional_model_inference'] is False
    source = (ROOT / 'lib/StoragePreflight.groovy').read_text()
    assert '--${kind}-reference-atlas' in source
    with pytest.raises(FileNotFoundError, match='Reference atlas manifest'):
        storage.reference_mapping_storage_model(observations=10, source_table_bytes=12000, atlas=str(tmp_path / 'absent'))


def test_reference_mapping_report_includes_both_units_and_export_copy(tmp_path, monkeypatch):
    image = tmp_path / 'sample.tif'
    image.write_bytes(b'dimension probe fixture')
    monkeypatch.setattr(storage, 'probe_dimensions', lambda _: (1000, 1000, 'test'))
    options = hierarchy_options(tmp_path)
    atlas = tmp_path / 'atlas'
    atlas.mkdir()
    (atlas / 'atlas_manifest.json').write_text('{"size_fixture":true}')
    args = storage.parser().parse_args(['--input', str(image), '--outdir', str(tmp_path),
        '--workdir', str(tmp_path), '--output-json', str(tmp_path / 'report.json'),
        '--start-point', 'convert', '--end-point', 'cluster_geojson',
        '--source-mpp', '.25', '--cell-profiles-enabled', 'true', '--cell-profiles-spatialdata', 'true',
        '--tissue-hierarchy-enabled', 'true', '--hierarchy-model-snapshot', options['model_snapshot']])
    baseline = storage.build_report(args)['stage_storage_models']
    args.cell_reference_atlas = args.region_reference_atlas = str(atlas)
    models = storage.build_report(args)['stage_storage_models']
    cell, region = models['cell_reference_mapping'], models['region_reference_mapping']
    assert cell['observation_count_allowance'] == models['cell_profiles']['estimated_cells']
    assert region['observation_count_allowance'] == models['tissue_hierarchy']['region_count_allowance']
    extra = cell['spatialdata_attachment_bytes'] + region['spatialdata_attachment_bytes']
    for key in ('published_bytes', 'retained_work_bytes', 'work_bytes'):
        assert models['spatialdata'][key] == baseline['spatialdata'][key] + extra
    args.tissue_hierarchy_enabled = 'false'
    with pytest.raises(ValueError, match='region-reference-atlas requires'):
        storage.build_report(args)
    args.region_reference_atlas = ''
    args.cell_profiles_enabled = 'false'
    with pytest.raises(ValueError, match='cell-reference-atlas requires'):
        storage.build_report(args)


def test_cell_hierarchy_linkage_accounts_for_immutable_copy_and_fragmentation():
    profile = {'estimated_cells': 10, 'scalar_tables_bytes': 1000, 'published_bytes': 5000}
    nuclear = storage.cell_hierarchy_link_storage_model(profile, {}, native_pixels=1000, physical_compartments=False)
    rings = storage.cell_hierarchy_link_storage_model(profile, {}, native_pixels=1000, physical_compartments=True)
    assert nuclear['overlap_rows_allowance'] == 40 and rings['overlap_rows_allowance'] == 80
    assert nuclear['profile_copy_bytes'] == 5000 and nuclear['source_registry_bytes'] == 1000
    assert rings['published_bytes'] > nuclear['published_bytes'] > profile['published_bytes']
    assert rings['work_bytes'] > rings['published_bytes']
    assert nuclear['worst_case_output_extra_bytes'] == 960 * 1024
    assert rings['additional_model_inference'] is False


def test_nextflow_wrapper_treats_yaml_boolean_false_as_off() -> None:
    source = (ROOT / "lib" / "StoragePreflight.groovy").read_text(encoding="utf-8")
    assert "rawMode instanceof Boolean && !rawMode ? 'off'" in source


def test_roi_bbox_fraction_is_clipped_and_padded(tmp_path: Path) -> None:
    roi = tmp_path / "sample.geojson"
    roi.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-10, 10], [50, 10], [50, 60], [-10, 60], [-10, 10]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Clipped box is 50 x 50 of a 100 x 100 image, then 15% crop allowance.
    assert storage.roi_bbox_fraction(roi, 100, 100) == 0.2875


def test_route_factors_remove_obsolete_growth_and_scale_both_routes() -> None:
    factors = storage.adjusted_stage_factors(
        storage.active_stages("uni2", "cluster_geojson"),
        cell_detection_mode="consensus",
        uni2_sampling_mode="both",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=5,
        uni2_save_tiles=False,
    )
    assert "grow_tissue" not in factors
    assert factors["uni2"][0] == storage.STAGE_FACTORS["uni2"][0] * 2
    assert factors["cluster_geojson"][1] == storage.STAGE_FACTORS["cluster_geojson"][1] * 2


def test_stardist_storage_accounts_for_pyramidal_crop_overviews(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"dimension fixture")
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 500, "test"))
    args = storage.parser().parse_args([
        "--input", str(image), "--outdir", str(tmp_path), "--workdir", str(tmp_path),
        "--output-json", str(tmp_path / "report.json"), "--start-point", "convert",
        "--end-point", "stardist", "--min-free-gib", "0",
    ])
    report = storage.build_report(args)
    model = report["stage_storage_models"]["stardist"]
    active_rgb_bytes = 1000 * 500 * 3
    expected_overviews = math.ceil(active_rgb_bytes * (storage.PYRAMID_FACTOR - 1.0))
    assert model["roi_crop_pyramid_factor"] == storage.PYRAMID_FACTOR
    assert model["roi_crop_overview_allowance_bytes"] == expected_overviews
    assert model["published_bytes"] == math.ceil(
        active_rgb_bytes * storage.STAGE_FACTORS["stardist"][0]
    ) + expected_overviews


def test_gigatime_channels_and_saved_tiles_raise_capacity_estimate() -> None:
    base = storage.adjusted_stage_factors(
        ["gigatime", "uni2"],
        cell_detection_mode="consensus",
        uni2_sampling_mode="cells",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=5,
        uni2_save_tiles=False,
    )
    larger = storage.adjusted_stage_factors(
        ["gigatime", "uni2"],
        cell_detection_mode="consensus",
        uni2_sampling_mode="cells",
        gigatime_enabled=True,
        marker_quantification_enabled=True,
        gigatime_channel_count=10,
        uni2_save_tiles=True,
    )
    assert larger["gigatime"][0] == base["gigatime"][0] * 2
    assert larger["uni2"][0] > base["uni2"][0]


def test_capacity_has_distinct_expected_and_restart_states() -> None:
    gib = storage.GIB
    assert storage.assess_capacity(100 * gib, 30 * gib, 60 * gib, 20 * gib) == "pass"
    assert storage.assess_capacity(70 * gib, 30 * gib, 60 * gib, 20 * gib) == "warning"
    assert storage.assess_capacity(45 * gib, 30 * gib, 60 * gib, 20 * gib) == "fail"


def test_report_groups_output_work_and_cache_on_one_filesystem(
    tmp_path: Path, monkeypatch,
) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"x" * 1024)
    outdir = tmp_path / "results"
    workdir = tmp_path / "work"
    cache = tmp_path / "cache"
    outdir.mkdir()
    workdir.mkdir()
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 500, "test"))
    monkeypatch.setattr(
        storage,
        "filesystem_info",
        lambda path: {
            "device": "same",
            "anchor": str(tmp_path),
            "available_bytes": 500 * storage.GIB,
            "capacity_bytes": 1000 * storage.GIB,
        },
    )
    args = storage.parser().parse_args(
        [
            "--input", str(image),
            "--outdir", str(outdir),
            "--workdir", str(workdir),
            "--output-json", str(outdir / "00_execution" / "storage_preflight.json"),
            "--start-point", "convert",
            "--end-point", "uni2",
            "--cache", f"hf={cache}",
            "--min-free-gib", "1",
        ]
    )
    report = storage.build_report(args)
    assert report["status"] == "pass"
    assert report["inputs"][0]["active_rgb_bytes"] == 1_500_000
    assert len(report["filesystems"]) == 1
    kinds = {row["kind"] for row in report["filesystems"][0]["paths"]}
    assert kinds == {"output", "work", "cache"}
    assert report["cache_inventory"][0]["label"] == "hf"
    assert report["filesystems"][0]["worst_case_increment_bytes"] > report["filesystems"][0]["expected_increment_bytes"]


def test_rellink_reduces_output_duplication(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "sample.tif"
    image.write_bytes(b"x")
    outdir = tmp_path / "results"
    workdir = tmp_path / "work"
    outdir.mkdir()
    workdir.mkdir()
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (10_000, 10_000, "test"))

    def build(mode: str) -> dict:
        args = storage.parser().parse_args(
            [
                "--input", str(image), "--outdir", str(outdir), "--workdir", str(workdir),
                "--output-json", str(outdir / "preflight.json"), "--start-point", "convert",
                "--end-point", "cluster_geojson", "--publish-dir-mode", mode, "--min-free-gib", "0",
            ]
        )
        return storage.build_report(args)

    copied = next(row for row in build("copy")["demands"] if row["kind"] == "output")
    linked = next(row for row in build("rellink")["demands"] if row["kind"] == "output")
    assert copied["expected_bytes"] > linked["expected_bytes"]


@pytest.mark.parametrize("spec,count,expected", [("", 1, 23), ("  , ", 5, 23),
    (None, 0, 23), (None, None, 23), (None, 5, 5), ("DAPI,CD3,dapi", 23, 2)])
def test_empty_channel_selection_is_full_model(spec, count, expected):
    assert storage.resolve_channel_count(spec, count) == expected


def test_wrapper_forwards_precision_compartments_scale_and_profiles():
    source = (ROOT / "lib" / "StoragePreflight.groovy").read_text()
    assert "Math.max(1, outputChannels.size())" not in source
    for flag in ("--gigatime-output-channels", "--gigatime-output-dtype", "--gigatime-output-format",
                 "--gigatime-export-ometiff", "--gigatime-export-output-dtype", "--gigatime-blockwise",
                 "--source-mpp", "--gigatime-target-mpp", "--physical-compartments",
                 "--cell-profiles-enabled", "--cell-neighborhood-radii-um", "--input-metadata-json",
                 "--hovernet-execution-mode", "--hovernet-target-mpp", "--hovernet-stream-core-size",
                 "--hovernet-stream-halo", "--hovernet-stream-batch-tiles", "--hovernet-export-contours"):
        assert flag in source


def test_hovernet_streaming_model_removes_slide_wide_arrays(tmp_path, monkeypatch):
    image = tmp_path / "sample.tif"
    image.write_bytes(b"dimension fixture")
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (20_000, 10_000, "test"))
    common = [
        "--input", str(image), "--outdir", str(tmp_path), "--workdir", str(tmp_path),
        "--output-json", str(tmp_path / "report.json"), "--start-point", "cell_consensus",
        "--end-point", "cell_consensus", "--source-mpp", ".25", "--min-free-gib", "0",
    ]
    streaming = storage.build_report(storage.parser().parse_args(common))["stage_storage_models"]["cell_consensus"]["hovernet"]
    legacy = storage.build_report(storage.parser().parse_args(common + ["--hovernet-execution-mode", "wsi"]))["stage_storage_models"]["cell_consensus"]["hovernet"]
    assert streaming["bounded_tile_batch"] is True
    assert streaming["slide_wide_prediction_bytes"] == 0
    assert streaming["batch_tiles"] == 15
    assert legacy["bounded_tile_batch"] is False
    assert legacy["slide_wide_prediction_bytes"] == legacy["inference_pixels"] * 20
    assert legacy["scratch_bytes"] > streaming["scratch_bytes"]


def test_full_float_store_and_second_export_pyramid_accounted():
    model = storage.gigatime_storage_model(3000)
    base = 3000 * 23 * 4
    assert model["primary_store_bytes"] == base
    assert model["exported_ometiff_bytes"] == base * 4 // 3
    assert model["published_bytes"] == base * 7 // 3
    assert model["work_bytes"] == model["published_bytes"]
    assert model["scratch_bytes"] == 0  # blockwise Zarr, no full-image scratch


@pytest.mark.parametrize("dtype,factor", [("uint8", 1), ("uint16", 2), ("float32", 4)])
def test_dtype_width_scales_arrays(dtype, factor):
    model = storage.gigatime_storage_model(3000, output_dtype=dtype, export_ometiff=False)
    assert model["level0_bytes"] == 3000 * 23 * factor
    assert model["published_bytes"] == model["level0_bytes"]


def test_direct_tiff_counts_staged_buffer_and_export_dtype_independently():
    model = storage.gigatime_storage_model(3000, output_format="ome_tiff", output_dtype="float32",
        export_channel_count=5, export_dtype="uint8")
    assert model["primary_store_bytes"] == 3000 * 23 * 4 * 4 // 3
    assert model["staged_level0_bytes"] == 3000 * 23 * 4
    assert model["exported_ometiff_bytes"] == 3000 * 5 * 4 // 3
    assert model["work_bytes"] == model["published_bytes"] + model["staged_level0_bytes"]


def test_no_store_has_no_large_published_array_and_dense_still_counts_all_channels():
    model = storage.gigatime_storage_model(1000, output_format="none")
    assert model["published_bytes"] == model["scratch_bytes"] == 0
    dense = storage.gigatime_storage_model(1000, channel_count=1, output_dtype="uint8", blockwise=False)
    assert dense["dense_accumulator_allowance_bytes"] >= 1000 * (23 + 1 + 1) * 4
    legacy = storage.gigatime_storage_model(1000, output_format="ome_tiff", output_dtype="uint8", blockwise=False)
    assert legacy["bytes_per_sample"] == 4


def input_estimate(mpp=None, source="unknown", pixels=1_000_000):
    return storage.InputEstimate("image.tif", 1000, 1000, 1000, "test", None, 1.,
                                 pixels * 3, "test", mpp, source)


def test_explicit_physical_resampling_accounts_for_upsampling_and_rounding():
    estimate = storage.inference_pixel_estimate(input_estimate(.5, "explicit_override"), .25)
    assert estimate["inference_area_scale"] == 4
    assert estimate["inference_pixels"] >= 4_000_000
    small = storage.inference_pixel_estimate(input_estimate(.125, "explicit_override"), .25)
    assert small["inference_area_scale"] == .25
    assert 250_000 <= small["inference_pixels"] < 260_000


def test_unknown_or_unvalidated_metadata_never_earns_downsampling_discount():
    unknown = storage.inference_pixel_estimate(input_estimate(), .25)
    stale = storage.inference_pixel_estimate(input_estimate(.125, "unvalidated_ome_metadata_max_axis"), .25)
    assert unknown["inference_area_scale"] == stale["inference_area_scale"] == 16
    assert "not_an_upper_bound" in unknown["reason"]
    bigger = storage.inference_pixel_estimate(input_estimate(2., "unvalidated_tiff_resolution_max_axis"), .25)
    assert bigger["inference_area_scale"] == 64


def test_selected_vsi_series_metadata_replaces_tiny_header_dimensions(tmp_path):
    image = tmp_path / "sample.vsi"
    image.write_bytes(b"tiny header")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "schema_version": 1,
        "inputs": [{
            "path": str(image),
            "width_px": 159_319,
            "height_px": 129_791,
            "physical_size_x_um": 0.1720394092,
            "physical_size_y_um": 0.1720398793,
            "backend": "bioformats_unflattened_series",
        }],
    }))
    args = storage.parser().parse_args([
        "--input", str(image), "--input-metadata-json", str(metadata),
        "--outdir", str(tmp_path / "out"), "--workdir", str(tmp_path / "work"),
        "--output-json", str(tmp_path / "report.json"),
        "--start-point", "convert", "--end-point", "gigatime",
        "--gigatime-output-format", "none", "--gigatime-export-ometiff", "false",
        "--min-free-gib", "0",
    ])
    report = storage.build_report(args)
    estimate = report["inputs"][0]
    assert estimate["width_px"] == 159_319
    assert estimate["height_px"] == 129_791
    assert estimate["dimension_source"] == "bioformats_unflattened_series"
    assert estimate["active_rgb_bytes"] == 159_319 * 129_791 * 3
    inference = report["inference_scale_estimates"][0]
    assert inference["inference_area_scale"] == 1
    assert inference["reason"] == "selected_series_metadata_without_downsampling_credit"
    item = storage.InputEstimate(**estimate)
    expected_cells = math.ceil(159_319 * 129_791 * 0.1720398793 ** 2 * 10_000 / 1e6)
    assert storage.estimated_cell_count([item], 10_000, 1.0) == expected_cells


def test_input_metadata_rejects_invalid_dimensions(tmp_path):
    metadata = tmp_path / "bad.json"
    metadata.write_text(json.dumps({
        "schema_version": 1,
        "inputs": [{"path": "sample.vsi", "width_px": 0, "height_px": 20}],
    }))
    with pytest.raises(ValueError, match="positive integers"):
        storage.load_input_metadata(metadata)


def test_metadata_probe_rejects_thousand_mpp_without_decoding(tmp_path):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "stale.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8), resolution=(10, 10), resolutionunit="CENTIMETER")
    mpp, origin = storage.probe_source_mpp(path)
    assert mpp is None
    assert origin == "implausible_tiff_mpp"


def test_physical_three_compartments_include_scratch_and_legacy_is_smaller():
    physical = storage.compartment_storage_model(1000)
    legacy = storage.compartment_storage_model(1000, physical=False)
    assert physical["compartment_count"] == 3
    assert physical["output_file_count_allowance"] == 4
    assert physical["published_bytes"] == 16_000
    assert physical["scratch_bytes"] == 4000
    assert physical["published_bytes"] == legacy["published_bytes"] * 4


def test_optional_profiles_count_feature_copies_and_radius_graphs():
    kwargs = dict(items=[input_estimate(.25, "explicit_override")], markers=True,
                  cellvit=False, radii_um=[25., 50.], density_per_mm2=10000., unknown_source_mpp=1.)
    base = storage.cell_profile_storage_model(uni2=False, **kwargs)
    full = storage.cell_profile_storage_model(uni2=True, **kwargs)
    assert full["estimated_cells"] == 625
    assert full["feature_arrays_bytes"] - base["feature_arrays_bytes"] == 625 * 2 * 1536 * 4
    assert full["graph_bytes"] > 0
    assert full["retained_work_bytes"] == full["published_bytes"] * 2
    assert full['neighborhood_dimensions_allowance'] == full['feature_dimensions_allowance'] * 3
    assert full['neighborhood_feature_tables_bytes'] == 625 * full['neighborhood_dimensions_allowance'] * 40
    assert full['scalar_tables_bytes'] > full['neighborhood_feature_tables_bytes']


def test_cohort_storage_accounts_for_new_outputs_without_copying_profiles(tmp_path, monkeypatch):
    image=tmp_path/'sample.tif'; image.write_bytes(b'probe')
    monkeypatch.setattr(storage,'probe_dimensions',lambda _:(1000,1000,'test'))
    args=storage.parser().parse_args(['--input',str(image),'--outdir',str(tmp_path),
        '--workdir',str(tmp_path),'--output-json',str(tmp_path/'report.json'),
        '--start-point','convert','--end-point','cluster_geojson',
        '--source-mpp','.25','--cell-profiles-enabled','true','--cohort-niches-enabled','true'])
    models=storage.build_report(args)['stage_storage_models']
    cohort=models['cohort_niches']
    assert cohort['estimated_cells']==models['cell_profiles']['estimated_cells']
    assert cohort['assignment_bytes']==cohort['estimated_cells']*2048
    assert cohort['published_bytes']>cohort['assignment_bytes']+cohort['model_bytes']
    assert cohort['work_bytes']==cohort['published_bytes']
    assert cohort['source_profiles_copied'] is False
    args.cell_profiles_enabled='false'
    with pytest.raises(ValueError,match='requires cell profiles'):
        storage.build_report(args)


def test_array_profiles_budget_neighbor_payloads_and_scaled_scratch():
    kwargs = dict(items=[input_estimate(.25, "explicit_override")], markers=True, uni2=True,
                  cellvit=True, radii_um=[25., 50.], density_per_mm2=10000., unknown_source_mpp=1.)
    wide = storage.cell_profile_storage_model(**kwargs)
    arrays = storage.cell_profile_storage_model(**kwargs, feature_storage="arrays")
    n, d = arrays['estimated_cells'], arrays['feature_dimensions_allowance']
    assert arrays['neighborhood_feature_tables_bytes'] == 0
    assert arrays['neighborhood_feature_arrays_bytes'] == n * d * 2 * 8
    assert arrays['niche_scaled_scratch_bytes'] == n * d * 3 * 2 * 8
    assert arrays['work_bytes'] == arrays['retained_work_bytes'] + arrays['niche_scaled_scratch_bytes']
    assert arrays['published_bytes'] < wide['published_bytes']
    assert arrays['feature_arrays_bytes'] == wide['feature_arrays_bytes']


def test_array_cohort_and_export_include_all_scaled_and_identity_copies(tmp_path, monkeypatch):
    image = tmp_path / 'sample.tif'; image.write_bytes(b'probe')
    monkeypatch.setattr(storage, 'probe_dimensions', lambda _: (1000, 1000, 'test'))
    args = storage.parser().parse_args(['--input', str(image), '--outdir', str(tmp_path),
        '--workdir', str(tmp_path), '--output-json', str(tmp_path / 'report.json'),
        '--start-point', 'convert', '--end-point', 'cluster_geojson', '--source-mpp', '.25',
        '--cell-profiles-enabled', 'true', '--cohort-niches-enabled', 'true',
        '--cell-profiles-spatialdata', 'true', '--cell-neighborhood-feature-storage', 'arrays'])
    models = storage.build_report(args)['stage_storage_models']
    profile, cohort, export = (models[name] for name in ('cell_profiles', 'cohort_niches', 'spatialdata'))
    assert cohort['scratch_bytes'] == profile['niche_scaled_scratch_bytes'] > 0
    assert cohort['work_bytes'] == cohort['published_bytes'] + cohort['scratch_bytes']
    assert export['array_own_group_export_bytes'] == profile['estimated_cells'] * profile['feature_dimensions_allowance'] * 8
    assert export['profile_copies_bytes'] == profile['published_bytes'] + export['array_own_group_export_bytes']
    assert '--cohort-niches-enabled' in (ROOT/'lib/StoragePreflight.groovy').read_text()


def test_report_defaults_use_full_float_panel_and_explicit_models(tmp_path, monkeypatch):
    image = tmp_path / "sample.tif"
    image.write_bytes(b"stub")
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 1000, "test"))
    args = storage.parser().parse_args(["--input", str(image), "--outdir", str(tmp_path),
        "--workdir", str(tmp_path), "--output-json", str(tmp_path / "report.json"),
        "--start-point", "cytoplasm", "--end-point", "marker_quantification",
        "--gigatime-output-channels", "", "--source-mpp", ".25", "--cell-profiles-enabled", "true"])
    report = storage.build_report(args)
    assert report["schema_version"] == 2
    assert report["configuration"]["gigatime_channel_count"] == 23
    assert report["configuration"]["gigatime_output_dtype"] == "float32"
    models = report["stage_storage_models"]
    assert models["gigatime"]["level0_bytes"] >= 1_000_000 * 23 * 4
    assert models["marker_quantification"]["channel_count"] == 23
    assert models["marker_quantification"]["compartment_count"] == 3
    assert models["cytoplasm"]["compartment_count"] == 3
    assert "cell_profiles" in models
    assert any("not measured" in value for value in report["limitations"])
    args.cell_profiles_spatialdata = "true"
    args.spatialdata_pyramid_levels = 2
    exported = storage.build_report(args)["stage_storage_models"]["spatialdata"]
    assert exported["native_raster_bytes"] == 1_000_000 * 7 * (1 + .25 + .0625)
    assert exported["profile_copies_bytes"] == models["cell_profiles"]["published_bytes"]
    assert exported["retained_work_bytes"] == exported["published_bytes"]
    args.source_mpp = 1000.
    with pytest.raises(ValueError, match="verify calibration"):
        storage.build_report(args)


def hierarchy_options(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"stub":true}')
    (snapshot / "model.safetensors").write_bytes(b"fixture, not model weights")
    return dict(model_tile_size=224, inner_size=90, target_mpp=.25, unknown_source_mpp=1.,
        max_components=1_000_000, regions_per_grid=4., model_snapshot=str(snapshot),
        weights_filename="", device="cpu", feature_batch=16, max_window_pixels=16_777_216,
        fit_limit=5000, components_per_block=64)


def test_hierarchy_retains_two_full_float_blocks_rasters_and_float64_sums(tmp_path):
    options = hierarchy_options(tmp_path)
    model = storage.hierarchy_storage_model([input_estimate(.25, "explicit_override")], **options)
    assert model["grid_estimates"][0]["stride_source_pixels"] == 90
    assert model["estimated_grid_rows"] >= 12 * 12  # ceil(1000/90) on each axis
    assert model["grid_feature_bytes"] == model["estimated_grid_rows"] * 2 * 1536 * 4
    assert model["region_feature_bytes"] == model["region_count_allowance"] * 2 * 1536 * 4
    assert model["region_float64_sums_bytes"] == model["region_feature_bytes"] * 2
    assert model["native_raster_bytes"] == 1_000_000 * 14
    assert model["alignment_scratch_bytes"] == model["grid_feature_bytes"]
    assert model["work_bytes"] == model["published_bytes"] + model["scratch_bytes"] + model["staged_model_copy_allowance_bytes"]
    assert model["worst_case_work_extra_bytes"] > model["worst_case_output_extra_bytes"] > 0


def test_hierarchy_unknown_mpp_is_conservative_and_component_cap_is_explicit(tmp_path):
    options = hierarchy_options(tmp_path)
    known = storage.hierarchy_storage_model([input_estimate(.25, "explicit_override")], **options)
    stale = storage.hierarchy_storage_model([input_estimate(.25, "unvalidated_ome_metadata_max_axis")], **options)
    assert stale["estimated_grid_rows"] > known["estimated_grid_rows"]
    assert stale["grid_estimates"][0]["planning_source_mpp"] == 1.
    limited = storage.hierarchy_storage_model([input_estimate(.25, "explicit_override")], **{**options, "max_components": 10})
    assert limited["region_count_allowance"] == limited["region_count_algorithm_limit"] == 10
    assert limited["worst_case_output_extra_bytes"] == limited["worst_case_work_extra_bytes"] == 0


def test_hierarchy_uses_existing_local_checkpoint_not_download_allowance(tmp_path):
    options = hierarchy_options(tmp_path)
    model = storage.hierarchy_storage_model([input_estimate()], **options)
    local = model["local_model_snapshot"]
    assert local["additional_download_bytes"] == 0
    assert local["existing_selected_bytes"] == sum(Path(path).stat().st_size for path in local["selected_files"])
    assert model["resource_planning"]["model_ram_allowance_bytes"] == local["existing_selected_bytes"] * 2
    assert "tissue_hierarchy" not in storage.CACHE_STAGES["hf"]
    with pytest.raises(ValueError, match="explicit local model"):
        storage.hierarchy_storage_model([input_estimate()], **{**options, "model_snapshot": ""})
    with pytest.raises(ValueError, match="will not download"):
        storage.hierarchy_storage_model([input_estimate()], **{**options, "weights_filename": "missing.bin"})


def measured_storage_fixture(tmp_path, *, unit="cell"):
    """Size/header-only fixture; intentionally not a biologically validated package."""
    package = tmp_path / "assay"
    package.mkdir()
    files = {name: "0" * 64 for name in ("measured_values.npy", "measured_observations.parquet",
        "measured_observations.csv", "measured_rows.csv")}
    for name in files:
        (package / name).write_bytes(b"size fixture")
    manifest = {"schema_version": "cellphenotyper.measured_assay.v1", "sample_id": "sample_A", "assay_id": "assay_1",
        "observation_unit": unit, "observation_count": 3, "matrix": {"shape": [3, 2], "dtype": "float64"}, "files": files}
    (package / "measured_assay_manifest.json").write_text(json.dumps(manifest))
    row = {"sample_id": "sample_A", "measured_assays": ["assay"], "measured_region_shapes": []}
    config = tmp_path / "measured.json"
    config.write_text(json.dumps([row]))
    return config, row, package


def test_measured_export_counts_all_rows_precision_tables_and_staging(tmp_path):
    config, _, package = measured_storage_fixture(tmp_path)
    model = storage.measured_export_storage_model(config)
    record = model["packages"][0]
    assert record["uncompressed_matrix_bytes"] == 3 * 2 * 8
    assert record["table_allowance_bytes"] == 3 * 4096
    assert record["path"] == str(package)
    assert model["published_bytes"] == record["published_copy_allowance_bytes"]
    assert model["work_bytes"] == model["published_bytes"] + model["staged_input_copy_allowance_bytes"]
    assert model["staged_input_copy_allowance_bytes"] == sum(path.stat().st_size for path in package.iterdir())
    assert "export separately verifies hashes" in model["limitations"]


def test_measured_region_geometry_sizes_resolve_relative_to_link(tmp_path):
    config, row, _ = measured_storage_fixture(tmp_path, unit="spatial_bin")
    with pytest.raises(ValueError, match="requires one explicit"):
        storage.measured_export_storage_model(config)
    bundle = tmp_path / "shape_bundle"
    bundle.mkdir()
    records = {}
    for name in ("shapes", "registry_table", "registry_manifest"):
        path = bundle / f"{name}.data"
        path.write_bytes(b"size fixture " * 30)
        records[name] = {"path": path.name, "sha256": "0" * 64}
    link = {"schema_version": "cellphenotyper.measured_region_shapes.v1", "sample_id": "sample_A", "assay_id": "assay_1", **records}
    (bundle / "link.json").write_text(json.dumps(link))
    row["measured_region_shapes"] = ["shape_bundle"]
    config.write_text(json.dumps([row]))
    model = storage.measured_export_storage_model(config)
    assert model["shape_bundles"][0]["published_geometry_allowance_bytes"] == (bundle / "shapes.data").stat().st_size * 4
    row["measured_region_shapes"].append("shape_bundle")
    config.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match="Duplicate measured shape bundle"):
        storage.measured_export_storage_model(config)


@pytest.mark.parametrize("mutation,error", [("wrong_sample", "identity mismatch"), ("duplicate", "Duplicate measured package"),
    ("float32", "precision"), ("escape", "escapes")])
def test_measured_storage_rejects_ambiguous_or_incompatible_declarations(tmp_path, mutation, error):
    config, row, package = measured_storage_fixture(tmp_path)
    manifest_path = package / "measured_assay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "wrong_sample":
        manifest["sample_id"] = "other"
    elif mutation == "duplicate":
        row["measured_assays"].append("assay")
    elif mutation == "float32":
        manifest["matrix"]["dtype"] = "float32"
    elif mutation == "escape":
        manifest["files"]["../measured.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    config.write_text(json.dumps([row]))
    with pytest.raises(ValueError, match=error):
        storage.measured_export_storage_model(config)


def test_report_applies_hierarchy_fragmentation_and_measured_copies_and_fails_low_space(tmp_path, monkeypatch):
    options = hierarchy_options(tmp_path)
    config, _, _ = measured_storage_fixture(tmp_path)
    image = tmp_path / "input.tif"
    image.write_bytes(b"size fixture")
    monkeypatch.setattr(storage, "probe_dimensions", lambda _: (1000, 1000, "test"))
    monkeypatch.setattr(storage, "filesystem_info", lambda path: {"device": "same", "anchor": str(tmp_path),
        "available_bytes": 100, "capacity_bytes": 1000 * storage.GIB})
    args = storage.parser().parse_args(["--input", str(image), "--outdir", str(tmp_path), "--workdir", str(tmp_path),
        "--output-json", str(tmp_path / "report.json"), "--start-point", "clustering", "--end-point", "cluster_mask",
        "--source-mpp", ".25", "--tissue-hierarchy-enabled", "true", "--hierarchy-model-snapshot", options["model_snapshot"],
        "--hierarchy-device", "cpu", "--cell-profiles-enabled", "true", "--cell-profiles-spatialdata", "true",
        "--cell-measured-assays", str(config), "--min-free-gib", "0"])
    report = storage.build_report(args)
    assert report["status"] == "fail"
    models = report["stage_storage_models"]
    output = next(row for row in report["demands"] if row["kind"] == "output")
    work = next(row for row in report["demands"] if row["kind"] == "work")
    assert output["worst_case_bytes"] >= output["expected_bytes"] + sum(m["published_bytes"] for m in models.values()) + models["tissue_hierarchy"]["worst_case_output_extra_bytes"]
    assert work["worst_case_bytes"] >= work["expected_bytes"] + sum(m["work_bytes"] for m in models.values()) + models["tissue_hierarchy"]["worst_case_work_extra_bytes"]
    assert "measured_spatialdata" in models
    args.cell_profiles_spatialdata = "false"
    with pytest.raises(ValueError, match="requires cell profiles and SpatialData"):
        storage.build_report(args)


def test_wrapper_forwards_hierarchy_measured_and_gpu_device_contracts():
    source = (ROOT / "lib/StoragePreflight.groovy").read_text()
    for flag in ("--tissue-hierarchy-enabled", "--hierarchy-model-snapshot", "--hierarchy-weights-filename",
        "--hierarchy-grid-model-tile-size", "--hierarchy-grid-inner-size", "--hierarchy-grid-target-mpp",
        "--hierarchy-max-components", "--hierarchy-feature-batch", "--hierarchy-max-window-pixels",
        "--hierarchy-fit-limit", "--hierarchy-components-per-block", "--hierarchy-device", "--cell-measured-assays"):
        assert flag in source
    assert "new File(measured).canonicalPath" in source
    assert "new File(snapshot).canonicalPath" in source
    config = (ROOT / "nextflow.config").read_text()
    assert "beforeScript = gpu_admission_before_script" in config.split("withLabel: gpu_capable", 1)[1]
    before_script = config.split("def gpu_admission_before_script = {", 1)[1].split("process {", 1)[0]
    hierarchy = before_script.split("processName.endsWith('PREPARE_HIERARCHY_FEATURES')", 1)[1].split("else if", 1)[0]
    assert "params.uni2_gpu_memory_gb as double" in hierarchy
    assert "params.tissue_hierarchy_device" in hierarchy and "!= 'cuda') return ''" in hierarchy
