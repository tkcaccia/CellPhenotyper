from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

jsonschema = pytest.importorskip("jsonschema")
yaml = pytest.importorskip("yaml")


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "nextflow_schema.json"
PARAMS_PATH = ROOT / "pipeline_paramers.yml"
sys.path.insert(0, str(ROOT / "bin"))

from validate_pipeline_params import load_mapping  # noqa: E402
from validate_pipeline_params import cross_field_errors


def test_pre_medsam_temperature_schedule_must_cool() -> None:
    assert not cross_field_errors({
        "medsam_pre_boundary_competition": True,
        "medsam_pre_boundary_initial_temperature": 2.0,
        "medsam_pre_boundary_final_temperature": 0.05,
    })
    assert cross_field_errors({
        "medsam_pre_boundary_competition": True,
        "medsam_pre_boundary_initial_temperature": 0.05,
        "medsam_pre_boundary_final_temperature": 2.0,
    })


@pytest.mark.parametrize("space", ["luminance", "lab", "od", "lab_od"])
def test_internal_boundary_gradient_spaces_are_valid(space: str) -> None:
    validate({
        "medsam_internal_gradient_space": space,
        "medsam_internal_gradient_sigma_px": 16.0,
        "medsam_internal_watershed_compactness": 0.001,
    })


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_cell_reference_atlas_requires_enabled_profile_stage():
    assert cross_field_errors({'cell_reference_atlas': '/frozen/atlas'}) == [
        'cell_reference_atlas requires cell_profiles_enable']
    assert not cross_field_errors({'cell_reference_atlas': '/frozen/atlas', 'cell_profiles_enable': True})


def test_external_cohort_bundle_requires_explicit_export_without_refitting():
    options = {'cohort_niches_bundle': '/completed/cohort', 'cell_profiles_enable': True,
               'cell_profiles_spatialdata': True}
    validate(options)
    assert not cross_field_errors(options)
    assert cross_field_errors({**options, 'cell_profiles_enable': False})
    assert cross_field_errors({**options, 'cell_profiles_spatialdata': False})
    assert cross_field_errors({**options, 'cohort_niches_enable': True})


@pytest.mark.parametrize('options', [
    {'cell_neighborhood_feature_storage': 'compressed'},
    {'cell_neighborhood_row_batch_size': 0},
    {'cell_neighborhood_column_batch_size': '7; false'},
    {'cell_niche_fit_limit': 9},
])
def test_array_neighborhood_schema_rejects_invalid_execution_options(options):
    with pytest.raises(jsonschema.ValidationError):
        validate(options)


@pytest.mark.parametrize('value', [False, 1, [], {}])
def test_external_cohort_bundle_schema_rejects_nonpaths(value):
    with pytest.raises(jsonschema.ValidationError):
        validate({'cohort_niches_bundle': value})


def validate(payload: dict) -> None:
    jsonschema.Draft7Validator(load_schema()).validate(payload)


def test_schema_is_valid_draft7_and_defaults_validate() -> None:
    schema = load_schema()
    jsonschema.Draft7Validator.check_schema(schema)
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    validate(params)


def test_grid_is_the_default_uni2_sampling_route() -> None:
    schema = load_schema()
    route = schema["definitions"]["representation_options"]["properties"]["uni2_sampling_mode"]
    assert route["default"] == "grid"
    assert yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))["uni2_sampling_mode"] == "grid"
    assert "uni2_sampling_mode             = 'grid'" in (ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "params.uni2_sampling_mode ?: 'grid'" in (ROOT / "main.nf").read_text(encoding="utf-8")


@pytest.mark.parametrize("storage", ["csv", "binary"])
def test_uni2_storage_schema(storage):
    validate({"uni2_embedding_storage": storage})


@pytest.mark.parametrize("encoder", ["uni2-h", "virchow", "virchow2", "phikon-v2"])
def test_registered_foundation_encoders_are_valid_for_grid_route(encoder: str) -> None:
    validate(
        {
            "analysis_intent": "tissue_domain_discovery",
            "uni2_sampling_mode": "grid",
            "uni2_encoder": encoder,
            "uni2_include_inner_square": True,
            "uni2_fuse_tile_inner_square": True,
            "uni2_use_roi_crop": True,
        }
    )


def test_unregistered_foundation_encoder_is_rejected() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate({"uni2_encoder": "arbitrary/hub-model"})


def test_pathsegmentor_requires_explicit_local_assets_and_gpu_capable_execution() -> None:
    valid = {
        "pathsegmentor_enable": True,
        "compute_device": "gpu",
        "pathsegmentor_repo": "/models/PathSegmentor",
        "pathsegmentor_config": "/models/PathSegmentor/config.yaml",
        "pathsegmentor_checkpoint": "/models/PathSegmentor/model.pt",
    }
    validate(valid)
    assert not cross_field_errors(valid)
    for missing in ("pathsegmentor_repo", "pathsegmentor_config", "pathsegmentor_checkpoint"):
        invalid = dict(valid)
        invalid[missing] = ""
        with pytest.raises(jsonschema.ValidationError):
            validate(invalid)
    with pytest.raises(jsonschema.ValidationError):
        validate({**valid, "compute_device": "cpu"})


def test_pathsegmentor_refinement_is_opt_in_and_output_scale_is_validated() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate({"pathsegmentor_guided_refine_enable": True, "pathsegmentor_enable": False})
    errors = cross_field_errors({
        "pathsegmentor_enable": True,
        "pathsegmentor_repo": "/repo",
        "pathsegmentor_config": "/config",
        "pathsegmentor_checkpoint": "/checkpoint",
        "pathsegmentor_target_mpp": 2.0,
        "pathsegmentor_output_mpp": 0.25,
    })
    assert "pathsegmentor_output_mpp must be at least pathsegmentor_target_mpp" in errors


@pytest.mark.parametrize("storage", ["", "npy", "float16", 1, None])
def test_uni2_storage_schema_rejects_unknown_format(storage):
    with pytest.raises(jsonschema.ValidationError):
        validate({"uni2_embedding_storage": storage})


@pytest.mark.parametrize("mode", ["provided", "brightfield_native"])
def test_neighborhood_support_mode_schema(mode):
    validate({"cell_neighborhood_support_mode": mode})


@pytest.mark.parametrize("mode", ["", "native", "expert", 1, None])
def test_neighborhood_support_mode_schema_rejects_invalid(mode):
    with pytest.raises(jsonschema.ValidationError):
        validate({"cell_neighborhood_support_mode": mode})


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        ("cell_phenotyping", "cells"),
        ("tissue_domain_discovery", "grid"),
        ("tissue_domain_discovery", "both"),
    ],
)
def test_valid_scientific_route_combinations(intent: str, route: str) -> None:
    validate(
        {
            "analysis_intent": intent,
            "uni2_sampling_mode": route,
            "uni2_encoder": "uni2-h",
            "uni2_include_inner_square": True,
            "uni2_fuse_tile_inner_square": True,
            "uni2_use_roi_crop": True,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis_intent": "cell_phenotyping", "uni2_sampling_mode": "grid"},
        {"analysis_intent": "tissue_domain_discovery", "uni2_sampling_mode": "cells"},
        {"analysis_intent": "virtual_staining", "gigatime_enable": False},
        {
            "analysis_intent": "outcome_prediction",
            "pathofmpred_enable": True,
            "pathofmpred_cancer": "",
        },
        {"cell_detection_mode": "consensus", "compute_device": "cpu"},
        {"grandqc_enable": False, "end_point": "stardist"},
        {"gigatime_output_format": "jpg"},
        {"gigatime_output_format": "none", "gigatime_export_ometiff": True},
        {"gigatime_output_dtype": "float16"},
        {"full_format": "tiff"},
        {
            "uni2_sampling_mode": "grid",
            "uni2_encoder": "uni2-h",
            "uni2_include_inner_square": False,
            "uni2_fuse_tile_inner_square": True,
            "uni2_use_roi_crop": True,
        },
    ],
)
def test_invalid_scientific_route_combinations(payload: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)


def test_schema_covers_public_scientific_and_qc_contracts() -> None:
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    for name in (
        "analysis_intent",
        "evidence_gate_mode",
        "input_resolution_max_metadata_conflict_fraction",
        "convert_channel_order",
        "convert_compression",
        "convert_jpeg_quality",
        "hovernet_cache_backend",
        "hovernet_execution_mode",
        "hovernet_stream_batch_tiles",
        "roi_validation_mode",
        "cell_consensus_fusion_acceptance_policy",
        "gigatime_seam_qc_mode",
        "uni2_sampling_mode",
        "cluster_stability_runs",
        "cluster_abstain_uncertain",
        "cluster_forced_count_sensitivity_acknowledged",
        "pathofmpred_cancer",
        "pathsegmentor_guided_refine_enable",
        "storage_preflight_mode",
        "storage_preflight_input_metadata",
        "storage_restart_duplication_factor",
    ):
        assert f'"{name}"' in text


@pytest.mark.parametrize("compression", ["JPEG", "LZW", "DEFLATE", "NONE", "UNCOMPRESSED"])
def test_schema_accepts_documented_input_compression_options(compression: str) -> None:
    validate({"convert_compression": compression, "convert_jpeg_quality": 95})


@pytest.mark.parametrize(
    "payload",
    [
        {"convert_compression": "ZIP"},
        {"convert_jpeg_quality": 0},
        {"convert_jpeg_quality": 101},
    ],
)
def test_schema_rejects_invalid_input_compression_options(payload: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)


@pytest.mark.parametrize("backend", ["zarr", "numpy"])
def test_schema_accepts_documented_hovernet_cache_backends(backend: str) -> None:
    validate({"hovernet_cache_backend": backend})


def test_schema_rejects_unknown_hovernet_cache_backend() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate({"hovernet_cache_backend": "float16"})


@pytest.mark.parametrize("mode", ["streaming_tiles", "wsi"])
def test_schema_accepts_documented_hovernet_execution_modes(mode: str) -> None:
    validate({"hovernet_execution_mode": mode})


@pytest.mark.parametrize("payload", [
    {"hovernet_execution_mode": "unbounded"},
    {"hovernet_stream_batch_tiles": 0},
    {"hovernet_stream_halo": 91},
    {"hovernet_tile_jpeg_quality": 101},
])
def test_schema_rejects_invalid_hovernet_streaming_settings(payload: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)


def test_yaml_unquoted_off_is_normalized_for_storage_preflight(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yml"
    params_path.write_text("storage_preflight_mode: off\n", encoding="utf-8")
    assert load_mapping(params_path)["storage_preflight_mode"] == "off"


@pytest.mark.parametrize(
    "overrides",
    [
        {"input_resolution_min_mpp": 0.5, "input_resolution_max_mpp": 0.5},
        {"uni2_tile_size": 64, "uni2_inner_square_fixed_px": 90},
        {"gigatime_skip_background_blocks": True},
        {
            "cluster_target_clusters": 2,
            "cluster_forced_count_sensitivity_acknowledged": False,
        },
    ],
)
def test_cli_validator_rejects_cross_field_or_experimental_settings(
    tmp_path: Path, overrides: dict
) -> None:
    payload = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    payload.update(overrides)
    params_path = tmp_path / "params.yml"
    params_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "bin" / "validate_pipeline_params.py"),
            "--params",
            str(params_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Parameter validation failed" in result.stderr


def test_forced_cluster_count_requires_explicit_sensitivity_acknowledgement() -> None:
    with pytest.raises(jsonschema.ValidationError):
        validate({"cluster_target_clusters": 2})
    validate(
        {
            "cluster_target_clusters": 2,
            "cluster_forced_count_sensitivity_acknowledged": True,
        }
    )


@pytest.mark.parametrize('overrides', [
    {'uni2_sampling_mode': 'cells'}, {'tissue_hierarchy_model_snapshot': None},
    {'tissue_hierarchy_context_field_um': 56}, {'tissue_hierarchy_fixed_k': 1},
    {'tissue_hierarchy_fixed_k': 6},
])
def test_hierarchy_configuration_rejects_incompatible_routes_and_scales(overrides):
    params = {'tissue_hierarchy_enable': True, 'uni2_sampling_mode': 'grid',
              'tissue_hierarchy_model_snapshot': '/local/snapshot'}
    params.update(overrides)
    assert cross_field_errors(params)


def test_measured_and_region_reference_products_must_be_enabled():
    assert cross_field_errors({'cell_measured_assays': 'assays.json'})
    assert cross_field_errors({'region_reference_atlas': '/atlas'})
    assert not cross_field_errors({'cell_measured_assays': 'assays.json', 'cell_profiles_enable': True, 'cell_profiles_spatialdata': True})


def test_linked_hierarchy_uses_the_same_refined_parent_definition():
    base = {'tissue_hierarchy_enable': True, 'cell_profiles_enable': True,
            'uni2_sampling_mode': 'grid', 'tissue_hierarchy_model_snapshot': '/local/snapshot'}
    assert not cross_field_errors(base)
    assert cross_field_errors({**base, 'cell_profiles_domain_source': 'cluster'})
    assert cross_field_errors({**base, 'tissue_hierarchy_parent_variant': 'another'})
    assert not cross_field_errors({**base, 'tissue_hierarchy_parent_variant': 'another', 'cluster_primary_variant': 'another'})
