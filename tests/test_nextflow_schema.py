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


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate(payload: dict) -> None:
    jsonschema.Draft7Validator(load_schema()).validate(payload)


def test_schema_is_valid_draft7_and_defaults_validate() -> None:
    schema = load_schema()
    jsonschema.Draft7Validator.check_schema(schema)
    params = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    validate(params)


@pytest.mark.parametrize(
    ("intent", "route"),
    [
        ("cell_phenotyping", "grid"),
        ("tissue_domain_discovery", "grid"),
    ],
)
def test_valid_scientific_route_combinations(intent: str, route: str) -> None:
    validate(
        {
            "analysis_intent": intent,
            "uni2_sampling_mode": route,
            "uni2_encoder": "virchow2",
            "uni2_include_inner_square": False,
            "uni2_fuse_tile_inner_square": False,
            "uni2_use_roi_crop": True,
        }
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"analysis_intent": "cell_phenotyping", "uni2_sampling_mode": "cells"},
        {"analysis_intent": "tissue_domain_discovery", "uni2_sampling_mode": "cells"},
        {"analysis_intent": "tissue_domain_discovery", "uni2_sampling_mode": "both"},
        {"analysis_intent": "virtual_staining", "gigatime_enable": False},
        {
            "analysis_intent": "outcome_prediction",
            "pathofmpred_enable": True,
            "pathofmpred_cancer": "",
        },
        {"cell_detection_mode": "consensus", "compute_device": "cpu"},
        {"grandqc_enable": False, "end_point": "stardist"},
        {"gigatime_output_format": "jpg"},
        {"gigatime_output_dtype": "float16"},
        {"full_format": "tiff"},
        {
            "uni2_sampling_mode": "grid",
            "uni2_encoder": "uni2-h",
            "uni2_include_inner_square": True,
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
        "roi_validation_mode",
        "cell_consensus_fusion_acceptance_policy",
        "gigatime_seam_qc_mode",
        "uni2_sampling_mode",
        "cluster_stability_runs",
        "cluster_abstain_uncertain",
        "cluster_forced_count_sensitivity_acknowledged",
        "pathofmpred_cancer",
        "storage_preflight_mode",
        "storage_restart_duplication_factor",
    ):
        assert f'"{name}"' in text


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
