#!/usr/bin/env python3
"""Validate a CellPhenotyper YAML/JSON parameter file against its public schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import jsonschema
import yaml


def load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle) if path.suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Parameter file must contain a mapping/object: {path}")
    # YAML 1.1 parsers interpret an unquoted `off` scalar as boolean false.
    if payload.get("storage_preflight_mode") is False:
        payload["storage_preflight_mode"] = "off"
    return payload


def format_path(parts: Any) -> str:
    path = ".".join(str(part) for part in parts)
    return path or "<root>"


def cross_field_errors(params: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    min_mpp = float(params.get("input_resolution_min_mpp", 0.05))
    max_mpp = float(params.get("input_resolution_max_mpp", 0.50))
    if min_mpp >= max_mpp:
        failures.append(
            "input_resolution_min_mpp must be smaller than input_resolution_max_mpp"
        )

    inner_px = int(params.get("uni2_inner_square_fixed_px", 90))
    tile_px = int(params.get("uni2_tile_size", 224))
    grid_stride_px = int(params.get("uni2_grid_stride_px", 0))
    if inner_px > tile_px:
        failures.append("uni2_inner_square_fixed_px cannot exceed uni2_tile_size")
    if grid_stride_px < 0 or grid_stride_px > tile_px:
        failures.append("uni2_grid_stride_px must be zero or no larger than uni2_tile_size")

    if params.get("pathsegmentor_guided_refine_enable") and not params.get(
        "pathsegmentor_enable"
    ):
        failures.append(
            "pathsegmentor_guided_refine_enable requires pathsegmentor_enable"
        )
    if params.get("pathsegmentor_enable"):
        if float(params.get("pathsegmentor_output_mpp", 2.0)) < float(
            params.get("pathsegmentor_target_mpp", 0.25)
        ):
            failures.append(
                "pathsegmentor_output_mpp must be at least pathsegmentor_target_mpp"
            )
        for name in (
            "pathsegmentor_repo",
            "pathsegmentor_config",
            "pathsegmentor_checkpoint",
        ):
            if not str(params.get(name, "")).strip():
                failures.append(f"pathsegmentor_enable requires {name}")

    if params.get("gigatime_skip_background_blocks") is True:
        failures.append(
            "gigatime_skip_background_blocks=true is outside the technically validated "
            "no-skip continuity path; use only as a declared experimental sensitivity run"
        )
    if params.get("medsam_pre_boundary_competition", True):
        initial_temperature = float(params.get("medsam_pre_boundary_initial_temperature", 2.0))
        final_temperature = float(params.get("medsam_pre_boundary_final_temperature", 0.05))
        if final_temperature > initial_temperature:
            failures.append(
                "medsam_pre_boundary_final_temperature cannot exceed "
                "medsam_pre_boundary_initial_temperature"
            )
    target_clusters = int(params.get("cluster_target_clusters", 0))
    if target_clusters >= 2 and not params.get(
        "cluster_forced_count_sensitivity_acknowledged", False
    ):
        failures.append(
            "cluster_target_clusters >=2 requires "
            "cluster_forced_count_sensitivity_acknowledged=true; forced counts are "
            "sensitivity-only and cannot be presented as the graph-derived primary result"
        )
    if params.get('tissue_hierarchy_enable'):
        # There is deliberately no production method selector: a request for
        # legacy K-means must not be silently accepted by the native route.
        if 'tissue_hierarchy_discovery_method' in params:
            failures.append('tissue_hierarchy_discovery_method is not a pipeline parameter; the hierarchy uses native kodama_graph')
        library = params.get('tissue_hierarchy_kodama_r_library')
        if library is not None and (not isinstance(library, str) or not library.strip() or
                library != library.strip() or any(ord(c) < 32 or ord(c) == 127 for c in library)):
            failures.append('tissue_hierarchy_kodama_r_library must be null or an explicit nonempty path without edge whitespace/control characters')
        margin = float(params.get('tissue_hierarchy_min_affinity_margin', .1))
        if not math.isfinite(margin) or not 0 <= margin <= 1:
            failures.append('tissue_hierarchy_min_affinity_margin must be finite in [0,1]')
        if params.get('cell_profiles_enable'):
            if params.get('cell_profiles_domain_source', 'refined') != 'refined':
                failures.append('Linked cell hierarchy requires refined cell-profile domains')
            parent_variant = params.get('tissue_hierarchy_parent_variant')
            if parent_variant and parent_variant != params.get('cluster_primary_variant', 'standard'):
                failures.append('Linked cell hierarchy requires the same primary parent variant')
        if params.get('uni2_sampling_mode', 'grid') not in {'grid', 'both'}:
            failures.append('tissue_hierarchy_enable requires grid or both UNI-2 sampling')
        if not params.get('tissue_hierarchy_model_snapshot'):
            failures.append('tissue_hierarchy_enable requires an explicit local model snapshot')
        if float(params.get('tissue_hierarchy_context_field_um', 224)) <= float(params.get('tissue_hierarchy_local_field_um', 56)):
            failures.append('Hierarchy context field must be strictly wider than its local field')
        fixed = int(params.get('tissue_hierarchy_fixed_k', 0))
        if fixed == 1 or fixed > int(params.get('tissue_hierarchy_max_k', 5)):
            failures.append('Hierarchy fixed K must be 0 (automatic) or 2..max_k')
        if int(params.get('tissue_hierarchy_fit_limit', 5000)) < int(params.get('tissue_hierarchy_min_observations', 20)):
            failures.append('Hierarchy fit_limit must be at least min_observations')
    if params.get('cell_reference_atlas') and not params.get('cell_profiles_enable'):
        failures.append('cell_reference_atlas requires cell_profiles_enable')
    if params.get('cohort_niches_enable') and not params.get('cell_profiles_enable'):
        failures.append('cohort_niches_enable requires cell_profiles_enable in the full pipeline')
    if params.get('cohort_niches_bundle'):
        if params.get('cohort_niches_enable'):
            failures.append('Choose cohort_niches_enable or cohort_niches_bundle, not both')
        if not (params.get('cell_profiles_enable') and params.get('cell_profiles_spatialdata')):
            failures.append('cohort_niches_bundle requires cell profiles and SpatialData export in the full pipeline')
    if int(params.get('cohort_niches_fixed_k', 0)) > int(params.get('cohort_niches_max_k', 8)):
        failures.append('cohort_niches_fixed_k cannot exceed cohort_niches_max_k')
    if params.get('region_reference_atlas') and not params.get('tissue_hierarchy_enable'):
        failures.append('region_reference_atlas requires tissue_hierarchy_enable')
    if params.get('cell_measured_assays') and not (params.get('cell_profiles_enable') and params.get('cell_profiles_spatialdata')):
        failures.append('cell_measured_assays requires cell profiles and SpatialData export')
    return failures


def validate(params_path: Path, schema_path: Path) -> None:
    params = load_mapping(params_path)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    validator = jsonschema.Draft7Validator(schema)
    failures = [
        f"{format_path(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(params),
            key=lambda error: (list(error.absolute_path), error.message),
        )
    ]
    if not failures:
        failures.extend(cross_field_errors(params))
    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures)
        raise ValueError(f"Parameter validation failed:\n{formatted}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "nextflow_schema.json",
    )
    args = parser.parse_args()
    try:
        validate(args.params, args.schema)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(f"[OK] Parameters satisfy {args.schema}: {args.params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
