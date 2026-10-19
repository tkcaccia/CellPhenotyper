#!/usr/bin/env python3
"""Validate a CellPhenotyper YAML/JSON parameter file against its public schema."""

from __future__ import annotations

import argparse
import json
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
    if inner_px > tile_px:
        failures.append("uni2_inner_square_fixed_px cannot exceed uni2_tile_size")

    if params.get("gigatime_skip_background_blocks") is True:
        failures.append(
            "gigatime_skip_background_blocks=true is outside the technically validated "
            "no-skip continuity path; use only as a declared experimental sensitivity run"
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
