#!/usr/bin/env python3
"""Smooth multiclass internal boundaries while preserving one polygon coverage.

The algorithm operates only on a predicted multiclass GeoJSON. Shared internal
interfaces are extracted once, smoothed once, and then polygonized together
with the unchanged outer tissue boundary. No reference annotation is accepted
by this program.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.errors import GEOSException
from shapely.geometry import LineString, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry.is_empty:
        return
    if geometry.geom_type in {"LineString", "LinearRing"}:
        yield LineString(geometry.coords)
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from iter_lines(part)


def smooth_line(
    line: LineString,
    coefficient: float,
    *,
    minimum_turn_degrees: float = 0.0,
    minimum_adjacent_length: float = 0.0,
) -> tuple[LineString, float]:
    """Apply one selective endpoint-preserving pass without adding vertices."""
    points = np.asarray(line.coords, dtype=np.float64)
    if points.shape[0] < 3 or coefficient == 0:
        return line, 0.0
    closed = bool(np.allclose(points[0], points[-1]))
    if closed:
        core = points[:-1]
        if core.shape[0] < 3:
            return line, 0.0
        incoming = core - np.roll(core, 1, axis=0)
        outgoing = np.roll(core, -1, axis=0) - core
        target = 0.5 * (np.roll(core, 1, axis=0) + np.roll(core, -1, axis=0))
        updated = core.copy()
        eligible = eligible_turns(
            incoming,
            outgoing,
            minimum_turn_degrees=minimum_turn_degrees,
            minimum_adjacent_length=minimum_adjacent_length,
        )
        updated[eligible] += float(coefficient) * (target[eligible] - core[eligible])
        updated = np.vstack((updated, updated[0]))
    else:
        updated = points.copy()
        incoming = points[1:-1] - points[:-2]
        outgoing = points[2:] - points[1:-1]
        target = 0.5 * (points[:-2] + points[2:])
        eligible = eligible_turns(
            incoming,
            outgoing,
            minimum_turn_degrees=minimum_turn_degrees,
            minimum_adjacent_length=minimum_adjacent_length,
        )
        interior = updated[1:-1]
        interior[eligible] += float(coefficient) * (
            target[eligible] - points[1:-1][eligible]
        )
        updated[1:-1] = interior
    displacement = float(np.linalg.norm(updated - points, axis=1).max(initial=0.0))
    return LineString(updated), displacement


def eligible_turns(
    incoming: np.ndarray,
    outgoing: np.ndarray,
    *,
    minimum_turn_degrees: float,
    minimum_adjacent_length: float,
) -> np.ndarray:
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    valid = (incoming_length > 1e-9) & (outgoing_length > 1e-9)
    cosine = np.ones(incoming_length.shape, dtype=np.float64)
    cosine[valid] = np.sum(incoming[valid] * outgoing[valid], axis=1) / (
        incoming_length[valid] * outgoing_length[valid]
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return (
        valid
        & (angles >= float(minimum_turn_degrees))
        & (incoming_length >= float(minimum_adjacent_length))
        & (outgoing_length >= float(minimum_adjacent_length))
    )


def smooth_shared_boundary_coverage(
    labeled_geometries: list[tuple[int, object]],
    *,
    smoothing_coefficient: float,
    smoothing_passes: int = 1,
    minimum_turn_degrees: float = 0.0,
    minimum_adjacent_length: float = 0.0,
) -> tuple[list[tuple[int, object]], dict]:
    """Return a mutually exclusive coverage with smoothed internal interfaces."""
    if not 0 <= float(smoothing_coefficient) < 1:
        raise ValueError("Smoothing coefficient must be in [0, 1)")
    if int(smoothing_passes) < 1:
        raise ValueError("Smoothing passes must be positive")
    if not 0 <= float(minimum_turn_degrees) <= 180:
        raise ValueError("Minimum turn angle must be in [0, 180]")
    if float(minimum_adjacent_length) < 0:
        raise ValueError("Minimum adjacent length must be nonnegative")
    labels = [int(label) for label, _ in labeled_geometries]
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("Expected at least two unique dissolved class geometries")
    geometries = [geometry for _, geometry in labeled_geometries]
    if any(geometry.is_empty or not geometry.is_valid for geometry in geometries):
        raise ValueError("Input class geometries must be nonempty and valid")
    for index, left in enumerate(geometries):
        for right in geometries[index + 1:]:
            if left.intersection(right).area > 1e-6:
                raise ValueError("Input class geometries must be mutually exclusive")

    tissue = unary_union(geometries)
    internal = []
    for index, left in enumerate(geometries):
        for right in geometries[index + 1:]:
            shared = left.boundary.intersection(right.boundary)
            shared_lines = list(iter_lines(shared))
            if len(shared_lines) > 1:
                try:
                    shared_lines = list(iter_lines(linemerge(unary_union(shared_lines))))
                except (GEOSException, TypeError, ValueError):
                    pass
            internal.extend(shared_lines)
    if not internal or float(smoothing_coefficient) == 0:
        return list(labeled_geometries), {
            "applied": False,
            "shared_line_count": len(internal),
            "maximum_vertex_displacement_px": 0.0,
        }

    smoothed = internal
    maximum_displacement = 0.0
    for _ in range(int(smoothing_passes)):
        next_lines = []
        for line in smoothed:
            result, displacement = smooth_line(
                line,
                float(smoothing_coefficient),
                minimum_turn_degrees=float(minimum_turn_degrees),
                minimum_adjacent_length=float(minimum_adjacent_length),
            )
            next_lines.append(result)
            maximum_displacement = max(maximum_displacement, displacement)
        smoothed = next_lines

    network = unary_union([tissue.boundary, *smoothed])
    pieces = [piece for piece in polygonize(network) if tissue.covers(piece.representative_point())]
    if not pieces:
        raise RuntimeError("Smoothed linework did not polygonize inside the tissue coverage")

    assigned = {label: [] for label in labels}
    for piece in pieces:
        point = piece.representative_point()
        matches = [
            label for label, geometry in labeled_geometries
            if geometry.covers(point)
        ]
        if len(matches) == 1:
            assigned[matches[0]].append(piece)
            continue
        areas = [float(piece.intersection(geometry).area) for geometry in geometries]
        assigned[labels[int(np.argmax(areas))]].append(piece)

    result = []
    for label in labels:
        if not assigned[label]:
            raise RuntimeError(f"Smoothing removed class {label}")
        geometry = unary_union(assigned[label])
        if geometry.is_empty or not geometry.is_valid:
            raise RuntimeError(f"Smoothing produced invalid class geometry for {label}")
        result.append((label, geometry))

    output_union = unary_union([geometry for _, geometry in result])
    coverage_difference = float(tissue.symmetric_difference(output_union).area)
    overlap_area = 0.0
    for index, (_, left) in enumerate(result):
        for _, right in result[index + 1:]:
            overlap_area += float(left.intersection(right).area)
    tolerance = max(1e-6, float(tissue.area) * 1e-12)
    if coverage_difference > tolerance or overlap_area > tolerance:
        raise RuntimeError("Smoothed boundaries do not form an exclusive complete coverage")
    return result, {
        "applied": True,
        "method": "shared_line_endpoint_preserving_laplacian",
        "smoothing_coefficient": float(smoothing_coefficient),
        "smoothing_passes": int(smoothing_passes),
        "minimum_turn_degrees": float(minimum_turn_degrees),
        "minimum_adjacent_length_px": float(minimum_adjacent_length),
        "shared_line_count": len(internal),
        "maximum_vertex_displacement_px": maximum_displacement,
        "outer_tissue_boundary_unchanged": True,
        "coverage_symmetric_difference_area_px2": coverage_difference,
        "multiclass_overlap_area_px2": overlap_area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficient", type=float, required=True)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--minimum-turn-degrees", type=float, default=0.0)
    parser.add_argument("--minimum-adjacent-length", type=float, default=0.0)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    by_label = {}
    properties = {}
    for feature in payload.get("features", []):
        label = int(feature.get("properties", {}).get("value"))
        by_label.setdefault(label, []).append(shape(feature["geometry"]))
        properties.setdefault(label, copy.deepcopy(feature.get("properties", {})))
    labeled = [(label, unary_union(parts)) for label, parts in sorted(by_label.items())]
    smoothed, diagnostics = smooth_shared_boundary_coverage(
        labeled,
        smoothing_coefficient=args.coefficient,
        smoothing_passes=args.passes,
        minimum_turn_degrees=args.minimum_turn_degrees,
        minimum_adjacent_length=args.minimum_adjacent_length,
    )
    output = copy.deepcopy(payload)
    output["features"] = [
        {
            "type": "Feature",
            "properties": properties[label],
            "geometry": mapping(geometry),
        }
        for label, geometry in smoothed
    ]
    output.setdefault("cellphenotyper_provenance", {})["shared_boundary_smoothing"] = diagnostics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
