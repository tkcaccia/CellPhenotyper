#!/usr/bin/env python3
"""Align StarDist, HoVer-Net and CellViT++ instances into canonical cell IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


DETECTOR_SCOPES = {
    "stardist": {
        "target": "general H&E nuclei",
        "known_non_exhaustive": False,
        "instance_fusion_role": "broad_scope",
    },
    "hovernet": {
        "target": "MoNuSAC-annotated epithelial, lymphocyte, macrophage, and neutrophil nuclei",
        "known_non_exhaustive": True,
        "instance_fusion_role": "scoped_support",
        "known_omissions": [
            "fibroblasts and other nuclei not annotated as positive classes in MoNuSAC",
        ],
        "reference": "https://github.com/simongraham/hovernet_inference#datasets",
    },
    "cellvitpp": {
        "target": "PanNuke five-class nuclei",
        "known_non_exhaustive": False,
        "instance_fusion_role": "broad_scope",
    },
}


@dataclass
class Cell:
    source: str
    source_id: str
    x: float
    y: float
    contour: list[list[float]]
    type_id: object = None
    cell_type: object = None
    probability: object = None


def load_stardist(path: Path) -> list[Cell]:
    result = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result.append(Cell("stardist", str(row["label"]), float(row["x"]), float(row["y"]), []))
    return result


def load_cells(path: Path, source: str, *, source_bytes: bytes | None = None) -> list[Cell]:
    payload = json.loads(path.read_bytes() if source_bytes is None else source_bytes)
    cells = payload.get("cells", payload)
    if isinstance(cells, dict):
        iterable = cells.items()
    else:
        iterable = enumerate(cells)
    result = []
    for fallback_id, item in iterable:
        if not isinstance(item, dict) or "centroid" not in item:
            continue
        centroid = item["centroid"]
        contour = item.get("contour") or []
        result.append(Cell(
            source, str(item.get("id", fallback_id)), float(centroid[0]), float(centroid[1]),
            [[float(p[0]), float(p[1])] for p in contour], item.get("type_id"),
            item.get("type_name", item.get("type")),
            item.get("type_prob", item.get("type_probabilities")),
        ))
    return result


def verify_cellvit_source(path: Path, expected_sha256: str) -> None:
    """Reject a source replacement while its detector-to-canonical map is made."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("CellViT source population changed during canonical fusion")


def cellvit_correspondence(record: dict, source_sha256: str) -> dict:
    """Retain the detector centroid; the canonical fused centroid may differ."""
    cell = record["members"].get("cellvitpp")
    return {} if cell is None else {"cellvitpp_x_px": cell.x, "cellvitpp_y_px": cell.y,
                                    "cellvitpp_source_sha256": source_sha256}


class DisjointSet:
    def __init__(self, cells: list[Cell]):
        self.parent = list(range(len(cells)))
        self.sources = [{cell.source} for cell in cells]

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b or self.sources[a] & self.sources[b]:
            return False
        if len(self.sources[a]) < len(self.sources[b]):
            a, b = b, a
        self.parent[b] = a
        self.sources[a] |= self.sources[b]
        return True


def candidate_edges(cells: list[Cell], by_source: dict[str, list[int]], radius: float) -> list[tuple[float, int, int]]:
    edges = []
    names = sorted(by_source)
    for pos, left_name in enumerate(names):
        left_ids = by_source[left_name]
        if not left_ids:
            continue
        left_xy = np.asarray([(cells[i].x, cells[i].y) for i in left_ids])
        for right_name in names[pos + 1:]:
            right_ids = by_source[right_name]
            if not right_ids:
                continue
            right_xy = np.asarray([(cells[i].x, cells[i].y) for i in right_ids])
            tree = cKDTree(right_xy)
            k = min(4, len(right_ids))
            distances, neighbors = tree.query(left_xy, k=k, distance_upper_bound=radius)
            distances = np.asarray(distances).reshape(len(left_ids), k)
            neighbors = np.asarray(neighbors).reshape(len(left_ids), k)
            for local_left in range(len(left_ids)):
                for dist, local_right in zip(distances[local_left], neighbors[local_left]):
                    if math.isfinite(float(dist)) and int(local_right) < len(right_ids):
                        edges.append((float(dist), left_ids[local_left], right_ids[int(local_right)]))
    return sorted(edges)


def choose_geometry(group: list[Cell], priority: list[str]) -> Cell | None:
    with_contours = [cell for cell in group if len(cell.contour) >= 3]
    if not with_contours:
        return None
    center = np.mean([(cell.x, cell.y) for cell in group], axis=0)
    rank = {name: i for i, name in enumerate(priority)}
    return min(
        with_contours,
        key=lambda cell: (
            rank.get(cell.source, len(rank)),
            np.linalg.norm(np.asarray((cell.x, cell.y)) - center),
            cell.source_id,
        ),
    )


def detector_role(source: str) -> str:
    return str(DETECTOR_SCOPES.get(source, {}).get("instance_fusion_role", "scoped_support"))


def component_role_evidence(group: list[Cell]) -> dict:
    sources = sorted({cell.source for cell in group})
    broad_sources = sorted(source for source in sources if detector_role(source) == "broad_scope")
    scoped_sources = sorted(source for source in sources if detector_role(source) == "scoped_support")
    required_broad = sorted(
        source for source in DETECTOR_SCOPES if detector_role(source) == "broad_scope"
    )
    has_broad_consensus = set(required_broad).issubset(broad_sources)
    if has_broad_consensus and scoped_sources:
        tier = "broad_consensus_with_scoped_support"
    elif has_broad_consensus:
        tier = "broad_consensus"
    elif broad_sources and scoped_sources:
        tier = "single_broad_with_scoped_support"
    elif broad_sources:
        tier = "single_broad_only"
    else:
        tier = "scoped_support_only"
    return {
        "instance_sources": sources,
        "broad_instance_sources": broad_sources,
        "broad_instance_support": len(broad_sources),
        "required_broad_instance_sources": required_broad,
        "has_broad_instance_consensus": has_broad_consensus,
        "scoped_support_sources": scoped_sources,
        "scoped_support_count": len(scoped_sources),
        "instance_evidence_tier": tier,
    }


def component_is_accepted(
    group: list[Cell],
    min_support: int,
    policy: str,
    *,
    mpp: float | None = None,
    match_radius_um: float | None = None,
) -> tuple[bool, str]:
    role_evidence = component_role_evidence(group)
    if len(role_evidence["instance_sources"]) < int(min_support):
        return False, "abstained_insufficient_detector_support"
    if policy == "broad_pair" and not role_evidence["has_broad_instance_consensus"]:
        return False, "abstained_missing_broad_detector_consensus"
    if policy == "broad_pair" and mpp is not None and match_radius_um is not None:
        broad_cells = [cell for cell in group if detector_role(cell.source) == "broad_scope"]
        broad_distances_um = [
            float(math.hypot(a.x - b.x, a.y - b.y) * mpp)
            for index, a in enumerate(broad_cells)
            for b in broad_cells[index + 1 :]
        ]
        if broad_distances_um and max(broad_distances_um) > float(match_radius_um):
            return False, "abstained_broad_detector_distance"
    if policy != "any_two" and policy != "broad_pair":
        raise ValueError(f"Unsupported fusion acceptance policy: {policy}")
    return True, "accepted_instance_fusion"


def summarize_group_agreement(group: list[Cell], mpp: float, match_radius_um: float) -> dict:
    """Describe role-aware detector agreement without implying calibrated accuracy."""
    role_evidence = component_role_evidence(group)
    sources = role_evidence["instance_sources"]
    broad_cells = [cell for cell in group if detector_role(cell.source) == "broad_scope"]
    all_distances_um = [
        float(math.hypot(a.x - b.x, a.y - b.y) * mpp)
        for index, a in enumerate(group)
        for b in group[index + 1 :]
    ]
    broad_distances_um = [
        float(math.hypot(a.x - b.x, a.y - b.y) * mpp)
        for index, a in enumerate(broad_cells)
        for b in broad_cells[index + 1 :]
    ]
    broad_max_distance_um = max(broad_distances_um, default=0.0)
    broad_mean_distance_um = float(np.mean(broad_distances_um)) if broad_distances_um else 0.0
    normalized_distance = broad_max_distance_um / max(float(match_radius_um), 1e-12)
    distance_score = max(0.0, 1.0 - min(1.0, normalized_distance))
    required_broad_count = max(1, len(role_evidence["required_broad_instance_sources"]))
    broad_support_fraction = role_evidence["broad_instance_support"] / required_broad_count
    agreement_score = broad_support_fraction * (0.5 + 0.5 * distance_score)
    if role_evidence["has_broad_instance_consensus"] and agreement_score >= 0.75:
        agreement_tier = "high"
    elif agreement_score >= 0.45:
        agreement_tier = "moderate"
    else:
        agreement_tier = "low"
    phenotype_sources = sorted(
        cell.source
        for cell in group
        if cell.cell_type is not None and str(cell.cell_type).strip()
    )
    return {
        **role_evidence,
        "instance_support": len(sources),
        "centroid_distance_um_mean": broad_mean_distance_um,
        "centroid_distance_um_max": broad_max_distance_um,
        "broad_centroid_distance_um_mean": broad_mean_distance_um,
        "broad_centroid_distance_um_max": broad_max_distance_um,
        "all_source_centroid_distance_um_mean": (
            float(np.mean(all_distances_um)) if all_distances_um else 0.0
        ),
        "all_source_centroid_distance_um_max": max(all_distances_um, default=0.0),
        "agreement_score": float(agreement_score),
        "agreement_tier": agreement_tier,
        "phenotype_evidence_methods": phenotype_sources,
        "phenotype_evidence_count": len(phenotype_sources),
        "phenotype_status": (
            "detector_specific_evidence_only" if phenotype_sources else "no_phenotype_evidence"
        ),
    }


def detector_agreement_benchmark(groups: list[list[Cell]], by_source: dict[str, list[int]], mpp: float) -> list[dict]:
    """Measure inter-detector agreement without presenting it as ground-truth accuracy."""
    from shapely.geometry import Polygon

    rows = []
    names = sorted(by_source)
    for left_pos, source_a in enumerate(names):
        for source_b in names[left_pos + 1:]:
            matched = []
            for group in groups:
                members = {cell.source: cell for cell in group}
                if source_a in members and source_b in members:
                    matched.append((members[source_a], members[source_b]))

            centroid_um = [
                float(math.hypot(a.x - b.x, a.y - b.y) * mpp)
                for a, b in matched
            ]
            ious = []
            hausdorff_um = []
            for a, b in matched:
                if len(a.contour) < 3 or len(b.contour) < 3:
                    continue
                pa, pb = Polygon(a.contour), Polygon(b.contour)
                if not pa.is_valid:
                    pa = pa.buffer(0)
                if not pb.is_valid:
                    pb = pb.buffer(0)
                if pa.is_empty or pb.is_empty or pa.area <= 0 or pb.area <= 0:
                    continue
                union = float(pa.union(pb).area)
                if union <= 0:
                    continue
                ious.append(float(pa.intersection(pb).area / union))
                hausdorff_um.append(float(pa.hausdorff_distance(pb) * mpp))

            def percentile(values, q):
                return float(np.percentile(values, q)) if values else None

            count_a = len(by_source[source_a])
            count_b = len(by_source[source_b])
            rows.append({
                "source_a": source_a,
                "source_b": source_b,
                "source_a_count": int(count_a),
                "source_b_count": int(count_b),
                "matched_pairs": int(len(matched)),
                "source_a_match_fraction": float(len(matched) / max(1, count_a)),
                "source_b_match_fraction": float(len(matched) / max(1, count_b)),
                "centroid_distance_um_median": percentile(centroid_um, 50),
                "centroid_distance_um_p95": percentile(centroid_um, 95),
                "boundary_pairs": int(len(ious)),
                "polygon_iou_median": percentile(ious, 50),
                "polygon_iou_p25": percentile(ious, 25),
                "polygon_iou_p75": percentile(ious, 75),
                "hausdorff_distance_um_median": percentile(hausdorff_um, 50),
                "hausdorff_distance_um_p95": percentile(hausdorff_um, 95),
            })
    return rows


def count_ratio(counts: dict[str, int], names: list[str]) -> float | None:
    values = [counts[name] for name in names if counts.get(name, 0) > 0]
    if len(values) < 2:
        return None
    return float(max(values) / min(values))


def detector_count_qc(by_source: dict[str, list[int]], warning_ratio: float) -> dict:
    """Report count imbalance without treating heterogeneous model scope as accuracy."""
    counts = {source: len(indices) for source, indices in sorted(by_source.items())}
    broad_scope = [
        source for source in counts
        if not DETECTOR_SCOPES.get(source, {}).get("known_non_exhaustive", False)
    ]
    broad_values = [counts[source] for source in broad_scope if counts[source] > 0]
    broad_mean = float(np.mean(broad_values)) if broad_values else None
    all_ratio = count_ratio(counts, list(counts))
    broad_ratio = count_ratio(counts, broad_scope)
    relative = {
        source: (float(count / broad_mean) if broad_mean else None)
        for source, count in counts.items()
    }
    missing_broad = [source for source in broad_scope if counts.get(source, 0) == 0]
    scoped_scope = [source for source in counts if source not in broad_scope]
    missing_scoped = [source for source in scoped_scope if counts.get(source, 0) == 0]
    if missing_broad:
        status = "failed_missing_broad_detector_output"
    elif broad_ratio is not None and broad_ratio > warning_ratio:
        status = "review_broad_scope_detector_imbalance"
    elif missing_scoped:
        status = "review_missing_scoped_support_output"
    elif all_ratio is not None and all_ratio > warning_ratio:
        status = "expected_non_exhaustive_scope_imbalance"
    else:
        status = "within_count_ratio_threshold"
    return {
        "status": status,
        "warning_ratio": float(warning_ratio),
        "counts": counts,
        "all_detector_max_min_ratio": all_ratio,
        "broad_scope_detectors": broad_scope,
        "scoped_support_detectors": scoped_scope,
        "missing_broad_scope_detectors": missing_broad,
        "missing_scoped_support_detectors": missing_scoped,
        "broad_scope_max_min_ratio": broad_ratio,
        "relative_to_broad_scope_mean": relative,
        "detector_scopes": {source: DETECTOR_SCOPES.get(source, {}) for source in counts},
        "interpretation": (
            "Raw detector counts are not accuracy estimates. HoVer-Net MoNuSAC is known to be "
            "non-exhaustive because the training labels omit classes such as fibroblasts."
        ),
    }


def detector_spatial_coverage(cells: list[Cell], width: int, height: int, bins: int = 16) -> dict:
    """Summarize whether count differences are global or caused by missing slide regions."""
    by_source: dict[str, list[Cell]] = defaultdict(list)
    for source in DETECTOR_SCOPES:
        by_source[source]
    for cell in cells:
        by_source[cell.source].append(cell)
    occupied: dict[str, set[tuple[int, int]]] = {}
    rows = {}
    for source, source_cells in sorted(by_source.items()):
        valid = [cell for cell in source_cells if 0 <= cell.x < width and 0 <= cell.y < height]
        source_bins = {
            (
                min(bins - 1, int(cell.x * bins / width)),
                min(bins - 1, int(cell.y * bins / height)),
            )
            for cell in valid
        }
        occupied[source] = source_bins
        rows[source] = {
            "valid_centroids": len(valid),
            "outside_image_centroids": len(source_cells) - len(valid),
            "occupied_bins": len(source_bins),
            "occupied_fraction": float(len(source_bins) / (bins * bins)),
            "centroid_bounds_xyxy": (
                [
                    float(min(cell.x for cell in valid)),
                    float(min(cell.y for cell in valid)),
                    float(max(cell.x for cell in valid)),
                    float(max(cell.y for cell in valid)),
                ] if valid else None
            ),
        }
    reference_bins = occupied.get("stardist", set()) & occupied.get("cellvitpp", set())
    hover_missing = reference_bins - occupied.get("hovernet", set())
    return {
        "grid_bins_xy": [bins, bins],
        "detectors": rows,
        "broad_scope_jointly_occupied_bins": len(reference_bins),
        "broad_scope_bins_without_hovernet": len(hover_missing),
        "broad_scope_bins_without_hovernet_fraction": (
            float(len(hover_missing) / len(reference_bins)) if reference_bins else None
        ),
    }


def allocate_unique_seed_pixels(records: list[dict], width: int, height: int) -> dict[int, tuple[int, int]]:
    """Reserve one unique mask pixel near every consensus centroid.

    Detector contours can overlap completely. Rasterizing them by priority can
    therefore erase an otherwise valid consensus cell. Reserved seed pixels are
    applied after polygon rasterization so every canonical cell ID survives.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid mask dimensions: width={width}, height={height}")

    occupied: set[tuple[int, int]] = set()
    seeds: dict[int, tuple[int, int]] = {}
    for record in records:
        center_x = min(width - 1, max(0, int(round(float(record["x"])))))
        center_y = min(height - 1, max(0, int(round(float(record["y"])))))
        selected = None
        radius = 0
        while selected is None:
            if radius == 0:
                candidates = [(center_x, center_y)]
            else:
                candidates = []
                for dx in range(-radius, radius + 1):
                    candidates.append((center_x + dx, center_y - radius))
                    candidates.append((center_x + dx, center_y + radius))
                for dy in range(-radius + 1, radius):
                    candidates.append((center_x - radius, center_y + dy))
                    candidates.append((center_x + radius, center_y + dy))
            for x, y in candidates:
                if 0 <= x < width and 0 <= y < height and (x, y) not in occupied:
                    selected = (x, y)
                    break
            radius += 1
            if radius > max(width, height):
                raise RuntimeError("Could not reserve a unique consensus seed pixel")
        occupied.add(selected)
        seeds[int(record["label"])] = selected
    return seeds


def write_mask(records: list[dict], width: int, height: int, output: Path, tile_size: int, compression: str) -> dict:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import mapping

    bins: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        polygon = record["polygon"]
        xmin, ymin, xmax, ymax = polygon.bounds
        for ty in range(max(0, int(ymin) // tile_size), min((height - 1) // tile_size, int(ymax) // tile_size) + 1):
            for tx in range(max(0, int(xmin) // tile_size), min((width - 1) // tile_size, int(xmax) // tile_size) + 1):
                bins[(ty, tx)].append(record)
    seed_pixels = allocate_unique_seed_pixels(records, width, height)
    seed_bins: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for label, (x, y) in seed_pixels.items():
        seed_bins[(y // tile_size, x // tile_size)].append((label, x, y))
    for record in records:
        seed_x, seed_y = seed_pixels[int(record["label"])]
        record["mask_seed_x"] = seed_x
        record["mask_seed_y"] = seed_y
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1, "dtype": "uint32",
        "tiled": True, "blockxsize": tile_size, "blockysize": tile_size, "compress": compression,
        "BIGTIFF": "YES", "transform": rasterio.Affine.identity(),
    }
    with rasterio.open(output, "w", **profile) as dst:
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                h, w = min(tile_size, height - y0), min(tile_size, width - x0)
                relevant = sorted(bins.get((y0 // tile_size, x0 // tile_size), []), key=lambda r: r["support"])
                shapes = [(mapping(r["polygon"]), int(r["label"])) for r in relevant]
                tile = rasterize(shapes, out_shape=(h, w), transform=rasterio.Affine.translation(x0, y0), fill=0, dtype="uint32") if shapes else np.zeros((h, w), dtype=np.uint32)
                for label, seed_x, seed_y in seed_bins.get((y0 // tile_size, x0 // tile_size), []):
                    tile[seed_y - y0, seed_x - x0] = label
                dst.write(tile, 1, window=Window(x0, y0, w, h))
    return {
        "reserved_seed_count": len(seed_pixels),
        "relocated_seed_count": sum(
            (x, y) != (
                min(width - 1, max(0, int(round(float(record["x"]))))),
                min(height - 1, max(0, int(round(float(record["y"]))))),
            )
            for record in records
            for x, y in [seed_pixels[int(record["label"])]]
        ),
    }


def validate_mask_label_coverage(path: Path, expected_count: int) -> dict:
    import rasterio

    seen = np.zeros(expected_count + 1, dtype=bool)
    invalid_labels: set[int] = set()
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            values = src.read(1, window=window)
            block_max = int(values.max(initial=0))
            if block_max > expected_count:
                invalid_labels.update(int(v) for v in np.unique(values[values > expected_count]))
            else:
                seen[values.reshape(-1).astype(np.int64, copy=False)] = True
    seen[0] = True
    missing = np.flatnonzero(~seen)
    if invalid_labels or missing.size:
        raise RuntimeError(
            "Consensus mask label coverage failed: "
            f"expected={expected_count}, present={int(seen[1:].sum())}, "
            f"missing={missing[:20].tolist()}, invalid={sorted(invalid_labels)[:20]}"
        )
    return {
        "expected_label_count": int(expected_count),
        "present_label_count": int(seen[1:].sum()),
        "missing_label_count": 0,
        "invalid_label_count": 0,
    }


def write_preview(image_path: Path, records: list[dict], output: Path, max_side: int) -> None:
    import pyvips

    source = pyvips.Image.new_from_file(str(image_path), access="sequential")
    scale = min(1.0, max_side / max(source.width, source.height))
    thumb = source.resize(scale) if scale < 1 else source
    if thumb.bands > 3:
        thumb = thumb[:3]
    if thumb.bands == 1:
        thumb = thumb.bandjoin([thumb, thumb])
    if thumb.format != "uchar":
        thumb = thumb.cast("uchar")
    array = np.ndarray(buffer=thumb.write_to_memory(), dtype=np.uint8, shape=(thumb.height, thumb.width, thumb.bands))
    canvas = Image.fromarray(array[:, :, :3])
    draw = ImageDraw.Draw(canvas)
    colors = {2: (255, 190, 0), 3: (0, 220, 120)}
    for record in records:
        x, y = record["x"] * scale, record["y"] * scale
        r = max(1, int(round(2 * scale)))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=colors.get(record["support"], (255, 255, 255)))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stardist-objects", required=True)
    parser.add_argument("--hovernet-cells", required=True)
    parser.add_argument("--cellvit-cells", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--match-radius-um", type=float, default=4.0)
    parser.add_argument("--default-mpp", type=float, default=0.25)
    parser.add_argument("--geometry-priority", default="cellvitpp,hovernet")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--compression", default="deflate")
    parser.add_argument("--preview-max-side", type=int, default=3000)
    parser.add_argument("--count-ratio-warning", type=float, default=2.0)
    parser.add_argument("--min-agreement-score", type=float, default=0.0)
    parser.add_argument(
        "--fusion-acceptance-policy", choices=("broad_pair", "any_two"), default="broad_pair"
    )
    args = parser.parse_args()

    from shapely.geometry import Polygon, mapping

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shift = json.loads(Path(args.shift).read_text())
    mpp = float(shift.get("source_mpp", shift.get("microns_per_pixel", args.default_mpp)))
    radius_px = args.match_radius_um / mpp
    cells = load_stardist(Path(args.stardist_objects))
    cells += load_cells(Path(args.hovernet_cells), "hovernet")
    cellvit_path = Path(args.cellvit_cells)
    cellvit_bytes = cellvit_path.read_bytes()
    cellvit_source_sha256 = hashlib.sha256(cellvit_bytes).hexdigest()
    cells += load_cells(cellvit_path, "cellvitpp", source_bytes=cellvit_bytes)
    del cellvit_bytes
    verify_cellvit_source(cellvit_path, cellvit_source_sha256)
    by_source: dict[str, list[int]] = defaultdict(list)
    for source in DETECTOR_SCOPES:
        by_source[source]
    for index, cell in enumerate(cells):
        by_source[cell.source].append(index)
    dsu = DisjointSet(cells)
    for _, left, right in candidate_edges(cells, by_source, radius_px):
        dsu.union(left, right)
    groups: dict[int, list[Cell]] = defaultdict(list)
    for index, cell in enumerate(cells):
        groups[dsu.find(index)].append(cell)
    agreement_rows = detector_agreement_benchmark(list(groups.values()), by_source, mpp)
    count_qc = detector_count_qc(by_source, args.count_ratio_warning)
    if count_qc["status"] != "within_count_ratio_threshold":
        print(
            "[WARN] Detector count QC: "
            f"status={count_qc['status']} counts={count_qc['counts']} "
            f"all_ratio={count_qc['all_detector_max_min_ratio']} "
            f"broad_scope_ratio={count_qc['broad_scope_max_min_ratio']}"
        )
    if count_qc["missing_broad_scope_detectors"]:
        raise RuntimeError(
            "Canonical instance fusion requires all broad-scope detectors; missing outputs: "
            + ",".join(count_qc["missing_broad_scope_detectors"])
        )

    priority = [value.strip() for value in args.geometry_priority.split(",") if value.strip()]
    records = []
    rejected_no_geometry = 0
    decision_by_member = {}
    component_decisions = []
    for group in groups.values():
        support = len({cell.source for cell in group})
        agreement = summarize_group_agreement(group, mpp, args.match_radius_um)
        accepted_by_policy, policy_decision = component_is_accepted(
            group,
            args.min_support,
            args.fusion_acceptance_policy,
            mpp=mpp,
            match_radius_um=args.match_radius_um,
        )
        if not accepted_by_policy:
            component_decisions.append(policy_decision)
            for cell in group:
                decision_by_member[(cell.source, cell.source_id)] = {
                    **agreement,
                    "decision": policy_decision,
                }
            continue
        representative = choose_geometry(group, priority)
        if representative is None:
            rejected_no_geometry += 1
            component_decisions.append("abstained_no_valid_geometry")
            for cell in group:
                decision_by_member[(cell.source, cell.source_id)] = {
                    **agreement,
                    "decision": "abstained_no_valid_geometry",
                }
            continue
        polygon = Polygon(representative.contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            rejected_no_geometry += 1
            component_decisions.append("abstained_no_valid_geometry")
            for cell in group:
                decision_by_member[(cell.source, cell.source_id)] = {
                    **agreement,
                    "decision": "abstained_no_valid_geometry",
                }
            continue
        if agreement["agreement_score"] < args.min_agreement_score:
            component_decisions.append("abstained_low_agreement_score")
            for cell in group:
                decision_by_member[(cell.source, cell.source_id)] = {
                    **agreement,
                    "decision": "abstained_low_agreement_score",
                }
            continue
        broad_members = [cell for cell in group if detector_role(cell.source) == "broad_scope"]
        center_members = broad_members or group
        center = np.mean([(cell.x, cell.y) for cell in center_members], axis=0)
        members = {cell.source: cell for cell in group}
        record = {
            "x": float(center[0]),
            "y": float(center[1]),
            "support": support,
            "members": members,
            "polygon": polygon,
            "geometry_source": representative.source,
            **agreement,
        }
        records.append(record)
        component_decisions.append("accepted_instance_fusion")
        for cell in group:
            decision_by_member[(cell.source, cell.source_id)] = {
                **agreement,
                "decision": "accepted_instance_fusion",
                "geometry_source": representative.source,
            }
    records.sort(key=lambda item: (item["y"], item["x"]))
    for label, record in enumerate(records, 1):
        record["label"] = label

    verify_cellvit_source(cellvit_path, cellvit_source_sha256)

    import pyvips
    image = pyvips.Image.new_from_file(args.image, access="sequential")
    spatial_coverage = detector_spatial_coverage(cells, image.width, image.height)
    mask_path = outdir / "labels.tif"
    mask_seed_summary = write_mask(records, image.width, image.height, mask_path, args.tile_size, args.compression)
    mask_coverage = validate_mask_label_coverage(mask_path, len(records))
    columns = [
        "label", "area", "y", "x", "xmin", "ymin", "xmax", "ymax",
        "mask_seed_x", "mask_seed_y", "instance_fusion_status", "consensus_support",
        "consensus_methods", "geometry_source", "agreement_score", "agreement_tier",
        "centroid_distance_um_mean", "centroid_distance_um_max",
        "phenotype_evidence_count", "phenotype_evidence_methods", "phenotype_status",
        "broad_instance_support", "broad_instance_sources", "scoped_support_count",
        "scoped_support_sources", "instance_evidence_tier",
    ]
    for source in ("stardist", "hovernet", "cellvitpp"):
        columns += [
            f"{source}_id", f"{source}_type_id", f"{source}_type",
            f"{source}_probability",
        ]
    columns += ["cellvitpp_x_px", "cellvitpp_y_px", "cellvitpp_source_sha256"]
    with (outdir / "objects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            xmin, ymin, xmax, ymax = record["polygon"].bounds
            row = {
                "label": record["label"], "area": record["polygon"].area, "y": record["y"], "x": record["x"],
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "mask_seed_x": record["mask_seed_x"],
                "mask_seed_y": record["mask_seed_y"],
                "instance_fusion_status": "accepted_instance_fusion",
                "consensus_support": record["support"], "consensus_methods": ";".join(sorted(record["members"])),
                "geometry_source": record["geometry_source"],
                "agreement_score": record["agreement_score"],
                "agreement_tier": record["agreement_tier"],
                "centroid_distance_um_mean": record["centroid_distance_um_mean"],
                "centroid_distance_um_max": record["centroid_distance_um_max"],
                "phenotype_evidence_count": record["phenotype_evidence_count"],
                "phenotype_evidence_methods": ";".join(record["phenotype_evidence_methods"]),
                "phenotype_status": record["phenotype_status"],
                "broad_instance_support": record["broad_instance_support"],
                "broad_instance_sources": ";".join(record["broad_instance_sources"]),
                "scoped_support_count": record["scoped_support_count"],
                "scoped_support_sources": ";".join(record["scoped_support_sources"]),
                "instance_evidence_tier": record["instance_evidence_tier"],
            }
            for source, cell in record["members"].items():
                row[f"{source}_id"] = cell.source_id
                row[f"{source}_type_id"] = cell.type_id
                row[f"{source}_type"] = cell.cell_type
                row[f"{source}_probability"] = json.dumps(cell.probability) if cell.probability is not None else ""
            row.update(cellvit_correspondence(record, cellvit_source_sha256))
            writer.writerow(row)
    accepted_assignment = {
        (source, cell.source_id): record["label"]
        for record in records
        for source, cell in record["members"].items()
    }
    method_combination_counts = defaultdict(int)
    for record in records:
        method_combination_counts[";".join(sorted(record["members"]))] += 1
    detector_assignment_counts = {}
    for source, indices in sorted(by_source.items()):
        accepted = sum((source, cells[index].source_id) in accepted_assignment for index in indices)
        detector_assignment_counts[source] = {
            "accepted_in_consensus": int(accepted),
            "not_in_canonical_fusion": int(len(indices) - accepted),
            "acceptance_fraction": float(accepted / max(1, len(indices))),
        }
    component_support = {}
    for group in groups.values():
        support = len({cell.source for cell in group})
        for cell in group:
            component_support[(cell.source, cell.source_id)] = support
    with (outdir / "alignment.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source", "source_id", "x", "y", "component_support", "consensus_label",
            "accepted", "decision", "agreement_score", "agreement_tier",
            "centroid_distance_um_mean", "centroid_distance_um_max", "geometry_source",
            "phenotype_status", "broad_instance_support", "scoped_support_count",
            "instance_evidence_tier",
        ])
        writer.writeheader()
        for cell in cells:
            key = (cell.source, cell.source_id)
            label = accepted_assignment.get(key, "")
            decision = decision_by_member.get(key, {"decision": "abstained_unresolved"})
            writer.writerow({
                "source": cell.source, "source_id": cell.source_id, "x": cell.x, "y": cell.y,
                "component_support": component_support.get(key, 1), "consensus_label": label,
                "accepted": bool(label),
                "decision": decision["decision"],
                "agreement_score": decision.get("agreement_score"),
                "agreement_tier": decision.get("agreement_tier"),
                "centroid_distance_um_mean": decision.get("centroid_distance_um_mean"),
                "centroid_distance_um_max": decision.get("centroid_distance_um_max"),
                "geometry_source": decision.get("geometry_source", ""),
                "phenotype_status": decision.get("phenotype_status", "no_phenotype_evidence"),
                "broad_instance_support": decision.get("broad_instance_support", 0),
                "scoped_support_count": decision.get("scoped_support_count", 0),
                "instance_evidence_tier": decision.get("instance_evidence_tier", "unresolved"),
            })
    benchmark_columns = [
        "source_a", "source_b", "source_a_count", "source_b_count", "matched_pairs",
        "source_a_match_fraction", "source_b_match_fraction",
        "centroid_distance_um_median", "centroid_distance_um_p95", "boundary_pairs",
        "polygon_iou_median", "polygon_iou_p25", "polygon_iou_p75",
        "hausdorff_distance_um_median", "hausdorff_distance_um_p95",
    ]
    with (outdir / "detector_agreement_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark_columns)
        writer.writeheader()
        writer.writerows(agreement_rows)
    (outdir / "detector_agreement_benchmark.json").write_text(json.dumps({
        "interpretation": "Inter-detector agreement; not accuracy against an independent reference standard.",
        "source_mpp": mpp,
        "detector_count_qc": count_qc,
        "detector_spatial_coverage": spatial_coverage,
        "pairs": agreement_rows,
    }, indent=2))
    with (outdir / "consensus_cells.geojson").open("w") as handle:
        handle.write('{"type":"FeatureCollection","features":[')
        for index, record in enumerate(records):
            if index:
                handle.write(",")
            properties = {
                "id": record["label"],
                "instance_fusion_status": "accepted_instance_fusion",
                "support": record["support"],
                "methods": sorted(record["members"]),
                "geometry_source": record["geometry_source"],
                "agreement_score": record["agreement_score"],
                "agreement_tier": record["agreement_tier"],
                "phenotype_status": record["phenotype_status"],
                "phenotype_evidence_methods": record["phenotype_evidence_methods"],
                "broad_instance_support": record["broad_instance_support"],
                "broad_instance_sources": record["broad_instance_sources"],
                "scoped_support_count": record["scoped_support_count"],
                "scoped_support_sources": record["scoped_support_sources"],
                "instance_evidence_tier": record["instance_evidence_tier"],
            }
            json.dump({"type": "Feature", "properties": properties, "geometry": mapping(record["polygon"])}, handle)
        handle.write("]}")
    from detector_candidate_review import export_candidates
    candidate_review = export_candidates(cells, decision_by_member, accepted_assignment, {
        "stardist_objects": args.stardist_objects, "hovernet_cells": args.hovernet_cells,
        "cellvit_cells": args.cellvit_cells, "image": args.image, "labels": mask_path,
        "shift": args.shift, "alignment_csv": outdir / "alignment.csv",
    }, outdir / "rejected_detector_candidates.geojson")
    summary = {
        "input_counts": {source: len(ids) for source, ids in by_source.items()},
        "consensus_count": len(records), "minimum_support": args.min_support,
        "fusion_acceptance_policy": args.fusion_acceptance_policy,
        "match_radius_um": args.match_radius_um, "match_radius_px": radius_px,
        "source_mpp": mpp, "rejected_no_geometry": rejected_no_geometry,
        "minimum_agreement_score": args.min_agreement_score,
        "rejected_candidate_review": candidate_review,
        "fusion_contract": {
            "instance_geometry": "one canonical instance from spatially matched detector predictions",
            "canonical_instance_detectors": sorted(
                source for source in DETECTOR_SCOPES if detector_role(source) == "broad_scope"
            ),
            "scoped_support_detectors": sorted(
                source for source in DETECTOR_SCOPES if detector_role(source) == "scoped_support"
            ),
            "acceptance_policy": args.fusion_acceptance_policy,
            "geometry_selection": f"priority order: {','.join(priority)}",
            "phenotype_semantics": "detector-specific evidence only; incompatible taxonomies are not voted into one class",
            "agreement_score_semantics": "descriptive broad-detector support and broad-centroid proximity; scoped support is reported separately and does not inflate the score",
        },
        "detector_count_qc": count_qc,
        "detector_spatial_coverage": spatial_coverage,
        "detector_agreement_benchmark": agreement_rows,
        "support_counts": {str(level): sum(r["support"] == level for r in records) for level in (2, 3)},
        "agreement_tier_counts": {
            level: sum(r["agreement_tier"] == level for r in records)
            for level in ("high", "moderate", "low")
        },
        "instance_evidence_tier_counts": dict(sorted({
            tier: sum(r["instance_evidence_tier"] == tier for r in records)
            for tier in {r["instance_evidence_tier"] for r in records}
        }.items())),
        "component_decision_counts": dict(sorted({
            decision: component_decisions.count(decision)
            for decision in set(component_decisions)
        }.items())),
        "detector_prediction_decision_counts": dict(sorted({
            decision: sum(value["decision"] == decision for value in decision_by_member.values())
            for decision in {value["decision"] for value in decision_by_member.values()}
        }.items())),
        "method_combination_counts": dict(sorted(method_combination_counts.items())),
        "detector_assignment_counts": detector_assignment_counts,
        "mask_seed_summary": mask_seed_summary,
        "mask_label_coverage": mask_coverage,
        "cellvit_source_binding": {"sha256": cellvit_source_sha256,
            "coordinates": "analysis_crop_level0_xy_pixels",
            "centroid_columns": ["cellvitpp_x_px", "cellvitpp_y_px"],
            "identity_column": "cellvitpp_source_sha256",
            "semantics": "exact normalized CellViT population and detector centroids; canonical fused centroids unchanged"},
    }
    write_preview(Path(args.image), records, outdir / "consensus_preview.png", args.preview_max_side)
    verify_cellvit_source(cellvit_path, cellvit_source_sha256)
    (outdir / "consensus_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
