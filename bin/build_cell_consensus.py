#!/usr/bin/env python3
"""Align StarDist, HoVer-Net and CellViT++ instances into canonical cell IDs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


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


def load_cells(path: Path, source: str) -> list[Cell]:
    payload = json.loads(path.read_text())
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
    args = parser.parse_args()

    from shapely.geometry import Polygon, mapping

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    shift = json.loads(Path(args.shift).read_text())
    mpp = float(shift.get("source_mpp", shift.get("microns_per_pixel", args.default_mpp)))
    radius_px = args.match_radius_um / mpp
    cells = load_stardist(Path(args.stardist_objects))
    cells += load_cells(Path(args.hovernet_cells), "hovernet")
    cells += load_cells(Path(args.cellvit_cells), "cellvitpp")
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(cells):
        by_source[cell.source].append(index)
    dsu = DisjointSet(cells)
    for _, left, right in candidate_edges(cells, by_source, radius_px):
        dsu.union(left, right)
    groups: dict[int, list[Cell]] = defaultdict(list)
    for index, cell in enumerate(cells):
        groups[dsu.find(index)].append(cell)

    priority = [value.strip() for value in args.geometry_priority.split(",") if value.strip()]
    records = []
    rejected_no_geometry = 0
    for group in groups.values():
        support = len({cell.source for cell in group})
        if support < args.min_support:
            continue
        representative = choose_geometry(group, priority)
        if representative is None:
            rejected_no_geometry += 1
            continue
        polygon = Polygon(representative.contour)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            rejected_no_geometry += 1
            continue
        center = np.mean([(cell.x, cell.y) for cell in group], axis=0)
        members = {cell.source: cell for cell in group}
        records.append({"x": float(center[0]), "y": float(center[1]), "support": support, "members": members, "polygon": polygon})
    records.sort(key=lambda item: (item["y"], item["x"]))
    for label, record in enumerate(records, 1):
        record["label"] = label

    import pyvips
    image = pyvips.Image.new_from_file(args.image, access="sequential")
    mask_path = outdir / "labels.tif"
    mask_seed_summary = write_mask(records, image.width, image.height, mask_path, args.tile_size, args.compression)
    mask_coverage = validate_mask_label_coverage(mask_path, len(records))
    columns = ["label", "area", "y", "x", "xmin", "ymin", "xmax", "ymax", "mask_seed_x", "mask_seed_y", "consensus_support", "consensus_methods"]
    for source in ("stardist", "hovernet", "cellvitpp"):
        columns += [
            f"{source}_id", f"{source}_type_id", f"{source}_type",
            f"{source}_probability",
        ]
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
                "consensus_support": record["support"], "consensus_methods": ";".join(sorted(record["members"])),
            }
            for source, cell in record["members"].items():
                row[f"{source}_id"] = cell.source_id
                row[f"{source}_type_id"] = cell.type_id
                row[f"{source}_type"] = cell.cell_type
                row[f"{source}_probability"] = json.dumps(cell.probability) if cell.probability is not None else ""
            writer.writerow(row)
    accepted_assignment = {
        (source, cell.source_id): record["label"]
        for record in records
        for source, cell in record["members"].items()
    }
    component_support = {}
    for group in groups.values():
        support = len({cell.source for cell in group})
        for cell in group:
            component_support[(cell.source, cell.source_id)] = support
    with (outdir / "alignment.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "source", "source_id", "x", "y", "component_support", "consensus_label", "accepted",
        ])
        writer.writeheader()
        for cell in cells:
            key = (cell.source, cell.source_id)
            label = accepted_assignment.get(key, "")
            writer.writerow({
                "source": cell.source, "source_id": cell.source_id, "x": cell.x, "y": cell.y,
                "component_support": component_support.get(key, 1), "consensus_label": label,
                "accepted": bool(label),
            })
    with (outdir / "consensus_cells.geojson").open("w") as handle:
        handle.write('{"type":"FeatureCollection","features":[')
        for index, record in enumerate(records):
            if index:
                handle.write(",")
            properties = {"id": record["label"], "support": record["support"], "methods": sorted(record["members"])}
            json.dump({"type": "Feature", "properties": properties, "geometry": mapping(record["polygon"])}, handle)
        handle.write("]}")
    summary = {
        "input_counts": {source: len(ids) for source, ids in by_source.items()},
        "consensus_count": len(records), "minimum_support": args.min_support,
        "match_radius_um": args.match_radius_um, "match_radius_px": radius_px,
        "source_mpp": mpp, "rejected_no_geometry": rejected_no_geometry,
        "support_counts": {str(level): sum(r["support"] == level for r in records) for level in (2, 3)},
        "mask_seed_summary": mask_seed_summary,
        "mask_label_coverage": mask_coverage,
    }
    (outdir / "consensus_summary.json").write_text(json.dumps(summary, indent=2))
    write_preview(Path(args.image), records, outdir / "consensus_preview.png", args.preview_max_side)


if __name__ == "__main__":
    main()
