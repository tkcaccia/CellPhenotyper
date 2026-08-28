#!/usr/bin/env python3
"""Select the final tissue component containing the most neoplastic cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def polygon_components(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for child in geometry.geoms:
            yield from polygon_components(child)


def load_sections(path: Path, cluster_variant: str) -> list[dict]:
    from shapely.geometry import shape

    payload = json.loads(path.read_text())
    sections = []
    for feature_index, feature in enumerate(payload.get("features", []), 1):
        geometry = shape(feature.get("geometry"))
        properties = dict(feature.get("properties") or {})
        value = properties.get("value", feature_index)
        classification = properties.get("classification", f"class_{value}")
        components = sorted(
            (part for part in polygon_components(geometry) if not part.is_empty and part.area > 0),
            key=lambda part: (-part.area, part.bounds),
        )
        for component_index, component in enumerate(components, 1):
            sections.append({
                "section_id": f"{cluster_variant}_class_{value}_component_{component_index}",
                "cluster_variant": cluster_variant,
                "class_value": value,
                "classification": classification,
                "source_feature_index": feature_index,
                "component_index": component_index,
                "geometry": component,
                "area_px2": float(component.area),
                "total_cells": 0,
                "neoplastic_cells": 0,
            })
    if not sections:
        raise RuntimeError(f"No polygonal tissue sections found in {path}")
    return sections


class SectionGrid:
    """Small fixed grid that limits exact point-in-polygon tests per cell."""

    def __init__(self, sections: list[dict], bin_size: int = 2048):
        self.sections = sections
        self.bin_size = max(64, int(bin_size))
        self.bins: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, section in enumerate(sections):
            xmin, ymin, xmax, ymax = section["geometry"].bounds
            for by in range(math.floor(ymin / self.bin_size), math.floor(ymax / self.bin_size) + 1):
                for bx in range(math.floor(xmin / self.bin_size), math.floor(xmax / self.bin_size) + 1):
                    self.bins[(bx, by)].append(index)

    def locate(self, x: float, y: float) -> int | None:
        from shapely.geometry import Point

        point = Point(x, y)
        candidates = self.bins.get((math.floor(x / self.bin_size), math.floor(y / self.bin_size)), ())
        for index in candidates:
            if self.sections[index]["geometry"].covers(point):
                return index
        return None


def count_cells(
    objects_csv: Path,
    sections: list[dict],
    neoplastic_names: set[str],
    bin_size: int = 2048,
) -> dict:
    grid = SectionGrid(sections, bin_size)
    seen = assigned = neoplastic_seen = 0
    with objects_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "cellvitpp_type"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"Consensus table is missing {sorted(missing)}. "
                "Run the named CellViT++ consensus stage before section selection."
            )
        for row in reader:
            seen += 1
            cell_type = str(row.get("cellvitpp_type", "")).strip().lower()
            is_neoplastic = cell_type in neoplastic_names
            neoplastic_seen += int(is_neoplastic)
            try:
                section_index = grid.locate(float(row["x"]), float(row["y"]))
            except (TypeError, ValueError):
                continue
            if section_index is None:
                continue
            assigned += 1
            sections[section_index]["total_cells"] += 1
            sections[section_index]["neoplastic_cells"] += int(is_neoplastic)
    return {
        "consensus_cells_seen": seen,
        "consensus_cells_assigned": assigned,
        "neoplastic_consensus_cells_seen": neoplastic_seen,
    }


def select_section(sections: list[dict]) -> dict:
    return min(
        sections,
        key=lambda section: (
            -section["neoplastic_cells"],
            -section["total_cells"],
            -section["area_px2"],
            section["section_id"],
        ),
    )


def source_mpp(shift_path: Path, fallback: float) -> float:
    data = json.loads(shift_path.read_text())
    value = data.get("source_mpp", data.get("microns_per_pixel", fallback))
    if value is None or float(value) <= 0:
        raise RuntimeError("No valid source MPP in shift.json; set --default-mpp")
    return float(value)


def write_sections_csv(sections: list[dict], path: Path, mpp: float, selected_id: str) -> None:
    columns = [
        "section_id", "cluster_variant", "class_value", "classification",
        "source_feature_index", "component_index", "area_px2", "area_mm2",
        "total_cells", "neoplastic_cells", "neoplastic_fraction", "selected",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for section in sorted(sections, key=lambda value: value["section_id"]):
            total = section["total_cells"]
            writer.writerow({
                **{key: section[key] for key in columns if key in section},
                "area_mm2": section["area_px2"] * mpp * mpp / 1_000_000.0,
                "neoplastic_fraction": section["neoplastic_cells"] / total if total else 0.0,
                "selected": section["section_id"] == selected_id,
            })


def write_geojson(section: dict, path: Path, origin: tuple[int, int] | None = None) -> None:
    from shapely.affinity import translate
    from shapely.geometry import mapping

    geometry = section["geometry"]
    coordinate_space = "roi_crop_level0_pixels"
    if origin is not None:
        geometry = translate(geometry, xoff=-origin[0], yoff=-origin[1])
        coordinate_space = "selected_section_crop_pixels"
    properties = {
        key: section[key]
        for key in (
            "section_id", "cluster_variant", "class_value", "classification",
            "total_cells", "neoplastic_cells", "area_px2",
        )
    }
    properties["coordinate_space"] = coordinate_space
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": properties, "geometry": mapping(geometry)}],
    }))


def write_crop(
    image_path: Path,
    section: dict,
    outdir: Path,
    mpp: float,
    padding_um: float,
    tile_size: int,
) -> dict:
    import numpy as np
    import pyvips
    import rasterio
    from PIL import Image, ImageDraw
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.affinity import translate
    from shapely.geometry import mapping

    source = pyvips.Image.new_from_file(str(image_path), access="random")
    padding_px = max(0, int(round(padding_um / mpp)))
    xmin, ymin, xmax, ymax = section["geometry"].bounds
    x0 = max(0, int(math.floor(xmin)) - padding_px)
    y0 = max(0, int(math.floor(ymin)) - padding_px)
    x1 = min(source.width, int(math.ceil(xmax)) + padding_px)
    y1 = min(source.height, int(math.ceil(ymax)) + padding_px)
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        raise RuntimeError("Selected tissue section has an empty clipped bounding box")

    local_geometry = translate(section["geometry"], xoff=-x0, yoff=-y0)
    mask_path = outdir / "selected_section_mask.tif"
    block = max(64, min(1024, int(tile_size)))
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": "uint8", "tiled": True, "blockxsize": block, "blockysize": block,
        "compress": "deflate", "BIGTIFF": "YES", "transform": rasterio.Affine.identity(),
    }
    with rasterio.open(mask_path, "w", **profile) as dst:
        for yy in range(0, height, block):
            for xx in range(0, width, block):
                h, w = min(block, height - yy), min(block, width - xx)
                tile = rasterize(
                    [(mapping(local_geometry), 1)], out_shape=(h, w),
                    transform=rasterio.Affine.translation(xx, yy), fill=0, dtype="uint8",
                )
                dst.write(tile, 1, window=Window(xx, yy, w, h))

    crop = source.crop(x0, y0, width, height)
    if crop.bands > 3:
        crop = crop[:3]
    if crop.bands == 1:
        crop = crop.bandjoin([crop, crop])
    mask = pyvips.Image.new_from_file(str(mask_path), access="sequential")
    masked = (mask > 0).ifthenelse(crop, 255).copy(xres=1000.0 / mpp, yres=1000.0 / mpp)
    crop_path = outdir / "selected_section.ome.tif"
    masked.tiffsave(
        str(crop_path), tile=True, tile_width=512, tile_height=512, pyramid=True,
        compression="jpeg", Q=92, bigtiff=True, resunit="cm",
    )

    scale = min(1.0, 3000.0 / max(width, height))
    thumb = masked.resize(scale) if scale < 1 else masked
    if thumb.format != "uchar":
        thumb = thumb.cast("uchar")
    array = np.ndarray(
        buffer=thumb.write_to_memory(), dtype=np.uint8,
        shape=(thumb.height, thumb.width, thumb.bands),
    )[:, :, :3].copy()
    preview = Image.fromarray(array)
    draw = ImageDraw.Draw(preview)
    polygons = [local_geometry] if local_geometry.geom_type == "Polygon" else list(local_geometry.geoms)
    for polygon in polygons:
        draw.line([(x * scale, y * scale) for x, y in polygon.exterior.coords], fill=(255, 80, 20), width=3)
    preview.save(outdir / "selected_section_preview.png")
    return {
        "crop_origin_x": x0, "crop_origin_y": y0, "crop_width": width,
        "crop_height": height, "source_mpp": mpp, "padding_um": padding_um,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-geojson", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--cluster-variant", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--neoplastic-names", default="neoplastic,tumor,tumour")
    parser.add_argument("--default-mpp", type=float, default=0.5)
    parser.add_argument("--padding-um", type=float, default=256.0)
    parser.add_argument("--spatial-bin-size", type=int, default=2048)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--require-neoplastic", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sections = load_sections(Path(args.cluster_geojson), args.cluster_variant)
    names = {name.strip().lower() for name in args.neoplastic_names.split(",") if name.strip()}
    counts = count_cells(Path(args.objects), sections, names, args.spatial_bin_size)
    selected = select_section(sections)
    if args.require_neoplastic and selected["neoplastic_cells"] == 0:
        raise RuntimeError("No named neoplastic consensus cells fall within any final tissue section")
    mpp = source_mpp(Path(args.shift), args.default_mpp)
    crop_metadata = write_crop(
        Path(args.image), selected, outdir, mpp, args.padding_um, args.tile_size,
    )
    write_sections_csv(sections, outdir / "section_neoplastic_counts.csv", mpp, selected["section_id"])
    write_geojson(selected, outdir / "selected_section.geojson")
    write_geojson(
        selected, outdir / "selected_section_crop.geojson",
        (crop_metadata["crop_origin_x"], crop_metadata["crop_origin_y"]),
    )
    summary = {
        "selection_rule": "maximum named neoplastic consensus-cell count; ties by total cells, area, section ID",
        "neoplastic_names": sorted(names),
        "section_count": len(sections),
        "selected_section_id": selected["section_id"],
        "selected_neoplastic_cells": selected["neoplastic_cells"],
        "selected_total_cells": selected["total_cells"],
        "selected_area_px2": selected["area_px2"],
        **counts,
        **crop_metadata,
    }
    (outdir / "selected_section_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "selected_section_shift.json").write_text(json.dumps(crop_metadata, indent=2))


if __name__ == "__main__":
    main()
