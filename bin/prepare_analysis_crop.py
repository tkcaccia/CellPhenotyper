#!/usr/bin/env python3
"""Create one GrandQC tissue/ROI crop shared by all downstream stages."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


def iter_xy(value: Any) -> Iterable[tuple[float, float]]:
    if isinstance(value, dict):
        if value.get("type") == "FeatureCollection":
            for feature in value.get("features", []):
                yield from iter_xy(feature)
        elif value.get("type") == "Feature":
            yield from iter_xy(value.get("geometry", {}))
        elif value.get("type") == "GeometryCollection":
            for geometry in value.get("geometries", []):
                yield from iter_xy(geometry)
        else:
            yield from iter_xy(value.get("coordinates", []))
    elif isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            yield float(value[0]), float(value[1])
        else:
            for item in value:
                yield from iter_xy(item)


def shift_coordinates(value: Any, dx: float, dy: float) -> Any:
    if isinstance(value, dict):
        return {key: shift_coordinates(item, dx, dy) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            shifted = list(value)
            shifted[0] = float(value[0]) + dx
            shifted[1] = float(value[1]) + dy
            return shifted
        return [shift_coordinates(item, dx, dy) for item in value]
    return value


def iter_polygon_rings(value: Any) -> Iterable[tuple[list, list[list]]]:
    """Yield exterior/interior rings from arbitrary GeoJSON containers."""
    if not isinstance(value, dict):
        return
    kind = value.get("type")
    if kind == "FeatureCollection":
        for feature in value.get("features", []):
            yield from iter_polygon_rings(feature)
    elif kind == "Feature":
        yield from iter_polygon_rings(value.get("geometry", {}))
    elif kind == "GeometryCollection":
        for geometry in value.get("geometries", []):
            yield from iter_polygon_rings(geometry)
    elif kind == "Polygon":
        rings = value.get("coordinates", [])
        if rings:
            yield rings[0], list(rings[1:])
    elif kind == "MultiPolygon":
        for polygon in value.get("coordinates", []):
            if polygon:
                yield polygon[0], list(polygon[1:])


def read_mask_2d(path: Path):
    import numpy as np
    import tifffile

    try:
        mask = np.asarray(tifffile.memmap(str(path)))
    except ValueError:
        mask = np.asarray(tifffile.imread(str(path)))
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise RuntimeError(f"GrandQC clean-tissue mask must be 2D, got {mask.shape}")
    return mask


def rasterize_roi_mask(roi: dict, shape: tuple[int, int], coordinate_size: tuple[int, int]):
    """Rasterize pixel-coordinate GeoJSON without allocating a full-resolution mask."""
    import numpy as np
    from PIL import Image, ImageDraw

    mask_h, mask_w = map(int, shape)
    coord_w, coord_h = map(int, coordinate_size)
    canvas = Image.new("L", (mask_w, mask_h), 0)
    draw = ImageDraw.Draw(canvas)

    def scaled_ring(ring):
        return [
            (
                min(mask_w - 1, max(0, int(round(float(point[0]) * mask_w / max(1, coord_w))))),
                min(mask_h - 1, max(0, int(round(float(point[1]) * mask_h / max(1, coord_h))))),
            )
            for point in ring
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]

    polygon_count = 0
    for exterior, holes in iter_polygon_rings(roi):
        outer = scaled_ring(exterior)
        if len(outer) < 3:
            continue
        draw.polygon(outer, fill=1)
        polygon_count += 1
        for hole in holes:
            inner = scaled_ring(hole)
            if len(inner) >= 3:
                draw.polygon(inner, fill=0)
    if polygon_count == 0:
        raise RuntimeError("ROI contains no polygon geometry")
    return np.asarray(canvas, dtype=bool)


def tissue_roi_bbox(
    clean_tissue_mask,
    roi: dict,
    image_size: tuple[int, int],
    pad: int,
) -> tuple[tuple[int, int, int, int], dict]:
    """Return a level-0 bounding box around GrandQC clean tissue intersected with ROI."""
    import numpy as np

    width, height = map(int, image_size)
    mask_h, mask_w = map(int, clean_tissue_mask.shape)
    roi_mask = rasterize_roi_mask(roi, (mask_h, mask_w), (width, height))
    support = (np.asarray(clean_tissue_mask) != 0) & roi_mask
    ys, xs = np.nonzero(support)
    if xs.size == 0:
        raise RuntimeError("GrandQC clean tissue and ROI do not overlap")

    mx0, my0 = int(xs.min()), int(ys.min())
    mx1, my1 = int(xs.max()) + 1, int(ys.max()) + 1
    x0 = max(0, int(math.floor(mx0 * width / mask_w)) - pad)
    y0 = max(0, int(math.floor(my0 * height / mask_h)) - pad)
    x1 = min(width, int(math.ceil(mx1 * width / mask_w)) + pad)
    y1 = min(height, int(math.ceil(my1 * height / mask_h)) + pad)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError(f"GrandQC tissue/ROI crop is empty after clamping: {(x0, y0, x1, y1)}")
    return (x0, y0, x1, y1), {
        "grandqc_mask_shape_yx": [mask_h, mask_w],
        "grandqc_clean_pixels": int(np.count_nonzero(clean_tissue_mask)),
        "roi_pixels_at_grandqc_scale": int(np.count_nonzero(roi_mask)),
        "tissue_roi_pixels_at_grandqc_scale": int(xs.size),
        "tissue_roi_bbox_mask_xyxy": [mx0, my0, mx1, my1],
    }


def read_shape_and_mpp(path: Path) -> tuple[int, int, float | None]:
    import tifffile
    from validate_input_resolution import read_resolution

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(series.axes).replace("S", "C")
        shape = tuple(map(int, series.shape))
        height = shape[axes.index("Y")]
        width = shape[axes.index("X")]
    try:
        info = read_resolution(path)
    except RuntimeError:
        return height, width, None
    if (height, width) != (info.height_px, info.width_px):
        raise ValueError("Crop source dimensions disagree with physical-resolution metadata")
    mpp_x = float(info.mpp_x)
    mpp_y = float(info.mpp_y if info.mpp_y is not None else info.mpp_x)
    mpp = (mpp_x + mpp_y) / 2.0
    anisotropy = abs(mpp_x - mpp_y) / mpp
    if anisotropy > 0.05:
        raise ValueError(
            f"Crop source X/Y pixel sizes differ by {anisotropy:.2%}; "
            "a scalar MPP would corrupt physical calibration."
        )
    return height, width, mpp


def resolve_zarr_array(root):
    """Resolve level 0 from a tifffile Zarr array or multiscale group."""
    if hasattr(root, "shape") and hasattr(root, "ndim"):
        return root
    if not hasattr(root, "keys"):
        raise RuntimeError("TIFF Zarr store contains neither an array nor a group")

    keys = list(root.keys())
    ordered = (["0"] if "0" in keys else []) + sorted(
        (key for key in keys if key != "0"),
        key=lambda key: (not str(key).isdigit(), int(key) if str(key).isdigit() else str(key)),
    )
    for key in ordered:
        child = root[key]
        if hasattr(child, "shape") and hasattr(child, "ndim"):
            return child
        if hasattr(child, "keys"):
            try:
                return resolve_zarr_array(child)
            except RuntimeError:
                pass
    raise RuntimeError(f"Could not resolve a Zarr array from TIFF group keys: {keys}")


def read_zarr_rgb_block(array, axes: str, x0: int, x1: int, y0: int, y1: int):
    """Read one spatial block and normalize it to YXC without loading the crop."""
    import numpy as np

    axes = axes.replace("S", "C")
    if len(axes) != array.ndim or "Y" not in axes or "X" not in axes:
        raise RuntimeError(f"Unsupported TIFF axes '{axes}' for Zarr shape {array.shape}")

    slicer: list[Any] = []
    retained_axes: list[str] = []
    for axis in axes:
        if axis == "Y":
            slicer.append(slice(y0, y1))
            retained_axes.append(axis)
        elif axis == "X":
            slicer.append(slice(x0, x1))
            retained_axes.append(axis)
        elif axis == "C":
            slicer.append(slice(None))
            retained_axes.append(axis)
        else:
            slicer.append(0)

    block = np.asarray(array[tuple(slicer)])
    order = [retained_axes.index("Y"), retained_axes.index("X")]
    if "C" in retained_axes:
        order.append(retained_axes.index("C"))
    block = np.transpose(block, axes=order)
    if block.ndim == 2:
        block = np.repeat(block[..., None], 3, axis=-1)
    elif block.shape[-1] == 1:
        block = np.repeat(block, 3, axis=-1)
    if block.ndim != 3 or block.shape[-1] < 3:
        raise RuntimeError(f"Could not normalize TIFF block to RGB: shape={block.shape}, axes={axes}")
    return block[..., :3]


def write_crop_with_tifffile_zarr(
    source: Path,
    output: Path,
    bbox: tuple[int, int, int, int],
    block_rows: int = 512,
) -> None:
    """Write a crop through a disk-backed buffer when libvips cannot decode it."""
    import numpy as np
    import tifffile
    import zarr

    x0, y0, x1, y1 = bbox
    height, width = y1 - y0, x1 - x0
    buffer_path = output.with_name(f".{output.name}.crop-buffer-{os.getpid()}.dat")
    buffer = None
    try:
        with tifffile.TiffFile(str(source)) as tif:
            series = tif.series[0]
            levels = getattr(series, "levels", None) or [series]
            level0 = levels[0]
            axes = str(getattr(level0, "axes", None) or series.axes)
            store = level0.aszarr()
            try:
                array = resolve_zarr_array(zarr.open(store, mode="r"))
                buffer = np.memmap(
                    buffer_path,
                    mode="w+",
                    dtype=array.dtype,
                    shape=(height, width, 3),
                )
                for out_y0 in range(0, height, max(1, int(block_rows))):
                    out_y1 = min(height, out_y0 + max(1, int(block_rows)))
                    buffer[out_y0:out_y1] = read_zarr_rgb_block(
                        array,
                        axes,
                        x0,
                        x1,
                        y0 + out_y0,
                        y0 + out_y1,
                    )
                buffer.flush()
            finally:
                if hasattr(store, "close"):
                    store.close()

        tifffile.imwrite(
            str(output),
            buffer,
            photometric="rgb",
            compression="zlib",
            bigtiff=True,
            tile=(512, 512),
        )
    finally:
        if buffer is not None:
            del buffer
        buffer_path.unlink(missing_ok=True)


def write_crop(source: Path, output: Path, bbox: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = bbox
    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(source), access="sequential", page=0)
        crop = image.crop(x0, y0, x1 - x0, y1 - y0)
        crop.tiffsave(
            str(output), tile=True, tile_width=512, tile_height=512,
            compression="deflate", bigtiff=True,
        )
        return "pyvips_streaming"
    except Exception as vips_error:
        write_crop_with_tifffile_zarr(source, output, bbox)
        return f"tifffile_zarr_memmap_fallback:{type(vips_error).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--clean-tissue-mask", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pad", type=int, default=200)
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    roi_path = Path(args.roi).resolve()
    clean_tissue_mask_path = Path(args.clean_tissue_mask).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    roi = json.loads(roi_path.read_text())
    height, width, source_mpp = read_shape_and_mpp(image_path)
    pad = max(0, int(args.pad))
    clean_tissue_mask = read_mask_2d(clean_tissue_mask_path)
    (x0, y0, x1, y1), support_summary = tissue_roi_bbox(
        clean_tissue_mask,
        roi,
        (width, height),
        pad,
    )

    crop_path = outdir / "crop_roi.tif"
    writer = write_crop(image_path, crop_path, (x0, y0, x1, y1))
    shifted_roi = shift_coordinates(roi, -x0, -y0)
    (outdir / "roi_all_crop.geojson").write_text(json.dumps(shifted_roi))

    shift = {
        "input_image": str(image_path),
        "roi_input_geojson": str(roi_path),
        "grandqc_clean_tissue_mask": str(clean_tissue_mask_path),
        "crop_tif": str(crop_path),
        "crop_bbox_xyxy": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "pad_pixels_used": pad,
        "offset_crop_to_original": {"dx": x0, "dy": y0},
        "crop_size": {"width": x1 - x0, "height": y1 - y0},
        "full_size": {"width": width, "height": height},
        "crop_basis": "grandqc_clean_tissue_intersection_roi",
        **support_summary,
    }
    if source_mpp and source_mpp > 0:
        shift["source_mpp"] = source_mpp
        shift["microns_per_pixel"] = source_mpp
    (outdir / "shift.json").write_text(json.dumps(shift, indent=2))
    summary = {**shift, "crop_writer": writer, "shared_by": ["StarDist", "HoVer-Net MoNuSAC", "CellViT++"]}
    (outdir / "crop_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[OK] GrandQC tissue/ROI crop={x1 - x0}x{y1 - y0} "
        f"origin=({x0},{y0}) support={support_summary['tissue_roi_pixels_at_grandqc_scale']} writer={writer}"
    )


if __name__ == "__main__":
    main()
