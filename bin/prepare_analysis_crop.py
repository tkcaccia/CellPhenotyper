#!/usr/bin/env python3
"""Create one GrandQC tissue/ROI crop shared by all downstream stages."""

from __future__ import annotations

import argparse
import hashlib
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
    import re
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(series.axes).replace("S", "C")
        shape = tuple(map(int, series.shape))
        height = shape[axes.index("Y")]
        width = shape[axes.index("X")]
        description = getattr(tif, "ome_metadata", None)
        if not description:
            try:
                description = tif.pages[0].tags["ImageDescription"].value
            except Exception:
                description = ""
        match = re.search(r'PhysicalSizeX="([0-9]+(?:\.[0-9]+)?)"', str(description))
        mpp = float(match.group(1)) if match else None
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


def pyramid_level_shapes(height: int, width: int, tile_size: int = 512) -> list[tuple[int, int]]:
    """Return 2x overview shapes until the image fits in one output tile."""
    shapes: list[tuple[int, int]] = []
    level_h, level_w = int(height), int(width)
    while max(level_h, level_w) > int(tile_size):
        level_h = max(1, (level_h + 1) // 2)
        level_w = max(1, (level_w + 1) // 2)
        shapes.append((level_h, level_w))
    return shapes


def downsample_rgb_2x_to_memmap(source, output_path: Path, block_rows: int = 64):
    """Create one lossless-base visualization overview with bounded memory.

    Edge pixels are replicated before a 2x2 box average. The returned memmap
    preserves the source dtype and is suitable for direct tiled TIFF writing.
    """
    import numpy as np

    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Pyramid source must be YX3 RGB, got {source.shape}")
    source_h, source_w, _ = map(int, source.shape)
    output_h, output_w = (source_h + 1) // 2, (source_w + 1) // 2
    output = np.memmap(output_path, mode="w+", dtype=source.dtype, shape=(output_h, output_w, 3))
    rows_per_block = max(1, int(block_rows))
    integer_source = np.issubdtype(source.dtype, np.integer)
    for out_y0 in range(0, output_h, rows_per_block):
        out_y1 = min(output_h, out_y0 + rows_per_block)
        source_y0 = 2 * out_y0
        source_y1 = min(source_h, 2 * out_y1)
        block = np.asarray(source[source_y0:source_y1, :, :])
        required_rows = 2 * (out_y1 - out_y0)
        if block.shape[0] < required_rows:
            block = np.concatenate([block, block[-1:, :, :]], axis=0)
        if block.shape[1] % 2:
            block = np.concatenate([block, block[:, -1:, :]], axis=1)
        reshaped = block.reshape(out_y1 - out_y0, 2, output_w, 2, 3)
        if integer_source:
            values = reshaped.astype(np.uint64).sum(axis=(1, 3))
            output[out_y0:out_y1] = ((values + 2) // 4).astype(source.dtype)
        else:
            output[out_y0:out_y1] = reshaped.mean(axis=(1, 3), dtype=np.float64).astype(source.dtype)
    output.flush()
    return output


def write_pyramidal_rgb_tiff(
    output: Path,
    level0,
    source_mpp: float | None,
    compression: str = "zlib",
    tile_size: int = 512,
) -> list[list[int]]:
    """Write a tiled RGB TIFF with SubIFD overviews and a bit-exact level 0."""
    import tifffile

    height, width, channels = map(int, level0.shape)
    if channels != 3:
        raise ValueError(f"Expected RGB level 0, got {level0.shape}")
    reduced_shapes = pyramid_level_shapes(height, width, tile_size)
    temporary_levels: list[Path] = []
    previous = level0
    try:
        with tifffile.TiffWriter(str(output), bigtiff=True) as tif:
            base_options = {
                "photometric": "rgb",
                "compression": compression,
                "tile": (tile_size, tile_size),
                "metadata": None,
            }
            if source_mpp:
                base_options.update({
                    "resolution": (10000 / source_mpp, 10000 / source_mpp),
                    "resolutionunit": "CENTIMETER",
                })
            tif.write(level0, subifds=len(reduced_shapes), **base_options)
            for level_index, expected_shape in enumerate(reduced_shapes, start=1):
                level_path = output.with_name(
                    f".{output.name}.pyramid-{os.getpid()}-level-{level_index}.dat"
                )
                temporary_levels.append(level_path)
                current = downsample_rgb_2x_to_memmap(previous, level_path)
                if tuple(map(int, current.shape[:2])) != expected_shape:
                    raise RuntimeError(
                        f"Pyramid level {level_index} shape mismatch: "
                        f"expected={expected_shape} observed={current.shape[:2]}"
                    )
                level_options = dict(base_options)
                if source_mpp:
                    factor = 2 ** level_index
                    level_options["resolution"] = (
                        10000 / (source_mpp * factor),
                        10000 / (source_mpp * factor),
                    )
                tif.write(current, subfiletype=1, **level_options)
                if previous is not level0:
                    previous.flush()
                    del previous
                    temporary_levels[-2].unlink(missing_ok=True)
                previous = current
        return [[height, width], *[[h, w] for h, w in reduced_shapes]]
    finally:
        if previous is not level0:
            previous.flush()
            del previous
        for path in temporary_levels:
            path.unlink(missing_ok=True)


def validate_crop_pyramid(
    path: Path,
    expected_shape: tuple[int, int],
    source_mpp: float | None,
    tile_size: int = 512,
) -> dict[str, Any]:
    """Fail closed when a WSI-scale crop is not tiled and pyramidal."""
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0]
        axes = str(series.axes).replace("S", "C")
        shapes = []
        for level in series.levels:
            level_axes = str(level.axes).replace("S", "C")
            level_shape = tuple(map(int, level.shape))
            shapes.append([level_shape[level_axes.index("Y")], level_shape[level_axes.index("X")]])
        observed = tuple(shapes[0])
        if observed != tuple(map(int, expected_shape)):
            raise RuntimeError(f"Crop shape mismatch: expected={expected_shape} observed={observed}")
        expected_levels = 1 + len(pyramid_level_shapes(*expected_shape, tile_size))
        if len(shapes) != expected_levels:
            raise RuntimeError(
                f"ROI crop pyramid is incomplete: expected_levels={expected_levels} "
                f"observed_levels={len(shapes)}"
            )
        if not tif.pages[0].is_tiled:
            raise RuntimeError("ROI crop level 0 must be tiled")
        for previous_shape, current_shape in zip(shapes, shapes[1:]):
            expected_reduced = [(previous_shape[0] + 1) // 2, (previous_shape[1] + 1) // 2]
            if current_shape != expected_reduced:
                raise RuntimeError(
                    f"ROI crop pyramid geometry is invalid: expected={expected_reduced} "
                    f"observed={current_shape}"
                )
        page = tif.pages[0]
        compression = getattr(page.compression, "name", str(page.compression))
        resolution_mpp = None
        if source_mpp:
            if "XResolution" not in page.tags or "ResolutionUnit" not in page.tags:
                raise RuntimeError("ROI crop is missing the source-MPP TIFF resolution tags")
            numerator, denominator = page.tags["XResolution"].value
            pixels_per_unit = float(numerator) / float(denominator)
            unit = int(page.tags["ResolutionUnit"].value)
            if unit != 3 or pixels_per_unit <= 0:  # centimetre
                raise RuntimeError(
                    "ROI crop resolution tags must contain positive pixels-per-centimetre values"
                )
            resolution_mpp = 10000.0 / pixels_per_unit
            if not math.isclose(resolution_mpp, source_mpp, rel_tol=1e-4, abs_tol=1e-6):
                raise RuntimeError(
                    f"ROI crop MPP mismatch: expected={source_mpp} observed={resolution_mpp}"
                )
        return {
            "required": expected_levels > 1,
            "validated": True,
            "level_count": len(shapes),
            "level_shapes_yx": shapes,
            "subifd": bool(getattr(page, "subifds", ())),
            "tiled": True,
            "tile_size_px": [int(page.tilelength), int(page.tilewidth)],
            "compression": compression,
            "level0_mpp_from_resolution_tags": resolution_mpp,
        }


def write_crop_with_tifffile_zarr(
    source: Path,
    output: Path,
    bbox: tuple[int, int, int, int],
    block_rows: int = 512,
    source_mpp: float | None = None,
) -> str:
    """Write a crop through a disk-backed buffer when libvips cannot decode it.

    Prefer the TIFF/Zarr adapter, but use the existing bounded segment reader
    when installed adapter versions cannot interoperate. Neither path decodes
    a whole slide into memory.
    """
    import numpy as np
    import tifffile

    x0, y0, x1, y1 = bbox
    height, width = y1 - y0, x1 - x0
    buffer_path = output.with_name(f".{output.name}.crop-buffer-{os.getpid()}.dat")
    buffer = None
    reader = None
    store = None
    try:
        with tifffile.TiffFile(str(source)) as tif:
            series = tif.series[0]
            levels = getattr(series, "levels", None) or [series]
            level0 = levels[0]
            axes = str(getattr(level0, "axes", None) or series.axes)
            try:
                import zarr
                store = level0.aszarr()
                array = resolve_zarr_array(zarr.open(store, mode="r"))
                dtype = array.dtype
                read_block = lambda xa, ya, xb, yb: read_zarr_rgb_block(array, axes, xa, xb, ya, yb)
                backend = "zarr_windows"
            except (ImportError, ValueError, TypeError, AttributeError) as adapter_error:
                if hasattr(store, "close"):
                    store.close()
                store = None
                from profile_cell_morphology import WindowReader
                reader = WindowReader(source)
                dtype = reader.dtype
                read_block = reader.read
                backend = f"{reader.backend}_adapter_fallback:{type(adapter_error).__name__}"
            buffer = np.memmap(buffer_path, mode="w+", dtype=dtype, shape=(height, width, 3))
            step = max(1, min(512, int(block_rows)))
            for out_y0 in range(0, height, step):
                out_y1 = min(height, out_y0 + step)
                for out_x0 in range(0, width, 512):
                    out_x1 = min(width, out_x0 + 512)
                    block = np.asarray(read_block(x0 + out_x0, y0 + out_y0, x0 + out_x1, y0 + out_y1))
                    if block.ndim == 2:
                        block = np.repeat(block[..., None], 3, axis=-1)
                    elif block.ndim == 3 and block.shape[-1] == 1:
                        block = np.repeat(block, 3, axis=-1)
                    if block.ndim != 3 or block.shape[-1] < 3:
                        raise ValueError(f"Crop block is not greyscale/RGB(A): {block.shape}")
                    buffer[out_y0:out_y1, out_x0:out_x1] = block[..., :3]
            buffer.flush()

        write_pyramidal_rgb_tiff(output, buffer, source_mpp, compression="zlib", tile_size=512)
        return f"{backend}_pyramidal_subifd"
    finally:
        if hasattr(store, "close"):
            store.close()
        if reader is not None:
            reader.close()
        if buffer is not None:
            del buffer
        buffer_path.unlink(missing_ok=True)


def write_crop(source: Path, output: Path, bbox: tuple[int, int, int, int], source_mpp: float | None = None) -> str:
    x0, y0, x1, y1 = bbox
    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(source), access="sequential", page=0)
        crop = image.crop(x0, y0, x1 - x0, y1 - y0)
        if source_mpp:
            crop = crop.copy(xres=1000.0 / source_mpp, yres=1000.0 / source_mpp)
        # Source OME XML has source dimensions/scale, not the crop's geometry.
        if crop.get_typeof("image-description"):
            crop.remove("image-description")
        crop.tiffsave(
            str(output), tile=True, tile_width=512, tile_height=512,
            pyramid=True, subifd=True, depth="onetile",
            compression="deflate", bigtiff=True, resunit="cm",
        )
        return "pyvips_streaming_pyramidal_subifd"
    except Exception as vips_error:
        backend = write_crop_with_tifffile_zarr(source, output, bbox, source_mpp=source_mpp)
        return f"tifffile_{backend}_memmap_fallback:{type(vips_error).__name__}"


def verified_report_scale(report_path: Path, image_path: Path, width: int, height: int) -> float:
    """Bind explicit calibration to the exact normalized image, not its filename."""
    report = json.loads(report_path.read_text())
    if report.get("status") != "pass":
        raise ValueError("Input resolution report did not pass")
    if (report.get("width_px"), report.get("height_px")) != (width, height):
        raise ValueError("Input resolution report/image dimensions differ")
    mx, my = float(report["mpp_x"]), float(report["mpp_y"])
    if not (math.isfinite(mx) and math.isfinite(my) and 0.01 <= mx <= 10 and math.isclose(mx, my, rel_tol=1e-4)):
        raise ValueError("Crop requires plausible isotropic physical calibration")
    expected = report.get("file_sha256")
    if not expected:
        raise ValueError("Resolution report must identify its image by SHA-256")
    digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Resolution report refers to different image content")
    return mx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--clean-tissue-mask", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pad", type=int, default=200)
    parser.add_argument("--resolution-json", help="Passed normalized-image resolution report; authoritative over stale TIFF metadata")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    roi_path = Path(args.roi).resolve()
    clean_tissue_mask_path = Path(args.clean_tissue_mask).resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    roi = json.loads(roi_path.read_text())
    height, width, source_mpp = read_shape_and_mpp(image_path)
    if args.resolution_json:
        source_mpp = verified_report_scale(Path(args.resolution_json), image_path, width, height)
    pad = max(0, int(args.pad))
    clean_tissue_mask = read_mask_2d(clean_tissue_mask_path)
    (x0, y0, x1, y1), support_summary = tissue_roi_bbox(
        clean_tissue_mask,
        roi,
        (width, height),
        pad,
    )

    crop_path = outdir / "crop_roi.tif"
    writer = write_crop(image_path, crop_path, (x0, y0, x1, y1), source_mpp)
    pyramid = validate_crop_pyramid(crop_path, (y1 - y0, x1 - x0), source_mpp)
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
        "crop_pyramid": pyramid,
        "full_size": {"width": width, "height": height},
        "crop_basis": "grandqc_clean_tissue_intersection_roi",
        "calibration_source": "passed_resolution_report" if args.resolution_json else "source_ome_metadata",
        "resolution_report": str(Path(args.resolution_json).resolve()) if args.resolution_json else None,
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
