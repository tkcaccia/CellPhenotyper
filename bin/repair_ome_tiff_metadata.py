#!/usr/bin/env python3
"""Losslessly replace a TIFF with a validated pyramidal OME-TIFF at a known MPP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from grow_to_tissue import pyramidize_with_raw2ometiff
from ome_tiff_metadata import create_tiff_memmap, label_storage_dtype, validate_ome_tiff


def describe(path: Path) -> dict[str, Any]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        physical_x = physical_y = None
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            pixels = next((node for node in root.iter() if node.tag.endswith("Pixels")), None)
            if pixels is not None:
                physical_x = pixels.attrib.get("PhysicalSizeX")
                physical_y = pixels.attrib.get("PhysicalSizeY")
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "is_ome": bool(tif.is_ome),
            "axes": series.axes,
            "shape": list(series.shape),
            "pyramid_levels": len(series.levels),
            "physical_size_x_um": float(physical_x) if physical_x else None,
            "physical_size_y_um": float(physical_y) if physical_y else None,
        }


def open_image(path: Path, rgb: bool):
    import pyvips

    if rgb:
        from extract_titan_section_embedding import open_vips_rgb

        image = open_vips_rgb(str(path))
        if image.bands < 3:
            raise RuntimeError(f"Expected RGB input, found {image.bands} band(s): {path}")
        return image[:3] if image.bands > 3 else image
    image = pyvips.Image.new_from_file(str(path), access="random")
    if image.bands != 1:
        raise RuntimeError(f"Expected one-channel label image, found {image.bands} bands: {path}")
    return image


def strip_inherited_ome_description(image):
    """Remove source OME XML before changing pixel type or channel layout."""
    staging = image.copy()
    if staging.get_typeof("image-description"):
        staging.remove("image-description")
    return staging


def write_label_flat(source, path: Path, mpp_x: float, mpp_y: float, block_rows: int = 512) -> None:
    import numpy as np

    dtype = label_storage_dtype(int(source.max()))
    if dtype.itemsize == 1:
        staging = strip_inherited_ome_description(source.cast("uchar"))
        staging.copy(xres=1000.0 / mpp_x, yres=1000.0 / mpp_y).tiffsave(
            str(path), tile=True, tile_width=512, tile_height=512, pyramid=False,
            compression="lzw", bigtiff=True, resunit="cm",
        )
        return

    numpy_dtypes = {
        "uchar": np.dtype("u1"), "char": np.dtype("i1"),
        "ushort": np.dtype("u2"), "short": np.dtype("i2"),
        "uint": np.dtype("u4"), "int": np.dtype("i4"),
    }
    source_dtype = numpy_dtypes.get(source.format)
    if source_dtype is None:
        raise RuntimeError(f"Unsupported label pixel format for repair: {source.format}")
    output = create_tiff_memmap(
        path,
        shape=(source.height, source.width),
        dtype=dtype,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    for y0 in range(0, source.height, max(1, int(block_rows))):
        height = min(max(1, int(block_rows)), source.height - y0)
        region = source.crop(0, y0, source.width, height)
        block = np.frombuffer(region.write_to_memory(), dtype=source_dtype).reshape(
            height, source.width,
        )
        output[y0:y0 + height] = block.astype(dtype, copy=False)
    output.flush()
    del output


def repair(path: Path, mpp_x: float, mpp_y: float, rgb: bool, max_workers: int) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = describe(path)
    source = open_image(path, rgb)
    expected_shape = (source.height, source.width)
    token = f"{os.getpid()}"
    flat = path.with_name(f".{path.name}.{token}.flat.tif")
    replacement = path.with_name(f".{path.name}.{token}.replacement.ome.tif")
    try:
        if rgb:
            staging = strip_inherited_ome_description(source)
            staging.copy(xres=1000.0 / mpp_x, yres=1000.0 / mpp_y).tiffsave(
                str(flat), tile=True, tile_width=512, tile_height=512, pyramid=False,
                compression="lzw", bigtiff=True, resunit="cm",
            )
        else:
            write_label_flat(source, flat, mpp_x, mpp_y)
        pyramidize_with_raw2ometiff(
            in_tif=str(flat),
            out_ome_tif=str(replacement),
            compression="LZW",
            max_workers=max(1, int(max_workers)),
            downsample="GAUSSIAN" if rgb else "SIMPLE",
            overwrite=True,
            keep_tmp=False,
            legacy=False,
        )
        validation = validate_ome_tiff(
            replacement,
            expected_shape=expected_shape,
            expected_mpp=(mpp_x, mpp_y),
        )
        converted = open_image(replacement, rgb)
        if (converted.width, converted.height, converted.bands) != (
            source.width,
            source.height,
            source.bands,
        ):
            raise RuntimeError("Converted TIFF dimensions or channel count changed")
        max_abs_difference = float((converted.cast("double") - source.cast("double")).abs().max())
        if max_abs_difference != 0.0:
            raise RuntimeError(f"Lossless pixel comparison failed: max difference={max_abs_difference}")
        os.replace(replacement, path)
        after = describe(path)
        return {
            "path": str(path),
            "rgb": rgb,
            "before": before,
            "after": after,
            "validation": validation,
            "max_abs_pixel_difference": max_abs_difference,
        }
    finally:
        flat.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mpp-x", type=float, required=True)
    parser.add_argument("--mpp-y", type=float, required=True)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--audit", default="")
    args = parser.parse_args()

    result = repair(Path(args.path), args.mpp_x, args.mpp_y, args.rgb, args.max_workers)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.audit:
        Path(args.audit).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
