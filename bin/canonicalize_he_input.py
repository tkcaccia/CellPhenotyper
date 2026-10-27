#!/usr/bin/env python3
"""Verify canonical 8-bit H&E RGB at pipeline entry; correct declared BGR once.

The RGB channel order cannot be recovered reliably from TIFF RGB tags alone.
For known mislabelled sources, --channel-order bgr performs a lossless,
whole-slide channel swap before any segmentation or GigaTIME inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _inspect(path: Path) -> dict:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError("Input contains no TIFF pages")
        page = tif.pages[0]
        shape = tuple(int(v) for v in page.shape)
        dtype = str(page.dtype)
        samples = int(page.samplesperpixel or 1)
        photometric = str(getattr(page.photometric, "name", page.photometric)).upper()
        if len(shape) != 3 or shape[-1] != 3 or samples != 3 or dtype != "uint8":
            raise ValueError(
                "GigaTIME and StarDist require a shared three-channel uint8 H&E image "
                f"at pipeline entry; observed shape={shape}, samples={samples}, dtype={dtype}. "
                "For higher-bit-depth input, first apply one documented whole-image intensity mapping."
            )
        if photometric not in {"RGB", "YCBCR"}:
            raise ValueError(f"Input is not RGB-compatible: photometric={photometric}")
        return {
            "shape_yxc": list(shape),
            "dtype": dtype,
            "photometric": photometric,
            "compression": str(getattr(page.compression, "name", page.compression)),
            "is_ome": bool(tif.ome_metadata),
        }


def canonicalize(path: Path, report_path: Path, *, channel_order: str,
                 source_resolution_report: Path) -> dict:
    if channel_order not in {"rgb", "bgr"}:
        raise ValueError("--channel-order must be rgb or bgr")
    before = _inspect(path)
    resolution = json.loads(source_resolution_report.read_text(encoding="utf-8"))
    mpp_x = float(resolution.get("mpp_x") or resolution.get("effective_mpp") or 0)
    mpp_y = float(resolution.get("mpp_y") or resolution.get("effective_mpp") or 0)
    if channel_order == "bgr" and (not 0 < mpp_x <= .5 or not 0 < mpp_y <= .5):
        raise ValueError("A validated source microns-per-pixel value is required before BGR correction")
    action = "declared_rgb_no_pixel_change"
    if channel_order == "bgr":
        import pyvips

        source = pyvips.Image.new_from_file(str(path), access="sequential", page=0)
        if source.bands != 3 or source.format != "uchar":
            raise ValueError("Streaming BGR correction requires three uint8 bands")
        rgb = source[2].bandjoin([source[1], source[0]]).copy(
            xres=1000.0 / mpp_x, yres=1000.0 / mpp_y)
        temporary = path.with_name(f".{path.name}.canonical-{os.getpid()}.tif")
        try:
            rgb.tiffsave(str(temporary), compression="lzw", tile=True,
                         tile_width=512, tile_height=512, pyramid=True,
                         bigtiff=True, properties=False)
            after = _inspect(temporary)
            if after["shape_yxc"] != before["shape_yxc"]:
                raise ValueError("BGR correction changed the level-zero image dimensions")
            # A new file is installed only after format and physical-spacing checks.
            from validate_input_resolution import read_resolution

            actual = read_resolution(temporary)
            if actual.mpp_x is None or actual.mpp_y is None:
                raise ValueError("Corrected image lost physical pixel spacing")
            if max(abs(actual.mpp_x / mpp_x - 1), abs(actual.mpp_y / mpp_y - 1)) > .001:
                raise ValueError("Corrected image physical spacing drifted")
            os.replace(temporary, path)
            action = "whole_slide_bgr_to_rgb_lossless_lzw"
        finally:
            temporary.unlink(missing_ok=True)
    result = {
        "schema_version": 1,
        "image": str(path.resolve()),
        "source_resolution_report": str(source_resolution_report.resolve()),
        "source_channel_order": channel_order,
        "canonical_channel_order": "rgb",
        "action": action,
        "mpp_x": mpp_x if mpp_x > 0 else None,
        "mpp_y": mpp_y if mpp_y > 0 else None,
        "intensity_mapping": "identity_uint8",
        "model_normalization": "GigaTIME applies its fixed RGB channel mean/std after 0.25-um/px resampling",
        "input_layout": before,
        "output_layout": _inspect(path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--source-resolution-report", type=Path, required=True)
    p.add_argument("--channel-order", choices=("rgb", "bgr"), default="rgb")
    args = p.parse_args()
    print(json.dumps(canonicalize(args.image, args.report,
                                  channel_order=args.channel_order,
                                  source_resolution_report=args.source_resolution_report), indent=2))


if __name__ == "__main__":
    main()
