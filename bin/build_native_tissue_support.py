#!/usr/bin/env python3
"""Experimental native bright-background subtraction for spatial graphs only.

This is a deterministic H&E rule, not a validated tissue segmentation model.
It never consumes annotations, cell masks or domain labels. It cannot recover
tissue excluded upstream, or distinguish optically identical glass and tissue.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from cell_profile_io import RasterReader, calibration, sha256_file


FORMAT = "cellphenotyper_native_brightfield_support"
VERSION = "1.1.0"
CODES = {"0": "upstream_excluded", "1": "retained_support",
         "2": "image_rule_excluded_bright_interior", "3": "retained_ambiguous_bright_pixel"}


def classify_rgb(rgb, *, radius_px=2, minimum_channel=248, maximum_chroma=8,
                 maximum_local_range=6, white_rgb=(255., 255., 255.)):
    """Classify exact native uint8 pixels with a square physical guard footprint.

    A whole neighbourhood must be near-white and low-chroma, with little
    luminance variation. Only its interior is excluded; uncertain edge pixels
    remain supported. Padding is never invented outside the actual image.
    """
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Native brightfield support requires explicit uint8 RGB H&E")
    normalized = np.minimum(rgb.astype(np.float32) * (255. / np.asarray(white_rgb, np.float32)), 255.)
    low, high = normalized.min(axis=-1), normalized.max(axis=-1)
    candidate = (low >= minimum_channel) & ((high - low) <= maximum_chroma)
    size = 2 * radius_px + 1
    # A missing external image pixel is not evidence of background.
    interior = ndimage.minimum_filter(candidate, size=size, mode="constant", cval=0)
    luminance = normalized.mean(axis=-1)
    local_range = (ndimage.maximum_filter(luminance, size=size, mode="nearest")
                   - ndimage.minimum_filter(luminance, size=size, mode="nearest"))
    return candidate, interior & (local_range <= maximum_local_range)


def sample_support_window(reader, bounds, native_shape):
    """Categorical pixel-centre sampling of the declared crop-extent raster."""
    x0, y0, x1, y1 = bounds
    height, width = native_shape
    xs = ((2 * np.arange(x0, x1, dtype=np.int64) + 1) * reader.width // (2 * width))
    ys = ((2 * np.arange(y0, y1, dtype=np.int64) + 1) * reader.height // (2 * height))
    block = reader.window(int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
    if block.ndim != 2 or block.dtype.kind not in "buif" or not np.isfinite(block).all() or (block < 0).any():
        raise ValueError("Upstream support must be a finite nonnegative two-dimensional raster")
    return block[np.ix_(ys - ys[0], xs - xs[0])] > 0


def estimate_background_white(rgb_reader, support_reader, *, max_samples=250_000):
    """Estimate slide background colour from bright upstream-excluded samples.

    A fixed global lattice makes this independent of processing tile size.
    The brightest fifth must contain >=128 samples, be bright, and have a
    narrow per-channel 10..90 percentile range. Otherwise use nominal white.
    This is scanner-colour adaptation, not tissue annotation or stain removal.
    """
    height, width = rgb_reader.height, rgb_reader.width
    step = max(1, math.ceil(math.sqrt(height * width / max_samples)))
    samples = []
    max_window = 0
    for y0 in range(0, height, 512):
        for x0 in range(0, width, 512):
            bounds = (x0, y0, min(width, x0+512), min(height, y0+512))
            ys, xs = np.arange((-y0) % step, bounds[3]-y0, step), np.arange((-x0) % step, bounds[2]-x0, step)
            if not len(ys) or not len(xs):
                continue
            base = sample_support_window(support_reader, bounds, (height, width))[np.ix_(ys, xs)]
            if base.all():
                continue
            rgb = rgb_reader.window(*bounds)[np.ix_(ys, xs)]
            max_window = max(max_window, (bounds[3]-y0)*(bounds[2]-x0))
            samples.append(rgb[~base])
    excluded = np.concatenate(samples) if samples else np.empty((0, 3), np.uint8)
    result = {"method": "brightest_fifth_of_fixed_lattice_upstream_excluded_pixels",
              "sample_stride_pixels": step, "excluded_sample_count": len(excluded),
              "max_image_window_pixels": max_window,
              "white_rgb": [255., 255., 255.], "status": "nominal_white_fallback"}
    if len(excluded) < 640:
        result["reason"] = "insufficient_upstream_excluded_samples"
        return result
    luminance = excluded.astype(np.float32).mean(axis=1)
    selected = excluded[luminance >= np.quantile(luminance, .8)]
    white = np.median(selected, axis=0)
    spread = np.quantile(selected, .9, axis=0) - np.quantile(selected, .1, axis=0)
    result.update({"bright_sample_count": len(selected), "bright_channel_p10_p90_range": spread.tolist()})
    if len(selected) < 128 or white.min() < 180 or spread.max() > 12 or white.max()-white.min() > 40:
        result["reason"] = "background_samples_not_consistently_bright"
        return result
    result.update({"status": "estimated_from_upstream_exclusions", "white_rgb": white.tolist()})
    return result


def build_support(image, support_mask, shift, resolution_json, outdir, *, tile_size=512,
                  guard_radius_um=.5, minimum_channel=248, maximum_chroma=8,
                  maximum_local_range=6, white_reference="auto"):
    start = time.monotonic()
    outdir = Path(outdir)
    if outdir.exists():
        raise FileExistsError("Native support requires a new output directory; existing results are preserved")
    if isinstance(tile_size, bool) or int(tile_size) != tile_size or tile_size < 16 or tile_size % 16:
        raise ValueError("tile_size must be a positive multiple of 16")
    for name, value in (("minimum_channel", minimum_channel), ("maximum_chroma", maximum_chroma),
                        ("maximum_local_range", maximum_local_range)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 255:
            raise ValueError(f"{name} must lie within 0..255")
    if not math.isfinite(guard_radius_um) or not 0 <= guard_radius_um <= 10:
        raise ValueError("guard_radius_um must be finite within 0..10 micrometres")
    if white_reference not in ("auto", "nominal"):
        raise ValueError("white_reference must be auto or nominal")
    paths = {"image": Path(image), "support_mask": Path(support_mask), "shift": Path(shift),
             "resolution_json": Path(resolution_json)}
    inputs = {key: {"path": str(path.resolve()), "sha256": sha256_file(path)} for key, path in paths.items()}
    cal = calibration(shift, resolution_json)
    radius_px = math.ceil(guard_radius_um / cal["mpp"])
    if radius_px > tile_size:
        raise ValueError("Physical guard exceeds tile size; increase tile_size or reduce guard_radius_um")
    height, width = cal["height"], cal["width"]
    counts = {key: 0 for key in CODES}
    stats = {"max_image_window_pixels": 0, "tiles": 0}
    output_paths = {"support": outdir / "support.tif", "reasons": outdir / "reasons.tif"}
    with RasterReader(image) as rgb_reader, RasterReader(support_mask) as support_reader:
        if rgb_reader.dtype != np.uint8 or rgb_reader.reader.shape != (height, width, 3):
            raise ValueError("Image must be uint8 RGB at the exact calibrated crop dimensions")
        if len(support_reader.reader.shape) != 2:
            raise ValueError("Support mask must be two dimensional")
        source_shape = [support_reader.height, support_reader.width]
        if support_reader.height > height or support_reader.width > width:
            raise ValueError("Support resolution must not exceed native crop resolution")
        white = estimate_background_white(rgb_reader, support_reader) if white_reference == "auto" else {
            "status": "explicit_nominal_white", "white_rgb": [255., 255., 255.]}
        outdir.mkdir(parents=True)

        def tiles():
            for y0 in range(0, height, tile_size):
                for x0 in range(0, width, tile_size):
                    x1, y1 = min(x0 + tile_size, width), min(y0 + tile_size, height)
                    xa, ya, xb, yb = max(0, x0-radius_px), max(0, y0-radius_px), min(width, x1+radius_px), min(height, y1+radius_px)
                    rgb = rgb_reader.window(xa, ya, xb, yb)
                    stats["max_image_window_pixels"] = max(stats["max_image_window_pixels"], rgb.shape[0] * rgb.shape[1])
                    candidate, excluded = classify_rgb(rgb, radius_px=radius_px, minimum_channel=minimum_channel,
                        maximum_chroma=maximum_chroma, maximum_local_range=maximum_local_range, white_rgb=white["white_rgb"])
                    view = np.s_[y0-ya:y1-ya, x0-xa:x1-xa]
                    candidate, excluded = candidate[view], excluded[view]
                    base = sample_support_window(support_reader, (x0, y0, x1, y1), (height, width))
                    reasons = np.where(base, np.where(excluded, 2, np.where(candidate, 3, 1)), 0).astype(np.uint8)
                    stats["tiles"] += 1
                    for code, count in enumerate(np.bincount(reasons.ravel(), minlength=4)):
                        counts[str(code)] += int(count)
                    # TIFF tile iterators require full tile shapes at image edges.
                    tile = np.zeros((tile_size, tile_size), np.uint8)
                    tile[:y1-y0, :x1-x0] = reasons
                    yield tile

        # One H&E pass, compressed categorical output; the second pass decodes
        # only its small uint8 tiles to produce the binary consumer contract.
        options = dict(shape=(height, width), dtype=np.uint8, tile=(tile_size, tile_size),
                       compression="deflate", photometric="minisblack", bigtiff=True,
                       resolution=(10000 / cal["mpp"], 10000 / cal["mpp"]), resolutionunit="CENTIMETER",
                       metadata={"axes": "YX"}, maxworkers=1)
        tifffile.imwrite(output_paths["reasons"], data=tiles(), **options)
    with RasterReader(output_paths["reasons"]) as reason_reader:
        def binary_tiles():
            for y0 in range(0, height, tile_size):
                for x0 in range(0, width, tile_size):
                    block = reason_reader.window(x0, y0, min(width, x0+tile_size), min(height, y0+tile_size))
                    tile = np.zeros((tile_size, tile_size), np.uint8)
                    tile[:block.shape[0], :block.shape[1]] = np.isin(block, (1, 3)).astype(np.uint8)
                    yield tile
        tifffile.imwrite(output_paths["support"], data=binary_tiles(), **options)
    for key, path in paths.items():
        if sha256_file(path) != inputs[key]["sha256"]:
            raise ValueError(f"Native support source changed during execution: {key}; incomplete output preserved")
    total = height * width
    stats["max_image_window_pixels"] = max(stats["max_image_window_pixels"], white.get("max_image_window_pixels", 0))
    manifest = {"format": FORMAT, "schema_version": VERSION, "complete": True,
        "method": "strict_near_white_low_chroma_low_variation_interior_subtraction",
        "inputs": inputs, "source_support_shape_yx": source_shape,
        "white_reference": white,
        "coordinate_frame": "crop_pixels", "shape_yx": [height, width],
        "origin_original_pixels_xy": cal["origin_px"].tolist(), "mpp_xy": [cal["mpp"]] * 2,
        "source_support_sampling": "nearest_native_pixel_centre_into_crop_extent_support",
        "parameters": {"minimum_channel": minimum_channel, "maximum_chroma": maximum_chroma,
            "maximum_local_range": maximum_local_range, "guard_radius_um": guard_radius_um,
            "guard_radius_px": radius_px, "actual_square_halfwidth_um": radius_px * cal["mpp"],
            "guard_footprint": "square_Chebyshev; missing_outside_image_is_not_background", "tile_size": tile_size,
            "threshold_intensity_basis": "uint8_equivalent_after_per_channel_white_reference_scaling_and_clipping"},
        "reason_codes": CODES, "pixel_counts": counts,
        "image_rule_excluded_fraction_of_upstream_support": counts["2"] / max(1, total-counts["0"]),
        "retained_fraction_of_crop": (counts["1"] + counts["3"]) / total,
        "outputs": {key: {"filename": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
                    for key, path in output_paths.items()},
        "producer_sha256": sha256_file(__file__), "runtime_seconds": time.monotonic()-start, **stats,
        "scope": "graph_and_density_support_only; canonical_cells_markers_compartments_and_domains_unchanged",
        "biological_validation": "not_established",
        "limitations": ["Near-white tissue and glass can be optically indistinguishable; pale-tissue loss remains possible.",
            "Darker, textured, coloured and sub-footprint gaps may remain supported.",
            "Cannot restore tissue excluded by upstream GrandQC or the analysis crop.",
            "Background colour estimation can be biased by bright artifacts; fallback and sample diagnostics require review.",
            "Native sampling and rule agreement are not calibrated biological confidence."]}
    (outdir / "support_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def verify_support_bundle(manifest_path, *, image_sha256, support_mask, shift, resolution_json):
    """Verify only declared in-bundle outputs; never follow input receipt paths."""
    manifest_path = Path(manifest_path)
    record = json.loads(manifest_path.read_text())
    if record.get("format") != FORMAT or record.get("schema_version") != VERSION or record.get("complete") is not True:
        raise ValueError("Native support bundle is incomplete or has an unsupported schema")
    expected = {"image": image_sha256, "support_mask": sha256_file(support_mask),
                "shift": sha256_file(shift), "resolution_json": sha256_file(resolution_json)}
    if not image_sha256 or any(record.get("inputs", {}).get(key, {}).get("sha256") != value for key, value in expected.items()):
        raise ValueError("Native support bundle source identities disagree with the canonical profile")
    cal = calibration(shift, resolution_json)
    if record.get("coordinate_frame") != "crop_pixels" or record.get("shape_yx") != [cal["height"], cal["width"]] or record.get("mpp_xy") != [cal["mpp"]] * 2 or record.get("origin_original_pixels_xy") != cal["origin_px"].tolist():
        raise ValueError("Native support bundle coordinate frame mismatch")
    for key, name in (("support", "support.tif"), ("reasons", "reasons.tif")):
        entry = record.get("outputs", {}).get(key, {})
        path = manifest_path.parent / name
        if entry.get("filename") != name or path.is_symlink() or not path.is_file() or entry.get("bytes") != path.stat().st_size or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"Native support output identity failure: {key}")
    if record.get("reason_codes") != CODES or record.get("biological_validation") != "not_established":
        raise ValueError("Native support bundle has an incompatible reason or validation contract")
    counts = {key: 0 for key in CODES}
    with RasterReader(manifest_path.parent / "support.tif") as binary, RasterReader(manifest_path.parent / "reasons.tif") as reasons, RasterReader(support_mask) as source:
        expected_shape = (cal["height"], cal["width"])
        if any(reader.reader.shape != expected_shape or reader.dtype != np.uint8 for reader in (binary, reasons)):
            raise ValueError("Native support outputs must be uint8 at the declared native dimensions")
        for y0 in range(0, cal["height"], 512):
            for x0 in range(0, cal["width"], 512):
                bounds = (x0, y0, min(cal["width"], x0+512), min(cal["height"], y0+512))
                codes, mask = reasons.window(*bounds), binary.window(*bounds)
                base = sample_support_window(source, bounds, expected_shape)
                if (codes > 3).any() or not np.array_equal(mask, np.isin(codes, (1, 3)).astype(np.uint8)) or not np.array_equal(codes == 0, ~base):
                    raise ValueError("Native support/reason rasters disagree with the upstream support contract")
                for code, count in enumerate(np.bincount(codes.ravel(), minlength=4)):
                    counts[str(code)] += int(count)
    if record.get("pixel_counts") != counts:
        raise ValueError("Native support pixel counts do not match the actual rasters")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "support-mask", "shift", "resolution-json", "outdir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--guard-radius-um", type=float, default=.5)
    p.add_argument("--minimum-channel", type=float, default=248)
    p.add_argument("--maximum-chroma", type=float, default=8)
    p.add_argument("--maximum-local-range", type=float, default=6)
    p.add_argument("--white-reference", choices=("auto", "nominal"), default="auto")
    result = build_support(**vars(p.parse_args()))
    print(json.dumps({"pixel_counts": result["pixel_counts"], "runtime_seconds": result["runtime_seconds"]}))


if __name__ == "__main__":
    main()
