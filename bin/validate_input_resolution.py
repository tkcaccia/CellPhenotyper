#!/usr/bin/env python3
"""Validate native physical resolution before cell-level WSI processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MICRONS_PER_INCH = 25_400.0
MICRONS_PER_CENTIMETER = 10_000.0


@dataclass
class ResolutionInfo:
    width_px: int
    height_px: int
    mpp_x: float | None
    mpp_y: float | None
    metadata_source: str


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _rational_to_float(value: Any) -> float | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = _positive_float(value[0])
        denominator = _positive_float(value[1])
        if numerator is not None and denominator is not None:
            return numerator / denominator
    return _positive_float(value)


def _parse_description(description: str) -> tuple[float | None, float | None, str]:
    if not description:
        return None, None, ""

    patterns = (
        (r'PhysicalSizeX=["\']([0-9]+(?:\.[0-9]+)?)', r'PhysicalSizeY=["\']([0-9]+(?:\.[0-9]+)?)', "ome-xml"),
        (r'(?i)\bMPP\s*=\s*([0-9]+(?:\.[0-9]+)?)', None, "image-description-mpp"),
        (r'(?i)\bmicrons[_ ]per[_ ]pixel[_ ]x\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
         r'(?i)\bmicrons[_ ]per[_ ]pixel[_ ]y\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
         "image-description-mpp-xy"),
    )
    for x_pattern, y_pattern, source in patterns:
        x_match = re.search(x_pattern, description)
        if not x_match:
            continue
        mpp_x = _positive_float(x_match.group(1))
        y_match = re.search(y_pattern, description) if y_pattern else None
        mpp_y = _positive_float(y_match.group(1)) if y_match else mpp_x
        if source == "ome-xml":
            x_unit_match = re.search(r'PhysicalSizeXUnit=["\']([^"\']+)', description)
            y_unit_match = re.search(r'PhysicalSizeYUnit=["\']([^"\']+)', description)
            mpp_x = _physical_size_to_microns(
                mpp_x, x_unit_match.group(1) if x_unit_match else None
            )
            mpp_y = _physical_size_to_microns(
                mpp_y, y_unit_match.group(1) if y_unit_match else None
            )
        if mpp_x is not None:
            return mpp_x, mpp_y or mpp_x, source
    return None, None, ""


def _physical_size_to_microns(value: float | None, unit: str | None) -> float | None:
    if value is None or not unit:
        return value
    normalized = unit.strip().lower().replace("μ", "µ")
    if normalized in {"µm", "um", "micrometer", "micrometre"}:
        return value
    if normalized in {"nm", "nanometer", "nanometre"}:
        return value / 1000.0
    if normalized in {"mm", "millimeter", "millimetre"}:
        return value * 1000.0
    return value


def _resolution_unit_microns(value: Any) -> float | None:
    name = str(value).upper()
    numeric = None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        try:
            numeric = int(getattr(value, "value"))
        except (TypeError, ValueError, AttributeError):
            pass
    if "INCH" in name or numeric == 2:
        return MICRONS_PER_INCH
    if "CENTIMETER" in name or "CENTIMETRE" in name or numeric == 3:
        return MICRONS_PER_CENTIMETER
    return None


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _json_scalar_or_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return [int(item) if isinstance(item, (int, bool)) else str(item) for item in value]
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def _ome_channel_names(ome_xml: str | None) -> list[str]:
    if not ome_xml:
        return []
    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return []
    names: list[str] = []
    for channel in root.findall(".//{*}Channel"):
        name = str(channel.attrib.get("Name") or channel.attrib.get("ID") or "").strip()
        if name:
            names.append(name)
    return names


def _inspect_with_tifffile(path: Path) -> dict[str, Any]:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series or not tif.pages:
            raise RuntimeError("TIFF contains no image series")
        series = tif.series[0]
        levels = list(getattr(series, "levels", None) or [series])
        page = series.pages[0]
        shape = [int(value) for value in series.shape]
        axes = str(getattr(series, "axes", "") or "")
        channel_count = None
        for axis in ("C", "S"):
            if axis in axes:
                channel_count = int(shape[axes.index(axis)])
                break
        if channel_count is None:
            channel_count = int(getattr(page, "samplesperpixel", 1) or 1)
        photometric = _enum_name(getattr(page, "photometric", None))
        return {
            "inspection_backend": "tifffile",
            "format": "OME-TIFF" if bool(tif.ome_metadata) else "TIFF",
            "is_ome": bool(tif.ome_metadata),
            "is_bigtiff": bool(tif.is_bigtiff),
            "series_count": int(len(tif.series)),
            "page_count": int(len(tif.pages)),
            "shape": shape,
            "axes": axes,
            "dtype": str(series.dtype),
            "bits_per_sample": _json_scalar_or_list(getattr(page, "bitspersample", None)),
            "channel_count": channel_count,
            "channel_names": _ome_channel_names(tif.ome_metadata),
            "photometric": photometric,
            "color_interpretation": "RGB" if str(photometric).upper() == "RGB" else photometric,
            "planar_configuration": _enum_name(getattr(page, "planarconfig", None)),
            "compression": _enum_name(getattr(page, "compression", None)),
            "extra_samples": [
                _enum_name(value) for value in (getattr(page, "extrasamples", None) or ())
            ],
            "is_tiled": bool(getattr(page, "is_tiled", False)),
            "tile_width_px": int(getattr(page, "tilewidth", 0) or 0) or None,
            "tile_height_px": int(getattr(page, "tilelength", 0) or 0) or None,
            "pyramid_level_count": int(len(levels)),
            "pyramid_level_shapes": [
                [int(value) for value in level.shape] for level in levels
            ],
        }


def _inspect_with_pyvips(path: Path) -> dict[str, Any]:
    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="sequential", page=0)
    format_bits = {
        "uchar": 8,
        "char": 8,
        "ushort": 16,
        "short": 16,
        "uint": 32,
        "int": 32,
        "float": 32,
        "double": 64,
        "complex": 64,
        "dpcomplex": 128,
    }
    return {
        "inspection_backend": "pyvips",
        "format": path.suffix.lower().lstrip(".").upper() or "unknown",
        "is_ome": False,
        "is_bigtiff": None,
        "series_count": None,
        "page_count": None,
        "shape": [int(image.height), int(image.width), int(image.bands)],
        "axes": "YXC",
        "dtype": str(image.format),
        "bits_per_sample": format_bits.get(str(image.format)),
        "channel_count": int(image.bands),
        "channel_names": [],
        "photometric": None,
        "color_interpretation": str(image.interpretation),
        "planar_configuration": None,
        "compression": None,
        "extra_samples": [],
        "is_tiled": None,
        "tile_width_px": None,
        "tile_height_px": None,
        "pyramid_level_count": 1,
        "pyramid_level_shapes": [[int(image.height), int(image.width), int(image.bands)]],
    }


def inspect_image_layout(path: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    for reader in (_inspect_with_tifffile, _inspect_with_pyvips):
        try:
            return reader(path), errors
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
    return {
        "inspection_backend": "unresolved",
        "format": path.suffix.lower().lstrip(".").upper() or "unknown",
    }, errors


def _add_mpp_candidate(
    candidates: list[dict[str, Any]], source: str, mpp_x: Any, mpp_y: Any
) -> None:
    parsed_x = _positive_float(mpp_x)
    parsed_y = _positive_float(mpp_y) or parsed_x
    if parsed_x is None or parsed_y is None:
        return
    candidate = {"source": source, "mpp_x": parsed_x, "mpp_y": parsed_y}
    if candidate not in candidates:
        candidates.append(candidate)


def inspect_mpp_candidates(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            for label, description in (
                ("ome-xml", tif.ome_metadata or ""),
                ("page-description", page.description or ""),
            ):
                mpp_x, mpp_y, parsed_source = _parse_description(description)
                if mpp_x is not None:
                    _add_mpp_candidate(
                        candidates, f"{label}:{parsed_source}", mpp_x, mpp_y
                    )
            tags = page.tags
            x_resolution = (
                _rational_to_float(tags["XResolution"].value)
                if "XResolution" in tags
                else None
            )
            y_resolution = (
                _rational_to_float(tags["YResolution"].value)
                if "YResolution" in tags
                else None
            )
            resolution_unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
            unit_microns = _resolution_unit_microns(resolution_unit)
            if unit_microns and x_resolution:
                _add_mpp_candidate(
                    candidates,
                    "tiff-resolution-tags",
                    unit_microns / x_resolution,
                    unit_microns / (y_resolution or x_resolution),
                )
    except Exception as exc:
        errors.append(f"tifffile MPP inspection: {exc}")

    try:
        import pyvips

        image = pyvips.Image.new_from_file(str(path), access="sequential", page=0)

        def get_property(name: str) -> Any:
            try:
                return image.get(name) if image.get_typeof(name) else None
            except Exception:
                return None

        openslide_x = _positive_float(get_property("openslide.mpp-x"))
        openslide_y = _positive_float(get_property("openslide.mpp-y"))
        _add_mpp_candidate(candidates, "openslide-mpp", openslide_x, openslide_y)
        description = str(get_property("image-description") or "")
        mpp_x, mpp_y, parsed_source = _parse_description(description)
        if mpp_x is not None:
            _add_mpp_candidate(
                candidates, f"pyvips:{parsed_source}", mpp_x, mpp_y
            )
        if not candidates:
            xres = _positive_float(getattr(image, "xres", None))
            yres = _positive_float(getattr(image, "yres", None))
            if xres is not None:
                _add_mpp_candidate(
                    candidates,
                    "pyvips-pixels-per-mm",
                    1000.0 / xres,
                    1000.0 / (yres or xres),
                )
    except Exception as exc:
        errors.append(f"pyvips MPP inspection: {exc}")
    return candidates, errors


def find_mpp_conflicts(
    candidates: list[dict[str, Any]], max_fraction: float
) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            x_denom = max(1.0e-12, (left["mpp_x"] + right["mpp_x"]) / 2.0)
            y_denom = max(1.0e-12, (left["mpp_y"] + right["mpp_y"]) / 2.0)
            fraction = max(
                abs(left["mpp_x"] - right["mpp_x"]) / x_denom,
                abs(left["mpp_y"] - right["mpp_y"]) / y_denom,
            )
            if fraction > max_fraction:
                conflicts.append(
                    {
                        "left_source": left["source"],
                        "right_source": right["source"],
                        "relative_difference": fraction,
                        "left_mpp_xy": [left["mpp_x"], left["mpp_y"]],
                        "right_mpp_xy": [right["mpp_x"], right["mpp_y"]],
                    }
                )
    return conflicts


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_with_tifffile(path: Path) -> ResolutionInfo:
    import tifffile

    with tifffile.TiffFile(str(path)) as tif:
        if not tif.pages:
            raise RuntimeError("TIFF contains no pages")
        page = tif.pages[0]
        width = int(page.imagewidth)
        height = int(page.imagelength)

        descriptions = [tif.ome_metadata or "", page.description or ""]
        for description in descriptions:
            mpp_x, mpp_y, source = _parse_description(description)
            if mpp_x is not None:
                return ResolutionInfo(width, height, mpp_x, mpp_y, source)

        tags = page.tags
        x_resolution = _rational_to_float(tags["XResolution"].value) if "XResolution" in tags else None
        y_resolution = _rational_to_float(tags["YResolution"].value) if "YResolution" in tags else None
        resolution_unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else None
        unit_microns = _resolution_unit_microns(resolution_unit)
        if unit_microns and x_resolution:
            mpp_x = unit_microns / x_resolution
            mpp_y = unit_microns / (y_resolution or x_resolution)
            return ResolutionInfo(width, height, mpp_x, mpp_y, "tiff-resolution-tags")

        return ResolutionInfo(width, height, None, None, "unresolved")


def _read_with_pyvips(path: Path) -> ResolutionInfo:
    import pyvips

    image = pyvips.Image.new_from_file(str(path), access="sequential", page=0)
    width = int(image.width)
    height = int(image.height)

    def get_property(name: str) -> Any:
        try:
            return image.get(name) if image.get_typeof(name) else None
        except Exception:
            return None

    mpp_x = _positive_float(get_property("openslide.mpp-x"))
    mpp_y = _positive_float(get_property("openslide.mpp-y"))
    if mpp_x is not None:
        return ResolutionInfo(width, height, mpp_x, mpp_y or mpp_x, "openslide-mpp")

    description = str(get_property("image-description") or "")
    mpp_x, mpp_y, source = _parse_description(description)
    if mpp_x is not None:
        return ResolutionInfo(width, height, mpp_x, mpp_y, f"pyvips-{source}")

    xres = _positive_float(getattr(image, "xres", None))
    yres = _positive_float(getattr(image, "yres", None))
    if xres is not None:
        return ResolutionInfo(width, height, 1000.0 / xres, 1000.0 / (yres or xres), "pyvips-pixels-per-mm")
    return ResolutionInfo(width, height, None, None, "unresolved")


def read_resolution(path: Path) -> ResolutionInfo:
    errors: list[str] = []
    for reader in (_read_with_tifffile, _read_with_pyvips):
        try:
            info = reader(path)
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")
            continue
        if info.mpp_x is not None:
            return info
        errors.append(f"{reader.__name__}: physical pixel size unresolved")
    raise RuntimeError("; ".join(errors))


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    path = Path(args.image).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Input image is missing or empty: {path}")

    warnings: list[str] = []
    failures: list[str] = []
    require_rgb = bool(getattr(args, "require_rgb", False))
    require_pyramid = bool(getattr(args, "require_pyramid", False))
    hash_file = bool(getattr(args, "hash_file", True))
    max_metadata_conflict_fraction = float(
        getattr(args, "max_metadata_conflict_fraction", 0.02)
    )
    layout, inspection_errors = inspect_image_layout(path)
    mpp_candidates, mpp_inspection_errors = inspect_mpp_candidates(path)
    plausible_mpp_candidates = [
        candidate
        for candidate in mpp_candidates
        if args.min_mpp <= candidate["mpp_x"] <= args.max_mpp
        and args.min_mpp <= candidate["mpp_y"] <= args.max_mpp
    ]
    out_of_range_mpp_candidates = [
        candidate for candidate in mpp_candidates if candidate not in plausible_mpp_candidates
    ]
    mpp_conflicts = find_mpp_conflicts(
        plausible_mpp_candidates, max_metadata_conflict_fraction
    )
    file_sha256 = None
    if hash_file:
        try:
            file_sha256 = sha256_file(path)
        except OSError as exc:
            failures.append(f"Unable to SHA-256 hash the input image: {exc}")

    try:
        info = read_resolution(path)
        read_error = None
    except Exception as exc:
        info = ResolutionInfo(0, 0, None, None, "unresolved")
        read_error = str(exc)

    override = _positive_float(args.override_mpp)
    if override is not None:
        if info.mpp_x is not None:
            warnings.append(
                f"Metadata MPP ({info.mpp_x:.6g}, {info.mpp_y:.6g}) was replaced by explicit override {override:.6g}."
            )
        info.mpp_x = override
        info.mpp_y = override
        info.metadata_source = "explicit-override"

    if mpp_conflicts:
        message = (
            f"Found {len(mpp_conflicts)} conflicting physical-resolution metadata pair(s) "
            f"above {max_metadata_conflict_fraction:.2%}."
        )
        if override is None:
            failures.append(message + " Supply an independently verified --override-mpp.")
        else:
            warnings.append(message + " The explicit override is authoritative for this run.")
    if out_of_range_mpp_candidates and plausible_mpp_candidates:
        ignored_sources = ", ".join(
            f"{candidate['source']}=({candidate['mpp_x']:.6g}, {candidate['mpp_y']:.6g})"
            for candidate in out_of_range_mpp_candidates
        )
        warnings.append(
            "Excluded physical-resolution candidate(s) outside the configured cell-analysis "
            f"sanity range [{args.min_mpp:g}, {args.max_mpp:g}] µm/px from metadata conflict "
            f"testing: {ignored_sources}."
        )

    color_interpretation = str(layout.get("color_interpretation") or "").upper()
    channel_count = layout.get("channel_count")
    rgb_compatible = any(
        token in color_interpretation for token in ("RGB", "YCBCR")
    )
    if require_rgb and (not rgb_compatible or int(channel_count or 0) != 3):
        failures.append(
            "Converted analysis input must be a three-channel RGB-compatible color image; "
            f"observed color={layout.get('color_interpretation')} channels={channel_count}."
        )
    pyramid_level_count = int(layout.get("pyramid_level_count") or 0)
    if require_pyramid and pyramid_level_count < 2:
        failures.append(
            "Converted analysis input must be pyramidal; "
            f"observed pyramid_level_count={pyramid_level_count}."
        )

    if info.mpp_x is None or info.mpp_y is None:
        failures.append(
            "Physical pixel size could not be resolved. Provide valid TIFF/OME metadata or --override-mpp."
        )
    else:
        if info.mpp_x < args.min_mpp or info.mpp_y < args.min_mpp:
            failures.append(
                f"MPP is below the metadata sanity limit {args.min_mpp:g} µm/px: "
                f"x={info.mpp_x:.6g}, y={info.mpp_y:.6g}."
            )
        if info.mpp_x > args.max_mpp or info.mpp_y > args.max_mpp:
            failures.append(
                f"Native resolution is too coarse for cell-level processing; maximum allowed is "
                f"{args.max_mpp:g} µm/px but image has x={info.mpp_x:.6g}, y={info.mpp_y:.6g}."
            )
        mean_mpp = (info.mpp_x + info.mpp_y) / 2.0
        anisotropy = abs(info.mpp_x - info.mpp_y) / mean_mpp
        if anisotropy > args.max_anisotropy_fraction:
            failures.append(
                f"X/Y MPP anisotropy {anisotropy:.2%} exceeds the allowed "
                f"{args.max_anisotropy_fraction:.2%}."
            )

    reference: dict[str, Any] | None = None
    byte_identical_to_reference = None
    conversion_relationship = None
    if args.reference_report:
        reference_path = Path(args.reference_report)
        reference = json.loads(reference_path.read_text())
        ref_mpp = _positive_float(reference.get("effective_mpp"))
        current_mpp = None
        if info.mpp_x is not None and info.mpp_y is not None:
            current_mpp = (info.mpp_x + info.mpp_y) / 2.0
        if ref_mpp is None or current_mpp is None:
            failures.append("Unable to compare converted MPP with the source-resolution report.")
        else:
            drift = abs(current_mpp - ref_mpp) / ref_mpp
            if drift > args.max_conversion_drift_fraction:
                failures.append(
                    f"Conversion changed physical resolution by {drift:.2%}; maximum allowed drift is "
                    f"{args.max_conversion_drift_fraction:.2%}."
                )
        reference_sha256 = reference.get("file_sha256")
        if file_sha256 and reference_sha256:
            byte_identical_to_reference = file_sha256 == reference_sha256
            conversion_relationship = (
                "byte_identical_copy"
                if byte_identical_to_reference
                else "rewritten_recompressed_or_resampled"
            )

    effective_mpp = None
    anisotropy = None
    upscale_to_cell_target = None
    if info.mpp_x is not None and info.mpp_y is not None:
        effective_mpp = (info.mpp_x + info.mpp_y) / 2.0
        anisotropy = abs(info.mpp_x - info.mpp_y) / effective_mpp
        upscale_to_cell_target = effective_mpp / args.cell_target_mpp
        if upscale_to_cell_target > 1.0:
            warnings.append(
                f"Cell-model input at {args.cell_target_mpp:g} µm/px requires "
                f"{upscale_to_cell_target:.3f}x linear upsampling; no spatial detail is created by resampling."
            )

    passed = not failures
    enforced = bool(args.strict)
    report: dict[str, Any] = {
        "schema_version": 2,
        "image": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": file_sha256,
        "hash_algorithm": "sha256" if hash_file else None,
        "image_layout": layout,
        "image_inspection_errors": inspection_errors,
        **asdict(info),
        "effective_mpp": effective_mpp,
        "mpp_candidates": mpp_candidates,
        "plausible_mpp_candidates": plausible_mpp_candidates,
        "out_of_range_mpp_candidates": out_of_range_mpp_candidates,
        "mpp_inspection_errors": mpp_inspection_errors,
        "mpp_conflicts": mpp_conflicts,
        "cell_target_mpp": args.cell_target_mpp,
        "linear_upsample_factor_to_cell_target": upscale_to_cell_target,
        "anisotropy_fraction": anisotropy,
        "requirements": {
            "three_channel_rgb": require_rgb,
            "pyramidal": require_pyramid,
            "full_file_sha256": hash_file,
        },
        "limits": {
            "min_mpp": args.min_mpp,
            "max_mpp": args.max_mpp,
            "max_anisotropy_fraction": args.max_anisotropy_fraction,
            "max_conversion_drift_fraction": args.max_conversion_drift_fraction,
            "max_metadata_conflict_fraction": max_metadata_conflict_fraction,
        },
        "strict": enforced,
        "status": "pass" if passed else ("fail" if enforced else "warning"),
        "warnings": warnings,
        "failures": failures,
        "metadata_read_error": read_error,
    }
    if reference is not None:
        report["reference_report"] = str(Path(args.reference_report).resolve())
        report["reference_effective_mpp"] = reference.get("effective_mpp")
        report["reference_file_sha256"] = reference.get("file_sha256")
        report["byte_identical_to_reference"] = byte_identical_to_reference
        report["conversion_relationship"] = conversion_relationship
    return report, passed or not enforced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-mpp", type=float, default=0.05)
    parser.add_argument("--max-mpp", type=float, default=0.50)
    parser.add_argument("--cell-target-mpp", type=float, default=0.25)
    parser.add_argument("--max-anisotropy-fraction", type=float, default=0.05)
    parser.add_argument("--max-conversion-drift-fraction", type=float, default=0.02)
    parser.add_argument("--max-metadata-conflict-fraction", type=float, default=0.02)
    parser.add_argument("--override-mpp", type=float, default=0.0)
    parser.add_argument("--reference-report", default="")
    parser.add_argument("--require-rgb", action="store_true")
    parser.add_argument("--require-pyramid", action="store_true")
    parser.add_argument("--skip-file-hash", dest="hash_file", action="store_false")
    parser.set_defaults(hash_file=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.min_mpp <= 0 or args.max_mpp <= args.min_mpp:
        parser.error("Require 0 < --min-mpp < --max-mpp")
    if args.cell_target_mpp <= 0:
        parser.error("--cell-target-mpp must be positive")
    if not 0 <= args.max_anisotropy_fraction < 1:
        parser.error("--max-anisotropy-fraction must be in [0, 1)")
    if not 0 <= args.max_conversion_drift_fraction < 1:
        parser.error("--max-conversion-drift-fraction must be in [0, 1)")
    if not 0 <= args.max_metadata_conflict_fraction < 1:
        parser.error("--max-metadata-conflict-fraction must be in [0, 1)")
    return args


def main() -> int:
    args = parse_args()
    try:
        report, accepted = validate(args)
    except Exception as exc:
        report = {
            "schema_version": 2,
            "image": str(Path(args.image).resolve()),
            "strict": bool(args.strict),
            "status": "fail" if args.strict else "warning",
            "warnings": [],
            "failures": [str(exc)],
        }
        accepted = not args.strict

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    stream = sys.stdout if accepted else sys.stderr
    print(
        f"[{'INFO' if accepted else 'ERROR'}] Input resolution status={report['status']} "
        f"mpp={report.get('effective_mpp')} report={report_path}",
        file=stream,
        flush=True,
    )
    for message in report.get("warnings", []):
        print(f"[WARN] {message}", file=stream, flush=True)
    for message in report.get("failures", []):
        print(f"[ERROR] {message}", file=stream, flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
