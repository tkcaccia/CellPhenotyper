#!/usr/bin/env python3
"""Validate native physical resolution before cell-level WSI processing."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
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
        if mpp_x is not None:
            return mpp_x, mpp_y or mpp_x, source
    return None, None, ""


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

    try:
        info = read_resolution(path)
        read_error = None
    except Exception as exc:
        info = ResolutionInfo(0, 0, None, None, "unresolved")
        read_error = str(exc)

    override = _positive_float(args.override_mpp)
    warnings: list[str] = []
    failures: list[str] = []
    if override is not None:
        if info.mpp_x is not None:
            warnings.append(
                f"Metadata MPP ({info.mpp_x:.6g}, {info.mpp_y:.6g}) was replaced by explicit override {override:.6g}."
            )
        info.mpp_x = override
        info.mpp_y = override
        info.metadata_source = "explicit-override"

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
        "schema_version": 1,
        "image": str(path),
        "file_size_bytes": path.stat().st_size,
        **asdict(info),
        "effective_mpp": effective_mpp,
        "cell_target_mpp": args.cell_target_mpp,
        "linear_upsample_factor_to_cell_target": upscale_to_cell_target,
        "anisotropy_fraction": anisotropy,
        "limits": {
            "min_mpp": args.min_mpp,
            "max_mpp": args.max_mpp,
            "max_anisotropy_fraction": args.max_anisotropy_fraction,
            "max_conversion_drift_fraction": args.max_conversion_drift_fraction,
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
    parser.add_argument("--override-mpp", type=float, default=0.0)
    parser.add_argument("--reference-report", default="")
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
    return args


def main() -> int:
    args = parse_args()
    try:
        report, accepted = validate(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
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
