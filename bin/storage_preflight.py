#!/usr/bin/env python3
"""Estimate CellPhenotyper disk demand before scheduling WSI stages."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


GIB = 1024**3
STAGE_ORDER = [
    "convert", "grandqc", "stardist", "cell_consensus", "tma", "tissue_mask",
    "cell_assignment", "cytoplasm", "gigatime", "marker_quantification",
    "grid_tiles", "uni2", "kodama", "clustering", "cluster_mask",
    "grow_tissue", "medsam_refine", "cluster_geojson", "neoplastic_section",
    "titan", "pathofmpred",
]

# Coefficients are intentionally conservative multiples of active RGB pixels.
# They estimate retained published files and retained/transient Nextflow work.
STAGE_FACTORS = {
    "convert": (0.50, 0.80),
    "grandqc": (0.12, 0.55),
    "stardist": (1.40, 2.10),
    "cell_consensus": (0.80, 1.80),
    "tma": (0.08, 0.15),
    "tissue_mask": (0.15, 0.25),
    "cell_assignment": (0.08, 0.15),
    "cytoplasm": (1.60, 1.90),
    "gigatime": (1.60, 2.30),
    "marker_quantification": (0.25, 0.45),
    "grid_tiles": (0.04, 0.12),
    "uni2": (1.20, 1.55),
    "kodama": (0.45, 0.75),
    "clustering": (0.12, 0.25),
    "cluster_mask": (0.90, 1.15),
    "grow_tissue": (0.18, 0.35),
    "medsam_refine": (0.35, 0.90),
    "cluster_geojson": (0.08, 0.15),
    "neoplastic_section": (0.70, 0.95),
    "titan": (0.03, 0.08),
    "pathofmpred": (0.02, 0.05),
}

CACHE_TARGET_GIB = {
    "stardist": 1.0,
    "grandqc": 2.0,
    "hf": 12.0,
    "titan": 12.0,
    "pathofmpred": 2.0,
    "singularity": 30.0,
}

CACHE_STAGES = {
    "stardist": {"stardist"},
    "grandqc": {"grandqc"},
    "hf": {"gigatime", "uni2"},
    "titan": {"titan"},
    "pathofmpred": {"pathofmpred"},
    "singularity": set(STAGE_ORDER),
}

KNOWN_IMAGE_SUFFIXES = (
    ".ome.tiff", ".ome.tif", ".tiff", ".tif", ".btf", ".czi", ".svs",
    ".ndpi", ".scn", ".mrxs", ".vms", ".vmu", ".png", ".jpeg", ".jpg",
)


@dataclass
class InputEstimate:
    path: str
    source_bytes: int
    width_px: int | None
    height_px: int | None
    dimension_source: str
    roi_path: str | None
    roi_bbox_fraction: float
    active_rgb_bytes: int
    estimate_basis: str


@dataclass
class Demand:
    kind: str
    label: str
    path: str
    expected_bytes: int
    worst_case_bytes: int
    note: str


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def strip_image_suffix(path: Path) -> str:
    lower = path.name.lower()
    for suffix in KNOWN_IMAGE_SUFFIXES:
        if lower.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def probe_dimensions(path: Path) -> tuple[int | None, int | None, str]:
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            axes = str(series.axes)
            shape = tuple(int(value) for value in series.shape)
            if "X" in axes and "Y" in axes:
                return shape[axes.index("X")], shape[axes.index("Y")], "tifffile"
    except Exception:
        pass
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as image:
            return int(image.width), int(image.height), "Pillow"
    except Exception:
        return None, None, "source_size_fallback"


def _coordinate_bounds(value, bounds: list[float]) -> None:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        bounds[0] = min(bounds[0], float(value[0]))
        bounds[1] = min(bounds[1], float(value[1]))
        bounds[2] = max(bounds[2], float(value[0]))
        bounds[3] = max(bounds[3], float(value[1]))
        return
    if isinstance(value, list):
        for item in value:
            _coordinate_bounds(item, bounds)


def roi_bbox_fraction(path: Path | None, width: int | None, height: int | None) -> float:
    if path is None or width is None or height is None or width <= 0 or height <= 0:
        return 1.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 1.0
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    for feature in payload.get("features") or []:
        _coordinate_bounds((feature.get("geometry") or {}).get("coordinates"), bounds)
    if not all(math.isfinite(value) for value in bounds):
        return 1.0
    x0, y0, x1, y1 = bounds
    clipped_width = max(0.0, min(float(width), x1) - max(0.0, x0))
    clipped_height = max(0.0, min(float(height), y1) - max(0.0, y0))
    fraction = clipped_width * clipped_height / float(width * height)
    # Include crop padding and avoid optimistic estimates for tiny annotations.
    return min(1.0, max(0.01, fraction * 1.15))


def matching_roi(image: Path, explicit_roi: Path | None, input_count: int) -> Path | None:
    if explicit_roi is not None and input_count == 1:
        return explicit_roi
    candidate = image.with_name(f"{strip_image_suffix(image)}.geojson")
    return candidate if candidate.is_file() else None


def estimate_input(
    path: Path,
    *,
    explicit_roi: Path | None,
    input_count: int,
    fallback_expansion: float,
) -> InputEstimate:
    source_bytes = path.stat().st_size
    width, height, source = probe_dimensions(path)
    roi = matching_roi(path, explicit_roi, input_count)
    fraction = roi_bbox_fraction(roi, width, height)
    if width and height:
        full_rgb = width * height * 3
        basis = f"{source} dimensions x ROI bounding-box fraction"
    else:
        full_rgb = int(source_bytes * fallback_expansion)
        basis = f"source bytes x fallback expansion {fallback_expansion:g}"
    active_rgb = max(source_bytes, int(math.ceil(full_rgb * fraction)))
    return InputEstimate(
        path=str(path.absolute()),
        source_bytes=source_bytes,
        width_px=width,
        height_px=height,
        dimension_source=source,
        roi_path=str(roi.absolute()) if roi else None,
        roi_bbox_fraction=fraction,
        active_rgb_bytes=active_rgb,
        estimate_basis=basis,
    )


def active_stages(start_point: str, end_point: str) -> list[str]:
    try:
        start = STAGE_ORDER.index(start_point)
        end = STAGE_ORDER.index(end_point)
    except ValueError as exc:
        raise ValueError(f"Unknown stage in window {start_point!r} -> {end_point!r}") from exc
    if start > end:
        raise ValueError(f"Stage window is reversed: {start_point} -> {end_point}")
    return STAGE_ORDER[start : end + 1]


def adjusted_stage_factors(
    stages: Iterable[str],
    *,
    cell_detection_mode: str,
    uni2_sampling_mode: str,
    gigatime_enabled: bool,
    marker_quantification_enabled: bool,
    gigatime_channel_count: int,
    uni2_save_tiles: bool,
) -> dict[str, tuple[float, float]]:
    factors = {stage: STAGE_FACTORS[stage] for stage in stages}
    if not gigatime_enabled:
        factors.pop("gigatime", None)
        factors.pop("marker_quantification", None)
    elif not marker_quantification_enabled:
        factors.pop("marker_quantification", None)
    if "gigatime" in factors:
        scale = max(1, gigatime_channel_count) / 5.0
        publish, work = factors["gigatime"]
        factors["gigatime"] = (publish * scale, work * max(1.0, scale))
    if cell_detection_mode != "consensus" and "cell_consensus" in factors:
        factors["cell_consensus"] = (0.05, 0.10)
    if cell_detection_mode == "cellvitpp" and "stardist" in factors:
        # This stage now prepares the common crop plus CellViT cell outputs;
        # it runs neither StarDist nor HoVer-Net. The bounded WSI controller
        # separately admits each shard against live space for retained vectors.
        factors["stardist"] = (2.20, 4.50)
        factors.pop("cell_consensus", None)
    if uni2_sampling_mode == "both":
        for stage in ("uni2", "kodama", "clustering", "cluster_mask", "medsam_refine", "cluster_geojson"):
            if stage in factors:
                publish, work = factors[stage]
                factors[stage] = (publish * 2.0, work * 2.0)
    if uni2_sampling_mode in {"grid", "both"}:
        factors.pop("grow_tissue", None)
    if uni2_save_tiles and "uni2" in factors:
        publish, work = factors["uni2"]
        factors["uni2"] = (publish + 3.0, work + 3.5)
    return factors


def directory_size(path: Path, max_files: int = 100_000) -> tuple[int, bool]:
    if not path.exists():
        return 0, True
    if path.is_file():
        return path.stat().st_size, True
    total = 0
    seen = 0
    for root, _, names in os.walk(path, followlinks=False):
        for name in names:
            seen += 1
            if seen > max_files:
                return total, False
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total, True


def nearest_existing(path: Path) -> Path:
    candidate = path.absolute()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def filesystem_info(path: Path) -> dict:
    anchor = nearest_existing(path)
    stat = os.stat(anchor)
    vfs = os.statvfs(anchor)
    return {
        "device": str(stat.st_dev),
        "anchor": str(anchor),
        "available_bytes": int(vfs.f_bavail * vfs.f_frsize),
        "capacity_bytes": int(vfs.f_blocks * vfs.f_frsize),
    }


def assess_capacity(available: int, expected: int, worst: int, reserve: int) -> str:
    if available < expected + reserve:
        return "fail"
    if available < worst + reserve:
        return "warning"
    return "pass"


def aggregate_filesystems(demands: list[Demand], reserve_bytes: int) -> list[dict]:
    grouped: dict[str, dict] = {}
    for demand in demands:
        info = filesystem_info(Path(demand.path))
        row = grouped.setdefault(
            info["device"],
            {
                **info,
                "paths": [],
                "expected_increment_bytes": 0,
                "worst_case_increment_bytes": 0,
            },
        )
        row["paths"].append(
            {
                "kind": demand.kind,
                "label": demand.label,
                "path": demand.path,
                "expected_bytes": demand.expected_bytes,
                "worst_case_bytes": demand.worst_case_bytes,
                "note": demand.note,
            }
        )
        row["expected_increment_bytes"] += demand.expected_bytes
        row["worst_case_increment_bytes"] += demand.worst_case_bytes
    for row in grouped.values():
        row["reserve_bytes"] = reserve_bytes
        row["status"] = assess_capacity(
            row["available_bytes"],
            row["expected_increment_bytes"],
            row["worst_case_increment_bytes"],
            reserve_bytes,
        )
        row["expected_headroom_bytes"] = (
            row["available_bytes"] - row["expected_increment_bytes"] - reserve_bytes
        )
        row["worst_case_headroom_bytes"] = (
            row["available_bytes"] - row["worst_case_increment_bytes"] - reserve_bytes
        )
    return sorted(grouped.values(), key=lambda row: str(row["device"]))


def collapse_cache_specs(specs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for label, path in sorted(specs, key=lambda item: len(item[1].absolute().parts)):
        absolute = path.absolute()
        if any(absolute == kept or kept in absolute.parents for _, kept in selected):
            continue
        selected.append((label, absolute))
    return selected


def build_report(args: argparse.Namespace) -> dict:
    inputs = [Path(value) for value in args.input]
    if not inputs:
        raise ValueError("At least one --input is required")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError(f"Input files do not exist: {', '.join(missing)}")
    roi = Path(args.roi_geojson) if args.roi_geojson else None
    estimates = [
        estimate_input(
            path,
            explicit_roi=roi,
            input_count=len(inputs),
            fallback_expansion=args.source_expansion_factor,
        )
        for path in inputs
    ]
    active_rgb_bytes = sum(item.active_rgb_bytes for item in estimates)
    stages = active_stages(args.start_point, args.end_point)
    factors = adjusted_stage_factors(
        stages,
        cell_detection_mode=args.cell_detection_mode,
        uni2_sampling_mode=args.uni2_sampling_mode,
        gigatime_enabled=parse_bool(args.gigatime_enabled),
        marker_quantification_enabled=parse_bool(args.marker_quantification_enabled),
        gigatime_channel_count=args.gigatime_channel_count,
        uni2_save_tiles=parse_bool(args.uni2_save_tiles),
    )
    safety = float(args.safety_factor)
    publish_raw = sum(active_rgb_bytes * value[0] for value in factors.values())
    work_raw = sum(active_rgb_bytes * value[1] for value in factors.values())
    largest_work_stage = max((active_rgb_bytes * value[1] for value in factors.values()), default=0)
    fixed_reports = int(0.25 * GIB)
    if args.publish_dir_mode == "copy":
        output_expected = int(math.ceil(publish_raw * safety)) + fixed_reports
        output_worst = output_expected + int(math.ceil(publish_raw))
        output_note = "durable copies plus bounded execution reports"
    else:
        output_expected = int(math.ceil(active_rgb_bytes * 0.03 * safety)) + fixed_reports
        output_worst = output_expected + int(math.ceil(active_rgb_bytes * 0.03))
        output_note = "relative links; bulk data remain dependent on the work cache"
    work_expected = int(math.ceil(work_raw * safety))
    work_worst = work_expected + int(
        math.ceil(max(work_raw, largest_work_stage) * args.restart_duplication_factor)
    )
    demands = [
        Demand("output", "published_results", str(Path(args.outdir).absolute()), output_expected, output_worst, output_note),
        Demand("work", "nextflow_work", str(Path(args.workdir).absolute()), work_expected, work_worst, "retained task outputs, transient accumulators, and retry/restart duplication"),
    ]

    cache_specs = []
    for value in args.cache:
        if "=" not in value:
            raise ValueError(f"Invalid --cache {value!r}; expected label=/path")
        label, raw_path = value.split("=", 1)
        if label not in CACHE_TARGET_GIB:
            raise ValueError(f"Unknown cache label {label!r}")
        if label == "stardist" and args.cell_detection_mode == "cellvitpp":
            continue
        if CACHE_STAGES[label].isdisjoint(factors):
            continue
        cache_specs.append((label, Path(raw_path)))
    cache_inventory = []
    for label, path in collapse_cache_specs(cache_specs):
        used, complete = directory_size(path)
        target = int(CACHE_TARGET_GIB[label] * GIB)
        increment = max(0, target - used)
        cache_inventory.append(
            {
                "label": label,
                "path": str(path),
                "current_bytes": used,
                "scan_complete": complete,
                "conservative_target_bytes": target,
                "expected_increment_bytes": increment,
            }
        )
        demands.append(
            Demand("cache", label, str(path), increment, increment, "conservative model/runtime cache allowance"),
        )

    reserve = int(args.min_free_gib * GIB)
    filesystems = aggregate_filesystems(demands, reserve)
    states = {row["status"] for row in filesystems}
    status = "fail" if "fail" in states else "warning" if "warning" in states else "pass"
    limitations = [
        "This is a capacity estimate, not a quota reservation; concurrent jobs can consume the reported headroom.",
        "ROI bounding boxes reduce estimates only when a matching GeoJSON is discoverable before execution.",
        "Source-size fallback is less reliable than TIFF dimension metadata, especially for highly compressed CZI inputs.",
        "Docker engine storage is not portable to inspect from Nextflow and is excluded unless supplied as a cache path.",
        "Worst case assumes retained prior work plus one restart-sized duplication allowance.",
    ]
    if parse_bool(args.uni2_save_tiles):
        limitations.append("UNI-2 tile saving is enabled and can dominate inode count as well as byte demand.")
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "policy_mode": args.mode,
        "stage_window": {"start": args.start_point, "end": args.end_point, "active": list(factors)},
        "configuration": {
            "publish_dir_mode": args.publish_dir_mode,
            "cell_detection_mode": args.cell_detection_mode,
            "uni2_sampling_mode": args.uni2_sampling_mode,
            "gigatime_enabled": parse_bool(args.gigatime_enabled),
            "marker_quantification_enabled": parse_bool(args.marker_quantification_enabled),
            "gigatime_channel_count": args.gigatime_channel_count,
            "uni2_save_tiles": parse_bool(args.uni2_save_tiles),
            "safety_factor": args.safety_factor,
            "restart_duplication_factor": args.restart_duplication_factor,
            "minimum_post_run_free_gib": args.min_free_gib,
            "source_expansion_factor": args.source_expansion_factor,
        },
        "inputs": [asdict(item) for item in estimates],
        "total_active_rgb_bytes": active_rgb_bytes,
        "stage_factors": {
            stage: {"published_per_rgb_byte": values[0], "work_per_rgb_byte": values[1]}
            for stage, values in factors.items()
        },
        "demands": [asdict(item) for item in demands],
        "cache_inventory": cache_inventory,
        "filesystems": filesystems,
        "limitations": limitations,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--roi-geojson", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--start-point", choices=STAGE_ORDER, required=True)
    ap.add_argument("--end-point", choices=STAGE_ORDER, required=True)
    ap.add_argument("--publish-dir-mode", choices=("copy", "rellink"), default="copy")
    ap.add_argument("--cell-detection-mode", choices=("stardist", "cellvitpp"), default="cellvitpp")
    ap.add_argument("--uni2-sampling-mode", choices=("cells", "grid", "both"), default="cells")
    ap.add_argument("--gigatime-enabled", default="true")
    ap.add_argument("--marker-quantification-enabled", default="true")
    ap.add_argument("--gigatime-channel-count", type=int, default=5)
    ap.add_argument("--uni2-save-tiles", default="false")
    ap.add_argument("--cache", action="append", default=[])
    ap.add_argument("--mode", choices=("off", "warn", "fail"), default="fail")
    ap.add_argument("--min-free-gib", type=float, default=20.0)
    ap.add_argument("--safety-factor", type=float, default=1.25)
    ap.add_argument("--restart-duplication-factor", type=float, default=1.0)
    ap.add_argument("--source-expansion-factor", type=float, default=8.0)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "off":
        return 0
    for label, value in (
        ("min-free-gib", args.min_free_gib),
        ("safety-factor", args.safety_factor),
        ("restart-duplication-factor", args.restart_duplication_factor),
        ("source-expansion-factor", args.source_expansion_factor),
    ):
        if value < 0:
            raise SystemExit(f"[ERROR] --{label} must be non-negative")
    try:
        report = build_report(args)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] Storage preflight could not be completed: {exc}")
        return 2
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    expected = sum(row["expected_increment_bytes"] for row in report["filesystems"])
    worst = sum(row["worst_case_increment_bytes"] for row in report["filesystems"])
    print(
        f"Storage preflight: status={report['status']} expected_increment={expected / GIB:.1f} GiB "
        f"restart_worst_case={worst / GIB:.1f} GiB report={output}"
    )
    for row in report["filesystems"]:
        print(
            f"  device={row['device']} status={row['status']} available={row['available_bytes'] / GIB:.1f} GiB "
            f"expected={row['expected_increment_bytes'] / GIB:.1f} GiB "
            f"worst={row['worst_case_increment_bytes'] / GIB:.1f} GiB reserve={row['reserve_bytes'] / GIB:.1f} GiB"
        )
    return 2 if report["status"] == "fail" and args.mode == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
