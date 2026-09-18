#!/usr/bin/env python3
"""Estimate CellPhenotyper disk demand before scheduling WSI stages."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


GIB = 1024**3
MODEL_CHANNEL_COUNT = 23
DTYPE_BYTES = {"uint8": 1, "uint16": 2, "float32": 4}
PYRAMID_FACTOR = 4.0 / 3.0  # uncompressed geometric-series allowance
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
    source_mpp: float | None = None
    mpp_source: str = "unknown"


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


def probe_source_mpp(path: Path) -> tuple[float | None, str]:
    """Read metadata only; it is a planning hint, not a passed calibration check."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            if tif.ome_metadata:
                units = {"µm": 1., "um": 1., "nm": .001, "mm": 1000.}
                for element in ET.fromstring(tif.ome_metadata).iter():
                    if element.tag.rsplit("}", 1)[-1] == "Pixels":
                        values = [float(element.attrib[f"PhysicalSize{axis}"]) *
                                  units[element.attrib.get(f"PhysicalSize{axis}Unit", "µm")]
                                  for axis in ("X", "Y")]
                        if all(.01 <= value <= 10. for value in values):
                            return max(values), "unvalidated_ome_metadata_max_axis"
                        return None, "implausible_ome_mpp"
            page = tif.pages[0]
            match = re.search(r"\bMPP\s*=\s*([0-9.eE+-]+)", page.description or "")
            if match and .01 <= float(match.group(1)) <= 10.:
                return float(match.group(1)), "unvalidated_aperio_metadata"
            unit = int(page.tags["ResolutionUnit"].value)
            micrometres = {2: 25400., 3: 10000.}[unit]
            values = []
            for axis in ("X", "Y"):
                value = page.tags[f"{axis}Resolution"].value
                resolution = value[0] / value[1] if isinstance(value, tuple) else float(value)
                values.append(micrometres / resolution)
            if all(.01 <= value <= 10. for value in values):
                return max(values), "unvalidated_tiff_resolution_max_axis"
            return None, "implausible_tiff_mpp"
    except Exception:
        return None, "unknown"


def load_input_metadata(path: Path | None) -> dict[str, dict]:
    """Load selected-series metadata produced by a trusted format reader."""
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("inputs"), list):
        raise ValueError("Storage preflight input metadata must use schema_version 1 and an inputs list")
    result: dict[str, dict] = {}
    for record in payload["inputs"]:
        if not isinstance(record, dict):
            raise ValueError("Storage preflight input metadata records must be objects")
        raw_path = record.get("path")
        width = record.get("width_px")
        height = record.get("height_px")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Storage preflight input metadata requires nonempty paths")
        if type(width) is not int or type(height) is not int or width < 1 or height < 1:
            raise ValueError("Storage preflight input dimensions must be positive integers")
        mpp_values = [record.get("physical_size_x_um"), record.get("physical_size_y_um")]
        if any(value is not None and (not isinstance(value, (int, float)) or not .01 <= float(value) <= 10.)
               for value in mpp_values):
            raise ValueError("Storage preflight physical sizes must be null or within 0.01..10 micrometres/pixel")
        canonical = str(Path(raw_path).resolve())
        if canonical in result:
            raise ValueError(f"Duplicate storage preflight metadata path: {canonical}")
        result[canonical] = record
    return result


def resolve_channel_count(spec: str | None, count: int | None = None) -> int:
    """Empty output selection means the complete model panel, never one channel."""
    if spec is not None:
        value = len({part.strip().lower() for part in spec.split(",") if part.strip()})
        value = value or MODEL_CHANNEL_COUNT
    else:
        value = MODEL_CHANNEL_COUNT if count in (None, 0) else int(count)
    if not 1 <= value <= MODEL_CHANNEL_COUNT:
        raise ValueError(f"GigaTIME channel count must be between 1 and {MODEL_CHANNEL_COUNT}")
    return value


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
    source_mpp: float = 0.,
    selected_series_metadata: dict | None = None,
) -> InputEstimate:
    source_bytes = path.stat().st_size
    if selected_series_metadata is not None:
        width = int(selected_series_metadata["width_px"])
        height = int(selected_series_metadata["height_px"])
        source = str(selected_series_metadata.get("backend") or "selected_series_metadata")
    else:
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
    if source_mpp > 0:
        mpp, mpp_source = source_mpp, "explicit_override"
    elif selected_series_metadata is not None:
        values = [selected_series_metadata.get("physical_size_x_um"), selected_series_metadata.get("physical_size_y_um")]
        valid = [float(value) for value in values if value is not None]
        mpp = max(valid) if valid else None
        mpp_source = "bioformats_selected_series_metadata" if mpp is not None else "selected_series_metadata_without_mpp"
    else:
        mpp, mpp_source = probe_source_mpp(path)
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
        source_mpp=mpp,
        mpp_source=mpp_source,
    )


def inference_pixel_estimate(item: InputEstimate, target_mpp: float,
                             unknown_source_mpp: float = 1.) -> dict:
    """Conservative planning area; do not apply RAM/disk-driven downsampling caps."""
    native = int(math.ceil(item.active_rgb_bytes / 3.))
    mpp = item.source_mpp
    if target_mpp <= 0:
        scale, reason = 1., "native_scale_no_positive_target_mpp"
    elif mpp is None:
        scale = max(1., (unknown_source_mpp / target_mpp) ** 2)
        reason = "unknown_mpp_declared_allowance_not_an_upper_bound"
    elif item.mpp_source == "bioformats_selected_series_metadata":
        # The same selected-series reader drives conversion. Use its pixel
        # dimensions but never claim a downsampling storage discount before
        # the converted-resolution validator has run.
        scale = max(1., (mpp / target_mpp) ** 2)
        reason = "selected_series_metadata_without_downsampling_credit"
    else:
        scale = (mpp / target_mpp) ** 2
        reason = "explicit_override_target_mpp"
        if item.mpp_source != "explicit_override":
            # Never earn a downsampling discount from metadata not yet validated.
            scale = max(1., scale, (unknown_source_mpp / target_mpp) ** 2)
            reason = "unvalidated_metadata_or_unknown_mpp_allowance_whichever_larger"
    # Rounding overhead bounds ceil(H*s)*ceil(W*s) using full-image dimensions.
    overhead = 0
    if item.width_px and item.height_px:
        overhead = math.ceil((item.width_px + item.height_px) * math.sqrt(scale) + 1)
    return {"native_pixels": native, "inference_area_scale": scale,
            "inference_pixels": int(math.ceil(native * scale)) + overhead,
            "mpp_source": item.mpp_source, "source_mpp": mpp,
            "target_mpp": target_mpp, "reason": reason,
            "unknown_source_mpp_allowance": unknown_source_mpp}


def gigatime_storage_model(pixels: int, *, channel_count: int = MODEL_CHANNEL_COUNT,
                           output_dtype: str = "float32", output_format: str = "zarr",
                           pyramid: bool = True, export_ometiff: bool = True,
                           export_channel_count: int | None = None,
                           export_dtype: str = "auto", blockwise: bool = True) -> dict:
    """Uncompressed arrays, no optimistic compression or sparse-tissue credit."""
    channel_count = resolve_channel_count(None, channel_count)
    dtype_bytes = DTYPE_BYTES[output_dtype]
    if output_format == "ome_tiff" and not blockwise:
        # The current legacy dense writer always stages/writes float32, even
        # when a smaller output dtype was requested. Do not underbudget it.
        dtype_bytes = max(dtype_bytes, 4)
    export_count = channel_count if export_channel_count is None else resolve_channel_count(None, export_channel_count)
    export_bytes = dtype_bytes if export_dtype == "auto" else DTYPE_BYTES[export_dtype]
    multiplier = PYRAMID_FACTOR if pyramid else 1.
    level0 = pixels * channel_count * dtype_bytes
    primary = 0 if output_format == "none" else math.ceil(level0 * (multiplier if output_format == "ome_tiff" else 1.))
    exported = math.ceil(pixels * export_count * export_bytes * multiplier) if export_ometiff and output_format != "none" else 0
    # Direct blockwise TIFF stages a full level-0 buffer. Zarr writes chunks.
    staged = level0 if output_format == "ome_tiff" and blockwise else 0
    # Dense inference may spill the full 23-channel float32 accumulator plus
    # one float32 weight plane and a selected-channel float32 final buffer.
    dense = math.ceil(pixels * (MODEL_CHANNEL_COUNT + 1 + channel_count) * 4 * 1.10) if not blockwise else 0
    scratch = staged + dense
    return {"basis": "uncompressed_array_accounting_not_measured_compression",
            "inference_pixels": pixels, "channel_count": channel_count,
            "bytes_per_sample": dtype_bytes, "level0_bytes": level0,
            "primary_store_bytes": primary, "exported_ometiff_bytes": exported,
            "staged_level0_bytes": staged, "dense_accumulator_allowance_bytes": dense,
            "published_bytes": primary + exported,
            "retained_work_bytes": primary + exported,
            "scratch_bytes": scratch, "work_bytes": primary + exported + scratch,
            "pyramid_factor": multiplier}


def compartment_storage_model(pixels: int, physical: bool = True) -> dict:
    count = 3 if physical else 1
    # The portable bundle contains a second path for the whole-cell TIFF.
    # It is hardlinked locally where possible, but publish-copy/cross-device
    # fallback can materialize both copies, so do not assume link preservation.
    files = count + int(physical)
    retained = pixels * files * 4  # canonical pipeline masks are uint32
    scratch = pixels * 4  # tiled expansion's disk-backed whole-cell array
    return {"basis": "uncompressed_uint32_label_arrays_with_portable_whole_bundle_copy_allowance", "compartment_count": count,
            "output_file_count_allowance": files,
            "published_bytes": retained, "retained_work_bytes": retained,
            "scratch_bytes": scratch, "work_bytes": retained + scratch}


def hovernet_storage_model(
    items: list[InputEstimate],
    *,
    execution_mode: str,
    target_mpp: float,
    core_size: int,
    halo: int,
    batch_tiles: int,
    export_contours: bool,
    density_per_mm2: float,
    unknown_source_mpp: float,
) -> dict:
    """Expose HoVer-Net's peak-disk shape without claiming measured compression.

    The streaming route is deliberately modelled as one bounded tile batch plus
    the incrementally accumulated compressed cell records.  The legacy WSI
    route retains the upstream full-slide float32x4 and int32 arrays.
    """
    if execution_mode not in {"streaming_tiles", "wsi"}:
        raise ValueError("HoVer-Net execution mode must be streaming_tiles or wsi")
    if not math.isfinite(target_mpp) or target_mpp <= 0:
        raise ValueError("HoVer-Net target MPP must be finite and positive")
    if core_size < 512 or halo < 92 or core_size + 2 * halo > 5000:
        raise ValueError("HoVer-Net streaming geometry requires core>=512, halo>=92, core+2*halo<=5000")
    if batch_tiles < 1:
        raise ValueError("HoVer-Net streaming batch size must be positive")

    inference = [inference_pixel_estimate(item, target_mpp, unknown_source_mpp) for item in items]
    inference_pixels = sum(row["inference_pixels"] for row in inference)
    cells = estimated_cell_count(items, density_per_mm2, unknown_source_mpp)
    # Planning allowances include JSON keys and variable-length IDs. The raw
    # upstream records contain contours even when the final compact output does
    # not; gzip ratios are intentionally not treated as guarantees.
    final_cell_record_bytes = 1024 if export_contours else 256
    published = cells * final_cell_record_bytes
    runtime_copy = 256 * 1024**2

    if execution_mode == "wsi":
        slide_wide = inference_pixels * (4 * 4 + 4)
        raw_json = cells * 4096
        scratch = slide_wide + raw_json + runtime_copy
        return {
            "basis": "uncompressed_slide_wide_float32x4_int32_arrays_and_raw_cell_JSON_allowance",
            "execution_mode": execution_mode,
            "inference_pixels": inference_pixels,
            "estimated_cells": cells,
            "slide_wide_prediction_bytes": slide_wide,
            "published_bytes": published,
            "retained_work_bytes": published,
            "scratch_bytes": scratch,
            "work_bytes": published + scratch,
            "bounded_tile_batch": False,
        }

    tile_side = core_size + 2 * halo
    tile_pixels = tile_side**2
    tile_count = 0
    for item, row in zip(items, inference):
        if item.width_px and item.height_px:
            linear_scale = math.sqrt(row["inference_area_scale"])
            tile_count += (
                math.ceil(item.width_px * linear_scale / core_size)
                * math.ceil(item.height_px * linear_scale / core_size)
            )
        else:
            tile_count += math.ceil(row["inference_pixels"] / core_size**2)
    tile_count = max(1, tile_count)
    active_batch_tiles = min(batch_tiles, tile_count)
    input_batch = active_batch_tiles * tile_pixels * 3
    cells_per_tile = math.ceil(tile_pixels * target_mpp**2 * density_per_mm2 / 1e6)
    raw_batch_json = active_batch_tiles * cells_per_tile * 4096
    scratch = input_batch + raw_batch_json + runtime_copy
    return {
        "basis": "bounded_uncompressed_tile_batch_plus_incremental_final_output_allowance",
        "execution_mode": execution_mode,
        "inference_pixels": inference_pixels,
        "estimated_cells": cells,
        "core_size_px": core_size,
        "halo_px": halo,
        "tile_side_px": tile_side,
        "tile_count_without_tissue_sparsity_credit": tile_count,
        "batch_tiles": active_batch_tiles,
        "batch_input_upper_bound_bytes": input_batch,
        "batch_raw_json_upper_bound_bytes": raw_batch_json,
        "accumulated_raw_cell_records_bytes": 0,
        "slide_wide_prediction_bytes": 0,
        "published_bytes": published,
        "retained_work_bytes": published,
        "scratch_bytes": scratch,
        "work_bytes": published + scratch,
        "bounded_tile_batch": True,
        "raw_tile_records_deleted_after_each_batch": True,
        "contours_exported": export_contours,
    }


def cell_profile_storage_model(items: list[InputEstimate], *, uni2: bool, markers: bool,
                               cellvit: bool, radii_um: list[float], density_per_mm2: float,
                               unknown_source_mpp: float, feature_storage: str = "table") -> dict:
    """Cell density, scalar tables and graph degree are explicit planning priors."""
    cells = estimated_cell_count(items, density_per_mm2, unknown_source_mpp)
    if feature_storage not in ("table", "arrays"):
        raise ValueError("Cell neighbourhood feature storage must be table or arrays")
    dimensions = 128 + (2 * 1536 if uni2 else 0) + (3 * MODEL_CHANNEL_COUNT * 4 if markers else 0) + (1024 if cellvit else 0)
    arrays = cells * dimensions * 4
    # CSV + Parquet, morphology, row-index and per-cell provenance allowances.
    tables = cells * (32768 + len(radii_um) * 4096)
    # Representation v2 can retain all own-cell and per-radius neighbour means
    # as wide numeric tables. Budget this even for a restricted selection: do
    # not understate the available-block default or assume CSV compression.
    neighborhood_dimensions = dimensions * (1 + len(radii_um))
    neighborhood_tables = cells * neighborhood_dimensions * 40 if feature_storage == "table" else 0
    # Own-cell axes reference existing arrays. Only per-radius finite-neighbour
    # means add float64 payloads; all numeric + missingness fitting axes may
    # additionally occupy a temporary float64 scaled matrix.
    neighborhood_arrays = cells * dimensions * len(radii_um) * 8 if feature_storage == "arrays" else 0
    fitting_scratch = cells * neighborhood_dimensions * 2 * 8 if feature_storage == "arrays" else 0
    tables += neighborhood_tables
    edges = sum(cells * min(max(0, cells - 1), math.ceil(math.pi * radius ** 2 * density_per_mm2 / 1e6 * 2.)) for radius in radii_um)
    graphs = edges * 16 + len(radii_um) * (cells + 1) * 8
    retained = arrays + tables + graphs + neighborhood_arrays
    return {"basis": "uncalibrated_cell_density_and_graph_degree_allowance",
            "estimated_cells": cells, "cell_density_per_mm2": density_per_mm2,
            "feature_dimensions_allowance": dimensions, "feature_arrays_bytes": arrays,
            "scalar_tables_bytes": tables, "graph_bytes": graphs,
            "neighborhood_dimensions_allowance": neighborhood_dimensions,
            "neighborhood_feature_tables_bytes": neighborhood_tables,
            "neighborhood_feature_storage": feature_storage,
            "neighborhood_feature_arrays_bytes": neighborhood_arrays,
            "niche_scaled_scratch_bytes": fitting_scratch,
            "graph_directed_edges_allowance": edges, "radii_um": radii_um,
            "published_bytes": retained, "retained_work_bytes": retained * 2,
            "scratch_bytes": fitting_scratch, "work_bytes": retained * 2 + fitting_scratch}


def reference_mapping_storage_model(*, observations: int, source_table_bytes: int, atlas: str) -> dict:
    """Budget copied query scalars plus additive interpretations, not atlas arrays."""
    manifest = Path(atlas) / 'atlas_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Reference atlas manifest is missing: {manifest}')
    atlas_bytes = manifest.stat().st_size
    # IDs, 7 interpretation columns and JSON bindings are variable-length.
    # These are explicit planning allowances, not mathematical maxima.
    interpretation = observations * 1024
    receipt = 4096 + observations * 256
    bundle = source_table_bytes + interpretation + receipt + atlas_bytes
    attachment = interpretation + 3 * (receipt + atlas_bytes)
    return {'basis': 'full_query_scalar_table_copy_and_additive_reference_interpretation',
            'observation_count_allowance': observations, 'source_table_copy_bytes': source_table_bytes,
            'interpretation_bytes': interpretation, 'receipt_allowance_bytes': receipt,
            'atlas_manifest_bytes': atlas_bytes, 'published_bytes': bundle,
            'retained_work_bytes': bundle, 'scratch_bytes': 0, 'work_bytes': bundle,
            'spatialdata_attachment_bytes': attachment, 'additional_model_inference': False,
            'note': 'Variable-length IDs/labels and JSON are uncalibrated allowances; reference arrays are not copied into the mapping/export bundle.'}


def estimated_cell_count(items: list[InputEstimate], density_per_mm2: float,
                         unknown_source_mpp: float) -> int:
    cells = 0
    for item in items:
        mpp = item.source_mpp or unknown_source_mpp
        if item.mpp_source not in {"explicit_override", "bioformats_selected_series_metadata"}:
            mpp = max(mpp, unknown_source_mpp)
        cells += math.ceil(item.active_rgb_bytes / 3. * mpp ** 2 * density_per_mm2 / 1e6)
    return cells


def cell_hierarchy_link_storage_model(profile, hierarchy, *, native_pixels, physical_compartments):
    """Retain the source profile and budget additive tables; no new encoder work."""
    cells = profile['estimated_cells']
    compartments = 2 if physical_compartments else 1
    # Four categorical overlaps per compartment is a planning prior, not a
    # mathematical bound. Disjoint nucleus/ring pixels bound the worst case.
    rows = min(native_pixels, cells * compartments * 4)
    relation_bytes = rows * 1024  # CSV/Parquet plus UID/status fields
    source_registry = profile['scalar_tables_bytes']
    summaries = cells * compartments * 2048
    ring_source = native_pixels * 4 if physical_compartments else 0
    retained = profile['published_bytes'] + source_registry + ring_source + relation_bytes + summaries
    fragmentation_extra = max(0, native_pixels - rows) * 1024
    return {'basis': 'immutable_profile_copy_source_registry_and_fractional_membership_tables',
            'estimated_cells': cells, 'compartment_count': compartments,
            'overlap_rows_allowance': rows, 'overlap_rows_pixel_bound': native_pixels,
            'relation_table_bytes': relation_bytes, 'source_registry_bytes': source_registry,
            'retained_ring_raster_bytes': ring_source,
            'summary_table_bytes': summaries, 'profile_copy_bytes': profile['published_bytes'],
            'published_bytes': retained, 'retained_work_bytes': retained,
            'scratch_bytes': relation_bytes, 'work_bytes': retained + relation_bytes,
            'worst_case_output_extra_bytes': fragmentation_extra,
            'worst_case_work_extra_bytes': fragmentation_extra * 2,
            'additional_model_inference': False,
            'note': 'Overlap row density is uncalibrated; worst case uses at most one relation per disjoint compartment pixel.'}


def hierarchy_storage_model(items, *, model_tile_size, inner_size, target_mpp,
                            unknown_source_mpp, max_components, regions_per_grid,
                            model_snapshot, weights_filename, device, feature_batch,
                            max_window_pixels, fit_limit, components_per_block):
    """Array accounting plus explicitly uncalibrated fragmentation/RAM priors."""
    if (min(model_tile_size, inner_size, max_components, feature_batch, max_window_pixels, fit_limit, components_per_block) < 1
            or inner_size > model_tile_size or not math.isfinite(target_mpp) or target_mpp <= 0
            or not math.isfinite(regions_per_grid) or regions_per_grid < 1
            or device not in {"cpu", "mps", "cuda"}):
        raise ValueError("Invalid hierarchy grid/resource bounds")
    if not model_snapshot:
        raise ValueError("Hierarchy preflight requires the explicit local model snapshot; no download is assumed")
    snapshot = Path(model_snapshot)
    filename = weights_filename or next((name for name in ("model.safetensors", "pytorch_model.bin") if (snapshot / name).is_file()), "")
    if not filename or Path(filename).name != filename:
        raise ValueError("Hierarchy checkpoint must be a local filename in the configured snapshot")
    model_files = [snapshot / "config.json", snapshot / filename]
    if any(not path.is_file() for path in model_files):
        raise ValueError("Hierarchy snapshot config/checkpoint is missing; preflight will not download it")
    model_bytes = sum(path.stat().st_size for path in model_files)
    grid_rows = region_count = region_cap = native_pixels = 0
    grids = []
    for item in items:
        mpp = item.source_mpp or unknown_source_mpp
        if item.mpp_source not in {"explicit_override", "bioformats_selected_series_metadata"}:
            mpp = max(mpp, unknown_source_mpp)
        field_pixels = max(1, round(model_tile_size * target_mpp / mpp))
        stride = min(field_pixels, max(1, round(inner_size * field_pixels / model_tile_size)))
        pixels = math.ceil(item.active_rgb_bytes / 3.)
        # Area + perimeter rounding allowance for a bounding-box grid. No
        # tissue/field-coverage credit: all rows remain in both NPY blocks.
        rows = math.ceil(pixels / stride ** 2)
        rows += math.ceil(((item.width_px or math.sqrt(pixels)) + (item.height_px or math.sqrt(pixels))) / stride) + 1
        cap = min(max_components, pixels)
        regions = min(cap, math.ceil(rows * regions_per_grid))
        native_pixels += pixels
        grid_rows += rows
        region_count += regions
        region_cap += cap
        grids.append({"path": item.path, "estimated_grid_rows": rows, "stride_source_pixels": stride,
                      "planning_source_mpp": mpp, "region_count_allowance": regions,
                      "algorithm_component_limit": cap})
    grid_features = grid_rows * 2 * 1536 * 4
    region_features = region_count * 2 * 1536 * 4
    # subdomain u32, status u8, region u32; original parent u32 and uncertainty u8 copies.
    rasters = native_pixels * (4 + 1 + 4 + 4 + 1)
    tables = (grid_rows + region_count) * 4096
    region_sums = region_count * 2 * 1536 * 8
    alignment_scratch = grid_features
    model_staging = model_bytes * len(items)  # fallback when executor staging copies rather than links
    retained = grid_features + region_features + rasters + tables
    scratch = region_sums + alignment_scratch
    additional_regions = max(0, region_cap - region_count)
    output_fragmentation_extra = additional_regions * (2 * 1536 * 4 + 4096)
    work_fragmentation_extra = output_fragmentation_extra + additional_regions * 2 * 1536 * 8
    return {"basis": "two_full_float32_feature_blocks_native_rasters_and_explicit_region_fragmentation_prior",
        "grid_estimates": grids, "estimated_grid_rows": grid_rows,
        "region_count_allowance": region_count, "region_count_algorithm_limit": region_cap,
        "regions_per_grid_allowance": regions_per_grid, "feature_blocks": 2, "features_per_block": 1536,
        "grid_feature_bytes": grid_features, "region_feature_bytes": region_features,
        "native_raster_bytes": rasters, "scalar_table_allowance_bytes": tables,
        "region_float64_sums_bytes": region_sums, "alignment_scratch_bytes": alignment_scratch,
        "published_bytes": retained, "retained_work_bytes": retained, "scratch_bytes": scratch,
        "staged_model_copy_allowance_bytes": model_staging,
        "work_bytes": retained + scratch + model_staging, "worst_case_output_extra_bytes": output_fragmentation_extra,
        "worst_case_work_extra_bytes": work_fragmentation_extra,
        "local_model_snapshot": {"path": str(snapshot.resolve()), "selected_files": [str(path.resolve()) for path in model_files],
            "existing_selected_bytes": model_bytes, "additional_download_bytes": 0},
        "resource_planning": {"device": device, "feature_batch": feature_batch,
            "source_window_and_copy_bytes": max_window_pixels * 3 * 2,
            "resized_uint8_and_float32_batch_bytes": feature_batch * 224 * 224 * 3 * 5,
            "model_ram_allowance_bytes": model_bytes * 2,
            "discovery_fit_and_projection_ram_allowance_bytes": min(grid_rows, fit_limit) * 1536 * 8 * 2 + grid_rows * components_per_block * 4 * 4,
            "note": "Uncalibrated RAM allowances, not GPU activation estimates or reservations; CUDA admission uses configured UNI2 memory, CPU/MPS do not lease CUDA."}}


def measured_export_storage_model(config_path):
    """Read only the user's explicit package/link declarations; do not scan home."""
    config_path = Path(config_path).resolve()
    rows = json.loads(config_path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("cell_measured_assays must be a nonempty JSON list")
    samples, packages, shape_records, published, staged = set(), [], [], 0, 0

    def directory(raw):
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Measured declarations require nonempty directory paths")
        path = Path(raw)
        path = path if path.is_absolute() else config_path.parent / path
        if not path.is_dir():
            raise ValueError(f"Declared measured directory is missing: {path}")
        return path.resolve()

    def package_file(root, relative):
        if not isinstance(relative, str):
            raise ValueError("Invalid measured package file declaration")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Measured package file is missing or escapes the declared package")
        return path

    for row in rows:
        sample = row.get("sample_id") if isinstance(row, dict) else None
        if (not isinstance(sample, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", sample)
                or sample in {".", ".."} or sample in samples):
            raise ValueError("Measured sample IDs must be unique safe path segments")
        samples.add(sample)
        raw_packages, raw_shapes = row.get("measured_assays", []), row.get("measured_region_shapes", [])
        if not isinstance(raw_packages, list) or not raw_packages or not isinstance(raw_shapes, list):
            raise ValueError("Measured sample requires measured_assays list and optional measured_region_shapes list")
        roots, links, assay_ids, region_assays = set(), set(), set(), set()
        for raw in raw_packages:
            root = directory(raw)
            if root in roots:
                raise ValueError("Duplicate measured package directory")
            roots.add(root)
            manifest_file = root / "measured_assay_manifest.json"
            manifest = json.loads(manifest_file.read_text())
            if not isinstance(manifest, dict):
                raise ValueError("Measured package manifest must be an object")
            assay_id = manifest.get("assay_id")
            if (manifest.get("schema_version") != "cellphenotyper.measured_assay.v1" or manifest.get("sample_id") != sample
                    or not isinstance(assay_id, str) or not assay_id or assay_id in assay_ids):
                raise ValueError("Measured package schema/sample/assay identity mismatch")
            assay_ids.add(assay_id)
            if manifest.get("observation_unit") == "spatial_bin":
                region_assays.add(assay_id)
            elif manifest.get("observation_unit") != "cell":
                raise ValueError("Unsupported measured observation unit")
            files = manifest.get("files", {})
            required = {"measured_values.npy", "measured_observations.parquet", "measured_observations.csv", "measured_rows.csv"}
            if not isinstance(files, dict) or not required <= set(files):
                raise ValueError("Measured package lacks expected matrix/table files")
            sizes = {name: package_file(root, name).stat().st_size for name in files}
            matrix_record = manifest.get("matrix", {})
            if not isinstance(matrix_record, dict):
                raise ValueError("Measured package matrix declaration must be an object")
            shape = matrix_record.get("shape", [])
            if (not isinstance(shape, list) or len(shape) != 2 or any(type(value) is not int or value < 0 for value in shape)
                    or shape[0] != manifest.get("observation_count") or shape[1] < 1
                    or matrix_record.get("dtype") != "float64"):
                raise ValueError("Measured package matrix dimensions/precision are invalid")
            matrix = max(sizes["measured_values.npy"], shape[0] * shape[1] * 8)
            # AnnData obs/string metadata may expand relative to compressed
            # Parquet. Use raw row priors plus actual declared table sizes.
            table = max(shape[0] * 4096, sizes["measured_observations.parquet"] + sizes["measured_observations.csv"])
            output = matrix + table + manifest_file.stat().st_size * 2
            existing = sum(sizes.values()) + manifest_file.stat().st_size
            published += output
            staged += existing
            packages.append({"sample_id": sample, "assay_id": assay_id, "path": str(root),
                "observation_unit": manifest["observation_unit"], "shape": shape,
                "uncompressed_matrix_bytes": matrix, "table_allowance_bytes": table,
                "published_copy_allowance_bytes": output, "existing_declared_bytes": existing})
        linked_assays = set()
        for raw in raw_shapes:
            root = directory(raw)
            if root in links:
                raise ValueError("Duplicate measured shape bundle")
            links.add(root)
            link_file = root / "link.json"
            link = json.loads(link_file.read_text())
            if not isinstance(link, dict):
                raise ValueError("Measured shape link must be an object")
            assay_id = link.get("assay_id")
            if (link.get("schema_version") != "cellphenotyper.measured_region_shapes.v1"
                    or link.get("sample_id") != sample or assay_id not in region_assays or assay_id in linked_assays):
                raise ValueError("Measured region shape link is duplicate, unused, or mismatched")
            linked_assays.add(assay_id)
            artifact_sizes = {}
            for name in ("shapes", "registry_table", "registry_manifest"):
                record = link.get(name, {})
                if not isinstance(record, dict):
                    raise ValueError("Shape link artifact declarations must be objects")
                raw_path = record.get("path")
                if not isinstance(raw_path, str) or not raw_path:
                    raise ValueError("Shape link lacks explicit artifact paths")
                path = Path(raw_path)
                path = path if path.is_absolute() else root / path
                if not path.is_file():
                    raise ValueError(f"Declared shape-link artifact is missing: {path}")
                artifact_sizes[name] = path.stat().st_size
            shape_bytes = artifact_sizes["shapes"] * 4
            published += shape_bytes
            staged += sum(artifact_sizes.values()) + link_file.stat().st_size
            shape_records.append({"sample_id": sample, "assay_id": assay_id, "bundle": str(root),
                "published_geometry_allowance_bytes": shape_bytes, "declared_artifact_bytes": artifact_sizes})
        if linked_assays != region_assays:
            raise ValueError("Every measured spatial-bin package requires one explicit shape-link bundle")
    return {"basis": "explicit_measured_package_dimensions_and_file_sizes_with_uncompressed_table_geometry_allowances",
        "configuration_path": str(config_path), "packages": packages, "shape_bundles": shape_records,
        "published_bytes": published, "retained_work_bytes": published,
        "scratch_bytes": 0, "staged_input_copy_allowance_bytes": staged,
        "work_bytes": published + staged,
        "limitations": "Preflight reads headers/declared file sizes, not assay biological validity; export separately verifies hashes. Compression, metadata and geometry factors are planning priors. Unrelated files inside declared directories are not inventoried."}


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
        model = gigatime_storage_model(3, channel_count=gigatime_channel_count)
        factors["gigatime"] = (model["published_bytes"] / 9., model["work_bytes"] / 9.)
    if "cytoplasm" in factors:
        model = compartment_storage_model(3)
        factors["cytoplasm"] = (model["published_bytes"] / 9., model["work_bytes"] / 9.)
    if cell_detection_mode != "consensus" and "cell_consensus" in factors:
        factors["cell_consensus"] = (0.05, 0.10)
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
    for name in ("min_free_gib", "restart_duplication_factor", "source_mpp", "gigatime_target_mpp"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in ("safety_factor", "source_expansion_factor", "unknown_source_mpp", "cell_density_per_mm2"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.safety_factor < 1.:
        raise ValueError("--safety-factor must be at least 1; estimates must not be discounted")
    if args.source_mpp and not .01 <= args.source_mpp <= 10.:
        raise ValueError("--source-mpp must be within 0.01..10 micrometres/pixel; verify calibration")
    channels = resolve_channel_count(args.gigatime_output_channels, args.gigatime_channel_count)
    export_channels = (resolve_channel_count(args.gigatime_export_channels)
                       if args.gigatime_export_channels.strip() else channels)
    if export_channels > channels:
        raise ValueError("Export channel count cannot exceed the stored GigaTIME panel")
    radii = sorted(set(float(value) for value in args.cell_neighborhood_radii_um.split(",") if value.strip()))
    if any(not math.isfinite(value) or value <= 0 for value in radii):
        raise ValueError("Cell-neighborhood radii must be finite and positive")
    inputs = [Path(value) for value in args.input]
    if not inputs:
        raise ValueError("At least one --input is required")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise ValueError(f"Input files do not exist: {', '.join(missing)}")
    roi = Path(args.roi_geojson) if args.roi_geojson else None
    metadata_path = Path(args.input_metadata_json) if args.input_metadata_json else None
    input_metadata = load_input_metadata(metadata_path)
    estimates = [
        estimate_input(
            path,
            explicit_roi=roi,
            input_count=len(inputs),
            fallback_expansion=args.source_expansion_factor,
            source_mpp=args.source_mpp,
            selected_series_metadata=input_metadata.get(str(path.resolve())),
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
        gigatime_channel_count=channels,
        uni2_save_tiles=parse_bool(args.uni2_save_tiles),
    )
    safety = float(args.safety_factor)
    stage_models = {stage: {"basis": "historical_uncalibrated_rgb_factor",
                           "published_bytes": math.ceil(active_rgb_bytes * value[0]),
                           "work_bytes": math.ceil(active_rgb_bytes * value[1])}
                    for stage, value in factors.items()}
    if "stardist" in stage_models:
        # The historical StarDist factor already included one flat RGB crop.
        # Budget the newly retained 2x overview series explicitly rather than
        # assuming that lossless TIFF compression will reduce its size.
        overview_bytes = math.ceil(active_rgb_bytes * (PYRAMID_FACTOR - 1.0))
        stardist_model = stage_models["stardist"]
        stardist_model.update(
            basis="historical_rgb_factor_plus_uncompressed_roi_crop_overviews",
            roi_crop_pyramid_factor=PYRAMID_FACTOR,
            roi_crop_overview_allowance_bytes=overview_bytes,
            published_bytes=stardist_model["published_bytes"] + overview_bytes,
            work_bytes=stardist_model["work_bytes"] + overview_bytes,
        )
    inference = [dict(path=item.path, **inference_pixel_estimate(item, args.gigatime_target_mpp,
                                                               args.unknown_source_mpp))
                 for item in estimates]
    if "gigatime" in factors:
        stage_models["gigatime"] = gigatime_storage_model(
            sum(row["inference_pixels"] for row in inference), channel_count=channels,
            output_dtype=args.gigatime_output_dtype, output_format=args.gigatime_output_format,
            pyramid=parse_bool(args.gigatime_output_pyramid),
            export_ometiff=parse_bool(args.gigatime_export_ometiff),
            export_channel_count=export_channels, export_dtype=args.gigatime_export_output_dtype,
            blockwise=parse_bool(args.gigatime_blockwise))
    if "cytoplasm" in factors:
        stage_models["cytoplasm"] = compartment_storage_model(math.ceil(active_rgb_bytes / 3.),
                                                              parse_bool(args.physical_compartments))
    if "cell_consensus" in stage_models and args.cell_detection_mode == "consensus":
        hovernet = hovernet_storage_model(
            estimates,
            execution_mode=args.hovernet_execution_mode,
            target_mpp=args.hovernet_target_mpp,
            core_size=args.hovernet_stream_core_size,
            halo=args.hovernet_stream_halo,
            batch_tiles=args.hovernet_stream_batch_tiles,
            export_contours=parse_bool(args.hovernet_export_contours),
            density_per_mm2=args.cell_density_per_mm2,
            unknown_source_mpp=args.unknown_source_mpp,
        )
        consensus = stage_models["cell_consensus"]
        consensus["hovernet"] = hovernet
        consensus["basis"] = "historical_consensus_output_prior_with_explicit_hovernet_peak_disk_model"
        consensus["published_bytes"] = max(consensus["published_bytes"], hovernet["published_bytes"])
        consensus["work_bytes"] = max(consensus["work_bytes"], hovernet["work_bytes"])
    if "marker_quantification" in factors:
        cells = estimated_cell_count(estimates, args.cell_density_per_mm2, args.unknown_source_mpp)
        compartments = 3 if parse_bool(args.physical_compartments) else 2
        # Four per-channel reductions plus counts/IDs/summary: full integrated
        # panel even when only a small visualization subset is stored.
        tables = cells * compartments * (MODEL_CHANNEL_COUNT * 4 * 24 + 1024)
        prior = stage_models["marker_quantification"]
        retained = max(tables, prior["published_bytes"])
        stage_models["marker_quantification"] = {
            "basis": "full_panel_scalar_table_allowance_or_rgb_prior_whichever_larger",
            "estimated_cells": cells, "channel_count": MODEL_CHANNEL_COUNT,
            "compartment_count": compartments, "published_bytes": retained,
            "retained_work_bytes": retained, "scratch_bytes": 0,
            "work_bytes": max(retained, prior["work_bytes"])}
    if parse_bool(args.cell_profiles_enabled):
        stage_models["cell_profiles"] = cell_profile_storage_model(estimates,
            uni2=parse_bool(args.cell_profiles_uni2_enabled), markers=parse_bool(args.cell_profiles_markers_enabled),
            cellvit=parse_bool(args.cellvit_embeddings_enabled), radii_um=radii,
            density_per_mm2=args.cell_density_per_mm2, unknown_source_mpp=args.unknown_source_mpp,
            feature_storage=args.cell_neighborhood_feature_storage)
        if args.cell_neighborhood_support_mode == "brightfield_native":
            # Two native uint8 rasters in the published profile, plus the
            # original producer bundle and its profile copy retained in work.
            # No compression discount; image decoding remains bounded in RAM.
            support_bytes = 2 * math.ceil(active_rgb_bytes / 3.)
            profile_model = stage_models["cell_profiles"]
            profile_model["native_support_bundle_bytes"] = support_bytes
            profile_model["published_bytes"] += support_bytes
            profile_model["work_bytes"] += 2 * support_bytes
            profile_model["retained_work_bytes"] = profile_model.get("retained_work_bytes", 0) + 2 * support_bytes
        if parse_bool(args.cell_profiles_spatialdata):
            if args.spatialdata_pyramid_levels < 0:
                raise ValueError("--spatialdata-pyramid-levels must be non-negative")
            pyramid = sum(4. ** -level for level in range(min(args.spatialdata_pyramid_levels, 32) + 1))
            native_pixels = math.ceil(active_rgb_bytes / 3.)
            rasters = math.ceil(native_pixels * (3 + 4) * pyramid)
            profile_copies = stage_models["cell_profiles"]["published_bytes"]
            own_group_export = (stage_models["cell_profiles"]["estimated_cells"] *
                stage_models["cell_profiles"]["feature_dimensions_allowance"] * 8
                if args.cell_neighborhood_feature_storage == "arrays" else 0)
            # Export preserves original arrays and adds separately named own:*
            # float64 groups alongside neighbour matrices. Do not omit this
            # intentional identity projection from disk planning.
            profile_copies += own_group_export
            total = rasters + profile_copies
            stage_models["spatialdata"] = {
                "basis": "uncompressed_native_uint8_RGB_uint32_labels_and_profile_graph_copies",
                "native_raster_bytes": rasters, "profile_copies_bytes": profile_copies,
                "array_own_group_export_bytes": own_group_export,
                "pyramid_levels": args.spatialdata_pyramid_levels, "pyramid_factor": pyramid,
                "published_bytes": total, "retained_work_bytes": total,
                "scratch_bytes": 0, "work_bytes": total}
    if parse_bool(args.cohort_niches_enabled):
        if not parse_bool(args.cell_profiles_enabled):
            raise ValueError('--cohort-niches-enabled requires cell profiles enabled')
        if args.cohort_niches_max_k < 2:
            raise ValueError('--cohort-niches-max-k must be at least 2')
        profile = stage_models['cell_profiles']
        dimensions = profile['neighborhood_dimensions_allowance']
        model_bytes = dimensions * (10 + args.cohort_niches_max_k + len(estimates) * 8) * 64
        assignment_bytes = profile['estimated_cells'] * 2048
        total = assignment_bytes + model_bytes + 1024**2
        fitting_scratch = profile['niche_scaled_scratch_bytes']
        stage_models['cohort_niches'] = {
            'basis': 'canonical_assignment_and_shared_scaling_centroid_JSON_allowance',
            'estimated_cells': profile['estimated_cells'], 'assignment_bytes': assignment_bytes,
            'model_bytes': model_bytes, 'published_bytes': total, 'retained_work_bytes': total,
            'scratch_bytes': fitting_scratch, 'work_bytes': total + fitting_scratch, 'additional_model_inference': False,
            'source_profiles_copied': False,
            'note': 'Variable-length IDs, feature names and JSON are planning allowances, not mathematical maxima.'}
    if parse_bool(args.tissue_hierarchy_enabled):
        stage_models["tissue_hierarchy"] = hierarchy_storage_model(estimates,
            model_tile_size=args.hierarchy_grid_model_tile_size, inner_size=args.hierarchy_grid_inner_size,
            target_mpp=args.hierarchy_grid_target_mpp, unknown_source_mpp=args.unknown_source_mpp,
            max_components=args.hierarchy_max_components, regions_per_grid=args.hierarchy_regions_per_grid,
            model_snapshot=args.hierarchy_model_snapshot, weights_filename=args.hierarchy_weights_filename,
            device=args.hierarchy_device, feature_batch=args.hierarchy_feature_batch,
            max_window_pixels=args.hierarchy_max_window_pixels, fit_limit=args.hierarchy_fit_limit,
            components_per_block=args.hierarchy_components_per_block)
        if parse_bool(args.cell_profiles_enabled):
            native_pixels = math.ceil(active_rgb_bytes / 3.)
            hierarchy = stage_models['tissue_hierarchy']
            links = cell_hierarchy_link_storage_model(stage_models['cell_profiles'], hierarchy,
                native_pixels=native_pixels, physical_compartments=parse_bool(args.physical_compartments))
            stage_models['cell_tissue_links'] = links
            if parse_bool(args.cell_profiles_spatialdata):
                # Five hierarchy rasters, their region features and the new
                # relationship table are additional to the original export.
                spatial = stage_models['spatialdata']
                rasters = math.ceil(hierarchy['native_raster_bytes'] * spatial['pyramid_factor'])
                rasters += math.ceil(links['retained_ring_raster_bytes'] * spatial['pyramid_factor'])
                tables = hierarchy['region_feature_bytes'] + hierarchy['scalar_table_allowance_bytes'] + links['relation_table_bytes'] + links['summary_table_bytes']
                extra = rasters + tables
                spatial.update(hierarchy_raster_bytes=rasters, hierarchy_table_bytes=tables,
                    published_bytes=spatial['published_bytes'] + extra,
                    retained_work_bytes=spatial['retained_work_bytes'] + extra,
                    work_bytes=spatial['work_bytes'] + extra,
                    worst_case_output_extra_bytes=hierarchy['worst_case_output_extra_bytes'] + links['worst_case_output_extra_bytes'],
                    worst_case_work_extra_bytes=hierarchy['worst_case_output_extra_bytes'] + links['worst_case_output_extra_bytes'])
    if args.cell_reference_atlas:
        if not parse_bool(args.cell_profiles_enabled):
            raise ValueError('--cell-reference-atlas requires cell profiles enabled')
        profile = stage_models['cell_profiles']
        tables = profile['scalar_tables_bytes'] + stage_models.get('cell_tissue_links', {}).get('summary_table_bytes', 0)
        stage_models['cell_reference_mapping'] = reference_mapping_storage_model(
            observations=profile['estimated_cells'], source_table_bytes=tables, atlas=args.cell_reference_atlas)
    if args.region_reference_atlas:
        if not parse_bool(args.tissue_hierarchy_enabled):
            raise ValueError('--region-reference-atlas requires tissue hierarchy enabled')
        hierarchy = stage_models['tissue_hierarchy']
        stage_models['region_reference_mapping'] = reference_mapping_storage_model(
            observations=hierarchy['region_count_allowance'],
            source_table_bytes=hierarchy['scalar_table_allowance_bytes'], atlas=args.region_reference_atlas)
    if 'spatialdata' in stage_models:
        attachment = sum(stage_models[name]['spatialdata_attachment_bytes']
                         for name in ('cell_reference_mapping', 'region_reference_mapping') if name in stage_models)
        if attachment:
            spatial = stage_models['spatialdata']
            spatial['reference_mapping_attachment_bytes'] = attachment
            for key in ('published_bytes', 'retained_work_bytes', 'work_bytes'):
                spatial[key] += attachment
    if args.cell_measured_assays:
        if not (parse_bool(args.cell_profiles_enabled) and parse_bool(args.cell_profiles_spatialdata)):
            raise ValueError("--cell-measured-assays requires cell profiles and SpatialData export enabled")
        stage_models["measured_spatialdata"] = measured_export_storage_model(args.cell_measured_assays)
    publish_raw = sum(value["published_bytes"] for value in stage_models.values())
    work_raw = sum(value["work_bytes"] for value in stage_models.values())
    fragmentation_output = sum(value.get("worst_case_output_extra_bytes", 0) for value in stage_models.values())
    fragmentation_work = sum(value.get("worst_case_work_extra_bytes", 0) for value in stage_models.values())
    largest_work_stage = max((value["work_bytes"] for value in stage_models.values()), default=0)
    # Keep old report readers working; these are now derived effective factors.
    factors = {stage: (value["published_bytes"] / max(1, active_rgb_bytes),
                       value["work_bytes"] / max(1, active_rgb_bytes))
               for stage, value in stage_models.items()}
    fixed_reports = int(0.25 * GIB)
    if args.publish_dir_mode == "copy":
        output_expected = int(math.ceil(publish_raw * safety)) + fixed_reports
        output_worst = output_expected + int(math.ceil(publish_raw + fragmentation_output * safety))
        output_note = "durable copies plus bounded execution reports and configured fragmentation-cap allowance"
    else:
        output_expected = int(math.ceil(active_rgb_bytes * 0.03 * safety)) + fixed_reports
        output_worst = output_expected + int(math.ceil(active_rgb_bytes * 0.03))
        output_note = "relative links; bulk data remain dependent on the work cache"
    work_expected = int(math.ceil(work_raw * safety))
    work_worst = work_expected + int(
        math.ceil(max(work_raw, largest_work_stage) * args.restart_duplication_factor + fragmentation_work * safety)
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
        "Worst case assumes retained prior work plus the configured restart-sized duplication allowance, with additional hierarchy fragmentation up to the configured component cap when enabled; these are not absolute resource upper bounds.",
        "GigaTIME arrays use uncompressed byte counts, including any second exported TIFF and its pyramid; real compression and filesystem overhead vary.",
        "Storage limits and automatic image-size caps are not used as a downsampling discount: strict target MPP can override them.",
        "Only explicit source-MPP override earns a downsampling discount. Unvalidated or missing metadata uses the larger of native, metadata-predicted and declared unknown-MPP area allowances; this is not a guaranteed upper bound.",
        "Full tissue bounding boxes are budgeted; tissue sparsity is not credited. Model memory, GPU memory, inode limits and filesystem quotas are separate constraints.",
        "Per-stage work includes retained arrays plus scratch; summing scratch across stages is conservative and does not model scheduling overlap exactly.",
        "Cell-profile counts, feature dimensions, scalar tables and graph density are planning priors, not measured or biologically calibrated estimates; unusually dense tissue or large radii can exceed them.",
        "Optional standalone atlas fitting and inspector exports are excluded. SpatialData native RGB/label and feature/graph copies are included only when cell-profile SpatialData export is enabled.",
    ]
    if parse_bool(args.uni2_save_tiles):
        limitations.append("UNI-2 tile saving is enabled and can dominate inode count as well as byte demand.")
    if "tissue_hierarchy" in stage_models:
        limitations.extend([
            "Hierarchy budgets both complete 1536-dimensional float32 field blocks without tissue/coverage sparsity discounts, native-resolution rasters, alignment scratch and regional float64 sum buffers. Grid and region counts remain planning allowances until the actual grid/topology is known.",
            "Hierarchy expected regions use an explicit regions-per-grid prior; worst demand adds fragmentation through the configured component cap. The existing local model files are inventoried without any download/cache-growth assumption. RAM allowances are not GPU activation benchmarks or scheduler reservations.",
        ])
    if "measured_spatialdata" in stage_models:
        limitations.append(stage_models["measured_spatialdata"]["limitations"])
    return {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "policy_mode": args.mode,
        "stage_window": {"start": args.start_point, "end": args.end_point, "active": list(factors)},
        "configuration": {
            "publish_dir_mode": args.publish_dir_mode,
            "cell_detection_mode": args.cell_detection_mode,
            "hovernet_execution_mode": args.hovernet_execution_mode,
            "hovernet_target_mpp": args.hovernet_target_mpp,
            "hovernet_stream_core_size": args.hovernet_stream_core_size,
            "hovernet_stream_halo": args.hovernet_stream_halo,
            "hovernet_stream_batch_tiles": args.hovernet_stream_batch_tiles,
            "hovernet_export_contours": parse_bool(args.hovernet_export_contours),
            "uni2_sampling_mode": args.uni2_sampling_mode,
            "gigatime_enabled": parse_bool(args.gigatime_enabled),
            "marker_quantification_enabled": parse_bool(args.marker_quantification_enabled),
            "gigatime_channel_count": channels,
            "gigatime_output_dtype": args.gigatime_output_dtype,
            "gigatime_output_format": args.gigatime_output_format,
            "gigatime_output_pyramid": parse_bool(args.gigatime_output_pyramid),
            "gigatime_export_ometiff": parse_bool(args.gigatime_export_ometiff),
            "gigatime_export_channel_count": export_channels,
            "gigatime_export_output_dtype": args.gigatime_export_output_dtype,
            "gigatime_blockwise": parse_bool(args.gigatime_blockwise),
            "gigatime_target_mpp": args.gigatime_target_mpp,
            "source_mpp_override": args.source_mpp,
            "input_metadata_json": str(metadata_path.resolve()) if metadata_path else None,
            "unknown_source_mpp_allowance": args.unknown_source_mpp,
            "physical_compartments": parse_bool(args.physical_compartments),
            "cell_profiles_enabled": parse_bool(args.cell_profiles_enabled),
            "cell_profiles_spatialdata": parse_bool(args.cell_profiles_spatialdata),
            "spatialdata_pyramid_levels": args.spatialdata_pyramid_levels,
            "tissue_hierarchy_enabled": parse_bool(args.tissue_hierarchy_enabled),
            "cell_reference_atlas": args.cell_reference_atlas or None,
            "region_reference_atlas": args.region_reference_atlas or None,
            "cell_measured_assays": str(Path(args.cell_measured_assays).resolve()) if args.cell_measured_assays else None,
            "uni2_save_tiles": parse_bool(args.uni2_save_tiles),
            "safety_factor": args.safety_factor,
            "restart_duplication_factor": args.restart_duplication_factor,
            "minimum_post_run_free_gib": args.min_free_gib,
            "source_expansion_factor": args.source_expansion_factor,
        },
        "inputs": [asdict(item) for item in estimates],
        "total_active_rgb_bytes": active_rgb_bytes,
        "inference_scale_estimates": inference,
        "stage_storage_models": stage_models,
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
    ap.add_argument("--input-metadata-json", default="",
                    help="Selected-series dimensions/physical sizes produced by a trusted format reader.")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--start-point", choices=STAGE_ORDER, required=True)
    ap.add_argument("--end-point", choices=STAGE_ORDER, required=True)
    ap.add_argument("--publish-dir-mode", choices=("copy", "rellink"), default="copy")
    ap.add_argument("--cell-detection-mode", choices=("consensus", "stardist"), default="consensus")
    ap.add_argument("--hovernet-execution-mode", choices=("streaming_tiles", "wsi"), default="streaming_tiles")
    ap.add_argument("--hovernet-target-mpp", type=float, default=.25)
    ap.add_argument("--hovernet-stream-core-size", type=int, default=4096)
    ap.add_argument("--hovernet-stream-halo", type=int, default=256)
    ap.add_argument("--hovernet-stream-batch-tiles", type=int, default=64)
    ap.add_argument("--hovernet-export-contours", default="false")
    ap.add_argument("--uni2-sampling-mode", choices=("cells", "grid", "both"), default="cells")
    ap.add_argument("--gigatime-enabled", default="true")
    ap.add_argument("--marker-quantification-enabled", default="true")
    ap.add_argument("--gigatime-channel-count", type=int, default=MODEL_CHANNEL_COUNT)
    ap.add_argument("--gigatime-output-channels", default=None,
                    help="Comma-separated names; an empty string means all 23 channels. Overrides channel count.")
    ap.add_argument("--gigatime-output-dtype", choices=DTYPE_BYTES, default="float32")
    ap.add_argument("--gigatime-output-format", choices=("zarr", "ome_tiff", "none"), default="none")
    ap.add_argument("--gigatime-output-pyramid", default="true")
    ap.add_argument("--gigatime-export-ometiff", default="false")
    ap.add_argument("--gigatime-export-channels", default="")
    ap.add_argument("--gigatime-export-output-dtype", choices=("auto", *DTYPE_BYTES), default="auto")
    ap.add_argument("--gigatime-blockwise", default="true")
    ap.add_argument("--gigatime-target-mpp", type=float, default=.25)
    ap.add_argument("--source-mpp", type=float, default=0., help="Explicit pipeline input-resolution override, in micrometres/pixel.")
    ap.add_argument("--unknown-source-mpp", type=float, default=1., help="Declared planning allowance when source MPP is unvalidated; not a verified upper bound.")
    ap.add_argument("--physical-compartments", default="true")
    ap.add_argument("--cell-profiles-enabled", default="false")
    ap.add_argument("--cohort-niches-enabled", default="false")
    ap.add_argument("--cohort-niches-max-k", type=int, default=8)
    ap.add_argument("--cell-neighborhood-support-mode", choices=("provided", "brightfield_native"), default="provided")
    ap.add_argument("--cell-neighborhood-feature-storage", choices=("table", "arrays"), default="table")
    ap.add_argument("--cell-profiles-spatialdata", default="false")
    ap.add_argument("--spatialdata-pyramid-levels", type=int, default=0)
    ap.add_argument("--cell-profiles-uni2-enabled", default="true")
    ap.add_argument("--cell-profiles-markers-enabled", default="true")
    ap.add_argument("--cellvit-embeddings-enabled", default="false")
    ap.add_argument("--cell-neighborhood-radii-um", default="25,50,100")
    ap.add_argument("--cell-density-per-mm2", type=float, default=10000., help="Uncalibrated dense-tissue planning prior for optional profiles and graphs.")
    ap.add_argument("--cell-measured-assays", default="", help="Explicit per-sample package/shape-bundle JSON used by the full pipeline.")
    ap.add_argument("--cell-reference-atlas", default="", help="Frozen cell atlas directory; mapping/export scalar-storage allowance only.")
    ap.add_argument("--region-reference-atlas", default="", help="Frozen region atlas directory; mapping/export scalar-storage allowance only.")
    ap.add_argument("--tissue-hierarchy-enabled", default="false")
    ap.add_argument("--hierarchy-grid-model-tile-size", type=int, default=224)
    ap.add_argument("--hierarchy-grid-inner-size", type=int, default=90)
    ap.add_argument("--hierarchy-grid-target-mpp", type=float, default=.25)
    ap.add_argument("--hierarchy-max-components", type=int, default=1_000_000)
    ap.add_argument("--hierarchy-regions-per-grid", type=float, default=4., help="Uncalibrated region fragmentation planning prior; worst case separately accounts for component cap.")
    ap.add_argument("--hierarchy-model-snapshot", default="")
    ap.add_argument("--hierarchy-weights-filename", default="")
    ap.add_argument("--hierarchy-device", choices=("cpu", "mps", "cuda"), default="cuda")
    ap.add_argument("--hierarchy-feature-batch", type=int, default=16)
    ap.add_argument("--hierarchy-max-window-pixels", type=int, default=16_777_216)
    ap.add_argument("--hierarchy-fit-limit", type=int, default=5000)
    ap.add_argument("--hierarchy-components-per-block", type=int, default=64)
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
