#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tifffile

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape as shp_shape
from shapely.geometry import Polygon
from shapely.geometry import mapping
from shapely.ops import polygonize, unary_union

from shared_boundary_smoothing import smooth_shared_boundary_coverage


PROVENANCE_SCHEMA = "cellphenotyper.vectorized_refinement_provenance.v1"
REFINEMENT_CODES = {0: "outside_tissue_support", 1: "original_label_unchanged", 2: "modified_original_label", 3: "inferred_from_originally_uncertain", 4: "unresolved_inside_tissue", 5: "inferred_new_label", 6: "removed_original_label", 7: "protected_original_core", 8: "pre_refinement_grown_assignment_unchanged"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raster_provenance_summary(mask_path, uncertainty_path=None, provenance_path=None,
                              metadata_path=None, *, tile_size=512, max_labels=65536):
    """Summarize exact level-zero pixels, never pixels of a smoothed polygon.

    Every native label is retained, including zero (which can contain unresolved
    tissue omitted by polygonization). Scan buffers are bounded in both axes;
    the existing polygon extraction has its own, separate memory requirements.
    """
    from profile_cell_morphology import WindowReader

    if bool(provenance_path) != bool(metadata_path) or (provenance_path and not uncertainty_path):
        raise ValueError("Refinement provenance requires uncertainty, provenance and metadata together")
    if tile_size < 1 or tile_size > 4096 or max_labels < 1:
        raise ValueError("Provenance tile size must be 1..4096 and max labels positive")
    paths = {"labels": mask_path}
    if uncertainty_path:
        paths["uncertainty"] = uncertainty_path
    if provenance_path:
        paths["provenance"] = provenance_path
        paths["refinement_metadata"] = metadata_path
    if len({Path(path).resolve() for path in paths.values()}) != len(paths):
        raise ValueError("Label, uncertainty, provenance and metadata must be distinct artifacts")
    inputs = {name: {"filename": Path(path).name, "sha256": file_sha256(path)} for name, path in paths.items()}
    metadata = json.loads(Path(metadata_path).read_text()) if metadata_path else None
    producer_binding = "unavailable"
    if metadata is not None:
        if metadata.get("schema_version") != "1.1.0" or metadata.get("coordinate_frame") != "analysis_crop":
            raise ValueError("Unsupported refinement metadata schema or coordinate frame")
        if metadata.get("provenance_codes") != {str(k): v for k, v in REFINEMENT_CODES.items()}:
            raise ValueError("Refinement provenance code legend does not match its schema")
        if metadata.get("original_uncertainty_codes_preserved") is not True:
            raise ValueError("Refinement metadata does not preserve original uncertainty")
        producer_binding = "legacy_unbound_output_files"
        if "output_binding_status" in metadata or "output_artifacts" in metadata:
            if metadata.get("output_binding_status") != "complete":
                raise ValueError("Refinement producer output binding is incomplete; finalize the label export first")
            artifacts = metadata.get("output_artifacts")
            if not isinstance(artifacts, dict) or set(artifacts) != {"labels", "uncertainty", "provenance"}:
                raise ValueError("Refinement producer binding must include all three exact output artifacts")
            for name, record in artifacts.items():
                if not isinstance(record, dict) or record.get("sha256") != inputs[name]["sha256"]:
                    raise ValueError(f"Refinement producer SHA256 mismatch for {name}")
                if record.get("size_bytes") != Path(paths[name]).stat().st_size:
                    raise ValueError(f"Refinement producer size mismatch for {name}")
            producer_binding = "exact_output_sha256_verified"
    counts = {}
    max_window = 0
    with ExitStack() as stack:
        readers = {}
        for name, path in paths.items():
            if name == "refinement_metadata":
                continue
            reader = WindowReader(path)
            stack.callback(reader.close)
            if len(reader.shape) != 2 or reader.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a two-dimensional integer raster")
            readers[name] = reader
        height, width = readers["labels"].shape
        if any(reader.shape != (height, width) for reader in readers.values()):
            raise ValueError("Uncertainty/provenance must exactly match native label raster alignment")
        if metadata is not None and metadata.get("shape_yx") != [height, width]:
            raise ValueError("Refinement metadata shape conflicts with the native rasters")
        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                x1, y1 = min(width, x0 + tile_size), min(height, y0 + tile_size)
                blocks = {name: reader.read(x0, y0, x1, y1) for name, reader in readers.items()}
                labels = blocks["labels"]
                if np.any(labels < 0):
                    raise ValueError("Label raster contains negative values")
                max_window = max(max_window, labels.size)
                if "uncertainty" in blocks and (np.any(blocks["uncertainty"] < 0) or np.any(blocks["uncertainty"] > 254)):
                    raise ValueError("Uncertainty raster must contain categorical codes 0..254")
                if "provenance" in blocks and not np.isin(blocks["provenance"], list(REFINEMENT_CODES)).all():
                    raise ValueError("Provenance raster contains codes absent from its legend")
                if "provenance" in blocks:
                    provenance, uncertain = blocks["provenance"], blocks["uncertainty"]
                    positive_codes = np.isin(provenance, [1, 2, 3, 5, 7, 8])
                    if np.any(positive_codes != (labels > 0)):
                        raise ValueError("Provenance label-presence semantics conflict with final labels")
                    if np.any((provenance == 0) & (uncertain != 0)) or np.any(np.isin(provenance, [1, 7]) & (uncertain != 0)):
                        raise ValueError("Unchanged/outside provenance conflicts with uncertainty codes")
                    if np.any((provenance == 3) & ((uncertain < 1) | (uncertain > 249))):
                        raise ValueError("Inferred original uncertainty must retain its source code")
                    for code, reason in ((2, 250), (5, 251), (6, 252), (8, 254)):
                        if np.any((provenance == code) & (uncertain != reason)):
                            raise ValueError("Refinement provenance conflicts with its uncertainty reason")
                    if np.any((provenance == 4) & ~(((uncertain >= 1) & (uncertain <= 249)) | (uncertain == 253))):
                        raise ValueError("Unresolved tissue must remain explicitly uncertain")
                for label, size in zip(*np.unique(labels, return_counts=True)):
                    key = str(int(label))
                    record = counts.setdefault(key, {"source_pixel_count": 0, "uncertainty_counts": {}, "provenance_counts": {}})
                    if len(counts) > max_labels:
                        raise ValueError("Native label count exceeds provenance-max-labels")
                    record["source_pixel_count"] += int(size)
                    selected = labels == label
                    for name in ("uncertainty", "provenance"):
                        if name in blocks:
                            for code, number in zip(*np.unique(blocks[name][selected], return_counts=True)):
                                code_key = str(int(code))
                                record[f"{name}_counts"][code_key] = record[f"{name}_counts"].get(code_key, 0) + int(number)
        io = {name: reader.backend for name, reader in readers.items()}
    if metadata is not None:
        totals = {}
        for record in counts.values():
            for code, number in record["provenance_counts"].items():
                name = REFINEMENT_CODES[int(code)]
                totals[name] = totals.get(name, 0) + number
        if totals != metadata.get("pixel_counts"):
            raise ValueError("Refinement metadata pixel counts conflict with the actual provenance raster")
    # A foreground aggregate makes binary output semantics explicit without
    # collapsing or discarding the native multiclass source inventory.
    foreground = {"source_pixel_count": 0, "uncertainty_counts": {}, "provenance_counts": {}}
    for label, record in counts.items():
        if label == "0":
            continue
        foreground["source_pixel_count"] += record["source_pixel_count"]
        for name in ("uncertainty_counts", "provenance_counts"):
            for code, number in record[name].items():
                foreground[name][code] = foreground[name].get(code, 0) + number
    for record in [*counts.values(), foreground]:
        total = record["source_pixel_count"]
        for name in ("uncertainty", "provenance"):
            record[f"{name}_fractions"] = {code: number / total for code, number in record[f"{name}_counts"].items()} if total else {}
        record["nonzero_uncertainty_fraction"] = ((total - record["uncertainty_counts"].get("0", 0)) / total) if total and uncertainty_path else None
        record["accepted_biological_confidence"] = None
    return {"schema_version": PROVENANCE_SCHEMA,
            "status": "refinement_provenance_available" if provenance_path else ("source_uncertainty_only_growth_provenance_unavailable" if uncertainty_path else "unavailable"),
            "inputs": inputs, "source_shape_yx": [height, width],
            "coordinate_frame": "analysis_crop_level_zero_pixel_edges",
            "label_summaries": counts, "foreground_summary": foreground,
            "source_label_zero_included": True,
            "refinement_legend": {str(k): v for k, v in REFINEMENT_CODES.items()} if metadata else None,
            "source_uncertainty_available": bool(uncertainty_path),
            "original_clustering_uncertainty_available": metadata.get("input_uncertainty_provided") if metadata else None,
            "producer_output_binding": producer_binding,
            "binding": ("Exact finalized producer output SHA256 values verified against every supplied raster; native counts and categorical semantics verified independently."
                        if producer_binding == "exact_output_sha256_verified" else
                        "Exact input file hashes and native raster counts; metadata paths alone are not identity evidence. Legacy producer metadata does not supply output hashes."),
            "interpretation": "Categorical computational provenance, not calibrated biological confidence. Code zero alone does not establish supported growth or biological correctness.",
            "io": {"backends": io, "scan_tile_size": tile_size, "maximum_scan_window_pixels": max_window,
                   "scope": "Bounded native provenance scan; polygon extraction is separate"}}


def write_geojson_with_provenance(args, features, summary, resolved_page, scale_x, scale_y, polygon_backend):
    summary_path = Path(args.provenance_summary_out or (str(args.out) + ".provenance.json"))
    if summary_path.resolve() == Path(args.out).resolve():
        raise ValueError("GeoJSON and provenance summary paths must differ")
    if summary_path.resolve().parent != Path(args.out).resolve().parent:
        raise ValueError("Provenance summary must be a sibling of its GeoJSON for portable linking")
    for target in (Path(args.out), summary_path):
        if target.resolve() in {Path(path).resolve() for path in (args.mask, args.uncertainty_mask, args.provenance_mask, args.provenance_metadata) if path}:
            raise ValueError("Vector outputs must not overwrite source artifacts")
    paths = {"labels": args.mask, "uncertainty": args.uncertainty_mask,
             "provenance": args.provenance_mask, "refinement_metadata": args.provenance_metadata}
    for name, record in summary["inputs"].items():
        if file_sha256(paths[name]) != record["sha256"]:
            raise ValueError(f"Source {name} changed during vectorization; no linked result can be emitted")
    summary["source_hashes_verified_after_vectorization"] = True
    identity = {name: getattr(args, name) for name in ("sample_key", "sample_id", "cluster_variant")}
    if any(identity.values()):
        if not all(identity.values()) or args.sample_key != f"{args.sample_id}::{args.cluster_variant}":
            raise ValueError("Provide matching sample-key, sample-id and cluster-variant together")
    summary["lineage"] = identity if any(identity.values()) else None
    summary["vectorization"] = {"page": resolved_page, "scale_to_level_zero_xy": [scale_x, scale_y],
        "backend": polygon_backend, "smooth_buffer_px": args.smooth_buffer, "smooth_passes": args.smooth_passes,
        "simplify_px": args.simplify, "preserve_topology": args.preserve_topology,
        "shared_boundary_simplify": args.shared_boundary_simplify,
        "simplify_max_categorical_difference_fraction": getattr(args, "simplify_max_categorical_difference_fraction", None),
        "simplify_search_steps": getattr(args, "simplify_search_steps", None),
        "simplification_method": ("shapely_coverage_simplify" if args.shared_boundary_simplify else
                                  ("per_geometry" if args.simplify > 0 else "none")),
        "native_pixel_edge_polygonization": (resolved_page == 0 and abs(scale_x - 1.0) < 1e-12 and abs(scale_y - 1.0) < 1e-12),
        "fill_holes": args.fill_holes,
        "minimum_polygon_area_px2": args.min_area, "binary": args.binary,
        "raster_connectivity": args.connectivity,
        "classification_group_prefix": args.group_prefix,
        "classification_group_map": {"filename": Path(args.group_map).name, "sha256": file_sha256(args.group_map)} if args.group_map else None,
        "feature_count": len(features),
        "coverage_diagnostics": getattr(args, "_coverage_diagnostics", None),
        "shared_boundary_smoothing": getattr(args, "_shared_boundary_smoothing_diagnostics", None),
        "summary_scope": "All native source pixels of the same label, including disconnected or vector-filtered components; NOT polygon-specific pixel fractions.",
        "geometry_authority": "Source rasters are authoritative. Smoothed, simplified, hole-filled or reduced-level vectors may include unsupported pixels or omit source pixels."}
    if not args.binary:
        geometries = [(int(feature["properties"]["value"]), shp_shape(feature["geometry"])) for feature in features]
        overlap_pairs = []
        total_overlap = 0.0
        tolerance = max(1e-6, abs(float(scale_x) * float(scale_y)) * 1e-6)
        for index, (left_value, left) in enumerate(geometries):
            for right_value, right in geometries[index + 1:]:
                overlap = float(left.intersection(right).area)
                if overlap > tolerance:
                    overlap_pairs.append({"values": [left_value, right_value], "area_px2": overlap})
                    total_overlap += overlap
        summary["vectorization"].update({
            "multiclass_geometry_mutually_exclusive": not overlap_pairs,
            "multiclass_overlap_area_px2": total_overlap,
            "multiclass_overlap_pairs": overlap_pairs,
        })
        if overlap_pairs:
            pairs = ", ".join(f"{row['values'][0]}-{row['values'][1]}" for row in overlap_pairs)
            raise ValueError(
                "Vectorization created overlapping cluster geometries "
                f"({pairs}); disable per-class hole filling/smoothing or use label-aware settings"
            )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    reference = {"file": summary_path.name, "sha256": file_sha256(summary_path)}
    for feature in features:
        label = str(int(feature["properties"]["value"]))
        record = summary["foreground_summary"] if args.binary else summary["label_summaries"].get(label)
        if record is None:
            raise ValueError("Vector label has no native source raster summary")
        feature["properties"]["raster_provenance"] = {
            "status": summary["status"], "summary": reference,
            "producer_output_binding": summary["producer_output_binding"],
            "summary_key": "foreground_summary" if args.binary else f"label_summaries/{label}",
            "scope": "all_native_source_pixels_of_this_label_not_this_vector_geometry",
            "source_pixel_count": record["source_pixel_count"],
            "nonzero_uncertainty_fraction": record["nonzero_uncertainty_fraction"],
            "uncertainty_fractions": record["uncertainty_fractions"],
            "provenance_fractions": record["provenance_fractions"],
            "calibrated_confidence": None}
    gj = {"type": "FeatureCollection", "features": features,
          "cellphenotyper_provenance": {"schema_version": PROVENANCE_SCHEMA, "status": summary["status"],
              "producer_output_binding": summary["producer_output_binding"],
              "summary": reference, "lineage": summary["lineage"],
              "geometry_authority": summary["vectorization"]["geometry_authority"]}}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(gj, handle, allow_nan=False, separators=(",", ":"))


def normalize_mask(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    raise ValueError(f"Mask must be 2D. Got shape={arr.shape}")


def _spatial_shape(shape):
    shape = tuple(int(x) for x in shape)
    if len(shape) == 2:
        return shape
    if len(shape) == 3 and shape[0] == 1:
        return shape[1:]
    if len(shape) == 3 and shape[-1] == 1:
        return shape[:2]
    raise ValueError(f"Mask page must be 2D. Got shape={shape}")


def read_tiff_page(path: str, page: int, max_page_side: int) -> tuple[np.ndarray, float, float, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        levels = list(getattr(series, "levels", []) or [])
        if levels:
            full_h, full_w = _spatial_shape(levels[0].shape)
            if page < 0:
                max_page_side = max(1, int(max_page_side))
                chosen = 0
                for idx, level in enumerate(levels):
                    lh, lw = _spatial_shape(level.shape)
                    chosen = idx
                    if max(lh, lw) <= max_page_side:
                        break
                page = chosen
            if page < 0 or page >= len(levels):
                raise ValueError(f"--page {page} out of range. This TIFF series has {len(levels)} pyramid levels.")
            level = levels[page]
            arr = level.asarray()
            page_h, page_w = _spatial_shape(arr.shape)
        else:
            if page < 0:
                page = 0
            if page >= len(tf.pages):
                raise ValueError(f"--page {page} out of range. This TIFF has {len(tf.pages)} pages.")
            full_h, full_w = _spatial_shape(tf.pages[0].shape)
            arr = tf.pages[page].asarray()
            page_h, page_w = _spatial_shape(arr.shape)
    scale_x = float(full_w) / float(page_w)
    scale_y = float(full_h) / float(page_h)
    return normalize_mask(arr), scale_x, scale_y, int(page)


def cast_for_rasterio(data: np.ndarray, binary: bool) -> np.ndarray:
    # rasterio supports: int8, uint8, int16, uint16, int32, float32, float64
    if binary:
        return (data > 0).astype(np.uint8)

    if data.dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.float32, np.float64):
        return data
    mx = int(data.max()) if data.size else 0
    mn = int(data.min()) if data.size else 0
    if mn >= 0 and mx <= 65535:
        return data.astype(np.uint16, copy=False)
    return data.astype(np.int32, copy=False)


def load_group_map(path: str | None) -> dict[int, str]:
    """
    JSON mapping file, value -> classification string.
    Example:
      {"1": "Tumor", "2": "Stroma", "3": "Immune"}
    """
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Group map not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out = {}
    for k, v in obj.items():
        out[int(k)] = str(v)
    return out


def classification_for_value(v: int, group_map: dict[int, str], prefix: str) -> str:
    # Always return something
    return group_map.get(v, f"{prefix}{v}")


def smooth_geom(g, smooth_buffer: float, smooth_passes: int,
                simplify: float, preserve_topology: bool):
    """
    Strong smoothing:
      - repeat buffer(+r) then buffer(-r) multiple times
      - simplify afterwards to reduce vertices / file size
    """
    if g.is_empty:
        return g

    out = g

    # multiple smoothing passes -> heavier smoothing
    if smooth_buffer and smooth_buffer > 0:
        passes = max(1, int(smooth_passes))
        for _ in range(passes):
            out = out.buffer(smooth_buffer, join_style=1, cap_style=1).buffer(
                -smooth_buffer, join_style=1, cap_style=1
            )

    if simplify and simplify > 0:
        out = out.simplify(simplify, preserve_topology=preserve_topology)

    if not out.is_valid:
        out = out.buffer(0)

    return out


def edge_match_polygon_coverage(labeled_geometries):
    """Node shared raster edges once and restore labels on the resulting faces."""
    grouped = {}
    for label, geometry in labeled_geometries:
        grouped.setdefault(int(label), []).append(geometry)
    label_regions = {label: unary_union(parts) for label, parts in grouped.items()}
    noded_boundaries = unary_union([geometry.boundary for _, geometry in labeled_geometries])
    faces = list(polygonize(noded_boundaries))
    repaired = []
    unassigned = 0
    for face in faces:
        point = face.representative_point()
        labels = [label for label, region in label_regions.items() if region.covers(point)]
        if len(labels) > 1:
            raise ValueError("Noded coverage face belongs to more than one raster label")
        if not labels:
            unassigned += 1
            continue
        repaired.append((labels[0], face))
    if not repaired:
        raise ValueError("Shared-edge noding produced no labelled polygon faces")
    return repaired, {
        "noded_face_count": len(faces),
        "labelled_face_count": len(repaired),
        "unassigned_background_face_count": unassigned,
    }


def _group_label_coverage(labels, geometries):
    grouped = {}
    for label, geometry in zip(labels, geometries):
        grouped.setdefault(int(label), []).append(geometry)
    return {
        label: unary_union(parts)
        for label, parts in sorted(grouped.items())
    }


def _polygon_segment_count(geometry):
    if geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return (len(geometry.exterior.coords) - 1) + sum(
            len(ring.coords) - 1 for ring in geometry.interiors
        )
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        return sum(_polygon_segment_count(child) for child in geometry.geoms)
    return 0


def _coverage_fidelity(reference, candidate):
    reference_union = unary_union(list(reference.values()))
    candidate_union = unary_union(list(candidate.values()))
    foreground_area = float(reference_union.area)
    if foreground_area <= 0:
        raise ValueError("Cannot measure simplification fidelity on empty foreground")
    per_label_area = {
        label: float(reference[label].symmetric_difference(candidate[label]).area)
        for label in reference
    }
    tissue_difference = float(reference_union.symmetric_difference(candidate_union).area)
    # A positive-label swap contributes to two per-class symmetric differences;
    # a tissue/background change contributes to one class and the union difference.
    categorical_difference = (sum(per_label_area.values()) + tissue_difference) / 2.0
    return {
        "categorical_difference_area_px2": categorical_difference,
        "categorical_difference_fraction": categorical_difference / foreground_area,
        "tissue_boundary_difference_area_px2": tissue_difference,
        "tissue_boundary_difference_fraction": tissue_difference / foreground_area,
        "per_label_symmetric_difference_area_px2": per_label_area,
        "per_label_symmetric_difference_fraction": {
            label: area / float(reference[label].area)
            for label, area in per_label_area.items()
        },
    }


def simplify_multiclass_coverage(labeled_geometries, tolerance: float, *,
                                 max_categorical_difference_fraction=None,
                                 search_steps=10, return_diagnostics=False):
    """Simplify a raster-derived coverage while enforcing an annotation-error budget."""
    try:
        import shapely
    except ImportError as exc:  # pragma: no cover - shapely is already a core dependency
        raise RuntimeError("Shared-boundary simplification requires Shapely >= 2.1") from exc
    if not hasattr(shapely, "coverage_simplify") or not hasattr(shapely, "coverage_is_valid"):
        raise RuntimeError("Shared-boundary simplification requires Shapely >= 2.1")
    labels = [int(label) for label, _ in labeled_geometries]
    geometries = np.asarray([geometry for _, geometry in labeled_geometries], dtype=object)
    if geometries.size == 0:
        return ([], {}) if return_diagnostics else []
    diagnostics = {
        "input_polygon_count": int(geometries.size),
        "input_coverage_valid": bool(shapely.coverage_is_valid(geometries)),
        "edge_match_repair_applied": False,
    }
    if not diagnostics["input_coverage_valid"]:
        validity = np.asarray(shapely.is_valid(geometries), dtype=bool)
        invalid_edges = np.asarray(shapely.coverage_invalid_edges(geometries), dtype=object)
        nonempty_edges = sum(not edge.is_empty for edge in invalid_edges)
        invalid_length = sum(float(edge.length) for edge in invalid_edges if not edge.is_empty)
        diagnostics.update({
            "invalid_geometry_count": int((~validity).sum()),
            "invalid_edge_set_count": int(nonempty_edges),
            "invalid_edge_length_px": float(invalid_length),
        })
        repaired, repair_diagnostics = edge_match_polygon_coverage(list(zip(labels, geometries.tolist())))
        diagnostics.update(repair_diagnostics)
        diagnostics["edge_match_repair_applied"] = True
        labels = [int(label) for label, _ in repaired]
        geometries = np.asarray([geometry for _, geometry in repaired], dtype=object)
        diagnostics["repaired_coverage_valid"] = bool(shapely.coverage_is_valid(geometries))
        if not diagnostics["repaired_coverage_valid"]:
            raise ValueError(
                "Noding shared raster boundaries did not produce a valid polygon coverage: "
                f"invalid_geometries={diagnostics['invalid_geometry_count']}/{diagnostics['input_polygon_count']}, "
                f"invalid_edge_sets={nonempty_edges}, invalid_edge_length={invalid_length:.6g}"
            )
    reference = _group_label_coverage(labels, geometries)
    labels = list(reference)
    geometries = np.asarray([reference[label] for label in labels], dtype=object)
    if not bool(shapely.coverage_is_valid(geometries)):
        raise ValueError("Dissolved label geometries do not form a valid polygon coverage")
    diagnostics["source_coordinate_count"] = int(sum(shapely.get_num_coordinates(g) for g in geometries))
    diagnostics["source_segment_count"] = int(sum(_polygon_segment_count(g) for g in geometries))
    diagnostics["requested_max_simplify_px"] = float(tolerance)
    diagnostics["max_categorical_difference_fraction"] = max_categorical_difference_fraction

    def evaluate(candidate_tolerance):
        candidate_geometries = (
            geometries.copy()
            if candidate_tolerance <= 0
            else np.asarray(shapely.coverage_simplify(
                geometries, float(candidate_tolerance), simplify_boundary=True
            ), dtype=object)
        )
        if candidate_geometries.shape != geometries.shape or any(g.is_empty for g in candidate_geometries):
            raise ValueError("Shared-boundary simplification removed or reordered a cluster geometry")
        if not bool(shapely.coverage_is_valid(candidate_geometries)):
            raise ValueError("Shared-boundary simplification produced an invalid polygon coverage")
        candidate = {label: geometry for label, geometry in zip(labels, candidate_geometries)}
        fidelity = _coverage_fidelity(reference, candidate)
        fidelity.update({
            "tolerance_px": float(candidate_tolerance),
            "coordinate_count": int(sum(shapely.get_num_coordinates(g) for g in candidate_geometries)),
            "segment_count": int(sum(_polygon_segment_count(g) for g in candidate_geometries)),
        })
        return candidate_geometries, fidelity

    trials = []
    if max_categorical_difference_fraction is None:
        simplified, selected = evaluate(float(tolerance))
        trials.append(selected)
    else:
        limit = float(max_categorical_difference_fraction)
        if limit < 0 or limit > 1:
            raise ValueError("Maximum categorical difference fraction must be in [0, 1]")
        if search_steps < 0:
            raise ValueError("Simplification search steps cannot be negative")
        baseline_geometries, baseline = evaluate(0.0)
        trials.append(baseline)
        upper_geometries, upper = evaluate(float(tolerance))
        trials.append(upper)
        if upper["categorical_difference_fraction"] <= limit:
            simplified, selected = upper_geometries, upper
        else:
            low_tolerance, high_tolerance = 0.0, float(tolerance)
            simplified, selected = baseline_geometries, baseline
            for _ in range(int(search_steps)):
                midpoint = (low_tolerance + high_tolerance) / 2.0
                midpoint_geometries, midpoint_result = evaluate(midpoint)
                trials.append(midpoint_result)
                if midpoint_result["categorical_difference_fraction"] <= limit:
                    low_tolerance = midpoint
                    simplified, selected = midpoint_geometries, midpoint_result
                else:
                    high_tolerance = midpoint
    diagnostics["simplification_trials"] = sorted(trials, key=lambda row: row["tolerance_px"])
    diagnostics["selected_simplify_px"] = selected["tolerance_px"]
    diagnostics["selected_fidelity"] = selected
    diagnostics["selected_coordinate_count"] = selected["coordinate_count"]
    diagnostics["selected_segment_count"] = selected["segment_count"]
    diagnostics["selected_coordinate_reduction_fraction"] = (
        1.0 - selected["coordinate_count"] / diagnostics["source_coordinate_count"]
    )
    diagnostics["selected_segment_reduction_fraction"] = (
        1.0 - selected["segment_count"] / diagnostics["source_segment_count"]
    )
    diagnostics["simplified_coverage_valid"] = True
    diagnostics["simplified_polygon_count"] = int(simplified.size)
    result = list(zip(labels, simplified.tolist()))
    return (result, diagnostics) if return_diagnostics else result


def valid_polygonal_parts(geometry):
    """Return valid positive-area Polygon parts without changing class identity."""
    if geometry.is_empty:
        return []
    if not geometry.is_valid:
        import shapely
        geometry = shapely.make_valid(geometry)
    if geometry.geom_type == "Polygon":
        return [geometry] if geometry.area > 0 else []
    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for child in geometry.geoms:
            parts.extend(valid_polygonal_parts(child))
        return parts
    return []


def drop_holes(geom):
    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return type(geom)(geom.exterior)
    if geom.geom_type == "MultiPolygon":
        return type(geom)([type(p)(p.exterior) for p in geom.geoms])
    return geom


def iter_polygons_from_mask(mask2d: np.ndarray, binary: bool, scale_x: float, scale_y: float,
                            connectivity: int = 4):
    data = cast_for_rasterio(mask2d, binary=binary)

    # Pixel coords: x=col, y=row, scaled back to level-0 pixel coordinates.
    transform = rasterio.Affine(float(scale_x), 0, 0, 0, float(scale_y), 0)

    for geom, val in shapes(data, mask=(data > 0), transform=transform, connectivity=int(connectivity)):
        yield geom, int(val)


def default_polygon_backend() -> str:
    try:
        import cv2  # noqa: F401
        return "opencv"
    except Exception:
        pass
    try:
        from skimage import measure  # noqa: F401
        return "skimage"
    except Exception:
        return "rasterio"


def contour_to_polygon(contour, scale_x: float, scale_y: float):
    pts = contour.reshape(-1, 2)
    if pts.shape[0] < 3:
        return None
    coords = [(float(x) * float(scale_x), float(y) * float(scale_y)) for x, y in pts]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def skimage_contour_to_polygon(contour, scale_x: float, scale_y: float):
    if contour.shape[0] < 3:
        return None
    coords = [(float(col) * float(scale_x), float(row) * float(scale_y)) for row, col in contour]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly


def iter_contours_from_mask(mask2d: np.ndarray, binary: bool, scale_x: float, scale_y: float, min_area: float, backend: str):
    if backend == "skimage":
        yield from iter_skimage_contours_from_mask(mask2d, binary, scale_x, scale_y, min_area)
        return
    import cv2

    data = cast_for_rasterio(mask2d, binary=binary)
    values = [1] if binary else [int(v) for v in np.unique(data) if int(v) > 0]
    area_scale = float(scale_x) * float(scale_y)
    for val in values:
        if binary:
            m = (data > 0).astype(np.uint8)
        else:
            m = (data == val).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if contour.shape[0] < 3:
                continue
            if min_area > 0 and (float(cv2.contourArea(contour)) * area_scale) < float(min_area):
                continue
            poly = contour_to_polygon(contour, scale_x, scale_y)
            if poly is not None:
                yield poly, val


def iter_skimage_contours_from_mask(mask2d: np.ndarray, binary: bool, scale_x: float, scale_y: float, min_area: float):
    from skimage import measure

    data = cast_for_rasterio(mask2d, binary=binary)
    values = [1] if binary else [int(v) for v in np.unique(data) if int(v) > 0]
    for val in values:
        m = (data > 0) if binary else (data == val)
        contours = measure.find_contours(m.astype(np.uint8), 0.5, fully_connected="high")
        for contour in contours:
            poly = skimage_contour_to_polygon(contour, scale_x, scale_y)
            if poly is None:
                continue
            if min_area > 0 and poly.area < float(min_area):
                continue
            yield poly, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="Input mask TIFF (binary or labeled; can be pyramidal/OME-TIFF)")
    ap.add_argument("--page", type=int, default=0,
                    help="TIFF page/pyramid level to read (0 = full-res; -1 = auto-select by --max-page-side)")
    ap.add_argument("--max-page-side", type=int, default=8192,
                    help="When --page -1, choose the first pyramid level with max(height,width) <= this value.")
    ap.add_argument("--out", required=True, help="Output GeoJSON file")
    ap.add_argument("--uncertainty-mask", help="Native aligned categorical uncertainty; never a confidence probability")
    ap.add_argument("--provenance-mask", help="Native refinement-provenance raster; requires its metadata and uncertainty")
    ap.add_argument("--provenance-metadata", help="Refinement schema/legend/counts JSON")
    ap.add_argument("--provenance-summary-out", help="Default: output GeoJSON path + .provenance.json")
    ap.add_argument("--provenance-tile-size", type=int, default=512)
    ap.add_argument("--provenance-max-labels", type=int, default=65536)
    ap.add_argument("--sample-key")
    ap.add_argument("--sample-id")
    ap.add_argument("--cluster-variant")

    ap.add_argument("--binary", action="store_true", help="Treat mask as binary foreground (mask>0)")
    ap.add_argument("--dissolve", action="store_true",
                    help="Binary only: merge all polygons into one MultiPolygon feature")
    ap.add_argument("--dissolve-by-value", action="store_true",
                    help="Labeled: merge polygons per value into one MultiPolygon per value (best for small GeoJSON).")
    ap.add_argument("--min-area", type=float, default=0.0, help="Drop polygons with area < this (pixel^2)")
    ap.add_argument("--polygon-backend", choices=("auto", "opencv", "skimage", "rasterio"), default="rasterio",
                    help="Polygon extraction backend. Contour backends are faster for one-geometry-per-label outputs.")
    ap.add_argument("--connectivity", type=int, choices=(4, 8), default=4,
                    help="Rasterio pixel connectivity; 4 avoids self-touching rings from diagonal-only contacts")

    # Strong smoothing controls
    ap.add_argument("--smooth-buffer", type=float, default=0.0,
                    help="Smoothing radius in pixels via buffer+/- (try 4–12 for strong smoothing)")
    ap.add_argument("--smooth-passes", type=int, default=1,
                    help="Repeat smoothing passes (try 2–4 for more smoothing)")
    ap.add_argument("--simplify", type=float, default=0.0,
                    help="Simplify tolerance in pixels (try 2–8 for strong simplification)")
    ap.add_argument("--preserve-topology", action="store_true",
                    help="Topology-preserving simplify (safer)")
    ap.add_argument("--shared-boundary-simplify", action="store_true",
                    help="For dissolved multiclass rasterio output, simplify all labels as one polygon coverage (Shapely >=2.1)")
    ap.add_argument("--simplify-max-categorical-difference-fraction", type=float,
                    help="Choose the strongest shared-boundary simplification up to --simplify whose exact polygon symmetric-difference fraction does not exceed this limit")
    ap.add_argument("--simplify-search-steps", type=int, default=10,
                    help="Binary-search iterations for fidelity-constrained shared-boundary simplification")
    ap.add_argument("--shared-boundary-smooth", action="store_true",
                    help="Selectively smooth long, sharp internal class-boundary vertices after coverage simplification")
    ap.add_argument("--shared-boundary-smoothing-coefficient", type=float, default=0.05,
                    help="Fraction of the local Laplacian displacement applied to eligible shared-boundary vertices")
    ap.add_argument("--shared-boundary-smoothing-passes", type=int, default=1,
                    help="Number of endpoint-preserving selective smoothing passes")
    ap.add_argument("--shared-boundary-minimum-turn-degrees", type=float, default=45.0,
                    help="Only smooth shared-boundary vertices turning by at least this angle")
    ap.add_argument("--shared-boundary-minimum-adjacent-length", type=float, default=16.0,
                    help="Only smooth vertices whose two adjacent segments are at least this many native pixels")

    ap.add_argument("--fill-holes", action="store_true",
                    help="Remove holes by keeping only exterior rings (after smoothing)")

    # Classification/grouping
    ap.add_argument("--group-map", default=None,
                    help="Optional JSON: value -> classification label (e.g. {'1':'Tumor'})")
    ap.add_argument("--group-prefix", default="group_",
                    help="Prefix used when group-map is not provided (default group_)")

    args = ap.parse_args()

    if args.shared_boundary_simplify:
        if args.binary or not args.dissolve_by_value or args.polygon_backend not in ("rasterio", "auto"):
            ap.error("--shared-boundary-simplify requires labeled --dissolve-by-value rasterio output")
        if args.simplify <= 0:
            ap.error("--shared-boundary-simplify requires --simplify > 0")
        if args.smooth_buffer > 0 or args.fill_holes:
            ap.error("Shared-boundary simplification is incompatible with per-class smoothing or hole filling")
    elif args.simplify_max_categorical_difference_fraction is not None:
        ap.error("--simplify-max-categorical-difference-fraction requires --shared-boundary-simplify")
    if args.shared_boundary_smooth and not args.shared_boundary_simplify:
        ap.error("--shared-boundary-smooth requires --shared-boundary-simplify")

    provenance_summary = raster_provenance_summary(args.mask, args.uncertainty_mask,
        args.provenance_mask, args.provenance_metadata, tile_size=args.provenance_tile_size,
        max_labels=args.provenance_max_labels)
    mask2d, scale_x, scale_y, resolved_page = read_tiff_page(args.mask, args.page, args.max_page_side)
    polygon_backend = args.polygon_backend
    if polygon_backend == "auto":
        polygon_backend = default_polygon_backend()
    print(
        f"[INFO] mask page={resolved_page} shape={mask2d.shape} "
        f"scale_x={scale_x:.6g} scale_y={scale_y:.6g} polygon_backend={polygon_backend}",
        flush=True,
    )
    group_map = load_group_map(args.group_map)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    features = []

    # ---------- Binary ----------
    if args.binary:
        if args.dissolve:
            geoms = []
            polygon_iter = (
                iter_contours_from_mask(mask2d, binary=True, scale_x=scale_x, scale_y=scale_y, min_area=args.min_area, backend=polygon_backend)
                if polygon_backend in ("opencv", "skimage")
                else ((shp_shape(geom), val) for geom, val in iter_polygons_from_mask(mask2d, binary=True, scale_x=scale_x, scale_y=scale_y, connectivity=args.connectivity))
            )
            for g, val in polygon_iter:
                if polygon_backend not in ("opencv", "skimage") and args.min_area > 0 and g.area < args.min_area:
                    continue
                if not g.is_empty:
                    geoms.append(g)

            if geoms:
                merged = unary_union(geoms)
                merged = smooth_geom(merged, args.smooth_buffer, args.smooth_passes,
                                     args.simplify, args.preserve_topology)
                if args.fill_holes:
                    merged = drop_holes(merged)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(merged),
                    "properties": {
                        "value": 1,
                        "classification": "foreground"
                    }
                })
        else:
            for geom, val in iter_polygons_from_mask(mask2d, binary=True, scale_x=scale_x, scale_y=scale_y, connectivity=args.connectivity):
                g = shp_shape(geom)
                if args.min_area > 0 and g.area < args.min_area:
                    continue
                g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                                args.simplify, args.preserve_topology)
                if args.fill_holes:
                    g = drop_holes(g)
                if g.is_empty:
                    continue
                features.append({
                    "type": "Feature",
                    "geometry": mapping(g),
                    "properties": {
                        "value": 1,
                        "classification": "foreground"
                    }
                })

        write_geojson_with_provenance(args, features, provenance_summary, resolved_page, scale_x, scale_y, polygon_backend)
        print(f"[OK] wrote {len(features)} feature(s) -> {args.out}")
        return

    # ---------- Labeled ----------
    if args.dissolve_by_value:
        # buckets: value -> list of geometries; then unary_union into one feature per value
        buckets: dict[int, list] = {}
        coverage_parts = []
        polygon_iter = (
            iter_contours_from_mask(mask2d, binary=False, scale_x=scale_x, scale_y=scale_y, min_area=args.min_area, backend=polygon_backend)
            if polygon_backend in ("opencv", "skimage")
            else ((shp_shape(geom), val) for geom, val in iter_polygons_from_mask(mask2d, binary=False, scale_x=scale_x, scale_y=scale_y, connectivity=args.connectivity))
        )
        for g, val in polygon_iter:
            if polygon_backend not in ("opencv", "skimage") and args.min_area > 0 and g.area < args.min_area:
                continue
            if g.is_empty:
                continue
            if args.shared_boundary_simplify:
                for part in valid_polygonal_parts(g):
                    if args.min_area <= 0 or part.area >= args.min_area:
                        coverage_parts.append((int(val), part))
            else:
                buckets.setdefault(val, []).append(g)

        labeled_geometries = []
        if args.shared_boundary_simplify:
            simplified_parts, args._coverage_diagnostics = simplify_multiclass_coverage(
                coverage_parts,
                args.simplify,
                max_categorical_difference_fraction=args.simplify_max_categorical_difference_fraction,
                search_steps=args.simplify_search_steps,
                return_diagnostics=True,
            )
            buckets = {}
            for val, geometry in simplified_parts:
                buckets.setdefault(int(val), []).append(geometry)
        for val, geoms in sorted(buckets.items(), key=lambda x: x[0]):
            merged = unary_union(geoms)
            merged = smooth_geom(merged, args.smooth_buffer, args.smooth_passes,
                                 0.0 if args.shared_boundary_simplify else args.simplify,
                                 args.preserve_topology)
            if args.fill_holes:
                merged = drop_holes(merged)
            labeled_geometries.append((int(val), merged))
        if args.shared_boundary_smooth:
            if len(labeled_geometries) < 2:
                args._shared_boundary_smoothing_diagnostics = {
                    "applied": False,
                    "reason": "fewer_than_two_positive_labels",
                }
            else:
                labeled_geometries, args._shared_boundary_smoothing_diagnostics = smooth_shared_boundary_coverage(
                    labeled_geometries,
                    smoothing_coefficient=args.shared_boundary_smoothing_coefficient,
                    smoothing_passes=args.shared_boundary_smoothing_passes,
                    minimum_turn_degrees=args.shared_boundary_minimum_turn_degrees,
                    minimum_adjacent_length=args.shared_boundary_minimum_adjacent_length,
                )
        for val, merged in labeled_geometries:
            features.append({
                "type": "Feature",
                "geometry": mapping(merged),
                "properties": {
                    "value": int(val),
                    "classification": classification_for_value(int(val), group_map, args.group_prefix)
                }
            })
    else:
        # one feature per connected region (each still gets classification based on its value)
        for geom, val in iter_polygons_from_mask(mask2d, binary=False, scale_x=scale_x, scale_y=scale_y, connectivity=args.connectivity):
            g = shp_shape(geom)
            if args.min_area > 0 and g.area < args.min_area:
                continue
            g = smooth_geom(g, args.smooth_buffer, args.smooth_passes,
                            args.simplify, args.preserve_topology)
            if args.fill_holes:
                g = drop_holes(g)
            if g.is_empty:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(g),
                "properties": {
                    "value": int(val),
                    "classification": classification_for_value(int(val), group_map, args.group_prefix)
                }
            })

    write_geojson_with_provenance(args, features, provenance_summary, resolved_page, scale_x, scale_y, polygon_backend)

    print(f"[OK] read: {args.mask} (page {resolved_page})")
    print(f"[OK] wrote {len(features)} feature(s) -> {args.out}")


if __name__ == "__main__":
    main()
