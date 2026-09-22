#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite
import tifffile

from ome_tiff_metadata import (
    create_tiff_memmap,
    label_storage_dtype,
    read_mpp_json,
    tiff_resolution_kwargs,
    validate_ome_tiff,
)

from grow_to_tissue import (
    DEFAULT_PALETTE,
    ensure_2d,
    load_mask_2d,
    load_tiff,
    pyramidize_with_raw2ometiff,
    save_preview_png,
)
from grow_to_tissue_core import compute_boundary, to_float_rgb
from medsam_border_refine import DEFAULT_MEDSAM_CHECKPOINT, MedSAMConfig, MedSAMUnavailableError, run_medsam_border_refine, _binary_dilate, _build_protected_core
from model_provenance import checkpoint_record, git_revision, sha256_file
from tissue_appearance_refine import refine_tissue_domains_by_appearance
from annealed_wand_boundary import annealed_wand_boundary_competition, fit_appearance_calibration


REFINEMENT_PROVENANCE = {0: "outside_tissue_support", 1: "original_label_unchanged", 2: "modified_original_label", 3: "inferred_from_originally_uncertain", 4: "unresolved_inside_tissue", 5: "inferred_new_label", 6: "removed_original_label", 7: "protected_original_core", 8: "pre_refinement_grown_assignment_unchanged"}
REFINEMENT_UNCERTAINTY = {250: "label_modified_by_refinement", 251: "new_label_inferred_by_refinement", 252: "original_label_removed", 253: "unresolved_inside_tissue", 254: "pre_refinement_growth_inferred"}
GRANDQC_KODAMA_OUTLIER_CODE = 5


def validate_uncertainty(mask, shape):
    values = np.asarray(mask)
    if values.shape != tuple(shape) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= 250) or not np.equal(values, np.floor(values)).all():
        raise ValueError("Clustering uncertainty must match the crop and contain integer codes 0..249; 250..254 are reserved for refinement")
    return values.astype(np.uint8, copy=False)


def refinement_constraints(original, tissue, uncertainty, core_radius, boundary_radius):
    """Protect confident interiors; permit uncertain pixels and boundary bands."""
    original = np.asarray(original)
    tissue = np.asarray(tissue, bool)
    uncertainty = validate_uncertainty(uncertainty, original.shape)
    hard_excluded = uncertainty == GRANDQC_KODAMA_OUTLIER_CODE
    tissue = tissue & ~hard_excluded
    uncertain = (uncertainty > 0) & ~hard_excluded
    protected = np.zeros_like(original)
    empty = np.zeros(original.shape, bool)
    for label in np.unique(original):
        if label <= 0:
            continue
        core = _build_protected_core((original == label) & tissue, empty, int(core_radius))
        protected[core & ~uncertain] = label
    boundary = label_boundaries(original) | compute_boundary(original > 0)
    editable = tissue & (protected == 0) & (uncertain | _binary_dilate(boundary, int(boundary_radius)) | (original == 0))
    return editable, protected


def enforce_refinement_constraints(result, original, tissue, editable=None, protected=None):
    result = np.asarray(result).copy()
    if editable is not None:
        result[~np.asarray(editable, bool)] = np.asarray(original)[~np.asarray(editable, bool)]
    if protected is not None:
        keep = np.asarray(protected) > 0
        result[keep] = np.asarray(protected)[keep]
    result[~np.asarray(tissue, bool)] = 0
    return result


def refinement_provenance(original, result, tissue, uncertainty=None, protected=None, source_labels=None):
    """Preserve original uncertainty codes even after inferred label assignment."""
    original, result, tissue = np.asarray(original), np.asarray(result), np.asarray(tissue, bool)
    incoming = np.zeros(original.shape, np.uint8) if uncertainty is None else validate_uncertainty(uncertainty, original.shape)
    provenance = np.zeros(original.shape, np.uint8)
    original_positive, final_positive = original > 0, result > 0
    provenance[tissue & final_positive & original_positive & (original == result)] = 1
    provenance[tissue & final_positive & original_positive & (original != result)] = 2
    provenance[tissue & final_positive & ~original_positive] = 5
    provenance[tissue & ~final_positive] = 4
    provenance[tissue & ~final_positive & original_positive] = 6
    provenance[tissue & final_positive & (incoming > 0)] = 3
    provenance[tissue & ~final_positive & (incoming > 0)] = 4
    if protected is not None:
        keep = (np.asarray(protected) > 0) & tissue
        if np.any(result[keep] != np.asarray(protected)[keep]):
            raise RuntimeError("Protected confident cores changed during refinement")
        provenance[keep & (incoming == 0)] = 7
    if source_labels is not None:
        source = np.asarray(source_labels)
        if source.shape != original.shape:
            raise ValueError("Original cluster support must match the refinement crop")
        # Growth already filled sparse observation masks before this stage.
        # Such labels are not accepted clustering observations, even if fixed
        # by a geometric protected-core constraint during refinement.
        grown_only = tissue & (source <= 0) & original_positive & final_positive & (original == result) & (incoming == 0)
        provenance[grown_only] = 8
    categorical = incoming.copy()
    for provenance_code, uncertainty_code in ((2, 250), (5, 251), (6, 252), (4, 253), (8, 254)):
        categorical[(provenance == provenance_code) & (incoming == 0)] = uncertainty_code
    categorical[~tissue] = 0
    return provenance, categorical


def apply_physical_parameters(args):
    options = {"medsam_core_erosion_um": ("medsam_core_erosion_radius", 1), "medsam_outer_dilation_um": ("medsam_outer_dilation_radius", 1), "medsam_smooth_um": ("medsam_smooth_radius", 1), "pre_boundary_radius_um": ("pre_boundary_radius", 1), "internal_boundary_radius_um": ("internal_boundary_radius", 1), "internal_gradient_sigma_um": ("internal_gradient_sigma_px", 1), "appearance_core_erosion_um": ("appearance_core_erosion_px", 1), "appearance_smooth_sigma_um": ("appearance_smooth_sigma_px", 1), "appearance_min_region_area_um2": ("appearance_min_region_area_px", 2)}
    chosen = {key: getattr(args, key, None) for key in options if getattr(args, key, None) is not None}
    args.physical_parameters = {}
    if not chosen:
        return
    if not args.resolution_json or json.loads(Path(args.resolution_json).read_text()).get("status") != "pass":
        raise ValueError("Micrometre refinement options require a verified resolution JSON with status=pass")
    mx, my = float(args.source_mpp_x), float(args.source_mpp_y)
    if min(mx, my) <= 0 or not np.isfinite([mx, my]).all() or not np.isclose(mx, my, rtol=.02):
        raise ValueError("Physical-radius morphology requires finite near-isotropic MPP")
    for key, value in chosen.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError("Physical refinement dimensions must be finite and nonnegative")
        target, power = options[key]
        converted = value / (mx * my if power == 2 else (mx + my) / 2)
        converted = float(converted) if target in {"appearance_smooth_sigma_px", "internal_gradient_sigma_px"} else int(np.ceil(converted))
        setattr(args, target, converted)
        args.physical_parameters[key] = {"requested": value, "effective_pixel_parameter": target, "effective_pixels": converted}


def write_refinement_provenance_outputs(args, final_labels, original_reader, tissue_reader, uncertainty_reader=None, protected_labels=None, source_reader=None):
    h, w = final_labels.shape
    provenance_path = Path(args.provenance_out or (str(args.out) + ".provenance.tif"))
    uncertainty_path = Path(args.uncertainty_out or (str(args.out) + ".uncertainty.tif"))
    metadata_path = Path(str(args.out) + ".provenance.json")
    output_paths = [provenance_path.resolve(), uncertainty_path.resolve(), metadata_path.resolve(), Path(args.out).resolve()]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Refinement label, uncertainty and provenance output paths must be distinct")
    input_paths = [Path(getattr(args, name)).resolve() for name in ("image", "seed_mask", "grown_mask", "tissue_mask", "clustering_uncertainty") if getattr(args, name, "")]
    for path in (provenance_path, uncertainty_path, metadata_path):
        if path.resolve() in input_paths:
            raise ValueError("Refinement provenance outputs must not overwrite input artifacts")
        if path.exists() and not getattr(args, "overwrite", False):
            raise FileExistsError(f"Preserving existing provenance output {path}; choose a new path or explicit --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
    provenance_out = create_tiff_memmap(provenance_path, shape=(h, w), dtype=np.dtype("uint8"), mpp_x=args.source_mpp_x, mpp_y=args.source_mpp_y)
    uncertainty_out = create_tiff_memmap(uncertainty_path, shape=(h, w), dtype=np.dtype("uint8"), mpp_x=args.source_mpp_x, mpp_y=args.source_mpp_y)
    counts = {}
    for y0 in range(0, h, max(1, int(args.stream_block_rows))):
        y1 = min(h, y0 + max(1, int(args.stream_block_rows)))
        original = ensure_2d(original_reader.read(y0, y1, 0, w), "original labels")
        tissue = tissue_reader.read(y0, y1, 0, w)
        uncertainty = uncertainty_reader.read(y0, y1, 0, w) if uncertainty_reader else None
        protected = protected_labels[y0:y1] if protected_labels is not None else None
        source = source_reader.read(y0, y1, 0, w) if source_reader else None
        provenance, categorical = refinement_provenance(original, final_labels[y0:y1], tissue, uncertainty, protected, source)
        provenance_out[y0:y1], uncertainty_out[y0:y1] = provenance, categorical
        ids, sizes = np.unique(provenance, return_counts=True)
        for label, size in zip(ids, sizes):
            counts[REFINEMENT_PROVENANCE[int(label)]] = counts.get(REFINEMENT_PROVENANCE[int(label)], 0) + int(size)
    provenance_out.flush(); uncertainty_out.flush()
    metadata = {"schema_version": "1.1.0", "provenance_path": str(provenance_path), "uncertainty_path": str(uncertainty_path), "input_clustering_uncertainty": args.clustering_uncertainty or None, "input_uncertainty_provided": bool(args.clustering_uncertainty), "original_uncertainty_codes_preserved": True, "original_uncertainty_preservation_scope": "inside_tissue_support; outside_support_is_code_0", "source_observation_support_provided": source_reader is not None, "original_label_reference": "pre_refinement_grown_labels", "provenance_codes": REFINEMENT_PROVENANCE, "additional_uncertainty_codes": REFINEMENT_UNCERTAINTY, "pixel_counts": counts, "physical_parameters": getattr(args, "physical_parameters", {}), "mpp_x": float(args.source_mpp_x), "mpp_y": float(args.source_mpp_y), "shape_yx": [int(h), int(w)], "coordinate_frame": "analysis_crop", "claim": "categorical computational provenance, not calibrated biological confidence"}
    # The label TIFF is pyramidized only later. Never bind an old destination or
    # a temporary flat checkpoint as though it were the finalized output.
    metadata["output_binding_status"] = "pending_final_label_export"
    metadata["staged_sidecar_artifacts"] = {
        name: {"filename": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in (("uncertainty", uncertainty_path), ("provenance", provenance_path))}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def finalize_refinement_provenance_outputs(args, metadata):
    """Bind the exact finalized label/sidecar files after successful pyramidization."""
    metadata_path = Path(str(args.out) + ".provenance.json")
    # JSON object keys are strings, including the categorical code legends.
    metadata = json.loads(json.dumps(metadata, allow_nan=False))
    if json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Refinement metadata changed before output binding completed")
    if metadata.get("output_binding_status") != "pending_final_label_export":
        raise ValueError("Only newly produced pending refinement metadata can be finalized")
    paths = {"labels": Path(args.out), "uncertainty": Path(metadata["uncertainty_path"]),
             "provenance": Path(metadata["provenance_path"])}
    artifacts = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Final refinement output is unavailable: {path}")
        with tifffile.TiffFile(path) as tif:
            shape = list(tif.series[0].levels[0].shape)
        if shape != metadata["shape_yx"]:
            raise ValueError(f"Final {name} shape differs from refinement provenance metadata")
        artifacts[name] = {"filename": path.name, "sha256": sha256_file(path),
                           "size_bytes": path.stat().st_size}
        if name != "labels" and artifacts[name] != metadata.get("staged_sidecar_artifacts", {}).get(name):
            raise ValueError(f"Refinement {name} sidecar changed after it was written and flushed")
    finalized = {**metadata, "output_binding_status": "complete", "output_artifacts": artifacts,
                 "output_binding_scope": "Exact finalized label TIFF after pyramidization plus flushed native uncertainty/provenance TIFFs"}
    metadata_path.write_text(json.dumps(finalized, indent=2) + "\n")
    return finalized


def sample_prefix(sample_id: str, out_path: str | Path) -> str:
    out_name = Path(out_path).name
    if out_name.endswith(".ome.tif"):
        out_name = out_name[:-8]
    if out_name.endswith(".tif"):
        out_name = out_name[:-4]
    return sample_id or out_name


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return (to_float_rgb(image) * 255.0).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    # Keep the tissue preview keyed to the numeric cluster ID, exactly as in
    # the KODAMA membership plot.  Enumerating only the labels visible in a
    # preview crop would shift colours whenever an earlier cluster was absent.
    for lid in unique:
        palette_index = (max(1, int(lid)) - 1) % len(DEFAULT_PALETTE)
        out[labels == lid] = DEFAULT_PALETTE[palette_index]
    return out


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b &= labels > 0
    return b


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    mask = np.asarray(mask).astype(bool)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_labels(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    labels = np.asarray(labels).astype(np.int32)
    colors = colorize_labels(labels).astype(np.float32)
    mask = labels > 0
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colors[mask]
    boundaries = label_boundaries(labels)
    rgb[boundaries] = colors[boundaries]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def boundary_compare(image: np.ndarray, raw_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).copy()
    b0 = compute_boundary(raw_mask)
    b1 = compute_boundary(final_mask)
    rgb[b0 & ~b1] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[b1 & ~b0] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[b0 & b1] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def heatmap(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob).astype(np.float32), 0.0, 1.0)
    out = np.zeros(p.shape + (3,), dtype=np.uint8)
    out[..., 0] = np.clip(255.0 * p, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(255.0 * (1.0 - np.abs(p - 0.5) * 2.0), 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(255.0 * (1.0 - p), 0, 255).astype(np.uint8)
    return out


def fit_panel(arr: np.ndarray, width: int = 420) -> Image.Image:
    img = Image.fromarray(arr)
    h = int(round(img.height * (width / img.width)))
    return img.resize((width, h), Image.Resampling.BILINEAR)


def make_panel(items: List[Tuple[str, np.ndarray]], out_path: Path, columns: int = 3) -> None:
    font = ImageFont.load_default()
    panels = [(title, fit_panel(arr)) for title, arr in items]
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    title_h = 22
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        r = idx // columns
        c = idx % columns
        x = c * cell_w
        y = r * (cell_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 6, y + 4), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_native_panel(items: List[Tuple[str, np.ndarray]], out_path: Path, columns: int = 2) -> None:
    """Create a QC panel without resizing the image crops."""
    font = ImageFont.load_default()
    panels = [(title, Image.fromarray(np.asarray(arr))) for title, arr in items]
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    title_h = 24
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        r = idx // columns
        c = idx % columns
        x = c * cell_w
        y = r * (cell_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 6, y + 5), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def preview_step_for_shape(shape: Tuple[int, int], max_side: int = 2048) -> int:
    h, w = int(shape[0]), int(shape[1])
    return max(1, int(np.ceil(max(h, w) / float(max_side))))


def preview_subsample(arr: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    if step == 1:
        return np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[::step, ::step]
    return arr[::step, ::step, ...]


def appearance_refine_working_scale(
    image: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    args,
    step: int,
    editable=None,
    protected=None,
    core_support=None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    step = max(1, int(step))
    return refine_tissue_domains_by_appearance(
        to_uint8_rgb(image),
        labels,
        tissue,
        clusters=int(args.appearance_clusters),
        core_erosion_px=max(1, int(round(args.appearance_core_erosion_px / step))),
        smooth_sigma=max(0.0, float(args.appearance_smooth_sigma_px) / step),
        min_region_area_px=max(1, int(round(args.appearance_min_region_area_px / (step * step)))),
        sample_pixels=int(args.appearance_sample_pixels),
        random_seed=int(args.appearance_random_seed),
        allow_multiclass=bool(getattr(args, "appearance_multiclass", False)),
        editable_mask=editable,
        protected_mask=protected,
        core_support=core_support,
        min_vote_fraction=float(getattr(args, "appearance_min_vote_fraction", 0.0)),
        min_vote_margin=float(getattr(args, "appearance_min_vote_margin", 0.0)),
    )


def pre_medsam_boundary_competition(
    image: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    args,
    *,
    editable=None,
    protected_labels=None,
    appearance_calibration=None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Run bounded annealed label competition before MedSAM inference."""
    source = np.asarray(labels)
    if not bool(getattr(args, "pre_boundary_competition", False)):
        return source.copy(), {"enabled": False, "applied": False, "changed_pixels": 0}
    step = max(1, int(args.pre_boundary_downsample))
    work_image = np.asarray(
        Image.fromarray(to_uint8_rgb(image)).resize(
            (int(np.ceil(source.shape[1] / step)), int(np.ceil(source.shape[0] / step))),
            Image.Resampling.BOX,
        )
    )
    work_labels = preview_subsample(source, step)
    work_tissue = preview_subsample(np.asarray(tissue, dtype=bool), step)
    work_editable = preview_subsample(np.asarray(editable, dtype=bool), step) if editable is not None else None
    work_protected = preview_subsample(np.asarray(protected_labels), step) if protected_labels is not None else None
    work_result, metadata = annealed_wand_boundary_competition(
        work_image,
        work_labels,
        work_tissue,
        editable_mask=work_editable,
        protected_labels=work_protected,
        boundary_radius=max(1, int(np.ceil(int(args.pre_boundary_radius) / step))),
        iterations=int(args.pre_boundary_iterations),
        initial_temperature=float(args.pre_boundary_initial_temperature),
        final_temperature=float(args.pre_boundary_final_temperature),
        data_weight=float(args.pre_boundary_data_weight),
        smoothness_weight=float(args.pre_boundary_smoothness_weight),
        edge_beta=float(args.pre_boundary_edge_beta),
        connectivity=int(getattr(args, "pre_boundary_connectivity", 8)),
        appearance_calibration=appearance_calibration,
    )
    y_index = np.minimum(np.arange(source.shape[0]) // step, work_result.shape[0] - 1)
    x_index = np.minimum(np.arange(source.shape[1]) // step, work_result.shape[1] - 1)
    expanded = work_result[np.ix_(y_index, x_index)]
    change = (expanded != source) & (expanded > 0) & (source > 0) & np.asarray(tissue, dtype=bool)
    if editable is not None:
        change &= np.asarray(editable, dtype=bool)
    if protected_labels is not None:
        change &= np.asarray(protected_labels) == 0
    result = source.copy()
    result[change] = expanded[change].astype(source.dtype, copy=False)
    result[~np.asarray(tissue, dtype=bool)] = 0
    metadata.update(
        working_downsample=step,
        working_shape_yx=list(map(int, work_result.shape)),
        changed_pixels_working=int(metadata.get("changed_pixels", 0)),
        changed_pixels=int(np.count_nonzero(result != source)),
        maximum_boundary_displacement_px=int(args.pre_boundary_radius) + step - 1,
        foreground_footprint_invariant=True,
        grandqc_support_is_hard_constraint=True,
    )
    return result.astype(source.dtype, copy=False), metadata


def load_tiff_memmap(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    try:
        return tifffile.memmap(path)
    except Exception:
        return load_tiff(path, name)


def first_zarr_array(root):
    if hasattr(root, "shape"):
        return root
    names = list(root.array_keys())
    if "0" in names:
        return root["0"]
    if names:
        return root[names[0]]
    for _, obj in root.items():
        if hasattr(obj, "shape"):
            return obj
    raise ValueError("TIFF zarr store did not expose an array.")


class TiffWindowReader:
    def __init__(self, path: str, name: str):
        self.path = path
        self.name = name
        self.kind = "memmap"
        self._tf = None
        self._store = None
        self._arr = None
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
        try:
            self._arr = tifffile.memmap(path)
        except Exception:
            try:
                import zarr
            except Exception as exc:
                raise RuntimeError(f"{name} is not memmappable and zarr is unavailable: {path}") from exc
            self.kind = "zarr"
            self._tf = tifffile.TiffFile(path)
            self._store = self._tf.series[0].aszarr()
            self._arr = first_zarr_array(zarr.open(self._store, mode="r"))
        self.shape = tuple(int(x) for x in self._arr.shape)
        self.dtype = np.dtype(self._arr.dtype)

    def spatial_shape(self) -> Tuple[int, int]:
        if len(self.shape) == 2:
            return int(self.shape[0]), int(self.shape[1])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return int(self.shape[0]), int(self.shape[1])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return int(self.shape[1]), int(self.shape[2])
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
        if len(self.shape) == 2:
            return np.asarray(self._arr[y0:y1, x0:x1])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return np.asarray(self._arr[y0:y1, x0:x1, ...])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return np.asarray(self._arr[:, y0:y1, x0:x1]).transpose(1, 2, 0)
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def read_stride(self, step: int) -> np.ndarray:
        step = max(1, int(step))
        if len(self.shape) == 2:
            return np.asarray(self._arr[::step, ::step])
        if len(self.shape) == 3 and self.shape[-1] in (1, 3, 4):
            return np.asarray(self._arr[::step, ::step, ...])
        if len(self.shape) == 3 and self.shape[0] in (1, 3, 4):
            return np.asarray(self._arr[:, ::step, ::step]).transpose(1, 2, 0)
        raise ValueError(f"{self.name} has unsupported TIFF shape: {self.shape}")

    def close(self) -> None:
        try:
            if self._store is not None and hasattr(self._store, "close"):
                self._store.close()
        finally:
            if self._tf is not None:
                self._tf.close()


class ScaledBinaryMaskReader:
    """Read a low-resolution mask in target-image coordinates without seams."""

    def __init__(self, path: str, target_shape: Tuple[int, int], name: str = "tissue mask"):
        self.reader = TiffWindowReader(path, name)
        self.source_shape = self.reader.spatial_shape()
        self.target_shape = tuple(map(int, target_shape))

    def read(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        target_h, target_w = self.target_shape
        source_h, source_w = self.source_shape
        y_index = np.minimum(
            source_h - 1,
            np.floor(np.arange(y0, y1, dtype=np.float64) * source_h / max(1, target_h)).astype(np.int64),
        )
        x_index = np.minimum(
            source_w - 1,
            np.floor(np.arange(x0, x1, dtype=np.float64) * source_w / max(1, target_w)).astype(np.int64),
        )
        sy0, sy1 = int(y_index.min()), int(y_index.max()) + 1
        sx0, sx1 = int(x_index.min()), int(x_index.max()) + 1
        source = ensure_2d(self.reader.read(sy0, sy1, sx0, sx1), "scaled tissue-mask block")
        return np.asarray(source[np.ix_(y_index - sy0, x_index - sx0)] != 0)

    def read_stride(self, step: int) -> np.ndarray:
        """Read a globally aligned preview without constructing the full-size mask."""
        step = max(1, int(step))
        target_h, target_w = self.target_shape
        source_h, source_w = self.source_shape
        target_y = np.arange(0, target_h, step, dtype=np.int64)
        target_x = np.arange(0, target_w, step, dtype=np.int64)
        source_y = np.minimum(source_h - 1, np.floor(target_y * source_h / max(1, target_h)).astype(np.int64))
        source_x = np.minimum(source_w - 1, np.floor(target_x * source_w / max(1, target_w)).astype(np.int64))
        sy0, sy1 = int(source_y.min()), int(source_y.max()) + 1
        sx0, sx1 = int(source_x.min()), int(source_x.max()) + 1
        source = ensure_2d(self.reader.read(sy0, sy1, sx0, sx1), "scaled tissue-mask preview")
        return np.asarray(source[np.ix_(source_y - sy0, source_x - sx0)] != 0)

    def close(self) -> None:
        self.reader.close()


def tiff_spatial_shape(path: str, name: str) -> Tuple[int, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    with tifffile.TiffFile(path) as tf:
        shape = tuple(int(x) for x in tf.series[0].shape)
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3 and shape[-1] in (1, 3, 4):
        return shape[0], shape[1]
    if len(shape) == 3 and shape[0] in (1, 3, 4):
        return shape[1], shape[2]
    raise ValueError(f"{name} has unsupported TIFF shape: {shape}")


def tile_starts(n: int, tile_size: int, overlap: int) -> List[int]:
    n = int(n)
    tile_size = max(1, int(tile_size))
    overlap = max(0, min(int(overlap), tile_size - 1))
    if n <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, n, stride))
    last = max(0, n - tile_size)
    starts = [s for s in starts if s <= last]
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def commit_bounds(y0: int, y1: int, x0: int, x1: int, shape: Tuple[int, int], overlap: int) -> Tuple[int, int, int, int]:
    h, w = int(shape[0]), int(shape[1])
    margin = max(0, int(overlap) // 2)
    cy0 = y0 if y0 <= 0 else min(y1, y0 + margin)
    cy1 = y1 if y1 >= h else max(y0, y1 - margin)
    cx0 = x0 if x0 <= 0 else min(x1, x0 + margin)
    cx1 = x1 if x1 >= w else max(x0, x1 - margin)
    return int(cy0), int(cy1), int(cx0), int(cx1)


def internal_refinement_gradient(
    image: np.ndarray,
    foreground: np.ndarray,
    *,
    space: str = "luminance",
    sigma_px: float = 0.0,
) -> np.ndarray:
    """Build a robust multichannel H&E gradient for boundary watershed."""
    from scipy import ndimage as ndi
    from skimage.color import rgb2gray
    from skimage.filters import sobel

    rgb = to_uint8_rgb(image)
    space = str(space).strip().lower()
    sigma_px = float(sigma_px)
    if sigma_px < 0 or not np.isfinite(sigma_px):
        raise ValueError("Internal-boundary gradient sigma must be finite and nonnegative")
    if space == "luminance" and sigma_px == 0:
        # Exact historical behavior for reproducibility.
        return sobel(rgb2gray(rgb)).astype(np.float32)

    lab = None
    od = None
    if space in {"luminance", "lab", "lab_od"}:
        from tissue_appearance_refine import rgb_to_lab
        lab = rgb_to_lab(rgb)
    if space in {"od", "lab_od"}:
        od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    if space == "luminance":
        features = lab[..., :1]
    elif space == "lab":
        features = lab
    elif space == "od":
        features = od
    elif space == "lab_od":
        features = np.concatenate((lab, od), axis=-1)
    else:
        raise ValueError(f"Unsupported internal-boundary gradient space: {space!r}")

    selected = features[np.asarray(foreground, dtype=bool)]
    if selected.size == 0:
        return np.zeros(np.asarray(foreground).shape, dtype=np.float32)
    center = np.median(selected, axis=0)
    q25, q75 = np.percentile(selected, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 0.1 * np.std(selected, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = np.clip((features - center) / scale, -10.0, 10.0)
    if sigma_px > 0:
        standardized = np.stack(
            [ndi.gaussian_filter(standardized[..., index], sigma=sigma_px)
             for index in range(standardized.shape[-1])], axis=-1,
        )
    gradient = np.zeros(np.asarray(foreground).shape, dtype=np.float32)
    for index in range(standardized.shape[-1]):
        channel = standardized[..., index]
        gradient += ndi.sobel(channel, axis=0) ** 2 + ndi.sobel(channel, axis=1) ** 2
    gradient = np.sqrt(gradient / standardized.shape[-1])
    selected_gradient = gradient[np.asarray(foreground, dtype=bool)]
    low, high = np.percentile(selected_gradient, [1, 99])
    return np.clip((gradient - low) / max(1e-6, high - low), 0.0, 1.0).astype(np.float32)


def refine_internal_cluster_boundaries(
    image: np.ndarray,
    labels: np.ndarray,
    radius: int,
    editable_mask=None,
    protected_mask=None,
    gradient_space: str = "luminance",
    gradient_sigma_px: float = 0.0,
    watershed_compactness: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Snap only inter-cluster boundaries to local full-resolution image gradients."""
    radius = max(0, int(radius))
    source = np.asarray(labels)
    if radius == 0 or source.size == 0:
        return source.copy(), {"enabled": int(radius > 0), "editable_pixels": 0, "changed_pixels": 0}

    foreground = source > 0
    internal = np.zeros(source.shape, dtype=bool)
    vertical = (source[:-1, :] != source[1:, :]) & foreground[:-1, :] & foreground[1:, :]
    horizontal = (source[:, :-1] != source[:, 1:]) & foreground[:, :-1] & foreground[:, 1:]
    internal[:-1, :] |= vertical
    internal[1:, :] |= vertical
    internal[:, :-1] |= horizontal
    internal[:, 1:] |= horizontal
    if not internal.any():
        return source.copy(), {"enabled": 1, "editable_pixels": 0, "changed_pixels": 0}

    from scipy.ndimage import distance_transform_edt
    from skimage.segmentation import watershed

    if float(watershed_compactness) < 0 or not np.isfinite(watershed_compactness):
        raise ValueError("Internal-boundary watershed compactness must be finite and nonnegative")

    editable = (distance_transform_edt(~internal) <= radius) & foreground
    if editable_mask is not None:
        editable &= np.asarray(editable_mask, bool)
    if protected_mask is not None:
        editable &= ~np.asarray(protected_mask, bool)
    markers = source.astype(np.int32, copy=True)
    markers[editable] = 0
    if not np.any(markers > 0):
        return source.copy(), {"enabled": 1, "editable_pixels": int(editable.sum()), "changed_pixels": 0}

    gradient = internal_refinement_gradient(
        image, foreground, space=gradient_space, sigma_px=gradient_sigma_px,
    )
    snapped = watershed(
        gradient, markers=markers, mask=foreground,
        connectivity=np.ones((3, 3), dtype=np.uint8),
        compactness=float(watershed_compactness),
    )
    result = source.copy()
    valid = editable & (snapped > 0)
    result[valid] = snapped[valid].astype(source.dtype, copy=False)
    return result, {
        "enabled": 1,
        "editable_pixels": int(editable.sum()),
        "changed_pixels": int(np.count_nonzero(result != source)),
        "radius_px": radius,
        "gradient_space": str(gradient_space),
        "gradient_sigma_px": float(gradient_sigma_px),
        "watershed_compactness": float(watershed_compactness),
    }


def make_medsam_config(args) -> MedSAMConfig:
    return MedSAMConfig(
        checkpoint=str(args.medsam_checkpoint),
        device=str(args.medsam_device),
        bbox_margin=int(args.medsam_bbox_margin),
        component_min_area=int(args.medsam_component_min_area),
        component_merge_distance=int(args.medsam_component_merge_distance),
        seed_dilation_radius=int(args.medsam_seed_dilation_radius),
        core_erosion_radius=int(args.medsam_core_erosion_radius),
        outer_dilation_radius=int(args.medsam_outer_dilation_radius),
        min_object_size=int(args.medsam_min_object_size),
        smooth_radius=int(args.medsam_smooth_radius),
        force_core_preservation=bool(args.medsam_force_core_preservation),
        save_debug=bool(args.medsam_save_debug),
        cluster_tile_size=int(args.medsam_cluster_tile_size),
        cluster_tile_overlap=int(args.medsam_cluster_tile_overlap),
    )


def medsam_model_provenance(config: MedSAMConfig) -> dict:
    checkpoint = Path(config.checkpoint).resolve()
    return {
        "source_repository": "https://github.com/bowang-lab/MedSAM",
        "requested_revision": None,
        "resolved_revision": git_revision(config.repo_dir),
        "cache_path": str(checkpoint.parent),
        "checkpoints": [checkpoint_record(checkpoint, logical_name="medsam_vit_b")],
    }


def pyramidize_or_fallback(tmp_flat: Path, out_path: str, args) -> None:
    try:
        pyramidize_with_raw2ometiff(
            in_tif=str(tmp_flat),
            out_ome_tif=str(out_path),
            compression=args.pyr_compression,
            max_workers=args.max_workers,
            downsample=args.downsample,
            overwrite=args.overwrite,
            keep_tmp=args.keep_tmp,
            legacy=args.legacy,
        )
        ome_summary = validate_ome_tiff(
            out_path,
            expected_shape=args.output_spatial_shape,
            expected_mpp=(args.source_mpp_x, args.source_mpp_y),
        )
        print(f"[INFO] Validated MedSAM OME-TIFF: {json.dumps(ome_summary, sort_keys=True)}")
    except Exception:
        out_file = Path(out_path)
        if out_file.exists():
            out_file.unlink()
        raise


def count_nonzero_2d_blocks(arr: np.ndarray, block_rows: int = 512) -> int:
    h = int(arr.shape[0])
    block_rows = max(1, int(block_rows))
    total = 0
    for y0 in range(0, h, block_rows):
        y1 = min(h, y0 + block_rows)
        total += int(np.count_nonzero(np.asarray(arr[y0:y1, :])))
    return total


def read_stride_tiled(reader: TiffWindowReader, step: int, tile_size: int = 4096) -> np.ndarray:
    """Downsample a WSI reader without forcing zarr/tifffile to stride over the full image."""
    step = max(1, int(step))
    h, w = reader.spatial_shape()
    if step == 1:
        return reader.read(0, h, 0, w)
    tile_size = max(step, int(tile_size))
    out_h = (h + step - 1) // step
    out_w = (w + step - 1) // step
    sample = reader.read(0, min(h, 1), 0, min(w, 1))
    if sample.ndim == 2:
        out = np.zeros((out_h, out_w), dtype=sample.dtype)
    else:
        out = np.zeros((out_h, out_w) + sample.shape[2:], dtype=sample.dtype)
    del sample
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        sy0 = ((y0 + step - 1) // step) * step
        if sy0 >= y1:
            continue
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            sx0 = ((x0 + step - 1) // step) * step
            if sx0 >= x1:
                continue
            block = reader.read(sy0, y1, sx0, x1)
            small = block[::step, ::step] if block.ndim == 2 else block[::step, ::step, ...]
            oy0 = sy0 // step
            ox0 = sx0 // step
            out[oy0:oy0 + small.shape[0], ox0:ox0 + small.shape[1], ...] = small
            del block, small
        gc.collect()
    return out


def array_stride_tiled(arr: np.ndarray, step: int, tile_size: int = 4096) -> np.ndarray:
    step = max(1, int(step))
    h, w = int(arr.shape[0]), int(arr.shape[1])
    if step == 1:
        return np.asarray(arr)
    tile_size = max(step, int(tile_size))
    out = np.zeros(((h + step - 1) // step, (w + step - 1) // step), dtype=arr.dtype)
    for y0 in range(0, h, tile_size):
        y1 = min(h, y0 + tile_size)
        sy0 = ((y0 + step - 1) // step) * step
        if sy0 >= y1:
            continue
        for x0 in range(0, w, tile_size):
            x1 = min(w, x0 + tile_size)
            sx0 = ((x0 + step - 1) // step) * step
            if sx0 >= x1:
                continue
            small = np.asarray(arr[sy0:y1:step, sx0:x1:step])
            oy0 = sy0 // step
            ox0 = sx0 // step
            out[oy0:oy0 + small.shape[0], ox0:ox0 + small.shape[1]] = small
            del small
        gc.collect()
    return out


def choose_random_tissue_crop(
    raw_preview: np.ndarray,
    refined_preview: np.ndarray,
    step: int,
    full_shape: Tuple[int, int],
    crop_size: int,
    seed: int,
) -> Tuple[int, int, int, int]:
    h, w = int(full_shape[0]), int(full_shape[1])
    crop_size = max(1, min(int(crop_size), h, w))
    tissue_preview = (np.asarray(raw_preview) > 0) | (np.asarray(refined_preview) > 0)
    coords = np.argwhere(tissue_preview)
    rng = np.random.default_rng(int(seed))
    if coords.size:
        py, px = coords[int(rng.integers(coords.shape[0]))]
        center_y = int(py) * int(step) + int(step) // 2
        center_x = int(px) * int(step) + int(step) // 2
    else:
        center_y = h // 2
        center_x = w // 2
    y0 = max(0, min(h - crop_size, center_y - crop_size // 2))
    x0 = max(0, min(w - crop_size, center_x - crop_size // 2))
    return int(y0), int(y0 + crop_size), int(x0), int(x0 + crop_size)


def medsam_change_map(image: np.ndarray, raw_labels: np.ndarray, refined_labels: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    raw = np.asarray(raw_labels)
    refined = np.asarray(refined_labels)
    raw_fg = raw > 0
    refined_fg = refined > 0
    raw_only = raw_fg & ~refined_fg
    final_only = refined_fg & ~raw_fg
    relabeled = (raw != refined) & raw_fg & refined_fg
    rgb[raw_only] = (0.50 * rgb[raw_only]) + 0.50 * np.array([255, 0, 0], dtype=np.float32)
    rgb[final_only] = (0.50 * rgb[final_only]) + 0.50 * np.array([0, 255, 255], dtype=np.float32)
    rgb[relabeled] = (0.45 * rgb[relabeled]) + 0.55 * np.array([255, 255, 0], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def grandqc_empty_exclusion_map(
    image: np.ndarray,
    tissue_support: np.ndarray,
    labels_before_support: np.ndarray,
    final_labels: np.ndarray,
) -> np.ndarray:
    """Visualize GrandQC empty-area negatives and verify final label exclusion."""
    rgb = to_uint8_rgb(image).astype(np.float32)
    support = np.asarray(tissue_support).astype(bool)
    before = np.asarray(labels_before_support)
    final = np.asarray(final_labels)
    if support.shape != before.shape or support.shape != final.shape or rgb.shape[:2] != support.shape:
        raise ValueError("GrandQC exclusion QC inputs must share one spatial shape")
    empty = ~support
    removed = empty & (before > 0)
    leaked = empty & (final > 0)
    rgb[empty] = (0.35 * rgb[empty]) + 0.65 * np.array([55, 120, 220], dtype=np.float32)
    rgb[removed] = (0.20 * rgb[removed]) + 0.80 * np.array([255, 40, 40], dtype=np.float32)
    rgb[leaked] = np.array([255, 0, 255], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def crop_change_summary(raw_labels: np.ndarray, refined_labels: np.ndarray) -> Dict[str, int]:
    raw = np.asarray(raw_labels)
    refined = np.asarray(refined_labels)
    raw_fg = raw > 0
    refined_fg = refined > 0
    return {
        "raw_pixels": int(np.count_nonzero(raw_fg)),
        "final_pixels": int(np.count_nonzero(refined_fg)),
        "removed_pixels": int(np.count_nonzero(raw_fg & ~refined_fg)),
        "added_pixels": int(np.count_nonzero(refined_fg & ~raw_fg)),
        "relabel_pixels": int(np.count_nonzero((raw != refined) & raw_fg & refined_fg)),
    }


def save_fullres_qc_crop(
    image_crop: np.ndarray,
    raw_crop: np.ndarray,
    refined_crop: np.ndarray,
    out_path: Path,
    crop_bounds: Tuple[int, int, int, int],
) -> Dict[str, object]:
    raw_crop = ensure_2d(np.asarray(raw_crop), "raw MedSAM QC crop")
    refined_crop = ensure_2d(np.asarray(refined_crop), "refined MedSAM QC crop")
    make_native_panel(
        [
            ("Original crop (native resolution)", to_uint8_rgb(image_crop)),
            ("Before MedSAM", overlay_labels(image_crop, raw_crop)),
            ("After MedSAM", overlay_labels(image_crop, refined_crop)),
            ("Changes: red removed, cyan added, yellow relabeled", medsam_change_map(image_crop, raw_crop, refined_crop)),
        ],
        out_path,
        columns=2,
    )
    y0, y1, x0, x1 = crop_bounds
    meta: Dict[str, object] = {
        "path": str(out_path),
        "y0": int(y0),
        "y1": int(y1),
        "x0": int(x0),
        "x1": int(x1),
        "height": int(y1 - y0),
        "width": int(x1 - x0),
    }
    meta.update(crop_change_summary(raw_crop, refined_crop))
    return meta


def write_stream_progress(path: Path | None, payload: Dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def run_large_image_streaming_medsam(args, med_cfg: MedSAMConfig) -> None:
    start = time.perf_counter()
    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = sample_prefix(args.sample_id, args.out)

    image_reader = TiffWindowReader(args.image, "image")
    seed_reader = TiffWindowReader(args.seed_mask, "seed mask")
    grown_reader = TiffWindowReader(args.grown_mask, "grown mask")
    tissue_reader = None
    uncertainty_reader = None
    protected_out = None
    editable_out = None
    constraint_paths = []
    try:
        shape = seed_reader.spatial_shape()
        if grown_reader.spatial_shape() != shape:
            raise ValueError(f"Step-15 and step-16 masks must share the same shape: {shape} vs {grown_reader.spatial_shape()}")
        if image_reader.spatial_shape() != shape:
            raise ValueError(f"Image and masks must share the same spatial shape: {image_reader.spatial_shape()} vs {shape}")
        tissue_reader = ScaledBinaryMaskReader(args.tissue_mask, shape)
        if args.clustering_uncertainty:
            uncertainty_reader = TiffWindowReader(args.clustering_uncertainty, "clustering uncertainty")
            if uncertainty_reader.spatial_shape() != shape:
                raise ValueError("Clustering uncertainty must exactly match the seed/grown crop")
            if args.stream_resume_tmp:
                raise ValueError("An uncertainty-guided stream must start a new checkpoint; legacy checkpoints do not contain protected-core provenance")

        label_dtype = seed_reader.dtype if np.issubdtype(seed_reader.dtype, np.integer) else np.uint16
        if np.dtype(label_dtype).itemsize > np.dtype(grown_reader.dtype).itemsize and np.issubdtype(grown_reader.dtype, np.integer):
            label_dtype = grown_reader.dtype
        if np.dtype(label_dtype).itemsize < 2:
            label_dtype = np.uint16
        label_dtype = np.dtype(label_dtype).newbyteorder("=")
        storage_dtype = label_dtype.newbyteorder(">")

        h, w = shape
        tile_size = max(512, int(args.medsam_cluster_tile_size))
        overlap = max(0, min(int(args.medsam_cluster_tile_overlap), tile_size - 1))
        block_rows = max(1, int(args.stream_block_rows))
        resume_tmp = Path(args.stream_resume_tmp).expanduser() if args.stream_resume_tmp else None
        resume_tiles = max(0, int(args.stream_resume_tiles))
        resume_grid_y = max(0, int(args.stream_resume_grid_y))
        resume_grid_x = max(0, int(args.stream_resume_grid_x))
        resume_by_grid = resume_grid_y > 0 and resume_grid_x > 0
        progress_path = Path(args.stream_progress_json).expanduser() if args.stream_progress_json else outdir / f"{prefix}_medsam_stream_progress.json"
        if resume_tmp is not None:
            if not resume_tmp.exists():
                raise FileNotFoundError(f"--stream-resume-tmp not found: {resume_tmp}")
            tmp_flat = resume_tmp
            refined_out = tifffile.memmap(tmp_flat)
            if tuple(refined_out.shape) != (h, w):
                raise ValueError(f"--stream-resume-tmp shape mismatch: {tuple(refined_out.shape)} vs {(h, w)}")
            storage_dtype = np.dtype(refined_out.dtype)
            label_dtype = storage_dtype.newbyteorder("=")
            print(
                f"[INFO] Resuming streaming MedSAM from checkpoint={tmp_flat} "
                f"skip_completed_foreground_tiles={resume_tiles}"
                + (
                    f" resume_after_grid=({resume_grid_y},{resume_grid_x})"
                    if resume_by_grid
                    else ""
                ),
                flush=True,
            )
        else:
            tmp_flat = outdir / f".tmp_refined_stream_{os.getpid()}_{int(time.time())}.tif"
            if tmp_flat.exists():
                tmp_flat.unlink()
            refined_out = create_tiff_memmap(
                tmp_flat,
                shape=(h, w),
                dtype=storage_dtype,
                mpp_x=args.source_mpp_x,
                mpp_y=args.source_mpp_y,
            )

        input_grown_pixels = 0
        if uncertainty_reader is not None:
            constraint_paths = [outdir / f".tmp_protected_{os.getpid()}.tif", outdir / f".tmp_editable_{os.getpid()}.tif"]
            protected_out = create_tiff_memmap(constraint_paths[0], shape=(h, w), dtype=label_dtype, mpp_x=args.source_mpp_x, mpp_y=args.source_mpp_y)
            editable_out = create_tiff_memmap(constraint_paths[1], shape=(h, w), dtype=np.dtype("uint8"), mpp_x=args.source_mpp_x, mpp_y=args.source_mpp_y)
            protected_out[:] = 0
            editable_out[:] = 0
        raw_pixels = 0
        grandqc_support_pixels = 0
        excluded_background_pixels = 0
        print(
            f"[INFO] Large-image streaming MedSAM enabled: shape={shape}, tile_size={tile_size}, "
            f"overlap={overlap}, image_reader={image_reader.kind}, grown_reader={grown_reader.kind}, device={med_cfg.device}",
            flush=True,
        )
        for y0 in range(0, h, block_rows):
            y1 = min(h, y0 + block_rows)
            block = ensure_2d(grown_reader.read(y0, y1, 0, w), "grown mask block").astype(label_dtype, copy=False)
            tissue_block = tissue_reader.read(y0, y1, 0, w)
            if uncertainty_reader is not None:
                uncertainty_block = validate_uncertainty(
                    uncertainty_reader.read(y0, y1, 0, w), block.shape
                )
                tissue_block = tissue_block & (
                    uncertainty_block != GRANDQC_KODAMA_OUTLIER_CODE
                )
                del uncertainty_block
            input_grown_pixels += int(np.count_nonzero(block))
            grandqc_support_pixels += int(np.count_nonzero(tissue_block))
            excluded_background_pixels += int(np.count_nonzero((block > 0) & ~tissue_block))
            block = np.where(tissue_block, block, 0).astype(label_dtype, copy=False)
            if resume_tmp is None:
                refined_out[y0:y1, :] = block
            raw_pixels += int(np.count_nonzero(block))
            if y0 == 0 or y1 == h or ((y0 // block_rows) % 50 == 0):
                action = "Initialized" if resume_tmp is None else "Checked"
                print(f"[INFO] {action} refined output rows {y0}:{y1}", flush=True)
            del block, tissue_block
        refined_out.flush()

        ys = tile_starts(h, tile_size, overlap)
        xs = tile_starts(w, tile_size, overlap)
        total_tiles = len(ys) * len(xs)
        precompetition_calibration = None
        precompetition_calibration_meta: Dict[str, object] = {"enabled": False}
        if args.pre_boundary_competition:
            calibration_step = max(
                int(args.pre_boundary_downsample),
                int(args.pre_boundary_calibration_downsample),
            )
            print(
                "[INFO] Fitting one slide-level Wald Lab/OD calibration "
                f"at step={calibration_step}",
                flush=True,
            )
            calibration_image = read_stride_tiled(
                image_reader, calibration_step, tile_size=tile_size
            )
            calibration_labels = ensure_2d(
                read_stride_tiled(grown_reader, calibration_step, tile_size=tile_size),
                "Wald calibration labels",
            )
            calibration_tissue = tissue_reader.read_stride(calibration_step)
            calibration_trusted = (calibration_labels > 0) & calibration_tissue
            if uncertainty_reader is not None:
                calibration_trusted &= ensure_2d(
                    read_stride_tiled(
                        uncertainty_reader, calibration_step, tile_size=tile_size
                    ),
                    "Wald calibration uncertainty",
                ) == 0
            precompetition_calibration = fit_appearance_calibration(
                calibration_image,
                calibration_labels,
                calibration_tissue,
                trusted_mask=calibration_trusted,
                max_pixels=int(args.pre_boundary_calibration_pixels),
                random_seed=int(args.pre_boundary_calibration_seed),
            )
            precompetition_calibration_meta = dict(
                precompetition_calibration.get("metadata", {})
            )
            precompetition_calibration_meta.update(
                enabled=True,
                downsample=int(calibration_step),
                working_shape_yx=list(map(int, calibration_labels.shape)),
            )
            del (
                calibration_image,
                calibration_labels,
                calibration_tissue,
                calibration_trusted,
            )
            gc.collect()
        processed_tiles = resume_tiles if resume_by_grid else 0
        resumed_tiles_skipped = resume_tiles if resume_by_grid else 0
        skipped_no_seed = 0
        failed_tiles = 0
        cleanup_added_pixels = 0
        cleanup_removed_pixels = 0
        cleanup_relabel_pixels = 0
        medsam_added_pixels = 0
        medsam_removed_pixels = 0
        medsam_relabel_pixels = 0
        medsam_outside_grandqc_pixels = 0
        image_guided_editable_pixels = 0
        image_guided_changed_pixels = 0
        precompetition_tiles = 0
        precompetition_editable_pixels = 0
        precompetition_changed_pixels = 0
        precompetition_initial_energy = 0.0
        precompetition_final_energy = 0.0
        tile_runtime_sec = 0.0
        image_encoder_calls = 0
        image_embedding_cache_hits = 0
        tile_cfg = replace(
            med_cfg,
            save_debug=False,
            cluster_tile_size=tile_size,
            cluster_tile_overlap=overlap,
        )

        for yi, y0 in enumerate(ys, start=1):
            y1 = min(h, y0 + tile_size)
            for xi, x0 in enumerate(xs, start=1):
                x1 = min(w, x0 + tile_size)
                cy0, cy1, cx0, cx1 = commit_bounds(y0, y1, x0, x1, shape, overlap)
                if cy1 <= cy0 or cx1 <= cx0:
                    continue
                if resume_by_grid and (yi < resume_grid_y or (yi == resume_grid_y and xi <= resume_grid_x)):
                    if (yi == resume_grid_y and xi == resume_grid_x) or (xi == 1 and yi % 5 == 0):
                        print(
                            f"[INFO] Resume grid-skip through grid {yi}/{len(ys)}, {xi}/{len(xs)} "
                            f"(foreground_tiles={resume_tiles})",
                            flush=True,
                        )
                        write_stream_progress(
                            progress_path,
                            {
                                "sample_id": args.sample_id,
                                "mode": "large_image_streaming",
                                "status": "resume_grid_skip",
                                "processed_tiles": int(processed_tiles),
                                "resumed_tiles_skipped": int(resumed_tiles_skipped),
                                "grid_y": int(yi),
                                "grid_x": int(xi),
                                "grid_y_total": int(len(ys)),
                                "grid_x_total": int(len(xs)),
                                "total_grid": int(total_tiles),
                                "elapsed_sec": round(float(time.perf_counter() - start), 3),
                                "checkpoint": str(tmp_flat),
                            },
                        )
                    continue
                tile_seed = ensure_2d(seed_reader.read(y0, y1, x0, x1), "seed mask tile").astype(label_dtype, copy=False)
                tile_tissue = tissue_reader.read(y0, y1, x0, x1)
                tile_seed = np.where(tile_tissue, tile_seed, 0).astype(label_dtype, copy=False)
                tile_editable = tile_protected = None
                if uncertainty_reader is not None:
                    tile_uncertainty = validate_uncertainty(uncertainty_reader.read(y0, y1, x0, x1), tile_seed.shape)
                    tile_tissue = tile_tissue & (tile_uncertainty != GRANDQC_KODAMA_OUTLIER_CODE)
                    tile_seed = np.where(tile_tissue, tile_seed, 0).astype(label_dtype, copy=False)
                    tile_original = ensure_2d(grown_reader.read(y0, y1, x0, x1), "grown constraint tile").astype(label_dtype, copy=False)
                    tile_editable, tile_protected = refinement_constraints(tile_original, tile_tissue, tile_uncertainty, med_cfg.core_erosion_radius, args.internal_boundary_radius)
                    py0, py1, px0, px1 = cy0-y0, cy1-y0, cx0-x0, cx1-x0
                    protected_out[cy0:cy1, cx0:cx1] = tile_protected[py0:py1, px0:px1]
                    editable_out[cy0:cy1, cx0:cx1] = tile_editable[py0:py1, px0:px1]
                    tile_seed = tile_seed.copy()
                    tile_seed[(tile_uncertainty > 0) | tile_editable] = 0
                    del tile_uncertainty, tile_original
                if not np.any(tile_seed > 0):
                    skipped_no_seed += 1
                    del tile_seed, tile_tissue
                    continue
                tile_grown = ensure_2d(grown_reader.read(y0, y1, x0, x1), "grown mask tile").astype(label_dtype, copy=False)
                tile_grown = np.where(tile_tissue, tile_grown, 0).astype(label_dtype, copy=False)
                if not np.any(tile_grown > 0):
                    skipped_no_seed += 1
                    del tile_seed, tile_grown, tile_tissue
                    continue
                if processed_tiles < resume_tiles:
                    processed_tiles += 1
                    resumed_tiles_skipped += 1
                    if processed_tiles == resume_tiles or processed_tiles % 25 == 0:
                        print(
                            f"[INFO] Resume skip foreground tile {processed_tiles}/{resume_tiles} "
                            f"(grid {yi}/{len(ys)}, {xi}/{len(xs)})",
                            flush=True,
                        )
                    write_stream_progress(
                        progress_path,
                        {
                            "sample_id": args.sample_id,
                            "mode": "large_image_streaming",
                            "status": "resume_skip",
                            "processed_tiles": int(processed_tiles),
                            "resumed_tiles_skipped": int(resumed_tiles_skipped),
                            "grid_y": int(yi),
                            "grid_x": int(xi),
                            "grid_y_total": int(len(ys)),
                            "grid_x_total": int(len(xs)),
                            "total_grid": int(total_tiles),
                            "elapsed_sec": round(float(time.perf_counter() - start), 3),
                            "checkpoint": str(tmp_flat),
                        },
                    )
                    del tile_seed, tile_grown, tile_tissue
                    continue
                tile_image = image_reader.read(y0, y1, x0, x1)
                premedsam_tile, tile_precompetition_meta = pre_medsam_boundary_competition(
                    tile_image,
                    tile_grown,
                    tile_tissue,
                    args,
                    editable=tile_editable,
                    protected_labels=tile_protected,
                    appearance_calibration=precompetition_calibration,
                )
                precompetition_tiles += 1
                precompetition_editable_pixels += int(tile_precompetition_meta.get("editable_pixels", 0))
                precompetition_initial_energy += float(tile_precompetition_meta.get("initial_energy", 0.0))
                precompetition_final_energy += float(tile_precompetition_meta.get("final_energy", 0.0))
                tile_start = time.perf_counter()
                try:
                    _, _, runtime_sec, tile_metadata, artifacts = run_medsam_border_refine(
                        image=tile_image,
                        seed_labels=tile_seed,
                        baseline_tissue_mask=premedsam_tile > 0,
                        config=tile_cfg,
                        baseline_label_map=premedsam_tile,
                        allowed_support_mask=tile_tissue,
                        editable_mask=tile_editable,
                        protected_labels=tile_protected,
                    )
                except MedSAMUnavailableError:
                    raise
                except Exception:
                    failed_tiles += 1
                    raise
                tile_runtime_sec += float(runtime_sec)
                image_encoder_calls += int(tile_metadata.get("image_encoder_calls", 0))
                image_embedding_cache_hits += int(tile_metadata.get("image_embedding_cache_hits", 0))
                medsam_tile = np.asarray(artifacts.get("label_map", premedsam_tile), dtype=label_dtype).copy()
                ly0, ly1 = cy0 - y0, cy1 - y0
                lx0, lx1 = cx0 - x0, cx1 - x0
                tile_tissue_commit = tile_tissue[ly0:ly1, lx0:lx1]
                medsam_outside_grandqc_pixels += int(
                    np.count_nonzero((medsam_tile[ly0:ly1, lx0:lx1] > 0) & ~tile_tissue_commit)
                )
                medsam_tile[~tile_tissue] = 0
                refined_tile = medsam_tile
                if args.image_guided_internal_refine:
                    refined_tile, boundary_meta = refine_internal_cluster_boundaries(
                        tile_image, refined_tile, int(args.internal_boundary_radius),
                        editable_mask=tile_editable,
                        protected_mask=(tile_protected > 0) if tile_protected is not None else None,
                        gradient_space=args.internal_gradient_space,
                        gradient_sigma_px=args.internal_gradient_sigma_px,
                        watershed_compactness=args.internal_watershed_compactness,
                    )
                    image_guided_editable_pixels += int(boundary_meta["editable_pixels"])
                    image_guided_changed_pixels += int(boundary_meta["changed_pixels"])
                refined_tile = enforce_refinement_constraints(refined_tile, tile_grown, tile_tissue, tile_editable, tile_protected)

                raw_commit = tile_grown[ly0:ly1, lx0:lx1]
                precompetition_changed_pixels += int(
                    np.count_nonzero(premedsam_tile[ly0:ly1, lx0:lx1] != raw_commit)
                )
                medsam_commit = medsam_tile[ly0:ly1, lx0:lx1]
                new_commit = refined_tile[ly0:ly1, lx0:lx1]
                raw_fg = raw_commit > 0
                medsam_fg = medsam_commit > 0
                new_fg = new_commit > 0
                medsam_removed_pixels += int(np.count_nonzero(raw_fg & ~medsam_fg))
                medsam_added_pixels += int(np.count_nonzero(medsam_fg & ~raw_fg))
                medsam_relabel_pixels += int(np.count_nonzero((raw_commit != medsam_commit) & raw_fg & medsam_fg))
                cleanup_removed_pixels += int(np.count_nonzero(raw_fg & ~new_fg))
                cleanup_added_pixels += int(np.count_nonzero(new_fg & ~raw_fg))
                cleanup_relabel_pixels += int(np.count_nonzero((raw_commit != new_commit) & raw_fg & new_fg))
                refined_out[cy0:cy1, cx0:cx1] = new_commit
                processed_tiles += 1
                if processed_tiles == 1 or processed_tiles % 10 == 0:
                    elapsed = time.perf_counter() - start
                    print(
                        f"[INFO] Streaming MedSAM tile {processed_tiles} processed "
                        f"(grid {yi}/{len(ys)}, {xi}/{len(xs)}; total_grid={total_tiles}; elapsed_sec={elapsed:.1f})",
                        flush=True,
                    )
                write_stream_progress(
                    progress_path,
                    {
                        "sample_id": args.sample_id,
                        "mode": "large_image_streaming",
                        "status": "running",
                        "processed_tiles": int(processed_tiles),
                        "resumed_tiles_skipped": int(resumed_tiles_skipped),
                        "tiles_skipped_no_seed": int(skipped_no_seed),
                        "grid_y": int(yi),
                        "grid_x": int(xi),
                        "grid_y_total": int(len(ys)),
                        "grid_x_total": int(len(xs)),
                        "total_grid": int(total_tiles),
                        "elapsed_sec": round(float(time.perf_counter() - start), 3),
                        "checkpoint": str(tmp_flat),
                    },
                )
                del tile_image, tile_seed, tile_grown, tile_tissue, tile_tissue_commit, premedsam_tile, medsam_tile, refined_tile, raw_commit, medsam_commit, new_commit
                gc.collect()

        # Enforce the tissue contract over the complete checkpoint, including
        # checkpoints created by older interrupted runs.
        final_background_clamp_pixels = 0
        final_outside_grandqc_pixels = 0
        for y0 in range(0, h, block_rows):
            y1 = min(h, y0 + block_rows)
            final_block = np.asarray(refined_out[y0:y1, :]).copy()
            tissue_block = tissue_reader.read(y0, y1, 0, w)
            final_background_clamp_pixels += int(np.count_nonzero((final_block > 0) & ~tissue_block))
            final_block[~tissue_block] = 0
            final_outside_grandqc_pixels += int(np.count_nonzero((final_block > 0) & ~tissue_block))
            refined_out[y0:y1, :] = final_block
            del final_block, tissue_block
        refined_out.flush()
        if final_outside_grandqc_pixels:
            raise RuntimeError(
                f"GrandQC empty-area contract violated: {final_outside_grandqc_pixels} final pixels remain outside support"
            )
        appearance_meta: Dict[str, object] = {"enabled": bool(args.appearance_refine), "applied": False, "changed_pixels": 0}
        if args.appearance_refine:
            appearance_step = max(1, int(args.appearance_downsample))
            print(f"[INFO] Running automatic appearance refinement at step={appearance_step}", flush=True)
            appearance_image = read_stride_tiled(image_reader, appearance_step, tile_size=tile_size)
            appearance_labels = ensure_2d(
                array_stride_tiled(refined_out, appearance_step, tile_size=tile_size),
                "appearance label preview",
            )
            appearance_tissue = tissue_reader.read_stride(appearance_step)
            appearance_trusted = None
            if uncertainty_reader is not None:
                appearance_source = ensure_2d(read_stride_tiled(seed_reader, appearance_step, tile_size=tile_size), "appearance original cluster support")
                appearance_trusted = (appearance_source > 0) & (appearance_source == appearance_labels)
                appearance_trusted &= read_stride_tiled(uncertainty_reader, appearance_step, tile_size=tile_size) == 0
                del appearance_source
            appearance_result, appearance_meta = appearance_refine_working_scale(
                appearance_image, appearance_labels, appearance_tissue, args, appearance_step,
                editable=array_stride_tiled(editable_out, appearance_step, tile_size=tile_size).astype(bool) if editable_out is not None else None,
                protected=(array_stride_tiled(protected_out, appearance_step, tile_size=tile_size) > 0) if protected_out is not None else None,
                core_support=appearance_trusted,
            )
            appearance_change = appearance_result != appearance_labels
            changed_full = 0
            x_indices = np.minimum(np.arange(w) // appearance_step, appearance_result.shape[1] - 1)
            for y0 in range(0, h, block_rows):
                y1 = min(h, y0 + block_rows)
                y_indices = np.minimum(np.arange(y0, y1) // appearance_step, appearance_result.shape[0] - 1)
                change_block = appearance_change[y_indices[:, None], x_indices[None, :]]
                if not np.any(change_block):
                    continue
                result_block = appearance_result[y_indices[:, None], x_indices[None, :]]
                tissue_block = tissue_reader.read(y0, y1, 0, w)
                if uncertainty_reader is not None:
                    uncertainty_block = validate_uncertainty(
                        uncertainty_reader.read(y0, y1, 0, w), tissue_block.shape
                    )
                    tissue_block = tissue_block & (
                        uncertainty_block != GRANDQC_KODAMA_OUTLIER_CODE
                    )
                    del uncertainty_block
                change_block &= tissue_block
                if editable_out is not None:
                    change_block &= (editable_out[y0:y1] > 0) & (protected_out[y0:y1] == 0)
                current_block = np.asarray(refined_out[y0:y1, :]).copy()
                current_block[change_block] = result_block[change_block].astype(current_block.dtype, copy=False)
                refined_out[y0:y1, :] = current_block
                changed_full += int(np.count_nonzero(change_block))
                del result_block, tissue_block, current_block, change_block
            refined_out.flush()
            appearance_meta["changed_pixels_full_resolution"] = int(changed_full)
            del appearance_image, appearance_labels, appearance_tissue, appearance_result, appearance_change, appearance_trusted
            gc.collect()
        if editable_out is not None:
            for y0 in range(0, h, block_rows):
                y1 = min(h, y0 + block_rows)
                original = ensure_2d(grown_reader.read(y0, y1, 0, w), "constraint original")
                tissue_block = tissue_reader.read(y0, y1, 0, w)
                uncertainty_block = validate_uncertainty(
                    uncertainty_reader.read(y0, y1, 0, w), tissue_block.shape
                )
                tissue_block = tissue_block & (
                    uncertainty_block != GRANDQC_KODAMA_OUTLIER_CODE
                )
                refined_out[y0:y1] = enforce_refinement_constraints(
                    refined_out[y0:y1], original, tissue_block,
                    editable_out[y0:y1], protected_out[y0:y1]
                )
            refined_out.flush()
        provenance_meta = write_refinement_provenance_outputs(args, refined_out, grown_reader, tissue_reader, uncertainty_reader, protected_out, seed_reader)
        print("[INFO] Streaming MedSAM loop complete; counting final refined pixels", flush=True)
        final_pixels = count_nonzero_2d_blocks(refined_out, block_rows=block_rows)
        diag_step = preview_step_for_shape(shape, max_side=2048)
        print(f"[INFO] Building tiled diagnostic previews with step={diag_step}", flush=True)
        image_preview = read_stride_tiled(image_reader, diag_step, tile_size=tile_size)
        seed_preview = ensure_2d(read_stride_tiled(seed_reader, diag_step, tile_size=tile_size), "seed mask preview")
        raw_preview = ensure_2d(read_stride_tiled(grown_reader, diag_step, tile_size=tile_size), "grown mask preview")
        refined_preview = array_stride_tiled(refined_out, diag_step, tile_size=tile_size)
        tissue_preview = tissue_reader.read_stride(diag_step)
        precompetition_preview = raw_preview.copy()
        precompetition_preview_meta = {"enabled": bool(args.pre_boundary_competition), "diagnostic_only": True}
        if args.pre_boundary_competition:
            preview_editable = array_stride_tiled(editable_out, diag_step, tile_size=tile_size).astype(bool) if editable_out is not None else None
            preview_protected = array_stride_tiled(protected_out, diag_step, tile_size=tile_size) if protected_out is not None else None
            precompetition_preview, precompetition_preview_meta = annealed_wand_boundary_competition(
                image_preview,
                raw_preview,
                tissue_preview,
                editable_mask=preview_editable,
                protected_labels=preview_protected,
                boundary_radius=max(1, int(np.ceil(args.pre_boundary_radius / diag_step))),
                iterations=int(args.pre_boundary_iterations),
                initial_temperature=float(args.pre_boundary_initial_temperature),
                final_temperature=float(args.pre_boundary_final_temperature),
                data_weight=float(args.pre_boundary_data_weight),
                smoothness_weight=float(args.pre_boundary_smoothness_weight),
                edge_beta=float(args.pre_boundary_edge_beta),
                connectivity=int(args.pre_boundary_connectivity),
            )
            precompetition_preview_meta["diagnostic_only"] = True
            precompetition_preview_meta["native_preview_step"] = int(diag_step)
        raw_vs_final_change = to_uint8_rgb(image_preview)
        raw_only = (raw_preview > 0) & ~(refined_preview > 0)
        final_only = (refined_preview > 0) & ~(raw_preview > 0)
        relabeled = (raw_preview != refined_preview) & (raw_preview > 0) & (refined_preview > 0)
        raw_vs_final_change[raw_only] = ((0.65 * raw_vs_final_change[raw_only]) + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        raw_vs_final_change[final_only] = ((0.65 * raw_vs_final_change[final_only]) + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
        raw_vs_final_change[relabeled] = ((0.55 * raw_vs_final_change[relabeled]) + 0.45 * np.array([255, 255, 0])).astype(np.uint8)

        save_png(outdir / f"{prefix}_medsam_seed_labels.png", overlay_labels(image_preview, seed_preview))
        save_png(outdir / f"{prefix}_medsam_raw_labels.png", overlay_labels(image_preview, raw_preview))
        save_png(outdir / f"{prefix}_medsam_precompetition_labels.png", overlay_labels(image_preview, precompetition_preview))
        save_png(outdir / f"{prefix}_medsam_refined_labels.png", overlay_labels(image_preview, refined_preview))
        save_png(outdir / f"{prefix}_medsam_boundary_compare.png", boundary_compare(image_preview, raw_preview > 0, refined_preview > 0))
        save_png(outdir / f"{prefix}_medsam_streaming_change_map.png", raw_vs_final_change)
        grandqc_exclusion_preview = grandqc_empty_exclusion_map(
            image_preview, tissue_preview, raw_preview, refined_preview,
        )
        save_png(outdir / f"{prefix}_medsam_grandqc_empty_exclusion.png", grandqc_exclusion_preview)
        save_png(Path(args.preview), overlay_labels(image_preview, refined_preview, alpha=float(args.preview_alpha)))

        panel_path = outdir / f"{prefix}_medsam_raw_vs_final_panel.png"
        make_panel(
            [
                ("Original image", to_uint8_rgb(image_preview)),
                ("Step-15 labels", overlay_labels(image_preview, seed_preview)),
                ("Grown labels before MedSAM", overlay_labels(image_preview, raw_preview)),
                ("Pre-MedSAM competition (diagnostic scale)", overlay_labels(image_preview, precompetition_preview)),
                ("GrandQC empty: blue; removed labels: red; leakage: magenta", grandqc_exclusion_preview),
                ("Streaming MedSAM refined labels", overlay_labels(image_preview, refined_preview)),
                ("What refinements changed", raw_vs_final_change),
            ],
            panel_path,
            columns=3,
        )

        random_fullres_qc = None
        if int(args.medsam_qc_crop_size) > 0:
            y0, y1, x0, x1 = choose_random_tissue_crop(
                raw_preview=raw_preview,
                refined_preview=refined_preview,
                step=diag_step,
                full_shape=shape,
                crop_size=int(args.medsam_qc_crop_size),
                seed=int(args.medsam_qc_random_seed),
            )
            qc_path = outdir / f"{prefix}_medsam_random_fullres_qc.png"
            print(
                f"[INFO] Writing native-resolution random MedSAM QC crop: "
                f"y={y0}:{y1}, x={x0}:{x1}, path={qc_path}",
                flush=True,
            )
            image_crop = image_reader.read(y0, y1, x0, x1)
            raw_crop = ensure_2d(grown_reader.read(y0, y1, x0, x1), "grown mask MedSAM QC crop")
            refined_crop = np.asarray(refined_out[y0:y1, x0:x1])
            random_fullres_qc = save_fullres_qc_crop(
                image_crop=image_crop,
                raw_crop=raw_crop,
                refined_crop=refined_crop,
                out_path=qc_path,
                crop_bounds=(y0, y1, x0, x1),
            )
            del image_crop, raw_crop, refined_crop

        summary = {
            "sample_id": args.sample_id,
            "mode": "large_image_streaming",
            "image_shape": [int(h), int(w)],
            "medsam_device": str(med_cfg.device),
            "source_mpp_x": float(args.source_mpp_x),
            "source_mpp_y": float(args.source_mpp_y),
            "medsam_runtime_sec": round(float(time.perf_counter() - start), 3),
            "medsam_tile_inference_sec": round(float(tile_runtime_sec), 3),
            "tiles_total": int(total_tiles),
            "tiles_processed": int(processed_tiles),
            "tiles_resumed_skipped": int(resumed_tiles_skipped),
            "tiles_skipped_no_seed": int(skipped_no_seed),
            "tiles_failed": int(failed_tiles),
            "tile_size": int(tile_size),
            "tile_overlap": int(overlap),
            "input_grown_pixels": int(input_grown_pixels),
            "raw_pixels": int(raw_pixels),
            "final_pixels": int(final_pixels),
            "grandqc_support_pixels": int(grandqc_support_pixels),
            "grandqc_empty_pixels": int((h * w) - grandqc_support_pixels),
            "excluded_background_pixels": int(excluded_background_pixels),
            "medsam_pixels_outside_grandqc_before_clamp": int(medsam_outside_grandqc_pixels),
            "final_background_clamp_pixels": int(final_background_clamp_pixels),
            "final_pixels_outside_grandqc": int(final_outside_grandqc_pixels),
            "tissue_mask_source_shape_yx": list(map(int, tissue_reader.source_shape)),
            "background_exclusion_contract": "grandqc_clean_tissue_intersection_roi",
            "grandqc_empty_area_role": "hard_negative_support_during_medsam",
            "cleanup_removed_pixels": int(cleanup_removed_pixels),
            "cleanup_added_pixels": int(cleanup_added_pixels),
            "cleanup_relabel_pixels": int(cleanup_relabel_pixels),
            "medsam_removed_pixels": int(medsam_removed_pixels),
            "medsam_added_pixels": int(medsam_added_pixels),
            "medsam_relabel_pixels": int(medsam_relabel_pixels),
            "pre_medsam_boundary_competition": {
                "enabled": bool(args.pre_boundary_competition),
                "method": "deterministic_mean_field_annealed_wand_frontier",
                "tiles_evaluated": int(precompetition_tiles),
                "editable_pixels_summed_over_tiles": int(precompetition_editable_pixels),
                "changed_pixels_in_nonoverlapping_commits": int(precompetition_changed_pixels),
                "initial_energy_summed_over_tiles": float(precompetition_initial_energy),
                "final_energy_summed_over_tiles": float(precompetition_final_energy),
                "energy_scope": "sum_over_overlapping_processing_tiles_not_a_global_energy",
                "appearance_calibration": precompetition_calibration_meta,
                "boundary_radius_px": int(args.pre_boundary_radius),
                "working_downsample": int(args.pre_boundary_downsample),
                "iterations": int(args.pre_boundary_iterations),
                "initial_temperature": float(args.pre_boundary_initial_temperature),
                "final_temperature": float(args.pre_boundary_final_temperature),
                "diagnostic_preview": precompetition_preview_meta,
                "foreground_footprint_invariant": True,
                "grandqc_support_is_hard_constraint": True,
            },
            "image_guided_internal_refine": bool(args.image_guided_internal_refine),
            "internal_boundary_radius_px": int(args.internal_boundary_radius),
            "internal_gradient_space": str(args.internal_gradient_space),
            "internal_gradient_sigma_px": float(args.internal_gradient_sigma_px),
            "internal_watershed_compactness": float(args.internal_watershed_compactness),
            "image_guided_editable_pixels": int(image_guided_editable_pixels),
            "image_guided_changed_pixels": int(image_guided_changed_pixels),
            "appearance_refinement": appearance_meta,
            "refinement_provenance": provenance_meta,
            "image_encoder_calls": image_encoder_calls,
            "image_embedding_cache_hits": image_embedding_cache_hits,
            "diagnostic_preview_step": int(diag_step),
            "image_reader": image_reader.kind,
            "grown_reader": grown_reader.kind,
            "seed_reader": seed_reader.kind,
            "model_provenance": medsam_model_provenance(med_cfg),
        }
        if random_fullres_qc is not None:
            summary["random_fullres_qc"] = random_fullres_qc
        refined_out.flush()
        del refined_out
        gc.collect()
        try:
            print(f"[INFO] Pyramidizing refined checkpoint to {args.out}", flush=True)
            pyramidize_or_fallback(tmp_flat, args.out, args)
        finally:
            if not args.keep_tmp and resume_tmp is None:
                try:
                    tmp_flat.unlink()
                except Exception:
                    pass

        provenance_meta = finalize_refinement_provenance_outputs(args, provenance_meta)
        summary["refinement_provenance"] = provenance_meta
        (outdir / f"{prefix}_medsam_summary.json").write_text(json.dumps(summary, indent=2))
        write_stream_progress(progress_path, {**summary, "status": "complete", "checkpoint": str(tmp_flat)})
        print(json.dumps(summary, indent=2))
        print(f"[OK] wrote streaming refined mask: {args.out}")
        print(f"[OK] wrote streaming refinement panel: {panel_path}")
    finally:
        image_reader.close()
        seed_reader.close()
        grown_reader.close()
        if tissue_reader is not None:
            tissue_reader.close()
        if uncertainty_reader is not None:
            uncertainty_reader.close()
        for path in constraint_paths:
            if path.exists():
                path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine the step-16 grown tissue mask with MedSAM while preserving the protected core and seeded labels."
    )
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--image", required=True, help="ROI image TIFF used by the pipeline")
    ap.add_argument("--seed-mask", required=True, help="Step-15 cluster mask TIFF")
    ap.add_argument("--grown-mask", required=True, help="Step-16 grown tissue mask TIFF")
    ap.add_argument("--tissue-mask", required=True, help="GrandQC clean-tissue intersection with the optional ROI")
    ap.add_argument("--clustering-uncertainty", default="", help="Aligned input categorical uncertainty; 0=accepted,1..249=original reasons")
    ap.add_argument("--uncertainty-out", default="", help="Default: output-label-path.uncertainty.tif; original reasons survive refinement")
    ap.add_argument("--provenance-out", default="", help="Default: output-label-path.provenance.tif")
    ap.add_argument("--out", required=True, help="Output refined OME-TIFF path")
    ap.add_argument("--resolution-json", default="",
                    help="Pipeline shift/resolution JSON containing authoritative source_mpp metadata.")
    ap.add_argument("--default-mpp", type=float, default=0.0,
                    help="Fallback MPP used only when --resolution-json has no physical size.")
    ap.add_argument("--preview", required=True, help="Output preview PNG path")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0)
    ap.add_argument("--preview-alpha", type=float, default=0.45)
    ap.add_argument("--pyr-compression", default="LZW")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--downsample", default="GAUSSIAN")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--large-image-mode", choices=("auto", "full", "stream"), default="auto",
                    help="Use streamed tiled MedSAM for WSI-scale inputs; 'auto' switches above --large-image-max-pixels.")
    ap.add_argument("--large-image-max-pixels", type=int, default=50_000_000,
                    help="Pixel threshold for automatic streamed MedSAM mode.")
    ap.add_argument("--stream-block-rows", type=int, default=512,
                    help="Rows per block when initializing streamed large-image outputs.")
    ap.add_argument("--stream-resume-tmp", default="",
                    help="Existing flat checkpoint TIFF from an interrupted streamed MedSAM run.")
    ap.add_argument("--stream-resume-tiles", type=int, default=0,
                    help="Number of previously completed foreground tiles to skip when resuming from --stream-resume-tmp.")
    ap.add_argument("--stream-resume-grid-y", type=int, default=0,
                    help="1-based grid row of the last completed streamed tile; skips earlier grid cells without re-reading them.")
    ap.add_argument("--stream-resume-grid-x", type=int, default=0,
                    help="1-based grid column of the last completed streamed tile; used with --stream-resume-grid-y.")
    ap.add_argument("--stream-progress-json", default="",
                    help="Optional progress JSON path written during streamed MedSAM.")
    ap.add_argument("--medsam-qc-crop-size", type=int, default=1024,
                    help="Native-resolution square crop size for random tissue MedSAM QC preview; set 0 to disable.")
    ap.add_argument("--medsam-qc-random-seed", type=int, default=1729,
                    help="Random seed used to choose the tissue location for the MedSAM QC crop.")

    ap.add_argument("--medsam-checkpoint", default=str(DEFAULT_MEDSAM_CHECKPOINT))
    ap.add_argument("--medsam-device", default="cuda")
    ap.add_argument("--medsam-bbox-margin", type=int, default=144)
    ap.add_argument("--medsam-component-min-area", type=int, default=200)
    ap.add_argument("--medsam-component-merge-distance", type=int, default=24)
    ap.add_argument("--medsam-seed-dilation-radius", type=int, default=8)
    ap.add_argument("--medsam-core-erosion-radius", type=int, default=43)
    ap.add_argument("--medsam-outer-dilation-radius", type=int, default=53)
    ap.add_argument("--medsam-min-object-size", type=int, default=5000)
    ap.add_argument("--medsam-smooth-radius", type=int, default=5)
    ap.add_argument("--medsam-cluster-tile-size", type=int, default=4096)
    ap.add_argument("--medsam-cluster-tile-overlap", type=int, default=512)
    ap.add_argument("--medsam-force-core-preservation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--medsam-save-debug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pre-boundary-competition", action=argparse.BooleanOptionalAction, default=True,
                    help="Before MedSAM, let neighbouring labels compete in a bounded uncertainty band by deterministic mean-field annealing.")
    ap.add_argument("--pre-boundary-radius", type=int, default=64,
                    help="Maximum native-pixel boundary error band used by pre-MedSAM competition.")
    ap.add_argument("--pre-boundary-downsample", type=int, default=4)
    ap.add_argument("--pre-boundary-calibration-downsample", type=int, default=16,
                    help="Bounded slide-level sampling scale used once for Wald Lab/OD calibration.")
    ap.add_argument("--pre-boundary-calibration-pixels", type=int, default=250000,
                    help="Maximum trusted slide-level pixels used to fit Wald appearance calibration.")
    ap.add_argument("--pre-boundary-calibration-seed", type=int, default=1729)
    ap.add_argument("--pre-boundary-iterations", type=int, default=16)
    ap.add_argument("--pre-boundary-initial-temperature", type=float, default=2.0)
    ap.add_argument("--pre-boundary-final-temperature", type=float, default=0.05)
    ap.add_argument("--pre-boundary-data-weight", type=float, default=1.0)
    ap.add_argument("--pre-boundary-smoothness-weight", type=float, default=0.3)
    ap.add_argument("--pre-boundary-edge-beta", type=float, default=0.7)
    ap.add_argument("--pre-boundary-connectivity", type=int, choices=[4, 8], default=8,
                    help="Potts/frontier neighbourhood; 8 reduces grid-aligned staircase boundaries with distance-weighted diagonals.")
    ap.add_argument("--image-guided-internal-refine", action=argparse.BooleanOptionalAction, default=True,
                    help="Refine inter-cluster boundaries with a full-resolution marker-controlled watershed.")
    ap.add_argument("--internal-boundary-radius", type=int, default=64,
                    help="Maximum pixel distance that image-guided refinement may move an inter-cluster boundary.")
    ap.add_argument("--internal-gradient-space", choices=("luminance", "lab", "od", "lab_od"), default="luminance",
                    help="H&E feature space used to construct the internal-boundary watershed gradient.")
    ap.add_argument("--internal-gradient-sigma-px", type=float, default=0.0,
                    help="Native-pixel Gaussian scale applied before the watershed gradient; zero reproduces the historical microtexture-sensitive gradient.")
    ap.add_argument("--internal-watershed-compactness", type=float, default=0.0,
                    help="Nonnegative watershed compactness penalty; zero is purely image-gradient driven.")
    ap.add_argument("--appearance-refine", action=argparse.BooleanOptionalAction, default=True,
                    help="Correct large coherent two-domain errors using H&E appearance and pipeline-derived label cores only.")
    ap.add_argument("--appearance-downsample", type=int, default=8)
    ap.add_argument("--appearance-clusters", type=int, default=24)
    ap.add_argument("--appearance-core-erosion-px", type=int, default=32)
    ap.add_argument("--appearance-smooth-sigma-px", type=float, default=24.0)
    ap.add_argument("--appearance-min-region-area-px", type=int, default=640000)
    ap.add_argument("--appearance-sample-pixels", type=int, default=250000)
    ap.add_argument("--appearance-random-seed", type=int, default=1)
    ap.add_argument("--appearance-multiclass", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--appearance-min-vote-fraction", type=float, default=0.0)
    ap.add_argument("--appearance-min-vote-margin", type=float, default=0.0)
    for option in ("medsam-core-erosion-um", "medsam-outer-dilation-um", "medsam-smooth-um", "pre-boundary-radius-um", "internal-boundary-radius-um", "internal-gradient-sigma-um", "appearance-core-erosion-um", "appearance-smooth-sigma-um", "appearance-min-region-area-um2"):
        ap.add_argument("--" + option, type=float, default=None)
    args = ap.parse_args()

    args.source_mpp_x, args.source_mpp_y = read_mpp_json(args.resolution_json, args.default_mpp)
    apply_physical_parameters(args)

    med_cfg = make_medsam_config(args)
    image_shape = tiff_spatial_shape(args.image, "image")
    seed_shape = tiff_spatial_shape(args.seed_mask, "seed mask")
    grown_shape = tiff_spatial_shape(args.grown_mask, "grown mask")
    if seed_shape != grown_shape:
        raise ValueError(f"Step-15 and step-16 masks must share the same shape: {seed_shape} vs {grown_shape}")
    if image_shape != seed_shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_shape} vs {seed_shape}")
    args.output_spatial_shape = seed_shape
    print(
        f"[INFO] Output physical resolution: mpp_x={args.source_mpp_x:.9g}, "
        f"mpp_y={args.source_mpp_y:.9g}"
    )
    pixel_count = int(seed_shape[0]) * int(seed_shape[1])
    if args.large_image_mode == "stream" or (
        args.large_image_mode == "auto" and pixel_count > int(args.large_image_max_pixels)
    ):
        run_large_image_streaming_medsam(args, med_cfg)
        return

    image_rgb = load_tiff_memmap(args.image, "image")
    seed_labels = ensure_2d(load_tiff_memmap(args.seed_mask, "seed mask"), "seed mask")
    grown_labels = ensure_2d(load_tiff_memmap(args.grown_mask, "grown mask"), "grown mask")
    label_dtype = seed_labels.dtype
    if np.issubdtype(seed_labels.dtype, np.integer):
        min_label = int(np.min(seed_labels, initial=0))
        max_label = int(np.max(seed_labels, initial=0))
        if min_label >= 0 and max_label <= np.iinfo(np.uint16).max:
            label_dtype = np.uint16
    seed_labels = seed_labels.astype(label_dtype, copy=False)
    grown_labels = grown_labels.astype(label_dtype, copy=False)

    if grown_labels.shape != seed_labels.shape:
        raise ValueError(f"Step-15 and step-16 masks must share the same shape: {seed_labels.shape} vs {grown_labels.shape}")

    if image_rgb.ndim == 3 and image_rgb.shape[:2] != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape[:2]} vs {seed_labels.shape}")
    if image_rgb.ndim == 2 and image_rgb.shape != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape} vs {seed_labels.shape}")

    diag_step = preview_step_for_shape(seed_labels.shape, max_side=2048)
    grown_before_support_preview = np.asarray(preview_subsample(grown_labels, diag_step)).copy()

    tissue_reader = ScaledBinaryMaskReader(args.tissue_mask, seed_labels.shape)
    try:
        tissue_support = tissue_reader.read(0, seed_labels.shape[0], 0, seed_labels.shape[1])
        tissue_source_shape = tissue_reader.source_shape
    finally:
        tissue_reader.close()
    excluded_seed_background_pixels = int(np.count_nonzero((seed_labels > 0) & ~tissue_support))
    excluded_background_pixels = int(np.count_nonzero((grown_labels > 0) & ~tissue_support))
    grandqc_support_pixels = int(np.count_nonzero(tissue_support))
    grandqc_empty_pixels = int(tissue_support.size - grandqc_support_pixels)
    seed_labels = np.where(tissue_support, seed_labels, 0).astype(label_dtype, copy=False)
    grown_labels = np.where(tissue_support, grown_labels, 0).astype(label_dtype, copy=False)
    input_uncertainty = None
    editable = protected_labels = None
    medsam_seed = seed_labels
    if args.clustering_uncertainty:
        input_uncertainty = validate_uncertainty(load_tiff_memmap(args.clustering_uncertainty, "clustering uncertainty"), grown_labels.shape)
        tissue_support = tissue_support & (input_uncertainty != GRANDQC_KODAMA_OUTLIER_CODE)
        seed_labels = np.where(tissue_support, seed_labels, 0).astype(label_dtype, copy=False)
        grown_labels = np.where(tissue_support, grown_labels, 0).astype(label_dtype, copy=False)
        editable, protected_labels = refinement_constraints(grown_labels, tissue_support, input_uncertainty, med_cfg.core_erosion_radius, args.internal_boundary_radius)
        medsam_seed = seed_labels.copy()
        medsam_seed[(input_uncertainty > 0) | editable] = 0

    premedsam_labels, precompetition_meta = pre_medsam_boundary_competition(
        image_rgb,
        grown_labels,
        tissue_support,
        args,
        editable=editable,
        protected_labels=protected_labels,
    )

    try:
        refined_tissue, probability_map, runtime_sec, med_meta, artifacts = run_medsam_border_refine(
            image=image_rgb,
            seed_labels=medsam_seed,
            baseline_tissue_mask=premedsam_labels > 0,
            config=med_cfg,
            baseline_label_map=premedsam_labels,
            allowed_support_mask=tissue_support,
            editable_mask=editable,
            protected_labels=protected_labels,
        )
    except MedSAMUnavailableError as exc:
        raise RuntimeError(f"MedSAM unavailable: {exc}") from exc

    medsam_labels = np.asarray(artifacts.get("label_map", premedsam_labels), dtype=label_dtype).copy()
    medsam_outside_grandqc_pixels = int(np.count_nonzero((medsam_labels > 0) & ~tissue_support))
    medsam_labels[~tissue_support] = 0
    refined_labels = medsam_labels
    boundary_meta = {"enabled": 0, "editable_pixels": 0, "changed_pixels": 0, "radius_px": 0}
    if args.image_guided_internal_refine:
        refined_labels, boundary_meta = refine_internal_cluster_boundaries(
            image_rgb, refined_labels, int(args.internal_boundary_radius),
            editable_mask=editable,
            protected_mask=(protected_labels > 0) if protected_labels is not None else None,
            gradient_space=args.internal_gradient_space,
            gradient_sigma_px=args.internal_gradient_sigma_px,
            watershed_compactness=args.internal_watershed_compactness,
        )
    appearance_meta: Dict[str, object] = {"enabled": bool(args.appearance_refine), "applied": False, "changed_pixels": 0}
    if args.appearance_refine:
        appearance_step = max(1, int(args.appearance_downsample))
        work_image = preview_subsample(image_rgb, appearance_step)
        work_labels = preview_subsample(refined_labels, appearance_step)
        work_tissue = preview_subsample(tissue_support, appearance_step)
        work_result, appearance_meta = appearance_refine_working_scale(
            work_image, work_labels, work_tissue, args, appearance_step,
            editable=preview_subsample(editable, appearance_step) if editable is not None else None,
            protected=preview_subsample(protected_labels > 0, appearance_step) if protected_labels is not None else None,
            core_support=((preview_subsample(input_uncertainty, appearance_step) == 0) & (preview_subsample(seed_labels, appearance_step) > 0) & (preview_subsample(seed_labels, appearance_step) == work_labels)) if input_uncertainty is not None else None,
        )
        work_change = work_result != work_labels
        if np.any(work_change):
            target_size = (refined_labels.shape[1], refined_labels.shape[0])
            change_full = np.array(
                Image.fromarray(work_change).resize(target_size, Image.Resampling.NEAREST), dtype=bool,
            )
            result_full = np.asarray(
                Image.fromarray(work_result).resize(target_size, Image.Resampling.NEAREST), dtype=refined_labels.dtype,
            )
            change_full &= tissue_support
            refined_labels[change_full] = result_full[change_full]
            appearance_meta["changed_pixels_full_resolution"] = int(np.count_nonzero(change_full))
        else:
            appearance_meta["changed_pixels_full_resolution"] = 0
    refined_labels = enforce_refinement_constraints(refined_labels, grown_labels, tissue_support, editable, protected_labels)
    final_outside_grandqc_pixels = int(np.count_nonzero((refined_labels > 0) & ~tissue_support))
    if final_outside_grandqc_pixels:
        raise RuntimeError(
            f"GrandQC empty-area contract violated: {final_outside_grandqc_pixels} final pixels remain outside support"
        )
    raw_medsam_labels = np.asarray(artifacts.get("raw_medsam_label_map", medsam_labels), dtype=label_dtype)
    protected_core = np.asarray(artifacts.get("protected_core", np.zeros_like(grown_labels)), dtype=bool)
    editable_band = np.asarray(artifacts.get("editable_band", np.zeros_like(grown_labels)), dtype=bool)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = sample_prefix(args.sample_id, args.out)
    provenance_original = TiffWindowReader(args.grown_mask, "original grown mask")
    provenance_tissue = ScaledBinaryMaskReader(args.tissue_mask, grown_labels.shape)
    provenance_uncertainty = TiffWindowReader(args.clustering_uncertainty, "clustering uncertainty") if args.clustering_uncertainty else None
    provenance_source = TiffWindowReader(args.seed_mask, "original clustering support")
    try:
        provenance_meta = write_refinement_provenance_outputs(args, refined_labels, provenance_original, provenance_tissue, provenance_uncertainty, protected_labels, provenance_source)
    finally:
        provenance_original.close()
        provenance_tissue.close()
        provenance_source.close()
        if provenance_uncertainty is not None:
            provenance_uncertainty.close()
    debug_dir = outdir / f"{prefix}_medsam_debug"

    if med_cfg.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(med_meta)
        meta["sample_id"] = args.sample_id
        meta["raw_medsam_pixels"] = int((raw_medsam_labels > 0).sum())
        meta["cleaned_medsam_pixels"] = int((medsam_labels > 0).sum())
        meta["final_pixels"] = int((refined_labels > 0).sum())
        (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        for name, arr in artifacts.items():
            np.save(debug_dir / f"{name}.npy", np.asarray(arr))
        if probability_map is not None:
            np.save(debug_dir / "probability_map.npy", np.asarray(probability_map, dtype=np.float32))

    summary = {
        "sample_id": args.sample_id,
        "source_mpp_x": float(args.source_mpp_x),
        "source_mpp_y": float(args.source_mpp_y),
        "medsam_runtime_sec": round(float(runtime_sec), 3),
        "baseline_pixels": int((grown_labels > 0).sum()),
        "raw_medsam_pixels": int((raw_medsam_labels > 0).sum()),
        "cleaned_medsam_pixels": int((medsam_labels > 0).sum()),
        "final_pixels": int((refined_labels > 0).sum()),
        "excluded_background_pixels": int(excluded_background_pixels),
        "excluded_seed_background_pixels": int(excluded_seed_background_pixels),
        "grandqc_support_pixels": int(grandqc_support_pixels),
        "grandqc_empty_pixels": int(grandqc_empty_pixels),
        "medsam_pixels_outside_grandqc_before_clamp": int(medsam_outside_grandqc_pixels),
        "final_pixels_outside_grandqc": int(final_outside_grandqc_pixels),
        "tissue_mask_source_shape_yx": list(map(int, tissue_source_shape)),
        "background_exclusion_contract": "grandqc_clean_tissue_intersection_roi",
        "grandqc_empty_area_role": "hard_negative_support_during_medsam",
        "medsam_removed_pixels": int(((grown_labels > 0) & ~(medsam_labels > 0)).sum()),
        "medsam_added_pixels": int(((medsam_labels > 0) & ~(grown_labels > 0)).sum()),
        "medsam_relabel_pixels": int(((grown_labels != medsam_labels) & (grown_labels > 0) & (medsam_labels > 0)).sum()),
        "pre_medsam_boundary_competition": precompetition_meta,
        "image_guided_changed_pixels": int(boundary_meta["changed_pixels"]),
        "protected_core_pixels": int(protected_core.sum()),
        "editable_band_pixels": int(editable_band.sum()),
        "image_guided_internal_refine": bool(args.image_guided_internal_refine),
        "internal_boundary_radius_px": int(args.internal_boundary_radius),
        "internal_gradient_space": str(args.internal_gradient_space),
        "internal_gradient_sigma_px": float(args.internal_gradient_sigma_px),
        "internal_watershed_compactness": float(args.internal_watershed_compactness),
        "image_guided_editable_pixels": int(boundary_meta["editable_pixels"]),
        "appearance_refinement": appearance_meta,
        "refinement_provenance": provenance_meta,
        "image_encoder_calls": int(med_meta.get("image_encoder_calls", 0)),
        "image_embedding_cache_hits": int(med_meta.get("image_embedding_cache_hits", 0)),
        "image_embedding_cache": med_meta.get("image_embedding_cache", "unavailable"),
        "model_provenance": medsam_model_provenance(med_cfg),
    }
    summary["diagnostic_preview_step"] = int(diag_step)

    image_preview = preview_subsample(image_rgb, diag_step)
    seed_preview = preview_subsample(seed_labels, diag_step)
    grown_preview = preview_subsample(grown_labels, diag_step)
    premedsam_preview = preview_subsample(premedsam_labels, diag_step)
    raw_medsam_preview = preview_subsample(raw_medsam_labels, diag_step)
    medsam_preview = preview_subsample(medsam_labels, diag_step)
    refined_preview = preview_subsample(refined_labels, diag_step)
    protected_preview = preview_subsample(protected_core, diag_step)
    editable_preview = preview_subsample(editable_band, diag_step)
    tissue_preview = preview_subsample(tissue_support, diag_step)
    prob_preview = preview_subsample(probability_map, diag_step) if probability_map is not None else None

    raw_vs_final_change = to_uint8_rgb(image_preview)
    raw_only = (grown_preview > 0) & ~(refined_preview > 0)
    final_only = (refined_preview > 0) & ~(grown_preview > 0)
    relabeled = (medsam_preview != refined_preview) & (medsam_preview > 0) & (refined_preview > 0)
    raw_vs_final_change[raw_only] = ((0.65 * raw_vs_final_change[raw_only]) + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    raw_vs_final_change[final_only] = ((0.65 * raw_vs_final_change[final_only]) + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
    raw_vs_final_change[relabeled] = ((0.55 * raw_vs_final_change[relabeled]) + 0.45 * np.array([255, 255, 0])).astype(np.uint8)

    save_png(outdir / f"{prefix}_medsam_seed_labels.png", overlay_labels(image_preview, seed_preview))
    save_png(outdir / f"{prefix}_medsam_protected_core.png", overlay_mask(image_preview, protected_preview, (255, 200, 0)))
    save_png(outdir / f"{prefix}_medsam_editable_band.png", overlay_mask(image_preview, editable_preview, (180, 0, 255)))
    save_png(outdir / f"{prefix}_medsam_tissue_support.png", overlay_mask(image_preview, tissue_preview, (0, 180, 80)))
    save_png(outdir / f"{prefix}_medsam_precompetition_labels.png", overlay_labels(image_preview, premedsam_preview))
    grandqc_exclusion_preview = grandqc_empty_exclusion_map(
        image_preview, tissue_preview, grown_before_support_preview, refined_preview,
    )
    save_png(outdir / f"{prefix}_medsam_grandqc_empty_exclusion.png", grandqc_exclusion_preview)
    save_png(outdir / f"{prefix}_medsam_raw_labels.png", overlay_labels(image_preview, raw_medsam_preview))
    save_png(outdir / f"{prefix}_medsam_cleaned_labels.png", overlay_labels(image_preview, medsam_preview))
    save_png(outdir / f"{prefix}_medsam_refined_labels.png", overlay_labels(image_preview, refined_preview))
    if prob_preview is not None:
        save_png(outdir / f"{prefix}_medsam_probability_heatmap.png", heatmap(prob_preview))
    save_png(outdir / f"{prefix}_medsam_boundary_compare.png", boundary_compare(image_preview, grown_preview > 0, refined_preview > 0))

    panel_path = outdir / f"{prefix}_medsam_raw_vs_final_panel.png"
    make_panel(
        [
            ("Original image", to_uint8_rgb(image_preview)),
            ("Step-15 labels", overlay_labels(image_preview, seed_preview)),
            ("Protected core", overlay_mask(image_preview, protected_preview, (255, 200, 0))),
            ("Editable band", overlay_mask(image_preview, editable_preview, (180, 0, 255))),
            ("GrandQC tissue and ROI support", overlay_mask(image_preview, tissue_preview, (0, 180, 80))),
            ("Pre-MedSAM annealed boundary competition", overlay_labels(image_preview, premedsam_preview)),
            ("GrandQC empty: blue; removed labels: red; leakage: magenta", grandqc_exclusion_preview),
            ("Raw MedSAM predictions", overlay_labels(image_preview, raw_medsam_preview)),
            ("After MedSAM cleanup", overlay_labels(image_preview, medsam_preview)),
            ("Image-guided final", overlay_labels(image_preview, refined_preview)),
            ("What refinements changed", raw_vs_final_change),
        ],
        panel_path,
        columns=3,
    )

    if int(args.medsam_qc_crop_size) > 0:
        y0, y1, x0, x1 = choose_random_tissue_crop(
            raw_preview=medsam_preview,
            refined_preview=refined_preview,
            step=diag_step,
            full_shape=seed_labels.shape,
            crop_size=int(args.medsam_qc_crop_size),
            seed=int(args.medsam_qc_random_seed),
        )
        image_crop = np.asarray(image_rgb[y0:y1, x0:x1, ...]) if image_rgb.ndim == 3 else np.asarray(image_rgb[y0:y1, x0:x1])
        raw_crop = np.asarray(medsam_labels[y0:y1, x0:x1])
        refined_crop = np.asarray(refined_labels[y0:y1, x0:x1])
        summary["random_fullres_qc"] = save_fullres_qc_crop(
            image_crop=image_crop,
            raw_crop=raw_crop,
            refined_crop=refined_crop,
            out_path=outdir / f"{prefix}_medsam_random_fullres_qc.png",
            crop_bounds=(y0, y1, x0, x1),
        )
        del image_crop, raw_crop, refined_crop
    tmp_flat = outdir / f".tmp_refined_flat_{os.getpid()}_{int(time.time())}.tif"
    maxlab = int(refined_labels.max()) if refined_labels.size else 0
    flat = refined_labels.astype(label_storage_dtype(maxlab))
    imwrite(
        tmp_flat,
        flat,
        bigtiff=True,
        byteorder=flat.dtype.byteorder if flat.dtype.itemsize > 1 else None,
        **tiff_resolution_kwargs(args.source_mpp_x, args.source_mpp_y, "YX"),
    )
    try:
        pyramidize_or_fallback(tmp_flat, args.out, args)
    finally:
        try:
            tmp_flat.unlink()
        except Exception:
            pass

    provenance_meta = finalize_refinement_provenance_outputs(args, provenance_meta)
    summary["refinement_provenance"] = provenance_meta
    (outdir / f"{prefix}_medsam_summary.json").write_text(json.dumps(summary, indent=2))
    save_preview_png(
        image_path=args.image,
        grown=refined_labels,
        out_png=args.preview,
        factor=args.preview_factor,
        size_threshold_mb=args.preview_threshold_mb,
        alpha=args.preview_alpha,
        default_value=0,
    )

    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote refined mask: {args.out}")
    print(f"[OK] wrote refinement panel: {panel_path}")


if __name__ == "__main__":
    main()
