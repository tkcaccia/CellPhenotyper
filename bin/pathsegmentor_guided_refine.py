#!/usr/bin/env python3
"""Experimental KODAMA-preserving refinement using PathSegmentor evidence.

Only a narrow band around existing cluster boundaries may change. GrandQC is a
hard support mask, cluster identities and cluster count are retained, and the
unmodified KODAMA mask remains a separate pipeline output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import tifffile


def boundary_pixels(labels: np.ndarray, tissue: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        a = labels[tuple(left)]
        b = labels[tuple(right)]
        valid = tissue[tuple(left)] & tissue[tuple(right)] & (a > 0) & (b > 0) & (a != b)
        boundary[tuple(left)] |= valid
        boundary[tuple(right)] |= valid
    return boundary


def semantic_prototypes(
    labels: np.ndarray,
    probabilities: np.ndarray,
    tissue: np.ndarray,
    core_erosion_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    cluster_ids = np.asarray(sorted(int(value) for value in np.unique(labels[tissue]) if value > 0), dtype=np.int32)
    if cluster_ids.size < 2:
        raise ValueError("PathSegmentor-guided refinement requires at least two KODAMA clusters")
    prototypes = []
    for cluster_id in cluster_ids:
        support = (labels == cluster_id) & tissue
        erosion = max(0, int(core_erosion_px))
        core = (
            ndi.binary_erosion(support, iterations=erosion, border_value=0)
            if erosion > 0
            else support.copy()
        )
        if not np.any(core):
            core = support
        if not np.any(core):
            raise ValueError(f"KODAMA cluster {cluster_id} has no semantic prototype support")
        prototypes.append(probabilities[:, core].mean(axis=1))
    return cluster_ids, np.asarray(prototypes, dtype=np.float32)


def refine_labels(
    labels: np.ndarray,
    probabilities: np.ndarray,
    tissue: np.ndarray,
    *,
    boundary_band_px: int,
    core_erosion_px: int,
    min_distance_improvement: float,
    assign_unlabeled: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if labels.shape != tissue.shape or probabilities.ndim != 3 or probabilities.shape[1:] != labels.shape:
        raise ValueError("Cluster, tissue and PathSegmentor arrays are not aligned")
    if not np.isfinite(probabilities).all():
        raise ValueError("PathSegmentor probability stack contains non-finite values")
    cluster_ids, prototypes = semantic_prototypes(labels, probabilities, tissue, core_erosion_px)
    boundary = boundary_pixels(labels, tissue)
    band = int(boundary_band_px)
    editable = (boundary.copy() if band <= 0 else ndi.binary_dilation(boundary, iterations=band)) & tissue
    if assign_unlabeled:
        editable |= tissue & (labels == 0)
    feature = np.moveaxis(probabilities, 0, -1)
    distances = np.stack(
        [np.mean((feature - prototype.reshape(1, 1, -1)) ** 2, axis=2) for prototype in prototypes],
        axis=0,
    )
    best_index = np.argmin(distances, axis=0)
    candidate = cluster_ids[best_index]
    current_distance = np.full(labels.shape, np.inf, dtype=np.float32)
    for index, cluster_id in enumerate(cluster_ids):
        current_distance[labels == cluster_id] = distances[index][labels == cluster_id]
    best_distance = np.take_along_axis(distances, best_index[None], axis=0)[0]
    improvement = current_distance - best_distance
    change = editable & (candidate != labels) & (
        (labels == 0) | (improvement >= float(min_distance_improvement))
    )
    refined = labels.copy()
    refined[change] = candidate[change]
    refined[~tissue] = 0
    metadata = {
        "cluster_ids": cluster_ids.tolist(),
        "semantic_prototypes": prototypes.tolist(),
        "boundary_band_px": int(boundary_band_px),
        "core_erosion_px": int(core_erosion_px),
        "min_distance_improvement": float(min_distance_improvement),
        "assign_unlabeled_within_grandqc_tissue": bool(assign_unlabeled),
        "editable_pixels": int(editable.sum()),
        "changed_pixels": int(change.sum()),
        "changed_fraction_of_tissue": float(change.sum() / max(1, tissue.sum())),
        "cluster_count_before": int(cluster_ids.size),
        "cluster_count_after": int(len([value for value in np.unique(refined[tissue]) if value > 0])),
        "scientific_role": "experimental_supervised_boundary_refinement_preserving_raw_kodama_output",
    }
    if metadata["cluster_count_after"] != metadata["cluster_count_before"]:
        raise RuntimeError("PathSegmentor refinement changed the KODAMA cluster count")
    return refined, change, metadata


def resize_labels(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.int32), mode="I")
    return np.asarray(image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST), dtype=np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-mask", type=Path, required=True)
    parser.add_argument("--tissue-mask", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--pathsegmentor-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--change-mask", type=Path, required=True)
    parser.add_argument("--boundary-band-um", type=float, default=16.0)
    parser.add_argument("--core-erosion-um", type=float, default=24.0)
    parser.add_argument("--min-distance-improvement", type=float, default=0.01)
    parser.add_argument("--assign-unlabeled-within-tissue", action="store_true")
    parser.add_argument("--block-rows", type=int, default=512)
    args = parser.parse_args()
    if min(args.boundary_band_um, args.core_erosion_um, args.min_distance_improvement) < 0:
        raise ValueError("Refinement distances and widths must be non-negative")

    manifest = json.loads(args.pathsegmentor_manifest.read_text(encoding="utf-8"))
    output_mpp = float(manifest["output_mpp"])
    probabilities = np.asarray(tifffile.imread(args.probabilities), dtype=np.float32)
    if probabilities.ndim != 3:
        raise ValueError("PathSegmentor probabilities must be CYX")
    coarse_shape = tuple(map(int, probabilities.shape[1:]))
    with tifffile.TiffFile(args.cluster_mask) as tif:
        cluster_native = tif.series[0].asarray(out="memmap")
    with tifffile.TiffFile(args.tissue_mask) as tif:
        tissue_native = tif.series[0].asarray(out="memmap")
    if cluster_native.ndim != 2 or tissue_native.ndim != 2 or cluster_native.shape != tissue_native.shape:
        raise ValueError("Native cluster and GrandQC tissue masks must be aligned 2D rasters")
    coarse_cluster = resize_labels(cluster_native, coarse_shape)
    coarse_tissue = resize_labels(tissue_native, coarse_shape) > 0
    refined, changed, metadata = refine_labels(
        coarse_cluster,
        probabilities,
        coarse_tissue,
        boundary_band_px=int(round(args.boundary_band_um / output_mpp)),
        core_erosion_px=int(round(args.core_erosion_um / output_mpp)),
        min_distance_improvement=args.min_distance_improvement,
        assign_unlabeled=args.assign_unlabeled_within_tissue,
    )
    height, width = map(int, cluster_native.shape)
    y_map = np.minimum(coarse_shape[0] - 1, (np.arange(height) * coarse_shape[0] // height)).astype(np.int64)
    x_map = np.minimum(coarse_shape[1] - 1, (np.arange(width) * coarse_shape[1] // width)).astype(np.int64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = tifffile.memmap(args.out, shape=(height, width), dtype=np.uint16, bigtiff=True, metadata={"axes": "YX"})
    change_out = tifffile.memmap(args.change_mask, shape=(height, width), dtype=np.uint8, bigtiff=True, metadata={"axes": "YX"})
    for y0 in range(0, height, max(1, args.block_rows)):
        y1 = min(height, y0 + max(1, args.block_rows))
        coarse_y = y_map[y0:y1]
        candidate = refined[np.ix_(coarse_y, x_map)]
        edit = changed[np.ix_(coarse_y, x_map)]
        tissue = np.asarray(tissue_native[y0:y1]) > 0
        original = np.asarray(cluster_native[y0:y1], dtype=np.uint16)
        block = np.where(edit, candidate, original).astype(np.uint16)
        block[~tissue] = 0
        out[y0:y1] = block
        change_out[y0:y1] = edit.astype(np.uint8)
    out.flush()
    change_out.flush()
    metadata.update(
        pathsegmentor_manifest=str(args.pathsegmentor_manifest.resolve()),
        raw_kodama_mask=str(args.cluster_mask.resolve()),
        grandqc_tissue_mask=str(args.tissue_mask.resolve()),
        output_mpp=output_mpp,
        native_shape_yx=[height, width],
        coarse_shape_yx=list(coarse_shape),
    )
    args.provenance.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
