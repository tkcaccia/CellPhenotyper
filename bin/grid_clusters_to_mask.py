#!/usr/bin/env python3
"""Rasterize clustered UNI2 grid cores directly into a memory-mapped TIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from labels_to_cluster_mask import (
    UNCERTAINTY_PALETTE,
    UNCERTAINTY_STATUS_NAMES,
    load_map,
    load_uncertainty_map,
    smallest_mask_dtype,
    write_preview_overlay_png,
)
from ome_tiff_metadata import create_tiff_memmap


REQUIRED_GRID_COLUMNS = {
    "label",
    "grid_row",
    "core_x0",
    "core_y0",
    "core_x1",
    "core_y1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-objects", required=True)
    parser.add_argument("--grid-metadata", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uncertainty-out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--default", type=int, default=0)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--uncertainty-preview", required=True)
    parser.add_argument("--preview-factor", type=int, default=10)
    parser.add_argument("--preview-threshold-mb", type=float, default=100.0)
    parser.add_argument("--preview-background", required=True)
    parser.add_argument("--preview-alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(Path(args.grid_metadata).read_text(encoding="utf-8"))
    height = int(metadata["image_height_px"])
    width = int(metadata["image_width_px"])
    mpp_x = float(metadata["source_mpp_x"])
    mpp_y = float(metadata["source_mpp_y"])

    grid = pd.read_csv(args.grid_objects, usecols=sorted(REQUIRED_GRID_COLUMNS))
    missing = sorted(REQUIRED_GRID_COLUMNS.difference(grid.columns))
    if missing:
        raise ValueError(f"Grid observation CSV is missing columns: {missing}")
    for column in REQUIRED_GRID_COLUMNS:
        grid[column] = pd.to_numeric(grid[column], errors="raise").astype(np.int64)
    if grid.empty or grid["label"].duplicated().any() or (grid["label"] <= 0).any():
        raise ValueError("Grid labels must be non-empty, unique positive integers")

    # Grid observations remain spatially valid even when the stability policy
    # abstains from interpreting their cluster.  Keep KODAMA's raw assignment
    # in the categorical baseline and carry abstention separately in the
    # uncertainty raster.  Turning an abstention into label 0 leaves large
    # seedless windows that a tiled refiner can only fill locally, producing
    # processing-grid mosaics.
    mapping = load_map(
        args.map,
        default_value=args.default,
        prefer_interpretable=False,
    )
    uncertainty_mapping = load_uncertainty_map(args.map)
    cluster_records = pd.read_csv(args.map)
    cluster_records.columns = [c.strip().strip('"').strip("'") for c in cluster_records.columns]
    if "exclude_from_downstream" in cluster_records.columns:
        exclusion_values = cluster_records["exclude_from_downstream"].astype(str).str.strip().str.lower()
        exclude_by_label = pd.Series(
            exclusion_values.isin({"true", "1", "yes", "y"}).to_numpy(),
            index=pd.to_numeric(cluster_records["label"], errors="raise").astype(np.int64),
        )
    else:
        exclude_by_label = pd.Series(False, index=mapping["label"], dtype=bool)
    cluster_by_label = mapping.set_index("label")["cluster"]
    uncertainty_by_label = uncertainty_mapping.set_index("label")["uncertainty_code"]
    grid["cluster"] = grid["label"].map(cluster_by_label)
    grid["uncertainty_code"] = grid["label"].map(uncertainty_by_label)
    grid["exclude_from_downstream"] = grid["label"].map(exclude_by_label).fillna(False).astype(bool)
    missing_labels = grid.loc[grid["cluster"].isna(), "label"].tolist()
    if missing_labels:
        raise ValueError(
            f"Clustering output is missing {len(missing_labels)} grid observations; "
            f"examples={missing_labels[:10]}"
        )
    missing_uncertainty = grid.loc[grid["uncertainty_code"].isna(), "label"].tolist()
    if missing_uncertainty:
        raise ValueError(
            f"Clustering output is missing uncertainty status for {len(missing_uncertainty)} grid observations; "
            f"examples={missing_uncertainty[:10]}"
        )
    grid["cluster"] = grid["cluster"].astype(np.int64)
    grid["uncertainty_code"] = grid["uncertainty_code"].astype(np.uint8)
    if args.default < 0 or int(grid["cluster"].min()) < 0:
        out_dtype = np.dtype(np.int32)
    else:
        out_dtype = smallest_mask_dtype(max(args.default, int(grid["cluster"].max())))

    output = create_tiff_memmap(
        args.out,
        shape=(height, width),
        dtype=out_dtype,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    output[:] = args.default
    uncertainty_output = create_tiff_memmap(
        args.uncertainty_out,
        shape=(height, width),
        dtype=np.dtype(np.uint8),
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    uncertainty_output[:] = 0
    written_pixels = 0
    abstained_pixels = 0
    for _, row_group in grid.groupby("grid_row", sort=False):
        row_y0 = row_group["core_y0"].unique()
        row_y1 = row_group["core_y1"].unique()
        if len(row_y0) != 1 or len(row_y1) != 1:
            raise ValueError("A logical grid row contains inconsistent vertical core bounds")
        y0 = max(0, int(row_y0[0]))
        y1 = min(height, int(row_y1[0]))
        if y1 <= y0:
            continue
        scanline = np.full(width, args.default, dtype=out_dtype)
        uncertainty_scanline = np.zeros(width, dtype=np.uint8)
        for record in row_group.itertuples(index=False):
            x0 = max(0, int(record.core_x0))
            x1 = min(width, int(record.core_x1))
            if x1 <= x0:
                continue
            scanline[x0:x1] = int(record.cluster)
            if bool(record.exclude_from_downstream):
                scanline[x0:x1] = int(args.default)
            uncertainty_scanline[x0:x1] = int(record.uncertainty_code)
            written_pixels += (x1 - x0) * (y1 - y0)
            if int(record.uncertainty_code) > 0:
                abstained_pixels += (x1 - x0) * (y1 - y0)
        output[y0:y1, :] = scanline
        uncertainty_output[y0:y1, :] = uncertainty_scanline
    output.flush()
    uncertainty_output.flush()

    observed_clusters = sorted(
        int(value) for value in np.unique(grid["cluster"]) if int(value) != args.default
    )
    preview_factor, preview_bytes = write_preview_overlay_png(
        output,
        args.preview,
        factor_if_large=args.preview_factor,
        size_threshold_mb=args.preview_threshold_mb,
        default_value=args.default,
        preview_background_path=args.preview_background,
        alpha=args.preview_alpha,
    )
    uncertainty_preview_factor, uncertainty_preview_bytes = write_preview_overlay_png(
        uncertainty_output,
        args.uncertainty_preview,
        factor_if_large=args.preview_factor,
        size_threshold_mb=args.preview_threshold_mb,
        default_value=0,
        preview_background_path=args.preview_background,
        alpha=max(float(args.preview_alpha), 0.65),
        palette_by_value=UNCERTAINTY_PALETTE,
        legend_items=[
            (UNCERTAINTY_STATUS_NAMES[code], tuple(color.tolist()))
            for code, color in sorted(UNCERTAINTY_PALETTE.items())
        ],
    )
    status_counts = {
        UNCERTAINTY_STATUS_NAMES[int(code)]: int(count)
        for code, count in grid["uncertainty_code"].value_counts().sort_index().items()
    }
    abstained_observations = int((grid["uncertainty_code"] > 0).sum())
    excluded_observations = int(grid["exclude_from_downstream"].sum())
    summary = {
        "schema_version": 2,
        "observation_type": "spatial_grid",
        "grid_observations": int(len(grid)),
        "accepted_observations": int(len(grid) - abstained_observations),
        "abstained_observations": abstained_observations,
        "excluded_grandqc_candidate_kodama_outlier_observations": excluded_observations,
        "accepted_observation_fraction": float(
            (len(grid) - abstained_observations) / max(1, len(grid))
        ),
        "uncertainty_status_counts": status_counts,
        "uncertainty_mask_codes": {
            str(code): name for code, name in UNCERTAINTY_STATUS_NAMES.items()
        },
        "abstained_assignment_policy": "retain_raw_kodama_cluster_with_uncertainty",
        "grandqc_candidate_policy": "exclude_only_when_cluster_conditioned_kodama_plot_outlier",
        "clusters": observed_clusters,
        "written_core_pixels": int(written_pixels),
        "abstained_core_pixels": int(abstained_pixels),
        "image_height_px": height,
        "image_width_px": width,
        "preview_downsample_factor": int(preview_factor),
        "preview_estimated_bytes": int(preview_bytes),
        "uncertainty_preview_downsample_factor": int(uncertainty_preview_factor),
        "uncertainty_preview_estimated_bytes": int(uncertainty_preview_bytes),
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"[INFO] Rasterized {len(grid)} clustered grid cores into {args.out}; "
        f"clusters={observed_clusters} pixels={written_pixels} "
        f"abstained_observations={abstained_observations}"
    )


if __name__ == "__main__":
    main()
