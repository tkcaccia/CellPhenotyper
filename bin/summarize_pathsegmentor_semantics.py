#!/usr/bin/env python3
"""Attach PathSegmentor semantic evidence to observations and KODAMA clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def observation_bounds(row: pd.Series, source_shape: tuple[int, int], cell_window_px: int) -> tuple[int, int, int, int]:
    height, width = source_shape
    if all(name in row.index and pd.notna(row[name]) for name in ("core_x0", "core_y0", "core_x1", "core_y1")):
        x0, y0, x1, y1 = (int(row[name]) for name in ("core_x0", "core_y0", "core_x1", "core_y1"))
    else:
        cx = int(round(float(row["x"])))
        cy = int(round(float(row["y"])))
        half = max(1, int(cell_window_px)) // 2
        x0, y0, x1, y1 = cx - half, cy - half, cx - half + max(1, cell_window_px), cy - half + max(1, cell_window_px)
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def summarize(
    probabilities: np.ndarray,
    manifest: dict,
    observations: pd.DataFrame,
    *,
    cell_window_um: float,
) -> pd.DataFrame:
    source_h, source_w = map(int, manifest["source_shape_yx"])
    source_mpp = float(manifest["source_mpp"])
    output_mpp = float(manifest["output_mpp"])
    prompt_ids = [str(record["id"]) for record in manifest["prompt_panel"]["prompts"]]
    if probabilities.shape[0] != len(prompt_ids):
        raise ValueError("PathSegmentor probability channels do not match the prompt manifest")
    if not {"label", "x", "y"}.issubset(observations.columns):
        raise ValueError("Observation table requires label,x,y")
    scale = source_mpp / output_mpp
    cell_window_px = max(1, int(round(float(cell_window_um) / source_mpp)))
    records: list[dict] = []
    for _, row in observations.iterrows():
        x0, y0, x1, y1 = observation_bounds(row, (source_h, source_w), cell_window_px)
        ox0 = max(0, min(probabilities.shape[2] - 1, int(np.floor(x0 * scale))))
        oy0 = max(0, min(probabilities.shape[1] - 1, int(np.floor(y0 * scale))))
        ox1 = max(ox0 + 1, min(probabilities.shape[2], int(np.ceil(x1 * scale))))
        oy1 = max(oy0 + 1, min(probabilities.shape[1], int(np.ceil(y1 * scale))))
        record = {"label": str(row["label"]), "x": float(row["x"]), "y": float(row["y"])}
        for index, prompt_id in enumerate(prompt_ids):
            values = probabilities[index, oy0:oy1, ox0:ox1]
            record[f"pathsegmentor_{prompt_id}_mean"] = float(values.mean())
            record[f"pathsegmentor_{prompt_id}_max"] = float(values.max())
        records.append(record)
    return pd.DataFrame.from_records(records)


def cluster_summary(semantic: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if not {"label", "cluster"}.issubset(clusters.columns):
        raise ValueError("KODAMA cluster table requires label and cluster")
    left = semantic.copy()
    right = clusters[[column for column in ("label", "cluster", "interpretable_cluster", "is_abstained") if column in clusters]].copy()
    left["label"] = left["label"].astype(str)
    right["label"] = right["label"].astype(str)
    joined = left.merge(right, on="label", how="left", validate="one_to_one")
    if joined["cluster"].isna().any():
        raise ValueError("Some PathSegmentor observations have no KODAMA cluster assignment")
    feature_columns = [column for column in joined if column.startswith("pathsegmentor_")]
    grouped = joined.groupby("cluster", sort=True, dropna=False)
    result = grouped[feature_columns].mean().reset_index()
    result.insert(1, "observations", grouped.size().to_numpy())
    result["semantic_role"] = "descriptive_supervised_evidence_not_cluster_ground_truth"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--cell-window-um", type=float, default=28.0)
    parser.add_argument("--observation-role", choices=["grid", "cell"], required=True)
    args = parser.parse_args()
    if args.cell_window_um <= 0:
        raise ValueError("--cell-window-um must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    probabilities = tifffile.imread(args.probabilities)
    if probabilities.ndim != 3:
        raise ValueError("PathSegmentor probabilities must have CYX shape")
    observations = pd.read_csv(args.observations)
    semantic = summarize(probabilities, manifest, observations, cell_window_um=args.cell_window_um)
    semantic.insert(1, "observation_role", args.observation_role)
    args.outdir.mkdir(parents=True, exist_ok=True)
    semantic_path = args.outdir / f"pathsegmentor_{args.observation_role}_semantics.csv.gz"
    semantic.to_csv(semantic_path, index=False, compression="gzip")
    summary_path = None
    if args.clusters is not None:
        summary = cluster_summary(semantic, pd.read_csv(args.clusters))
        summary_path = args.outdir / "pathsegmentor_cluster_semantics.csv"
        summary.to_csv(summary_path, index=False)
    receipt = {
        "schema_version": "1.0.0",
        "observation_role": args.observation_role,
        "observation_count": int(len(semantic)),
        "semantic_role": "descriptive_supervised_evidence_not_cell_type_or_cluster_ground_truth",
        "probability_manifest": str(args.manifest.resolve()),
        "semantic_table": semantic_path.name,
        "cluster_summary": summary_path.name if summary_path else None,
    }
    (args.outdir / "pathsegmentor_annotation_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
