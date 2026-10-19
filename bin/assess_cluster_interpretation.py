#!/usr/bin/env python3
"""Create spatial, marker, and blinded-review evidence for cluster interpretation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw


MARKER_METADATA_COLUMNS = {
    "label_id",
    "mask_name",
    "value_semantics",
    "calibrated_probability",
    "measured_protein_abundance",
    "area_px",
    "centroid_y_px",
    "centroid_x_px",
    "bbox_ymin_px",
    "bbox_xmin_px",
    "bbox_ymax_px",
    "bbox_xmax_px",
}

SPATIAL_COLUMNS = [
    "cluster",
    "sampled_observations",
    "mean_same_cluster_fraction",
    "median_same_cluster_fraction",
]
MARKER_COLUMNS = [
    "cluster",
    "marker",
    "observations",
    "mean",
    "median",
    "overall_mean",
    "standardized_mean_difference",
]

UNCERTAINTY_COLORS = {
    "abstained_ambiguous_assignment": "#E69F00",
    "abstained_seed_instability": "#56B4E9",
    "abstained_ambiguous_assignment_and_seed_instability": "#CC79A7",
    "abstained_other": "#333333",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--marker-quant-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--spatial-k", type=int, default=15)
    parser.add_argument("--spatial-max-observations", type=int, default=50000)
    parser.add_argument("--review-per-cluster", type=int, default=20)
    parser.add_argument("--review-radius-px", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def normalize_ids(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip().str.strip('"').str.strip("'")
    numeric = pd.to_numeric(raw, errors="coerce")
    integer_like = numeric.notna() & np.isclose(numeric.fillna(0) % 1, 0)
    raw.loc[integer_like] = numeric.loc[integer_like].astype(np.int64).astype(str)
    return raw


def load_observations(cluster_path: str, object_path: str) -> tuple[pd.DataFrame, str]:
    clusters = pd.read_csv(cluster_path)
    objects = pd.read_csv(object_path)
    required_cluster = {"label", "cluster"}
    required_objects = {"label", "x", "y"}
    if missing := sorted(required_cluster.difference(clusters.columns)):
        raise ValueError(f"Cluster CSV is missing columns: {missing}")
    if missing := sorted(required_objects.difference(objects.columns)):
        raise ValueError(f"Object CSV is missing columns: {missing}")
    clusters["label_key"] = normalize_ids(clusters["label"])
    objects["label_key"] = normalize_ids(objects["label"])
    cluster_column = "interpretable_cluster" if "interpretable_cluster" in clusters.columns else "cluster"
    clusters["assessment_cluster"] = pd.to_numeric(clusters[cluster_column], errors="coerce")
    keep_cluster_columns = [
        column
        for column in (
            "label_key",
            "cluster",
            "assessment_cluster",
            "assignment_vote_fraction",
            "assignment_vote_margin",
            "stability_fraction",
            "uncertainty_reason",
            "is_abstained",
            "interpretation_status",
        )
        if column in clusters.columns
    ]
    observations = objects.merge(
        clusters[keep_cluster_columns], on="label_key", how="inner", validate="one_to_one"
    )
    observations["x"] = pd.to_numeric(observations["x"], errors="coerce")
    observations["y"] = pd.to_numeric(observations["y"], errors="coerce")
    observations = observations[np.isfinite(observations["x"]) & np.isfinite(observations["y"])].copy()
    observation_type = (
        "spatial_grid"
        if {"core_x0", "core_y0", "core_x1", "core_y1"}.issubset(observations.columns)
        else "contextual_cell"
    )
    return observations, observation_type


def stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("assessment_cluster", sort=True))
    target_by_group = {
        cluster: max(1, int(round(maximum * len(group) / len(frame))))
        for cluster, group in groups
    }
    selected: list[np.ndarray] = []
    for cluster, group in groups:
        count = min(len(group), target_by_group[cluster])
        selected.append(rng.choice(group.index.to_numpy(), size=count, replace=False))
    indices = np.concatenate(selected)
    if len(indices) > maximum:
        indices = rng.choice(indices, size=maximum, replace=False)
    elif len(indices) < maximum:
        remaining = frame.index.difference(indices).to_numpy()
        if len(remaining):
            extra = rng.choice(remaining, size=min(maximum - len(indices), len(remaining)), replace=False)
            indices = np.concatenate([indices, extra])
    return frame.loc[np.sort(indices)].copy()


def spatial_coherence(frame: pd.DataFrame, k: int, maximum: int, seed: int) -> tuple[dict, pd.DataFrame]:
    accepted = frame[frame["assessment_cluster"].notna()].copy()
    if len(accepted) < 3:
        return {"status": "not_evaluated", "reason": "fewer_than_three_interpretable_observations"}, pd.DataFrame()
    sampled = stratified_sample(accepted, max(3, maximum), seed)
    used_k = max(1, min(int(k), len(sampled) - 1))
    coords = sampled[["x", "y"]].to_numpy(dtype=np.float64)
    labels = sampled["assessment_cluster"].to_numpy(dtype=np.int64)
    neighbor_index = cKDTree(coords).query(coords, k=used_k + 1, workers=1)[1]
    if neighbor_index.ndim == 1:
        neighbor_index = neighbor_index[:, None]
    neighbor_index = neighbor_index[:, 1:]
    same_fraction = np.mean(labels[neighbor_index] == labels[:, None], axis=1)
    sampled["spatial_neighbor_same_cluster_fraction"] = same_fraction
    cluster_rows = (
        sampled.groupby("assessment_cluster", sort=True)
        .agg(
            sampled_observations=("label_key", "size"),
            mean_same_cluster_fraction=("spatial_neighbor_same_cluster_fraction", "mean"),
            median_same_cluster_fraction=("spatial_neighbor_same_cluster_fraction", "median"),
        )
        .reset_index()
        .rename(columns={"assessment_cluster": "cluster"})
    )
    prevalence = sampled["assessment_cluster"].value_counts(normalize=True)
    chance_agreement = float(np.square(prevalence.to_numpy(dtype=float)).sum())
    observed_agreement = float(same_fraction.mean())
    summary = {
        "status": "descriptive_only",
        "sampled_observations": int(len(sampled)),
        "spatial_k": int(used_k),
        "mean_same_cluster_neighbor_fraction": observed_agreement,
        "chance_same_cluster_fraction_from_prevalence": chance_agreement,
        "excess_over_prevalence_chance": observed_agreement - chance_agreement,
        "interpretation": (
            "Spatial coherence can reflect biology, smoothing, or spatial confounding and is not accuracy."
        ),
    }
    return summary, cluster_rows


def map_points_to_grid(points: pd.DataFrame, grids: pd.DataFrame) -> pd.Series:
    x_starts = np.sort(grids["core_x0"].astype(float).unique())
    y_starts = np.sort(grids["core_y0"].astype(float).unique())
    x_bin = np.searchsorted(x_starts, points["centroid_x_px"].to_numpy(float), side="right") - 1
    y_bin = np.searchsorted(y_starts, points["centroid_y_px"].to_numpy(float), side="right") - 1
    valid = (x_bin >= 0) & (y_bin >= 0)
    keys = np.full(len(points), "", dtype=object)
    keys[valid] = np.char.add(
        np.char.add(x_starts[x_bin[valid]].astype(str), ":"),
        y_starts[y_bin[valid]].astype(str),
    )
    grid_keys = grids["core_x0"].astype(float).astype(str) + ":" + grids["core_y0"].astype(float).astype(str)
    key_to_index = pd.Series(grids.index.to_numpy(), index=grid_keys).to_dict()
    matched_index = pd.Series(keys).map(key_to_index)
    matched_cluster = pd.Series(np.nan, index=points.index, dtype=float)
    matched = matched_index.notna().to_numpy()
    if matched.any():
        target = grids.loc[matched_index[matched].astype(int).to_numpy()]
        source = points.loc[matched]
        inside = (
            (source["centroid_x_px"].to_numpy(float) >= target["core_x0"].to_numpy(float))
            & (source["centroid_x_px"].to_numpy(float) < target["core_x1"].to_numpy(float))
            & (source["centroid_y_px"].to_numpy(float) >= target["core_y0"].to_numpy(float))
            & (source["centroid_y_px"].to_numpy(float) < target["core_y1"].to_numpy(float))
        )
        matched_positions = np.flatnonzero(matched)[inside]
        matched_cluster.iloc[matched_positions] = target.loc[inside, "assessment_cluster"].to_numpy(float)
    return matched_cluster


def marker_enrichment(
    observations: pd.DataFrame, observation_type: str, marker_quant_dir: str
) -> tuple[dict, pd.DataFrame]:
    marker_files = sorted(Path(marker_quant_dir).glob("*_nuclei_gigatime_mean_intensity.csv"))
    if not marker_files:
        return {"status": "not_available", "reason": "nuclei_marker_table_not_found"}, pd.DataFrame()
    markers = pd.read_csv(marker_files[0])
    if "label_id" not in markers.columns:
        return {"status": "not_available", "reason": "label_id_missing"}, pd.DataFrame()
    marker_columns = [
        column
        for column in markers.columns
        if column not in MARKER_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(markers[column])
    ]
    if not marker_columns:
        return {"status": "not_available", "reason": "numeric_marker_columns_missing"}, pd.DataFrame()

    if observation_type == "contextual_cell":
        markers["label_key"] = normalize_ids(markers["label_id"])
        cluster_by_label = observations.set_index("label_key")["assessment_cluster"]
        markers["assessment_cluster"] = markers["label_key"].map(cluster_by_label)
    else:
        markers["assessment_cluster"] = map_points_to_grid(markers, observations)
    markers = markers[markers["assessment_cluster"].notna()].copy()
    if markers.empty:
        return {"status": "not_available", "reason": "no_markers_mapped_to_interpretable_clusters"}, pd.DataFrame()

    rows: list[dict] = []
    for marker in marker_columns:
        values = pd.to_numeric(markers[marker], errors="coerce")
        overall_mean = float(values.mean())
        overall_sd = float(values.std(ddof=1))
        for cluster, group in markers.assign(_value=values).groupby("assessment_cluster", sort=True):
            group_values = group["_value"].dropna()
            if group_values.empty:
                continue
            rows.append(
                {
                    "cluster": int(cluster),
                    "marker": marker,
                    "observations": int(len(group_values)),
                    "mean": float(group_values.mean()),
                    "median": float(group_values.median()),
                    "overall_mean": overall_mean,
                    "standardized_mean_difference": (
                        float((group_values.mean() - overall_mean) / overall_sd)
                        if math.isfinite(overall_sd) and overall_sd > 0
                        else np.nan
                    ),
                }
            )
    enrichment = pd.DataFrame(rows)
    return {
        "status": "descriptive_only",
        "marker_table": str(marker_files[0]),
        "mapped_marker_objects": int(len(markers)),
        "markers": int(len(marker_columns)),
        "interpretation": "GigaTIME markers are same-image virtual-marker proxies, not independent reference measurements.",
    }, enrichment


def write_blinded_review_packet(
    frame: pd.DataFrame,
    outdir: Path,
    *,
    per_cluster: int,
    radius: float,
    seed: int,
) -> dict:
    accepted = frame[frame["assessment_cluster"].notna()].copy()
    rng = np.random.default_rng(seed)
    selected_parts = []
    for _, group in accepted.groupby("assessment_cluster", sort=True):
        count = min(max(1, int(per_cluster)), len(group))
        selected_parts.append(group.loc[rng.choice(group.index.to_numpy(), size=count, replace=False)])
    if not selected_parts:
        return {"status": "not_available", "reason": "no_interpretable_clusters"}
    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    selected["review_id"] = [f"R{index:04d}" for index in range(1, len(selected) + 1)]
    form = selected[["review_id"]].copy()
    form["tissue_interpretation"] = ""
    form["reviewer_confidence"] = ""
    form["artifact_present"] = ""
    form["notes"] = ""
    form.to_csv(outdir / "blinded_review_form.csv", index=False)
    key_columns = ["review_id", "label_key", "assessment_cluster", "x", "y"]
    for column in ("assignment_vote_margin", "stability_fraction", "interpretation_status"):
        if column in selected.columns:
            key_columns.append(column)
    selected[key_columns].rename(columns={"assessment_cluster": "cluster"}).to_csv(
        outdir / "blinded_review_key.csv", index=False
    )

    features = []
    for row in selected.itertuples(index=False):
        x = float(row.x)
        y = float(row.y)
        coordinates = [[
            [x - radius, y - radius],
            [x + radius, y - radius],
            [x + radius, y + radius],
            [x - radius, y + radius],
            [x - radius, y - radius],
        ]]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": coordinates},
                "properties": {"review_id": row.review_id},
            }
        )
    (outdir / "blinded_review_regions.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "ready_for_independent_review",
        "regions": int(len(selected)),
        "regions_per_cluster_requested": int(per_cluster),
        "random_seed": int(seed),
        "blinding": "review form and GeoJSON omit cluster; answer key is separate",
    }


def normalized_interpretation_status(frame: pd.DataFrame) -> pd.Series:
    if "interpretation_status" not in frame.columns:
        return pd.Series(
            np.where(frame["assessment_cluster"].notna(), "accepted", "abstained_other"),
            index=frame.index,
            dtype=str,
        )
    status = frame["interpretation_status"].fillna("abstained_other").astype(str)
    known = status.eq("accepted") | status.isin(UNCERTAINTY_COLORS)
    status = status.where(known, "abstained_other")
    return status.where(frame["assessment_cluster"].isna(), "accepted")


def write_abstention_geojson(frame: pd.DataFrame, observation_type: str, outdir: Path) -> dict:
    status = normalized_interpretation_status(frame)
    abstained = frame.loc[status.ne("accepted")].copy()
    abstained["status_name"] = status.loc[abstained.index]
    features = []
    is_grid = observation_type == "spatial_grid"
    for row in abstained.itertuples(index=False):
        if is_grid:
            x0, y0 = float(row.core_x0), float(row.core_y0)
            x1, y1 = float(row.core_x1), float(row.core_y1)
            geometry = {
                "type": "Polygon",
                "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
            }
        else:
            geometry = {"type": "Point", "coordinates": [float(row.x), float(row.y)]}
        properties = {
            "label": str(row.label_key),
            "assigned_cluster": int(row.cluster) if pd.notna(row.cluster) else None,
            "interpretation_status": str(row.status_name),
            "observation_unit": observation_type,
        }
        for column in ("assignment_vote_fraction", "assignment_vote_margin", "stability_fraction"):
            value = getattr(row, column, None)
            properties[column] = float(value) if value is not None and pd.notna(value) else None
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    output_path = outdir / "cluster_abstentions.geojson"
    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "observation_unit": observation_type,
            "coordinate_space": "level_0_pixels",
            "purpose": "Explicit locations excluded from interpretable cluster outputs",
            "geometry_semantics": "grid core polygons" if is_grid else "cell-centroid points",
        },
        "features": features,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "status": "available",
        "file": output_path.name,
        "abstained_observations": int(len(abstained)),
        "status_counts": {
            str(name): int(count) for name, count in abstained["status_name"].value_counts().items()
        },
    }


def write_spatial_uncertainty_plot(
    frame: pd.DataFrame,
    observation_type: str,
    outdir: Path,
    seed: int,
) -> dict:
    status = normalized_interpretation_status(frame)
    rng = np.random.default_rng(seed)
    accepted_idx = frame.index[status.eq("accepted")].to_numpy()
    abstained_idx = frame.index[status.ne("accepted")].to_numpy()
    accepted_cap = 200_000
    abstained_cap = 100_000
    if len(accepted_idx) > accepted_cap:
        accepted_idx = rng.choice(accepted_idx, accepted_cap, replace=False)
    if len(abstained_idx) > abstained_cap:
        abstained_idx = rng.choice(abstained_idx, abstained_cap, replace=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return write_spatial_uncertainty_plot_pillow(
            frame,
            status,
            accepted_idx,
            abstained_idx,
            observation_type,
            outdir,
        )

    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    if len(accepted_idx):
        accepted = frame.loc[accepted_idx]
        ax.scatter(
            accepted["x"], accepted["y"], s=0.8, c="#BDBDBD", alpha=0.35,
            linewidths=0, rasterized=True, label=f"accepted n={status.eq('accepted').sum():,}",
        )
    for name, color in UNCERTAINTY_COLORS.items():
        selected_index = abstained_idx[status.loc[abstained_idx].to_numpy() == name]
        if not len(selected_index):
            continue
        selected = frame.loc[selected_index]
        total = int(status.eq(name).sum())
        ax.scatter(
            selected["x"], selected["y"], s=2.0, c=color, alpha=0.85,
            linewidths=0, rasterized=True, label=f"{name} n={total:,}",
        )
    ax.set_title(
        f"Spatial uncertainty and abstention | {observation_type}\n"
        "Gray observations are accepted; colored observations are excluded from interpretation"
    )
    ax.set_xlabel("level-0 x coordinate (px)")
    ax.set_ylabel("level-0 y coordinate (px)")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=False, markerscale=4, fontsize=8)
    output_path = outdir / "cluster_spatial_uncertainty.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "status": "available",
        "file": output_path.name,
        "accepted_points_rendered": int(len(accepted_idx)),
        "abstained_points_rendered": int(len(abstained_idx)),
        "rendering_note": "Large classes are deterministically sampled for display; counts use all observations.",
    }


def write_spatial_uncertainty_plot_pillow(
    frame: pd.DataFrame,
    status: pd.Series,
    accepted_idx: np.ndarray,
    abstained_idx: np.ndarray,
    observation_type: str,
    outdir: Path,
) -> dict:
    width, height = 1800, 1400
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 110, 190
    canvas = Image.new("RGB", (width, height), (250, 249, 245))
    draw = ImageDraw.Draw(canvas)
    selected_idx = np.concatenate([accepted_idx, abstained_idx])
    selected = frame.loc[selected_idx] if len(selected_idx) else frame.iloc[:0]
    if len(selected):
        x = selected["x"].to_numpy(dtype=float)
        y = selected["y"].to_numpy(dtype=float)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)

        def project(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            px = margin_left + (
                (points["x"].to_numpy(dtype=float) - x_min) / x_span
            ) * (width - margin_left - margin_right)
            py = margin_top + (
                (points["y"].to_numpy(dtype=float) - y_min) / y_span
            ) * (height - margin_top - margin_bottom)
            return px.astype(int), py.astype(int)

        accepted = frame.loc[accepted_idx]
        px, py = project(accepted)
        for xx, yy in zip(px, py):
            draw.point((int(xx), int(yy)), fill=(189, 189, 189))
        for name, color in UNCERTAINTY_COLORS.items():
            idx = abstained_idx[status.loc[abstained_idx].to_numpy() == name]
            points = frame.loc[idx]
            px, py = project(points)
            rgb = tuple(int(color[pos : pos + 2], 16) for pos in (1, 3, 5))
            for xx, yy in zip(px, py):
                draw.ellipse((int(xx) - 2, int(yy) - 2, int(xx) + 2, int(yy) + 2), fill=rgb)

    title = f"Spatial uncertainty and abstention | {observation_type}"
    draw.text((margin_left, 32), title, fill=(20, 42, 51))
    draw.text(
        (margin_left, 58),
        "Gray observations are accepted; colored observations are excluded from interpretation",
        fill=(55, 70, 76),
    )
    legend_y = height - margin_bottom + 30
    legend = [("accepted", "#BDBDBD"), *UNCERTAINTY_COLORS.items()]
    for label, color in legend:
        rgb = tuple(int(color[pos : pos + 2], 16) for pos in (1, 3, 5))
        count = int(status.eq(label).sum())
        if count == 0 and label != "accepted":
            continue
        draw.rectangle((margin_left, legend_y, margin_left + 18, legend_y + 18), fill=rgb)
        draw.text((margin_left + 28, legend_y + 1), f"{label} n={count:,}", fill=(25, 25, 25))
        legend_y += 28
    output_path = outdir / "cluster_spatial_uncertainty.png"
    canvas.save(output_path)
    return {
        "status": "available",
        "file": output_path.name,
        "renderer": "pillow_fallback",
        "accepted_points_rendered": int(len(accepted_idx)),
        "abstained_points_rendered": int(len(abstained_idx)),
        "rendering_note": "Large classes are deterministically sampled for display; counts use all observations.",
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    observations, observation_type = load_observations(args.clusters, args.objects)
    spatial_summary, spatial_rows = spatial_coherence(
        observations, args.spatial_k, args.spatial_max_observations, args.seed
    )
    marker_summary, marker_rows = marker_enrichment(
        observations, observation_type, args.marker_quant_dir
    )
    review_summary = write_blinded_review_packet(
        observations,
        outdir,
        per_cluster=args.review_per_cluster,
        radius=args.review_radius_px,
        seed=args.seed,
    )
    abstention_map_summary = write_abstention_geojson(
        observations, observation_type, outdir
    )
    spatial_uncertainty_summary = write_spatial_uncertainty_plot(
        observations, observation_type, outdir, args.seed
    )
    spatial_rows.reindex(columns=SPATIAL_COLUMNS).to_csv(
        outdir / "cluster_spatial_coherence.csv", index=False
    )
    marker_rows.reindex(columns=MARKER_COLUMNS).to_csv(
        outdir / "cluster_marker_enrichment.csv", index=False
    )
    interpretable = int(observations["assessment_cluster"].notna().sum())
    report = {
        "schema_version": 1,
        "sample_id": args.sample_id,
        "cluster_variant": args.variant,
        "observation_type": observation_type,
        "status": "independent_review_required",
        "observations": int(len(observations)),
        "interpretable_observations": interpretable,
        "abstained_observations": int(len(observations) - interpretable),
        "spatial_coherence": spatial_summary,
        "marker_enrichment": marker_summary,
        "blinded_review_packet": review_summary,
        "abstention_map": abstention_map_summary,
        "spatial_uncertainty_figure": spatial_uncertainty_summary,
        "interpretation_constraint": (
            "Stability, spatial coherence, and virtual-marker enrichment do not establish a biological class. "
            "Independent annotations or assays and blinded expert review remain required."
        ),
    }
    (outdir / "cluster_interpretation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[INFO] Cluster interpretation evidence: observations={len(observations)} "
        f"interpretable={interpretable} type={observation_type}",
        flush=True,
    )


if __name__ == "__main__":
    main()
