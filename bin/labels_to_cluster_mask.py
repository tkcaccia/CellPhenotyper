#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image, ImageDraw

from tiff_preview import read_tiled_tiff_preview

DEFAULT_PALETTE = np.array([
    [0, 114, 178],
    [230, 159, 0],
    [0, 158, 115],
    [204, 121, 167],
    [213, 94, 0],
    [86, 180, 233],
    [240, 228, 66],
    [0, 0, 0],
    [51, 34, 136],
    [136, 204, 238],
    [68, 170, 153],
    [17, 119, 51],
    [153, 153, 51],
    [221, 204, 119],
    [204, 102, 119],
    [136, 34, 85],
], dtype=np.uint8)

UNCERTAINTY_STATUS_CODES = {
    "accepted": 0,
    "none": 0,
    "ambiguous_assignment": 1,
    "seed_instability": 2,
    "ambiguous_assignment_and_seed_instability": 3,
    "abstained_ambiguous_assignment": 1,
    "abstained_seed_instability": 2,
    "abstained_ambiguous_assignment_and_seed_instability": 3,
    "grandqc_artifact_kodama_outlier": 5,
    "abstained_grandqc_artifact_kodama_outlier": 5,
}
UNCERTAINTY_STATUS_NAMES = {
    0: "accepted_or_background",
    1: "abstained_ambiguous_assignment",
    2: "abstained_seed_instability",
    3: "abstained_ambiguous_assignment_and_seed_instability",
    4: "abstained_other",
    5: "excluded_grandqc_candidate_kodama_outlier",
}
UNCERTAINTY_PALETTE = {
    1: np.array([230, 159, 0], dtype=np.uint8),
    2: np.array([86, 180, 233], dtype=np.uint8),
    3: np.array([204, 121, 167], dtype=np.uint8),
    4: np.array([51, 51, 51], dtype=np.uint8),
    5: np.array([215, 48, 39], dtype=np.uint8),
}


def read_mask_2d(path: str) -> np.ndarray:
    m = tiff.imread(path)
    if m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask TIFF, got shape={m.shape}")
    if not np.issubdtype(m.dtype, np.integer):
        m = m.astype(np.int64)
    return m


def open_mask_2d(path: str) -> np.ndarray:
    try:
        m = tiff.memmap(path)
    except Exception:
        with tiff.TiffFile(path) as tif:
            m = tif.pages[0].asarray(out="memmap")
    if m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask TIFF, got shape={m.shape}")
    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError(f"Expected integer label mask TIFF, got dtype={m.dtype}")
    return m


def smallest_mask_dtype(max_value: int) -> np.dtype:
    max_value = int(max_value)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def load_map(
    csv_path: str,
    default_value: int | None = None,
    *,
    prefer_interpretable: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    if "label" not in df.columns or "cluster" not in df.columns:
        raise ValueError('CSV must contain columns: "label","cluster"')

    cluster_column = (
        "interpretable_cluster"
        if prefer_interpretable and "interpretable_cluster" in df.columns
        else "cluster"
    )
    df = df[["label", cluster_column]].copy().rename(columns={cluster_column: "cluster"})
    raw_label = df["label"].astype(str).str.strip().str.strip('"').str.strip("'")
    label_num = pd.to_numeric(raw_label, errors="coerce")
    if label_num.isna().any():
        tail_digits = raw_label.str.extract(r"([0-9]+)$", expand=False)
        label_num = label_num.fillna(pd.to_numeric(tail_digits, errors="coerce"))
    if label_num.isna().any():
        bad = raw_label[label_num.isna()].head(10).tolist()
        raise ValueError(
            "Label column contains values that cannot be mapped to integer IDs. "
            f"Examples: {bad}"
        )
    df["label"] = label_num.astype(np.int64)

    cl = pd.to_numeric(df["cluster"], errors="coerce")
    allow_abstention = cluster_column == "interpretable_cluster"
    if cl.isna().any() and not allow_abstention:
        bad = df.loc[cl.isna(), "cluster"].head(10).tolist()
        raise ValueError(
            "Cluster column contains non-numeric values (cannot write as numeric mask). "
            f"Examples: {bad}"
        )
    if allow_abstention:
        if default_value is None:
            keep = cl.notna()
            df = df.loc[keep].copy()
            cl = cl.loc[keep]
        else:
            cl = cl.fillna(int(default_value))
    df["cluster"] = cl.astype(np.int64)

    dup_conflicts = df.groupby("label")["cluster"].nunique(dropna=False)
    dup_conflicts = dup_conflicts[dup_conflicts > 1]
    if len(dup_conflicts) > 0:
        bad_ids = dup_conflicts.index[:10].tolist()
        raise ValueError(
            "Same label is assigned to multiple clusters in mapping CSV. "
            f"Example label IDs: {bad_ids}"
        )
    df = df.drop_duplicates(subset=["label"], keep="first")
    return df


def uncertainty_code(status: str) -> int:
    normalized = str(status).strip()
    if normalized in UNCERTAINTY_STATUS_CODES:
        return UNCERTAINTY_STATUS_CODES[normalized]
    return 0 if normalized == "accepted" else 4


def load_uncertainty_map(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    if "label" not in df.columns:
        raise ValueError('CSV must contain column "label"')
    raw_label = df["label"].astype(str).str.strip().str.strip('"').str.strip("'")
    label_num = pd.to_numeric(raw_label, errors="coerce")
    if label_num.isna().any():
        tail_digits = raw_label.str.extract(r"([0-9]+)$", expand=False)
        label_num = label_num.fillna(pd.to_numeric(tail_digits, errors="coerce"))
    if label_num.isna().any():
        bad = raw_label[label_num.isna()].head(10).tolist()
        raise ValueError(f"Uncertainty labels cannot be mapped to integer IDs. Examples: {bad}")
    status = (
        df["interpretation_status"].fillna("abstained_other").astype(str)
        if "interpretation_status" in df.columns
        else pd.Series("accepted", index=df.index, dtype=str)
    )
    reason = (
        df["uncertainty_reason"].fillna("other").astype(str)
        if "uncertainty_reason" in df.columns
        else status
    )
    if "is_abstained" in df.columns:
        raw_abstained = df["is_abstained"].astype(str).str.strip().str.lower()
        is_abstained = raw_abstained.isin({"true", "1", "yes", "y"})
    elif "interpretable_cluster" in df.columns:
        is_abstained = pd.to_numeric(df["interpretable_cluster"], errors="coerce").isna()
    else:
        is_abstained = status.ne("accepted")
    codes = reason.map(uncertainty_code).astype(np.uint8)
    codes.loc[~is_abstained] = 0
    out = pd.DataFrame(
        {
            "label": label_num.astype(np.int64),
            "uncertainty_code": codes,
            "interpretation_status": status,
        }
    )
    conflicts = out.groupby("label")["uncertainty_code"].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError(
            "Same label has conflicting interpretation statuses. "
            f"Examples: {conflicts[conflicts > 1].index[:10].tolist()}"
        )
    return out.drop_duplicates(subset=["label"], keep="first")


def downsample_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    f = int(factor)
    if f <= 1:
        return img
    return img[::f, ::f]


def row_blocks(n_rows: int, block_rows: int):
    block_rows = max(1, int(block_rows))
    for y0 in range(0, int(n_rows), block_rows):
        yield y0, min(int(n_rows), y0 + block_rows)


def remap_mask_chunked(
    labels: np.ndarray,
    lut: np.ndarray,
    out_path: str,
    out_dtype: np.dtype,
    default_value: int,
    block_rows: int,
) -> tuple[np.ndarray, dict]:
    out = tiff.memmap(out_path, shape=labels.shape, dtype=out_dtype, bigtiff=True)
    present_labels = set()
    observed_clusters = set()
    foreground_px = 0
    mapped_px = 0

    for y0, y1 in row_blocks(labels.shape[0], block_rows):
        block = np.asarray(labels[y0:y1, :])
        out_block = lut[block]
        out[y0:y1, :] = out_block

        fg = block != 0
        foreground_px += int(fg.sum())
        mapped = fg & (out_block != default_value)
        mapped_px += int(mapped.sum())

        if fg.any():
            present_labels.update(int(x) for x in np.unique(block[fg]))
        if mapped.any():
            observed_clusters.update(int(x) for x in np.unique(out_block[mapped]))

    out.flush()
    stats = {
        "present_labels": np.array(sorted(present_labels), dtype=np.int64),
        "observed_clusters": np.array(sorted(observed_clusters), dtype=np.int64),
        "foreground_px": foreground_px,
        "mapped_px": mapped_px,
    }
    return out, stats


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported preview image shape: {arr.shape}")
        arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported preview image shape: {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr_f = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_f)
    if not finite.any():
        return np.zeros(arr.shape[:2] + (3,), dtype=np.uint8)
    lo = float(np.percentile(arr_f[finite], 1.0))
    hi = float(np.percentile(arr_f[finite], 99.0))
    if hi <= lo:
        hi = lo + 1.0
    arr_f = (arr_f - lo) * (255.0 / (hi - lo))
    return np.clip(arr_f, 0, 255).astype(np.uint8)


def _preview_metadata(path: str) -> tuple[tuple[int, ...], np.dtype, int]:
    with tiff.TiffFile(path) as tif:
        page = tif.pages[0]
        shape = tuple(page.shape)
        dtype = np.dtype(page.dtype)
        samples = 1
        if len(shape) == 3:
            if shape[-1] in (3, 4):
                samples = shape[-1]
            elif shape[0] in (3, 4):
                samples = shape[0]
        return shape, dtype, samples


def read_preview_background(
    path: str,
    expected_shape: tuple[int, int],
    downsample_factor: int = 1,
    allow_full_read: bool = True,
) -> np.ndarray | None:
    f = max(1, int(downsample_factor))
    bg = None
    if f > 1:
        try:
            mm = tiff.memmap(path)
            if mm.ndim > 3:
                mm = mm[0]
            if mm.ndim == 2:
                bg = np.asarray(mm[::f, ::f])
            elif mm.ndim == 3 and mm.shape[0] in (3, 4) and mm.shape[-1] not in (3, 4):
                bg = np.asarray(mm[:, ::f, ::f])
            elif mm.ndim == 3:
                bg = np.asarray(mm[::f, ::f, :])
        except Exception:
            bg = None

        if bg is None:
            try:
                bg = read_tiled_tiff_preview(path, f)
            except Exception as exc:
                print(f"[WARN] Could not read tiled TIFF preview via Zarr: {exc}", flush=True)

    if bg is None:
        if not allow_full_read:
            return None
        bg = tiff.imread(path)
    if bg.ndim > 3:
        bg = bg[0]
    bg_rgb = to_uint8_rgb(bg)
    expected_small_shape = (
        (expected_shape[0] + f - 1) // f,
        (expected_shape[1] + f - 1) // f,
    )
    if bg_rgb.shape[:2] not in (expected_shape, expected_small_shape):
        raise ValueError(
            f"Preview background shape {bg_rgb.shape[:2]} does not match mask shape {expected_shape} "
            f"or downsampled shape {expected_small_shape}. "
            "Expected crop_roi.tif aligned with labels."
        )
    return bg_rgb


def colorize_cluster_mask(
    cluster_mask: np.ndarray,
    default_value: int,
    palette_by_value: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    out = np.zeros(cluster_mask.shape + (3,), dtype=np.uint8)
    cluster_ids = np.unique(cluster_mask)
    cluster_ids = cluster_ids[cluster_ids != default_value]
    for cid in cluster_ids:
        value = int(cid)
        color = (
            palette_by_value.get(value, np.array([51, 51, 51], dtype=np.uint8))
            if palette_by_value is not None
            else DEFAULT_PALETTE[(max(1, value) - 1) % len(DEFAULT_PALETTE)]
        )
        out[cluster_mask == cid] = color
    return out


def add_preview_legend(image: np.ndarray, legend_items: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    if not legend_items:
        return image
    row_height = 28
    legend_height = 18 + row_height * len(legend_items)
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + legend_height), (250, 250, 250))
    canvas.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.shape[0] + 10
    for label, color in legend_items:
        draw.rectangle((12, y + 3, 28, y + 19), fill=tuple(int(v) for v in color))
        draw.text((36, y + 2), label, fill=(25, 25, 25))
        y += row_height
    return np.asarray(canvas)


def write_preview_overlay_png(
    cluster_mask: np.ndarray,
    out_png: str,
    factor_if_large: int,
    size_threshold_mb: float,
    default_value: int,
    preview_background_path: str,
    alpha: float,
    palette_by_value: dict[int, np.ndarray] | None = None,
    legend_items: list[tuple[str, tuple[int, int, int]]] | None = None,
) -> tuple[int, int]:
    bg_shape, bg_dtype, bg_samples = _preview_metadata(preview_background_path)
    bg_pixels = int(np.prod(bg_shape[:2]))
    bg_est_bytes = bg_pixels * bg_samples * int(bg_dtype.itemsize)
    est_bytes = int(bg_est_bytes + cluster_mask.nbytes)
    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = int(factor_if_large) if est_bytes > threshold_bytes else 1
    use_factor = max(1, use_factor)

    mask_small = downsample_nearest(cluster_mask, use_factor)
    bg_small = read_preview_background(
        preview_background_path,
        cluster_mask.shape,
        downsample_factor=use_factor,
        allow_full_read=(use_factor == 1),
    )
    if bg_small is None:
        bg_small = np.full(mask_small.shape + (3,), 245, dtype=np.uint8)
    elif bg_small.shape[:2] != mask_small.shape:
        bg_small = downsample_nearest(bg_small, use_factor)

    overlay_rgb = colorize_cluster_mask(
        mask_small,
        default_value=default_value,
        palette_by_value=palette_by_value,
    )
    fg = mask_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    a = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - a) * out[fg] + a * overlay_rgb[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    out = add_preview_legend(out, legend_items or [])
    Image.fromarray(out).save(out_png)
    return use_factor, est_bytes


def main():
    ap = argparse.ArgumentParser(
        description="Remap a labeled mask to a cluster-valued mask using cluster.csv, and optionally write a preview."
    )
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF (e.g., labels_cyto.tif)")
    ap.add_argument("--map", required=True, help='CSV mapping with columns "label","cluster"')
    ap.add_argument("--out", required=True, help="Output TIFF cluster mask (pixel values = cluster)")
    ap.add_argument("--uncertainty-out", default=None,
                    help="Optional uint8 TIFF where 0 is accepted/background and positive values are abstention reasons.")
    ap.add_argument("--summary", default=None,
                    help="Optional JSON summary of accepted and abstained observations.")
    ap.add_argument("--default", type=int, default=0,
                    help="Value for labels not found in CSV (default 0)")
    ap.add_argument("--compress", default="none", choices=["none", "zlib", "lzma"],
                    help="TIFF compression (default none)")
    ap.add_argument("--block-rows", type=int, default=1024,
                    help="Rows per chunk for memory-safe mask remapping (default 1024).")

    ap.add_argument("--preview", default=None,
                    help="Optional preview image path (e.g., cluster_mask_preview.png)")
    ap.add_argument("--uncertainty-preview", default=None,
                    help="Optional H&E overlay showing categorical abstention reasons.")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor used only when preview image is larger than threshold.")
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB).")
    ap.add_argument("--preview-background", default=None,
                    help="Background image TIFF for preview overlay (typically crop_roi.tif).")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored cluster mask in preview (0..1).")

    args = ap.parse_args()

    labels = open_mask_2d(args.mask)
    df = load_map(args.map, default_value=args.default)
    uncertainty_df = load_uncertainty_map(args.map)

    max_lab = int(labels.max())
    if max_lab > 50_000_000:
        raise ValueError(
            f"Max label is {max_lab:,}, LUT would be huge. "
            "If your labels are sparse with giant IDs, relabel mask to 1..N first."
        )

    lab_ids = df["label"].to_numpy(dtype=np.int64)
    clus = df["cluster"].to_numpy(dtype=np.int64)
    valid = (lab_ids >= 0) & (lab_ids <= max_lab)

    out_max_value = int(max(args.default, int(clus[valid].max()) if valid.any() else args.default))
    if args.default < 0 or (valid.any() and int(clus[valid].min()) < 0):
        lut_dtype = np.int32 if out_max_value <= np.iinfo(np.int32).max else np.int64
    else:
        lut_dtype = smallest_mask_dtype(out_max_value)

    # LUT (look-up table): lut[label] = cluster. Keep the output dtype tied to cluster IDs,
    # not the potentially large cell-label IDs.
    lut = np.full(max_lab + 1, args.default, dtype=lut_dtype)
    lut[lab_ids[valid]] = clus[valid]

    if args.compress != "none":
        print(
            f"[INFO] Chunked cluster-mask writer uses uncompressed BigTIFF; "
            f"requested compression '{args.compress}' is ignored to keep RAM bounded."
        )

    out_tif, remap_stats = remap_mask_chunked(
        labels,
        lut,
        args.out,
        out_dtype=lut_dtype,
        default_value=args.default,
        block_rows=args.block_rows,
    )

    uncertainty_tif = None
    uncertainty_stats = None
    if args.uncertainty_out:
        uncertainty_lut = np.zeros(max_lab + 1, dtype=np.uint8)
        uncertainty_labels = uncertainty_df["label"].to_numpy(dtype=np.int64)
        uncertainty_codes = uncertainty_df["uncertainty_code"].to_numpy(dtype=np.uint8)
        uncertainty_valid = (uncertainty_labels >= 0) & (uncertainty_labels <= max_lab)
        uncertainty_lut[uncertainty_labels[uncertainty_valid]] = uncertainty_codes[uncertainty_valid]
        uncertainty_tif, uncertainty_stats = remap_mask_chunked(
            labels,
            uncertainty_lut,
            args.uncertainty_out,
            out_dtype=np.dtype(np.uint8),
            default_value=0,
            block_rows=args.block_rows,
        )

    present_labels = remap_stats["present_labels"]
    present_labels = present_labels[(present_labels >= 0) & (present_labels <= max_lab)]
    mapped_label_ids = np.intersect1d(present_labels, lab_ids[valid], assume_unique=False)
    expected_clusters = np.unique(lut[mapped_label_ids])
    expected_clusters = expected_clusters[expected_clusters != args.default]
    observed_clusters = remap_stats["observed_clusters"]

    if mapped_label_ids.size == 0:
        raise ValueError(
            "No mask labels matched the clustering map. "
            "Check that 'label' IDs in cluster CSV match IDs in the mask."
        )
    if expected_clusters.size > 1 and observed_clusters.size <= 1:
        raise ValueError(
            "Cluster mask collapsed to one group although mapping contains multiple clusters. "
            f"Expected cluster IDs in mapped labels: {expected_clusters.tolist()}, "
            f"observed in output mask: {observed_clusters.tolist()}"
        )

    preview_factor_used = None
    preview_estimated_mb = None
    # preview
    if args.preview:
        if args.preview_background:
            preview_factor_used, preview_est_bytes = write_preview_overlay_png(
                out_tif,
                args.preview,
                factor_if_large=args.preview_factor,
                size_threshold_mb=args.preview_threshold_mb,
                default_value=args.default,
                preview_background_path=args.preview_background,
                alpha=args.preview_alpha,
            )
            preview_estimated_mb = preview_est_bytes / (1024.0 * 1024.0)
        else:
            small = downsample_nearest(out_tif, factor=max(1, int(args.preview_factor)))
            mx = int(small.max())
            if mx == 0:
                view = small.astype(np.uint8)
            else:
                view = (small.astype(np.float32) * (255.0 / mx)).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(view).save(args.preview)
            preview_factor_used = max(1, int(args.preview_factor))
            preview_estimated_mb = out_tif.nbytes / (1024.0 * 1024.0)

    uncertainty_preview_factor = None
    if args.uncertainty_preview:
        if uncertainty_tif is None:
            raise ValueError("--uncertainty-preview requires --uncertainty-out")
        if not args.preview_background:
            raise ValueError("--uncertainty-preview requires --preview-background")
        uncertainty_preview_factor, _ = write_preview_overlay_png(
            uncertainty_tif,
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
        for code, count in uncertainty_df["uncertainty_code"].value_counts().sort_index().items()
    }
    observations = int(len(uncertainty_df))
    abstained_observations = int((uncertainty_df["uncertainty_code"] > 0).sum())
    if args.summary:
        summary = {
            "schema_version": 2,
            "observation_type": "contextual_cell",
            "observations": observations,
            "accepted_observations": observations - abstained_observations,
            "abstained_observations": abstained_observations,
            "accepted_observation_fraction": float(
                (observations - abstained_observations) / max(1, observations)
            ),
            "uncertainty_status_counts": status_counts,
            "uncertainty_mask_codes": {
                str(code): name for code, name in UNCERTAINTY_STATUS_NAMES.items()
            },
            "foreground_pixels": int(remap_stats["foreground_px"]),
            "interpretable_cluster_pixels": int(remap_stats["mapped_px"]),
            "abstained_pixels": int(
                uncertainty_stats["mapped_px"] if uncertainty_stats is not None else 0
            ),
            "preview_downsample_factor": int(preview_factor_used or 1),
            "uncertainty_preview_downsample_factor": int(uncertainty_preview_factor or 1),
        }
        Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # report
    fg = int(remap_stats["foreground_px"])
    mapped = int(remap_stats["mapped_px"])
    print(
        f"[INFO] mask labels present={len(present_labels)} "
        f"mapped_labels={len(mapped_label_ids)} "
        f"unmapped_labels={len(present_labels) - len(mapped_label_ids)}"
    )
    print(f"[INFO] expected clusters in mapped labels: {expected_clusters.tolist()}")
    print(f"[INFO] observed clusters in output mask: {observed_clusters.tolist()}")
    print(
        f"[INFO] uncertainty observations: accepted={observations - abstained_observations:,} "
        f"abstained={abstained_observations:,} status_counts={status_counts}"
    )
    print(f"[INFO] wrote cluster mask: {args.out}")
    if args.preview:
        if args.preview_background:
            print(
                "[INFO] wrote overlay preview: "
                f"{args.preview} (factor={preview_factor_used}, "
                f"threshold_mb={args.preview_threshold_mb}, "
                f"estimated_mb={preview_estimated_mb:.1f})"
            )
        else:
            print(f"[INFO] wrote grayscale preview (x{preview_factor_used}): {args.preview}")
    print(f"[INFO] foreground_px={fg:,}  mapped_px={mapped:,}  ({mapped / max(1, fg):.3f})")


if __name__ == "__main__":
    main()
