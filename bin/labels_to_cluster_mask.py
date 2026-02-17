#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image

DEFAULT_PALETTE = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=np.uint8)


def read_mask_2d(path: str) -> np.ndarray:
    m = tiff.imread(path)
    if m.ndim > 2:
        m = m[0]
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask TIFF, got shape={m.shape}")
    if not np.issubdtype(m.dtype, np.integer):
        m = m.astype(np.int64)
    return m


def load_map(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    if "label" not in df.columns or "cluster" not in df.columns:
        raise ValueError('CSV must contain columns: "label","cluster"')

    df = df[["label", "cluster"]].copy()
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
    if cl.isna().any():
        bad = df.loc[cl.isna(), "cluster"].head(10).tolist()
        raise ValueError(
            "Cluster column contains non-numeric values (cannot write as numeric mask). "
            f"Examples: {bad}"
        )
    df["cluster"] = cl.astype(np.int64)

    dup_conflicts = df.groupby("label")["cluster"].nunique()
    dup_conflicts = dup_conflicts[dup_conflicts > 1]
    if len(dup_conflicts) > 0:
        bad_ids = dup_conflicts.index[:10].tolist()
        raise ValueError(
            "Same label is assigned to multiple clusters in mapping CSV. "
            f"Example label IDs: {bad_ids}"
        )
    df = df.drop_duplicates(subset=["label"], keep="first")
    return df


def downsample_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    f = int(factor)
    if f <= 1:
        return img
    return img[::f, ::f]


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


def read_preview_background(path: str, expected_shape: tuple[int, int]) -> np.ndarray:
    bg = tiff.imread(path)
    if bg.ndim > 3:
        bg = bg[0]
    bg_rgb = to_uint8_rgb(bg)
    if bg_rgb.shape[:2] != expected_shape:
        raise ValueError(
            f"Preview background shape {bg_rgb.shape[:2]} does not match mask shape {expected_shape}. "
            "Expected crop_roi.tif aligned with labels."
        )
    return bg_rgb


def colorize_cluster_mask(cluster_mask: np.ndarray, default_value: int) -> np.ndarray:
    out = np.zeros(cluster_mask.shape + (3,), dtype=np.uint8)
    cluster_ids = np.unique(cluster_mask)
    cluster_ids = cluster_ids[cluster_ids != default_value]
    for idx, cid in enumerate(cluster_ids):
        out[cluster_mask == cid] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return out


def write_preview_overlay_png(
    cluster_mask: np.ndarray,
    out_png: str,
    factor_if_large: int,
    size_threshold_mb: float,
    default_value: int,
    preview_background_path: str,
    alpha: float,
) -> tuple[int, int]:
    bg = read_preview_background(preview_background_path, cluster_mask.shape)
    est_bytes = int(bg.nbytes + cluster_mask.nbytes)
    threshold_bytes = int(float(size_threshold_mb) * 1024 * 1024)
    use_factor = int(factor_if_large) if est_bytes > threshold_bytes else 1
    use_factor = max(1, use_factor)

    bg_small = downsample_nearest(bg, use_factor)
    mask_small = downsample_nearest(cluster_mask, use_factor)

    overlay_rgb = colorize_cluster_mask(mask_small, default_value=default_value)
    fg = mask_small != default_value

    out = bg_small.astype(np.float32, copy=True)
    a = float(max(0.0, min(1.0, alpha)))
    out[fg] = (1.0 - a) * out[fg] + a * overlay_rgb[fg].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out).save(out_png)
    return use_factor, est_bytes


def main():
    ap = argparse.ArgumentParser(
        description="Remap a labeled mask to a cluster-valued mask using cluster.csv, and optionally write a preview."
    )
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF (e.g., labels_cyto.tif)")
    ap.add_argument("--map", required=True, help='CSV mapping with columns "label","cluster"')
    ap.add_argument("--out", required=True, help="Output TIFF cluster mask (pixel values = cluster)")
    ap.add_argument("--default", type=int, default=0,
                    help="Value for labels not found in CSV (default 0)")
    ap.add_argument("--compress", default="none", choices=["none", "zlib", "lzma"],
                    help="TIFF compression (default none)")

    ap.add_argument("--preview", default=None,
                    help="Optional preview image path (e.g., cluster_mask_preview.png)")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor used only when preview image is larger than threshold.")
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0,
                    help="Downsample preview only when estimated image+mask size exceeds this threshold (MB).")
    ap.add_argument("--preview-background", default=None,
                    help="Background image TIFF for preview overlay (typically crop_roi.tif).")
    ap.add_argument("--preview-alpha", type=float, default=0.45,
                    help="Overlay alpha for colored cluster mask in preview (0..1).")

    args = ap.parse_args()

    labels = read_mask_2d(args.mask)
    df = load_map(args.map)

    max_lab = int(labels.max())
    if max_lab > 50_000_000:
        raise ValueError(
            f"Max label is {max_lab:,}, LUT would be huge. "
            "If your labels are sparse with giant IDs, relabel mask to 1..N first."
        )

    # LUT (look-up table): lut[label] = cluster
    lut = np.full(max_lab + 1, args.default, dtype=np.int64)
    lab_ids = df["label"].to_numpy(dtype=np.int64)
    clus = df["cluster"].to_numpy(dtype=np.int64)

    valid = (lab_ids >= 0) & (lab_ids <= max_lab)
    lut[lab_ids[valid]] = clus[valid]

    out = lut[labels]

    present_labels = np.unique(labels[labels != 0]).astype(np.int64, copy=False)
    present_labels = present_labels[(present_labels >= 0) & (present_labels <= max_lab)]
    mapped_label_ids = np.intersect1d(present_labels, lab_ids[valid], assume_unique=False)
    expected_clusters = np.unique(lut[mapped_label_ids])
    expected_clusters = expected_clusters[expected_clusters != args.default]
    observed_clusters = np.unique(out[(labels != 0) & (out != args.default)])

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

    # choose dtype
    mx = int(out.max())
    if mx <= np.iinfo(np.uint16).max:
        out_tif = out.astype(np.uint16)
    elif mx <= np.iinfo(np.uint32).max:
        out_tif = out.astype(np.uint32)
    else:
        out_tif = out.astype(np.uint64)

    comp = None if args.compress == "none" else args.compress
    tiff.imwrite(args.out, out_tif, compression=comp)

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

    # report
    fg = int((labels != 0).sum())
    mapped = int((out != args.default).sum())
    print(
        f"[INFO] mask labels present={len(present_labels)} "
        f"mapped_labels={len(mapped_label_ids)} "
        f"unmapped_labels={len(present_labels) - len(mapped_label_ids)}"
    )
    print(f"[INFO] expected clusters in mapped labels: {expected_clusters.tolist()}")
    print(f"[INFO] observed clusters in output mask: {observed_clusters.tolist()}")
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
