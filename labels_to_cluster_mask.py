#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff


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

    df["label"] = (
        df["label"].astype(str).str.strip().str.strip('"').str.strip("'").astype(np.int64)
    )

    cl = pd.to_numeric(df["cluster"], errors="coerce")
    if cl.isna().any():
        bad = df.loc[cl.isna(), "cluster"].head(10).tolist()
        raise ValueError(
            "Cluster column contains non-numeric values (cannot write as numeric mask). "
            f"Examples: {bad}"
        )
    df["cluster"] = cl.astype(np.int64)
    return df


def downsample_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Nearest-neighbor downsample by an integer factor.
    Keeps discrete labels/cluster IDs intact.
    """
    f = int(factor)
    if f <= 1:
        return img
    return img[::f, ::f]


def write_preview_png(cluster_mask: np.ndarray, out_png: str, factor: int):
    """
    Writes a small PNG preview (factor x smaller in each dimension).
    Stores cluster IDs as grayscale for quick QC.
    """
    small = downsample_nearest(cluster_mask, factor=factor)

    # Scale to 8-bit for viewing (NOT for analysis)
    mx = int(small.max())
    if mx == 0:
        view = small.astype(np.uint8)
    else:
        view = (small.astype(np.float32) * (255.0 / mx)).round().clip(0, 255).astype(np.uint8)

    tiff.imwrite(out_png, view)  # tifffile can write PNG if extension is .png


def main():
    ap = argparse.ArgumentParser(
        description="Remap a labeled mask to a cluster-valued mask using cluster.csv, and optionally write a preview."
    )
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF (e.g., labels_cyto.tif)")
    ap.add_argument("--map", required=True, help='CSV mapping with columns "label","cluster"')
    ap.add_argument("--out", required=True, help="Output TIFF cluster mask (pixel values = cluster)")
    ap.add_argument("--default", type=int, default=0,
                    help="Value for labels not found in CSV (default 0)")
    ap.add_argument("--compress", default="zlib", choices=["none", "zlib", "lzma"],
                    help="TIFF compression (default zlib)")

    ap.add_argument("--preview", default=None,
                    help="Optional preview image path (e.g., cluster_mask_preview.png)")
    ap.add_argument("--preview-factor", type=int, default=10,
                    help="Downsample factor for preview (default 10 => 10x smaller per dimension)")

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

    # preview
    if args.preview:
        write_preview_png(out_tif, args.preview, factor=args.preview_factor)

    # report
    fg = int((labels != 0).sum())
    mapped = int((out != args.default).sum())
    print(f"[INFO] wrote cluster mask: {args.out}")
    if args.preview:
        print(f"[INFO] wrote preview (x{args.preview_factor} downsample): {args.preview}")
    print(f"[INFO] foreground_px={fg:,}  mapped_px={mapped:,}  ({mapped / max(1, fg):.3f})")


if __name__ == "__main__":
    main()
