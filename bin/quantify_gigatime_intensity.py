#!/usr/bin/env python3
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile


def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantify per-label marker intensities from a GigaTIME multichannel TIFF."
    )
    ap.add_argument("--image", required=True, help="Input GigaTIME TIFF/OME-TIFF")
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF")
    ap.add_argument("--mask-name", required=True, help="Mask label used in outputs, e.g. nuclei or cyto")
    ap.add_argument("--out-mean-csv", required=True, help="Output CSV with mean intensity per marker")
    ap.add_argument("--out-stats-csv", required=True, help="Output CSV with area/mean/sum/max per marker")
    ap.add_argument("--out-summary-json", required=True, help="Output JSON summary")
    return ap.parse_args()


def _extract_channel_names(tf: tifffile.TiffFile, n_channels: int) -> list[str]:
    ome_xml = getattr(tf, "ome_metadata", None)
    if ome_xml:
        try:
            root = ET.fromstring(ome_xml)
            ns = {}
            if root.tag.startswith("{"):
                ns["ome"] = root.tag.split("}", 1)[0][1:]
            pixels_path = ".//ome:Pixels" if ns else ".//Pixels"
            channel_path = "ome:Channel" if ns else "Channel"
            pixels = root.find(pixels_path, ns)
            if pixels is not None:
                names = []
                for idx, ch in enumerate(pixels.findall(channel_path, ns)):
                    names.append((ch.attrib.get("Name") or f"channel_{idx + 1}").strip())
                if len(names) == n_channels:
                    return names
        except Exception:
            pass
    return [f"channel_{idx + 1}" for idx in range(n_channels)]


def _normalize_image_axes(arr: np.ndarray, axes: str | None) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim != 3:
        raise ValueError(f"Unsupported image shape after squeeze: {arr.shape}")

    axes = (axes or "").upper()
    if "S" in axes and "C" not in axes:
        axes = axes.replace("S", "C")
    if not axes:
        smallest_dim = int(np.argmin(arr.shape))
        if arr.shape[smallest_dim] <= 64:
            if smallest_dim == 0:
                axes = "CYX"
            elif smallest_dim == 2:
                axes = "YXC"
            else:
                axes = "YCX"
        elif arr.shape[0] <= 64 and arr.shape[1] > 64 and arr.shape[2] > 64:
            axes = "CYX"
        elif arr.shape[-1] <= 64 and arr.shape[0] > 64 and arr.shape[1] > 64:
            axes = "YXC"
        else:
            axes = "CYX"

    if set(axes) == set("CYX"):
        order = [axes.index("C"), axes.index("Y"), axes.index("X")]
        return np.moveaxis(arr, order, (0, 1, 2))

    raise ValueError(f"Unsupported image axes '{axes}' for shape {arr.shape}")


def _read_multichannel_image(path: str) -> tuple[np.ndarray, list[str]]:
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        axes = ""
        try:
            axes = tf.series[0].axes
        except Exception:
            axes = ""
        image = _normalize_image_axes(arr, axes)
        channel_names = _extract_channel_names(tf, image.shape[0])
    return image.astype(np.float32, copy=False), channel_names


def _read_label_mask(path: str) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        arr = np.asarray(tf.asarray())
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported mask shape after squeeze: {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int64, copy=False)
    return arr


def quantify(image_cyx: np.ndarray, mask_yx: np.ndarray, channel_names: list[str]) -> tuple[list[dict], list[dict]]:
    if image_cyx.ndim != 3:
        raise ValueError(f"Expected CYX image, got shape {image_cyx.shape}")
    if mask_yx.ndim != 2:
        raise ValueError(f"Expected YX mask, got shape {mask_yx.shape}")
    if image_cyx.shape[1:] != mask_yx.shape:
        raise ValueError(
            f"Image/mask shape mismatch: image YX={image_cyx.shape[1:]}, mask={mask_yx.shape}"
        )

    mask_flat = mask_yx.reshape(-1)
    positive = mask_flat > 0
    if not np.any(positive):
        return [], []

    labels_sorted, inverse = np.unique(mask_flat[positive].astype(np.int64, copy=False), return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64, copy=False)
    n_labels = len(labels_sorted)

    means_by_channel: list[np.ndarray] = []
    sums_by_channel: list[np.ndarray] = []
    max_by_channel: list[np.ndarray] = []

    for ch in range(image_cyx.shape[0]):
        values = image_cyx[ch].reshape(-1)[positive].astype(np.float64, copy=False)
        sums = np.bincount(inverse, weights=values, minlength=n_labels)
        means = sums / counts
        maxes = np.full(n_labels, -np.inf, dtype=np.float64)
        np.maximum.at(maxes, inverse, values)
        means_by_channel.append(means)
        sums_by_channel.append(sums)
        max_by_channel.append(maxes)

    mean_rows: list[dict] = []
    stats_rows: list[dict] = []
    for idx, label_id in enumerate(labels_sorted.tolist()):
        base = {
            "label_id": int(label_id),
            "area_px": int(counts[idx]),
        }
        mean_row = dict(base)
        stats_row = dict(base)
        for ch_name, means, sums, maxes in zip(channel_names, means_by_channel, sums_by_channel, max_by_channel):
            safe_name = ch_name.strip() or "unnamed"
            mean_row[safe_name] = float(means[idx])
            stats_row[f"{safe_name}__mean"] = float(means[idx])
            stats_row[f"{safe_name}__sum"] = float(sums[idx])
            stats_row[f"{safe_name}__max"] = float(maxes[idx])
        mean_rows.append(mean_row)
        stats_rows.append(stats_row)

    return mean_rows, stats_rows


def write_csv(path: str, rows: list[dict]) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path_obj.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label_id", "area_px"])
        return

    fieldnames = list(rows[0].keys())
    with path_obj.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    image_cyx, channel_names = _read_multichannel_image(args.image)
    mask_yx = _read_label_mask(args.mask)
    mean_rows, stats_rows = quantify(image_cyx, mask_yx, channel_names)

    write_csv(args.out_mean_csv, mean_rows)
    write_csv(args.out_stats_csv, stats_rows)

    summary = {
        "mask_name": args.mask_name,
        "image": str(Path(args.image).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "image_shape_cyx": list(map(int, image_cyx.shape)),
        "mask_shape_yx": list(map(int, mask_yx.shape)),
        "channel_names": channel_names,
        "objects_quantified": len(mean_rows),
    }
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[OK] quantified {len(mean_rows)} objects for mask={args.mask_name} "
        f"channels={len(channel_names)} image={args.image}"
    )


if __name__ == "__main__":
    main()
