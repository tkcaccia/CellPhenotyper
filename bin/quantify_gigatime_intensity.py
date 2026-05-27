#!/usr/bin/env python3
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import zarr


def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantify per-label marker intensities from a GigaTIME multichannel image store."
    )
    ap.add_argument("--image", required=True, help="Input GigaTIME TIFF, Zarr store, or GigaTIME output directory")
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF")
    ap.add_argument("--mask-name", required=True, help="Mask label used in outputs, e.g. nuclei or cyto")
    ap.add_argument("--out-quant-csv", required=True, help="Output wide CSV with mcMicro-like per-object quantification")
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
    try:
        desc = tf.pages[0].description
        if isinstance(desc, str) and desc.strip().startswith("{"):
            meta = json.loads(desc)
            names = meta.get("Channel", {}).get("Name")
            if isinstance(names, list) and len(names) == n_channels:
                return [str(v).strip() or f"channel_{idx + 1}" for idx, v in enumerate(names)]
    except Exception:
        pass
    return [f"channel_{idx + 1}" for idx in range(n_channels)]


def _normalize_axes(axes: str | None, shape: tuple[int, ...]) -> str:
    axes = (axes or "").upper()
    if "S" in axes and "C" not in axes:
        axes = axes.replace("S", "C")
    if not axes:
        smallest_dim = int(np.argmin(shape))
        if shape[smallest_dim] <= 64:
            if smallest_dim == 0:
                axes = "CYX"
            elif smallest_dim == 2:
                axes = "YXC"
            else:
                axes = "YCX"
        elif shape[0] <= 64 and shape[1] > 64 and shape[2] > 64:
            axes = "CYX"
        elif shape[-1] <= 64 and shape[0] > 64 and shape[1] > 64:
            axes = "YXC"
        else:
            axes = "CYX"
    return axes


def _load_storage_metadata(image_path: str) -> dict:
    sidecar = Path(image_path).with_name("gigatime_metadata.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_gigatime_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.is_dir() and path.suffix.lower() != ".zarr":
        candidates = [
            path / "gigatime_probs.zarr",
            path / "gigatime_probs.ome.tif",
            path / "gigatime_probs.ome.tiff",
            path / "gigatime_probs.tif",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No GigaTIME image found in directory: {path}")
    return path


def _extract_channel_names_from_attrs(attrs, n_channels: int) -> list[str] | None:
    try:
        omero = attrs.get("omero")
        if isinstance(omero, dict):
            channels = omero.get("channels")
            if isinstance(channels, list):
                names = [str(ch.get("label") or f"channel_{idx + 1}").strip() for idx, ch in enumerate(channels)]
                if len(names) == n_channels:
                    return names
    except Exception:
        pass
    try:
        names = attrs.get("channel_names")
        if isinstance(names, list) and len(names) == n_channels:
            return [str(v).strip() or f"channel_{idx + 1}" for idx, v in enumerate(names)]
    except Exception:
        pass
    return None


def _select_zarr_array(zobj):
    if hasattr(zobj, "shape") and hasattr(zobj, "dtype"):
        return zobj, getattr(zobj, "attrs", {})
    for key in ("0", "probs"):
        try:
            child = zobj[key]
            if hasattr(child, "shape") and hasattr(child, "dtype"):
                return child, getattr(zobj, "attrs", {})
        except Exception:
            pass
    for key in getattr(zobj, "keys", lambda: [])():
        try:
            child = zobj[key]
            if hasattr(child, "shape") and hasattr(child, "dtype"):
                return child, getattr(zobj, "attrs", {})
        except Exception:
            continue
    raise ValueError(f"Could not resolve a readable array from zarr object: {type(zobj)}")


def _axes_from_zarr_attrs(root_attrs, array_attrs, shape: tuple[int, ...]) -> str:
    axes = array_attrs.get("axes")
    if isinstance(axes, str):
        return _normalize_axes(axes, shape)
    multiscales = root_attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        try:
            axes_entries = multiscales[0].get("axes") or []
            axis_names = "".join(str(entry.get("name", "")).upper()[:1] for entry in axes_entries)
            if axis_names:
                return _normalize_axes(axis_names, shape)
        except Exception:
            pass
    return _normalize_axes(None, shape)


class LazyImageReader:
    def __init__(self, path: str):
        self.path = _resolve_gigatime_image_path(path)
        self.storage_meta = _load_storage_metadata(str(self.path))
        self.tf = None
        self.series = None
        self.root_attrs = {}
        self.array_attrs = {}

        if self.path.is_dir() or self.path.suffix.lower() == ".zarr":
            zobj = zarr.open(str(self.path), mode="r")
            self.arr, self.root_attrs = _select_zarr_array(zobj)
            self.array_attrs = getattr(self.arr, "attrs", {})
            self.axes = _axes_from_zarr_attrs(self.root_attrs, self.array_attrs, tuple(int(v) for v in self.arr.shape))
        else:
            self.tf = tifffile.TiffFile(str(self.path))
            self.series = self.tf.series[0]
            self.arr = zarr.open(self.series.aszarr(), mode="r")
            self.axes = _normalize_axes(getattr(self.series, "axes", ""), tuple(int(v) for v in self.arr.shape))

        shape = tuple(int(v) for v in self.arr.shape)
        if len(shape) == 2:
            self.height, self.width = shape
            self.channels = 1
            self.axes = "CYX"
        elif len(shape) == 3 and set(self.axes) == set("CYX"):
            self.channels = int(shape[self.axes.index("C")])
            self.height = int(shape[self.axes.index("Y")])
            self.width = int(shape[self.axes.index("X")])
        else:
            raise ValueError(f"Unsupported image axes '{self.axes}' for shape {shape}")

        channel_names = None
        if self.tf is not None:
            channel_names = _extract_channel_names(self.tf, self.channels)
        if channel_names is None:
            channel_names = _extract_channel_names_from_attrs(self.root_attrs, self.channels)
        if channel_names is None:
            channel_names = _extract_channel_names_from_attrs(self.array_attrs, self.channels)
        if channel_names is None:
            channel_names = [f"channel_{idx + 1}" for idx in range(self.channels)]
        self.channel_names = channel_names

        sidecar_channels = self.path.with_name("gigatime_channels.json")
        if sidecar_channels.exists():
            try:
                names = json.loads(sidecar_channels.read_text(encoding="utf-8"))
                if isinstance(names, list) and len(names) == self.channels:
                    self.channel_names = [str(v).strip() or f"channel_{i + 1}" for i, v in enumerate(names)]
            except Exception:
                pass

        self.scale_max = None
        if str(self.storage_meta.get("output_dtype", "")).lower() == "uint16":
            self.scale_max = float(self.storage_meta.get("storage_scale_max") or 65535.0)

    def close(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyImageReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if self.channels == 1:
            block = np.asarray(self.arr[y0:y1, x0:x1])[None, ...]
        elif self.axes == "CYX":
            block = np.asarray(self.arr[:, y0:y1, x0:x1])
        elif self.axes == "YXC":
            block = np.moveaxis(np.asarray(self.arr[y0:y1, x0:x1, :]), -1, 0)
        elif self.axes == "YCX":
            block = np.moveaxis(np.asarray(self.arr[y0:y1, :, x0:x1]), 1, 0)
        else:
            raise ValueError(f"Unsupported image axes '{self.axes}'")
        block = block.astype(np.float32, copy=False)
        if self.scale_max and self.scale_max > 0:
            block = block / self.scale_max
        return block


class LazyMaskReader:
    def __init__(self, path: str):
        self.tf = tifffile.TiffFile(path)
        self.series = self.tf.series[0]
        arr = zarr.open(self.series.aszarr(), mode="r")
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"Unsupported mask shape after squeeze: {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"Unsupported mask shape after squeeze: {arr.shape}")
        self.arr = arr
        self.height = int(arr.shape[0])
        self.width = int(arr.shape[1])

    def close(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyMaskReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, y0: int, y1: int, x0: int, x1: int, target_shape: tuple[int, int]) -> np.ndarray:
        ty, tx = target_shape
        if (self.height, self.width) == target_shape:
            out = np.asarray(self.arr[y0:y1, x0:x1])
        else:
            y_idx = np.minimum(
                np.round(np.linspace(y0 * self.height / ty, (y1 - 1) * self.height / ty, y1 - y0)).astype(np.int64),
                self.height - 1,
            )
            x_idx = np.minimum(
                np.round(np.linspace(x0 * self.width / tx, (x1 - 1) * self.width / tx, x1 - x0)).astype(np.int64),
                self.width - 1,
            )
            out = np.asarray(self.arr.oindex[y_idx, x_idx])
        out = np.squeeze(out)
        if out.ndim != 2:
            raise ValueError(f"Unsupported mask block shape after squeeze: {out.shape}")
        if not np.issubdtype(out.dtype, np.integer):
            out = np.rint(out).astype(np.int64, copy=False)
        return out


def _scan_max_label(mask_reader: LazyMaskReader, target_shape: tuple[int, int], block_size: int) -> int:
    ty, tx = target_shape
    max_label = 0
    for y0 in range(0, ty, block_size):
        y1 = min(ty, y0 + block_size)
        for x0 in range(0, tx, block_size):
            x1 = min(tx, x0 + block_size)
            block = mask_reader.read_block(y0, y1, x0, x1, target_shape)
            if block.size:
                max_label = max(max_label, int(block.max()))
    return max_label


def quantify_blockwise(
    image_reader: LazyImageReader,
    mask_reader: LazyMaskReader,
    channel_names: list[str],
    *,
    mask_name: str,
    block_size: int = 1024,
) -> tuple[list[dict], list[dict], list[dict]]:
    target_shape = (image_reader.height, image_reader.width)
    max_label = _scan_max_label(mask_reader, target_shape, block_size)
    if max_label <= 0:
        return [], [], []

    counts = np.zeros(max_label + 1, dtype=np.int64)
    sum_y = np.zeros(max_label + 1, dtype=np.float64)
    sum_x = np.zeros(max_label + 1, dtype=np.float64)
    min_y = np.full(max_label + 1, np.iinfo(np.int64).max, dtype=np.int64)
    min_x = np.full(max_label + 1, np.iinfo(np.int64).max, dtype=np.int64)
    max_y = np.full(max_label + 1, -1, dtype=np.int64)
    max_x = np.full(max_label + 1, -1, dtype=np.int64)
    sums_by_channel = np.zeros((len(channel_names), max_label + 1), dtype=np.float64)
    max_by_channel = np.full((len(channel_names), max_label + 1), -np.inf, dtype=np.float32)

    ty, tx = target_shape
    for y0 in range(0, ty, block_size):
        y1 = min(ty, y0 + block_size)
        for x0 in range(0, tx, block_size):
            x1 = min(tx, x0 + block_size)
            mask_block = mask_reader.read_block(y0, y1, x0, x1, target_shape)
            positive = mask_block > 0
            if not np.any(positive):
                continue
            labels = mask_block[positive].astype(np.int64, copy=False)
            counts += np.bincount(labels, minlength=max_label + 1)

            y_local, x_local = np.nonzero(positive)
            abs_y = y_local.astype(np.int64, copy=False) + int(y0)
            abs_x = x_local.astype(np.int64, copy=False) + int(x0)
            sum_y += np.bincount(labels, weights=abs_y.astype(np.float64, copy=False), minlength=max_label + 1)
            sum_x += np.bincount(labels, weights=abs_x.astype(np.float64, copy=False), minlength=max_label + 1)
            np.minimum.at(min_y, labels, abs_y)
            np.minimum.at(min_x, labels, abs_x)
            np.maximum.at(max_y, labels, abs_y)
            np.maximum.at(max_x, labels, abs_x)

            image_block = image_reader.read_block(y0, y1, x0, x1)
            for ch in range(image_block.shape[0]):
                values = image_block[ch][positive].astype(np.float64, copy=False)
                sums_by_channel[ch] += np.bincount(labels, weights=values, minlength=max_label + 1)
                np.maximum.at(max_by_channel[ch], labels, values)

    labels_sorted = np.flatnonzero(counts > 0)
    labels_sorted = labels_sorted[labels_sorted > 0]
    if labels_sorted.size == 0:
        return [], [], []

    quant_rows: list[dict] = []
    mean_rows: list[dict] = []
    stats_rows: list[dict] = []
    for label_id in labels_sorted.tolist():
        base = {
            "label_id": int(label_id),
            "mask_name": mask_name,
            "area_px": int(counts[label_id]),
            "centroid_y_px": float(sum_y[label_id] / counts[label_id]),
            "centroid_x_px": float(sum_x[label_id] / counts[label_id]),
            "bbox_ymin_px": int(min_y[label_id]),
            "bbox_xmin_px": int(min_x[label_id]),
            "bbox_ymax_px": int(max_y[label_id] + 1),
            "bbox_xmax_px": int(max_x[label_id] + 1),
        }
        mean_row = dict(base)
        stats_row = dict(base)
        quant_row = dict(base)
        for ch_idx, ch_name in enumerate(channel_names):
            safe_name = ch_name.strip() or "unnamed"
            mean_val = float(sums_by_channel[ch_idx, label_id] / counts[label_id])
            sum_val = float(sums_by_channel[ch_idx, label_id])
            max_val = float(max_by_channel[ch_idx, label_id])
            mean_row[safe_name] = mean_val
            stats_row[f"{safe_name}__mean"] = mean_val
            stats_row[f"{safe_name}__sum"] = sum_val
            stats_row[f"{safe_name}__max"] = max_val
            quant_row[f"{safe_name}__mean"] = mean_val
            quant_row[f"{safe_name}__sum"] = sum_val
            quant_row[f"{safe_name}__max"] = max_val
        quant_rows.append(quant_row)
        mean_rows.append(mean_row)
        stats_rows.append(stats_row)

    return quant_rows, mean_rows, stats_rows


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
    with LazyImageReader(args.image) as image_reader, LazyMaskReader(args.mask) as mask_reader:
        image_shape_cyx = [int(image_reader.channels), int(image_reader.height), int(image_reader.width)]
        mask_shape_yx = [int(mask_reader.height), int(mask_reader.width)]
        channel_names = list(image_reader.channel_names)
        quant_rows, mean_rows, stats_rows = quantify_blockwise(
            image_reader,
            mask_reader,
            channel_names,
            mask_name=args.mask_name,
        )

    write_csv(args.out_quant_csv, quant_rows)
    write_csv(args.out_mean_csv, mean_rows)
    write_csv(args.out_stats_csv, stats_rows)

    summary = {
        "mask_name": args.mask_name,
        "image": str(Path(args.image).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "image_shape_cyx": image_shape_cyx,
        "mask_shape_yx": mask_shape_yx,
        "channel_names": channel_names,
        "objects_quantified": len(quant_rows),
        "quantification_csv": str(Path(args.out_quant_csv).resolve()),
    }
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[OK] quantified {len(quant_rows)} objects for mask={args.mask_name} "
        f"channels={len(channel_names)} image={args.image}"
    )


if __name__ == "__main__":
    main()
