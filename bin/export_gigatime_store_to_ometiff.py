#!/usr/bin/env python3
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile
import zarr


def parse_args():
    ap = argparse.ArgumentParser(description="Export a GigaTIME Zarr/TIFF store to OME-TIFF blockwise.")
    ap.add_argument("--input", required=True, help="Input GigaTIME directory, .zarr store, or .ome.tif")
    ap.add_argument("--output", required=True, help="Output OME-TIFF path")
    ap.add_argument("--output-channels", default="", help="Comma-separated subset of marker channels to export.")
    ap.add_argument("--tile-size", type=int, default=2048, help="Output tile size")
    ap.add_argument("--compression", default="deflate", help="TIFF compression codec")
    ap.add_argument("--predictor", action="store_true", help="Enable TIFF predictor")
    return ap.parse_args()


def resolve_output_channels(spec: str | None, available: list[str]) -> tuple[list[int], list[str]]:
    if not spec or not str(spec).strip():
        return list(range(len(available))), list(available)

    chosen: list[str] = []
    seen: set[str] = set()
    for raw in str(spec).split(","):
        name = raw.strip()
        if not name:
            continue
        match = next((cand for cand in available if cand.lower() == name.lower()), None)
        if match is None:
            raise ValueError(
                f"Unknown channel '{name}'. Available channels: {', '.join(available)}"
            )
        if match not in seen:
            chosen.append(match)
            seen.add(match)
    if not chosen:
        raise ValueError("No valid export channels were selected")
    return [available.index(name) for name in chosen], chosen


def _resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir() and path.suffix.lower() != ".zarr":
        for cand in (
            path / "gigatime_probs.zarr",
            path / "gigatime_probs.ome.tif",
            path / "gigatime_probs.ome.tiff",
            path / "gigatime_probs.tif",
        ):
            if cand.exists():
                return cand
    return path


def _load_sidecar_json(image_path: Path, filename: str, default):
    sidecar = image_path.with_name(filename)
    if not sidecar.exists():
        return default
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return default


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


def _extract_tiff_channel_names(tf: tifffile.TiffFile, n_channels: int) -> list[str] | None:
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
    return None


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
        return zobj, getattr(zobj, "attrs", {}), {}
    root_attrs = getattr(zobj, "attrs", {})
    for key in ("0", "probs"):
        try:
            child = zobj[key]
            if hasattr(child, "shape") and hasattr(child, "dtype"):
                return child, getattr(child, "attrs", {}), root_attrs
        except Exception:
            pass
    for key in getattr(zobj, "keys", lambda: [])():
        try:
            child = zobj[key]
            if hasattr(child, "shape") and hasattr(child, "dtype"):
                return child, getattr(child, "attrs", {}), root_attrs
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


class LazyGigaTIMEReader:
    def __init__(self, input_path: Path):
        self.path = _resolve_input_path(str(input_path))
        self.storage_meta = _load_sidecar_json(self.path, "gigatime_metadata.json", {})
        self.channel_names = None
        self.scale_max = None
        self.tf = None
        self.series = None
        self.root_attrs = {}
        self.array_attrs = {}

        if self.path.is_dir() or self.path.suffix.lower() == ".zarr":
            zobj = zarr.open(str(self.path), mode="r")
            self.arr, self.array_attrs, self.root_attrs = _select_zarr_array(zobj)
            self.axes = _axes_from_zarr_attrs(self.root_attrs, self.array_attrs, tuple(int(v) for v in self.arr.shape))
        else:
            self.tf = tifffile.TiffFile(str(self.path))
            self.series = self.tf.series[0]
            self.arr = zarr.open(self.series.aszarr(), mode="r")
            self.axes = _normalize_axes(getattr(self.series, "axes", ""), tuple(int(v) for v in self.arr.shape))

        shape = tuple(int(v) for v in self.arr.shape)
        if len(shape) == 2:
            self.channels = 1
            self.height, self.width = shape
            self.axes = "CYX"
        elif len(shape) == 3 and set(self.axes) == set("CYX"):
            self.channels = int(shape[self.axes.index("C")])
            self.height = int(shape[self.axes.index("Y")])
            self.width = int(shape[self.axes.index("X")])
        else:
            raise ValueError(f"Unsupported image axes '{self.axes}' for shape {shape}")

        if self.tf is not None:
            self.channel_names = _extract_tiff_channel_names(self.tf, self.channels)
        if self.channel_names is None:
            self.channel_names = _extract_channel_names_from_attrs(self.root_attrs, self.channels)
        if self.channel_names is None:
            self.channel_names = _extract_channel_names_from_attrs(self.array_attrs, self.channels)
        if self.channel_names is None:
            self.channel_names = [f"channel_{idx + 1}" for idx in range(self.channels)]

        sidecar_names = _load_sidecar_json(self.path, "gigatime_channels.json", None)
        if isinstance(sidecar_names, list) and len(sidecar_names) == self.channels:
            self.channel_names = [str(v).strip() or f"channel_{idx + 1}" for idx, v in enumerate(sidecar_names)]

        if str(self.storage_meta.get("output_dtype", "")).lower() == "uint16":
            self.scale_max = float(self.storage_meta.get("storage_scale_max") or 65535.0)

    def close(self):
        if self.tf is not None:
            try:
                self.tf.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def read_tile_yxc(self, y0: int, y1: int, x0: int, x1: int, channel_indices: list[int] | None = None) -> np.ndarray:
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
        if self.scale_max and self.scale_max > 0:
            block = np.clip(block, 0, self.scale_max).astype(np.uint16, copy=False)
        elif block.dtype != np.uint16:
            block = np.clip(block, 0.0, 1.0)
            block = np.round(block * 65535.0).astype(np.uint16)
        if channel_indices is not None:
            block = block[channel_indices, ...]
        return np.moveaxis(block, 0, -1)


def iter_tiles(height: int, width: int, tile_size: int):
    for y0 in range(0, int(height), int(tile_size)):
        y1 = min(int(height), y0 + int(tile_size))
        for x0 in range(0, int(width), int(tile_size)):
            x1 = min(int(width), x0 + int(tile_size))
            yield y0, y1, x0, x1


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with LazyGigaTIMEReader(input_path) as reader:
        channel_indices, channel_names = resolve_output_channels(args.output_channels, reader.channel_names)
        h, w, c = reader.height, reader.width, len(channel_names)
        meta = reader.storage_meta or {}
        tiff_kwargs = {
            "shape": (h, w, c),
            "dtype": np.uint16,
            "tile": (int(args.tile_size), int(args.tile_size)),
            "compression": args.compression,
            "planarconfig": "CONTIG",
            "photometric": "MINISBLACK",
            "metadata": {"axes": "YXS", "Channel": {"Name": channel_names}},
            "bigtiff": True,
        }
        if args.predictor:
            tiff_kwargs["predictor"] = True
        effective_mpp = meta.get("effective_mpp")
        if effective_mpp:
            res = float(10_000.0 / float(effective_mpp))
            tiff_kwargs["resolution"] = (res, res)
            tiff_kwargs["resolutionunit"] = "CENTIMETER"

        total_tiles = int(math.ceil(h / float(args.tile_size)) * math.ceil(w / float(args.tile_size)))

        def tile_iterator():
            done = 0
            for y0, y1, x0, x1 in iter_tiles(h, w, args.tile_size):
                done += 1
                if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                    print(f"[INFO] export tile {done}/{total_tiles}", flush=True)
                yield reader.read_tile_yxc(y0, y1, x0, x1, channel_indices=channel_indices)

        tifffile.imwrite(output_path, tile_iterator(), **tiff_kwargs)
        print(f"[OK] wrote {output_path} shape={(h, w, c)} channels={channel_names}", flush=True)


if __name__ == "__main__":
    main()
