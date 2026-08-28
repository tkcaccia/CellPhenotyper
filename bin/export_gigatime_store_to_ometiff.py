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
    ap.add_argument("--output-dtype", choices=["auto", "uint8", "uint16"], default="auto",
                    help="OME-TIFF storage dtype. 'auto' preserves the GigaTIME store dtype when possible.")
    ap.add_argument("--predictor", action="store_true", help="Enable TIFF predictor")
    ap.add_argument("--no-pyramid", action="store_true",
                    help="Write only the full-resolution level instead of a QuPath-friendly pyramid.")
    ap.add_argument("--pyramid-cache-max-gib", type=float, default=0.75,
                    help="Maximum RAM to use for caching one lower-resolution pyramid level.")
    ap.add_argument("--verification-json", default="",
                    help="Optional path for a JSON verification report. Defaults to <output>.verify.json.")
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
    candidates = [image_path.with_name(filename)]
    if image_path.is_dir():
        candidates.append(image_path / filename)
    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            continue
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
        gigatime = attrs.get("gigatime")
        if isinstance(gigatime, dict):
            for key in ("store_channels", "channels", "model_channels"):
                names = gigatime.get(key)
                if isinstance(names, list) and len(names) == n_channels:
                    return [str(v).strip() or f"channel_{idx + 1}" for idx, v in enumerate(names)]
    except Exception:
        pass
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
        self.store_output_dtype = None
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
        self.store_output_dtype = str(
            self.storage_meta.get("output_dtype")
            or (self.root_attrs.get("gigatime") or {}).get("output_dtype")
            or self.array_attrs.get("output_dtype")
            or getattr(self.arr, "dtype", "uint16")
        ).lower()

    def resolve_export_dtype(self, requested: str) -> np.dtype:
        if requested != "auto":
            return np.dtype(requested)
        if self.store_output_dtype in {"uint8", "|u1"}:
            return np.dtype("uint8")
        if self.store_output_dtype in {"uint16", "<u2", ">u2"}:
            return np.dtype("uint16")
        if np.issubdtype(getattr(self.arr, "dtype", np.dtype("uint16")), np.uint8):
            return np.dtype("uint8")
        return np.dtype("uint16")

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

    def read_tile_yxc(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        channel_indices: list[int] | None = None,
        export_dtype: np.dtype | None = None,
    ) -> np.ndarray:
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
        if channel_indices is not None:
            block = block[channel_indices, ...]
        block = self._convert_block_cyx(block, export_dtype)
        return np.moveaxis(block, 0, -1)

    def _convert_block_cyx(self, block: np.ndarray, export_dtype: np.dtype | None = None) -> np.ndarray:
        export_dtype = np.dtype(export_dtype or np.uint16)
        if export_dtype == np.dtype("uint8"):
            if block.dtype == np.uint8:
                block = block.astype(np.uint8, copy=False)
            elif np.issubdtype(block.dtype, np.floating):
                block = np.round(np.clip(block, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                max_val = float(self.scale_max or np.iinfo(block.dtype).max)
                block = np.round(np.clip(block.astype(np.float32), 0.0, max_val) * (255.0 / max_val)).astype(np.uint8)
        elif export_dtype == np.dtype("uint16"):
            if block.dtype == np.uint16:
                block = block.astype(np.uint16, copy=False)
            elif block.dtype == np.uint8:
                block = (block.astype(np.uint16) * 257)
            elif np.issubdtype(block.dtype, np.floating):
                block = np.round(np.clip(block, 0.0, 1.0) * 65535.0).astype(np.uint16)
            else:
                max_val = float(self.scale_max or np.iinfo(block.dtype).max)
                block = np.round(np.clip(block.astype(np.float32), 0.0, max_val) * (65535.0 / max_val)).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported export dtype: {export_dtype}")
        return block

    def _read_decimated_block_cyx_orthogonal(
        self,
        ys: np.ndarray,
        xs: np.ndarray,
        channel_indices: list[int] | None,
        export_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        channels = list(channel_indices) if channel_indices is not None else list(range(self.channels))
        try:
            if self.axes == "CYX":
                if len(channels) == 1:
                    block = np.asarray(self.arr.oindex[channels[0], ys, xs])[None, ...]
                else:
                    block = np.asarray(self.arr.oindex[channels, ys, xs])
            elif self.axes == "YXC":
                block = np.moveaxis(np.asarray(self.arr.oindex[ys, xs, channels]), -1, 0)
            elif self.axes == "YCX":
                block = np.moveaxis(np.asarray(self.arr.oindex[ys, channels, xs]), 1, 0)
            elif self.channels == 1:
                block = np.asarray(self.arr.oindex[ys, xs])[None, ...]
            else:
                raise ValueError(f"Unsupported image axes '{self.axes}'")
        except Exception:
            # Some zarr-backed TIFF stores do not expose efficient orthogonal
            # indexing. Fall back to row-wise reads; slower, but bounded memory.
            rows = []
            for y in ys:
                if self.channels == 1:
                    row = np.asarray(self.arr[int(y), xs])[None, :]
                elif self.axes == "CYX":
                    row = np.asarray(self.arr[channels, int(y), xs])
                elif self.axes == "YXC":
                    row = np.moveaxis(np.asarray(self.arr[int(y), xs, :])[:, channels], -1, 0)
                elif self.axes == "YCX":
                    row = np.asarray(self.arr[int(y), channels, :])[:, xs]
                else:
                    raise
                rows.append(row)
            block = np.stack(rows, axis=1)
        return self._convert_block_cyx(block, export_dtype)

    def read_tile_cyx(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        channel_indices: list[int] | None = None,
        export_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        return np.moveaxis(
            self.read_tile_yxc(
                y0,
                y1,
                x0,
                x1,
                channel_indices=channel_indices,
                export_dtype=export_dtype,
            ),
            -1,
            0,
        )

    def read_decimated_tile_yxc(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        downsample: int,
        channel_indices: list[int] | None = None,
        export_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        mag = max(1, int(downsample))
        ys = np.minimum(np.arange(int(y0), int(y1), dtype=np.int64) * mag, self.height - 1)
        xs = np.minimum(np.arange(int(x0), int(x1), dtype=np.int64) * mag, self.width - 1)
        if mag > 1:
            block = self._read_decimated_block_cyx_orthogonal(
                ys,
                xs,
                channel_indices=channel_indices,
                export_dtype=export_dtype,
            )
            return np.ascontiguousarray(np.moveaxis(block, 0, -1))
        src_y0 = int(ys[0])
        src_y1 = int(ys[-1]) + 1
        src_x0 = int(xs[0])
        src_x1 = int(xs[-1]) + 1
        tile = self.read_tile_yxc(
            src_y0,
            src_y1,
            src_x0,
            src_x1,
            channel_indices=channel_indices,
            export_dtype=export_dtype,
        )
        return np.ascontiguousarray(tile[ys - src_y0, :, :][:, xs - src_x0, :])

    def read_decimated_tile_cyx(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        downsample: int,
        channel_indices: list[int] | None = None,
        export_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        return np.moveaxis(
            self.read_decimated_tile_yxc(
                y0,
                y1,
                x0,
                x1,
                downsample=downsample,
                channel_indices=channel_indices,
                export_dtype=export_dtype,
            ),
            -1,
            0,
        )


def iter_tiles(height: int, width: int, tile_shape: tuple[int, int] | None):
    tile_h, tile_w = tile_shape if tile_shape is not None else (int(height), int(width))
    for y0 in range(0, int(height), int(tile_h)):
        y1 = min(int(height), y0 + int(tile_h))
        for x0 in range(0, int(width), int(tile_w)):
            x1 = min(int(width), x0 + int(tile_w))
            yield y0, y1, x0, x1


def compute_pyramid_level_shapes(height: int, width: int, tile_size: int) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    h = int(height)
    w = int(width)
    while max(h, w) > int(tile_size):
        h = max(1, int(math.ceil(h / 2.0)))
        w = max(1, int(math.ceil(w / 2.0)))
        levels.append((h, w))
    return levels


def choose_tile_shape(height: int, width: int, requested: int) -> tuple[int, int] | None:
    tile_h = min(int(height), int(requested))
    tile_w = min(int(width), int(requested))
    if tile_h >= 16:
        tile_h = max(16, (tile_h // 16) * 16)
    if tile_w >= 16:
        tile_w = max(16, (tile_w // 16) * 16)
    if tile_h < 16 or tile_w < 16:
        return None
    return (tile_h, tile_w) if tile_h > 0 and tile_w > 0 else None


def _ome_pixels_and_channels(ome_xml: str):
    root = ET.fromstring(ome_xml)
    ns = {}
    if root.tag.startswith("{"):
        ns["ome"] = root.tag.split("}", 1)[0][1:]
    pixels = root.find(".//ome:Pixels" if ns else ".//Pixels", ns)
    if pixels is None:
        raise ValueError("OME metadata does not contain a Pixels element")
    channels = pixels.findall("ome:Channel" if ns else "Channel", ns)
    return pixels, channels


def verify_written_ome_tiff(
    path: Path,
    *,
    expected_shape_cyx: tuple[int, int, int],
    expected_dtype: np.dtype,
    expected_channels: list[str],
    require_pyramid: bool,
    expected_physical_size_um: float | None = None,
) -> dict:
    expected_c, expected_y, expected_x = (int(v) for v in expected_shape_cyx)
    with tifffile.TiffFile(path) as tf:
        if not tf.is_ome:
            raise ValueError(f"{path} is not marked as an OME-TIFF")
        series = tf.series[0]
        axes = str(getattr(series, "axes", "")).upper()
        shape = tuple(int(v) for v in series.shape)
        if "C" not in axes or "Y" not in axes or "X" not in axes:
            raise ValueError(f"OME-TIFF axes must include C,Y,X; got axes={axes!r} shape={shape}")
        observed = {
            axis: int(shape[axes.index(axis)])
            for axis in ("C", "Y", "X")
        }
        expected = {"C": expected_c, "Y": expected_y, "X": expected_x}
        if observed != expected:
            raise ValueError(f"OME-TIFF shape mismatch: observed {observed}, expected {expected}")
        if np.dtype(series.dtype) != np.dtype(expected_dtype):
            raise ValueError(f"OME-TIFF dtype mismatch: observed {series.dtype}, expected {expected_dtype}")
        if require_pyramid and len(series.levels) <= 1:
            raise ValueError("OME-TIFF pyramid is required but no pyramid levels were written")
        if require_pyramid:
            prev_y, prev_x = expected_y, expected_x
            for idx, level in enumerate(series.levels[1:], start=1):
                level_axes = str(getattr(level, "axes", axes)).upper()
                level_shape = tuple(int(v) for v in level.shape)
                ly = int(level_shape[level_axes.index("Y")])
                lx = int(level_shape[level_axes.index("X")])
                if ly > prev_y or lx > prev_x:
                    raise ValueError(
                        f"Pyramid level {idx} is not downsampled: {(ly, lx)} after {(prev_y, prev_x)}"
                    )
                prev_y, prev_x = ly, lx
        if not tf.ome_metadata:
            raise ValueError("OME-TIFF metadata is empty")
        pixels, channels = _ome_pixels_and_channels(tf.ome_metadata)
        size_c = int(pixels.attrib.get("SizeC", "-1"))
        if size_c != expected_c:
            raise ValueError(f"OME SizeC mismatch: observed {size_c}, expected {expected_c}")
        physical_size = {
            "x": pixels.attrib.get("PhysicalSizeX"),
            "x_unit": pixels.attrib.get("PhysicalSizeXUnit"),
            "y": pixels.attrib.get("PhysicalSizeY"),
            "y_unit": pixels.attrib.get("PhysicalSizeYUnit"),
        }
        if expected_physical_size_um is not None:
            for axis in ("x", "y"):
                value = physical_size[axis]
                if value is None:
                    raise ValueError(f"OME PhysicalSize{axis.upper()} is missing")
                observed_mpp = float(value)
                if not math.isclose(observed_mpp, float(expected_physical_size_um), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError(
                        f"OME PhysicalSize{axis.upper()} mismatch: "
                        f"observed {observed_mpp}, expected {expected_physical_size_um}"
                    )
        observed_names = [
            (ch.attrib.get("Name") or f"channel_{idx + 1}").strip()
            for idx, ch in enumerate(channels)
        ]
        if observed_names != list(expected_channels):
            raise ValueError(
                "OME channel names mismatch: "
                f"observed={observed_names}, expected={list(expected_channels)}"
            )
        level_shapes = [tuple(int(v) for v in level.shape) for level in series.levels]
        resolution = None
        try:
            page0 = tf.pages[0]
            resolution = {
                "x_resolution": tuple(int(v) for v in page0.tags["XResolution"].value),
                "y_resolution": tuple(int(v) for v in page0.tags["YResolution"].value),
                "resolution_unit": str(page0.tags.get("ResolutionUnit").value),
            }
        except Exception:
            resolution = None
        report = {
            "path": str(path),
            "is_ome": True,
            "axes": axes,
            "shape": list(shape),
            "dtype": str(series.dtype),
            "levels": len(series.levels),
            "level_shapes": [list(v) for v in level_shapes],
            "size_c": size_c,
            "channel_names": observed_names,
            "dimension_order": pixels.attrib.get("DimensionOrder"),
            "size_x": int(pixels.attrib.get("SizeX", "-1")),
            "size_y": int(pixels.attrib.get("SizeY", "-1")),
            "physical_size_um": physical_size,
            "resolution": resolution,
        }
        print(
            f"[OK] verified OME-TIFF {path} axes={axes} shape={shape} "
            f"dtype={series.dtype} levels={len(series.levels)} SizeC={size_c}",
            flush=True,
        )
        return report


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with LazyGigaTIMEReader(input_path) as reader:
        channel_indices, channel_names = resolve_output_channels(args.output_channels, reader.channel_names)
        h, w, c = reader.height, reader.width, len(channel_names)
        meta = dict((reader.root_attrs.get("gigatime") or {}))
        meta.update(reader.array_attrs.get("gigatime") or {})
        meta.update(reader.storage_meta or {})
        export_dtype = reader.resolve_export_dtype(args.output_dtype)
        pyramid_shapes = [] if args.no_pyramid else compute_pyramid_level_shapes(h, w, args.tile_size)
        effective_mpp = meta.get("effective_mpp")
        physical_size_um = float(effective_mpp) if effective_mpp else None
        metadata = {"axes": "CYX", "Channel": {"Name": channel_names}}
        if physical_size_um is not None:
            metadata.update(
                {
                    "PhysicalSizeX": physical_size_um,
                    "PhysicalSizeY": physical_size_um,
                    "PhysicalSizeXUnit": "\u00b5m",
                    "PhysicalSizeYUnit": "\u00b5m",
                }
            )
        tiff_kwargs = {
            "shape": (c, h, w),
            "dtype": export_dtype,
            "tile": choose_tile_shape(h, w, args.tile_size),
            "compression": args.compression,
            "photometric": "MINISBLACK",
            "metadata": metadata,
        }
        if tiff_kwargs["tile"] is None:
            tiff_kwargs.pop("tile")
        if args.predictor:
            tiff_kwargs["predictor"] = True
        base_res = None
        if physical_size_um is not None:
            base_res = float(10_000.0 / physical_size_um)
            tiff_kwargs["resolution"] = (base_res, base_res)
            tiff_kwargs["resolutionunit"] = "CENTIMETER"

        def tile_iterator(
            level_h: int,
            level_w: int,
            *,
            tile_shape: tuple[int, int] | None,
            downsample: int = 1,
            cache_array: np.ndarray | None = None,
        ):
            done = 0
            step_h, step_w = tile_shape if tile_shape is not None else (int(level_h), int(level_w))
            level_tiles = int(math.ceil(level_h / float(step_h)) * math.ceil(level_w / float(step_w)))
            for ch_idx in range(c):
                one_channel_index = [channel_indices[ch_idx]]
                for y0, y1, x0, x1 in iter_tiles(level_h, level_w, tile_shape):
                    done += 1
                    if done == 1 or done == level_tiles * c or done % max(1, (level_tiles * c) // 20) == 0:
                        print(
                            f"[INFO] export level downsample={downsample} tile {done}/{level_tiles * c}",
                            flush=True,
                        )
                    if downsample == 1:
                        tile = reader.read_tile_cyx(
                            y0,
                            y1,
                            x0,
                            x1,
                            channel_indices=one_channel_index,
                            export_dtype=export_dtype,
                        )
                    else:
                        tile = reader.read_decimated_tile_cyx(
                            y0,
                            y1,
                            x0,
                            x1,
                            downsample=downsample,
                            channel_indices=one_channel_index,
                            export_dtype=export_dtype,
                        )
                    if cache_array is not None:
                        cache_array[ch_idx, y0:y1, x0:x1] = tile[0]
                    yield np.ascontiguousarray(tile[0])

        def cache_for_level(level_h: int, level_w: int) -> np.ndarray | None:
            max_bytes = max(0.0, float(args.pyramid_cache_max_gib)) * (1024 ** 3)
            nbytes = int(c) * int(level_h) * int(level_w) * np.dtype(export_dtype).itemsize
            if max_bytes <= 0 or nbytes > max_bytes:
                return None
            print(
                f"[INFO] caching pyramid level shape={(c, level_h, level_w)} bytes={nbytes}",
                flush=True,
            )
            return np.empty((c, int(level_h), int(level_w)), dtype=export_dtype)

        def cached_tile_iterator(
            previous_cyx: np.ndarray,
            level_h: int,
            level_w: int,
            *,
            tile_shape: tuple[int, int] | None,
            downsample: int,
            cache_array: np.ndarray | None = None,
        ):
            done = 0
            step_h, step_w = tile_shape if tile_shape is not None else (int(level_h), int(level_w))
            level_tiles = int(math.ceil(level_h / float(step_h)) * math.ceil(level_w / float(step_w)))
            prev_h = int(previous_cyx.shape[1])
            prev_w = int(previous_cyx.shape[2])
            for ch_idx in range(c):
                prev_plane = previous_cyx[ch_idx]
                for y0, y1, x0, x1 in iter_tiles(level_h, level_w, tile_shape):
                    done += 1
                    if done == 1 or done == level_tiles * c or done % max(1, (level_tiles * c) // 20) == 0:
                        print(
                            f"[INFO] export level downsample={downsample} tile {done}/{level_tiles * c} cached_prev",
                            flush=True,
                        )
                    ys = np.minimum(np.arange(int(y0), int(y1), dtype=np.int64) * 2, prev_h - 1)
                    xs = np.minimum(np.arange(int(x0), int(x1), dtype=np.int64) * 2, prev_w - 1)
                    tile = np.ascontiguousarray(prev_plane[np.ix_(ys, xs)])
                    if cache_array is not None:
                        cache_array[ch_idx, y0:y1, x0:x1] = tile
                    yield tile

        with tifffile.TiffWriter(output_path, bigtiff=True, ome=True) as tif:
            tif.write(
                tile_iterator(h, w, tile_shape=tiff_kwargs.get("tile"), downsample=1),
                subifds=len(pyramid_shapes),
                **tiff_kwargs,
            )
            previous_level_cache = None
            for level_idx, (level_h, level_w) in enumerate(pyramid_shapes, start=1):
                mag = 2 ** level_idx
                level_kwargs = dict(tiff_kwargs)
                level_kwargs["shape"] = (c, level_h, level_w)
                level_kwargs["metadata"] = None
                level_tile = choose_tile_shape(level_h, level_w, args.tile_size)
                if level_tile is None:
                    level_kwargs.pop("tile", None)
                else:
                    level_kwargs["tile"] = level_tile
                if base_res is not None:
                    level_kwargs["resolution"] = (base_res / mag, base_res / mag)
                level_cache = cache_for_level(level_h, level_w)
                if previous_level_cache is None:
                    data_iterator = tile_iterator(
                        level_h,
                        level_w,
                        tile_shape=level_kwargs.get("tile"),
                        downsample=mag,
                        cache_array=level_cache,
                    )
                else:
                    data_iterator = cached_tile_iterator(
                        previous_level_cache,
                        level_h,
                        level_w,
                        tile_shape=level_kwargs.get("tile"),
                        downsample=mag,
                        cache_array=level_cache,
                    )
                tif.write(
                    data_iterator,
                    subfiletype=1,
                    **level_kwargs,
                )
                if level_cache is not None:
                    previous_level_cache = level_cache
        print(
            f"[OK] wrote {output_path} shape={(h, w, c)} "
            f"dtype={export_dtype} pyramid_levels={len(pyramid_shapes)} channels={channel_names}",
            flush=True,
        )
        verification = verify_written_ome_tiff(
            output_path,
            expected_shape_cyx=(c, h, w),
            expected_dtype=export_dtype,
            expected_channels=channel_names,
            require_pyramid=not args.no_pyramid,
            expected_physical_size_um=physical_size_um,
        )
        verification_path = (
            Path(args.verification_json).resolve()
            if str(args.verification_json).strip()
            else output_path.with_suffix(output_path.suffix + ".verify.json")
        )
        verification_path.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[OK] wrote verification report {verification_path}", flush=True)


if __name__ == "__main__":
    main()
