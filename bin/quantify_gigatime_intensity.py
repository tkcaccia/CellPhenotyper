#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import OrderedDict
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_provenance import sha256_file
from profile_cell_morphology import WindowReader

MARKER_SCHEMA_VERSION = "cellphenotyper.gigatime.v1"
CANONICAL_CHANNEL_NAMES = ["DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68", "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67", "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin"]
BACKGROUND_CHANNELS = ["TRITC", "Cy5"]
COMPARTMENT_SEMANTICS = {
    "nuclei": "canonical_nuclear_mask",
    "cyto": "legacy_named_whole_cell_approximation_including_nucleus",
    "ring": "perinuclear_ring_excluding_all_nuclear_pixels",
}
ZARR_STORAGE_SCHEMA = "cellphenotyper.gigatime_zarr_storage.v1"
ZARR_STORAGE_MANIFEST = "cellphenotyper_storage_manifest.json"


def zarr_file_inventory(store_path):
    """Hash actual encoded store files; missing Zarr chunks otherwise read as fill."""
    root = Path(store_path)
    records = {}
    def fail_walk(error):
        raise error
    for directory, folders, files in os.walk(root, onerror=fail_walk):
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Zarr receipt does not permit symlinked internal directories")
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if relative == ZARR_STORAGE_MANIFEST:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Zarr receipt requires ordinary internal files")
            records[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            if len(records) > 2_000_000:
                raise ValueError("Zarr inventory exceeds the bounded 2-million-file limit")
    return records


def finalize_zarr_storage(store_path, metadata):
    """Publish native completion last; receipt binds metadata and encoded chunks."""
    if metadata.get("inference_complete") is not True:
        raise ValueError("Cannot finalize Zarr before integrated inference completes")
    root = zarr.open_group(str(store_path), mode="r+")
    root.attrs["gigatime"] = metadata
    root.attrs["gigatime_storage"] = {"schema_version": ZARR_STORAGE_SCHEMA,
        "state": "complete", "manifest": ZARR_STORAGE_MANIFEST}
    receipt = {"schema_version": ZARR_STORAGE_SCHEMA, "state": "complete", "array_path": "0",
        "marker_schema_sha256": metadata.get("marker_schema", {}).get("schema_sha256"),
        "files": zarr_file_inventory(store_path)}
    (Path(store_path) / ZARR_STORAGE_MANIFEST).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def validate_zarr_storage(image_reader):
    if getattr(image_reader.arr, "path", None) != "0":
        raise ValueError("Native GigaTIME Zarr receipt requires the explicit level-zero array '0'")
    state = dict(image_reader.root_attrs.get("gigatime_storage", {}))
    if state != {"schema_version": ZARR_STORAGE_SCHEMA, "state": "complete", "manifest": ZARR_STORAGE_MANIFEST}:
        raise ValueError("missing or incomplete native Zarr completion receipt")
    path = image_reader.path / ZARR_STORAGE_MANIFEST
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing native Zarr storage inventory")
    receipt = json.loads(path.read_text())
    if not isinstance(receipt, dict):
        raise ValueError("invalid native Zarr storage receipt object")
    if receipt.get("schema_version") != ZARR_STORAGE_SCHEMA or receipt.get("state") != "complete" or receipt.get("array_path") != "0":
        raise ValueError("invalid native Zarr storage receipt")
    if receipt.get("marker_schema_sha256") != image_reader.storage_meta.get("marker_schema", {}).get("schema_sha256"):
        raise ValueError("Zarr receipt marker schema does not match native metadata")
    if receipt.get("files") != zarr_file_inventory(image_reader.path):
        raise ValueError("Zarr encoded-file inventory mismatch: missing, extra or changed metadata/chunks")
    return {"status": "exact_encoded_file_inventory_verified", "files": len(receipt["files"]),
            "manifest_sha256": sha256_file(path)}


def marker_schema_hash(schema):
    payload = {k: v for k, v in schema.items() if k != "schema_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build_marker_schema(metadata, mask_paths):
    checkpoints = metadata.get("model_provenance", {}).get("checkpoints", [])
    valid = [item.get("sha256") for item in checkpoints if item.get("status") == "verified" and re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))]
    if len(valid) != 1:
        raise ValueError("GigaTIME schema requires exactly one verified model checkpoint SHA256")
    names = list(metadata.get("model_channels", CANONICAL_CHANNEL_NAMES))
    if names != CANONICAL_CHANNEL_NAMES:
        raise ValueError("GigaTIME model marker order differs from the versioned 23-channel schema")
    schema = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "marker_names": names,
        "channel_roles": {name: ("background_control" if name in BACKGROUND_CHANNELS else ("nuclear_counterstain" if name == "DAPI" else "predicted_biological_marker")) for name in names},
        "default_phenotype_excluded_channels": list(BACKGROUND_CHANNELS),
        "checkpoint_sha256": valid[0],
        "prediction_precision": "float32", "reduction_precision": "float64",
        "model_arithmetic": metadata.get("model_arithmetic", "unspecified"),
        "prediction_settings": metadata.get("prediction_settings", {}),
        "prediction_precision_semantics": "float32 blended field before storage encoding; model arithmetic is recorded separately",
        "value_semantics": "uncalibrated_virtual_marker_score",
        "inference_shape_yx": list(metadata["inference_shape_yx"]),
        "coordinate_contract": {key: metadata.get(key) for key in ("resolution_contract", "original_shape_yx", "source_mpp", "effective_mpp", "downsample_factor")},
        "compartments": {name: {"semantics": COMPARTMENT_SEMANTICS[name], "mask_sha256": sha256_file(path)} for name, path in mask_paths.items() if path},
    }
    schema["schema_sha256"] = marker_schema_hash(schema)
    return schema


def validate_restart_contract(image_reader, mask_name, mask_path, *, expected_schema=None,
                              checkpoint_sha256=None, allow_legacy_research=False):
    """Reject silent marker/precision/checkpoint/compartment changes on restart."""
    metadata = image_reader.storage_meta
    actual = metadata.get("marker_schema")
    if expected_schema is not None:
        expected_schema = expected_schema.get("marker_schema", expected_schema)
    problems = []
    problems.extend(getattr(image_reader, "metadata_conflicts", []))
    storage_integrity = None
    if getattr(image_reader, "is_zarr", False):
        try:
            storage_integrity = validate_zarr_storage(image_reader)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            problems.append(str(exc))
    if metadata.get("inference_complete") is not True:
        problems.append("inference completion is not recorded")
    if not isinstance(actual, dict):
        problems.append("missing versioned marker schema")
        actual = {}
    else:
        if actual.get("schema_version") != MARKER_SCHEMA_VERSION:
            problems.append("unsupported marker schema version")
        if actual.get("schema_sha256") != marker_schema_hash(actual):
            problems.append("marker schema fingerprint mismatch")
        if actual.get("marker_names") != CANONICAL_CHANNEL_NAMES:
            problems.append("model marker schema differs from canonical 23-channel order")
        if actual.get("prediction_precision") != "float32" or actual.get("reduction_precision") != "float64":
            problems.append("prediction/reduction precision contract mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", str(actual.get("checkpoint_sha256", ""))):
            problems.append("missing verified checkpoint identity")
    if expected_schema is not None and marker_schema_hash(actual) != marker_schema_hash(expected_schema):
        problems.append("requested marker schema does not match the stored inference schema")
    if checkpoint_sha256 and actual.get("checkpoint_sha256") != checkpoint_sha256:
        problems.append("requested checkpoint SHA256 differs from stored inference checkpoint")
    if list(image_reader.channel_names) != CANONICAL_CHANNEL_NAMES:
        problems.append("stored marker subset/order does not contain all 23 canonical markers")
    if list(getattr(image_reader, "native_channel_names", image_reader.channel_names)) != list(image_reader.channel_names):
        problems.append("channel sidecar conflicts with image header marker names")
    if metadata.get("store_channels") not in (None, list(image_reader.channel_names)):
        problems.append("storage metadata marker order conflicts with image header")
    if actual.get("default_phenotype_excluded_channels") != BACKGROUND_CHANNELS:
        problems.append("background-channel exclusion contract mismatch")
    recorded_checkpoints = [record.get("sha256") for record in metadata.get("model_provenance", {}).get("checkpoints", []) if record.get("status") == "verified"]
    if recorded_checkpoints != [actual.get("checkpoint_sha256")]:
        problems.append("checkpoint schema conflicts with inference provenance")
    dtype = np.dtype(image_reader.arr.dtype)
    if dtype != np.dtype("float32") or image_reader.scale_max is not None:
        problems.append(f"stored precision {dtype} is not unquantized float32")
    if metadata.get("output_dtype") not in (None, "float32"):
        problems.append("storage metadata precision differs from authoritative float32")
    if actual.get("inference_shape_yx") != [image_reader.height, image_reader.width]:
        problems.append("stored image dimensions differ from inference schema")
    compartment = actual.get("compartments", {}).get(mask_name)
    if compartment is None:
        problems.append(f"compartment '{mask_name}' was not bound to this inference schema")
    elif compartment.get("semantics") != COMPARTMENT_SEMANTICS.get(mask_name) or compartment.get("mask_sha256") != sha256_file(mask_path):
        problems.append(f"compartment '{mask_name}' mask or semantics differs from inference schema")
    if problems and (not allow_legacy_research or expected_schema is not None or checkpoint_sha256 is not None):
        raise ValueError("Incompatible GigaTIME restart: " + "; ".join(problems) + ". Use authoritative integrated tables or rerun with all-channel float32 storage; --allow-legacy-research explicitly permits non-equivalent exploratory quantification only.")
    return {
        "status": "legacy_research_not_equivalent" if problems else "equivalent_float32_restart",
        "equivalent_to_authoritative_integrated": not problems,
        "differences": problems, "marker_schema": actual or None,
        "source_storage_dtype": str(dtype), "reduction_precision": "float64",
        "storage_integrity": storage_integrity,
    }


GIGATIME_VALUE_SEMANTICS = {
    "value_type": "uncalibrated_virtual_marker_score",
    "bounded_range": [0.0, 1.0],
    "calibrated_probability": False,
    "measured_protein_abundance": False,
    "research_use_only": True,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantify per-label uncalibrated GigaTIME virtual-marker scores."
    )
    ap.add_argument("--image", required=True, help="Input GigaTIME TIFF, Zarr store, or GigaTIME output directory")
    ap.add_argument("--mask", required=True, help="Input labeled mask TIFF")
    ap.add_argument("--mask-name", required=True, help="Mask label used in outputs, e.g. nuclei or cyto")
    ap.add_argument("--out-quant-csv", required=True, help="Output wide CSV with mcMicro-like per-object quantification")
    ap.add_argument("--out-mean-csv", required=True, help="Output CSV with mean virtual-marker score")
    ap.add_argument("--out-stats-csv", required=True, help="Output CSV with area/mean/sum/max virtual-marker scores")
    ap.add_argument("--out-summary-json", required=True, help="Output JSON summary")
    ap.add_argument("--expected-schema", help="Expected gigatime_marker_schema.json or metadata JSON")
    ap.add_argument("--checkpoint-sha256", help="Required inference checkpoint SHA256")
    ap.add_argument("--allow-legacy-research", action="store_true", help="Explicitly quantify old/subset/quantized stores as non-equivalent exploratory output")
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


class _TiffPlaneWindows(WindowReader):
    """Reuse bounded segment decoding for one CYX TIFF channel page."""
    def __init__(self, tf, page):
        if isinstance(page, tifffile.TiffFrame):
            page = page.aspage()
        self._tf, self._page = tf, page
        self._array = None
        self._cache, self._cache_bytes, self._cache_limit = OrderedDict(), 0, 4*1024**2
        self._height, self._width = int(page.imagelength), int(page.imagewidth)
        self._samples, self._separate = 1, False
        self.shape = (self._height, self._width)
        self.dtype = np.dtype(page.dtype)
        self._tile_w = int(page.tilewidth) if page.is_tiled else self._width
        self._tile_h = int(page.tilelength) if page.is_tiled else int(page.rowsperstrip)
        self._ncols, self._nrows = math.ceil(self._width/self._tile_w), math.ceil(self._height/self._tile_h)
        self.backend = "tiff_channel_segment_windows"
        if int(page.samplesperpixel) != 1 or int(page.imagedepth) != 1:
            raise ValueError("GigaTIME TIFF requires scalar channel pages (CYX)")
        if self._tile_h*self._tile_w*self.dtype.itemsize > 128*1024**2:
            raise ValueError("GigaTIME TIFF strips exceed bounded-memory limit; retile the scalar channels")

    def close(self):
        self._cache.clear()


class LazyImageReader:
    def __init__(self, path: str):
        self.path = _resolve_gigatime_image_path(path)
        self.storage_meta = _load_storage_metadata(str(self.path))
        self.tf = None
        self.series = None
        self.root_attrs = {}
        self.array_attrs = {}
        self.channel_windows = None
        self.metadata_conflicts = []
        self.is_zarr = self.path.is_dir() or self.path.suffix.lower() == ".zarr"

        if self.is_zarr:
            zobj = zarr.open(str(self.path), mode="r")
            self.arr, self.root_attrs = _select_zarr_array(zobj)
            self.array_attrs = getattr(self.arr, "attrs", {})
            self.axes = _axes_from_zarr_attrs(self.root_attrs, self.array_attrs, tuple(int(v) for v in self.arr.shape))
            native_metadata = self.root_attrs.get("gigatime")
            if isinstance(native_metadata, dict):
                # Diagnostic additions such as seam-QC may be appended to the
                # external sidecar later; authoritative numerical fields may not.
                for key in ("marker_schema", "inference_complete", "model_provenance", "store_channels", "output_dtype", "storage_scale_max",
                            "inference_shape_yx", "original_shape_yx", "source_mpp", "effective_mpp", "downsample_factor", "resolution_contract"):
                    if key in self.storage_meta and self.storage_meta.get(key) != native_metadata.get(key):
                        self.metadata_conflicts.append(f"external metadata conflicts with native Zarr {key}")
                self.storage_meta = dict(native_metadata)
            else:
                self.metadata_conflicts.append("native Zarr inference metadata is absent")
        else:
            self.tf = tifffile.TiffFile(str(self.path))
            self.series = self.tf.series[0].levels[0]
            self.axes = _normalize_axes(getattr(self.series, "axes", ""), tuple(int(v) for v in self.series.shape))
            if len(self.series.shape) == 2:
                self.axes = "CYX"
            elif self.axes != "CYX":
                raise ValueError(f"GigaTIME TIFF must have explicit scalar CYX axes, got {self.axes}")
            self.arr = SimpleNamespace(shape=self.series.shape, dtype=self.series.dtype)
            try:
                self.channel_windows = [_TiffPlaneWindows(self.tf, page) for page in self.series.pages]
            except Exception:
                self.tf.close()
                raise

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
        self.native_channel_names = list(channel_names)
        if self.is_zarr:
            multiscales = self.root_attrs.get("multiscales")
            if isinstance(multiscales, list) and multiscales and self.array_attrs.get("axes"):
                entries = multiscales[0].get("axes", [])
                root_axes = "".join((item.get("name", "") if isinstance(item, dict) else str(item)).upper()[:1] for item in entries)
                if root_axes and root_axes != self.axes:
                    self.metadata_conflicts.append("Zarr root/array coordinate axes conflict")
            array_names = _extract_channel_names_from_attrs(self.array_attrs, self.channels)
            if array_names is not None and array_names != self.native_channel_names:
                self.metadata_conflicts.append("Zarr root/array channel names conflict")
            if self.array_attrs.get("output_dtype") not in (None, self.storage_meta.get("output_dtype")):
                self.metadata_conflicts.append("Zarr array/native precision metadata conflict")
            if self.array_attrs.get("storage_scale_max") != self.storage_meta.get("storage_scale_max"):
                self.metadata_conflicts.append("Zarr array/native storage scaling conflict")

        sidecar_channels = self.path.with_name("gigatime_channels.json")
        if sidecar_channels.exists():
            try:
                names = json.loads(sidecar_channels.read_text(encoding="utf-8"))
                if isinstance(names, list) and len(names) == self.channels:
                    self.channel_names = [str(v).strip() or f"channel_{i + 1}" for i, v in enumerate(names)]
            except Exception:
                pass

        self.scale_max = None
        storage_scale = self.storage_meta.get("storage_scale_max")
        if storage_scale is not None:
            self.scale_max = float(storage_scale)
        elif np.issubdtype(np.dtype(self.arr.dtype), np.integer):
            self.scale_max = float(np.iinfo(np.dtype(self.arr.dtype)).max)

    def close(self) -> None:
        for reader in self.channel_windows or []:
            reader.close()
        try:
            self.tf.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyImageReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if self.channel_windows is not None:
            block = np.stack([reader.read(x0, y0, x1, y1) for reader in self.channel_windows])
        elif self.channels == 1 and len(self.arr.shape) == 2:
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
        self.path = str(path)
        self.reader = WindowReader(path)
        if len(self.reader.shape) != 2 or self.reader.dtype.kind not in "ui":
            self.reader.close()
            raise ValueError("Marker compartment mask must be a 2D integer label raster")
        self.height, self.width = self.reader.shape

    def close(self) -> None:
        try:
            self.reader.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyMaskReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, y0: int, y1: int, x0: int, x1: int, target_shape: tuple[int, int]) -> np.ndarray:
        ty, tx = target_shape
        if (self.height, self.width) == target_shape:
            out = self.reader.read(x0, y0, x1, y1)
        else:
            y_idx = np.minimum(
                np.round(np.linspace(y0 * self.height / ty, (y1 - 1) * self.height / ty, y1 - y0)).astype(np.int64),
                self.height - 1,
            )
            x_idx = np.minimum(
                np.round(np.linspace(x0 * self.width / tx, (x1 - 1) * self.width / tx, x1 - x0)).astype(np.int64),
                self.width - 1,
            )
            view = self.reader.read(int(x_idx[0]), int(y_idx[0]), int(x_idx[-1]+1), int(y_idx[-1]+1))
            out = view[np.ix_(y_idx-y_idx[0], x_idx-x_idx[0])]
        if out.ndim != 2:
            raise ValueError(f"Unsupported mask block shape after squeeze: {out.shape}")
        if np.any(out < 0):
            raise ValueError("Negative compartment labels are invalid")
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
            values_at_cells = image_block[:, positive]
            if not np.isfinite(values_at_cells).all() or np.any(values_at_cells < 0) or np.any(values_at_cells > 1):
                raise ValueError("Stored virtual-marker scores within cell masks must be finite and in [0,1]")
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
            "value_semantics": GIGATIME_VALUE_SEMANTICS["value_type"],
            "calibrated_probability": False,
            "measured_protein_abundance": False,
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


def write_csv(path: str, rows: list[dict], fieldnames=None) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path_obj.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(fieldnames or ["label_id", "area_px"])
        return

    fieldnames = fieldnames or list(rows[0].keys())
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
        expected = json.loads(Path(args.expected_schema).read_text()) if args.expected_schema else None
        restart_contract = validate_restart_contract(image_reader, args.mask_name, args.mask,
            expected_schema=expected, checkpoint_sha256=args.checkpoint_sha256,
            allow_legacy_research=args.allow_legacy_research)
        quant_rows, mean_rows, stats_rows = quantify_blockwise(
            image_reader,
            mask_reader,
            channel_names,
            mask_name=args.mask_name,
        )

    for rows in (quant_rows, mean_rows, stats_rows):
        for row in rows:
            row["quantification_status"] = restart_contract["status"]
    base_fields = ["label_id", "mask_name", "value_semantics", "calibrated_probability",
        "measured_protein_abundance", "area_px", "centroid_y_px", "centroid_x_px",
        "bbox_ymin_px", "bbox_xmin_px", "bbox_ymax_px", "bbox_xmax_px", "quantification_status"]
    statistic_fields = [f"{name}__{stat}" for name in channel_names for stat in ("mean", "sum", "max")]
    write_csv(args.out_quant_csv, quant_rows, base_fields+statistic_fields)
    write_csv(args.out_mean_csv, mean_rows, base_fields+channel_names)
    write_csv(args.out_stats_csv, stats_rows, base_fields+statistic_fields)

    summary = {
        "mask_name": args.mask_name,
        "image": str(Path(args.image).resolve()),
        "mask": str(Path(args.mask).resolve()),
        "image_shape_cyx": image_shape_cyx,
        "mask_shape_yx": mask_shape_yx,
        "channel_names": channel_names,
        "objects_quantified": len(quant_rows),
        "value_semantics": dict(GIGATIME_VALUE_SEMANTICS),
        "restart_contract": restart_contract,
        "marker_schema": restart_contract["marker_schema"],
        "authority_status": restart_contract["status"],
        "mask_compartment_contract": (
            "Scores were summarized over pixels assigned to the named segmentation compartment."
        ),
        "quantification_csv": str(Path(args.out_quant_csv).resolve()),
    }
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[OK] quantified {len(quant_rows)} objects for mask={args.mask_name} "
        f"channels={len(channel_names)} authority_status={restart_contract['status']} image={args.image}"
    )


if __name__ == "__main__":
    main()
