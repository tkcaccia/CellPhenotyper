"""Source-bound scalar/array neighbourhood columns without fitting runtimes.

NPY segments retain their source precision on disk. Requested blocks are
returned as float64; NaNs mean unavailable, not zeros. Loading checks every
payload once in bounded windows. Later reads never construct a wide global
DataFrame, and recheck() explicitly revalidates the original source bytes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FORMAT = "cellphenotyper_neighborhood_feature_store"
VERSION = "1.0.0"
STORE_PATH = "neighborhood_features/feature_store.json"
KEYS = ["sample_id", "cell_id", "cell_uid"]
BATCH_ROWS = 4096
ARRAY_BLOCK_BYTES = 8 * 1024**2


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, expected_sha256=None):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Duplicate neighbourhood JSON field: {name}")
            result[name] = value
        return result
    def nonfinite(value):
        raise ValueError(f"Nonfinite neighbourhood JSON value: {value}")
    raw = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Neighbourhood metadata SHA256 mismatch while reading")
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    json.dumps(value, allow_nan=False)
    if not isinstance(value, dict):
        raise ValueError("Neighbourhood metadata must be an object")
    return value


def _inside(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or Path(name).is_absolute()
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Neighbourhood source path must be a contained relative path")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or escaping neighbourhood source: {name}")
    return path


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _names(value, context, *, empty=False):
    if (not isinstance(value, list) or (not value and not empty)
            or any(not isinstance(name, str) or not name.strip() or re.search(r"[\x00-\x1f\x7f]", name) for name in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{context} must contain unique nonempty literal column names")
    return value


def _profile_axes(root):
    table = _inside(root, "cell_profiles.parquet")
    rows = _inside(root, "feature_rows.csv")
    parquet = pq.ParquetFile(table)
    columns = _names(parquet.schema_arrow.names, "Canonical profile columns")
    if not set(KEYS) <= set(columns):
        raise ValueError("Canonical profile identity columns are missing")
    count, seen_pairs, seen_uids = 0, set(), set()
    with rows.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != KEYS:
            raise ValueError("Neighbourhood feature_rows must use exact canonical identity columns")
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=KEYS):
            frame = batch.to_pandas()
            for values in frame.itertuples(index=False, name=None):
                if (any(not isinstance(value, str) or not value.strip()
                        or value != value.strip() or re.search(r"[\x00-\x1f\x7f]", value) for value in values)
                        or next(reader, None) != list(values)):
                    raise ValueError("Neighbourhood feature_rows identities/order differ from canonical profile")
                pair = values[:2]
                if pair in seen_pairs or values[2] in seen_uids:
                    raise ValueError("Duplicate canonical neighbourhood identities")
                seen_pairs.add(pair); seen_uids.add(values[2]); count += 1
        if next(reader, None) is not None or count != parquet.metadata.num_rows:
            raise ValueError("Neighbourhood feature_rows row count differs")
    return count, columns, {"cell_profiles.parquet": _sha(table), "feature_rows.csv": _sha(rows)}


def _numeric_float64(values):
    array = np.asarray(values)
    if array.dtype.kind not in "fiub":
        raise ValueError("Requested neighbourhood feature is not real numeric data")
    if array.dtype.kind == "f" and array.dtype.itemsize > 8:
        raise ValueError("Neighbourhood floating precision exceeds lossless float64 conversion")
    converted = array.astype(np.float64, copy=False)
    if np.isinf(converted).any():
        raise ValueError("Neighbourhood features cannot contain infinity")
    if array.dtype.kind in "iu":
        # Comparing against Python integers avoids uint64 boundary overflow in
        # a float64 -> integer round trip. Ordinary counts stay on the fast path.
        large = np.abs(converted) >= 2**53
        if large.any() and any(int(number) != int(original) for number, original in
                               zip(converted[large].flat, array[large].flat)):
            raise ValueError("Integer neighbourhood features cannot be represented exactly as float64")
    return converted


def _validate_store(root, store, count, scalar_columns, row_hash, sources):
    if (set(store) != {"format", "schema_version", "cell_count", "feature_rows_sha256", "groups"}
            or store.get("format") != FORMAT or store.get("schema_version") != VERSION
            or not isinstance(store.get("cell_count"), int) or isinstance(store.get("cell_count"), bool)
            or store.get("cell_count") != count
            or store.get("feature_rows_sha256") != row_hash):
        raise ValueError("Neighbourhood feature store format/count/row identity mismatch")
    groups = store["groups"]
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Neighbourhood feature store requires declared groups")
    _names(list(groups), "Neighbourhood groups")
    locations, arrays, used_paths = {}, {}, {}
    for group, record in groups.items():
        if not isinstance(record, dict) or set(record) != {"columns", "segments"}:
            raise ValueError("Neighbourhood group requires exact columns and segments")
        columns = _names(record["columns"], "Neighbourhood group columns")
        if set(columns) & (set(scalar_columns) | set(locations)):
            raise ValueError("Neighbourhood logical columns overlap scalar columns or another group")
        segments = record["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError("Neighbourhood group requires numeric segments")
        flattened = []
        for segment in segments:
            if (not isinstance(segment, dict)
                    or set(segment) != {"path", "sha256", "shape", "dtype", "columns"}):
                raise ValueError("Neighbourhood segment requires exact source/axis metadata")
            axis = _names(segment["columns"], "Neighbourhood segment columns")
            name = segment["path"]
            path = _inside(root, name)
            if path.suffix != ".npy" or not _digest(segment["sha256"]):
                raise ValueError("Neighbourhood segments require NPY files and SHA256")
            shape = segment["shape"]
            if (not isinstance(shape, list) or len(shape) != 2
                    or any(not isinstance(n, int) or isinstance(n, bool) for n in shape)
                    or shape != [count, len(axis)]):
                raise ValueError("Neighbourhood segment row/feature shape mismatch")
            if path in used_paths and used_paths[path] != name:
                raise ValueError("Neighbourhood segment path aliases are ambiguous")
            used_paths[path] = name
            if name not in arrays:
                actual = _sha(path)
                if actual != segment["sha256"]:
                    raise ValueError(f"Neighbourhood segment SHA256 mismatch: {name}")
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                if (values.ndim != 2 or values.dtype.kind not in "fiu"
                        or (values.dtype.kind == "f" and values.dtype.itemsize > 8)):
                    raise ValueError("Neighbourhood segments must be real numeric matrices")
                if path.stat().st_size != values.offset + values.nbytes:
                    raise ValueError("Neighbourhood NPY segment has extra or incomplete payload bytes")
                step = max(1, ARRAY_BLOCK_BYTES // max(1, values.shape[1] * max(values.dtype.itemsize, 8)))
                for begin in range(0, count, step):
                    _numeric_float64(values[begin:begin + step])
                arrays[name] = values
                sources[name] = actual
            values = arrays[name]
            try:
                dtype = np.dtype(segment["dtype"])
            except (TypeError, ValueError) as error:
                raise ValueError("Invalid neighbourhood segment dtype") from error
            if (not isinstance(segment["dtype"], str) or values.dtype != dtype
                    or list(values.shape) != shape or sources[name] != segment["sha256"]):
                raise ValueError("Neighbourhood segment source shape/dtype/hash mismatch")
            for offset, column in enumerate(axis):
                if column in flattened:
                    raise ValueError("Neighbourhood segment columns are duplicated")
                locations[column] = (name, offset)
            flattened.extend(axis)
        if flattened != columns:
            raise ValueError("Neighbourhood group axis must exactly match ordered segment columns")
    return locations, arrays


def finalize_store(profile_dir, groups):
    """Validate completed arrays, then write one completion JSON; return descriptor.

    The caller writes the narrow canonical table and feature_rows first, then
    includes the returned descriptor in its final cell_profiles_manifest.json.
    Existing array payloads are referenced in place and never copied/modified.
    """
    root = Path(profile_dir).resolve()
    destination = root / STORE_PATH
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Neighbourhood feature store requires a new completion path")
    if not destination.parent.resolve().is_relative_to(root):
        raise ValueError("Neighbourhood feature store directory escapes profile root")
    count, columns, sources = _profile_axes(root)
    # Round trip to reject non-JSON/nonfinite metadata and detach caller state.
    groups = json.loads(json.dumps(groups, allow_nan=False))
    record = {"format": FORMAT, "schema_version": VERSION, "cell_count": count,
              "feature_rows_sha256": sources["feature_rows.csv"], "groups": groups}
    _validate_store(root, record, count, columns, sources["feature_rows.csv"], sources)
    for name, expected in sources.items():
        if _sha(_inside(root, name)) != expected:
            raise ValueError("Neighbourhood source changed while finalizing feature store")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return {"path": STORE_PATH, "sha256": _sha(destination)}


class FeatureColumns:
    """Read ordered scalar/array feature blocks with immutable source receipts."""

    def __init__(self, profile_dir, *, manifest=None):
        self.root = Path(profile_dir).resolve()
        header = _inside(self.root, "cell_profiles_manifest.json")
        header_hash = _sha(header)
        actual_manifest = _json(header, header_hash)
        if manifest is not None and manifest != actual_manifest:
            raise ValueError("Supplied neighbourhood profile manifest differs from source bytes")
        self.manifest = actual_manifest
        self.cell_count, self.scalar_columns, self.source_files = _profile_axes(self.root)
        if (not isinstance(self.manifest.get("cell_count"), int)
                or isinstance(self.manifest.get("cell_count"), bool)
                or self.manifest.get("cell_count") != self.cell_count):
            raise ValueError("Neighbourhood profile manifest cell count mismatch")
        for name, digest in self.source_files.items():
            expected = self.manifest.get("files", {}).get(name)
            if self.manifest.get("neighborhood_feature_store") is not None and not _digest(expected):
                raise ValueError(f"Array-backed neighbourhood profiles require canonical source hashes: {name}")
            if expected is not None and expected != digest:
                raise ValueError(f"Neighbourhood profile manifest SHA256 mismatch: {name}")
        self.source_files["cell_profiles_manifest.json"] = header_hash
        self.store_record, self.groups, self._locations, self._arrays = None, {}, {}, {}
        descriptor = self.manifest.get("neighborhood_feature_store")
        if descriptor is not None:
            if (not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}
                    or not _digest(descriptor["sha256"])):
                raise ValueError("Invalid neighbourhood feature store descriptor")
            path = _inside(self.root, descriptor["path"])
            if _sha(path) != descriptor["sha256"]:
                raise ValueError("Neighbourhood feature store SHA256 mismatch")
            self.source_files[descriptor["path"]] = descriptor["sha256"]
            self.store_record = _json(path, descriptor["sha256"])
            self._locations, self._arrays = _validate_store(self.root, self.store_record, self.cell_count,
                self.scalar_columns, self.source_files["feature_rows.csv"], self.source_files)
            self.groups = self.store_record["groups"]
            for name, group in self.groups.items():
                if name.startswith("own:"):
                    original = self.manifest.get("feature_blocks", {}).get(name[4:])
                    if (not isinstance(original, dict) or len(group["segments"]) != 1
                            or any(group["segments"][0].get(key) != original.get(key)
                                   for key in ("path", "sha256", "shape"))):
                        raise ValueError("Own-cell neighbourhood groups must reference the unchanged original feature block")
        self.group_definitions = {}
        summary_name = self.manifest.get("neighborhoods", {}).get("summary")
        if summary_name is not None:
            path = _inside(self.root, summary_name)
            actual = _sha(path)
            if self.manifest.get("files", {}).get(summary_name) != actual:
                raise ValueError("Neighbourhood feature definition summary SHA256 mismatch")
            self.source_files[summary_name] = actual
            summary = _json(path, actual)
            definitions = summary.get("feature_group_definitions", {})
            if not isinstance(definitions, dict):
                raise ValueError("Neighbourhood feature group definitions must be an object")
            self.group_definitions = definitions
            declared = summary.get("feature_groups")
            if self.groups and (not isinstance(declared, dict) or any(
                    declared.get(name) != record["columns"] for name, record in self.groups.items())):
                raise ValueError("Neighbourhood stored axes differ from summary feature groups")
        self.available_columns = self.scalar_columns + list(self._locations)

    def recheck(self):
        for name, expected in self.source_files.items():
            if _sha(_inside(self.root, name)) != expected:
                raise ValueError(f"Neighbourhood feature source changed: {name}")

    def _rows(self, rows):
        if isinstance(rows, slice):
            return np.fromiter(range(*rows.indices(self.cell_count)), dtype=np.int64)
        values = np.asarray(rows)
        if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
            raise ValueError("Neighbourhood rows must be a slice or one-dimensional integer indices")
        if (values < 0).any() or (values >= self.cell_count).any():
            raise ValueError("Neighbourhood row index is out of range")
        return values.astype(np.int64, copy=False)

    def read(self, rows, columns):
        """Return a requested float64 block, preserving row order/repetitions."""
        if isinstance(columns, (str, bytes)):
            raise ValueError("Requested neighbourhood columns must be a sequence, not a string")
        columns = _names(list(columns), "Requested neighbourhood columns", empty=True)
        if set(columns) - set(self.available_columns):
            raise ValueError("Requested neighbourhood feature columns are unavailable")
        indices = self._rows(rows)
        output = np.empty((len(indices), len(columns)), dtype=np.float64)
        if not len(indices) or not columns:
            return output
        scalar = [name for name in columns if name not in self._locations]
        if scalar:
            destinations = [columns.index(name) for name in scalar]
            cursor = 0
            parquet = pq.ParquetFile(_inside(self.root, "cell_profiles.parquet"))
            batch_rows = max(1, min(BATCH_ROWS, ARRAY_BLOCK_BYTES // (8 * len(scalar))))
            for batch in parquet.iter_batches(batch_size=batch_rows, columns=scalar):
                selected = np.flatnonzero((indices >= cursor) & (indices < cursor + len(batch)))
                if len(selected):
                    frame = batch.to_pandas()
                    for name in scalar:
                        if not pd.api.types.is_numeric_dtype(frame[name].dtype):
                            raise ValueError(f"Requested neighbourhood scalar is not numeric: {name}")
                    values = frame.iloc[indices[selected] - cursor].to_numpy(dtype=np.float64, na_value=np.nan)
                    # Validate integer precision before pandas combines dtypes.
                    for name in scalar:
                        series = frame[name].iloc[indices[selected] - cursor]
                        if series.dtype.kind in "iu":
                            observed = series.dropna()
                            dtype = getattr(observed.dtype, "numpy_dtype", observed.dtype)
                            _numeric_float64(observed.to_numpy(dtype=dtype))
                    output[np.ix_(selected, destinations)] = _numeric_float64(values)
                cursor += len(batch)
        by_segment = {}
        for destination, column in enumerate(columns):
            if column in self._locations:
                path, offset = self._locations[column]
                by_segment.setdefault(path, []).append((destination, offset))
        for path, offsets in by_segment.items():
            output[:, [item[0] for item in offsets]] = _numeric_float64(
                self._arrays[path][np.ix_(indices, [item[1] for item in offsets])])
        return output
