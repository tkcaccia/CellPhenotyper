"""Portable, hash-bound UNI2 shards; storage integrity is not encoder provenance.

The feature payload is a little-endian, C-order float32/float64 matrix without a
header. A small CSV preserves all row metadata (especially string cell IDs).
JSON is committed last. Interrupted bundles are deliberately not reusable.
"""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


FORMAT = "cellphenotyper_uni2_binary"
SCHEMA_VERSION = "1.0.0"
SUFFIX = ".embedding.json"
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _names(values, name):
    if not isinstance(values, list) or not values or any(
        not isinstance(item, str) or not item or item.strip() != item or
        any(ord(char) < 32 for char in item) for item in values
    ) or len(set(values)) != len(values):
        raise ValueError(f"Invalid or duplicate {name}")
    return values


def _ids(rows):
    if "cell_id" not in rows:
        raise ValueError("Binary UNI2 rows require cell_id")
    values = rows["cell_id"].tolist()
    if any(not isinstance(item, (str, int, np.integer)) or isinstance(item, (bool, np.bool_)) for item in values):
        raise ValueError("cell_id must contain exact strings or integers, never floating-point IDs")
    ids = [str(item) for item in values]
    _names(ids, "cell_id values")
    return ids


def _payload_path(manifest_path, name):
    if not isinstance(name, str) or not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError("Binary UNI2 payload paths must be same-directory file names")
    path = manifest_path.parent / name
    if path.resolve().parent != manifest_path.parent.resolve():
        raise ValueError("Binary UNI2 payload escapes its manifest directory")
    return path


def _manifest(path):
    path = Path(path)
    if not path.name.endswith(SUFFIX) or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Invalid binary UNI2 manifest name or oversized manifest")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result
    record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(record, dict) or record.get("format") != FORMAT or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported binary UNI2 format/schema")
    if record.get("dtype") not in {"<f4", "<f8"} or record.get("order") != "C":
        raise ValueError("Binary UNI2 requires little-endian float32/float64 in C order")
    shape = record.get("shape")
    if not isinstance(shape, list) or len(shape) != 2:
        raise ValueError("Binary UNI2 shape must be [rows, features]")
    n, d = (_positive_integer(value, "shape dimension") for value in shape)
    names = _names(record.get("feature_names"), "feature names")
    columns = _names(record.get("row_columns"), "row columns")
    if len(names) != d or "cell_id" not in columns or set(names) & set(columns):
        raise ValueError("Binary UNI2 feature/row schema mismatch")
    if "embedding_mode" in record and record["embedding_mode"] not in {"tile", "inner_square", "nuclei", "cyto"}:
        raise ValueError("Unknown binary UNI2 embedding_mode")
    stem = path.name[:-len(SUFFIX)]
    for kind, suffix in (("features", ".features.bin"), ("rows", ".rows.csv")):
        payload = _payload_path(path, record.get(f"{kind}_file"))
        if payload.name != stem + suffix:
            raise ValueError("Binary UNI2 payload does not belong to this shard stem")
        if not isinstance(record.get(f"{kind}_sha256"), str) or not _HASH.fullmatch(record[f"{kind}_sha256"]):
            raise ValueError("Binary UNI2 requires payload SHA256 hashes")
        size = _positive_integer(record.get(f"{kind}_size_bytes"), f"{kind} size")
        if not payload.is_file() or payload.stat().st_size != size:
            raise ValueError(f"Binary UNI2 {kind} payload missing or byte-size mismatch")
    if record["features_size_bytes"] != n * d * np.dtype(record["dtype"]).itemsize:
        raise ValueError("Binary UNI2 shape/dtype does not match feature byte size")
    return record


def binary_shard_paths(path):
    """Return manifest, rows and feature paths after structural validation."""
    path = Path(path)
    record = _manifest(path)
    return path, _payload_path(path, record["rows_file"]), _payload_path(path, record["features_file"])


def _block_rows(values, requested=4096):
    _positive_integer(requested, "block_rows")
    return min(requested, max(1, (8 * 1024 * 1024) // (values.shape[1] * values.dtype.itemsize)))


def _finite(values, block_rows=4096):
    step = _block_rows(values, block_rows)
    for start in range(0, len(values), step):
        if not np.isfinite(values[start:start + step]).all():
            raise ValueError("Binary UNI2 feature vectors must be finite")


def write_binary_shard(path_or_stem, rows, values, feature_names=None, *, embedding_mode=None):
    """Write one new shard, preserving floating-point precision and row order.

    Only metadata is represented as a DataFrame; feature writes and validation
    use bounded blocks. No existing payload is ever overwritten.
    """
    path = Path(path_or_stem)
    if not path.name.endswith(SUFFIX):
        path = path.with_name(path.name + SUFFIX)
    stem = path.name[:-len(SUFFIX)]
    rows = pd.DataFrame(rows).copy()
    rows.columns = _names(list(rows.columns), "row columns")
    rows["cell_id"] = _ids(rows)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != len(rows) or not all(values.shape):
        raise ValueError("Binary UNI2 row/feature shape mismatch or empty shard")
    if values.dtype.kind != "f" or values.dtype.itemsize not in (4, 8):
        raise ValueError("Binary UNI2 features require float32 or float64; no implicit lossy cast")
    names = _names(list(feature_names) if feature_names is not None else [f"feat_{i + 1}" for i in range(values.shape[1])], "feature names")
    if len(names) != values.shape[1] or set(names) & set(rows.columns):
        raise ValueError("Binary UNI2 feature/row schema mismatch")
    if embedding_mode is not None and embedding_mode not in {"tile", "inner_square", "nuclei", "cyto"}:
        raise ValueError("Unknown binary UNI2 embedding_mode")
    _finite(values)
    features = path.with_name(stem + ".features.bin")
    metadata = path.with_name(stem + ".rows.csv")
    if any(item.exists() for item in (path, features, metadata)):
        raise FileExistsError("Binary UNI2 shard or incomplete payload already exists; use a fresh destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(f"<f{values.dtype.itemsize}")
    step = _block_rows(values)
    with features.open("xb") as handle:
        for start in range(0, len(values), step):
            block = np.ascontiguousarray(values[start:start + step], dtype=dtype)
            handle.write(memoryview(block).cast("B"))
    with metadata.open("x", encoding="utf-8", newline="") as handle:
        rows.to_csv(handle, index=False, lineterminator="\n")
    record = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "dtype": dtype.str,
              "shape": list(values.shape), "order": "C", "feature_names": names,
              "row_columns": list(rows.columns), "features_file": features.name, "rows_file": metadata.name,
              "features_sha256": sha256_file(features), "rows_sha256": sha256_file(metadata),
              "features_size_bytes": features.stat().st_size, "rows_size_bytes": metadata.stat().st_size}
    if embedding_mode is not None:
        record["embedding_mode"] = embedding_mode
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def _open_binary_shard(path, block_rows):
    path = Path(path)
    record = _manifest(path)
    metadata, features = (_payload_path(path, record[key]) for key in ("rows_file", "features_file"))
    for kind, payload in (("rows", metadata), ("features", features)):
        if sha256_file(payload) != record[f"{kind}_sha256"]:
            raise ValueError(f"Binary UNI2 {kind} SHA256 mismatch")
    with metadata.open(encoding="utf-8", newline="") as handle:
        if next(csv.reader(handle), None) != record["row_columns"]:
            raise ValueError("Binary UNI2 metadata header mismatch")
    values = np.memmap(features, dtype=record["dtype"], mode="r", shape=tuple(record["shape"]), order="C")
    _finite(values, block_rows)
    return metadata, values, record


def read_binary_shard(path, *, verify_hashes=True, block_rows=4096):
    """Return (metadata rows, read-only feature memmap, validated manifest).

    Metadata memory scales with shard row count; feature scanning is bounded.
    Hashes are mandatory: the argument exists only to make that policy explicit.
    Use iter_binary_blocks to also bound metadata parsing.
    """
    if verify_hashes is not True:
        raise ValueError("Binary UNI2 hash verification cannot be disabled")
    metadata, values, record = _open_binary_shard(path, block_rows)
    rows = pd.read_csv(metadata, dtype={"cell_id": str}, keep_default_na=False, float_precision="round_trip")
    if list(rows.columns) != record["row_columns"] or len(rows) != record["shape"][0]:
        raise ValueError("Binary UNI2 metadata dimensions/schema mismatch")
    _ids(rows)
    return rows, values, record


def iter_binary_blocks(path, *, block_rows=4096):
    metadata, values, record = _open_binary_shard(path, block_rows)
    step = _block_rows(values, block_rows)
    seen, start = set(), 0
    for rows in pd.read_csv(metadata, dtype={"cell_id": str}, keep_default_na=False,
                            float_precision="round_trip", chunksize=step):
        if list(rows.columns) != record["row_columns"] or start + len(rows) > len(values):
            raise ValueError("Binary UNI2 metadata dimensions/schema mismatch")
        ids = _ids(rows)
        if seen.intersection(ids):
            raise ValueError("Duplicate cell_id values across binary UNI2 row blocks")
        seen.update(ids)
        yield rows.reset_index(drop=True), values[start:start + len(rows)]
        start += len(rows)
    if start != len(values):
        raise ValueError("Binary UNI2 metadata row count mismatch")


def discover_embedding_shards(root, *, storage=None):
    """Find logical shards; reject mixed formats and unowned binary payloads.

    Does not establish encoder provenance. Consumers still verify extraction
    receipts and read each selected shard to check hashes, IDs and finiteness.
    """
    root = Path(root)
    if storage not in {None, "csv", "binary"}:
        raise ValueError("embedding storage must be csv or binary")
    if root.is_file():
        if root.name.endswith(SUFFIX):
            binary_shard_paths(root)
            if storage == "csv":
                raise ValueError("Binary UNI2 supplied where CSV was requested")
        elif root.name.endswith(".rows.csv") or not (root.name.endswith(".csv") or root.name.endswith(".csv.gz")) or storage == "binary":
            raise ValueError("Unsupported UNI2 shard file")
        return [root]
    files, visited = [], set()
    for directory, _dirs, names in os.walk(root, followlinks=True):
        real = Path(directory).resolve()
        if real in visited:
            raise ValueError("Repeated or cyclic staged UNI2 directory")
        visited.add(real)
        files.extend(Path(directory) / name for name in names if "embeddings_shard" in name)
    binaries = sorted(p for p in files if p.name.endswith(SUFFIX))
    csv_files = sorted(p for p in files if ".csv" in p.name and not p.name.endswith(".rows.csv"))
    payloads = {p for p in files if p.name.endswith((".features.bin", ".rows.csv"))}
    owned = set()
    for manifest in binaries:
        _, rows, features = binary_shard_paths(manifest)
        if rows in owned or features in owned:
            raise ValueError("Binary UNI2 payload is shared by multiple shards")
        owned.update((rows, features))
    if payloads != owned:
        raise ValueError("Orphan or incomplete binary UNI2 payload inventory")
    if binaries and csv_files:
        raise ValueError("Mixed CSV and binary UNI2 shards are not permitted")
    if (storage == "csv" and binaries) or (storage == "binary" and csv_files):
        raise ValueError("UNI2 shard storage does not match requested format")
    return binaries or csv_files
