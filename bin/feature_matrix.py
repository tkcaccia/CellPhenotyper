#!/usr/bin/env python3
"""Isolated, CPU-only candidate for lossless CellViT float32 feature storage.

This does not replace the production CSV contract. A completed bundle contains
an immutable manifest, a gzip float32 matrix and a gzip row index. Readers
verify all digests and each row binding before publishing a reconstructed CSV.
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import time


VERSION = 1
DTYPE = "float32-le"
MATRIX = "features.f32.gz"
INDEX = "row_index.csv.gz"
MANIFEST = "manifest.json"
INDEX_HEADER = ["cell_id", "source_id", "x", "y", "row_sha256"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _fsync_file(path):
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def _row_digest(meta, packed):
    payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload + b"\0" + packed).hexdigest()


def _header(feature_count):
    return ["cell_id", "source_id", "x", "y"] + [f"feat_{i}" for i in range(1, feature_count + 1)]


def pack(source, destination, source_manifest=None):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if not source.is_file():
        raise ValueError(f"missing source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    started = time.perf_counter()
    try:
        source_sha256 = digest(source)
        count = 0
        seen = set()
        with gzip.open(source, "rt", encoding="utf-8", newline="") as input_stream:
            source_rows = csv.reader(input_stream)
            source_header = next(source_rows)
            if len(source_header) < 5 or source_header != _header(len(source_header) - 4):
                raise ValueError("unexpected feature CSV header or order")
            feature_count = len(source_header) - 4
            row_bytes = 4 * feature_count
            with open(temporary / MATRIX, "wb") as raw_matrix, open(temporary / INDEX, "wb") as raw_index:
                with gzip.GzipFile(filename="", fileobj=raw_matrix, mode="wb", mtime=0, compresslevel=6) as matrix_stream:
                    with gzip.GzipFile(filename="", fileobj=raw_index, mode="wb", mtime=0, compresslevel=6) as index_gzip:
                        import io
                        with io.TextIOWrapper(index_gzip, encoding="utf-8", newline="") as text_index:
                            index_rows = csv.writer(text_index)
                            index_rows.writerow(INDEX_HEADER)
                            for row in source_rows:
                                if len(row) != len(source_header):
                                    raise ValueError(f"row {count} has wrong column count")
                                meta = row[:4]
                                if not meta[0] or meta[0] in seen:
                                    raise ValueError(f"missing or duplicate cell ID at row {count}")
                                seen.add(meta[0])
                                packed = bytearray(row_bytes)
                                for i, value in enumerate(row[4:]):
                                    numeric = float(value)
                                    if not math.isfinite(numeric):
                                        raise ValueError(f"nonfinite feature at row {count}, column {i}")
                                    blob = struct.pack("<f", numeric)
                                    if struct.unpack("<f", blob)[0] != numeric:
                                        raise ValueError(f"feature loses precision at row {count}, column {i}")
                                    packed[4 * i:4 * i + 4] = blob
                                matrix_stream.write(packed)
                                index_rows.writerow(meta + [_row_digest(meta, packed)])
                                count += 1
        for name in (MATRIX, INDEX):
            _fsync_file(temporary / name)
        provenance = None
        if source_manifest is not None:
            source_manifest = Path(source_manifest)
            scientific = json.loads(source_manifest.read_text(encoding="utf-8"))
            feature_set = scientific.get("feature_set", {})
            if (scientific.get("schema") != "cellphenotyper-cell-features"
                    or feature_set.get("sha256") != source_sha256
                    or feature_set.get("row_count") != count
                    or feature_set.get("feature_columns") != source_header[4:]):
                raise ValueError("source feature manifest does not match CSV")
            provenance = {
                "scientific_manifest_sha256": digest(source_manifest),
                "feature_set_id": feature_set.get("feature_set_id"),
                "frame_id": scientific.get("image", {}).get("frame_id"),
                "original_csv_name": feature_set.get("path"),
            }
        manifest = {
            "schema": "cellvit-float32-feature-matrix", "version": VERSION,
            "dtype": DTYPE, "row_count": count, "feature_count": feature_count,
            "columns": source_header, "row_index_columns": INDEX_HEADER,
            "source": {"name": source.name, "sha256": source_sha256, "size": source.stat().st_size},
            "matrix": {"name": MATRIX, "sha256": digest(temporary / MATRIX), "size": (temporary / MATRIX).stat().st_size},
            "index": {"name": INDEX, "sha256": digest(temporary / INDEX), "size": (temporary / INDEX).stat().st_size},
        }
        if provenance is not None:
            manifest["scientific_provenance"] = provenance
        with open(temporary / MANIFEST, "w", encoding="utf-8") as output:
            json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {**manifest, "pack_seconds": time.perf_counter() - started}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validated(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "cellvit-float32-feature-matrix" or manifest.get("version") != VERSION:
        raise ValueError("unsupported manifest")
    if manifest.get("dtype") != DTYPE:
        raise ValueError("unsupported matrix dtype")
    n, d = manifest.get("row_count"), manifest.get("feature_count")
    if not isinstance(n, int) or n < 0 or not isinstance(d, int) or d < 1:
        raise ValueError("invalid matrix shape")
    if manifest.get("columns") != _header(d) or manifest.get("row_index_columns") != INDEX_HEADER:
        raise ValueError("invalid ordered columns")
    for key, expected_name in (("matrix", MATRIX), ("index", INDEX)):
        entry = manifest[key]
        path = bundle / expected_name
        if entry.get("name") != expected_name or entry.get("size") != path.stat().st_size or entry.get("sha256") != digest(path):
            raise ValueError(f"{key} size or checksum mismatch")
    return manifest


def iter_rows(bundle):
    manifest = _validated(bundle)
    bundle = Path(bundle)
    d, n = manifest["feature_count"], manifest["row_count"]
    with gzip.open(bundle / MATRIX, "rb") as matrix_stream, gzip.open(bundle / INDEX, "rt", encoding="utf-8", newline="") as index_stream:
        index_rows = csv.reader(index_stream)
        if next(index_rows) != INDEX_HEADER:
            raise ValueError("index header mismatch")
        seen = set()
        for row_number, entry in enumerate(index_rows):
            if row_number >= n or len(entry) != 5:
                raise ValueError("index shape mismatch")
            meta, check = entry[:4], entry[4]
            if not meta[0] or meta[0] in seen:
                raise ValueError("missing or duplicate cell ID")
            seen.add(meta[0])
            packed = matrix_stream.read(4 * d)
            if len(packed) != 4 * d:
                raise ValueError("truncated matrix")
            if _row_digest(meta, packed) != check:
                raise ValueError(f"row {row_number} ID/matrix binding mismatch")
            values = struct.unpack(f"<{d}f", packed)
            if not all(map(math.isfinite, values)):
                raise ValueError(f"nonfinite matrix row {row_number}")
            yield meta, values
        if len(seen) != n or matrix_stream.read(1):
            raise ValueError("matrix or index row count mismatch")


def restore_csv(bundle, destination, require_original_sha256=True):
    bundle, destination = Path(bundle), Path(destination)
    manifest = _validated(bundle)
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    started = time.perf_counter()
    try:
        import io
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text_stream:
                    writer = csv.writer(text_stream)
                    writer.writerow(manifest["columns"])
                    for meta, values in iter_rows(bundle):
                        writer.writerow(meta + list(values))
            raw.flush()
            os.fsync(raw.fileno())
        if require_original_sha256 and digest(name) != manifest["source"]["sha256"]:
            raise ValueError("restored CSV checksum differs from source")
        os.replace(name, destination)
        return {"path": str(destination), "sha256": digest(destination),
                "bytes": destination.stat().st_size, "restore_seconds": time.perf_counter() - started}
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def main():
    cli = argparse.ArgumentParser()
    commands = cli.add_subparsers(dest="command", required=True)
    build = commands.add_parser("pack")
    build.add_argument("source")
    build.add_argument("destination")
    build.add_argument("--source-manifest")
    restore = commands.add_parser("restore")
    restore.add_argument("bundle")
    restore.add_argument("destination")
    args = cli.parse_args()
    result = pack(args.source, args.destination, args.source_manifest) if args.command == "pack" else restore_csv(args.bundle, args.destination)
    if args.command == "pack":
        result = {key: result[key] for key in ("row_count", "feature_count", "matrix", "index", "pack_seconds")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
