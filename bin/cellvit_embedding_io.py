"""Portable CellViT tokens bound to exact source pixels and detector identities.

Consumers use only NumPy/pandas/stdlib and never follow recorded upstream paths
or load CellViT/PyTorch. A receipt binds provenance; it is not biological proof.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT = "cellphenotyper_cellvit_embeddings"
VERSION = "1.0.0"
COMPLETION = "cellvit_embeddings_completion.json"
FILES = {"embeddings": "cellvit_embeddings.npy", "ids": "cellvit_embedding_ids.csv",
         "metadata": "cellvit_embeddings_metadata.json", "retained_population": "cellvit_cells.json",
         "raw_population": "cellvit_raw_population.csv"}
ID_COLUMNS = ["embedding_row", "cellvitpp_id", "x_px", "y_px", "source_graph_row"]
RAW_COLUMNS = ["source_graph_row", "cellvitpp_id", "x_px", "y_px"]
REPRESENTATION = "mean CellViT feature tokens within detected nucleus bounding box"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate CellViT JSON field: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite CellViT JSON constant: {value}")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=reject)


def file_record(path):
    path = Path(path)
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def population(payload):
    cells = payload.get("cells", payload) if isinstance(payload, dict) else payload
    if not isinstance(cells, (list, dict)):
        raise ValueError("CellViT population must be a list or dictionary")
    result, seen = [], set()
    for fallback, cell in (cells.items() if isinstance(cells, dict) else enumerate(cells)):
        if not isinstance(cell, dict) or "centroid" not in cell or cell.get("id", fallback) is None:
            raise ValueError("Every CellViT population member needs an ID and centroid")
        identity = str(cell.get("id", fallback))
        if not identity or identity.strip() != identity or re.search(r"[\x00-\x1f\x7f]", identity) or identity in seen:
            raise ValueError("Invalid or duplicate CellViT population ID")
        xy = np.asarray(cell["centroid"], dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("CellViT population needs finite two-dimensional centroids")
        seen.add(identity)
        result.append((identity, xy))
    return result


def capture_inputs(image, shift, resolution_json, clean_tissue_mask=None):
    """Producer-side capture; caller rechecks after inference and before receipt."""
    if not resolution_json:
        raise ValueError("Source-bound CellViT embeddings require --resolution-json with a passed report")
    paths = {"image": Path(image).resolve(), "shift": Path(shift).resolve(),
             "resolution_json": Path(resolution_json).resolve()}
    if clean_tissue_mask:
        paths["clean_tissue_mask"] = Path(clean_tissue_mask).resolve()
    inputs = {key: file_record(path) for key, path in paths.items()}
    from cell_profile_io import calibration, RasterReader
    cal = calibration(paths["shift"], paths["resolution_json"])
    shift_data = read_json(paths["shift"])
    declared_mpp = shift_data.get("source_mpp", shift_data.get("microns_per_pixel"))
    if declared_mpp is None or not np.isfinite(float(declared_mpp)) or not np.isclose(float(declared_mpp), cal["mpp"], rtol=1e-4, atol=0):
        raise ValueError("CellViT shift calibration contradicts the passed resolution report")
    with RasterReader(paths["image"]) as raster:
        if (raster.width, raster.height) != (cal["width"], cal["height"]):
            raise ValueError("CellViT source image geometry differs from shift crop size")
    geometry = {"source_mpp": cal["mpp"], "crop_size_px": [cal["width"], cal["height"]],
                "crop_origin_px": cal["origin_px"].tolist(),
                "coordinate_system": "analysis_crop_level0_xy_pixels"}
    check_inputs(paths, inputs)
    return {"inputs": inputs, "geometry": geometry}, paths


def check_inputs(paths, expected):
    if {key: file_record(path) for key, path in paths.items()} != expected:
        raise ValueError("CellViT source inputs changed during execution")


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def feature_definition(execution):
    """Exclude specimen hashes/paths; retain all representation-defining settings."""
    runtime = execution.get("runtime_identity", {})
    checkpoint = execution.get("checkpoint", {})
    definition = {"representation": REPRESENTATION, "model": execution.get("model"),
                  "taxonomy": execution.get("taxonomy"), "checkpoint_sha256": checkpoint.get("sha256"),
                  "checkpoint_scope": "segmentation model with embedded heads; external taxonomy classifier weights not captured",
                  "source_mpp": execution.get("geometry", {}).get("source_mpp"),
                  "preprocessing": execution.get("preprocessing"), "amp": execution.get("amp"),
                  "amp_semantics": "requested --enforce_amp; otherwise checkpoint/runtime default, not measured autocast",
                  "runtime": runtime.get("portable_identity"),
                  "rescaling": "CellViT process_wsi behavior bound to actual package source and commanded source_mpp"}
    # Native image dimensions/bands are provenance, not a reference feature
    # definition: slides of different size still share physical preprocessing.
    if isinstance(definition["preprocessing"], dict):
        definition["preprocessing"] = {key: value for key, value in definition["preprocessing"].items()
                                        if key not in ("native_size_px", "source_bands")}
    compatible = (runtime.get("status") == "verified_configured_python_entrypoint"
                  and runtime.get("package_version") == "1.0.9"
                  and execution.get("model") in ("HIPT", "SAM")
                  and execution.get("taxonomy") in ("pannuke", "binary")
                  and _hash(checkpoint.get("sha256")) and checkpoint.get("status") == "verified"
                  and isinstance(execution.get("amp"), bool)
                  and bool(definition["preprocessing"]) and bool(definition["runtime"]))
    return definition, bool(compatible)


def _csv(path, columns=None):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if not header or len(set(header)) != len(header) or (columns is not None and header != columns):
            raise ValueError("Invalid CellViT identity CSV schema/order")
        rows = list(reader)
        if any(len(row) != len(header) for row in rows):
            raise ValueError("Malformed CellViT identity CSV row")
    return pd.DataFrame(rows, columns=header)


def _indices(frame, name, *, count=None):
    if name not in frame:
        raise ValueError(f"CellViT identity CSV is missing {name}")
    values = frame[name].astype(str)
    if not values.str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"CellViT {name} must contain literal nonnegative integers")
    try:
        integers = np.asarray([int(value) for value in values], dtype=np.int64)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"CellViT {name} integer is out of range") from exc
    if count is not None and not np.array_equal(integers, np.arange(count)):
        raise ValueError(f"CellViT {name} row order is not exact")
    frame[name] = integers
    return integers


def _coordinates(frame):
    try:
        xy = np.asarray([[float(x), float(y)] for x, y in zip(frame.x_px, frame.y_px)], dtype=np.float64).reshape(-1, 2)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid CellViT CSV centroids") from exc
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite CellViT CSV centroids")
    frame["x_px"], frame["y_px"] = xy[:, 0], xy[:, 1]
    return xy


def _validate_ids(frame):
    if "cellvitpp_id" not in frame:
        raise ValueError("CellViT IDs are missing cellvitpp_id")
    values = frame.cellvitpp_id.astype(str)
    if values.duplicated().any() or values.eq("").any() or values.str.strip().ne(values).any() or values.str.contains(r"[\x00-\x1f\x7f]", regex=True).any():
        raise ValueError("Invalid or duplicate literal CellViT IDs")


def _matrix(path, count):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 2 or data.shape[0] != count or data.shape[1] < 1 or data.dtype != np.dtype("float32"):
        raise ValueError("CellViT embedding matrix must be aligned float32 N x D")
    for start in range(0, count, 4096):
        if not np.isfinite(data[start:start + 4096]).all():
            raise ValueError("Nonfinite CellViT embedding vector")
    return data


def _payloads(root, execution=None):
    ids = _csv(root / FILES["ids"], ID_COLUMNS if execution is not None else None)
    _validate_ids(ids)
    _indices(ids, "embedding_row", count=len(ids))
    data = _matrix(root / FILES["embeddings"], len(ids))
    if execution is None:
        if ("x_px" in ids) != ("y_px" in ids):
            raise ValueError("Legacy CellViT IDs contain partial centroid coordinates")
        if "x_px" in ids:
            _coordinates(ids)
        return data, ids
    raw = _csv(root / FILES["raw_population"], RAW_COLUMNS)
    _validate_ids(raw)
    _indices(raw, "source_graph_row", count=len(raw))
    raw_xy, retained_xy = _coordinates(raw), _coordinates(ids)
    source_rows = _indices(ids, "source_graph_row")
    if len(set(source_rows)) != len(source_rows) or (source_rows >= len(raw)).any():
        raise ValueError("CellViT source_graph_row must select distinct raw population rows")
    retained = population(read_json(root / FILES["retained_population"]))
    if [identity for identity, _ in retained] != ids.cellvitpp_id.tolist():
        raise ValueError("CellViT retained population IDs/order differ from embedding rows")
    expected_xy = np.asarray([xy for _, xy in retained], dtype=np.float64).reshape(-1, 2)
    if not np.array_equal(expected_xy, retained_xy):
        raise ValueError("CellViT retained population centroids differ from embedding IDs")
    if raw.cellvitpp_id.iloc[source_rows].tolist() != ids.cellvitpp_id.tolist() or not np.array_equal(raw_xy[source_rows], retained_xy):
        raise ValueError("CellViT retained population differs from its raw source rows")
    width, height = execution["geometry"]["crop_size_px"]
    if ((retained_xy < 0).any() or (retained_xy[:, 0] >= width).any() or (retained_xy[:, 1] >= height).any()):
        raise ValueError("Retained CellViT centroids fall outside their source crop")
    return data, ids


def _validate_execution(execution):
    if not isinstance(execution, dict):
        raise ValueError("Missing CellViT execution binding")
    inputs = execution.get("inputs", {})
    if not {"image", "shift", "resolution_json"} <= set(inputs):
        raise ValueError("CellViT execution requires exact image/shift/resolution bindings")
    for record in inputs.values():
        if not isinstance(record, dict) or not _hash(record.get("sha256")) or not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ValueError("Invalid CellViT source hash/size binding")
    geometry = execution.get("geometry", {})
    try:
        mpp, size, origin = float(geometry["source_mpp"]), geometry["crop_size_px"], geometry["crop_origin_px"]
        if not np.isfinite(mpp) or not .01 <= mpp <= 10 or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size) or len(origin) != 2 or not np.isfinite(origin).all():
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid CellViT source calibration/geometry") from None
    if geometry.get("coordinate_system") != "analysis_crop_level0_xy_pixels":
        raise ValueError("Unsupported CellViT source coordinate system")
    prep = execution.get("preprocessing", {})
    if (prep.get("native_size_px") != size or prep.get("bands") != 3
            or prep.get("tile_size_px") != [512, 512] or prep.get("compression") != "jpeg"
            or prep.get("jpeg_quality") != 92 or prep.get("pyramid") is not True
            or prep.get("bigtiff") is not True or not prep.get("channel_conversion")
            or not isinstance(execution.get("amp"), bool)):
        raise ValueError("Incomplete or contradictory CellViT physical preprocessing/AMP")
    if not _hash(execution.get("prepared_image", {}).get("sha256")):
        raise ValueError("Missing CellViT prepared-image identity")
    definition, compatible = feature_definition(execution)
    # A claimed verified runtime must carry all immutable identity components.
    runtime = execution.get("runtime_identity", {})
    if runtime.get("status") == "verified_configured_python_entrypoint":
        portable = runtime.get("portable_identity", {})
        for key in ("package_source_sha256", "executable_sha256", "interpreter_sha256"):
            if not _hash(portable.get(key)):
                raise ValueError("Incomplete verified CellViT runtime identity")
        if portable.get("package_version") != runtime.get("package_version") or not portable.get("entry_point"):
            raise ValueError("Contradictory CellViT runtime package identity")
    return definition, compatible


def _bundle_paths(root, *, required):
    """Resolve every fixed payload before opening any payload, including legacy."""
    root = Path(root).resolve()
    paths = {}
    for key, name in {**FILES, "receipt": COMPLETION}.items():
        path = root / name
        try:
            contained = path.resolve().is_relative_to(root)
        except (OSError, RuntimeError):
            contained = False
        if not contained:
            raise ValueError("CellViT artifact path escapes its bundle or contains a symlink cycle")
        if path.is_file():
            paths[key] = path
        elif key != "receipt" and required:
            raise ValueError(f"Missing CellViT bundle payload: {key}")
    return paths


def _validate_metadata(root, metadata, data, ids, raw_count):
    expected = {"schema_version": 1, "status": "exported", "binding_contract": f"{FORMAT}/{VERSION}",
                "features_file": FILES["embeddings"], "ids_file": FILES["ids"],
                "representation": REPRESENTATION, "dtype": "float32",
                "raw_cell_count": raw_count, "retained_cell_count": len(ids),
                "excluded_cell_count": raw_count - len(ids), "embedding_dimension": data.shape[1],
                "missing_embedding_count": 0, "coordinate_system": "analysis_crop_level0_xy_pixels",
                "alignment": "upstream_row_order_verified_by_centroids_then_filtered_by_id",
                "centroid_absolute_tolerance_px": .05, "centroid_relative_tolerance": 0.,
                "retained_centroid_comparison": "exact_raw_json_coordinates"}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("CellViT metadata contradicts its population/array/alignment contract")
    upstream = metadata.get("upstream_artifacts", {})
    if set(upstream) != {"raw_cells_json", "retained_cells_json", "raw_graph"}:
        raise ValueError("CellViT metadata lacks raw/retained/graph source identities")
    for record in upstream.values():
        if (not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}
                or not _hash(record.get("sha256")) or type(record.get("size_bytes")) is not int
                or record["size_bytes"] < 0):
            raise ValueError("Invalid CellViT upstream artifact identity")
    if upstream["retained_cells_json"] != file_record(root / FILES["retained_population"]):
        raise ValueError("CellViT retained population differs from producer source identity")


def complete_embedding_bundle(outdir, execution, *, source_paths=None):
    """Call only after all source postchecks and final wrapper metadata writes."""
    root = Path(outdir).resolve()
    receipt_path = root / COMPLETION
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("CellViT embedding completion already exists")
    paths = _bundle_paths(root, required=True)
    files = {key: {"path": FILES[key], **file_record(path)} for key, path in paths.items()}
    definition, compatible = _validate_execution(execution)
    data, ids = _payloads(root, execution)
    metadata = read_json(root / FILES["metadata"])
    raw_count = len(_csv(root / FILES["raw_population"], RAW_COLUMNS))
    _validate_metadata(root, metadata, data, ids, raw_count)
    record = {"format": FORMAT, "schema_version": VERSION, "complete": True,
              "execution": execution, "files": files,
              "shape": list(data.shape), "dtype": "float32", "raw_cell_count": raw_count,
              "retained_cell_count": len(ids), "excluded_cell_count": raw_count - len(ids),
              "feature_definition": definition, "reference_compatible": compatible,
              "interpretation": "Source-bound H&E morphology tokens, not molecular measurements or biological validation"}
    for key, name in FILES.items():
        if files[key] != {"path": name, **file_record(root / name)}:
            raise ValueError("CellViT payload changed before completion")
    if source_paths is not None:
        check_inputs(source_paths, execution["inputs"])
    with receipt_path.open("xb") as handle:
        handle.write(json_bytes(record))
    return record


def load_cellvit_embedding_bundle(source, *, expected_inputs, expected_geometry):
    root = Path(source).resolve()
    receipt_path = root / COMPLETION
    paths = _bundle_paths(root, required=False)
    before = {key: file_record(path) for key, path in paths.items()}
    metadata = read_json(root / FILES["metadata"])
    if not receipt_path.is_file():
        if metadata.get("binding_contract"):
            raise ValueError("Source-bound CellViT embeddings lack their completion receipt")
        data, ids = _payloads(root)
        if {key: file_record(path) for key, path in paths.items()} != before:
            raise ValueError("Legacy CellViT artifacts changed during validation")
        retained_hash = sha256(paths["retained_population"]) if "retained_population" in paths else None
        return data, ids, {"status": "legacy_unverified", "reference_compatible": False,
                           "feature_definition": {"representation": metadata.get("representation"), "source_binding": "legacy_unverified"},
                           "retained_population_sha256": retained_hash, "receipt": None, "summary": metadata}, paths
    receipt_bytes = receipt_path.read_bytes()
    record = read_json(receipt_path)
    if record.get("format") != FORMAT or record.get("schema_version") != VERSION or record.get("complete") is not True:
        raise ValueError("Unsupported CellViT embedding completion schema")
    execution = record.get("execution")
    definition, compatible = _validate_execution(execution)
    if not {"image", "shift", "resolution_json"} <= set(expected_inputs or {}):
        raise ValueError("CellViT source validation requires expected image/shift/resolution hashes")
    for key, expected in expected_inputs.items():
        if execution["inputs"].get(key, {}).get("sha256") != expected:
            raise ValueError(f"CellViT source input hash mismatch: {key}")
    for key in ("source_mpp", "crop_size_px", "crop_origin_px"):
        if (expected_geometry or {}).get(key) != execution["geometry"].get(key):
            raise ValueError(f"CellViT source geometry/calibration mismatch: {key}")
    if set(record.get("files", {})) != set(FILES):
        raise ValueError("CellViT completion payload inventory differs from contract")
    paths = {}
    for key, name in FILES.items():
        artifact = record["files"][key]
        path = root / name
        if artifact.get("path") != name or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("CellViT payload path escapes its bundle or is missing")
        if artifact != {"path": name, **file_record(path)}:
            raise ValueError(f"CellViT payload hash/size mismatch: {key}")
        paths[key] = path
    data, ids = _payloads(root, execution)
    if (record.get("shape") != list(data.shape) or record.get("dtype") != "float32"
            or record.get("retained_cell_count") != len(ids)
            or record.get("raw_cell_count") != len(_csv(paths["raw_population"], RAW_COLUMNS))
            or record.get("excluded_cell_count") != record["raw_cell_count"] - len(ids)
            or record.get("feature_definition") != definition or record.get("reference_compatible") != compatible):
        raise ValueError("CellViT completion geometry/identity/feature definition differs from payloads")
    _validate_metadata(root, metadata, data, ids, record["raw_cell_count"])
    if any(file_record(path) != before.get(key) for key, path in paths.items()):
        raise ValueError("CellViT payload changed during validation")
    for key, path in paths.items():
        if file_record(path) != {k: v for k, v in record["files"][key].items() if k != "path"}:
            raise ValueError("CellViT payload changed during validation")
    if receipt_path.read_bytes() != receipt_bytes:
        raise ValueError("CellViT completion changed during validation")
    paths["receipt"] = receipt_path
    return data, ids, {"status": "verified_exact_inputs_and_population", "reference_compatible": compatible,
                       "feature_definition": definition, "retained_population_sha256": record["files"]["retained_population"]["sha256"],
                       "receipt": record, "summary": metadata}, paths
