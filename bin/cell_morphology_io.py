"""Portable, source-bound canonical morphology tables; no image/model imports.

Hashes establish file identity, not biological validity or a signed attestation.
Legacy tables remain inspectable but cannot assert reference compatibility.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

import numpy as np

FORMAT = "cellphenotyper_morphology"
VERSION = "2.0.0"
CONTRACT = f"{FORMAT}/{VERSION}"
TABLE = "cell_morphology.csv"
SUMMARY = "morphology_summary.json"
COMPLETION = "morphology_completion.json"
OD_EPSILON = 1.0 / 255.0
OD_MAX = math.log(256.0)
TEXTURE_LEVELS = 32
GEOMETRY_FIELDS = ["mask_pixel_count", "area_um2", "perimeter_um", "circularity", "solidity",
    "boundary_irregularity", "eccentricity", "major_axis_um", "minor_axis_um", "orientation_rad",
    "mask_centroid_x", "mask_centroid_y", "mask_centroid_x_um", "mask_centroid_y_um", "connected_components"]
APPEARANCE_FIELDS = [f"rgb_od_{channel}_{stat}" for channel in ("red", "green", "blue", "mean")
    for stat in ("mean", "std", "p10", "p90")] + ["od_glcm_contrast", "od_glcm_homogeneity",
    "od_glcm_entropy_bits", "od_gradient_mean_per_um", "od_texture_pair_count", "image_saturated_fraction"]
FIELDS = ["label", "x", "y", "geometry_source", "geometry_representation", "morphology_status",
    "morphology_contract", "texture_status", "mask_bbox_xmin", "mask_bbox_ymin", "mask_bbox_xmax",
    "mask_bbox_ymax", "touches_image_edge"] + GEOMETRY_FIELDS + APPEARANCE_FIELDS
GEOMETRY_FEATURES = ["area_um2", "perimeter_um", "eccentricity", "solidity", "circularity",
                     "boundary_irregularity", "major_axis_um", "minor_axis_um"]
TEXTURE_FEATURES = [name for name in APPEARANCE_FIELDS if name.startswith(("rgb_od_", "od_glcm_", "od_gradient_"))]


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate morphology JSON field: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"Invalid morphology JSON constant: {value}")
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("Morphology metadata must be a JSON object")
    return value


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(path):
    """Hash exact file bytes, or a contained deterministic Zarr directory tree."""
    root = Path(path).resolve(strict=True)
    if root.is_file():
        return {"kind": "file", "sha256": sha256(root), "size_bytes": root.stat().st_size}
    digest, size, count = hashlib.sha256(), 0, 0
    def visit(node, relative, ancestors):
        nonlocal size, count
        resolved = node.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved in ancestors:
            raise ValueError("Morphology source tree contains an escape or cycle")
        if resolved.is_dir():
            digest.update(json_bytes(["directory", relative]))
            for child in sorted(resolved.iterdir(), key=lambda item: item.name):
                visit(child, relative + "/" + child.name, ancestors | {resolved})
        elif resolved.is_file():
            value = identity(resolved)
            digest.update(json_bytes(["file", relative, value]))
            size += value["size_bytes"]
            count += 1
        else:
            raise ValueError("Morphology source tree contains an unsupported entry")
    visit(root, "", set())
    return {"kind": "directory", "sha256": digest.hexdigest(), "size_bytes": size, "file_count": count}


def capture_sources(paths):
    return {name: identity(path) for name, path in paths.items() if path is not None}


def check_sources(paths, expected):
    if capture_sources(paths) != expected:
        raise ValueError("Morphology source or producer changed during execution")


def producer_identity():
    paths = {name: Path(__file__).with_name(name) for name in ("profile_cell_morphology.py", "cell_morphology_io.py")}
    return {"files": capture_sources(paths), "runtime": {"numpy": np.__version__,
        **{name: importlib.metadata.version(name) for name in ("scipy", "tifffile")}}}, paths


def texture_sampling(mpp_x, mpp_y, lag_um=.5):
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in (mpp_x, mpp_y, lag_um)):
        raise ValueError("Texture MPP and requested physical lag must be positive and finite")
    sx, sy = [max(1, int(math.floor(float(lag_um) / float(mpp) + .5))) for mpp in (mpp_x, mpp_y)]
    offsets = [[sx, 0], [0, sy], [sx, sy], [-sx, sy]]
    physical = [[x * mpp_x, y * mpp_y] for x, y in offsets]
    return {"requested_axial_lag_um": float(lag_um), "offset_rounding": "nearest positive integer, half up",
            "source_mpp_xy": [float(mpp_x), float(mpp_y)], "offsets_px": offsets,
            "effective_offsets_um": physical, "effective_distances_um": [math.hypot(*v) for v in physical],
            "pair_membership": "both endpoints in same canonical nucleus; no background pairs",
            "directions": "axial x/y plus both diagonals; diagonals use the same axial steps"}


def intensity_settings(dtype, white_level=None):
    dtype = np.dtype(dtype)
    if dtype.kind not in "uif":
        raise ValueError("Nuclear RGB must have real integer or floating-point intensity values")
    maximum = float(np.iinfo(dtype).max) if dtype.kind in "ui" else None
    if white_level is None:
        if maximum is None:
            raise ValueError("Floating-point RGB requires --white-level")
        white_level = maximum
    white_level = float(white_level)
    if not math.isfinite(white_level) or white_level <= 0:
        raise ValueError("White level must be positive and finite")
    standard = (white_level == maximum) if maximum is not None else white_level == 1.0
    policy = {"policy": "full_range_normalized_transmittance"} if standard else {
        "policy": "nonstandard_explicit_white_calibration",
        "scale_relative_to_integer_dtype_max": white_level / maximum if maximum else None,
        "floating_white_scale": white_level if maximum is None else None}
    return {"source_dtype": str(dtype), "white_level": white_level,
            "white_level_origin": "dtype_max" if maximum == white_level else "explicit",
            "normalization_policy": policy,
            "formula": "-ln((RGB/white_level + epsilon)/(1+epsilon))",
            "transmittance_epsilon": OD_EPSILON, "od_range": [0.0, OD_MAX],
            "levels": TEXTURE_LEVELS, "quantization": "floor(OD/OD_max*levels), clipped to [0,levels-1]",
            "color_semantics": "first three RGB channels; alpha ignored; no stain deconvolution or stain normalization"}


def feature_definitions(settings, producer):
    shared = {"contract": CONTRACT, "producer": producer, "source_mpp_xy": settings["texture"]["source_mpp_xy"]}
    intensity = {key: value for key, value in settings["intensity"].items()
                 if key not in ("source_dtype", "white_level", "white_level_origin")}
    return {"morphology": {**shared, "method": "canonical_raster_geometry", "units": "micrometres_and_dimensionless",
        "perimeter": "exact exposed pixel edges including holes; grid dependent", "axes": "pixel-area second moments",
        "feature_names": GEOMETRY_FEATURES}, "nuclear_texture": {**shared,
        "method": "intranuclear_RGB_optical_density_appearance_proxy", "stain_deconvolved": False,
        "intensity": intensity, "sampling": settings["texture"], "feature_names": TEXTURE_FEATURES}}


def _safe_paths(table):
    table = Path(table).absolute()
    root = table.parent.resolve()
    paths = {"table": table, "summary": table.with_name(SUMMARY), "completion": table.with_name(COMPLETION)}
    for path in paths.values():
        try:
            valid = path.resolve().is_relative_to(root)
        except (OSError, RuntimeError):
            valid = False
        if not valid:
            raise ValueError("Morphology artifact path escapes its bundle or contains a cycle")
    return paths


def read_table(path, *, strict=False):
    import pandas as pd
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if not header or len(header) != len(set(header)) or "label" not in header or (strict and header != FIELDS):
            raise ValueError("Invalid morphology CSV schema/order")
        rows = list(reader)
    if any(len(row) != len(header) for row in rows):
        raise ValueError("Malformed morphology CSV row")
    frame = pd.DataFrame(rows, columns=header)
    labels = frame.label.astype(str)
    if (labels.duplicated().any() or labels.eq("").any() or labels.str.strip().ne(labels).any()
            or labels.str.contains(r"[\x00-\x1f\x7f]", regex=True).any()):
        raise ValueError("Invalid or duplicate morphology labels")
    if strict:
        if not labels.str.fullmatch(r"[1-9][0-9]*").all() or not frame.morphology_contract.eq(CONTRACT).all():
            raise ValueError("Morphology rows do not declare the exact canonical contract")
        nonnumeric = {"label", "geometry_source", "geometry_representation", "morphology_status", "morphology_contract", "texture_status"}
        missing = {"orientation_rad", "od_glcm_contrast", "od_glcm_homogeneity", "od_glcm_entropy_bits", "od_gradient_mean_per_um"}
        for name in set(FIELDS) - nonnumeric:
            try:
                values = np.asarray([float(value) if value != "" else np.nan for value in frame[name]], dtype=float)
            except ValueError as exc:
                raise ValueError(f"Nonnumeric morphology measurement: {name}") from exc
            if np.isinf(values).any() or (name not in missing and not np.isfinite(values).all()):
                raise ValueError(f"Invalid missing/nonfinite morphology measurement: {name}")
            frame[name] = values
        no_pairs = frame.od_texture_pair_count.eq(0)
        if not frame.texture_status.eq(np.where(no_pairs, "no_pairs_at_requested_physical_lag", "ok")).all():
            raise ValueError("Morphology texture status contradicts pair count")
        for name in missing - {"orientation_rad"}:
            if not np.array_equal(frame[name].isna().to_numpy(), no_pairs.to_numpy()):
                raise ValueError("Morphology missing texture values contradict pair count")
    return frame


def complete_morphology(outdir, *, sources, source_paths, producer, producer_paths, settings, geometry):
    paths = _safe_paths(Path(outdir) / TABLE)
    if paths["completion"].exists() or paths["completion"].is_symlink():
        raise FileExistsError("Morphology completion already exists")
    before = {name: identity(paths[name]) for name in ("table", "summary")}
    frame = read_table(paths["table"], strict=True)
    summary = read_json(paths["summary"])
    definitions = feature_definitions(settings, producer)
    if (summary.get("binding_contract") != CONTRACT or summary.get("cells") != len(frame)
            or summary.get("settings") != settings or summary.get("geometry") != geometry
            or summary.get("producer") != producer or summary.get("source_identities") != sources):
        raise ValueError("Morphology summary contradicts its execution binding")
    record = {"format": FORMAT, "schema_version": VERSION, "complete": True,
        "artifacts": {name: {"path": paths[name].name, **before[name]} for name in before},
        "cell_count": len(frame), "feature_columns": FIELDS, "sources": sources, "producer": producer,
        "settings": settings, "settings_sha256": hashlib.sha256(json_bytes(settings)).hexdigest(),
        "geometry": geometry, "feature_definitions": definitions,
        "reference_compatible": "resolution_json" in sources,
        "interpretation": "H&E-derived nuclear appearance, not calibrated chromatin or molecular measurements"}
    if before != {name: identity(paths[name]) for name in before}:
        raise ValueError("Morphology artifacts changed before completion")
    check_sources(source_paths, sources)
    check_sources(producer_paths, producer["files"])
    if producer_identity()[0] != producer:
        raise ValueError("Morphology producer/runtime changed before completion")
    with paths["completion"].open("xb") as handle:
        handle.write(json_bytes(record))
    return record


def load_morphology_table(table, *, expected_inputs, expected_geometry, expected_ids):
    """Return table, portable verification record and source paths to recheck."""
    paths = _safe_paths(table)
    existing = {name: path for name, path in paths.items() if path.is_file()}
    before = capture_sources(existing)
    frame = read_table(paths["table"], strict="completion" in existing)
    summary = read_json(paths["summary"]) if "summary" in existing else None
    if "completion" not in existing:
        if "morphology_contract" in frame or (summary and summary.get("binding_contract")):
            raise ValueError("Source-bound morphology lacks its completion receipt")
        check_sources(existing, before)
        return frame, {"status": "legacy_unverified", "reference_compatible": False,
            "source_sha256": before,
            "feature_definitions": {name: {"method": "legacy_morphology_unverified"} for name in ("morphology", "nuclear_texture")}}, existing
    record = read_json(paths["completion"])
    if record.get("format") != FORMAT or record.get("schema_version") != VERSION or record.get("complete") is not True:
        raise ValueError("Unsupported morphology completion contract")
    if not {"image", "labels", "objects", "shift"} <= set(expected_inputs):
        raise ValueError("Morphology verification requires expected image/mask/objects/shift identities")
    expected = {key: value for key, value in expected_inputs.items() if value is not None}
    if set(record.get("sources", {})) != set(expected):
        raise ValueError("Morphology source inventory differs from canonical profile")
    for key, value in expected.items():
        if record["sources"][key].get("sha256") != value:
            raise ValueError(f"Morphology source input hash mismatch: {key}")
    if record.get("geometry") != expected_geometry:
        raise ValueError("Morphology calibration/geometry differs from canonical profile")
    if frame.label.tolist() != list(expected_ids):
        raise ValueError("Morphology population/order differs from canonical objects")
    if set(record.get("artifacts", {})) != {"table", "summary"}:
        raise ValueError("Morphology receipt artifact inventory mismatch")
    for name in ("table", "summary"):
        if record["artifacts"][name] != {"path": paths[name].name, **before.get(name, {})}:
            raise ValueError(f"Morphology artifact hash/size mismatch: {name}")
    settings, producer = record.get("settings", {}), record.get("producer", {})
    if record.get("settings_sha256") != hashlib.sha256(json_bytes(settings)).hexdigest():
        raise ValueError("Morphology settings fingerprint mismatch")
    try:
        correct_intensity = intensity_settings(settings["intensity"]["source_dtype"], settings["intensity"]["white_level"])
        correct_texture = texture_sampling(*expected_geometry["mpp_xy"], settings["texture"]["requested_axial_lag_um"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid morphology normalization/physical settings") from exc
    if settings.get("intensity") != correct_intensity or settings.get("texture") != correct_texture:
        raise ValueError("Morphology normalization/physical offsets contradict the contract")
    if set(producer.get("files", {})) != {"profile_cell_morphology.py", "cell_morphology_io.py"} or not producer.get("runtime"):
        raise ValueError("Morphology lacks actual producer identity")
    for value in producer["files"].values():
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) or value.get("kind") != "file" or value.get("size_bytes", 0) <= 0:
            raise ValueError("Invalid morphology producer identity")
    if (record.get("feature_definitions") != feature_definitions(settings, producer)
            or record.get("feature_columns") != FIELDS or record.get("cell_count") != len(frame)
            or record.get("reference_compatible") is not ("resolution_json" in expected)
            or not summary or summary.get("binding_contract") != CONTRACT
            or summary.get("cells") != len(frame) or summary.get("settings") != settings
            or summary.get("geometry") != expected_geometry or summary.get("producer") != producer
            or summary.get("source_identities") != record["sources"]):
        raise ValueError("Morphology receipt/summary/feature definition is inconsistent")
    check_sources(existing, before)
    return frame, {"status": "verified_exact_inputs_and_population", "reference_compatible": record["reference_compatible"],
        "feature_definitions": record["feature_definitions"], "receipt": record, "source_sha256": before}, existing
