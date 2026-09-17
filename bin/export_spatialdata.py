#!/usr/bin/env python3
"""Export real SpatialData/Zarr elements in original-slide micrometres.

Uses the official Image2DModel, Labels2DModel, ShapesModel and TableModel APIs:
https://spatialdata.scverse.org/en/stable/api/models.html
https://spatialdata.scverse.org/en/stable/api/SpatialData.html

TIFF pixels are decoded only through bounded RasterReader windows. Native
resolution is retained; feature blocks remain separate AnnData obsm arrays,
with missing values intact and no transcript-count normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cell_profile_io import RasterReader, calibration, sha256_file


CELL_LABELS = "canonical_cells"
CELL_RING_LABELS = "canonical_perinuclear_ring"
HE_IMAGE = "he_image"
TISSUE_SHAPES = "tissue_domains"
COORDINATE_SYSTEM = "original_um"
HIERARCHY_LABELS = {"parent": "hierarchy_parent_domains", "subdomain": "hierarchy_subdomains",
                    "region": "hierarchy_regions", "status": "hierarchy_status",
                    "parent_uncertainty": "hierarchy_parent_uncertainty"}
NATIVE_SUPPORT_LABELS = {"support": "neighborhood_tissue_support",
                         "reasons": "neighborhood_tissue_support_reasons"}
NATIVE_SUPPORT_CODES = {"0": "upstream_excluded", "1": "retained_support",
    "2": "image_rule_excluded_bright_interior", "3": "retained_ambiguous_bright_pixel"}
REFERENCE_FIELDS = ("reference_assignment", "reference_status", "reference_distance",
    "reference_radius", "reference_margin", "reference_nearest_group", "reference_atlas_id")


def _window(path, x0, y0, x1, y1, channels):
    with RasterReader(path) as reader:
        array = reader.window(x0, y0, x1, y1)
    if channels:
        return array[None] if array.ndim == 2 else np.moveaxis(array, -1, 0)
    return array


def lazy_raster(path, tile_size=1024, channels=False):
    """Construct a lazy native-resolution Dask array without retaining TIFF handles."""
    import dask.array as da
    from dask import delayed

    if tile_size < 16:
        raise ValueError("tile_size must be at least 16 pixels")
    path = str(Path(path).resolve())
    with RasterReader(path) as reader:
        height, width, dtype = reader.height, reader.width, reader.dtype
        shape = reader.reader.shape
    if not channels and len(shape) != 2:
        raise ValueError("Canonical labels must be a two-dimensional integer raster")
    count = shape[2] if len(shape) == 3 else 1
    rows = []
    for y0 in range(0, height, tile_size):
        columns = []
        for x0 in range(0, width, tile_size):
            y1, x1 = min(height, y0 + tile_size), min(width, x0 + tile_size)
            block_shape = (count, y1 - y0, x1 - x0) if channels else (y1 - y0, x1 - x0)
            task = delayed(_window, pure=True)(path, x0, y0, x1, y1, channels)
            columns.append(da.from_delayed(task, shape=block_shape, dtype=dtype))
        rows.append(da.concatenate(columns, axis=2 if channels else 1))
    return da.concatenate(rows, axis=1 if channels else 0)


def physical_transform(cal, source="crop_pixels"):
    from spatialdata.transformations import Affine

    matrix = np.eye(3, dtype=float)
    if source in ("crop_pixels", "original_pixels"):
        matrix[0, 0] = matrix[1, 1] = cal["mpp"]
    if source == "crop_pixels":
        matrix[:2, 2] = cal["origin_px"] * cal["mpp"]
    elif source not in ("original_pixels", "original_um"):
        raise ValueError("Specify tissue coordinates explicitly: crop_pixels, original_pixels, or original_um")
    return Affine(matrix, input_axes=("x", "y"), output_axes=("x", "y"))


def _contained(root, path):
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("A profile artifact path escapes its profile directory")
    return candidate


def native_support_elements(root, manifest, cal, source_hashes, tile_size):
    """Export only a profile-bound categorical bundle, without opening upstream paths.

    The original coarse mask is not an export argument. Its recorded identity
    must agree with the profile, but its pixels and the image-based rule are not
    recomputed here. Both bundled native rasters are checked in full.
    """
    from spatialdata.models import Labels2DModel

    graph = manifest.get("neighborhoods", {}).get("graph_support", {})
    if not graph.get("producer_manifest"):
        if (any(str(name).startswith("neighborhood_support/") for name in manifest.get("files", {}))
                or str(graph.get("path", "")).startswith("neighborhood_support/")):
            raise ValueError("Native neighborhood support bundle lacks its producer-manifest linkage")
        return {}, None, None
    expected_paths = {"manifest": "neighborhood_support/support_manifest.json",
        "support": "neighborhood_support/support.tif", "reasons": "neighborhood_support/reasons.tif"}
    if (graph.get("path_basis") != "profile_directory"
            or graph.get("producer_manifest") != expected_paths["manifest"]
            or graph.get("path") != expected_paths["support"]):
        raise ValueError("Native neighborhood support requires its exact contained profile bundle paths")
    paths, hashes = {}, {}
    for key, relative in expected_paths.items():
        raw = root / relative
        if raw.is_symlink() or raw.parent.is_symlink():
            raise ValueError("Native neighborhood support bundle must not contain symlinks")
        path = _contained(root, relative)
        expected = manifest.get("files", {}).get(relative)
        if (not path.is_file() or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected):
            raise ValueError(f"Native neighborhood support profile artifact hash mismatch: {key}")
        paths[key], hashes[key] = path, expected
    if (graph.get("producer_manifest_sha256") != hashes["manifest"]
            or graph.get("sha256") != hashes["support"]
            or manifest.get("inputs", {}).get("native_support_mask_sha256") != hashes["support"]):
        raise ValueError("Native neighborhood support graph/profile identities disagree")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate native neighborhood support receipt key")
            result[key] = value
        return result

    receipt = json.loads(paths["manifest"].read_text(), object_pairs_hook=unique_object)
    if (receipt.get("format") != "cellphenotyper_native_brightfield_support"
            or receipt.get("schema_version") not in {"1.0.0", "1.1.0"} or receipt.get("complete") is not True
            or receipt.get("reason_codes") != NATIVE_SUPPORT_CODES
            or receipt.get("method") != "strict_near_white_low_chroma_low_variation_interior_subtraction"
            or receipt.get("biological_validation") != "not_established"):
        raise ValueError("Unsupported native neighborhood support schema or categorical semantics")
    for source in ("image", "shift", "resolution_json"):
        expected = source_hashes.get(source)
        if (not expected or manifest.get("inputs", {}).get(source + "_sha256") != expected
                or receipt.get("inputs", {}).get(source, {}).get("sha256") != expected):
            raise ValueError(f"Native neighborhood support source binding mismatch: {source}")
    upstream_hash = receipt.get("inputs", {}).get("support_mask", {}).get("sha256")
    if (not isinstance(upstream_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", upstream_hash)
            or manifest.get("inputs", {}).get("support_mask_sha256") != upstream_hash):
        raise ValueError("Native neighborhood support upstream mask identity disagrees with the profile")
    shape = [cal["height"], cal["width"]]
    mpp = [cal["mpp"]] * 2
    origin = (cal["origin_px"] * cal["mpp"]).tolist()
    if (receipt.get("coordinate_frame") != "crop_pixels" or receipt.get("shape_yx") != shape
            or receipt.get("mpp_xy") != mpp
            or receipt.get("origin_original_pixels_xy") != cal["origin_px"].tolist()
            or graph.get("source_coordinates") != "crop_pixels" or graph.get("explicit_native_support") is not True
            or graph.get("shape_yx") != shape or graph.get("mpp_xy") != mpp or graph.get("origin_um_xy") != origin
            or graph.get("view_window_xyxy") != [0, 0, cal["width"], cal["height"]]
            or graph.get("graph_and_density_resampling") != "none"):
        raise ValueError("Native neighborhood support geometry disagrees with the exact exported crop")
    for key in NATIVE_SUPPORT_LABELS:
        output = receipt.get("outputs", {}).get(key, {})
        if (output.get("filename") != paths[key].name or output.get("sha256") != hashes[key]
                or output.get("bytes") != paths[key].stat().st_size):
            raise ValueError("Native neighborhood support producer output identity mismatch")
    counts = np.zeros(4, dtype=np.int64)
    with RasterReader(paths["support"]) as support, RasterReader(paths["reasons"]) as reasons:
        if any(reader.reader.shape != tuple(shape) or reader.dtype != np.dtype("uint8")
               for reader in (support, reasons)):
            raise ValueError("Native neighborhood support rasters require exact uint8 crop geometry")
        for y in range(0, shape[0], tile_size):
            for x in range(0, shape[1], tile_size):
                bounds = (x, y, min(x + tile_size, shape[1]), min(y + tile_size, shape[0]))
                mask, codes = support.window(*bounds), reasons.window(*bounds)
                if (np.any(codes > 3) or np.any(mask > 1)
                        or not np.array_equal(mask, np.isin(codes, (1, 3)))):
                    raise ValueError("Native neighborhood support binary/reason categorical consistency failure")
                counts += np.bincount(codes.ravel(), minlength=4)
    if receipt.get("pixel_counts") != {str(key): int(value) for key, value in enumerate(counts)}:
        raise ValueError("Native neighborhood support reason counts disagree with its native pixels")
    info = {"paths": paths, "hashes": hashes}
    _verify_native_support_sources(info)
    elements = {NATIVE_SUPPORT_LABELS[key]: Labels2DModel.parse(lazy_raster(paths[key], tile_size),
        dims=("y", "x"), transformations={COORDINATE_SYSTEM: physical_transform(cal)})
        for key in NATIVE_SUPPORT_LABELS}
    record = {"status": "verified_profile_bound_native_categorical_bundle", "label_elements": NATIVE_SUPPORT_LABELS,
        "source_profile_paths_to_label_elements": {expected_paths[key]: name for key, name in NATIVE_SUPPORT_LABELS.items()},
        "source_manifest_profile_path": expected_paths["manifest"], "producer_manifest_sha256": hashes["manifest"],
        "producer_manifest_json": json.dumps(receipt, sort_keys=True),
        "source_output_hashes": {key: hashes[key] for key in NATIVE_SUPPORT_LABELS},
        "semantics": "Categorical graph/density support, NOT cell instance masks; cells remain linked only to canonical_cells.",
        "support_values": {"0": "excluded_from_neighborhood_graph_and_density_support", "1": "retained_support"},
        "reason_codes": NATIVE_SUPPORT_CODES, "biological_validation": "not_established",
        "native_pixel_verification": "all_pixels_uint8_binary_reason_consistency_and_counts",
        "upstream_support_verification": "receipt_hash_matches_profile; upstream_pixels_not_reopened_or_recomputed",
        "image_rule_verification": "source_image_identity_verified; classification_not_recomputed",
        "pyramid_policy": "native_single_scale_categorical_elements"}
    return elements, record, info


def _verify_native_support_sources(info):
    for key, path in info["paths"].items():
        if path.is_symlink() or not path.is_file() or sha256_file(path) != info["hashes"][key]:
            raise ValueError("Native neighborhood support bundle changed during SpatialData export")


def _verify_native_support_roundtrip(reloaded, table, info, record, tile_size, cal):
    from spatialdata.transformations import get_transformation

    _verify_native_support_sources(info)
    expected_transform = physical_transform(cal).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    for key, name in NATIVE_SUPPORT_LABELS.items():
        actual = reloaded.labels[name]
        transform = get_transformation(actual, to_coordinate_system=COORDINATE_SYSTEM)
        if not np.array_equal(transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), expected_transform):
            raise RuntimeError("SpatialData round trip changed native neighborhood support transform")
        with RasterReader(info["paths"][key]) as reader:
            if tuple(actual.shape) != (reader.height, reader.width) or actual.dtype != reader.dtype:
                raise RuntimeError("SpatialData round trip changed native neighborhood support shape/dtype")
            for y in range(0, reader.height, tile_size):
                for x in range(0, reader.width, tile_size):
                    x1, y1 = min(x + tile_size, reader.width), min(y + tile_size, reader.height)
                    if not np.array_equal(actual.data[y:y1, x:x1].compute(), reader.window(x, y, x1, y1)):
                        raise RuntimeError("SpatialData round trip changed native neighborhood support pixels")
    if (json.loads(table.uns["cellphenotyper"]["neighborhood_support_json"]) != record
            or reloaded.attrs["cellphenotyper"].get("neighborhood_support") != record):
        raise RuntimeError("SpatialData round trip changed native neighborhood support semantics/receipt")
    _verify_native_support_sources(info)


def load_cell_profile(profile_dir, cal):
    root = Path(profile_dir).resolve()
    manifest_path = root / "cell_profiles_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("observation_unit", "cell") != "cell":
        raise ValueError("Canonical-label annotation requires a cell profile")
    table_path = root / "cell_profiles.parquet"
    if table_path.is_file():
        cells = pd.read_parquet(table_path)
    else:
        table_path = root / "cell_profiles.csv"
        # Detector identifiers are labels, not measurements. Do not strip zero
        # padding while falling back from the portable Parquet table to CSV.
        identity_columns = ("cell_uid", "cell_id", "sample_id", "stardist_id", "cellvitpp_id", "hovernet_id", "consensus_id")
        cells = pd.read_csv(table_path, dtype={key: str for key in identity_columns})
    required = {"cell_uid", "cell_id", "sample_id", "x_um", "y_um"}
    if not required <= set(cells):
        raise ValueError(f"Cell profile is missing {sorted(required - set(cells))}")
    for column in ("cell_uid", "cell_id", "sample_id"):
        if cells[column].isna().any() or cells[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing profile identity: {column}")
        cells[column] = cells[column].astype(str)
    if cells.cell_uid.duplicated().any() or cells.cell_id.duplicated().any() or cells.sample_id.nunique() != 1:
        raise ValueError("SpatialData export requires one sample and unique canonical cell IDs")
    if manifest.get("cell_count", len(cells)) != len(cells):
        raise ValueError("Cell count does not match the profile manifest")
    if manifest.get("files", {}).get(table_path.name) not in (None, sha256_file(table_path)):
        raise ValueError("Cell profile table does not match its manifest hash")
    ids_path = root / "feature_rows.csv"
    rows = pd.read_csv(ids_path, dtype=str, keep_default_na=False)
    identity = ["sample_id", "cell_id", "cell_uid"]
    if not set(identity) <= set(rows) or not rows[identity].equals(cells[identity]):
        raise ValueError("Feature rows do not align with canonical table identities")
    if not np.isclose(float(manifest["source_mpp"]), cal["mpp"], rtol=1e-6):
        raise ValueError("Profile and image pixel calibration disagree")
    if not np.allclose(manifest["crop_origin_um"], cal["origin_px"] * cal["mpp"], atol=1e-6):
        raise ValueError("Profile and image crop offsets disagree")
    if manifest.get("crop_size_px") != [cal["width"], cal["height"]]:
        raise ValueError("Profile and image crop dimensions disagree")
    xy = cells[["x_um", "y_um"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite profile coordinates")
    origin = cal["origin_px"] * cal["mpp"]
    size = np.array([cal["width"], cal["height"]]) * cal["mpp"]
    if (xy < origin).any() or (xy >= origin + size).any():
        raise ValueError("Cell profile coordinates lie outside the declared crop")
    if {"x_crop_px", "y_crop_px"} <= set(cells):
        expected = (cells[["x_crop_px", "y_crop_px"]].to_numpy(float) + cal["origin_px"]) * cal["mpp"]
        if not np.allclose(xy, expected, atol=1e-6):
            raise ValueError("Crop and original-physical cell coordinates disagree")
    try:
        instance_ids = np.asarray([int(value) for value in cells.cell_id], dtype=np.int64)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Canonical cell IDs must be positive integer label values") from exc
    if (instance_ids <= 0).any() or len(np.unique(instance_ids)) != len(instance_ids):
        raise ValueError("Canonical IDs do not map uniquely to positive raster labels")
    return root, cells, manifest, instance_ids


def verify_label_identity(path, expected_ids, cal, tile_size):
    expected = set(map(int, expected_ids))
    observed = set()
    with RasterReader(path) as reader:
        if len(reader.reader.shape) != 2 or reader.dtype.kind not in "ui":
            raise ValueError("Canonical labels must have an integer YX dtype")
        if (reader.width, reader.height) != (cal["width"], cal["height"]):
            raise ValueError("Canonical labels and native crop dimensions differ")
        if expected and max(expected) > np.iinfo(reader.dtype).max:
            raise ValueError("Canonical IDs exceed the label raster dtype")
        for y0 in range(0, reader.height, tile_size):
            for x0 in range(0, reader.width, tile_size):
                values = np.unique(reader.window(x0, y0, min(reader.width, x0 + tile_size), min(reader.height, y0 + tile_size)))
                seen = set(map(int, values)) - {0}
                if seen - expected:
                    raise ValueError(f"Label raster contains foreign cell IDs: {sorted(seen - expected)[:5]}")
                observed.update(seen)
        dtype = reader.dtype
    if observed != expected:
        raise ValueError(f"Canonical cells missing from label raster: {sorted(expected - observed)[:5]}")
    return dtype


def _safe_obs(frame):
    """Keep string missingness portable across AnnData/pandas releases."""
    result = frame.copy()
    for name in result:
        if isinstance(result[name].dtype, pd.CategoricalDtype):
            result[name] = result[name].cat.remove_unused_categories()
        elif pd.api.types.is_string_dtype(result[name].dtype) or result[name].dtype == object:
            result[name] = pd.Categorical(result[name].fillna("").astype(str))
    return result


def cell_table(root, cells, manifest, instance_ids, label_dtype, tile_size):
    import anndata as ad
    import dask.array as da
    from scipy import sparse
    from spatialdata.models import TableModel
    from spatialdata import sanitize_table

    obs = cells.copy()
    obs.index = pd.Index(cells.cell_uid, name="cell_uid_index")
    if {"spatial_region", "instance_id"} & {str(name).lower() for name in obs}:
        raise ValueError("Profile contains reserved SpatialData annotation columns")
    obs["spatial_region"] = pd.Categorical([CELL_LABELS] * len(obs))
    obs["instance_id"] = instance_ids.astype(label_dtype)
    table = ad.AnnData(X=sparse.csr_matrix((len(obs), 0), dtype=np.float32), obs=_safe_obs(obs))
    table.obsm["spatial"] = cells[["x_um", "y_um"]].to_numpy(float)
    for name, record in manifest.get("feature_blocks", {}).items():
        if name.lower() == "spatial" or "/" in name:
            raise ValueError(f"Feature block name cannot be represented safely in obsm: {name}")
        path = _contained(root, record["path"])
        if record.get("sha256") not in (None, sha256_file(path)):
            raise ValueError(f"Feature block {name} content does not match manifest hash")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != len(cells) or list(values.shape) != record["shape"] or values.dtype.kind not in "fiu":
            raise ValueError(f"Feature block {name} is not row-aligned numeric data")
        table.obsm[name] = da.from_array(values, chunks=(tile_size, values.shape[1]), asarray=False)
    for name, record in manifest.get("spatial_graphs", {}).items():
        path = _contained(root, record["path"])
        if record.get("sha256") not in (None, sha256_file(path)):
            raise ValueError(f"Spatial graph {name} content does not match manifest hash")
        graph = sparse.load_npz(path).tocsr()
        if graph.shape != (len(cells), len(cells)) or not np.isfinite(graph.data).all():
            raise ValueError(f"Spatial graph {name} is not a finite canonical-row adjacency matrix")
        key = "neighbors_" + str(name).replace(".", "_") + "um"
        if "/" in key or key in table.obsp:
            raise ValueError("Spatial graph names are not unique portable obsp keys")
        table.obsp[key] = graph
    table.uns["cellphenotyper"] = {"observation_unit": "cell", "coordinate_system": COORDINATE_SYSTEM,
        "profile_manifest_json": json.dumps(manifest, sort_keys=True),
        "feature_semantics": "separate feature blocks; virtual markers are predictions, not measured abundance",
        "X_semantics": "empty; use the explicitly named obsm feature groups"}
    # Neighborhood phenotype columns can contain ':' and other characters that
    # SpatialData forbids in keys. Use its public naming API and retain a fully
    # reversible source-to-export mapping; never change IDs or values.
    original_names = {attr: list(getattr(table, attr).keys()) for attr in ("obs", "obsm", "obsp")}
    sanitize_table(table, inplace=True)
    name_mapping = {attr: dict(zip(names, getattr(table, attr).keys())) for attr, names in original_names.items()}
    table.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(name_mapping, sort_keys=True)
    return TableModel.parse(table, region=CELL_LABELS, region_key="spatial_region", instance_key="instance_id")


def tissue_elements(path, source_coordinates, cal):
    import anndata as ad
    import geopandas as gpd
    from scipy import sparse
    from shapely.geometry import shape
    from spatialdata.models import ShapesModel, TableModel

    transform = physical_transform(cal, source_coordinates)
    payload = json.loads(Path(path).read_text())
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Tissue annotation must be a GeoJSON FeatureCollection")
    rows, geometries = [], []
    for index, feature in enumerate(payload.get("features", []), 1):
        geometry = shape(feature["geometry"])
        if geometry.geom_type not in ("Polygon", "MultiPolygon") or geometry.is_empty or not geometry.is_valid:
            raise ValueError(f"Tissue feature {index} must be a valid nonempty polygon/multipolygon")
        if not np.isfinite(geometry.bounds).all():
            raise ValueError("Tissue polygon contains nonfinite coordinates")
        properties = feature.get("properties") or {}
        rows.append({"domain_instance_id": index, "geojson_feature_id": str(feature.get("id", index)),
            "domain_label": str(properties.get("value", properties.get("cluster", "unknown"))),
            "source_properties_json": json.dumps(properties, sort_keys=True)})
        geometries.append(geometry)
    if not rows:
        raise ValueError("Tissue GeoJSON contains no polygon features")
    frame = pd.DataFrame(rows).set_index("domain_instance_id", drop=False)
    polygons = gpd.GeoDataFrame(frame.copy(), geometry=geometries)
    polygons.index = pd.Index(frame.index, name="domain_instance_id")
    shapes = ShapesModel.parse(polygons, transformations={COORDINATE_SYSTEM: transform})
    obs = frame.copy()
    obs.index = pd.Index([f"domain:{i}" for i in frame.index], name="domain_uid")
    obs["spatial_region"] = pd.Categorical([TISSUE_SHAPES] * len(obs))
    table = ad.AnnData(X=sparse.csr_matrix((len(obs), 0), dtype=np.float32), obs=_safe_obs(obs))
    table.uns["cellphenotyper"] = {"observation_unit": "tissue_region", "source_coordinate_system": source_coordinates,
        "value_semantics": "unsupervised domain labels; no expert biological identity inferred"}
    return shapes, TableModel.parse(table, region=TISSUE_SHAPES, region_key="spatial_region", instance_key="domain_instance_id")


def measured_cell_table(package, profile_root, cells, profile_manifest, instance_ids,
                        label_dtype, tile_size, *, region_link=None):
    """Verify an imported assay and link it to its explicit observation geometry.

    Source assay files need not remain mounted: their hashes are import-time
    provenance. Every packaged output and the current canonical profile lineage
    are verified here. No matching or biological validation is inferred.
    """
    import anndata as ad
    import dask.array as da
    import pyarrow.parquet as pq
    from spatialdata import sanitize_table
    from spatialdata.models import TableModel

    root = Path(package).resolve()
    is_region = region_link is not None
    unit, identity_key = ("spatial_bin", "region_uid") if is_region else ("cell", "cell_uid")
    spatial_element = region_link["shape_name"] if is_region else CELL_LABELS
    manifest_path = root / "measured_assay_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "cellphenotyper.measured_assay.v1" or manifest.get("modality") != "measured_assay":
        raise ValueError("Measured package requires cellphenotyper.measured_assay.v1")
    if manifest.get("observation_unit") != unit or manifest.get("identity_key") != identity_key:
        raise ValueError("Region/spatial_bin measured packages require explicit region-shape linkage; they cannot annotate canonical cell labels")
    if ((not is_region and (manifest.get("reference_design") != "same_section_registered"
                           or "visium" in str(manifest.get("assay_platform", "")).lower()))
            or (is_region and manifest.get("reference_design") not in {"same_section_registered", "serial_section_region_level"})):
        raise ValueError("Cell-linked measured data must be same-section; serial sections and Visium remain region-level")
    if manifest.get("coordinate_system") != "original_slide_micrometres":
        raise ValueError("Measured package has incompatible coordinate units/frame")
    coordinate_contract = manifest.get("coordinates", {})
    if (coordinate_contract.get("target_frame") != "original_slide_micrometres"
            or coordinate_contract.get("target_units") != "um"
            or coordinate_contract.get("source_units") not in {"pixel", "um"}
            or not coordinate_contract.get("source_frame")):
        raise ValueError("Measured package coordinate lineage is incomplete or inconsistent")
    if manifest.get("predicted_features_modified") is not False:
        raise ValueError("Measured package must explicitly preserve predicted features")
    sample = str(cells.sample_id.iloc[0])
    if manifest.get("sample_id") != sample or manifest.get("observation_count") != len(cells):
        raise ValueError("Measured package sample or cell count differs from canonical cells")
    registry = manifest.get("canonical_registry", {})
    registry_manifest_hash = region_link["registry_manifest_sha256"] if is_region else sha256_file(profile_root / "cell_profiles_manifest.json")
    if (registry.get("manifest_sha256") != registry_manifest_hash
            or registry.get("sample_id") != sample or registry.get("observation_unit") != unit):
        raise ValueError("Measured package canonical profile manifest lineage mismatch")
    table_name = Path(str(registry.get("table_path", ""))).name
    if not is_region and table_name not in {"cell_profiles.csv", "cell_profiles.parquet"}:
        raise ValueError("Measured package has no recognized canonical table lineage")
    expected = registry.get("table_sha256")
    registry_table_hash = region_link["registry_table_sha256"] if is_region else sha256_file(profile_root / table_name)
    if (not expected or expected != profile_manifest.get("files", {}).get(table_name)
            or expected != registry_table_hash):
        raise ValueError("Measured package canonical table lineage mismatch")
    files = manifest.get("files", {})
    required = {"measured_observations.csv", "measured_observations.parquet", "measured_values.npy", "measured_rows.csv"}
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Measured package is missing hashed output artifacts")
    for name, digest in files.items():
        path = _contained(root, name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or sha256_file(path) != digest:
            raise ValueError(f"Measured package artifact hash mismatch: {name}")
    parquet = root / "measured_observations.parquet"
    if pq.ParquetFile(parquet).metadata.num_rows != len(cells):
        raise ValueError("Measured table must contain every canonical cell exactly once")
    obs = pd.read_parquet(parquet)
    rows = pd.read_csv(root / "measured_rows.csv", dtype=str, keep_default_na=False)
    identity = ["sample_id", identity_key]
    if not set(identity) <= set(obs) or not set(identity) <= set(rows):
        raise ValueError("Measured table or row index is missing stable cell identity")
    if (obs[identity_key].isna().any() or obs[identity_key].duplicated().any()
            or not obs[identity].equals(cells[identity]) or not rows[identity].equals(cells[identity])):
        raise ValueError("Measured package row order/identities differ from canonical cells; no implicit reordering")
    if not is_region and "cell_id" in obs and obs.cell_id.astype(str).tolist() != cells.cell_id.astype(str).tolist():
        raise ValueError("Measured package original integer-label identity mismatch")
    if is_region and ("cell_uid" in obs or "cell_id" in obs):
        raise ValueError("Region measured tables must not assert individual-cell identities")
    if is_region and ("area_um2" not in obs or not np.allclose(obs.area_um2.to_numpy(float), cells.area_um2.to_numpy(float), rtol=1e-9, atol=1e-9)):
        raise ValueError("Measured package region areas disagree with registry")
    if not {"x_um", "y_um"} <= set(obs) or not np.allclose(obs[["x_um", "y_um"]].to_numpy(float), cells[["x_um", "y_um"]].to_numpy(float), rtol=0, atol=1e-9):
        raise ValueError("Measured package canonical physical coordinates disagree")
    registration = manifest.get("registration", {})
    try:
        landmarks = float(registration["independent_landmarks"])
        median = float(registration["median_error_um"])
        p95 = float(registration["p95_error_um"])
        acceptance = float(registration["acceptance_p95_um"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Measured package is missing registration acceptance evidence") from None
    if (registration.get("status") != "passed" or registration.get("locked_before_prediction_review") is not True
            or not all(math.isfinite(value) for value in (landmarks, median, p95, acceptance))
            or landmarks < 3 or landmarks != int(landmarks) or not 0 <= median <= p95 <= acceptance or acceptance <= 0
            or registration.get("target_observations_sha256") != expected
            or registration.get("source_observations_sha256") != manifest.get("inputs", {}).get("measured_table_sha256")):
        raise ValueError("Measured package registration gate/lineage is not valid")
    matrix_record = manifest.get("matrix", {})
    if (matrix_record.get("path") != "measured_values.npy" or matrix_record.get("row_index") != "measured_rows.csv"
            or matrix_record.get("dtype") != "float64" or matrix_record.get("sha256") != files["measured_values.npy"]):
        raise ValueError("Measured matrix schema/precision/hash mismatch")
    matrix = np.load(root / "measured_values.npy", mmap_mode="r", allow_pickle=False)
    markers = manifest.get("markers", [])
    columns = matrix_record.get("feature_names", [])
    if (not isinstance(markers, list) or not markers or matrix.dtype != np.float64
            or matrix.shape != (len(cells), len(markers)) or list(matrix.shape) != matrix_record.get("shape")
            or not isinstance(columns, list) or len(columns) != len(markers) or len(set(columns)) != len(columns)
            or not all(isinstance(name, str) and name.startswith("measured__") for name in columns)
            or not set(columns) <= set(obs)):
        raise ValueError("Measured matrix shape/marker schema does not match the observation table")
    marker_names = [record.get("name") for record in markers]
    if (marker_names != matrix_record.get("marker_names") or len(set(marker_names)) != len(markers)
            or any(not isinstance(name, str) or not name for name in marker_names)
            or [record.get("output_column") for record in markers] != columns):
        raise ValueError("Measured marker order/identity mismatch")
    for marker in markers:
        if any(not isinstance(marker.get(field), str) or not marker[field].strip()
               for field in ("units", "measurement_type", "normalization", "column")):
            raise ValueError("Measured markers require declared units and assay-specific semantics")
    for start in range(0, len(cells), tile_size):
        values = matrix[start:start + tile_size]
        if np.isinf(values).any() or not np.array_equal(values, obs.iloc[start:start + tile_size][columns].to_numpy(float), equal_nan=True):
            raise ValueError("Measured matrix and table values/NaNs differ")
    required_match = {"measured_matched", "measured_marker_count", "measured_status", "assay_observation_id", "assay_id"}
    if not required_match <= set(obs) or obs.measured_matched.dtype.kind != "b":
        raise ValueError("Measured match-status schema is missing or ambiguous")
    matched = obs.measured_matched.to_numpy()
    measured_ids = obs.loc[matched, "assay_observation_id"]
    if measured_ids.isna().any() or measured_ids.astype(str).eq("").any() or measured_ids.duplicated().any():
        raise ValueError("Measured source IDs must be one-to-one for matched cells")
    counts = obs[columns].notna().sum(axis=1).to_numpy()
    statuses = np.where(~matched, "unmatched", np.where(counts == 0, "matched_all_markers_missing", "matched"))
    if (not np.array_equal(counts, obs.measured_marker_count.to_numpy())
            or not np.array_equal(statuses, obs.measured_status.to_numpy()) or (counts[~matched] != 0).any()
            or int(matched.sum()) != manifest.get("matched_observation_count")
            or len(cells) - int(matched.sum()) != manifest.get("unmatched_observation_count")):
        raise ValueError("Measured match status/counts disagree with actual values")
    matching = manifest.get("matching", {})
    fraction = float(matched.mean())
    try:
        minimum = float(matching["minimum_matched_fraction"])
        declared = float(matching["matched_fraction"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Measured matching gate is incomplete") from None
    expected_matching = "provided_region_ids" if is_region else "provided_one_to_one_ids"
    if (matching.get("method") != expected_matching or matching.get("independent_of_predicted_markers") is not True
            or matching.get("eligible_observation_count") != len(cells) or not 0 < minimum <= fraction <= 1
            or not math.isclose(fraction, declared, abs_tol=1e-8)
            or not math.isclose(fraction, manifest.get("matched_fraction", -1), abs_tol=1e-8)):
        raise ValueError("Measured package matching/coverage gate failed")
    try:
        maximum_distance = float(matching["maximum_match_distance_um"])
        registered = obs.loc[matched, ["registered_x_um", "registered_y_um"]].to_numpy(float)
        distances = obs.loc[matched, "match_distance_um"].to_numpy(float)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Measured package physical match-distance evidence is missing") from None
    actual_distances = np.linalg.norm(registered - cells.loc[matched, ["x_um", "y_um"]].to_numpy(float), axis=1)
    if (not math.isfinite(maximum_distance) or maximum_distance <= 0 or not np.isfinite(registered).all()
            or not np.isfinite(distances).all() or (actual_distances > maximum_distance).any()
            or not np.allclose(distances, actual_distances, rtol=0, atol=1e-9)
            or obs.loc[~matched, "assay_observation_id"].notna().any()):
        raise ValueError("Measured package physical match-distance/identity gate failed")
    assay_id = manifest.get("assay_id")
    if not isinstance(assay_id, str) or not assay_id or not obs.assay_id.eq(assay_id).all():
        raise ValueError("Measured assay identity mismatch")
    name = "measured_assay_" + hashlib.sha256(assay_id.encode()).hexdigest()[:16]
    obs = obs.drop(columns=columns).copy()
    if any(str(column).startswith("predicted__") for column in obs) or {"spatial_region", "instance_id"} & set(obs):
        raise ValueError("Measured table contains predicted/reserved fields; modalities must remain separate")
    obs.index = pd.Index(cells[identity_key], name=f"{identity_key}_index")
    obs["spatial_region"] = pd.Categorical([spatial_element] * len(cells))
    obs["instance_id"] = instance_ids.astype(label_dtype)
    var = pd.DataFrame({"marker_name": marker_names, "units": [m["units"] for m in markers],
        "measurement_type": [m["measurement_type"] for m in markers],
        "normalization": [m["normalization"] for m in markers],
        "source_column": [m["column"] for m in markers], "measured_feature_name": columns},
        index=pd.Index([f"marker_{index:04d}" for index in range(len(markers))], name="marker_id"))
    table = ad.AnnData(X=da.from_array(matrix, chunks=(tile_size, len(markers)), asarray=False),
                      obs=_safe_obs(obs), var=_safe_obs(var))
    table.obsm["spatial"] = cells[["x_um", "y_um"]].to_numpy(float)
    table.uns["cellphenotyper"] = {"modality": "measured_assay", "assay_id": assay_id,
        "value_semantics": manifest["value_semantics"], "coordinate_system": COORDINATE_SYSTEM,
        "package_manifest_sha256": sha256_file(manifest_path), "package_manifest_json": json.dumps(manifest, sort_keys=True),
        "X_semantics": "independently measured assay values; float64; original units; NaNs preserved; no normalization or imputation"}
    original = {attr: list(getattr(table, attr).keys()) for attr in ("obs", "var", "obsm")}
    sanitize_table(table, inplace=True)
    table.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(
        {attr: dict(zip(names, getattr(table, attr).keys())) for attr, names in original.items()}, sort_keys=True)
    table = TableModel.parse(table, region=spatial_element, region_key="spatial_region", instance_key="instance_id")
    record = {"table": name, "assay_id": assay_id, "observation_unit": unit, "modality": "measured_assay",
        "spatial_element": spatial_element, "identity_key": identity_key,
        "package_manifest_sha256": sha256_file(manifest_path),
        "canonical_registry_manifest_sha256": registry["manifest_sha256"],
        "marker_names": marker_names, "matched_observations": int(matched.sum()), "value_dtype": "float64"}
    if is_region:
        record["region_shape_link"] = region_link
    else:
        record["canonical_profile_manifest_sha256"] = registry["manifest_sha256"]
        record["matched_cells"] = int(matched.sum())
    return name, table, matrix, record


def measured_region_elements(package, link_path, cal, sample_id, source_hashes):
    """Read explicitly registered polygons, bind registry IDs and physical frame."""
    import geopandas as gpd
    from shapely.geometry import shape
    from shapely.affinity import affine_transform
    from spatialdata.models import ShapesModel
    from integrate_measured_assay import read_table, numeric_values, validate_ids, verified_artifact

    link_path = Path(link_path).resolve()
    link = json.loads(link_path.read_text())
    package_manifest = json.loads((Path(package) / "measured_assay_manifest.json").read_text())
    if link.get("schema_version") != "cellphenotyper.measured_region_shapes.v1":
        raise ValueError("Measured region shapes require cellphenotyper.measured_region_shapes.v1")
    if (link.get("assay_id") != package_manifest.get("assay_id") or link.get("sample_id") != sample_id
            or package_manifest.get("sample_id") != sample_id or package_manifest.get("observation_unit") != "spatial_bin"):
        raise ValueError("Measured region shape-link assay/sample/observation identity mismatch")
    frame = link.get("source_coordinates")
    if frame not in {"crop_pixels", "original_pixels", "original_um"} or link.get("target_coordinates") != "original_slide_micrometres":
        raise ValueError("Region shape-link must declare source and target coordinate frames")
    if link.get("coordinate_anchor") != "geometry_centroid":
        raise ValueError("Region shape-link requires coordinate_anchor=geometry_centroid")
    if any(link.get(f"target_{name}_sha256") != source_hashes[name] for name in ("image", "shift")):
        raise ValueError("Region shape-link target image/shift hash mismatch")
    transform_hash = package_manifest.get("registration", {}).get("transform_artifact", {}).get("sha256")
    if not transform_hash or link.get("registration_transform_sha256") != transform_hash:
        raise ValueError("Region shape-link registration transform lineage mismatch")
    shape_record = verified_artifact(link.get("shapes"), link_path.parent)
    registry_record = verified_artifact(link.get("registry_table"), link_path.parent)
    registry_manifest_record = verified_artifact(link.get("registry_manifest"), link_path.parent)
    lineage = package_manifest.get("canonical_registry", {})
    if (registry_record["sha256"] != lineage.get("table_sha256")
            or registry_manifest_record["sha256"] != lineage.get("manifest_sha256")):
        raise ValueError("Region shape-link registry lineage mismatch")
    registry_manifest = json.loads(Path(registry_manifest_record["path"]).read_text())
    regions = read_table(registry_record["path"], max_observations=5_000_000)
    validate_ids(regions, "region_uid")
    original_table_name = Path(str(lineage.get("table_path", ""))).name
    if (registry_manifest.get("schema_version") != "cellphenotyper.region_registry.v1"
            or registry_manifest.get("observation_unit") != "spatial_bin"
            or registry_manifest.get("coordinate_system") != "original_slide_micrometres"
            or registry_manifest.get("sample_id") != sample_id or "sample_id" not in regions
            or set(regions.sample_id) != {sample_id} or "cell_uid" in regions or "cell_id" in regions
            or regions.empty or registry_manifest.get("region_count") != len(regions)
            or registry_manifest.get("files", {}).get(original_table_name) != registry_record["sha256"]):
        raise ValueError("Region registry sample/schema/count/table lineage does not match exported specimen")
    regions[["x_um", "y_um", "area_um2"]] = numeric_values(regions, ["x_um", "y_um", "area_um2"], allow_missing=False)
    if (regions.area_um2 <= 0).any():
        raise ValueError("Region registry areas must be positive")
    for definition in (registry_manifest.get("region_definition", {}), package_manifest.get("region_definition", {})):
        if definition.get("artifact", {}).get("sha256") != shape_record["sha256"]:
            raise ValueError("Registered polygon content differs from the imported region definition")
    geojson = json.loads(Path(shape_record["path"]).read_text())
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Registered region shapes must be a GeoJSON FeatureCollection")
    geometries = {}
    for feature in geojson.get("features", []):
        uid = (feature.get("properties") or {}).get("region_uid")
        if not isinstance(uid, str) or not uid or uid != uid.strip() or uid in geometries:
            raise ValueError("Each registered polygon must have one unique explicit region_uid")
        geometry = shape(feature["geometry"])
        if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid or geometry.is_empty or not np.isfinite(geometry.bounds).all():
            raise ValueError("Registered regions require valid, finite nonempty polygon/multipolygon geometry")
        geometries[uid] = geometry
    if set(geometries) != set(regions.region_uid):
        raise ValueError("Registered polygon region identities differ from the complete registry")
    scale = 1. if frame == "original_um" else cal["mpp"]
    offset = cal["origin_px"] * cal["mpp"] if frame == "crop_pixels" else np.zeros(2)
    lower = cal["origin_px"] * cal["mpp"]
    upper = lower + np.array([cal["width"], cal["height"]]) * cal["mpp"]
    ordered = [geometries[uid] for uid in regions.region_uid]
    for row, geometry in zip(regions.itertuples(index=False), ordered):
        physical = affine_transform(geometry, [scale, 0, 0, scale, *offset])
        centre = np.array([physical.centroid.x, physical.centroid.y])
        if (not np.allclose(centre, [row.x_um, row.y_um], rtol=0, atol=1e-6)
                or not math.isclose(physical.area, row.area_um2, rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("Registered region centroid/physical area disagrees with explicit registry and coordinate frame")
        bounds = np.array(physical.bounds)
        if (bounds[:2] < lower - 1e-6).any() or (bounds[2:] > upper + 1e-6).any():
            raise ValueError("Registered region lies outside the exported image crop; no implicit clipping/subsetting")
    indices = np.arange(1, len(regions) + 1, dtype=np.int64)
    shape_name = "measured_regions_" + hashlib.sha256(link["assay_id"].encode()).hexdigest()[:16]
    geometry_frame = gpd.GeoDataFrame({"region_uid": regions.region_uid.to_numpy(),
        "sample_id": regions.sample_id.to_numpy(), "area_um2": regions.area_um2.to_numpy()},
        geometry=ordered, index=pd.Index(indices, name="region_instance_id"))
    shapes = ShapesModel.parse(geometry_frame, transformations={COORDINATE_SYSTEM: physical_transform(cal, frame)})
    record = {"shape_name": shape_name, "shape_link_path": str(link_path), "shape_link_sha256": sha256_file(link_path),
        "registry_manifest_sha256": registry_manifest_record["sha256"], "registry_table_sha256": registry_record["sha256"],
        "shapes_sha256": shape_record["sha256"], "source_coordinates": frame,
        "target_coordinates": "original_slide_micrometres", "coordinate_anchor": "geometry_centroid",
        "target_image_sha256": source_hashes["image"], "target_shift_sha256": source_hashes["shift"],
        "registration_transform_sha256": transform_hash, "region_count": len(regions),
        "identity_mapping": "polygon region_uid matched exactly to registry order; region instance IDs are not cell labels"}
    return shapes, regions, registry_manifest, indices, record


def hierarchy_elements(profile_dir, labels, directory, cal, source_hashes, tile_size):
    """Export discovered regions and a many-to-many cell/region relation.

    Region instances retain their own raster IDs. Overlap rows deliberately do
    not annotate the cell or region element as though they were unique cells:
    they are a separate relation with explicit foreign-key metadata.
    """
    import anndata as ad
    import dask.array as da
    from scipy import sparse
    from spatialdata import sanitize_table
    from spatialdata.models import Labels2DModel, TableModel
    from link_cell_tissue_hierarchy import load_verified_hierarchy, load_verified_links

    info = load_verified_hierarchy(profile_dir, labels, directory, tile_size=tile_size)
    summary = info["summary"]
    for key, source in (("image", "image"), ("shift_json", "shift"), ("resolution_json", "resolution_json")):
        if (not source_hashes.get(source)
                or summary.get("inputs", {}).get(key, {}).get("sha256") != source_hashes[source]):
            raise ValueError(f"Exported hierarchy {key} differs from the actual export source SHA256")
    overlap = load_verified_links(profile_dir, info)
    if not isinstance(overlap, pd.DataFrame):
        raise ValueError("Verified hierarchy links must return their exact overlap table")
    transform = {COORDINATE_SYSTEM: physical_transform(cal)}
    export_rasters = {HIERARCHY_LABELS[key]: path for key, path in info["raster_paths"].items()}
    compartment_labels = {"nucleus": CELL_LABELS}
    if "perinuclear_ring" in info["manifest"]["hierarchy_links"]["compartments"]:
        # load_verified_links verifies this relocated copy against both the
        # original compartment lineage and the exact joint overlap counts.
        ring_path = _contained(Path(profile_dir).resolve(), "hierarchy_source/labels_perinuclear_ring.tif")
        export_rasters[CELL_RING_LABELS] = ring_path
        compartment_labels["perinuclear_ring"] = CELL_RING_LABELS
    info["export_raster_paths"] = export_rasters
    label_elements = {name: Labels2DModel.parse(lazy_raster(path, tile_size),
        dims=("y", "x"), transformations=transform) for name, path in export_rasters.items()}
    regions, region_manifest = info["regions"], info["region_manifest"]
    if {"spatial_region", "instance_id"} & set(regions):
        raise ValueError("Hierarchy region profile contains reserved SpatialData columns")
    obs = regions.copy()
    obs.index = pd.Index(regions.region_uid.astype(str), name="region_uid_index")
    obs["spatial_region"] = pd.Categorical([HIERARCHY_LABELS["region"]] * len(regions))
    obs["instance_id"] = regions.region_id.astype(np.int64).to_numpy()
    region_table = ad.AnnData(X=sparse.csr_matrix((len(obs), 0), dtype=np.float32), obs=_safe_obs(obs))
    region_table.obsm["spatial"] = regions[["x_um", "y_um"]].to_numpy(float)
    matrices = {}
    for name, record in region_manifest.get("feature_blocks", {}).items():
        if name.lower() == "spatial" or "/" in name:
            raise ValueError(f"Hierarchy feature block name is reserved/unsafe: {name}")
        matrix = np.load(_contained(info["region_root"], record["path"]), mmap_mode="r", allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[0] != len(regions) or matrix.dtype.kind not in "fiu":
            raise ValueError("Hierarchy region feature matrix is not numeric and region-row aligned")
        feature_names = record.get("feature_names")
        if (not isinstance(feature_names, list) or len(feature_names) != matrix.shape[1]
                or any(not isinstance(value, str) or not value for value in feature_names)
                or len(set(feature_names)) != len(feature_names)):
            raise ValueError("Hierarchy region feature columns lack an exact unique feature-name schema")
        matrices[name] = matrix
        region_table.obsm[name] = da.from_array(matrix, chunks=(tile_size, max(1, matrix.shape[1])), asarray=False)
    region_table.uns["cellphenotyper"] = {"observation_unit": "tissue_region",
        "hierarchy_id": summary["hierarchy_id"], "coordinate_system": COORDINATE_SYSTEM,
        "target_labels": HIERARCHY_LABELS["region"],
        "assignment_status": "accepted_regions" if len(regions) else "no_accepted_regions; parent tissue and unresolved status retained",
        "region_profile_manifest_json": json.dumps(region_manifest, sort_keys=True),
        "feature_semantics": "Actual area-weighted source-grid feature means; not constituent distributions or cell embeddings",
        "X_semantics": "empty; named obsm blocks retain original region features; no normalization or imputation"}
    original = {attr: list(getattr(region_table, attr).keys()) for attr in ("obs", "obsm")}
    sanitize_table(region_table, inplace=True)
    mapping = {attr: dict(zip(names, getattr(region_table, attr).keys())) for attr, names in original.items()}
    region_table.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(mapping, sort_keys=True)
    # An empty table has no represented instances, so the official annotation
    # model requires an empty region list. Preserve the intended label element
    # separately; do not invent a dummy region row to satisfy a viewer.
    region_table = TableModel.parse(region_table, region=HIERARCHY_LABELS["region"] if len(regions) else [],
                                    region_key="spatial_region", instance_key="instance_id")
    relation_obs = overlap.copy()
    relation_obs.index = pd.Index([f"overlap_{i:012d}" for i in range(len(overlap))], name="overlap_row_id")
    relation = ad.AnnData(X=sparse.csr_matrix((len(overlap), 0), dtype=np.float32), obs=_safe_obs(relation_obs))
    relation.uns["cellphenotyper"] = {"observation_unit": "cell_compartment_hierarchy_overlap",
        "hierarchy_id": summary["hierarchy_id"], "modality": "exact_geometric_relation",
        "foreign_keys_json": json.dumps({"cell_uid": {"table": "cells", "key": "cell_uid"},
            "cell_id": {"labels": CELL_LABELS, "key": "raster_value"},
            "compartment": {"geometry_labels": compartment_labels, "instance_key": "cell_id"},
            "parent_domain_id": {"labels": HIERARCHY_LABELS["parent"], "key": "raster_value"},
            "subdomain_id": {"labels": HIERARCHY_LABELS["subdomain"], "key": "raster_value"},
            "region_id": {"labels": HIERARCHY_LABELS["region"], "key": "raster_value"},
            "region_uid": {"table": "hierarchy_region_profiles", "key": "region_uid"}}, sort_keys=True),
        "semantics": "Joint pixel partition per canonical cell and compartment; repeated cell IDs are intentional many-to-many relations, not additional cells. Region/subdomain ID 0 and empty region_uid are unresolved/background, never a fabricated region. Status and parent-uncertainty codes retain producer meanings.",
        "parent_uncertainty_code_semantics": "Provided parent raster codes 0..254 retain their exact upstream meaning; synthetic table sentinel 255 means the parent uncertainty raster was not supplied, not confidence or a raster label.",
        "parent_uncertainty_available": info["manifest"]["hierarchy_links"].get("parent_uncertainty_available", "parent_uncertainty" in info["raster_paths"]),
        "hierarchy_links_json": json.dumps(info["manifest"]["hierarchy_links"], sort_keys=True)}
    original_obs = list(relation.obs.columns)
    sanitize_table(relation, inplace=True)
    relation.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(
        {"obs": dict(zip(original_obs, relation.obs.columns))}, sort_keys=True)
    # No region/instance annotation: there can be several relation rows for a
    # single cell and for a single region. The explicit keys above are required.
    relation = TableModel.parse(relation)
    record = {"hierarchy_id": summary["hierarchy_id"], "status": "verified_exact_native_maps_and_overlap_links",
        "hierarchy_summary_sha256": sha256_file(Path(directory) / "hierarchy_summary.json"),
        "region_profile_manifest_sha256": sha256_file(info["region_root"] / "region_profiles_manifest.json"),
        "label_elements": {key: HIERARCHY_LABELS[key] for key in info["raster_paths"]}, "region_table": "hierarchy_region_profiles",
        "compartment_label_elements": compartment_labels,
        "overlap_table": "cell_hierarchy_overlaps", "region_count": len(regions), "overlap_rows": len(overlap),
        "status_codes": summary["status_codes"], "parent_uncertainty_semantics": summary.get("parent_uncertainty_semantics", "not_provided"),
        "source_inputs": summary["inputs"], "source_output_hashes": summary["outputs"],
        "links": info["manifest"]["hierarchy_links"]}
    return label_elements, {"hierarchy_region_profiles": region_table, "cell_hierarchy_overlaps": relation}, info, matrices, mapping, record


def measured_profile_registry(package_header, profile_root, cells, manifest):
    """Resolve an assay's exact registry through one verified additive link step.

    An already imported assay remains bound to its original registry, not to a
    newly asserted hash. The linker preserves that registry byte-for-byte. Only
    a derivation adding hierarchy fields can use it, after every original cell
    column and feature/graph definition has been shown unchanged.
    """
    current_hash = sha256_file(profile_root / "cell_profiles_manifest.json")
    expected_hash = package_header.get("canonical_registry", {}).get("manifest_sha256")
    if expected_hash == current_hash:
        return profile_root, manifest, {"status": "current_profile_registry",
            "registry_manifest_sha256": current_hash, "exported_profile_manifest_sha256": current_hash}
    links = manifest.get("hierarchy_links", {})
    if not links:
        return profile_root, manifest, {"status": "current_profile_registry"}
    source_root = _contained(profile_root, links.get("source_root", "hierarchy_source"))
    files = links.get("source_files", {})
    if not isinstance(files, dict) or "cell_profiles_manifest.json" not in files:
        raise ValueError("Additive hierarchy lineage lacks the exact original measured registry")
    for name, digest in files.items():
        if not isinstance(digest, str) or sha256_file(_contained(source_root, name)) != digest:
            raise ValueError("Additive hierarchy original registry hash mismatch")
    source_manifest_path = source_root / "cell_profiles_manifest.json"
    source_hash = sha256_file(source_manifest_path)
    if expected_hash != source_hash or source_hash != links.get("parent_profile_manifest_sha256"):
        raise ValueError("Measured package is bound to neither the current nor the exact additive-source registry")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("hierarchy_links"):
        raise ValueError("Nested additive hierarchy measured-registry lineage is unsupported; import against the final registry")
    for key in ("sample_id", "observation_unit", "cell_count", "source_mpp", "crop_origin_um", "crop_size_px",
                "inputs", "feature_blocks", "spatial_graphs"):
        if source_manifest.get(key) != manifest.get(key):
            raise ValueError(f"Additive hierarchy derivation changed an original registry field: {key}")
    table_path = source_root / "cell_profiles.parquet"
    if table_path.is_file():
        base_cells = pd.read_parquet(table_path)
    else:
        table_path = source_root / "cell_profiles.csv"
        base_cells = pd.read_csv(table_path, dtype={name: str for name in
            ("sample_id", "cell_id", "cell_uid", "stardist_id", "cellvitpp_id", "hovernet_id", "consensus_id")})
    if (files.get(table_path.name) != sha256_file(table_path)
            or source_manifest.get("files", {}).get(table_path.name) != sha256_file(table_path)
            or len(base_cells) != len(cells) or not set(base_cells) <= set(cells)):
        raise ValueError("Additive hierarchy source canonical table provenance or schema mismatch")
    for name in base_cells:
        try:
            pd.testing.assert_series_equal(base_cells[name].reset_index(drop=True), cells[name].reset_index(drop=True),
                check_names=False, check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise ValueError(f"Additive hierarchy changed an original canonical cell value/order: {name}") from exc
    if any(not name.startswith("hierarchy_") for name in set(cells) - set(base_cells)):
        raise ValueError("Measured registry derivation added fields outside the declared hierarchy-only operation")
    return source_root, source_manifest, {"status": "verified_additive_hierarchy_source_registry",
        "registry_manifest_sha256": source_hash, "exported_profile_manifest_sha256": current_hash,
        "source_table_sha256": sha256_file(table_path),
        "semantics": "Measured package remains bound to the byte-identical pre-link registry; all original cell values, feature definitions and graphs unchanged; no package rebinding"}


def _verify_hierarchy_roundtrip(reloaded, store, tables, info, matrices, mapping, tile_size, cal):
    """Check exact native maps, transforms, region features and relation rows."""
    import anndata as ad
    from spatialdata.transformations import get_transformation

    expected_transform = physical_transform(cal).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    for name, path in info["export_raster_paths"].items():
        actual = reloaded.labels[name]
        transform = get_transformation(actual, to_coordinate_system=COORDINATE_SYSTEM)
        if not np.array_equal(transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), expected_transform):
            raise RuntimeError("SpatialData round trip changed hierarchy raster transform")
        with RasterReader(path) as reader:
            if tuple(actual.shape) != (reader.height, reader.width) or actual.dtype != reader.dtype:
                raise RuntimeError("SpatialData round trip changed hierarchy native shape/dtype")
            for y in range(0, reader.height, tile_size):
                for x in range(0, reader.width, tile_size):
                    x1, y1 = min(x + tile_size, reader.width), min(y + tile_size, reader.height)
                    if not np.array_equal(actual.data[y:y1, x:x1].compute(), reader.window(x, y, x1, y1)):
                        raise RuntimeError("SpatialData round trip changed hierarchy native label pixels")
    for name, expected in tables.items():
        actual = ad.experimental.read_lazy(store["tables"][name])
        # Dataset2D's lazy columns are not order-preserving in AnnData 0.12.
        # Verify the written DataFrame order and lazy key set independently.
        written_order = list(store["tables"][name]["obs"].attrs["column-order"])
        if (actual.obs_names.tolist() != expected.obs_names.tolist()
                or set(actual.obs.columns) != set(expected.obs.columns) or written_order != list(expected.obs.columns)):
            raise RuntimeError("SpatialData round trip changed hierarchy table row identity/schema")
        for column in expected.obs:
            one = np.asarray(actual.obs[column].values)
            two = np.asarray(expected.obs[column].values)
            same = np.array_equal(one, two, equal_nan=True) if two.dtype.kind in "fiu" else np.array_equal(one, two)
            if not same:
                raise RuntimeError(f"SpatialData round trip changed hierarchy table values: {name}/{column}")
        # Values may be lists or arrays after Zarr. Compare the public linkage
        # fields individually, not with an ambiguous array mapping comparison.
        for field in ("region_key", "instance_key", "region"):
            actual_value = actual.uns.get("spatialdata_attrs", {}).get(field)
            actual_value = actual_value.compute() if hasattr(actual_value, "compute") else actual_value
            if not np.array_equal(np.asarray(actual_value), np.asarray(expected.uns.get("spatialdata_attrs", {}).get(field))):
                raise RuntimeError("SpatialData round trip changed hierarchy table-to-label linkage")
        if name == "hierarchy_region_profiles":
            for source_name, matrix in {"spatial": info["regions"][["x_um", "y_um"]].to_numpy(float), **matrices}.items():
                values = actual.obsm[mapping["obsm"][source_name]]
                if values.shape != matrix.shape or values.dtype != matrix.dtype:
                    raise RuntimeError("SpatialData round trip changed hierarchy region feature precision/shape")
                for start in range(0, len(matrix), tile_size):
                    block = values[start:start + tile_size]
                    block = block.compute() if hasattr(block, "compute") else np.asarray(block)
                    if not np.array_equal(block, matrix[start:start + tile_size], equal_nan=True):
                        raise RuntimeError("SpatialData round trip changed hierarchy region features/NaNs")


def _reference_profile_sources(profile_dir, unit):
    """Small registry/feature sources, not another copy of images or profiles."""
    root = Path(profile_dir).resolve()
    stem = "cell_profiles" if unit == "cell" else "region_profiles"
    header = root / f"{stem}_manifest.json"
    manifest = json.loads(header.read_text())
    paths = {header}
    for name in (f"{stem}.csv", f"{stem}.parquet", "feature_rows.csv"):
        if (root / name).is_file():
            paths.add(root / name)
    for records in (manifest.get("feature_blocks", {}), manifest.get("spatial_graphs", {})):
        for record in records.values():
            paths.add(_contained(root, record["path"]))
    return {path: sha256_file(path) for path in paths}


def attach_reference_mapping(table, profile_dir, receipt_path, unit, table_name):
    """Attach only verified interpretation columns; never alter discovery.

    The portable receipt and exact frozen atlas manifest remain available in
    the exported store. Absolute source paths are used only for in-flight
    integrity checks and are not required for portable readback.
    """
    from reference_mapping_io import load_reference_mapping

    before = _reference_profile_sources(profile_dir, unit)
    frame, record, artifacts = load_reference_mapping(receipt_path, profile_dir, expected_unit=unit)
    if set(artifacts) != {"receipt", "assignments", "atlas_manifest"}:
        raise ValueError("Reference mapping loader must identify its exact three portable artifacts")
    if _reference_profile_sources(profile_dir, unit) != before:
        raise ValueError("Reference mapping source profile changed while validating")
    receipt_bytes = Path(artifacts["receipt"]).read_bytes()
    if json.loads(receipt_bytes) != record:
        raise ValueError("Reference mapping receipt changed while validating")
    atlas_bytes = Path(artifacts["atlas_manifest"]).read_bytes()
    # Parse as well as preserve exact bytes; this is metadata, not executable
    # content or a request to follow archived file paths.
    json.loads(atlas_bytes)
    metadata_artifacts = {}
    for name, supplied in artifacts.items():
        path = Path(supplied).resolve()
        actual = sha256_file(path)
        bound = {"assignments": record["assignments"]["sha256"],
                 "atlas_manifest": record["atlas"]["sha256"]}.get(name)
        if bound is not None and actual != bound:
            raise ValueError("Reference mapping payload changed after validation")
        if name in {"receipt", "atlas_manifest"}:
            loaded_bytes = receipt_bytes if name == "receipt" else atlas_bytes
            if hashlib.sha256(loaded_bytes).hexdigest() != actual:
                raise ValueError("Reference mapping metadata changed while reading")
        before[path] = actual
        metadata_artifacts[name] = {"filename": path.name, "sha256": actual, "size_bytes": path.stat().st_size}
    uid = "cell_uid" if unit == "cell" else "region_uid"
    instance = "cell_id" if unit == "cell" else "region_id"
    if not {uid, instance, "sample_id", *REFERENCE_FIELDS} <= set(frame):
        raise ValueError("Reference mapping lacks exact observation identities or interpretation columns")
    if {name for name in frame if name.startswith("reference_")} != set(REFERENCE_FIELDS):
        raise ValueError("Reference mapping has an unsupported interpretation schema")
    field_mapping = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    for source_name in (uid, instance, "sample_id"):
        exported_name = field_mapping["obs"].get(source_name)
        if (exported_name is None or len(frame) != table.n_obs
                or frame[source_name].astype(str).tolist() != table.obs[exported_name].astype(str).tolist()):
            raise ValueError("Reference mapping IDs/order differ from the exact exported observation table")
    if frame[uid].astype(str).tolist() != table.obs_names.tolist():
        raise ValueError("Reference mapping row order differs from the spatial table index")
    existing = {str(name).casefold() for name in table.obs} | {str(name).casefold() for name in field_mapping["obs"]}
    if existing & set(REFERENCE_FIELDS):
        raise ValueError("Reference mapping interpretation columns collide with existing profile fields")
    interpretations = frame.loc[:, list(REFERENCE_FIELDS)].copy()
    interpretations.index = table.obs.index
    safe = _safe_obs(interpretations)
    for name in REFERENCE_FIELDS:
        table.obs[name] = safe[name]
        field_mapping["obs"][name] = name
    table.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(field_mapping, sort_keys=True)
    portable = {"status": "verified_exact_profile_and_reference_mapping", "observation_unit": unit,
        "table": table_name, "columns": list(REFERENCE_FIELDS), "artifacts": metadata_artifacts,
        "receipt": record, "receipt_json": receipt_bytes.decode("utf-8"),
        "atlas_manifest_json": atlas_bytes.decode("utf-8"),
        "interpretation": "Separate frozen-reference interpretation; original discovery labels and measurements are unchanged; distances are not calibrated probabilities"}
    table.uns["cellphenotyper"]["reference_mapping_json"] = json.dumps(portable, sort_keys=True, allow_nan=False)
    return portable, {"sources": before, "table": table_name, "frame": safe,
                      "metadata_json": table.uns["cellphenotyper"]["reference_mapping_json"]}


def _verify_reference_mapping_sources(checks):
    for check in checks.values():
        for path, expected in check["sources"].items():
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Reference mapping or bound profile source changed during export: {path.name}")


def _verify_reference_mapping_roundtrip(store, reloaded, records, checks):
    import anndata as ad

    actual_records = reloaded.attrs.get("cellphenotyper", {}).get("reference_mappings")
    if json.dumps(actual_records, sort_keys=True, allow_nan=False) != json.dumps(records, sort_keys=True, allow_nan=False):
        raise RuntimeError("SpatialData round trip changed portable reference mapping provenance")
    for check in checks.values():
        table = ad.experimental.read_lazy(store["tables"][check["table"]])
        expected = check["frame"]
        if table.obs_names.tolist() != expected.index.tolist():
            raise RuntimeError("SpatialData round trip changed reference-mapped observation identities")
        if table.uns["cellphenotyper"].get("reference_mapping_json") != check["metadata_json"]:
            raise RuntimeError("SpatialData round trip changed reference receipt or exact atlas metadata")
        for name in REFERENCE_FIELDS:
            actual = np.asarray(table.obs[name].values)
            original = expected[name].to_numpy()
            same = np.array_equal(actual, original, equal_nan=True) if original.dtype.kind in "fiu" else np.array_equal(actual, original)
            if not same or (original.dtype.kind in "fiu" and actual.dtype != original.dtype):
                raise RuntimeError(f"SpatialData round trip changed reference interpretation values/precision: {name}")


def attach_cohort_niches(table, profile_dir, bundle, sample_id):
    """Add source-bound exploratory cohort labels without replacing discovery.

    The shared reader uses no fitting runtime. Its full portable record is kept
    unchanged, including the distinction between engineering verification and
    biological validation; no reference interpretation is inferred here.
    """
    from cohort_niche_io import COHORT_COLUMNS, load_cohort_bundle, verify_cohort_sources

    frame, record = load_cohort_bundle(bundle, profile_dir=profile_dir, sample_id=sample_id)
    verify_cohort_sources(bundle, profile_dir, record)
    identities = ("sample_id", "cell_id", "cell_uid")
    if (frame.columns.duplicated().any()
            or set(frame) != {*identities, *COHORT_COLUMNS}):
        raise ValueError("Cohort niches require exact identity and interpretation columns")
    field_mapping = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    for name in identities:
        exported_name = field_mapping["obs"].get(name)
        if (exported_name is None or len(frame) != table.n_obs
                or frame[name].tolist() != table.obs[exported_name].astype(str).tolist()):
            raise ValueError("Cohort niche IDs/order differ from the exact exported cell table")
    if frame.cell_uid.tolist() != table.obs_names.tolist():
        raise ValueError("Cohort niche row order differs from the spatial table index")
    existing = {str(name).casefold() for name in table.obs} | {str(name).casefold() for name in field_mapping["obs"]}
    if existing & {name.casefold() for name in COHORT_COLUMNS}:
        raise ValueError("Cohort niche columns collide with existing profile fields")
    if "cohort_niches_json" in table.uns["cellphenotyper"]:
        raise ValueError("Cohort niche metadata already exists; attachment cannot overwrite it")
    metadata_json = json.dumps(record, sort_keys=True, allow_nan=False)
    interpretations = frame.loc[:, COHORT_COLUMNS].copy()
    interpretations.index = table.obs.index
    safe = _safe_obs(interpretations)
    for name in COHORT_COLUMNS:
        table.obs[name] = safe[name]
        field_mapping["obs"][name] = name
    mapping_json = json.dumps(field_mapping, sort_keys=True)
    table.uns["cellphenotyper"]["field_name_mapping_json"] = mapping_json
    table.uns["cellphenotyper"]["cohort_niches_json"] = metadata_json
    return record, {"bundle": bundle, "profile_dir": profile_dir, "record": record,
                    "frame": safe, "metadata_json": metadata_json, "field_mapping_json": mapping_json}


def _verify_cohort_niche_sources(check):
    if check is not None:
        from cohort_niche_io import verify_cohort_sources
        verify_cohort_sources(check["bundle"], check["profile_dir"], check["record"])


def _verify_cohort_niche_roundtrip(store, reloaded, check):
    import anndata as ad
    from cohort_niche_io import COHORT_COLUMNS

    actual_record = reloaded.attrs.get("cellphenotyper", {}).get("cohort_niches")
    if json.dumps(actual_record, sort_keys=True, allow_nan=False) != check["metadata_json"]:
        raise RuntimeError("SpatialData round trip changed portable cohort niche provenance")
    table_store = store["tables"]["cells"]
    table = ad.experimental.read_lazy(table_store)
    if table.obs_names.tolist() != check["frame"].index.tolist():
        raise RuntimeError("SpatialData round trip changed cohort niche cell identities")
    metadata = table.uns["cellphenotyper"]
    if (metadata.get("cohort_niches_json") != check["metadata_json"]
            or metadata.get("field_name_mapping_json") != check["field_mapping_json"]):
        raise RuntimeError("SpatialData round trip changed cohort niche metadata or field mapping")
    for name in COHORT_COLUMNS:
        # Read only the five observation columns, not cell feature matrices.
        # read_elem preserves nullable integer masks and categoricals; a plain
        # NumPy equality test can lose nullable IDs or mishandle pd.NA.
        values = ad.io.read_elem(table_store["obs"][name])
        actual = pd.Series(values, index=check["frame"].index, name=name)
        try:
            pd.testing.assert_series_equal(actual, check["frame"][name],
                                           check_exact=True, check_dtype=True, check_categorical=True)
        except AssertionError as error:
            raise RuntimeError(f"SpatialData round trip changed cohort niche values/precision: {name}") from error


def _neighborhood_feature_block(reader, begin, end, columns):
    """A bounded task, shared by local threaded Dask export and readback."""
    return reader.read(slice(begin, end), columns)


def attach_neighborhood_features(table, profile_dir, manifest, tile_size):
    """Export derived logical feature groups separately from original obsm/obs."""
    import dask.array as da
    from dask import delayed
    from neighborhood_feature_io import FeatureColumns

    reader = FeatureColumns(profile_dir, manifest=manifest)
    if reader.store_record is None:
        raise ValueError("Array-backed neighbourhood export requires a declared feature store")
    if reader.cell_count != table.n_obs:
        raise ValueError("Neighbourhood feature store does not match exported cell count")
    mapping = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    keys = set(table.obsm) | set(mapping["obsm"])
    existing = {str(key).casefold() for key in keys}
    groups = {}
    for number, (name, group) in enumerate(reader.groups.items()):
        target = f"neighborhood_group_{number:04d}"
        logical_key = "neighborhood_features/" + name
        if target.casefold() in existing or logical_key.casefold() in existing:
            raise ValueError("Neighbourhood group export name collides with original feature blocks")
        columns = group["columns"]
        rows = []
        # Each task returns at most tile_size x 256 float64 values. Explicit
        # delayed keys avoid tokenizing/hashing large mmap arrays in the reader.
        token = reader.manifest["neighborhood_feature_store"]["sha256"]
        for begin in range(0, reader.cell_count, tile_size):
            end = min(reader.cell_count, begin + tile_size)
            pieces = []
            for start in range(0, len(columns), 256):
                axis = columns[start:start + 256]
                value = delayed(_neighborhood_feature_block, pure=False)(reader, begin, end, axis,
                    dask_key_name=f"neighborhood-{token}-{number}-{begin}-{start}")
                pieces.append(da.from_delayed(value, shape=(end - begin, len(axis)), dtype=np.float64))
            rows.append(da.concatenate(pieces, axis=1))
        table.obsm[target] = (da.concatenate(rows, axis=0) if rows else
                              da.from_array(np.empty((0, len(columns)), dtype=np.float64)))
        mapping["obsm"][logical_key] = target
        existing.update((target.casefold(), logical_key.casefold()))
        groups[name] = {"obsm": target, "logical_columns": columns,
            "shape": [reader.cell_count, len(columns)], "export_dtype": "float64",
            "source_segments": group["segments"],
            "aggregation_definition": reader.group_definitions.get(name),
            "aggregation_definition_status": "source_declared" if name in reader.group_definitions else "unavailable"}
    record = {"status": "verified_array_backed_neighborhood_features", "table": "cells",
        "source_store": reader.store_record, "source_files": reader.source_files, "groups": groups,
        "semantics": "Separate derived neighbourhood groups; canonical scalar observations and original feature blocks remain unchanged; NaNs are not imputed; float32 source values are exactly promoted to float64."}
    metadata_json = json.dumps(record, sort_keys=True, allow_nan=False)
    if "neighborhood_features_json" in table.uns["cellphenotyper"]:
        raise ValueError("Neighbourhood feature metadata already exists")
    table.uns["cellphenotyper"]["neighborhood_features_json"] = metadata_json
    table.uns["cellphenotyper"]["field_name_mapping_json"] = json.dumps(mapping, sort_keys=True)
    return record, {"reader": reader, "record": record, "metadata_json": metadata_json,
                    "cell_uids": table.obs_names.tolist(), "mapping": mapping["obsm"]}


def _verify_neighborhood_feature_roundtrip(store, reloaded, check, tile_size):
    import anndata as ad

    record = reloaded.attrs.get("cellphenotyper", {}).get("neighborhood_features")
    if json.dumps(record, sort_keys=True, allow_nan=False) != check["metadata_json"]:
        raise RuntimeError("SpatialData round trip changed neighbourhood feature provenance")
    table = ad.experimental.read_lazy(store["tables"]["cells"])
    if table.obs_names.tolist() != check["cell_uids"]:
        raise RuntimeError("SpatialData round trip changed neighbourhood feature cell order")
    if table.uns["cellphenotyper"].get("neighborhood_features_json") != check["metadata_json"]:
        raise RuntimeError("SpatialData round trip changed neighbourhood feature axis/aggregation metadata")
    mapping = json.loads(table.uns["cellphenotyper"]["field_name_mapping_json"])
    if mapping["obsm"] != check["mapping"]:
        raise RuntimeError("SpatialData round trip changed neighbourhood feature name mapping")
    for name, record in check["record"]["groups"].items():
        values = table.obsm[record["obsm"]]
        if values.dtype != np.dtype("float64") or list(values.shape) != record["shape"]:
            raise RuntimeError("SpatialData round trip changed neighbourhood feature shape/precision")
        for begin in range(0, values.shape[0], tile_size):
            end = min(begin + tile_size, values.shape[0])
            for start in range(0, values.shape[1], 256):
                columns = record["logical_columns"][start:start + 256]
                expected = check["reader"].read(slice(begin, end), columns)
                actual = values[begin:end, start:start + len(columns)]
                actual = actual.compute() if hasattr(actual, "compute") else np.asarray(actual)
                if not np.array_equal(actual, expected, equal_nan=True):
                    raise RuntimeError(f"SpatialData round trip changed neighbourhood feature values/NaNs: {name}")


def export_spatialdata(profile_dir, image, labels, shift, outdir, *, tissue_geojson,
                       tissue_coordinates, resolution_json=None, tile_size=1024, workers=2,
                       pyramid_levels=0, measured_assay=(), measured_region_shapes=(), expected_sample_id=None,
                       hierarchy_dir=None, cell_reference_mapping=None, region_reference_mapping=None,
                       cohort_niches=None):
    import dask
    import anndata as ad
    import spatialdata as sd
    import zarr
    from spatialdata.models import Image2DModel, Labels2DModel

    outdir = Path(outdir).resolve()
    if outdir.exists():
        raise FileExistsError("SpatialData exports require a new output directory")
    if tile_size < 16 or workers < 1 or pyramid_levels < 0:
        raise ValueError("tile_size>=16, workers>=1 and pyramid_levels>=0 are required")
    if region_reference_mapping is not None and hierarchy_dir is None:
        raise ValueError("Region reference mapping requires the exact discovered --hierarchy-dir")
    cal = calibration(shift, resolution_json)
    root, cells, manifest, instance_ids = load_cell_profile(profile_dir, cal)
    if manifest.get("hierarchy_links") and hierarchy_dir is None:
        raise ValueError("Linked cell profiles require --hierarchy-dir to preserve tissue hierarchy and explicit overlap relations")
    if expected_sample_id is not None and (manifest.get("sample_id") != str(expected_sample_id)
                                           or set(cells.sample_id.astype(str)) != {str(expected_sample_id)}):
        raise ValueError("Canonical profile manifest/table sample_id differs from the requested sample-id")
    source_hashes = {"image": sha256_file(image), "labels": sha256_file(labels), "shift": sha256_file(shift),
        "tissue_geojson": sha256_file(tissue_geojson), "profile_manifest": sha256_file(root / "cell_profiles_manifest.json")}
    if resolution_json:
        source_hashes["resolution_json"] = sha256_file(resolution_json)
    bound_rasters, unbound_rasters = [], []
    for name in ("image", "labels"):
        expected = manifest.get("inputs", {}).get(f"{name}_sha256")
        if expected is None:
            unbound_rasters.append(name)
        elif expected != source_hashes[name]:
            raise ValueError(f"Canonical profile {name} raster hash mismatch; IDs/shape alone are insufficient lineage")
        else:
            bound_rasters.append(name)
    raster_lineage = {"status": "verified_image_and_labels" if not unbound_rasters else "legacy_or_partial_raster_lineage",
        "verified_profile_bound_rasters": bound_rasters, "missing_profile_raster_hashes": unbound_rasters,
        "limitation": "None" if not unbound_rasters else "Missing source raster hashes cannot establish that legacy profile features came from these exact pixels; ID/coordinate checks are weaker evidence."}
    measured_tables, measured_matrices, measured_records, region_shapes = {}, {}, [], {}
    region_links = {}
    for link_path in measured_region_shapes or ():
        link = json.loads(Path(link_path).read_text())
        assay_id = link.get("assay_id")
        if not isinstance(assay_id, str) or not assay_id:
            raise ValueError("Each measured region shape-link requires an explicit assay_id")
        if assay_id in region_links:
            raise ValueError("Duplicate measured region shape-link for the same assay_id")
        region_links[assay_id] = link_path
    used_links = set()
    with RasterReader(image) as reader:
        if (reader.width, reader.height) != (cal["width"], cal["height"]):
            raise ValueError("Original crop image and shift dimensions differ")
        if min(reader.height, reader.width) // (2 ** pyramid_levels) < 1:
            raise ValueError("Requested pyramid has more levels than the image permits")
        channels = reader.reader.shape[2] if len(reader.reader.shape) == 3 else 1
    label_dtype = verify_label_identity(labels, instance_ids, cal, tile_size)
    native_labels, native_record, native_info = native_support_elements(root, manifest, cal, source_hashes, tile_size)
    hierarchy_labels, hierarchy_tables, hierarchy_record = {}, {}, None
    if hierarchy_dir is not None:
        hierarchy_labels, hierarchy_tables, hierarchy_info, hierarchy_matrices, hierarchy_mapping, hierarchy_record = hierarchy_elements(
            root, labels, hierarchy_dir, cal, source_hashes, tile_size)
    for package in measured_assay or ():
        package_header = json.loads((Path(package) / "measured_assay_manifest.json").read_text())
        if package_header.get("observation_unit") == "spatial_bin":
            assay_id = package_header.get("assay_id")
            if assay_id not in region_links:
                raise ValueError("Spatial-bin measured packages require an explicit region-shape linkage via --measured-region-shapes")
            shapes, regions, registry_manifest, region_ids, link_record = measured_region_elements(
                package, region_links[assay_id], cal, str(cells.sample_id.iloc[0]), source_hashes)
            name, table, matrix, record = measured_cell_table(package, root, regions, registry_manifest,
                region_ids, np.dtype("int64"), tile_size, region_link=link_record)
            region_shapes[link_record["shape_name"]] = shapes
            used_links.add(assay_id)
        else:
            registry_root, registry_manifest, registry_binding = measured_profile_registry(package_header, root, cells, manifest)
            name, table, matrix, record = measured_cell_table(package, registry_root, cells, registry_manifest, instance_ids, label_dtype, tile_size)
            record["profile_registry_binding"] = registry_binding
        if name in measured_tables:
            raise ValueError("Duplicate measured assay ID/package; each linked assay requires a distinct declared assay_id")
        measured_tables[name], measured_matrices[name] = table, matrix
        measured_records.append(record)
    if set(region_links) != used_links:
        raise ValueError("Unused measured region shape-link: no uniquely associated spatial-bin assay package")
    transformations = {COORDINATE_SYSTEM: physical_transform(cal)}
    factors = [2] * pyramid_levels if pyramid_levels else None
    channel_names = {1: ["intensity"], 3: ["red", "green", "blue"], 4: ["red", "green", "blue", "alpha"]}[channels]
    image_element = Image2DModel.parse(lazy_raster(image, tile_size, channels=True), dims=("c", "y", "x"),
        c_coords=channel_names, transformations=transformations, scale_factors=factors)
    labels_element = Labels2DModel.parse(lazy_raster(labels, tile_size), dims=("y", "x"),
        transformations=transformations, scale_factors=factors)
    polygons, domain_table = tissue_elements(tissue_geojson, tissue_coordinates, cal)
    cells_table = cell_table(root, cells, manifest, instance_ids, label_dtype, tile_size)
    neighborhood_record, neighborhood_check = {"status": "not_provided"}, None
    if manifest.get("neighborhood_feature_store") is not None:
        neighborhood_record, neighborhood_check = attach_neighborhood_features(cells_table, root, manifest, tile_size)
    reference_records, reference_checks = {}, {}
    if cell_reference_mapping is not None:
        reference_records["cell"], reference_checks["cell"] = attach_reference_mapping(
            cells_table, root, cell_reference_mapping, "cell", "cells")
    if region_reference_mapping is not None:
        reference_records["region"], reference_checks["region"] = attach_reference_mapping(
            hierarchy_tables["hierarchy_region_profiles"], hierarchy_info["region_root"],
            region_reference_mapping, "tissue_region", "hierarchy_region_profiles")
    cohort_record, cohort_check = {"status": "not_provided"}, None
    if cohort_niches is not None:
        cohort_record, cohort_check = attach_cohort_niches(
            cells_table, root, cohort_niches, str(cells.sample_id.iloc[0]))
    if native_record is not None:
        cells_table.uns["cellphenotyper"]["neighborhood_support_json"] = json.dumps(native_record, sort_keys=True)
    provenance = {"schema_version": "1.0.0", "sample_id": str(cells.sample_id.iloc[0]),
        "coordinate_system": COORDINATE_SYSTEM, "coordinate_units": "micrometre",
        "crop_origin_px": cal["origin_px"].tolist(), "crop_origin_um": (cal["origin_px"] * cal["mpp"]).tolist(),
        "source_mpp": cal["mpp"], "native_crop_shape_yx": [cal["height"], cal["width"]],
        "canonical_cell_count": len(cells), "tissue_polygon_count": len(polygons),
        "tiff_read_policy": "bounded_RasterReader_windows", "tile_size": tile_size, "dask_workers": workers,
        "feature_blocks": list(manifest.get("feature_blocks", {})), "pyramid_levels": pyramid_levels,
        "canonical_raster_lineage": raster_lineage,
        "measured_modalities_status": ("verified_separate_cell_and_or_region_tables" if region_shapes else "verified_separate_cell_tables") if measured_records else "not_provided; predicted marker features remain explicitly predicted",
        "measured_modalities": measured_records,
        "reference_mappings": reference_records,
        "cohort_niches": cohort_record,
        "neighborhood_features": neighborhood_record,
        "tissue_hierarchy": hierarchy_record if hierarchy_record is not None else {"status": "not_provided"},
        "versions": {name: importlib.metadata.version(name) for name in ("spatialdata", "anndata", "zarr", "dask", "tifffile")},
        "source_hashes": source_hashes}
    if native_record is not None:
        provenance["neighborhood_support"] = native_record
    sdata = sd.SpatialData(images={HE_IMAGE: image_element}, labels={CELL_LABELS: labels_element, **hierarchy_labels, **native_labels},
        shapes={TISSUE_SHAPES: polygons, **region_shapes}, tables={"cells": cells_table, "tissue_domain_annotations": domain_table, **measured_tables, **hierarchy_tables},
        attrs={"cellphenotyper": provenance})
    outdir.parent.mkdir(parents=True, exist_ok=True)
    _verify_reference_mapping_sources(reference_checks)
    _verify_cohort_niche_sources(cohort_check)
    if neighborhood_check is not None:
        neighborhood_check["reader"].recheck()
    with dask.config.set(scheduler="threads", num_workers=workers):
        sdata.write(outdir, overwrite=False)
    _verify_reference_mapping_sources(reference_checks)
    _verify_cohort_niche_sources(cohort_check)
    if neighborhood_check is not None:
        neighborhood_check["reader"].recheck()
    # Verify via the public APIs without eagerly reloading all cell embeddings.
    reloaded = sd.SpatialData.read(outdir, selection=("images", "labels", "shapes"))
    if set(reloaded.images) != {HE_IMAGE} or set(reloaded.labels) != {CELL_LABELS, *hierarchy_labels, *native_labels} or set(reloaded.shapes) != {TISSUE_SHAPES, *region_shapes}:
        raise RuntimeError("SpatialData round trip changed spatial elements")
    if region_shapes:
        from spatialdata.transformations import get_transformation
        for name, expected_shapes in region_shapes.items():
            actual_shapes = reloaded.shapes[name]
            expected_transform = get_transformation(expected_shapes, to_coordinate_system=COORDINATE_SYSTEM)
            actual_transform = get_transformation(actual_shapes, to_coordinate_system=COORDINATE_SYSTEM)
            if (actual_shapes.index.tolist() != expected_shapes.index.tolist()
                    or actual_shapes.region_uid.tolist() != expected_shapes.region_uid.tolist()
                    or not all(a.equals_exact(b, tolerance=0) for a, b in zip(actual_shapes.geometry, expected_shapes.geometry))
                    or not np.array_equal(actual_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                                          expected_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))):
                raise RuntimeError("SpatialData round trip changed registered region identity/geometry/transform")
    reloaded_table = ad.experimental.read_lazy(zarr.open_group(outdir, mode="r")["tables"]["cells"])
    if reloaded_table.obs_names.tolist() != cells.cell_uid.tolist():
        raise RuntimeError("SpatialData round trip changed canonical cell identities")
    if set(np.asarray(reloaded_table.obs["instance_id"].values).astype(int)) != set(instance_ids):
        raise RuntimeError("SpatialData round trip changed table-to-label identities")
    for name, expected_matrix in measured_matrices.items():
        measured = ad.experimental.read_lazy(zarr.open_group(outdir, mode="r")["tables"][name])
        expected_table = measured_tables[name]
        if (measured.obs_names.tolist() != expected_table.obs_names.tolist()
                or not np.array_equal(np.asarray(measured.obs["instance_id"].values).astype(int), expected_table.obs.instance_id.to_numpy())):
            raise RuntimeError("SpatialData round trip changed measured-table observation identities")
        if measured.X.dtype != np.float64 or measured.X.shape != expected_matrix.shape:
            raise RuntimeError("SpatialData round trip changed measured precision/shape")
        for start in range(0, expected_matrix.shape[0], tile_size):
            actual = measured.X[start:start + tile_size]
            actual = actual.compute() if hasattr(actual, "compute") else np.asarray(actual)
            if not np.array_equal(actual, expected_matrix[start:start + tile_size], equal_nan=True):
                raise RuntimeError("SpatialData round trip changed measured values or NaNs")
    if hierarchy_record is not None:
        _verify_hierarchy_roundtrip(reloaded, zarr.open_group(outdir, mode="r"), hierarchy_tables,
            hierarchy_info, hierarchy_matrices, hierarchy_mapping, tile_size, cal)
    if native_record is not None:
        _verify_native_support_roundtrip(reloaded, reloaded_table, native_info, native_record, tile_size, cal)
    if reference_checks:
        _verify_reference_mapping_roundtrip(zarr.open_group(outdir, mode="r"), reloaded, reference_records, reference_checks)
        _verify_reference_mapping_sources(reference_checks)
    if cohort_check is not None:
        _verify_cohort_niche_roundtrip(zarr.open_group(outdir, mode="r"), reloaded, cohort_check)
        _verify_cohort_niche_sources(cohort_check)
    if neighborhood_check is not None:
        _verify_neighborhood_feature_roundtrip(zarr.open_group(outdir, mode="r"), reloaded, neighborhood_check, tile_size)
        neighborhood_check["reader"].recheck()
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "image", "labels", "shift", "outdir", "tissue-geojson"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tissue-coordinates", choices=("crop_pixels", "original_pixels", "original_um"), required=True)
    parser.add_argument("--resolution-json")
    parser.add_argument("--hierarchy-dir", help="Verified discovered hierarchy paired with an explicitly linked cell-profile bundle")
    parser.add_argument("--cell-reference-mapping", help="Verified .mapping.json receipt bound to the exact final cell-profile registry")
    parser.add_argument("--region-reference-mapping", help="Verified .mapping.json receipt bound to the supplied hierarchy's region profiles")
    parser.add_argument("--cohort-niches", help="Source-verified shared cohort niche bundle bound to the exact final cell profiles; no fitting is performed")
    parser.add_argument("--sample-id", dest="expected_sample_id", help="Reject a canonical profile from a different sample")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pyramid-levels", type=int, default=0)
    parser.add_argument("--measured-assay", action="append", default=[],
                        help="Repeatable imported cell-level or spatial-bin measured-assay package")
    parser.add_argument("--measured-region-shapes", action="append", default=[],
                        help="Repeatable explicit region-shape link JSON for each spatial-bin assay; never links regions to cells")
    provenance = export_spatialdata(**vars(parser.parse_args()))
    print(json.dumps({"spatialdata": "written_and_read_back", "cell_count": provenance["canonical_cell_count"],
        "polygons": provenance["tissue_polygon_count"], "coordinate_system": COORDINATE_SYSTEM,
        "measured_assays": len(provenance["measured_modalities"]),
        "reference_mappings": list(provenance["reference_mappings"]),
        "cohort_niches": provenance["cohort_niches"].get("cohort_niche_model_id") is not None}))


if __name__ == "__main__":
    main()
