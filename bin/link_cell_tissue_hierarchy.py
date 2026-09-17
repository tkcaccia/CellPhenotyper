#!/usr/bin/env python3
"""Link canonical cells to tissue hierarchy using exact native-pixel overlaps.

Inputs remain immutable. Membership fractions describe measured raster area,
not calibrated probabilities; unresolved tissue and background remain explicit.
All raster processing uses bounded two-dimensional tiles, including sparse IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from cell_profile_io import RasterReader, sha256_file

SCHEMA = "cellphenotyper.cell_hierarchy_links.v1"
KEYS = ["sample_id", "cell_id", "cell_uid"]
RASTERS = {"parent": "parent_domains.ome.tif", "subdomain": "subdomain_mask.ome.tif",
           "region": "region_mask.ome.tif", "status": "hierarchy_status.ome.tif"}
OVERLAP_COLUMNS = KEYS + ["compartment", "parent_domain_id", "subdomain_id", "region_id",
    "region_uid", "hierarchy_status_code", "parent_uncertainty_code", "overlap_pixels",
    "compartment_pixels", "area_um2", "overlap_fraction"]


def _path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Artifact paths must be safe relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Artifact path is missing or escapes its directory: {name}")
    return path


def _verify(root, name, digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"Missing valid SHA256 for {name}")
    path = _path(root, name)
    if sha256_file(path) != digest:
        raise ValueError(f"SHA256 mismatch: {name}")
    return path


def _integers(values, name, positive=False):
    # Parse decimal text directly: float conversion would alias IDs above 2**53.
    result = []
    for value in values:
        text = str(value)
        if not re.fullmatch(r"[0-9]+", text):
            raise ValueError(f"{name} must be nonnegative decimal integers")
        integer = int(text)
        if integer > np.iinfo(np.int64).max or (positive and integer == 0):
            raise ValueError(f"{name} outside positive signed 64-bit raster range")
        result.append(integer)
    return np.asarray(result, dtype=np.int64)


def _profile(source):
    manifest = json.loads((source / "cell_profiles_manifest.json").read_text())
    if manifest.get("observation_unit") != "cell":
        raise ValueError("Expected canonical cell observations")
    files = manifest.get("files", {})
    for name in ("cell_profiles.parquet", "feature_rows.csv"):
        _verify(source, name, files.get(name))
    for name, digest in files.items():
        _verify(source, name, digest)
    cells = pd.read_parquet(source / "cell_profiles.parquet")
    if not set(KEYS) <= set(cells) or len(cells) != manifest.get("cell_count"):
        raise ValueError("Canonical cell count or identity columns disagree")
    if cells.cell_uid.duplicated().any() or cells.cell_id.duplicated().any() or cells[KEYS].isna().any().any():
        raise ValueError("Duplicate or missing canonical cell identities")
    if not cells.sample_id.astype(str).eq(str(manifest.get("sample_id"))).all():
        raise ValueError("Canonical profile contains foreign sample IDs")
    rows = pd.read_csv(source / "feature_rows.csv", dtype=str, keep_default_na=False)
    if not rows.equals(cells[KEYS].astype(str)):
        raise ValueError("Feature rows differ from canonical cell order")
    numeric_ids = _integers(cells.cell_id, "cell_id", positive=True)
    if len(set(map(int, numeric_ids))) != len(cells):
        raise ValueError("Canonical cell IDs alias the same numeric raster identity")
    for group, record in manifest.get("feature_blocks", {}).items():
        path = _verify(source, record["path"], record.get("sha256"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != tuple(record["shape"]) or values.shape[0] != len(cells):
            raise ValueError(f"Feature block shape mismatch: {group}")
    for record in manifest.get("spatial_graphs", {}).values():
        from scipy import sparse
        path = _verify(source, record["path"], record.get("sha256"))
        if sparse.load_npz(path).shape != (len(cells), len(cells)):
            raise ValueError("Spatial graph dimensions differ from canonical rows")
    if manifest.get("neighborhood_feature_store") is not None:
        from neighborhood_feature_io import FeatureColumns
        FeatureColumns(source, manifest=manifest)
    return manifest, cells


def _raster(reader, shape):
    if tuple(reader.reader.shape) != tuple(shape) or reader.dtype.kind not in "ui":
        raise ValueError("Hierarchy and cell rasters must be integer native-crop YX")


def _tiles(shape, tile_size):
    for y in range(0, shape[0], tile_size):
        for x in range(0, shape[1], tile_size):
            yield x, y, min(x + tile_size, shape[1]), min(y + tile_size, shape[0])


def load_verified_hierarchy(profile_dir, labels, hierarchy_dir, *, tile_size=512):
    """Read-only shared validator for linker/exporter; returns verified paths/data.

The exact provided label file is bound to the cell manifest. Image, shift and
resolution identities must agree between manifests, and their resolved physical
geometry must agree numerically. No historical paths are dereferenced.
"""
    if not 2 <= tile_size <= 4096:
        raise ValueError("tile_size must be in [2, 4096]")
    source, root = Path(profile_dir).resolve(), Path(hierarchy_dir).resolve()
    manifest, cells = _profile(source)
    summary_path = root / "hierarchy_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("sample_id") != manifest.get("sample_id"):
        raise ValueError("Hierarchy sample identity differs from canonical profile")
    if summary.get("parent_labels_immutable") is not True or not summary.get("hierarchy_id"):
        raise ValueError("Hierarchy must declare immutable parents and an identity")
    canonical, inputs = manifest.get("inputs", {}), summary.get("inputs", {})
    if not canonical.get("labels_sha256") or sha256_file(labels) != canonical["labels_sha256"]:
        raise ValueError("Canonical labels SHA256 differs from provided raster")
    required = {"image": "image_sha256", "shift_json": "shift_sha256",
                "resolution_json": "resolution_json_sha256", "parent_mask": "domain_mask_sha256"}
    if "parent_uncertainty" in inputs or canonical.get("domain_uncertainty_sha256"):
        required["parent_uncertainty"] = "domain_uncertainty_sha256"
    for key, field in required.items():
        digest = canonical.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or inputs.get(key, {}).get("sha256") != digest:
            raise ValueError(f"Hierarchy {key} source SHA256 differs from canonical profile")
    geometry = summary.get("geometry", {})
    if geometry.get("coordinate_space") != "original_slide_micrometres" or manifest.get("coordinate_system") != "original_slide_micrometres":
        raise ValueError("Explicit original-slide micrometre coordinate frames required")
    mpp = float(manifest.get("source_mpp", 0))
    size, origin = manifest.get("crop_size_px", []), manifest.get("crop_origin_um", [])
    if not np.isfinite(mpp) or not .01 <= mpp <= 10 or len(size) != 2 or len(origin) != 2:
        raise ValueError("Canonical calibration is missing or invalid")
    size = _integers(size, "crop_size_px", positive=True)
    expected = {"shape_yx": size[::-1], "mpp_xy": [mpp, mpp], "origin_um_xy": origin,
                "origin_px_xy": np.asarray(origin, float) / mpp}
    for key, values in expected.items():
        actual = np.asarray(geometry.get(key, []), float)
        if actual.shape != (2,) or not np.isfinite(actual).all() or not np.allclose(actual, values, rtol=1e-7, atol=1e-7):
            raise ValueError(f"Hierarchy geometry differs from canonical crop: {key}")
    shape = tuple(map(int, geometry["shape_yx"]))
    status_legend = summary.get("status_codes", {})
    if (not isinstance(status_legend, dict) or not status_legend
            or any(not isinstance(key, str) or not key.isdigit() or str(int(key)) != key
                   or not 0 <= int(key) <= 12 or not isinstance(value, str) or not value.strip()
                   for key, value in status_legend.items())):
        raise ValueError("Invalid categorical hierarchy status legend")
    declared_status_codes = np.array([int(key) for key in status_legend], dtype=np.uint8)
    outputs = summary.get("outputs", {})
    names = dict(RASTERS)
    if "parent_uncertainty" in required:
        names["parent_uncertainty"] = "parent_uncertainty.ome.tif"
    required_outputs = list(names.values()) + ["region_profiles/region_profiles_manifest.json"]
    for name in required_outputs:
        _verify(root, name, outputs.get(name))
    for name, digest in outputs.items():
        _verify(root, name, digest)
    paths = {key: root / name for key, name in names.items()}
    if sha256_file(paths["parent"]) != canonical["domain_mask_sha256"]:
        raise ValueError("Copied parent raster differs from canonical parent")
    if "parent_uncertainty" in paths and sha256_file(paths["parent_uncertainty"]) != canonical["domain_uncertainty_sha256"]:
        raise ValueError("Copied uncertainty raster differs from canonical uncertainty")
    region_root = root / "region_profiles"
    region_manifest = json.loads((region_root / "region_profiles_manifest.json").read_text())
    if (region_manifest.get("observation_unit") != "tissue_region" or
        region_manifest.get("hierarchy_id") != summary["hierarchy_id"] or
        region_manifest.get("coordinate_space") != "original_slide_micrometres"):
        raise ValueError("Region profile hierarchy/coordinate identity mismatch")
    for name in ("region_profiles.csv", "feature_rows.csv"):
        _verify(region_root, name, region_manifest.get("files", {}).get(name))
    for name, digest in region_manifest.get("files", {}).items():
        _verify(region_root, name, digest)
    regions = pd.read_csv(region_root / "region_profiles.csv", dtype={"region_uid": str, "region_id": str, "sample_id": str},
        keep_default_na=False, float_precision="round_trip")
    fields = {"region_uid", "region_id", "sample_id", "parent_domain_id", "subdomain_id", "area_um2", "x_um", "y_um"}
    if not fields <= set(regions) or len(regions) != region_manifest.get("region_count") or len(regions) != summary.get("regions"):
        raise ValueError("Region table schema/count mismatch")
    if regions.region_uid.duplicated().any() or regions.region_uid.eq("").any() or regions.region_id.duplicated().any() or not regions.sample_id.eq(str(manifest["sample_id"])).all():
        raise ValueError("Region identity missing, duplicate or foreign")
    region_rows = pd.read_csv(region_root / "feature_rows.csv", dtype=str, keep_default_na=False)
    if not region_rows.equals(regions[["region_uid", "region_id", "sample_id"]].astype(str)):
        raise ValueError("Region feature order differs from table")
    for record in region_manifest.get("feature_blocks", {}).values():
        path = _verify(region_root, record["path"], region_manifest["files"].get(record["path"]))
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != tuple(record["shape"]) or matrix.shape[0] != len(regions):
            raise ValueError("Region feature shape mismatch")
    ids = _integers(regions.region_id, "region_id", True)
    if len(set(map(int, ids))) != len(regions):
        raise ValueError("Region IDs alias the same numeric raster identity")
    parent_ids = _integers(regions.parent_domain_id, "parent_domain_id", True)
    subdomain_ids = _integers(regions.subdomain_id, "subdomain_id", True)
    region_lookup = {int(r): (int(p), int(s)) for r, p, s in zip(ids, parent_ids, subdomain_ids)}
    sd_parent = {}
    for p, s in zip(parent_ids, subdomain_ids):
        if int(s) in sd_parent and sd_parent[int(s)] != int(p):
            raise ValueError("One subdomain cannot belong to multiple parents")
        sd_parent[int(s)] = int(p)
    stats, observed_cells = defaultdict(lambda: [0, 0, 0]), Counter()
    canonical_ids = set(map(int, _integers(cells.cell_id, "cell_id", True)))
    with ExitStack() as stack:
        readers = {key: stack.enter_context(RasterReader(path)) for key, path in paths.items()}
        cell_reader = stack.enter_context(RasterReader(labels))
        for reader in [cell_reader, *readers.values()]:
            _raster(reader, shape)
        for bounds in _tiles(shape, tile_size):
            x, y, x1, y1 = bounds
            blocks = {key: reader.window(*bounds) for key, reader in readers.items()}
            p, s, r, status = (blocks[key] for key in ("parent", "subdomain", "region", "status"))
            if any(np.any(v < 0) or np.any(v > np.iinfo(np.int64).max) for v in blocks.values()):
                raise ValueError("Raster identity outside nonnegative signed 64-bit range")
            u = blocks.get("parent_uncertainty", np.zeros(p.shape, np.uint8))
            # Codes 11/12 retain native graph isolates/ambiguous assignments as
            # unresolved parent tissue. They must not be lost when cells link
            # to the new KODAMA hierarchy. Unknown or undeclared codes fail.
            if np.any(status > 12) or not np.isin(status, declared_status_codes).all() or np.any(u > 254):
                raise ValueError("Invalid categorical hierarchy/uncertainty status code")
            if np.any((p == 0) != (status == 0)) or np.any((s > 0) != (r > 0)) or np.any((r > 0) != (status == 1)):
                raise ValueError("Hierarchy parent/subdomain/region/status rasters disagree")
            # The producer can abstain for an entire grid core when its parent
            # support is uncertain. Those neighbouring pixels retain raw code
            # 0 even though their hierarchy status is 10; never inflate their
            # literal upstream uncertainty area to the whole grid core.
            if np.any((p > 0) & (u > 0) & (status != 10)):
                raise ValueError("Parent uncertainty must remain unresolved hierarchy status 10")
            if np.any((r > 0) & ((p == 0) | (u > 0))):
                raise ValueError("Accepted region overlaps background or uncertain parent")
            labels_block = cell_reader.window(*bounds)
            if np.any(labels_block < 0):
                raise ValueError("Canonical nuclear raster contains negative labels")
            cell_values, counts = np.unique(labels_block, return_counts=True)
            if any(int(v) not in canonical_ids for v in cell_values if v > 0):
                raise ValueError("Cell raster contains noncanonical cell IDs")
            observed_cells.update({int(v): int(c) for v, c in zip(cell_values, counts) if v > 0})
            selected = r > 0
            if selected.any():
                # Aggregate joint IDs once, not a full tile scan per region;
                # fragmented tissue must not cause quadratic raster work.
                keys, inverse, multiplicity = np.unique(np.column_stack([v[selected].astype(np.int64) for v in (r, p, s)]),
                    axis=0, return_inverse=True, return_counts=True)
                yy, xx = np.nonzero(selected)
                order = np.argsort(inverse, kind="stable")
                starts = np.r_[0, np.cumsum(multiplicity)[:-1]]
                sums_x = np.add.reduceat(2 * (xx[order] + x) + 1, starts)
                sums_y = np.add.reduceat(2 * (yy[order] + y) + 1, starts)
                for (region, parent, subdomain), count, sum_x, sum_y in zip(keys, multiplicity, sums_x, sums_y):
                    rid = int(region)
                    if rid not in region_lookup:
                        raise ValueError("Region raster contains an unregistered identity")
                    if region_lookup[rid] != (int(parent), int(subdomain)):
                        raise ValueError("Raster parent/subdomain membership differs from region table")
                    # Integer doubled pixel centres keep sums exact across tiles.
                    stats[rid][0] += int(count)
                    stats[rid][1] += int(sum_x)
                    stats[rid][2] += int(sum_y)
    if set(observed_cells) != canonical_ids:
        raise ValueError("Canonical cells are absent from the exact nuclear raster")
    if set(stats) != set(region_lookup):
        raise ValueError("Region table contains identities absent from the raster")
    for row in regions.itertuples():
        count, sum_x, sum_y = stats[int(row.region_id)]
        expected_values = [count * mpp * mpp, sum_x / (2 * count) * mpp + origin[0], sum_y / (2 * count) * mpp + origin[1]]
        values = np.asarray([row.area_um2, row.x_um, row.y_um], float)
        if not np.isfinite(values).all() or not np.allclose(values, expected_values, rtol=1e-7, atol=1e-7):
            raise ValueError("Region area/centroid differs from native raster geometry")
    return {"manifest": manifest, "cells": cells, "summary": summary, "regions": regions,
            "region_manifest": region_manifest, "region_root": region_root, "raster_paths": paths,
            "geometry": geometry, "hierarchy_root": root, "hierarchy_summary_sha256": sha256_file(summary_path),
            "labels": Path(labels).resolve(), "nuclear_pixels": dict(observed_cells), "tile_size": tile_size}


def _verify_ring(path, info):
    path = Path(path)
    summary_path = path.parent / "compartment_summary.json"
    if not summary_path.is_file():
        raise ValueError("Ring labels require their compartment_summary.json provenance")
    report = json.loads(summary_path.read_text())
    manifest = info["manifest"]
    provenance = manifest.get("tables", {}).get("compartment_qc", {})
    if not provenance.get("mask_lineage_verified") or provenance.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("Ring compartment summary is not bound to the canonical profile")
    for key, field in (("labels", "labels_sha256"), ("shift", "shift_sha256"), ("resolution", "resolution_json_sha256")):
        if report.get("inputs", {}).get(key, {}).get("sha256") != manifest["inputs"][field]:
            raise ValueError(f"Ring {key} input differs from canonical profile")
    digest = sha256_file(path)
    if report.get("output_artifacts", {}).get("perinuclear_ring", {}).get("sha256") != digest:
        raise ValueError("Ring raster SHA256 differs from compartment provenance")
    return digest


def compute_overlaps(info, *, ring_labels=None, max_overlap_records=10_000_000):
    """Exact sparse joint-partition counts; no label-sized dense arrays."""
    cells, shape = info["cells"], tuple(info["geometry"]["shape_yx"])
    canonical_ids = set(map(int, _integers(cells.cell_id, "cell_id", True)))
    counts = {"nucleus": Counter()}
    compartments = {"nucleus": info["labels"]}
    if ring_labels is not None:
        _verify_ring(ring_labels, info)
        compartments["perinuclear_ring"] = Path(ring_labels)
        counts["perinuclear_ring"] = Counter()
    with ExitStack() as stack:
        readers = {key: stack.enter_context(RasterReader(path)) for key, path in info["raster_paths"].items()}
        compartment_readers = {key: stack.enter_context(RasterReader(path)) for key, path in compartments.items()}
        for reader in compartment_readers.values():
            _raster(reader, shape)
        for bounds in _tiles(shape, info["tile_size"]):
            blocks = {key: reader.window(*bounds) for key, reader in readers.items()}
            nuclear = compartment_readers["nucleus"].window(*bounds)
            for compartment, reader in compartment_readers.items():
                labels = nuclear if compartment == "nucleus" else reader.window(*bounds)
                if np.any(labels < 0):
                    raise ValueError("Compartment raster contains negative labels")
                if compartment != "nucleus" and np.any((labels > 0) & (nuclear > 0)):
                    raise ValueError("Perinuclear ring overlaps nuclear pixels")
                selected = labels > 0
                if not selected.any():
                    continue
                # 255 is a link-table-only sentinel for an unavailable upstream
                # uncertainty raster; never misreport unavailable as code 0.
                u = blocks.get("parent_uncertainty", np.full(labels.shape, 255, np.uint8))
                values = [labels[selected], *[blocks[k][selected] for k in ("parent", "subdomain", "region", "status")], u[selected]]
                if any(np.any(v > np.iinfo(np.int64).max) for v in values):
                    raise ValueError("Compartment ID exceeds signed 64-bit range")
                keys = np.column_stack([v.astype(np.int64) for v in values])
                unique, multiplicity = np.unique(keys, axis=0, return_counts=True)
                for key, count in zip(unique, multiplicity):
                    if int(key[0]) not in canonical_ids:
                        raise ValueError("Compartment raster contains noncanonical cell IDs")
                    counts[compartment][tuple(map(int, key))] += int(count)
                if sum(len(c) for c in counts.values()) > max_overlap_records:
                    raise ValueError("Exact overlap table exceeds max_overlap_records memory guard")
    region_uids = {int(row.region_id): row.region_uid for row in info["regions"].itertuples()}
    rows = []
    mpp = float(info["manifest"]["source_mpp"])
    for compartment, counter in counts.items():
        grouped, totals = defaultdict(list), Counter()
        for key, count in counter.items():
            grouped[key[0]].append((key, count))
            totals[key[0]] += count
        for cell in cells.itertuples():
            cid = int(cell.cell_id)
            for (_, parent, subdomain, region, status, uncertainty), count in sorted(grouped[cid]):
                rows.append([str(cell.sample_id), str(cell.cell_id), str(cell.cell_uid), compartment,
                    parent, subdomain, region, region_uids.get(region, ""), status, uncertainty,
                    count, totals[cid], count * mpp * mpp, count / totals[cid]])
    return pd.DataFrame(rows, columns=OVERLAP_COLUMNS)


def _summaries(cells, overlaps, compartments):
    result = cells.copy()
    if any(c.startswith("hierarchy_") for c in cells):
        raise ValueError("Canonical table already contains hierarchy summaries; use its original parent")
    for compartment in compartments:
        groups = {uid: frame for uid, frame in overlaps[overlaps.compartment == compartment].groupby("cell_uid", sort=False)}
        records = []
        for cell in cells.itertuples():
            rows = groups.get(str(cell.cell_uid))
            if rows is None:
                records.append({"pixels": 0, "status": "empty_compartment", "background_fraction": np.nan,
                    "unresolved_fraction": np.nan, "parent_uncertain_fraction": np.nan, "accepted_fraction": np.nan,
                    "parent_zero_fraction": np.nan, "unassigned_parent_uncertain_fraction": np.nan,
                    "unassigned_parent_uncertainty_unavailable_fraction": np.nan,
                    "parent_count": 0, "region_count": 0, "dominant_parent_id": 0, "dominant_parent_fraction": np.nan,
                    "dominant_region_id": 0, "dominant_region_uid": "", "dominant_region_fraction": np.nan,
                    "crosses_parent_boundary": False, "crosses_region_boundary": False})
                continue
            total = int(rows.compartment_pixels.iloc[0])
            parents = rows[rows.parent_domain_id > 0].groupby("parent_domain_id").overlap_pixels.sum()
            regions = rows[rows.region_id > 0].groupby("region_id").overlap_pixels.sum()
            # Ties use the smallest numeric ID, recorded solely as a summary.
            parent = int(parents.idxmax()) if len(parents) else 0
            region = int(regions.idxmax()) if len(regions) else 0
            parent_zero = rows.parent_domain_id == 0
            parent_zero_pixels = int(rows.loc[parent_zero, "overlap_pixels"].sum())
            # Parent label 0 is not necessarily empty tissue: raw code 253 (or
            # another unresolved code) can explicitly describe unassigned tissue.
            background = int(rows.loc[parent_zero & rows.parent_uncertainty_code.eq(0), "overlap_pixels"].sum())
            unassigned_uncertain = int(rows.loc[parent_zero & rows.parent_uncertainty_code.between(1, 254), "overlap_pixels"].sum())
            unassigned_unavailable = int(rows.loc[parent_zero & rows.parent_uncertainty_code.eq(255), "overlap_pixels"].sum())
            unresolved = int(rows.loc[(rows.parent_domain_id > 0) & (rows.region_id == 0), "overlap_pixels"].sum()) + unassigned_uncertain
            uncertainty_available = not rows.parent_uncertainty_code.eq(255).all()
            uncertain = int(rows.loc[(rows.parent_domain_id > 0) & rows.parent_uncertainty_code.between(1, 254), "overlap_pixels"].sum())
            accepted = int(rows.loc[rows.region_id > 0, "overlap_pixels"].sum())
            if unassigned_unavailable:
                status = "no_parent_assignment_uncertainty_unavailable" if unassigned_unavailable == total else "partially_unassigned_parent_uncertainty_unavailable"
            else:
                status = "background_only" if background == total else ("fully_accepted" if accepted == total else "partially_or_fully_unresolved")
            records.append({"pixels": total, "status": status,
                "background_fraction": background / total if not unassigned_unavailable else np.nan, "unresolved_fraction": unresolved / total,
                "parent_uncertain_fraction": uncertain / total if uncertainty_available else np.nan, "accepted_fraction": accepted / total,
                "parent_zero_fraction": parent_zero_pixels / total,
                "unassigned_parent_uncertain_fraction": unassigned_uncertain / total if uncertainty_available else np.nan,
                "unassigned_parent_uncertainty_unavailable_fraction": unassigned_unavailable / total,
                "parent_count": len(parents), "region_count": len(regions),
                "dominant_parent_id": parent, "dominant_parent_fraction": int(parents[parent]) / total if parent else 0.,
                "dominant_region_id": region, "dominant_region_uid": str(rows.loc[rows.region_id == region, "region_uid"].iloc[0]) if region else "",
                "dominant_region_fraction": int(regions[region]) / total if region else 0.,
                "crosses_parent_boundary": len(parents) + int(parent_zero_pixels > 0) > 1,
                "crosses_region_boundary": len(regions) + int(unresolved > 0 or parent_zero_pixels > 0) > 1})
        summaries = pd.DataFrame(records, index=result.index).add_prefix(f"hierarchy_{compartment}_")
        result = pd.concat([result, summaries], axis=1)
    return result


def load_verified_links(profile_dir, hierarchy_info):
    """Validate completed immutable link tables for export; never recompute them."""
    root = Path(profile_dir).resolve()
    manifest, cells = _profile(root)
    record = manifest.get("hierarchy_links", {})
    if (record.get("schema") != SCHEMA or record.get("hierarchy_summary_sha256") != hierarchy_info["hierarchy_summary_sha256"] or
        record.get("hierarchy_id") != hierarchy_info["summary"]["hierarchy_id"] or
        record.get("labels_sha256") != manifest.get("inputs", {}).get("labels_sha256")):
        raise ValueError("Cell hierarchy link provenance differs from verified hierarchy")
    path = _verify(root, record.get("table"), record.get("table_sha256"))
    table = pd.read_parquet(path)
    if list(table.columns) != OVERLAP_COLUMNS:
        raise ValueError("Unexpected hierarchy overlap table schema")
    if len(table) != record.get("overlap_rows") or table.duplicated(KEYS + ["compartment", "parent_domain_id", "subdomain_id", "region_id", "hierarchy_status_code", "parent_uncertainty_code"]).any():
        raise ValueError("Duplicate overlap rows or wrong recorded row count")
    cells_by_uid = cells.set_index("cell_uid")
    region_by_id = {int(r.region_id): r for r in hierarchy_info["regions"].itertuples()}
    compartments = record.get("compartments")
    if not isinstance(compartments, list) or "nucleus" not in compartments or not set(compartments) <= {"nucleus", "perinuclear_ring"}:
        raise ValueError("Invalid overlap compartment inventory")
    if not set(table.compartment) <= set(compartments):
        raise ValueError("Overlap table contains undeclared compartments")
    for name in ["parent_domain_id", "subdomain_id", "region_id", "hierarchy_status_code", "parent_uncertainty_code", "overlap_pixels", "compartment_pixels"]:
        _integers(table[name], name, positive=name in {"overlap_pixels", "compartment_pixels"})
    for row in table.itertuples():
        if row.cell_uid not in cells_by_uid.index:
            raise ValueError("Overlap table contains noncanonical cell UID")
        cell = cells_by_uid.loc[row.cell_uid]
        if str(cell.cell_id) != str(row.cell_id) or str(cell.sample_id) != str(row.sample_id):
            raise ValueError("Overlap cell foreign keys disagree")
        if row.region_id:
            region = region_by_id.get(int(row.region_id))
            if region is None or row.region_uid != region.region_uid or row.parent_domain_id != int(region.parent_domain_id) or row.subdomain_id != int(region.subdomain_id):
                raise ValueError("Overlap region foreign keys disagree")
        elif row.region_uid or row.subdomain_id:
            raise ValueError("Unresolved overlap has a fabricated region/subdomain")
    expected_fraction = table.overlap_pixels.to_numpy(float) / table.compartment_pixels.to_numpy(float)
    expected_area = table.overlap_pixels.to_numpy(float) * float(manifest["source_mpp"]) ** 2
    if not np.allclose(table.overlap_fraction, expected_fraction, rtol=0, atol=1e-14) or not np.allclose(table.area_um2, expected_area, rtol=1e-12, atol=1e-12):
        raise ValueError("Overlap fractions/physical areas disagree with integer counts")
    for (uid, compartment), rows in table.groupby(["cell_uid", "compartment"], sort=False):
        if rows.compartment_pixels.nunique() != 1 or int(rows.overlap_pixels.sum()) != int(rows.compartment_pixels.iloc[0]):
            raise ValueError("Overlap partition does not sum to compartment size")
        if compartment == "nucleus" and int(rows.compartment_pixels.iloc[0]) != hierarchy_info["nuclear_pixels"][int(cells_by_uid.loc[uid].cell_id)]:
            raise ValueError("Nuclear overlap denominator differs from exact canonical raster")
    if set(table.loc[table.compartment == "nucleus", "cell_uid"]) != set(cells.cell_uid):
        raise ValueError("Nuclear overlap table omits canonical cells")
    original_columns = record.get("original_columns", [])
    if not original_columns or not set(original_columns) <= set(cells):
        raise ValueError("Link manifest lacks retained canonical column inventory")
    source_files = record.get("source_files", {})
    for name in ("cell_profiles_manifest.json", "cell_profiles.parquet"):
        _verify(root, f"hierarchy_source/{name}", source_files.get(name))
    for name, digest in source_files.items():
        _verify(root, f"hierarchy_source/{name}", digest)
    original_manifest = json.loads((root / "hierarchy_source/cell_profiles_manifest.json").read_text())
    original = pd.read_parquet(root / "hierarchy_source/cell_profiles.parquet")
    if (record.get("parent_profile_manifest_sha256") != source_files["cell_profiles_manifest.json"] or
        original_manifest.get("files", {}).get("cell_profiles.parquet") != source_files["cell_profiles.parquet"] or
        original_manifest.get("sample_id") != manifest.get("sample_id") or
        original_manifest.get("inputs") != manifest.get("inputs") or list(original) != original_columns or
        not original.equals(cells[original_columns])):
        raise ValueError("Hierarchy linkage changed the original canonical cell values or provenance")
    derivation = manifest.get("cell_hierarchy", {})
    if derivation.get("source_profile_manifest_sha256") != record["parent_profile_manifest_sha256"]:
        raise ValueError("Cell hierarchy derivation identity mismatch")
    recalculated = _summaries(cells[original_columns], table, compartments)
    if not recalculated.equals(cells):
        raise ValueError("Canonical hierarchy summaries disagree with exact overlaps")
    # A self-consistent table/hash pair is not evidence of geometric accuracy.
    # Recompute native integer counts, including preserved optional ring masks.
    ring = root / "hierarchy_source/labels_perinuclear_ring.tif" if "perinuclear_ring" in compartments else None
    if ring is not None and record.get("ring_labels_sha256") != source_files.get("labels_perinuclear_ring.tif"):
        raise ValueError("Preserved ring raster identity differs from link provenance")
    actual = compute_overlaps(hierarchy_info, ring_labels=ring)
    if not actual.equals(table):
        raise ValueError("Hierarchy overlap assignments differ from exact native rasters")
    return table


def link_profiles(profile_dir, labels, hierarchy_dir, outdir, *, ring_labels=None, tile_size=512, max_overlap_records=10_000_000):
    source, out = Path(profile_dir).resolve(), Path(outdir).resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise FileExistsError("Linked profiles require a separate new output directory")
    info = load_verified_hierarchy(source, labels, hierarchy_dir, tile_size=tile_size)
    if info["manifest"].get("hierarchy_links"):
        raise ValueError("Already linked profile; use the original unlinked parent")
    overlaps = compute_overlaps(info, ring_labels=ring_labels, max_overlap_records=max_overlap_records)
    compartments = ["nucleus"] + (["perinuclear_ring"] if ring_labels is not None else [])
    profiles = _summaries(info["cells"], overlaps, compartments)
    # Copy every existing artifact, not only the subset this stage understands.
    # Dereference only validated in-directory links; never preserve shared writes.
    for child in source.rglob("*"):
        if child.is_symlink() and not child.resolve().is_relative_to(source):
            raise ValueError("Canonical profile contains an external symlink")
    shutil.copytree(source, out, symlinks=False)
    if (out / "hierarchy_source").exists():
        raise ValueError("Source profile contains a reserved hierarchy_source directory")
    (out / "hierarchy_source").mkdir()
    source_files = {}
    for name in ("cell_profiles_manifest.json", "cell_profiles.parquet", "cell_profiles.csv"):
        if (source / name).is_file():
            shutil.copy2(source / name, out / "hierarchy_source" / name)
            source_files[name] = sha256_file(out / "hierarchy_source" / name)
    if ring_labels is not None:
        for path in (Path(ring_labels), Path(ring_labels).parent / "compartment_summary.json"):
            name = "labels_perinuclear_ring.tif" if path == Path(ring_labels) else path.name
            shutil.copy2(path, out / "hierarchy_source" / name)
            source_files[name] = sha256_file(out / "hierarchy_source" / name)
    profiles.to_csv(out / "cell_profiles.csv", index=False)
    profiles.to_parquet(out / "cell_profiles.parquet", index=False)
    overlaps.to_csv(out / "cell_hierarchy_overlaps.csv", index=False)
    overlaps.to_parquet(out / "cell_hierarchy_overlaps.parquet", index=False)
    manifest = info["manifest"]
    record = {"schema": SCHEMA, "hierarchy_summary_sha256": info["hierarchy_summary_sha256"],
        "hierarchy_id": info["summary"]["hierarchy_id"], "labels_sha256": manifest["inputs"]["labels_sha256"],
        "table": "cell_hierarchy_overlaps.parquet", "table_csv": "cell_hierarchy_overlaps.csv",
        "table_sha256": sha256_file(out / "cell_hierarchy_overlaps.parquet"), "overlap_rows": len(overlaps),
        "parent_profile_manifest_sha256": sha256_file(source / "cell_profiles_manifest.json"),
        "source_root": "hierarchy_source", "source_files": source_files,
        "compartments": compartments, "original_columns": list(info["cells"].columns), "tile_size": tile_size,
        "parent_uncertainty_available": "parent_uncertainty" in info["raster_paths"],
        "semantics": "Exact joint native-pixel partition: parent/subdomain/region/status/parent uncertainty. Fractions use the entire compartment. Parent-zero fraction includes all parent label 0 pixels; p0 with raw codes 1..254 contributes to unassigned-parent-uncertain and unresolved fractions, not background. Background means parent-map label 0 with raw code 0, not independently verified absent tissue. Parent-uncertain fraction covers only positive-parent pixels with raw codes 1..254. With available raw uncertainty, accepted+unresolved+background partitions the compartment. Link-only code 255 means unavailable upstream raster: uncertainty fractions are NaN and parent-zero pixels have explicit unavailable-assignment status/fraction, never confident background. No probability calibration or inferred fills. Dominant labels are summaries only, numeric-ID ties deterministic; empty rings are NaN, not zero.",
        "existing_features_and_graphs": "Copied byte-identically in original canonical row order"}
    if ring_labels is not None:
        record["ring_labels_sha256"] = sha256_file(ring_labels)
    manifest["hierarchy_links"] = record
    manifest["cell_hierarchy"] = {"source_profile_manifest_sha256": record["parent_profile_manifest_sha256"],
        "hierarchy_id": record["hierarchy_id"], "hierarchy_summary_sha256": record["hierarchy_summary_sha256"],
        "source_root": "hierarchy_source", "source_files": source_files}
    manifest["files"].update({name: sha256_file(out / name) for name in (
        "cell_profiles.csv", "cell_profiles.parquet", "cell_hierarchy_overlaps.csv", "cell_hierarchy_overlaps.parquet")})
    (out / "cell_profiles_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    load_verified_links(out, info)
    return profiles, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "labels", "hierarchy-dir", "outdir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ring-labels")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-overlap-records", type=int, default=10_000_000)
    args = vars(parser.parse_args(argv))
    cells, manifest = link_profiles(**args)
    print(json.dumps({"cells": len(cells), "overlap_rows": manifest["hierarchy_links"]["overlap_rows"], "hierarchy_id": manifest["hierarchy_links"]["hierarchy_id"]}))


if __name__ == "__main__":
    main()
