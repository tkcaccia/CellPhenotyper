#!/usr/bin/env python3
"""Serve a local, read-only cell inspector without pre-rendering cell thumbnails.

All paths are supplied at startup. HTTP clients can request cell IDs, coordinates,
and named feature groups, never filesystem paths. Expert notes are a separate
display layer and are not used by profiling, neighbour retrieval, or review flags.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import threading
from collections import Counter
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from cell_profile_io import RasterReader, calibration, sha256_file
from export_spatialdata import load_cell_profile


HTML_PATH = Path(__file__).resolve().parents[1] / "resources" / "cell_inspector.html"


def clean_json(value):
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def truth(value):
    return str(value).lower() in ("true", "1", "1.0", "yes")


def cohort_directory(value):
    bundle = Path(value).resolve()
    if bundle.is_file():
        if bundle.name != "cohort_niches_completion.json":
            raise ValueError("Select the cohort directory or its exact completion receipt")
        bundle = bundle.parent
    return bundle


def spatial_field(name):
    return name.startswith(("tissue_", "hierarchy_", "niche_", "reference_", "neighbor", "boundary_", "distance_")) or (name.startswith("r") and "um_" in name)


def review_reasons(row):
    reasons = []
    if str(row.get("agreement_tier", "")).lower() in ("low", "moderate"):
        reasons.append("detector_disagreement")
    morphology = str(row.get("morphology__morphology_status", row.get("morphology_status", "")))
    if morphology and morphology.lower() not in ("ok", "nan", "none", "<na>"):
        reasons.append("morphology_issue")
    if truth(row.get("morphology__touches_image_edge", row.get("touches_image_edge", False))):
        reasons.append("image_edge")
    if "tissue_domain" in row and str(row["tissue_domain"]).lower() in ("0", "0.0", "unknown", "nan", "<na>"):
        reasons.append("unknown_domain")
    domain_status = str(row.get("tissue_domain_status", "")).lower()
    if domain_status in ("uncertain", "unknown_or_outside_support", "unavailable"):
        reasons.append("uncertain_domain")
    niche = str(row.get("niche_status", "")).lower()
    if niche and niche not in ("assigned", "stable", "assigned_fixed_k", "assigned_exploratory",
                                "single_niche_no_supported_subdivision", "nan", "none", "<na>"):
        reasons.append("niche_review")
    if "phenotype" in row and str(row["phenotype"]).lower() in ("unknown", "unmatched", "nan", "<na>", ""):
        reasons.append("unknown_phenotype")
    stability = row.get("niche_stability")
    if isinstance(stability, (int, float, np.number)) and np.isfinite(stability) and stability < 0.8:
        reasons.append("unstable_niche")
    if str(row.get("reference_status", "")) in ("outside_reference", "ambiguous_reference", "insufficient_reference", "missing_features"):
        reasons.append("unknown_reference")
    cohort_status = str(row.get("cohort_niche_status", "")).lower()
    if cohort_status and cohort_status not in ("assigned", "stable", "assigned_fixed_k", "assigned_exploratory",
            "single_niche_no_supported_subdivision", "nan", "none", "<na>"):
        reasons.append("cohort_niche_review")
    cohort_stability = row.get("cohort_niche_stability")
    if isinstance(cohort_stability, (int, float, np.number)) and np.isfinite(cohort_stability) and cohort_stability < 0.8:
        reasons.append("unstable_cohort_niche")
    return sorted(set(reasons))


class CellInspector:
    def __init__(self, profile_dir, image, labels, shift, resolution_json=None, expert_annotations=None,
                 hierarchy_dir=None, region_annotations=None, rejected_candidates=None, candidate_sources=None,
                 cohort_niches=None):
        if candidate_sources is not None and not rejected_candidates:
            raise ValueError("Candidate source verification requires --rejected-candidates")
        profile_manifest_sha = (sha256_file(Path(profile_dir).resolve() / "cell_profiles_manifest.json")
                                if cohort_niches is not None else None)
        self.cal = calibration(shift, resolution_json)
        self.root, self.cells, self.manifest, instance_ids = load_cell_profile(profile_dir, self.cal)
        self.image, self.labels = Path(image).resolve(), Path(labels).resolve()
        source_hashes = self.manifest.get("inputs", {})
        if not isinstance(source_hashes, dict):
            raise ValueError("Profile input provenance must be an object")
        self.raster_provenance = {}
        for name, path in (("image", self.image), ("labels", self.labels)):
            key = name + "_sha256"
            if key not in source_hashes:
                self.raster_provenance[name] = {"status": "legacy_missing_sha256", "sha256": None}
                continue
            expected = source_hashes[key]
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                raise ValueError(f"Profile {key} is not a valid source hash")
            if sha256_file(path) != expected.lower():
                raise ValueError(f"Inspector {name} SHA256 does not match the canonical profile source")
            self.raster_provenance[name] = {"status": "verified_sha256", "sha256": expected.lower()}
        self.raster_identity_status = ("verified_sha256" if all(record["status"] == "verified_sha256"
            for record in self.raster_provenance.values()) else "legacy_partial_or_unverified")
        self.instance_ids = instance_ids
        for path, is_labels in ((self.image, False), (self.labels, True)):
            with RasterReader(path) as reader:
                if (reader.width, reader.height) != (self.cal["width"], self.cal["height"]):
                    raise ValueError("Inspector rasters must have the calibrated native crop dimensions")
                if is_labels and (len(reader.reader.shape) != 2 or reader.dtype.kind not in "ui"):
                    raise ValueError("Canonical labels must be integer YX")
                if not is_labels and reader.dtype != np.dtype("uint8"):
                    raise ValueError("Inspector H&E display requires uint8 RGB/grey data; supply an explicitly converted display crop")
        self.uid_index = dict(zip(self.cells.cell_uid, range(len(self.cells))))
        self.label_index = dict(zip(map(int, instance_ids), range(len(self.cells))))
        self.xy = self.cells[["x_um", "y_um"]].to_numpy(float) / self.cal["mpp"] - self.cal["origin_px"]
        self.tree = cKDTree(self.xy)
        self.blocks = {}
        self.stats = {}
        self.stats_lock = threading.Lock()
        for name, record in self.manifest.get("feature_blocks", {}).items():
            path = (self.root / record["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Feature block path escapes the profile directory")
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.ndim != 2 or values.shape[1] == 0 or values.shape[0] != len(self.cells) or list(values.shape) != record["shape"] or values.dtype.kind not in "fiu":
                raise ValueError(f"Invalid row-aligned numeric feature block {name}")
            if record.get("sha256") not in (None, sha256_file(path)):
                raise ValueError(f"Feature block hash mismatch: {name}")
            self.blocks[name] = values
        self.neighborhood_reader = None
        self.neighborhood_source_stamps = {}
        self.neighborhood_display_columns = []
        self.neighborhood_lock = threading.RLock()
        if self.manifest.get("neighborhood_feature_store") is not None:
            from neighborhood_feature_io import FeatureColumns
            self.neighborhood_reader = FeatureColumns(self.root, manifest=self.manifest)
            self.neighborhood_display_columns = [column
                for group, record in self.neighborhood_reader.groups.items() if not group.startswith("own:")
                for column in record["columns"] if spatial_field(column)]
            self.verify_neighborhood_attachment(force=True)
        self.annotations = {}
        if expert_annotations:
            notes = json.loads(Path(expert_annotations).read_text())
            if not isinstance(notes, dict) or not isinstance(notes.get("annotations"), list):
                raise ValueError("Expert annotations require an annotations list with cell_uid keys")
            for note in notes["annotations"]:
                uid = note.get("cell_uid") if isinstance(note, dict) else None
                if uid not in self.uid_index or uid in self.annotations:
                    raise ValueError("Expert note has a foreign or duplicate canonical cell UID")
                self.annotations[uid] = note
        self.cohort_niches = None
        self.cohort_record = None
        self.cohort_bundle = None
        self.cohort_source_stamps = {}
        self.cohort_lock = threading.Lock()
        if cohort_niches is not None:
            from cohort_niche_io import COHORT_COLUMNS, KEYS, load_cohort_bundle
            bundle = cohort_directory(cohort_niches)
            frame, record = load_cohort_bundle(bundle, profile_dir=self.root,
                                               sample_id=self.manifest["sample_id"])
            if record["source_identity"]["manifest_sha256"] != profile_manifest_sha:
                raise ValueError("Canonical profile changed while attaching cohort niches")
            if (list(frame.columns) != list(KEYS) + list(COHORT_COLUMNS)
                    or frame[list(KEYS)].values.tolist() != self.cells[list(KEYS)].values.tolist()):
                raise ValueError("Cohort niches must preserve the exact canonical identities and source row order")
            if set(COHORT_COLUMNS) & set(self.cells.columns):
                raise ValueError("Canonical profile already contains cohort fields; refusing an ambiguous attachment")
            self.cohort_niches = frame[list(COHORT_COLUMNS)].reset_index(drop=True)
            self.cohort_record = record
            self.cohort_bundle = bundle
            self.verify_cohort_attachment(force=True)
        self.review = []
        for index, values in enumerate(self.cells.itertuples(index=False, name=None)):
            row = dict(zip(self.cells.columns, values))
            if self.cohort_niches is not None:
                row.update(self.cohort_niches.iloc[index].to_dict())
            flags = review_reasons(row)
            if flags:
                self.review.append({"cell_uid": row["cell_uid"], "cell_id": row["cell_id"], "reasons": flags, "row": index})
        self.review.sort(key=lambda item: (-len(item["reasons"]), item["cell_uid"]))
        self.flags = {record["cell_uid"]: record["reasons"] for record in self.review}
        self.hierarchy_overlaps = None
        self.hierarchy_overlap_indices = {}
        if self.manifest.get('hierarchy_links'):
            if not hierarchy_dir:
                raise ValueError('Linked cell profiles require their verified hierarchy for inspection')
            from link_cell_tissue_hierarchy import load_verified_hierarchy, load_verified_links
            verified = load_verified_hierarchy(self.root, self.labels, hierarchy_dir)
            self.hierarchy_overlaps = load_verified_links(self.root, verified)
            self.hierarchy_overlap_indices = self.hierarchy_overlaps.groupby('cell_uid', sort=False).indices
        self.regions = RegionInspector(self, hierarchy_dir, shift, resolution_json, region_annotations) if hierarchy_dir else None
        if region_annotations and not hierarchy_dir:
            raise ValueError("Region annotations require a verified hierarchy directory")
        self.rejected_candidates = None
        if rejected_candidates:
            from detector_candidate_review import CandidateReview
            self.rejected_candidates = CandidateReview(self, rejected_candidates, shift, resolution_json, candidate_sources)
        self.verify_cohort_attachment(force=True)
        self.verify_neighborhood_attachment(force=True)

    def verify_neighborhood_attachment(self, *, force=False):
        """Recheck changed backing files, not full array bytes on every click."""
        if self.neighborhood_reader is None:
            return
        with self.neighborhood_lock:
            stamps = {}
            for name in self.neighborhood_reader.source_files:
                path = self.root / name
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    raise ValueError("Neighbourhood source path escapes its verified profile directory")
                stat, link = path.stat(), path.lstat()
                stamps[path] = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                               stat.st_ctime_ns, link.st_ino, link.st_mtime_ns, link.st_ctime_ns)
            if force or stamps != self.neighborhood_source_stamps:
                self.neighborhood_reader.recheck()
                if self.neighborhood_source_stamps and any(
                        value[:3] != self.neighborhood_source_stamps[path][:3] for path, value in stamps.items()):
                    # A byte-identical atomic replacement or contained symlink
                    # retarget must remap NPYs, not retain an old inode's mmap.
                    from neighborhood_feature_io import FeatureColumns
                    replacement = FeatureColumns(self.root, manifest=self.manifest)
                    if replacement.source_files != self.neighborhood_reader.source_files:
                        raise ValueError("Neighbourhood source inventory changed during refresh")
                    self.neighborhood_reader = replacement
                self.neighborhood_source_stamps = stamps

    def neighborhood_metadata(self):
        if self.neighborhood_reader is None:
            return {"status": "scalar_profile_fields"}
        return clean_json({"status": "verified_array_backed", "source_files": self.neighborhood_reader.source_files,
            "groups": self.neighborhood_reader.groups, "group_definitions": self.neighborhood_reader.group_definitions,
            "displayed_columns": self.neighborhood_display_columns,
            "own_feature_policy": "Own feature groups remain original similarity blocks; not added to the spatial detail fields.",
            "source_monitoring": "Startup byte verification and per-request file-stat/resolved-path checks; changed sources require full digest recheck. Not a continuous cryptographic audit."})

    def verify_cohort_attachment(self, *, force=False):
        """Rehash changed sources, without rereading large features per cell click."""
        if self.cohort_record is None:
            return
        from cohort_niche_io import verify_cohort_sources
        with self.cohort_lock:
            paths = [(root, root / name) for root, key in ((self.root, "source_files"),
                     (self.cohort_bundle, "bundle_files")) for name in self.cohort_record[key]]
            stamps = {}
            for root, path in paths:
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("Cohort attachment source path escapes its verified directory")
                stat = path.stat()
                link = path.lstat()
                stamps[path] = (str(resolved), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
                               stat.st_ctime_ns, link.st_ino, link.st_mtime_ns, link.st_ctime_ns)
            if force or stamps != self.cohort_source_stamps:
                verify_cohort_sources(self.cohort_bundle, self.root, self.cohort_record)
                self.cohort_source_stamps = stamps

    def row_index(self, uid):
        if uid not in self.uid_index:
            raise KeyError("Unknown canonical cell UID")
        return self.uid_index[uid]

    def metadata(self):
        self.verify_cohort_attachment()
        self.verify_neighborhood_attachment()
        counts = Counter(reason for row in self.review for reason in row["reasons"])
        return {"sample_id": self.manifest["sample_id"], "cell_count": len(self.cells),
            "width": self.cal["width"], "height": self.cal["height"], "source_mpp": self.cal["mpp"],
            "first_cell_uid": self.cells.cell_uid.iloc[0] if len(self.cells) else None,
            "feature_blocks": [{"name": name, "dimension": int(values.shape[1]),
                "reference_compatible": self.manifest["feature_blocks"][name].get("reference_compatible")}
                for name, values in self.blocks.items()],
            "marker_columns": self.manifest.get("biological_marker_features", []),
            "availability_counts": self.manifest.get("availability", {}),
            "review_count": len(self.review), "review_reason_counts": dict(counts),
            "expert_annotations": len(self.annotations), "read_only": True,
            "raster_identity_status": self.raster_identity_status, "raster_provenance": self.raster_provenance,
            "regions": self.regions.metadata() if self.regions else {"status": "not_provided", "region_count": 0},
            "rejected_candidates": self.rejected_candidates.metadata() if self.rejected_candidates else {
                "status": "not_provided", "candidate_count": 0},
            "cohort_niches": ({"status": "verified_source_bound", "record": clean_json(self.cohort_record),
                               "source_monitoring": "File-stat and resolved-path checks on cohort-bearing requests; changed files require a full digest recheck, not a continuous cryptographic audit."}
                              if self.cohort_record is not None else {"status": "not_provided"}),
            "neighborhood_features": self.neighborhood_metadata(),
            "interpretation": "H&E predictions and descriptive distances; no calibrated cell-type or molecular probabilities."}

    def search(self, term, limit=30):
        self.verify_cohort_attachment()
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be in [1,100]")
        term = term.strip()
        indices = np.flatnonzero(self.cells.cell_uid.str.contains(term, regex=False).to_numpy() |
                                 self.cells.cell_id.str.contains(term, regex=False).to_numpy())[:limit]
        return [{"cell_uid": self.cells.cell_uid.iloc[i], "cell_id": self.cells.cell_id.iloc[i],
                 "review_reasons": self.flags.get(self.cells.cell_uid.iloc[i], [])} for i in indices]

    def cell_map(self, limit=6000):
        indices = np.linspace(0, max(len(self.cells) - 1, 0), min(limit, len(self.cells)), dtype=int)
        return {"sampled": len(indices) < len(self.cells), "total": len(self.cells),
            "points": [[float(self.xy[i, 0]), float(self.xy[i, 1])] for i in indices]}

    def queue(self, offset=0, limit=40, reason=""):
        self.verify_cohort_attachment()
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("Review offset/limit out of bounds")
        rows = [row for row in self.review if not reason or reason in row["reasons"]]
        return {"total": len(rows), "rows": [{k: v for k, v in row.items() if k != "row"} for row in rows[offset:offset + limit]]}

    def pick(self, x, y, radius_um=30):
        if not np.isfinite([x, y, radius_um]).all() or radius_um <= 0 or radius_um > 100:
            raise ValueError("Coordinates/radius must be finite; radius in (0,100] um")
        if not 0 <= x < self.cal["width"] or not 0 <= y < self.cal["height"]:
            raise ValueError("Pick coordinates outside the crop")
        with RasterReader(self.labels) as reader:
            value = int(reader.window(int(x), int(y), int(x) + 1, int(y) + 1)[0, 0])
        if value > 0:
            if value not in self.label_index:
                raise ValueError("Clicked label has no canonical profile; check raster/profile pairing")
            index, method = self.label_index[value], "exact_label"
        else:
            distance, index = self.tree.query([x, y])
            if not np.isfinite(distance) or distance * self.cal["mpp"] > radius_um:
                return {"status": "no_nearby_cell"}
            method = "nearest_centroid"
        return {"status": "selected", "cell_uid": self.cells.cell_uid.iloc[index], "method": method}

    def bounds(self, uid, size=384):
        if not 64 <= size <= 1024:
            raise ValueError("Native window size must be in [64,1024] pixels")
        index = self.row_index(uid)
        x, y = self.xy[index]
        x0, y0 = max(0, int(round(x)) - size // 2), max(0, int(round(y)) - size // 2)
        return (x0, y0, min(self.cal["width"], x0 + size), min(self.cal["height"], y0 + size))

    def detail(self, uid, size=384):
        self.verify_cohort_attachment()
        index = self.row_index(uid)
        row = self.cells.iloc[index].to_dict()
        controls = {key for record in self.manifest.get("tables", {}).values()
                    for key in record.get("background_columns", [])}
        markers = {key: value for key, value in row.items() if key.startswith("predicted__") and key.endswith(("__mean", "__sum", "__max")) and key not in controls}
        detector = {key: value for key, value in row.items() if key == "phenotype" or key.startswith(("stardist", "cellvitpp", "hovernet", "consensus", "agreement", "phenotype_", "broad_", "scoped_", "geometry_source"))}
        # Also visible in the existing evidence table, without implying that a
        # legacy ID/shape match proves source-image or segmentation identity.
        detector["source_raster_identity"] = self.raster_identity_status
        morphology = {key: value for key, value in row.items() if key.startswith("morphology__")}
        spatial = {key: value for key, value in row.items() if spatial_field(key)}
        if self.neighborhood_reader is not None:
            with self.neighborhood_lock:
                self.verify_neighborhood_attachment()
                before = dict(self.neighborhood_source_stamps)
                values = self.neighborhood_reader.read(np.array([index], dtype=np.int64), self.neighborhood_display_columns)
                self.verify_neighborhood_attachment()
                if before != self.neighborhood_source_stamps:
                    raise ValueError("Neighbourhood sources changed during cell detail retrieval; retry after source stabilization")
                spatial.update(zip(self.neighborhood_display_columns, values[0]))
        overlap_indices = self.hierarchy_overlap_indices.get(uid, [])
        overlaps = (self.hierarchy_overlaps.iloc[overlap_indices[:512]].to_dict('records')
                    if self.hierarchy_overlaps is not None else [])
        blocks = {name: {"dimension": int(values.shape[1]), "available": bool(np.isfinite(values[index]).all()),
            "missing_dimensions": int((~np.isfinite(values[index])).sum()),
            "reference_compatible": self.manifest["feature_blocks"][name].get("reference_compatible")}
            for name, values in self.blocks.items()}
        return clean_json({"cell_uid": uid, "cell_id": row["cell_id"], "sample_id": row["sample_id"],
            "coordinates_um": [row["x_um"], row["y_um"]], "crop_xy": self.xy[index].tolist(),
            "window": dict(zip(["x0", "y0", "x1", "y1"], self.bounds(uid, size))),
            "markers": markers, "detector_evidence": detector, "morphology": morphology,
            "technical_marker_controls": {key: row[key] for key in controls if key in row},
            "raster_provenance": self.raster_provenance,
            "compartment_evidence": {key: value for key, value in row.items() if key.startswith("compartment__")},
            "feature_availability": {key: value for key, value in row.items() if key.endswith("_available")},
            "spatial": spatial, "feature_blocks": blocks, "review_reasons": self.flags.get(uid, []),
            "neighborhood_features": self.neighborhood_metadata(),
            "cohort_niche": self.cohort_niches.iloc[index].to_dict() if self.cohort_niches is not None else {},
            "tissue_membership_overlaps": overlaps, "tissue_membership_overlap_count": len(overlap_indices),
            "tissue_membership_overlaps_truncated": len(overlap_indices) > 512,
            "expert_annotation": self.annotations.get(uid), "expert_annotation_used_for_inference": False})

    def png(self, uid, size=384, overlay=False):
        bounds = self.bounds(uid, size)
        if overlay:
            with RasterReader(self.labels) as reader:
                mask = reader.window(*bounds) == self.instance_ids[self.row_index(uid)]
            boundary = mask & ~binary_erosion(mask, border_value=0)
            pixels = np.zeros((*mask.shape, 4), dtype=np.uint8)
            pixels[boundary] = [255, 157, 20, 255]
        else:
            with RasterReader(self.image) as reader:
                pixels = reader.window(*bounds)
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return output.getvalue()

    def block_stats(self, name):
        with self.stats_lock:
            if name in self.stats:
                return self.stats[name]
            values = self.blocks[name]
            count, mean, m2 = 0, np.zeros(values.shape[1]), np.zeros(values.shape[1])
            for start in range(0, len(values), 4096):
                block = values[start:start + 4096]
                block = block[np.isfinite(block).all(axis=1)].astype(np.float64)
                n = len(block)
                if not n:
                    continue
                block_mean = block.mean(axis=0)
                delta = block_mean - mean
                m2 += ((block - block_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                mean += delta * n / (count + n)
                count += n
            scales = np.sqrt(m2 / count) if count else np.ones(values.shape[1])
            scales[scales < 1e-8] = 1.
            self.stats[name] = (mean, scales, count)
            return self.stats[name]

    def similar(self, uid, groups, weights=None, k=12, marker=None, min_difference=None):
        # RegionInspector reuses this distance routine but has no cohort-cell
        # layer or cohort review flags.
        if getattr(self, "cohort_record", None) is not None:
            self.verify_cohort_attachment()
        index = self.row_index(uid)
        if not groups or len(set(groups)) != len(groups) or not set(groups) <= set(self.blocks):
            raise ValueError("Select distinct available feature groups explicitly")
        weights = np.ones(len(groups)) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != (len(groups),) or not np.isfinite(weights).all() or (weights <= 0).any() or not 1 <= k <= 100:
            raise ValueError("Positive finite block weights and k in [1,100] required")
        if bool(marker) != (min_difference is not None):
            raise ValueError("Marker discordance requires a column and minimum difference")
        if marker and (marker not in self.manifest.get("biological_marker_features", []) or not np.isfinite(min_difference) or min_difference < 0):
            raise ValueError("Invalid declared marker or discordance threshold")
        if any(not np.isfinite(self.blocks[name][index]).all() for name in groups):
            return {"status": "missing_features", "results": []}
        scales = {name: self.block_stats(name)[1] for name in groups}
        marker_values = pd.to_numeric(self.cells[marker], errors="coerce").to_numpy(float) if marker else None
        if marker and not np.isfinite(marker_values[index]):
            return {"status": "missing_marker", "results": []}
        distances = np.full(len(self.cells), np.inf)
        for start in range(0, len(self.cells), 4096):
            stop = min(len(self.cells), start + 4096)
            valid = np.ones(stop - start, dtype=bool)
            squared = np.zeros(stop - start)
            for name, weight in zip(groups, weights):
                block = self.blocks[name][start:stop]
                present = np.isfinite(block).all(axis=1)
                valid &= present
                # Do not impute missing vectors, even for the distance accumulator.
                delta = (block[present] - self.blocks[name][index]) / scales[name]
                squared[present] += weight * np.mean(delta ** 2, axis=1)
            if marker:
                difference = np.abs(marker_values[start:stop] - marker_values[index])
                valid &= np.isfinite(difference) & (difference >= min_difference)
            chunk = distances[start:stop]
            chunk[valid] = np.sqrt(squared[valid] / weights.sum())
        distances[index] = np.inf
        eligible = np.flatnonzero(np.isfinite(distances))
        ordered = eligible[np.lexsort((self.cells.cell_uid.iloc[eligible].to_numpy(), distances[eligible]))[:k]]
        results = []
        for i in ordered:
            result = {"cell_uid": self.cells.cell_uid.iloc[i], "cell_id": self.cells.cell_id.iloc[i],
                "sample_id": self.cells.sample_id.iloc[i], "distance": float(distances[i]),
                "review_reasons": self.flags.get(self.cells.cell_uid.iloc[i], [])}
            if marker:
                result["predicted_marker_difference"] = float(abs(marker_values[i] - marker_values[index]))
            results.append(result)
        return {"status": "matched" if results else "no_matching_cells", "results": results,
            "groups": groups, "weights": weights.tolist(),
            "distance_semantics": "weighted RMS of per-block mean standardized squared differences; not calibrated confidence",
            "scaling_scope": "finite feature vectors in this specimen only", "expert_annotations_used": False}


class RegionInspector:
    """Verified hierarchy observations plus bounded, explicitly linked cell review.

    A region mean is not a distribution of its source grid features. Only actual
    constituent-cell marker values are summarized as distributions here.
    """
    def __init__(self, parent, directory, shift, resolution_json, annotations=None):
        from cell_reference_atlas import load_profile
        self.parent, self.root = parent, Path(directory).resolve()
        self.summary = json.loads((self.root / "hierarchy_summary.json").read_text())
        summary = self.summary
        if parent.raster_identity_status != "verified_sha256":
            raise ValueError("Region inspection requires verified canonical image and labels SHA256")
        if summary.get("sample_id") != parent.manifest["sample_id"]:
            raise ValueError("Hierarchy sample ID differs from the canonical cell sample")
        geometry = summary.get("geometry", {})
        expected = {"shape_yx": [parent.cal["height"], parent.cal["width"]],
                    "mpp_xy": [parent.cal["mpp"]] * 2,
                    "origin_um_xy": (parent.cal["origin_px"] * parent.cal["mpp"]).tolist()}
        if geometry.get("coordinate_space") != "original_slide_micrometres":
            raise ValueError("Hierarchy must declare original-slide micrometre coordinates")
        for key, value in expected.items():
            actual = np.asarray(geometry.get(key, []), float)
            if actual.shape != np.asarray(value).shape or not np.isfinite(actual).all() or not np.allclose(actual, value, rtol=1e-7, atol=1e-7):
                raise ValueError(f"Hierarchy geometry differs from canonical crop: {key}")
        inputs = summary.get("inputs", {})
        required_inputs = {"shift_json": sha256_file(shift),
                           "image": parent.manifest.get("inputs", {}).get("image_sha256"),
                           "parent_mask": parent.manifest.get("inputs", {}).get("domain_mask_sha256")}
        if resolution_json:
            required_inputs["resolution_json"] = sha256_file(resolution_json)
        else:
            raise ValueError("Region inspection requires the verified physical-resolution report")
        canonical_sources = parent.manifest.get("inputs", {})
        if canonical_sources.get("domain_uncertainty_sha256"):
            required_inputs["parent_uncertainty"] = canonical_sources["domain_uncertainty_sha256"]
        if (canonical_sources.get("shift_sha256") != required_inputs["shift_json"] or
            canonical_sources.get("resolution_json_sha256") != required_inputs["resolution_json"]):
            raise ValueError("Region inspection shift/resolution SHA256 must match canonical profile provenance")
        for key, expected_hash in required_inputs.items():
            if not expected_hash or inputs.get(key, {}).get("sha256") != expected_hash:
                raise ValueError(f"Hierarchy {key} source SHA256 differs from canonical profile inputs")
        outputs = summary.get("outputs", {})
        required_files = ("region_mask.ome.tif", "hierarchy_status.ome.tif", "parent_domains.ome.tif",
                          "grid_subdomains.csv", "region_profiles/region_profiles_manifest.json")
        self.parent_uncertainty = self.root / "parent_uncertainty.ome.tif" if "parent_uncertainty" in inputs else None
        if self.parent_uncertainty:
            required_files += ("parent_uncertainty.ome.tif",)
        for name in required_files:
            record = outputs.get(name)
            digest = record.get("sha256") if isinstance(record, dict) else record
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Hierarchy lacks a valid output SHA256 for {name}; regenerate its provenance")
            if sha256_file(self.root / name) != digest:
                raise ValueError(f"Hierarchy output SHA256 mismatch: {name}")
        if sha256_file(self.root / "parent_domains.ome.tif") != required_inputs["parent_mask"]:
            raise ValueError("Hierarchy parent raster differs from canonical domain mask")
        if self.parent_uncertainty and sha256_file(self.parent_uncertainty) != inputs["parent_uncertainty"].get("sha256"):
            raise ValueError("Hierarchy parent uncertainty copy differs from its declared source")
        manifest_path = self.root / "region_profiles" / "region_profiles_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("hierarchy_id") != summary.get("hierarchy_id") or not manifest.get("hierarchy_id"):
            raise ValueError("Region profile hierarchy identity mismatch")
        if manifest.get("coordinate_space") != "original_slide_micrometres":
            raise ValueError("Region profile coordinate space is not explicit")
        # The summary binds the manifest; the manifest in turn must bind every
        # table and numeric block, not merely assert a compatible shape.
        names = ["region_profiles.csv", "feature_rows.csv"] + [r["path"] for r in manifest.get("feature_blocks", {}).values()]
        for name in names:
            path = (manifest_path.parent / name).resolve()
            if not path.is_relative_to(manifest_path.parent):
                raise ValueError("Region feature path escapes its profile directory")
            if manifest.get("files", {}).get(name) != sha256_file(path):
                raise ValueError(f"Region profile file SHA256 mismatch or missing: {name}")
        profile = load_profile(manifest_path.parent, list(manifest.get("feature_blocks", {})))
        if profile.unit != "tissue_region" or not profile.cells.sample_id.eq(summary["sample_id"]).all():
            raise ValueError("Region profiles have a foreign sample or observation unit")
        self.cells, self.blocks, self.manifest = profile.cells.copy(), profile.blocks, manifest
        self.cells["cell_uid"], self.cells["cell_id"] = self.cells.region_uid, self.cells.region_id
        self.uid_index = dict(zip(self.cells.region_uid, range(len(self.cells))))
        try:
            self.ids = self.cells.region_id.astype(np.int64).to_numpy()
        except (ValueError, OverflowError) as exc:
            raise ValueError("Region IDs must be positive raster integers") from exc
        if (self.ids <= 0).any() or len(set(self.ids)) != len(self.ids):
            raise ValueError("Region IDs must be unique positive raster integers")
        self.id_index = dict(zip(map(int, self.ids), range(len(self.ids))))
        self.grid = pd.read_csv(self.root / "grid_subdomains.csv", dtype={"sample_id": str})
        grid_columns = {"label", "sample_id", "parent_domain_id", "subdomain_id", "status_code", "x_um", "y_um"}
        if (not grid_columns <= set(self.grid) or self.grid.label.duplicated().any() or
            not self.grid.sample_id.eq(summary["sample_id"]).all()):
            raise ValueError("Hierarchy grid identities are missing, duplicate or foreign")
        by_grid = self.grid.set_index("label")
        for row in self.cells.itertuples():
            if int(row.representative_grid_id) not in by_grid.index:
                raise ValueError("Region representative is absent from the hierarchy grid")
            representative = by_grid.loc[int(row.representative_grid_id)]
            if (representative.status_code != 1 or representative.parent_domain_id != int(row.parent_domain_id) or
                representative.subdomain_id != int(row.subdomain_id) or not np.allclose(
                [representative.x_um, representative.y_um], [float(row.representative_x_um), float(row.representative_y_um)], rtol=1e-7, atol=1e-7)):
                raise ValueError("Region representative grid identity or coordinate mismatch")
        self.xy = self.cells[["representative_x_um", "representative_y_um"]].to_numpy(float) / parent.cal["mpp"] - parent.cal["origin_px"]
        if not np.isfinite(self.xy).all() or np.any(self.xy < 0) or np.any(self.xy >= [parent.cal["width"], parent.cal["height"]]):
            raise ValueError("Region representative coordinates are outside the native crop")
        self.mask, self.status, self.parent_mask = [self.root / name for name in required_files[:3]]
        self.stats, self.stats_lock, self.flags = {}, threading.Lock(), {}
        self.spatial_stats, self.membership, self.status_counts = {}, np.zeros(len(parent.cells), np.int64), Counter()
        self._scan_rasters()
        self.member_indices = {int(value): np.flatnonzero(self.membership == value) for value in self.ids}
        self.member_weights = {value: np.ones(len(ids)) for value, ids in self.member_indices.items()}
        self.membership_method = 'centroid inside region mask; crossing nuclei are not fractionally assigned'
        self.marker_mean_weighting = 'equal centroid-member cells'
        if parent.hierarchy_overlaps is not None:
            nuclear = parent.hierarchy_overlaps
            nuclear = nuclear[(nuclear.compartment == 'nucleus') & (nuclear.region_id > 0)]
            grouped = nuclear.groupby(['region_id', 'cell_uid'], sort=True).overlap_fraction.sum()
            for value in self.ids:
                rows = grouped.loc[int(value)] if int(value) in grouped.index.get_level_values(0) else pd.Series(dtype=float)
                self.member_indices[int(value)] = np.array([parent.uid_index[uid] for uid in rows.index], dtype=int)
                self.member_weights[int(value)] = rows.to_numpy(float)
            self.membership_method = 'any exact nuclear pixel overlap; cell may belong to multiple regions; marker means weighted by whole-nucleus overlap fraction'
            self.marker_mean_weighting = 'whole-nucleus overlap fraction'
        self.markers = parent.manifest.get("biological_marker_features", [])
        # Retrieval may optionally filter actual region means of predicted cell
        # markers. Missing constituent values are never turned into zero.
        self.manifest = {**manifest, "biological_marker_features": self.markers}
        for marker in self.markers:
            values = pd.to_numeric(parent.cells[marker], errors="coerce").to_numpy(float)
            self.cells[marker] = [float(np.average(values[ids][np.isfinite(values[ids])], weights=self.member_weights[value][np.isfinite(values[ids])]))
                if np.isfinite(values[ids]).any() else np.nan for value, ids in self.member_indices.items()]
        self.annotations = {}
        if annotations:
            payload = json.loads(Path(annotations).read_text())
            if not isinstance(payload, dict) or not isinstance(payload.get("annotations"), list):
                raise ValueError("Region notes require an annotations list with region_uid keys")
            for note in payload["annotations"]:
                uid = note.get("region_uid") if isinstance(note, dict) else None
                if uid not in self.uid_index or uid in self.annotations:
                    raise ValueError("Region note has a foreign or duplicate region UID")
                self.annotations[uid] = note

    def _scan_rasters(self):
        parent = self.parent
        labels = self.ids
        with ExitStack() as stack:
            paths = [self.mask, self.status, self.parent_mask] + ([self.parent_uncertainty] if self.parent_uncertainty else [])
            readers = [stack.enter_context(RasterReader(path)) for path in paths]
            regions, status, domains = readers[:3]
            for reader in readers:
                if (reader.width, reader.height) != (parent.cal["width"], parent.cal["height"]) or len(reader.reader.shape) != 2 or reader.dtype.kind not in "ui":
                    raise ValueError("Hierarchy rasters must be integer native-crop YX")
            for y in range(0, regions.height, 512):
                for x in range(0, regions.width, 512):
                    bounds = (x, y, min(x+512, regions.width), min(y+512, regions.height))
                    block, uncertainty, parents = (reader.window(*bounds) for reader in (regions, status, domains))
                    if np.any((block > 0) != (uncertainty == 1)) or np.any((block > 0) & (parents == 0)):
                        raise ValueError("Region/status/parent raster assignments disagree")
                    if self.parent_uncertainty and np.any((block > 0) & (readers[3].window(*bounds) != 0)):
                        raise ValueError("Accepted region overlaps unresolved/inferred parent uncertainty")
                    for code, count in zip(*np.unique(uncertainty, return_counts=True)):
                        if str(int(code)) not in self.summary.get("status_codes", {}):
                            raise ValueError("Hierarchy raster contains an undeclared status code")
                        self.status_counts[int(code)] += int(count)
                    candidates = np.flatnonzero((parent.xy[:, 0] >= x) & (parent.xy[:, 0] < bounds[2]) &
                                               (parent.xy[:, 1] >= y) & (parent.xy[:, 1] < bounds[3]))
                    xy = np.floor(parent.xy[candidates]).astype(int)
                    self.membership[candidates] = block[xy[:, 1]-y, xy[:, 0]-x]
                    for value in np.unique(block):
                        if value == 0:
                            continue
                        if value not in self.id_index:
                            raise ValueError("Region raster has an ID absent from its profile")
                        row = self.cells.iloc[self.id_index[value]]
                        yy, xx = np.nonzero(block == value)
                        if np.any(parents[yy, xx] != int(row.parent_domain_id)):
                            raise ValueError("Region crosses its declared parent domain")
                        record = self.spatial_stats.setdefault(int(value), {"count": 0, "sum_x": 0., "sum_y": 0., "x0": regions.width, "y0": regions.height, "x1": 0, "y1": 0})
                        record["count"] += len(xx); record["sum_x"] += float(xx.sum()) + (x+.5)*len(xx); record["sum_y"] += float(yy.sum()) + (y+.5)*len(yy)
                        for key, v in (("x0", int(xx.min())+x), ("y0", int(yy.min())+y)):
                            record[key] = min(record[key], v)
                        for key, v in (("x1", int(xx.max())+x+1), ("y1", int(yy.max())+y+1)):
                            record[key] = max(record[key], v)
            if set(self.spatial_stats) != set(labels):
                raise ValueError("A region profile is absent from its raster")
            for i, value in enumerate(labels):
                record, row = self.spatial_stats[int(value)], self.cells.iloc[i]
                expected = [record["count"] * parent.cal["mpp"]**2,
                            (record["sum_x"]/record["count"] + parent.cal["origin_px"][0]) * parent.cal["mpp"],
                            (record["sum_y"]/record["count"] + parent.cal["origin_px"][1]) * parent.cal["mpp"]]
                if not np.allclose(np.array([row.area_um2, row.x_um, row.y_um], float), expected, rtol=1e-6, atol=1e-6):
                    raise ValueError("Region area/centroid disagrees with its native raster")

    def row_index(self, uid):
        if uid not in self.uid_index:
            raise KeyError("Unknown region UID")
        return self.uid_index[uid]

    block_stats = CellInspector.block_stats

    def metadata(self):
        return {"status": "verified_sha256", "region_count": len(self.cells), "hierarchy_id": self.summary["hierarchy_id"],
            "first_region_uid": self.cells.region_uid.iloc[0] if len(self.cells) else None,
            "feature_blocks": [{"name": name, "dimension": int(values.shape[1])} for name, values in self.blocks.items()],
            "marker_columns": self.markers, "cells_outside_accepted_regions": int((self.membership == 0).sum()),
            "cell_membership_method": self.membership_method,
            "cells_without_nuclear_region_overlap": (len(self.parent.cells) - len(set(int(i) for ids in self.member_indices.values() for i in ids))) if self.parent.hierarchy_overlaps is not None else None,
            "parent_uncertainty_status": "hash_verified_and_excluded_from_regions" if self.parent_uncertainty else "not_provided",
            "status_pixel_counts": {self.summary["status_codes"][str(code)]: count for code, count in self.status_counts.items()},
            "interpretation": "Unsupervised connected regions; abstention/stability are descriptive, not calibrated biological uncertainty."}

    def search(self, term="", limit=30):
        if not 1 <= limit <= 100:
            raise ValueError("Region search limit must be in [1,100]")
        indices = np.flatnonzero(self.cells.region_uid.str.contains(term.strip(), regex=False).to_numpy() |
                                 self.cells.region_id.str.contains(term.strip(), regex=False).to_numpy())[:limit]
        return [{"region_uid": self.cells.region_uid.iloc[i], "region_id": self.cells.region_id.iloc[i],
                 "parent_domain_id": int(self.cells.parent_domain_id.iloc[i]), "area_um2": float(self.cells.area_um2.iloc[i])} for i in indices]

    def pick(self, x, y):
        if not np.isfinite([x, y]).all() or not 0 <= x < self.parent.cal["width"] or not 0 <= y < self.parent.cal["height"]:
            raise ValueError("Region pick coordinates outside the crop")
        with RasterReader(self.mask) as reader, RasterReader(self.status) as status:
            value = int(reader.window(int(x), int(y), int(x)+1, int(y)+1)[0, 0])
            code = int(status.window(int(x), int(y), int(x)+1, int(y)+1)[0, 0])
        if value == 0:
            with RasterReader(self.parent_mask) as reader:
                parent_id = int(reader.window(int(x), int(y), int(x)+1, int(y)+1)[0, 0])
            raw_uncertainty = None
            if self.parent_uncertainty:
                with RasterReader(self.parent_uncertainty) as reader:
                    raw_uncertainty = int(reader.window(int(x), int(y), int(x)+1, int(y)+1)[0, 0])
            if parent_id > 0:
                status_name = "unresolved_parent_tissue"
                reason = self.summary["status_codes"][str(code)]
            elif raw_uncertainty is None:
                status_name = "no_parent_assignment_uncertainty_unavailable"
                reason = "Parent label 0; original uncertainty was not supplied, so background is not established"
            elif raw_uncertainty > 0:
                status_name = "unassigned_parent_uncertain_tissue"
                reason = f"Parent label 0 with preserved upstream uncertainty code {raw_uncertainty}; unresolved assignment, not background"
            else:
                status_name = "parent_map_background"
                reason = "Parent label 0 and upstream uncertainty code 0; parent-map background, not independent proof that tissue is absent"
            return {"status": status_name, "reason": reason, "parent_domain_id": parent_id,
                    "hierarchy_status_code": code, "parent_uncertainty_code": raw_uncertainty,
                    "parent_uncertainty_available": raw_uncertainty is not None}
        return {"status": "selected", "region_uid": self.cells.region_uid.iloc[self.id_index[value]], "method": "exact_region_raster"}

    def bounds(self, uid, size=384, x=None, y=None):
        if not 64 <= size <= 1024:
            raise ValueError("Native region window size must be in [64,1024] pixels")
        index = self.row_index(uid)
        if (x is None) != (y is None):
            raise ValueError("Region window centre requires both x and y")
        x, y = self.xy[index] if x is None else (x, y)
        if not np.isfinite([x, y]).all() or not 0 <= x < self.parent.cal["width"] or not 0 <= y < self.parent.cal["height"]:
            raise ValueError("Region window centre outside the crop")
        x0, y0 = max(0, int(round(x))-size//2), max(0, int(round(y))-size//2)
        return x0, y0, min(self.parent.cal["width"], x0+size), min(self.parent.cal["height"], y0+size)

    def detail(self, uid, size=384, x=None, y=None):
        index = self.row_index(uid)
        row = self.cells.iloc[index]
        ids = self.member_indices[int(self.ids[index])]
        weights = self.member_weights[int(self.ids[index])]
        weights_by_cell = dict(zip(ids, weights))
        distance = ((self.parent.xy[ids] - self.xy[index])**2).sum(axis=1)
        ordered = ids[np.lexsort((self.parent.cells.cell_uid.iloc[ids].to_numpy(), distance))[:12]]
        representatives = [{"cell_uid": self.parent.cells.cell_uid.iloc[i], "cell_id": self.parent.cells.cell_id.iloc[i],
                            "nuclear_overlap_fraction": float(weights_by_cell[i]) if self.parent.hierarchy_overlaps is not None else None,
                            "distance_to_representative_um": float(np.linalg.norm(self.parent.xy[i]-self.xy[index]) * self.parent.cal["mpp"])} for i in ordered]
        distributions = {}
        for name in self.markers:
            values = pd.to_numeric(self.parent.cells[name].iloc[ids], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            distributions[name] = {"count": len(finite), "missing": int(len(values)-len(finite)),
                "mean": float(np.average(finite, weights=weights[np.isfinite(values)])) if len(finite) else None,
                "mean_weighting": self.marker_mean_weighting,
                "quantile_weighting": 'unweighted among overlapping cells' if self.parent.hierarchy_overlaps is not None else 'equal centroid-member cells',
                "quantiles_p05_p50_p95": np.quantile(finite, [.05, .5, .95]).tolist() if len(finite) else None}
        blocks = {name: {"dimension": int(values.shape[1]), "available": bool(np.isfinite(values[index]).all()),
            "missing_dimensions": int((~np.isfinite(values[index])).sum()), "feature_definition": self.manifest["feature_blocks"][name]["feature_definition"],
            "within_region_feature_distribution": None, "distribution_status": "not_exported; stored vector is the area-weighted grid mean"} for name, values in self.blocks.items()}
        stats = self.spatial_stats[int(self.ids[index])]
        grid = self.grid[(self.grid.subdomain_id == int(row.subdomain_id)) & (self.grid.status_code == 1)]
        stability = {}
        for name in ("seed_stability", "scale_agreement", "centroid_margin", "affinity_margin", "own_affinity_fraction", "graph_degree"):
            values = pd.to_numeric(grid[name], errors="coerce").to_numpy(float) if name in grid else np.array([])
            finite = values[np.isfinite(values)]
            stability[name] = {"count": len(finite), "missing": int(len(grid)-len(finite)),
                               "quantiles_p05_p50_p95": np.quantile(finite, [.05, .5, .95]).tolist() if len(finite) else None}
        return clean_json({"region_uid": uid, "region_id": row.region_id, "sample_id": row.sample_id,
            "parent_domain_id": row.parent_domain_id, "subdomain_id": row.subdomain_id, "area_um2": row.area_um2,
            "coordinates_um": [row.x_um, row.y_um], "crop_xy": self.xy[index].tolist(),
            "window": dict(zip(("x0", "y0", "x1", "y1"), self.bounds(uid, size, x, y))),
            "region_bounds_px": {key: stats[key] for key in ("x0", "y0", "x1", "y1")},
            "representative_grid_id": row.representative_grid_id, "representative_cell_selection": "nearest member-cell centroids to the real representative grid observation; not phenotype prototypes",
            "cell_membership": self.membership_method, "fractional_nuclear_cell_count": float(weights.sum()) if self.parent.hierarchy_overlaps is not None else None,
            "cell_count": len(ids), "representative_cells": representatives, "predicted_marker_distributions": distributions,
            "feature_blocks": blocks, "uncertainty": {"status": "accepted_subdomain", "calibrated_probability": None,
                "parent_uncertainty_status": "hash_verified_and_excluded" if self.parent_uncertainty else "not_provided",
                "boundary_uncertainty": None, "scope": "Region rasters contain accepted pixels only; unresolved parent tissue is separate, never silently assigned"},
            "subdomain_grid_stability": stability,
            "stability_scope": "All accepted grid observations of this subdomain; may include other connected regions of the same subdomain. Not calibrated probability.",
            "expert_annotation": self.annotations.get(uid), "expert_annotation_used_for_inference": False})

    def png(self, uid, size=384, overlay=False, x=None, y=None):
        bounds = self.bounds(uid, size, x, y)
        if overlay:
            with RasterReader(self.mask) as reader:
                mask = reader.window(*bounds) == self.ids[self.row_index(uid)]
            pixels = np.zeros((*mask.shape, 4), np.uint8)
            pixels[mask] = [25, 165, 186, 45]
            pixels[mask & ~binary_erosion(mask, border_value=1)] = [10, 128, 153, 255]
        else:
            with RasterReader(self.parent.image) as reader:
                pixels = reader.window(*bounds)
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return output.getvalue()

    def similar(self, *args, **kwargs):
        result = CellInspector.similar(self, *args, **kwargs)
        for row in result["results"]:
            row["region_uid"], row["region_id"] = row.pop("cell_uid"), row.pop("cell_id")
            row.pop("review_reasons")
        if result["status"] == "no_matching_cells":
            result["status"] = "no_matching_regions"
        result["observation_unit"] = "tissue_region"
        result["cell_membership"] = self.membership_method
        result["marker_mean_weighting"] = self.marker_mean_weighting
        result["marker_semantics"] = (
            "whole-nucleus overlap-fraction-weighted mean of finite predicted cell-marker values for nuclear-overlap member cells"
            if self.marker_mean_weighting == 'whole-nucleus overlap fraction' else
            "mean of finite predicted cell-marker values for centroid-member cells"
        ) + "; missing values remain unknown and are never zero-filled; not measured markers"
        return result


def make_server(inspector, port=8765):
    html = HTML_PATH.read_bytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(clean_json(body), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if len(self.path) > 8192:
                self.respond(414, {"error": "Request too long"})
                return
            allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed_hosts:
                self.respond(403, {"error": "Only localhost hosts are accepted"})
                return
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + value for value in allowed_hosts}:
                self.respond(403, {"error": "Cross-origin requests are disabled"})
                return
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query, keep_blank_values=True)
            if any(len(values) != 1 for values in params.values()):
                self.respond(400, {"error": "Duplicate query parameters"})
                return
            args = {key: values[0] for key, values in params.items()}
            routes = {"/": set(), "/api/metadata": set(), "/api/map": set(),
                "/api/search": {"q", "limit"}, "/api/review": {"offset", "limit", "reason"},
                "/api/pick": {"x", "y", "radius_um"}, "/api/cell": {"uid", "size"},
                "/api/window.png": {"uid", "size"}, "/api/overlay.png": {"uid", "size"},
                "/api/similar": {"uid", "groups", "weights", "k", "marker", "min_difference"},
                "/api/regions": {"q", "limit"}, "/api/region-pick": {"x", "y"},
                "/api/region": {"uid", "size", "x", "y"},
                "/api/region-window.png": {"uid", "size", "x", "y"},
                "/api/region-overlay.png": {"uid", "size", "x", "y"},
                "/api/rejected-candidates": {"offset", "limit", "source"},
                "/api/rejected-candidate": {"uid", "size"},
                "/api/rejected-candidate-window.png": {"uid", "size"},
                "/api/rejected-candidate-overlay.png": {"uid", "size"},
                "/api/region-similar": {"uid", "groups", "weights", "k", "marker", "min_difference"}}
            if parsed.path not in routes:
                self.respond(404, {"error": "Unknown endpoint"})
                return
            if not set(args) <= routes[parsed.path]:
                self.respond(400, {"error": "Unsupported query parameter; paths are not accepted"})
                return
            try:
                region_route = parsed.path.startswith("/api/region")
                if region_route and inspector.regions is None:
                    raise KeyError("No verified hierarchy was supplied")
                if parsed.path.startswith("/api/rejected-candidate") and inspector.rejected_candidates is None:
                    raise KeyError("No verified rejected-candidate layer was supplied")
                if parsed.path == "/":
                    self.respond(200, html, "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/metadata":
                    result = inspector.metadata()
                elif parsed.path == "/api/map":
                    result = inspector.cell_map()
                elif parsed.path == "/api/search":
                    result = inspector.search(args.get("q", ""), int(args.get("limit", 30)))
                elif parsed.path == "/api/review":
                    result = inspector.queue(int(args.get("offset", 0)), int(args.get("limit", 40)), args.get("reason", ""))
                elif parsed.path == "/api/pick":
                    result = inspector.pick(float(args["x"]), float(args["y"]), float(args.get("radius_um", 30)))
                elif parsed.path in ("/api/window.png", "/api/overlay.png"):
                    self.respond(200, inspector.png(args["uid"], int(args.get("size", 384)), parsed.path.endswith("overlay.png")), "image/png")
                    return
                elif parsed.path == "/api/cell":
                    result = inspector.detail(args["uid"], int(args.get("size", 384)))
                elif parsed.path == "/api/rejected-candidates":
                    result = inspector.rejected_candidates.queue(int(args.get("offset", 0)),
                        int(args.get("limit", 25)), args.get("source", ""))
                elif parsed.path == "/api/rejected-candidate":
                    result = inspector.rejected_candidates.detail(args["uid"], int(args.get("size", 384)))
                elif parsed.path in ("/api/rejected-candidate-window.png", "/api/rejected-candidate-overlay.png"):
                    self.respond(200, inspector.rejected_candidates.png(args["uid"], int(args.get("size", 384)),
                        parsed.path.endswith("overlay.png")), "image/png")
                    return
                elif parsed.path == "/api/regions":
                    result = inspector.regions.search(args.get("q", ""), int(args.get("limit", 30)))
                elif parsed.path == "/api/region-pick":
                    result = inspector.regions.pick(float(args["x"]), float(args["y"]))
                elif parsed.path in ("/api/region", "/api/region-window.png", "/api/region-overlay.png"):
                    window = {"size": int(args.get("size", 384)),
                              "x": float(args["x"]) if "x" in args else None,
                              "y": float(args["y"]) if "y" in args else None}
                    if parsed.path == "/api/region":
                        result = inspector.regions.detail(args["uid"], **window)
                    else:
                        self.respond(200, inspector.regions.png(args["uid"], overlay=parsed.path.endswith("overlay.png"), **window), "image/png")
                        return
                else:
                    target = inspector.regions if region_route else inspector
                    result = target.similar(args["uid"], [g for g in args.get("groups", "").split(",") if g],
                        [float(w) for w in args["weights"].split(",")] if args.get("weights") else None,
                        int(args.get("k", 12)), args.get("marker") or None,
                        float(args["min_difference"]) if args.get("min_difference") else None)
                self.respond(200, result)
            except KeyError as exc:
                self.respond(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except Exception:
                self.respond(500, {"error": "Inspector could not read the requested data; check the supplied dataset"})

        def do_POST(self):
            self.respond(405, {"error": "Read-only inspector; annotation writes are disabled"})

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def atlas_launch_inputs(atlas_manifest, sample_id):
    """Resolve a user-selected portable specimen atlas, never an HTTP path."""
    path = Path(atlas_manifest).absolute()
    atlas = json.loads(path.read_text())
    specimens = [item for item in atlas.get("specimens", []) if item.get("sample_id") == sample_id]
    if len(specimens) != 1:
        raise ValueError("Select one exact sample ID from the atlas manifest")
    launch = specimens[0].get("cell_inspector", {})
    if launch.get("status") != "available" or launch.get("path_base") != "specimen_atlas_json_directory":
        raise ValueError("The atlas has no complete portable inspector launch for this sample")
    sources = launch.get("datasets", {})
    required = {"profile_dir", "image", "labels", "shift"}
    if not required <= sources.keys() or not sources.keys() <= required | {"resolution_json", "hierarchy_dir", "cohort_niches"}:
        raise ValueError("Inspector launch must contain only the explicit supported source paths")
    resolved = {}
    for name, value in sources.items():
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError("Portable atlas source paths must be relative")
        source = Path(os.path.abspath(path.parent / value))
        if not source.is_relative_to(path.parent.parent):
            raise ValueError("Portable inspector path escapes the results tree")
        resolved[name] = source
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile-dir", "image", "labels", "shift"):
        parser.add_argument("--" + name)
    parser.add_argument("--resolution-json")
    parser.add_argument("--expert-annotations")
    parser.add_argument("--hierarchy-dir", help="Verified native-crop hierarchy outputs, including region_profiles/")
    parser.add_argument("--cohort-niches", help="Explicit source-bound shared cohort niche bundle; separate from per-slide niches and reference assignments")
    parser.add_argument("--region-annotations", help="Display-only JSON annotations keyed by region_uid")
    parser.add_argument("--rejected-candidates", help="Source-bound rejected_detector_candidates.geojson; separate review observations, never canonical cells")
    for source in ("stardist-objects", "hovernet-cells", "cellvit-cells", "alignment-csv"):
        parser.add_argument("--candidate-" + source, help="Optional explicit original source for candidate-payload verification; supply all four candidate source paths")
    parser.add_argument("--atlas-manifest", help="Portable specimen_atlas.json; use with --sample-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and print metadata without starting a server")
    parser.add_argument("--port", type=int, default=8765)
    args = vars(parser.parse_args())
    candidate_sources = {name: args.pop("candidate_" + name) for name in
        ("stardist_objects", "hovernet_cells", "cellvit_cells", "alignment_csv")}
    args["candidate_sources"] = candidate_sources if any(candidate_sources.values()) else None
    port = args.pop("port")
    atlas, sample_id, check_only = args.pop("atlas_manifest"), args.pop("sample_id"), args.pop("check_only")
    if atlas:
        if not sample_id or any(args.get(name) for name in ("profile_dir", "image", "labels", "shift", "resolution_json", "hierarchy_dir")):
            parser.error("--atlas-manifest requires --sample-id and cannot be mixed with explicit dataset paths")
        selected = atlas_launch_inputs(atlas, sample_id)
        explicit_cohort = args.get("cohort_niches")
        if explicit_cohort and selected.get("cohort_niches") and (
                cohort_directory(explicit_cohort) != cohort_directory(selected["cohort_niches"])):
            parser.error("Explicit --cohort-niches conflicts with the atlas-declared cohort bundle")
        args.update(selected)
        if explicit_cohort:
            args["cohort_niches"] = explicit_cohort
    elif sample_id or any(not args.get(name) for name in ("profile_dir", "image", "labels", "shift")):
        parser.error("Supply --profile-dir, --image, --labels and --shift, or --atlas-manifest with --sample-id")
    inspector = CellInspector(**args)
    if check_only:
        print(json.dumps(clean_json(inspector.metadata()), allow_nan=False))
        return
    server = make_server(inspector, port)
    print(f"Read-only cell inspector: http://127.0.0.1:{server.server_port} ({len(inspector.cells)} cells)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
