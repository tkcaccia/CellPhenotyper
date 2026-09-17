#!/usr/bin/env python3
"""Export rejected detector observations for review, never as canonical cells."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import io
import re
from collections import Counter
from pathlib import Path

import numpy as np

from cell_profile_io import calibration, sha256_file


SCHEMA = "cellphenotyper.rejected_detector_candidates/1"
SOURCES = ("stardist", "hovernet", "cellvitpp")
SOURCE_KEYS = ("stardist_objects", "hovernet_cells", "cellvit_cells")
VERIFICATION_KEYS = (*SOURCE_KEYS, "alignment_csv")
INTERPRETATION = "Excluded by fusion policy; not an established false detection or a canonical cell"


def candidate_calibration(shift, resolution=None):
    data = json.loads(Path(shift).read_text())
    size = data.get("crop_size", {})
    if any(type(size.get(key)) not in (int, float) or not np.isfinite(size[key])
           or size[key] <= 0 or int(size[key]) != size[key] for key in ("width", "height")):
        raise ValueError("Candidate crop dimensions must be positive integer pixels")
    box = data.get("crop_bbox_xyxy")
    if box is not None:
        if not isinstance(box, dict) or not {"x0", "y0", "x1", "y1"} <= set(box):
            raise ValueError("Candidate crop bounding box is incomplete")
        if any(type(box[key]) not in (int, float) or not np.isfinite(box[key]) or int(box[key]) != box[key] for key in box):
            raise ValueError("Candidate crop bounding box must use integer pixels")
        offset = data.get("offset_crop_to_original", {"dx": box["x0"], "dy": box["y0"]})
        if (box["x1"]-box["x0"] != size["width"] or box["y1"]-box["y0"] != size["height"]
                or (offset.get("dx"), offset.get("dy")) != (box["x0"], box["y0"])):
            raise ValueError("Candidate crop bounding box, offset and dimensions disagree")
    if resolution or data.get("source_mpp", data.get("microns_per_pixel")) is not None:
        return calibration(shift, resolution)
    # Legacy crop shifts can bind pixel geometry without any physical scale.
    # Validate geometry through the shared reader; the unit value is discarded,
    # never exported or used for physical coordinates or an inference setting.
    result = calibration({**data, "source_mpp": 1.0})
    result["mpp"] = None
    result["calibration_source"] = "physical_scale_unavailable"
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


def candidate_geometry(cell):
    """Keep source vertices without repair; never invent a StarDist contour."""
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    point = {"type": "Point", "coordinates": [float(cell.x), float(cell.y)]}
    if not cell.contour:
        return point, "centroid_only", "source_contour_not_available"
    try:
        ring = np.asarray(cell.contour, dtype=float)
    except (ValueError, TypeError):
        return point, "centroid_only", "source_contour_nonfinite_or_malformed"
    if ring.ndim != 2 or ring.shape[1] != 2 or not np.isfinite(ring).all():
        return point, "centroid_only", "source_contour_nonfinite_or_malformed"
    if len(ring) < 3:
        return point, "centroid_only", "source_contour_has_fewer_than_three_vertices"
    vertices = ring.tolist()
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])
    polygon = Polygon(vertices)
    if polygon.is_valid and polygon.area > 0:
        return {"type": "Polygon", "coordinates": [vertices]}, "source_polygon", None
    return ({"type": "LineString", "coordinates": vertices}, "invalid_source_outline",
            explain_validity(polygon))


def load_detector_inputs(inputs):
    from build_cell_consensus import load_stardist, load_cells
    cells = load_stardist(Path(inputs["stardist_objects"]))
    cells += load_cells(Path(inputs["hovernet_cells"]), "hovernet")
    cells += load_cells(Path(inputs["cellvit_cells"]), "cellvitpp")
    return cells


def read_alignment(cells, path):
    """Read the authoritative source table without guessing missing decisions."""
    decisions, accepted = {}, set()
    cell_index = {(cell.source, cell.source_id): cell for cell in cells}
    if len(cell_index) != len(cells):
        raise ValueError("Detector predictions require unique source IDs")
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "source_id", "x", "y", "accepted", "consensus_label", "decision"}
        fields = reader.fieldnames or []
        if not required <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError("Alignment requires unique columns and explicit source IDs, coordinates and decisions")
        for row in reader:
            key = (row["source"], row["source_id"])
            if key in decisions:
                raise ValueError("Duplicate detector ID in alignment table")
            flag = row["accepted"].lower()
            if flag not in ("true", "false") or bool(row["consensus_label"]) != (flag == "true"):
                raise ValueError("Alignment accepted flag and canonical assignment disagree")
            if not row["decision"] or (row["decision"] == "accepted_instance_fusion") != (flag == "true"):
                raise ValueError("Alignment accepted fusion decision contradicts its assignment")
            if flag == "true":
                accepted.add(key)
            decisions[key] = {k: v for k, v in row.items() if k not in {"source", "source_id", "x", "y", "accepted", "consensus_label"}}
            cell = cell_index.get(key)
            if cell is None or not np.allclose([float(row["x"]), float(row["y"])], [cell.x, cell.y], rtol=0, atol=1e-6):
                raise ValueError("Alignment centroid does not match its detector observation")
    if set(decisions) != set(cell_index):
        raise ValueError("Alignment must cover exactly all detector predictions")
    return decisions, accepted


def export_candidates(cells, decisions, accepted, inputs, output):
    """Decisions and accepted keys must partition the exact input observations."""
    from cell_profile_io import RasterReader
    from build_cell_consensus import DETECTOR_SCOPES
    keys = [(cell.source, cell.source_id) for cell in cells]
    if len(set(keys)) != len(keys) or any(source not in SOURCES or not uid for source, uid in keys):
        raise ValueError("Detector predictions require unique nonempty source IDs")
    if set(keys) != set(decisions) or not set(accepted) <= set(keys):
        raise ValueError("Candidate decisions must cover exactly the detector predictions")
    required = {"image", "labels", "shift", "alignment_csv", *SOURCE_KEYS}
    if not required <= set(inputs) or not set(inputs) <= required | {"resolution_json"}:
        raise ValueError("Candidate source binding requires image, labels, shift and all detector inputs")
    alignment_decisions, alignment_accepted = read_alignment(cells, inputs["alignment_csv"])
    if set(accepted) != alignment_accepted or any(decisions[key].get("decision") != alignment_decisions[key]["decision"] for key in keys):
        raise ValueError("Candidate accepted fusion decisions differ from the source alignment table")
    # All review evidence comes from the hash-bound alignment CSV, including
    # integrated exports; it can be exactly rechecked by a later inspector.
    decisions = alignment_decisions
    output = Path(output)
    if output.exists() or output.resolve() in {Path(p).resolve() for p in inputs.values()}:
        raise ValueError("Candidate export requires a fresh output path; source files are preserved")
    cal = candidate_calibration(inputs["shift"], inputs.get("resolution_json"))
    for key in ("image", "labels"):
        with RasterReader(inputs[key]) as reader:
            if (reader.width, reader.height) != (cal["width"], cal["height"]):
                raise ValueError("Candidate source rasters must match the calibrated crop dimensions")
            if key == "labels" and (len(reader.reader.shape) != 2 or reader.dtype.kind not in "ui"):
                raise ValueError("Candidate canonical-label binding requires an integer YX raster")
    hashes = {key: sha256_file(path) for key, path in inputs.items()}
    namespace = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    features = []
    for cell in sorted(cells, key=lambda c: (c.source, c.source_id)):
        key = (cell.source, cell.source_id)
        decision = decisions[key]
        if not isinstance(decision, dict) or not isinstance(decision.get("decision"), str) or not decision["decision"]:
            raise ValueError("Each detector observation needs its explicit fusion decision")
        if key in accepted:
            if decision["decision"] != "accepted_instance_fusion":
                raise ValueError("Accepted detector assignment contradicts its fusion decision")
            continue
        if decision["decision"] == "accepted_instance_fusion":
            raise ValueError("Rejected detector observation has an accepted fusion decision")
        if not np.isfinite([cell.x, cell.y]).all() or not (0 <= cell.x < cal["width"] and 0 <= cell.y < cal["height"]):
            raise ValueError("Rejected candidate centroid is nonfinite or outside the native crop")
        geometry, status, reason = candidate_geometry(cell)
        identity = json.dumps([namespace, *key], separators=(",", ":"))
        uid = "detector_candidate:" + hashlib.sha256(identity.encode()).hexdigest()
        properties = {"candidate_uid": uid, "source": cell.source, "source_id": cell.source_id,
            "detector_scope": DETECTOR_SCOPES[cell.source],
            "observation_unit": "rejected_detector_prediction", "canonical_assignment": None,
            "decision": decision["decision"], "crop_xy": [cell.x, cell.y],
            "coordinates_um": ((np.array([cell.x, cell.y]) + cal["origin_px"]) * cal["mpp"]).tolist() if cal["mpp"] is not None else None,
            "geometry_status": status, "geometry_issue": reason,
            "fusion_evidence": {k: v for k, v in decision.items() if k != "decision"},
            "detector_phenotype_evidence": {"type_id": cell.type_id, "type": cell.cell_type,
                "probability": cell.probability},
            "interpretation": INTERPRETATION}
        features.append({"type": "Feature", "id": uid, "properties": properties, "geometry": geometry})
    metadata = {"schema": SCHEMA, "coordinate_frame": "analysis_crop_pixels", "read_only": True,
        "input_sha256": hashes, "identity_namespace": namespace,
        "crop_size_px": [cal["width"], cal["height"]], "crop_origin_px": cal["origin_px"].tolist(),
        "crop_origin_um": (cal["origin_px"] * cal["mpp"]).tolist() if cal["mpp"] is not None else None,
        "source_mpp": cal["mpp"], "calibration_source": cal["calibration_source"], "input_prediction_count": len(cells),
        "accepted_prediction_count": len(accepted), "candidate_count": len(features),
        "decision_counts": dict(Counter(f["properties"]["decision"] for f in features)),
        "geometry_counts": dict(Counter(f["properties"]["geometry_status"] for f in features)),
        "canonical_population_changed": False,
        "fusion_evidence_scope": "Original fusion scores/distances retained, not recomputed using review calibration; this export does not establish the original fusion MPP",
        "claim": "Review observations only; detector agreement is not independent accuracy"}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of previous review exports.
    with output.open("x") as handle:
        json.dump(json_safe({"type": "FeatureCollection", "metadata": metadata, "features": features}), handle, allow_nan=False)
    return metadata


class CandidateReview:
    """A separate source-bound observation layer; no canonical mutations."""
    def __init__(self, parent, path, shift, resolution=None, source_inputs=None):
        from shapely.geometry import shape
        self.parent = parent
        path = Path(path)
        if path.stat().st_size > 512 * 1024**2:
            raise ValueError("Candidate review GeoJSON exceeds the 512 MiB startup limit")
        with path.open("rb") as handle:
            review_bytes = handle.read(512 * 1024**2 + 1)
        if len(review_bytes) > 512 * 1024**2:
            raise ValueError("Candidate review GeoJSON exceeds the 512 MiB startup limit")
        payload = json.loads(review_bytes)
        review_digest = hashlib.sha256(review_bytes).hexdigest()
        del review_bytes
        meta = payload.get("metadata", {})
        features = payload.get("features")
        if (payload.get("type") != "FeatureCollection" or meta.get("schema") != SCHEMA
                or meta.get("coordinate_frame") != "analysis_crop_pixels" or not isinstance(features, list)
                or meta.get("read_only") is not True or meta.get("canonical_population_changed") is not False):
            raise ValueError("Unsupported rejected-candidate review schema or coordinate frame")
        hashes = meta.get("input_sha256", {})
        required = {"image", "labels", "shift", "alignment_csv", *SOURCE_KEYS}
        if not required <= set(hashes) or not set(hashes) <= required | {"resolution_json"} or any(
                not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()):
            raise ValueError("Rejected candidates require exact source hashes")
        for key, source in (("image", parent.image), ("labels", parent.labels), ("shift", shift)):
            verified = parent.raster_provenance.get(key, {})
            actual = verified.get("sha256") if verified.get("status") == "verified_sha256" else sha256_file(source)
            if hashes[key] != actual:
                raise ValueError(f"Rejected-candidate {key} SHA256 does not match this specimen")
        if "resolution_json" in hashes and (not resolution or hashes["resolution_json"] != sha256_file(resolution)):
            raise ValueError("Rejected-candidate resolution SHA256 does not match this specimen")
        namespace = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        cal = parent.cal
        if (meta.get("identity_namespace") != namespace or meta.get("crop_size_px") != [cal["width"], cal["height"]]
                or not np.array_equal(meta.get("crop_origin_px"), cal["origin_px"])
                or (meta.get("source_mpp") is not None and (meta["source_mpp"] != cal["mpp"]
                    or not np.array_equal(meta.get("crop_origin_um"), cal["origin_px"] * cal["mpp"])))
                or (meta.get("source_mpp") is None and (meta.get("crop_origin_um") is not None
                    or meta.get("calibration_source") != "physical_scale_unavailable"))):
            raise ValueError("Rejected-candidate coordinate calibration/identity binding mismatch")
        counts = [meta.get(k) for k in ("candidate_count", "accepted_prediction_count", "input_prediction_count")]
        if (any(type(v) is not int or v < 0 for v in counts) or counts[0] != len(features)
                or counts[0] + counts[1] != counts[2]):
            raise ValueError("Rejected-candidate population accounting mismatch")
        canonical_keys = {(source, str(value)) for source in SOURCES if source + "_id" in parent.cells
                          for value in parent.cells[source + "_id"] if str(value) not in ("", "nan", "<NA>")}
        self.records, source_keys = {}, set()
        for feature in features:
            props = feature.get("properties", {})
            key = (props.get("source"), props.get("source_id"))
            if key[0] not in SOURCES or not isinstance(key[1], str) or not key[1] or key in source_keys or key in canonical_keys:
                raise ValueError("Rejected candidate has duplicate, invalid or canonical detector identity")
            expected = "detector_candidate:" + hashlib.sha256(json.dumps([namespace, *key], separators=(",", ":")).encode()).hexdigest()
            if (feature.get("type") != "Feature" or feature.get("id") != expected
                    or props.get("candidate_uid") != expected or "cell_uid" in props or "cell_id" in props
                    or props.get("canonical_assignment", "missing") is not None
                    or props.get("observation_unit") != "rejected_detector_prediction"
                    or not isinstance(props.get("decision"), str) or not props["decision"]
                    or props["decision"] == "accepted_instance_fusion"):
                raise ValueError("Rejected-candidate identity or exclusion status is invalid")
            xy = np.asarray(props.get("crop_xy"), float)
            physical = np.asarray(props.get("coordinates_um"), float)
            bad_physical = (props.get("coordinates_um") is not None if meta.get("source_mpp") is None else
                (physical.shape != (2,) or not np.allclose(physical, (xy + cal["origin_px"]) * cal["mpp"], rtol=0, atol=cal["mpp"]*1e-6)))
            if (xy.shape != (2,) or not np.isfinite(xy).all()
                    or not 0 <= xy[0] < cal["width"] or not 0 <= xy[1] < cal["height"]
                    or bad_physical):
                raise ValueError("Rejected-candidate centroid calibration mismatch")
            geometry = feature.get("geometry", {})
            status = props.get("geometry_status")
            expected_type = {"source_polygon": "Polygon", "invalid_source_outline": "LineString", "centroid_only": "Point"}.get(status)
            if expected_type is None or geometry.get("type") != expected_type:
                raise ValueError("Candidate geometry and declared geometry status disagree")
            coords = np.asarray(geometry.get("coordinates"), float)
            if not np.isfinite(coords).all() or (status != "source_polygon" and not props.get("geometry_issue")):
                raise ValueError("Invalid candidate geometry must have finite coordinates and an explicit reason")
            if status == "source_polygon" and (coords.ndim != 3 or coords.shape[0] != 1
                    or coords.shape[1] < 4 or coords.shape[2] != 2 or not np.array_equal(coords[0, 0], coords[0, -1])):
                raise ValueError("Candidate source polygon requires its explicit closed exterior ring")
            geom = shape(geometry)
            if (geom.is_empty or (status == "source_polygon" and (not geom.is_valid or geom.area <= 0
                    or props.get("geometry_issue") is not None))
                    or (status == "centroid_only" and not np.array_equal(coords, xy))):
                raise ValueError("Malformed candidate source geometry")
            # Wildly displaced vertices are not silently clamped into the display crop.
            if coords.size and np.max(np.abs(coords)) > max(cal["width"], cal["height"]) * 10:
                raise ValueError("Candidate outline is implausibly far outside its native crop")
            self.records[expected] = feature
            source_keys.add(key)
        if (dict(Counter(f["properties"]["decision"] for f in features)) != meta.get("decision_counts")
                or dict(Counter(f["properties"]["geometry_status"] for f in features)) != meta.get("geometry_counts")):
            raise ValueError("Rejected-candidate summary counts mismatch")
        payload_status = "unverified_source_payload"
        if source_inputs is not None:
            if set(source_inputs) != set(VERIFICATION_KEYS) or not all(source_inputs.values()):
                raise ValueError("Candidate verification requires all four explicit detector/alignment inputs")
            for key, source in source_inputs.items():
                if sha256_file(source) != hashes[key]:
                    raise ValueError(f"Candidate verification {key} SHA256 mismatch")
            source_cells = load_detector_inputs(source_inputs)
            decisions, accepted = read_alignment(source_cells, source_inputs["alignment_csv"])
            source_cal = candidate_calibration(shift, resolution if "resolution_json" in hashes else None)
            if meta.get("source_mpp") != source_cal["mpp"] or meta.get("calibration_source") != source_cal["calibration_source"]:
                raise ValueError("Candidate producer calibration does not match its explicit source inputs")
            original = {(cell.source, cell.source_id): cell for cell in source_cells}
            if (source_keys != set(original) - accepted or len(original) != counts[2] or len(accepted) != counts[1]):
                raise ValueError("Candidate payload does not contain exactly the source alignment's rejected population")
            from build_cell_consensus import DETECTOR_SCOPES
            for feature in self.records.values():
                props = feature["properties"]
                key = (props["source"], props["source_id"])
                cell = original[key]
                geometry, status, issue = candidate_geometry(cell)
                if (feature["geometry"] != geometry or props["geometry_status"] != status
                        or props.get("geometry_issue") != issue or props["crop_xy"] != [cell.x, cell.y]):
                    raise ValueError("Candidate geometry/centroid does not match its explicit detector source")
                source_um = ((np.array([cell.x, cell.y]) + source_cal["origin_px"]) * source_cal["mpp"]).tolist() if source_cal["mpp"] is not None else None
                if props.get("coordinates_um") != source_um:
                    raise ValueError("Candidate physical coordinates do not match its explicit source calibration")
                expected = {"decision": decisions[key]["decision"],
                    "fusion_evidence": {k: v for k, v in decisions[key].items() if k != "decision"},
                    "detector_phenotype_evidence": json_safe({"type_id": cell.type_id, "type": cell.cell_type, "probability": cell.probability})}
                # Strict JSON comparison also rejects numeric/string/bool evidence
                # substitutions that Python's loose scalar equality can conceal.
                for field, value in expected.items():
                    if json.dumps(props.get(field), sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
                        raise ValueError(f"Candidate {field} does not match its explicit source evidence")
                if "detector_scope" in props and props["detector_scope"] != DETECTOR_SCOPES[cell.source]:
                    raise ValueError("Candidate detector scope does not match the declared source detector")
            # Fail if an explicitly supplied source changed during parsing.
            if any(sha256_file(source) != hashes[key] for key, source in source_inputs.items()):
                raise ValueError("Candidate verification input changed during source recheck")
            payload_status = "verified_against_explicit_sources"
        self.info = {**meta, "status": "source_verified" if source_inputs is not None else "raster_bound_payload_unverified",
            "raster_binding_status": "verified_sha256", "candidate_payload_status": payload_status,
            "review_file_sha256": review_digest, "review_file_hash_role": "Observed checksum of loaded bytes only, not an independent trusted payload receipt",
            "geometry_loading": "Source GeoJSON held in memory; native H&E decoded only in bounded windows",
            "source_verification": ("Explicit detector/alignment files hash-matched; complete rejected population, original geometry, centroids, decisions and displayed evidence rechecked"
                if source_inputs is not None else "Image/labels/shift hashes verified only; candidate geometry/evidence and detector/alignment hash declarations remain unverified"),
            "claim": "Read-only observations; source consistency is not independent detection accuracy"}

    def metadata(self):
        return self.info

    def queue(self, offset=0, limit=25, source=""):
        if offset < 0 or not 1 <= limit <= 100 or source not in ("", *SOURCES):
            raise ValueError("Rejected-candidate queue offset/limit/source is invalid")
        rows = [f["properties"] for f in self.records.values() if not source or f["properties"]["source"] == source]
        fields = ("candidate_uid", "source", "source_id", "decision", "geometry_status")
        return {"total": len(rows), "candidate_payload_status": self.info["candidate_payload_status"],
                "rows": [{k: p[k] for k in fields} for p in rows[offset:offset+limit]]}

    def bounds(self, uid, size):
        if not 64 <= size <= 1024:
            raise ValueError("Native candidate window must be in [64,1024] pixels")
        xy = self.records[uid]["properties"]["crop_xy"]
        x0, y0 = [max(0, int(round(v)) - size//2) for v in xy]
        return x0, y0, min(self.parent.cal["width"], x0+size), min(self.parent.cal["height"], y0+size)

    def detail(self, uid, size=384):
        feature = self.records[uid]
        return {**feature["properties"], "geometry": feature["geometry"],
            "display_coordinates_um": ((np.array(feature["properties"]["crop_xy"]) + self.parent.cal["origin_px"]) * self.parent.cal["mpp"]).tolist(),
            "display_coordinate_calibration": self.parent.cal["calibration_source"],
            "producer_calibration": self.info["calibration_source"],
            "fusion_evidence_scope": self.info.get("fusion_evidence_scope", "Original source values, not recalibrated by this review layer"),
            "window": dict(zip(("x0", "y0", "x1", "y1"), self.bounds(uid, size))),
            "read_only": True, "source_raster_identity": "verified_sha256",
            "candidate_payload_status": self.info["candidate_payload_status"],
            "interpretation": INTERPRETATION if self.info["candidate_payload_status"] == "verified_against_explicit_sources" else
                "Supplied observation reported as excluded by fusion; source geometry/evidence not reverified. Not a canonical cell or an established false detection",
            "profile_available": False, "used_for_inference": False}

    def png(self, uid, size=384, overlay=False):
        from PIL import Image, ImageDraw
        from cell_profile_io import RasterReader
        bounds = self.bounds(uid, size)
        if not overlay:
            with RasterReader(self.parent.image) as reader:
                image = Image.fromarray(reader.window(*bounds))
        else:
            x0, y0, x1, y1 = bounds
            image = Image.new("RGBA", (x1-x0, y1-y0))
            draw = ImageDraw.Draw(image)
            feature = self.records[uid]
            geometry = feature["geometry"]
            if geometry["type"] == "Point":
                x, y = geometry["coordinates"]; x -= x0; y -= y0
                draw.line([(x-5, y), (x+5, y)], fill=(199, 36, 112, 255), width=2)
                draw.line([(x, y-5), (x, y+5)], fill=(199, 36, 112, 255), width=2)
            else:
                coords = geometry["coordinates"][0] if geometry["type"] == "Polygon" else geometry["coordinates"]
                draw.line([(x-x0, y-y0) for x, y in coords],
                    fill=(199, 36, 112, 255) if geometry["type"] == "Polygon" else (180, 105, 0, 255), width=2)
        output = io.BytesIO(); image.save(output, format="PNG")
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("alignment-csv", "stardist-objects", "hovernet-cells", "cellvit-cells", "image", "labels", "shift", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--resolution-json", help="Optional passed physical-resolution report; missing physical scale remains unavailable")
    args = vars(parser.parse_args())
    cells = load_detector_inputs(args)
    decisions, accepted = read_alignment(cells, args["alignment_csv"])
    inputs = {key: args[key] for key in ("image", "labels", "shift", "alignment_csv", *SOURCE_KEYS)}
    if args["resolution_json"]:
        inputs["resolution_json"] = args["resolution_json"]
    result = export_candidates(cells, decisions, accepted, inputs, args["output"])
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
