#!/usr/bin/env python3
"""Attach frozen Bora PLS tissue names to accepted, unchanged region geometry.

This CPU-only handoff consumes ``bora classify-regions`` products. It preserves
all cells, images and existing provenance through checked relative references.
Anonymous group annotations remain available at their original project path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from bora_contextual_fullslide import _rebase_project, _validate_project, sha256
from export_experiment import atomic_json


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _geometry(features: list) -> str:
    return hashlib.sha256(json.dumps([f["geometry"] for f in features],
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def export_identity_project(base_project: Path, identity_dir: Path,
                            output: Path, *, tissue_id: str | None = None) -> dict:
    base = Path(base_project).resolve(strict=True)
    identity = Path(identity_dir).resolve(strict=True)
    output = Path(output).absolute()
    if output.name != "experiment.cellphenotyper.json" or not output.parent.parent.is_dir():
        raise ValueError("Use experiment.cellphenotyper.json in a new output folder")
    if output.parent.exists():
        raise FileExistsError("Identity project output folder already exists")
    project = _validate_project(base)
    tissues = [item for item in project["tissues"]
               if tissue_id is None or item["tissue_id"] == tissue_id]
    if len(tissues) != 1:
        raise ValueError("Select exactly one source tissue with --tissue-id")
    tissue = tissues[0]
    image = tissue["image"]
    report_path = identity / "report.json"
    prediction_path = identity / "tissue-annotations.geojson"
    descriptors_path = identity / "descriptors.npz"
    report, predicted = _read(report_path), _read(prediction_path)
    if (report.get("format") != "bora-region-identity-v1" or
            report.get("complete") is not True or
            report.get("geometry_unchanged") is not True or
            report.get("biological_accuracy_qualified") is not False or
            report.get("responses_are_calibrated_probabilities") is not False):
        raise ValueError("Bora identity report lacks complete, unchanged-geometry provenance")
    if (report.get("source_frame_id") != image["frame_id"] or
            report.get("source_image_sha256") != image["sha256"] or
            report.get("native_shape_yx") != [image["height"], image["width"]] or
            report.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]]):
        raise ValueError("Bora identity report belongs to another source image frame")
    inputs = report.get("input_sha256", {})
    if set(inputs) != {"contours", "tokens", "model", "registration"} or any(
            not isinstance(value, str) or len(value) != 64 or
            any(ch not in "0123456789abcdef" for ch in value)
            for value in inputs.values()):
        raise ValueError("Bora identity report lacks frozen input hashes")
    artifacts = [item for item in tissue.get("artifacts", [])
                 if item.get("role") == "tissue_annotations" and
                 item.get("sha256") == inputs["contours"]]
    if len(artifacts) != 1:
        raise ValueError("Bora identity must bind exactly one accepted source annotation")
    source_artifact = artifacts[0]
    source_path = (base.parent / source_artifact["path"]).resolve(strict=True)
    source = _read(source_path)
    source_features, result_features = source.get("features"), predicted.get("features")
    if (source.get("type") != "FeatureCollection" or
            predicted.get("type") != "FeatureCollection" or
            not isinstance(source_features, list) or not source_features or
            not isinstance(result_features, list) or
            len(source_features) != len(result_features) or
            _geometry(source_features) != _geometry(result_features) or
            report.get("geometry_sha256") != _geometry(source_features)):
        raise ValueError("Tissue identity changed accepted geometry or feature order")
    for before, after in zip(source_features, result_features):
        original, named = before.get("properties", {}), after.get("properties", {})
        if (named.get("source_group_id") != original.get("value") or
                not isinstance(named.get("classification"), dict) or
                not named["classification"].get("name")):
            raise ValueError("Tissue prediction lost original group identity or tissue name")
    expected_outputs = report.get("output_sha256", {})
    for path in (prediction_path, descriptors_path):
        if expected_outputs.get(path.name) != sha256(path):
            raise ValueError("Bora identity output hash differs from its report")

    stage = Path(tempfile.mkdtemp(prefix=f".{output.parent.name}.staging-",
                                  dir=output.parent.parent))
    try:
        combined = _rebase_project(base, stage / output.name)
        target = next(item for item in combined["tissues"]
                      if item["tissue_id"] == tissue["tissue_id"])
        artifact = next(item for item in target["artifacts"]
                        if item.get("role") == "tissue_annotations" and
                        item.get("sha256") == inputs["contours"])
        original_ref = copy.deepcopy(artifact)
        for path in (prediction_path, descriptors_path, report_path):
            destination = stage / path.name
            with path.open("rb") as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            if sha256(path) != sha256(destination):
                raise ValueError("Copied Bora identity product differs from its source")
        artifact.update({"artifact_id": "bora-identity-" + sha256(prediction_path)[:20],
            "label": "Bora · predicted tissue types", "path": prediction_path.name,
            "sha256": sha256(prediction_path), "predicted_not_expert_annotation": True,
            "training_eligible": False, "review_only": False,
            "provenance": {"algorithm": "bora-region-identity-v1",
                "class_identity": "frozen_neighbor_context_PLS_prediction",
                "accepted_geometry": original_ref,
                "identity_report": {"path": report_path.name, "sha256": sha256(report_path)},
                "descriptors": {"path": descriptors_path.name,
                                "sha256": sha256(descriptors_path)},
                "biological_accuracy_qualified": False}})
        combined["settings"]["bora_annotations_mode"] = "ordinary_editable"
        combined["settings"]["bora_status"] = (
            "Frozen Neighbor-context PLS tissue predictions on accepted smooth regions; "
            "biological identity unvalidated")
        atomic_json(stage / output.name, combined, overwrite=False)
        _validate_project(stage / output.name)
        if output.parent.exists():
            raise FileExistsError("Identity output folder appeared before publication")
        os.rename(stage, output.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"project": str(output), "tissue_id": tissue["tissue_id"],
            "feature_count": len(result_features), "geometry_unchanged": True,
            "source_annotation_sha256": inputs["contours"],
            "tissue_annotation_sha256": sha256(prediction_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tissue-id")
    args = parser.parse_args(argv)
    print(json.dumps(export_identity_project(args.base_project, args.identity_dir,
        args.output, tissue_id=args.tissue_id), sort_keys=True))


if __name__ == "__main__":
    main()
