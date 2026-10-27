#!/usr/bin/env python3
"""Compare CellViT++ nuclei with independent StarDist; optionally export tight boundary proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_geometry_refinement import (TuningThresholds, instance_metrics, label_geojson,
    label_mapping, rasterize_cells, sha256_file, tune_geometry, validate_frame, validate_labels)


def read_labels(path: Path, max_pixels: int) -> np.ndarray:
    with tifffile.TiffFile(path) as handle:
        shape = handle.series[0].shape
        if len(shape) != 2 or int(np.prod(shape)) > max_pixels:
            raise ValueError(f"Instance mask must be 2D and at most {max_pixels} pixels: {path}")
        return validate_labels(handle.asarray(), str(path))


def load_inputs(values: dict, max_pixels: int) -> tuple[np.ndarray, np.ndarray, dict, dict, dict]:
    allow_partial = values.get("allow_partial_image_cells", False)
    quarantine_invalid = values.get("quarantine_invalid_geometry", False)
    quarantine_overlapping = values.get("quarantine_overlapping_geometry", False)
    if type(allow_partial) is not bool:
        raise ValueError("allow_partial_image_cells must be an explicit boolean")
    if type(quarantine_invalid) is not bool:
        raise ValueError("quarantine_invalid_geometry must be an explicit boolean")
    if type(quarantine_overlapping) is not bool:
        raise ValueError("quarantine_overlapping_geometry must be an explicit boolean")
    if allow_partial and values.get("cellvit_labels"):
        raise ValueError("Partial-image opt-in requires the original contour JSON route without --cellvit-labels; a cropped label mask cannot prove unchanged original contours")
    if quarantine_invalid and values.get("cellvit_labels"):
        raise ValueError("Invalid-geometry quarantine requires original contour JSON without --cellvit-labels; no source instance population is inferred from an incomplete raster")
    if quarantine_overlapping and values.get("cellvit_labels"):
        raise ValueError("Overlap-geometry quarantine requires original contour JSON without --cellvit-labels; no collision population is inferred from an already resolved raster")
    source = Path(values["source_image"])
    source_hash = sha256_file(source)
    try:
        with tifffile.TiffFile(source) as handle:
            shape = (handle.pages[0].imagelength, handle.pages[0].imagewidth)
    except tifffile.TiffFileError:
        from PIL import Image
        with Image.open(source) as handle:
            shape = (handle.height, handle.width)
    if int(np.prod(shape)) > max_pixels:
        raise ValueError(f"Source exceeds the bounded {max_pixels}-pixel crop limit")
    payload = json.loads(Path(values["cellvit_cells"]).read_text()) if values.get("cellvit_cells") else None
    embedded_frame = None
    if isinstance(payload, dict):
        candidates = (payload, payload.get("metadata", {}))
        for context in candidates:
            if isinstance(context, dict) and "frame" in context:
                embedded_frame = context["frame"]
                break
            if isinstance(context, dict) and "source_image_sha256" in context:
                embedded_frame = context
                break
    if values.get("cellvit_metadata"):
        meta_a = json.loads(Path(values["cellvit_metadata"]).read_text())
    elif embedded_frame is not None:
        meta_a = embedded_frame
    else:
        raise ValueError("Provide --cellvit-metadata or a complete native frame embedded in the CellViT cells JSON")
    meta_b = json.loads(Path(values["stardist_metadata"]).read_text())
    frame = validate_frame(meta_a, meta_b, shape, source_hash)
    if embedded_frame is not None:
        validate_frame(frame, embedded_frame, shape, source_hash)
    if values.get("cellvit_labels"):
        original = read_labels(Path(values["cellvit_labels"]), max_pixels)
        mapping = label_mapping(original, payload)
    elif payload is not None:
        original, mapping = rasterize_cells(payload, shape, allow_partial_image_cells=allow_partial,
            quarantine_invalid_geometry=quarantine_invalid, quarantine_overlapping_geometry=quarantine_overlapping)
    else:
        raise ValueError("Provide --cellvit-labels or --cellvit-cells")
    reference = read_labels(Path(values["stardist_labels"]), max_pixels)
    if original.shape != shape or reference.shape != shape:
        raise ValueError("Source image and detector labels must share the exact native grid")
    provenance = {key: {"path": str(Path(values[key]).resolve()), "sha256": sha256_file(Path(values[key]))}
                  for key in ("source_image", "cellvit_metadata", "stardist_metadata", "cellvit_labels", "cellvit_cells", "stardist_labels") if values.get(key)}
    for metadata, key in ((meta_a, "cellvit_labels" if values.get("cellvit_labels") else "cellvit_cells"), (meta_b, "stardist_labels")):
        declared_hash = metadata.get("input_sha256")
        if declared_hash and declared_hash != provenance[key]["sha256"]:
            raise ValueError(f"Declared detector output hash differs: {key}")
    return original, reference, frame, mapping, provenance


def _write_new(path: Path, content: str) -> None:
    with path.open("x") as handle:
        handle.write(content)


def write_outputs(outdir: Path, original: np.ndarray, candidate: np.ndarray, report: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    names = ("original_labels.tif", "tuned_candidate_labels.tif", "segmentation_qc.json", "segmentation_qc.csv", "segmentation_qc.geojson")
    if any((outdir / name).exists() for name in names):
        raise FileExistsError("Output files already exist; use a new output directory to preserve evidence")
    geojson = label_geojson(candidate, report["cells"], report["frame"])
    for name, mask in ((names[0], original), (names[1], candidate)):
        tifffile.imwrite(outdir / name, mask, compression="deflate", metadata={"axes": "YX", "coordinate_space": report["frame"]["coordinate_space"]})
    report["outputs"] = {name: {"path": str((outdir / name).resolve()), "sha256": sha256_file(outdir / name)} for name in names[:2]}
    _write_new(outdir / names[2], json.dumps(report, indent=2, allow_nan=False) + "\n")
    keys = sorted({key for row in report["cells"] for key in row} - {"original_cell"})
    with (outdir / names[3]).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in report["cells"]:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items() if key != "original_cell"})
    _write_new(outdir / names[4], json.dumps(geojson, allow_nan=False) + "\n")


def calibrate(manifest_path: Path, output: Path, max_pixels: int) -> None:
    """Select gates on calibration subjects; score the frozen choice on validation only."""
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
        raise ValueError("Calibration manifest cases must be an array of objects")
    subjects = {}
    for case in cases:
        subject, split = case.get("subject_id"), case.get("split")
        if not isinstance(subject, str) or not subject.strip() or split not in ("calibration", "validation", "test"):
            raise ValueError("Every calibration case needs subject_id and split calibration/validation/test")
        if subject in subjects and subjects[subject] != split:
            raise ValueError("Subject leakage between calibration, validation or test splits")
        subjects[subject] = split
    if not {"calibration", "validation"}.issubset(set(subjects.values())):
        raise ValueError("At least one disjoint calibration and validation subject is required")
    loaded = []
    source_splits = {}
    for case in cases:
        if case["split"] == "test":
            continue  # Do not stat, hash or open held-out test images or labels.
        if case.get("quarantine_invalid_geometry", False) or case.get("quarantine_overlapping_geometry", False):
            raise ValueError("Invalid-geometry quarantine is review-only; geometry calibration requires assessable original instances")
        values = dict(case)
        for key in ("source_image", "cellvit_metadata", "stardist_metadata", "cellvit_labels", "cellvit_cells", "stardist_labels", "expert_labels", "expert_metadata"):
            if values.get(key):
                path = Path(values[key])
                values[key] = str(path if path.is_absolute() else manifest_path.parent / path)
        if not values.get("expert_labels") or not values.get("expert_metadata") or case.get("reference_quality") != "expert_instance_labels":
            raise ValueError("Calibration/validation require explicit expert_instance_labels, expert_metadata and an expert label mask")
        original, reference, frame, mapping, provenance = load_inputs(values, max_pixels)
        source_hash = frame["source_image_sha256"]
        if source_hash in source_splits and source_splits[source_hash] != case["split"]:
            raise ValueError("Identical source image appears in calibration and validation splits")
        source_splits[source_hash] = case["split"]
        expert = read_labels(Path(values["expert_labels"]), max_pixels)
        if expert.shape != original.shape:
            raise ValueError("Expert instance labels are not in the validated native frame")
        expert_metadata = json.loads(Path(values["expert_metadata"]).read_text())
        validate_frame(frame, expert_metadata, original.shape, frame["source_image_sha256"])
        provenance["expert_labels"] = {"path": values["expert_labels"], "sha256": sha256_file(Path(values["expert_labels"]))}
        provenance["expert_metadata"] = {"path": values["expert_metadata"], "sha256": sha256_file(Path(values["expert_metadata"]))}
        if expert_metadata.get("input_sha256") and expert_metadata["input_sha256"] != provenance["expert_labels"]["sha256"]:
            raise ValueError("Declared expert output hash differs")
        loaded.append((case, original, reference, frame, mapping, expert, provenance))
    options = [None] + [TuningThresholds(min_iou=value) for value in (0.7, 0.8, 0.9, 0.95)]
    selection = []
    for gates in options:
        subject_scores = {}
        for case, original, reference, frame, mapping, expert, _ in loaded:
            if case["split"] != "calibration":
                continue
            candidate, _ = tune_geometry(original, reference, frame, mapping, auto_adjust=gates is not None,
                allow_partial_image_cells=case.get("allow_partial_image_cells", False), thresholds=gates)
            subject_scores.setdefault(case["subject_id"], []).append(instance_metrics(candidate, expert)["panoptic_quality"])
        selection.append({"thresholds": asdict(gates) if gates else None, "auto_adjust": gates is not None,
                          "calibration_subject_mean_panoptic_quality": float(np.mean([np.mean(scores) for scores in subject_scores.values()]))})
    best = max(range(len(options)), key=lambda i: (selection[i]["calibration_subject_mean_panoptic_quality"], i == 0,
                                                  options[i].min_iou if options[i] else 1.0))
    evaluated = []
    for case, original, reference, frame, mapping, expert, provenance in loaded:
        candidate, report = tune_geometry(original, reference, frame, mapping, auto_adjust=options[best] is not None,
            allow_partial_image_cells=case.get("allow_partial_image_cells", False), thresholds=options[best])
        evaluated.append({"subject_id": case["subject_id"], "split": case["split"], "provenance": provenance,
                          "original": instance_metrics(original, expert), "candidate": instance_metrics(candidate, expert),
                          "adjusted_cell_count": report["adjusted_cell_count"]})
    result = {"schema": "cellphenotyper.segmentation-calibration/v1", "manifest_sha256": sha256_file(manifest_path),
              "selection_split": "calibration", "validation_used_for_selection": False, "test_labels_opened": False,
              "excluded_test_case_count": sum(case["split"] == "test" for case in cases),
              "selection_candidates": selection, "selected_thresholds": selection[best]["thresholds"],
              "selected_auto_adjust": options[best] is not None, "evaluations": evaluated,
              "claim_limit": "Expert-reference geometry calibration on declared subjects; no cell-type accuracy inference or clinical validation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(output, json.dumps(result, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-image", "cellvit-labels", "cellvit-cells", "stardist-labels", "cellvit-metadata", "stardist-metadata", "outdir", "calibration-manifest", "calibration-output"):
        parser.add_argument("--" + name)
    parser.add_argument("--auto-adjust", action="store_true", help="Export tight one-to-one boundary candidates; originals remain saved")
    parser.add_argument("--allow-partial-image-cells", action="store_true", help="Explicitly review visible footprints of valid original contours crossing image bounds; original centroids must be in-image, partial cells are always review-required and never auto-adjusted")
    parser.add_argument("--quarantine-invalid-geometry", action="store_true", help="Preserve invalid original rings as unavailable review rows; omit only their derived raster footprints, never repair geometry or invent pixels")
    parser.add_argument("--quarantine-overlapping-geometry", action="store_true", help="Explicit review-only quarantine of ALL cells sharing native pixel centres; preserve every original contour/ID/type, emit unavailable/null agreement, and assign no winner or replacement pixels")
    parser.add_argument("--max-pixels", type=int, default=64_000_000, help="Finite crop memory guard; never tile/rescale implicitly")
    defaults = TuningThresholds()
    for name, default in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=int if name == "min_overlap_pixels" else float, default=default)
    args = parser.parse_args(argv)
    if args.max_pixels < 1:
        parser.error("--max-pixels must be positive")
    if args.calibration_manifest:
        if args.quarantine_invalid_geometry or args.quarantine_overlapping_geometry:
            parser.error("Geometry quarantine is review-only; calibration requires assessable original geometry")
        if args.allow_partial_image_cells:
            parser.error("Set allow_partial_image_cells explicitly per contour-JSON case in the calibration manifest")
        if not args.calibration_output:
            parser.error("--calibration-output is required with --calibration-manifest")
        calibrate(Path(args.calibration_manifest), Path(args.calibration_output), args.max_pixels)
        return
    required = ("source_image", "stardist_labels", "stardist_metadata", "outdir")
    missing = ["--" + key.replace("_", "-") for key in required if not getattr(args, key)]
    if missing or not (args.cellvit_labels or args.cellvit_cells):
        parser.error("Required: " + ", ".join(missing + ([] if args.cellvit_labels or args.cellvit_cells else ["--cellvit-labels or --cellvit-cells"])))
    original, reference, frame, mapping, provenance = load_inputs(vars(args), args.max_pixels)
    thresholds = TuningThresholds(**{key: getattr(args, key) for key in asdict(defaults)})
    candidate, report = tune_geometry(original, reference, frame, mapping, auto_adjust=args.auto_adjust,
        allow_partial_image_cells=args.allow_partial_image_cells,
        quarantine_invalid_geometry=args.quarantine_invalid_geometry,
        quarantine_overlapping_geometry=args.quarantine_overlapping_geometry, thresholds=thresholds)
    report["inputs"] = provenance
    write_outputs(Path(args.outdir), original, candidate, report)
    print(json.dumps({"outdir": str(Path(args.outdir).resolve()), "cells": report["original_cell_count"],
                      "review_cells": report["review_cell_count"], "boundary_candidates": report["adjusted_cell_count"],
                      "reference_quality": report["reference_quality"]}))


if __name__ == "__main__":
    main()
