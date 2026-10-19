#!/usr/bin/env python3
"""Explicitly register existing GigaTIME files in a NEW experiment index.

No model, workflow engine, remote client or GPU library is imported.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from export_experiment import (ExperimentError, GIGATIME_SCORE_SEMANTICS, _identifier,
                               _number, _path, _text, export_experiment, sha256,
                               validate_experiment)


def inspect_marker_image(path):
    """Return explicit OME page names and TIFF header facts, without pixels."""
    path = _path(str(path), Path.cwd(), "marker image")
    try:
        import tifffile
        with tifffile.TiffFile(path) as image:
            if not image.is_ome or len(image.series) != 1 or set(image.series[0].axes) - set("CYX"):
                raise ExperimentError("Select a single-series OME-TIFF marker image with C/Y/X axes")
            xml = ET.fromstring(image.ome_metadata)
            pixels = next((node for node in xml.iter() if node.tag.rsplit("}", 1)[-1] == "Pixels"), None)
            names = [node.attrib.get("Name") for node in pixels
                     if node.tag.rsplit("}", 1)[-1] == "Channel"] if pixels is not None else []
            if len(names) != len(image.pages):
                raise ExperimentError("OME channel names and TIFF pages differ; use an explicitly supported marker image")
            result = []
            for index, page in enumerate(image.pages):
                if len(page.shape) != 2:
                    raise ExperimentError(f"TIFF page {index} is not one scalar marker plane")
                result.append(dict(page=index, marker=names[index], width=int(page.shape[1]),
                                   height=int(page.shape[0]), dtype=page.dtype.name,
                                   requires_display_max=page.dtype.kind == "f"))
            return result
    except (ImportError, OSError, ValueError, ET.ParseError) as exc:
        if isinstance(exc, ExperimentError):
            raise
        raise ExperimentError(f"Cannot inspect marker image {path}: {exc}") from exc


def register_gigatime(experiment, output, *, tissue_id, marker_image, channels,
                      full_image=False, extent=None, scores=None, mask=None,
                      mask_name=None, segmentation_unit=None, score_columns=None,
                      source_coordinate_convention=None, display_max=None,
                      id_column="label_id", x_column="centroid_x_px", y_column="centroid_y_px",
                      artifact_prefix="gigatime"):
    """Clone a verified experiment and add explicit bindings to one tissue."""
    source = _path(str(experiment), Path.cwd(), "source experiment")
    destination = Path(os.path.abspath(Path(output).expanduser()))
    if os.path.lexists(destination):
        raise ExperimentError(f"Output already exists: {destination}. Choose a NEW filename; source projects are never overwritten")
    before = sha256(source)
    spec = validate_experiment(source)
    if sha256(source) != before:
        raise ExperimentError("Source experiment changed during validation; no output was created")
    tissues = [tissue for tissue in spec["tissues"] if tissue["tissue_id"] == tissue_id]
    if len(tissues) != 1:
        available = ", ".join(tissue["tissue_id"] for tissue in spec["tissues"])
        raise ExperimentError(f"Unknown tissue_id '{tissue_id}'. Choose one of: {available}")
    tissue = tissues[0]
    prefix = _identifier(artifact_prefix, "artifact-prefix")
    if bool(full_image) == (extent is not None):
        raise ExperimentError("Choose exactly one of --full-image or --extent X Y WIDTH HEIGHT; placement is never guessed")
    if full_image:
        placement = dict(x=0, y=0, width=tissue["image"]["width"], height=tissue["image"]["height"])
    else:
        if not isinstance(extent, (list, tuple)) or len(extent) != 4:
            raise ExperimentError("--extent needs X Y WIDTH HEIGHT in native H&E pixels")
        placement = dict(zip(("x", "y", "width", "height"), extent))
    marker_path = _path(str(marker_image), Path.cwd(), "marker image")
    available = inspect_marker_image(marker_path)
    if not channels:
        raise ExperimentError("Select at least one --channel MARKER PAGE; use --inspect-marker-image to list choices")
    ranges = {}
    for marker, maximum in display_max or []:
        if marker in ranges:
            raise ExperimentError(f"Display range was supplied twice for marker '{marker}'")
        try:
            ranges[marker] = _number(float(maximum), "display-max", positive=True)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"--display-max for '{marker}' must be a finite positive number") from exc
    selected, selected_markers, selected_pages = [], set(), set()
    for marker, page in channels:
        marker = _text(marker, "channel marker")
        try:
            if isinstance(page, bool):
                raise ValueError("boolean page")
            index = int(page)
            if str(index) != str(page):
                raise ValueError("noninteger page")
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"Page for '{marker}' must be a nonnegative integer") from exc
        if index < 0 or index >= len(available) or available[index]["marker"] != marker:
            choices = ", ".join(f"{item['marker']} page {item['page']}" for item in available)
            raise ExperimentError(f"Marker/page '{marker}' {index} does not match the OME header. Available: {choices}")
        if marker in selected_markers or index in selected_pages:
            raise ExperimentError("Selected marker names and pages must be unique")
        selected_markers.add(marker); selected_pages.add(index)
        channel = dict(marker=marker, page=index)
        if marker in ranges:
            channel["display_max"] = ranges[marker]
        elif available[index]["requires_display_max"]:
            raise ExperimentError(f"Floating marker '{marker}' needs --display-max '{marker}' VALUE; display scaling is explicit")
        selected.append(channel)
    if set(ranges) - selected_markers:
        raise ExperimentError("--display-max names a marker that was not selected with --channel")
    common = dict(frame_id=tissue["image"]["frame_id"], image_sha256=tissue["image"]["sha256"])
    marker = dict(common, artifact_id=f"{prefix}:markers", role="gigatime_marker_image", format="ome-tiff",
                  path=str(marker_path), score_semantics=GIGATIME_SCORE_SEMANTICS, channels=selected, extent=placement)
    added = [marker]
    score_options = (mask, mask_name, segmentation_unit, score_columns, source_coordinate_convention)
    if not scores and any(value is not None for value in score_options):
        raise ExperimentError("Mask/score-column/coordinate options require --scores; omit them for marker-image-only registration")
    if scores:
        required = dict(mask=mask, mask_name=mask_name, segmentation_unit=segmentation_unit,
                        score_column=score_columns, source_coordinate_convention=source_coordinate_convention)
        missing = ["--" + key.replace("_", "-") for key, value in required.items() if not value]
        if missing:
            raise ExperimentError("Cell scores require explicit " + ", ".join(missing))
        if segmentation_unit not in ("cell", "nucleus"):
            raise ExperimentError("--segmentation-unit must be cell or nucleus; grid masks are unsupported")
        if source_coordinate_convention not in ("zero-based-pixel-index", "native-image-pixel-centres"):
            raise ExperimentError("Choose an explicit --source-coordinate-convention")
        mask_path = _path(str(mask), Path.cwd(), "segmentation mask")
        score_path = _path(str(scores), Path.cwd(), "cell score CSV")
        if not (score_path.name.lower().endswith(".csv") or score_path.name.lower().endswith(".csv.gz")):
            raise ExperimentError("--scores must be an existing CSV or CSV.GZ file")
        raster = available[int(selected[0]["page"])]
        sx = _number(placement["width"], "extent.width", positive=True) / raster["width"]
        sy = _number(placement["height"], "extent.height", positive=True) / raster["height"]
        affine = ([[sx, 0, placement["x"] + sx / 2], [0, sy, placement["y"] + sy / 2], [0, 0, 1]]
                  if source_coordinate_convention == "zero-based-pixel-index" else [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mask_artifact = dict(common, artifact_id=f"{prefix}:mask", role="cell_segmentation_mask", format="tiff",
                             path=str(mask_path), mask_name=mask_name, segmentation_unit=segmentation_unit)
        score_artifact = dict(common, artifact_id=f"{prefix}:scores", role="gigatime_cell_scores",
                              format="csv.gz" if score_path.name.lower().endswith(".gz") else "csv", path=str(score_path),
                              id_column=id_column, x_column=x_column, y_column=y_column,
                              marker_columns=[dict(marker=name, column=column) for name, column in score_columns],
                              mask_name=mask_name, mask_artifact_id=mask_artifact["artifact_id"],
                              source_artifact_id=marker["artifact_id"], source_to_image_affine=affine,
                              source_coordinate_convention=source_coordinate_convention,
                              score_semantics=GIGATIME_SCORE_SEMANTICS, value_encoding="native_export_values")
        added.extend([mask_artifact, score_artifact])
    existing = {artifact["artifact_id"] for artifact in tissue["artifacts"]}
    if existing.intersection(artifact["artifact_id"] for artifact in added):
        raise ExperimentError("Artifact IDs already exist in this tissue. Choose a different --artifact-prefix; existing layers are preserved")
    tissue["artifacts"].extend(added)
    # Existing relative paths resolve against the source index. New assets are
    # explicit absolute inputs, and export rebases both for the new destination.
    if sha256(source) != before:
        raise ExperimentError("Source experiment changed while preparing registration; no output was created")
    return export_experiment(spec, destination, base_dir=source.parent, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Register existing GigaTIME outputs in a NEW copy of a verified CellPhenotyper experiment. Never starts inference.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--experiment", type=Path, help="Existing experiment index; all tissues and prior assets are retained")
    mode.add_argument("--inspect-marker-image", type=Path, help="List OME marker names/pages/dimensions/dtypes without creating files")
    parser.add_argument("--output", type=Path, help="NEW destination index; existing paths and aliases are refused")
    parser.add_argument("--tissue-id", help="Existing tissue ID to receive the layers")
    parser.add_argument("--marker-image", type=Path, help="Existing GigaTIME OME-TIFF")
    parser.add_argument("--channel", action="append", nargs=2, metavar=("MARKER", "PAGE"), help="Exact OME marker name and zero-based page; repeat to select channels")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--full-image", action="store_true", help="Explicitly confirm the marker raster spans the entire selected H&E image")
    placement.add_argument("--extent", nargs=4, type=float, metavar=("X", "Y", "WIDTH", "HEIGHT"), help="Explicit marker placement in native H&E pixel edges")
    parser.add_argument("--display-max", action="append", nargs=2, metavar=("MARKER", "VALUE"), help="Explicit display maximum; required for floating marker pages")
    parser.add_argument("--scores", type=Path, help="Optional existing cell CSV/CSV.GZ; values stay unchanged")
    parser.add_argument("--mask", type=Path, help="Native H&E-size integer cell/nucleus label TIFF")
    parser.add_argument("--mask-name", help="Explicit compartment matching any CSV mask_name column")
    parser.add_argument("--segmentation-unit", choices=("cell", "nucleus"))
    parser.add_argument("--score-column", action="append", nargs=2, metavar=("MARKER", "COLUMN"), help="Explicit marker-to-score column mapping; repeat as needed")
    parser.add_argument("--source-coordinate-convention", choices=("zero-based-pixel-index", "native-image-pixel-centres"), help="Required with --scores; selects the deterministic affine without guessing")
    parser.add_argument("--id-column", default="label_id")
    parser.add_argument("--x-column", default="centroid_x_px")
    parser.add_argument("--y-column", default="centroid_y_px")
    parser.add_argument("--artifact-prefix", default="gigatime", help="Stable ID prefix; change it to add another independent layer group")
    args = parser.parse_args(argv)
    try:
        if args.inspect_marker_image:
            if args.output:
                raise ExperimentError("Inspection creates no project; remove --output or use --experiment")
            for item in inspect_marker_image(args.inspect_marker_image):
                note = " (requires --display-max)" if item["requires_display_max"] else ""
                print(f"page {item['page']}: {item['marker'] or '<unnamed>'} | {item['width']}x{item['height']} | {item['dtype']}{note}")
            return 0
        missing = [flag for flag, value in (("--output", args.output), ("--tissue-id", args.tissue_id),
                                           ("--marker-image", args.marker_image), ("--channel", args.channel)) if not value]
        if missing:
            raise ExperimentError("Registration requires " + ", ".join(missing) + "; see --help")
        payload = register_gigatime(args.experiment, args.output, tissue_id=args.tissue_id,
            marker_image=args.marker_image, channels=args.channel, full_image=args.full_image,
            extent=args.extent, scores=args.scores, mask=args.mask, mask_name=args.mask_name,
            segmentation_unit=args.segmentation_unit, score_columns=args.score_column,
            source_coordinate_convention=args.source_coordinate_convention, display_max=args.display_max,
            id_column=args.id_column, x_column=args.x_column, y_column=args.y_column,
            artifact_prefix=args.artifact_prefix)
        print(f"Ready to open in wsiTools: {args.output}\nRetained {len(payload['tissues'])} tissues and existing project data. Source project and asset bytes are unchanged; no inference was started.")
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        action = "inspect marker image" if args.inspect_marker_image else "register GigaTIME outputs"
        print(f"Cannot {action}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
