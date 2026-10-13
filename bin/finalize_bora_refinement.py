#!/usr/bin/env python3
"""Bind Bora outputs to CellPhenotyper's support and provenance contract.

Bora owns the pixel refinement and GeoJSON conversion.  This adapter does not
alter Bora's labels; it verifies the hard GrandQC/KODAMA support constraints,
creates the categorical uncertainty/provenance sidecars used downstream, and
records immutable tool provenance for restart and release audits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bora_support_contract import verify_support
from geojson_coordinates import ORIGINAL_FRAME, rebase_geojson_file
from grow_to_tissue import save_preview_png
from model_provenance import sha256_file
from ome_tiff_metadata import read_mpp_json, restore_ome_physical_size
from refine_grown_tissue_medsam import (
    GRANDQC_KODAMA_OUTLIER_CODE,
    ScaledBinaryMaskReader,
    TiffWindowReader,
    finalize_refinement_provenance_outputs,
    write_refinement_provenance_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and register Bora tissue refinement outputs."
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-mask", required=True)
    parser.add_argument("--baseline-mask", required=True)
    parser.add_argument("--tissue-mask", required=True)
    parser.add_argument("--uncertainty-mask", required=True)
    parser.add_argument("--resolution-json", default="")
    parser.add_argument(
        "--shift-json",
        required=True,
        help="Analysis-crop transform used to publish GeoJSON in original-image pixels.",
    )
    parser.add_argument("--refined-mask", required=True)
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--bora-report", required=True)
    parser.add_argument("--uncertainty-out", required=True)
    parser.add_argument("--provenance-out", required=True)
    parser.add_argument("--geojson-provenance-out", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--bora-repository", required=True)
    parser.add_argument("--bora-revision", required=True)
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument("--preview-factor", type=int, default=10)
    parser.add_argument("--preview-threshold-mb", type=float, default=100.0)
    parser.add_argument("--preview-alpha", type=float, default=0.45)
    parser.add_argument("--default-mpp", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def artifact(path: str | Path) -> dict:
    source = Path(path)
    return {
        "filename": source.name,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def main() -> None:
    args = parse_args()
    block_rows = max(1, int(args.block_rows))
    mpp_x, mpp_y = read_mpp_json(args.resolution_json, args.default_mpp)
    ome_validation = restore_ome_physical_size(
        args.refined_mask,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )
    final_reader = TiffWindowReader(args.refined_mask, "Bora refined labels")
    baseline_reader = TiffWindowReader(args.baseline_mask, "pre-Bora labels")
    source_reader = TiffWindowReader(args.source_mask, "source cluster labels")
    uncertainty_reader = TiffWindowReader(
        args.uncertainty_mask, "clustering uncertainty"
    )
    shape = final_reader.spatial_shape()
    for name, reader in (
        ("baseline mask", baseline_reader),
        ("source mask", source_reader),
        ("uncertainty mask", uncertainty_reader),
    ):
        if reader.spatial_shape() != shape:
            raise ValueError(
                f"{name} shape {reader.spatial_shape()} does not match Bora output {shape}"
            )
    tissue_reader = ScaledBinaryMaskReader(args.tissue_mask, shape)
    try:
        support_counts = verify_support(
            final_reader,
            baseline_reader,
            tissue_reader,
            uncertainty_reader,
            block_rows,
            hard_outlier_code=GRANDQC_KODAMA_OUTLIER_CODE,
        )
        coordinate_transform = rebase_geojson_file(args.geojson, args.shift_json)
        report_path = Path(args.bora_report)
        report = json.loads(report_path.read_text())
        report["cellphenotyper_contract"] = {
            "sample_id": args.sample_id,
            "bora_repository": args.bora_repository,
            "bora_revision": args.bora_revision,
            "support_verification": support_counts,
            "geojson_coordinate_system": {
                "coordinate_space": ORIGINAL_FRAME,
                "crop_to_original_offset_xy": [
                    coordinate_transform["dx"],
                    coordinate_transform["dy"],
                ],
                "shift_json_sha256": coordinate_transform["source_sha256"],
            },
            "ome_validation": ome_validation,
            "provenance_metadata": Path(
                str(args.refined_mask) + ".provenance.json"
            ).name,
            "uncertainty_output": Path(args.uncertainty_out).name,
            "provenance_output": Path(args.provenance_out).name,
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")

        args.out = args.refined_mask
        args.clustering_uncertainty = args.uncertainty_mask
        args.stream_block_rows = block_rows
        args.source_mpp_x = float(mpp_x or 0.0)
        args.source_mpp_y = float(mpp_y or 0.0)
        args.physical_parameters = {}
        metadata = write_refinement_provenance_outputs(
            args,
            final_reader._arr,
            baseline_reader,
            tissue_reader,
            uncertainty_reader=uncertainty_reader,
            source_reader=source_reader,
        )
        metadata = finalize_refinement_provenance_outputs(args, metadata)
        metadata.update(
            {
                "refinement_engine": "Bora",
                "bora_repository": args.bora_repository,
                "bora_revision": args.bora_revision,
                "bora_report": artifact(args.bora_report),
                "support_verification": support_counts,
                "ome_validation": ome_validation,
                "support_policy": (
                    "The upstream cluster-label footprint is authoritative and may "
                    "include GrandQC artifact candidates. Bora cannot add foreground "
                    "outside that footprint; only KODAMA-confirmed artifact outliers "
                    "(uncertainty code 5) are hard-negative support. The scaled tissue "
                    "mask is reported as a diagnostic rather than used to erase "
                    "candidate tissue inherited from the upstream mask."
                ),
            }
        )
        metadata_path = Path(str(args.refined_mask) + ".provenance.json")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        save_preview_png(
            args.image,
            final_reader._arr,
            args.preview,
            factor=args.preview_factor,
            size_threshold_mb=args.preview_threshold_mb,
            alpha=args.preview_alpha,
        )

        geojson_provenance = {
            "schema_version": "cellphenotyper.bora_geojson_provenance.v1",
            "sample_id": args.sample_id,
            "converter": "Bora",
            "bora_repository": args.bora_repository,
            "bora_revision": args.bora_revision,
            "inputs": {
                "refined_labels": artifact(args.refined_mask),
                "uncertainty": artifact(args.uncertainty_out),
                "refinement_provenance": artifact(args.provenance_out),
            },
            "output": artifact(args.geojson),
            "bora_report": artifact(args.bora_report),
            "support_verification": support_counts,
            "coordinate_system": {
                "coordinate_space": ORIGINAL_FRAME,
                "crop_to_original_offset_xy": [
                    coordinate_transform["dx"],
                    coordinate_transform["dy"],
                ],
                "crop_size": coordinate_transform["crop_size"],
                "full_size": coordinate_transform["full_size"],
                "shift_json": artifact(args.shift_json),
            },
            "claim": (
                "Topology-aware categorical vectorization of Bora-refined labels; "
                "not calibrated biological confidence."
            ),
        }
        Path(args.geojson_provenance_out).write_text(
            json.dumps(geojson_provenance, indent=2) + "\n"
        )
    finally:
        tissue_reader.close()
        uncertainty_reader.close()
        source_reader.close()
        baseline_reader.close()
        final_reader.close()


if __name__ == "__main__":
    main()
