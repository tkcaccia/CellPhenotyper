"""Attach one verified global Bora exact GeoJSON for exploratory review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .bora_fullslide_action import GENERALIZED_ALGORITHM, native_geometry_identity, sha256, verify
    from .export_experiment import export_experiment, validate_experiment
else:
    from bora_fullslide_action import GENERALIZED_ALGORITHM, native_geometry_identity, sha256, verify
    from export_experiment import export_experiment, validate_experiment


def export(base_project: Path, manifest_path: Path, bora_output: Path,
           destination: Path, requested_algorithm: str | None = None) -> dict:
    if not all(path.is_absolute() for path in (base_project, manifest_path, bora_output, destination)):
        raise ValueError("All project/Bora paths must be absolute")
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    base_project = base_project.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    bora_output = bora_output.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Choose a new project JSON path with an existing parent")
    manifest = json.loads(manifest_path.read_text())
    chain = verify(manifest_path, bora_output, manifest, requested_algorithm)
    if (chain.get("verified_algorithm_id") != GENERALIZED_ALGORITHM or
            not chain.get("algorithm_evidence_receipt_sha256")):
        raise ValueError("Bora project export lacks verified hard-group algorithm evidence")
    if (manifest.get("identity_status") not in {"exploratory_transfer", "externally_validated"} or
            chain.get("tissue_identity_status") != manifest["identity_status"]):
        raise ValueError("Bora identity status differs from the source-bound manifest")
    native = native_geometry_identity(manifest, bora_output)
    previous = validate_experiment(base_project)
    matches = [tissue for tissue in previous["tissues"]
               if (tissue["image"].get("sha256") == native["source_image_sha256"] and
                   tissue["image"].get("frame_id") == native["source_frame_id"] and
                   [tissue["image"].get("height"), tissue["image"].get("width")] == native["native_shape_yx"] and
                   [tissue["image"].get("mpp_x"), tissue["image"].get("mpp_y")] == native["native_mpp_xy"])]
    if len(matches) != 1:
        raise ValueError("Find exactly one project tissue in the Bora native image/frame/shape/MPP")
    tissue = matches[0]
    chain_path = bora_output / "CHAIN_COMPLETE.json"
    artifact_id = "bora-global-exact-" + sha256(chain_path)[:20]
    if any(item.get("artifact_id") == artifact_id for item in tissue["artifacts"]):
        raise ValueError("The exact Bora review boundary is already in this project")
    exact = bora_output / "vectors/exact/mask.geojson"
    tissue["artifacts"].append({
        "artifact_id": artifact_id, "role": "bora_exploratory_boundary_review",
        "label": "Bora anonymous-group boundaries (exploratory review; exact)",
        "path": str(exact), "sha256": sha256(exact), "format": "geojson",
        "review_only": True,
        "source_frame_id": native["source_frame_id"],
        "image_sha256": native["source_image_sha256"],
        "provenance": {"producer": "bora-wsi-global-v1",
                       "algorithm_id": GENERALIZED_ALGORITHM,
                       "algorithm_evidence_receipt_sha256": chain["algorithm_evidence_receipt_sha256"],
                       "prediction_unit": "global_anonymous_Bora-AF_group",
                       "chain_complete_sha256": sha256(chain_path),
                       "manifest_sha256": chain["manifest_sha256"],
                       "native_shape_yx": native["native_shape_yx"],
                       "native_mpp_xy": native["native_mpp_xy"],
                       "identity_status": chain["tissue_identity_status"],
                       "evidence_class": chain["evidence_class"],
                       "geometry": "exact mask-to-vector, 100 um2 minimum part/hole area"},
    })
    result = export_experiment(previous, destination, base_dir=base_project.parent)
    matching = [row for row in result["tissues"] if row["tissue_id"] == tissue["tissue_id"]]
    if len(matching) != 1 or not any(row["artifact_id"] == artifact_id for row in matching[0]["artifacts"]):
        raise ValueError("Published project lost the exact Bora review boundary")
    return {"schema": "cellphenotyper-bora-project-export-v1", "complete": True,
            "project": str(destination), "project_sha256": sha256(destination),
            "bora_chain_sha256": sha256(chain_path), "tissue_id": tissue["tissue_id"],
            "artifact_id": artifact_id, "artifact_role": "bora_exploratory_boundary_review",
            "review_only": True, "algorithm_id": GENERALIZED_ALGORITHM,
            "algorithm_evidence_receipt_sha256": chain["algorithm_evidence_receipt_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bora-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-algorithm", help=(
        "Defaults to bora-global-hard-group-v1; colon H&E-guided export requires a separate contract."))
    args = parser.parse_args()
    print(json.dumps(export(args.base_project, args.manifest, args.bora_output,
                            args.output, args.requested_algorithm), sort_keys=True))


if __name__ == "__main__":
    main()
