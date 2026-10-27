"""Fail-closed handoff between one Nextflow CPU task and Bora's global CLI.

The Bora CLI performs the authoritative, source-bound admission.  This small
wrapper checks the pipeline intent before launch and the published result
afterwards; it never starts an encoder case or retries a failed chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


REQUIRED_OUTPUTS = (
    "ADMISSION.json",
    "CHAIN_COMPLETE.json",
    "rgb/image/rgb.npy",
    "af/readout/group_map.npy",
    "af/readout/APPEARANCE_FIT_COMPLETE.json",
    "descriptors.npz",
    "semantic/semantic-mask.uint8.dat",
    "semantic/ASSEMBLY_COMPLETE.json",
    "vectors/exact/mask.uint8.dat",
    "vectors/exact/mask.geojson",
    "vectors/review/mask.geojson",
    "vectors/GLOBAL_VECTOR_COMPLETE.json",
)

# This route assigns a hard tissue identity to each complete Bora-AF group.
# The colon H&E boundary candidate is a different, currently unqualified
# whole-slide algorithm and must never be inferred from a smooth GeoJSON.
GENERALIZED_ALGORITHM = "bora-global-hard-group-v1"
COLON_HE_ALGORITHM = "bora-colon-he-guided-v1"


def require_generalized_algorithm(spec: dict, requested_algorithm: str | None = None) -> str:
    declared = spec.get("algorithm_id")
    requested = declared if requested_algorithm is None else requested_algorithm
    if declared != GENERALIZED_ALGORITHM or requested != GENERALIZED_ALGORITHM:
        if declared == COLON_HE_ALGORITHM or requested == COLON_HE_ALGORITHM:
            raise ValueError("The colon H&E-guided Bora method has no source-bound whole-slide completion contract")
        if declared is None:
            raise ValueError("Declare algorithm_id explicitly; the old hard-group Bora method cannot be an implicit default or stand in for refined semantic Bora")
        raise ValueError("Unsupported or mismatched Bora algorithm identity; refined semantic Bora requires its separate H&E-refined chain")
    return GENERALIZED_ALGORITHM


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _geojson_header(path: Path) -> dict:
    """Read only Bora's bounded metadata prefix, never the full WSI geometry."""
    with path.open("rb") as stream:
        prefix = stream.read(64 * 1024)
    marker = b',"features":['
    end = prefix.find(marker)
    if end < 0:
        raise ValueError("Bora GeoJSON lacks its bounded source metadata header")
    try:
        return json.loads(prefix[:end] + b"}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Bora GeoJSON source metadata is malformed") from exc


def native_geometry_identity(spec: dict, output: Path) -> dict:
    """Bind native mask/vector coordinates to the verified dual-MPP source."""
    inputs = spec.get("inputs", {})
    registration_ref = inputs.get("registration", {}) if isinstance(inputs, dict) else {}
    if not isinstance(registration_ref, dict):
        raise ValueError("Bora registration binding is missing")
    registration_path = Path(registration_ref.get("path") or "")
    if (not registration_path.is_absolute() or not registration_path.is_file() or
            sha256(registration_path) != registration_ref.get("sha256")):
        raise ValueError("Bora registration differs from the source-bound manifest")
    registration = json.loads(registration_path.read_text())
    if registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or registration.get("verified") is not True:
        raise ValueError("Bora requires verified dual-MPP registration")
    working = registration.get("image")
    native = registration.get("cell_full_image")
    if not isinstance(working, dict) or not isinstance(native, dict):
        raise ValueError("Bora registration lacks working or native image identity")
    if (working.get("frame_id") != spec.get("frame_id") or
            working.get("sha256") != spec.get("expected", {}).get("image_sha256")):
        raise ValueError("Bora working image differs from the registered manifest frame")
    assembly = json.loads((output / "semantic/ASSEMBLY_COMPLETE.json").read_text())
    vector = json.loads((output / "vectors/GLOBAL_VECTOR_COMPLETE.json").read_text())
    native_shape = [native.get("height"), native.get("width")]
    native_mpp = [native.get("mpp_x"), native.get("mpp_y")]
    if (any(type(value) is not int or value <= 0 for value in native_shape) or
            any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in native_mpp) or
            not isinstance(native.get("frame_id"), str) or not native["frame_id"] or
            not isinstance(native.get("sha256"), str) or len(native["sha256"]) != 64):
        raise ValueError("Bora native registration has invalid image identity")
    expected = dict(native_shape_yx=native_shape, native_mpp_xy=native_mpp,
                    source_frame_id=native["frame_id"], source_image_sha256=native["sha256"])
    for label, record in (("assembly", assembly), ("vector", vector),
                          ("exact GeoJSON", _geojson_header(output / "vectors/exact/mask.geojson")),
                          ("review GeoJSON", _geojson_header(output / "vectors/review/mask.geojson"))):
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Bora {label} native frame, image, shape or MPP differs from registration")
        if label.endswith("GeoJSON") and (record.get("type") != "FeatureCollection" or
                record.get("coordinate_space") != "native_image_pixel_edges" or
                record.get("origin") != "top_left" or
                record.get("source_semantic_mask_sha256") != assembly.get("semantic_mask_sha256") or
                record.get("global_model_binding_sha256") != assembly.get("global_model_binding_sha256")):
            raise ValueError(f"Bora {label} coordinate/source metadata is invalid")
    if (assembly.get("working_frame_id") != working["frame_id"] or
            assembly.get("working_image_sha256") != working["sha256"] or
            assembly.get("source_sha256", {}).get("registration") != sha256(registration_path) or
            vector.get("source_semantic_mask_sha256") != assembly.get("semantic_mask_sha256") or
            vector.get("global_model_binding_sha256") != assembly.get("global_model_binding_sha256")):
        raise ValueError("Bora native geometry differs from the bound working source or semantic mask")
    return expected


def prepare(manifest_path: Path, output: Path, required_group_count: int | None = None,
            build_receipt_path: Path | None = None,
            requested_algorithm: str | None = None) -> dict:
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("Use an existing absolute Bora manifest path")
    if not output.is_absolute() or output.exists():
        raise ValueError("Bora output must be an absolute, new path; no overwrite/retry")
    if not output.parent.is_dir():
        raise ValueError("The Bora output parent directory must exist for container binding")
    spec = json.loads(manifest_path.read_text())
    require_generalized_algorithm(spec, requested_algorithm)
    if build_receipt_path is not None:
        receipt = json.loads(build_receipt_path.read_text())
        if (receipt.get("schema") != "cellphenotyper-bora-manifest-build-v1" or
                receipt.get("complete") is not True or
                receipt.get("manifest_sha256") != sha256(manifest_path) or
                receipt.get("inventory_sha256") != spec.get("expected", {}).get("inventory_sha256") or
                receipt.get("evidence_class") != "verified_gpu_inventory"):
            raise ValueError("Manifest build receipt is absent or differs from the admitted source")
    if spec.get("schema") != "bora-wsi-global-v1":
        raise ValueError("Expected the generalized Bora v1 manifest")
    run_root = Path(spec.get("run_root", ""))
    if run_root.is_absolute():
        output_root = output.resolve()
        run_root = run_root.resolve()
        if output_root == run_root or output_root.is_relative_to(run_root) or run_root.is_relative_to(output_root):
            raise ValueError("Bora output and source GPU case root must be separate")
    expected = spec.get("expected", {})
    if expected.get("scale_factors") != [1, 2, 4] or expected.get("case_count", 0) < 1:
        raise ValueError("A complete three-scale token inventory is required")
    if spec.get("evidence_class") != "verified_gpu_inventory":
        raise ValueError("Real full-slide runs require verified GPU inventory evidence")
    if spec.get("identity_status") not in {"exploratory_transfer", "externally_validated"}:
        raise ValueError("Declare the classifier transfer evidence status")
    inputs = spec.get("inputs", {})
    receipt = inputs.get("cluster_receipt", {})
    cluster = inputs.get("cluster_csv", {})
    if not receipt.get("path") or not cluster.get("path"):
        raise ValueError("The source-bound KODAMA receipt and CSV are required")
    receipt_path, csv_path = Path(receipt["path"]), Path(cluster["path"])
    if not receipt_path.is_absolute() or not csv_path.is_absolute():
        raise ValueError("KODAMA input paths must be absolute")
    if not receipt_path.is_file() or not csv_path.is_file():
        raise ValueError("KODAMA input files are missing")
    if sha256(receipt_path) != receipt.get("sha256") or sha256(csv_path) != cluster.get("sha256"):
        raise ValueError("KODAMA files differ from the manifest hashes")
    provenance = json.loads(receipt_path.read_text())
    groups = expected.get("group_count")
    if (provenance.get("native_fit_reused") is not True or
            provenance.get("posthoc_merge") is not False or
            provenance.get("requested_groups") != groups or
            provenance.get("obtained_groups") != groups or
            provenance.get("csv_sha256") != cluster["sha256"]):
        raise ValueError("KODAMA must be a direct fit of exactly the supplied group CSV")
    if required_group_count is not None and groups != required_group_count:
        raise ValueError("KODAMA group count differs from the requested direct fit")
    return spec


def verify(manifest_path: Path, output: Path, spec: dict,
           requested_algorithm: str | None = None) -> dict:
    algorithm = require_generalized_algorithm(spec, requested_algorithm)
    if not output.is_dir():
        raise ValueError("Bora output directory was not created")
    absent = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if absent:
        raise ValueError("Bora chain is incomplete: " + ", ".join(absent))
    chain = json.loads((output / "CHAIN_COMPLETE.json").read_text())
    admission = json.loads((output / "ADMISSION.json").read_text())
    vector = json.loads((output / "vectors/GLOBAL_VECTOR_COMPLETE.json").read_text())
    assembly_path = output / "semantic/ASSEMBLY_COMPLETE.json"
    assembly = json.loads(assembly_path.read_text())
    semantic_hash = sha256(output / "semantic/semantic-mask.uint8.dat")
    source = assembly.get("source_sha256", {})
    inputs = spec.get("inputs", {})
    if not isinstance(source, dict) or not isinstance(inputs, dict):
        raise ValueError("Bora semantic source binding is malformed")
    if (chain.get("algorithm_id", algorithm) != algorithm or
            assembly.get("format") != "bora-global-group-materialization-v1" or
            assembly.get("prediction_unit") != "global_anonymous_Bora-AF_group" or
            assembly.get("probability_shards_persisted") is not False or
            assembly.get("global_fit_count_declared") != 1 or
            assembly.get("window_fit_count_declared") != 0 or
            assembly.get("independently_fitted_windows") is not False or
            assembly.get("working_frame_id") != spec.get("frame_id") or
            assembly.get("working_image_sha256") != spec.get("expected", {}).get("image_sha256") or
            assembly.get("semantic_mask_sha256") != semantic_hash or
            assembly.get("tissue_identity_status") != chain.get("tissue_identity_status") or
            source.get("model") != spec.get("expected", {}).get("classifier_sha256") or
            source.get("model") != inputs.get("classifier_model", {}).get("sha256") or
            source.get("global_group_map") != sha256(output / "af/readout/group_map.npy") or
            source.get("global_appearance_fit") != sha256(output / "af/readout/APPEARANCE_FIT_COMPLETE.json") or
            source.get("global_descriptors") != sha256(output / "descriptors.npz") or
            source.get("registration") != inputs.get("registration", {}).get("sha256") or
            source.get("support") != inputs.get("support_mask", {}).get("sha256") or
            vector.get("source_semantic_mask_sha256") != semantic_hash or
            vector.get("global_model_binding_sha256") != assembly.get("global_model_binding_sha256")):
        raise ValueError("Bora semantic output is not a source-bound generalized hard-group prediction")
    if (chain.get("schema") != "bora-wsi-global-chain-result-v1" or
            chain.get("complete") is not True or
            chain.get("manifest_sha256") != sha256(manifest_path) or
            chain.get("inventory_sha256") != spec["expected"]["inventory_sha256"] or
            chain.get("sample_id") != spec["sample_id"] or
            chain.get("frame_id") != spec["frame_id"] or
            chain.get("evidence_class") != "verified_gpu_inventory" or
            chain.get("gpu_execution_qualified") is not True or
            chain.get("exact_100_um2_floor") is not True or
            chain.get("exact_raster_parity") is not True or
            admission.get("inventory_sha256") != chain["inventory_sha256"] or
            chain.get("vector_receipt_sha256") != sha256(output / "vectors/GLOBAL_VECTOR_COMPLETE.json") or
            vector.get("minimum_area_um2") != 100.0 or
            vector.get("exact_vector_matches_cleaned_mask") is not True):
        raise ValueError("Bora completion lacks source, inventory, geometry or raster parity")
    for relative in ("exact/mask.uint8.dat", "exact/mask.geojson", "review/mask.geojson"):
        path = output / "vectors" / relative
        if (vector.get("output_sha256", {}).get(relative) != sha256(path) or
                chain.get("output_sha256", {}).get(relative) != sha256(path)):
            raise ValueError(f"Bora vector hash differs: {relative}")
    native_geometry_identity(spec, output)
    return dict(chain, verified_algorithm_id=algorithm,
                algorithm_evidence_receipt_sha256=sha256(assembly_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bora-executable", default="bora")
    parser.add_argument("--required-group-count", type=int)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--requested-algorithm", help=(
        "Explicit algorithm request; defaults to bora-global-hard-group-v1. "
        "The colon H&E-guided route is blocked until its own completion contract exists."))
    args = parser.parse_args()
    spec = prepare(args.manifest, args.output, args.required_group_count,
                   args.build_receipt, args.requested_algorithm)
    subprocess.run([args.bora_executable, "wsi-global", "--manifest", str(args.manifest),
                    "--output", str(args.output)], check=True)
    print(json.dumps(verify(args.manifest, args.output, spec, args.requested_algorithm), sort_keys=True))


if __name__ == "__main__":
    main()
