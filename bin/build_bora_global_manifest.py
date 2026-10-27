"""Build one Bora v1 manifest from completed, source-bound slide evidence.

No hash or group count is supplied as authority. The imported Bora admission
validates every token case and its lifecycle before BUILD_COMPLETE is written.
This script starts no GPU work and never retries a failed build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

INPUT_NAMES = frozenset({"registration", "grid_csv", "cluster_csv", "cluster_receipt",
                         "grid_metadata", "support_mask", "working_image", "model",
                         "classifier_model"})


def stages(group_count: int) -> tuple[str, ...]:
    return (("coarse", "observations", "selected", "rgb9", "rgb19", "replay",
             "rgb_image", "evidence", "af_inputs") +
            tuple(f"af_{index}" for index in range(1, group_count + 1)) +
            ("af_readout", "descriptors", "classifier", "vectors"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _file(value: object, label: str) -> Path:
    path = Path(value) if isinstance(value, str) else Path()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be one existing absolute file")
    return path.resolve()


def _directory(value: object, label: str) -> Path:
    path = Path(value) if isinstance(value, str) else Path()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} must be one existing absolute directory")
    return path.resolve()


def build(spec: dict, *, package_root: Path | None = None,
          source_tree_digest_fn=None, capture_runtime_fn=None) -> dict:
    if spec.get("schema") != "cellphenotyper-bora-global-inputs-v1":
        raise ValueError("Expected a versioned Bora input specification")
    if spec.get("evidence_class") != "verified_gpu_inventory":
        raise ValueError("Real-slide manifest building requires verified GPU inventory evidence")
    run_root = _directory(spec.get("run_root"), "run_root")
    producer = _directory(spec.get("producer_source"), "producer_source")
    plan_path = _file(str(run_root / "plans/spatial-chunks.json"), "frozen plan")
    inventory_path = _file(str(run_root / "output/spatial-chunks/CHUNK_INVENTORY.json"), "complete inventory")
    plan = json.loads(plan_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("all_cleanup_verified") is not True or
            inventory.get("completed_case_count") != plan.get("chunk_plan", {}).get("case_count") or
            inventory.get("assembled_tensor_written") is not False):
        raise ValueError("Token inventory is missing, incomplete or lacks lifecycle cleanup proof")
    scales = [row.get("factor") for row in plan.get("chunk_plan", {}).get("scales", [])]
    if scales != [1, 2, 4]:
        raise ValueError("The frozen producer does not declare all three scales [1,2,4]")
    raw_inputs = spec.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != INPUT_NAMES:
        raise ValueError("Declare exactly the nine Bora coordinate, image and model inputs")
    inputs = {name: {"path": str(_file(raw_inputs[name], name)),
                     "sha256": digest(_file(raw_inputs[name], name))} for name in sorted(INPUT_NAMES)}
    cluster = json.loads(Path(inputs["cluster_receipt"]["path"]).read_text())
    groups = cluster.get("obtained_groups")
    if (type(groups) is not int or not 2 <= groups <= 32 or
            cluster.get("native_fit_reused") is not True or
            cluster.get("posthoc_merge") is not False or
            cluster.get("requested_groups") != groups or
            cluster.get("csv_sha256") != inputs["cluster_csv"]["sha256"] or
            cluster.get("original_grid_sha256") != inputs["grid_csv"]["sha256"] or
            cluster.get("original_image_sha256") != inputs["working_image"]["sha256"]):
        raise ValueError("KODAMA receipt is not a direct fit bound to these grid/image bytes")
    required_groups = spec.get("required_group_count")
    if required_groups is not None and (type(required_groups) is not int or required_groups != groups):
        raise ValueError("Direct KODAMA group count differs from the requested slide count")
    if package_root is None or source_tree_digest_fn is None or capture_runtime_fn is None:
        import bora
        from bora.wsi_global_manifest import source_tree_digest
        from bora.wsi_global_runtime import capture_runtime
        package_root = package_root or Path(bora.__file__).resolve().parent
        source_tree_digest_fn = source_tree_digest_fn or source_tree_digest
        capture_runtime_fn = capture_runtime_fn or capture_runtime
    package = Path(package_root).resolve()
    fastpls_site = _directory(spec.get("fastpls_site"), "fastpls_site")
    vendor = fastpls_site / "bora/_vendor/fastpls"
    cores = sorted(vendor.glob("_core*.so"))
    if len(cores) != 1:
        raise ValueError("Pinned fastPLS site requires exactly one native core extension")
    libraries = spec.get("native_libraries")
    if not isinstance(libraries, list) or not libraries:
        raise ValueError("Declare the CPU runtime's native library paths")
    libraries = [str(_file(value, "native library")) for value in libraries]
    limits = spec.get("limits")
    if not isinstance(limits, dict) or set(limits.get("stage_seconds", {})) != set(stages(groups)):
        raise ValueError("Every CPU Bora stage needs its own finite limit")
    expected = {
        "plan_sha256": digest(plan_path), "inventory_sha256": digest(inventory_path),
        "canonical_sha256": plan["chunk_plan_canonical_sha256"],
        "producer_sha256": plan["package_sha256"]["chunk_producer.py"],
        "image_sha256": plan["input_sha256"]["image"],
        "model_sha256": plan["input_sha256"]["model"],
        "classifier_sha256": inputs["classifier_model"]["sha256"],
        "case_count": plan["chunk_plan"]["case_count"], "scale_factors": scales,
        "group_count": groups, "case_prefix": plan["case_prefix"],
        "working_frame_id": plan["working_frame_id"],
    }
    result = {
        "schema": "bora-wsi-global-v1", "sample_id": spec["sample_id"],
        "frame_id": plan["working_frame_id"],
        "identity_status": spec["identity_status"],
        "evidence_class": "verified_gpu_inventory",
        "run_root": str(run_root), "producer_source": str(producer),
        "expected": expected,
        "runtime": {
            "bora_package_root": str(package),
            "source_tree_sha256": source_tree_digest_fn(package),
            "cpu": capture_runtime_fn(libraries),
            "fastpls": {"site": str(fastpls_site), "core_sha256": digest(cores[0]),
                        "api_sha256": digest(_file(str(vendor / "api.py"), "fastPLS API")),
                        "init_sha256": digest(_file(str(vendor / "__init__.py"), "fastPLS package"))},
        },
        "inputs": inputs, "palette_rgb": spec["palette_rgb"], "limits": limits,
    }
    return result


def publish(spec: dict, output: Path, *, build_fn=build, load_manifest_fn=None,
            admit_fn=None) -> dict:
    if not output.is_absolute() or output.exists() or not output.parent.is_dir():
        raise ValueError("Manifest output must be one new absolute directory with an existing parent")
    manifest = build_fn(spec)
    output.mkdir()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    try:
        if load_manifest_fn is None or admit_fn is None:
            from bora.wsi_global_admit import admit
            from bora.wsi_global_manifest import load_manifest
            load_manifest_fn = load_manifest_fn or load_manifest
            admit_fn = admit_fn or admit
        normalized = load_manifest_fn(manifest_path)
        admitted = admit_fn(normalized)
        if admitted.get("admitted") is not True or admitted.get("gpu_execution_qualified") is not True:
            raise ValueError("Full authentic GPU inventory admission failed")
        receipt = {"schema": "cellphenotyper-bora-manifest-build-v1", "complete": True,
                   "manifest_sha256": digest(manifest_path),
                   "inventory_sha256": manifest["expected"]["inventory_sha256"],
                   "plan_sha256": manifest["expected"]["plan_sha256"],
                   "source_tree_sha256": manifest["runtime"]["source_tree_sha256"],
                   "sample_id": manifest["sample_id"], "frame_id": manifest["frame_id"],
                   "group_count": manifest["expected"]["group_count"],
                   "case_count": manifest["expected"]["case_count"],
                   "evidence_class": "verified_gpu_inventory"}
        (output / "BUILD_COMPLETE.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        return receipt
    except BaseException as exc:
        (output / "BUILD_FAILED.json").write_text(json.dumps(
            {"complete": False, "no_automatic_retry": True,
             "reason": f"{type(exc).__name__}: {exc}"}, sort_keys=True) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source = _file(str(args.inputs), "input specification")
    print(json.dumps(publish(json.loads(source.read_text()), args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
