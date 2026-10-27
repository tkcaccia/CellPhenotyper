"""Manifest hashes and groups must come from frozen bytes, never a caller guess."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bin.build_bora_global_manifest import build, digest, publish, stages


def fixture(tmp_path: Path):
    root = tmp_path / "tokens"
    (root / "plans").mkdir(parents=True)
    (root / "output/spatial-chunks").mkdir(parents=True)
    producer = tmp_path / "producer"
    producer.mkdir()
    paths = {}
    for name in ("registration", "grid_csv", "cluster_csv", "cluster_receipt",
                 "grid_metadata", "support_mask", "working_image", "model",
                 "classifier_model"):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = path
    receipt = {"native_fit_reused": True, "posthoc_merge": False,
               "requested_groups": 2, "obtained_groups": 2,
               "csv_sha256": digest(paths["cluster_csv"]),
               "original_grid_sha256": digest(paths["grid_csv"]),
               "original_image_sha256": digest(paths["working_image"])}
    paths["cluster_receipt"].write_text(json.dumps(receipt))
    plan = {"chunk_plan": {"case_count": 3, "scales": [{"factor": x} for x in (1, 2, 4)]},
            "chunk_plan_canonical_sha256": "a" * 64,
            "package_sha256": {"chunk_producer.py": "b" * 64},
            "input_sha256": {"image": digest(paths["working_image"]),
                             "model": digest(paths["model"])},
            "case_prefix": "fixed", "working_frame_id": "frame"}
    (root / "plans/spatial-chunks.json").write_text(json.dumps(plan))
    inventory = {"all_cleanup_verified": True, "completed_case_count": 3,
                 "assembled_tensor_written": False}
    (root / "output/spatial-chunks/CHUNK_INVENTORY.json").write_text(json.dumps(inventory))
    fastpls = tmp_path / "fastpls"
    vendor = fastpls / "bora/_vendor/fastpls"
    vendor.mkdir(parents=True)
    for name in ("_core.test.so", "api.py", "__init__.py"):
        (vendor / name).write_text(name)
    native = tmp_path / "libvips.so"
    native.write_text("native")
    spec = {"schema": "cellphenotyper-bora-global-inputs-v1", "sample_id": "PM016",
            "identity_status": "exploratory_transfer", "evidence_class": "verified_gpu_inventory",
            "run_root": str(root), "producer_source": str(producer),
            "inputs": {name: str(path) for name, path in paths.items()},
            "fastpls_site": str(fastpls), "native_libraries": [str(native)],
            "required_group_count": 2, "palette_rgb": [[210, 35, 75], [35, 170, 70]],
            "limits": {"total_seconds": 7200, "admission_seconds": 900,
                       "stage_seconds": {name: 360 for name in stages(2)},
                       "memory_budget_bytes": 1024**3,
                       "stage_disk_budget_bytes": 2 * 1024**3,
                       "minimum_free_bytes": 0}}
    extras = {"package_root": producer, "source_tree_digest_fn": lambda _: "c" * 64,
              "capture_runtime_fn": lambda libraries: {"pinned": libraries}}
    return spec, paths, extras


def test_manifest_derives_exact_source_inventory_group_and_input_hashes(tmp_path: Path):
    spec, paths, extras = fixture(tmp_path)
    result = build(spec, **extras)
    assert result["schema"] == "bora-wsi-global-v1"
    assert result["expected"]["case_count"] == 3
    assert result["expected"]["group_count"] == 2
    assert result["expected"]["scale_factors"] == [1, 2, 4]
    assert result["expected"]["image_sha256"] == digest(paths["working_image"])
    assert result["inputs"]["cluster_receipt"]["sha256"] == digest(paths["cluster_receipt"])
    assert result["runtime"]["source_tree_sha256"] == "c" * 64


@pytest.mark.parametrize("mutation,message", [
    ("incomplete", "cleanup"), ("one_scale", "three scales"),
    ("merged", "direct fit"), ("wrong_groups", "group count"),
])
def test_incomplete_or_mismatched_inputs_fail_closed(tmp_path: Path, mutation: str, message: str):
    spec, paths, extras = fixture(tmp_path)
    root = Path(spec["run_root"])
    if mutation == "incomplete":
        inventory = root / "output/spatial-chunks/CHUNK_INVENTORY.json"
        data = json.loads(inventory.read_text())
        data["all_cleanup_verified"] = False
        inventory.write_text(json.dumps(data))
    elif mutation == "one_scale":
        plan = root / "plans/spatial-chunks.json"
        data = json.loads(plan.read_text())
        data["chunk_plan"]["scales"] = [{"factor": 1}]
        plan.write_text(json.dumps(data))
    elif mutation == "merged":
        receipt = paths["cluster_receipt"]
        data = json.loads(receipt.read_text())
        data["posthoc_merge"] = True
        receipt.write_text(json.dumps(data))
    else:
        spec["required_group_count"] = 3
    with pytest.raises(ValueError, match=message):
        build(spec, **extras)


def test_synthetic_inventory_cannot_claim_real_gpu_proof(tmp_path: Path):
    spec, _, extras = fixture(tmp_path)
    spec["evidence_class"] = "cpu_synthetic_fixture"
    with pytest.raises(ValueError, match="verified GPU"):
        build(spec, **extras)


def test_frozen_pm016_shape_derives_real_plan_identity_but_needs_authoritative_admission(tmp_path: Path):
    spec, paths, extras = fixture(tmp_path)
    plan_path = Path(spec["run_root"]) / "plans/spatial-chunks.json"
    plan = json.loads(plan_path.read_text())
    plan.update(working_frame_id="cellvit-frame-f19a898f7318e2725c960901",
                working_mpp_xy=[0.500003036313874, 0.5000017316490221],
                source_shape_yx=[7381, 11952], case_prefix="PM016-bora-native")
    plan["chunk_plan"]["case_count"] = 60
    plan["package_sha256"]["chunk_producer.py"] = "770a77d684a98ce87bc129debe2c6d26df60131b09d782ecdc3cde9b0cfe92e0"
    plan_path.write_text(json.dumps(plan))
    inventory_path = Path(spec["run_root"]) / "output/spatial-chunks/CHUNK_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["completed_case_count"] = 60
    inventory_path.write_text(json.dumps(inventory))
    result = build(spec, **extras)
    assert result["expected"]["case_count"] == 60
    assert result["expected"]["producer_sha256"] == plan["package_sha256"]["chunk_producer.py"]
    assert result["frame_id"] == plan["working_frame_id"]
    assert result["expected"]["plan_sha256"] == digest(plan_path)
    output = tmp_path / "proposed-manifest"
    with pytest.raises(ValueError, match="Imported verifier differs from frozen producer"):
        publish(spec, output, build_fn=lambda _: result,
                load_manifest_fn=lambda path: json.loads(path.read_text()),
                admit_fn=lambda _: (_ for _ in ()).throw(
                    ValueError("Imported verifier differs from frozen producer")))
    assert not (output / "BUILD_COMPLETE.json").exists()
    failed = json.loads((output / "BUILD_FAILED.json").read_text())
    assert failed["complete"] is False and failed["no_automatic_retry"] is True
