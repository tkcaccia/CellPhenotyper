"""Bounded absolute-grid reconstruction and registered H&E edge padding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from bin import materialize_bora_semantic_tokens as module
from bin.export_experiment import sha256


SOURCE_ROOT = Path(os.environ.get("BORA_SEMANTIC_SOURCE_ROOT",
    str(Path(__file__).resolve().parents[2] / "Bora" / "src")))
pytestmark = pytest.mark.skipif(
    not (SOURCE_ROOT / "bora" / "chunk_plan.py").is_file(),
    reason="Optional Bora semantic integration source unavailable; set BORA_SEMANTIC_SOURCE_ROOT")


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def _frame(path, mpp=.5):
    with Image.open(path) as image:
        w, h = image.size
    value = {"sha256": sha256(path), "width": w, "height": h,
        "mpp_x": mpp, "mpp_y": mpp, "origin_x": 0, "origin_y": 0,
        "coordinate_space": "crop-native-pixels",
        "coordinate_convention": "pixel-edge-origin_centres-in-native-pixels"}
    value["frame_id"] = "cellvit-frame-" + hashlib.sha256(
        json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]
    return value


def _fixture(root):
    if not (SOURCE_ROOT / "bora/chunk_plan.py").is_file():
        pytest.skip("Optional Bora integration source is unavailable")
    sys.path.insert(0, str(SOURCE_ROOT))
    from bora.chunk_plan import make_chunk_plan
    from bora.chunk_store import relative_name

    h, w = 17, 20
    he = np.empty((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            he[y, x] = [x + 10, y + 20, x + y + 30]
    native = root / "native.tif"
    working = root / "working.tif"
    Image.fromarray(he).save(native)
    Image.fromarray(he).save(working)
    he_path = root / "he.npy"
    np.save(he_path, he)
    registration = {"format": "cellphenotyper-dual-mpp-registration-v1",
        "verified": True,
        "support_sampling": "absolute pixel centre, nearest, full-frame GrandQC mask",
        "cell_full_image": _frame(native), "image": _frame(working),
        "tissue_to_cell_full_affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    reg_path = _write_json(root / "registration.json", registration)
    evidence_path = _write_json(root / "he-evidence.json", {
        "format": "coordinator-bora-global-evidence-v1",
        "source_sha256": {"registration": sha256(reg_path)},
        "output_sha256": {"working_he": sha256(he_path)},
        "working_frame_id": registration["image"]["frame_id"],
        "working_image_sha256": registration["image"]["sha256"],
        "working_shape_yx": [h, w]})
    checkpoint = root / "checkpoint.bin"
    checkpoint.write_bytes(b"pinned synthetic Virchow2 checkpoint identity")
    plan = make_chunk_plan(h, w)
    chunk_root = root / "frozen-chunks"
    chunk_root.mkdir()
    hashes = {}
    expected = np.empty((2, 2, 1280), np.float32)
    for case in plan["cases"]:
        shape = case["chunk_shape_yxc"]
        arr = np.empty(shape, np.float32)
        for y in range(shape[0]):
            for x in range(shape[1]):
                global_y = case["token_y0"] + y
                global_x = case["token_x0"] + x
                arr[y, x] = case["scale_factor"] * 100 + global_y * 10 + global_x
        path = chunk_root / relative_name(case)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        hashes[str(case["case_id"])] = sha256(path)
        if case["scale_factor"] == 1:
            expected[case["token_y0"]:case["token_y1"],
                     case["token_x0"]:case["token_x1"]] = arr
    inventory_path = _write_json(chunk_root / "CHUNK_INVENTORY.json", {
        "format": "coordinator-bora-token-chunk-inventory-v1",
        "plan": plan, "sha256_by_case": hashes,
        "working_frame_id": registration["image"]["frame_id"],
        "working_image_sha256": registration["image"]["sha256"],
        "checkpoint_sha256": sha256(checkpoint),
        "completed_case_count": plan["case_count"],
        "all_cleanup_verified": True,
        "one_global_absolute_field": True})
    source_hashes = {name: sha256(SOURCE_ROOT / "bora" / name)
                     for name in module.SOURCE_MODULES}
    files = {"registration": reg_path, "native_image": native,
             "working_image": working, "working_he": he_path,
             "he_evidence": evidence_path, "token_inventory": inventory_path,
             "virchow2_checkpoint": checkpoint}
    spec = {"format": module.SPEC_FORMAT,
        "bora_source_root": str(SOURCE_ROOT),
        "bora_source_sha256": source_hashes,
        "inputs": {key: {"path": str(path), "sha256": sha256(path)}
                   for key, path in files.items()}}
    spec_path = _write_json(root / "spec.json", spec)
    return {"spec": spec_path, "files": files, "chunk_root": chunk_root,
            "he": he, "expected": expected, "plan": plan}


@pytest.mark.skipif(not (SOURCE_ROOT / "bora" / "chunk_plan.py").is_file(),
                    reason="Reviewed Bora chunk source is unavailable")
def test_reconstructs_exact_grid_and_white_padded_he_edges(tmp_path):
    case = _fixture(tmp_path)
    output = tmp_path / "materialized"
    receipt = module.materialize(case["spec"], output,
        memory_budget_bytes=16 * 1024**2, disk_budget_bytes=10 * 1024**2)
    tokens = np.load(output / module.OUTPUTS["virchow2_tokens"])
    he_tokens = np.load(output / module.OUTPUTS["he_tokens"])
    np.testing.assert_array_equal(tokens, case["expected"])
    assert tokens.dtype == np.float32 and he_tokens.dtype == np.float32
    assert he_tokens.shape == (2, 2, 3)
    padded = np.full((14, 14, 3), 255, np.uint8)
    padded[:3, :6] = case["he"][14:17, 14:20]
    np.testing.assert_allclose(he_tokens[1, 1], padded.mean((0, 1)) / 255,
                               rtol=0, atol=5e-8)
    assert receipt["all_scale_chunk_hashes_verified"] is True
    assert receipt["no_chunk_gap_or_overlap"] is True
    assert receipt["source_sha256"]["token_inventory"] == sha256(case["files"]["token_inventory"])
    assert receipt["output_sha256"]["virchow2_tokens"] == \
        sha256(output / module.OUTPUTS["virchow2_tokens"])
    assert json.loads((output / "MATERIALIZATION_COMPLETE.json").read_text()) == receipt
    verified, _ = module.verify_materialization(output)
    assert verified == receipt


@pytest.mark.parametrize("problem", ["scale1_hash", "scale2_hash", "scale4_hash",
                                    "plan_gap", "scale2_shape", "he_hash",
                                    "checkpoint_hash", "frame_mismatch", "incomplete"])
def test_rejects_incomplete_or_tampered_frozen_source(tmp_path, problem):
    case = _fixture(tmp_path)
    inventory_path = case["files"]["token_inventory"]
    inventory = json.loads(inventory_path.read_text())
    spec = json.loads(case["spec"].read_text())
    if problem.endswith("_hash") and problem.startswith("scale"):
        factor = int(problem[5])
        cid = next(c["case_id"] for c in case["plan"]["cases"]
                   if c["scale_factor"] == factor)
        chunk = case["chunk_root"] / f"scale-{factor}" / f"case-{cid:05d}.npy"
        with chunk.open("ab") as stream:
            stream.write(b"tamper")
    elif problem == "plan_gap":
        inventory["plan"]["cases"][0]["token_x0"] = 1
        _write_json(inventory_path, inventory)
        spec["inputs"]["token_inventory"]["sha256"] = sha256(inventory_path)
        _write_json(case["spec"], spec)
    elif problem == "scale2_shape":
        inventory["plan"]["scales"][1]["token_shape_yxc"][0] += 1
        _write_json(inventory_path, inventory)
        spec["inputs"]["token_inventory"]["sha256"] = sha256(inventory_path)
        _write_json(case["spec"], spec)
    elif problem == "he_hash":
        with case["files"]["working_he"].open("ab") as stream:
            stream.write(b"tamper")
    elif problem == "checkpoint_hash":
        case["files"]["virchow2_checkpoint"].write_bytes(b"replacement")
    elif problem == "frame_mismatch":
        inventory["working_frame_id"] = "different-frame"
        _write_json(inventory_path, inventory)
        spec["inputs"]["token_inventory"]["sha256"] = sha256(inventory_path)
        _write_json(case["spec"], spec)
    else:
        inventory["all_cleanup_verified"] = False
        _write_json(inventory_path, inventory)
        spec["inputs"]["token_inventory"]["sha256"] = sha256(inventory_path)
        _write_json(case["spec"], spec)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        module.materialize(case["spec"], output,
            memory_budget_bytes=16 * 1024**2, disk_budget_bytes=10 * 1024**2)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path):
    case = _fixture(tmp_path)
    output = tmp_path / "keep"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="new absolute output folder"):
        module.materialize(case["spec"], output)
    assert sentinel.read_text() == "unchanged"
