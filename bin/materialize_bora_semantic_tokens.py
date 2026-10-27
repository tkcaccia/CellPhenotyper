"""Materialize source-bound Virchow2/H&E semantic token fields without a GPU.

Read a complete three-scale frozen chunk inventory, copy only scale-1
Virchow2 spatial tokens into one disk-backed YXC-1280 float32 field, and
calculate the registered H&E mean in identical 14×14 working-pixel cells.
Output is a new immutable folder with a hash-bound receipt; this stage does
not fit a classifier, predict tissue labels, or modify a project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

if __package__:
    from .bora_semantic_roi_intake import _frame, _source_modules, SOURCE_MODULES
    from .export_experiment import sha256
else:
    from bora_semantic_roi_intake import _frame, _source_modules, SOURCE_MODULES
    from export_experiment import sha256


FORMAT = "cellphenotyper-bora-semantic-token-materialization-v1"
SPEC_FORMAT = "cellphenotyper-bora-semantic-token-materialization-spec-v1"
INPUTS = ("registration", "native_image", "working_image", "working_he",
          "he_evidence", "token_inventory", "virchow2_checkpoint")
OUTPUTS = {"virchow2_tokens": "virchow2-spatial-tokens.npy",
           "he_tokens": "registered-he-tokens.npy"}
STRIDE = 14
CHANNELS = 1280
MAX_SPEC_BYTES = 2 * 1024 * 1024
MIN_MEMORY = 16 * 1024 * 1024


def _json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SPEC_BYTES:
        raise ValueError(f"Missing, symlinked or oversized JSON: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _ref(value: dict, name: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{name} requires an absolute path and SHA-256")
    path = Path(value["path"])
    digest = value["sha256"]
    if (not path.is_absolute() or not path.is_file() or path.is_symlink() or
            not isinstance(digest, str) or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest) or
            sha256(path) != digest):
        raise ValueError(f"Frozen {name} source bytes differ")
    return path.resolve(strict=True)


def _input_contract(spec: dict) -> tuple[dict, dict, dict, dict]:
    if spec.get("format") != SPEC_FORMAT or not isinstance(spec.get("inputs"), dict) or \
            set(spec["inputs"]) != set(INPUTS):
        raise ValueError("Complete frozen semantic token materialization spec required")
    inputs = {name: _ref(spec["inputs"][name], name) for name in INPUTS}
    registration = _json(inputs["registration"])
    if (registration.get("format") != "cellphenotyper-dual-mpp-registration-v1" or
            registration.get("verified") is not True or
            registration.get("support_sampling") !=
                "absolute pixel centre, nearest, full-frame GrandQC mask"):
        raise ValueError("Verified dual-MPP registration is required")
    native = registration.get("cell_full_image")
    working = registration.get("image")
    nh, nw, nmx, nmy = _frame(native, inputs["native_image"])
    wh, ww, wmx, wmy = _frame(working, inputs["working_image"])
    sx, sy = nw / ww, nh / wh
    affine = registration.get("tissue_to_cell_full_affine")
    expected = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
    if (not isinstance(affine, list) or len(affine) != 3 or
            any(not isinstance(row, list) or len(row) != 3 for row in affine) or
            any(not math.isclose(float(actual), wanted, rel_tol=1e-12, abs_tol=1e-9)
                for row, goal in zip(affine, expected) for actual, wanted in zip(row, goal)) or
            not math.isclose(nw * nmx, ww * wmx, rel_tol=1e-6, abs_tol=1e-6) or
            not math.isclose(nh * nmy, wh * wmy, rel_tol=1e-6, abs_tol=1e-6)):
        raise ValueError("Native/working physical field or pixel-edge affine differs")
    evidence = _json(inputs["he_evidence"])
    if (evidence.get("format") != "coordinator-bora-global-evidence-v1" or
            evidence.get("source_sha256", {}).get("registration") !=
                spec["inputs"]["registration"]["sha256"] or
            evidence.get("output_sha256", {}).get("working_he") !=
                spec["inputs"]["working_he"]["sha256"] or
            evidence.get("working_frame_id") != working["frame_id"] or
            evidence.get("working_image_sha256") != working["sha256"] or
            evidence.get("working_shape_yx") != [wh, ww]):
        raise ValueError("Registered H&E evidence differs from working image")
    he = np.load(inputs["working_he"], mmap_mode="r", allow_pickle=False)
    if he.shape != (wh, ww, 3) or he.dtype != np.uint8:
        raise ValueError("Registered working H&E must be YXC-3 uint8")
    inventory = _json(inputs["token_inventory"])
    if (inventory.get("format") != "coordinator-bora-token-chunk-inventory-v1" or
            inventory.get("working_frame_id") != working["frame_id"] or
            inventory.get("working_image_sha256") != working["sha256"] or
            inventory.get("checkpoint_sha256") !=
                spec["inputs"]["virchow2_checkpoint"]["sha256"] or
            inventory.get("all_cleanup_verified") is not True or
            inventory.get("one_global_absolute_field") is not True or
            inventory.get("completed_case_count") !=
                inventory.get("plan", {}).get("case_count")):
        raise ValueError("Incomplete, unregistered or wrong-checkpoint token inventory")
    root = Path(spec.get("bora_source_root", ""))
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("An absolute reviewed Bora source root is required")
    modules = _source_modules(root)
    if (not isinstance(spec.get("bora_source_sha256"), dict) or
            set(spec["bora_source_sha256"]) != set(SOURCE_MODULES) or
            spec["bora_source_sha256"] != modules["sha256"]):
        raise ValueError("Reviewed Bora chunk reader/plan source hashes differ")
    if inputs["token_inventory"].name != "CHUNK_INVENTORY.json":
        raise ValueError("Token inventory must be the frozen chunk-root inventory")
    grids = {factor: modules["token_grid"].ChunkGrid(
        inputs["token_inventory"].parent / f"scale-{factor}", factor=factor)
        for factor in (1, 2, 4)}
    expected_shape = (math.ceil(wh / STRIDE), math.ceil(ww / STRIDE), CHANNELS)
    if grids[1].shape != expected_shape:
        raise ValueError("Scale-1 token geometry differs from registered working H&E")
    for grid in grids.values():
        grid.verify_all()
    if sha256(inputs["token_inventory"]) != spec["inputs"]["token_inventory"]["sha256"]:
        raise ValueError("Chunk inventory changed during verification")
    return inputs, registration, grids, dict(native=native, working=working,
                                             he=he, shape=expected_shape)


def materialize(spec_path: Path, output: Path, *,
                memory_budget_bytes: int = 128 * 1024**2,
                disk_budget_bytes: int = 16 * 1024**3) -> dict:
    spec_path = Path(spec_path)
    output = Path(output)
    if (not spec_path.is_absolute() or not output.is_absolute() or
            output.exists() or output.is_symlink() or not output.parent.is_dir()):
        raise ValueError("Use an absolute frozen spec and a new absolute output folder")
    if (type(memory_budget_bytes) is not int or memory_budget_bytes < MIN_MEMORY or
            type(disk_budget_bytes) is not int or disk_budget_bytes < 1):
        raise ValueError("A finite positive disk and >=16 MiB memory budget is required")
    spec = _json(spec_path)
    spec_sha = sha256(spec_path)
    inputs, registration, grids, context = _input_contract(spec)
    inventory_root = inputs["token_inventory"].parent
    source_root = Path(spec["bora_source_root"])
    resolved_output = output.resolve(strict=False)
    if resolved_output.is_relative_to(inventory_root) or \
            resolved_output.is_relative_to(source_root.resolve(strict=True)):
        raise ValueError("Materialized output must be outside frozen chunks and source code")
    th, tw, _ = context["shape"]
    wh, ww = context["working"]["height"], context["working"]["width"]
    total_bytes = th * tw * (CHANNELS + 3) * 4 + 4096
    if total_bytes > disk_budget_bytes or \
            shutil.disk_usage(output.parent).free < total_bytes + 1024**3:
        raise ValueError("Insufficient disk budget/free space for two token fields")
    row_bytes = tw * CHANNELS * 4
    rows = min(16, (memory_budget_bytes // 2) // row_bytes,
               (256 * 1024**2) // row_bytes)
    if rows < 1:
        raise ValueError("One complete token row exceeds bounded memory")
    source_sha = sha256(Path(__file__))
    staged = Path(tempfile.mkdtemp(prefix=".semantic-token-materialization-",
                                   dir=output.parent))
    try:
        token_path = staged / OUTPUTS["virchow2_tokens"]
        he_path = staged / OUTPUTS["he_tokens"]
        token_out = np.lib.format.open_memmap(token_path, mode="w+",
                                              dtype=np.float32, shape=(th, tw, CHANNELS))
        he_out = np.lib.format.open_memmap(he_path, mode="w+",
                                           dtype=np.float32, shape=(th, tw, 3))
        for y in range(0, th, rows):
            y1 = min(y + rows, th)
            token_out[y:y1] = grids[1].read_block(y, y1, 0, tw)
            for row in range(y, y1):
                he_strip = np.full((STRIDE, tw * STRIDE, 3), 255, np.uint8)
                actual = max(0, min(STRIDE, wh - row * STRIDE))
                if actual:
                    he_strip[:actual, :ww] = context["he"][row * STRIDE:
                                                     row * STRIDE + actual]
                he_out[row] = he_strip.reshape(STRIDE, tw, STRIDE, 3).mean(
                    axis=(0, 2), dtype=np.float64) / 255.0
        token_out.flush(); he_out.flush()
        del token_out, he_out
        for grid in grids.values():
            grid.verify_all()
        if (sha256(spec_path) != spec_sha or
                any(sha256(inputs[name]) != spec["inputs"][name]["sha256"]
                    for name in INPUTS) or
                sha256(Path(__file__)) != source_sha or
                any(sha256(source_root / "bora" / name) != digest
                    for name, digest in spec["bora_source_sha256"].items())):
            raise ValueError("A frozen semantic token source changed during materialization")
        outputs = {key: sha256(staged / relative) for key, relative in OUTPUTS.items()}
        receipt = dict(format=FORMAT, complete=True,
            source_spec_path=str(spec_path.resolve(strict=True)),
            source_spec_sha256=spec_sha,
            source_sha256={name: spec["inputs"][name]["sha256"] for name in INPUTS},
            bora_source_sha256=spec["bora_source_sha256"],
            materializer_sha256=source_sha,
            registration_sha256=spec["inputs"]["registration"]["sha256"],
            inventory_sha256=spec["inputs"]["token_inventory"]["sha256"],
            checkpoint_sha256=spec["inputs"]["virchow2_checkpoint"]["sha256"],
            working_frame_id=context["working"]["frame_id"],
            working_image_sha256=context["working"]["sha256"],
            native_frame_id=context["native"]["frame_id"],
            native_image_sha256=context["native"]["sha256"],
            working_shape_yx=[wh, ww], token_shape_yxc=list(context["shape"]),
            he_token_shape_yxc=[th, tw, 3], token_stride_working_yx=[14, 14],
            token_origin_working_yx=[0, 0],
            token_layout="YXC-1280 float32 scale-1 exact chunk placement",
            he_rule="registered uint8 H&E white-padded at true bottom/right edge; 14x14 mean divided by 255",
            all_scale_chunk_hashes_verified=True,
            no_chunk_gap_or_overlap=True,
            scale_2_and_4_metadata_verified=True,
            output_paths=OUTPUTS, output_sha256=outputs,
            memory_budget_bytes=memory_budget_bytes,
            disk_budget_bytes=disk_budget_bytes)
        receipt_path = staged / "MATERIALIZATION_COMPLETE.json"
        with receipt_path.open("w") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if sum(path.stat().st_size for path in staged.iterdir()) > disk_budget_bytes:
            raise ValueError("Materialized outputs exceeded the declared disk budget")
        os.rename(staged, output)
        return receipt
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def verify_materialization(folder: Path) -> tuple[dict, dict]:
    """Recheck the frozen sources, chunk hashes and both published arrays."""
    folder = Path(folder)
    receipt_path = folder / "MATERIALIZATION_COMPLETE.json"
    receipt = _json(receipt_path)
    if (receipt.get("format") != FORMAT or receipt.get("complete") is not True or
            receipt.get("all_scale_chunk_hashes_verified") is not True or
            receipt.get("no_chunk_gap_or_overlap") is not True or
            receipt.get("scale_2_and_4_metadata_verified") is not True or
            receipt.get("output_paths") != OUTPUTS or
            not isinstance(receipt.get("output_sha256"), dict) or
            set(receipt["output_sha256"]) != set(OUTPUTS)):
        raise ValueError("Semantic token materialization receipt is incomplete")
    spec_path = Path(receipt.get("source_spec_path", ""))
    if (not spec_path.is_absolute() or
            sha256(spec_path) != receipt.get("source_spec_sha256")):
        raise ValueError("Semantic token materialization spec changed")
    spec = _json(spec_path)
    inputs, registration, grids, context = _input_contract(spec)
    if (receipt.get("source_sha256") != {
            name: spec["inputs"][name]["sha256"] for name in INPUTS} or
            receipt.get("bora_source_sha256") != spec["bora_source_sha256"] or
            receipt.get("materializer_sha256") != sha256(Path(__file__)) or
            receipt.get("registration_sha256") !=
                spec["inputs"]["registration"]["sha256"] or
            receipt.get("inventory_sha256") !=
                spec["inputs"]["token_inventory"]["sha256"] or
            receipt.get("checkpoint_sha256") !=
                spec["inputs"]["virchow2_checkpoint"]["sha256"] or
            receipt.get("working_frame_id") != context["working"]["frame_id"] or
            receipt.get("working_image_sha256") != context["working"]["sha256"] or
            receipt.get("native_frame_id") != context["native"]["frame_id"] or
            receipt.get("native_image_sha256") != context["native"]["sha256"] or
            receipt.get("working_shape_yx") !=
                [context["working"]["height"], context["working"]["width"]] or
            receipt.get("token_shape_yxc") != list(context["shape"]) or
            receipt.get("he_token_shape_yxc") !=
                [context["shape"][0], context["shape"][1], 3] or
            receipt.get("token_stride_working_yx") != [14, 14] or
            receipt.get("token_origin_working_yx") != [0, 0]):
        raise ValueError("Semantic materialization source/frame contract changed")
    for key, relative in OUTPUTS.items():
        path = folder / relative
        if (not path.is_file() or path.is_symlink() or
                sha256(path) != receipt["output_sha256"].get(key)):
            raise ValueError(f"Semantic materialized {key} bytes changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        channels = CHANNELS if key == "virchow2_tokens" else 3
        if array.dtype != np.float32 or list(array.shape) != [
                context["shape"][0], context["shape"][1], channels]:
            raise ValueError(f"Semantic materialized {key} shape/dtype changed")
    return receipt, spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-budget-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--disk-budget-bytes", type=int, default=16 * 1024**3)
    args = parser.parse_args()
    print(json.dumps(materialize(args.spec, args.output,
        memory_budget_bytes=args.memory_budget_bytes,
        disk_budget_bytes=args.disk_budget_bytes), sort_keys=True))


if __name__ == "__main__":
    main()
