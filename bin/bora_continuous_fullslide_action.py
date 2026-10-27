"""CPU-only, source-pinned continuous Bora continuation for one complete slide.

The input is a *finished* global RGB9/RGB19 replay.  This action never fits a
model, requests a GPU, retries a shard, or assigns biological tissue labels.
It creates one new result directory; interrupted/failed directories are kept
for inspection and can never be resumed or overwritten by this action.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback


FORMAT = "cellphenotyper-bora-continuous-fullslide-run-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
MIB = 1024**2
GIB = 1024**3
MAX_NATIVE_PIXELS = 1_000_000_000
MAX_SHARDS = 4096
DEPENDENCIES = {
    "numpy": "numpy", "scipy": "scipy", "rasterio": "rasterio",
    "shapely": "shapely", "scikit-image": "skimage", "Pillow": "PIL",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: str, name: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"{name} requires a lowercase SHA-256 digest")
    return value


def _json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MIB:
        raise ValueError(f"Missing or oversized input receipt: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Input receipt must be an object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    pending = path.with_name("." + path.name + ".tmp")
    with pending.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def bora_source_identity(root: Path) -> dict:
    """Hash the *whole* Python source tree and package metadata, not a version."""
    root = Path(root).resolve(strict=True)
    package = root / "src" / "bora"
    metadata_path = root / "pyproject.toml"
    if not package.is_dir() or not metadata_path.is_file():
        raise ValueError("Staged Bora source lacks src/bora or pyproject.toml")
    paths = [metadata_path, *sorted(package.rglob("*.py"))]
    if len(paths) < 3 or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Staged Bora source contains missing/symlinked source")
    files = {str(path.relative_to(root)): sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return {"root": str(root), "sha256": digest, "file_count": len(files),
            "files": files}


def runtime_identity(sif: Path | None = None) -> dict:
    """Bind the *running* interpreter and imported library installations.

    A pinned SIF is an additional source artifact; its presence does not by
    itself prove that this Python process is executing inside that SIF.
    """
    executable = Path(sys.executable).resolve(strict=True)
    deps = {}
    for distribution, module in DEPENDENCIES.items():
        spec = util.find_spec(module)
        if spec is None or spec.origin is None:
            raise ValueError(f"Required runtime dependency missing: {distribution}")
        origin = Path(spec.origin).resolve(strict=True)
        deps[distribution] = {"version": metadata.version(distribution),
                              "module_file": str(origin),
                              "module_sha256": sha256(origin)}
    result = {"executor_kind": "running_python_process",
              "python_executable": str(executable),
              "python_executable_sha256": sha256(executable),
              "python_version": sys.version,
              "dependencies": deps,
              "execution_environment": {name: os.environ.get(name) for name in
                  ("PYTHONNOUSERSITE", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                   "ROCR_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "pinned_sif": None}
    if sif is not None:
        sif = Path(sif).resolve(strict=True)
        if sif.suffix != ".sif" or not sif.is_file():
            raise ValueError("Pinned runtime image must be an existing .sif")
        result["pinned_sif"] = {"path": str(sif), "sha256": sha256(sif),
                                "execution_proven": False}
    return result


def identity_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def core_grid(shape_yx: tuple[int, int], core_pixels: int, halo_pixels: int,
              class_count: int) -> list[tuple[int, int, int, int]]:
    height, width = shape_yx
    if (type(core_pixels) is not int or not 1 <= core_pixels <= 3072 or
            type(halo_pixels) is not int or not 1 <= halo_pixels <= 512 or
            type(class_count) is not int or not 2 <= class_count <= 32 or
            type(height) is not int or type(width) is not int or
            min(height, width) < 1 or height * width > MAX_NATIVE_PIXELS):
        raise ValueError("Invalid bounded continuous slide/core parameters")
    cores = [(y, x, min(core_pixels, height-y), min(core_pixels, width-x))
             for y in range(0, height, core_pixels)
             for x in range(0, width, core_pixels)]
    if len(cores) > MAX_SHARDS:
        raise ValueError("Continuous slide requires more than 4096 shards")
    for y, x, h, w in cores:
        padded_h = min(height, y+h+halo_pixels) - max(0, y-halo_pixels)
        padded_w = min(width, x+w+halo_pixels) - max(0, x-halo_pixels)
        if padded_h * padded_w > 16 * MIB:
            raise ValueError("A padded shard exceeds Bora's 16M-pixel window bound")
    return cores


def _estimated_bytes(shape_yx: tuple[int, int], cores, halo_pixels,
                     class_count: int) -> int:
    height, width = shape_yx
    shards = 0
    for y, x, h, w in cores:
        ph = min(height, y+h+halo_pixels) - max(0, y-halo_pixels)
        pw = min(width, x+w+halo_pixels) - max(0, x-halo_pixels)
        shards += ph * pw * class_count * 4 + MIB
    # The action keeps source-bound support.npy, while the global chain
    # independently rematerializes support and writes assembly/vector masks.
    return shards + 8 * height * width + 2 * GIB


def _load_bora(source_root: Path):
    src = (source_root / "src").resolve(strict=True)
    if "bora" in sys.modules:
        current = Path(sys.modules["bora"].__file__).resolve()
        if src not in current.parents:
            raise ValueError("Another Bora installation is already imported")
    sys.path.insert(0, str(src))
    from bora import wsi_continuous_chain, wsi_continuous_probability_adapter
    for module in (wsi_continuous_chain, wsi_continuous_probability_adapter):
        if src not in Path(module.__file__).resolve().parents:
            raise ValueError("Bora module resolved outside the pinned source tree")
    return wsi_continuous_chain, wsi_continuous_probability_adapter


def run(*, replay_folder: Path, registration_path: Path,
        registered_support_tiff: Path, base_project: Path,
        bora_source: Path, expected_bora_sha256: str,
        expected_runtime_sha256: str, output_folder: Path,
        runtime_sif: Path | None = None,
        core_pixels: int = 2048, halo_pixels: int = 128,
        p9_weight: float = .5, disk_budget_bytes: int = 16 * GIB,
        memory_budget_bytes: int = 512 * MIB) -> dict:
    """Generate every shard once, globally assemble/vectorize, then export."""
    source_root = Path(bora_source).resolve(strict=True)
    source = bora_source_identity(source_root)
    if source["sha256"] != _sha(expected_bora_sha256, "Expected Bora source"):
        raise ValueError("Pinned Bora source-tree SHA-256 differs from requested source")
    runtime = runtime_identity(runtime_sif)
    runtime_sha = identity_digest(runtime)
    if runtime_sha != _sha(expected_runtime_sha256, "Expected runtime"):
        raise ValueError("Running interpreter/dependency/SIF identity differs from request")
    if type(disk_budget_bytes) is not int or disk_budget_bytes < 4 * GIB or \
            type(memory_budget_bytes) is not int or memory_budget_bytes < 64 * MIB or \
            type(p9_weight) not in (float, int) or not math.isfinite(p9_weight) or \
            not 0 <= p9_weight <= 1:
        raise ValueError("Invalid finite CPU resource or RGB9 weight bound")
    chain, adapter = _load_bora(source_root)
    replay_folder = Path(replay_folder).resolve(strict=True)
    registration_path = Path(registration_path).resolve(strict=True)
    registered_support_tiff = Path(registered_support_tiff).resolve(strict=True)
    base_project = Path(base_project).resolve(strict=True)
    registration, _ = adapter._registration(registration_path)
    native = registration["cell_full_image"]
    working = registration["image"]
    shape = native["height"], native["width"]
    replay = _json(replay_folder / "TOKEN_REPLAY_COMPLETE.json")
    classes = replay.get("class_ids")
    if (replay.get("format") != adapter.REPLAY_FORMAT or
            replay.get("method") != "continuous-deployment-head-v1" or
            replay.get("working_shape_yx") != [working["height"], working["width"]] or
            replay.get("token_shape_yx") != [(working["height"]+13)//14,
                                               (working["width"]+13)//14] or
            replay.get("whole_field_recipe") is not True or
            replay.get("independently_fitted_windows") is not False or
            replay.get("probability_regularization_applied") is not False or
            not isinstance(classes, list) or len(classes) < 2 or len(classes) > 32):
        raise ValueError("Incomplete or incompatible frozen RGB9/RGB19 replay")
    hashes = replay.get("output_sha256", {})
    if not isinstance(hashes, dict) or set(hashes) != {"p9", "p19", "combined"}:
        raise ValueError("Frozen replay output inventory is incomplete")
    replay_hashes = {}
    for name in ("p9", "p19", "combined"):
        path = replay_folder / (name + ".npy")
        digest = sha256(path)
        if digest != _sha(hashes[name], name + " replay"):
            raise ValueError(f"Frozen {name} replay bytes changed")
        replay_hashes[name] = digest
    source_inputs = {"registration_sha256": sha256(registration_path),
                     "registered_support_tiff_sha256": sha256(registered_support_tiff),
                     "base_project_sha256": sha256(base_project),
                     "replay_receipt_sha256": sha256(replay_folder / "TOKEN_REPLAY_COMPLETE.json"),
                     "replay_output_sha256": replay_hashes}
    if source_inputs["registered_support_tiff_sha256"] != registration["full_support_mask"]["sha256"]:
        raise ValueError("Registered GrandQC TIFF differs from registration")
    cores = core_grid(shape, core_pixels, halo_pixels, len(classes))
    estimate = _estimated_bytes(shape, cores, halo_pixels, len(classes))
    output_folder = Path(output_folder).absolute()
    if output_folder.exists():
        raise FileExistsError(output_folder)
    if not output_folder.parent.is_dir() or source_root in output_folder.parents or \
            replay_folder in output_folder.parents:
        raise ValueError("Output needs an existing parent outside frozen input/source trees")
    if (disk_budget_bytes < estimate or
            shutil.disk_usage(output_folder.parent).free < estimate):
        raise ValueError("Insufficient declared or available disk for all shards and global masks")
    if memory_budget_bytes < 5 * ((min(core_pixels + 2*halo_pixels, shape[0]) *
                                   min(core_pixels + 2*halo_pixels, shape[1]) *
                                   len(classes) * 4) + MIB):
        raise ValueError("Insufficient declared memory for the largest probability shard")
    output_folder.mkdir()
    stage = "preflight"
    shards = []
    try:
        preflight = {"format": FORMAT, "complete": False,
                     "source": source, "runtime": runtime,
                     "runtime_sha256": runtime_sha,
                     "input_paths": {"replay_folder": str(replay_folder),
                                     "registration": str(registration_path),
                                     "registered_support_tiff": str(registered_support_tiff),
                                     "base_project": str(base_project)},
                     "input_sha256": source_inputs,
                     "native_shape_yx": list(shape),
                     "native_frame_id": native["frame_id"],
                     "native_mpp_xy": [native["mpp_x"], native["mpp_y"]],
                     "class_ids": classes,
                     "core_pixels": core_pixels, "halo_pixels": halo_pixels,
                     "p9_weight": float(p9_weight), "shard_count": len(cores),
                     "disk_budget_bytes": disk_budget_bytes,
                     "memory_budget_bytes": memory_budget_bytes,
                     "estimated_peak_output_bytes": estimate,
                     "cpu_only": True, "retry_count": 0,
                     "biological_accuracy_qualified": False}
        _write_json(output_folder / "PREFLIGHT.json", preflight)
        stage = "native_support"
        (output_folder / "inputs").mkdir()
        support_path = output_folder / "inputs" / "support.npy"
        chain._write_native_support(registered_support_tiff, support_path,
            shape, memory_budget_bytes=memory_budget_bytes)
        support_sha = sha256(support_path)
        (output_folder / "shards").mkdir()
        for index, core in enumerate(cores):
            stage = f"shard_{index:04d}"
            if bora_source_identity(source_root)["sha256"] != source["sha256"]:
                raise ValueError("Pinned Bora source changed between shards")
            folder = output_folder / "shards" / f"{index:04d}"
            adapter.produce_continuous_probability_shard(
                replay_folder, registration_path, support_path, support_sha,
                core, halo_pixels, folder, p9_weight=p9_weight,
                disk_budget_bytes=disk_budget_bytes,
                memory_budget_bytes=memory_budget_bytes)
            shards.append(folder)
        stage = "global_chain"
        if bora_source_identity(source_root)["sha256"] != source["sha256"]:
            raise ValueError("Pinned Bora source changed before global chain")
        chain_receipt = chain.run_continuous_global_chain(
            shards, registration_path, registered_support_tiff,
            output_folder / "global", disk_budget_bytes=disk_budget_bytes,
            memory_budget_bytes=memory_budget_bytes, max_shards=MAX_SHARDS)
        if chain_receipt.get("shard_count") != len(cores) or \
                chain_receipt.get("review_only") is not True:
            raise ValueError("Global continuous chain did not complete all review-only shards")
        stage = "project_export"
        if bora_source_identity(source_root)["sha256"] != source["sha256"] or \
                identity_digest(runtime_identity(runtime_sif)) != runtime_sha:
            raise ValueError("Pinned Bora source or runtime changed before project publication")
        if (sha256(base_project) != source_inputs["base_project_sha256"] or
                sha256(registration_path) != source_inputs["registration_sha256"] or
                sha256(registered_support_tiff) != source_inputs["registered_support_tiff_sha256"] or
                sha256(replay_folder / "TOKEN_REPLAY_COMPLETE.json") != source_inputs["replay_receipt_sha256"] or
                any(sha256(replay_folder / (name + ".npy")) != digest
                    for name, digest in replay_hashes.items())):
            raise ValueError("Frozen input changed before project publication")
        if sha256(support_path) != support_sha:
            raise ValueError("Native support changed before project publication")
        if __package__:
            from .export_bora_continuous_project import export
            from .export_experiment import validate_experiment
        else:
            from export_bora_continuous_project import export
            from export_experiment import validate_experiment
        pending = output_folder / "project.pending.cellphenotyper.json"
        export(base_project, registration_path, registered_support_tiff,
               output_folder / "global", pending)
        if bora_source_identity(source_root)["sha256"] != source["sha256"] or \
                identity_digest(runtime_identity(runtime_sif)) != runtime_sha:
            raise ValueError("Pinned source/runtime changed during project export")
        final = output_folder / "experiment.cellphenotyper.json"
        pending.rename(final)
        validate_experiment(final)
        stage = "completion"
        result = {"format": FORMAT, "complete": True, "cpu_only": True,
                  "review_only": True,
                  "biological_accuracy_qualified": False,
                  "source_tree_sha256": source["sha256"],
                  "runtime_sha256": runtime_sha,
                  "preflight_sha256": sha256(output_folder / "PREFLIGHT.json"),
                  "shard_count": len(cores),
                  "shard_receipt_sha256": [sha256(folder / "SHARD_COMPLETE.json")
                                            for folder in shards],
                  "chain_receipt_sha256": sha256(output_folder / "global" /
                                                  "CONTINUOUS_CHAIN_COMPLETE.json"),
                  "project_path": str(final), "project_sha256": sha256(final),
                  "retry_count": 0}
        _write_json(output_folder / "RUN_COMPLETE.json", result)
        return result
    except BaseException as exc:
        published = output_folder / "experiment.cellphenotyper.json"
        if published.is_file() and not (output_folder / "RUN_COMPLETE.json").is_file():
            try:
                published.rename(output_folder / "project.unaccepted.cellphenotyper.json")
            except OSError:
                pass
        failure = {"format": FORMAT, "complete": False,
                   "failed_stage": stage, "exception_type": type(exc).__name__,
                   "message": str(exc)[:2000],
                   "traceback": traceback.format_exc(limit=8)[-8000:],
                   "completed_shards": len(shards),
                   "completed_shard_receipt_sha256": [sha256(folder / "SHARD_COMPLETE.json")
                        for folder in shards if (folder / "SHARD_COMPLETE.json").is_file()],
                   "retry_count": 0, "output_preserved": True}
        try:
            _write_json(output_folder / "RUN_FAILED.json", failure)
        except OSError:
            # A full disk can prevent even a small failure receipt. Preserve
            # the original error and the unfinished folder for diagnosis.
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("identity", help="Print source/runtime pins before a run")
    identity.add_argument("--bora-source", required=True, type=Path)
    identity.add_argument("--runtime-sif", type=Path)
    launch = sub.add_parser("run", help="Run exactly one new CPU-only slide folder")
    for option in ("replay-folder", "registration", "registered-support-tiff",
                   "base-project", "bora-source", "output"):
        launch.add_argument("--" + option, required=True, type=Path)
    launch.add_argument("--expected-bora-sha256", required=True)
    launch.add_argument("--expected-runtime-sha256", required=True)
    launch.add_argument("--runtime-sif", type=Path)
    launch.add_argument("--core-pixels", type=int, default=2048)
    launch.add_argument("--halo-pixels", type=int, default=128)
    launch.add_argument("--p9-weight", type=float, default=.5)
    launch.add_argument("--disk-budget-bytes", type=int, default=16 * GIB)
    launch.add_argument("--memory-budget-bytes", type=int, default=512 * MIB)
    args = parser.parse_args()
    if args.command == "identity":
        source = bora_source_identity(args.bora_source)
        runtime = runtime_identity(args.runtime_sif)
        print(json.dumps({"bora_source_sha256": source["sha256"],
                          "runtime_sha256": identity_digest(runtime),
                          "runtime": runtime}, sort_keys=True))
        return
    result = run(replay_folder=args.replay_folder,
        registration_path=args.registration,
        registered_support_tiff=args.registered_support_tiff,
        base_project=args.base_project, bora_source=args.bora_source,
        expected_bora_sha256=args.expected_bora_sha256,
        expected_runtime_sha256=args.expected_runtime_sha256,
        output_folder=args.output, runtime_sif=args.runtime_sif,
        core_pixels=args.core_pixels, halo_pixels=args.halo_pixels,
        p9_weight=args.p9_weight, disk_budget_bytes=args.disk_budget_bytes,
        memory_budget_bytes=args.memory_budget_bytes)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
