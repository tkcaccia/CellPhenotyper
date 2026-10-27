"""Opt-in CPU-only V5 manuscript transfer for one fully materialized slide.

This is separate from the expert-evaluated Bora route.  It never launches an
encoder, GPU job or retry, and it never publishes an expert annotation.  Each
new result folder is exclusive; a failed folder is retained for diagnosis.
Run under an independently bounded host supervisor for real full slides.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

if __package__:
    from .bora_continuous_fullslide_action import (
        GIB, MAX_NATIVE_PIXELS, MAX_SHARDS, bora_source_identity,
        identity_digest, runtime_identity, sha256, _write_json,
    )
    from .export_experiment import validate_experiment
    from .export_bora_manuscript_transfer_project import export
else:
    from bora_continuous_fullslide_action import (
        GIB, MAX_NATIVE_PIXELS, MAX_SHARDS, bora_source_identity,
        identity_digest, runtime_identity, sha256, _write_json,
    )
    from export_experiment import validate_experiment
    from export_bora_manuscript_transfer_project import export


FORMAT = "cellphenotyper-bora-manuscript-transfer-fullslide-v1"
V5_MODEL_SHA256 = "21a17b968c85db0d592cf1c014f9ef00d1019105732e54e3784afa00b067d847"
V5_POLICY_SHA256 = "6cf74e443e84f3832fca53863a50c84d728065337f6ff876cbb435209f305fbf"
V5_CENTROID_SHA256 = "fdd1396d58c1ab0fc32746c47a90eb4deaf1514db8c4038fd2b3f172f6b00044"
DEFAULT_REFINEMENT = {
    "format": "bora-local-he-boundary-v1", "uncertainty_margin": .30,
    "maximum_distance_um": 12.0, "he_blur_um": .5,
    "he_edge_sigma": .12, "smoothness": 1.5, "iterations": 4,
}


def _cpu_environment() -> None:
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(key) not in (None, "", "-1"):
            raise ValueError(f"CPU-only transfer refuses an allocated GPU environment: {key}")
        os.environ[key] = "-1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["PYTHONNOUSERSITE"] = "1"


def _runtime_identity() -> dict:
    """Pin the native OpenCV and sklearn code used beyond the base stack."""
    import cv2
    import sklearn

    result = runtime_identity()
    cv2_root = Path(cv2.__file__).resolve(strict=True).parent
    cv2_binaries = sorted(cv2_root.glob("cv2*.so"))
    if len(cv2_binaries) != 1:
        raise ValueError("Exactly one native OpenCV binary is required")
    result["semantic_dependencies"] = {
        "opencv": {"version": cv2.__version__,
                   "binary": str(cv2_binaries[0]),
                   "binary_sha256": sha256(cv2_binaries[0])},
        "scikit-learn": {"version": metadata.version("scikit-learn"),
                         "module": str(Path(sklearn.__file__).resolve(strict=True)),
                         "module_sha256": sha256(Path(sklearn.__file__))},
    }
    result["cellphenotyper_action_sha256"] = sha256(Path(__file__))
    return result


def _load_bora(source_root: Path):
    src = (source_root / "src").resolve(strict=True)
    if "bora" in sys.modules and src not in Path(sys.modules["bora"].__file__).resolve().parents:
        raise ValueError("Another Bora installation is already imported")
    sys.path.insert(0, str(src))
    from bora import (wsi_manuscript_transfer_registration,
                      wsi_manuscript_transfer, wsi_manuscript_transfer_native,
                      wsi_manuscript_transfer_vector,
                      wsi_semantic_materialization)
    modules = (wsi_manuscript_transfer_registration, wsi_manuscript_transfer,
               wsi_manuscript_transfer_native,
               wsi_manuscript_transfer_vector, wsi_semantic_materialization)
    if any(src not in Path(module.__file__).resolve().parents for module in modules):
        raise ValueError("Manuscript-transfer module resolved outside pinned Bora source")
    return modules


def _ordinary(path: Path, name: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be an absolute ordinary file")
    return path.resolve(strict=True)


def _materialize_native_support(source_tiff: Path, target_npy: Path,
                                registration: dict, *,
                                memory_budget_bytes: int) -> int:
    """Map a possibly coarse GrandQC TIFF onto exact native pixel centres."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    native = registration["cell_full_image"]
    source = registration["full_support_mask"]
    nh, nw = native["height"], native["width"]
    sh, sw = source["height"], source["width"]
    if (registration.get("support_sampling") !=
            "absolute pixel centre, nearest, full-frame GrandQC mask" or
            any(type(value) is not int or value < 1 for value in (nh, nw, sh, sw)) or
            sh > nh or sw > nw or
            type(memory_budget_bytes) is not int or memory_budget_bytes < 64*1024**2 or
            target_npy.exists() or target_npy.is_symlink() or
            sha256(source_tiff) != source["sha256"]):
        raise ValueError("Registered full-frame support source or CPU bound differs")
    output = np.lib.format.open_memmap(target_npy, mode="w+", dtype=np.uint8,
                                        shape=(nh, nw))
    supported = 0
    try:
        with rasterio.open(source_tiff) as dataset:
            if (dataset.count != 1 or dataset.dtypes[0] not in ("uint8", "uint16", "bool") or
                    (dataset.height, dataset.width) != (sh, sw)):
                raise ValueError("GrandQC source TIFF does not match registered coarse frame")
            allowed = ((0, 1, 255, 65535) if dataset.dtypes[0] == "uint16"
                       else (0, 1, 255))
            for y in range(0, nh, 128):
                y1 = min(y + 128, nh)
                sy = ((2*np.arange(y, y1, dtype=np.int64)+1)*sh)//(2*nh)
                for x in range(0, nw, 4096):
                    x1 = min(x + 4096, nw)
                    sx = ((2*np.arange(x, x1, dtype=np.int64)+1)*sw)//(2*nw)
                    sy0, sy1 = int(sy[0]), int(sy[-1]) + 1
                    sx0, sx1 = int(sx[0]), int(sx[-1]) + 1
                    raw = dataset.read(1, window=Window(sx0, sy0, sx1-sx0, sy1-sy0))
                    if not np.isin(raw, allowed).all():
                        raise ValueError("GrandQC support TIFF is not binary")
                    block = (raw[np.ix_(sy-sy0, sx-sx0)] > 0).astype(np.uint8)
                    output[y:y1, x:x1] = block
                    supported += int(np.count_nonzero(block))
        output.flush()
    finally:
        del output
    if supported < 1 or sha256(source_tiff) != source["sha256"]:
        raise ValueError("Registered GrandQC support changed or contains no tissue")
    return supported


def _core_grid(shape: tuple[int, int], core_pixels: int, context: tuple[int, int],
               output_halo_pixels: int) -> list[tuple[int, int, int, int]]:
    height, width = shape
    if (type(core_pixels) is not int or not 128 <= core_pixels <= 3072 or
            type(output_halo_pixels) is not int or not 1 <= output_halo_pixels <= 256 or
            any(type(value) is not int or value < 0 for value in context) or
            height * width > MAX_NATIVE_PIXELS):
        raise ValueError("Invalid bounded native core plan")
    cores = [(y, x, min(core_pixels, height-y), min(core_pixels, width-x))
             for y in range(0, height, core_pixels)
             for x in range(0, width, core_pixels)]
    if len(cores) > MAX_SHARDS:
        raise ValueError("Slide needs more than the permitted number of native shards")
    for y, x, h, w in cores:
        source_h = min(height, y+h+output_halo_pixels+context[0]) - \
            max(0, y-output_halo_pixels-context[0])
        source_w = min(width, x+w+output_halo_pixels+context[1]) - \
            max(0, x-output_halo_pixels-context[1])
        if source_h * source_w > 20_000_000:
            raise ValueError("One H&E refinement source window exceeds Bora's 20M-pixel limit")
    return cores


def _estimate_bytes(shape, cores, token_shape, output_halo_pixels):
    height, width = shape
    shard_pixels = sum((min(height, y+h+output_halo_pixels)-max(0, y-output_halo_pixels)) *
                       (min(width, x+w+output_halo_pixels)-max(0, x-output_halo_pixels))
                       for y, x, h, w in cores)
    # Three native shard byte masks, native support, two assembled byte masks,
    # exact raster, vectorization scratch and reserve for potentially complex
    # whole-slide GeoJSON.  No source image or token field is copied.
    return 3*shard_pixels + 7*height*width + 11*token_shape[0]*token_shape[1] + 2*GIB


def run(*, model_path: Path, policy_path: Path,
        materialization_folder: Path, registration_path: Path,
        registered_support_tiff: Path, base_project: Path,
        bora_source: Path, expected_bora_sha256: str,
        expected_runtime_sha256: str, output_folder: Path,
        acknowledge_exploratory: bool = False,
        core_pixels: int = 2048, output_halo_pixels: int = 3,
        memory_budget_bytes: int = 3*GIB,
        disk_budget_bytes: int = 16*GIB,
        deadline_seconds: int = 24*3600,
        refinement_spec: dict | None = None) -> dict:
    if acknowledge_exploratory is not True:
        raise ValueError("Explicit --acknowledge-exploratory is required")
    _cpu_environment()
    if (type(memory_budget_bytes) is not int or not GIB <= memory_budget_bytes <= 12*GIB or
            type(disk_budget_bytes) is not int or not GIB <= disk_budget_bytes <= 128*GIB or
            type(deadline_seconds) is not int or not 60 <= deadline_seconds <= 25*3600):
        raise ValueError("Finite CPU memory, disk and deadline bounds required")
    source_root = Path(bora_source).resolve(strict=True)
    source = bora_source_identity(source_root)
    if source["sha256"] != expected_bora_sha256:
        raise ValueError("Pinned Bora source tree differs from request")
    runtime = _runtime_identity()
    runtime_sha = identity_digest(runtime)
    if runtime_sha != expected_runtime_sha256:
        raise ValueError("Running Python/dependencies differ from request")
    (registration_module, scorer, native_module,
     vector_module, materialization_module) = _load_bora(source_root)
    model_path = _ordinary(model_path, "frozen V5 model")
    policy_path = _ordinary(policy_path, "frozen V5 policy")
    registration_path = _ordinary(registration_path, "dual-MPP registration")
    registered_support_tiff = _ordinary(registered_support_tiff, "native support TIFF")
    base_project = _ordinary(base_project, "base CellPhenotyper project")
    if (sha256(model_path) != V5_MODEL_SHA256 or
            sha256(policy_path) != V5_POLICY_SHA256 or
            sha256(policy_path.parent / "V5_TRANSFER_CENTROIDS.npz") != V5_CENTROID_SHA256):
        raise ValueError("Exact frozen V5 model, policy and centroid bytes required")
    materialization_folder = Path(materialization_folder).resolve(strict=True)
    materialized, _, material_sha = materialization_module.verify_semantic_materialization(
        materialization_folder, registration_path,
        memory_budget_bytes=memory_budget_bytes,
        registration_validator=registration_module._registration)
    registration, _ = registration_module._registration(registration_path)
    native, working = registration["cell_full_image"], registration["image"]
    if sha256(registered_support_tiff) != registration["full_support_mask"]["sha256"]:
        raise ValueError("Native support differs from verified registration")
    spec_path = Path(materialized["source_spec_path"])
    if sha256(spec_path) != materialized["source_spec_sha256"]:
        raise ValueError("Materialization specification changed")
    spec = json.loads(spec_path.read_text())
    native_he = _ordinary(Path(spec["inputs"]["native_image"]["path"]), "native H&E")
    if sha256(native_he) != native["sha256"]:
        raise ValueError("Native H&E differs from verified materialization")
    validate_experiment(base_project)
    shape = native["height"], native["width"]
    refinement, context = native_module._context_pixels(
        refinement_spec or DEFAULT_REFINEMENT, (native["mpp_x"], native["mpp_y"]))
    cores = _core_grid(shape, core_pixels, context, output_halo_pixels)
    token_shape = ((working["height"]+13)//14, (working["width"]+13)//14)
    estimated = _estimate_bytes(shape, cores, token_shape, output_halo_pixels)
    if estimated > disk_budget_bytes:
        raise ValueError("Declared disk budget cannot fit the full-slide output plan")
    output_folder = Path(output_folder)
    if not output_folder.is_absolute() or not output_folder.parent.is_dir() or \
            output_folder.exists() or output_folder.is_symlink():
        raise ValueError("Use a new absolute result folder with an existing parent")
    output_folder = output_folder.resolve(strict=False)
    if (any(output_folder.is_relative_to(path) for path in
            (source_root, materialization_folder, model_path.parent)) or
            shutil.disk_usage(output_folder.parent).free < estimated):
        raise ValueError("Result must be outside frozen inputs and have free disk")
    # The action is CPU-only, but its in-process native calls cannot prove a
    # finite shutdown after an OS/native hang. An outer host supervisor must
    # impose its own walltime and verify all owned processes before any retry.
    frozen = {"model_sha256": sha256(model_path),
              "policy_sha256": sha256(policy_path),
              "centroid_sha256": sha256(policy_path.parent / "V5_TRANSFER_CENTROIDS.npz"),
              "materialization_receipt_sha256": material_sha,
              "registration_sha256": sha256(registration_path),
              "support_tiff_sha256": sha256(registered_support_tiff),
              "native_he_sha256": sha256(native_he),
              "base_project_sha256": sha256(base_project)}
    start = time.monotonic()
    output_folder.mkdir()
    stage = "preflight"
    shards = []
    try:
        _write_json(output_folder / "PREFLIGHT.json", {
            "format": FORMAT, "complete": False, "cpu_only": True,
            "exploratory_opt_in": True, "biological_accuracy_qualified": False,
            "source": source, "runtime": runtime, "runtime_sha256": runtime_sha,
            "inputs_sha256": frozen, "native_shape_yx": list(shape),
            "native_mpp_xy": [native["mpp_x"], native["mpp_y"]],
            "registered_support_shape_yx":
                [registration["full_support_mask"]["height"],
                 registration["full_support_mask"]["width"]],
            "support_conversion":
                "registered-GrandQC-TIFF/absolute-centre-nearest-binary-to-native-uint8-NPY/v2",
            "core_pixels": core_pixels, "output_halo_pixels": output_halo_pixels,
            "shard_count": len(cores), "refinement_spec": refinement,
            "memory_budget_bytes": memory_budget_bytes,
            "disk_budget_bytes": disk_budget_bytes,
            "estimated_peak_output_bytes": estimated,
            "deadline_seconds": deadline_seconds, "retry_count": 0})
        stage = "native_support"
        support_path = output_folder / "support.npy"
        _materialize_native_support(registered_support_tiff, support_path,
            registration, memory_budget_bytes=memory_budget_bytes)
        support_sha = sha256(support_path)
        stage = "token_transfer"
        scorer.publish_manuscript_transfer_token_field(
            model_path, registration_path, materialization_folder, policy_path,
            output_folder / "tokens", memory_budget_bytes=memory_budget_bytes,
            disk_budget_bytes=disk_budget_bytes)
        token_receipt, _ = scorer.verify_manuscript_transfer_token_field(
            output_folder / "tokens", registration_path)
        if token_receipt.get("materialization_receipt_sha256") != material_sha:
            raise ValueError("Transfer scored a different materialization")
        (output_folder / "shards").mkdir()
        for index, core in enumerate(cores):
            if time.monotonic() - start > deadline_seconds - 120:
                raise TimeoutError("CPU deadline admission closed before next native shard")
            if bora_source_identity(source_root)["sha256"] != source["sha256"]:
                raise ValueError("Pinned Bora source changed between native shards")
            stage = f"native_shard_{index:04d}"
            folder = output_folder / "shards" / f"{index:04d}"
            native_module.publish_transfer_native_shard(
                output_folder / "tokens", registration_path, native_he,
                support_path, support_sha, core, folder, refinement,
                registered_support_tiff=registered_support_tiff,
                output_halo_pixels=output_halo_pixels,
                memory_budget_bytes=memory_budget_bytes,
                disk_budget_bytes=disk_budget_bytes)
            shards.append(folder)
        stage = "global_assembly"
        native_module.assemble_transfer_native_shards(
            shards, registration_path, support_path, support_sha,
            output_folder / "assembly",
            registered_support_tiff=registered_support_tiff,
            disk_budget_bytes=disk_budget_bytes)
        if time.monotonic() - start > deadline_seconds - 120:
            raise TimeoutError("CPU deadline admission closed before global vectorization")
        stage = "global_vector"
        vector_module.publish_transfer_global_vectors(
            output_folder / "assembly", support_path, support_sha,
            output_folder / "vectors", registration_path=registration_path,
            registered_support_tiff=registered_support_tiff,
            disk_budget_bytes=disk_budget_bytes)
        stage = "project_export"
        if (bora_source_identity(source_root)["sha256"] != source["sha256"] or
                identity_digest(_runtime_identity()) != runtime_sha or
                any(sha256(path) != digest for path, digest in (
                    (model_path, frozen["model_sha256"]),
                    (policy_path, frozen["policy_sha256"]),
                    (native_he, frozen["native_he_sha256"]),
                    (registration_path, frozen["registration_sha256"]),
                    (registered_support_tiff, frozen["support_tiff_sha256"]),
                    (base_project, frozen["base_project_sha256"]))) or
                sha256(materialization_folder / "MATERIALIZATION_COMPLETE.json") != material_sha):
            raise ValueError("Pinned source/runtime/inputs changed before project export")
        pending = output_folder / "project.pending.cellphenotyper.json"
        export(base_project, registration_path, materialization_folder,
               registered_support_tiff, output_folder, pending)
        final = output_folder / "experiment.cellphenotyper.json"
        pending.rename(final)
        validate_experiment(final)
        stage = "completion"
        report = {"format": FORMAT, "complete": True, "cpu_only": True,
            "review_only": True, "biological_accuracy_qualified": False,
            "shard_count": len(shards),
            "shard_receipt_sha256": [sha256(folder / "SHARD_COMPLETE.json")
                                      for folder in shards],
            "token_receipt_sha256": sha256(output_folder / "tokens" /
                                           "TRANSFER_TOKEN_COMPLETE.json"),
            "assembly_receipt_sha256": sha256(output_folder / "assembly" /
                                              "ASSEMBLY_COMPLETE.json"),
            "vector_receipt_sha256": sha256(output_folder / "vectors" /
                                            "GLOBAL_VECTOR_COMPLETE.json"),
            "project_path": str(final), "project_sha256": sha256(final),
            "source_tree_sha256": source["sha256"],
            "runtime_sha256": runtime_sha, "retry_count": 0}
        _write_json(output_folder / "RUN_COMPLETE.json", report)
        return report
    except BaseException as exc:
        project = output_folder / "experiment.cellphenotyper.json"
        if project.is_file() and not (output_folder / "RUN_COMPLETE.json").is_file():
            try:
                project.rename(output_folder / "project.unaccepted.cellphenotyper.json")
            except OSError:
                pass
        failure = {"format": FORMAT, "complete": False, "failed_stage": stage,
                   "exception_type": type(exc).__name__, "message": str(exc)[:2000],
                   "traceback": traceback.format_exc(limit=8)[-8000:],
                   "completed_shards": len(shards), "retry_count": 0,
                   "output_preserved": True}
        try:
            _write_json(output_folder / "RUN_FAILED.json", failure)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("identity")
    identity.add_argument("--bora-source", required=True, type=Path)
    launch = sub.add_parser("run")
    for name in ("model", "policy", "materialization-folder", "registration",
                 "registered-support-tiff", "base-project", "bora-source", "output"):
        launch.add_argument("--" + name, required=True, type=Path)
    launch.add_argument("--expected-bora-sha256", required=True)
    launch.add_argument("--expected-runtime-sha256", required=True)
    launch.add_argument("--acknowledge-exploratory", action="store_true")
    launch.add_argument("--core-pixels", type=int, default=2048)
    launch.add_argument("--output-halo-pixels", type=int, default=3)
    launch.add_argument("--memory-budget-bytes", type=int, default=3*GIB)
    launch.add_argument("--disk-budget-bytes", type=int, default=16*GIB)
    launch.add_argument("--deadline-seconds", type=int, default=24*3600)
    args = parser.parse_args()
    if args.command == "identity":
        _cpu_environment()
        source = bora_source_identity(args.bora_source)
        runtime = _runtime_identity()
        print(json.dumps({"bora_source_sha256": source["sha256"],
                          "runtime_sha256": identity_digest(runtime),
                          "runtime": runtime}, sort_keys=True))
        return
    print(json.dumps(run(model_path=args.model, policy_path=args.policy,
        materialization_folder=args.materialization_folder,
        registration_path=args.registration,
        registered_support_tiff=args.registered_support_tiff,
        base_project=args.base_project, bora_source=args.bora_source,
        expected_bora_sha256=args.expected_bora_sha256,
        expected_runtime_sha256=args.expected_runtime_sha256,
        output_folder=args.output,
        acknowledge_exploratory=args.acknowledge_exploratory,
        core_pixels=args.core_pixels, output_halo_pixels=args.output_halo_pixels,
        memory_budget_bytes=args.memory_budget_bytes,
        disk_budget_bytes=args.disk_budget_bytes,
        deadline_seconds=args.deadline_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
