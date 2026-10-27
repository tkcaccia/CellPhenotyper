"""Run one source-bound, CPU-only semantic Bora slide into one new result folder.

This action starts from a completed CellPhenotyper Virchow2 materialization and
passing independent-expert same-slide holdout. It scores one global token field,
projects bounded native shards, refines them against registered H&E, assembles
one global boundary, and exports a locked exploratory project revision. No
hard-group fallback, GPU command, retry, or resume path exists. Failed folders
remain sealed for diagnosis and a new run must choose a new output directory.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

if __package__:
    from .bora_continuous_fullslide_action import (
        GIB, MIB, MAX_NATIVE_PIXELS, MAX_SHARDS, _write_json,
        bora_source_identity, core_grid, identity_digest, runtime_identity,
        sha256,
    )
    from .export_bora_semantic_project import _evaluation
    from .export_bora_refined_semantic_project import export
    from .export_experiment import validate_experiment
    from .materialize_bora_semantic_tokens import OUTPUTS, verify_materialization
else:
    from bora_continuous_fullslide_action import (
        GIB, MIB, MAX_NATIVE_PIXELS, MAX_SHARDS, _write_json,
        bora_source_identity, core_grid, identity_digest, runtime_identity,
        sha256,
    )
    from export_bora_semantic_project import _evaluation
    from export_bora_refined_semantic_project import export
    from export_experiment import validate_experiment
    from materialize_bora_semantic_tokens import OUTPUTS, verify_materialization


FORMAT = "cellphenotyper-bora-refined-semantic-fullslide-run-v1"
DEFAULT_REFINEMENT = {
    # Recovered on the registered 4090-pixel colon crop. Keep the same
    # native H&E boundary parameters in the slide controller; the semantic
    # model must still pass its independent tissue-specific evaluation.
    "format": "bora-local-he-boundary-v1", "uncertainty_margin": .30,
    "maximum_distance_um": 12.0, "he_blur_um": .5,
    "he_edge_sigma": .12, "smoothness": 1.5, "iterations": 4,
}
OWN_SOURCE_NAMES = (
    "bora_refined_semantic_fullslide_action.py",
    "bora_continuous_fullslide_action.py",
    "bora_fullslide_action.py",
    "export_bora_refined_semantic_project.py",
    "export_bora_semantic_project.py",
    "export_bora_continuous_project.py",
    "export_experiment.py",
    "bora_semantic_roi_intake.py",
    "materialize_bora_semantic_tokens.py",
)


def _cpu_environment() -> None:
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        current = os.environ.get(key)
        if current not in (None, "", "-1"):
            raise ValueError(f"CPU-only semantic Bora refuses an allocated GPU environment: {key}")
        os.environ[key] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ["PYTHONNOUSERSITE"] = "1"


def _own_source_identity() -> dict:
    folder = Path(__file__).resolve().parent
    files = {name: sha256(_ordinary(folder / name, name))
             for name in OWN_SOURCE_NAMES}
    return {"files": files, "sha256": identity_digest(files)}


def semantic_runtime_identity() -> dict:
    """Pin the base Python stack plus the native OpenCV refinement binary."""
    import cv2
    result = runtime_identity()
    result["semantic_dependencies"] = _semantic_dependency_identity()
    spec = importlib.util.find_spec("cv2")
    if spec is None or spec.origin is None:
        raise ValueError("Native H&E refinement requires OpenCV")
    package = Path(spec.origin).resolve(strict=True).parent
    binaries = sorted((*package.glob("cv2*.so"), *package.glob("cv2*.pyd")))
    if len(binaries) != 1 or not binaries[0].is_file():
        raise ValueError("One identifiable OpenCV native binary is required")
    result["opencv"] = {"version": cv2.__version__,
                         "module_file": str(Path(spec.origin).resolve(strict=True)),
                         "module_sha256": sha256(Path(spec.origin)),
                         "native_binary": str(binaries[0].resolve(strict=True)),
                         "native_binary_sha256": sha256(binaries[0])}
    return result


def _semantic_dependency_identity() -> dict:
    """Pin the CPU model reader/fitter and TIFF stack actually used by Bora."""
    dependencies = {
        "scikit-learn": ("sklearn", "sklearn.linear_model._logistic"),
        "tifffile": ("tifffile",),
        "threadpoolctl": ("threadpoolctl",),
        "joblib": ("joblib",),
    }
    result = {}
    for distribution, modules in dependencies.items():
        files = {}
        for name in modules:
            spec = importlib.util.find_spec(name)
            if spec is None or spec.origin is None:
                raise ValueError(f"Semantic runtime dependency is missing: {name}")
            path = Path(spec.origin).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Semantic runtime dependency is not a file: {name}")
            files[name] = {"path": str(path), "sha256": sha256(path)}
        result[distribution] = {"version": metadata.version(distribution),
                                "modules": files}
    return result


def _ordinary(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be an ordinary absolute file: {path}")
    return path.resolve(strict=True)


def _load_bora(source_root: Path):
    source = (source_root / "src").resolve(strict=True)
    if "bora" in sys.modules and source not in Path(sys.modules["bora"].__file__).resolve().parents:
        raise ValueError("Another Bora package is already imported")
    sys.path.insert(0, str(source))
    from bora import (wsi_continuous_chain, wsi_continuous_probability_adapter,
                      wsi_semantic_probability, wsi_semantic_chain,
                      wsi_semantic_refinement, wsi_semantic_refined_chain)
    from bora.local_semantic import LocalSemanticModel
    from bora.boundary_refinement import parameters
    for module in (wsi_continuous_chain, wsi_continuous_probability_adapter,
                   wsi_semantic_probability, wsi_semantic_chain,
                   wsi_semantic_refinement, wsi_semantic_refined_chain):
        if source not in Path(module.__file__).resolve().parents:
            raise ValueError("Bora semantic module resolved outside pinned source")
    return (wsi_continuous_chain, wsi_continuous_probability_adapter,
            wsi_semantic_probability, wsi_semantic_chain,
            wsi_semantic_refinement, wsi_semantic_refined_chain,
            LocalSemanticModel, parameters)


def _estimate(shape: tuple[int, int], cores: list[tuple[int, int, int, int]],
              source_halo: int, output_halo: int, classes: int,
              token_shape: tuple[int, int]) -> int:
    height, width = shape
    native_windows = 0
    for y, x, h, w in cores:
        sh = min(height, y+h+source_halo) - max(0, y-source_halo)
        sw = min(width, x+w+source_halo) - max(0, x-source_halo)
        rh = min(height, y+h+output_halo) - max(0, y-output_halo)
        rw = min(width, x+w+output_halo) - max(0, x-output_halo)
        if sh*sw > 16_777_216 or rh*rw > 20_000_000 or max(rh, rw) > 8192:
            raise ValueError("Source/refined shard exceeds Bora's bounded native window")
        native_windows += (sh*sw + rh*rw) * classes * 4 + 2*MIB
    pixels = height * width
    # Two global chains (source evidence and refined publication), support,
    # token scores and a conservative 2 GiB free-space margin.
    return native_windows + 32*pixels + token_shape[0]*token_shape[1]*classes*4 + 2*GIB


def _checkpoint(folder: Path, stage: str, completed: int, receipt: Path | None,
                source_sha: str, runtime_sha: str) -> None:
    _write_json(folder / "PROGRESS.json", {
        "format": FORMAT, "complete": False, "cpu_only": True,
        "last_completed_stage": stage, "completed_shards": completed,
        "receipt_path": str(receipt) if receipt else None,
        "receipt_sha256": sha256(receipt) if receipt else None,
        "source_tree_sha256": source_sha, "runtime_sha256": runtime_sha,
        "retry_count": 0,
    })


def run(*, roi_spec_path: Path, evaluation_folder: Path,
        materialization_folder: Path, registration_path: Path,
        registered_support_tiff: Path, base_project: Path,
        bora_source: Path, expected_bora_sha256: str,
        expected_runtime_sha256: str, output_folder: Path,
        refinement_spec: dict | None = None,
        core_pixels: int = 2048, source_halo_pixels: int = 96,
        output_halo_pixels: int = 3,
        memory_budget_bytes: int = 3*GIB,
        disk_budget_bytes: int = 64*GIB,
        deadline_seconds: int = 24*3600) -> dict:
    _cpu_environment()
    source_root = Path(bora_source).resolve(strict=True)
    source_identity = bora_source_identity(source_root)
    if source_identity["sha256"] != expected_bora_sha256:
        raise ValueError("Pinned Bora source tree differs from requested SHA-256")
    own_source = _own_source_identity()
    runtime = semantic_runtime_identity()
    runtime_sha = identity_digest(runtime)
    if runtime_sha != expected_runtime_sha256:
        raise ValueError("Running Python/dependency identity differs from requested SHA-256")
    if (type(memory_budget_bytes) is not int or not 512*MIB <= memory_budget_bytes <= 12*GIB or
            type(disk_budget_bytes) is not int or disk_budget_bytes < 2*GIB or
            type(deadline_seconds) is not int or not 60 <= deadline_seconds <= 25*3600 or
            type(core_pixels) is not int or type(source_halo_pixels) is not int or
            type(output_halo_pixels) is not int or not 1 <= output_halo_pixels <= 256):
        raise ValueError("CPU-only semantic resource/deadline bounds are invalid")
    (support_module, adapter, scorer, source_global, refiner, refined_global,
     model_type, check_parameters) = _load_bora(source_root)
    source_dir = source_root / "src"
    roi_spec_path = _ordinary(roi_spec_path, "expert ROI specification")
    registration_path = _ordinary(registration_path, "registration")
    registered_support_tiff = _ordinary(registered_support_tiff, "native GrandQC TIFF")
    base_project = _ordinary(base_project, "base CellPhenotyper project")
    validate_experiment(base_project)
    materialization_folder = Path(materialization_folder).resolve(strict=True)
    evaluation_folder = Path(evaluation_folder).resolve(strict=True)
    materialized, material_spec = verify_materialization(materialization_folder)
    evaluation, roi_spec, model_path = _evaluation(
        roi_spec_path, evaluation_folder, source_dir, registration_path)
    if (materialized["registration_sha256"] != sha256(registration_path) or
            materialized["inventory_sha256"] !=
                roi_spec["inputs"]["token_inventory"]["sha256"] or
            materialized["checkpoint_sha256"] !=
                roi_spec["inputs"]["virchow2_checkpoint"]["sha256"] or
            material_spec["bora_source_root"] != str(source_dir) or
            materialized["source_sha256"]["working_he"] !=
                roi_spec["inputs"]["working_he"]["sha256"]):
        raise ValueError("Materialized Virchow2/H&E token field differs from passing expert ROI evaluation")
    registration, _ = adapter._registration(registration_path)
    native, working = registration["cell_full_image"], registration["image"]
    if sha256(registered_support_tiff) != registration["full_support_mask"]["sha256"]:
        raise ValueError("Registered native support TIFF changed")
    native_image = _ordinary(Path(roi_spec["inputs"]["native_image"]["path"]),
                             "registered native H&E")
    if sha256(native_image) != native["sha256"]:
        raise ValueError("Registered native H&E changed")
    model = model_type.load(model_path)
    classes = list(map(int, model.classes_))
    if classes != evaluation["class_ids"] or model.class_names_ != \
            {int(k): v for k, v in evaluation["class_names"].items()}:
        raise ValueError("Expert-evaluated semantic model classes changed")
    shape = native["height"], native["width"]
    if shape[0]*shape[1] > MAX_NATIVE_PIXELS:
        raise ValueError("Native slide exceeds one-billion-pixel admission bound")
    refinement = check_parameters(refinement_spec or DEFAULT_REFINEMENT)
    context_y, context_x = refiner._context_pixels(
        refinement, [native["mpp_x"], native["mpp_y"]])
    if source_halo_pixels < output_halo_pixels + max(context_y, context_x):
        raise ValueError("Source halo is smaller than H&E search/blur/smoothing context")
    cores = core_grid(shape, core_pixels, source_halo_pixels, len(classes))
    token_shape = ((working["height"]+13)//14, (working["width"]+13)//14)
    estimate = _estimate(shape, cores, source_halo_pixels,
                         output_halo_pixels, len(classes), token_shape)
    output_folder = Path(output_folder)
    if not output_folder.is_absolute():
        raise ValueError("New slide output folder must be an absolute path")
    if not output_folder.parent.is_dir():
        raise ValueError("New output folder needs an existing ordinary parent")
    output_parent = output_folder.parent.resolve(strict=True)
    canonical_output = output_parent / output_folder.name
    if (output_folder.exists() or output_folder.is_symlink() or
            any(canonical_output.is_relative_to(path) for path in
                (source_root, materialization_folder, evaluation_folder)) or
            disk_budget_bytes < estimate or
            shutil.disk_usage(output_parent).free < estimate):
        raise ValueError("New output folder or disk admission failed; no overwrite/retry")
    output_folder = canonical_output
    largest_h = min(shape[0], core_pixels+2*source_halo_pixels)
    largest_w = min(shape[1], core_pixels+2*source_halo_pixels)
    if memory_budget_bytes < 5*(largest_h*largest_w*len(classes)*4 + MIB):
        raise ValueError("Declared memory cannot fit the largest native source shard")
    start = time.monotonic()
    def admit(stage: str):
        if time.monotonic()-start > deadline_seconds-30:
            raise TimeoutError(f"Deadline admission closed before {stage}")
        if bora_source_identity(source_root)["sha256"] != source_identity["sha256"]:
            raise ValueError("Pinned Bora source changed before " + stage)
        if _own_source_identity()["sha256"] != own_source["sha256"]:
            raise ValueError("CellPhenotyper semantic controller/export source changed before " + stage)
    frozen = {
        "roi_spec_sha256": sha256(roi_spec_path),
        "evaluation_sha256": sha256(evaluation_folder / "EVALUATION.json"),
        "materialization_receipt_sha256": sha256(
            materialization_folder / "MATERIALIZATION_COMPLETE.json"),
        "model_sha256": sha256(model_path),
        "registration_sha256": sha256(registration_path),
        "native_he_sha256": sha256(native_image),
        "support_tiff_sha256": sha256(registered_support_tiff),
        "base_project_sha256": sha256(base_project),
    }
    output_folder.mkdir()
    stage = "preflight"
    completed = 0
    source_folders: list[Path] = []
    refined_folders: list[Path] = []
    try:
        _write_json(output_folder / "PREFLIGHT.json", {
            "format": FORMAT, "complete": False, "cpu_only": True,
            "algorithm_id": "bora-wsi-native-he-refined-semantic-chain-v1",
            "review_only": True, "source": source_identity,
            "cellphenotyper_source": own_source,
            "runtime": runtime, "runtime_sha256": runtime_sha,
            "frozen_input_sha256": frozen,
            "native_shape_yx": list(shape), "native_mpp_xy":
                [native["mpp_x"], native["mpp_y"]],
            "class_ids": classes, "class_names": evaluation["class_names"],
            "core_pixels": core_pixels, "source_halo_pixels": source_halo_pixels,
            "output_halo_pixels": output_halo_pixels,
            "refinement_spec": refinement, "planned_shards": len(cores),
            "memory_budget_bytes": memory_budget_bytes,
            "disk_budget_bytes": disk_budget_bytes,
            "estimated_peak_output_bytes": estimate,
            "deadline_seconds": deadline_seconds, "retry_count": 0,
        })
        stage = "support"
        admit(stage)
        (output_folder / "01_inputs").mkdir()
        native_support = output_folder / "01_inputs/support.npy"
        support_module._write_native_support(registered_support_tiff, native_support,
            shape, memory_budget_bytes=memory_budget_bytes)
        support_sha = sha256(native_support)
        _checkpoint(output_folder, stage, completed, native_support,
                    source_identity["sha256"], runtime_sha)
        stage = "global_token_scoring"
        admit(stage)
        tokens = materialization_folder / OUTPUTS["virchow2_tokens"]
        he_tokens = materialization_folder / OUTPUTS["he_tokens"]
        token_folder = output_folder / "02_semantic_token"
        scorer.publish_semantic_token_field(model_path, tokens, registration_path,
            token_folder, he_token_path=he_tokens,
            materialization_folder=materialization_folder,
            memory_budget_bytes=memory_budget_bytes,
            disk_budget_bytes=disk_budget_bytes)
        _checkpoint(output_folder, stage, completed,
                    token_folder / "TOKEN_SEMANTIC_COMPLETE.json",
                    source_identity["sha256"], runtime_sha)
        (output_folder / "03_native_source_shards").mkdir()
        (output_folder / "04_native_he_refined_shards").mkdir()
        for index, core in enumerate(cores):
            stage = f"source_shard_{index:04d}"
            admit(stage)
            source_folder = output_folder / "03_native_source_shards" / f"{index:04d}"
            scorer.produce_semantic_probability_shard(token_folder,
                registration_path, native_support, support_sha, core,
                source_halo_pixels, source_folder,
                disk_budget_bytes=disk_budget_bytes,
                memory_budget_bytes=memory_budget_bytes)
            source_folders.append(source_folder)
            _checkpoint(output_folder, stage, completed,
                source_folder / "SHARD_COMPLETE.json", source_identity["sha256"], runtime_sha)
            stage = f"he_refined_shard_{index:04d}"
            admit(stage)
            refined_folder = output_folder / "04_native_he_refined_shards" / f"{index:04d}"
            refiner.produce_native_he_refined_semantic_shard(
                source_folder, token_folder, registration_path, native_image,
                native_support, support_sha, refined_folder, refinement,
                output_halo_pixels=output_halo_pixels,
                disk_budget_bytes=disk_budget_bytes,
                memory_budget_bytes=memory_budget_bytes)
            refined_folders.append(refined_folder)
            completed += 1
            _checkpoint(output_folder, stage, completed,
                refined_folder / "SHARD_COMPLETE.json", source_identity["sha256"], runtime_sha)
        stage = "source_global_evidence"
        admit(stage)
        source_output = output_folder / "05_source_global_evidence"
        source_global.run_semantic_global_chain(source_folders, token_folder,
            registration_path, registered_support_tiff, source_output,
            disk_budget_bytes=disk_budget_bytes,
            memory_budget_bytes=memory_budget_bytes)
        _checkpoint(output_folder, stage, completed,
            source_output / "SEMANTIC_CHAIN_COMPLETE.json",
            source_identity["sha256"], runtime_sha)
        stage = "native_he_refined_global"
        admit(stage)
        refined_output = output_folder / "06_refined_global"
        refined_global.run_semantic_refined_global_chain(refined_folders,
            source_folders, token_folder, model_path, tokens, he_tokens,
            registration_path, native_image, registered_support_tiff,
            native_support, refined_output,
            disk_budget_bytes=disk_budget_bytes,
            memory_budget_bytes=memory_budget_bytes)
        _checkpoint(output_folder, stage, completed,
            refined_output / "REFINED_SEMANTIC_CHAIN_COMPLETE.json",
            source_identity["sha256"], runtime_sha)
        stage = "project_export"
        admit(stage)
        if identity_digest(semantic_runtime_identity()) != runtime_sha or any(
                sha256(path) != digest for path, digest in (
                    (roi_spec_path, frozen["roi_spec_sha256"]),
                    (evaluation_folder / "EVALUATION.json", frozen["evaluation_sha256"]),
                    (materialization_folder / "MATERIALIZATION_COMPLETE.json",
                     frozen["materialization_receipt_sha256"]),
                    (model_path, frozen["model_sha256"]),
                    (registration_path, frozen["registration_sha256"]),
                    (native_image, frozen["native_he_sha256"]),
                    (registered_support_tiff, frozen["support_tiff_sha256"]),
                    (base_project, frozen["base_project_sha256"]))):
            raise ValueError("Frozen source/runtime changed before project publication")
        pending = output_folder / "project.pending.cellphenotyper.json"
        export(base_project, roi_spec_path, evaluation_folder, registration_path,
            registered_support_tiff, materialization_folder, token_folder,
            tokens, he_tokens, source_folders, refined_folders,
            source_output, refined_output, source_dir, pending)
        final = output_folder / "experiment.cellphenotyper.json"
        pending.rename(final)
        validate_experiment(final)
        _checkpoint(output_folder, stage, completed, final,
                    source_identity["sha256"], runtime_sha)
        stage = "complete"
        result = {
            "format": FORMAT, "complete": True, "cpu_only": True,
            "review_only": True, "algorithm_id":
                "bora-wsi-native-he-refined-semantic-chain-v1",
            "source_tree_sha256": source_identity["sha256"],
            "cellphenotyper_source_sha256": own_source["sha256"],
            "runtime_sha256": runtime_sha,
            "preflight_sha256": sha256(output_folder / "PREFLIGHT.json"),
            "shard_count": completed,
            "source_chain_sha256": sha256(source_output / "SEMANTIC_CHAIN_COMPLETE.json"),
            "refined_chain_sha256": sha256(
                refined_output / "REFINED_SEMANTIC_CHAIN_COMPLETE.json"),
            "project_path": str(final), "project_sha256": sha256(final),
            "retry_count": 0,
        }
        _write_json(output_folder / "RUN_COMPLETE.json", result)
        return result
    except BaseException as exc:
        published = output_folder / "experiment.cellphenotyper.json"
        if published.is_file() and not (output_folder / "RUN_COMPLETE.json").exists():
            try:
                published.rename(output_folder / "project.unaccepted.cellphenotyper.json")
            except OSError:
                pass
        failure = {"format": FORMAT, "complete": False,
                   "failed_stage": stage, "exception_type": type(exc).__name__,
                   "message": str(exc)[:2000],
                   "traceback": traceback.format_exc(limit=8)[-8000:],
                   "completed_shards": completed, "retry_count": 0,
                   "output_preserved": True, "next_stage_blocked": True}
        try:
            _write_json(output_folder / "RUN_FAILED.json", failure)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("identity", help="Print source/runtime pins")
    identity.add_argument("--bora-source", type=Path, required=True)
    launch = sub.add_parser("run", help="Run one new CPU-only slide folder")
    for name in ("roi-spec", "evaluation-folder", "materialization-folder",
                 "registration", "registered-support-tiff", "base-project",
                 "bora-source", "output"):
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--expected-bora-sha256", required=True)
    launch.add_argument("--expected-runtime-sha256", required=True)
    launch.add_argument("--compute-device", required=True, choices=("cpu",))
    launch.add_argument("--refinement-spec", type=Path)
    launch.add_argument("--core-pixels", type=int, default=2048)
    launch.add_argument("--source-halo-pixels", type=int, default=96)
    launch.add_argument("--output-halo-pixels", type=int, default=3)
    launch.add_argument("--memory-budget-bytes", type=int, default=3*GIB)
    launch.add_argument("--disk-budget-bytes", type=int, default=64*GIB)
    launch.add_argument("--deadline-seconds", type=int, default=24*3600)
    args = parser.parse_args()
    _cpu_environment()
    if args.command == "identity":
        source = bora_source_identity(args.bora_source)
        runtime = semantic_runtime_identity()
        print(json.dumps({"bora_source_sha256": source["sha256"],
                          "cellphenotyper_source_sha256": _own_source_identity()["sha256"],
                          "runtime_sha256": identity_digest(runtime),
                          "runtime": runtime}, sort_keys=True))
        return
    refinement = None
    if args.refinement_spec:
        refinement = json.loads(_ordinary(args.refinement_spec, "refinement spec").read_text())
    result = run(roi_spec_path=args.roi_spec,
        evaluation_folder=args.evaluation_folder,
        materialization_folder=args.materialization_folder,
        registration_path=args.registration,
        registered_support_tiff=args.registered_support_tiff,
        base_project=args.base_project,
        bora_source=args.bora_source,
        expected_bora_sha256=args.expected_bora_sha256,
        expected_runtime_sha256=args.expected_runtime_sha256,
        output_folder=args.output, refinement_spec=refinement,
        core_pixels=args.core_pixels,
        source_halo_pixels=args.source_halo_pixels,
        output_halo_pixels=args.output_halo_pixels,
        memory_budget_bytes=args.memory_budget_bytes,
        disk_budget_bytes=args.disk_budget_bytes,
        deadline_seconds=args.deadline_seconds)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
