#!/usr/bin/env python3
"""Continue a registered Virchow2/KODAMA slide through contextual Bora.

This CPU-only route consumes a complete, source-pinned PCA token lattice and a
head fitted for the *same image*. It never trains from its own predictions.
The native pixel mask remains exact; the guarded, diagonal GeoJSON is the
ordinary editable tissue annotation in a new CellPhenotyper project.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy import ndimage as ndi

from export_experiment import atomic_json, validate_experiment


SCHEMA = "cellphenotyper-bora-contextual-fullslide-input-v1"
COMPLETE = "cellphenotyper-bora-contextual-fullslide-complete-v1"
BOUNDARY_SPEC = {"format": "bora-local-he-boundary-v1",
    "uncertainty_margin": .30, "maximum_distance_um": 12.0,
    "he_blur_um": .5, "he_edge_sigma": .12,
    "smoothness": 1.5, "iterations": 4}
CLASS_NAMES = {1: "KODAMA Group 1", 2: "KODAMA Group 2",
               3: "KODAMA Group 3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(root: Path) -> str:
    root = root.resolve(strict=True)
    sources = [root / "pyproject.toml", *sorted((root / "src/bora").rglob("*.py"))]
    if len(sources) < 3 or any(not p.is_file() or p.is_symlink() for p in sources):
        raise ValueError("Bora source must be an ordinary complete source tree")
    rows = {str(p.relative_to(root)): sha256(p) for p in sources}
    return hashlib.sha256(json.dumps(rows, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def _input(record: dict, key: str) -> Path:
    item = record.get(key)
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError(f"{key} requires one path and SHA-256")
    path = Path(item["path"])
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{key} must be an ordinary absolute file")
    if sha256(path) != item["sha256"]:
        raise ValueError(f"{key} SHA-256 mismatch")
    return path.resolve(strict=True)


def _manifest(path: Path) -> tuple[dict, dict[str, Path], dict]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError("Use an ordinary absolute contextual Bora manifest")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Expected the explicit contextual Bora input schema")
    mode = manifest.get("head_mode")
    if mode not in ("fit_from_kodama_tiles", "pinned_same_slide_head"):
        raise ValueError("Select explicit KODAMA tile fitting or a pinned same-slide head")
    files = {key: _input(manifest["inputs"], key) for key in (
        "pca_field", "pca_projection", "token_support", "native_support", "native_he",
        "base_project", "kodama_binding")}
    if mode == "pinned_same_slide_head":
        files.update({key: _input(manifest["inputs"], key)
                      for key in ("numeric_head", "fit_receipt")})
        receipt = json.loads(files["fit_receipt"].read_text())
        if (files["numeric_head"].name != "numeric_head.npz" or
                files["fit_receipt"].name != "FIT_COMPLETE.json" or
                files["numeric_head"].parent != files["fit_receipt"].parent or
                receipt.get("format") != "bora-kodama-three-group-head-v1" or
                receipt.get("status") != "COMPLETE" or
                receipt.get("class_ids") != [1, 2, 3] or
                receipt.get("model_sha256") != sha256(files["numeric_head"]) or
                receipt.get("source_image_sha256") != sha256(files["native_he"]) or
                receipt.get("source_frame_id") != manifest.get("source_frame_id") or
                receipt.get("pca_projection_sha256") !=
                    manifest.get("pca_projection_sha256") or
                receipt.get("training_annotation_sha256") !=
                    sha256(files["kodama_binding"])):
            raise ValueError("The numeric head is not pinned to this same slide")
    else:
        files["labeled_tiles"] = _input(manifest["inputs"], "labeled_tiles")
    kodama = json.loads(files["kodama_binding"].read_text())
    if (kodama.get("group_count") != 3 or
            kodama.get("source_image_sha256") != sha256(files["native_he"]) or
            kodama.get("class_ids") != [1, 2, 3] or
            kodama.get("pca_field_sha256") != sha256(files["pca_field"]) or
            kodama.get("pca_projection_sha256") !=
                manifest.get("pca_projection_sha256") or
            kodama.get("token_support_sha256") != sha256(files["token_support"]) or
            kodama.get("source_frame_id") != manifest.get("source_frame_id") or
            kodama.get("working_shape_yx") != manifest.get("working_shape_yx")):
        raise ValueError("KODAMA groups do not bind this source slide")
    if (mode == "fit_from_kodama_tiles" and
            kodama.get("labeled_tiles_sha256") != sha256(files["labeled_tiles"])):
        raise ValueError("KODAMA binding does not pin the labeled tile table")
    tissue = [item for item in validate_experiment(files["base_project"])["tissues"]
              if item["image"]["sha256"] == sha256(files["native_he"])]
    if len(tissue) != 1:
        raise ValueError("Base project must contain exactly this registered native H&E")
    image = tissue[0]["image"]
    if (manifest.get("native_shape_yx") != [image["height"], image["width"]] or
            manifest.get("native_mpp_xy") != [image["mpp_x"], image["mpp_y"]] or
            manifest.get("source_frame_id") != image["frame_id"]):
        raise ValueError("Native shape, MPP or frame differs from project")
    if (not isinstance(manifest.get("working_shape_yx"), list) or
            len(manifest["working_shape_yx"]) != 2 or
            any(type(v) is not int or v <= 0 for v in manifest["working_shape_yx"]) or
            manifest.get("working_mpp_xy") != [0.5, 0.5] or
            any(abs(manifest["native_shape_yx"][axis] *
                    manifest["native_mpp_xy"][1-axis] -
                    manifest["working_shape_yx"][axis] * 0.5) > .5
                for axis in (0, 1))):
        raise ValueError("Working 0.5-µm grid and native slide differ in physical extent")
    token = np.load(files["pca_field"], mmap_mode="r", allow_pickle=False)
    token_support = np.load(files["token_support"], mmap_mode="r", allow_pickle=False)
    native_support = np.load(files["native_support"], mmap_mode="r", allow_pickle=False)
    working = manifest.get("working_shape_yx")
    if (token.ndim != 3 or token.dtype != np.float32 or
            token.shape[2] != 32 or
            not isinstance(working, list) or len(working) != 2 or
            any(type(v) is not int or v <= 0 for v in working) or
            token.shape[:2] != ((working[0] + 13) // 14,
                                (working[1] + 13) // 14) or
            token_support.shape != token.shape[:2] or
            token_support.dtype != bool or
            native_support.shape != (image["height"], image["width"]) or
            native_support.dtype not in (np.dtype("bool"), np.dtype("uint8")) or
            not np.isfinite(token).all()):
        raise ValueError("Registered PCA/support arrays have incompatible shapes")
    projection_sha = manifest.get("pca_projection_sha256")
    if (not isinstance(projection_sha, str) or len(projection_sha) != 64 or
            any(char not in "0123456789abcdef" for char in projection_sha) or
            projection_sha != sha256(files["pca_projection"])):
        raise ValueError("A frozen PCA projection SHA-256 is required")
    spec = manifest.get("analysis")
    if not isinstance(spec, dict) or set(spec) != {
            "context_sigma_tokens", "prior_exponent", "training_prior",
            "boundary_spec", "core_pixels", "halo_pixels",
            "regularization_c"}:
        raise ValueError("Context, prior, boundary and bounded core settings are required")
    sigma, exponent = spec["context_sigma_tokens"], spec["prior_exponent"]
    prior = np.asarray(spec["training_prior"], np.float32)
    if (type(sigma) not in (int, float) or not math.isfinite(sigma) or
            not 0 <= sigma <= 8 or type(exponent) not in (int, float) or
            not math.isfinite(exponent) or not 0 <= exponent <= 1 or
            prior.shape != (3,) or not np.isfinite(prior).all() or
            np.any(prior <= 0) or not np.isclose(prior.sum(), 1, atol=1e-6) or
            type(spec["regularization_c"]) not in (int, float) or
            not math.isfinite(spec["regularization_c"]) or
            not 0 < spec["regularization_c"] <= 10):
        raise ValueError("Context/prior settings are invalid")
    if (type(spec["core_pixels"]) is not int or
            not 256 <= spec["core_pixels"] <= 2048 or
            type(spec["halo_pixels"]) is not int or
            not 2 <= spec["halo_pixels"] <= 256):
        raise ValueError("Native core/halo settings are outside bounded limits")
    if spec["boundary_spec"] != BOUNDARY_SPEC:
        raise ValueError("Use the versioned recovered native H&E boundary specification")
    mx, my = manifest["native_mpp_xy"]
    required_halo = max(math.ceil(12 / mx + 3*.5 / mx) + 4 + 6,
                        math.ceil(12 / my + 3*.5 / my) + 4 + 6)
    if spec["halo_pixels"] < required_halo:
        raise ValueError("Native halo lacks the full H&E boundary context")
    return manifest, files, tissue[0]


def _samples_from_kodama_tiles(table: Path, field: np.ndarray,
                           token_support: np.ndarray, sigma_tokens: float,
                           contextualize_pca,
                           native_shape: tuple[int, int],
                           working_shape: tuple[int, int], prior: np.ndarray,
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample registered KODAMA representatives from this slide's context field."""
    scale_y = native_shape[0] / working_shape[0]
    scale_x = native_shape[1] / working_shape[1]
    samples, labels, seen = [], [], set()
    with table.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["tile_id", "native_x_px", "native_y_px",
                                 "kodama_group"]:
            raise ValueError("Labeled tiles need the exact registered four-column schema")
        for row in reader:
            tile_id = row["tile_id"]
            if not tile_id or tile_id in seen:
                raise ValueError("KODAMA tile IDs must be unique and nonempty")
            seen.add(tile_id)
            try:
                x, y = float(row["native_x_px"]), float(row["native_y_px"])
                group = int(row["kodama_group"])
            except (TypeError, ValueError) as exc:
                raise ValueError("KODAMA tile coordinate or group is invalid") from exc
            if (not math.isfinite(x) or not math.isfinite(y) or
                    not 0 <= x < native_shape[1] or
                    not 0 <= y < native_shape[0] or group not in (1, 2, 3)):
                raise ValueError("KODAMA tile is outside the registered slide or group set")
            ty = min(field.shape[0]-1, int(y / scale_y / 14))
            tx = min(field.shape[1]-1, int(x / scale_x / 14))
            if not token_support[ty, tx]:
                raise ValueError("A KODAMA training tile maps outside token tissue support")
            # Gaussian filtering uses a four-sigma finite footprint. Sample
            # only the needed neighborhood so a whole-slide fit remains bounded.
            radius = math.ceil(4 * sigma_tokens) + 1
            y0, y1 = max(0, ty-radius), min(field.shape[0], ty+radius+1)
            x0, x1 = max(0, tx-radius), min(field.shape[1], tx+radius+1)
            local = contextualize_pca(np.asarray(field[y0:y1, x0:x1]),
                np.asarray(token_support[y0:y1, x0:x1]), sigma_tokens)
            samples.append(local[ty-y0, tx-x0])
            labels.append(group)
    counts = np.bincount(labels, minlength=4)[1:]
    if len(samples) < 30 or np.any(counts < 10):
        raise ValueError("Each KODAMA group requires at least 10 source tiles")
    empirical = counts / counts.sum()
    if not np.allclose(empirical, prior, atol=1e-5, rtol=0):
        raise ValueError("Training prior differs from the bound KODAMA tile counts")
    return (np.asarray(samples, np.float32),
            np.asarray(labels, np.int32), {"tile_count": len(samples),
                  "tiles_per_group": counts.astype(int).tolist(),
                  "training_prior": empirical.tolist()})


def _project_window(tokens: np.ndarray, origin: tuple[int, int],
                    shape: tuple[int, int], scale: tuple[float, float]) -> np.ndarray:
    """Bilinear native-centre projection in absolute coordinates, never per-core."""
    oy, ox = origin
    height, width = shape
    yy = ((np.arange(oy, oy + height, dtype=np.float64) + .5) /
          scale[1]) / 14 - .5
    xx = ((np.arange(ox, ox + width, dtype=np.float64) + .5) /
          scale[0]) / 14 - .5
    ty0, tx0 = max(0, math.floor(float(yy[0]))), max(0, math.floor(float(xx[0])))
    ty1 = min(tokens.shape[0], math.ceil(float(yy[-1])) + 2)
    tx1 = min(tokens.shape[1], math.ceil(float(xx[-1])) + 2)
    block = np.asarray(tokens[ty0:ty1, tx0:tx1])
    grid_y, grid_x = np.meshgrid(yy - ty0, xx - tx0, indexing="ij")
    output = np.empty((height, width, 3), np.float32)
    for channel in range(3):
        output[..., channel] = ndi.map_coordinates(block[..., channel],
            [grid_y, grid_x], order=1, mode="nearest", prefilter=False)
    np.clip(output, 0, 1, out=output)
    output /= output.sum(axis=-1, keepdims=True)
    return output


def _export_project(base: Path, source_tissue: dict, geojson: Path,
                    receipt: Path, output: Path) -> None:
    project = copy.deepcopy(json.loads(base.read_text()))
    # The new project sits beside this run, not beside the input manifest.
    # Rebase every existing relative asset reference without copying data.
    for old_tissue in project["tissues"]:
        for asset in (old_tissue["image"], *old_tissue["artifacts"],
                      *old_tissue["feature_sets"]):
            old_path = asset["path"]
            if Path(old_path).is_absolute():
                raise ValueError("Base project contains an absolute asset path")
            asset["path"] = os.path.relpath(
                (base.parent / old_path).resolve(strict=True), output.parent)
    tissue = next(item for item in project["tissues"]
                  if item["tissue_id"] == source_tissue["tissue_id"])
    if any(item.get("artifact_id", "").startswith("bora-contextual-")
           for item in tissue["artifacts"]):
        raise ValueError("Base project already contains contextual Bora annotation")
    tissue["artifacts"].append({
        "artifact_id": "bora-contextual-" + sha256(geojson)[:20],
        "role": "tissue_annotations", "label": "Bora · KODAMA 3 groups",
        "path": os.path.relpath(geojson, output.parent),
        "sha256": sha256(geojson), "format": "geojson",
        "source_frame_id": tissue["image"]["frame_id"],
        "image_sha256": tissue["image"]["sha256"],
        "predicted_not_expert_annotation": True,
        "training_eligible": False, "review_only": False,
        "provenance": {"algorithm": COMPLETE,
            "completion_path": os.path.relpath(receipt, output.parent),
            "completion_sha256": sha256(receipt),
            "class_identity": "anonymous_KODAMA_groups"}})
    project["settings"]["bora_annotations_mode"] = "ordinary_editable"
    project["settings"]["bora_status"] = (
        "Predicted KODAMA groups; editable tissue annotation; biological identity unvalidated")
    atomic_json(output, project, overwrite=False)
    validate_experiment(output)


def run(manifest_path: Path, bora_source: Path, expected_source_sha256: str,
        output: Path, *, deadline_seconds: int = 14400,
        disk_budget_bytes: int = 32 * 1024**3) -> dict:
    """One source-bound CPU run; each output is new and failures are retained."""
    start = time.monotonic()
    if (output.exists() or not output.parent.is_dir() or
            type(deadline_seconds) is not int or not 600 <= deadline_seconds <= 86400 or
            type(disk_budget_bytes) is not int or disk_budget_bytes < 1024**3):
        raise ValueError("Use a fresh output with bounded time and disk")
    source_sha = _source_identity(bora_source)
    if source_sha != expected_source_sha256:
        raise ValueError("Pinned Bora source tree differs from request")
    manifest, files, tissue = _manifest(manifest_path)
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(key) not in (None, "", "-1"):
            raise ValueError("Contextual Bora is CPU-only")
        os.environ[key] = "-1"
    source_package = (bora_source.resolve(strict=True) / "src/bora")
    if ("bora" in sys.modules and
            source_package not in Path(sys.modules["bora"].__file__).resolve().parents):
        raise ValueError("Another Bora installation is already imported")
    sys.path.insert(0, str(bora_source.resolve(strict=True) / "src"))
    from bora.kodama_bora import (
        contextualize_pca, fit_three_group_model, score_three_group_model,
        refine_three_group_native_window,
        assemble_wsi_label_cores, publish_three_group_contours)
    import rasterio
    from rasterio.windows import Window
    for name in ("bora.kodama_bora", "bora.contextual_token_head",
                 "bora.boundary_refinement", "bora.wsi_label_assembly",
                 "bora.wsi_global_vector", "bora.wsi_review_smoothing"):
        module = sys.modules.get(name)
        if module is None or source_package not in Path(module.__file__).resolve().parents:
            raise ValueError(f"{name} resolved outside the pinned Bora source")
    if _source_identity(bora_source) != source_sha:
        raise ValueError("Bora source changed during import")
    spec = manifest["analysis"]
    native_h, native_w = manifest["native_shape_yx"]
    working_h, working_w = manifest["working_shape_yx"]
    estimated_bytes = (native_h*native_w*6 +
                       2 * 1024**3)
    if (estimated_bytes > disk_budget_bytes or
            shutil.disk_usage(output.parent).free < estimated_bytes + 1024**3):
        raise ValueError("Full-slide core, vector and staging budget is insufficient")
    scale = (native_w / working_w, native_h / working_h)
    native_support = np.load(files["native_support"], mmap_mode="r", allow_pickle=False)
    if native_support.dtype == np.uint8 and np.any(native_support > 1):
        raise ValueError("Native support must be binary")
    source_hashes = {key: sha256(path) for key, path in files.items()}
    source_hashes["manifest"] = sha256(manifest_path)
    output.mkdir()
    stage = "preflight"
    try:
        atomic_json(output / "RUN_STARTED.json", {"schema": COMPLETE,
            "complete": False, "source_sha256": source_sha,
            "input_sha256": source_hashes, "analysis": spec})
        def admit(name: str) -> None:
            if time.monotonic() - start > deadline_seconds:
                raise TimeoutError(f"Deadline expired before {name}")
            if _source_identity(bora_source) != source_sha:
                raise ValueError("Bora source changed during run")
        stage = "global_context"
        admit(stage)
        field = np.load(files["pca_field"], mmap_mode="r", allow_pickle=False)
        token_support = np.load(files["token_support"], mmap_mode="r", allow_pickle=False)
        if manifest["head_mode"] == "fit_from_kodama_tiles":
            samples, labels, sample_receipt = _samples_from_kodama_tiles(
                files["labeled_tiles"], field, token_support,
                spec["context_sigma_tokens"], contextualize_pca,
                (native_h, native_w), (working_h, working_w),
                np.asarray(spec["training_prior"], np.float32))
            samples_path = output / "kodama_training_samples.npz"
            np.savez_compressed(samples_path, features=samples, labels=labels)
            atomic_json(output / "SAMPLE_COMPLETE.json", sample_receipt)
            fit_manifest = output / "bora_fit_manifest.json"
            atomic_json(fit_manifest, {
                "format": "bora-kodama-three-group-fit-v1",
                "source_frame_id": manifest["source_frame_id"],
                "source_image_sha256": source_hashes["native_he"],
                "training_annotation_sha256": source_hashes["kodama_binding"],
                "pca_projection_sha256": manifest["pca_projection_sha256"],
                "class_ids": [1, 2, 3],
                "regularization_c": spec["regularization_c"],
                "training_prior": spec["training_prior"],
                "samples": {"path": samples_path.name,
                            "sha256": sha256(samples_path)}})
            model_folder = output / "model"
            fit_receipt = fit_three_group_model(fit_manifest, model_folder)
        else:
            model_folder = files["numeric_head"].parent
            fit_receipt = json.loads(files["fit_receipt"].read_text())
        if (fit_receipt.get("format") != "bora-kodama-three-group-head-v1" or
                fit_receipt.get("status") != "COMPLETE" or
                fit_receipt.get("model_sha256") !=
                    sha256(model_folder / "numeric_head.npz") or
                fit_receipt.get("source_frame_id") !=
                    manifest["source_frame_id"] or
                fit_receipt.get("source_image_sha256") !=
                    source_hashes["native_he"] or
                fit_receipt.get("training_annotation_sha256") !=
                    source_hashes["kodama_binding"] or
                fit_receipt.get("pca_projection_sha256") !=
                    manifest["pca_projection_sha256"] or
                np.asarray(fit_receipt.get("training_prior", [])).shape != (3,) or
                not np.allclose(fit_receipt["training_prior"],
                    spec["training_prior"], rtol=0, atol=1e-6) or
                fit_receipt.get("regularization_c") !=
                    spec["regularization_c"] or
                fit_receipt.get("class_ids") != [1, 2, 3]):
            raise ValueError("Bora fitted head/model does not bind this slide and PCA projection")
        model_binding = hashlib.sha256(json.dumps({"inputs": source_hashes,
            "source": source_sha, "analysis": spec,
            "model_sha256": fit_receipt["model_sha256"]}, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        score_manifest = output / "bora_score_manifest.json"
        atomic_json(score_manifest, {
            "format": "bora-kodama-three-group-score-v1",
            "registration_method": "same_global_token_grid",
            "source_frame_id": manifest["source_frame_id"],
            "source_image_sha256": source_hashes["native_he"],
            "pca_projection_sha256": manifest["pca_projection_sha256"],
            "field": {"path": str(files["pca_field"]),
                      "sha256": source_hashes["pca_field"]},
            "support": {"path": str(files["token_support"]),
                        "sha256": source_hashes["token_support"]}})
        token_scores = output / "token_scores"
        score_receipt = score_three_group_model(score_manifest, model_folder,
            token_scores, sigma_tokens=spec["context_sigma_tokens"],
            prior_exponent=spec["prior_exponent"])
        probability_path = token_scores / "three_group_probability.npy"
        if (score_receipt.get("status") != "COMPLETE" or
                score_receipt.get("probability_sha256") != sha256(probability_path) or
                score_receipt.get("model_sha256") != fit_receipt["model_sha256"] or
                score_receipt.get("source_image_sha256") !=
                    source_hashes["native_he"] or
                score_receipt.get("source_frame_id") !=
                    manifest["source_frame_id"]):
            raise ValueError("Bora contextual probability receipt is incomplete")
        probability = np.load(probability_path, mmap_mode="r", allow_pickle=False)
        stage = "native_he_refinement"
        admit(stage)
        shard_root = output / "native_shards"
        shard_root.mkdir()
        windows = []
        core = spec["core_pixels"]
        halo = spec["halo_pixels"]
        # The output halo overlaps neighboring cores. An additional source
        # halo admits the full finite H&E search/smoothing context.
        output_halo = 2
        with rasterio.open(files["native_he"]) as image:
            if (image.count != 3 or image.dtypes != ("uint8",) * 3 or
                    (image.height, image.width) != (native_h, native_w)):
                raise ValueError("Native H&E must be a registered RGB uint8 TIFF")
            index = 0
            for y in range(0, native_h, core):
                for x in range(0, native_w, core):
                    admit(f"native_core_{index}")
                    h, w = min(core, native_h-y), min(core, native_w-x)
                    oy, ox = max(0, y-halo), max(0, x-halo)
                    ey, ex = min(native_h, y+h+halo), min(native_w, x+w+halo)
                    if (ey-oy)*(ex-ox) > 16_777_216:
                        raise ValueError("Native source window exceeds bounded size")
                    scores = _project_window(probability, (oy, ox),
                        (ey-oy, ex-ox), scale)
                    he = np.moveaxis(image.read((1, 2, 3),
                        window=Window(ox, oy, ex-ox, ey-oy)), 0, -1)
                    support = np.asarray(native_support[oy:ey, ox:ex]).astype(bool)
                    labels, _ = refine_three_group_native_window(scores, he,
                        support, manifest["native_mpp_xy"],
                        spec["boundary_spec"])
                    wy, wx = max(0, y-output_halo), max(0, x-output_halo)
                    wey = min(native_h, y+h+output_halo)
                    wex = min(native_w, x+w+output_halo)
                    selected = labels[wy-oy:wey-oy, wx-ox:wex-ox]
                    selected_support = support[wy-oy:wey-oy, wx-ox:wex-ox]
                    categorical = np.where(selected_support, selected, 0)
                    if (not np.isin(categorical, (0, 1, 2, 3)).all() or
                            np.any((categorical != 0) != selected_support)):
                        raise ValueError("Native labels disagree with source tissue support")
                    path = shard_root / f"{index:05d}.npy"
                    np.save(path, categorical.astype(np.uint8), allow_pickle=False)
                    windows.append({"labels_path": str(path),
                        "labels_sha256": sha256(path),
                        "origin_yx": [wy, wx], "core_yxhw": [y, x, h, w],
                        "global_model_binding_sha256": model_binding,
                        "source_frame_id": manifest["source_frame_id"],
                        "source_image_sha256": source_hashes["native_he"],
                        "tissue_support_sha256": source_hashes["native_support"],
                        "independently_fitted": False})
                    index += 1
        del probability
        stage = "exact_global_assembly"
        admit(stage)
        assembly = output / "assembly"
        assemble_wsi_label_cores(windows, (native_h, native_w),
            [1, 2, 3], assembly,
            tissue_support_path=files["native_support"],
            tissue_support_sha256=source_hashes["native_support"],
            native_he_path=files["native_he"],
            global_model_binding_sha256=model_binding,
            source_frame_id=manifest["source_frame_id"],
            source_image_sha256=source_hashes["native_he"],
            native_mpp_xy=manifest["native_mpp_xy"],
            disk_budget_bytes=disk_budget_bytes)
        stage = "global_vector_and_guarded_outer_geometry"
        admit(stage)
        contours = output / "contours"
        contour_receipt = publish_three_group_contours(
            assembly, files["native_he"], contours, CLASS_NAMES,
            minimum_area_um2=1000.0,
            minimum_free_bytes=1024**3,
            deadline_seconds=max(60, min(7200,
                deadline_seconds - int(time.monotonic()-start))))
        geojson = contours / "tissue_annotations/BORA_TISSUE_ANNOTATIONS.geojson"
        if (contour_receipt.get("status") != "COMPLETE" or
                contour_receipt.get("predicted_tissue_geojson_sha256") != sha256(geojson) or
                contour_receipt.get("component_count", 0) < 1):
            raise ValueError("Guarded outer geometry did not pass Bora's native QA")
        for key, path in files.items():
            if sha256(path) != source_hashes[key]:
                raise ValueError(f"Frozen input changed during run: {key}")
        if sha256(manifest_path) != source_hashes["manifest"]:
            raise ValueError("Manifest changed during run")
        admit("project_export")
        products = output / "PIPELINE_PRODUCTS.json"
        atomic_json(products, {"schema": COMPLETE, "complete": True,
            "source_sha256": source_sha, "model_binding_sha256": model_binding,
            "input_sha256": source_hashes,
            "token_probability_sha256": sha256(probability_path),
            "fit_receipt_sha256": sha256(model_folder / "FIT_COMPLETE.json"),
            "score_receipt_sha256": sha256(token_scores / "SCORE_COMPLETE.json"),
            "native_assembly_sha256": sha256(assembly / "ASSEMBLY_COMPLETE.json"),
            "contour_complete_sha256": sha256(contours / "COMPLETE.json"),
            "annotation_sha256": sha256(geojson),
            "core_count": len(windows), "cpu_only": True,
            "biological_accuracy_qualified": False,
            "elapsed_seconds": time.monotonic()-start})
        project = output / "experiment.cellphenotyper.json"
        _export_project(files["base_project"], tissue, geojson,
                        products, project)
        complete = output / "RUN_COMPLETE.json"
        atomic_json(complete, {"schema": COMPLETE, "complete": True,
            "products_sha256": sha256(products),
            "project_sha256": sha256(project),
            "source_sha256": source_sha,
            "model_binding_sha256": model_binding,
            "elapsed_seconds": time.monotonic()-start})
        return {"project": str(project), "annotation": str(geojson),
                "completion": str(complete)}
    except BaseException as exc:
        atomic_json(output / "RUN_FAILED.json", {"schema": COMPLETE,
            "complete": False, "stage": stage, "error": str(exc),
            "elapsed_seconds": time.monotonic()-start})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    identity = actions.add_parser("identity", help="hash the reviewed Bora source tree")
    identity.add_argument("--bora-source", type=Path, required=True)
    launch = actions.add_parser("run", help="run one fresh, bounded CPU continuation")
    launch.add_argument("--manifest", type=Path, required=True)
    launch.add_argument("--bora-source", type=Path, required=True)
    launch.add_argument("--expected-bora-source-sha256", required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--deadline-seconds", type=int, default=14400)
    launch.add_argument("--disk-budget-bytes", type=int, default=32*1024**3)
    args = parser.parse_args()
    if args.action == "identity":
        print(json.dumps({"bora_source_sha256":
                          _source_identity(args.bora_source)}, sort_keys=True))
        return
    print(json.dumps(run(args.manifest, args.bora_source,
        args.expected_bora_source_sha256, args.output,
        deadline_seconds=args.deadline_seconds,
        disk_budget_bytes=args.disk_budget_bytes), sort_keys=True))


if __name__ == "__main__":
    main()
