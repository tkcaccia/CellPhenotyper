#!/usr/bin/env python3
"""Produce independently encoded local/wider physical UNI2 fields for hierarchy.

No expert annotations are read. Each field is a separate image crop and model
forward, not a renamed subset of tokens from a shared context. Enlarging the
physical field while retaining 224 model pixels changes effective model MPP:
this is an explicit scale/domain shift requiring specimen-level validation.

Outputs are checksum-bound float32 NPY matrices, small keyed row tables and
embedding manifests accepted by discover_tissue_hierarchy.py. A completed field
is reused only after input, feature definition, row and array hashes match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cell_profile_io import RasterReader, sha256_file
from discover_tissue_hierarchy import validate_feature_definitions, validate_geometry, validate_grid
from model_provenance import package_version


SCHEMA_VERSION = "1.0.0"
MODEL_ID = "MahmoodLab/UNI2-h"
MODEL_INPUT_PIXELS = 224
PRODUCER_CODE_FILES = (
    "prepare_hierarchy_features.py", "extract_uni2_embeddings.py", "model_provenance.py",
    "uni2_grid.py", "uni2_embedding_io.py", "discover_tissue_hierarchy.py", "hierarchy_kodama.py",
    "build_cell_profiles.py", "cell_profile_io.py", "profile_cell_morphology.py",
    "cell_morphology_io.py", "cell_phenotype_io.py", "cellvit_embedding_io.py", "ome_tiff_metadata.py",
)


def verify_source_identity(expected):
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Source changed during hierarchy feature generation: {path}")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def inspect_snapshot(snapshot, weights_filename=None):
    """Identify actual local config+weights bytes, not the mutable Hub tag."""
    snapshot = Path(snapshot).resolve()
    config_path = snapshot / "config.json"
    config_digest = sha256_file(config_path)
    config = json.loads(config_path.read_text())
    if sha256_file(config_path) != config_digest:
        raise RuntimeError("UNI2 config changed during snapshot inspection")
    if config.get("architecture") not in {"vit_giant_patch14_224", "vit_huge_patch14_224"}:
        raise ValueError("The local model config must declare a compatible UNI2-h giant/huge patch14 base architecture")
    if weights_filename is None:
        weights_filename = next((name for name in ("model.safetensors", "pytorch_model.bin") if (snapshot / name).is_file()), None)
    if not weights_filename or Path(weights_filename).name != weights_filename:
        raise ValueError("Specify a local UNI2 checkpoint filename, not a relative/absolute path")
    weights_path = snapshot / weights_filename
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if weights_path.suffix not in {".safetensors", ".bin", ".pth", ".pt"}:
        raise ValueError("Unsupported UNI2 checkpoint format")
    records = {name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in (("config.json", config_path), (weights_filename, weights_path))}
    if records["config.json"]["sha256"] != config_digest:
        raise RuntimeError("UNI2 config changed during snapshot inspection")
    cfg = config.get("pretrained_cfg", {})
    mean, std = np.array(cfg.get("mean", [.485, .456, .406])), np.array(cfg.get("std", [.229, .224, .225]))
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid model RGB normalization")
    revision = "sha256:" + canonical_hash(records)
    return {"model_id": MODEL_ID, "model_revision": revision, "weights_filename": weights_filename,
            "architecture": config["architecture"], "files": records, "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(), "snapshot_location": str(snapshot),
            "snapshot_revision_name": snapshot.name if snapshot.parent.name == "snapshots" else None,
            "identity_basis": "SHA256 of exact local checkpoint and config, not a mutable repository ref",
            "architecture_recipe_source": "https://huggingface.co/MahmoodLab/UNI2-h",
            "architecture_recipe": "patch14,depth24,heads24,embed1536,SwiGLUPacked,SiLU,8registers,no_embed_class,strict_state_dict"}


def resolve_fields(local_um, context_um, geometry, max_window_pixels=16777216):
    mpp = np.asarray(geometry["mpp_xy"], float)
    if not np.isclose(mpp[0], mpp[1], rtol=.001, atol=0):
        raise ValueError("Square physical fields currently require near-isotropic MPP (0.1% tolerance)")
    requested = np.array([local_um, context_um], float)
    if not np.isfinite(requested).all() or np.any(requested <= 0) or context_um <= local_um:
        raise ValueError("Context field must be strictly wider than a positive local field")
    widths = np.maximum(1, np.floor(requested / mpp.mean() + .5).astype(int))
    if widths[1] <= widths[0]:
        raise ValueError("Local and context fields become equal after source-pixel rounding")
    if np.any(widths.astype(float) ** 2 > max_window_pixels):
        raise ValueError("Requested field exceeds the bounded image-window pixel limit")
    return {name: {"requested_field_um": float(requested[i]), "field_pixels": int(widths[i]),
                   "actual_field_um_xy": (widths[i] * mpp).tolist(),
                   "effective_model_mpp_xy": (widths[i] * mpp / MODEL_INPUT_PIXELS).tolist()}
            for i, name in enumerate(("local", "context"))}


def feature_definitions(model, fields, geometry, min_coverage):
    definitions = {}
    for name, field in fields.items():
        definitions[name] = {"model_id": model["model_id"], "model_revision": model["model_revision"],
                             "pooling": "cls", "preprocessing": "RGB uint8 unchanged; complete square field resized directly to 224x224 with PIL bicubic; model mean/std normalization; no center crop, no per-tile contrast scaling",
                             "normalization_mean": model["normalization_mean"], "normalization_std": model["normalization_std"],
                             "field_width_source_px": field["field_pixels"], "input_context_width_source_px": field["field_pixels"],
                             "requested_field_um": field["requested_field_um"], "effective_model_mpp_xy": field["effective_model_mpp_xy"],
                             "model_input_size_px": MODEL_INPUT_PIXELS, "independent_transformer_context": True,
                             "field_generation": "separate_physical_image_crop_and_forward", "outside_crop_fill": "white_RGB_255",
                             "minimum_in_image_field_fraction": min_coverage,
                             "scale_shift_warning": "Changing field width changes effective model MPP; not validated as equivalent to UNI2 training scale"}
    return validate_feature_definitions(definitions, geometry)


def field_bounds(grid, width, shape):
    h, w = shape
    x0 = grid.x.to_numpy(np.int64) - width // 2
    y0 = grid.y.to_numpy(np.int64) - width // 2
    x1, y1 = x0 + width, y0 + width
    visible_x = np.maximum(0, np.minimum(w, x1) - np.maximum(0, x0))
    visible_y = np.maximum(0, np.minimum(h, y1) - np.maximum(0, y0))
    coverage = visible_x * visible_y / float(width * width)
    return x0, y0, x1, y1, coverage


def read_resized_field(reader, x0, y0, width):
    """Read only the requested physical field; retain its entire spatial extent."""
    x0, y0, width = int(x0), int(y0), int(width)
    x1, y1 = x0 + width, y0 + width
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(reader.width, x1), min(reader.height, y1)
    tile = np.full((width, width, 3), 255, np.uint8)
    if sx1 > sx0 and sy1 > sy0:
        block = reader.window(sx0, sy0, sx1, sy1)
        if block.dtype != np.uint8 or block.ndim != 3 or block.shape[2] != 3:
            raise ValueError("Hierarchy UNI2 fields require a calibrated RGB uint8 analysis crop")
        tile[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = block
    return np.asarray(Image.fromarray(tile).resize((MODEL_INPUT_PIXELS, MODEL_INPUT_PIXELS), Image.Resampling.BICUBIC))


def checked_relative_file(directory, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Feature bundle paths must be local relative paths")
    path = Path(directory) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_complete_field(directory, request_key, definition):
    manifest_path = directory / "embedding_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("request_key") != request_key or manifest.get("feature_definition") != definition:
        raise ValueError("Existing field has different inputs or feature definition; use a new output directory")
    for key in ("matrix", "rows"):
        record = manifest[key]
        path = checked_relative_file(directory, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Cached {key} checksum mismatch; preserving artifacts and refusing stale reuse")
    values = np.load(directory / manifest["matrix"]["path"], mmap_mode="r", allow_pickle=False)
    if list(values.shape) != manifest["matrix"]["shape"] or values.dtype != np.dtype("float32"):
        raise ValueError("Cached matrix shape/dtype mismatch")
    return manifest


def produce_field(reader, grid, field, definition, directory, request_key, encode, batch_size, min_coverage, source_inputs=None, sample_id=None, source_guard=None):
    directory.mkdir(exist_ok=True)
    cached = verify_complete_field(directory, request_key, definition)
    if cached is not None:
        return cached, True
    marker = directory / "field_request.json"
    if any(directory.iterdir()):
        if not marker.exists() or json.loads(marker.read_text()).get("request_key") != request_key:
            raise FileExistsError(f"Unrecognized existing field outputs in {directory}; preserving them")
    write_json(marker, {"request_key": request_key, "feature_definition": definition, "status": "in_progress"})
    x0, y0, x1, y1, coverage = field_bounds(grid, field["field_pixels"], (reader.height, reader.width))
    eligible = coverage >= min_coverage
    indices = np.flatnonzero(eligible)
    matrix = None
    temporary = directory / "features.inprogress.npy"
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        images = np.stack([read_resized_field(reader, x0[i], y0[i], field["field_pixels"]) for i in selected])
        features = np.asarray(encode(images), np.float32)
        if features.ndim != 2 or features.shape[0] != len(selected) or features.shape[1] < 1 or not np.isfinite(features).all():
            raise ValueError("Encoder returned invalid/nonfinite feature rows")
        if matrix is None:
            matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(grid), features.shape[1]))
            for row in range(0, len(grid), 4096):
                matrix[row:row+4096] = np.nan
        if features.shape[1] != matrix.shape[1]:
            raise ValueError("Encoder feature dimension changed between batches")
        matrix[selected] = features
    if matrix is None:
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(grid), 1536))
        for row in range(0, len(grid), 4096):
            matrix[row:row+4096] = np.nan
    matrix.flush()
    shape = list(matrix.shape)
    del matrix
    temporary.replace(directory / "features.npy")
    rows = pd.DataFrame({"cell_id": grid.label.astype(str), "cx": grid.x, "cy": grid.y, "observation_type": "grid",
                         "source_mpp": np.mean(definition["source_mpp_xy"]), "extraction_tile_size": field["field_pixels"],
                         "field_x0": x0, "field_y0": y0, "field_x1": x1, "field_y1": y1,
                         "in_image_field_fraction": coverage, "feature_status": np.where(eligible, "encoded", "insufficient_crop_coverage")})
    rows.to_csv(directory / "feature_rows.csv", index=False)
    manifest = {"schema_version": SCHEMA_VERSION, "format": "cellphenotyper_grid_embeddings_npy", "request_key": request_key,
                "source_inputs": source_inputs or {}, "sample_id": sample_id,
                "matrix": {"path": "features.npy", "sha256": sha256_file(directory / "features.npy"), "shape": shape, "dtype": "float32"},
                "rows": {"path": "feature_rows.csv", "sha256": sha256_file(directory / "feature_rows.csv"), "count": len(grid)},
                "feature_names": [f"feat_{i+1}" for i in range(shape[1])], "feature_definition": definition,
                "observations_encoded": int(eligible.sum()), "observations_missing": int((~eligible).sum()),
                "missing_semantics": "All-NaN feature rows retain grid IDs and parent support; no boundary padding treated as observed tissue"}
    # Do not publish a complete field from a mixed image/model/code population.
    # Incomplete arrays remain recoverable, but never acquire a success receipt.
    if source_guard is not None:
        source_guard()
    write_json(directory / "embedding_manifest.json", manifest)
    write_json(marker, {"request_key": request_key, "status": "complete"})
    return manifest, False


def default_encoder_factory(args, model):
    from extract_uni2_embeddings import load_local_uni2_encoder
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    encoder = load_local_uni2_encoder(args.model_snapshot, model["weights_filename"], args.device)
    # Detect replacement between preflight fingerprinting and actual model load.
    for filename, record in model["files"].items():
        if sha256_file(Path(args.model_snapshot) / filename) != record["sha256"]:
            raise RuntimeError("The UNI2 snapshot changed while loading; refusing mislabeled feature provenance")
    return encoder


def prepare(args, *, encoder_factory=None):
    if not 0 < args.minimum_field_coverage <= 1 or args.batch < 1 or args.batch > 256:
        raise ValueError("Field coverage must be in (0,1] and batch size in [1,256]")
    if args.torch_threads < 1 or args.max_window_pixels < 1:
        raise ValueError("Threads and window-pixel memory bound must be positive")
    root = Path(args.outdir)
    input_paths = {name: Path(getattr(args, name)).absolute() for name in
                   ("image", "grid_objects", "grid_metadata", "shift_json", "resolution_json")}
    # Capture before opening images or parsing geometry/grid metadata. A hash
    # calculated only after parsing cannot identify the bytes actually read.
    inputs = {name: {"sha256": sha256_file(path)} for name, path in input_paths.items()}
    code_root = Path(__file__).resolve().parent
    code = {name: sha256_file(code_root / name) for name in PRODUCER_CODE_FILES}
    model = inspect_snapshot(args.model_snapshot, args.weights_filename)
    sources = {str(input_paths[name]): value["sha256"] for name, value in inputs.items()}
    sources.update({str(code_root / name): digest for name, digest in code.items()})
    sources.update({str(Path(args.model_snapshot).absolute() / name): record["sha256"] for name, record in model["files"].items()})
    def source_guard():
        verify_source_identity(sources)
    source_guard()
    with RasterReader(args.image) as reader:
        probe = reader.window(0, 0, min(1, reader.width), min(1, reader.height))
        if probe.dtype != np.uint8 or probe.shape != (1, 1, 3):
            raise ValueError("Hierarchy feature production requires an RGB uint8 analysis crop")
        geometry = validate_geometry(json.loads(Path(args.grid_metadata).read_text()), json.loads(Path(args.shift_json).read_text()), json.loads(Path(args.resolution_json).read_text()), (reader.height, reader.width))
        grid = validate_grid(pd.read_csv(args.grid_objects), (reader.height, reader.width))
        fields = resolve_fields(args.local_field_um, args.context_field_um, geometry, args.max_window_pixels)
        definitions = feature_definitions(model, fields, geometry, args.minimum_field_coverage)
        source_guard()
        runtime = {name: package_version(name) for name in ("numpy", "pandas", "Pillow", "torch", "torchvision", "timm", "safetensors", "tifffile")}
        request = {"schema_version": SCHEMA_VERSION, "sample_id": args.sample_id, "inputs": inputs, "feature_definitions": definitions,
                   "model_files": model["files"], "geometry": geometry, "code": code, "runtime": runtime,
                   "seed": args.seed, "device": args.device, "batch": args.batch, "torch_threads": args.torch_threads, "precision": "float32"}
        request_key = canonical_hash(request)
        marker = root / "hierarchy_feature_request.json"
        if root.exists() and any(root.iterdir()):
            if not marker.exists() or json.loads(marker.read_text()).get("request_key") != request_key:
                raise FileExistsError("Output directory contains a different or unrecognized feature request; use a fresh directory")
        root.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"request_key": request_key, "request": request})
        # Load the large model only if an uncached eligible field needs it.
        encoder = None
        def encode(images):
            nonlocal encoder
            if encoder is None:
                encoder = (encoder_factory or default_encoder_factory)(args, model)
            return encoder(images)
        manifests, reused = {}, {}
        for name in ("local", "context"):
            source_guard()
            manifests[name], reused[name] = produce_field(reader, grid, fields[name], definitions[name], root / name, request_key, encode, args.batch, args.minimum_field_coverage, source_inputs=inputs, sample_id=args.sample_id, source_guard=source_guard)
            source_guard()
            print(f"[hierarchy-features] {name}: encoded={manifests[name]['observations_encoded']} missing={manifests[name]['observations_missing']} reused={reused[name]}", flush=True)
        write_json(root / "embedding_metadata.json", definitions)
        summary = {"schema_version": SCHEMA_VERSION, "request_key": request_key, "sample_id": args.sample_id,
                   "geometry": geometry, "model": model, "feature_definitions": definitions, "reused": reused,
                   "model_provenance": {"source_repository": f"https://huggingface.co/{model['model_id']}",
                                        "repo_id": model["model_id"], "requested_revision": "local_snapshot",
                                        "resolved_revision": model["model_revision"], "cache_path": model["snapshot_location"],
                                        "checkpoints": [{"logical_name": name, "sha256": record["sha256"], "size_bytes": record["size_bytes"], "cache_path": str(Path(model["snapshot_location"]) / name)} for name, record in model["files"].items()],
                                        "used_model": any(value["observations_encoded"] > 0 for value in manifests.values())},
                   "blocks": {name: {"directory": name, "manifest": f"{name}/embedding_manifest.json", "shape": manifests[name]["matrix"]["shape"]} for name in manifests},
                   "hierarchy_cli_inputs": {"local_embeddings": "local", "context_embeddings": "context", "embedding_metadata": "embedding_metadata.json"},
                   "memory_bounds": {"maximum_source_window_pixels": args.max_window_pixels, "model_batch_size": args.batch, "matrix_storage": "disk-backed float32 NPY", "full_slide_image_decode": False},
                   "scientific_status": "Independent physical fields, not validated histological subdomains. Larger fields downsampled to 224px alter effective model MPP and require held-out scale/quality benchmarking.",
                   "no_expert_annotations_used": True}
        source_guard()
        summary["sources_unchanged_after_generation"] = True
        summary["source_identity"] = sources
        summary["encoder_loaded_this_invocation"] = encoder is not None
        write_json(root / "hierarchy_features_summary.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "grid-objects", "grid-metadata", "shift-json", "resolution-json", "model-snapshot", "outdir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--weights-filename", default=None)
    parser.add_argument("--local-field-um", type=float, required=True)
    parser.add_argument("--context-field-um", type=float, required=True)
    parser.add_argument("--minimum-field-coverage", type=float, default=1.0, help="Incomplete image fields abstain by default; white padding is explicitly recorded if lowered")
    parser.add_argument("--max-window-pixels", type=int, default=16777216)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv=None):
    print(json.dumps(prepare(parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
