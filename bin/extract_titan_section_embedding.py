#!/usr/bin/env python3
"""Extract an official TITAN slide vector from a selected tissue section."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np


def source_patch_size(source_mpp: float, target_mpp: float, patch_size: int) -> int:
    if source_mpp <= 0 or target_mpp <= 0 or patch_size <= 0:
        raise ValueError("MPP and patch size must be positive")
    return max(1, int(round(patch_size * target_mpp / source_mpp)))


def feature_columns(dimension: int = 768) -> list[str]:
    return [f"titan_{index:03d}" for index in range(dimension)]


def install_local_titan_file_resolver(model_ref: str) -> Path | None:
    """Keep official TITAN helper lookups inside a complete local snapshot."""
    model_path = Path(model_ref).expanduser()
    model_dir = model_path.resolve()
    if not model_dir.is_dir():
        if model_path.is_absolute() or model_ref.startswith(("./", "../", "~")):
            raise FileNotFoundError(
                f"TITAN local model directory is not accessible: {model_ref}"
            )
        return None

    import huggingface_hub
    from transformers import PreTrainedTokenizerFast
    from transformers.dynamic_module_utils import HF_MODULES_CACHE

    # Transformers 4.57 does not copy nested relative imports when loading this
    # local remote-code repository. Stage the snapshot's Python modules only.
    module_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / model_dir.name
    module_dir.mkdir(parents=True, exist_ok=True)
    for package_dir in (Path(HF_MODULES_CACHE), module_dir.parent, module_dir):
        (package_dir / "__init__.py").touch(exist_ok=True)
    for source in model_dir.glob("*.py"):
        shutil.copy2(source, module_dir / source.name)

    original_download = huggingface_hub.hf_hub_download
    original_tokenizer_load = PreTrainedTokenizerFast.from_pretrained.__func__

    def resolve_from_snapshot(repo_id: str, filename: str, *args, **kwargs):
        if repo_id == "MahmoodLab/TITAN":
            candidate = model_dir / filename
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"TITAN local snapshot is missing required file: {candidate}"
                )
            return str(candidate)
        return original_download(repo_id, filename, *args, **kwargs)

    huggingface_hub.hf_hub_download = resolve_from_snapshot

    @classmethod
    def load_tokenizer_from_snapshot(cls, pretrained_model_name_or_path, *args, **kwargs):
        if str(pretrained_model_name_or_path) == "MahmoodLab/TITAN":
            pretrained_model_name_or_path = str(model_dir)
            kwargs["local_files_only"] = True
        return original_tokenizer_load(cls, pretrained_model_name_or_path, *args, **kwargs)

    PreTrainedTokenizerFast.from_pretrained = load_tokenizer_from_snapshot
    return model_dir


def read_source_mpp(path: Path, fallback: float) -> float:
    payload = json.loads(path.read_text())
    value = payload.get("source_mpp", payload.get("microns_per_pixel", fallback))
    if value is None or float(value) <= 0:
        raise RuntimeError("No valid source MPP in selected-section metadata")
    return float(value)


def auto_batch_size(requested: int, gpu: int) -> int:
    if requested > 0:
        return requested
    try:
        import subprocess

        memory_mib = int(subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0])
    except Exception:
        return 8
    if memory_mib >= 70 * 1024:
        return 128
    if memory_mib >= 40 * 1024:
        return 64
    if memory_mib >= 20 * 1024:
        return 24
    if memory_mib >= 14 * 1024:
        return 12
    return 4


def unwrap_patch_features(output):
    import torch

    if isinstance(output, dict):
        for key in ("image_features", "pooler_output", "last_hidden_state"):
            if key in output:
                output = output[key]
                break
    elif hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    elif isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"Unsupported CONCH v1.5 output type: {type(output)!r}")
    if output.ndim == 3:
        output = output[:, 0, :]
    if output.ndim != 2 or output.shape[1] != 768:
        raise RuntimeError(f"Expected CONCH v1.5 patch features [N,768], got {tuple(output.shape)}")
    return output


def tile_candidates(mask_path: Path, source_tile: int, min_coverage: float):
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(mask_path) as mask:
        for y in range(0, mask.height, source_tile):
            for x in range(0, mask.width, source_tile):
                h, w = min(source_tile, mask.height - y), min(source_tile, mask.width - x)
                tile = mask.read(1, window=Window(x, y, w, h))
                coverage = float(np.count_nonzero(tile)) / float(source_tile * source_tile)
                if coverage >= min_coverage:
                    yield x, y, w, h, coverage


def vips_region_to_pil(image, x: int, y: int, width: int, height: int, source_tile: int, patch_size: int):
    from PIL import Image

    region = image.crop(x, y, width, height)
    if region.bands > 3:
        region = region[:3]
    if region.bands == 1:
        region = region.bandjoin([region, region])
    if width != source_tile or height != source_tile:
        region = region.embed(0, 0, source_tile, source_tile, extend="white")
    if source_tile != patch_size:
        region = region.resize(patch_size / source_tile, kernel="lanczos3")
    if region.format != "uchar":
        region = region.cast("uchar")
    array = np.ndarray(
        buffer=region.write_to_memory(), dtype=np.uint8,
        shape=(region.height, region.width, region.bands),
    )[:, :, :3].copy()
    return Image.fromarray(array, mode="RGB")


def write_embedding_csv(path: Path, sample_id: str, section_id: str, embedding: np.ndarray) -> None:
    values = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if values.size != 768 or not np.isfinite(values).all():
        raise RuntimeError(f"TITAN must emit 768 finite values, got shape {values.shape}")
    columns = ["sample_id", "section_id", *feature_columns()]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow([sample_id, section_id, *[f"{value:.9g}" for value in values]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--section-summary", required=True)
    parser.add_argument("--shift", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model", default="MahmoodLab/TITAN")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--target-mpp", type=float, default=0.5)
    parser.add_argument("--default-mpp", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--min-tissue-coverage", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--save-patch-preview", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import h5py
    import huggingface_hub
    import pyvips
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("TITAN is configured as a GPU-only pipeline stage, but CUDA is unavailable")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.section_summary).read_text())
    section_id = str(summary["selected_section_id"])
    mpp = read_source_mpp(Path(args.shift), args.default_mpp)
    source_tile = source_patch_size(mpp, args.target_mpp, args.patch_size)
    batch_size = auto_batch_size(args.batch_size, args.gpu)
    local_model_dir = install_local_titan_file_resolver(args.model)

    model_source = str(local_model_dir) if local_model_dir else args.model
    model = AutoModel.from_pretrained(
        model_source, revision=args.revision, trust_remote_code=True,
        local_files_only=args.offline,
    ).eval().to(device)
    conch, transform = model.return_conch()
    conch = conch.eval().to(device)
    image = pyvips.Image.new_from_file(args.image, access="random")
    candidates = list(tile_candidates(Path(args.mask), source_tile, args.min_tissue_coverage))
    if not candidates:
        raise RuntimeError("No TITAN patches passed the selected-section tissue threshold")

    h5_path = outdir / "titan_patch_features.h5"
    with h5py.File(h5_path, "w") as handle:
        feature_ds = handle.create_dataset(
            "features", shape=(len(candidates), 768), dtype="float32",
            chunks=(min(256, len(candidates)), 768), compression="gzip",
        )
        coords = np.zeros((len(candidates), 2), dtype=np.int64)
        coverage = np.zeros(len(candidates), dtype=np.float32)
        for start in range(0, len(candidates), batch_size):
            stop = min(len(candidates), start + batch_size)
            images = []
            for index in range(start, stop):
                x, y, width, height, fraction = candidates[index]
                images.append(vips_region_to_pil(
                    image, x, y, width, height, source_tile, args.patch_size,
                ))
                coords[index] = [round(x / source_tile) * args.patch_size, round(y / source_tile) * args.patch_size]
                coverage[index] = fraction
            batch = torch.stack([transform(patch) for patch in images]).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                features = unwrap_patch_features(conch(batch))
            feature_ds[start:stop] = features.float().cpu().numpy()
        coord_ds = handle.create_dataset("coords", data=coords, compression="gzip")
        handle.create_dataset("tissue_coverage", data=coverage, compression="gzip")
        coord_ds.attrs["patch_size_level0"] = args.patch_size
        handle.attrs["source_mpp"] = mpp
        handle.attrs["target_mpp"] = args.target_mpp
        handle.attrs["source_patch_size"] = source_tile
        handle.attrs["coordinate_space"] = "regularized_20x_grid_pixels"

    with h5py.File(h5_path, "r") as handle:
        patch_features = torch.from_numpy(handle["features"][:]).unsqueeze(0).to(device)
        patch_coords = torch.from_numpy(handle["coords"][:]).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embedding = model.encode_slide_from_patch_features(
            patch_features, patch_coords, args.patch_size,
        )
    embedding = embedding.float().cpu().numpy().reshape(-1)
    write_embedding_csv(outdir / "titan_embedding.csv", args.sample_id, section_id, embedding)
    np.save(outdir / "titan_embedding.npy", embedding.astype(np.float32))

    gpu_name = torch.cuda.get_device_name(args.gpu)
    metadata = {
        "sample_id": args.sample_id,
        "section_id": section_id,
        "model": args.model,
        "resolved_model_source": model_source,
        "revision": args.revision,
        "patch_encoder": "CONCH v1.5",
        "patch_count": len(candidates),
        "patch_feature_dimension": 768,
        "slide_embedding_dimension": int(embedding.size),
        "source_mpp": mpp,
        "target_mpp": args.target_mpp,
        "source_patch_size": source_tile,
        "patch_size_level0": args.patch_size,
        "min_tissue_coverage": args.min_tissue_coverage,
        "batch_size": batch_size,
        "device": str(device),
        "gpu_name": gpu_name,
        "hf_home": os.environ.get("HF_HOME", ""),
        "local_model_snapshot": str(local_model_dir) if local_model_dir else "",
        "huggingface_hub_version": huggingface_hub.__version__,
    }
    (outdir / "titan_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
