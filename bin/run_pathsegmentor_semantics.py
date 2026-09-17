#!/usr/bin/env python3
"""Run a pinned PathSegmentor checkpoint as an optional semantic evidence layer.

The output is a coarse, physically calibrated probability stack constrained by
the supplied GrandQC clean-tissue mask. It does not modify KODAMA labels and it
does not convert semantic masks into cell instances.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import tifffile


PROMPT_ID = re.compile(r"[a-z][a-z0-9_]{1,63}")


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(block), b""):
            digest.update(data)
    return digest.hexdigest()


def load_prompt_panel(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("PathSegmentor prompt panel requires a non-empty prompts list")
    seen: set[str] = set()
    for item in prompts:
        if not isinstance(item, dict):
            raise ValueError("Every PathSegmentor prompt must be an object")
        prompt_id = str(item.get("id", ""))
        text = str(item.get("text", "")).strip()
        if not PROMPT_ID.fullmatch(prompt_id) or prompt_id in seen:
            raise ValueError(f"Invalid or duplicate PathSegmentor prompt id: {prompt_id!r}")
        if not text or any(ord(char) < 32 for char in text):
            raise ValueError(f"Invalid PathSegmentor prompt text for {prompt_id}")
        seen.add(prompt_id)
    return payload


def axis_starts(length: int, window: int, stride: int) -> list[int]:
    if min(length, window, stride) < 1:
        raise ValueError("length, window and stride must be positive")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def blend_window(height: int, width: int, floor: float = 0.05) -> np.ndarray:
    if height < 1 or width < 1 or not 0 < floor <= 1:
        raise ValueError("Invalid blend-window geometry")
    wy = np.hanning(max(3, height))[:height]
    wx = np.hanning(max(3, width))[:width]
    return np.maximum(np.outer(wy, wx), float(floor)).astype(np.float32)


def _checkpoint_state(path: Path, torch: Any) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError("PathSegmentor checkpoint does not contain a tensor state dictionary")
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def load_pathsegmentor(repo: Path, config: Path, checkpoint: Path, device: str):
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PathSegmentor is configured as a CUDA-only experimental stage")
    repo = repo.resolve()
    if not (repo / "modeling").is_dir() or not (repo / "utilities").is_dir():
        raise ValueError("--repo is not a PathSegmentor source checkout")
    for path, label in ((config, "config"), (checkpoint, "checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"PathSegmentor {label} is missing: {path}")

    sys.path.insert(0, str(repo))
    from modeling import build_model
    from utilities.arguments import load_opt_from_config_files
    from utilities.model import align_and_update_state_dicts

    opt = load_opt_from_config_files([str(config.resolve())])
    opt.update(
        CUDA=True,
        device=torch.device("cuda", 0),
        world_size=1,
        local_size=1,
        rank=0,
        local_rank=0,
        env_info="CellPhenotyper single-GPU inference",
    )
    model = build_model(opt)
    supplied = _checkpoint_state(checkpoint, torch)
    aligned = align_and_update_state_dicts(model.state_dict(), supplied)
    model_state = model.state_dict()
    loaded_numel = sum(int(model_state[key].numel()) for key in aligned)
    total_numel = sum(int(value.numel()) for value in model_state.values())
    coverage = loaded_numel / max(1, total_numel)
    if coverage < 0.995:
        raise RuntimeError(
            f"PathSegmentor checkpoint tensor coverage is {coverage:.6f}; at least 0.995 is required"
        )
    incompat = model.load_state_dict(aligned, strict=False)
    materially_missing = [key for key in incompat.missing_keys if model_state[key].numel() > 0]
    if materially_missing or incompat.unexpected_keys:
        raise RuntimeError(
            "PathSegmentor checkpoint is incompatible: "
            f"missing={len(materially_missing)} unexpected={len(incompat.unexpected_keys)}"
        )
    model = model.eval().cuda()
    return model, opt, coverage


def infer_prompt_probabilities(model: Any, image: Image.Image, prompts: list[str]):
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import InterpolationMode, Resize
    from modeling.language.loss import vl_similarity

    resized = Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC)(image.convert("RGB"))
    image_np = np.asarray(resized)
    tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).cuda()
    model.task_switch.update(spatial=False, visual=False, grounding=True, audio=False)
    data = {"image": tensor, "text": prompts, "height": 1024, "width": 1024}
    with torch.inference_mode():
        results, image_size, extra = model.evaluate_demo([data])
        pred_masks = results["pred_masks"][0]
        visual = results["pred_captions"][0]
        text = extra["grounding_class"]
        visual = visual / (visual.norm(dim=-1, keepdim=True) + 1e-7)
        text = text / (text.norm(dim=-1, keepdim=True) + 1e-7)
        similarity = vl_similarity(
            visual, text, temperature=model.sem_seg_head.predictor.lang_encoder.logit_scale
        )
        matched = similarity.max(0).indices
        selected = pred_masks[matched]
        probability = F.interpolate(
            selected[:, None], image_size[-2:], mode="bilinear", align_corners=False
        )[:, 0, :1024, :1024].sigmoid()
        confidence = similarity.max(0).values
    return probability.float().cpu().numpy(), confidence.float().cpu().numpy(), matched.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tissue-mask", type=Path, required=True)
    parser.add_argument("--resolution-json", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-panel", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--target-mpp", type=float, default=0.25)
    parser.add_argument("--output-mpp", type=float, default=2.0)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap-fraction", type=float, default=0.25)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.01)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args()

    if args.target_mpp <= 0 or args.output_mpp < args.target_mpp:
        raise ValueError("output MPP must be at least the positive PathSegmentor target MPP")
    if args.tile_size != 1024:
        raise ValueError("The pinned PathSegmentor adapter requires 1024-pixel model inputs")
    if not 0 <= args.overlap_fraction < 0.9 or not 0 <= args.min_tissue_fraction <= 1:
        raise ValueError("Invalid overlap or tissue-fraction threshold")

    panel = load_prompt_panel(args.prompt_panel)
    prompt_records = panel["prompts"]
    prompts = [str(item["text"]) for item in prompt_records]
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Reuse the pipeline's tested tiled readers without loading the WSI in RAM.
    from extract_uni2_embeddings import LabelReader, RegionReader, read_resolution_json_mpp

    source_mpp = read_resolution_json_mpp(args.resolution_json)
    if source_mpp is None or source_mpp <= 0:
        raise ValueError("PathSegmentor requires a valid source MPP report")
    image_reader = RegionReader(args.image, level=0, force_full_image=False)
    tissue_reader = LabelReader(args.tissue_mask)
    height, width = map(int, image_reader.shape[:2])
    if tuple(tissue_reader.shape) != (height, width):
        raise ValueError("PathSegmentor image and GrandQC tissue mask are not aligned")

    model, _opt, state_coverage = load_pathsegmentor(
        args.repo, args.config, args.checkpoint, args.device
    )
    source_window = max(1, int(round(args.tile_size * args.target_mpp / source_mpp)))
    stride = max(1, int(round(source_window * (1.0 - args.overlap_fraction))))
    out_h = max(1, int(math.ceil(height * source_mpp / args.output_mpp)))
    out_w = max(1, int(math.ceil(width * source_mpp / args.output_mpp)))
    n_prompts = len(prompts)

    sum_path = args.outdir / ".semantic_sum.float32"
    weight_path = args.outdir / ".semantic_weight.float32"
    sums = np.memmap(sum_path, mode="w+", dtype=np.float32, shape=(n_prompts, out_h, out_w))
    weights = np.memmap(weight_path, mode="w+", dtype=np.float32, shape=(out_h, out_w))
    sums[:] = 0
    weights[:] = 0
    tile_rows: list[dict[str, Any]] = []
    scale_out = source_mpp / args.output_mpp

    for y0 in axis_starts(height, source_window, stride):
        for x0 in axis_starts(width, source_window, stride):
            tissue = tissue_reader.read(x0, y0, source_window, source_window) > 0
            tissue_fraction = float(tissue.mean())
            if tissue_fraction < args.min_tissue_fraction:
                continue
            rgb = image_reader.read(x0, y0, source_window, source_window)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[:, :, None], 3, axis=2)
            rgb = np.asarray(rgb[:, :, :3])
            if rgb.dtype != np.uint8:
                finite = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
                if finite.max(initial=0) <= 1.0:
                    finite *= 255.0
                rgb = np.clip(finite, 0, 255).astype(np.uint8)
            probs, confidence, matched = infer_prompt_probabilities(
                model, Image.fromarray(rgb, mode="RGB"), prompts
            )
            ox0 = int(round(x0 * scale_out))
            oy0 = int(round(y0 * scale_out))
            ox1 = min(out_w, int(round((x0 + source_window) * scale_out)))
            oy1 = min(out_h, int(round((y0 + source_window) * scale_out)))
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            target_shape = (oy1 - oy0, ox1 - ox0)
            from PIL import Image as PILImage
            weight = blend_window(*target_shape)
            support = np.asarray(
                PILImage.fromarray(tissue.astype(np.uint8)).resize(
                    (target_shape[1], target_shape[0]), resample=PILImage.Resampling.NEAREST
                ), dtype=bool
            )
            weight *= support
            weights[oy0:oy1, ox0:ox1] += weight
            for index in range(n_prompts):
                probability = np.asarray(
                    PILImage.fromarray(probs[index], mode="F").resize(
                        (target_shape[1], target_shape[0]), resample=PILImage.Resampling.BILINEAR
                    ), dtype=np.float32
                )
                sums[index, oy0:oy1, ox0:ox1] += probability * weight
                tile_rows.append(
                    {
                        "tile_x0": x0,
                        "tile_y0": y0,
                        "source_window_px": source_window,
                        "tissue_fraction": tissue_fraction,
                        "prompt_id": prompt_records[index]["id"],
                        "text_similarity": float(confidence[index]),
                        "matched_query": int(matched[index]),
                    }
                )

    denominator = np.maximum(np.asarray(weights), 1e-7)
    output_path = args.outdir / "pathsegmentor_probabilities.ome.tif"
    output = np.empty((n_prompts, out_h, out_w), dtype=np.float32)
    covered = np.asarray(weights) > 0
    for index in range(n_prompts):
        output[index] = np.asarray(sums[index]) / denominator
        output[index, ~covered] = 0.0
    tifffile.imwrite(
        output_path,
        output,
        bigtiff=True,
        compression="zlib",
        metadata={"axes": "CYX", "Channel": {"Name": [item["id"] for item in prompt_records]}},
        resolution=(10000.0 / args.output_mpp, 10000.0 / args.output_mpp),
        resolutionunit="CENTIMETER",
    )
    scores_path = args.outdir / "pathsegmentor_tile_scores.csv.gz"
    import gzip
    with gzip.open(scores_path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0]) if tile_rows else [
            "tile_x0", "tile_y0", "source_window_px", "tissue_fraction", "prompt_id",
            "text_similarity", "matched_query",
        ])
        writer.writeheader()
        writer.writerows(tile_rows)

    revision_result = subprocess.run(
        ["git", "-C", str(args.repo.resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    repository_revision = revision_result.stdout.strip() if revision_result.returncode == 0 else None
    manifest = {
        "schema_version": "1.0.0",
        "scientific_role": "supervised_semantic_evidence_not_unsupervised_cluster_identity",
        "image": str(args.image.resolve()),
        "image_sha256": sha256_file(args.image),
        "tissue_mask": str(args.tissue_mask.resolve()),
        "tissue_mask_sha256": sha256_file(args.tissue_mask),
        "prompt_panel": panel,
        "prompt_panel_sha256": sha256_file(args.prompt_panel),
        "pathsegmentor_repository": str(args.repo.resolve()),
        "pathsegmentor_repository_revision": repository_revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_tensor_coverage": state_coverage,
        "source_shape_yx": [height, width],
        "source_mpp": source_mpp,
        "target_mpp": args.target_mpp,
        "output_mpp": args.output_mpp,
        "output_shape_cyx": [n_prompts, out_h, out_w],
        "source_window_px": source_window,
        "overlap_fraction": args.overlap_fraction,
        "min_tissue_fraction": args.min_tissue_fraction,
        "processed_tile_prompt_rows": len(tile_rows),
        "probability_stack": output_path.name,
        "tile_scores": scores_path.name,
    }
    (args.outdir / "pathsegmentor_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del sums, weights
    sum_path.unlink(missing_ok=True)
    weight_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
