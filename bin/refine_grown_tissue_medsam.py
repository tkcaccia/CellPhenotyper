#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite
import tifffile

from grow_to_tissue import (
    DEFAULT_PALETTE,
    ensure_2d,
    load_mask_2d,
    load_tiff,
    pyramidize_with_raw2ometiff,
    save_preview_png,
)
from grow_to_tissue_core import compute_boundary, to_float_rgb
from medsam_border_refine import DEFAULT_MEDSAM_CHECKPOINT, MedSAMConfig, MedSAMUnavailableError, run_medsam_border_refine


def sample_prefix(sample_id: str, out_path: str | Path) -> str:
    out_name = Path(out_path).name
    if out_name.endswith(".ome.tif"):
        out_name = out_name[:-8]
    if out_name.endswith(".tif"):
        out_name = out_name[:-4]
    return sample_id or out_name


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return (to_float_rgb(image) * 255.0).astype(np.uint8)


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    for idx, lid in enumerate(unique):
        out[labels == lid] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return out


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    b = np.zeros(labels.shape, dtype=bool)
    b[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    b[1:, :] |= labels[1:, :] != labels[:-1, :]
    b &= labels > 0
    return b


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    mask = np.asarray(mask).astype(bool)
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def overlay_labels(image: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    labels = np.asarray(labels).astype(np.int32)
    colors = colorize_labels(labels).astype(np.float32)
    mask = labels > 0
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * colors[mask]
    boundaries = label_boundaries(labels)
    rgb[boundaries] = colors[boundaries]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def boundary_compare(image: np.ndarray, raw_mask: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).copy()
    b0 = compute_boundary(raw_mask)
    b1 = compute_boundary(final_mask)
    rgb[b0 & ~b1] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[b1 & ~b0] = np.array([0, 255, 255], dtype=np.uint8)
    rgb[b0 & b1] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def heatmap(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob).astype(np.float32), 0.0, 1.0)
    out = np.zeros(p.shape + (3,), dtype=np.uint8)
    out[..., 0] = np.clip(255.0 * p, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(255.0 * (1.0 - np.abs(p - 0.5) * 2.0), 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(255.0 * (1.0 - p), 0, 255).astype(np.uint8)
    return out


def fit_panel(arr: np.ndarray, width: int = 420) -> Image.Image:
    img = Image.fromarray(arr)
    h = int(round(img.height * (width / img.width)))
    return img.resize((width, h), Image.Resampling.BILINEAR)


def make_panel(items: List[Tuple[str, np.ndarray]], out_path: Path, columns: int = 3) -> None:
    font = ImageFont.load_default()
    panels = [(title, fit_panel(arr)) for title, arr in items]
    cell_w = max(img.width for _, img in panels)
    cell_h = max(img.height for _, img in panels)
    title_h = 22
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_w, rows * (cell_h + title_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (title, img) in enumerate(panels):
        r = idx // columns
        c = idx % columns
        x = c * cell_w
        y = r * (cell_h + title_h)
        canvas.paste(img, (x, y + title_h))
        draw.text((x + 6, y + 4), title, fill=(0, 0, 0), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def preview_step_for_shape(shape: Tuple[int, int], max_side: int = 2048) -> int:
    h, w = int(shape[0]), int(shape[1])
    return max(1, int(np.ceil(max(h, w) / float(max_side))))


def preview_subsample(arr: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    if step == 1:
        return np.asarray(arr)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[::step, ::step]
    return arr[::step, ::step, ...]


def load_tiff_memmap(path: str, name: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    try:
        return tifffile.memmap(path)
    except Exception:
        return load_tiff(path, name)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine the step-16 grown tissue mask with MedSAM while preserving the protected core and seeded labels."
    )
    ap.add_argument("--sample-id", required=True)
    ap.add_argument("--image", required=True, help="ROI image TIFF used by the pipeline")
    ap.add_argument("--seed-mask", required=True, help="Step-15 cluster mask TIFF")
    ap.add_argument("--grown-mask", required=True, help="Step-16 grown tissue mask TIFF")
    ap.add_argument("--out", required=True, help="Output refined OME-TIFF path")
    ap.add_argument("--preview", required=True, help="Output preview PNG path")
    ap.add_argument("--preview-factor", type=int, default=10)
    ap.add_argument("--preview-threshold-mb", type=float, default=100.0)
    ap.add_argument("--preview-alpha", type=float, default=0.45)
    ap.add_argument("--pyr-compression", default="LZW")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--downsample", default="GAUSSIAN")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")

    ap.add_argument("--medsam-checkpoint", default=str(DEFAULT_MEDSAM_CHECKPOINT))
    ap.add_argument("--medsam-device", default="cuda")
    ap.add_argument("--medsam-bbox-margin", type=int, default=144)
    ap.add_argument("--medsam-component-min-area", type=int, default=200)
    ap.add_argument("--medsam-component-merge-distance", type=int, default=24)
    ap.add_argument("--medsam-seed-dilation-radius", type=int, default=8)
    ap.add_argument("--medsam-core-erosion-radius", type=int, default=36)
    ap.add_argument("--medsam-outer-dilation-radius", type=int, default=44)
    ap.add_argument("--medsam-min-object-size", type=int, default=5000)
    ap.add_argument("--medsam-smooth-radius", type=int, default=5)
    ap.add_argument("--medsam-cluster-tile-size", type=int, default=4096)
    ap.add_argument("--medsam-cluster-tile-overlap", type=int, default=512)
    ap.add_argument("--medsam-force-core-preservation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--medsam-save-debug", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    image_rgb = load_tiff_memmap(args.image, "image")
    seed_labels = ensure_2d(load_tiff_memmap(args.seed_mask, "seed mask"), "seed mask")
    grown_labels = ensure_2d(load_tiff_memmap(args.grown_mask, "grown mask"), "grown mask")
    label_dtype = seed_labels.dtype
    if np.issubdtype(seed_labels.dtype, np.integer):
        min_label = int(np.min(seed_labels, initial=0))
        max_label = int(np.max(seed_labels, initial=0))
        if min_label >= 0 and max_label <= np.iinfo(np.uint16).max:
            label_dtype = np.uint16
    seed_labels = seed_labels.astype(label_dtype, copy=False)
    grown_labels = grown_labels.astype(label_dtype, copy=False)

    if grown_labels.shape != seed_labels.shape:
        raise ValueError(f"Step-15 and step-16 masks must share the same shape: {seed_labels.shape} vs {grown_labels.shape}")

    if image_rgb.ndim == 3 and image_rgb.shape[:2] != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape[:2]} vs {seed_labels.shape}")
    if image_rgb.ndim == 2 and image_rgb.shape != seed_labels.shape:
        raise ValueError(f"Image and masks must share the same spatial shape: {image_rgb.shape} vs {seed_labels.shape}")

    med_cfg = MedSAMConfig(
        checkpoint=str(args.medsam_checkpoint),
        device=str(args.medsam_device),
        bbox_margin=int(args.medsam_bbox_margin),
        component_min_area=int(args.medsam_component_min_area),
        component_merge_distance=int(args.medsam_component_merge_distance),
        seed_dilation_radius=int(args.medsam_seed_dilation_radius),
        core_erosion_radius=int(args.medsam_core_erosion_radius),
        outer_dilation_radius=int(args.medsam_outer_dilation_radius),
        min_object_size=int(args.medsam_min_object_size),
        smooth_radius=int(args.medsam_smooth_radius),
        force_core_preservation=bool(args.medsam_force_core_preservation),
        save_debug=bool(args.medsam_save_debug),
        cluster_tile_size=int(args.medsam_cluster_tile_size),
        cluster_tile_overlap=int(args.medsam_cluster_tile_overlap),
    )

    try:
        refined_tissue, probability_map, runtime_sec, med_meta, artifacts = run_medsam_border_refine(
            image=image_rgb,
            seed_labels=seed_labels,
            baseline_tissue_mask=grown_labels > 0,
            config=med_cfg,
            baseline_label_map=grown_labels,
        )
    except MedSAMUnavailableError as exc:
        raise RuntimeError(f"MedSAM unavailable: {exc}") from exc

    refined_labels = np.asarray(artifacts.get("label_map", grown_labels), dtype=label_dtype)
    raw_labels = np.asarray(artifacts.get("raw_medsam_label_map", refined_labels), dtype=label_dtype)
    protected_core = np.asarray(artifacts.get("protected_core", np.zeros_like(grown_labels)), dtype=bool)
    editable_band = np.asarray(artifacts.get("editable_band", np.zeros_like(grown_labels)), dtype=bool)

    outdir = Path(args.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = sample_prefix(args.sample_id, args.out)
    debug_dir = outdir / f"{prefix}_medsam_debug"

    if med_cfg.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(med_meta)
        meta["sample_id"] = args.sample_id
        meta["raw_pixels"] = int((raw_labels > 0).sum())
        meta["final_pixels"] = int((refined_labels > 0).sum())
        (debug_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        for name, arr in artifacts.items():
            np.save(debug_dir / f"{name}.npy", np.asarray(arr))
        if probability_map is not None:
            np.save(debug_dir / "probability_map.npy", np.asarray(probability_map, dtype=np.float32))

    summary = {
        "sample_id": args.sample_id,
        "medsam_runtime_sec": round(float(runtime_sec), 3),
        "raw_pixels": int((raw_labels > 0).sum()),
        "final_pixels": int((refined_labels > 0).sum()),
        "cleanup_removed_pixels": int(((raw_labels > 0) & ~(refined_labels > 0)).sum()),
        "cleanup_added_pixels": int(((refined_labels > 0) & ~(raw_labels > 0)).sum()),
        "cleanup_relabel_pixels": int(((raw_labels != refined_labels) & (raw_labels > 0) & (refined_labels > 0)).sum()),
        "protected_core_pixels": int(protected_core.sum()),
        "editable_band_pixels": int(editable_band.sum()),
    }
    diag_step = preview_step_for_shape(seed_labels.shape, max_side=2048)
    summary["diagnostic_preview_step"] = int(diag_step)
    (outdir / f"{prefix}_medsam_summary.json").write_text(json.dumps(summary, indent=2))

    image_preview = preview_subsample(image_rgb, diag_step)
    seed_preview = preview_subsample(seed_labels, diag_step)
    raw_preview = preview_subsample(raw_labels, diag_step)
    refined_preview = preview_subsample(refined_labels, diag_step)
    protected_preview = preview_subsample(protected_core, diag_step)
    editable_preview = preview_subsample(editable_band, diag_step)
    prob_preview = preview_subsample(probability_map, diag_step) if probability_map is not None else None

    raw_vs_final_change = to_uint8_rgb(image_preview)
    raw_only = (raw_preview > 0) & ~(refined_preview > 0)
    final_only = (refined_preview > 0) & ~(raw_preview > 0)
    relabeled = (raw_preview != refined_preview) & (raw_preview > 0) & (refined_preview > 0)
    raw_vs_final_change[raw_only] = ((0.65 * raw_vs_final_change[raw_only]) + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
    raw_vs_final_change[final_only] = ((0.65 * raw_vs_final_change[final_only]) + 0.35 * np.array([0, 255, 255])).astype(np.uint8)
    raw_vs_final_change[relabeled] = ((0.55 * raw_vs_final_change[relabeled]) + 0.45 * np.array([255, 255, 0])).astype(np.uint8)

    save_png(outdir / f"{prefix}_medsam_seed_labels.png", overlay_labels(image_preview, seed_preview))
    save_png(outdir / f"{prefix}_medsam_protected_core.png", overlay_mask(image_preview, protected_preview, (255, 200, 0)))
    save_png(outdir / f"{prefix}_medsam_editable_band.png", overlay_mask(image_preview, editable_preview, (180, 0, 255)))
    save_png(outdir / f"{prefix}_medsam_raw_labels.png", overlay_labels(image_preview, raw_preview))
    save_png(outdir / f"{prefix}_medsam_refined_labels.png", overlay_labels(image_preview, refined_preview))
    if prob_preview is not None:
        save_png(outdir / f"{prefix}_medsam_probability_heatmap.png", heatmap(prob_preview))
    save_png(outdir / f"{prefix}_medsam_boundary_compare.png", boundary_compare(image_preview, raw_preview > 0, refined_preview > 0))

    panel_path = outdir / f"{prefix}_medsam_raw_vs_final_panel.png"
    make_panel(
        [
            ("Original image", to_uint8_rgb(image_preview)),
            ("Step-15 labels", overlay_labels(image_preview, seed_preview)),
            ("Protected core", overlay_mask(image_preview, protected_preview, (255, 200, 0))),
            ("Editable band", overlay_mask(image_preview, editable_preview, (180, 0, 255))),
            ("Immediately after MedSAM", overlay_labels(image_preview, raw_preview)),
            ("After refinements", overlay_labels(image_preview, refined_preview)),
            ("What refinements changed", raw_vs_final_change),
        ],
        panel_path,
        columns=3,
    )

    tmp_flat = outdir / f".tmp_refined_flat_{os.getpid()}_{int(time.time())}.tif"
    maxlab = int(refined_labels.max()) if refined_labels.size else 0
    flat = refined_labels.astype(np.uint16) if maxlab <= 65535 else refined_labels.astype(np.uint32)
    imwrite(tmp_flat, flat)
    try:
        pyramidize_with_raw2ometiff(
            in_tif=str(tmp_flat),
            out_ome_tif=str(args.out),
            compression=args.pyr_compression,
            max_workers=args.max_workers,
            downsample=args.downsample,
            overwrite=args.overwrite,
            keep_tmp=args.keep_tmp,
            legacy=args.legacy,
        )
    except Exception as exc:
        print(f"[WARN] Pyramidal conversion failed ({exc}). Writing flat TIFF fallback to: {args.out}")
        out_path = Path(args.out)
        if out_path.exists():
            if args.overwrite:
                out_path.unlink()
            else:
                raise
        shutil.copyfile(tmp_flat, args.out)
    finally:
        try:
            tmp_flat.unlink()
        except Exception:
            pass

    save_preview_png(
        image_path=args.image,
        grown=refined_labels,
        out_png=args.preview,
        factor=args.preview_factor,
        size_threshold_mb=args.preview_threshold_mb,
        alpha=args.preview_alpha,
        default_value=0,
    )

    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote refined mask: {args.out}")
    print(f"[OK] wrote refinement panel: {panel_path}")


if __name__ == "__main__":
    main()
