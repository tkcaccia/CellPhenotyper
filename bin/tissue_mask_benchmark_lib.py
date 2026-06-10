from __future__ import annotations

import colorsys
import html
import importlib.util
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

try:
    import maxflow
except Exception:
    maxflow = None
import numpy as np
import pandas as pd
try:
    import pywt
except Exception:
    pywt = None
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage import color, feature, filters, measure, morphology, segmentation, transform, util

from grow_to_tissue_core import (
    DEFAULT_CONFIG,
    METHODS as CORE_METHODS,
    GrowConfig,
    build_background_likelihood,
    build_feature_stack,
    build_stats_masks,
    build_tissue_probability,
    cleanup_mask,
    compute_boundary,
    config_dict,
    grow_labels_within_mask,
    keep_seeded_components,
    make_signal_bundle,
    model_distance,
    normalize_method_output,
    read_tiff_level0,
    refine_probability_gate,
    refine_probability_random_walker,
    resize_binary,
    robust_center_scale,
    run_method as run_core_method,
    to_float_rgb,
    write_mask_tiff,
)

THUMB_SIZE = 420


def _require_optional(module: object | None, package_name: str) -> None:
    if module is None:
        raise RuntimeError(f"Optional dependency '{package_name}' is not installed in the current runtime")


@dataclass(frozen=True)
class SampleCase:
    sample: str
    root: str
    crop_roi: str
    seed_mask: str
    reference_mask: str | None = None
    historical_mask: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    selectable: bool
    production: bool
    config_updates: dict[str, object]
    runner: Callable[[np.ndarray, np.ndarray, np.ndarray | None, GrowConfig], np.ndarray]
    notes: str = ""


def _path_rank(case: SampleCase) -> tuple[int, int, int]:
    root = Path(case.root)
    score = 0
    if "realfiles" in root.as_posix():
        score += 8
    if case.reference_mask:
        score += 2
    if case.historical_mask:
        score += 1
    if not root.is_symlink():
        score += 1
    return (score, -len(root.as_posix()), -len(case.sample))


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_image_shape(path: Path) -> str:
    try:
        with tifffile.TiffFile(str(path)) as tif:
            arr = tif.series[0].levels[0].shape if getattr(tif.series[0], "levels", None) else tif.series[0].shape
        return "x".join(str(int(x)) for x in arr)
    except Exception:
        return ""


def discover_input_manifest_rows(repo_root: str | Path) -> list[dict[str, str]]:
    repo_root = Path(repo_root).resolve()
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for image_path in sorted((repo_root / 'Data').glob('*.ome.tif')):
        sample_id = image_path.name.replace('.ome.tif', '')
        geojson_path = image_path.with_suffix('').with_suffix('.geojson')
        key = (sample_id, str(image_path))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'sample_id': sample_id,
            'image_path': str(image_path),
            'roi_geojson_path': str(geojson_path) if geojson_path.exists() else '',
            'image_shape': _read_image_shape(image_path),
            'notes': 'repo_example_raw_input; benchmark requires upstream cluster mask generation',
        })

    for case in discover_example_samples(repo_root):
        key = (case.sample, case.crop_roi)
        if key in seen:
            continue
        seen.add(key)
        notes = [f'benchmarkable_from:{case.root}']
        if case.reference_mask:
            notes.append('has_reference_mask')
        if case.historical_mask:
            notes.append('has_historical_grown_mask')
        rows.append({
            'sample_id': case.sample,
            'image_path': case.crop_roi,
            'roi_geojson_path': '',
            'image_shape': _read_image_shape(Path(case.crop_roi)),
            'notes': ';'.join(notes),
        })

    return rows


def discover_example_samples(repo_root: str | Path) -> list[SampleCase]:
    repo_root = Path(repo_root).resolve()
    candidates: dict[str, SampleCase] = {}

    sample_dirs: list[Path] = []
    sample_dirs.extend(sorted(repo_root.glob('results_*/02_stardist/*')))
    sample_dirs.extend(sorted(repo_root.glob('*/02_stardist/*')))

    for crop_roi in repo_root.rglob('crop_roi.tif'):
        if crop_roi.parent.name == 'stardist_out' and crop_roi.parent.parent.parent.name == '02_stardist':
            sample_dirs.append(crop_roi.parent.parent)

    seen_dirs: set[str] = set()
    for sample_dir in sample_dirs:
        if not sample_dir.is_dir():
            continue
        sample_dir_key = str(sample_dir.resolve()) if sample_dir.exists() else str(sample_dir)
        if sample_dir_key in seen_dirs:
            continue
        seen_dirs.add(sample_dir_key)
        crop_roi = sample_dir / 'stardist_out' / 'crop_roi.tif'
        if not crop_roi.exists():
            continue
        if sample_dir.parent.name != '02_stardist':
            continue
        results_root = sample_dir.parent.parent
        sample = sample_dir.name
        seed_mask = _first_existing([
            results_root / '10_cluster_mask' / sample / f'{sample}_cluster_mask.tif',
            results_root / '15_cluster_mask' / sample / f'{sample}_cluster_mask.tif',
        ])
        if seed_mask is None:
            continue
        reference_mask = _first_existing([
            results_root / '04_roi_mask' / sample / f'{sample}_input_roi_mask.tif',
            results_root / '06_roi_mask' / sample / f'{sample}_input_roi_mask.tif',
        ])
        historical_mask = _first_existing([
            results_root / '11_grown_tissue' / sample / f'{sample}_grown_mask.ome.tif',
            results_root / '16_grown_tissue' / sample / f'{sample}_grown_mask.ome.tif',
        ])
        case = SampleCase(
            sample=sample,
            root=str(results_root),
            crop_roi=str(crop_roi),
            seed_mask=str(seed_mask),
            reference_mask=str(reference_mask) if reference_mask is not None else None,
            historical_mask=str(historical_mask) if historical_mask is not None else None,
        )
        prev = candidates.get(sample)
        if prev is None or _path_rank(case) > _path_rank(prev):
            candidates[sample] = case
    return sorted(candidates.values(), key=lambda c: c.sample)


def fit_panel(rgb: np.ndarray, max_side: int = THUMB_SIZE) -> Image.Image:
    img = Image.fromarray(rgb)
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if new_size != img.size:
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (max_side, max_side), (255, 255, 255))
    x = (max_side - img.size[0]) // 2
    y = (max_side - img.size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.integer):
        maxv = np.iinfo(arr.dtype).max or 255
        rgb = (arr.astype(np.float32) / float(maxv) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        rgbf = arr.astype(np.float32)
        if rgbf.max(initial=0.0) <= 1.5:
            rgbf *= 255.0
        rgb = rgbf.clip(0, 255).astype(np.uint8)
    return rgb


def colorize_labels(lbl: np.ndarray) -> np.ndarray:
    lbl = np.asarray(lbl)
    h, w = lbl.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    ids = [int(x) for x in np.unique(lbl) if int(x) != 0]
    for i, lab in enumerate(ids):
        hue = ((i * 137) % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
        out[lbl == lab] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def binary_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask).astype(np.uint8) * 255)


def save_binary_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), binary_to_uint8(mask), compression='deflate', photometric='minisblack')


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color_rgb=(255, 80, 0), alpha: float = 0.45) -> np.ndarray:
    rgb = to_uint8_rgb(image).astype(np.float32)
    mask = np.asarray(mask).astype(bool)
    overlay = np.zeros_like(rgb)
    overlay[mask] = np.array(color_rgb, dtype=np.float32)
    out = rgb.copy()
    out[mask] = (1.0 - alpha) * out[mask] + alpha * overlay[mask]
    return out.clip(0, 255).astype(np.uint8)


def overlay_seed_and_tissue(image: np.ndarray, seed_labels: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    rgb = overlay_mask_on_image(image, tissue_mask, color_rgb=(255, 140, 0), alpha=0.35)
    seed_boundary = compute_boundary(seed_labels > 0)
    tissue_boundary = compute_boundary(tissue_mask)
    rgb[tissue_boundary] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[seed_boundary] = np.array([0, 255, 255], dtype=np.uint8)
    return rgb


def overlay_boundaries(image: np.ndarray, tissue_mask: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).copy()
    rgb[compute_boundary(tissue_mask)] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[compute_boundary(seed_labels > 0)] = np.array([0, 255, 255], dtype=np.uint8)
    return rgb


def overlay_vs_reference(image: np.ndarray, tissue_mask: np.ndarray, reference_mask: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    rgb = to_uint8_rgb(image).copy()
    pred_b = compute_boundary(tissue_mask)
    ref_b = compute_boundary(reference_mask)
    rgb[pred_b] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[ref_b] = np.array([0, 255, 0], dtype=np.uint8)
    rgb[pred_b & ref_b] = np.array([255, 255, 0], dtype=np.uint8)
    rgb[compute_boundary(seed_labels > 0)] = np.array([0, 255, 255], dtype=np.uint8)
    return rgb


def mask_difference_panel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    panel = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    panel[tp] = np.array([220, 220, 220], dtype=np.uint8)
    panel[fp] = np.array([255, 0, 0], dtype=np.uint8)
    panel[fn] = np.array([0, 80, 255], dtype=np.uint8)
    return panel


def save_method_artifacts(
    outdir: str | Path,
    image: np.ndarray,
    seed_labels: np.ndarray,
    tissue_mask: np.ndarray,
    tissue_labels: np.ndarray,
    reference_labels: np.ndarray | None,
    method_name: str,
    metrics: dict[str, object],
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_binary_mask(outdir / 'tissue_mask_binary.tif', tissue_mask)
    write_mask_tiff(outdir / 'tissue_mask_labels.tif', tissue_labels.astype(np.uint16))
    write_mask_tiff(outdir / 'cell_mask_labels.tif', seed_labels.astype(np.uint16))
    fit_panel(overlay_mask_on_image(image, tissue_mask), THUMB_SIZE).save(outdir / 'overlay_tissue_on_image.png')
    fit_panel(overlay_seed_and_tissue(image, seed_labels, tissue_mask), THUMB_SIZE).save(outdir / 'overlay_cell_and_tissue.png')
    fit_panel(overlay_boundaries(image, tissue_mask, seed_labels), THUMB_SIZE).save(outdir / 'overlay_boundaries.png')
    fit_panel(colorize_labels(tissue_labels), THUMB_SIZE).save(outdir / 'labels_color.png')
    if reference_labels is not None:
        reference_mask = reference_labels > 0
        fit_panel(overlay_vs_reference(image, tissue_mask, reference_mask, seed_labels), THUMB_SIZE).save(outdir / 'overlay_reference.png')
        fit_panel(mask_difference_panel(tissue_mask, reference_mask), THUMB_SIZE).save(outdir / 'reference_diff.png')
    (outdir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (outdir / 'method.txt').write_text(f'{method_name}\n')


def grayscale(image: np.ndarray) -> np.ndarray:
    return color.rgb2gray(to_float_rgb(image)).astype(np.float32)


def _quantize_gray(image: np.ndarray, levels: int = 16) -> np.ndarray:
    gray = grayscale(image)
    return np.clip(np.floor(gray * levels), 0, levels - 1).astype(np.uint8)


def _normalized_texture(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(arr)] = 0.0
    return out.astype(np.float32)


def lbp_texture_map(image: np.ndarray) -> np.ndarray:
    gray = _quantize_gray(image, levels=32)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
    return _normalized_texture(lbp)


def gabor_texture_map(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    accum = np.zeros_like(gray, dtype=np.float32)
    for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
        real, imag = filters.gabor(gray, frequency=0.15, theta=theta)
        accum += np.sqrt(real.astype(np.float32) ** 2 + imag.astype(np.float32) ** 2)
    return _normalized_texture(accum)


def laws_texture_map(image: np.ndarray) -> np.ndarray:
    gray = grayscale(image)
    kernels = [
        np.array([1, 2, 1], dtype=np.float32),
        np.array([-1, 0, 1], dtype=np.float32),
        np.array([-1, 2, -1], dtype=np.float32),
    ]
    energies = []
    for a in kernels:
        for b in kernels:
            kernel = np.outer(a, b)
            resp = ndi.convolve(gray, kernel, mode='reflect')
            energies.append(np.abs(resp))
    return _normalized_texture(np.mean(np.stack(energies, axis=0), axis=0))


def wavelet_texture_map(image: np.ndarray) -> np.ndarray:
    _require_optional(pywt, 'PyWavelets')
    gray = grayscale(image)
    cA, (cH, cV, cD) = pywt.dwt2(gray, 'db2')
    energy = np.sqrt(cH ** 2 + cV ** 2 + cD ** 2)
    energy_up = transform.resize(energy, gray.shape, preserve_range=True, anti_aliasing=True)
    return _normalized_texture(energy_up)


def glcm_texture_map(image: np.ndarray) -> np.ndarray:
    gray_q = _quantize_gray(image, levels=16)
    h, w = gray_q.shape
    block = 16
    contrast = np.zeros_like(gray_q, dtype=np.float32)
    homogeneity = np.zeros_like(gray_q, dtype=np.float32)
    for y in range(0, h, block):
        for x in range(0, w, block):
            patch = gray_q[y:min(h, y + block), x:min(w, x + block)]
            if patch.size < 4:
                continue
            glcm = feature.graycomatrix(patch, distances=[1], angles=[0, np.pi / 4], levels=16, symmetric=True, normed=True)
            c = feature.graycoprops(glcm, 'contrast').mean()
            hg = feature.graycoprops(glcm, 'homogeneity').mean()
            contrast[y:min(h, y + block), x:min(w, x + block)] = c
            homogeneity[y:min(h, y + block), x:min(w, x + block)] = hg
    mix = _normalized_texture(contrast) * 0.65 + (1.0 - _normalized_texture(homogeneity)) * 0.35
    return _normalized_texture(mix)


def build_probability_from_feature_stack(feature_stack: np.ndarray, base: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    inside_mask, outside_mask = build_stats_masks(base, seed_binary, cfg)
    inside_vals = feature_stack[inside_mask]
    outside_vals = feature_stack[outside_mask]
    if inside_vals.size == 0 or outside_vals.size == 0:
        return build_tissue_probability(np.dstack([feature_stack[..., 0]] * 3), base, seed_binary, cfg)
    in_center, in_scale = robust_center_scale(inside_vals, cfg.stats_min_scale)
    out_center, out_scale = robust_center_scale(outside_vals, cfg.stats_min_scale)
    dist_in = model_distance(feature_stack, in_center, in_scale)
    dist_out = model_distance(feature_stack, out_center, out_scale)
    score = dist_out - dist_in
    in_score = float(np.median(score[inside_mask]))
    out_score = float(np.median(score[outside_mask]))
    mid = 0.5 * (in_score + out_score)
    scale = max(abs(in_score - out_score) * 0.5, 0.05)
    prob = 1.0 / (1.0 + np.exp(-(score - mid) / scale))
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def similarity_region_expand(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
    seed_labels = np.asarray(seed_labels)
    seed_binary = seed_labels > 0
    base_mask = np.asarray(base_mask).astype(bool) if base_mask is not None and np.size(base_mask) else seed_binary.copy()
    feats = build_feature_stack(image)
    bg_like = build_background_likelihood(image)
    _, indices = ndi.distance_transform_edt(~seed_binary, return_indices=True)
    nearest_label = seed_labels[tuple(indices)]
    search_global = morphology.binary_dilation(base_mask, morphology.disk(max(16, cfg.boundary_band // 2)))
    out = np.zeros_like(seed_binary, dtype=bool)
    label_ids = [int(x) for x in np.unique(seed_labels) if int(x) != 0]
    for lab in label_ids:
        core = seed_labels == lab
        territory = nearest_label == lab
        local_search = morphology.binary_dilation(core | (base_mask & territory), morphology.disk(max(20, cfg.boundary_band // 2)))
        local_search &= search_global & territory
        if not local_search.any():
            out |= core
            continue
        center, scale = robust_center_scale(feats[core], cfg.stats_min_scale)
        dist = model_distance(feats, center, scale)
        core_vals = dist[core]
        thr = float(np.quantile(core_vals, 0.995) + 0.80)
        allow = local_search & (dist <= thr) & (bg_like < 0.82)
        region = core.copy()
        for _ in range(18):
            frontier = morphology.binary_dilation(region, morphology.disk(1)) & ~region
            frontier &= allow
            if not frontier.any():
                break
            region |= frontier
        region = morphology.remove_small_holes(region, area_threshold=max(256, cfg.final_hole_area // 8))
        out |= region
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, cfg, keep_largest=True)


def refine_graph_cut(base: np.ndarray, image: np.ndarray, prob: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    _require_optional(maxflow, 'PyMaxflow')
    base = np.asarray(base).astype(bool)
    prob = np.asarray(prob, dtype=np.float32)
    seed_binary = np.asarray(seed_binary).astype(bool)
    if prob.shape != base.shape:
        prob = resize_binary(prob, base.shape).astype(np.float32)
    rgb = to_float_rgb(image)
    work_prob = prob
    work_base = base
    work_seed = seed_binary
    factor = 1
    if max(base.shape) > cfg.no_downsample_max_side:
        factor = max(1, int(cfg.image_downsample))
        work_prob = prob[::factor, ::factor]
        work_base = base[::factor, ::factor]
        work_seed = seed_binary[::factor, ::factor]
        rgb = rgb[::factor, ::factor]
    band = morphology.binary_dilation(work_base, morphology.disk(cfg.boundary_band)) | work_base
    inside = morphology.binary_erosion(work_base, morphology.disk(max(1, cfg.boundary_band // 3))) | work_seed
    outside = ~morphology.binary_dilation(work_base, morphology.disk(max(2, cfg.boundary_band // 2)))
    mask = band | outside
    coords = np.argwhere(mask)
    if coords.size == 0:
        out = base.copy()
        out[seed_binary] = True
        return out
    index = -np.ones(mask.shape, dtype=np.int32)
    index[mask] = np.arange(coords.shape[0], dtype=np.int32)
    graph = maxflow.Graph[float](coords.shape[0], coords.shape[0] * 4)
    graph.add_nodes(coords.shape[0])
    sigma = 0.10
    pairwise_w = 1.6
    for y, x in coords:
        node = int(index[y, x])
        if inside[y, x]:
            graph.add_tedge(node, 1e6, 0.0)
        elif outside[y, x]:
            graph.add_tedge(node, 0.0, 1e6)
        else:
            p = float(np.clip(work_prob[y, x], 1e-4, 1 - 1e-4))
            graph.add_tedge(node, -math.log(p), -math.log(1.0 - p))
        for ny, nx in ((y + 1, x), (y, x + 1)):
            if ny >= mask.shape[0] or nx >= mask.shape[1] or not mask[ny, nx]:
                continue
            node2 = int(index[ny, nx])
            diff = float(np.linalg.norm(rgb[y, x] - rgb[ny, nx]))
            weight = pairwise_w * math.exp(-(diff * diff) / max(2.0 * sigma * sigma, 1e-6)) + 0.02
            graph.add_edge(node, node2, weight, weight)
    graph.maxflow()
    out_small = np.zeros(mask.shape, dtype=bool)
    for y, x in coords:
        node = int(index[y, x])
        out_small[y, x] = graph.get_segment(node) == 0
    if factor > 1:
        out = resize_binary(out_small, base.shape)
    else:
        out = out_small
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_baseline_morphology(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    bg_like = build_background_likelihood(image)
    allow = bg_like < float(np.quantile(bg_like, 0.88))
    radius = max(10, min(42, int(round(math.sqrt(float(seed_binary.sum()) / math.pi) * 0.18)) + 8))
    out = morphology.binary_dilation(seed_binary, morphology.disk(radius)) & allow
    out = morphology.binary_closing(out, morphology.disk(max(2, radius // 5)))
    out = morphology.remove_small_holes(out, area_threshold=max(1024, cfg.final_hole_area // 4))
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_seed_region_grow(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
    return similarity_region_expand(image, seed_labels, base_mask, cfg)


def method_graphcut_stats(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    if base_mask is None:
        base_mask = CORE_METHODS['union_watershed'](image, seed_labels, cfg)
    prob = build_tissue_probability(image, base_mask, seed_binary, cfg)
    return refine_graph_cut(base_mask, image, prob, seed_binary, cfg)


def _texture_probability(image: np.ndarray, base_mask: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig, mode: str) -> np.ndarray:
    gray = grayscale(image)
    od = make_signal_bundle(image)['od']
    if mode == 'lbp':
        extra = lbp_texture_map(image)
    elif mode == 'gabor':
        extra = gabor_texture_map(image)
    elif mode == 'glcm':
        extra = glcm_texture_map(image)
    elif mode == 'laws':
        extra = laws_texture_map(image)
    elif mode == 'wavelet':
        extra = wavelet_texture_map(image)
    else:
        raise KeyError(mode)
    stack = np.stack([
        _normalized_texture(gray),
        _normalized_texture(od),
        extra,
        _normalized_texture(filters.sobel(gray)),
    ], axis=-1)
    return build_probability_from_feature_stack(stack, base_mask, seed_binary, cfg)


def _texture_rw_method(mode: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None, GrowConfig], np.ndarray]:
    def runner(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
        seed_binary = seed_labels > 0
        if base_mask is None:
            base_mask = CORE_METHODS['union_watershed'](image, seed_labels, cfg)
        prob = _texture_probability(image, np.asarray(base_mask).astype(bool), seed_binary, cfg, mode)
        out = refine_probability_random_walker(np.asarray(base_mask).astype(bool), prob, seed_binary, cfg)
        out[seed_binary] = True
        return cleanup_mask(out, cfg, keep_largest=True)
    return runner


def method_mrf_smooth_stats(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    if base_mask is None:
        base_mask = CORE_METHODS['union_stats_hybrid'](image, seed_labels, cfg)
    prob = build_tissue_probability(image, base_mask, seed_binary, cfg)
    graphcut = refine_graph_cut(np.asarray(base_mask).astype(bool), image, prob, seed_binary, cfg)
    smooth = morphology.binary_closing(graphcut, morphology.disk(3))
    smooth = morphology.binary_opening(smooth, morphology.disk(1))
    smooth[seed_binary] = True
    return cleanup_mask(smooth, cfg, keep_largest=True)


def _unavailable_method(reason: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None, GrowConfig], np.ndarray]:
    def runner(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
        raise RuntimeError(reason)
    return runner


def build_method_specs() -> list[MethodSpec]:
    specs: list[MethodSpec] = [
        MethodSpec('baseline_morphology', 'baseline_morphology', True, False, {}, method_baseline_morphology, 'Dilate/close/fill baseline'),
        MethodSpec('seed_region_grow', 'seed_region_grow', True, False, {}, method_seed_region_grow, 'Texture/color region growing from seeds'),
        MethodSpec('graphcut_stats', 'graphcut_stats', True, False, {'boundary_band': 40}, method_graphcut_stats, 'Graph cut from tissue probability'),
        MethodSpec('lbp_texture_rw', 'lbp_texture_rw', True, False, {'boundary_band': 40}, _texture_rw_method('lbp'), 'LBP-guided random walker'),
        MethodSpec('gabor_texture_rw', 'gabor_texture_rw', True, False, {'boundary_band': 40}, _texture_rw_method('gabor'), 'Gabor-guided random walker'),
        MethodSpec('glcm_texture_rw', 'glcm_texture_rw', True, False, {'boundary_band': 40}, _texture_rw_method('glcm'), 'GLCM-guided random walker'),
        MethodSpec('laws_texture_rw', 'laws_texture_rw', True, False, {'boundary_band': 40}, _texture_rw_method('laws'), 'Laws texture random walker'),
        MethodSpec('wavelet_texture_rw', 'wavelet_texture_rw', True, False, {'boundary_band': 40}, _texture_rw_method('wavelet'), 'Wavelet-guided random walker'),
        MethodSpec('mrf_smooth_stats', 'mrf_smooth_stats', True, False, {'boundary_band': 40}, method_mrf_smooth_stats, 'Graph-cut/Potts-like smoothing'),
    ]
    production_methods = [
        ('seed_only', {}, 'Seed-only lower bound'),
        ('union_seeded', {}, 'Image-seeded union mask'),
        ('union_watershed', {'boundary_band': 40}, 'Recommended stable production candidate'),
        ('gcdt_coarse_to_fine', {'gcdt_rough_radius': 18, 'gcdt_cycles': 3}, 'Nature 2022-inspired coarse rough label + channel-difference threshold refinement'),
        ('dual_transition_refine', {'dual_cycles': 6, 'dual_band_radius': 10}, 'Paper-inspired dual positive/negative transition refinement'),
        ('cell_texture_expand', {'texture_init_expand_radius': 10, 'texture_step_radius': 4, 'texture_max_iters': 18, 'texture_cycles': 10, 'texture_cycle_radius': 12}, 'Ten-cycle segment-preserving texture expansion from cell mask with stronger contact-border relabeling'),
        ('union_random_walker', {'boundary_band': 40}, 'Random walker from image-guided base'),
        ('union_mgac', {'mgac_iters': 24, 'boundary_band': 40}, 'Morphological geodesic active contour'),
        ('union_chan_vese', {'chan_vese_iters': 50, 'boundary_band': 40}, 'Chan-Vese active contour'),
        ('union_superpixel_proto', {'boundary_band': 40}, 'Superpixel prototype refinement'),
        ('union_stats_hybrid', {'boundary_band': 40}, 'Probability gate + random walker hybrid'),
        ('od_stats_rw', {'boundary_band': 40}, 'Optical-density random walker'),
        ('union_bg_trim', {'boundary_band': 40}, 'Background-aware conservative trim'),
        ('lab_ab_watershed', {'boundary_band': 40}, 'Lab gradient watershed'),
        ('hed_watershed', {'boundary_band': 40}, 'HED gradient watershed'),
    ]
    for name, cfg_updates, notes in production_methods:
        def make_runner(method_name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None, GrowConfig], np.ndarray]:
            def runner(image: np.ndarray, seed_labels: np.ndarray, base_mask: np.ndarray | None, cfg: GrowConfig) -> np.ndarray:
                _, labels, _ = run_core_method(method_name, image, seed_labels, cfg)
                return labels
            return runner
        specs.append(MethodSpec(name, name, True, True, cfg_updates, make_runner(name), notes))
    pathsam_reason = (
        'Real Path-SAM2 is not installed in the current runtime. '
        'The external SAM2PATH repository is available for inspection, but it is training-oriented and '
        'does not provide a benchmark-ready inference path with suitable pathology weights for these examples.'
    )
    specs.extend([
        MethodSpec('path_sam2_semantic', 'path_sam2_semantic', False, False, {}, _unavailable_method(pathsam_reason), 'Path-SAM2 status probe only'),
        MethodSpec('path_sam2_semantic_plus_cleanup', 'path_sam2_semantic_plus_cleanup', False, False, {}, _unavailable_method(pathsam_reason), 'Path-SAM2 status probe only'),
    ])
    return specs


BENCHMARK_METHODS = build_method_specs()
BENCHMARK_METHOD_MAP = {m.name: m for m in BENCHMARK_METHODS}


def seed_component_count(seed_labels: np.ndarray) -> int:
    return int(measure.label(seed_labels > 0, connectivity=1).max())


def estimate_background_mask(image: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    bg_like = build_background_likelihood(image)
    border = np.zeros(bg_like.shape, dtype=bool)
    border[:8, :] = True
    border[-8:, :] = True
    border[:, :8] = True
    border[:, -8:] = True
    border_vals = bg_like[border]
    threshold = float(np.quantile(border_vals, 0.78)) if border_vals.size else 0.78
    obvious = bg_like >= max(0.72, threshold)
    obvious &= ~morphology.binary_dilation(seed_labels > 0, morphology.disk(4))
    return obvious


def segmentwise_reference_scores(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict[str, float]:
    pred_ids = [int(x) for x in np.unique(pred_labels) if int(x) != 0]
    gt_ids = [int(x) for x in np.unique(gt_labels) if int(x) != 0]
    if not gt_ids:
        return {
            'ref_segment_count_gt': 0.0,
            'ref_segment_count_pred': float(len(pred_ids)),
            'ref_segment_count_error': float(len(pred_ids)),
            'ref_segment_min_dice': 1.0,
            'ref_segment_mean_dice': 1.0,
            'ref_segment_mean_iou': 1.0,
            'ref_segment_mean_boundary_f1': 1.0,
        }
    dice_mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    scores_cache: dict[tuple[int, int], dict[str, float]] = {}
    for i, pid in enumerate(pred_ids):
        pm = pred_labels == pid
        p_sum = float(pm.sum())
        for j, gid in enumerate(gt_ids):
            gm = gt_labels == gid
            g_sum = float(gm.sum())
            inter = float(np.logical_and(pm, gm).sum())
            dice = 0.0 if (p_sum + g_sum) == 0 else (2.0 * inter) / (p_sum + g_sum)
            dice_mat[i, j] = dice
            scores_cache[(pid, gid)] = score_binary_masks(pm, gm)
    if pred_ids and gt_ids:
        row_ind, col_ind = linear_sum_assignment(1.0 - dice_mat)
    else:
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)
    matched_gt = {}
    for ri, ci in zip(row_ind, col_ind):
        pid = pred_ids[int(ri)]
        gid = gt_ids[int(ci)]
        matched_gt[gid] = scores_cache[(pid, gid)]
    per_gt = [matched_gt.get(gid, {'dice': 0.0, 'iou': 0.0, 'boundary_f1': 0.0}) for gid in gt_ids]
    return {
        'ref_segment_count_gt': float(len(gt_ids)),
        'ref_segment_count_pred': float(len(pred_ids)),
        'ref_segment_count_error': float(abs(len(pred_ids) - len(gt_ids))),
        'ref_segment_min_dice': float(min(s['dice'] for s in per_gt)),
        'ref_segment_mean_dice': float(np.mean([s['dice'] for s in per_gt])),
        'ref_segment_mean_iou': float(np.mean([s['iou'] for s in per_gt])),
        'ref_segment_mean_boundary_f1': float(np.mean([s['boundary_f1'] for s in per_gt])),
    }


def score_binary_masks(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    union = float(np.logical_or(pred, gt).sum())
    dice = 1.0 if pred_sum + gt_sum == 0 else (2.0 * inter) / (pred_sum + gt_sum)
    iou = 1.0 if union == 0 else inter / union
    pred_b = compute_boundary(pred)
    gt_b = compute_boundary(gt)
    if not pred_b.any() and not gt_b.any():
        boundary_f1 = 1.0
    else:
        pred_hit = morphology.binary_dilation(gt_b, morphology.disk(tolerance)) & pred_b
        gt_hit = morphology.binary_dilation(pred_b, morphology.disk(tolerance)) & gt_b
        precision = float(pred_hit.sum()) / float(pred_b.sum()) if pred_b.any() else 0.0
        recall = float(gt_hit.sum()) / float(gt_b.sum()) if gt_b.any() else 0.0
        boundary_f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {'dice': dice, 'iou': iou, 'boundary_f1': boundary_f1}


def compute_unsupervised_metrics(image: np.ndarray, seed_labels: np.ndarray, tissue_mask: np.ndarray, runtime_sec: float) -> dict[str, float]:
    seed_binary = seed_labels > 0
    tissue_mask = np.asarray(tissue_mask).astype(bool)
    obvious_bg = estimate_background_mask(image, seed_labels)
    area = float(tissue_mask.sum())
    perimeter = float(compute_boundary(tissue_mask).sum())
    seed_area = float(seed_binary.sum())
    cell_retention_ratio = 1.0 if seed_area == 0 else float(np.logical_and(tissue_mask, seed_binary).sum()) / seed_area
    background_leakage_est = 0.0 if area == 0 else float(np.logical_and(tissue_mask, obvious_bg).sum()) / area
    num_components = int(measure.label(tissue_mask, connectivity=1).max())
    seed_components = max(1, seed_component_count(seed_labels))
    fragmentation_proxy = max(num_components - 1, 0) / seed_components
    compactness = 0.0 if perimeter == 0 or area == 0 else float(4.0 * math.pi * area / (perimeter * perimeter))
    roughness_proxy = 0.0 if perimeter == 0 or area == 0 else float(perimeter / max(2.0 * math.sqrt(math.pi * area), 1e-6) - 1.0)
    halo = morphology.binary_dilation(seed_binary, morphology.disk(20))
    continuity = float(tissue_mask[halo].mean()) if halo.any() else 0.0
    filled = morphology.remove_small_holes(tissue_mask, area_threshold=5000)
    hole_ratio = 0.0 if area == 0 else float(np.logical_and(filled, ~tissue_mask).sum()) / area
    total_pixels = float(tissue_mask.size)
    tissue_area_ratio = 0.0 if total_pixels == 0 else area / total_pixels
    composite_score = (
        8.0 * cell_retention_ratio
        + 1.8 * continuity
        + 1.4 * compactness
        - 2.8 * background_leakage_est
        - 0.8 * fragmentation_proxy
        - 0.4 * hole_ratio
        - 0.2 * roughness_proxy
        - 0.05 * min(runtime_sec / 10.0, 10.0)
    )
    if cell_retention_ratio < 0.999999:
        composite_score -= 25.0 * (1.0 - cell_retention_ratio)
    return {
        'runtime_sec': float(runtime_sec),
        'cell_retention_ratio': float(cell_retention_ratio),
        'background_leakage_est': float(background_leakage_est),
        'num_components': float(num_components),
        'seed_components': float(seed_components),
        'fragmentation_proxy': float(fragmentation_proxy),
        'compactness': float(compactness),
        'roughness_proxy': float(roughness_proxy),
        'continuity_around_cells': float(continuity),
        'hole_ratio': float(hole_ratio),
        'tissue_area_pixels': float(area),
        'tissue_area_ratio': float(tissue_area_ratio),
        'composite_score': float(composite_score),
    }


def run_spec_on_sample(spec: MethodSpec, image: np.ndarray, seed_labels: np.ndarray, base_labels: np.ndarray | None, cfg: GrowConfig) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()
    output = spec.runner(image, seed_labels, base_labels, cfg)
    runtime_sec = float(time.perf_counter() - t0)
    tissue_mask, labels = normalize_method_output(output, seed_labels)
    return tissue_mask, labels.astype(np.uint16), runtime_sec


def default_config_for_spec(spec: MethodSpec) -> GrowConfig:
    return GrowConfig(**{**config_dict(DEFAULT_CONFIG), **spec.config_updates})


def score_candidate_program(program_path: str | Path, sample: SampleCase) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    spec = importlib.util.spec_from_file_location('candidate_program', str(program_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    image = read_tiff_level0(sample.crop_roi)
    seed_labels = np.asarray(read_tiff_level0(sample.seed_mask))
    if seed_labels.ndim == 3:
        seed_labels = seed_labels[..., 0]
    base_labels = np.asarray(read_tiff_level0(sample.historical_mask)) if sample.historical_mask else seed_labels.copy()
    output = mod.run_refinement(image, seed_labels, base_labels)
    tissue_mask, labels = normalize_method_output(output, seed_labels)
    metrics = compute_unsupervised_metrics(image, seed_labels, tissue_mask, runtime_sec=0.0)
    if sample.reference_mask:
        ref_labels = np.asarray(read_tiff_level0(sample.reference_mask))
        if ref_labels.shape != seed_labels.shape:
            ref_labels = resize_binary(ref_labels > 0, seed_labels.shape)
            ref_labels = ref_labels.astype(np.uint8)
        metrics.update({f'reference_{k}': v for k, v in score_binary_masks(labels > 0, ref_labels > 0).items()})
        metrics.update(segmentwise_reference_scores(labels, ref_labels))
    return metrics, tissue_mask, labels


def summarize_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby('method', as_index=False).agg(
        samples=('sample', 'count'),
        min_cell_retention_ratio=('cell_retention_ratio', 'min'),
        mean_cell_retention_ratio=('cell_retention_ratio', 'mean'),
        mean_composite_score=('composite_score', 'mean'),
        min_composite_score=('composite_score', 'min'),
        mean_background_leakage_est=('background_leakage_est', 'mean'),
        mean_fragmentation_proxy=('fragmentation_proxy', 'mean'),
        mean_compactness=('compactness', 'mean'),
        mean_runtime_sec=('runtime_sec', 'mean'),
    )
    if 'reference_dice' in df.columns:
        ref = df.groupby('method', as_index=False).agg(
            mean_reference_dice=('reference_dice', 'mean'),
            min_reference_dice=('reference_dice', 'min'),
            mean_ref_segment_mean_dice=('ref_segment_mean_dice', 'mean'),
        )
        grouped = grouped.merge(ref, on='method', how='left')
    grouped = grouped.sort_values(
        ['min_cell_retention_ratio', 'mean_composite_score', 'min_composite_score', 'mean_background_leakage_est', 'mean_fragmentation_proxy'],
        ascending=[False, False, False, True, True],
        kind='mergesort',
    ).reset_index(drop=True)
    grouped.insert(0, 'rank', np.arange(1, len(grouped) + 1))
    return grouped


def save_comparison_panel(
    path: str | Path,
    image: np.ndarray,
    seed_labels: np.ndarray,
    method_to_labels: dict[str, np.ndarray],
    historical_labels: np.ndarray | None = None,
    evolved_labels: np.ndarray | None = None,
) -> None:
    panels: list[tuple[str, np.ndarray]] = [
        ('original image', to_uint8_rgb(image)),
        ('cell mask', colorize_labels(seed_labels)),
    ]
    if historical_labels is not None:
        panels.append(('historical 16_grown_tissue', colorize_labels(historical_labels)))
    for method_name, labels in method_to_labels.items():
        panels.append((method_name, colorize_labels(labels)))
    if evolved_labels is not None:
        panels.append(('best evolved candidate', colorize_labels(evolved_labels)))
    tiles = [fit_panel(rgb, THUMB_SIZE) for _, rgb in panels]
    cols = min(4, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    tile_w = THUMB_SIZE
    caption_h = 34
    canvas = Image.new('RGB', (cols * tile_w, rows * (THUMB_SIZE + caption_h)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, ((title, _), tile) in enumerate(zip(panels, tiles)):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * (THUMB_SIZE + caption_h)
        canvas.paste(tile, (x, y + caption_h))
        draw.text((x + 8, y + 8), title, fill=(0, 0, 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_global_report(path: str | Path, repo_root: Path, samples: Sequence[SampleCase], ranking: pd.DataFrame, best_method: str | None) -> None:
    lines = [
        '# Tissue Mask Benchmark',
        '',
        f'Repository: `{repo_root}`',
        f'Images discovered: **{len(samples)}**',
        '',
        '## Discovered samples',
    ]
    for case in samples:
        lines.append(f'- `{case.sample}` from `{case.root}`')
    lines.extend(['', '## Ranking'])
    if ranking.empty:
        lines.append('- No results yet.')
    else:
        for _, row in ranking.head(12).iterrows():
            lines.append(
                f"- `{row['method']}`: rank={int(row['rank'])}, mean composite={row['mean_composite_score']:.4f}, min retention={row['min_cell_retention_ratio']:.4f}, mean leakage={row['mean_background_leakage_est']:.4f}, mean fragmentation={row['mean_fragmentation_proxy']:.4f}"
            )
    if best_method:
        lines.extend(['', '## Recommendation', f'- Current unsupervised winner: `{best_method}`'])
    lines.extend([
        '',
        '## Notes',
        '- Ranking is based on unsupervised/task-driven metrics.',
        '- ROI overlap metrics are saved only as diagnostic reference where `06_roi_mask` exists; they are not the primary ranking objective.',
    ])
    Path(path).write_text('\n'.join(lines) + '\n')


def write_per_image_report(path: str | Path, sample: str, rows: pd.DataFrame) -> None:
    lines = [f'# {sample}', '', '## Methods']
    ordered = rows.sort_values(['composite_score', 'cell_retention_ratio'], ascending=[False, False])
    for _, row in ordered.iterrows():
        lines.append(
            f"- `{row['method']}`: composite={row['composite_score']:.4f}, retention={row['cell_retention_ratio']:.4f}, leakage={row['background_leakage_est']:.4f}, fragmentation={row['fragmentation_proxy']:.4f}, runtime={row['runtime_sec']:.2f}s"
        )
    Path(path).write_text('\n'.join(lines) + '\n')


def write_gallery_index(path: str | Path, ranking: pd.DataFrame, samples: Sequence[SampleCase], results_root: Path) -> None:
    rows = []
    for case in samples:
        rel = Path('comparison_panels') / f'{case.sample}.png'
        rows.append(f"<div class='card'><h3>{html.escape(case.sample)}</h3><a href='{rel.as_posix()}'><img src='{rel.as_posix()}' loading='lazy'></a></div>")
    best_table = ''
    if not ranking.empty:
        best_table = ranking.head(12).to_html(index=False, float_format=lambda x: f'{x:.4f}' if isinstance(x, float) else str(x))
    content = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Tissue Mask Benchmark</title>
<style>
body {{ font-family: Helvetica, Arial, sans-serif; margin: 20px; background: #faf7f2; color: #1f1a17; }}
img {{ width: 100%; border: 1px solid #ddd; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
.card {{ background: white; padding: 12px; border-radius: 10px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
th {{ background: #f0e8dd; }}
</style></head>
<body>
<h1>Tissue Mask Benchmark</h1>
<p>Results root: <code>{html.escape(str(results_root))}</code></p>
<h2>Method ranking</h2>
{best_table}
<h2>Comparison panels</h2>
<div class='grid'>
{''.join(rows)}
</div>
</body></html>
"""
    Path(path).write_text(content)
