from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation, transform


@dataclass(frozen=True)
class GrowConfig:
    image_downsample: int = 8
    no_downsample_max_side: int = 1024
    chroma_close_radius: int = 12
    chroma_min_obj_area: int = 30000
    chroma_hole_area: int = 30000
    image_keep_largest: bool = True
    bridge_close_radius: int = 6
    boundary_band: int = 48
    boundary_sigma: float = 2.0
    boundary_quantile: float = 0.65
    random_walker_beta: int = 100
    mgac_iters: int = 40
    mgac_balloon: float = 0.5
    mgac_threshold: float = 0.25
    chan_vese_iters: int = 60
    final_open_radius: int = 2
    final_close_radius: int = 6
    final_hole_area: int = 20000
    final_min_obj_area: int = 20000
    boundary_tolerance: int = 3
    stats_min_scale: float = 0.05
    stats_core_radius: int = 10
    stats_prune_prob: float = 0.30
    stats_grow_prob: float = 0.58
    stats_outer_margin: int = 24
    rescue_min_area: int = 5000
    rescue_max_gap: int = 96
    rescue_score_floor: float = 0.52
    rescue_score_quantile: float = 0.25
    adaptive_band_scale: float = 0.04
    adaptive_band_min: int = 20
    adaptive_band_max: int = 72
    superpixel_size: int = 72
    superpixel_compactness: float = 10.0
    superpixel_fg_ratio: float = 0.70
    superpixel_bg_ratio: float = 0.70
    superpixel_expand_margin: int = 8
    texture_init_expand_radius: int = 10
    texture_step_radius: int = 4
    texture_max_iters: int = 18
    texture_accept_quantile: float = 0.90
    texture_accept_scale: float = 1.25
    texture_accept_floor: float = 1.60
    texture_bg_ratio: float = 1.05
    texture_prob_floor: float = 0.42
    texture_prob_quantile: float = 0.12
    texture_bg_quantile: float = 0.78
    texture_hole_area: int = 12000
    texture_min_obj_area: int = 12000
    texture_cycles: int = 10
    texture_cycle_radius: int = 12
    texture_cycle_accept_step: float = 0.20
    texture_cycle_bg_ratio_step: float = 0.10
    texture_cycle_prob_decay: float = 0.05
    texture_newarea_min_pixels: int = 2048
    texture_contact_iters: int = 4
    texture_contact_improve_ratio: float = 0.995
    dual_cycles: int = 6
    dual_band_radius: int = 10
    dual_temperature: float = 0.75
    dual_identity_weight: float = 0.90
    dual_expand_prob: float = 0.58
    dual_bg_prob_max: float = 0.42
    dual_prob_weight: float = 0.60
    dual_bg_weight: float = 1.10
    dual_relabel_margin: float = 0.04
    gcdt_rough_radius: int = 18
    gcdt_cycles: int = 3
    gcdt_core_threshold: float = 0.70
    gcdt_close_radius: int = 3
    gcdt_hole_area: int = 8000
    gcdt_min_obj_area: int = 2000


DEFAULT_CONFIG = GrowConfig()


def with_cfg(cfg: GrowConfig, **updates) -> GrowConfig:
    return GrowConfig(**{**config_dict(cfg), **updates})


def read_tiff_level0(path: str | Path) -> np.ndarray:
    path = str(path)
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if getattr(series, "levels", None):
            arr = series.levels[0].asarray()
        else:
            arr = series.asarray()
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image


def to_float_rgb(image: np.ndarray) -> np.ndarray:
    image = ensure_rgb(image)
    if np.issubdtype(image.dtype, np.integer):
        maxv = np.iinfo(image.dtype).max
        out = image.astype(np.float32) / float(maxv if maxv > 0 else 255)
    else:
        out = image.astype(np.float32)
        if out.max(initial=0) > 1.5:
            out = out / 255.0
    return np.clip(out, 0.0, 1.0)


def resize_binary(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    out = transform.resize(
        mask.astype(np.uint8),
        shape,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return out >= 0.5


def resize_labels(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    out = transform.resize(
        mask.astype(np.int32),
        shape,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    )
    return np.rint(out).astype(mask.dtype)


def maybe_downsample(image: np.ndarray, cfg: GrowConfig) -> Tuple[np.ndarray, int]:
    h, w = image.shape[:2]
    if max(h, w) <= cfg.no_downsample_max_side:
        return image, 1
    factor = max(1, int(cfg.image_downsample))
    out = image[::factor, ::factor]
    return out, factor


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return mask.astype(bool)
    props = measure.regionprops(labels)
    largest = max(props, key=lambda p: p.area).label
    return labels == largest


def keep_seeded_components(mask: np.ndarray, seed_binary: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return seed_binary.copy()
    touched = np.unique(labels[seed_binary])
    touched = touched[touched > 0]
    if touched.size == 0:
        return seed_binary.copy()
    out = np.isin(labels, touched)
    out[seed_binary] = True
    return out


def cleanup_mask(mask: np.ndarray, cfg: GrowConfig, keep_largest: bool | None = None) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if cfg.final_open_radius > 0:
        mask = morphology.binary_opening(mask, morphology.disk(cfg.final_open_radius))
    if cfg.final_close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(cfg.final_close_radius))
    if cfg.final_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=cfg.final_hole_area)
    if cfg.final_min_obj_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=cfg.final_min_obj_area)
    use_keep_largest = cfg.image_keep_largest if keep_largest is None else keep_largest
    if use_keep_largest and mask.any():
        mask = keep_largest_component(mask)
    return mask.astype(bool)


def build_chroma_candidate(image: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    rgb = to_float_rgb(image)
    work, factor = maybe_downsample(rgb, cfg)
    lab = color.rgb2lab(work)
    chroma = np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2]))
    finite = chroma[np.isfinite(chroma)]
    threshold = filters.threshold_otsu(finite) if finite.size else 0.0
    mask = chroma > threshold
    if cfg.chroma_close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(cfg.chroma_close_radius))
    if cfg.chroma_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=cfg.chroma_hole_area)
    if cfg.chroma_min_obj_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=cfg.chroma_min_obj_area)
    if cfg.image_keep_largest and mask.any():
        mask = keep_largest_component(mask)
    if factor > 1:
        mask = resize_binary(mask, rgb.shape[:2])
    return mask.astype(bool)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    finite = np.isfinite(signal)
    if not finite.any():
        return np.zeros_like(signal, dtype=np.float32)
    vals = signal[finite]
    lo = np.percentile(vals, 1)
    hi = np.percentile(vals, 99)
    if hi <= lo:
        hi = vals.max(initial=lo + 1e-6)
    out = np.clip((signal - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    out[~finite] = 0.0
    return out.astype(np.float32)


def optical_density_strength(image: np.ndarray) -> np.ndarray:
    rgb = to_float_rgb(image)
    od = -np.log(np.clip(rgb, 1.0 / 255.0, 1.0))
    return od.mean(axis=-1)


def hed_strength(image: np.ndarray) -> np.ndarray:
    rgb = to_float_rgb(image)
    hed = color.rgb2hed(rgb)
    hematoxylin = np.maximum(hed[..., 0], 0.0)
    eosin = np.maximum(hed[..., 1], 0.0)
    return hematoxylin + 0.35 * eosin


def chroma_strength(image: np.ndarray) -> np.ndarray:
    rgb = to_float_rgb(image)
    lab = color.rgb2lab(rgb)
    return np.sqrt(np.square(lab[..., 1]) + np.square(lab[..., 2]))


def make_signal_bundle(image: np.ndarray) -> Dict[str, np.ndarray]:
    bundle = {
        "chroma": normalize_signal(chroma_strength(image)),
        "od": normalize_signal(optical_density_strength(image)),
        "hed": normalize_signal(hed_strength(image)),
    }
    bundle["mix"] = normalize_signal(0.45 * bundle["chroma"] + 0.35 * bundle["od"] + 0.20 * bundle["hed"])
    return bundle


def _channel_gradient(channel: np.ndarray, op: str = "sobel", sigma: float = 1.0) -> np.ndarray:
    sm = filters.gaussian(channel.astype(np.float32), sigma=sigma)
    if op == "scharr":
        grad = filters.scharr(sm)
    else:
        grad = filters.sobel(sm)
    return normalize_signal(grad)


def rgb_gradient_fusion(image: np.ndarray, op: str = "sobel", rule: str = "max", sigma: float = 1.0) -> np.ndarray:
    rgb = to_float_rgb(image)
    grads = np.stack([_channel_gradient(rgb[..., c], op=op, sigma=sigma) for c in range(3)], axis=-1)
    if rule == "mean":
        fused = grads.mean(axis=-1)
    elif rule == "l2":
        fused = np.sqrt(np.mean(np.square(grads), axis=-1))
    else:
        fused = grads.max(axis=-1)
    return normalize_signal(fused)


def lab_ab_gradient_fusion(image: np.ndarray, op: str = "sobel", sigma: float = 1.0) -> np.ndarray:
    rgb = to_float_rgb(image)
    lab = color.rgb2lab(rgb)
    ga = _channel_gradient(lab[..., 1], op=op, sigma=sigma)
    gb = _channel_gradient(lab[..., 2], op=op, sigma=sigma)
    return normalize_signal(np.maximum(ga, gb))


def hed_gradient_fusion(image: np.ndarray, op: str = "sobel", sigma: float = 1.0) -> np.ndarray:
    rgb = to_float_rgb(image)
    hed = color.rgb2hed(rgb)
    gh = _channel_gradient(np.maximum(hed[..., 0], 0.0), op=op, sigma=sigma)
    ge = _channel_gradient(np.maximum(hed[..., 1], 0.0), op=op, sigma=sigma)
    return normalize_signal(np.maximum(gh, ge))


def build_feature_stack(image: np.ndarray) -> np.ndarray:
    rgb = to_float_rgb(image)
    lab = color.rgb2lab(rgb)
    hsv = color.rgb2hsv(rgb)
    hed = color.rgb2hed(rgb)
    gray = color.rgb2gray(rgb)
    grad = filters.sobel(filters.gaussian(gray, sigma=1.0))
    gray_mu = filters.gaussian(gray, sigma=2.0)
    gray_var = np.maximum(filters.gaussian(np.square(gray), sigma=2.0) - np.square(gray_mu), 0.0)
    texture = np.sqrt(gray_var)
    od = optical_density_strength(image)
    feats = [
        normalize_signal(lab[..., 0]),
        normalize_signal(np.abs(lab[..., 1])),
        normalize_signal(np.abs(lab[..., 2])),
        normalize_signal(hsv[..., 1]),
        normalize_signal(od),
        normalize_signal(np.maximum(hed[..., 0], 0.0)),
        normalize_signal(np.maximum(hed[..., 1], 0.0)),
        normalize_signal(grad),
        normalize_signal(texture),
    ]
    return np.stack(feats, axis=-1).astype(np.float32)


def robust_center_scale(values: np.ndarray, min_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float32)
    if vals.ndim == 1:
        vals = vals[:, None]
    center = np.median(vals, axis=0)
    mad = np.median(np.abs(vals - center), axis=0)
    scale = np.maximum(1.4826 * mad, min_scale)
    return center.astype(np.float32), scale.astype(np.float32)


def model_distance(feature_stack: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (feature_stack - center) / scale
    return np.mean(np.square(z), axis=-1)


def build_stats_masks(base: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> Tuple[np.ndarray, np.ndarray]:
    inner = morphology.binary_erosion(base, morphology.disk(max(2, cfg.stats_core_radius)))
    if inner.sum() < 2048:
        inner = morphology.binary_erosion(base, morphology.disk(2))
    if inner.sum() < 2048:
        inner = base.copy()

    outer_inner = morphology.binary_dilation(base, morphology.disk(max(2, cfg.boundary_band // 3)))
    outer_outer = morphology.binary_dilation(base, morphology.disk(max(outer_inner.shape[0] * 0 + 2, cfg.boundary_band + cfg.stats_outer_margin)))
    outer = outer_outer & ~outer_inner
    outer &= ~seed_binary
    if outer.sum() < 2048:
        outer = ~morphology.binary_dilation(base, morphology.disk(max(2, cfg.boundary_band // 2)))
    return inner.astype(bool), outer.astype(bool)


def build_tissue_probability(image: np.ndarray, base: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    feats = build_feature_stack(image)
    inside_mask, outside_mask = build_stats_masks(base, seed_binary, cfg)
    inside_vals = feats[inside_mask]
    outside_vals = feats[outside_mask]
    if inside_vals.size == 0 or outside_vals.size == 0:
        return make_signal_bundle(image)["mix"]

    in_center, in_scale = robust_center_scale(inside_vals, cfg.stats_min_scale)
    out_center, out_scale = robust_center_scale(outside_vals, cfg.stats_min_scale)
    dist_in = model_distance(feats, in_center, in_scale)
    dist_out = model_distance(feats, out_center, out_scale)
    score = dist_out - dist_in

    in_score = float(np.median(score[inside_mask]))
    out_score = float(np.median(score[outside_mask]))
    mid = 0.5 * (in_score + out_score)
    scale = max(abs(in_score - out_score) * 0.5, 0.05)
    prob = 1.0 / (1.0 + np.exp(-(score - mid) / scale))
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def threshold_signal(signal: np.ndarray) -> np.ndarray:
    vals = signal[np.isfinite(signal)]
    if vals.size == 0:
        return np.zeros_like(signal, dtype=bool)
    thr = filters.threshold_otsu(vals) if np.unique(vals).size > 1 else float(vals.mean())
    return signal > thr


def bridge_seed_and_candidate(candidate: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    out = candidate.copy()
    out[seed_binary] = True
    if cfg.bridge_close_radius > 0:
        out = morphology.binary_closing(out, morphology.disk(cfg.bridge_close_radius))
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def boundary_band(mask: np.ndarray, width: int) -> np.ndarray:
    if width <= 0:
        return mask.copy()
    dil = morphology.binary_dilation(mask, morphology.disk(width))
    ero = morphology.binary_erosion(mask, morphology.disk(max(1, width // 2)))
    band = np.logical_xor(dil, ero)
    return band


def adaptive_boundary_width(mask: np.ndarray, cfg: GrowConfig) -> int:
    area = float(np.count_nonzero(mask))
    if area <= 0:
        return int(cfg.boundary_band)
    eq_radius = np.sqrt(area / np.pi)
    width = int(round(eq_radius * float(cfg.adaptive_band_scale)))
    width = max(int(cfg.adaptive_band_min), min(int(cfg.adaptive_band_max), width))
    return max(4, width)


def prepare_working_scale(
    signal: np.ndarray,
    base: np.ndarray,
    seed_binary: np.ndarray,
    cfg: GrowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    h, w = base.shape
    if max(h, w) <= cfg.no_downsample_max_side:
        return signal, base.astype(bool), seed_binary.astype(bool), 1
    factor = max(1, int(cfg.image_downsample))
    return signal[::factor, ::factor], base[::factor, ::factor].astype(bool), seed_binary[::factor, ::factor].astype(bool), factor


def refine_local_threshold(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    band = boundary_band(base, cfg.boundary_band)
    if not band.any():
        out = base.copy()
        out[seed_binary] = True
        return out
    vals = signal[band]
    thr = np.quantile(vals, cfg.boundary_quantile) if vals.size else 0.5
    local = signal >= thr
    out = base | (local & morphology.binary_dilation(base, morphology.disk(cfg.boundary_band)))
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_watershed(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signal_w, base_w, seed_w, factor = prepare_working_scale(signal, base, seed_binary, cfg)
    band = boundary_band(base_w, cfg.boundary_band)
    if not band.any():
        return base.copy()
    grad = filters.sobel(filters.gaussian(signal_w, sigma=cfg.boundary_sigma))
    markers = np.zeros(base_w.shape, dtype=np.int32)
    inside = morphology.binary_erosion(base_w, morphology.disk(max(1, cfg.boundary_band // 3)))
    outside = ~morphology.binary_dilation(base_w, morphology.disk(max(2, cfg.boundary_band // 2)))
    markers[inside] = 1
    markers[outside] = 2
    seg = segmentation.watershed(grad, markers=markers, mask=inside | band | outside)
    out = seg == 1
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_watershed_topography(base: np.ndarray, topography: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    topo_w, base_w, seed_w, factor = prepare_working_scale(topography, base, seed_binary, cfg)
    band = boundary_band(base_w, cfg.boundary_band)
    if not band.any():
        return base.copy()
    markers = np.zeros(base_w.shape, dtype=np.int32)
    inside = morphology.binary_erosion(base_w, morphology.disk(max(1, cfg.boundary_band // 3)))
    outside = ~morphology.binary_dilation(base_w, morphology.disk(max(2, cfg.boundary_band // 2)))
    markers[inside] = 1
    markers[outside] = 2
    seg = segmentation.watershed(topo_w, markers=markers, mask=inside | band | outside)
    out = seg == 1
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_mgac(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signal_w, base_w, seed_w, factor = prepare_working_scale(signal, base, seed_binary, cfg)
    smoothed = filters.gaussian(signal_w, sigma=cfg.boundary_sigma)
    gimg = segmentation.inverse_gaussian_gradient(smoothed)
    out = segmentation.morphological_geodesic_active_contour(
        gimg,
        num_iter=cfg.mgac_iters,
        init_level_set=base_w.astype(np.int8),
        smoothing=1,
        threshold=cfg.mgac_threshold,
        balloon=cfg.mgac_balloon,
    )
    out = np.asarray(out, dtype=bool)
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_chan_vese(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signal_w, base_w, seed_w, factor = prepare_working_scale(signal, base, seed_binary, cfg)
    out = segmentation.morphological_chan_vese(
        signal_w,
        num_iter=cfg.chan_vese_iters,
        init_level_set=base_w.astype(bool),
        smoothing=1,
        lambda1=1,
        lambda2=1,
    )
    out = np.asarray(out, dtype=bool)
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_random_walker(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signal_w, base_w, seed_w, factor = prepare_working_scale(signal, base, seed_binary, cfg)
    band = boundary_band(base_w, cfg.boundary_band)
    markers = np.zeros(base_w.shape, dtype=np.int32)
    inside = morphology.binary_erosion(base_w, morphology.disk(max(1, cfg.boundary_band // 3)))
    outside = ~morphology.binary_dilation(base_w, morphology.disk(max(2, cfg.boundary_band // 2)))
    markers[inside] = 1
    markers[outside] = 2
    if np.unique(markers).size < 3:
        return base.copy()
    labels = segmentation.random_walker(1.0 - signal_w, markers, beta=cfg.random_walker_beta, mode="cg_j")
    out = labels == 1
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = bridge_seed_and_candidate(out, seed_binary, cfg)
    return out


def refine_probability_gate(base: np.ndarray, prob: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    core = morphology.binary_erosion(base, morphology.disk(max(1, cfg.stats_core_radius)))
    if core.sum() == 0:
        core = seed_binary.copy()
    expanded = morphology.binary_dilation(base, morphology.disk(cfg.boundary_band))
    out = core.copy()
    out |= base & (prob >= cfg.stats_prune_prob)
    out |= expanded & (prob >= cfg.stats_grow_prob)
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def trim_boundary_by_probability(base: np.ndarray, prob: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    out = np.asarray(base).astype(bool).copy()
    band = boundary_band(out, cfg.boundary_band)
    if not band.any():
        out[seed_binary] = True
        return out
    protected = morphology.binary_dilation(seed_binary, morphology.disk(max(2, cfg.stats_core_radius // 2)))
    trim = band & ~protected & (prob < cfg.stats_prune_prob)
    out[trim] = False
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def trim_boundary_by_signal(base: np.ndarray, signal: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    out = np.asarray(base).astype(bool).copy()
    band = boundary_band(out, cfg.boundary_band)
    if not band.any():
        out[seed_binary] = True
        return out
    protected = morphology.binary_dilation(seed_binary, morphology.disk(max(2, cfg.stats_core_radius // 2)))
    inner = out & band & ~protected
    vals = signal[inner]
    if vals.size == 0:
        vals = signal[out & ~protected]
    if vals.size == 0:
        out[seed_binary] = True
        return out
    thr = float(np.quantile(vals, cfg.boundary_quantile))
    trim = band & ~protected & (signal < thr)
    out[trim] = False
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def refine_probability_random_walker(base: np.ndarray, prob: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    prob_w, base_w, seed_w, factor = prepare_working_scale(prob, base, seed_binary, cfg)
    markers = np.zeros(base_w.shape, dtype=np.int32)
    inside = morphology.binary_erosion(base_w, morphology.disk(max(1, cfg.stats_core_radius // max(1, factor))))
    if inside.sum() == 0:
        inside = seed_w.copy()
    outside_far = ~morphology.binary_dilation(base_w, morphology.disk(max(2, cfg.boundary_band // max(1, factor))))
    outside_low = prob_w <= np.quantile(prob_w[outside_far], 0.75) if outside_far.any() else np.zeros_like(prob_w, dtype=bool)
    outside = outside_far & outside_low
    if outside.sum() == 0:
        outside = outside_far
    markers[inside] = 1
    markers[outside] = 2
    if np.unique(markers).size < 3:
        return refine_probability_gate(base, prob, seed_binary, cfg)
    labels = segmentation.random_walker(1.0 - prob_w, markers, beta=max(90, cfg.random_walker_beta), mode="cg_j")
    out = labels == 1
    if factor > 1:
        out = resize_binary(out, base.shape)
    out |= seed_binary
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def refine_superpixel_prototypes(base: np.ndarray, image: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    rgb = to_float_rgb(image)
    feats = build_feature_stack(image)
    h, w = base.shape
    n_segments = max(128, int(round((h * w) / float(max(16, cfg.superpixel_size) ** 2))))
    segments = segmentation.slic(
        rgb,
        n_segments=n_segments,
        compactness=float(cfg.superpixel_compactness),
        start_label=1,
        convert2lab=True,
        channel_axis=-1,
    )

    base = np.asarray(base).astype(bool)
    band = boundary_band(base, cfg.boundary_band)
    expand_mask = morphology.binary_dilation(base, morphology.disk(max(2, cfg.superpixel_expand_margin)))
    core = morphology.binary_erosion(base, morphology.disk(max(2, cfg.stats_core_radius)))
    if core.sum() == 0:
        core = base.copy()
    outside = ~morphology.binary_dilation(base, morphology.disk(max(2, cfg.boundary_band + cfg.stats_outer_margin)))
    outside |= ~(band | base)

    seg_ids = np.unique(segments)
    feat_dim = feats.shape[-1]
    seg_means = np.zeros((seg_ids.size, feat_dim), dtype=np.float32)
    fg_ids = []
    bg_ids = []
    band_ids = []
    seg_id_to_idx = {int(sid): idx for idx, sid in enumerate(seg_ids)}

    for sid in seg_ids:
        seg = segments == sid
        seg_feats = feats[seg]
        idx = seg_id_to_idx[int(sid)]
        seg_means[idx] = seg_feats.mean(axis=0)
        frac_core = float(core[seg].mean())
        frac_seed = float(seed_binary[seg].mean())
        frac_out = float(outside[seg].mean())
        if max(frac_core, frac_seed) >= float(cfg.superpixel_fg_ratio):
            fg_ids.append(int(sid))
        elif frac_out >= float(cfg.superpixel_bg_ratio):
            bg_ids.append(int(sid))
        elif np.any(seg & (band | expand_mask)):
            band_ids.append(int(sid))

    if not fg_ids or not bg_ids:
        out = base.copy()
        out[seed_binary] = True
        return out

    fg_center, fg_scale = robust_center_scale(seg_means[[seg_id_to_idx[s] for s in fg_ids]], cfg.stats_min_scale)
    bg_center, bg_scale = robust_center_scale(seg_means[[seg_id_to_idx[s] for s in bg_ids]], cfg.stats_min_scale)

    out = base.copy()
    out[seed_binary] = True
    protected = morphology.binary_dilation(seed_binary, morphology.disk(max(2, cfg.stats_core_radius // 2)))
    for sid in band_ids:
        seg = segments == sid
        if np.any(seg & protected):
            out[seg] = True
            continue
        feat = seg_means[seg_id_to_idx[sid]]
        d_fg = float(np.mean(np.square((feat - fg_center) / fg_scale)))
        d_bg = float(np.mean(np.square((feat - bg_center) / bg_scale)))
        if d_fg <= d_bg:
            out[seg] = True
        else:
            out[seg] = False

    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def composite_edge_map(image: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    edge = np.maximum.reduce([
        rgb_gradient_fusion(image, op="scharr", rule="max", sigma=cfg.boundary_sigma),
        lab_ab_gradient_fusion(image, op="scharr", sigma=cfg.boundary_sigma),
        hed_gradient_fusion(image, op="scharr", sigma=cfg.boundary_sigma),
    ])
    return normalize_signal(edge)


def build_background_likelihood(image: np.ndarray) -> np.ndarray:
    rgb = to_float_rgb(image)
    hsv = color.rgb2hsv(rgb)
    lab = color.rgb2lab(rgb)
    gray = color.rgb2gray(rgb)
    gray_mu = filters.gaussian(gray, sigma=2.0)
    gray_var = np.maximum(filters.gaussian(np.square(gray), sigma=2.0) - np.square(gray_mu), 0.0)
    texture = normalize_signal(np.sqrt(gray_var))
    light = normalize_signal(lab[..., 0])
    sat = normalize_signal(hsv[..., 1])
    od = normalize_signal(optical_density_strength(image))
    bg_like = 0.35 * light + 0.25 * (1.0 - sat) + 0.25 * (1.0 - od) + 0.15 * (1.0 - texture)
    return normalize_signal(bg_like)


def limit_to_boundary_band(base: np.ndarray, candidate: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = np.asarray(base).astype(bool)
    candidate = np.asarray(candidate).astype(bool)
    band = boundary_band(base, cfg.boundary_band)
    out = base.copy()
    out[band] = candidate[band]
    protected = morphology.binary_erosion(base, morphology.disk(max(2, cfg.stats_core_radius)))
    if protected.sum() == 0:
        protected = base & ~band
    out[protected] = True
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def refine_band_mgac(base: np.ndarray, gimage: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    g_w, base_w, seed_w, factor = prepare_working_scale(gimage, base, seed_binary, cfg)
    out = segmentation.morphological_geodesic_active_contour(
        np.clip(g_w, 0.0, 1.0),
        num_iter=cfg.mgac_iters,
        init_level_set=base_w.astype(np.int8),
        smoothing=2,
        threshold=cfg.mgac_threshold,
        balloon=0.0,
    )
    out = np.asarray(out, dtype=bool)
    if factor > 1:
        out = resize_binary(out, base.shape)
    out = limit_to_boundary_band(base, out, seed_binary, cfg)
    return out


def conservative_background_trim(base: np.ndarray, image: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = np.asarray(base).astype(bool)
    band = boundary_band(base, cfg.boundary_band)
    if not band.any():
        out = base.copy()
        out[seed_binary] = True
        return out

    prob = build_tissue_probability(image, base, seed_binary, cfg)
    edge = composite_edge_map(image, cfg)
    bg_like = build_background_likelihood(image)
    protected = morphology.binary_dilation(seed_binary, morphology.disk(max(2, cfg.stats_core_radius // 2)))
    outer = ~morphology.binary_dilation(base, morphology.disk(max(2, cfg.boundary_band + cfg.stats_outer_margin)))

    bg_vals = bg_like[outer]
    bg_thr = float(np.quantile(bg_vals, 0.55)) if bg_vals.size else 0.65
    edge_vals = edge[band & base & ~protected]
    edge_thr = float(np.quantile(edge_vals, 0.35)) if edge_vals.size else 0.25

    trim = (
        band
        & ~protected
        & (prob < max(cfg.stats_prune_prob, 0.35))
        & (bg_like >= bg_thr)
        & (edge <= edge_thr)
    )
    out = base.copy()
    out[trim] = False
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def seeded_image_mask(image: np.ndarray, seed_labels: np.ndarray, candidate: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    out = bridge_seed_and_candidate(candidate, seed_binary, cfg)
    out = cleanup_mask(out, cfg, keep_largest=True)
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    out[seed_binary] = True
    return out


def initial_segment_expand(seed_labels: np.ndarray, support_mask: np.ndarray, radius: int) -> np.ndarray:
    seed_binary = seed_labels > 0
    support_mask = np.asarray(support_mask).astype(bool)
    expanded = morphology.binary_dilation(seed_binary, morphology.disk(max(1, int(radius))))
    expanded &= support_mask
    expanded |= seed_binary
    labels = grow_labels_within_mask(seed_labels, expanded)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def build_outer_background_mask(image: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    bg_like = build_background_likelihood(image)
    border = np.zeros(bg_like.shape, dtype=bool)
    border[:8, :] = True
    border[-8:, :] = True
    border[:, :8] = True
    border[:, -8:] = True
    border_vals = bg_like[border]
    threshold = float(np.quantile(border_vals, float(cfg.texture_bg_quantile))) if border_vals.size else 0.78
    obvious_bg = bg_like >= max(0.72, threshold)
    obvious_bg &= ~morphology.binary_dilation(seed_binary, morphology.disk(max(4, cfg.texture_init_expand_radius)))
    if obvious_bg.sum() < 2048:
        obvious_bg = border & ~morphology.binary_dilation(seed_binary, morphology.disk(max(4, cfg.texture_init_expand_radius)))
    return obvious_bg.astype(bool)


def fill_small_holes_by_nearest_label(labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32)
    union = labels > 0
    filled = morphology.remove_small_holes(union, area_threshold=max(0, int(cfg.texture_hole_area)))
    if np.array_equal(filled, union):
        return labels
    reassigned = grow_labels_within_mask(labels.astype(np.uint16), filled)
    reassigned[labels > 0] = labels[labels > 0]
    return reassigned.astype(np.int32)


def cleanup_label_map(labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32)
    seed_binary = labels > 0
    if not seed_binary.any():
        return labels
    union = cleanup_mask(seed_binary, with_cfg(
        cfg,
        final_hole_area=max(cfg.final_hole_area, cfg.texture_hole_area),
        final_min_obj_area=max(cfg.final_min_obj_area, cfg.texture_min_obj_area),
    ), keep_largest=False)
    out = grow_labels_within_mask(labels.astype(np.uint16), union).astype(np.int32)
    out[seed_binary] = labels[seed_binary]
    out = fill_small_holes_by_nearest_label(out, cfg)
    return out.astype(np.int32)


def build_label_models(
    feature_stack: np.ndarray,
    labels: np.ndarray,
    cfg: GrowConfig,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    models: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    for lab in [int(x) for x in np.unique(labels) if int(x) != 0]:
        mask = labels == lab
        vals = feature_stack[mask]
        if vals.size == 0:
            continue
        center, scale = robust_center_scale(vals, cfg.stats_min_scale)
        dist = model_distance(vals, center, scale)
        thr = float(np.quantile(dist, float(cfg.texture_accept_quantile))) if dist.size else 1.0
        thr = max(float(cfg.texture_accept_floor), thr * float(cfg.texture_accept_scale))
        models[lab] = (center, scale, thr)
    return models


def build_growth_support_mask(image: np.ndarray, seed_binary: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    bg_like = build_background_likelihood(image)
    od = make_signal_bundle(image)["od"]
    border = np.zeros(bg_like.shape, dtype=bool)
    border[:8, :] = True
    border[-8:, :] = True
    border[:, :8] = True
    border[:, -8:] = True
    border_vals = bg_like[border]
    thr = float(np.quantile(border_vals, max(0.88, float(cfg.texture_bg_quantile)))) if border_vals.size else 0.90
    obvious_blank = (bg_like >= thr) & (od <= 0.18)
    obvious_blank &= ~morphology.binary_dilation(seed_binary, morphology.disk(max(4, cfg.texture_init_expand_radius)))
    support = ~obvious_blank
    support |= seed_binary
    return support.astype(bool)


def build_label_models_from_masks(
    feature_stack: np.ndarray,
    source_masks: Dict[int, np.ndarray],
    cfg: GrowConfig,
    relax: float = 0.0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray, float]]:
    models: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    for lab, mask in source_masks.items():
        vals = feature_stack[np.asarray(mask).astype(bool)]
        if vals.size == 0:
            continue
        center, scale = robust_center_scale(vals, cfg.stats_min_scale)
        dist = model_distance(vals, center, scale)
        thr = float(np.quantile(dist, float(cfg.texture_accept_quantile))) if dist.size else 1.0
        thr = max(float(cfg.texture_accept_floor), thr * float(cfg.texture_accept_scale))
        thr *= (1.0 + max(0.0, float(relax)))
        models[int(lab)] = (center, scale, thr)
    return models


def stable_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    temperature = max(1e-6, float(temperature))
    z = logits / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    denom = np.sum(e, axis=-1, keepdims=True)
    denom = np.maximum(denom, 1e-8)
    return (e / denom).astype(np.float32)


def dual_transition_refine_labels(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    if not seed_binary.any():
        return seed_labels

    feats = build_feature_stack(image)
    support_mask = build_growth_support_mask(image, seed_binary, cfg)
    labels = initial_segment_expand(seed_labels, support_mask, cfg.texture_init_expand_radius)

    bg_mask = build_outer_background_mask(image, seed_binary, cfg)
    if bg_mask.sum() < 2048:
        bg_mask = ~morphology.binary_dilation(labels > 0, morphology.disk(max(8, cfg.dual_band_radius)))
        bg_mask &= ~seed_binary
    bg_vals = feats[bg_mask]
    if bg_vals.size == 0:
        bg_vals = feats.reshape(-1, feats.shape[-1])
    bg_center, bg_scale = robust_center_scale(bg_vals, cfg.stats_min_scale)

    for _ in range(max(1, int(cfg.dual_cycles))):
        union = labels > 0
        lab_ids = [int(x) for x in np.unique(labels) if int(x) != 0]
        if not lab_ids:
            break

        source_masks = {lab: labels == lab for lab in lab_ids}
        models = build_label_models_from_masks(feats, source_masks, cfg, relax=0.10)
        if not models:
            break

        prob = build_tissue_probability(image, union, seed_binary, cfg)
        bg_like = build_background_likelihood(image)

        frontier = morphology.binary_dilation(union, morphology.disk(max(1, int(cfg.dual_band_radius))))
        frontier &= support_mask
        frontier &= ~seed_binary
        contact = morphology.binary_dilation(contact_zone_mask(labels), morphology.disk(1))
        candidate = frontier | contact
        if not candidate.any():
            break

        rr, cc = np.nonzero(candidate)
        pix_feats = feats[rr, cc]
        num_seg = len(lab_ids)
        logits = np.zeros((rr.size, num_seg + 1), dtype=np.float32)
        lab_to_idx = {lab: i for i, lab in enumerate(lab_ids)}

        for lab, idx in lab_to_idx.items():
            center, scale, _thr = models[lab]
            d_lab = model_distance(pix_feats, center, scale)
            logits[:, idx] = (
                -d_lab
                + float(cfg.dual_prob_weight) * prob[rr, cc]
                - 0.15 * bg_like[rr, cc]
            )

        d_bg = model_distance(pix_feats, bg_center, bg_scale)
        logits[:, -1] = (
            -float(cfg.dual_bg_weight) * d_bg
            + (1.0 - prob[rr, cc])
            + 0.35 * bg_like[rr, cc]
        )

        current_labels = labels[rr, cc]
        for i, cur in enumerate(current_labels):
            cur = int(cur)
            if cur > 0 and cur in lab_to_idx:
                logits[i, lab_to_idx[cur]] += float(cfg.dual_identity_weight)

        probs = stable_softmax(logits, temperature=cfg.dual_temperature)
        best_idx = np.argmax(probs, axis=1)
        best_prob = probs[np.arange(probs.shape[0]), best_idx]
        bg_prob = probs[:, -1]

        new_labels = labels.copy()
        changed = 0
        for i, (r, c) in enumerate(zip(rr, cc)):
            cur = int(labels[r, c])
            winner = int(best_idx[i])
            if winner == num_seg:
                continue
            winner_lab = lab_ids[winner]
            if cur == 0:
                if best_prob[i] >= float(cfg.dual_expand_prob) and bg_prob[i] <= float(cfg.dual_bg_prob_max):
                    new_labels[r, c] = winner_lab
                    changed += 1
            elif winner_lab != cur:
                cur_idx = lab_to_idx.get(cur)
                cur_prob = probs[i, cur_idx] if cur_idx is not None else 0.0
                if best_prob[i] >= float(cur_prob) + float(cfg.dual_relabel_margin):
                    new_labels[r, c] = winner_lab
                    changed += 1

        if changed == 0:
            break
        labels = cleanup_label_map(new_labels, cfg)
        labels[seed_binary] = seed_labels[seed_binary]

    labels = cleanup_label_map(labels, cfg)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def ordered_rgb_difference_maps(image: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = to_float_rgb(image)
    chans = {"r": rgb[..., 0], "g": rgb[..., 1], "b": rgb[..., 2]}
    diffs: Dict[str, np.ndarray] = {}
    for a in ("r", "g", "b"):
        for b in ("r", "g", "b"):
            if a == b:
                continue
            diffs[f"{a}-{b}"] = chans[a] - chans[b]
    return diffs


def cleanup_candidate_mask(mask: np.ndarray, seed_mask: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    mask |= seed_mask
    if cfg.gcdt_close_radius > 0:
        mask = morphology.binary_closing(mask, morphology.disk(int(cfg.gcdt_close_radius)))
    if cfg.gcdt_hole_area > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=int(cfg.gcdt_hole_area))
    if cfg.gcdt_min_obj_area > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(cfg.gcdt_min_obj_area))
    mask = keep_seeded_components(mask, seed_mask)
    mask |= seed_mask
    return mask


def gcdt_best_candidate_for_label(
    diff_maps: Dict[str, np.ndarray],
    rough_mask: np.ndarray,
    seed_mask: np.ndarray,
    cfg: GrowConfig,
) -> np.ndarray:
    rough_mask = np.asarray(rough_mask).astype(bool)
    seed_mask = np.asarray(seed_mask).astype(bool)
    if not rough_mask.any():
        return seed_mask.copy()

    dist = ndi.distance_transform_edt(rough_mask)
    if dist.max(initial=0.0) > 0:
        core = rough_mask & ((dist / float(dist.max())) >= float(cfg.gcdt_core_threshold))
    else:
        core = seed_mask.copy()
    if core.sum() < max(64, int(seed_mask.sum())):
        core = morphology.binary_dilation(seed_mask, morphology.disk(2)) & rough_mask
    if core.sum() == 0:
        core = seed_mask.copy()

    best_score = -1e9
    best_mask = seed_mask.copy()
    quantiles = (0.35, 0.45, 0.55, 0.65, 0.75)

    for _name, diff in diff_maps.items():
        vals = diff[rough_mask]
        if vals.size == 0:
            continue
        thresholds = [float(np.quantile(vals, q)) for q in quantiles]
        uniq = np.unique(np.round(vals.astype(np.float32), 4))
        if uniq.size > 1:
            try:
                thresholds.append(float(filters.threshold_otsu(vals)))
            except Exception:
                pass
        for thr in thresholds:
            cand = rough_mask & (diff >= thr)
            cand = cleanup_candidate_mask(cand, seed_mask, cfg)
            if not cand.any():
                continue
            core_iou = score_binary_masks(cand, core)["iou"]
            coverage = float(np.count_nonzero(cand)) / max(1.0, float(np.count_nonzero(rough_mask)))
            seed_cover = float(np.count_nonzero(cand & seed_mask)) / max(1.0, float(np.count_nonzero(seed_mask)))
            score = (1.40 * core_iou) + (0.20 * coverage) + (0.20 * seed_cover)
            if score > best_score:
                best_score = score
                best_mask = cand

    return best_mask.astype(bool)


def gcdt_coarse_to_fine_labels(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    if not seed_binary.any():
        return seed_labels

    support_mask = build_growth_support_mask(image, seed_binary, cfg)
    diff_maps = ordered_rgb_difference_maps(image)
    labels = seed_labels.copy()

    for _ in range(max(1, int(cfg.gcdt_cycles))):
        union = labels > 0
        rough_union = morphology.binary_dilation(union, morphology.disk(max(1, int(cfg.gcdt_rough_radius))))
        rough_union &= support_mask
        rough_union |= union
        rough_labels = grow_labels_within_mask(labels.astype(np.uint16), rough_union).astype(np.int32)

        new_labels = labels.copy()
        for lab in [int(x) for x in np.unique(labels) if int(x) != 0]:
            rough_mask = rough_labels == lab
            if not rough_mask.any():
                continue
            seed_mask = labels == lab
            cand = gcdt_best_candidate_for_label(diff_maps, rough_mask, seed_mask, cfg)
            new_labels[cand] = lab
        labels = cleanup_label_map(new_labels, cfg)
        labels[seed_binary] = seed_labels[seed_binary]

    labels = cleanup_label_map(labels, cfg)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def contact_zone_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32)
    pos = labels > 0
    if not pos.any():
        return np.zeros_like(labels, dtype=bool)
    zone = np.zeros_like(pos, dtype=bool)
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rolled = np.roll(labels, shift, axis=(0, 1))
        zone |= pos & (rolled > 0) & (rolled != labels)
    zone &= pos
    return zone


def refine_contact_border_labels(
    feature_stack: np.ndarray,
    labels: np.ndarray,
    models: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    cfg: GrowConfig,
) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32).copy()
    if len(models) < 2:
        return labels
    improve_ratio = float(cfg.texture_contact_improve_ratio)
    for _ in range(max(0, int(cfg.texture_contact_iters))):
        zone = morphology.binary_dilation(contact_zone_mask(labels), morphology.disk(1))
        zone &= labels > 0
        if not zone.any():
            break
        rr, cc = np.nonzero(zone)
        changed = 0
        for r, c in zip(rr, cc):
            current = int(labels[r, c])
            if current == 0 or current not in models:
                continue
            r0 = max(0, r - 1)
            r1 = min(labels.shape[0], r + 2)
            c0 = max(0, c - 1)
            c1 = min(labels.shape[1], c + 2)
            neigh = [int(x) for x in np.unique(labels[r0:r1, c0:c1]) if int(x) != 0]
            if len(neigh) < 2:
                continue
            feat = feature_stack[r, c]
            best_lab = current
            cur_center, cur_scale, _ = models[current]
            best_dist = float(np.mean(np.square((feat - cur_center) / cur_scale)))
            for lab in neigh:
                if lab == current or lab not in models:
                    continue
                center, scale, _ = models[lab]
                d_lab = float(np.mean(np.square((feat - center) / scale)))
                if d_lab < best_dist * improve_ratio:
                    best_lab = lab
                    best_dist = d_lab
            if best_lab != current:
                labels[r, c] = best_lab
                changed += 1
        if changed == 0:
            break
    return labels


def grow_by_segment_texture_cycles(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    if not seed_binary.any():
        return seed_labels

    feats = build_feature_stack(image)
    support_mask = build_growth_support_mask(image, seed_binary, cfg)
    labels = initial_segment_expand(seed_labels, support_mask, cfg.texture_init_expand_radius)
    outer_bg = ~support_mask
    if outer_bg.sum() < 2048:
        outer_bg = build_outer_background_mask(image, seed_binary, cfg)
    bg_vals = feats[outer_bg]
    if bg_vals.size == 0:
        bg_vals = feats.reshape(-1, feats.shape[-1])
    bg_center, bg_scale = robust_center_scale(bg_vals, cfg.stats_min_scale)

    prev_added: Dict[int, np.ndarray] = {}
    cycle_count = max(1, int(cfg.texture_cycles))
    cycle_radius = max(1, int(cfg.texture_cycle_radius))

    for cycle_idx in range(cycle_count):
        union = labels > 0
        prob = build_tissue_probability(image, union, seed_binary, cfg)
        bg_like = build_background_likelihood(image)
        base_prob_thr = float(cfg.texture_prob_floor)
        inside_prob = prob[union]
        if inside_prob.size:
            base_prob_thr = max(base_prob_thr, float(np.quantile(inside_prob, float(cfg.texture_prob_quantile))) * 0.95)
        prob_thr = max(0.20, base_prob_thr - cycle_idx * float(cfg.texture_cycle_prob_decay))

        source_masks: Dict[int, np.ndarray] = {}
        for lab in [int(x) for x in np.unique(labels) if int(x) != 0]:
            current = labels == lab
            recent = prev_added.get(lab)
            if recent is not None and int(np.count_nonzero(recent)) >= int(cfg.texture_newarea_min_pixels):
                source_masks[lab] = recent | morphology.binary_dilation(recent, morphology.disk(max(1, cfg.texture_step_radius)))
            else:
                source_masks[lab] = current
        models = build_label_models_from_masks(
            feats,
            source_masks,
            cfg,
            relax=cycle_idx * float(cfg.texture_cycle_accept_step),
        )
        if not models:
            break

        new_labels = labels.copy()
        cycle_added: Dict[int, np.ndarray] = {}
        accepted_total = 0
        best_score = np.full(union.shape, -np.inf, dtype=np.float32)
        best_label = np.zeros(union.shape, dtype=np.int32)

        for lab, (center, scale, thr) in models.items():
            current = labels == lab
            current_area = int(np.count_nonzero(current))
            recent = prev_added.get(lab)
            recent_area = int(np.count_nonzero(recent)) if recent is not None else 0
            growth_ratio = (recent_area / max(1.0, float(current_area))) if current_area > 0 else 0.0
            undergrown = growth_ratio < 0.02
            local_radius = cycle_radius + (4 if undergrown else 0)
            local_prob_thr = max(0.15, prob_thr - (0.08 if undergrown else 0.0))
            cand = morphology.binary_dilation(current, morphology.disk(local_radius))
            cand &= ~union
            cand &= support_mask
            cand &= prob >= local_prob_thr
            if not cand.any():
                continue
            d_lab = model_distance(feats[cand], center, scale)
            d_bg = model_distance(feats[cand], bg_center, bg_scale)
            bg_ratio = float(cfg.texture_bg_ratio) + cycle_idx * float(cfg.texture_cycle_bg_ratio_step) + (0.25 if undergrown else 0.0)
            accept = (d_lab <= thr) & (d_lab <= (d_bg * bg_ratio))
            if not np.any(accept):
                continue
            rr, cc = np.nonzero(cand)
            score = (-d_lab + 0.35 * prob[cand] - 0.20 * bg_like[cand]).astype(np.float32)
            keep = accept & (score > best_score[rr, cc])
            if not np.any(keep):
                continue
            rr = rr[keep]
            cc = cc[keep]
            best_score[rr, cc] = score[keep]
            best_label[rr, cc] = lab

        if np.any(best_label > 0):
            new_labels[best_label > 0] = best_label[best_label > 0]
            for lab in [int(x) for x in np.unique(best_label) if int(x) != 0]:
                added_mask = best_label == lab
                cycle_added[lab] = added_mask
                accepted_total += int(np.count_nonzero(added_mask))

        labels = cleanup_label_map(new_labels, cfg)
        relabel_source_masks: Dict[int, np.ndarray] = {}
        for lab in [int(x) for x in np.unique(labels) if int(x) != 0]:
            current = labels == lab
            recent = cycle_added.get(lab)
            if recent is not None and int(np.count_nonzero(recent)) >= int(cfg.texture_newarea_min_pixels):
                relabel_source_masks[lab] = current | recent
            else:
                relabel_source_masks[lab] = current
        relabel_models = build_label_models_from_masks(
            feats,
            relabel_source_masks,
            cfg,
            relax=cycle_idx * float(cfg.texture_cycle_accept_step),
        )
        labels = refine_contact_border_labels(feats, labels, relabel_models, cfg)
        labels = cleanup_label_map(labels, cfg)
        labels[seed_binary] = seed_labels[seed_binary]
        prev_added = cycle_added
        if accepted_total == 0:
            break

    labels = cleanup_label_map(labels, cfg)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def grow_by_segment_texture_similarity(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    if not seed_binary.any():
        return seed_labels

    feats = build_feature_stack(image)
    support_mask = ~build_outer_background_mask(image, seed_binary, cfg)
    labels = initial_segment_expand(seed_labels, support_mask, cfg.texture_init_expand_radius)
    union = labels > 0

    outer_bg = build_outer_background_mask(image, seed_binary, cfg)
    bg_vals = feats[outer_bg]
    if bg_vals.size == 0:
        bg_vals = feats[~morphology.binary_dilation(union, morphology.disk(max(8, cfg.texture_init_expand_radius + cfg.texture_step_radius)))]
    if bg_vals.size == 0:
        bg_vals = feats.reshape(-1, feats.shape[-1])
    bg_center, bg_scale = robust_center_scale(bg_vals, cfg.stats_min_scale)

    for _ in range(max(1, int(cfg.texture_max_iters))):
        union = labels > 0
        prob = build_tissue_probability(image, union, seed_binary, cfg)
        inside_prob = prob[union]
        prob_thr = float(cfg.texture_prob_floor)
        if inside_prob.size:
            prob_thr = max(prob_thr, float(np.quantile(inside_prob, float(cfg.texture_prob_quantile))) * 0.95)

        models = build_label_models(feats, labels, cfg)
        if not models:
            break

        frontier = morphology.binary_dilation(union, morphology.disk(max(1, int(cfg.texture_step_radius))))
        frontier &= ~union
        frontier &= support_mask
        frontier &= prob >= prob_thr
        if not frontier.any():
            break

        ownership = grow_labels_within_mask(labels.astype(np.uint16), union | frontier).astype(np.int32)
        new_labels = labels.copy()
        accepted_total = 0

        for lab, (center, scale, thr) in models.items():
            cand = frontier & (ownership == lab)
            if not cand.any():
                continue
            d_lab = model_distance(feats[cand], center, scale)
            d_bg = model_distance(feats[cand], bg_center, bg_scale)
            accept = (d_lab <= thr) & (d_lab <= (d_bg * float(cfg.texture_bg_ratio)))
            if not np.any(accept):
                continue
            rr, cc = np.nonzero(cand)
            rr = rr[accept]
            cc = cc[accept]
            new_labels[rr, cc] = lab
            accepted_total += int(accept.sum())

        labels = new_labels
        if accepted_total == 0:
            break

    labels = cleanup_label_map(labels, cfg)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def rescue_supported_components(
    base: np.ndarray,
    candidate: np.ndarray,
    score: np.ndarray,
    seed_binary: np.ndarray,
    cfg: GrowConfig,
) -> np.ndarray:
    out = np.asarray(base).astype(bool).copy()
    candidate = np.asarray(candidate).astype(bool)
    score = np.asarray(score, dtype=np.float32)
    extra = candidate & ~out
    if not extra.any():
        out[seed_binary] = True
        return out

    labels = measure.label(extra, connectivity=1)
    if labels.max() == 0:
        out[seed_binary] = True
        return out

    dist_to_base = ndi.distance_transform_edt(~out)
    for region in measure.regionprops(labels):
        if region.area < cfg.rescue_min_area:
            continue
        comp = labels == region.label
        if float(dist_to_base[comp].min(initial=np.inf)) > float(cfg.rescue_max_gap):
            continue
        comp_scores = score[comp]
        if comp_scores.size == 0:
            continue
        comp_q = float(np.quantile(comp_scores, cfg.rescue_score_quantile))
        comp_mean = float(np.mean(comp_scores))
        if max(comp_q, comp_mean) < float(cfg.rescue_score_floor):
            continue
        out[comp] = True

    out[seed_binary] = True
    return out


MethodFn = Callable[[np.ndarray, np.ndarray, GrowConfig], np.ndarray]


def finalize_label_output(labels: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    labels = np.asarray(labels).astype(np.int32)
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    labels[seed_binary] = seed_labels[seed_binary]
    labels = cleanup_label_map(labels, cfg)
    labels[seed_binary] = seed_labels[seed_binary]
    return labels.astype(np.int32)


def normalize_method_output(output: np.ndarray, seed_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(output)
    seed_labels = np.asarray(seed_labels).astype(np.int32)
    seed_binary = seed_labels > 0
    is_label_map = np.issubdtype(arr.dtype, np.integer) and not np.array_equal(np.unique(arr), np.array([0], dtype=arr.dtype)) and not set(np.unique(arr).tolist()).issubset({0, 1})
    if is_label_map:
        labels = arr.astype(np.int32)
        labels[seed_binary] = seed_labels[seed_binary]
        tissue_mask = labels > 0
        return tissue_mask.astype(bool), labels.astype(np.uint16)
    tissue_mask = arr.astype(bool)
    tissue_mask[seed_binary] = True
    labels = grow_labels_within_mask(seed_labels, tissue_mask)
    labels[seed_binary] = seed_labels[seed_binary]
    return tissue_mask.astype(bool), labels.astype(np.uint16)


def method_seed_only(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    return cleanup_mask(seed_labels > 0, cfg, keep_largest=False)


def method_chroma_seeded(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    candidate = build_chroma_candidate(image, cfg)
    return seeded_image_mask(image, seed_labels, candidate, cfg)


def method_od_seeded(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    candidate = threshold_signal(make_signal_bundle(image)["od"])
    return seeded_image_mask(image, seed_labels, candidate, cfg)


def method_hed_seeded(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    candidate = threshold_signal(make_signal_bundle(image)["hed"])
    return seeded_image_mask(image, seed_labels, candidate, cfg)


def method_union_seeded(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    candidate = build_chroma_candidate(image, cfg) | threshold_signal(signals["od"]) | threshold_signal(signals["hed"])
    return seeded_image_mask(image, seed_labels, candidate, cfg)


def method_union_local(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_union_seeded(image, seed_labels, cfg)
    out = refine_local_threshold(base, signals["mix"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_union_seeded(image, seed_labels, cfg)
    out = refine_watershed(base, signals["mix"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_stats_local(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    out = refine_probability_gate(base, prob, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_stats_rw(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    out = refine_probability_random_walker(base, prob, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_stats_hybrid(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    local = refine_probability_gate(base, prob, seed_binary, cfg)
    rw = refine_probability_random_walker(base, prob, seed_binary, cfg)
    out = (local | rw) & morphology.binary_dilation(base | seed_binary, morphology.disk(max(8, cfg.boundary_band)))
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_stats_rw_adaptive(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    band_cfg = with_cfg(cfg, boundary_band=adaptive_boundary_width(base, cfg))
    out = refine_probability_random_walker(base, prob, seed_binary, band_cfg)
    return cleanup_mask(out, band_cfg, keep_largest=True)


def method_union_stats_hybrid_adaptive(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    band_cfg = with_cfg(cfg, boundary_band=adaptive_boundary_width(base, cfg))
    local = refine_probability_gate(base, prob, seed_binary, band_cfg)
    rw = refine_probability_random_walker(base, prob, seed_binary, band_cfg)
    out = (local | rw) & morphology.binary_dilation(base | seed_binary, morphology.disk(max(8, band_cfg.boundary_band)))
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, band_cfg, keep_largest=True)


def method_union_superpixel_proto(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    out = refine_superpixel_prototypes(base, image, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_superpixel_proto_adaptive(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    band_cfg = with_cfg(cfg, boundary_band=adaptive_boundary_width(base, cfg))
    out = refine_superpixel_prototypes(base, image, seed_binary, band_cfg)
    return cleanup_mask(out, band_cfg, keep_largest=True)


def method_union_prob_mgac(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    prob_stop = segmentation.inverse_gaussian_gradient(filters.gaussian(prob, sigma=cfg.boundary_sigma))
    out = refine_band_mgac(base, prob_stop, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_hybrid_mgac(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    prob_stop = segmentation.inverse_gaussian_gradient(filters.gaussian(prob, sigma=cfg.boundary_sigma))
    edge_stop = np.clip(1.0 - composite_edge_map(image, cfg), 0.0, 1.0)
    gimage = np.clip(0.65 * prob_stop + 0.35 * edge_stop, 0.0, 1.0)
    out = refine_band_mgac(base, gimage, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_bg_trim(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    out = conservative_background_trim(base, image, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_rgb_sobel_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = method_union_seeded(image, seed_labels, cfg)
    topo = 1.0 - rgb_gradient_fusion(image, op="sobel", rule="max", sigma=cfg.boundary_sigma)
    out = refine_watershed_topography(base, topo, seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_rgb_scharr_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = method_union_seeded(image, seed_labels, cfg)
    topo = 1.0 - rgb_gradient_fusion(image, op="scharr", rule="max", sigma=cfg.boundary_sigma)
    out = refine_watershed_topography(base, topo, seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_lab_ab_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = method_union_seeded(image, seed_labels, cfg)
    topo = 1.0 - lab_ab_gradient_fusion(image, op="scharr", sigma=cfg.boundary_sigma)
    out = refine_watershed_topography(base, topo, seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_hed_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    base = method_union_seeded(image, seed_labels, cfg)
    topo = 1.0 - hed_gradient_fusion(image, op="scharr", sigma=cfg.boundary_sigma)
    out = refine_watershed_topography(base, topo, seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_watershed_recall(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    recall_cfg = GrowConfig(**{**config_dict(cfg), "image_keep_largest": False})
    signals = make_signal_bundle(image)
    candidate = (
        build_chroma_candidate(image, recall_cfg)
        | threshold_signal(signals["od"])
        | threshold_signal(signals["hed"])
    )
    base = seeded_image_mask(image, seed_labels, candidate, recall_cfg)
    out = refine_watershed(base, signals["mix"], seed_labels > 0, recall_cfg)
    out = rescue_supported_components(out, candidate, signals["mix"], seed_labels > 0, recall_cfg)
    return cleanup_mask(out, recall_cfg, keep_largest=False)


def method_union_watershed_trim(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base = method_union_watershed(image, seed_labels, cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    out = trim_boundary_by_probability(base, prob, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_watershed_signal_trim(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    signals = make_signal_bundle(image)
    base = method_union_watershed(image, seed_labels, cfg)
    out = trim_boundary_by_signal(base, signals["mix"], seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_mgac(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_union_local(image, seed_labels, cfg)
    out = refine_mgac(base, signals["mix"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_chan_vese(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_union_local(image, seed_labels, cfg)
    out = refine_chan_vese(base, signals["mix"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_union_random_walker(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_union_local(image, seed_labels, cfg)
    out = refine_random_walker(base, signals["mix"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_local(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_od_seeded(image, seed_labels, cfg)
    out = refine_local_threshold(base, signals["od"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_watershed(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_od_seeded(image, seed_labels, cfg)
    out = refine_watershed(base, signals["od"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_mgac(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    signals = make_signal_bundle(image)
    base = method_od_seeded(image, seed_labels, cfg)
    out = refine_mgac(base, signals["od"], seed_labels > 0, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_stats_local(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base_cfg = GrowConfig(**{**config_dict(cfg), "boundary_quantile": 0.60, "boundary_band": min(cfg.boundary_band, 40)})
    base = method_od_local(image, seed_labels, base_cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    out = refine_probability_gate(base, prob, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_stats_rw(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base_cfg = GrowConfig(**{**config_dict(cfg), "boundary_quantile": 0.60, "boundary_band": min(cfg.boundary_band, 40)})
    base = method_od_local(image, seed_labels, base_cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    out = refine_probability_random_walker(base, prob, seed_binary, cfg)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_od_stats_hybrid(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    seed_binary = seed_labels > 0
    base_cfg = GrowConfig(**{**config_dict(cfg), "boundary_quantile": 0.60, "boundary_band": min(cfg.boundary_band, 40)})
    base = method_od_local(image, seed_labels, base_cfg)
    prob = build_tissue_probability(image, base, seed_binary, cfg)
    local = refine_probability_gate(base, prob, seed_binary, cfg)
    rw = refine_probability_random_walker(base, prob, seed_binary, cfg)
    out = (local | rw) & morphology.binary_dilation(base | seed_binary, morphology.disk(max(8, cfg.boundary_band)))
    out[seed_binary] = True
    out = keep_seeded_components(out, seed_binary)
    return cleanup_mask(out, cfg, keep_largest=True)


def method_cell_texture_expand_singlepass(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    labels = grow_by_segment_texture_similarity(image, seed_labels, cfg)
    return finalize_label_output(labels, seed_labels, with_cfg(
        cfg,
        final_hole_area=max(cfg.final_hole_area, cfg.texture_hole_area),
        final_min_obj_area=max(cfg.final_min_obj_area, cfg.texture_min_obj_area),
    ))


def method_dual_transition_refine(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    # Paper-inspired: combine positive segment prototypes and negative background prototypes
    # with transition-like per-pixel probabilities and identity regularization.
    labels = dual_transition_refine_labels(image, seed_labels, cfg)
    return finalize_label_output(labels, seed_labels, with_cfg(
        cfg,
        final_hole_area=max(cfg.final_hole_area, cfg.texture_hole_area),
        final_min_obj_area=max(cfg.final_min_obj_area, cfg.texture_min_obj_area),
    ))


def method_gcdt_coarse_to_fine(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    # Nature 2022-inspired adaptation:
    # rough coarse label + generalized channel-difference threshold candidate selection.
    labels = gcdt_coarse_to_fine_labels(image, seed_labels, cfg)
    return finalize_label_output(labels, seed_labels, with_cfg(
        cfg,
        final_hole_area=max(cfg.final_hole_area, cfg.gcdt_hole_area),
        final_min_obj_area=max(cfg.final_min_obj_area, cfg.gcdt_min_obj_area),
    ))


def method_cell_texture_expand(image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig) -> np.ndarray:
    labels = grow_by_segment_texture_cycles(image, seed_labels, cfg)
    return finalize_label_output(labels, seed_labels, with_cfg(
        cfg,
        final_hole_area=max(cfg.final_hole_area, cfg.texture_hole_area),
        final_min_obj_area=max(cfg.final_min_obj_area, cfg.texture_min_obj_area),
    ))


METHODS: Dict[str, MethodFn] = {
    "seed_only": method_seed_only,
    "chroma_seeded": method_chroma_seeded,
    "od_seeded": method_od_seeded,
    "hed_seeded": method_hed_seeded,
    "union_seeded": method_union_seeded,
    "union_local": method_union_local,
    "union_watershed": method_union_watershed,
    "union_stats_local": method_union_stats_local,
    "union_stats_rw": method_union_stats_rw,
    "union_stats_hybrid": method_union_stats_hybrid,
    "union_stats_rw_adaptive": method_union_stats_rw_adaptive,
    "union_stats_hybrid_adaptive": method_union_stats_hybrid_adaptive,
    "union_superpixel_proto": method_union_superpixel_proto,
    "union_superpixel_proto_adaptive": method_union_superpixel_proto_adaptive,
    "union_prob_mgac": method_union_prob_mgac,
    "union_hybrid_mgac": method_union_hybrid_mgac,
    "union_bg_trim": method_union_bg_trim,
    "rgb_sobel_watershed": method_rgb_sobel_watershed,
    "rgb_scharr_watershed": method_rgb_scharr_watershed,
    "lab_ab_watershed": method_lab_ab_watershed,
    "hed_watershed": method_hed_watershed,
    "union_watershed_recall": method_union_watershed_recall,
    "union_watershed_trim": method_union_watershed_trim,
    "union_watershed_signal_trim": method_union_watershed_signal_trim,
    "union_mgac": method_union_mgac,
    "union_chan_vese": method_union_chan_vese,
    "union_random_walker": method_union_random_walker,
    "od_local": method_od_local,
    "od_watershed": method_od_watershed,
    "od_mgac": method_od_mgac,
    "od_stats_local": method_od_stats_local,
    "od_stats_rw": method_od_stats_rw,
    "od_stats_hybrid": method_od_stats_hybrid,
    "cell_texture_expand_singlepass": method_cell_texture_expand_singlepass,
    "dual_transition_refine": method_dual_transition_refine,
    "gcdt_coarse_to_fine": method_gcdt_coarse_to_fine,
    "cell_texture_expand": method_cell_texture_expand,
}


def grow_labels_within_mask(seed_labels: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    tissue_mask = tissue_mask.astype(bool)
    seed_labels = np.asarray(seed_labels)
    seed_binary = seed_labels > 0
    if not tissue_mask.any() or not seed_binary.any():
        return seed_labels.astype(np.uint16)
    dt, indices = ndi.distance_transform_edt(~seed_binary, return_indices=True)
    nearest = seed_labels[tuple(indices)]
    out = np.where(tissue_mask, nearest, 0)
    out[seed_binary] = seed_labels[seed_binary]
    return out.astype(np.uint16)


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    eroded = morphology.binary_erosion(mask, morphology.disk(1))
    return np.logical_xor(mask, eroded)


def score_binary_masks(pred: np.ndarray, gt: np.ndarray, tolerance: int = 3) -> Dict[str, float]:
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

    return {
        "dice": dice,
        "iou": iou,
        "boundary_f1": boundary_f1,
        "pred_pixels": pred_sum,
        "gt_pixels": gt_sum,
        "intersection_pixels": inter,
    }


def run_method(name: str, image: np.ndarray, seed_labels: np.ndarray, cfg: GrowConfig = DEFAULT_CONFIG) -> Tuple[np.ndarray, np.ndarray, float]:
    if name not in METHODS:
        raise KeyError(f"Unknown method: {name}")
    start = time.perf_counter()
    output = METHODS[name](image, seed_labels, cfg)
    tissue_mask, labels = normalize_method_output(output, seed_labels)
    elapsed = time.perf_counter() - start
    return tissue_mask.astype(bool), labels.astype(np.uint16), elapsed


def write_mask_tiff(path: str | Path, labels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        np.asarray(labels),
        compression="deflate",
        ome=path.suffix.lower().endswith(".tif"),
        photometric="minisblack",
    )


def _overlay_boundaries(base_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None, seed: np.ndarray | None = None) -> np.ndarray:
    rgb = (to_float_rgb(base_rgb) * 255.0).astype(np.uint8).copy()
    if gt is not None:
        gt_b = compute_boundary(gt)
        rgb[gt_b] = np.array([0, 255, 0], dtype=np.uint8)
    pred_b = compute_boundary(pred)
    rgb[pred_b] = np.array([255, 0, 0], dtype=np.uint8)
    if gt is not None:
        both = pred_b & compute_boundary(gt)
        rgb[both] = np.array([255, 255, 0], dtype=np.uint8)
    if seed is not None:
        seed_b = compute_boundary(seed > 0)
        rgb[seed_b] = np.array([0, 255, 255], dtype=np.uint8)
    return rgb


def _difference_panel(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    panel = np.zeros((*pred.shape, 3), dtype=np.uint8)
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    panel[tp] = np.array([220, 220, 220], dtype=np.uint8)
    panel[fp] = np.array([255, 0, 0], dtype=np.uint8)
    panel[fn] = np.array([0, 80, 255], dtype=np.uint8)
    return panel


def save_overlay(path: str | Path, image: np.ndarray, pred: np.ndarray, gt: np.ndarray, seed_labels: np.ndarray | None = None, title: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb = (to_float_rgb(image) * 255.0).astype(np.uint8)
    pred_panel = _overlay_boundaries(rgb, pred, None, seed_labels)
    gt_panel = _overlay_boundaries(rgb, gt, None, seed_labels)
    combo_panel = _overlay_boundaries(rgb, pred, gt, seed_labels)
    diff_panel = _difference_panel(pred, gt)

    panels = [rgb, pred_panel, gt_panel, combo_panel, diff_panel]
    labels = ["image", "pred", "gt", "overlay", "diff"]
    pil_panels = [Image.fromarray(p) for p in panels]
    max_h = max(p.height for p in pil_panels)
    widths = [p.width for p in pil_panels]
    caption_h = 24
    canvas = Image.new("RGB", (sum(widths), max_h + caption_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel, label in zip(pil_panels, labels):
        canvas.paste(panel, (x, caption_h))
        draw.text((x + 6, 4), label, fill=(0, 0, 0))
        x += panel.width
    if title:
        draw.text((6, max_h + 4), title, fill=(0, 0, 0))
    canvas.save(path)


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby("method", as_index=False)
        .agg(
            min_dice=("dice", "min"),
            mean_dice=("dice", "mean"),
            mean_iou=("iou", "mean"),
            mean_boundary_f1=("boundary_f1", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
            samples=("sample", "count"),
        )
    )
    grouped = grouped.sort_values(
        ["min_dice", "mean_dice", "mean_iou", "mean_boundary_f1"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
    return grouped


def config_dict(cfg: GrowConfig = DEFAULT_CONFIG) -> Dict[str, object]:
    return asdict(cfg)
