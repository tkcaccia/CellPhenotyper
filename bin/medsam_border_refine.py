from __future__ import annotations

import inspect
import os
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from grow_to_tissue_core import keep_seeded_components, to_float_rgb


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REMOVE_SMALL_HOLES_HAS_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_holes).parameters
_REMOVE_SMALL_OBJECTS_HAS_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_objects).parameters
DEFAULT_MEDSAM_REPO = Path(os.environ.get("MEDSAM_REPO_DIR", str(PROJECT_ROOT / "third_party" / "MedSAM")))
DEFAULT_MEDSAM_CHECKPOINT = Path(
    os.environ.get(
        "MEDSAM_CHECKPOINT",
        str(DEFAULT_MEDSAM_REPO / "work_dir" / "MedSAM" / "medsam_vit_b.pth"),
    )
)
LARGE_MASK_MAX_PIXELS = int(os.environ.get("MEDSAM_LARGE_MASK_MAX_PIXELS", "8000000"))


def _log(message: str) -> None:
    print(f"[MedSAM] {message}", flush=True)


class MedSAMUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MedSAMConfig:
    checkpoint: str = str(DEFAULT_MEDSAM_CHECKPOINT)
    device: str = "cuda"
    bbox_margin: int = 144
    component_min_area: int = 200
    component_merge_distance: int = 24
    seed_dilation_radius: int = 8
    core_erosion_radius: int = 43
    outer_dilation_radius: int = 53
    min_object_size: int = 5000
    smooth_radius: int = 5
    force_core_preservation: bool = True
    save_debug: bool = False
    repo_dir: str = str(DEFAULT_MEDSAM_REPO)
    cluster_tile_size: int = 4096
    cluster_tile_overlap: int = 512


def _disk(radius: int):
    radius = max(0, int(radius))
    return morphology.disk(radius) if radius > 0 else None


def _large_mask_step(shape: Tuple[int, int], max_pixels: int = LARGE_MASK_MAX_PIXELS) -> int:
    pixels = int(shape[0]) * int(shape[1])
    if pixels <= max_pixels:
        return 1
    return max(2, int(np.ceil(np.sqrt(float(pixels) / float(max_pixels)))))


def _downsample_bool_any(mask: np.ndarray, step: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool, copy=False)
    step = max(1, int(step))
    if step <= 1:
        return mask
    h, w = mask.shape
    out_h = int(np.ceil(h / float(step)))
    out_w = int(np.ceil(w / float(step)))
    pad_h = out_h * step - h
    pad_w = out_w * step - w
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    return mask.reshape(out_h, step, out_w, step).max(axis=(1, 3))


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool, copy=False)
    selem = _disk(radius)
    if selem is None:
        return mask
    step = _large_mask_step(mask.shape)
    if step > 1:
        small = _downsample_bool_any(mask, step)
        small_radius = max(1, int(np.ceil(float(radius) / float(step))))
        small_selem = _disk(small_radius)
        small_out = ndi.binary_dilation(small, structure=small_selem)
        return _upsample_bool_mask(small_out, mask.shape)
    return ndi.binary_dilation(mask, structure=selem)


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool, copy=False)
    selem = _disk(radius)
    if selem is None:
        return mask
    step = _large_mask_step(mask.shape)
    if step > 1:
        small = _downsample_bool_any(mask, step)
        small_radius = max(1, int(np.ceil(float(radius) / float(step))))
        small_selem = _disk(small_radius)
        small_out = ndi.binary_erosion(small, structure=small_selem)
        return _upsample_bool_mask(small_out, mask.shape) & mask
    return ndi.binary_erosion(mask, structure=selem)


def _to_uint8_rgb_for_resize(crop: np.ndarray) -> np.ndarray:
    image = np.asarray(crop)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape for MedSAM crop: {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if np.issubdtype(image.dtype, np.integer):
        maxv = float(np.iinfo(image.dtype).max or 255)
        scaled = np.clip(image.astype(np.float32, copy=False) / maxv, 0.0, 1.0)
        return np.ascontiguousarray(np.round(scaled * 255.0).astype(np.uint8))
    arr = image.astype(np.float32, copy=False)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))


def _bbox_from_mask(mask: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int]:
    mask = np.asarray(mask).astype(bool, copy=False)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, 0, mask.shape[0], mask.shape[1]
    y0 = max(0, int(rows[0]) - int(margin))
    y1 = min(mask.shape[0], int(rows[-1]) + 1 + int(margin))
    x0 = max(0, int(cols[0]) - int(margin))
    x1 = min(mask.shape[1], int(cols[-1]) + 1 + int(margin))
    return y0, y1, x0, x1


def _tile_starts(start: int, end: int, size: int, overlap: int) -> List[int]:
    start = int(start)
    end = int(end)
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    if end <= start:
        return []
    if end - start <= size:
        return [max(0, end - size)]
    stride = max(1, size - overlap)
    starts = list(range(start, end, stride))
    last = max(start, end - size)
    starts = [s for s in starts if s <= last]
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(max(0, s) for s in starts))


def _iter_mask_tiles(mask: np.ndarray, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    mask = np.asarray(mask).astype(bool, copy=False)
    if not mask.any():
        return []
    y0, y1, x0, x1 = _bbox_from_mask(mask, margin=0)
    ys = _tile_starts(y0, y1, tile_size, overlap)
    xs = _tile_starts(x0, x1, tile_size, overlap)
    h, w = mask.shape
    windows: List[Tuple[int, int, int, int]] = []
    for yy0 in ys:
        yy1 = min(h, yy0 + max(1, int(tile_size)))
        for xx0 in xs:
            xx1 = min(w, xx0 + max(1, int(tile_size)))
            if mask[yy0:yy1, xx0:xx1].any():
                windows.append((yy0, yy1, xx0, xx1))
    return windows


def _normalize_crop(crop: np.ndarray) -> np.ndarray:
    # Resize before converting to float32. On WSI crops, converting the full
    # crop to float first can allocate multiple GB and stall before CUDA is used.
    from PIL import Image

    rgb_u8 = _to_uint8_rgb_for_resize(crop)
    pil = Image.fromarray(rgb_u8, mode="RGB")
    if pil.size != (1024, 1024):
        pil = pil.resize((1024, 1024), Image.Resampling.BILINEAR)
    img_1024 = np.asarray(pil, dtype=np.float32) / 255.0
    img_1024 -= float(img_1024.min(initial=0.0))
    denom = float(img_1024.max(initial=0.0))
    if denom > 0:
        img_1024 /= denom
    return np.clip(img_1024, 0.0, 1.0)


@lru_cache(maxsize=4)
def _load_medsam_model(checkpoint: str, device: str, repo_dir: str):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise MedSAMUnavailableError(f"Checkpoint not found: {checkpoint_path}")

    repo_path = Path(repo_dir)
    if not repo_path.is_dir():
        raise MedSAMUnavailableError(f"MedSAM repo not found: {repo_path}")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    try:
        import torch
    except Exception as exc:  # pragma: no cover - env-specific
        raise MedSAMUnavailableError(f"PyTorch unavailable: {exc}") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise MedSAMUnavailableError("CUDA device requested but torch.cuda.is_available() is False")

    try:
        from segment_anything import sam_model_registry
    except Exception as exc:  # pragma: no cover - env-specific
        raise MedSAMUnavailableError(f"segment_anything import failed from MedSAM repo: {exc}") from exc

    _log(f"loading model on device={device} checkpoint={checkpoint_path}")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
    model = model.to(device)
    model.eval()
    return model


def _infer_box_prompt(crop_rgb: np.ndarray, box_xyxy: np.ndarray, config: MedSAMConfig) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model = _load_medsam_model(config.checkpoint, config.device, config.repo_dir)
    img_1024 = _normalize_crop(crop_rgb)
    tensor = torch.tensor(img_1024, dtype=torch.float32, device=config.device).permute(2, 0, 1).unsqueeze(0)

    with torch.inference_mode():
        image_embedding = model.image_encoder(tensor)
        scale = np.array(
            [1024.0 / crop_rgb.shape[1], 1024.0 / crop_rgb.shape[0], 1024.0 / crop_rgb.shape[1], 1024.0 / crop_rgb.shape[0]],
            dtype=np.float32,
        )
        box_1024 = (box_xyxy.astype(np.float32) * scale)[None, :]
        box_torch = torch.as_tensor(box_1024, dtype=torch.float32, device=config.device)[:, None, :]
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)
        low_res_logits, _ = model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        prob = torch.sigmoid(low_res_logits)
        prob = torch.nn.functional.interpolate(
            prob,
            size=crop_rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        prob_np = prob.squeeze().detach().cpu().numpy().astype(np.float32)
    pred_mask = prob_np >= 0.5
    return pred_mask, prob_np


def _upsample_bool_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).astype(bool, copy=False)
    if mask.shape == shape:
        return mask
    sy = max(1, int(np.ceil(shape[0] / float(mask.shape[0]))))
    sx = max(1, int(np.ceil(shape[1] / float(mask.shape[1]))))
    up = np.repeat(np.repeat(mask, sy, axis=0), sx, axis=1)
    return up[: shape[0], : shape[1]]


def _obvious_background_mask(crop_rgb: np.ndarray, max_side: int = 4096) -> np.ndarray:
    """
    Detect clearly white background without materializing several full-resolution
    float images. The mask only needs to be conservative, so we estimate it on a
    downsampled view and expand back to level-0.
    """
    image = np.asarray(crop_rgb)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 3:
        image = image[..., :3]
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape for background detection: {image.shape}")

    h, w = int(image.shape[0]), int(image.shape[1])
    step = max(1, int(np.ceil(max(h, w) / float(max_side))))
    work = image[::step, ::step]
    rgb = to_float_rgb(work)

    vmax = rgb.max(axis=-1)
    vmin = rgb.min(axis=-1)
    gray = rgb.mean(axis=-1)
    saturation = (vmax - vmin) / np.maximum(vmax, 1e-3)
    mask_small = (gray > 0.93) & (saturation < 0.10) & (vmax > 0.92)
    return _upsample_bool_mask(mask_small, (h, w))


def _build_component_groups(seed_binary: np.ndarray, baseline_mask: np.ndarray, config: MedSAMConfig) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    if int(baseline_mask.size) > int(LARGE_MASK_MAX_PIXELS) * 4:
        baseline_bool = baseline_mask.astype(bool, copy=False)
        seed_bool = seed_binary.astype(bool, copy=False)
        seed_area = int((seed_bool & baseline_bool).sum())
        labels = np.zeros(baseline_bool.shape, dtype=np.int32)
        if seed_area >= int(config.component_min_area) and baseline_bool.any():
            labels[baseline_bool] = 1
            return labels, [{
                "group_id": 1,
                "seed_area": seed_area,
                "baseline_component_pixels": int(baseline_bool.sum()),
            }]
        return labels, []

    baseline_groups = measure.label(_binary_dilate(baseline_mask.astype(bool), config.component_merge_distance), connectivity=2)
    labels = np.zeros_like(baseline_groups, dtype=np.int32)
    groups: List[Dict[str, int]] = []
    next_gid = 1
    for prop in measure.regionprops(baseline_groups):
        group_mask = baseline_groups == prop.label
        group_seed = seed_binary & group_mask
        group_area = int(group_seed.sum())
        if group_area < int(config.component_min_area):
            continue
        labels[group_mask] = next_gid
        groups.append({
            "group_id": int(next_gid),
            "seed_area": group_area,
            "baseline_component_pixels": int((baseline_mask & group_mask).sum()),
        })
        next_gid += 1
    if groups:
        return labels, groups

    support = _binary_dilate(seed_binary, config.seed_dilation_radius)
    merged = _binary_dilate(support, config.component_merge_distance)
    fallback = measure.label(merged, connectivity=2)
    groups = []
    labels = np.zeros_like(fallback, dtype=np.int32)
    next_gid = 1
    for prop in measure.regionprops(fallback):
        group_mask = fallback == prop.label
        group_seed = seed_binary & group_mask
        group_area = int(group_seed.sum())
        if group_area < int(config.component_min_area):
            continue
        labels[group_mask] = next_gid
        groups.append({"group_id": int(next_gid), "seed_area": group_area, "baseline_component_pixels": 0})
        next_gid += 1
    return labels, groups




def _build_protected_core(baseline: np.ndarray, seed_binary: np.ndarray, radius: int) -> np.ndarray:
    support = baseline.astype(bool)
    seed_binary = seed_binary.astype(bool)
    radius = max(0, int(radius))
    if radius == 0:
        core = support.copy()
    else:
        step = _large_mask_step(support.shape)
        if step > 1:
            small_support = _downsample_bool_any(support, step)
            small_radius = max(1, int(np.ceil(float(radius) / float(step))))
            dist = ndi.distance_transform_edt(small_support)
            core = _upsample_bool_mask(dist > float(small_radius), support.shape) & support
        else:
            # Use an exact interior-distance core on small masks. Large WSI masks
            # use the downsampled path above to avoid multi-GB float64 EDT arrays.
            dist = ndi.distance_transform_edt(support)
            core = dist > float(radius)
        if not core.any():
            core = seed_binary.copy()
    core |= seed_binary
    core &= support | seed_binary
    return core.astype(bool)


def _nearest_seed_labels(seed_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _nearest_existing_labels(seed_labels, mask)


def _nearest_existing_labels(label_map: np.ndarray, target_mask: np.ndarray | None = None) -> np.ndarray:
    labels = np.asarray(label_map)
    out = np.zeros(labels.shape, dtype=labels.dtype)
    fg = labels > 0
    if not fg.any():
        return out
    if target_mask is None:
        target_mask = np.ones(labels.shape, dtype=bool)
    else:
        target_mask = np.asarray(target_mask).astype(bool)
    if not target_mask.any():
        return out
    cc = measure.label(target_mask, connectivity=2)
    for prop in measure.regionprops(cc):
        y0, x0, y1, x1 = prop.bbox
        pad = 32
        while True:
            yy0 = max(0, y0 - pad)
            xx0 = max(0, x0 - pad)
            yy1 = min(labels.shape[0], y1 + pad)
            xx1 = min(labels.shape[1], x1 + pad)
            sub_labels = labels[yy0:yy1, xx0:xx1]
            sub_target = cc[yy0:yy1, xx0:xx1] == prop.label
            fg_sub = sub_labels > 0
            if fg_sub.any() or (yy0 == 0 and xx0 == 0 and yy1 == labels.shape[0] and xx1 == labels.shape[1]):
                break
            pad *= 2
        if not fg_sub.any():
            continue
        _, indices = ndi.distance_transform_edt(~fg_sub, return_indices=True)
        nearest = sub_labels[tuple(indices)]
        sub_out = out[yy0:yy1, xx0:xx1]
        sub_out[sub_target] = nearest[sub_target].astype(labels.dtype, copy=False)
        out[yy0:yy1, xx0:xx1] = sub_out
    out[fg & target_mask] = labels[fg & target_mask]
    return out


def _remove_isolated_label_components(label_map: np.ndarray, seed_labels: np.ndarray, min_area: int = 0) -> np.ndarray:
    labels = np.asarray(label_map)
    seeds = np.asarray(seed_labels)
    out = np.zeros_like(labels)
    for lid in [int(x) for x in np.unique(labels) if x > 0]:
        mask = labels == lid
        cc = measure.label(mask, connectivity=2)
        if cc.max() == 0:
            continue
        keep_ids = {int(x) for x in np.unique(cc[seeds == lid]) if x > 0}
        if not keep_ids:
            counts = np.bincount(cc.ravel())
            if counts.size > 1:
                keep_ids = {int(np.argmax(counts[1:]) + 1)}
        keep_mask = np.isin(cc, list(keep_ids))
        if int(min_area) > 0:
            filtered_keep = np.zeros_like(keep_mask, dtype=bool)
            counts = np.bincount(cc.ravel())
            for cid in keep_ids:
                cid = int(cid)
                cid_mask = cc == cid
                has_seed = np.any((seeds == lid) & cid_mask)
                if has_seed or (cid < counts.size and counts[cid] >= int(min_area)):
                    filtered_keep |= cid_mask
            keep_mask = filtered_keep
        out[keep_mask] = lid
    return out


def _apply_white_background_exclusion(label_map: np.ndarray, background_mask: np.ndarray, seed_labels: np.ndarray, protected_core_labels: np.ndarray) -> np.ndarray:
    out = np.asarray(label_map).copy()
    forbidden = np.asarray(background_mask).astype(bool)
    forbidden &= np.asarray(seed_labels) <= 0
    forbidden &= np.asarray(protected_core_labels) <= 0
    out[forbidden] = 0
    return out


def _fill_label_holes(label_map: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map)
    filled = ndi.binary_fill_holes(labels > 0)
    new_pixels = filled & ~(labels > 0)
    if not new_pixels.any():
        return labels.copy()
    out = labels.copy()
    nearest = _nearest_existing_labels(labels, new_pixels)
    out[new_pixels] = nearest[new_pixels]
    out[seed_labels > 0] = seed_labels[seed_labels > 0]
    return out


def _smooth_union_border(mask: np.ndarray, seed_binary: np.ndarray, protected_core: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask).astype(bool)
    radius = max(0, int(radius) + 1)
    if radius > 0:
        selem = morphology.disk(radius)
        out = morphology.opening(out, selem)
        out = morphology.closing(out, selem)
        out = ndi.binary_fill_holes(out)
    out |= seed_binary.astype(bool)
    out |= protected_core.astype(bool)
    out = keep_seeded_components(out, seed_binary.astype(bool))
    return out.astype(bool)


def _prune_thin_structures(mask: np.ndarray, seed_binary: np.ndarray, protected_core: np.ndarray, radius: int) -> np.ndarray:
    out = np.asarray(mask).astype(bool)
    radius = max(0, int(radius))
    if radius > 0:
        opened = morphology.opening(out, morphology.disk(radius))
        opened |= seed_binary.astype(bool)
        opened |= protected_core.astype(bool)
        opened = ndi.binary_fill_holes(opened)
        out = keep_seeded_components(opened, seed_binary.astype(bool))
    out |= seed_binary.astype(bool)
    out |= protected_core.astype(bool)
    return out.astype(bool)


def _conservative_cleanup(mask: np.ndarray, baseline: np.ndarray, seed_binary: np.ndarray, protected_core: np.ndarray, editable_band: np.ndarray, outer_envelope: np.ndarray, background_mask: np.ndarray, config: MedSAMConfig) -> np.ndarray:
    out = mask.astype(bool)
    if int(config.smooth_radius) > 0:
        out = morphology.closing(out, morphology.disk(int(config.smooth_radius)))
    hole_threshold = max(int(config.min_object_size), 256)
    if _REMOVE_SMALL_HOLES_HAS_MAX_SIZE:
        # New API removes holes <= max_size; threshold - 1 preserves the old < threshold behavior.
        out = morphology.remove_small_holes(out, max_size=hole_threshold - 1)
    else:
        out = morphology.remove_small_holes(out, area_threshold=hole_threshold)
    if int(config.min_object_size) > 0:
        object_threshold = int(config.min_object_size)
        if _REMOVE_SMALL_OBJECTS_HAS_MAX_SIZE:
            out = morphology.remove_small_objects(out, max_size=object_threshold - 1)
        else:
            out = morphology.remove_small_objects(out, min_size=object_threshold)
    out &= outer_envelope
    out |= seed_binary
    if bool(config.force_core_preservation):
        out |= protected_core
    out = keep_seeded_components(out, seed_binary)
    out[~editable_band] = baseline[~editable_band]
    out &= outer_envelope | baseline
    out |= seed_binary
    if bool(config.force_core_preservation):
        out |= protected_core
    out = ndi.binary_fill_holes(out)
    out[np.asarray(background_mask).astype(bool) & ~seed_binary & ~protected_core] = False
    out = keep_seeded_components(out, seed_binary)
    out |= seed_binary
    if bool(config.force_core_preservation):
        out |= protected_core
    return out.astype(bool)


def run_medsam_border_refine(
    image: np.ndarray,
    seed_labels: np.ndarray,
    baseline_tissue_mask: np.ndarray,
    config: MedSAMConfig,
    baseline_label_map: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, float, Dict[str, object], Dict[str, np.ndarray]]:
    start = time.perf_counter()
    seed_labels = np.asarray(seed_labels)
    _log(f"start image_shape={getattr(image, 'shape', None)} seed_shape={seed_labels.shape} baseline_shape={np.asarray(baseline_tissue_mask).shape} device={config.device}")
    label_dtype = seed_labels.dtype
    if np.issubdtype(label_dtype, np.integer):
        min_label = int(seed_labels.min(initial=0))
        max_label = int(seed_labels.max(initial=0))
        if min_label >= 0 and max_label <= np.iinfo(np.uint16).max:
            label_dtype = np.uint16
    if seed_labels.dtype != label_dtype:
        seed_labels = seed_labels.astype(label_dtype, copy=False)
    seed_binary = seed_labels > 0
    baseline = np.asarray(baseline_tissue_mask).astype(bool)
    if baseline.shape != seed_binary.shape:
        raise ValueError(f"Baseline mask and seed mask shape mismatch: {baseline.shape} vs {seed_binary.shape}")

    if baseline_label_map is not None:
        baseline_labels = np.asarray(baseline_label_map).astype(label_dtype, copy=False)
        if baseline_labels.shape != seed_labels.shape:
            raise ValueError(
                f"Baseline label map and seed mask shape mismatch: {baseline_labels.shape} vs {seed_labels.shape}"
            )
        baseline_labels = baseline_labels.copy()
        baseline_labels[~baseline] = 0
        unlabeled_baseline = baseline & (baseline_labels <= 0)
        if unlabeled_baseline.any():
            fallback_labels = _nearest_seed_labels(seed_labels, unlabeled_baseline)
            baseline_labels[unlabeled_baseline] = fallback_labels[unlabeled_baseline]
    else:
        baseline_labels = _nearest_seed_labels(seed_labels, baseline)
    final_labels = baseline_labels.copy()
    score_dtype = np.float16
    best_score = np.full(baseline.shape, -np.inf, dtype=score_dtype)
    best_score[baseline] = 0.25
    probability_map = np.zeros(baseline.shape, dtype=score_dtype) if config.save_debug else None
    background_mask = _obvious_background_mask(image)
    region_meta: List[Dict[str, object]] = []

    protected_core = np.zeros(baseline.shape, dtype=bool)
    editable_band = np.zeros(baseline.shape, dtype=bool)
    outer_envelope = np.zeros(baseline.shape, dtype=bool)
    protected_core_labels = np.zeros(seed_labels.shape, dtype=label_dtype)

    unique_labels = [int(x) for x in np.unique(seed_labels) if x > 0]
    _log(f"labels={unique_labels} large_mask_step={_large_mask_step(seed_labels.shape)}")

    for lid in unique_labels:
        label_start = time.perf_counter()
        _log(f"cluster {lid}: preparing border tiles")
        label_seed = seed_labels == lid
        label_baseline = baseline_labels == lid
        if not label_baseline.any():
            label_baseline = label_seed.copy()
        label_support = label_baseline | label_seed
        if int(label_seed.sum()) < int(config.component_min_area):
            _log(f"cluster {lid}: skipped tiny seed area={int(label_seed.sum())}")
            continue

        label_core = _build_protected_core(label_support, label_seed, int(config.core_erosion_radius))
        label_outer = _binary_dilate(label_support, int(config.outer_dilation_radius))
        label_outer |= label_support
        label_band = label_outer & ~label_core

        protected_core |= label_core
        editable_band |= label_band
        outer_envelope |= label_outer
        protected_core_labels[label_core] = lid
        final_labels[label_core] = lid
        best_score[label_core] = 2.0

        tile_mask = label_band | label_seed
        windows = _iter_mask_tiles(
            tile_mask,
            tile_size=int(config.cluster_tile_size),
            overlap=int(config.cluster_tile_overlap),
        )
        _log(
            f"cluster {lid}: seed_px={int(label_seed.sum())} baseline_px={int(label_baseline.sum())} "
            f"tile_count={len(windows)} prep_sec={time.perf_counter() - label_start:.1f}"
        )

        for tile_idx, (y0, y1, x0, x1) in enumerate(windows, start=1):
            tile_start = time.perf_counter()
            crop_rgb = image[y0:y1, x0:x1]
            crop_seed = label_seed[y0:y1, x0:x1]
            crop_support = label_support[y0:y1, x0:x1]
            crop_core = label_core[y0:y1, x0:x1]
            crop_outer = label_outer[y0:y1, x0:x1]
            crop_band = label_band[y0:y1, x0:x1]
            if crop_rgb.size == 0 or not crop_band.any():
                continue

            prompt_source = crop_seed if crop_seed.any() else crop_support
            prompt_region = _binary_dilate(
                prompt_source,
                max(1, int(config.seed_dilation_radius) + int(config.component_merge_distance)),
            )
            prompt_region &= crop_outer | crop_support
            if not prompt_region.any():
                prompt_region = crop_support | crop_seed
            if not prompt_region.any():
                continue

            py0, py1, px0, px1 = _bbox_from_mask(prompt_region, margin=max(2, int(config.seed_dilation_radius)))
            box_xyxy = np.array([px0, py0, px1, py1], dtype=np.float32)

            _log(
                f"cluster {lid} tile {tile_idx}/{len(windows)}: "
                f"crop={crop_rgb.shape[:2]} box={[int(px0), int(py0), int(px1), int(py1)]}"
            )
            pred_mask, pred_prob = _infer_box_prompt(crop_rgb, box_xyxy, config)
            _log(f"cluster {lid} tile {tile_idx}/{len(windows)}: inference done")
            pred_mask = np.asarray(pred_mask, dtype=bool)
            pred_prob = np.asarray(pred_prob, dtype=np.float32)
            crop_bg = background_mask[y0:y1, x0:x1]
            pred_mask &= crop_outer
            pred_mask[crop_bg & ~crop_support] = False

            candidate_mask = pred_mask & crop_band
            sub_scores = best_score[y0:y1, x0:x1]
            sub_labels = final_labels[y0:y1, x0:x1]
            update = candidate_mask & (pred_prob > sub_scores)
            sub_labels[update] = lid
            sub_scores[update] = pred_prob[update]
            sub_labels[crop_core] = lid
            sub_labels[crop_seed] = lid
            sub_scores[crop_core] = 2.0
            sub_scores[crop_seed] = 2.0
            final_labels[y0:y1, x0:x1] = sub_labels
            best_score[y0:y1, x0:x1] = sub_scores
            if probability_map is not None:
                probability_map[y0:y1, x0:x1] = np.maximum(
                    probability_map[y0:y1, x0:x1],
                    (pred_prob * crop_band.astype(np.float32)).astype(score_dtype, copy=False),
                )
            region_meta.append(
                {
                    "label_id": int(lid),
                    "tile_index": int(tile_idx),
                    "tile_count": int(len(windows)),
                    "seed_area": int(crop_seed.sum()),
                    "crop_box": [int(x0), int(y0), int(x1), int(y1)],
                    "prompt_box": [int(px0), int(py0), int(px1), int(py1)],
                    "editable_pixels": int(crop_band.sum()),
                    "predicted_pixels": int(pred_mask.sum()),
                    "runtime_sec": round(float(time.perf_counter() - tile_start), 3),
                }
            )
        _log(f"cluster {lid}: done total_sec={time.perf_counter() - label_start:.1f}")

    raw_medsam_labels = final_labels.copy()

    final_mask = _conservative_cleanup(final_labels > 0, baseline, seed_binary, protected_core, editable_band, outer_envelope, background_mask, config)
    final_labels[~final_mask] = 0
    final_labels[seed_binary] = seed_labels[seed_binary]
    final_labels[protected_core_labels > 0] = protected_core_labels[protected_core_labels > 0]

    strict_min_area = max(int(config.min_object_size) + 2000, int(round(int(config.min_object_size) * 1.35)))
    final_labels = _apply_white_background_exclusion(final_labels, background_mask, seed_labels, protected_core_labels)
    final_labels = _remove_isolated_label_components(final_labels, seed_labels, min_area=strict_min_area)
    final_labels = _fill_label_holes(final_labels, seed_labels)
    final_labels = _apply_white_background_exclusion(final_labels, background_mask, seed_labels, protected_core_labels)
    final_labels = _remove_isolated_label_components(final_labels, seed_labels, min_area=strict_min_area)
    final_labels[seed_binary] = seed_labels[seed_binary]
    final_labels[protected_core_labels > 0] = protected_core_labels[protected_core_labels > 0]

    final_mask = _smooth_union_border(final_labels > 0, seed_binary, protected_core, max(2, int(config.smooth_radius)))
    final_mask = _prune_thin_structures(final_mask, seed_binary, protected_core, max(3, int(config.smooth_radius) + 2))
    final_mask[np.asarray(background_mask).astype(bool) & ~seed_binary & ~protected_core] = False
    final_mask = ndi.binary_fill_holes(final_mask)
    final_mask |= seed_binary
    final_mask |= protected_core
    final_labels[~final_mask] = 0

    unlabeled = final_mask & (final_labels == 0)
    if unlabeled.any():
        refill = _nearest_existing_labels(final_labels, unlabeled)
        final_labels[unlabeled] = refill[unlabeled]

    final_labels[seed_binary] = seed_labels[seed_binary]
    final_labels[protected_core_labels > 0] = protected_core_labels[protected_core_labels > 0]
    final_labels = _apply_white_background_exclusion(final_labels, background_mask, seed_labels, protected_core_labels)
    final_labels = _remove_isolated_label_components(final_labels, seed_labels, min_area=strict_min_area)
    final_labels = _fill_label_holes(final_labels, seed_labels)
    final_labels = _apply_white_background_exclusion(final_labels, background_mask, seed_labels, protected_core_labels)
    final_labels[seed_binary] = seed_labels[seed_binary]
    final_labels[protected_core_labels > 0] = protected_core_labels[protected_core_labels > 0]

    final_mask = final_labels > 0
    runtime = time.perf_counter() - start
    meta: Dict[str, object] = {
        "device": config.device,
        "checkpoint": str(config.checkpoint),
        "repo_dir": str(config.repo_dir),
        "label_count": len(unique_labels),
        "processed_group_count": len(region_meta),
        "baseline_pixels": int(baseline.sum()),
        "final_pixels": int(final_mask.sum()),
        "protected_core_pixels": int(protected_core.sum()),
        "editable_band_pixels": int(editable_band.sum()),
        "runtime_sec": float(runtime),
        "regions": region_meta,
    }
    artifacts = {
        "protected_core": protected_core.astype(np.uint8),
        "editable_band": editable_band.astype(np.uint8),
        "raw_medsam_label_map": raw_medsam_labels.astype(label_dtype, copy=False),
        "label_map": final_labels.astype(label_dtype, copy=False),
    }
    if config.save_debug:
        artifacts["baseline_mask"] = baseline.astype(np.uint8)
        artifacts["baseline_labels"] = baseline_labels.astype(label_dtype, copy=False)
        artifacts["protected_core_labels"] = protected_core_labels.astype(label_dtype, copy=False)
        artifacts["outer_envelope"] = outer_envelope.astype(np.uint8)
        artifacts["final_mask"] = final_mask.astype(np.uint8)
        artifacts["probability_map"] = probability_map.astype(np.float16, copy=False) if probability_map is not None else np.zeros((1,), dtype=np.float16)
    prob_out = probability_map.astype(np.float16, copy=False) if probability_map is not None else None
    return final_mask.astype(bool), prob_out, float(runtime), meta, artifacts
