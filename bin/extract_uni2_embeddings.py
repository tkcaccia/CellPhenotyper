#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WSI-friendly CellSAM mask -> nucleus-centered tiles -> encoder embeddings (UNI/UNI2-h etc)

FIX in this version:
- Use ONE global intensity scaling for the whole image (per-channel p1..p99 by sampling)
  instead of the broken "clip to 0..255 then cast" conversion in the old to_rgb_uint8().
  That old conversion is here: to_rgb_uint8() clips then casts.  (See your file lines 58-69)
- Optional: --force-full-image loads the image into one numpy array to guarantee that every
  nucleus tile is extracted from the exact same image array (no backend differences).
"""

import argparse
import os
import sys
import math
import random
import shutil
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import tifffile
import torchvision.transforms as T


# --------------------------
# Encoder utilities (kept compatible with your current script)
# --------------------------
def build_dinov2_transform(img_size: int = 224):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])


def pool_tokens(x: torch.Tensor, mode: str) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() != 3:
        raise ValueError(f"Unexpected embedding tensor shape: {tuple(x.shape)}")

    cls = x[:, 0]
    if x.size(1) > 1:
        patch = x[:, 1:]
        mean_patch = patch.mean(1)
    else:
        mean_patch = cls

    if mode == "cls":
        return cls
    if mode == "mean_patch":
        return mean_patch
    if mode == "mean_all":
        return x.mean(1)
    if mode == "cls_mean_concat":
        return torch.cat([cls, mean_patch], dim=-1)
    raise ValueError(f"Unknown pooling mode: {mode}")


def _maybe_hf_login(token: Optional[str]):
    if token:
        try:
            from huggingface_hub import login
            login(token=token)
        except Exception as e:
            print(f"[WARN] Could not login to HuggingFace Hub: {e}", file=sys.stderr)


def _clear_hf_repo_cache(repo_id: str) -> None:
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        hub_root = Path(hf_hub_cache)
    else:
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        hub_root = Path(hf_home) / "hub"

    repo_dir = hub_root / f"models--{repo_id.replace('/', '--')}"
    if repo_dir.exists():
        print(f"[WARN] Clearing corrupted HF cache: {repo_dir}", file=sys.stderr)
        shutil.rmtree(repo_dir, ignore_errors=True)

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        xet_dir = Path(hf_home) / "xet"
        if xet_dir.exists():
            print(f"[WARN] Clearing HF xet cache: {xet_dir}", file=sys.stderr)
            shutil.rmtree(xet_dir, ignore_errors=True)


def _load_timm_hf_legacy_checkpoint(model: Any, repo_id: str) -> Any:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = None
        for key in ("state_dict", "model", "model_state_dict"):
            maybe = checkpoint.get(key)
            if isinstance(maybe, dict):
                state_dict = maybe
                break
        if state_dict is None and all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("Unsupported checkpoint format for legacy timm HF load")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Legacy load missing keys: {len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] Legacy load unexpected keys: {len(unexpected)}", file=sys.stderr)
    return model


def load_encoder(encoder: str, backend: str, pooling: str, img_size: int, hf_token: Optional[str]):
    presets: Dict[str, Dict[str, Any]] = {
        "dinov2_vitb14": {"backend": "dinov2_hub", "id": "dinov2_vitb14", "pooling": "cls", "tag": "dinov2_vitb14"},
        "dinov2_vitl14": {"backend": "dinov2_hub", "id": "dinov2_vitl14", "pooling": "cls", "tag": "dinov2_vitl14"},
        "virchow":  {"backend": "timm_hf", "id": "hf-hub:paige-ai/Virchow",  "pooling": "cls_mean_concat", "tag": "virchow"},
        "virchow2": {"backend": "timm_hf", "id": "hf-hub:paige-ai/Virchow2", "pooling": "cls_mean_concat", "tag": "virchow2"},
        "uni":   {"backend": "timm_hf", "id": "hf-hub:MahmoodLab/UNI",    "pooling": "cls", "tag": "uni"},
        "uni2-h": {
            "backend": "timm_hf",
            "id": "hf-hub:MahmoodLab/UNI2-h",
            "pooling": "cls",
            "tag": "uni2-h",
            "timm_kwargs": {
                "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
                "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
                "reg_tokens": 8, "dynamic_img_size": True,
            },
            "needs_swiglu_silu": True,
        },
        "phikon-v2": {"backend": "hf_transformers", "id": "owkin/phikon-v2", "pooling": "cls", "tag": "phikon-v2"},
    }

    if backend == "auto":
        if encoder in presets:
            backend_used = presets[encoder]["backend"]
            enc_id = presets[encoder]["id"]
            preset = presets[encoder]
        else:
            if encoder.startswith("hf-hub:"):
                backend_used = "timm_hf"
                enc_id = encoder
                preset = {}
            elif "/" in encoder:
                backend_used = "hf_transformers"
                enc_id = encoder
                preset = {}
            else:
                backend_used = "dinov2_hub"
                enc_id = encoder
                preset = {}
    else:
        backend_used = backend
        enc_id = encoder if encoder not in presets else presets[encoder]["id"]
        preset = presets.get(encoder, {})

    pooling_used = preset.get("pooling", "cls") if pooling == "auto" else pooling
    tag = preset.get("tag", encoder.replace("/", "_").replace(":", "_"))

    _maybe_hf_login(hf_token)

    if backend_used == "dinov2_hub":
        print(f"[INFO] Loading DINOv2 (torch.hub) '{enc_id}' ...")
        model = torch.hub.load("facebookresearch/dinov2", enc_id)
        model.eval()
        transform = build_dinov2_transform(img_size=img_size)
        return model, transform, backend_used, pooling_used, tag

    if backend_used == "timm_hf":
        print(f"[INFO] Loading timm model '{enc_id}' ...")
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        timm_kwargs = dict(preset.get("timm_kwargs", {}))
        if preset.get("needs_swiglu_silu", False) or encoder in ("virchow", "virchow2"):
            from timm.layers import SwiGLUPacked
            timm_kwargs.setdefault("mlp_layer", SwiGLUPacked)
            timm_kwargs.setdefault("act_layer", torch.nn.SiLU)

        repo_id = enc_id.split("hf-hub:", 1)[1] if enc_id.startswith("hf-hub:") else None
        legacy_weights_msg = "weights_only=True"  # torch>=2.6 + legacy .tar checkpoint
        legacy_tar_msg = "legacy .tar format"
        try:
            model = timm.create_model(enc_id, pretrained=True, **timm_kwargs)
        except (RuntimeError, OSError) as e:
            msg = str(e)
            corrupted = (
                "PytorchStreamReader failed reading file" in msg
                or "archive is corrupted" in msg
                or "invalid header" in msg
            )
            if repo_id and corrupted:
                print(
                    f"[WARN] Corrupted cached weights detected for '{repo_id}'. Clearing cache and retrying once.",
                    file=sys.stderr,
                )
                _clear_hf_repo_cache(repo_id)
                try:
                    model = timm.create_model(enc_id, pretrained=True, **timm_kwargs)
                except (RuntimeError, OSError) as e2:
                    msg2 = str(e2)
                    if legacy_weights_msg in msg2 and legacy_tar_msg in msg2:
                        print(
                            f"[WARN] Falling back to legacy checkpoint load for '{repo_id}'.",
                            file=sys.stderr,
                        )
                        model = timm.create_model(enc_id, pretrained=False, **timm_kwargs)
                        model = _load_timm_hf_legacy_checkpoint(model, repo_id)
                    else:
                        raise
            elif repo_id and legacy_weights_msg in msg and legacy_tar_msg in msg:
                print(
                    f"[WARN] Falling back to legacy checkpoint load for '{repo_id}'.",
                    file=sys.stderr,
                )
                model = timm.create_model(enc_id, pretrained=False, **timm_kwargs)
                model = _load_timm_hf_legacy_checkpoint(model, repo_id)
            else:
                raise
        model.eval()

        cfg = resolve_data_config(getattr(model, "pretrained_cfg", {}), model=model)
        transform = create_transform(**cfg)
        return model, transform, backend_used, pooling_used, tag

    if backend_used == "hf_transformers":
        print(f"[INFO] Loading Transformers vision backbone '{enc_id}' ...")
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(enc_id)
        model = AutoModel.from_pretrained(enc_id)
        model.eval()
        return (model, processor), None, backend_used, pooling_used, tag

    raise ValueError(f"Unknown backend: {backend_used}")


# --------------------------
# Region readers (image + mask)
# --------------------------
def _normalize_hwc(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
    return arr


def _shape_as_hwc(shp: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(shp) == 2:
        return (shp[0], shp[1], 1)
    if len(shp) == 3:
        if shp[0] in (1, 3, 4) and shp[2] not in (1, 3, 4):
            return (shp[1], shp[2], shp[0])
        return (shp[0], shp[1], shp[2])
    raise ValueError(f"Unsupported array shape: {shp}")


def _slice_with_pad_np(img: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    H, W = img.shape[0], img.shape[1]
    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    pad_left = sx0 - x0
    pad_top = sy0 - y0

    if img.ndim == 2:
        out = np.zeros((h, w), dtype=img.dtype)
        out[pad_top:pad_top + (sy1 - sy0), pad_left:pad_left + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
        return out

    C = img.shape[2]
    out = np.zeros((h, w, C), dtype=img.dtype)
    out[pad_top:pad_top + (sy1 - sy0), pad_left:pad_left + (sx1 - sx0), :] = img[sy0:sy1, sx0:sx1, :]
    return out


def _slice_with_pad_zarr(z, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    shp = z.shape
    HWC = _shape_as_hwc(shp)
    H, W, C = HWC

    x1, y1 = x0 + w, y0 + h
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x1), min(H, y1)
    pad_left = sx0 - x0
    pad_top = sy0 - y0

    if len(shp) == 2:
        out = np.zeros((h, w), dtype=z.dtype)
        patch = z[sy0:sy1, sx0:sx1]
        out[pad_top:pad_top + (sy1 - sy0), pad_left:pad_left + (sx1 - sx0)] = patch
        return np.asarray(out)

    if len(shp) == 3 and shp[0] in (1, 3, 4) and shp[2] not in (1, 3, 4):
        out = np.zeros((h, w, shp[0]), dtype=z.dtype)
        patch = z[:, sy0:sy1, sx0:sx1]
        patch = np.moveaxis(np.asarray(patch), 0, -1)
        out[pad_top:pad_top + (sy1 - sy0), pad_left:pad_left + (sx1 - sx0), :] = patch
        if out.shape[-1] == 4:
            out = out[..., :3]
        return out

    out = np.zeros((h, w, shp[2]), dtype=z.dtype)
    patch = np.asarray(z[sy0:sy1, sx0:sx1, :])
    out[pad_top:pad_top + (sy1 - sy0), pad_left:pad_left + (sx1 - sx0), :] = patch
    if out.shape[-1] == 4:
        out = out[..., :3]
    return out


class RegionReader:
    """For RGB image (openslide if possible, else tifffile+zarr if pyramid, else full numpy load)."""
    def __init__(self, path: Path, level: int = 0, force_full_image: bool = False):
        self.path = Path(path).resolve()
        self.level = level
        self.backend = None
        self._osr = None
        self._tif = None
        self._z = None
        self._np = None
        self.shape = None  # (H,W,C) or (H,W)

        if force_full_image:
            arr = tifffile.imread(str(self.path))
            arr = _normalize_hwc(arr)
            self._np = arr
            self.backend = "numpy_forced"
            self.shape = arr.shape
            return

        # Try openslide
        try:
            import openslide
            self._osr = openslide.OpenSlide(str(self.path))
            self.backend = "openslide"
            w, h = self._osr.level_dimensions[level]
            self.shape = (h, w, 3)
            return
        except Exception:
            pass

        # Try tifffile + zarr
        try:
            import zarr
            self._tif = tifffile.TiffFile(str(self.path))
            series = self._tif.series[0]
            levels = getattr(series, "levels", None)
            if levels is None or len(levels) == 0:
                arr = series.asarray()
                arr = _normalize_hwc(arr)
                self._np = arr
                self.backend = "numpy"
                self.shape = arr.shape
                return

            lvl = levels[level]
            store = lvl.aszarr()
            self._z = zarr.open(store, mode="r")
            self.backend = "tifffile_zarr"
            self.shape = _shape_as_hwc(self._z.shape)
            return
        except Exception:
            arr = tifffile.imread(str(self.path))
            arr = _normalize_hwc(arr)
            self._np = arr
            self.backend = "numpy"
            self.shape = arr.shape

    def read(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        if self.backend == "openslide":
            tile = self._osr.read_region((x0, y0), self.level, (w, h)).convert("RGB")
            return np.asarray(tile)
        if self.backend == "tifffile_zarr":
            return _slice_with_pad_zarr(self._z, x0, y0, w, h)
        return _slice_with_pad_np(self._np, x0, y0, w, h)


class LabelReader:
    """For 2D label mask; tries tifffile+zarr window reads; else full load fallback."""
    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.backend = None
        self._tif = None
        self._z = None
        self._np = None
        self.shape = None  # (H,W)

        # Try tifffile + zarr
        try:
            import zarr
            self._tif = tifffile.TiffFile(str(self.path))
            series = self._tif.series[0]
            levels = getattr(series, "levels", None)
            if levels is None or len(levels) == 0:
                arr = series.asarray()
                if arr.ndim != 2:
                    raise RuntimeError(f"Mask must be 2D, got {arr.shape}")
                self._np = arr
                self.backend = "numpy"
                self.shape = arr.shape
                return

            # if mask is pyramidal, use level 0 by default
            lvl = levels[0]
            store = lvl.aszarr()
            self._z = zarr.open(store, mode="r")
            if len(self._z.shape) != 2:
                raise RuntimeError(f"Mask zarr must be 2D, got {self._z.shape}")
            self.backend = "tifffile_zarr"
            self.shape = (self._z.shape[0], self._z.shape[1])
            return
        except Exception:
            arr = tifffile.imread(str(self.path))
            if arr.ndim != 2:
                raise RuntimeError(f"Mask must be 2D, got {arr.shape}")
            self._np = arr
            self.backend = "numpy"
            self.shape = arr.shape

    def read(self, x0: int, y0: int, w: int, h: int) -> np.ndarray:
        if self.backend == "tifffile_zarr":
            return _slice_with_pad_zarr(self._z, x0, y0, w, h)
        return _slice_with_pad_np(self._np, x0, y0, w, h)


def _ensure_rgb(tile: np.ndarray) -> np.ndarray:
    if tile.ndim == 2:
        tile = np.stack([tile, tile, tile], axis=-1)
    if tile.ndim == 3 and tile.shape[-1] == 1:
        tile = np.repeat(tile, 3, axis=-1)
    if tile.ndim == 3 and tile.shape[-1] >= 3:
        tile = tile[..., :3]
    return tile


def compute_global_percentiles(img_reader: RegionReader,
                               sample_tiles: int = 200,
                               sample_size: int = 512,
                               seed: int = 0,
                               p_lo: float = 1.0,
                               p_hi: float = 99.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate per-channel intensity percentiles on the SAME image reader,
    so we can map all tiles consistently to uint8.
    """
    H, W = img_reader.shape[0], img_reader.shape[1]
    rng = random.Random(seed)

    vals = [[] for _ in range(3)]
    for _ in tqdm(range(sample_tiles), desc="Sampling image for global scaling"):
        x0 = rng.randint(0, max(0, W - sample_size))
        y0 = rng.randint(0, max(0, H - sample_size))
        t = img_reader.read(x0, y0, sample_size, sample_size)
        t = _ensure_rgb(t).astype(np.float32, copy=False)
        for c in range(3):
            v = t[..., c].ravel()
            if v.size:
                # subsample to limit memory
                if v.size > 20000:
                    idx = np.random.default_rng(seed).choice(v.size, size=20000, replace=False)
                    v = v[idx]
                vals[c].append(v)

    lo = np.zeros((3,), dtype=np.float32)
    hi = np.zeros((3,), dtype=np.float32)
    for c in range(3):
        if not vals[c]:
            lo[c], hi[c] = 0.0, 255.0
            continue
        vv = np.concatenate(vals[c])
        lo[c] = np.percentile(vv, p_lo)
        hi[c] = np.percentile(vv, p_hi)
        if not np.isfinite(lo[c]) or not np.isfinite(hi[c]) or hi[c] <= lo[c]:
            lo[c], hi[c] = float(vv.min()), float(vv.max())
            if hi[c] <= lo[c]:
                lo[c], hi[c] = 0.0, 255.0

    print(f"[INFO] global scaling p{p_lo}/p{p_hi}: lo={lo.tolist()} hi={hi.tolist()}")
    return lo, hi


def to_rgb_uint8_global(tile: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary dtype tile to uint8 using one global (per-channel) linear mapping.
    """
    tile = _ensure_rgb(tile).astype(np.float32, copy=False)
    out = np.empty(tile.shape, dtype=np.uint8)
    for c in range(3):
        denom = (hi[c] - lo[c]) if (hi[c] > lo[c]) else 1.0
        x = (tile[..., c] - lo[c]) / denom
        x = np.clip(x, 0.0, 1.0) * 255.0
        out[..., c] = x.astype(np.uint8)
    return out


# --------------------------
# Streaming centroid computation (mask-only pass)
# --------------------------
def ensure_len(arr: np.ndarray, n: int, fill: float = 0) -> np.ndarray:
    if arr.shape[0] >= n:
        return arr
    out = np.empty((n,), dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    out[arr.shape[0]:] = 0 if fill == 0 else fill
    return out


def compute_centroids_streaming(mask_reader: LabelReader, block: int = 4096, verbose: bool = True):
    H, W = mask_reader.shape
    counts = np.zeros((1,), dtype=np.int64)
    sumx = np.zeros((1,), dtype=np.float64)
    sumy = np.zeros((1,), dtype=np.float64)

    for y0 in tqdm(range(0, H, block), disable=not verbose, desc="Mask blocks (centroids)"):
        h = min(block, H - y0)
        for x0 in range(0, W, block):
            w = min(block, W - x0)
            m = mask_reader.read(x0, y0, w, h)
            if m.size == 0:
                continue
            m = m.astype(np.int64, copy=False)
            mx = int(m.max())
            if mx >= counts.shape[0]:
                newn = mx + 1
                counts = ensure_len(counts, newn)
                sumx = ensure_len(sumx, newn)
                sumy = ensure_len(sumy, newn)

            flat = m.ravel()
            c = np.bincount(flat, minlength=mx + 1)

            yy = (np.arange(h, dtype=np.float64) + y0)[:, None]
            xx = (np.arange(w, dtype=np.float64) + x0)[None, :]
            wy = np.bincount(flat, weights=np.broadcast_to(yy, (h, w)).ravel(), minlength=mx + 1)
            wx = np.bincount(flat, weights=np.broadcast_to(xx, (h, w)).ravel(), minlength=mx + 1)

            counts[:mx + 1] += c
            sumx[:mx + 1] += wx
            sumy[:mx + 1] += wy

    labels = np.nonzero(counts[1:] > 0)[0] + 1
    area = counts[labels].astype(np.int64)
    cx = (sumx[labels] / counts[labels]).astype(np.float64)
    cy = (sumy[labels] / counts[labels]).astype(np.float64)
    return labels.astype(np.int64), cx, cy, area


# --------------------------
# Folder helpers
# --------------------------
def tile_folder(base: Path, grid_r: int, grid_c: int) -> Path:
    p = base / f"grid_r{grid_r:02d}_c{grid_c:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cell_subfolder(tile_base: Path, cell_id: int, per_dir: int = 5000) -> Path:
    b = cell_id // per_dir
    p = tile_base / f"cells_{b:05d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------
# Main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--image-level", type=int, default=0, help="Pyramid level for IMAGE (mask assumed same grid)")
    p.add_argument("--force-full-image", action="store_true",
                   help="Load the entire image into ONE numpy array (recommended for crop_roi.tif)")

    p.add_argument("--grid", type=str, default="10x10", help="Grid like 10x10")

    p.add_argument("--tile-size", type=int, default=224, help="Per-cell crop size for encoder")
    p.add_argument("--zero-outside-mask", action="store_true")
    p.add_argument("--outside-fill", type=int, default=0)

    p.add_argument("--save-tiles", action="store_true")
    p.add_argument("--tiles-root", default=None, help="Root folder for saved tiles (default: <outdir>/tiles)")
    p.add_argument("--tiles-format", default="png", choices=["png", "tif"])
    p.add_argument("--bucket-size", type=int, default=5000, help="cells per subfolder bucket when saving tiles")

    p.add_argument("--min-area", type=int, default=0)

    p.add_argument("--encoder", type=str, default="uni2-h")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "dinov2_hub", "timm_hf", "hf_transformers"])
    p.add_argument("--pooling", type=str, default="auto", choices=["auto", "cls", "mean_patch", "mean_all", "cls_mean_concat"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--hf-token", type=str, default=None)

    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--torch-threads", type=int, default=16)

    p.add_argument("--rows-per-csv", type=int, default=10000)
    p.add_argument("--mask-block", type=int, default=4096)

    # NEW: global scaling sampling
    p.add_argument("--scale-samples", type=int, default=200)
    p.add_argument("--scale-tile", type=int, default=512)
    p.add_argument("--scale-seed", type=int, default=0)
    p.add_argument("--scale-plo", type=float, default=1.0)
    p.add_argument("--scale-phi", type=float, default=99.0)

    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gr, gc = args.grid.lower().split("x")
    GR, GC = int(gr), int(gc)

    tiles_root = Path(args.tiles_root).resolve() if args.tiles_root else (outdir / "tiles")
    if args.save_tiles:
        tiles_root.mkdir(parents=True, exist_ok=True)

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device={device}")

    # Load encoder
    model_obj, transform, backend_used, pooling_used, tag = load_encoder(
        encoder=args.encoder, backend=args.backend, pooling=args.pooling, img_size=args.img_size, hf_token=args.hf_token
    )
    print(f"[INFO] encoder={args.encoder} backend={backend_used} pooling={pooling_used} tag={tag}")

    if backend_used == "hf_transformers":
        model, processor = model_obj
        model.to(device)
    else:
        model = model_obj
        processor = None
        model.to(device)
    model.eval()

    # Readers
    img_path = Path(args.image).expanduser().resolve()
    mask_path = Path(args.mask).expanduser().resolve()

    img_reader = RegionReader(img_path, level=args.image_level, force_full_image=args.force_full_image)
    mask_reader = LabelReader(mask_path)

    H, W = mask_reader.shape
    print(f"[INFO] mask shape={H}x{W} (backend={mask_reader.backend})")
    if img_reader.shape is not None:
        ih, iw = img_reader.shape[0], img_reader.shape[1]
        print(f"[INFO] image shape={ih}x{iw} (backend={img_reader.backend})")
        if (ih != H) or (iw != W):
            print(f"[WARN] image(level={args.image_level}) shape={ih}x{iw} differs from mask={H}x{W}")
            print("[WARN] Ensure mask and selected image level are aligned!")

    # NEW: compute global scaling once
    lo, hi = compute_global_percentiles(
        img_reader,
        sample_tiles=int(args.scale_samples),
        sample_size=int(args.scale_tile),
        seed=int(args.scale_seed),
        p_lo=float(args.scale_plo),
        p_hi=float(args.scale_phi),
    )

    # Pass 1: centroids/area streaming
    labels, cx, cy, area = compute_centroids_streaming(mask_reader, block=args.mask_block, verbose=True)
    if args.min_area > 0:
        keep = area >= args.min_area
        labels, cx, cy, area = labels[keep], cx[keep], cy[keep], area[keep]
    N = labels.shape[0]
    print(f"[INFO] cells after filter: {N}")

    # Assign each cell to a grid tile based on centroid
    tile_w = int(math.ceil(W / GC))
    tile_h = int(math.ceil(H / GR))
    grid_r = np.clip((cy // tile_h).astype(np.int64), 0, GR - 1)
    grid_c = np.clip((cx // tile_w).astype(np.int64), 0, GC - 1)
    tile_id = grid_r * GC + grid_c

    order = np.argsort(tile_id)
    labels, cx, cy, area, tile_id = labels[order], cx[order], cy[order], area[order], tile_id[order]

    tile_starts = np.searchsorted(tile_id, np.arange(GR * GC), side="left")
    tile_ends   = np.searchsorted(tile_id, np.arange(GR * GC), side="right")

    half = args.tile_size // 2

    row_parts: List[pd.DataFrame] = []
    row_count = 0
    shard_idx = 0
    feat_dim = None
    feat_cols: Optional[List[str]] = None

    def flush_csv(tile_out: Path):
        nonlocal row_parts, row_count, shard_idx
        if row_count == 0:
            return
        df = pd.concat(row_parts, ignore_index=True)
        out_path = tile_out / f"{tag}_embeddings_shard{shard_idx:04d}.csv.gz"
        df.to_csv(out_path, index=False, compression="gzip")
        row_parts = []
        row_count = 0
        shard_idx += 1

    batch_items = []
    batch_meta = []

    def add_example(tile_rgb_u8: np.ndarray, meta: Dict[str, Any]):
        pil = Image.fromarray(tile_rgb_u8, mode="RGB")
        if backend_used == "hf_transformers":
            batch_items.append(pil)
        else:
            batch_items.append(transform(pil))
        batch_meta.append(meta)

    def run_batch(tile_out: Path):
        nonlocal feat_dim, feat_cols, batch_items, batch_meta, row_parts, row_count
        if not batch_items:
            return

        with torch.inference_mode():
            if backend_used == "hf_transformers":
                inputs = processor(list(batch_items), return_tensors="pt")
                images = inputs["pixel_values"].to(device, non_blocking=True)
                outputs = model(pixel_values=images)
                feats = getattr(outputs, "last_hidden_state", None) or outputs[0]
            else:
                images = torch.stack(batch_items, dim=0).to(device, non_blocking=True)
                feats = model(images)

            if isinstance(feats, dict):
                if "last_hidden_state" in feats:
                    feats = feats["last_hidden_state"]
                elif "x_norm_clstoken" in feats:
                    feats = feats["x_norm_clstoken"]
                else:
                    feats = next(iter(feats.values()))

            feats = pool_tokens(feats, pooling_used).detach().cpu().numpy()

        if feat_dim is None:
            feat_dim = int(feats.shape[1])
            feat_cols = [f"feat_{j+1}" for j in range(feat_dim)]

        meta_df = pd.DataFrame.from_records(batch_meta)
        feat_df = pd.DataFrame(feats, columns=feat_cols)
        row_parts.append(pd.concat([meta_df.reset_index(drop=True), feat_df], axis=1))
        row_count += feats.shape[0]

        batch_items, batch_meta = [], []
        if row_count >= args.rows_per_csv:
            flush_csv(tile_out)

    # Pass 2: process grid tiles
    for tr in range(GR):
        for tc in range(GC):
            tid = tr * GC + tc
            s, e = tile_starts[tid], tile_ends[tid]
            if s == e:
                continue

            x0_core = tc * tile_w
            y0_core = tr * tile_h
            x1_core = min(W, x0_core + tile_w)
            y1_core = min(H, y0_core + tile_h)

            tile_out = tile_folder(outdir, tr, tc)
            tile_tiles = tile_folder(tiles_root, tr, tc) if args.save_tiles else None

            shard_idx = 0
            row_parts = []
            row_count = 0
            feat_dim = None
            feat_cols = None

            for i in tqdm(range(s, e), desc=f"grid {tr:02d},{tc:02d}", leave=False):
                lab = int(labels[i])
                x = int(round(float(cx[i])))
                y = int(round(float(cy[i])))
                a = int(area[i])

                if not (x0_core <= x < x1_core and y0_core <= y < y1_core):
                    continue

                x0 = x - half
                y0 = y - half

                img_tile_raw = img_reader.read(x0, y0, args.tile_size, args.tile_size)

                # FIX: consistent mapping to uint8 using global scaling
                img_tile = to_rgb_uint8_global(img_tile_raw, lo=lo, hi=hi)

                if args.zero_outside_mask:
                    m_tile = mask_reader.read(x0, y0, args.tile_size, args.tile_size).astype(np.int64, copy=False)
                    keep_m = (m_tile == lab)
                    fill = int(args.outside_fill)
                    fill = 0 if fill < 0 else (255 if fill > 255 else fill)
                    img_tile = img_tile.copy()
                    img_tile[~keep_m, :] = fill

                tile_path = ""
                if args.save_tiles and tile_tiles is not None:
                    sub = cell_subfolder(tile_tiles, lab, per_dir=int(args.bucket_size))
                    tile_name = f"cell_{lab:08d}_x{x}_y{y}.{args.tiles_format}"
                    outp = sub / tile_name
                    tile_path = str(outp)
                    if args.tiles_format == "png":
                        Image.fromarray(img_tile).save(outp)
                    else:
                        tifffile.imwrite(outp, img_tile, compression="zlib")

                meta = {
                    "cell_id": lab,
                    "cx": x,
                    "cy": y,
                    "area_px": a,
                    "tile_x0": int(x0),
                    "tile_y0": int(y0),
                    "grid_r": int(tr),
                    "grid_c": int(tc),
                    "tile_path": tile_path,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "image_level": int(args.image_level),
                    "zero_outside_mask": bool(args.zero_outside_mask),
                }

                add_example(img_tile, meta)
                if len(batch_items) >= int(args.batch):
                    run_batch(tile_out)

            run_batch(tile_out)
            flush_csv(tile_out)

    print("🎉 Done.")


if __name__ == "__main__":
    main()
