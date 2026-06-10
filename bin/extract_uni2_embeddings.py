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
import json
import os
import sys
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Optional, Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Import imagecodecs early; later imports can leave JPEG decode bound to stubs.
try:
    import imagecodecs  # noqa: F401
    _ = imagecodecs.jpeg8_decode
except Exception:
    imagecodecs = None  # type: ignore

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


def pool_from_token_parts(cls: Optional[torch.Tensor],
                          patch: Optional[torch.Tensor],
                          mode: str) -> torch.Tensor:
    if cls is None and patch is None:
        raise ValueError("Cannot pool embeddings without cls or patch tokens")

    if patch is None and cls is not None:
        # Preserve legacy behavior for backends that already return pooled 2D embeddings.
        return cls

    if patch is not None and patch.dim() != 3:
        raise ValueError(f"Unexpected patch token shape: {tuple(patch.shape)}")
    if cls is not None and cls.dim() != 2:
        raise ValueError(f"Unexpected cls token shape: {tuple(cls.shape)}")

    if patch is not None and patch.size(1) > 0:
        mean_patch = patch.mean(1)
    elif cls is not None:
        mean_patch = cls
    else:
        raise ValueError("Cannot compute mean_patch without cls or patch tokens")

    cls_token = cls if cls is not None else mean_patch

    if mode == "cls":
        return cls_token
    if mode == "mean_patch":
        return mean_patch
    if mode == "mean_all":
        if cls is not None and patch is not None and patch.size(1) > 0:
            return torch.cat([cls.unsqueeze(1), patch], dim=1).mean(1)
        return mean_patch
    if mode == "cls_mean_concat":
        return torch.cat([cls_token, mean_patch], dim=-1)
    raise ValueError(f"Unknown pooling mode: {mode}")


def extract_cls_and_patch_tokens(feats: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if isinstance(feats, dict):
        if "x_norm_patchtokens" in feats or "x_norm_clstoken" in feats:
            cls = feats.get("x_norm_clstoken")
            patch = feats.get("x_norm_patchtokens")
            if cls is not None and cls.dim() == 3 and cls.size(1) == 1:
                cls = cls[:, 0]
            return cls, patch
        if "last_hidden_state" in feats:
            feats = feats["last_hidden_state"]
        else:
            feats = next(iter(feats.values()))

    if torch.is_tensor(feats):
        if feats.dim() == 2:
            return feats, None
        if feats.dim() == 3:
            cls = feats[:, 0]
            patch = feats[:, 1:] if feats.size(1) > 1 else None
            # Some ViT variants, including UNI2-h, prepend extra prefix/register
            # tokens after CLS. For inner-square-style derivation we need only the
            # square patch-token grid, so strip any leading non-patch tokens if a
            # perfect-square tail exists.
            if patch is not None and patch.size(1) > 0:
                n = int(patch.size(1))
                side = int(round(math.sqrt(n)))
                if side * side != n:
                    for extra_prefix in range(1, min(32, n)):
                        tail = n - extra_prefix
                        if tail <= 0:
                            break
                        tail_side = int(round(math.sqrt(tail)))
                        if tail_side * tail_side == tail:
                            patch = patch[:, extra_prefix:]
                            break
            return cls, patch

    raise ValueError(f"Unsupported feature container for token extraction: {type(feats)}")


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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return default


def _hf_offline_enabled() -> bool:
    return _env_flag("HF_HUB_OFFLINE", False) or _env_flag("TRANSFORMERS_OFFLINE", False)


def _load_hf_repo_config(repo_id: str, local_files_only: bool = False) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
        local_files_only=local_files_only,
    )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _create_timm_model_from_repo_config(repo_id: str, timm_kwargs: Dict[str, Any], local_files_only: bool = False):
    import timm

    repo_cfg = _load_hf_repo_config(repo_id, local_files_only=local_files_only)
    architecture = repo_cfg.get("architecture")
    if not architecture:
        raise RuntimeError(f"HF repo config for '{repo_id}' does not define an architecture")

    model = timm.create_model(architecture, pretrained=False, **timm_kwargs)
    repo_pretrained_cfg = repo_cfg.get("pretrained_cfg") or {}
    if repo_pretrained_cfg:
        base_cfg = getattr(model, "pretrained_cfg", {}) or {}
        merged_cfg = dict(base_cfg)
        merged_cfg.update(repo_pretrained_cfg)
        model.pretrained_cfg = merged_cfg
    return model


def _load_timm_hf_legacy_checkpoint(model: Any, repo_id: str, local_files_only: bool = False) -> Any:
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename="pytorch_model.bin",
        local_files_only=local_files_only,
    )
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


def _load_timm_hf_checkpoint(model: Any, repo_id: str, local_files_only: bool = False) -> Any:
    """
    Load timm HF checkpoint from local/cache first.
    Tries safetensors, then legacy pytorch_model.bin.
    """
    from huggingface_hub import hf_hub_download

    # Preferred modern format.
    try:
        ckpt_safe = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_files_only=local_files_only,
        )
        from safetensors.torch import load_file as load_safetensors

        state_dict = load_safetensors(ckpt_safe, device="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] safetensors load missing keys: {len(missing)}", file=sys.stderr)
        if unexpected:
            print(f"[WARN] safetensors load unexpected keys: {len(unexpected)}", file=sys.stderr)
        return model
    except Exception as safetensor_err:
        print(f"[WARN] safetensors load failed, trying legacy bin: {safetensor_err}", file=sys.stderr)

    # Backward-compatible legacy format.
    return _load_timm_hf_legacy_checkpoint(model, repo_id, local_files_only=local_files_only)


def _is_transient_hf_error(msg: str) -> bool:
    lowered = (msg or "").lower()
    transient_markers = [
        "cannot send a request, as the client has been closed",
        "connection timed out",
        "timed out",
        "timeout",
        "temporary failure in name resolution",
        "connection reset by peer",
        "remote end closed connection",
        "connection aborted",
        "connection refused",
        "read timed out",
        "name or service not known",
        "service unavailable",
        "too many requests",
    ]
    return any(marker in lowered for marker in transient_markers)


def _create_timm_model_with_fallback(enc_id: str, timm_kwargs: Dict[str, Any], repo_id: Optional[str]):
    import timm

    legacy_weights_msg = "weights_only=True"  # torch>=2.6 + legacy .tar checkpoint
    legacy_tar_msg = "legacy .tar format"

    # In strict offline mode, avoid timm's remote resolution path.
    if repo_id and _hf_offline_enabled():
        print(f"[INFO] HF offline mode enabled; loading '{repo_id}' from local cache only.", file=sys.stderr)
        model = _create_timm_model_from_repo_config(repo_id, timm_kwargs, local_files_only=True)
        return _load_timm_hf_checkpoint(model, repo_id, local_files_only=True)

    try:
        return timm.create_model(enc_id, pretrained=True, **timm_kwargs)
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
                return timm.create_model(enc_id, pretrained=True, **timm_kwargs)
            except (RuntimeError, OSError) as e2:
                msg2 = str(e2)
                if legacy_weights_msg in msg2 and legacy_tar_msg in msg2:
                    print(
                        f"[WARN] Falling back to legacy checkpoint load for '{repo_id}'.",
                        file=sys.stderr,
                    )
                    model = _create_timm_model_from_repo_config(repo_id, timm_kwargs, local_files_only=_hf_offline_enabled())
                    return _load_timm_hf_checkpoint(model, repo_id, local_files_only=_hf_offline_enabled())
                raise
        if repo_id and legacy_weights_msg in msg and legacy_tar_msg in msg:
            print(
                f"[WARN] Falling back to legacy checkpoint load for '{repo_id}'.",
                file=sys.stderr,
            )
            model = _create_timm_model_from_repo_config(repo_id, timm_kwargs, local_files_only=_hf_offline_enabled())
            return _load_timm_hf_checkpoint(model, repo_id, local_files_only=_hf_offline_enabled())
        raise


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
        max_attempts = max(1, int(os.environ.get("UNI2_HF_LOAD_RETRIES", "4")))
        retry_delay = max(1.0, float(os.environ.get("UNI2_HF_LOAD_RETRY_DELAY_SEC", "10")))
        model = None
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                model = _create_timm_model_with_fallback(enc_id, timm_kwargs, repo_id)
                break
            except Exception as e:
                last_err = e
                msg = str(e)
                if attempt < max_attempts and _is_transient_hf_error(msg):
                    sleep_s = retry_delay * attempt
                    print(
                        f"[WARN] UNI2 model load transient failure (attempt {attempt}/{max_attempts}): {e}",
                        file=sys.stderr,
                    )
                    print(f"[WARN] Retrying UNI2 model load in {sleep_s:.1f}s...", file=sys.stderr)
                    time.sleep(sleep_s)
                    continue
                raise
        if model is None and last_err is not None:
            raise last_err
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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if float(den) == 0:
            return None
        return float(num) / float(den)
    try:
        return float(value)
    except Exception:
        return None


def read_source_mpp(path: str) -> Optional[float]:
    unit_to_um = {
        2: 25_400.0,  # inch
        3: 10_000.0,  # centimeter
    }
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tags = page.tags
            desc = None
            try:
                desc = tags["ImageDescription"].value
            except Exception:
                desc = None
            if isinstance(desc, str):
                for pattern in (
                    r'PhysicalSizeX="([0-9]+(?:\.[0-9]+)?)"',
                    r"MPP\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                ):
                    m = re.search(pattern, desc)
                    if m:
                        return float(m.group(1))

            unit = None
            try:
                unit = int(tags["ResolutionUnit"].value)
            except Exception:
                unit = None
            if unit in unit_to_um:
                vals = []
                for tag_name in ("XResolution", "YResolution"):
                    try:
                        res = _safe_float(tags[tag_name].value)
                    except Exception:
                        res = None
                    if res and res > 0:
                        vals.append(unit_to_um[unit] / res)
                if vals:
                    return float(sum(vals) / len(vals))
    except Exception:
        return None
    return None


def infer_source_mpp(path: str) -> Optional[float]:
    direct = read_source_mpp(path)
    if direct:
        return direct

    image_path = Path(path).resolve()
    shift_path = image_path.with_name("shift.json")
    if not shift_path.exists():
        return None
    try:
        shift = json.loads(shift_path.read_text())
    except Exception:
        return None
    input_image = shift.get("input_image")
    if not input_image:
        return None
    input_path = Path(str(input_image))
    candidate_paths: List[Path] = []
    if input_path.is_absolute():
        candidate_paths.append(input_path)
    for parent in [image_path.parent, *image_path.parents]:
        candidate_paths.append(parent / input_path)
        if not input_path.is_absolute():
            try:
                candidate_paths.extend(parent.glob(f"input*/{input_path.name}"))
            except Exception:
                pass
    seen = set()
    for candidate in candidate_paths:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        mpp = read_source_mpp(str(resolved))
        if mpp:
            print(f"[INFO] UNI2 source MPP recovered from {resolved}", flush=True)
            return mpp
    return None


def resolve_extraction_tile_size(model_tile_size: int, source_mpp: Optional[float], target_mpp: float) -> Tuple[int, Optional[float]]:
    model_tile_size = max(1, int(model_tile_size))
    if not source_mpp or source_mpp <= 0 or target_mpp <= 0:
        return model_tile_size, source_mpp
    extraction_tile_size = max(1, int(round(float(model_tile_size) * float(target_mpp) / float(source_mpp))))
    effective_mpp = float(source_mpp) * (float(extraction_tile_size) / float(model_tile_size))
    return extraction_tile_size, effective_mpp


def _resolve_zarr_array(zobj: Any, level: int = 0):
    """
    tifffile+zarr can return either an Array (older behavior) or a Group with
    level-indexed arrays (newer behavior, e.g. keys '0','1','2').
    Normalize both cases to a concrete array-like object with `.shape`.
    """
    if hasattr(zobj, "shape"):
        return zobj

    if hasattr(zobj, "array_keys"):
        keys = list(zobj.array_keys())
        if not keys:
            raise RuntimeError("Zarr group has no arrays")

        preferred = str(level)
        if preferred in keys:
            return zobj[preferred]

        numeric_keys = [k for k in keys if str(k).isdigit()]
        if numeric_keys:
            numeric_keys = sorted(numeric_keys, key=lambda x: int(x))
            return zobj[numeric_keys[0]]

        return zobj[sorted(keys)[0]]

    raise RuntimeError(f"Unsupported zarr object type: {type(zobj)}")


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
            self._z = _resolve_zarr_array(zarr.open(store, mode="r"), level=level)
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

        if self.path.is_dir() or str(self.path).lower().endswith(".zarr"):
            try:
                import zarr
                self._z = _resolve_zarr_array(zarr.open(str(self.path), mode="r"), level=0)
                if len(self._z.shape) != 2:
                    raise RuntimeError(f"Mask zarr must be 2D, got {self._z.shape}")
                self.backend = "tifffile_zarr"
                self.shape = (self._z.shape[0], self._z.shape[1])
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
                if arr.ndim != 2:
                    raise RuntimeError(f"Mask must be 2D, got {arr.shape}")
                self._np = arr
                self.backend = "numpy"
                self.shape = arr.shape
                return

            # if mask is pyramidal, use level 0 by default
            lvl = levels[0]
            store = lvl.aszarr()
            self._z = _resolve_zarr_array(zarr.open(store, mode="r"), level=0)
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


def compute_inner_square_bounds(
    tile_size: int,
    center_x_local: int,
    center_y_local: int,
    area_px: int,
    fixed_px: int,
    factor: float,
    min_px: int,
    max_px: int,
) -> Tuple[int, int, int, int]:
    if fixed_px > 0:
        side = int(fixed_px)
    else:
        eq_d = math.sqrt(max(1.0, float(area_px)) * 4.0 / math.pi)
        side = int(round(eq_d * max(0.0, float(factor))))
    side = max(int(min_px), side)
    if max_px > 0:
        side = min(side, int(max_px))
    side = max(1, min(int(tile_size), side))

    half = side // 2
    x0 = max(0, center_x_local - half)
    y0 = max(0, center_y_local - half)
    x1 = min(tile_size, x0 + side)
    y1 = min(tile_size, y0 + side)

    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return int(x0), int(y0), int(x1), int(y1)


def build_inner_square_mask(
    tile_size: int,
    center_x_local: int,
    center_y_local: int,
    area_px: int,
    fixed_px: int,
    factor: float,
    min_px: int,
    max_px: int,
) -> np.ndarray:
    """Build a centered square mask in tile coordinates.

    Side length is computed from equivalent diameter unless fixed_px is provided.
    """
    x0, y0, x1, y1 = compute_inner_square_bounds(
        tile_size=tile_size,
        center_x_local=center_x_local,
        center_y_local=center_y_local,
        area_px=area_px,
        fixed_px=fixed_px,
        factor=factor,
        min_px=min_px,
        max_px=max_px,
    )

    keep = np.zeros((tile_size, tile_size), dtype=bool)
    keep[y0:y1, x0:x1] = True
    return keep


def derive_inner_square_style_embeddings(
    cls: Optional[torch.Tensor],
    patch: Optional[torch.Tensor],
    batch_meta: List[Dict[str, Any]],
    tile_size: int,
    img_size: int,
    pooling_mode: str,
    fixed_px: int,
    factor: float,
    min_px: int,
    max_px: int,
) -> Optional[np.ndarray]:
    if patch is None or patch.dim() != 3:
        return None

    n_tokens = int(patch.shape[1])
    token_side = int(round(math.sqrt(n_tokens)))
    if token_side * token_side != n_tokens:
        return None

    patch_pitch = float(img_size) / float(token_side)
    token_centers = (np.arange(token_side, dtype=np.float32) + 0.5) * patch_pitch
    grid_x, grid_y = np.meshgrid(token_centers, token_centers, indexing="xy")
    token_x = grid_x.reshape(-1)
    token_y = grid_y.reshape(-1)

    center_local = tile_size // 2
    scale = float(img_size) / float(tile_size)
    keep_rows = []
    for meta in batch_meta:
        area_px = int(meta.get("area_model_px", meta.get("area_px", 0)))
        x0, y0, x1, y1 = compute_inner_square_bounds(
            tile_size=tile_size,
            center_x_local=center_local,
            center_y_local=center_local,
            area_px=area_px,
            fixed_px=fixed_px,
            factor=factor,
            min_px=min_px,
            max_px=max_px,
        )
        keep = (
            (token_x >= (x0 * scale)) &
            (token_x < (x1 * scale)) &
            (token_y >= (y0 * scale)) &
            (token_y < (y1 * scale))
        )
        if not np.any(keep):
            center_idx = (token_side // 2) * token_side + (token_side // 2)
            keep[min(center_idx, n_tokens - 1)] = True
        keep_rows.append(keep)

    keep_mask = torch.as_tensor(np.stack(keep_rows, axis=0), device=patch.device, dtype=patch.dtype)
    denom = keep_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    selected_mean = (patch * keep_mask.unsqueeze(-1)).sum(dim=1) / denom
    cls_token = cls if cls is not None else selected_mean

    if pooling_mode == "cls_mean_concat":
        out = torch.cat([cls_token, selected_mean], dim=-1)
    else:
        out = selected_mean
    return out.detach().cpu().numpy()


# --------------------------
# Main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--image", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--paired-inner-square-outdir", default="",
                   help="Optional second output directory for paired inner_square embeddings")
    p.add_argument("--paired-inner-square-mode", default="token_subset",
                   choices=["masked_forward", "token_subset"],
                   help=(
                       "How to produce --paired-inner-square-outdir. "
                       "masked_forward runs the masked inner-square image through the encoder in the same batch as the tile image; "
                       "token_subset averages patch tokens from the tile forward pass."
                   ))

    p.add_argument("--image-level", type=int, default=0, help="Pyramid level for IMAGE (mask assumed same grid)")
    p.add_argument("--force-full-image", action="store_true",
                   help="Load the entire image into ONE numpy array (recommended for crop_roi.tif)")

    p.add_argument("--grid", type=str, default="10x10", help="Grid like 10x10")

    p.add_argument("--tile-size", type=int, default=224, help="Per-cell crop size for encoder")
    p.add_argument("--target-mpp", type=float, default=0.0,
                   help="Target microns-per-pixel represented by encoder input pixels. 0 keeps source pixels unchanged.")
    p.add_argument("--default-source-mpp", type=float, default=0.25,
                   help="Fallback source MPP if TIFF metadata does not provide calibration.")
    p.add_argument("--zero-outside-mask", action="store_true")
    p.add_argument("--outside-fill", type=int, default=0)

    p.add_argument("--save-tiles", action="store_true")
    p.add_argument("--tiles-root", default=None, help="Root folder for saved tiles (default: <outdir>/tiles)")
    p.add_argument("--tiles-format", default="png", choices=["png", "tif"])
    p.add_argument("--bucket-size", type=int, default=5000, help="cells per subfolder bucket when saving tiles")

    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--mask-context-mode", type=str, default="label",
                   choices=["none", "label", "inner_square", "label_and_inner_square"],
                   help="Masking mode applied when --zero-outside-mask is enabled.")
    p.add_argument("--inner-square-factor", type=float, default=1.0,
                   help="Inner-square side as factor of equivalent cell diameter.")
    p.add_argument("--inner-square-min-px", type=int, default=32,
                   help="Minimum side for the inner-square mask.")
    p.add_argument("--inner-square-max-px", type=int, default=0,
                   help="Maximum side for inner-square mask (0 disables max).")
    p.add_argument("--inner-square-fixed-px", type=int, default=0,
                   help="Fixed inner-square side in pixels (overrides factor when >0).")

    p.add_argument("--encoder", type=str, default="uni2-h")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "dinov2_hub", "timm_hf", "hf_transformers"])
    p.add_argument("--pooling", type=str, default="auto", choices=["auto", "cls", "mean_patch", "mean_all", "cls_mean_concat"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--hf-token", type=str, default=None)

    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--torch-threads", type=int, default=16)

    p.add_argument("--rows-per-csv", type=int, default=10000)
    p.add_argument("--mask-block", type=int, default=4096)
    p.add_argument("--max-cells", type=int, default=0,
                   help="Process only the first N labels after filtering; intended for quick validation only.")
    p.add_argument("--disable-grid-resume", action="store_true",
                   help="Disable grid-level resume from completed UNI2 shard folders.")

    # NEW: global scaling sampling
    p.add_argument("--scale-samples", type=int, default=200)
    p.add_argument("--scale-tile", type=int, default=512)
    p.add_argument("--scale-seed", type=int, default=0)
    p.add_argument("--scale-plo", type=float, default=1.0)
    p.add_argument("--scale-phi", type=float, default=99.0)

    return p.parse_args()


def main():
    args = parse_args()
    paired_mode = str(args.paired_inner_square_mode)
    print(f"[INFO] paired_inner_square_mode={paired_mode}", flush=True)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    secondary_outdir = Path(args.paired_inner_square_outdir).resolve() if args.paired_inner_square_outdir else None
    if secondary_outdir is not None:
        secondary_outdir.mkdir(parents=True, exist_ok=True)

    gr, gc = args.grid.lower().split("x")
    GR, GC = int(gr), int(gc)

    tiles_root = Path(args.tiles_root).resolve() if args.tiles_root else (outdir / "tiles")
    if args.save_tiles:
        tiles_root.mkdir(parents=True, exist_ok=True)

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    requested_device = str(args.device).strip().lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif requested_device == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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

    detected_source_mpp = infer_source_mpp(str(img_path))
    source_mpp = detected_source_mpp if detected_source_mpp and detected_source_mpp > 0 else float(args.default_source_mpp)
    extraction_tile_size, effective_mpp = resolve_extraction_tile_size(
        model_tile_size=int(args.tile_size),
        source_mpp=source_mpp,
        target_mpp=float(args.target_mpp),
    )
    inner_scale = float(extraction_tile_size) / float(max(1, int(args.tile_size)))
    inner_square_fixed_px = int(round(int(args.inner_square_fixed_px) * inner_scale)) if int(args.inner_square_fixed_px) > 0 else 0
    inner_square_min_px = int(round(int(args.inner_square_min_px) * inner_scale)) if int(args.inner_square_min_px) > 0 else 0
    inner_square_max_px = int(round(int(args.inner_square_max_px) * inner_scale)) if int(args.inner_square_max_px) > 0 else 0
    print(
        "[INFO] UNI2 MPP calibration: "
        f"detected_source_mpp={detected_source_mpp} source_mpp={source_mpp:.6f} "
        f"target_mpp={float(args.target_mpp):.6f} effective_mpp={effective_mpp} "
        f"model_tile_size={int(args.tile_size)} extraction_tile_size={int(extraction_tile_size)}",
        flush=True,
    )

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
    if int(args.max_cells) > 0:
        n_keep = min(int(args.max_cells), labels.shape[0])
        labels, cx, cy, area = labels[:n_keep], cx[:n_keep], cy[:n_keep], area[:n_keep]
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

    half = extraction_tile_size // 2

    def init_writer_state(tile_out: Path) -> Dict[str, Any]:
        return {
            "tile_out": tile_out,
            "row_parts": [],
            "row_count": 0,
            "total_rows": 0,
            "shard_idx": 0,
            "feat_dim": None,
            "feat_cols": None,
        }

    def append_writer_rows(state: Dict[str, Any], meta_df: pd.DataFrame, feats_np: np.ndarray):
        if feats_np is None:
            return
        if state["feat_dim"] is None:
            state["feat_dim"] = int(feats_np.shape[1])
            state["feat_cols"] = [f"feat_{j+1}" for j in range(state["feat_dim"])]
        feat_df = pd.DataFrame(feats_np, columns=state["feat_cols"])
        state["row_parts"].append(pd.concat([meta_df.reset_index(drop=True), feat_df], axis=1))
        state["row_count"] += feats_np.shape[0]
        state["total_rows"] += feats_np.shape[0]

    def flush_writer(state: Dict[str, Any]):
        if state["row_count"] == 0:
            return
        df = pd.concat(state["row_parts"], ignore_index=True)
        out_path = state["tile_out"] / f"{tag}_embeddings_shard{state['shard_idx']:04d}.csv.gz"
        df.to_csv(out_path, index=False, compression="gzip")
        state["row_parts"] = []
        state["row_count"] = 0
        state["shard_idx"] += 1

    def grid_marker_path(tile_out: Path) -> Path:
        return tile_out / f".{tag}_grid_complete.json"

    def grid_marker_matches_current_mode(marker_path: Path) -> bool:
        try:
            marker = json.loads(marker_path.read_text())
        except Exception:
            return False
        return str(marker.get("paired_inner_square_mode", "")) == paired_mode

    def grid_is_complete(tile_out: Path, secondary_tile_out: Optional[Path]) -> bool:
        primary_marker = grid_marker_path(tile_out)
        if not primary_marker.exists():
            return False
        if secondary_tile_out is not None and not grid_marker_matches_current_mode(primary_marker):
            return False
        if not any(tile_out.glob(f"{tag}_embeddings_shard*.csv.gz")):
            return False
        if secondary_tile_out is not None:
            secondary_marker = grid_marker_path(secondary_tile_out)
            if not secondary_marker.exists():
                return False
            if not grid_marker_matches_current_mode(secondary_marker):
                return False
            if not any(secondary_tile_out.glob(f"{tag}_embeddings_shard*.csv.gz")):
                return False
        return True

    def reset_incomplete_grid(tile_out: Path, secondary_tile_out: Optional[Path], tile_tiles: Optional[Path]) -> None:
        for path in (tile_out, secondary_tile_out, tile_tiles):
            if path is not None and path.exists():
                shutil.rmtree(path)
            if path is not None:
                path.mkdir(parents=True, exist_ok=True)

    def write_grid_marker(state: Dict[str, Any], tr: int, tc: int, start_idx: int, end_idx: int) -> None:
        marker = {
            "grid_r": int(tr),
            "grid_c": int(tc),
            "index_start": int(start_idx),
            "index_end": int(end_idx),
            "rows_written": int(state["total_rows"]),
            "shards": int(state["shard_idx"]),
            "encoder": str(args.encoder),
            "tag": str(tag),
            "paired_inner_square_mode": str(paired_mode),
            "completed_at_unix": time.time(),
        }
        grid_marker_path(state["tile_out"]).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")

    batch_primary_items = []
    batch_secondary_items = []
    batch_primary_meta = []
    batch_secondary_meta = []

    def prepare_model_item(tile_rgb_u8: np.ndarray):
        pil = Image.fromarray(tile_rgb_u8, mode="RGB")
        if backend_used == "hf_transformers":
            return pil
        return transform(pil)

    def add_example(tile_rgb_u8: np.ndarray,
                    meta: Dict[str, Any],
                    secondary_rgb_u8: Optional[np.ndarray] = None,
                    secondary_meta: Optional[Dict[str, Any]] = None):
        batch_primary_items.append(prepare_model_item(tile_rgb_u8))
        batch_primary_meta.append(meta)
        if secondary_outdir is not None and paired_mode == "masked_forward":
            if secondary_rgb_u8 is None:
                raise RuntimeError("paired masked_forward mode requires a secondary inner-square image")
            batch_secondary_items.append(prepare_model_item(secondary_rgb_u8))
            batch_secondary_meta.append(secondary_meta if secondary_meta is not None else dict(meta))

    def forward_items(items: List[Any], need_tokens: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if backend_used == "hf_transformers":
            inputs = processor(list(items), return_tensors="pt")
            images = inputs["pixel_values"].to(device, non_blocking=True)
            outputs = model(pixel_values=images)
            raw_feats = getattr(outputs, "last_hidden_state", None)
            if raw_feats is None:
                raw_feats = outputs[0]
        else:
            images = torch.stack(items, dim=0).to(device, non_blocking=True)
            raw_feats = model.forward_features(images) if need_tokens and hasattr(model, "forward_features") else model(images)

        cls_tokens, patch_tokens = extract_cls_and_patch_tokens(raw_feats)
        pooled = pool_from_token_parts(cls_tokens, patch_tokens, pooling_used)
        return pooled, cls_tokens, patch_tokens

    def run_batch(primary_state: Dict[str, Any], secondary_state: Optional[Dict[str, Any]]):
        nonlocal batch_primary_items, batch_secondary_items, batch_primary_meta, batch_secondary_meta
        if not batch_primary_items:
            return

        with torch.inference_mode():
            if secondary_state is not None and paired_mode == "masked_forward":
                n_primary = len(batch_primary_items)
                if len(batch_secondary_items) != n_primary:
                    raise RuntimeError("paired masked_forward mode lost primary/secondary batch alignment")
                pooled, _, _ = forward_items(batch_primary_items + batch_secondary_items, need_tokens=False)
                pooled_np = pooled.detach().cpu().numpy()
                tile_feats = pooled_np[:n_primary]
                inner_square_style_feats = pooled_np[n_primary:]
            else:
                need_tokens = secondary_state is not None and paired_mode == "token_subset"
                pooled, cls_tokens, patch_tokens = forward_items(batch_primary_items, need_tokens=need_tokens)
                tile_feats = pooled.detach().cpu().numpy()
                inner_square_style_feats = None
                if secondary_state is not None:
                    inner_square_style_feats = derive_inner_square_style_embeddings(
                        cls=cls_tokens,
                        patch=patch_tokens,
                        batch_meta=batch_primary_meta,
                        tile_size=int(args.tile_size),
                        img_size=int(args.img_size),
                        pooling_mode=pooling_used,
                        fixed_px=int(args.inner_square_fixed_px),
                        factor=float(args.inner_square_factor),
                        min_px=int(args.inner_square_min_px),
                        max_px=int(args.inner_square_max_px),
                    )
                    if inner_square_style_feats is None:
                        raise RuntimeError(
                            "Combined tile+inner_square token_subset mode requires patch-token access; "
                            f"backend={backend_used} encoder={args.encoder} did not expose compatible tokens."
                        )

        primary_meta_df = pd.DataFrame.from_records(batch_primary_meta)
        append_writer_rows(primary_state, primary_meta_df, tile_feats)
        if secondary_state is not None:
            secondary_records = batch_secondary_meta if paired_mode == "masked_forward" else batch_primary_meta
            secondary_meta_df = pd.DataFrame.from_records(secondary_records)
            append_writer_rows(secondary_state, secondary_meta_df, inner_square_style_feats)

        batch_primary_items, batch_secondary_items = [], []
        batch_primary_meta, batch_secondary_meta = [], []
        if primary_state["row_count"] >= args.rows_per_csv:
            flush_writer(primary_state)
        if secondary_state is not None and secondary_state["row_count"] >= args.rows_per_csv:
            flush_writer(secondary_state)

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
            secondary_tile_out = tile_folder(secondary_outdir, tr, tc) if secondary_outdir is not None else None
            tile_tiles = tile_folder(tiles_root, tr, tc) if args.save_tiles else None

            resume_enabled = not bool(args.disable_grid_resume)
            if resume_enabled and grid_is_complete(tile_out, secondary_tile_out):
                print(f"[INFO] UNI2 resume: skipping completed grid {tr:02d},{tc:02d}", flush=True)
                continue
            reset_incomplete_grid(tile_out, secondary_tile_out, tile_tiles)

            primary_state = init_writer_state(tile_out)
            secondary_state = init_writer_state(secondary_tile_out) if secondary_tile_out is not None else None

            for i in tqdm(range(s, e), desc=f"grid {tr:02d},{tc:02d}", leave=False):
                lab = int(labels[i])
                x = int(round(float(cx[i])))
                y = int(round(float(cy[i])))
                a = int(area[i])

                if not (x0_core <= x < x1_core and y0_core <= y < y1_core):
                    continue

                x0 = x - half
                y0 = y - half

                img_tile_raw = img_reader.read(x0, y0, extraction_tile_size, extraction_tile_size)

                # FIX: consistent mapping to uint8 using global scaling
                img_tile = to_rgb_uint8_global(img_tile_raw, lo=lo, hi=hi)
                secondary_img_tile = None

                if args.zero_outside_mask:
                    m_tile = mask_reader.read(x0, y0, extraction_tile_size, extraction_tile_size).astype(np.int64, copy=False)
                    keep_label = (m_tile == lab)
                    keep_square = build_inner_square_mask(
                        tile_size=extraction_tile_size,
                        center_x_local=int(x - x0),
                        center_y_local=int(y - y0),
                        area_px=int(a),
                        fixed_px=int(inner_square_fixed_px),
                        factor=float(args.inner_square_factor),
                        min_px=int(inner_square_min_px),
                        max_px=int(inner_square_max_px),
                    )

                    if args.mask_context_mode == "none":
                        keep_m = np.ones_like(keep_label, dtype=bool)
                    elif args.mask_context_mode == "inner_square":
                        keep_m = keep_square
                    elif args.mask_context_mode == "label_and_inner_square":
                        keep_m = keep_label & keep_square
                    else:
                        keep_m = keep_label

                    # Avoid generating fully blank tiles due to empty intersections.
                    if not np.any(keep_m):
                        if np.any(keep_label):
                            keep_m = keep_label
                        elif np.any(keep_square):
                            keep_m = keep_square
                        else:
                            keep_m = np.ones_like(keep_label, dtype=bool)

                    fill = int(args.outside_fill)
                    fill = 0 if fill < 0 else (255 if fill > 255 else fill)
                    img_tile = img_tile.copy()
                    img_tile[~keep_m, :] = fill

                secondary_center = None
                if secondary_state is not None and paired_mode == "masked_forward":
                    sec_x = x
                    sec_y = y
                    sec_a = a
                    sec_mask_path = mask_path

                    sec_x0 = sec_x - half
                    sec_y0 = sec_y - half
                    if sec_x0 == x0 and sec_y0 == y0 and not args.zero_outside_mask:
                        secondary_img_tile = img_tile.copy()
                    else:
                        secondary_raw = img_reader.read(sec_x0, sec_y0, extraction_tile_size, extraction_tile_size)
                        secondary_img_tile = to_rgb_uint8_global(secondary_raw, lo=lo, hi=hi)
                    keep_square = build_inner_square_mask(
                        tile_size=extraction_tile_size,
                        center_x_local=int(sec_x - sec_x0),
                        center_y_local=int(sec_y - sec_y0),
                        area_px=int(sec_a),
                        fixed_px=int(inner_square_fixed_px),
                        factor=float(args.inner_square_factor),
                        min_px=int(inner_square_min_px),
                        max_px=int(inner_square_max_px),
                    )
                    fill = 255
                    secondary_img_tile[~keep_square, :] = fill
                    secondary_center = {
                        "cx": int(sec_x),
                        "cy": int(sec_y),
                        "area_px": int(sec_a),
                        "area_model_px": int(round(float(sec_a) / max(inner_scale * inner_scale, 1e-6))),
                        "tile_x0": int(sec_x0),
                        "tile_y0": int(sec_y0),
                        "grid_r": int(np.clip(sec_y // tile_h, 0, GR - 1)),
                        "grid_c": int(np.clip(sec_x // tile_w, 0, GC - 1)),
                        "mask_path": str(sec_mask_path),
                    }

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
                    "area_model_px": int(round(float(a) / max(inner_scale * inner_scale, 1e-6))),
                    "tile_x0": int(x0),
                    "tile_y0": int(y0),
                    "model_tile_size": int(args.tile_size),
                    "extraction_tile_size": int(extraction_tile_size),
                    "source_mpp": float(source_mpp) if source_mpp else None,
                    "target_mpp": float(args.target_mpp) if float(args.target_mpp) > 0 else None,
                    "effective_mpp": float(effective_mpp) if effective_mpp else None,
                    "grid_r": int(tr),
                    "grid_c": int(tc),
                    "tile_path": tile_path,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "image_level": int(args.image_level),
                    "zero_outside_mask": bool(args.zero_outside_mask),
                    "mask_context_mode": str(args.mask_context_mode),
                }

                secondary_meta = None
                if secondary_state is not None and paired_mode == "masked_forward":
                    secondary_meta = dict(meta)
                    if secondary_center is not None:
                        secondary_meta.update(secondary_center)
                    secondary_meta["tile_path"] = ""
                    secondary_meta["zero_outside_mask"] = True
                    secondary_meta["mask_context_mode"] = "inner_square"
                    secondary_meta["outside_fill"] = 255

                add_example(img_tile, meta, secondary_img_tile, secondary_meta)
                if len(batch_primary_items) >= int(args.batch):
                    run_batch(primary_state, secondary_state)

            run_batch(primary_state, secondary_state)
            flush_writer(primary_state)
            if secondary_state is not None:
                flush_writer(secondary_state)
            write_grid_marker(primary_state, tr, tc, s, e)
            if secondary_state is not None:
                write_grid_marker(secondary_state, tr, tc, s, e)

    print("🎉 Done.")


if __name__ == "__main__":
    main()
