#!/usr/bin/env python3
import argparse
import contextlib
import csv
import gc
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image
import tifffile
import torch
from torch import nn
import zarr
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects

try:
    import pyvips
except Exception:
    pyvips = None

if pyvips is not None:
    try:
        pyvips.cache_set_max(0)
        pyvips.cache_set_max_files(0)
        pyvips.cache_set_max_mem(64 * 1024 * 1024)
    except Exception:
        pass


CHANNEL_NAMES = [
    "DAPI",
    "TRITC",
    "Cy5",
    "PD-1",
    "CD14",
    "CD4",
    "T-bet",
    "CD34",
    "CD68",
    "CD16",
    "CD11c",
    "CD138",
    "CD20",
    "CD3",
    "CD8",
    "PD-L1",
    "CK",
    "Ki67",
    "Tryptase",
    "Actin-D",
    "Caspase3-D",
    "PHH3-B",
    "Transgelin",
]

CHANNEL_INDEX_BY_NAME = {name: idx for idx, name in enumerate(CHANNEL_NAMES)}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VGGBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class GigaTIMEModel(nn.Module):
    def __init__(self, num_classes: int = 23, input_channels: int = 3):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(x0_4)


def normalize_to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3]
        elif arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr[:3], 0, -1)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[0] == 1:
            arr = np.repeat(np.moveaxis(arr, 0, -1), 3, axis=-1)
        else:
            raise ValueError(f"Unsupported crop image shape: {arr.shape}")
    else:
        raise ValueError(f"Unsupported crop image shape: {arr.shape}")

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32, copy=False)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for ch in range(arr.shape[-1]):
        plane = arr[..., ch]
        finite = np.isfinite(plane)
        if not finite.any():
            continue
        vals = plane[finite]
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        if hi <= lo:
            scaled = np.zeros_like(plane, dtype=np.float32)
        else:
            scaled = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
        out[..., ch] = np.round(scaled * 255.0).astype(np.uint8)
    return out


def _safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        num, den = value
        if float(den) == 0.0:
            return None
        return float(num) / float(den)
    try:
        return float(value)
    except Exception:
        return None


def _read_mpp_from_tiff(path: str) -> float | None:
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
                vals = []
                for pattern in (
                    r'PhysicalSizeX="([0-9]+(?:\.[0-9]+)?)"',
                    r'PhysicalSizeY="([0-9]+(?:\.[0-9]+)?)"',
                    r"MPP\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                ):
                    m = re.search(pattern, desc)
                    if m:
                        vals.append(float(m.group(1)))
                if vals:
                    return float(sum(vals) / len(vals))

            unit = None
            try:
                unit = int(tags["ResolutionUnit"].value)
            except Exception:
                unit = None

            if unit in unit_to_um:
                xres = None
                yres = None
                try:
                    xres = _safe_float(tags["XResolution"].value)
                except Exception:
                    xres = None
                try:
                    yres = _safe_float(tags["YResolution"].value)
                except Exception:
                    yres = None

                vals = []
                if xres and xres > 0:
                    vals.append(unit_to_um[unit] / xres)
                if yres and yres > 0:
                    vals.append(unit_to_um[unit] / yres)
                if vals:
                    return float(sum(vals) / len(vals))
    except Exception:
        return None
    return None


def _read_mpp_from_shift(shift_path: Path) -> float | None:
    try:
        shift = json.loads(shift_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("source_mpp", "mpp", "microns_per_pixel", "microns_per_pixel_x"):
        value = _safe_float(shift.get(key))
        if value and value > 0:
            return float(value)
    return None


def infer_source_mpp(crop_path: str, shift_json: str | None = None) -> float | None:
    crop_file = Path(crop_path)
    crop_candidates = [crop_file]
    try:
        resolved_crop = crop_file.resolve()
        if resolved_crop != crop_file:
            crop_candidates.append(resolved_crop)
    except Exception:
        resolved_crop = crop_file

    for cand in crop_candidates:
        direct = _read_mpp_from_tiff(str(cand))
        if direct:
            print(f"[INFO] GigaTIME source MPP recovered from crop TIFF {cand}", flush=True)
            return direct

    shift_candidates: list[Path] = []
    if shift_json and str(shift_json).strip():
        shift_candidates.append(Path(shift_json))
    for cand in crop_candidates:
        shift_candidates.append(cand.with_name("shift.json"))

    seen_shift: set[Path] = set()
    for raw_shift in shift_candidates:
        try:
            shift_path = raw_shift.resolve()
        except Exception:
            shift_path = raw_shift
        if shift_path in seen_shift or not shift_path.exists():
            continue
        seen_shift.add(shift_path)

        shifted_mpp = _read_mpp_from_shift(shift_path)
        if shifted_mpp:
            print(f"[INFO] GigaTIME source MPP recovered from shift metadata {shift_path}", flush=True)
            return shifted_mpp

        try:
            shift = json.loads(shift_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        input_image = shift.get("input_image")
        if not input_image:
            continue

        input_path = Path(str(input_image))
        candidates: list[Path] = []
        if input_path.is_absolute():
            candidates.append(input_path)
        else:
            for parent in [shift_path.parent, shift_path.parent.parent, *resolved_crop.parents]:
                candidates.append(parent / input_path)
                try:
                    candidates.extend(parent.glob(f"input*/{input_path.name}"))
                except Exception:
                    pass

        seen_images: set[Path] = set()
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except Exception:
                resolved = cand
            if resolved in seen_images or not resolved.exists():
                continue
            seen_images.add(resolved)
            mpp = _read_mpp_from_tiff(str(resolved))
            if mpp:
                print(f"[INFO] GigaTIME source MPP recovered from original image {resolved}", flush=True)
                return mpp

    return None


def estimate_prediction_gib(
    height: int,
    width: int,
    num_channels: int = len(CHANNEL_NAMES),
    *,
    bytes_per_sample: int | None = None,
) -> float:
    if bytes_per_sample is None:
        bytes_per_sample = np.dtype(np.float32).itemsize
    bytes_total = int(num_channels) * int(height) * int(width) * int(bytes_per_sample)
    return float(bytes_total) / float(1024 ** 3)


def resolve_output_channels(spec: str | None, *, default_all: bool = True) -> tuple[list[int], list[str]]:
    if not spec or not str(spec).strip():
        if not default_all:
            return [], []
        return list(range(len(CHANNEL_NAMES))), list(CHANNEL_NAMES)

    names: list[str] = []
    seen: set[str] = set()
    for raw in str(spec).split(","):
        name = raw.strip()
        if not name:
            continue
        key = next((cand for cand in CHANNEL_NAMES if cand.lower() == name.lower()), None)
        if key is None:
            raise ValueError(
                f"Unknown GigaTIME channel '{name}'. Available channels: {', '.join(CHANNEL_NAMES)}"
            )
        if key not in seen:
            names.append(key)
            seen.add(key)

    if not names:
        raise ValueError("No valid GigaTIME output channels were selected")
    return [CHANNEL_INDEX_BY_NAME[name] for name in names], names


def choose_downsample_factor(
    orig_h: int,
    orig_w: int,
    source_mpp: float | None,
    target_mpp: float,
    auto_threshold_mpix: float,
    max_side: int,
    max_output_gib: float,
    *,
    num_channels: int,
    bytes_per_sample: int,
    strict_target_mpp: bool,
    enable_max_side_fallback: bool = True,
) -> tuple[int, dict]:
    factor = 1
    reason = "native"

    if source_mpp and source_mpp > 0 and target_mpp > 0:
        requested_factor = float(target_mpp) / float(source_mpp)
        factor = max(1, int(math.floor(requested_factor + 0.5)))
        reason = "target_mpp"
    else:
        if strict_target_mpp and target_mpp > 0:
            raise ValueError(
                "GigaTIME --strict-target-mpp was requested, but source MPP could not be resolved. "
                "Provide TIFF physical pixel-size metadata or stage/pass the StarDist shift.json so the "
                "original calibrated image can be found."
            )
        mpix = (orig_h * orig_w) / 1_000_000.0
        if enable_max_side_fallback and (mpix > float(auto_threshold_mpix) or max(orig_h, orig_w) > int(max_side)):
            factor = max(1, int(math.ceil(max(orig_h, orig_w) / float(max_side))))
            reason = "max_side_fallback"

    if not (strict_target_mpp and source_mpp and source_mpp > 0 and target_mpp > 0):
        while True:
            h = int(math.ceil(orig_h / float(factor)))
            w = int(math.ceil(orig_w / float(factor)))
            est_gib = estimate_prediction_gib(
                h,
                w,
                num_channels=num_channels,
                bytes_per_sample=bytes_per_sample,
            )
            if est_gib <= float(max_output_gib):
                break
            factor += 1
            reason = "disk_budget"

    final_h = int(math.ceil(orig_h / float(factor)))
    final_w = int(math.ceil(orig_w / float(factor)))
    meta = {
        "selected_factor": int(factor),
        "selected_shape_yx": [final_h, final_w],
        "estimated_prediction_gib": float(
            estimate_prediction_gib(
                final_h,
                final_w,
                num_channels=num_channels,
                bytes_per_sample=bytes_per_sample,
            )
        ),
        "selection_reason": reason,
    }
    if source_mpp and source_mpp > 0:
        meta["source_mpp"] = float(source_mpp)
        meta["effective_mpp"] = float(source_mpp * factor)
        if target_mpp > 0:
            meta["requested_downsample_factor"] = float(target_mpp) / float(source_mpp)
            meta["mpp_error_fraction"] = float((source_mpp * factor - target_mpp) / target_mpp)
    return factor, meta


def _read_page_lazy_downsampled(path: str, page: int, factor: int) -> tuple[np.ndarray, tuple[int, int], int]:
    if pyvips is not None:
        try:
            image_vips = pyvips.Image.new_from_file(path, access="random", page=int(page))
            width = int(image_vips.width)
            height = int(image_vips.height)
            target_width = max(1, int(round(width / max(1, factor))))
            thumb = pyvips.Image.thumbnail(path, target_width, page=int(page))
            bands = min(int(thumb.bands), 3)
            arr = np.ndarray(
                buffer=thumb.write_to_memory(),
                dtype=np.uint8,
                shape=(int(thumb.height), int(thumb.width), int(thumb.bands)),
            ).copy()
            if bands == 1:
                arr = arr[..., 0]
            elif bands > 3:
                arr = arr[..., :3]
            return normalize_to_uint8_rgb(arr), (height, width), factor
        except Exception:
            pass

    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--page {page} out of range for {path} ({len(tf.pages)} pages)")
        store = tf.pages[page].aszarr()
        arr = zarr.open(store, mode="r")
        shape = tuple(int(v) for v in arr.shape)

        if arr.ndim == 2:
            sampled = np.asarray(arr[::factor, ::factor])
        elif arr.ndim == 3:
            if arr.shape[-1] in (1, 3, 4):
                sampled = np.asarray(arr[::factor, ::factor, ...])
            elif arr.shape[0] in (1, 3, 4):
                sampled = np.asarray(arr[..., ::factor, ::factor])
            else:
                raise ValueError(f"Unsupported crop image shape: {shape}")
        else:
            raise ValueError(f"Unsupported crop image shape: {shape}")
    return normalize_to_uint8_rgb(sampled), shape[:2] if len(shape) >= 2 else shape, factor


def inspect_crop_image(
    path: str,
    page: int,
    auto_threshold_mpix: float,
    max_side: int,
    target_mpp: float,
    max_output_gib: float,
    *,
    num_output_channels: int,
    bytes_per_sample: int,
    strict_target_mpp: bool,
    shift_json: str | None = None,
    enable_max_side_fallback: bool = True,
) -> dict:
    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--page {page} out of range for {path} ({len(tf.pages)} pages)")
        page_shape = tuple(int(v) for v in tf.pages[page].shape)

    if len(page_shape) == 2:
        orig_h, orig_w = page_shape
    elif len(page_shape) == 3:
        if page_shape[-1] in (1, 3, 4):
            orig_h, orig_w = page_shape[0], page_shape[1]
        elif page_shape[0] in (1, 3, 4):
            orig_h, orig_w = page_shape[1], page_shape[2]
        else:
            raise ValueError(f"Unsupported crop image shape: {page_shape}")
    else:
        raise ValueError(f"Unsupported crop image shape: {page_shape}")

    source_mpp = infer_source_mpp(path, shift_json=shift_json)
    factor, selection = choose_downsample_factor(
        orig_h=orig_h,
        orig_w=orig_w,
        source_mpp=source_mpp,
        target_mpp=target_mpp,
        auto_threshold_mpix=auto_threshold_mpix,
        max_side=max_side,
        max_output_gib=max_output_gib,
        num_channels=num_output_channels,
        bytes_per_sample=bytes_per_sample,
        strict_target_mpp=strict_target_mpp,
        enable_max_side_fallback=enable_max_side_fallback,
    )
    meta = {
        "original_shape_yx": [int(orig_h), int(orig_w)],
        "inference_shape_yx": [int(math.ceil(orig_h / float(factor))), int(math.ceil(orig_w / float(factor)))],
        "downsample_factor": int(factor),
        "downsample_applied": bool(factor > 1),
        "target_mpp": float(target_mpp) if target_mpp > 0 else None,
        "max_output_gib": float(max_output_gib),
    }
    meta.update(selection)
    return meta


def read_crop_image(
    path: str,
    page: int,
    plan: dict,
) -> np.ndarray:
    factor = int(plan["downsample_factor"])
    if factor > 1:
        return _read_page_lazy_downsampled(path, page, factor)[0]

    with tifffile.TiffFile(path) as tf:
        arr = tf.pages[page].asarray()
    return normalize_to_uint8_rgb(arr)


class LazyCropReader:
    def __init__(self, path: str, page: int, factor: int):
        self.path = path
        self.page = int(page)
        self.factor = max(1, int(factor))
        self.vips_image = None
        use_pyvips = pyvips is not None
        if use_pyvips:
            try:
                with tifffile.TiffFile(path) as tf:
                    page_shape = tuple(int(v) for v in tf.pages[self.page].shape)
                if len(page_shape) >= 2:
                    if len(page_shape) == 3 and page_shape[-1] in (1, 3, 4):
                        pixel_count = int(page_shape[0]) * int(page_shape[1])
                    elif len(page_shape) == 3 and page_shape[0] in (1, 3, 4):
                        pixel_count = int(page_shape[1]) * int(page_shape[2])
                    else:
                        pixel_count = int(page_shape[0]) * int(page_shape[1])
                    use_pyvips = pixel_count <= int(os.environ.get("GIGATIME_PYVIPS_MAX_PIXELS", "1000000000"))
            except Exception:
                use_pyvips = pyvips is not None
        if use_pyvips:
            try:
                self.vips_image = pyvips.Image.new_from_file(path, access="random", page=self.page)
            except Exception:
                self.vips_image = None

        if self.vips_image is not None:
            self.orig_w = int(self.vips_image.width)
            self.orig_h = int(self.vips_image.height)
            self.shape = (self.orig_h, self.orig_w, int(self.vips_image.bands))
            self.layout = "YXC"
            self.final_h = int(math.ceil(self.orig_h / float(self.factor)))
            self.final_w = int(math.ceil(self.orig_w / float(self.factor)))
            self.tf = None
            self.page_obj = None
            self.arr = None
            return

        self.tf = tifffile.TiffFile(path)
        if self.page < 0 or self.page >= len(self.tf.pages):
            raise ValueError(f"--page {self.page} out of range for {path} ({len(self.tf.pages)} pages)")
        self.page_obj = self.tf.pages[self.page]
        self.arr = zarr.open(self.page_obj.aszarr(), mode="r")
        self.shape = tuple(int(v) for v in self.arr.shape)

        if len(self.shape) == 2:
            self.orig_h, self.orig_w = self.shape
            self.layout = "YX"
        elif len(self.shape) == 3:
            if self.shape[-1] in (1, 3, 4):
                self.orig_h, self.orig_w = self.shape[0], self.shape[1]
                self.layout = "YXC"
            elif self.shape[0] in (1, 3, 4):
                self.orig_h, self.orig_w = self.shape[1], self.shape[2]
                self.layout = "CYX"
            else:
                raise ValueError(f"Unsupported crop image shape: {self.shape}")
        else:
            raise ValueError(f"Unsupported crop image shape: {self.shape}")

        self.final_h = int(math.ceil(self.orig_h / float(self.factor)))
        self.final_w = int(math.ceil(self.orig_w / float(self.factor)))

    def close(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyCropReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _read_native_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if self.vips_image is not None:
            region = self.vips_image.crop(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
            arr = np.ndarray(
                buffer=region.write_to_memory(),
                dtype=np.uint8,
                shape=(int(region.height), int(region.width), int(region.bands)),
            ).copy()
            if arr.shape[-1] == 1:
                return arr[..., 0]
            if arr.shape[-1] > 3:
                return arr[..., :3]
            return arr
        if self.layout == "YX":
            return np.asarray(self.arr[y0:y1, x0:x1])
        if self.layout == "YXC":
            return np.asarray(self.arr[y0:y1, x0:x1, ...])
        if self.layout == "CYX":
            return np.asarray(self.arr[..., y0:y1, x0:x1])
        raise ValueError(f"Unsupported layout: {self.layout}")

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        if y1 <= y0 or x1 <= x0:
            raise ValueError(f"Invalid region: {(y0, y1, x0, x1)}")

        ry0 = max(0, int(y0))
        rx0 = max(0, int(x0))
        ry1 = min(self.final_h, int(y1))
        rx1 = min(self.final_w, int(x1))

        sy0 = ry0 * self.factor
        sx0 = rx0 * self.factor
        sy1 = min(self.orig_h, ry1 * self.factor)
        sx1 = min(self.orig_w, rx1 * self.factor)
        native = self._read_native_region(sy0, sy1, sx0, sx1)

        if self.factor > 1:
            if native.ndim == 2:
                sampled = native[::self.factor, ::self.factor]
            elif native.ndim == 3 and self.layout == "YXC":
                sampled = native[::self.factor, ::self.factor, ...]
            elif native.ndim == 3 and self.layout == "CYX":
                sampled = native[..., ::self.factor, ::self.factor]
            else:
                raise ValueError(f"Unsupported sampled block shape: {native.shape}")
        else:
            sampled = native

        block = normalize_to_uint8_rgb(sampled)
        want_h = int(y1 - y0)
        want_w = int(x1 - x0)
        pad_h = want_h - block.shape[0]
        pad_w = want_w - block.shape[1]
        if pad_h > 0 or pad_w > 0:
            mode = "reflect" if block.shape[0] > 1 and block.shape[1] > 1 else "edge"
            block = np.pad(block, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)), mode=mode)
        return block[:want_h, :want_w, :]


class LazyMaskReader:
    def __init__(self, path: str):
        self.path = str(path)
        self.tf = tifffile.TiffFile(self.path)
        self.series = self.tf.series[0]
        arr = zarr.open(self.series.aszarr(), mode="r")
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError(f"Unsupported mask shape after squeeze: {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"Unsupported mask shape after squeeze: {arr.shape}")
        self.arr = arr
        self.height = int(arr.shape[0])
        self.width = int(arr.shape[1])

    def close(self) -> None:
        try:
            self.tf.close()
        except Exception:
            pass

    def __enter__(self) -> "LazyMaskReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read_block(self, y0: int, y1: int, x0: int, x1: int, target_shape: tuple[int, int]) -> np.ndarray:
        ty, tx = target_shape
        if (self.height, self.width) == target_shape:
            out = np.asarray(self.arr[y0:y1, x0:x1])
        else:
            y_idx = np.minimum(
                np.round(np.linspace(y0 * self.height / ty, (y1 - 1) * self.height / ty, y1 - y0)).astype(np.int64),
                self.height - 1,
            )
            x_idx = np.minimum(
                np.round(np.linspace(x0 * self.width / tx, (x1 - 1) * self.width / tx, x1 - x0)).astype(np.int64),
                self.width - 1,
            )
            out = np.asarray(self.arr.oindex[y_idx, x_idx])
        out = np.squeeze(out)
        if out.ndim != 2:
            raise ValueError(f"Unsupported mask block shape after squeeze: {out.shape}")
        if not np.issubdtype(out.dtype, np.integer):
            out = np.rint(out).astype(np.int64, copy=False)
        return out


def _scan_max_label(mask_reader: LazyMaskReader, target_shape: tuple[int, int], block_size: int) -> int:
    ty, tx = target_shape
    max_label = 0
    for y0 in range(0, ty, block_size):
        y1 = min(ty, y0 + block_size)
        for x0 in range(0, tx, block_size):
            x1 = min(tx, x0 + block_size)
            block = mask_reader.read_block(y0, y1, x0, x1, target_shape)
            if block.size:
                max_label = max(max_label, int(block.max()))
    return max_label


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["label_id", "area_px"])
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TileQuantifier:
    def __init__(
        self,
        *,
        mask_name: str,
        mask_path: str,
        target_shape: tuple[int, int],
        channel_names: list[str],
        block_size: int,
        max_label: int | None = None,
    ):
        self.mask_name = str(mask_name)
        self.mask_path = str(mask_path)
        self.target_shape = (int(target_shape[0]), int(target_shape[1]))
        self.channel_names = list(channel_names)
        self.mask_reader = LazyMaskReader(mask_path)
        self.max_label = int(max_label) if max_label is not None else _scan_max_label(
            self.mask_reader,
            self.target_shape,
            block_size,
        )
        self.enabled = self.max_label > 0
        self.counts = np.zeros(self.max_label + 1, dtype=np.int64) if self.enabled else None
        self.sum_y = np.zeros(self.max_label + 1, dtype=np.float64) if self.enabled else None
        self.sum_x = np.zeros(self.max_label + 1, dtype=np.float64) if self.enabled else None
        self.min_y = np.full(self.max_label + 1, np.iinfo(np.int64).max, dtype=np.int64) if self.enabled else None
        self.min_x = np.full(self.max_label + 1, np.iinfo(np.int64).max, dtype=np.int64) if self.enabled else None
        self.max_y = np.full(self.max_label + 1, -1, dtype=np.int64) if self.enabled else None
        self.max_x = np.full(self.max_label + 1, -1, dtype=np.int64) if self.enabled else None
        self.sums_by_channel = (
            np.zeros((len(self.channel_names), self.max_label + 1), dtype=np.float64)
            if self.enabled
            else None
        )
        self.max_by_channel = (
            np.full((len(self.channel_names), self.max_label + 1), -np.inf, dtype=np.float32)
            if self.enabled
            else None
        )

    def close(self) -> None:
        self.mask_reader.close()

    def accumulate_tile(self, y0: int, y1: int, x0: int, x1: int, tile_probs_cyx: np.ndarray) -> None:
        if not self.enabled:
            return
        mask_block = self.mask_reader.read_block(y0, y1, x0, x1, self.target_shape)
        positive = mask_block > 0
        if not np.any(positive):
            return

        labels = mask_block[positive].astype(np.int64, copy=False)
        self.counts += np.bincount(labels, minlength=self.max_label + 1)

        y_local, x_local = np.nonzero(positive)
        abs_y = y_local.astype(np.int64, copy=False) + int(y0)
        abs_x = x_local.astype(np.int64, copy=False) + int(x0)
        self.sum_y += np.bincount(labels, weights=abs_y.astype(np.float64, copy=False), minlength=self.max_label + 1)
        self.sum_x += np.bincount(labels, weights=abs_x.astype(np.float64, copy=False), minlength=self.max_label + 1)
        np.minimum.at(self.min_y, labels, abs_y)
        np.minimum.at(self.min_x, labels, abs_x)
        np.maximum.at(self.max_y, labels, abs_y)
        np.maximum.at(self.max_x, labels, abs_x)

        for ch_idx in range(tile_probs_cyx.shape[0]):
            values = tile_probs_cyx[ch_idx][positive].astype(np.float64, copy=False)
            self.sums_by_channel[ch_idx] += np.bincount(labels, weights=values, minlength=self.max_label + 1)
            np.maximum.at(self.max_by_channel[ch_idx], labels, values)

    def write_outputs(self, outdir: Path, sample_id: str) -> dict:
        outdir.mkdir(parents=True, exist_ok=True)

        labels_sorted = np.asarray([], dtype=np.int64)
        if self.enabled:
            labels_sorted = np.flatnonzero(self.counts > 0)
            labels_sorted = labels_sorted[labels_sorted > 0]

        quant_csv = outdir / f"{sample_id}_{self.mask_name}_gigatime_quantification.csv"
        mean_csv = outdir / f"{sample_id}_{self.mask_name}_gigatime_mean_intensity.csv"
        stats_csv = outdir / f"{sample_id}_{self.mask_name}_gigatime_intensity_stats.csv"
        summary_json = outdir / f"{sample_id}_{self.mask_name}_gigatime_intensity_summary.json"

        base_fields = [
            "label_id",
            "mask_name",
            "area_px",
            "centroid_y_px",
            "centroid_x_px",
            "bbox_ymin_px",
            "bbox_xmin_px",
            "bbox_ymax_px",
            "bbox_xmax_px",
        ]
        mean_fields = base_fields + [ch.strip() or "unnamed" for ch in self.channel_names]
        stats_fields = list(base_fields)
        quant_fields = list(base_fields)
        for ch_name in self.channel_names:
            safe_name = ch_name.strip() or "unnamed"
            stats_fields.extend([f"{safe_name}__mean", f"{safe_name}__sum", f"{safe_name}__max"])
            quant_fields.extend([f"{safe_name}__mean", f"{safe_name}__sum", f"{safe_name}__max"])

        object_count = int(labels_sorted.size)
        with quant_csv.open("w", newline="", encoding="utf-8") as quant_fh, \
                mean_csv.open("w", newline="", encoding="utf-8") as mean_fh, \
                stats_csv.open("w", newline="", encoding="utf-8") as stats_fh:
            quant_writer = csv.DictWriter(quant_fh, fieldnames=quant_fields)
            mean_writer = csv.DictWriter(mean_fh, fieldnames=mean_fields)
            stats_writer = csv.DictWriter(stats_fh, fieldnames=stats_fields)
            quant_writer.writeheader()
            mean_writer.writeheader()
            stats_writer.writeheader()

            for label_id_np in labels_sorted:
                label_id = int(label_id_np)
                count = int(self.counts[label_id])
                if count <= 0:
                    continue
                base = {
                    "label_id": label_id,
                    "mask_name": self.mask_name,
                    "area_px": count,
                    "centroid_y_px": float(self.sum_y[label_id] / count),
                    "centroid_x_px": float(self.sum_x[label_id] / count),
                    "bbox_ymin_px": int(self.min_y[label_id]),
                    "bbox_xmin_px": int(self.min_x[label_id]),
                    "bbox_ymax_px": int(self.max_y[label_id] + 1),
                    "bbox_xmax_px": int(self.max_x[label_id] + 1),
                }
                quant_row = dict(base)
                mean_row = dict(base)
                stats_row = dict(base)
                for ch_idx, ch_name in enumerate(self.channel_names):
                    safe_name = ch_name.strip() or "unnamed"
                    mean_val = float(self.sums_by_channel[ch_idx, label_id] / count)
                    sum_val = float(self.sums_by_channel[ch_idx, label_id])
                    max_val = float(self.max_by_channel[ch_idx, label_id])
                    mean_row[safe_name] = mean_val
                    stats_row[f"{safe_name}__mean"] = mean_val
                    stats_row[f"{safe_name}__sum"] = sum_val
                    stats_row[f"{safe_name}__max"] = max_val
                    quant_row[f"{safe_name}__mean"] = mean_val
                    quant_row[f"{safe_name}__sum"] = sum_val
                    quant_row[f"{safe_name}__max"] = max_val
                quant_writer.writerow(quant_row)
                mean_writer.writerow(mean_row)
                stats_writer.writerow(stats_row)

        summary = {
            "mask_name": self.mask_name,
            "mask": str(Path(self.mask_path).resolve()),
            "mask_shape_yx": [int(self.mask_reader.height), int(self.mask_reader.width)],
            "image_shape_cyx": [len(self.channel_names), int(self.target_shape[0]), int(self.target_shape[1])],
            "channel_names": list(self.channel_names),
            "objects_quantified": object_count,
            "quantification_csv": str(quant_csv.resolve()),
        }
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "mask_name": self.mask_name,
            "objects_quantified": object_count,
            "quant_csv": str(quant_csv),
            "mean_csv": str(mean_csv),
            "stats_csv": str(stats_csv),
            "summary_json": str(summary_json),
        }


class JpegTileExporter:
    def __init__(
        self,
        *,
        outdir: Path,
        full_shape_yx: tuple[int, int],
        tile_size: int,
        quality: int,
        preview_max_side: int,
        save_tiles: bool,
        channel_indices: list[int],
        channel_names: list[str],
    ):
        self.outdir = Path(outdir)
        self.full_shape_yx = (int(full_shape_yx[0]), int(full_shape_yx[1]))
        self.tile_size = int(tile_size)
        self.quality = int(quality)
        self.save_tiles = bool(save_tiles)
        self.channel_indices = list(channel_indices)
        self.channel_names = list(channel_names)
        self.tiles_dir = self.outdir / "jpg_tiles" if self.save_tiles else None
        self.image_dir = self.outdir / "jpg_image"
        if self.tiles_dir is not None:
            self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.saved_tiles = {name: 0 for name in self.channel_names}
        jpeg_limit = 65535
        effective_max_side = min(max(0, int(preview_max_side)), jpeg_limit)
        self.preview_scale = min(1.0, float(effective_max_side) / float(max(self.full_shape_yx))) if effective_max_side > 0 else 0.0
        self.preview_arrays = {}
        if self.preview_scale > 0.0:
            preview_h = max(1, int(round(self.full_shape_yx[0] * self.preview_scale)))
            preview_w = max(1, int(round(self.full_shape_yx[1] * self.preview_scale)))
            for name in self.channel_names:
                self.preview_arrays[name] = np.zeros((preview_h, preview_w), dtype=np.uint8)

    def save_tile(self, y0: int, y1: int, x0: int, x1: int, tile_probs_cyx: np.ndarray) -> None:
        for out_idx, channel_name in enumerate(self.channel_names):
            tile_uint8 = np.round(np.clip(tile_probs_cyx[out_idx], 0.0, 1.0) * 255.0).astype(np.uint8)
            if self.tiles_dir is not None and int(tile_uint8.max()) > 0:
                marker_dir = self.tiles_dir / channel_name
                marker_dir.mkdir(parents=True, exist_ok=True)
                tile_path = marker_dir / f"y{int(y0):06d}_x{int(x0):06d}.jpg"
                Image.fromarray(tile_uint8, mode="L").save(
                    tile_path,
                    format="JPEG",
                    quality=self.quality,
                    optimize=True,
                    progressive=True,
                )
                self.saved_tiles[channel_name] += 1
            if self.preview_scale > 0.0:
                py0 = int(round(y0 * self.preview_scale))
                py1 = int(round(y1 * self.preview_scale))
                px0 = int(round(x0 * self.preview_scale))
                px1 = int(round(x1 * self.preview_scale))
                py1 = max(py1, py0 + 1)
                px1 = max(px1, px0 + 1)
                preview = Image.fromarray(tile_uint8, mode="L").resize(
                    (px1 - px0, py1 - py0),
                    Image.Resampling.BILINEAR,
                )
                self.preview_arrays[channel_name][py0:py1, px0:px1] = np.asarray(preview, dtype=np.uint8)

    def write_outputs(self, metadata: dict) -> None:
        for channel_name, arr in self.preview_arrays.items():
            preview_path = self.image_dir / f"{channel_name}.jpg"
            Image.fromarray(arr, mode="L").save(
                preview_path,
                format="JPEG",
                quality=max(60, min(95, self.quality)),
                optimize=True,
                progressive=True,
            )
        manifest = {
            "image_shape_yx": [int(self.full_shape_yx[0]), int(self.full_shape_yx[1])],
            "tile_size": int(self.tile_size),
            "channels": list(self.channel_names),
            "jpeg_quality": int(self.quality),
            "full_image_scale": float(self.preview_scale),
            "save_tiles": bool(self.save_tiles),
            "saved_tiles": dict(self.saved_tiles),
            "metadata": metadata,
        }
        (self.outdir / "gigatime_jpg_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def output_dtype_info(output_dtype: str) -> tuple[np.dtype, int, float | None]:
    if output_dtype == "uint8":
        return np.dtype(np.uint8), np.dtype(np.uint8).itemsize, 255.0
    if output_dtype == "uint16":
        return np.dtype(np.uint16), np.dtype(np.uint16).itemsize, 65535.0
    if output_dtype == "float32":
        return np.dtype(np.float32), np.dtype(np.float32).itemsize, None
    raise ValueError(f"Unsupported output dtype: {output_dtype}")


def _system_mem_available_gib() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            values = {}
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    values[parts[0].rstrip(":")] = float(parts[1]) / (1024.0 * 1024.0)
            if values.get("MemAvailable"):
                return float(values["MemAvailable"])
            if values.get("MemFree"):
                return float(values["MemFree"])
    except Exception:
        return None
    return None


def _cuda_mem_free_gib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        return float(free_bytes) / (1024.0 ** 3), float(total_bytes) / (1024.0 ** 3)
    except Exception:
        return None, None


def adapt_gigatime_hardware(args: argparse.Namespace, device: torch.device) -> dict:
    mem_available = _system_mem_available_gib()
    cuda_free, cuda_total = _cuda_mem_free_gib(device)
    requested = {
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "max_output_gib": float(args.max_output_gib),
    }
    if not bool(args.auto_hardware):
        return {
            "enabled": False,
            "requested": requested,
            "effective": dict(requested),
            "system_mem_available_gib": mem_available,
            "cuda_mem_free_gib": cuda_free,
            "cuda_mem_total_gib": cuda_total,
        }

    task_memory = float(args.task_memory_gb) if float(args.task_memory_gb or 0) > 0 else None
    usable_mem = mem_available
    if task_memory is not None:
        usable_mem = min(usable_mem, task_memory) if usable_mem is not None else task_memory
    if usable_mem is not None:
        usable_mem = max(0.0, usable_mem - float(args.min_free_system_gb))

    if usable_mem is not None and usable_mem < 4.0:
        raise RuntimeError(
            "GigaTIME cannot run safely: estimated usable system memory after reserve is "
            f"{usable_mem:.2f} GiB. Increase RAM/swap or lower concurrent workload before running."
        )

    effective_batch = max(1, int(args.batch_size))
    effective_block = max(256, int(args.block_size))
    effective_budget = max(0.25, float(args.max_output_gib))

    if usable_mem is not None:
        if usable_mem < 8.0:
            effective_block = min(effective_block, 512)
            effective_budget = min(effective_budget, 0.75)
            effective_batch = 1
        elif usable_mem < 16.0:
            effective_block = min(effective_block, 768)
            effective_budget = min(effective_budget, 1.25)
            effective_batch = 1
        elif usable_mem < 24.0:
            effective_block = min(effective_block, 1024)
            effective_budget = min(effective_budget, 2.0)
            effective_batch = 1
        else:
            effective_block = min(effective_block, 1536)
            effective_budget = min(effective_budget, 4.0)

    if cuda_free is not None:
        if cuda_free < 4.0:
            effective_batch = 1
            effective_block = min(effective_block, 512)
        elif cuda_free < 8.0:
            effective_batch = min(effective_batch, 1)
            effective_block = min(effective_block, 1024)
        elif cuda_free < 12.0:
            effective_batch = min(effective_batch, 2)

    # TIFF/JPEG encoders are happiest with multiples of 16; Zarr also benefits from regular chunks.
    effective_block = max(256, int(effective_block))
    effective_block = max(256, (effective_block // 16) * 16)
    args.batch_size = int(effective_batch)
    args.block_size = int(effective_block)
    args.max_output_gib = float(effective_budget)

    return {
        "enabled": True,
        "requested": requested,
        "effective": {
            "batch_size": int(args.batch_size),
            "block_size": int(args.block_size),
            "max_output_gib": float(args.max_output_gib),
        },
        "system_mem_available_gib": mem_available,
        "task_memory_gb": task_memory,
        "min_free_system_gb": float(args.min_free_system_gb),
        "usable_system_mem_gib": usable_mem,
        "cuda_mem_free_gib": cuda_free,
        "cuda_mem_total_gib": cuda_total,
    }


def _zarr_directory_store(path: Path):
    if hasattr(zarr, "DirectoryStore"):
        return zarr.DirectoryStore(str(path))
    if hasattr(zarr.storage, "DirectoryStore"):
        return zarr.storage.DirectoryStore(str(path))
    if hasattr(zarr.storage, "LocalStore"):
        return zarr.storage.LocalStore(str(path))
    raise RuntimeError("No supported directory-backed Zarr store implementation found")


def _zarr_compressor():
    try:
        return zarr.Blosc(cname="zstd", clevel=9, shuffle=2)
    except Exception:
        return None


def _zarr_root_attrs(metadata: dict, channel_names: list[str]) -> dict:
    channel_labels = [{"label": name, "active": True} for name in channel_names]
    root_attrs = {
        "multiscales": [{
            "version": "0.4",
            "name": "gigatime_probs",
            "datasets": [{"path": "0"}],
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }],
        "omero": {
            "name": "GigaTIME probabilities",
            "channels": channel_labels,
        },
        "gigatime": metadata,
    }
    return root_attrs


def build_background_skip_mask(
    source_path: str,
    page: int,
    factor: int,
    final_h: int,
    final_w: int,
    *,
    downsample: int,
    close_radius: int,
    min_obj_area: int,
    hole_area: int,
) -> tuple[np.ndarray, dict]:
    small_factor = max(1, int(downsample))
    rgb_small, _, _ = _read_page_lazy_downsampled(source_path, page, small_factor)
    if rgb_small.ndim != 3 or rgb_small.shape[-1] != 3:
        raise ValueError(f"Unexpected RGB preview shape for skip mask: {rgb_small.shape}")

    lab = rgb2lab(rgb_small)
    chroma = np.sqrt(lab[..., 1] * lab[..., 1] + lab[..., 2] * lab[..., 2])
    thresh = threshold_otsu(chroma)
    mask = chroma > thresh
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
    if hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=hole_area)
    if min_obj_area > 0:
        mask = remove_small_objects(mask, min_size=min_obj_area)
    mask = np.asarray(mask, dtype=bool)

    mask_h, mask_w = mask.shape
    scale_y = float(final_h) / float(mask_h)
    scale_x = float(final_w) / float(mask_w)
    meta = {
        "enabled": True,
        "mask_shape_yx": [int(mask_h), int(mask_w)],
        "mask_downsample_factor": int(small_factor),
        "mask_scale_yx": [scale_y, scale_x],
        "tissue_fraction": float(mask.mean()) if mask.size else 0.0,
        "close_radius": int(close_radius),
        "min_obj_area": int(min_obj_area),
        "hole_area": int(hole_area),
    }
    return mask, meta


def load_background_skip_mask(path: str, final_h: int, final_w: int) -> tuple[np.ndarray, dict]:
    mask_path = Path(path)
    suffix = mask_path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(mask_path)
    elif suffix in {".tif", ".tiff"}:
        mask = tifffile.imread(mask_path)
    else:
        mask = np.asarray(Image.open(mask_path))
    mask = np.asarray(mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Unsupported background skip mask shape: {mask.shape}")
    mask = mask > 0
    mask_h, mask_w = mask.shape
    meta = {
        "enabled": True,
        "source": str(mask_path),
        "mask_shape_yx": [int(mask_h), int(mask_w)],
        "mask_scale_yx": [float(final_h) / float(mask_h), float(final_w) / float(mask_w)],
        "tissue_fraction": float(mask.mean()) if mask.size else 0.0,
    }
    return mask, meta


def region_tissue_fraction(
    mask_small: np.ndarray,
    final_h: int,
    final_w: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> float:
    mask_h, mask_w = mask_small.shape
    my0 = max(0, int(math.floor(y0 * mask_h / float(final_h))))
    my1 = min(mask_h, max(my0 + 1, int(math.ceil(y1 * mask_h / float(final_h)))))
    mx0 = max(0, int(math.floor(x0 * mask_w / float(final_w))))
    mx1 = min(mask_w, max(mx0 + 1, int(math.ceil(x1 * mask_w / float(final_w)))))
    view = mask_small[my0:my1, mx0:mx1]
    if view.size == 0:
        return 0.0
    return float(view.mean())


def axis_patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    steps = int(math.ceil((length - patch_size) / float(stride))) + 1
    padded_length = max(length, (steps - 1) * stride + patch_size)
    return list(range(0, padded_length - patch_size + 1, stride))


def contributing_patch_starts(
    axis_starts: list[int],
    start: int,
    end: int,
    patch_size: int,
) -> list[int]:
    selected = [pos for pos in axis_starts if pos < end and (pos + patch_size) > start]
    if selected:
        return selected
    return [axis_starts[0] if axis_starts else 0]


def iter_output_tiles(height: int, width: int, tile_size: int) -> Iterable[tuple[int, int, int, int]]:
    for y0 in range(0, int(height), int(tile_size)):
        y1 = min(int(height), y0 + int(tile_size))
        for x0 in range(0, int(width), int(tile_size)):
            x1 = min(int(width), x0 + int(tile_size))
            yield y0, y1, x0, x1


def normalize_patch_batch(batch_rgb: np.ndarray) -> torch.Tensor:
    batch = batch_rgb.astype(np.float32) / 255.0
    batch = (batch - MEAN[None, None, None, :]) / STD[None, None, None, :]
    batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.from_numpy(batch)


def pad_to_tiling_shape(image: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w, _ = image.shape
    starts_y = axis_patch_starts(h, patch_size, stride)
    starts_x = axis_patch_starts(w, patch_size, stride)
    steps_y = len(starts_y)
    steps_x = len(starts_x)
    pad_h = max(0, (steps_y - 1) * stride + patch_size - h)
    pad_w = max(0, (steps_x - 1) * stride + patch_size - w)
    mode = "reflect" if h > 1 and w > 1 else "edge"
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    positions = [(y, x) for y in starts_y for x in starts_x]
    return padded, positions


def batched(items: list[tuple[int, int]], batch_size: int) -> Iterable[list[tuple[int, int]]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def make_blend_window(patch_size: int) -> np.ndarray:
    if patch_size <= 1:
        return np.ones((1, 1), dtype=np.float32)

    # Raised-cosine weights reduce seams at tile borders while keeping all
    # pixels non-zero so padded image borders still receive valid predictions.
    axis = np.hanning(patch_size).astype(np.float32)
    axis = np.maximum(axis, 1.0e-3)
    window = np.outer(axis, axis)
    window /= float(window.max())
    return window.astype(np.float32, copy=False)


def run_region_inference(
    image_rgb: np.ndarray,
    positions: list[tuple[int, int]],
    model: nn.Module,
    device: torch.device,
    patch_size: int,
    batch_size: int,
    channel_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    selected_channels = list(channel_indices) if channel_indices is not None else list(range(len(CHANNEL_NAMES)))
    accum = np.zeros((len(selected_channels), h, w), dtype=np.float32)
    counts = np.zeros((1, h, w), dtype=np.float32)
    blend_window = make_blend_window(patch_size)[None, :, :]
    use_autocast = device.type == "cuda"

    with torch.no_grad():
        for batch_positions in batched(positions, batch_size):
            patch_batch = np.stack(
                [image_rgb[y:y + patch_size, x:x + patch_size, :] for y, x in batch_positions],
                axis=0,
            )
            tensor = normalize_patch_batch(patch_batch).to(device, non_blocking=device.type == "cuda")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                logits = model(tensor)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                if selected_channels != list(range(len(CHANNEL_NAMES))):
                    probs = probs[:, selected_channels, :, :]

            for pred, (y, x) in zip(probs, batch_positions):
                weighted = pred * blend_window
                accum[:, y:y + patch_size, x:x + patch_size] += weighted
                counts[:, y:y + patch_size, x:x + patch_size] += blend_window

    np.maximum(counts, 1.0, out=counts)
    return accum, counts


def encode_probability_tile(tile_cyx: np.ndarray, output_dtype: str) -> np.ndarray:
    return np.moveaxis(encode_probability_cyx(tile_cyx, output_dtype), 0, -1)


def encode_probability_cyx(tile_cyx: np.ndarray, output_dtype: str) -> np.ndarray:
    tile = np.clip(tile_cyx, 0.0, 1.0)
    if output_dtype == "uint8":
        return np.round(tile * 255.0).astype(np.uint8)
    if output_dtype == "uint16":
        return np.round(tile * 65535.0).astype(np.uint16)
    if output_dtype == "float32":
        return tile.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported output dtype: {output_dtype}")


def compute_pyramid_level_shapes(height: int, width: int, tile_size: int) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    h = int(height)
    w = int(width)
    while max(h, w) > int(tile_size):
        h = max(1, int(math.ceil(h / 2.0)))
        w = max(1, int(math.ceil(w / 2.0)))
        levels.append((h, w))
    return levels


def choose_tiff_tile_shape(
    height: int,
    width: int,
    preferred_tile_size: int,
    compression: str,
) -> tuple[int, int] | None:
    tile_h = min(int(height), int(preferred_tile_size))
    tile_w = min(int(width), int(preferred_tile_size))
    if str(compression).strip().lower() == "jpeg":
        if tile_h >= 16:
            tile_h = max(16, (tile_h // 16) * 16)
        if tile_w >= 16:
            tile_w = max(16, (tile_w // 16) * 16)
        if tile_h < 16 or tile_w < 16:
            return None
    return int(tile_h), int(tile_w)


def build_ome_pyramid_metadata(channel_names: list[str], level_shapes: list[tuple[int, int]]) -> dict:
    metadata: dict = {
        "axes": "CYX",
        "Channel": {"Name": list(channel_names)},
    }
    if level_shapes:
        metadata["MapAnnotation"] = {
            "Namespace": "openmicroscopy.org/PyramidResolution",
            **{str(idx + 1): f"{int(w)} {int(h)}" for idx, (h, w) in enumerate(level_shapes)},
        }
    return metadata


def iter_cyx_tiff_tiles_from_level0(
    level0: np.ndarray,
    *,
    out_shape_yx: tuple[int, int],
    tile_shape: tuple[int, int] | None,
    downsample: int = 1,
) -> Iterable[np.ndarray]:
    """Yield one CYX TIFF tile at a time from the staged level-0 memmap.

    `tifffile` otherwise materializes non-contiguous pyramid views as large
    contiguous arrays. For WSI-sized GigaTIME outputs that can exceed RAM.
    """
    out_h, out_w = (int(out_shape_yx[0]), int(out_shape_yx[1]))
    mag = max(1, int(downsample))
    channel_count = int(level0.shape[0])
    if tile_shape is None:
        for ch_idx in range(channel_count):
            yield np.ascontiguousarray(level0[ch_idx, 0:out_h * mag:mag, 0:out_w * mag:mag])
        return

    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    for ch_idx in range(channel_count):
        for y0 in range(0, out_h, tile_h):
            y1 = min(out_h, y0 + tile_h)
            src_y0 = y0 * mag
            src_y1 = y1 * mag
            for x0 in range(0, out_w, tile_w):
                x1 = min(out_w, x0 + tile_w)
                src_x0 = x0 * mag
                src_x1 = x1 * mag
                yield np.ascontiguousarray(level0[ch_idx, src_y0:src_y1:mag, src_x0:src_x1:mag])


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("GigaTIME requested CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(repo_id: str, token: str | None, device: torch.device, offline: bool) -> GigaTIMEModel:
    local_dir = snapshot_download(repo_id=repo_id, token=token, local_files_only=offline)
    weights_path = Path(local_dir) / "model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"GigaTIME weights not found: {weights_path}")

    model = GigaTIMEModel(num_classes=len(CHANNEL_NAMES), input_channels=3)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_quantifiers(
    *,
    nuclei_mask_path: str,
    cyto_mask_path: str,
    target_shape: tuple[int, int],
    channel_names: list[str],
    block_size: int,
) -> list[TileQuantifier]:
    quantifiers: list[TileQuantifier] = []
    shared_max_label: int | None = None
    if nuclei_mask_path:
        nuclei_quantifier = TileQuantifier(
            mask_name="nuclei",
            mask_path=nuclei_mask_path,
            target_shape=target_shape,
            channel_names=channel_names,
            block_size=block_size,
        )
        shared_max_label = int(nuclei_quantifier.max_label)
        quantifiers.append(nuclei_quantifier)
    if cyto_mask_path:
        quantifiers.append(
            TileQuantifier(
                mask_name="cyto",
                mask_path=cyto_mask_path,
                target_shape=target_shape,
                channel_names=channel_names,
                block_size=block_size,
                max_label=shared_max_label,
            )
        )
    return quantifiers


def consume_tile_outputs(
    *,
    quantifiers: list[TileQuantifier],
    jpg_exporter: JpegTileExporter | None,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    tile_probs_all_cyx: np.ndarray,
    jpg_channel_indices: list[int],
) -> None:
    for quantifier in quantifiers:
        quantifier.accumulate_tile(y0, y1, x0, x1, tile_probs_all_cyx)
    if jpg_exporter is not None:
        jpg_exporter.save_tile(y0, y1, x0, x1, tile_probs_all_cyx[jpg_channel_indices, :, :])


def finalize_aux_outputs(
    *,
    sample_id: str,
    quantifiers: list[TileQuantifier],
    quant_dir: Path | None,
    jpg_exporter: JpegTileExporter | None,
    metadata: dict,
) -> None:
    summaries = []
    try:
        if quant_dir is not None:
            quant_dir.mkdir(parents=True, exist_ok=True)
            for quantifier in quantifiers:
                summaries.append(quantifier.write_outputs(quant_dir, sample_id))
        if jpg_exporter is not None:
            jpg_exporter.write_outputs(metadata)
        if quant_dir is not None:
            (quant_dir / f"{sample_id}_gigatime_integrated_quantification_summary.json").write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "summaries": summaries,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        for quantifier in quantifiers:
            quantifier.close()


def blockwise_write_ometiff_outputs(
    source_path: str,
    page: int,
    outdir: Path,
    model: nn.Module,
    device: torch.device,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    factor: int,
    tile_size: int,
    output_dtype: str,
    compression: str,
    predictor: bool,
    jpeg_quality: int,
    pyramid: bool,
    metadata: dict,
    output_channel_indices: list[int],
    output_channel_names: list[str],
    quantifiers: list[TileQuantifier],
    quant_dir: Path | None,
    jpg_exporter: JpegTileExporter | None,
    jpg_channel_indices: list[int],
    sample_id: str,
    resume_level0_buffer: bool,
    skip_background_blocks: bool,
    skip_background_mask_path: str,
    skip_background_downsample: int,
    skip_background_min_fraction: float,
    skip_background_close_radius: int,
    skip_background_min_obj_area: int,
    skip_background_hole_area: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    final_h, final_w = (int(v) for v in metadata["inference_shape_yx"])
    channel_count = len(output_channel_names)
    output_dtype_np = output_dtype_info(output_dtype)[0]
    n_tiles_y = int(math.ceil(final_h / float(tile_size)))
    n_tiles_x = int(math.ceil(final_w / float(tile_size)))
    total_tiles = int(n_tiles_y * n_tiles_x)
    pyramid_shapes = compute_pyramid_level_shapes(final_h, final_w, tile_size) if pyramid else []
    tmp_level0_path = outdir / "_gigatime_level0.cyx.bin"
    if skip_background_blocks:
        print(
            "[WARN] GigaTIME OME-TIFF background block skipping is disabled for this writer; "
            "the previous coarse skip-mask path can exceed RAM on WSI inputs. "
            "The OME-TIFF itself is still written tile-by-tile.",
            flush=True,
        )
        skip_background_blocks = False

    level0_tile = choose_tiff_tile_shape(final_h, final_w, tile_size, compression)
    final_tiff_kwargs = {
        "compression": compression,
        "metadata": build_ome_pyramid_metadata(output_channel_names, pyramid_shapes),
    }
    if level0_tile is not None:
        final_tiff_kwargs["tile"] = level0_tile
    if compression.lower() == "jpeg":
        final_tiff_kwargs["compressionargs"] = {"level": int(max(1, min(100, jpeg_quality)))}
    elif predictor:
        final_tiff_kwargs["predictor"] = True
    if metadata.get("effective_mpp"):
        res = float(10_000.0 / float(metadata["effective_mpp"]))
        final_tiff_kwargs["resolution"] = (res, res)
        final_tiff_kwargs["resolutionunit"] = "CENTIMETER"

    expected_level0_bytes = int(channel_count * final_h * final_w * np.dtype(output_dtype_np).itemsize)
    reuse_level0 = (
        bool(resume_level0_buffer)
        and tmp_level0_path.exists()
        and tmp_level0_path.stat().st_size == expected_level0_bytes
    )
    if reuse_level0:
        print(f"[INFO] Reusing preserved level-0 buffer: {tmp_level0_path}", flush=True)
    else:
        skipped_tiles = 0
        skip_mask_small = None
        skip_meta = {"enabled": False}
        if skip_background_blocks:
            if skip_background_mask_path:
                print(f"[INFO] GigaTIME loading external background-skip mask from {skip_background_mask_path}", flush=True)
                skip_mask_small, skip_meta = load_background_skip_mask(
                    path=skip_background_mask_path,
                    final_h=final_h,
                    final_w=final_w,
                )
            else:
                print("[INFO] GigaTIME building coarse background-skip mask", flush=True)
                skip_mask_small, skip_meta = build_background_skip_mask(
                    source_path=source_path,
                    page=page,
                    factor=factor,
                    final_h=final_h,
                    final_w=final_w,
                    downsample=skip_background_downsample,
                    close_radius=skip_background_close_radius,
                    min_obj_area=skip_background_min_obj_area,
                    hole_area=skip_background_hole_area,
                )
            print(
                "[INFO] GigaTIME background skip mask "
                f"shape={tuple(skip_meta['mask_shape_yx'])} "
                f"tissue_fraction={skip_meta['tissue_fraction']:.3f} "
                f"min_fraction={skip_background_min_fraction:.3f}",
                flush=True,
            )
        metadata["background_skip"] = dict(skip_meta)
        metadata["background_skip"]["min_fraction"] = float(skip_background_min_fraction)

        with LazyCropReader(source_path, page, factor) as reader:
            if reader.final_h != final_h or reader.final_w != final_w:
                raise ValueError(
                    f"Planned inference shape {(final_h, final_w)} does not match reader shape "
                    f"{(reader.final_h, reader.final_w)}"
                )
            starts_y = axis_patch_starts(final_h, patch_size, stride)
            starts_x = axis_patch_starts(final_w, patch_size, stride)
            level0 = np.memmap(
                tmp_level0_path,
                mode="w+",
                dtype=output_dtype_np,
                shape=(channel_count, final_h, final_w),
            )
            try:
                done = 0
                for y0, y1, x0, x1 in iter_output_tiles(final_h, final_w, tile_size):
                    if done == 0:
                        print(
                            f"[INFO] GigaTIME first tile region y=({y0},{y1}) x=({x0},{x1})",
                            flush=True,
                        )
                    if skip_mask_small is not None:
                        frac = region_tissue_fraction(skip_mask_small, final_h, final_w, y0, y1, x0, x1)
                        if frac < float(skip_background_min_fraction):
                            skipped_tiles += 1
                            done += 1
                            if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                                print(
                                    f"[INFO] GigaTIME blockwise tile {done}/{total_tiles} "
                                    f"(skipped blank region frac={frac:.3f})",
                                    flush=True,
                                )
                            continue
                    ys = contributing_patch_starts(starts_y, y0, y1, patch_size)
                    xs = contributing_patch_starts(starts_x, x0, x1, patch_size)
                    ry0 = ys[0]
                    rx0 = xs[0]
                    ry1 = ys[-1] + patch_size
                    rx1 = xs[-1] + patch_size
                    region_rgb = reader.read_region(ry0, ry1, rx0, rx1)
                    local_positions = [(yy - ry0, xx - rx0) for yy in ys for xx in xs]
                    accum, counts = run_region_inference(
                        image_rgb=region_rgb,
                        positions=local_positions,
                        model=model,
                        device=device,
                        patch_size=patch_size,
                        batch_size=batch_size,
                    )
                    off_y = y0 - ry0
                    off_x = x0 - rx0
                    tile_probs = accum[:, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)] / counts[
                        :, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)
                    ]
                    consume_tile_outputs(
                        quantifiers=quantifiers,
                        jpg_exporter=jpg_exporter,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                        tile_probs_all_cyx=tile_probs,
                        jpg_channel_indices=jpg_channel_indices,
                    )
                    tile_probs = tile_probs[output_channel_indices, :, :]
                    level0[:, y0:y1, x0:x1] = encode_probability_cyx(tile_probs, output_dtype)
                    del region_rgb, local_positions, accum, counts, tile_probs
                    if done % 10 == 0:
                        gc.collect()
                    done += 1
                    if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                        print(f"[INFO] GigaTIME blockwise tile {done}/{total_tiles}")
                level0.flush()
            finally:
                del level0
        metadata["background_skip"]["skipped_tiles"] = int(skipped_tiles)
        metadata["background_skip"]["processed_tiles"] = int(total_tiles - skipped_tiles)
        metadata["background_skip"]["total_tiles"] = int(total_tiles)

        # Persist integrated marker quantification before the potentially long
        # TIFF pyramid write, so a writer failure does not discard all-marker
        # single-cell summaries already computed during inference.
        finalize_aux_outputs(
            sample_id=sample_id,
            quantifiers=quantifiers,
            quant_dir=quant_dir,
            jpg_exporter=jpg_exporter,
            metadata=metadata,
        )

    final_path = outdir / "gigatime_probs.ome.tif"
    write_succeeded = False
    try:
        level0 = np.memmap(
            tmp_level0_path,
            mode="r",
            dtype=output_dtype_np,
            shape=(channel_count, final_h, final_w),
        )
        try:
            base_res = float(10_000.0 / float(metadata["effective_mpp"])) if metadata.get("effective_mpp") else None
            with tifffile.TiffWriter(final_path, bigtiff=True, ome=True) as tif:
                print(
                    f"[INFO] GigaTIME writing OME-TIFF level 0 shape={(channel_count, final_h, final_w)} "
                    f"tile={level0_tile}",
                    flush=True,
                )
                tif.write(
                    iter_cyx_tiff_tiles_from_level0(
                        level0,
                        out_shape_yx=(final_h, final_w),
                        tile_shape=level0_tile,
                        downsample=1,
                    ),
                    shape=(channel_count, final_h, final_w),
                    dtype=output_dtype_np,
                    subifds=len(pyramid_shapes),
                    **final_tiff_kwargs,
                )
                for level_idx, level_shape in enumerate(pyramid_shapes, start=1):
                    mag = 2 ** level_idx
                    level_kwargs = dict(final_tiff_kwargs)
                    level_kwargs["metadata"] = None
                    level_tile = choose_tiff_tile_shape(level_shape[0], level_shape[1], tile_size, compression)
                    if level_tile is None:
                        level_kwargs.pop("tile", None)
                    else:
                        level_kwargs["tile"] = level_tile
                    if base_res is not None:
                        level_kwargs["resolution"] = (base_res / mag, base_res / mag)
                        level_kwargs["resolutionunit"] = "CENTIMETER"
                    print(
                        f"[INFO] GigaTIME writing OME-TIFF pyramid level {level_idx} "
                        f"shape={(channel_count, int(level_shape[0]), int(level_shape[1]))} "
                        f"tile={level_tile}",
                        flush=True,
                    )
                    tif.write(
                        iter_cyx_tiff_tiles_from_level0(
                            level0,
                            out_shape_yx=(int(level_shape[0]), int(level_shape[1])),
                            tile_shape=level_tile,
                            downsample=mag,
                        ),
                        shape=(channel_count, int(level_shape[0]), int(level_shape[1])),
                        dtype=output_dtype_np,
                        subfiletype=1,
                        **level_kwargs,
                    )
            write_succeeded = True
        finally:
            del level0
    except Exception:
        if final_path.exists():
            final_path.unlink()
        raise
    finally:
        if write_succeeded and tmp_level0_path.exists():
            tmp_level0_path.unlink()
        elif not write_succeeded and tmp_level0_path.exists():
            print(f"[WARN] Preserving staged level-0 buffer after TIFF write failure: {tmp_level0_path}")

    (outdir / "gigatime_channels.json").write_text(json.dumps(output_channel_names, indent=2), encoding="utf-8")
    (outdir / "gigatime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if reuse_level0:
        finalize_aux_outputs(
            sample_id=sample_id,
            quantifiers=quantifiers,
            quant_dir=quant_dir,
            jpg_exporter=jpg_exporter,
            metadata=metadata,
        )


def blockwise_write_zarr_outputs(
    source_path: str,
    page: int,
    outdir: Path,
    model: nn.Module,
    device: torch.device,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    factor: int,
    tile_size: int,
    output_dtype: str,
    metadata: dict,
    skip_background_blocks: bool,
    skip_background_mask_path: str,
    skip_background_downsample: int,
    skip_background_min_fraction: float,
    skip_background_close_radius: int,
    skip_background_min_obj_area: int,
    skip_background_hole_area: int,
    output_channel_indices: list[int],
    output_channel_names: list[str],
    quantifiers: list[TileQuantifier],
    quant_dir: Path | None,
    jpg_exporter: JpegTileExporter | None,
    jpg_channel_indices: list[int],
    sample_id: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    final_h, final_w = (int(v) for v in metadata["inference_shape_yx"])
    chunk_y = min(int(tile_size), final_h)
    chunk_x = min(int(tile_size), final_w)
    print(
        f"[INFO] GigaTIME zarr init shape={(len(output_channel_names), final_h, final_w)} "
        f"chunks={(1, chunk_y, chunk_x)}",
        flush=True,
    )
    zarr_path = outdir / "gigatime_probs.zarr"
    if zarr_path.exists():
        if zarr_path.is_dir():
            for child in zarr_path.iterdir():
                if child.is_dir():
                    import shutil
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            zarr_path.unlink()
    store = _zarr_directory_store(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    root.attrs.update(_zarr_root_attrs(metadata, output_channel_names))
    compressor = _zarr_compressor()
    dataset_kwargs = {
        "shape": (len(output_channel_names), final_h, final_w),
        "chunks": (1, chunk_y, chunk_x),
        "dtype": output_dtype_info(output_dtype)[0],
        "overwrite": True,
        "fill_value": output_dtype_info(output_dtype)[0].type(0),
    }
    if compressor is not None:
        dataset_kwargs["compressor"] = compressor
    arr = root.create_dataset("0", **dataset_kwargs)
    print("[INFO] GigaTIME zarr dataset created", flush=True)
    arr.attrs.update({
        "axes": "CYX",
        "channel_names": list(output_channel_names),
        "storage_scale_max": float(metadata.get("storage_scale_max") or 65535.0),
        "output_dtype": output_dtype,
    })

    n_tiles_y = int(math.ceil(final_h / float(tile_size)))
    n_tiles_x = int(math.ceil(final_w / float(tile_size)))
    total_tiles = int(n_tiles_y * n_tiles_x)
    skipped_tiles = 0
    skip_mask_small = None
    skip_meta = {"enabled": False}

    if skip_background_blocks:
        if skip_background_mask_path:
            print(f"[INFO] GigaTIME loading external background-skip mask from {skip_background_mask_path}", flush=True)
            skip_mask_small, skip_meta = load_background_skip_mask(
                path=skip_background_mask_path,
                final_h=final_h,
                final_w=final_w,
            )
        else:
            print("[INFO] GigaTIME building coarse background-skip mask", flush=True)
            skip_mask_small, skip_meta = build_background_skip_mask(
                source_path=source_path,
                page=page,
                factor=factor,
                final_h=final_h,
                final_w=final_w,
                downsample=skip_background_downsample,
                close_radius=skip_background_close_radius,
                min_obj_area=skip_background_min_obj_area,
                hole_area=skip_background_hole_area,
            )
        print(
            "[INFO] GigaTIME background skip mask "
            f"shape={tuple(skip_meta['mask_shape_yx'])} "
            f"tissue_fraction={skip_meta['tissue_fraction']:.3f} "
            f"min_fraction={skip_background_min_fraction:.3f}",
            flush=True,
        )
    metadata["background_skip"] = dict(skip_meta)
    metadata["background_skip"]["min_fraction"] = float(skip_background_min_fraction)

    with LazyCropReader(source_path, page, factor) as reader:
        print(
            f"[INFO] GigaTIME reader ready final_shape={(reader.final_h, reader.final_w)} "
            f"using={'pyvips' if reader.vips_image is not None else 'tifffile-zarr'}",
            flush=True,
        )
        if reader.final_h != final_h or reader.final_w != final_w:
            raise ValueError(
                f"Planned inference shape {(final_h, final_w)} does not match reader shape "
                f"{(reader.final_h, reader.final_w)}"
            )
        starts_y = axis_patch_starts(final_h, patch_size, stride)
        starts_x = axis_patch_starts(final_w, patch_size, stride)
        done = 0
        for y0, y1, x0, x1 in iter_output_tiles(final_h, final_w, tile_size):
            if done == 0:
                print(
                    f"[INFO] GigaTIME first tile region y=({y0},{y1}) x=({x0},{x1})",
                    flush=True,
                )
            if skip_mask_small is not None:
                frac = region_tissue_fraction(skip_mask_small, final_h, final_w, y0, y1, x0, x1)
                if frac < float(skip_background_min_fraction):
                    skipped_tiles += 1
                    done += 1
                    if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                        print(
                            f"[INFO] GigaTIME blockwise tile {done}/{total_tiles} "
                            f"(skipped blank region frac={frac:.3f})",
                            flush=True,
                        )
                    continue
            ys = contributing_patch_starts(starts_y, y0, y1, patch_size)
            xs = contributing_patch_starts(starts_x, x0, x1, patch_size)
            ry0 = ys[0]
            rx0 = xs[0]
            ry1 = ys[-1] + patch_size
            rx1 = xs[-1] + patch_size
            region_rgb = reader.read_region(ry0, ry1, rx0, rx1)
            local_positions = [(yy - ry0, xx - rx0) for yy in ys for xx in xs]
            accum, counts = run_region_inference(
                image_rgb=region_rgb,
                positions=local_positions,
                model=model,
                device=device,
                patch_size=patch_size,
                batch_size=batch_size,
                channel_indices=output_channel_indices,
            )
            off_y = y0 - ry0
            off_x = x0 - rx0
            tile_probs = accum[:, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)] / counts[
                :, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)
            ]
            consume_tile_outputs(
                quantifiers=quantifiers,
                jpg_exporter=jpg_exporter,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                tile_probs_all_cyx=tile_probs,
                jpg_channel_indices=jpg_channel_indices,
            )
            arr[:, y0:y1, x0:x1] = encode_probability_cyx(tile_probs, output_dtype)
            del region_rgb, local_positions, accum, counts, tile_probs
            if done % 10 == 0:
                gc.collect()
            done += 1
            if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                print(f"[INFO] GigaTIME blockwise tile {done}/{total_tiles}", flush=True)

    metadata["background_skip"]["skipped_tiles"] = int(skipped_tiles)
    metadata["background_skip"]["processed_tiles"] = int(total_tiles - skipped_tiles)
    metadata["background_skip"]["total_tiles"] = int(total_tiles)

    (outdir / "gigatime_channels.json").write_text(json.dumps(output_channel_names, indent=2), encoding="utf-8")
    (outdir / "gigatime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finalize_aux_outputs(
        sample_id=sample_id,
        quantifiers=quantifiers,
        quant_dir=quant_dir,
        jpg_exporter=jpg_exporter,
        metadata=metadata,
    )


def blockwise_process_without_store(
    source_path: str,
    page: int,
    outdir: Path,
    model: nn.Module,
    device: torch.device,
    *,
    patch_size: int,
    stride: int,
    batch_size: int,
    factor: int,
    tile_size: int,
    metadata: dict,
    skip_background_blocks: bool,
    skip_background_mask_path: str,
    skip_background_downsample: int,
    skip_background_min_fraction: float,
    skip_background_close_radius: int,
    skip_background_min_obj_area: int,
    skip_background_hole_area: int,
    quantifiers: list[TileQuantifier],
    quant_dir: Path | None,
    jpg_exporter: JpegTileExporter | None,
    jpg_channel_indices: list[int],
    sample_id: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    final_h, final_w = (int(v) for v in metadata["inference_shape_yx"])
    n_tiles_y = int(math.ceil(final_h / float(tile_size)))
    n_tiles_x = int(math.ceil(final_w / float(tile_size)))
    total_tiles = int(n_tiles_y * n_tiles_x)
    skipped_tiles = 0
    skip_mask_small = None
    skip_meta = {"enabled": False}

    if skip_background_blocks:
        if skip_background_mask_path:
            print(f"[INFO] GigaTIME loading external background-skip mask from {skip_background_mask_path}", flush=True)
            skip_mask_small, skip_meta = load_background_skip_mask(
                path=skip_background_mask_path,
                final_h=final_h,
                final_w=final_w,
            )
        else:
            print("[INFO] GigaTIME building coarse background-skip mask", flush=True)
            skip_mask_small, skip_meta = build_background_skip_mask(
                source_path=source_path,
                page=page,
                factor=factor,
                final_h=final_h,
                final_w=final_w,
                downsample=skip_background_downsample,
                close_radius=skip_background_close_radius,
                min_obj_area=skip_background_min_obj_area,
                hole_area=skip_background_hole_area,
            )
        print(
            "[INFO] GigaTIME background skip mask "
            f"shape={tuple(skip_meta['mask_shape_yx'])} "
            f"tissue_fraction={skip_meta['tissue_fraction']:.3f} "
            f"min_fraction={skip_background_min_fraction:.3f}",
            flush=True,
        )
    metadata["background_skip"] = dict(skip_meta)
    metadata["background_skip"]["min_fraction"] = float(skip_background_min_fraction)

    with LazyCropReader(source_path, page, factor) as reader:
        print(
            f"[INFO] GigaTIME reader ready final_shape={(reader.final_h, reader.final_w)} "
            f"using={'pyvips' if reader.vips_image is not None else 'tifffile-zarr'}",
            flush=True,
        )
        if reader.final_h != final_h or reader.final_w != final_w:
            raise ValueError(
                f"Planned inference shape {(final_h, final_w)} does not match reader shape "
                f"{(reader.final_h, reader.final_w)}"
            )
        starts_y = axis_patch_starts(final_h, patch_size, stride)
        starts_x = axis_patch_starts(final_w, patch_size, stride)
        done = 0
        for y0, y1, x0, x1 in iter_output_tiles(final_h, final_w, tile_size):
            if done == 0:
                print(
                    f"[INFO] GigaTIME first tile region y=({y0},{y1}) x=({x0},{x1})",
                    flush=True,
                )
            if skip_mask_small is not None:
                frac = region_tissue_fraction(skip_mask_small, final_h, final_w, y0, y1, x0, x1)
                if frac < float(skip_background_min_fraction):
                    skipped_tiles += 1
                    done += 1
                    if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                        print(
                            f"[INFO] GigaTIME blockwise tile {done}/{total_tiles} "
                            f"(skipped blank region frac={frac:.3f})",
                            flush=True,
                        )
                    continue
            ys = contributing_patch_starts(starts_y, y0, y1, patch_size)
            xs = contributing_patch_starts(starts_x, x0, x1, patch_size)
            ry0 = ys[0]
            rx0 = xs[0]
            ry1 = ys[-1] + patch_size
            rx1 = xs[-1] + patch_size
            region_rgb = reader.read_region(ry0, ry1, rx0, rx1)
            local_positions = [(yy - ry0, xx - rx0) for yy in ys for xx in xs]
            accum, counts = run_region_inference(
                image_rgb=region_rgb,
                positions=local_positions,
                model=model,
                device=device,
                patch_size=patch_size,
                batch_size=batch_size,
            )
            off_y = y0 - ry0
            off_x = x0 - rx0
            tile_probs = accum[:, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)] / counts[
                :, off_y:off_y + (y1 - y0), off_x:off_x + (x1 - x0)
            ]
            consume_tile_outputs(
                quantifiers=quantifiers,
                jpg_exporter=jpg_exporter,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                tile_probs_all_cyx=tile_probs,
                jpg_channel_indices=jpg_channel_indices,
            )
            del region_rgb, local_positions, accum, counts, tile_probs
            if done % 10 == 0:
                gc.collect()
            done += 1
            if done == 1 or done == total_tiles or done % max(1, total_tiles // 20) == 0:
                print(f"[INFO] GigaTIME blockwise tile {done}/{total_tiles}", flush=True)

    metadata["background_skip"]["skipped_tiles"] = int(skipped_tiles)
    metadata["background_skip"]["processed_tiles"] = int(total_tiles - skipped_tiles)
    metadata["background_skip"]["total_tiles"] = int(total_tiles)
    (outdir / "gigatime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (outdir / "gigatime_channels.json").write_text(json.dumps(CHANNEL_NAMES, indent=2), encoding="utf-8")
    finalize_aux_outputs(
        sample_id=sample_id,
        quantifiers=quantifiers,
        quant_dir=quant_dir,
        jpg_exporter=jpg_exporter,
        metadata=metadata,
    )


def run_inference(
    image_rgb: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    disk_backed: bool,
    scratch_dir: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    padded, positions = pad_to_tiling_shape(image_rgb, patch_size, stride)
    h, w = image_rgb.shape[:2]
    padded_h, padded_w = padded.shape[:2]

    if disk_backed:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        accum = np.memmap(
            scratch_dir / "_gigatime_accum_float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(len(CHANNEL_NAMES), padded_h, padded_w),
        )
        counts = np.memmap(
            scratch_dir / "_gigatime_counts_float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(1, padded_h, padded_w),
        )
        accum[:] = 0.0
        counts[:] = 0.0
    else:
        accum = np.zeros((len(CHANNEL_NAMES), padded_h, padded_w), dtype=np.float32)
        counts = np.zeros((1, padded_h, padded_w), dtype=np.float32)
    blend_window = make_blend_window(patch_size)
    blend_window = blend_window[None, :, :]

    use_autocast = device.type == "cuda"

    with torch.no_grad():
        for batch_positions in batched(positions, batch_size):
            patch_batch = np.stack(
                [padded[y:y + patch_size, x:x + patch_size, :] for y, x in batch_positions],
                axis=0,
            )
            tensor = normalize_patch_batch(patch_batch).to(device, non_blocking=device.type == "cuda")
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_autocast
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                logits = model(tensor)
                probs = torch.sigmoid(logits).float().cpu().numpy()

            for pred, (y, x) in zip(probs, batch_positions):
                weighted = pred * blend_window
                accum[:, y:y + patch_size, x:x + patch_size] += weighted
                counts[:, y:y + patch_size, x:x + patch_size] += blend_window

    np.maximum(counts, 1.0, out=counts)
    if disk_backed:
        accum.flush()
        counts.flush()
    return accum, counts, (h, w)


def write_outputs(
    outdir: Path,
    accum: np.ndarray,
    counts: np.ndarray,
    valid_shape: tuple[int, int],
    compression: str,
    predictor: bool,
    metadata: dict,
    output_channel_indices: list[int],
    output_channel_names: list[str],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    h, w = valid_shape
    final_path = outdir / "_gigatime_final_float32.dat"
    final = np.memmap(
        final_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(output_channel_names), h, w),
    )
    denom = counts[0, :h, :w]
    for out_ch, src_ch in enumerate(output_channel_indices):
        final[out_ch] = accum[src_ch, :h, :w] / denom
    final.flush()
    tifffile.imwrite(
        outdir / "gigatime_probs.ome.tif",
        final,
        metadata={"axes": "CYX", "Channel": {"Name": output_channel_names}},
        compression=compression,
        predictor=predictor,
        bigtiff=final.nbytes >= (4 * 1024 * 1024 * 1024),
    )
    (outdir / "gigatime_channels.json").write_text(json.dumps(output_channel_names, indent=2), encoding="utf-8")
    (outdir / "gigatime_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        os.remove(final_path)
    except OSError:
        pass
    for temp_name in ("_gigatime_accum_float32.dat", "_gigatime_counts_float32.dat"):
        try:
            os.remove(outdir / temp_name)
        except OSError:
            pass


def parse_args():
    ap = argparse.ArgumentParser(description="Run GigaTIME virtual mIF inference on a crop image.")
    ap.add_argument("--image", required=True, help="Crop image TIFF path")
    ap.add_argument("--shift-json", default="",
                    help="Optional StarDist shift.json used to recover the calibrated source image MPP.")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--repo-id", default="prov-gigatime/GigaTIME", help="Hugging Face repo ID")
    ap.add_argument("--page", type=int, default=0, help="TIFF page to read")
    ap.add_argument("--patch-size", type=int, default=256, help="Inference patch size (default from model card: 256)")
    ap.add_argument("--stride", type=int, default=128, help="Sliding-window stride")
    ap.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    ap.add_argument("--auto-hardware", action="store_true",
                    help="Adapt GigaTIME batch/tile/output budget to live system RAM and GPU memory.")
    ap.add_argument("--task-memory-gb", type=float, default=0.0,
                    help="Memory budget assigned by the workflow engine; 0 means use live MemAvailable only.")
    ap.add_argument("--min-free-system-gb", type=float, default=6.0,
                    help="RAM reserve kept free when auto-adapting GigaTIME settings.")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    ap.add_argument("--compression", default="deflate", help="Output TIFF compression")
    ap.add_argument("--auto-threshold-mpix", type=float, default=100.0,
                    help="Apply automatic pre-inference downsampling above this image size (default: 100 MP).")
    ap.add_argument("--max-side", type=int, default=4096,
                    help="Maximum image side length used for inference after automatic downsampling (default: 4096).")
    ap.add_argument("--target-mpp", type=float, default=0.25,
                    help="Target physical resolution in microns-per-pixel for inference when source calibration is available.")
    ap.add_argument("--max-output-gib", type=float, default=32.0,
                    help="Maximum estimated uncompressed prediction stack size in GiB before extra downsampling is applied.")
    ap.add_argument("--disk-buffer-threshold-gib", type=float, default=4.0,
                    help="Switch to disk-backed accumulation above this estimated output size in GiB.")
    ap.add_argument("--output-format", choices=["ome_tiff", "zarr", "none"], default="ome_tiff",
                    help="Output storage backend for the virtual mIF stack.")
    ap.add_argument("--output-dtype", choices=["float32", "uint16", "uint8"], default="uint16",
                    help="Output storage dtype for the virtual mIF stack.")
    ap.add_argument("--output-channels", default="",
                    help="Comma-separated subset of marker channels to save, e.g. 'DAPI,PD-1'.")
    ap.add_argument("--jpg-markers", default="DAPI,PD-1",
                    help="Comma-separated marker channels to export as reconstructed whole-image JPEGs.")
    ap.add_argument("--jpg-quality", type=int, default=75,
                    help="JPEG quality for saved GigaTIME marker images.")
    ap.add_argument("--jpg-preview-max-side", type=int, default=4096,
                    help="Maximum side length for the reconstructed whole-image JPEGs.")
    ap.add_argument("--jpg-save-tiles", action="store_true",
                    help="Persist per-tile JPEGs in addition to the reconstructed whole-image JPEGs.")
    ap.add_argument("--nuclei-mask", default="", help="Optional nuclei label mask used for on-the-fly quantification.")
    ap.add_argument("--cyto-mask", default="", help="Optional cytoplasm label mask used for on-the-fly quantification.")
    ap.add_argument("--quant-dir", default="", help="Optional output directory for integrated marker quantification CSV/JSON files.")
    ap.add_argument("--strict-target-mpp", action="store_true",
                    help="Honor the requested target MPP when source calibration is available instead of relaxing scale for disk budget.")
    ap.add_argument("--pyramid", action="store_true",
                    help="Write OME-TIFF outputs with SubIFD pyramid levels.")
    ap.add_argument("--resume-level0-buffer", action="store_true",
                    help="Reuse an existing staged level-0 CYX buffer in the output directory.")
    ap.add_argument("--block-size", type=int, default=2048,
                    help="Tile size used for blockwise full-resolution writing.")
    ap.add_argument("--blockwise", action="store_true",
                    help="Write outputs tile-by-tile instead of accumulating a whole-image stack in memory or memmaps.")
    ap.add_argument("--predictor", action="store_true",
                    help="Enable TIFF predictor during output compression.")
    ap.add_argument("--skip-background-blocks", action="store_true",
                    help="Estimate a coarse tissue mask from the H&E crop and skip writing blank output blocks.")
    ap.add_argument("--skip-background-mask-path", default="",
                    help="Optional precomputed coarse tissue mask (PNG/TIFF/NPY) used for blank-block skipping.")
    ap.add_argument("--skip-background-downsample", type=int, default=32,
                    help="Downsample factor used to build the coarse tissue mask for blank-block skipping.")
    ap.add_argument("--skip-background-min-fraction", type=float, default=0.02,
                    help="Minimum coarse tissue fraction required to run a block instead of leaving it as zero fill.")
    ap.add_argument("--skip-background-close-radius", type=int, default=8,
                    help="Morphological closing radius for the coarse tissue mask used in blank-block skipping.")
    ap.add_argument("--skip-background-min-obj-area", type=int, default=2048,
                    help="Minimum coarse tissue object area retained in the blank-block skip mask.")
    ap.add_argument("--skip-background-hole-area", type=int, default=2048,
                    help="Maximum coarse hole area filled in the blank-block skip mask.")
    return ap.parse_args()


def main():
    args = parse_args()
    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    offline = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    device = resolve_device(args.device)
    hardware_meta = adapt_gigatime_hardware(args, device)
    print(
        "[INFO] GigaTIME hardware adaptation "
        f"enabled={hardware_meta['enabled']} "
        f"requested={hardware_meta['requested']} "
        f"effective={hardware_meta['effective']} "
        f"system_mem_available_gib={hardware_meta.get('system_mem_available_gib')} "
        f"cuda_mem_free_gib={hardware_meta.get('cuda_mem_free_gib')} "
        f"cuda_mem_total_gib={hardware_meta.get('cuda_mem_total_gib')}",
        flush=True,
    )
    output_channel_indices, output_channel_names = resolve_output_channels(args.output_channels, default_all=True)
    jpg_channel_indices, jpg_channel_names = resolve_output_channels(args.jpg_markers, default_all=False)
    sample_id = Path(args.outdir).name.replace("gigatime_", "", 1)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    effective_output_dtype = str(args.output_dtype)
    effective_predictor = bool(args.predictor)
    if args.output_format == "ome_tiff" and str(args.compression).strip().lower() == "jpeg":
        if effective_output_dtype != "uint8":
            print(
                f"[WARN] JPEG-compressed OME-TIFF requires uint8 storage; coercing output dtype from {effective_output_dtype} to uint8",
                flush=True,
            )
            effective_output_dtype = "uint8"
        if effective_predictor:
            print("[WARN] Disabling TIFF predictor for JPEG-compressed OME-TIFF output", flush=True)
            effective_predictor = False

    output_dtype, output_bytes_per_sample, output_scale_max = output_dtype_info(effective_output_dtype)
    disable_max_side_fallback = args.output_format == "none" and bool(args.strict_target_mpp)
    image_meta = inspect_crop_image(
        args.image,
        args.page,
        args.auto_threshold_mpix,
        args.max_side,
        args.target_mpp,
        args.max_output_gib,
        num_output_channels=(len(output_channel_names) if args.output_format != "none" else len(jpg_channel_names)),
        bytes_per_sample=output_bytes_per_sample,
        strict_target_mpp=bool(args.strict_target_mpp),
        shift_json=str(args.shift_json or ""),
        enable_max_side_fallback=not disable_max_side_fallback,
    )
    image_meta["channels"] = list(CHANNEL_NAMES)
    image_meta["store_channels"] = list(output_channel_names) if args.output_format != "none" else []
    image_meta["jpg_markers"] = list(jpg_channel_names)
    image_meta["model_channels"] = list(CHANNEL_NAMES)
    image_meta["output_format"] = args.output_format
    image_meta["output_dtype"] = effective_output_dtype
    image_meta["strict_target_mpp"] = bool(args.strict_target_mpp)
    image_meta["blockwise"] = bool(args.blockwise or effective_output_dtype != "float32" or args.output_format == "zarr")
    image_meta["block_size"] = int(args.block_size)
    image_meta["hardware_adaptation"] = hardware_meta
    image_meta["predictor"] = bool(effective_predictor)
    image_meta["pyramidal"] = bool(args.pyramid and args.output_format == "ome_tiff")
    if output_scale_max is not None:
        image_meta["storage_scale_max"] = float(output_scale_max)

    preview_rgb = read_crop_image(args.image, args.page, image_meta) if not image_meta["blockwise"] else None
    input_shape = preview_rgb.shape if preview_rgb is not None else tuple(image_meta["inference_shape_yx"]) + (3,)
    print(f"[INFO] GigaTIME input image shape={input_shape} dtype=uint8", flush=True)
    if image_meta["downsample_applied"]:
        print(
            "[INFO] GigaTIME downsampled large image "
            f"{tuple(image_meta['original_shape_yx'])} -> {tuple(image_meta['inference_shape_yx'])} "
            f"(factor={image_meta['downsample_factor']})",
            flush=True,
        )
    if image_meta.get("source_mpp"):
        print(
            "[INFO] GigaTIME physical scale "
            f"source_mpp={image_meta['source_mpp']:.4f} "
            f"effective_mpp={image_meta['effective_mpp']:.4f} "
            f"target_mpp={image_meta['target_mpp']:.4f}",
            flush=True,
        )
    print(f"[INFO] GigaTIME device={device} repo_id={args.repo_id} offline={offline}", flush=True)
    print(f"[INFO] GigaTIME tiling patch_size={args.patch_size} stride={args.stride} blend=raised_cosine", flush=True)
    print(
        f"[INFO] GigaTIME saving JPEG markers={','.join(jpg_channel_names)} "
        f"(count={len(jpg_channel_names)})",
        flush=True,
    )
    if args.output_format != "none":
        print(
            f"[INFO] GigaTIME persisted store channels={','.join(output_channel_names)} "
            f"(count={len(output_channel_names)})",
            flush=True,
        )
    print(
            "[INFO] GigaTIME output budget "
            f"estimated_output_gib={image_meta['estimated_prediction_gib']:.2f} "
            f"selection_reason={image_meta['selection_reason']} "
            f"output_format={args.output_format} output_dtype={args.output_dtype}",
            flush=True,
        )

    model = load_model(args.repo_id, token=token, device=device, offline=offline)
    target_shape = tuple(int(v) for v in image_meta["inference_shape_yx"])
    quantifiers = build_quantifiers(
        nuclei_mask_path=str(args.nuclei_mask or ""),
        cyto_mask_path=str(args.cyto_mask or ""),
        target_shape=target_shape,
        channel_names=list(output_channel_names if args.output_format != "none" else jpg_channel_names),
        block_size=max(512, int(args.block_size)),
    )
    quant_dir = Path(args.quant_dir).resolve() if str(args.quant_dir or "").strip() else None
    jpg_exporter = None
    if jpg_channel_names:
        selected_for_jpg = output_channel_indices if args.output_format != "none" else list(range(len(CHANNEL_NAMES)))
        jpg_channel_positions = [selected_for_jpg.index(idx) for idx in jpg_channel_indices if idx in selected_for_jpg]
        jpg_exporter = JpegTileExporter(
            outdir=Path(args.outdir),
            full_shape_yx=target_shape,
            tile_size=int(args.block_size),
            quality=int(args.jpg_quality),
            preview_max_side=int(args.jpg_preview_max_side),
            save_tiles=bool(args.jpg_save_tiles),
            channel_indices=jpg_channel_positions,
            channel_names=jpg_channel_names,
        )
    if image_meta["blockwise"]:
        print(f"[INFO] GigaTIME accumulation mode=blockwise_tiled tile_size={args.block_size}", flush=True)
        if args.output_format == "zarr":
            blockwise_write_zarr_outputs(
                source_path=args.image,
                page=args.page,
                outdir=Path(args.outdir),
                model=model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                factor=int(image_meta["downsample_factor"]),
                tile_size=args.block_size,
                output_dtype=args.output_dtype,
                metadata=image_meta,
                skip_background_blocks=bool(args.skip_background_blocks),
                skip_background_mask_path=str(args.skip_background_mask_path or ""),
                skip_background_downsample=int(args.skip_background_downsample),
                skip_background_min_fraction=float(args.skip_background_min_fraction),
                skip_background_close_radius=int(args.skip_background_close_radius),
                skip_background_min_obj_area=int(args.skip_background_min_obj_area),
                skip_background_hole_area=int(args.skip_background_hole_area),
                output_channel_indices=output_channel_indices,
                output_channel_names=output_channel_names,
                quantifiers=quantifiers,
                quant_dir=quant_dir,
                jpg_exporter=jpg_exporter,
                jpg_channel_indices=jpg_channel_indices,
                sample_id=sample_id,
            )
            out_path = Path(args.outdir) / "gigatime_probs.zarr"
        elif args.output_format == "ome_tiff":
            blockwise_write_ometiff_outputs(
                source_path=args.image,
                page=args.page,
                outdir=Path(args.outdir),
                model=model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                factor=int(image_meta["downsample_factor"]),
                tile_size=args.block_size,
                output_dtype=effective_output_dtype,
                compression=args.compression,
                predictor=bool(effective_predictor),
                jpeg_quality=int(args.jpg_quality),
                pyramid=bool(args.pyramid),
                metadata=image_meta,
                output_channel_indices=output_channel_indices,
                output_channel_names=output_channel_names,
                quantifiers=quantifiers,
                quant_dir=quant_dir,
                jpg_exporter=jpg_exporter,
                jpg_channel_indices=jpg_channel_indices,
                sample_id=sample_id,
                resume_level0_buffer=bool(args.resume_level0_buffer),
                skip_background_blocks=bool(args.skip_background_blocks),
                skip_background_mask_path=str(args.skip_background_mask_path or ""),
                skip_background_downsample=int(args.skip_background_downsample),
                skip_background_min_fraction=float(args.skip_background_min_fraction),
                skip_background_close_radius=int(args.skip_background_close_radius),
                skip_background_min_obj_area=int(args.skip_background_min_obj_area),
                skip_background_hole_area=int(args.skip_background_hole_area),
            )
            out_path = Path(args.outdir) / "gigatime_probs.ome.tif"
        else:
            blockwise_process_without_store(
                source_path=args.image,
                page=args.page,
                outdir=Path(args.outdir),
                model=model,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                factor=int(image_meta["downsample_factor"]),
                tile_size=int(args.block_size),
                metadata=image_meta,
                skip_background_blocks=bool(args.skip_background_blocks),
                skip_background_mask_path=str(args.skip_background_mask_path or ""),
                skip_background_downsample=int(args.skip_background_downsample),
                skip_background_min_fraction=float(args.skip_background_min_fraction),
                skip_background_close_radius=int(args.skip_background_close_radius),
                skip_background_min_obj_area=int(args.skip_background_min_obj_area),
                skip_background_hole_area=int(args.skip_background_hole_area),
                quantifiers=quantifiers,
                quant_dir=quant_dir,
                jpg_exporter=jpg_exporter,
                jpg_channel_indices=jpg_channel_indices,
                sample_id=sample_id,
            )
            out_path = Path(args.outdir)
        print(
            f"[OK] GigaTIME wrote {out_path} "
            f"shape={((len(output_channel_names) if args.output_format != 'none' else len(jpg_channel_names)), image_meta['inference_shape_yx'][0], image_meta['inference_shape_yx'][1])}"
        )
        return

    disk_backed = float(image_meta["estimated_prediction_gib"]) > float(args.disk_buffer_threshold_gib)
    print(f"[INFO] GigaTIME accumulation mode={'disk' if disk_backed else 'memory'}")
    accum, counts, valid_shape = run_inference(
        image_rgb=preview_rgb,
        model=model,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        disk_backed=disk_backed,
        scratch_dir=Path(args.outdir),
    )
    write_outputs(
        Path(args.outdir),
        accum,
        counts,
        valid_shape,
        args.compression,
        bool(args.predictor),
        image_meta,
        output_channel_indices,
        output_channel_names,
    )
    print(
        f"[OK] GigaTIME wrote {(Path(args.outdir) / 'gigatime_probs.ome.tif')} "
        f"shape={(len(output_channel_names), valid_shape[0], valid_shape[1])}"
    )


if __name__ == "__main__":
    main()
