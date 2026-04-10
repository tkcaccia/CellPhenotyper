#!/usr/bin/env python3
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from huggingface_hub import snapshot_download
import tifffile
import torch
from torch import nn


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


def read_crop_image(path: str, page: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        if page < 0 or page >= len(tf.pages):
            raise ValueError(f"--page {page} out of range for {path} ({len(tf.pages)} pages)")
        arr = tf.pages[page].asarray()

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


def normalize_patch_batch(batch_rgb: np.ndarray) -> torch.Tensor:
    batch = batch_rgb.astype(np.float32) / 255.0
    batch = (batch - MEAN[None, None, None, :]) / STD[None, None, None, :]
    batch = np.transpose(batch, (0, 3, 1, 2))
    return torch.from_numpy(batch)


def pad_to_tiling_shape(image: np.ndarray, patch_size: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w, _ = image.shape
    if h <= patch_size:
        steps_y = 1
    else:
        steps_y = int(math.ceil((h - patch_size) / float(stride))) + 1
    if w <= patch_size:
        steps_x = 1
    else:
        steps_x = int(math.ceil((w - patch_size) / float(stride))) + 1

    pad_h = max(0, (steps_y - 1) * stride + patch_size - h)
    pad_w = max(0, (steps_x - 1) * stride + patch_size - w)
    mode = "reflect" if h > 1 and w > 1 else "edge"
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    positions = [(y, x) for y in range(0, padded.shape[0] - patch_size + 1, stride) for x in range(0, padded.shape[1] - patch_size + 1, stride)]
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


def run_inference(
    image_rgb: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
) -> np.ndarray:
    padded, positions = pad_to_tiling_shape(image_rgb, patch_size, stride)
    h, w = image_rgb.shape[:2]
    padded_h, padded_w = padded.shape[:2]

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

    counts = np.maximum(counts, 1.0)
    merged = accum / counts
    return merged[:, :h, :w].astype(np.float32, copy=False)


def write_outputs(outdir: Path, predictions: np.ndarray, compression: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        outdir / "gigatime_probs.ome.tif",
        predictions,
        metadata={"axes": "CYX", "Channel": {"Name": CHANNEL_NAMES}},
        compression=compression,
        bigtiff=predictions.nbytes >= (4 * 1024 * 1024 * 1024),
    )
    (outdir / "gigatime_channels.json").write_text(json.dumps(CHANNEL_NAMES, indent=2), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Run GigaTIME virtual mIF inference on a crop image.")
    ap.add_argument("--image", required=True, help="Crop image TIFF path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--repo-id", default="prov-gigatime/GigaTIME", help="Hugging Face repo ID")
    ap.add_argument("--page", type=int, default=0, help="TIFF page to read")
    ap.add_argument("--patch-size", type=int, default=256, help="Inference patch size (default from model card: 256)")
    ap.add_argument("--stride", type=int, default=128, help="Sliding-window stride")
    ap.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    ap.add_argument("--compression", default="deflate", help="Output TIFF compression")
    return ap.parse_args()


def main():
    args = parse_args()
    token = (os.environ.get("HF_TOKEN") or "").strip() or None
    offline = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    device = resolve_device(args.device)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    image_rgb = read_crop_image(args.image, args.page)
    print(f"[INFO] GigaTIME input image shape={image_rgb.shape} dtype={image_rgb.dtype}")
    print(f"[INFO] GigaTIME device={device} repo_id={args.repo_id} offline={offline}")
    print(f"[INFO] GigaTIME tiling patch_size={args.patch_size} stride={args.stride} blend=raised_cosine")

    model = load_model(args.repo_id, token=token, device=device, offline=offline)
    predictions = run_inference(
        image_rgb=image_rgb,
        model=model,
        device=device,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    write_outputs(Path(args.outdir), predictions, args.compression)
    print(
        f"[OK] GigaTIME wrote {(Path(args.outdir) / 'gigatime_probs.ome.tif')} "
        f"shape={tuple(predictions.shape)}"
    )


if __name__ == "__main__":
    main()
