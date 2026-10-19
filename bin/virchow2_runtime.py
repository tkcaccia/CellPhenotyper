#!/usr/bin/env python3
"""Pinned Virchow2 inference shared by non-overlapping tile and Bora passes.

Importing this module never starts a GPU job. Every model call is made by an
explicit pipeline stage on its allocated device.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
CHECKPOINT_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
PATCH = 224
TOKEN_PATCH = 14
CHANNELS = 1280


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, device: str = "cuda"):
    """Load only the audited Virchow2 checkpoint, with no hub fallback."""
    import torch
    import timm
    from timm.layers import SwiGLUPacked
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != REVISION or sha256(checkpoint / "model.safetensors") != CHECKPOINT_SHA256:
        raise ValueError("Virchow2 checkpoint revision or SHA256 differs from Bora's audited model")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = timm.create_model(
            config["architecture"], pretrained=False, **config["model_args"],
            mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,
        )
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True, assign=True)
    return model.to(device).eval().requires_grad_(False)


def _forward(model, patches: np.ndarray, device: str) -> np.ndarray:
    import torch

    if patches.dtype != np.uint8 or patches.ndim != 4 or patches.shape[1:] != (PATCH, PATCH, 3):
        raise ValueError("Virchow2 expects N x 224 x 224 x 3 uint8 RGB tiles")
    data = torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).to(device).float().div_(255)
    mean = data.new_tensor([.485, .456, .406])[None, :, None, None]
    std = data.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
        tokens = model((data - mean) / std)
    if tokens.ndim != 3 or tokens.shape[1:] != (261, CHANNELS):
        raise ValueError(f"Unexpected Virchow2 token shape: {tuple(tokens.shape)}")
    return tokens.float().cpu().numpy()


def pooled_tiles(model, patches: np.ndarray, device: str = "cuda") -> np.ndarray:
    values = _forward(model, patches, device)
    # Bora's full-token field uses patch tokens; KODAMA uses a tile summary.
    return np.concatenate((values[:, 0, :], values[:, 5:, :].mean(1)), axis=1).astype(np.float32)


def spatial_tokens_nonoverlap(model, image: np.ndarray, *, device: str = "cuda", batch: int = 4):
    """Encode one calibrated field without overlapping encoder windows."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 RGB working H&E")
    if batch < 1:
        raise ValueError("Positive batch size required")
    height, width = image.shape[:2]
    rows, cols = (height + PATCH - 1) // PATCH, (width + PATCH - 1) // PATCH
    padded = np.pad(
        image, ((0, rows * PATCH - height), (0, cols * PATCH - width), (0, 0)),
        mode="constant", constant_values=255,
    )
    output = np.empty((rows * 16, cols * 16, CHANNELS), dtype=np.float16)
    positions = [(row, col) for row in range(rows) for col in range(cols)]
    for start in range(0, len(positions), batch):
        group = positions[start:start + batch]
        patches = np.stack([padded[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
                            for r, c in group])
        values = _forward(model, patches, device)[:, 5:, :].reshape(-1, 16, 16, CHANNELS)
        for (row, col), value in zip(group, values):
            output[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16] = value
    cropped = output[:(height + TOKEN_PATCH - 1) // TOKEN_PATCH,
                     :(width + TOKEN_PATCH - 1) // TOKEN_PATCH]
    if not np.isfinite(cropped).all():
        raise ValueError("Nonfinite Virchow2 spatial tokens")
    return cropped, dict(tile_size_px=PATCH, tile_stride_px=PATCH,
                         tile_overlap_px=0, grid_rows=rows, grid_cols=cols,
                         spatial_token_shape=list(cropped.shape),
                         source_padding="white", excluded_tokens="CLS plus four registers")
