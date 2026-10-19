#!/usr/bin/env python3
"""Small runtime hardware probes shared by GPU inference wrappers."""

from __future__ import annotations

import os
import subprocess


def assigned_gpu_selector(logical_index: int | str = 0) -> str:
    """Map a container-visible CUDA index to the physical nvidia-smi selector."""
    assigned = os.environ.get("CELLPHENOTYPER_GPU_INDEX", "").strip()
    if assigned:
        return assigned

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible.lower() not in {"none", "void", "no", "false", "-1"}:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        try:
            index = int(logical_index)
        except (TypeError, ValueError):
            index = 0
        if 0 <= index < len(devices):
            return devices[index]
    return str(logical_index)


def query_gpu_memory_mib(logical_index: int | str = 0) -> tuple[int, int]:
    """Return usable-free and total MiB for the assigned GPU.

    A small safety margin is removed from the live free-memory reading because
    CUDA context creation and model loading occur after this probe.
    """
    selector = assigned_gpu_selector(logical_index)
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            selector,
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    fields = [field.strip() for field in output.strip().splitlines()[0].split(",")]
    if len(fields) != 2:
        raise RuntimeError(f"Unexpected nvidia-smi memory output: {output!r}")
    free_mib, total_mib = (int(float(field)) for field in fields)
    reserve_mib = max(768, min(4096, int(total_mib * 0.08)))
    return max(0, min(free_mib, total_mib) - reserve_mib), total_mib


def auto_batch_from_free_vram(
    requested: int,
    logical_index: int | str,
    tiers: tuple[tuple[int, int], ...],
    fallback: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Choose a batch from live usable VRAM unless the user set it explicitly."""
    if requested > 0:
        value = requested
    else:
        try:
            usable_mib, _ = query_gpu_memory_mib(logical_index)
            value = fallback
            for threshold_mib, candidate in sorted(tiers, reverse=True):
                if usable_mib >= threshold_mib:
                    value = candidate
                    break
        except Exception:
            value = fallback
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value
