#!/usr/bin/env python3
"""Pure hardware-policy helpers for GigaTIME runtime adaptation."""

from __future__ import annotations

import math


def choose_gigatime_hardware_settings(
    *,
    enabled: bool,
    profile: str,
    requested_batch: int,
    requested_block: int,
    requested_output_gib: float,
    max_auto_batch: int,
    max_auto_block: int,
    max_auto_output_gib: float,
    task_memory_gib: float | None,
    min_free_system_gib: float,
    system_mem_available_gib: float | None,
    cuda_mem_free_gib: float | None,
    cuda_mem_total_gib: float | None,
    use_cuda: bool,
) -> dict:
    requested = {
        "batch_size": int(requested_batch),
        "block_size": int(requested_block),
        "max_output_gib": float(requested_output_gib),
    }
    profile = str(profile or "balanced").strip().lower()
    if profile not in {"conservative", "balanced", "aggressive"}:
        profile = "balanced"
    if not enabled:
        return {
            "enabled": False,
            "profile": profile,
            "requested": requested,
            "effective": dict(requested),
            "system_mem_available_gib": system_mem_available_gib,
            "cuda_mem_free_gib": cuda_mem_free_gib,
            "cuda_mem_total_gib": cuda_mem_total_gib,
        }

    reserve = float(min_free_system_gib)
    if system_mem_available_gib is None:
        usable_mem = max(0.0, task_memory_gib - reserve) if task_memory_gib is not None else None
    else:
        host_usable = max(0.0, float(system_mem_available_gib) - reserve)
        usable_mem = min(host_usable, task_memory_gib) if task_memory_gib is not None else host_usable
    if usable_mem is not None and usable_mem < 4.0:
        raise RuntimeError(
            "GigaTIME cannot run safely: estimated usable system memory after reserve is "
            f"{usable_mem:.2f} GiB. Increase RAM/swap or lower concurrent workload before running."
        )

    batch_cap = max(1, int(max_auto_batch))
    block_cap = max(256, int(max_auto_block))
    budget_cap = max(0.25, float(max_auto_output_gib))

    mem_batch, mem_block, mem_budget = 1, 512, 0.75
    if usable_mem is None:
        mem_block, mem_budget = 1024, 2.0
    elif usable_mem < 8.0:
        mem_block, mem_budget = 512, 0.75
    elif usable_mem < 12.0:
        mem_batch, mem_block, mem_budget = 2, 768, 1.25
    elif usable_mem < 20.0:
        mem_batch, mem_block, mem_budget = 4, 1024, 2.0
    elif usable_mem < 32.0:
        mem_batch, mem_block, mem_budget = 6, 1536, 4.0
    elif usable_mem < 64.0:
        mem_batch, mem_block, mem_budget = 8, 2048, 8.0
    else:
        mem_batch, mem_block, mem_budget = 16, 3072, 16.0

    gpu_batch, gpu_block = batch_cap, block_cap
    if use_cuda and cuda_mem_free_gib is None:
        gpu_batch = max(1, int(requested_batch))
        gpu_block = min(block_cap, max(256, int(requested_block)))
    elif use_cuda and cuda_mem_free_gib is not None:
        if cuda_mem_free_gib < 4.0:
            gpu_batch, gpu_block = 1, 512
        elif cuda_mem_free_gib < 8.0:
            gpu_batch, gpu_block = 2, 1024
        elif cuda_mem_free_gib < 12.0:
            gpu_batch, gpu_block = 4, 1024
        elif cuda_mem_free_gib < 20.0:
            gpu_batch, gpu_block = 8, 1536
        elif cuda_mem_free_gib < 32.0:
            gpu_batch, gpu_block = 12, 2048
        elif cuda_mem_free_gib < 48.0:
            gpu_batch, gpu_block = 16, 2048
        else:
            gpu_batch, gpu_block = 24, 3072

    scale = {
        "conservative": {"batch": 0.5, "block": 0.75, "budget": 0.5},
        "balanced": {"batch": 1.0, "block": 1.0, "budget": 1.0},
        "aggressive": {"batch": 1.5, "block": 1.25, "budget": 1.5},
    }[profile]
    # CUDA batches consume VRAM. Host RAM controls region and output buffers,
    # but must not throttle a capable GPU to batch one or two.
    batch_target = gpu_batch if use_cuda else min(mem_batch, gpu_batch)
    auto_batch = max(1, int(math.floor(batch_target * scale["batch"])))
    auto_block = max(256, int(math.floor(min(mem_block, gpu_block) * scale["block"])))
    auto_budget = max(0.25, float(min(mem_budget, budget_cap) * scale["budget"]))

    effective_batch = min(batch_cap, max(max(1, int(requested_batch)), auto_batch))
    effective_block = min(block_cap, max(max(256, int(requested_block)), auto_block))
    effective_budget = min(budget_cap, max(max(0.25, float(requested_output_gib)), auto_budget))
    effective_batch = min(effective_batch, max(1, int(gpu_batch if use_cuda else min(mem_batch, gpu_batch))))
    effective_block = min(effective_block, max(256, int(min(mem_block, gpu_block))))
    effective_budget = min(effective_budget, max(0.25, float(mem_budget)))
    effective_block = max(256, (int(effective_block) // 16) * 16)

    return {
        "enabled": True,
        "profile": profile,
        "requested": requested,
        "effective": {
            "batch_size": int(effective_batch),
            "block_size": int(effective_block),
            "max_output_gib": float(effective_budget),
        },
        "limits": {
            "max_auto_batch": int(batch_cap),
            "max_auto_block_size": int(block_cap),
            "max_auto_output_gib": float(budget_cap),
        },
        "derived_caps": {
            "memory_batch": int(mem_batch),
            "memory_block_size": int(mem_block),
            "memory_output_gib": float(mem_budget),
            "gpu_batch": int(gpu_batch),
            "gpu_block_size": int(gpu_block),
        },
        "system_mem_available_gib": system_mem_available_gib,
        "task_memory_gb": task_memory_gib,
        "min_free_system_gb": reserve,
        "usable_system_mem_gib": usable_mem,
        "cuda_mem_free_gib": cuda_mem_free_gib,
        "cuda_mem_total_gib": cuda_mem_total_gib,
    }
