#!/usr/bin/env python3
"""Import one separately approved Chiamaka CellViT case; never launch a GPU.

The frozen Chiamaka lifecycle adapter owns the GPU case and its authorization.
This CPU-only bridge verifies that adapter's bound terminal result, then copies
its immutable scientific output into one canonical CellPhenotyper shard.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

from cellvit_models import digest_file
from cellvit_shards import atomic_json, get_shard, validate_plan, verify_complete_shard
from cellvit_fullslide_action import (
    _adopt_shard, _require_capacity, _require_declared_coverage,
    _require_prior_cleanup,
)

RESULT_SCHEMA = "pm016-cellvit-case-result-v1"
BINDING_SCHEMA = "pm016-cellvit-case-binding-v1"
CHECKPOINT_SCHEMA = "pm016-cellvit-case-checkpoint-v1"
GiB = 1024 ** 3


def _checked_json(path: Path, expected_sha: str, label: str) -> dict:
    if not path.is_file() or path.is_symlink() or digest_file(path) != expected_sha:
        raise ValueError(f"{label} is absent, symlinked or differs from its SHA-256 pin")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _checked_case(args, plan: dict, shard: dict) -> tuple[Path, dict, dict]:
    case = args.case_dir.resolve(strict=True)
    if case.name != shard["id"] or case.parent.name != "cases":
        raise ValueError("Case path must be the exact cases/<shard-id> directory")
    binding_path = case / "GPU_CASE_BINDING.json"
    binding = _checked_json(binding_path, args.binding_sha256, "approved case binding")
    source_manifest = Path(binding.get("source_manifest_path", ""))
    bound_plan = Path(binding.get("plan_path", ""))
    if (not source_manifest.is_absolute() or not source_manifest.is_file() or
            source_manifest.is_symlink() or
            digest_file(source_manifest) != args.gpu_source_sha256 or
            binding.get("source_manifest_sha256") != args.gpu_source_sha256 or
            not bound_plan.is_absolute() or not bound_plan.is_file() or
            bound_plan.is_symlink() or
            digest_file(bound_plan) != args.plan_file_sha256):
        raise ValueError("Frozen GPU source manifest or plan bytes differ from approval")
    result_path = case / "GPU_CASE_RESULT.json"
    result = _checked_json(result_path, args.case_result_sha256, "terminal case result")
    marker = case / "output" / "complete.json"
    checkpoint_path = case / "GPU_CASE_CHECKPOINT.json"
    supervisor_path = case / "SUPERVISOR_RESULT.json"
    if (marker.is_symlink() or checkpoint_path.is_symlink() or
            supervisor_path.is_symlink() or not marker.is_file() or
            not checkpoint_path.is_file() or not supervisor_path.is_file()):
        raise ValueError("Bound scientific checkpoint or supervisor evidence is missing")
    if (binding.get("schema") != BINDING_SCHEMA or
            binding.get("host") != "Chiamaka" or binding.get("uct_excluded") is not True or
            binding.get("shard_id") != shard["id"] or
            binding.get("case_dir") != str(case) or
            binding.get("plan_sha256") != plan["identity_sha256"] or
            binding.get("plan_file_sha256") != args.plan_file_sha256 or
            binding.get("image_sha256") != plan["image"]["sha256"] or
            binding.get("gpu_source_sha256") != args.gpu_source_sha256 or
            not isinstance(binding.get("runtime_sha256"), str)):
        raise ValueError("Case binding differs from the exact PM016 source, image or plan")
    if (result.get("schema") != RESULT_SCHEMA or result.get("success") is not True or
            result.get("payload_returncode") != 0 or
            result.get("cleanup_verified") is not True or result.get("failures") != [] or
            result.get("case_id") != binding.get("case_id") or
            result.get("shard_id") != shard["id"] or
            result.get("plan_sha256") != plan["identity_sha256"] or
            result.get("binding_sha256") != args.binding_sha256 or
            result.get("approved_scope_sha256") != binding.get("approved_scope_sha256") or
            result.get("gpu_source_sha256") != args.gpu_source_sha256 or
            result.get("runtime_sha256") != binding["runtime_sha256"] or
            result.get("scope") != binding.get("scope") or
            result.get("complete_sha256") != digest_file(marker) or
            result.get("checkpoint_sha256") != digest_file(checkpoint_path) or
            result.get("supervisor_evidence_sha256") != digest_file(supervisor_path)):
        raise ValueError("Frozen one-case terminal result is failed or differs from bound evidence")
    cleanup = result.get("cleanup", {})
    if any(cleanup.get(key) is not True for key in (
            "owned_processes_absent", "scope_inactive",
            "assigned_gpu_contexts_baseline_only", "bounded_checks_complete")):
        raise ValueError("Frozen result lacks bounded terminal process/GPU-context cleanup")
    if (re.fullmatch(r"[0-9a-f]{64}", result.get("gpu_context_baseline_sha256") or "") is None or
            result["gpu_context_baseline_sha256"] != cleanup.get("gpu_context_baseline_sha256")):
        raise ValueError("Frozen assigned-GPU context baseline is unbound")
    resources = result.get("resources", {})
    if (resources.get("allocated_gpus") != 1 or resources.get("cpus") != 2 or
            type(resources.get("peak_memory_bytes")) is not int or
            not 0 <= resources["peak_memory_bytes"] <= 12 * GiB):
        raise ValueError("Frozen case resources differ from the one-GPU envelope")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if (checkpoint.get("schema") != CHECKPOINT_SCHEMA or
            checkpoint.get("complete") is not True or
            checkpoint.get("case_id") != binding["case_id"] or
            checkpoint.get("binding_sha256") != args.binding_sha256 or
            checkpoint.get("plan_sha256") != plan["identity_sha256"] or
            checkpoint.get("complete_sha256") != digest_file(marker)):
        raise ValueError("Application checkpoint is not bound to the case and shard")
    supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
    events = supervisor.get("events", [])
    finals = [event for event in events if event.get("event") == "final_assigned_contexts"]
    if (supervisor.get("success") is not True or
            supervisor.get("cleanup_verified") is not True or
            supervisor.get("payload_returncode") != 0 or
            supervisor.get("cpu_test") is not False or
            len(finals) < 2 or any(
                not isinstance(event.get("contexts"), list) or
                any(row.get("owner") != "baseline" for row in event["contexts"])
                for event in finals[-2:]) or
            not any(event.get("event") == "two_empty_owned_snapshots" for event in events)):
        raise ValueError("Supervisor terminal GPU and owned-process observations are incomplete")
    if result.get("cell_count") != checkpoint.get("cell_count"):
        raise ValueError("Case result and checkpoint cell counts differ")
    verified = verify_complete_shard(plan, shard, case / "output")
    if (digest_file(marker) != result["complete_sha256"] or
            verified["cell_count"] != result["cell_count"]):
        raise ValueError("Scientific output changed after the frozen result")
    return case, binding, result


def import_case(args) -> dict:
    """Copy one checked case into the only canonical shard without a GPU action."""
    root = args.bundle_dir.resolve(strict=True)
    if not args.bundle_dir.is_absolute() or args.reserve_gib < 0 or args.case_budget_gib <= 0:
        raise ValueError("Persistent absolute bundle and positive disk budget are required")
    lock = root.parent / f".{root.name}.cellvit-action.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / "cellvit_shard_plan.json"
        if digest_file(plan_path) != args.plan_file_sha256:
            raise ValueError("Canonical plan bytes differ from case approval")
        plan = validate_plan(json.loads(plan_path.read_text(encoding="utf-8")),
                             args.image, args.shift)
        _require_declared_coverage(plan, args.expected_shards)
        shard = get_shard(plan, args.shard_id)
        _require_prior_cleanup(root, plan)
        _require_capacity(root, args.reserve_gib, args.case_budget_gib)
        case, binding, result = _checked_case(args, plan, shard)
        run_root_value = binding.get("run_root")
        if not isinstance(run_root_value, str) or not Path(run_root_value).is_absolute():
            raise ValueError("Approved case must bind an absolute campaign run root")
        run_root = Path(run_root_value).resolve(strict=True)
        if (root != case.parent.parent or root == run_root or
                not root.is_relative_to(run_root)):
            raise ValueError("Approved case must bind this canonical CellPhenotyper result root")
        receipt = {
            "schema": "cellvit-bound-case-cleanup-v1",
            "attestation_state": "published", "draft": False,
            "shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
            "checkpoint_sha256": result["complete_sha256"],
            "cleanup_verified": True, "owned_processes_absent": True,
            "scope_inactive": True, "assigned_gpu_contexts_baseline_only": True,
            "gpu_context_baseline_sha256": result["gpu_context_baseline_sha256"],
            "bounded_checks_complete": True,
            "case_result_sha256": args.case_result_sha256,
            "case_binding_sha256": args.binding_sha256,
            "supervisor_evidence_sha256": result["supervisor_evidence_sha256"],
            "approved_scope_sha256": result["approved_scope_sha256"],
        }
        with tempfile.TemporaryDirectory(prefix=".case-import-", dir=root) as temporary:
            attestation = Path(temporary) / "CELLVIT_GPU_CLEANUP.json"
            atomic_json(attestation, receipt)
            adopted = _adopt_shard(SimpleNamespace(
                adopt_from=case / "output", expected_checkpoint_sha256=result["complete_sha256"],
                adoption_cleanup_receipt=attestation,
                adoption_cleanup_receipt_sha256=digest_file(attestation),
                allow_case_owned_source=True), root, plan, shard)
        return {"schema": "cellvit-approved-case-import-v1",
                "shard_id": shard["id"], "case_result_sha256": args.case_result_sha256,
                "canonical_checkpoint_sha256": adopted["checkpoint_sha256"],
                "cell_count": adopted["cell_count"], "gpu_started": False}
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--case-dir", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--shift", required=True, type=Path)
    p.add_argument("--shard-id", required=True)
    p.add_argument("--case-result-sha256", required=True)
    p.add_argument("--binding-sha256", required=True)
    p.add_argument("--gpu-source-sha256", required=True)
    p.add_argument("--plan-file-sha256", required=True)
    p.add_argument("--expected-shards", type=int, default=104)
    p.add_argument("--reserve-gib", type=float, default=40.0)
    p.add_argument("--case-budget-gib", type=float, default=4.0)
    return p


if __name__ == "__main__":
    print(json.dumps(import_case(parser().parse_args()), sort_keys=True))
