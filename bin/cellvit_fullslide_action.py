#!/usr/bin/env python3
"""CellViT++ full-slide bridge with a sequential host-run campaign action.

Planning, pilot adoption, storage conversion and finalization are CPU-only.
The ``run`` action launches one bounded native case at a time through the
Chiamaka lifecycle supervisor, then verifies it before the next core.
Standalone ``shard`` remains disabled. No failed case is retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
from argparse import Namespace

from cellvit_models import digest_file, preflight_models
from cellvit_runtime_binding import verified_pydeps
from cellvit_shards import atomic_json, get_shard, validate_plan

HEADS = ("pannuke", "consep", "lizard", "nucls_main", "nucls_super",
         "ocelot", "midog", "panoptils")
CLEANUP_FILE = "CELLVIT_GPU_CLEANUP.json"
CPU_EMPTY_SCHEMA = "cellvit-clean-mask-empty-cpu-v1"
FAILURE_FILE = "GPU_BUNDLE_FAILED.json"
ACTIVE_FILE = "GPU_CASE_ACTIVE.json"
COMPLETE_SCHEMA = "cellphenotyper-cellvit-fullslide-complete-v2"
CAMPAIGN_SOURCE_FILE = "CELLVIT_CAMPAIGN_SOURCE.json"
ACCEPTED_CLEANUP_SCHEMAS = {
    "cellvit-pilot-cleanup-attestation-v1",
    "cellvit-bound-case-cleanup-v1",
    "cellvit-direct-supervisor-cleanup-v1",
    CPU_EMPTY_SCHEMA,
}
PM016_IMAGE_SHA256 = "dc379f5336ce5341bc4fd07a889100396cae4319a899c0fcc82eeb786c3d74a7"
PILOT_SCHEMA = "cellvit-pilot-cleanup-attestation-v1"
PILOT_ARTIFACTS = ("binding", "supervisor_result", "application_checkpoint", "complete_marker")


def _published_pilot_provenance(root: Path, plan: dict, shard: dict,
                                receipt: dict, marker_sha: str) -> None:
    """Check a *published data-adoption* claim against preserved read-only evidence.

    Historical pilots cannot inherit the later one-case baseline/guardian claim.
    Their current-state audit and original successful case are separate evidence.
    """
    if receipt.get("schema") != PILOT_SCHEMA:
        return
    audit_path = Path(receipt.get("readonly_verification_path") or "")
    audit_sha = receipt.get("readonly_verification_sha256")
    if not audit_path.is_file() and isinstance(audit_sha, str):
        # A copied attestation retains its publisher's absolute path. The
        # identical audit may be packaged under this single result root.
        audit_path = root / "evidence" / "READONLY_VERIFICATION.json"
    if (not audit_path.is_absolute() or not audit_path.is_file() or
            audit_path.is_symlink() or
            digest_file(audit_path) != audit_sha):
        raise ValueError("Published pilot read-only verification file/hash differs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    try:
        published = datetime.fromisoformat(receipt["published_utc"].replace("Z", "+00:00"))
        observed = datetime.fromisoformat(audit["finished_utc"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError("Published pilot timestamps are absent or invalid") from error
    hashes = receipt.get("source_artifacts_sha256", {})
    paths = receipt.get("remote_evidence_paths", {})
    final = receipt.get("final_recheck", {})
    cases = audit.get("cases", [])
    matching = [item for item in cases if item.get("shard_id") == shard["id"] and
                item.get("case_id") == receipt.get("source_case_id")]
    if (not isinstance(receipt.get("attestor"), str) or
            not receipt["attestor"].strip() or published.tzinfo is None or
            observed.tzinfo is None or published < observed or
            receipt.get("plan_file_sha256") != digest_file(root / "cellvit_shard_plan.json") or
            audit.get("plan_identity_sha256") != plan["identity_sha256"] or
            audit.get("plan_file_sha256") != receipt["plan_file_sha256"] or
            audit.get("nvidia_smi_exit_code") != 0 or
            audit.get("assigned_gpu_compute_contexts") != [] or
            receipt.get("source_host") != audit.get("host") or
            receipt.get("assigned_gpu_uuid") != audit.get("assigned_gpu_uuid") or
            not isinstance(hashes, dict) or set(hashes) != set(PILOT_ARTIFACTS) or
            not isinstance(paths, dict) or
            set(paths) != set(PILOT_ARTIFACTS) | {"plan"} or
            any(not isinstance(paths[key], str) or
                not Path(paths[key]).is_absolute() for key in paths) or
            not isinstance(final, dict) or
            final.get("hash_mismatches") != 0 or
            final.get("saved_supervisor_success") is not True or
            final.get("saved_supervisor_cleanup_verified") is not True or
            final.get("scope_inactive") is not True or
            final.get("assigned_gpu_compute_contexts") != 0 or
            final.get("r7_administrator_gate_resolved") is not False or
            hashes.get("complete_marker") != marker_sha or len(matching) != 1):
        raise ValueError("Published pilot provenance differs from the exact plan/case/audit")
    case = matching[0]
    expected = {"binding": "binding_sha256",
                "supervisor_result": "supervisor_result_sha256",
                "application_checkpoint": "checkpoint_file_sha256",
                "complete_marker": "complete_marker_sha256"}
    if (receipt.get("source_scope") != case.get("scope") or
            any(hashes[key] != case.get(field) for key, field in expected.items()) or
            case.get("bound_file_hash_mismatches") != [] or
            case.get("declared_output_hash_mismatches") != [] or
            case.get("checkpoint_output_hash_mismatches") != [] or
            case.get("saved_supervisor_success") is not True or
            case.get("saved_supervisor_cleanup_verified") is not True or
            case.get("saved_payload_returncode") != 0 or
            case.get("saved_supervisor_failures") != [] or
            case.get("saved_two_empty_owned_snapshots_event") is not True or
            case.get("saved_final_assigned_contexts") != [[], []] or
            case.get("same_recorded_births_alive") != [] or
            case.get("active_marker_present") is not False or
            case.get("failure_marker_present") is not False or
            case.get("scope_properties", {}).get("ControlGroup") != "" or
            case.get("scope_properties", {}).get("ActiveState") != "inactive"):
        raise ValueError("Published pilot lacks exact historical case and current cleanup evidence")


def _require_declared_coverage(plan: dict, expected_shards: int) -> None:
    contract = plan.get("run_contract", {})
    requested = contract.get("requested_taxonomies")
    explicit_eight = (isinstance(requested, list) and len(requested) == len(HEADS)
                      and set(requested) == set(HEADS))
    if (contract.get("model") != "SAM" or contract.get("amp") is not True or
            contract.get("classification_device") != "cpu" or
            requested != ["all"] and not explicit_eight):
        raise ValueError("Full-slide completion requires the frozen eight-head SAM/AMP run contract")
    count = len(plan["shards"])
    if expected_shards and count != expected_shards:
        raise ValueError("Full-slide plan differs from the expected core count")
    if plan["image"]["sha256"] == PM016_IMAGE_SHA256 and (
            count != 104 or plan["core_size"] != 1920 or plan["halo"] != 960 or
            (plan["image"]["width"], plan["image"]["height"]) != (23904, 14762)):
        raise ValueError("PM016 requires its complete 104-core 1920/960 plan")


def _plan(root: Path) -> dict:
    path = root / "cellvit_shard_plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing immutable CellViT shard plan: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _free_bytes(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def _require_capacity(root: Path, reserve_gib: float, case_budget_gib: float) -> dict:
    free = _free_bytes(root)
    minimum = int((reserve_gib + case_budget_gib) * 1024**3)
    if free < minimum:
        raise RuntimeError(f"CellViT shard disk admission failed: {free} free bytes; "
                           f"need at least {minimum} including retained-output reserve")
    return {"free_bytes_before": free, "minimum_bytes": minimum,
            "reserve_gib": reserve_gib, "case_budget_gib": case_budget_gib}


def _cleanup_receipt(root: Path, plan: dict, shard: dict) -> dict:
    folder = root / "shards" / shard["id"]
    marker = folder / "complete.json"
    receipt_file = folder / CLEANUP_FILE
    if not marker.is_file():
        raise RuntimeError(f"Shard {shard['id']} lacks completion or independently verified GPU cleanup")
    from cellvit_shards import ARCHIVE_NAME, ARCHIVE_RECEIPT, verify_complete_shard
    compact_file = folder / ARCHIVE_RECEIPT
    if compact_file.is_file():
        # Compaction preserves the terminal attestation inside the lossless
        # archive. Verify the complete archive and then this exact member;
        # never materialize a second loose file merely to resume a campaign.
        verify_complete_shard(plan, shard, folder)
        compact = json.loads(compact_file.read_text(encoding="utf-8"))
        expected = compact["members"].get(CLEANUP_FILE)
        if expected is None and receipt_file.is_file():
            # Older pilot archives predate supervisor receipts; their
            # separately published pilot attestation remains a loose file.
            data = receipt_file.read_bytes()
        elif (not isinstance(expected, dict) or
              not isinstance(expected.get("bytes"), int) or
              not 0 < expected["bytes"] <= 1024 * 1024 or
              not isinstance(expected.get("sha256"), str) or
              not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])):
            raise RuntimeError(f"Shard {shard['id']} archive lacks a checked GPU cleanup receipt")
        elif receipt_file.is_file():
            data = receipt_file.read_bytes()
        else:
            with tarfile.open(folder / ARCHIVE_NAME, "r:gz") as archive:
                matches = [member for member in archive.getmembers()
                           if member.name == CLEANUP_FILE]
                if len(matches) != 1 or not matches[0].isfile() or matches[0].size != expected["bytes"]:
                    raise RuntimeError(f"Shard {shard['id']} archived GPU cleanup member differs")
                stream = archive.extractfile(matches[0])
                if stream is None:
                    raise RuntimeError(f"Shard {shard['id']} archived GPU cleanup is unreadable")
                with stream:
                    data = stream.read(expected["bytes"] + 1)
        if expected is not None and (len(data) != expected["bytes"] or
                                     hashlib.sha256(data).hexdigest() != expected["sha256"]):
            raise RuntimeError(f"Shard {shard['id']} archived GPU cleanup hash or size differs")
        receipt = json.loads(data)
    else:
        if not receipt_file.is_file():
            raise RuntimeError(f"Shard {shard['id']} lacks completion or independently verified GPU cleanup")
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if (receipt.get("schema") not in ACCEPTED_CLEANUP_SCHEMAS or
            receipt.get("attestation_state") != "published" or
            receipt.get("draft") is not False):
        raise RuntimeError(f"Shard {shard['id']} lacks a published terminal cleanup attestation")
    expected = {"shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
                "checkpoint_sha256": digest_file(marker)}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Shard {shard['id']} GPU cleanup receipt has a different identity")
    if receipt.get("schema") == CPU_EMPTY_SCHEMA:
        from cellvit_shards import _read_record_json
        verify_complete_shard(plan, shard, folder)
        metadata = _read_record_json(folder, "cellvit_metadata.json")
        evidence = metadata.get("empty_core_evidence")
        if (receipt.get("cpu_only") is not True or receipt.get("gpu_started") is not False or
                not isinstance(evidence, dict) or
                evidence.get("policy") != "grandqc-zero-clean-core-conservative-mask-cover-v1" or
                evidence.get("nonzero_mask_pixels") != 0 or
                evidence.get("core_xyxy") != shard["core"] or
                evidence.get("plan_sha256") != plan["identity_sha256"] or
                evidence.get("image_sha256") != plan["image"]["sha256"] or
                evidence.get("full_shift_sha256") != plan["shift_sha256"] or
                evidence.get("clean_tissue_mask_sha256") != plan["run_contract"].get("clean_tissue_mask_sha256") or
                receipt.get("empty_core_evidence") != evidence):
            raise RuntimeError(f"Shard {shard['id']} CPU-empty provenance differs from its checked core")
        return receipt
    for key in ("cleanup_verified", "owned_processes_absent", "scope_inactive",
                "bounded_checks_complete"):
        if receipt.get(key) is not True:
            raise RuntimeError(f"Shard {shard['id']} GPU cleanup is uncertain: {key}")
    if not (receipt.get("assigned_gpu_contexts_empty") is True or (
            receipt.get("assigned_gpu_contexts_baseline_only") is True and
            isinstance(receipt.get("gpu_context_baseline_sha256"), str) and
            len(receipt["gpu_context_baseline_sha256"]) == 64)):
        raise RuntimeError(f"Shard {shard['id']} assigned GPU contexts are uncertain")
    _published_pilot_provenance(root, plan, shard, receipt, digest_file(marker))
    return receipt


def _require_prior_cleanup(root: Path, plan: dict) -> None:
    if (root / FAILURE_FILE).exists():
        raise RuntimeError("A prior CellViT GPU case failed; this bundle is latched")
    active = root / ACTIVE_FILE
    if active.exists():
        observed = json.loads(active.read_text(encoding="utf-8"))
        if observed.get("plan_sha256") != plan["identity_sha256"]:
            raise RuntimeError("A previous GPU action has an incompatible active marker")
        _cleanup_receipt(root, plan, get_shard(plan, observed.get("shard_id", "")))
        active.unlink()
    for shard in plan["shards"]:
        folder = root / "shards" / shard["id"]
        if not folder.exists():
            continue
        _cleanup_receipt(root, plan, shard)


def _latch(root: Path, action: str, shard_id: str, reason: str) -> None:
    latch = root / FAILURE_FILE
    if not latch.exists():
        atomic_json(latch, {"schema": "cellvit-gpu-bundle-failure-v1",
                            "action": action, "shard_id": shard_id, "reason": reason,
                            "no_retry_or_successor": True})


def _campaign_source(root: Path, args) -> dict:
    """Freeze the actual local inference and export code across resumed cores."""
    import ast
    code = Path(__file__).resolve().parent
    seeds = [code / name for name in (
        "run_cellvitpp.py", "cellvit_models.py", "cellvit_shards.py",
        "cellvit_fullslide_action.py", "cellvit_fullslide_supervisor.py",
        "cellvit_scope_payload.py",
        "cellvit_runtime_binding.py",
        "feature_matrix.py", "classify_cellvit_features.py",
        "index_cellvit_partitions.py", "export_cellvit_partition_experiment.py",
        "combine_cellvit_bora_project.py")]
    files = set(seeds)
    pending = list(seeds)
    while pending:
        candidate = pending.pop()
        tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                dependency = code / f"{name}.py"
                if dependency.is_file() and dependency not in files:
                    files.add(dependency)
                    pending.append(dependency)
    files.add(code.parent / "resources" / "cellvit_model_catalog.json")
    observed = {str(path): digest_file(path) for path in sorted(files)}
    runtime = {}
    if args.runtime_sif:
        runtime = {"sif": str(args.runtime_sif.resolve(strict=True)),
                   "sif_sha256": args.runtime_sif_sha256,
                   "python": args.runtime_python,
                   "executable": args.runtime_executable,
                   "pydeps": str(args.runtime_pydeps.resolve(strict=True)) if args.runtime_pydeps else None,
                   "pydeps_sha256": args.runtime_pydeps_sha256 or None}
    runtime["case_settings"] = {"cpus": args.cpus, "gpu": args.gpu,
                                "batch_size": args.batch_size,
                                "memory_mb": args.memory_mb,
                                "app_memory_mb": args.app_memory_mb,
                                "case_tasks_max": args.case_tasks_max}
    source_path = root / CAMPAIGN_SOURCE_FILE
    if source_path.exists():
        existing = json.loads(source_path.read_text(encoding="utf-8"))
        if (existing.get("schema") != "cellvit-fullslide-source-v1" or
                existing.get("files") != observed or existing.get("runtime") != runtime):
            raise RuntimeError("CellViT campaign source changed; preserve this bundle and start a versioned recovery")
        return existing
    value = {"schema": "cellvit-fullslide-source-v1", "files": observed,
             "runtime": runtime}
    atomic_json(source_path, value)
    return value


def _preflight(args) -> dict:
    """CPU-only native wheel/source check in the exact selected runtime."""
    root = args.bundle_dir.resolve()
    _validate_case_limits(args)
    pydeps = verified_pydeps(args.runtime_pydeps, args.runtime_pydeps_sha256)
    if pydeps and not args.runtime_sif:
        raise ValueError("Runtime pydeps require the pinned Singularity image")
    if args.runtime_sif:
        if (not args.runtime_sif.is_absolute() or
                not re.fullmatch(r"[0-9a-f]{64}", args.runtime_sif_sha256 or "") or
                not args.runtime_python.startswith("/") or
                not args.runtime_executable.startswith("/") or
                digest_file(args.runtime_sif) != args.runtime_sif_sha256):
            raise ValueError("Pinned Singularity image, SHA-256 and in-image CellViT paths are required")
    if not (root / "cellvit_shard_plan.json").exists():
        _run_action(Namespace(**{**vars(args), "action": "plan"}))
    plan = validate_plan(_plan(root), args.image, args.shift)
    _require_declared_coverage(plan, args.expected_shards)
    _require_prior_cleanup(root, plan)
    source = _campaign_source(root, args)
    script = Path(__file__).with_name("run_cellvitpp.py")
    if args.runtime_sif:
        singularity = shutil.which("singularity")
        if not singularity:
            raise RuntimeError("Singularity is required for the pinned CellViT runtime")
        source_root = script.parents[1].resolve()
        command = [singularity, "exec", "--cleanenv",
                   "--env", "PYTHONDONTWRITEBYTECODE=1"]
        if pydeps:
            command += ["--env", f"PYTHONPATH={pydeps}",
                        "--bind", f"{pydeps}:{pydeps}"]
        command += ["--bind", f"{source_root}:{source_root}", str(args.runtime_sif),
                   args.runtime_python, str(script), "--preflight-only",
                   "--executable", args.runtime_executable]
    else:
        command = [args.python, str(script), "--preflight-only",
                   "--executable", args.executable]
    if args.amp:
        command.append("--amp")
    result = subprocess.run(command, text=True, capture_output=True,
                            timeout=180, check=True)
    details = json.loads(result.stdout)
    if details.get("status") != "ready" or details.get("cuda_probed") is not False:
        raise RuntimeError("CellViT runtime preflight did not return CPU-only ready status")
    receipt = {"schema": "cellvit-fullslide-cpu-preflight-v1",
               "plan_sha256": plan["identity_sha256"],
               "image_sha256": plan["image"]["sha256"],
               "source_manifest_sha256": digest_file(root / CAMPAIGN_SOURCE_FILE),
               "runtime_sif_sha256": args.runtime_sif_sha256 if args.runtime_sif else None,
               "runtime_pydeps_sha256": args.runtime_pydeps_sha256 or None,
               "native_runtime": details, "gpu_started": False}
    receipt_path = root / "CELLVIT_PREFLIGHT.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("Existing CPU preflight receipt differs from the current runtime")
    else:
        atomic_json(receipt_path, receipt)
    return receipt


def _base_command(args, *, action: str, shard_id: str = "") -> list[str]:
    runtime = bool(args.runtime_sif)
    pydeps = verified_pydeps(args.runtime_pydeps, args.runtime_pydeps_sha256)
    if pydeps and not runtime:
        raise ValueError("Runtime pydeps require the pinned Singularity image")
    if runtime and (not re.fullmatch(r"[0-9a-f]{64}", args.runtime_sif_sha256 or "") or
                    digest_file(args.runtime_sif) != args.runtime_sif_sha256):
        raise RuntimeError("Pinned CellViT Singularity image changed before CPU storage work")
    command = [args.runtime_python if runtime else args.python,
               str(Path(__file__).with_name("run_cellvitpp.py")),
               "--image", str(args.image), "--shift", str(args.shift),
               "--clean-tissue-mask", str(args.clean_tissue_mask),
               "--outdir", str(args.bundle_dir), "--model", args.model,
               "--executable", args.runtime_executable if runtime else args.executable,
               "--classifier-cache", str(args.classifier_cache),
               "--classification-device", "cpu", "--taxonomies", "all",
               "--default-mpp", "0", "--gpu", str(args.gpu),
               "--batch-size", str(args.batch_size), "--cpus", str(args.cpus),
               "--memory-mb", str(args.app_memory_mb), "--ray-workers", "1",
               "--ray-worker-cpus", str(args.cpus),
               "--shard-core-size", str(args.core_size),
               "--shard-halo", str(args.halo)]
    if args.amp:
        command.append("--amp")
    command += ({"plan": ["--shard-plan"], "shard": ["--shard-id", shard_id],
                 "probe_empty": ["--shard-id", shard_id, "--probe-empty-clean-core"],
                 "cpu_empty": ["--shard-id", shard_id, "--cpu-empty-core"],
                 "pack": ["--pack-features-shard-id", shard_id],
                 "compact": ["--compact-shard-id", shard_id],
                 "finalize": ["--finalize-shards"]})[action]
    if runtime and action != "shard":
        singularity = shutil.which("singularity")
        if not singularity:
            raise RuntimeError("Singularity is required for CellViT CPU storage work")
        cellvit_cache = Path(os.environ.get("CELLVIT_CACHE", "")) if action == "cpu_empty" else None
        if cellvit_cache is not None and (not cellvit_cache.is_absolute() or
                                          not cellvit_cache.is_dir()):
            raise RuntimeError("CPU empty-core export requires an absolute existing CELLVIT_CACHE")
        source_root = Path(__file__).resolve().parents[1]
        bind_dirs = sorted({str(path.resolve()) for path in (
            source_root, args.image.parent, args.shift.parent,
            args.clean_tissue_mask.parent, args.bundle_dir,
            args.classifier_cache, *((pydeps,) if pydeps else ()),
            *((cellvit_cache,) if cellvit_cache else ()))})
        return [singularity, "exec", "--cleanenv",
                "--env", "PYTHONDONTWRITEBYTECODE=1",
                *(["--env", f"PYTHONPATH={pydeps}"] if pydeps else []),
                *(["--env", f"CELLVIT_CACHE={cellvit_cache}"] if cellvit_cache else []),
                *[item for directory in bind_dirs
                  for item in ("--bind", f"{directory}:{directory}")],
                str(args.runtime_sif), *command]
    return command


def _validate_case_limits(args) -> None:
    if (args.cpus != 2 or not 1024 <= args.memory_mb <= 12288 or
            not 1024 <= args.app_memory_mb <= args.memory_mb - 2048 or
            not 256 <= args.case_tasks_max <= 1024):
        raise ValueError("CellViT case requires 2 CPUs, bounded tasks, and at least 2 GiB memory headroom")


def _adopt_shard(args, root: Path, plan: dict, shard: dict) -> dict:
    """Copy one previously completed case into this slide's canonical bundle.

    Old pilot outputs remain immutable. The source checkpoint, every scientific
    record and an external terminal-cleanup attestation are checked before and
    after copying; a hardlink cannot make the pilot mutable through this run.
    """
    from cellvit_shards import ARCHIVE_RECEIPT, verify_complete_shard
    source = args.adopt_from.resolve(strict=True)
    target = root / "shards" / shard["id"]
    allowed_case_output = (getattr(args, "allow_case_owned_source", False) is True and
                           source == root / "cases" / shard["id"] / "output")
    if source == target or (root in source.parents and not allowed_case_output) or target.exists():
        raise FileExistsError("Adoption requires a separate completed source and new canonical shard")
    if not args.expected_checkpoint_sha256 or not args.adoption_cleanup_receipt_sha256:
        raise ValueError("Pin both prior checkpoint and cleanup receipt SHA-256")
    marker = source / "complete.json"
    if digest_file(marker) != args.expected_checkpoint_sha256:
        raise ValueError("Prior pilot checkpoint SHA-256 differs from adoption pin")
    verify_complete_shard(plan, shard, source)
    cleanup = args.adoption_cleanup_receipt.resolve(strict=True)
    if digest_file(cleanup) != args.adoption_cleanup_receipt_sha256:
        raise ValueError("Prior pilot cleanup receipt SHA-256 differs from adoption pin")
    if (source / ARCHIVE_RECEIPT).is_file():
        archived = json.loads((source / ARCHIVE_RECEIPT).read_text(encoding="utf-8"))
        member = archived.get("members", {}).get(CLEANUP_FILE, {})
        if member and (member.get("sha256") != args.adoption_cleanup_receipt_sha256 or
                       member.get("bytes") != cleanup.stat().st_size):
            raise ValueError("Adoption cleanup evidence differs from the archived source receipt")
    receipt = json.loads(cleanup.read_text(encoding="utf-8"))
    if (receipt.get("schema") not in ACCEPTED_CLEANUP_SCHEMAS or
            receipt.get("attestation_state") != "published" or
            receipt.get("draft") is not False):
        raise ValueError("Prior pilot cleanup attestation is draft or unpublished")
    expected = {"shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
                "checkpoint_sha256": args.expected_checkpoint_sha256}
    if receipt.get("schema") == CPU_EMPTY_SCHEMA:
        if source.parent.name != "shards" or source.name != shard["id"] or (
                receipt != _cleanup_receipt(source.parent.parent, plan, shard)):
            raise ValueError("Prior CPU-empty core lacks its checked mask provenance")
    elif any(receipt.get(key) != value for key, value in expected.items()) or any(
            receipt.get(key) is not True for key in (
                "cleanup_verified", "owned_processes_absent", "scope_inactive",
                "bounded_checks_complete")) or not (
                receipt.get("assigned_gpu_contexts_empty") is True or (
                    receipt.get("assigned_gpu_contexts_baseline_only") is True and
                    isinstance(receipt.get("gpu_context_baseline_sha256"), str) and
                    len(receipt["gpu_context_baseline_sha256"]) == 64)):
        raise ValueError("Prior pilot lacks exact independently verified terminal cleanup")
    _published_pilot_provenance(root, plan, shard, receipt,
                                args.expected_checkpoint_sha256)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".adopt-{shard['id']}-", dir=target.parent) as temporary:
        staged = Path(temporary) / "scientific"
        staged.mkdir()
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Adoption refuses symlinked pilot evidence")
            if path.is_dir():
                continue
            relative = path.relative_to(source)
            if relative.name in ("cellvit_input.tif", "input.tif"):
                raise ValueError("Release redundant input crops before adopting a pilot")
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if digest_file(destination) != digest_file(path):
                raise ValueError(f"Pilot record changed while copying: {relative}")
        shutil.copyfile(cleanup, staged / CLEANUP_FILE)
        verify_complete_shard(plan, shard, staged)
        if digest_file(staged / "complete.json") != args.expected_checkpoint_sha256:
            raise ValueError("Copied pilot checkpoint differs")
        os.replace(staged, target)
    _cleanup_receipt(root, plan, shard)
    return {"action": "adopt", "shard_id": shard["id"],
            "plan_sha256": plan["identity_sha256"],
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "cleanup_receipt_sha256": args.adoption_cleanup_receipt_sha256,
            "cell_count": verify_complete_shard(plan, shard, target)["cell_count"],
            "gpu_started": False, "source_preserved": True}



def _verify_final_project(project_path: Path, plan: dict, manifest: dict,
                          manifest_path: Path, index_path: Path) -> dict:
    """A resumed JSON must describe this exact checked full-slide population."""
    from export_experiment import validate_experiment
    project = validate_experiment(project_path)
    tissues = project.get("tissues", [])
    if len(tissues) != 1:
        raise ValueError("Existing CellViT project must contain exactly this full-slide tissue")
    tissue = tissues[0]
    image = tissue.get("image", {})
    expected_image = plan["image"]
    if any(image.get(key) != expected_image.get(key) for key in
           ("sha256", "width", "height", "mpp_x", "mpp_y", "frame_id",
            "coordinate_convention", "coordinate_space", "origin_x", "origin_y")):
        raise ValueError("Existing CellViT project image differs from the complete plan")
    cells = tissue.get("cell_classification", {})
    if (not isinstance(cells, dict) or
            cells.get("cell_count") != manifest["cell_count"] or
            not isinstance(cells.get("index"), dict) or
            not isinstance(cells.get("feature_manifest"), dict)):
        raise ValueError("Existing CellViT project lacks the checked cell population")
    def exact_reference(reference: dict, path: Path) -> bool:
        raw = reference.get("path")
        if (not isinstance(raw, str) or Path(raw).is_absolute() or "://" in raw or
                reference.get("sha256") != digest_file(path)):
            return False
        return (project_path.parent / raw).resolve(strict=True) == path.resolve(strict=True)
    if (not exact_reference(cells, manifest_path) or
            not exact_reference(cells["feature_manifest"], manifest_path) or
            cells["index"].get("format") != "sqlite" or
            not exact_reference(cells["index"], index_path)):
        raise ValueError("Existing CellViT project points to another partition manifest or index")
    return project


def _run_action(args) -> dict:
    if not args.bundle_dir.is_absolute():
        raise ValueError("CellViT full-slide bundle path must be absolute and persistent across Nextflow tasks")
    root = args.bundle_dir.resolve()
    if args.model != "SAM":
        raise ValueError("The eight-head full-slide route requires the SAM backbone")
    if args.reserve_gib < 0 or args.case_budget_gib <= 0 or args.case_timeout_seconds <= 0:
        raise ValueError("Disk reserve, one-case budget and timeout must be positive")
    if args.expected_shards < 0:
        raise ValueError("Expected shard count cannot be negative")
    if args.action == "plan":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Plan requires a new empty bundle: {root}")
        root.mkdir(parents=True, exist_ok=True)
        models = preflight_models("SAM", list(HEADS), args.classifier_cache)
        missing = [model["model_id"] for model in models if model["status"] != "available"]
        if missing:
            raise RuntimeError(f"All eight pretrained heads are required: unavailable {missing}")
        capacity = _require_capacity(root, args.reserve_gib, args.case_budget_gib)
        subprocess.run(_base_command(args, action="plan"), check=True, stdout=sys.stderr)
        plan = validate_plan(_plan(root), args.image, args.shift)
        _require_declared_coverage(plan, args.expected_shards)
        receipt = {"action": "plan", "plan_sha256": plan["identity_sha256"],
                   "shard_count": len(plan["shards"]), "requested_heads": list(HEADS),
                   "image_sha256": plan["image"]["sha256"], "disk_admission": capacity,
                   "gpu_started": False}
        atomic_json(root / "CELLVIT_PLAN_COMPLETE.json", receipt)
        return receipt

    plan = validate_plan(_plan(root), args.image, args.shift)
    _require_declared_coverage(plan, args.expected_shards)
    if args.action == "adopt":
        if not args.shard_id or not args.adopt_from or not args.adoption_cleanup_receipt:
            raise ValueError("Adopt one named shard with source and terminal-cleanup receipt")
        _require_prior_cleanup(root, plan)
        _require_capacity(root, args.reserve_gib, args.case_budget_gib)
        return _adopt_shard(args, root, plan, get_shard(plan, args.shard_id))
    if args.action == "shard":
        if not args.shard_id:
            raise ValueError("One explicit --shard-id is required; no automatic successor exists")
        shard = get_shard(plan, args.shard_id)
        _require_prior_cleanup(root, plan)
        if (root / "shards" / shard["id"]).exists():
            raise FileExistsError(f"Preserve existing shard evidence: {shard['id']}")
        capacity = _require_capacity(root, args.reserve_gib, args.case_budget_gib)
        if not args.supervisor or not Path(args.supervisor).is_file():
            raise RuntimeError("A reviewed GPU lifecycle supervisor is required before one shard")
        raise RuntimeError("No compatible direct-supervisor implementation is approved; "
                           "the frozen Chiamaka launcher uses --binding/--binding-sha256. "
                           "Run that approved one-case adapter separately, then import its "
                           "terminal result with cellvit_approved_case_import.py")

    if args.action in ("pack", "compact"):
        if not args.shard_id:
            raise ValueError("One explicit --shard-id is required for a CPU storage action")
        shard = get_shard(plan, args.shard_id)
        _require_prior_cleanup(root, plan)
        _cleanup_receipt(root, plan, shard)
        capacity = _require_capacity(root, args.reserve_gib, args.case_budget_gib)
        subprocess.run(_base_command(args, action=args.action, shard_id=shard["id"]),
                       check=True, stdout=sys.stderr)
        from cellvit_shards import verify_complete_shard
        verify_complete_shard(plan, shard, root / "shards" / shard["id"])
        return {"action": args.action, "shard_id": shard["id"],
                "plan_sha256": plan["identity_sha256"], "disk_admission": capacity,
                "gpu_started": False, "feature_vectors_retained": True}

    if args.action != "finalize":
        raise ValueError("action must be plan, shard or finalize")
    _require_prior_cleanup(root, plan)
    for shard in plan["shards"]:
        _cleanup_receipt(root, plan, shard)
    capacity = _require_capacity(root, args.reserve_gib, args.case_budget_gib)
    from cellvit_shards import finalize_shards
    from index_cellvit_partitions import build_partition_index, validate_partition_index
    from export_cellvit_partition_experiment import export_partition_experiment
    from export_experiment import validate_experiment
    manifest = finalize_shards(plan, root, publish=False)
    if manifest["partition_count"] != len(plan["shards"]):
        raise ValueError("Refusing a partial CellViT full-slide manifest")
    manifest_path = root / "cellvit_partition_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing partition manifest differs from verified complete shards")
    else:
        atomic_json(manifest_path, manifest)
    index_path = root / "cellvit_cells.sqlite"
    if not index_path.exists():
        build_partition_index(manifest_path, args.image, index_path)
    validate_partition_index(index_path, manifest_path,
                             index_path.with_suffix(".features.json"), args.image)
    project_path = root / "experiment.cellphenotyper.json"
    if not project_path.exists():
        export_partition_experiment(manifest_path, args.image, index_path,
                                    project_path, name=args.name, tissue_id=args.tissue_id)
    _verify_final_project(project_path, plan, manifest, manifest_path, index_path)
    receipt = {"action": "finalize", "plan_sha256": plan["identity_sha256"],
               "schema": COMPLETE_SCHEMA,
               "planned_shards": len(plan["shards"]),
               "complete_shards": manifest["partition_count"],
               "shard_count": manifest["partition_count"], "cell_count": manifest["cell_count"],
               "every_core_verified": manifest["partition_count"] == len(plan["shards"]),
               "source_image_sha256": plan["image"]["sha256"],
               "partition_manifest_sha256": digest_file(manifest_path),
               "index_sha256": digest_file(index_path),
               "project_sha256": digest_file(project_path),
               "requested_heads": list(HEADS), "feature_vectors_retained": True,
               "viewport_index": str(index_path), "project": str(project_path),
               "gpu_started": False, "disk_admission": capacity}
    complete_path = root / "CELLVIT_PROJECT_COMPLETE.json"
    if complete_path.exists():
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        for key in ("schema", "plan_sha256", "planned_shards", "complete_shards",
                    "shard_count", "cell_count", "source_image_sha256",
                    "partition_manifest_sha256", "index_sha256", "project_sha256",
                    "requested_heads", "feature_vectors_retained"):
            if existing.get(key) != receipt[key]:
                raise ValueError("Existing full-slide completion receipt differs from verified outputs")
        return existing
    atomic_json(complete_path, receipt)
    return receipt


def _probe_empty_core(args, shard: dict) -> dict | None:
    """Read the declared GrandQC mask in the pinned CPU runtime before GPU admission."""
    command = _base_command(args, action="probe_empty", shard_id=shard["id"])
    affinity = shutil.which("taskset")
    if not affinity:
        raise RuntimeError("CPU-only CellViT mask probe requires taskset")
    result = subprocess.run([affinity, "-c", "0,1", *command], text=True,
                            capture_output=True, timeout=180, check=True)
    observed = json.loads(result.stdout)
    if observed.get("shard") != shard["id"] or not isinstance(observed.get("empty_clean_core"), bool):
        raise RuntimeError("CellViT CPU mask probe returned an incompatible core identity")
    evidence = observed.get("evidence")
    if observed["empty_clean_core"] != (isinstance(evidence, dict)):
        raise RuntimeError("CellViT CPU mask probe has inconsistent empty-core evidence")
    return evidence


def _run_cpu_empty_core(args, root: Path, plan: dict, shard: dict, evidence: dict) -> dict:
    """Publish one zero-cell core without assigning or starting a GPU process."""
    from cellvit_shards import verify_complete_shard
    case_dir = root / "cases" / shard["id"]
    case_dir.mkdir(parents=True, exist_ok=False)
    log_path = case_dir / "CPU_EMPTY_STDOUT.log"
    command = _base_command(args, action="cpu_empty", shard_id=shard["id"])
    affinity = shutil.which("taskset")
    if not affinity:
        raise RuntimeError("CPU-only CellViT empty-core export requires taskset")
    with log_path.open("xb") as log:
        process = subprocess.Popen([affinity, "-c", "0,1", *command],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=1800)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
            raise RuntimeError("CPU empty-core export timed out; preserve this case and verify cleanup") from error
        log.flush()
        os.fsync(log.fileno())
    if returncode != 0:
        raise RuntimeError(f"CPU empty-core export exited {returncode}; preserve its case log")
    folder = root / "shards" / shard["id"]
    marker = verify_complete_shard(plan, shard, folder)
    metadata = json.loads((folder / "cellvit_metadata.json").read_text(encoding="utf-8"))
    if marker.get("cell_count") != 0 or metadata.get("empty_core_evidence") != evidence:
        raise RuntimeError("CPU empty-core checkpoint differs from the independently probed mask")
    receipt = {"schema": CPU_EMPTY_SCHEMA, "attestation_state": "published",
               "draft": False, "shard_id": shard["id"],
               "plan_sha256": plan["identity_sha256"],
               "checkpoint_sha256": digest_file(folder / "complete.json"),
               "cpu_only": True, "gpu_started": False,
               "empty_core_evidence": evidence,
               "case_log_sha256": digest_file(log_path)}
    atomic_json(folder / CLEANUP_FILE, receipt)
    return _cleanup_receipt(root, plan, shard)


def _campaign(args) -> dict:
    """Run the complete WSI through sequential, verified native CellViT cases.

    Existing checked cores are reused.  One failure latches this bundle and
    prevents every successor; a failed core is never retried in place.
    """
    from cellvit_fullslide_supervisor import supervise_case
    from cellvit_shards import (ARCHIVE_RECEIPT, FEATURE_MATRIX_RECEIPT,
                                verify_complete_shard)

    root = args.bundle_dir.resolve()
    _validate_case_limits(args)
    if args.expected_shards <= 0:
        raise ValueError("Run requires the declared full-slide core count")
    if not args.gpu_uuid:
        raise ValueError("Run requires the assigned --gpu-uuid")
    if args.max_new_shards < 0:
        raise ValueError("Maximum new CellViT cores cannot be negative")
    pydeps = verified_pydeps(args.runtime_pydeps, args.runtime_pydeps_sha256)
    if pydeps and not args.runtime_sif:
        raise ValueError("Runtime pydeps require the pinned Singularity image")
    if args.runtime_sif:
        if (not args.runtime_sif.is_absolute() or
                not re.fullmatch(r"[0-9a-f]{64}", args.runtime_sif_sha256 or "") or
                not args.runtime_python.startswith("/") or
                not args.runtime_executable.startswith("/") or
                digest_file(args.runtime_sif) != args.runtime_sif_sha256):
            raise ValueError("Pinned Singularity image, SHA-256 and in-image CellViT paths are required")
    if not (root / "cellvit_shard_plan.json").exists():
        _run_action(Namespace(**{**vars(args), "action": "plan"}))
    plan = validate_plan(_plan(root), args.image, args.shift)
    _require_declared_coverage(plan, args.expected_shards)
    _require_prior_cleanup(root, plan)
    _campaign_source(root, args)
    _preflight(args)
    if (root / "CELLVIT_PROJECT_COMPLETE.json").exists():
        finalized = _run_action(Namespace(**{**vars(args), "action": "finalize"}))
    else:
        newly_completed = 0
        for position, shard in enumerate(plan["shards"], 1):
            _campaign_source(root, args)
            shard_dir = root / "shards" / shard["id"]
            marker = shard_dir / "complete.json"
            if marker.is_file():
                _cleanup_receipt(root, plan, shard)
                verify_complete_shard(plan, shard, shard_dir)
            else:
                if shard_dir.exists() and any(shard_dir.iterdir()):
                    raise RuntimeError(f"Incomplete CellViT core is preserved for review: {shard['id']}")
                _require_capacity(root, args.reserve_gib, args.case_budget_gib)
                empty_evidence = _probe_empty_core(args, shard)
                if empty_evidence is not None:
                    try:
                        _run_cpu_empty_core(args, root, plan, shard, empty_evidence)
                    except BaseException as error:
                        if (root / "cases" / shard["id"]).exists():
                            _latch(root, "cpu_empty", shard["id"],
                                   f"{type(error).__name__}: {error}")
                        raise
                else:
                    active = root / ACTIVE_FILE
                    atomic_json(active, {"plan_sha256": plan["identity_sha256"],
                                         "shard_id": shard["id"],
                                         "case_position": position, "case_count": len(plan["shards"])})
                    try:
                        native_args = args
                        if args.runtime_sif:
                            native_args = Namespace(**{**vars(args),
                                                       "python": args.runtime_python,
                                                       "executable": args.runtime_executable})
                        supervise_case(args, root, plan, shard,
                                       _base_command(native_args, action="shard", shard_id=shard["id"]))
                        _cleanup_receipt(root, plan, shard)
                        verify_complete_shard(plan, shard, shard_dir)
                    except BaseException as error:
                        if not (root / "cases" / shard["id"]).exists():
                            # Admission failed before a GPU case was attempted.
                            active.unlink(missing_ok=True)
                        else:
                            _latch(root, "run", shard["id"],
                                   f"{type(error).__name__}: {error}")
                        raise
                    active.unlink()
                newly_completed += 1
            if not (shard_dir / ARCHIVE_RECEIPT).exists():
                if not (shard_dir / FEATURE_MATRIX_RECEIPT).exists():
                    _run_action(Namespace(**{**vars(args), "action": "pack",
                                             "shard_id": shard["id"]}))
                _run_action(Namespace(**{**vars(args), "action": "compact",
                                         "shard_id": shard["id"]}))
            verify_complete_shard(plan, shard, shard_dir)
            atomic_json(root / "CELLVIT_CAMPAIGN_PROGRESS.json", {
                "schema": "cellphenotyper-cellvit-campaign-progress-v1",
                "plan_sha256": plan["identity_sha256"], "last_verified_core": shard["id"],
                "verified_position": position, "planned_cores": len(plan["shards"])})
            if args.max_new_shards and newly_completed >= args.max_new_shards:
                return {"action": "run", "status": "bounded_partial",
                        "plan_sha256": plan["identity_sha256"],
                        "newly_completed_shards": newly_completed,
                        "last_verified_core": shard["id"],
                        "planned_shards": len(plan["shards"]),
                        "project_published": False}
        finalized = _run_action(Namespace(**{**vars(args), "action": "finalize"}))

    if args.bora_geojson or args.bora_receipt or args.combined_output:
        if not all((args.bora_geojson, args.bora_receipt,
                    args.expected_bora_sha256, args.combined_output)):
            raise ValueError("Bora merge requires GeoJSON, receipt, exact hash and output together")
        from combine_cellvit_bora_project import combine
        combined = combine(root / "experiment.cellphenotyper.json",
                           root / "CELLVIT_PROJECT_COMPLETE.json",
                           args.bora_geojson, args.bora_receipt,
                           args.combined_output,
                           expected_bora_sha256=args.expected_bora_sha256,
                           expected_shards=args.expected_shards)
        finalized = {**finalized, "combined_project": combined["project"],
                     "bora_components": combined["bora_components"]}
    return finalized


def run_action(args) -> dict:
    """Serialize actions across Nextflow invocations without deleting a lock file."""
    if not args.bundle_dir.is_absolute():
        raise ValueError("CellViT full-slide bundle path must be absolute")
    root = args.bundle_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.parent / f".{root.name}.cellvit-action.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another CellViT action owns this persistent bundle") from exc
        if args.action == "preflight":
            return _preflight(args)
        return _campaign(args) if args.action == "run" else _run_action(args)
    finally:
        os.close(descriptor)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", required=True,
                   choices=("plan", "adopt", "shard", "pack", "compact", "finalize",
                            "preflight", "run"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True)
    p.add_argument("--clean-tissue-mask", type=Path, required=True)
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--classifier-cache", type=Path, required=True)
    p.add_argument("--model", default="SAM")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--executable", default="cellvit-inference")
    p.add_argument("--supervisor", type=Path)
    p.add_argument("--adopt-from", type=Path)
    p.add_argument("--expected-checkpoint-sha256", default="")
    p.add_argument("--adoption-cleanup-receipt", type=Path)
    p.add_argument("--adoption-cleanup-receipt-sha256", default="")
    p.add_argument("--shard-id", default="")
    p.add_argument("--name", default="CellViT++ full-slide cell review")
    p.add_argument("--tissue-id", default="tissue")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--gpu-uuid", default="",
                   help="The single assigned Chiamaka GPU UUID for the bounded run action")
    p.add_argument("--runtime-sif", type=Path,
                   help="Pinned CellViT GPU Singularity image; host supervisor remains outside")
    p.add_argument("--runtime-sif-sha256", default="")
    p.add_argument("--runtime-python", default="")
    p.add_argument("--runtime-executable", default="")
    p.add_argument("--runtime-pydeps", type=Path,
                   help="Pinned host directory of supplemental in-container Python modules")
    p.add_argument("--runtime-pydeps-sha256", default="",
                   help="Recursive filename/content SHA-256 of --runtime-pydeps")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--memory-mb", type=int, default=12288)
    p.add_argument("--app-memory-mb", type=int, default=8192,
                   help="Memory advertised to CellViT/Ray; leave headroom below case MemoryMax")
    p.add_argument("--case-tasks-max", type=int, default=512,
                   help="Finite systemd pids/threads cap for one CellViT Ray case")
    p.add_argument("--core-size", type=int, default=1920)
    p.add_argument("--halo", type=int, default=960)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--reserve-gib", type=float, default=40.0)
    p.add_argument("--case-budget-gib", type=float, default=4.0)
    p.add_argument("--case-timeout-seconds", type=int, default=7200)
    p.add_argument("--expected-shards", type=int, default=0)
    p.add_argument("--max-new-shards", type=int, default=0,
                   help="Stop after this many new verified cores (CPU-empty or GPU); zero means complete the slide")
    p.add_argument("--bora-geojson", type=Path)
    p.add_argument("--bora-receipt", type=Path)
    p.add_argument("--expected-bora-sha256", default="")
    p.add_argument("--combined-output", type=Path)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    print(json.dumps(run_action(options), sort_keys=True))
