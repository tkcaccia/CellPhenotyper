"""Run one official CellViT++ WSI partition in a bounded Chiamaka user unit.

The scientific command remains ``run_cellvitpp.py --shard-id``.  This module
only owns its lifetime.  A successful return is insufficient: two bounded
checks must find the unit inactive, its cgroup empty, and the assigned GPU's
compute contexts back at their pre-case baseline before a cleanup receipt is
published.  An unavailable observation fails closed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

from cellvit_models import digest_file
from cellvit_runtime_binding import verified_pydeps
from cellvit_shards import atomic_json, verify_complete_shard

MIN_CASE_TASKS = 256
MAX_CASE_TASKS = 1024


def _task_headroom(cgroup: str, *, tasks_max: int,
                   cgroup_root: Path = Path("/sys/fs/cgroup")) -> list[dict]:
    """Verify the live service and every ancestor can admit Ray's task budget."""
    if not cgroup.startswith("/user.slice/") or ".." in cgroup:
        raise RuntimeError("CellViT cgroup path is invalid")
    current = cgroup_root / cgroup.lstrip("/")
    unit_group = current
    minimum_free = max(256, tasks_max // 2)
    observations = []
    while True:
        limit_path = current / "pids.max"
        count_path = current / "pids.current"
        raw_limit = limit_path.read_text(encoding="utf-8").strip() if limit_path.is_file() else None
        if current == unit_group and raw_limit in (None, "max"):
            raise RuntimeError("CellViT case cgroup lacks its finite pids.max")
        used = None
        limit = None
        if raw_limit not in (None, "max"):
            if not count_path.is_file():
                raise RuntimeError(f"Finite CellViT pids controller cannot be read: {current}")
            try:
                used = int(count_path.read_text(encoding="utf-8").strip())
                limit = int(raw_limit)
            except ValueError as error:
                raise RuntimeError("CellViT cgroup pids telemetry is invalid") from error
            if used < 0 or limit < used or limit - used < minimum_free:
                raise RuntimeError(f"CellViT Ray task headroom is below {minimum_free}: {current}")
            if current == unit_group and limit != tasks_max:
                raise RuntimeError("CellViT case cgroup pids.max differs from requested TasksMax")
        observations.append({"cgroup": str(current.relative_to(cgroup_root)),
                             "pids_max": limit, "pids_current": used,
                             "pids_free": None if limit is None else limit - used})
        if current == cgroup_root:
            break
        current = current.parent
    return observations


def _command(argv: list[str], *, timeout: float = 10, check: bool = True,
             stdout_path: Path | None = None,
             stderr_path: Path | None = None) -> subprocess.CompletedProcess:
    if stdout_path is None and stderr_path is None:
        return subprocess.run(argv, text=True, capture_output=True,
                              timeout=timeout, check=check)
    if stdout_path is None or stderr_path is None:
        raise ValueError("Both bounded case log destinations are required")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        return subprocess.run(argv, text=True, stdout=stdout, stderr=stderr,
                              timeout=timeout, check=check)


def _contexts(gpu_uuid: str) -> list[int]:
    result = _command([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits"], timeout=10)
    pids = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2:
            raise RuntimeError("Assigned GPU context telemetry has an unsupported format")
        if values[1] == gpu_uuid:
            pids.append(int(values[0]))
    return sorted(pids)


def _terminal_unit_state(unit: str, *, allow_failed: bool,
                         live_cgroup: str | None = None) -> dict:
    result = _command(["systemctl", "--user", "show", unit,
                       "--property=ActiveState,Result,MainPID,ControlGroup", "--no-pager"],
                      timeout=10)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines()
                  if "=" in line)
    active = fields.get("ActiveState")
    if active != "inactive" and not (allow_failed and active == "failed"):
        raise RuntimeError(f"CellViT case unit is not terminal: {fields}")
    try:
        main_pid = int(fields["MainPID"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("CellViT case MainPID cannot be verified") from error
    if main_pid != 0:
        raise RuntimeError("CellViT case still has a main process")
    if not fields.get("Result"):
        raise RuntimeError("CellViT case result is unavailable")
    if not allow_failed and fields["Result"] != "success":
        raise RuntimeError("CellViT case unit did not finish successfully")
    cgroup = fields.get("ControlGroup")
    if cgroup is None or (cgroup and (not cgroup.startswith("/user.slice/") or ".." in cgroup)):
        raise RuntimeError("CellViT case cgroup identity is unavailable")
    if cgroup:
        processes = Path("/sys/fs/cgroup") / cgroup.lstrip("/") / "cgroup.procs"
        if not processes.is_file():
            raise RuntimeError("CellViT case cgroup process list cannot be read")
        if processes.read_text(encoding="utf-8").strip():
            raise RuntimeError("CellViT case cgroup retains processes")
    if live_cgroup:
        if not live_cgroup.startswith("/user.slice/") or ".." in live_cgroup:
            raise RuntimeError("CellViT live cgroup identity is invalid")
        old_group = Path("/sys/fs/cgroup") / live_cgroup.lstrip("/")
        if old_group.exists():
            previous_processes = old_group / "cgroup.procs"
            if not previous_processes.is_file():
                raise RuntimeError("Former CellViT cgroup process list cannot be read")
            if previous_processes.read_text(encoding="utf-8").strip():
                raise RuntimeError("Former CellViT cgroup retains processes")
    return {"active_state": active, "unit_result": fields["Result"],
            "main_pid": main_pid, "control_group": cgroup,
            "cgroup_processes_absent": True,
            "former_live_cgroup": live_cgroup,
            "former_live_cgroup_processes_absent": True if live_cgroup else None}


def _unit_state(unit: str) -> dict:
    return _terminal_unit_state(unit, allow_failed=False)


def _failure_unit_state(unit: str, live_cgroup: str | None) -> dict:
    return _terminal_unit_state(unit, allow_failed=True, live_cgroup=live_cgroup)


def _clean_snapshots(unit: str, gpu_uuid: str, baseline: list[int],
                     *, failed_case: bool = False,
                     live_cgroup: str | None = None) -> list[dict]:
    snapshots = []
    for index in range(2):
        state = (_failure_unit_state(unit, live_cgroup) if failed_case
                 else _unit_state(unit))
        contexts = _contexts(gpu_uuid)
        if contexts != baseline:
            raise RuntimeError("Assigned GPU contexts differ from the pre-case baseline")
        snapshots.append({**state, "assigned_gpu_context_pids": contexts})
        if index == 0:
            time.sleep(1)
    return snapshots


class UnitPending(RuntimeError):
    """The inert payload is waiting for systemd to register its new unit."""


def _live_unit_state(unit: str, memory_mb: int, tasks_max: int,
                     *, cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict:
    result = _command(["systemctl", "--user", "show", unit,
                       "--property=ActiveState,ControlGroup,MemoryMax,MemorySwapMax,TasksMax",
                       "--no-pager"], timeout=10)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines()
                  if "=" in line)
    if fields.get("ActiveState") in (None, "inactive", "activating"):
        raise UnitPending("CellViT unit is not active yet")
    if fields.get("ActiveState") != "active":
        raise RuntimeError("CellViT unit failed before resource admission")
    try:
        memory = int(fields["MemoryMax"])
        swap = int(fields["MemorySwapMax"])
        tasks = int(fields["TasksMax"])
    except (KeyError, ValueError) as error:
        raise RuntimeError("Chiamaka memory/tasks resource controls are unavailable") from error
    if not 0 < memory <= memory_mb * 1024**2 or swap != 0 or tasks != tasks_max:
        raise RuntimeError("Chiamaka does not enforce this live case's memory/swap/tasks limits")
    cgroup = fields.get("ControlGroup", "")
    if not cgroup.startswith("/user.slice/") or ".." in cgroup:
        raise RuntimeError("CellViT live unit lacks its expected user cgroup")
    processes = cgroup_root / cgroup.lstrip("/") / "cgroup.procs"
    if not processes.is_file() or not processes.read_text(encoding="utf-8").strip():
        raise RuntimeError("CellViT live cgroup has no readable owned processes")
    return {"unit": unit, "memory_max_bytes": memory, "memory_swap_max_bytes": swap,
            "tasks_max": tasks,
            "control_group": cgroup, "cpu_affinity": [0, 1]}


def supervise_case(args, root: Path, plan: dict, shard: dict, native_command: list[str]) -> dict:
    """Launch exactly one case; leave its evidence intact on any failure."""
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{32,40}", args.gpu_uuid or ""):
        raise ValueError("--gpu-uuid must name the assigned NVIDIA GPU UUID")
    if not 60 <= args.case_timeout_seconds <= 7200:
        raise ValueError("One CellViT GPU case must have a finite 60–7200 second limit")
    if args.cpus != 2 or not 1024 <= args.memory_mb <= 12288:
        raise ValueError("Chiamaka campaign is bounded to 2 CPUs and at most 12 GiB")
    if (not MIN_CASE_TASKS <= args.case_tasks_max <= MAX_CASE_TASKS or
            not 1024 <= args.app_memory_mb <= args.memory_mb - 2048):
        raise ValueError("CellViT/Ray needs a finite task cap and at least 2 GiB cgroup memory headroom")
    taskset = shutil.which("taskset")
    if not taskset or not {0, 1}.issubset(os.sched_getaffinity(0)):
        raise RuntimeError("Chiamaka CPUs 0–1 are unavailable for the bounded case")
    runtime = {}
    if args.runtime_sif:
        pydeps = verified_pydeps(getattr(args, "runtime_pydeps", None),
                                 getattr(args, "runtime_pydeps_sha256", ""))
        sif = args.runtime_sif.resolve(strict=True)
        if digest_file(sif) != args.runtime_sif_sha256:
            raise RuntimeError("CellViT Singularity image changed before this GPU case")
        singularity = shutil.which("singularity")
        if not singularity:
            raise RuntimeError("Pinned CellViT Singularity runtime is unavailable")
        cache = os.environ.get("CELLVIT_CACHE", "")
        if not cache or not Path(cache).is_absolute() or not Path(cache).is_dir():
            raise RuntimeError("CELLVIT_CACHE must be an existing absolute host folder")
        inputs = (args.image, args.shift, args.clean_tissue_mask,
                  args.classifier_cache, root, Path(__file__).resolve().parents[1],
                  Path(cache), *((pydeps,) if pydeps else ()))
        bind_dirs = sorted({str((path if Path(path).is_dir() else Path(path).parent).resolve())
                            for path in inputs})
        runtime = {"sif": str(sif), "sif_sha256": args.runtime_sif_sha256,
                   "singularity": singularity,
                   "singularity_sha256": digest_file(singularity),
                   "bind_dirs": bind_dirs, "cellvit_cache": cache,
                   "pydeps": str(pydeps) if pydeps else None,
                   "pydeps_sha256": getattr(args, "runtime_pydeps_sha256", "") or None}
    case_dir = root / "cases" / shard["id"]
    if case_dir.exists():
        raise FileExistsError(f"Preserve previous CellViT case evidence: {case_dir}")
    case_dir.mkdir(parents=True)
    case_id = uuid.uuid4().hex[:12]
    unit = f"cellvit-{shard['id']}-{case_id}.service"
    baseline = _contexts(args.gpu_uuid)
    scope = {
        "schema": "cellvit-fullslide-one-case-scope-v1",
        "case_id": case_id, "unit": unit, "shard_id": shard["id"],
        "plan_sha256": plan["identity_sha256"],
        "image_sha256": plan["image"]["sha256"],
        "gpu_uuid": args.gpu_uuid, "baseline_context_pids": baseline,
        "cpus": args.cpus, "memory_mb": args.memory_mb,
        "app_memory_mb": args.app_memory_mb,
        "tasks_max": args.case_tasks_max, "swap_max_bytes": 0,
        "timeout_seconds": args.case_timeout_seconds,
        "native_argv": native_command,
        "native_source_sha256": digest_file(Path(native_command[1])),
        "resource_admission": "live case gate required before GPU work",
        "runtime": runtime,
        "no_retry": True,
    }
    atomic_json(case_dir / "CASE_SCOPE.json", scope)
    env = [f"CUDA_VISIBLE_DEVICES={args.gpu_uuid}"]
    thread_caps = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
    env += [f"{key}={value}" for key, value in thread_caps.items()]
    for key in ("PATH", "LD_LIBRARY_PATH", "CELLVIT_CACHE", "CELLVIT_TMP_ROOT"):
        if key in os.environ:
            env.append(f"{key}={os.environ[key]}")
    command = ["systemd-run", "--user", "--wait", "--same-dir",
               f"--unit={unit}", f"--property=MemoryMax={args.memory_mb}M",
               "--property=MemorySwapMax=0",
               f"--property=TasksMax={args.case_tasks_max}",
               f"--property=RuntimeMaxSec={args.case_timeout_seconds}",
               "--property=TimeoutStopSec=30"]
    command += ["--setenv=" + value for value in env]
    if runtime:
        native_command = [runtime["singularity"], "exec", "--nv", "--cleanenv",
                          "--env", f"CUDA_VISIBLE_DEVICES={args.gpu_uuid}",
                          "--env", f"CELLVIT_CACHE={runtime['cellvit_cache']}",
                          "--env", "PYTHONDONTWRITEBYTECODE=1",
                          *[item for key, value in thread_caps.items()
                            for item in ("--env", f"{key}={value}")],
                          *(["--env", f"PYTHONPATH={runtime['pydeps']}"]
                            if runtime["pydeps"] else []),
                          *[item for directory in runtime["bind_dirs"]
                            for item in ("--bind", f"{directory}:{directory}")],
                          runtime["sif"], *native_command]
    gate_path = case_dir / "CASE_LIVE_ADMISSION.json"
    payload = Path(__file__).with_name("cellvit_scope_payload.py")
    command += [sys.executable, str(payload), "--gate", str(gate_path),
                "--unit", unit, "--plan-sha256", plan["identity_sha256"],
                "--", taskset, "-c", "0,1", *native_command]
    process = None
    try:
        with (case_dir / "CASE_STDOUT.log").open("xb") as stdout, \
             (case_dir / "CASE_STDERR.log").open("xb") as stderr:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       start_new_session=True, close_fds=True)
            deadline = time.monotonic() + 25
            while True:
                if process.poll() is not None:
                    raise RuntimeError("CellViT user unit exited before live resource admission")
                try:
                    live = _live_unit_state(unit, args.memory_mb, args.case_tasks_max)
                    break
                except UnitPending:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("CellViT live resource admission timed out")
                    time.sleep(0.2)
            # Keep the observed exact cgroup before checking ancestor pids.
            # A headroom failure must still permit bounded terminal cleanup.
            live["task_headroom"] = _task_headroom(
                live["control_group"], tasks_max=args.case_tasks_max)
            atomic_json(gate_path, {"schema": "cellvit-live-scope-admission-v1",
                                    "unit": unit, "plan_sha256": plan["identity_sha256"],
                                    "resource_controls_verified": True,
                                    **live})
            returncode = process.wait(timeout=args.case_timeout_seconds + 90)
        atomic_json(case_dir / "CASE_LAUNCH_RESULT.json", {
            "returncode": returncode,
            "stdout_path": "CASE_STDOUT.log", "stderr_path": "CASE_STDERR.log"})
        if returncode != 0:
            raise RuntimeError(f"CellViT case {shard['id']} exited {returncode}")
        snapshots = _clean_snapshots(unit, args.gpu_uuid, baseline)
        shard_dir = root / "shards" / shard["id"]
        verified = verify_complete_shard(plan, shard, shard_dir)
        marker = shard_dir / "complete.json"
        receipt = {
            "schema": "cellvit-direct-supervisor-cleanup-v1",
            "attestation_state": "published", "draft": False,
            "shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
            "checkpoint_sha256": digest_file(marker),
            "cleanup_verified": True, "owned_processes_absent": True,
            "scope_inactive": True, "assigned_gpu_contexts_empty": not baseline,
            "assigned_gpu_contexts_baseline_only": bool(baseline),
            "gpu_context_baseline_sha256": __import__("hashlib").sha256(
                json.dumps(baseline, separators=(",", ":")).encode()).hexdigest(),
            "bounded_checks_complete": True,
            "case_id": case_id, "unit": unit,
            "scope_sha256": digest_file(case_dir / "CASE_SCOPE.json"),
            "live_admission_sha256": digest_file(gate_path),
            "cleanup_snapshots": snapshots,
            "cell_count": verified["cell_count"],
        }
        atomic_json(shard_dir / "CELLVIT_GPU_CLEANUP.json", receipt)
        return receipt
    except BaseException:
        if process is not None:
            try:
                _command(["systemctl", "--user", "stop", unit], timeout=30,
                         check=False)
            except (subprocess.TimeoutExpired, OSError):
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            try:
                if "live" not in locals() or not live.get("control_group"):
                    raise RuntimeError("CellViT live cgroup was not captured before failure")
                observations = _clean_snapshots(
                    unit, args.gpu_uuid, baseline, failed_case=True,
                    live_cgroup=live["control_group"])
                cleanup = {
                    "schema": "cellvit-failed-case-terminal-cleanup-v1",
                    "unit": unit, "shard_id": shard["id"],
                    "scientific_case_succeeded": False,
                    "cleanup_verified": True, "no_retry_or_successor": True,
                    "assigned_gpu_baseline_pids": baseline,
                    "terminal_snapshots": observations,
                }
            except BaseException as cleanup_error:
                cleanup = {"schema": "cellvit-failed-case-terminal-cleanup-v1",
                           "unit": unit, "shard_id": shard["id"],
                           "scientific_case_succeeded": False,
                           "cleanup_verified": False, "no_retry_or_successor": True,
                           "reason": str(cleanup_error)}
            atomic_json(case_dir / "CASE_FAILURE_CLEANUP.json", cleanup)
        raise
