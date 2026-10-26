"""Sequential WSI admission and terminal-cleanup tests; no model or GPU run."""
from argparse import Namespace
import hashlib
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
import cellvit_fullslide_action as action
import cellvit_fullslide_supervisor as supervisor
from cellvit_models import digest_file
from cellvit_runtime_binding import pydeps_tree_sha256, verified_pydeps
from cellvit_shards import (ARCHIVE_NAME, ARCHIVE_RECEIPT, SHARD_FILES,
                            atomic_json, compact_shard, complete_shard)
from cellvit_models import CLASSIFICATION_HEADER
from test_cellvit_partition_project import _source


def _options(root, image):
    mask = root / "mask.png"
    mask.write_bytes(b"synthetic-mask")
    cache = root / "classifier-cache"
    cache.mkdir()
    return action.parser().parse_args([
        "--action", "run", "--image", str(image),
        "--shift", str(root / "shift.json"),
        "--clean-tissue-mask", str(mask), "--bundle-dir", str(root),
        "--classifier-cache", str(cache), "--python", sys.executable,
        "--expected-shards", "2", "--gpu-uuid",
        "GPU-00000000-0000-0000-0000-000000000000",
        "--reserve-gib", "0", "--case-budget-gib", "0.00001", "--amp",
    ])


def _archived_first_core(tmp_path):
    root, image, manifest = _source(tmp_path)
    # The synthetic source fixture publishes a partition manifest. A live
    # campaign archives each core before publishing that manifest.
    manifest.unlink()
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    folder = root / "shards" / shard["id"]
    receipt = {
        "schema": "cellvit-direct-supervisor-cleanup-v1",
        "attestation_state": "published", "draft": False,
        "shard_id": shard["id"], "plan_sha256": plan["identity_sha256"],
        "checkpoint_sha256": digest_file(folder / "complete.json"),
        "cleanup_verified": True, "owned_processes_absent": True,
        "scope_inactive": True, "bounded_checks_complete": True,
        "assigned_gpu_contexts_empty": True,
    }
    atomic_json(folder / action.CLEANUP_FILE, receipt)
    compact_shard(plan, shard, folder)
    assert not (folder / action.CLEANUP_FILE).exists()
    return root, image, plan, shard


def test_archived_cleanup_receipt_allows_resumed_campaign(tmp_path, monkeypatch):
    root, image, plan, first = _archived_first_core(tmp_path)
    folder = root / "shards" / first["id"]
    archive_hash = digest_file(folder / ARCHIVE_NAME)
    shutil.rmtree(root / "shards" / plan["shards"][1]["id"])
    assert action._cleanup_receipt(root, plan, first)["shard_id"] == first["id"]
    action._require_prior_cleanup(root, plan)
    monkeypatch.setattr(action, "_require_declared_coverage", lambda *_: None)
    monkeypatch.setattr(action, "_campaign_source", lambda *_: {"status": "pinned"})
    monkeypatch.setattr(action, "_preflight", lambda *_: {"status": "ready"})
    monkeypatch.setattr(action, "_probe_empty_core", lambda *_: None)
    attempted = []

    def stop_at_next_core(args, bundle, actual_plan, shard, command):
        attempted.append(shard["id"])
        raise RuntimeError("next core reached")

    monkeypatch.setattr(supervisor, "supervise_case", stop_at_next_core)
    with pytest.raises(RuntimeError, match="next core reached"):
        action.run_action(_options(root, image))
    assert attempted == [plan["shards"][1]["id"]]
    assert digest_file(folder / ARCHIVE_NAME) == archive_hash
    assert not (folder / action.CLEANUP_FILE).exists()


def test_archived_cleanup_member_must_match_compaction_receipt(tmp_path):
    root, _, plan, first = _archived_first_core(tmp_path)
    folder = root / "shards" / first["id"]
    receipt_file = folder / ARCHIVE_RECEIPT
    receipt = json.loads(receipt_file.read_text())
    receipt["members"][action.CLEANUP_FILE]["sha256"] = "0" * 64
    atomic_json(receipt_file, receipt)
    with pytest.raises(RuntimeError, match="cleanup hash or size differs"):
        action._cleanup_receipt(root, plan, first)


def test_archived_core_can_be_adopted_with_exact_external_receipt(tmp_path):
    root, _, plan, first = _archived_first_core(tmp_path)
    source = root / "shards" / first["id"]
    archive_receipt = json.loads((source / ARCHIVE_RECEIPT).read_text())
    expected = archive_receipt["members"][action.CLEANUP_FILE]
    with tarfile.open(source / ARCHIVE_NAME, "r:gz") as archive:
        stream = archive.extractfile(action.CLEANUP_FILE)
        assert stream is not None
        with stream:
            evidence = stream.read()
    assert len(evidence) == expected["bytes"]
    assert hashlib.sha256(evidence).hexdigest() == expected["sha256"]
    external = tmp_path / "prior-cleanup-evidence.json"
    external.write_bytes(evidence)
    target = tmp_path / "fresh-bundle"
    target.mkdir()
    atomic_json(target / "cellvit_shard_plan.json", plan)
    options = Namespace(adopt_from=source, expected_checkpoint_sha256=digest_file(source / "complete.json"),
                        adoption_cleanup_receipt=external,
                        adoption_cleanup_receipt_sha256=digest_file(external),
                        allow_case_owned_source=False)
    altered = tmp_path / "altered-evidence.json"
    altered.write_bytes(evidence + b"\n")
    options.adoption_cleanup_receipt = altered
    options.adoption_cleanup_receipt_sha256 = digest_file(altered)
    with pytest.raises(ValueError, match="differs from the archived source"):
        action._adopt_shard(options, target, plan, first)
    assert not (target / "shards").exists()
    options.adoption_cleanup_receipt = external
    options.adoption_cleanup_receipt_sha256 = digest_file(external)
    result = action._adopt_shard(options, target, plan, first)
    assert result["shard_id"] == first["id"]
    assert action._cleanup_receipt(target, plan, first)["shard_id"] == first["id"]
    assert not (source / action.CLEANUP_FILE).exists()


def test_cpu_empty_attestation_remains_checked_after_compaction(tmp_path):
    root, _, manifest = _source(tmp_path)
    manifest.unlink()
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    plan["run_contract"]["clean_tissue_mask_sha256"] = "a" * 64
    folder = root / "shards" / shard["id"]
    shutil.rmtree(folder)
    folder.mkdir()
    for name in SHARD_FILES:
        (folder / name).write_text("{}")
    (folder / "cellvit_cells.json").write_text('{"cells":[]}')
    (folder / "cellvit_cell_classification.json").write_text(json.dumps({
        "models": [{"model_id": f"head-{index}"} for index in range(8)], "cells": []}))
    with (folder / "cellvit_cell_classification.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(CLASSIFICATION_HEADER)
    evidence = {"policy": "grandqc-zero-clean-core-conservative-mask-cover-v1",
                "core_xyxy": shard["core"], "nonzero_mask_pixels": 0,
                "plan_sha256": plan["identity_sha256"],
                "image_sha256": plan["image"]["sha256"],
                "full_shift_sha256": plan["shift_sha256"],
                "clean_tissue_mask_sha256": "a" * 64}
    (folder / "cellvit_metadata.json").write_text(json.dumps({
        "model": "SAM", "requested_taxonomies": ["all"],
        "model_provenance": {"checkpoints": [{"sha256": "synthetic"}]},
        "empty_core_evidence": evidence}))
    complete_shard(plan, shard, folder)
    receipt = {"schema": action.CPU_EMPTY_SCHEMA, "attestation_state": "published",
               "draft": False, "shard_id": shard["id"],
               "plan_sha256": plan["identity_sha256"],
               "checkpoint_sha256": digest_file(folder / "complete.json"),
               "cpu_only": True, "gpu_started": False,
               "empty_core_evidence": evidence}
    atomic_json(folder / action.CLEANUP_FILE, receipt)
    assert action._cleanup_receipt(root, plan, shard) == receipt
    compact_shard(plan, shard, folder)
    assert not (folder / action.CLEANUP_FILE).exists()
    assert action._cleanup_receipt(root, plan, shard) == receipt
    with pytest.raises(RuntimeError, match="CPU-empty provenance"):
        action._cleanup_receipt(
            root, {**plan, "run_contract": {**plan["run_contract"],
                                           "clean_tissue_mask_sha256": "b" * 64}}, shard)
    with tarfile.open(folder / ARCHIVE_NAME, "r:gz") as archive:
        stream = archive.extractfile(action.CLEANUP_FILE)
        assert stream is not None
        with stream:
            external_bytes = stream.read()
    external = tmp_path / "cpu-empty-evidence.json"
    external.write_bytes(external_bytes)
    target = tmp_path / "next-version"
    target.mkdir()
    atomic_json(target / "cellvit_shard_plan.json", plan)
    options = Namespace(adopt_from=folder,
                        expected_checkpoint_sha256=digest_file(folder / "complete.json"),
                        adoption_cleanup_receipt=external,
                        adoption_cleanup_receipt_sha256=digest_file(external),
                        allow_case_owned_source=False)
    adopted = action._adopt_shard(options, target, plan, shard)
    assert adopted["shard_id"] == shard["id"]
    assert action._cleanup_receipt(target, plan, shard) == receipt


def test_first_failed_core_latches_campaign_without_running_successor(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shutil.rmtree(root / "shards")
    options = _options(root, image)
    monkeypatch.setattr(action, "_require_declared_coverage", lambda *_: None)
    monkeypatch.setattr(action, "_preflight", lambda *_: {"status": "ready"})
    monkeypatch.setattr(action, "_probe_empty_core", lambda *_: None)
    calls = []

    def fail_case(args, bundle, actual_plan, shard, command):
        calls.append(shard["id"])
        assert command[-2:] == ["--shard-id", shard["id"]]
        (bundle / "cases" / shard["id"]).mkdir(parents=True)
        raise RuntimeError("synthetic native failure")

    monkeypatch.setattr(supervisor, "supervise_case", fail_case)
    with pytest.raises(RuntimeError, match="synthetic native failure"):
        action.run_action(options)
    assert calls == [plan["shards"][0]["id"]]
    assert (root / "GPU_BUNDLE_FAILED.json").is_file()
    with pytest.raises(RuntimeError, match="latched"):
        action.run_action(options)
    assert calls == [plan["shards"][0]["id"]]


def test_supervisor_receipt_needs_two_clean_unit_and_gpu_observations(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    args = Namespace(gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                     case_timeout_seconds=120, cpus=2, memory_mb=4096,
                     app_memory_mb=2048, case_tasks_max=512,
                     runtime_sif=None)
    observations = []
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda *_: {0, 1}, raising=False)
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(supervisor, "_contexts", lambda uuid: [])
    monkeypatch.setattr(supervisor, "_live_unit_state", lambda unit, memory_mb, tasks_max: {
        "unit": unit, "memory_max_bytes": memory_mb * 1024**2,
        "tasks_max": tasks_max, "control_group": "/user.slice/test", "cpu_affinity": [0, 1]})
    monkeypatch.setattr(supervisor, "_task_headroom", lambda *args, **kwargs: [{"pids_max": 512}])
    monkeypatch.setattr(supervisor, "_unit_state", lambda unit: (
        observations.append(unit) or {"active_state": "inactive",
                                      "control_group": "", "cgroup_processes_absent": True}))
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process())
    receipt = supervisor.supervise_case(args, root, plan, shard,
                                         [sys.executable, str(ROOT / "bin/run_cellvitpp.py")])
    assert len(observations) == 2
    assert receipt["cleanup_verified"] is True
    assert receipt["checkpoint_sha256"] == digest_file(
        root / "shards" / shard["id"] / "complete.json")
    assert (root / "shards" / shard["id"] / "CELLVIT_GPU_CLEANUP.json").is_file()


def test_supervisor_uncertain_cleanup_never_publishes_receipt(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    args = Namespace(gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                     case_timeout_seconds=120, cpus=2, memory_mb=4096,
                     app_memory_mb=2048, case_tasks_max=512,
                     runtime_sif=None)
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda *_: {0, 1}, raising=False)
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(supervisor, "_contexts", lambda uuid: [])
    monkeypatch.setattr(supervisor, "_live_unit_state", lambda unit, memory_mb, tasks_max: {
        "unit": unit, "memory_max_bytes": memory_mb * 1024**2,
        "tasks_max": tasks_max, "control_group": "/user.slice/test", "cpu_affinity": [0, 1]})
    monkeypatch.setattr(supervisor, "_task_headroom", lambda *args, **kwargs: [{"pids_max": 512}])
    monkeypatch.setattr(supervisor, "_failure_unit_state", lambda unit, live_cgroup: (_ for _ in ()).throw(
        RuntimeError("unknown cgroup state")))
    monkeypatch.setattr(supervisor, "_unit_state", lambda unit: (_ for _ in ()).throw(
        RuntimeError("unknown cgroup state")))
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, "", ""))
    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match="unknown cgroup state"):
        supervisor.supervise_case(args, root, plan, shard,
                                  [sys.executable, str(ROOT / "bin/run_cellvitpp.py")])
    assert not (root / "shards" / shard["id"] / "CELLVIT_GPU_CLEANUP.json").exists()
    cases = list((root / "cases" / shard["id"]).glob("CASE_FAILURE_CLEANUP.json"))
    assert len(cases) == 1


def test_failed_oom_unit_can_have_verified_cleanup_without_success(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    args = Namespace(gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                     case_timeout_seconds=120, cpus=2, memory_mb=8192,
                     app_memory_mb=4096, case_tasks_max=512,
                     runtime_sif=None)
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda *_: {0, 1}, raising=False)
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(supervisor, "_contexts", lambda uuid: [])
    monkeypatch.setattr(supervisor, "_live_unit_state", lambda unit, memory_mb, tasks_max: {
        "unit": unit, "memory_max_bytes": memory_mb * 1024**2,
        "tasks_max": tasks_max, "control_group": "/user.slice/test", "cpu_affinity": [0, 1]})
    monkeypatch.setattr(supervisor, "_task_headroom", lambda *args, **kwargs: [{"pids_max": 512}])
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0,
                            "ActiveState=failed\nResult=oom-kill\nMainPID=0\nControlGroup=\n", ""))
    monkeypatch.setattr(supervisor, "_failure_unit_state", lambda unit, live_cgroup: {
        "active_state": "failed", "unit_result": "oom-kill", "main_pid": 0,
        "control_group": "", "cgroup_processes_absent": True,
        "former_live_cgroup": live_cgroup,
        "former_live_cgroup_processes_absent": True})
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            return 1

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match="exited 1"):
        supervisor.supervise_case(args, root, plan, shard,
                                  [sys.executable, str(ROOT / "bin/run_cellvitpp.py")])
    failure = json.loads((root / "cases" / shard["id"] / "CASE_FAILURE_CLEANUP.json").read_text())
    assert failure["cleanup_verified"] is True
    assert failure["scientific_case_succeeded"] is False
    assert len(failure["terminal_snapshots"]) == 2
    assert all(item["active_state"] == "failed" for item in failure["terminal_snapshots"])
    assert failure["no_retry_or_successor"] is True
    assert not (root / "shards" / shard["id"] / "CELLVIT_GPU_CLEANUP.json").exists()


def test_failed_unit_requires_zero_main_pid_and_empty_cgroup(monkeypatch):
    unit = "cellvit-r0000c0000-abcdef012345.service"
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0,
                            "ActiveState=failed\nResult=oom-kill\nMainPID=0\nControlGroup=\n", ""))
    observed = supervisor._failure_unit_state(unit, None)
    assert observed["unit_result"] == "oom-kill"
    assert observed["cgroup_processes_absent"] is True
    with pytest.raises(RuntimeError, match="not terminal"):
        supervisor._unit_state(unit)
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0,
                            "ActiveState=failed\nResult=oom-kill\nMainPID=42\nControlGroup=\n", ""))
    with pytest.raises(RuntimeError, match="main process"):
        supervisor._failure_unit_state(unit, None)


def test_live_gate_requires_ray_task_headroom_memory_and_zero_swap(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    child = root / "user.slice" / "test.service"
    child.mkdir(parents=True)
    for directory, limit, used in ((root, "max", 3000),
                                   (root / "user.slice", "max", 3000),
                                   (child, "512", 1)):
        (directory / "pids.max").write_text(f"{limit}\n")
        (directory / "pids.current").write_text(f"{used}\n")
    (child / "cgroup.procs").write_text("42\n")
    fields = {"ActiveState": "active", "ControlGroup": "/user.slice/test.service",
              "MemoryMax": str(12288 * 1024**2), "MemorySwapMax": "0",
              "TasksMax": "512"}
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0,
                            "".join(f"{key}={value}\n" for key, value in fields.items()), ""))
    live = supervisor._live_unit_state("cellvit-test.service", 12288, 512,
                                       cgroup_root=root)
    assert live["tasks_max"] == 512
    assert live["memory_swap_max_bytes"] == 0
    assert supervisor._task_headroom(
        live["control_group"], tasks_max=512, cgroup_root=root)[0]["pids_free"] == 511
    # The unified hierarchy root may not expose pids.current when unbounded.
    (root / "pids.current").unlink()
    assert supervisor._task_headroom(
        live["control_group"], tasks_max=512, cgroup_root=root)[-1]["pids_max"] is None
    (root / "pids.max").unlink()
    assert supervisor._task_headroom(
        live["control_group"], tasks_max=512, cgroup_root=root)[-1]["pids_max"] is None
    fields["TasksMax"] = "128"
    with pytest.raises(RuntimeError, match="memory/swap/tasks"):
        supervisor._live_unit_state("cellvit-test.service", 12288, 512,
                                    cgroup_root=root)
    fields["TasksMax"] = "512"
    fields["MemorySwapMax"] = "1048576"
    with pytest.raises(RuntimeError, match="memory/swap/tasks"):
        supervisor._live_unit_state("cellvit-test.service", 12288, 512,
                                    cgroup_root=root)
    fields["MemorySwapMax"] = "0"
    (child / "pids.current").write_text("300\n")
    with pytest.raises(RuntimeError, match="headroom"):
        supervisor._task_headroom(live["control_group"], tasks_max=512,
                                  cgroup_root=root)


def test_headroom_gate_failure_retains_live_cgroup_for_cleanup(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    args = Namespace(gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                     case_timeout_seconds=120, cpus=2, memory_mb=12288,
                     app_memory_mb=8192, case_tasks_max=512, runtime_sif=None)
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda *_: {0, 1}, raising=False)
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: "/usr/bin/taskset")
    monkeypatch.setattr(supervisor, "_contexts", lambda uuid: [])
    monkeypatch.setattr(supervisor, "_live_unit_state", lambda unit, memory_mb, tasks_max: {
        "unit": unit, "memory_max_bytes": memory_mb * 1024**2,
        "memory_swap_max_bytes": 0, "tasks_max": tasks_max,
        "control_group": "/user.slice/test.service", "cpu_affinity": [0, 1]})
    monkeypatch.setattr(supervisor, "_task_headroom", lambda cgroup, **kwargs: (_ for _ in ()).throw(
        RuntimeError("ancestor pids telemetry unavailable")))
    observed = []
    monkeypatch.setattr(supervisor, "_failure_unit_state", lambda unit, live_cgroup: (
        observed.append(live_cgroup) or {"active_state": "inactive", "unit_result": "success",
            "main_pid": 0, "control_group": "", "cgroup_processes_absent": True,
            "former_live_cgroup": live_cgroup,
            "former_live_cgroup_processes_absent": True}))
    monkeypatch.setattr(supervisor, "_command", lambda argv, **kwargs:
                        subprocess.CompletedProcess(argv, 0, "", ""))
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *args, **kwargs: Process())
    with pytest.raises(RuntimeError, match="ancestor pids telemetry unavailable"):
        supervisor.supervise_case(args, root, plan, shard,
                                  [sys.executable, str(ROOT / "bin/run_cellvitpp.py")])
    case = root / "cases" / shard["id"]
    failure = json.loads((case / "CASE_FAILURE_CLEANUP.json").read_text())
    assert failure["cleanup_verified"] is True
    assert len(failure["terminal_snapshots"]) == 2
    assert observed == ["/user.slice/test.service"] * 2
    assert not (case / "CASE_LIVE_ADMISSION.json").exists()
    assert not (root / "shards" / shard["id"] / "CELLVIT_GPU_CLEANUP.json").exists()


def test_ray_app_memory_is_below_case_limit_and_case_tasks_bounded(tmp_path):
    root, image, _ = _source(tmp_path)
    args = _options(root, image)
    args.memory_mb = 12288
    args.app_memory_mb = 8192
    args.case_tasks_max = 512
    command = action._base_command(args, action="shard", shard_id="r0000c0000")
    assert command[command.index("--memory-mb") + 1] == "8192"
    assert args.memory_mb - args.app_memory_mb == 4096
    assert args.case_tasks_max <= supervisor.MAX_CASE_TASKS


def test_supervisor_pins_source_and_wraps_native_command_in_sif(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    plan = json.loads((root / "cellvit_shard_plan.json").read_text())
    shard = plan["shards"][0]
    sif = tmp_path / "cellvit.sif"
    sif.write_bytes(b"pinned-test-runtime")
    pydeps = tmp_path / "pydeps"
    (pydeps / "pyvips").mkdir(parents=True)
    (pydeps / "pyvips" / "__init__.py").write_text("# pinned fixture\n")
    cache = tmp_path / "model-cache"
    cache.mkdir()
    monkeypatch.setenv("CELLVIT_CACHE", str(cache))
    mask = tmp_path / "mask.tif"
    mask.write_bytes(b"test-mask")
    classifier = tmp_path / "classifiers"
    classifier.mkdir()
    args = Namespace(gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                     case_timeout_seconds=120, cpus=2, memory_mb=4096,
                     app_memory_mb=2048, case_tasks_max=512,
                     runtime_sif=sif, runtime_sif_sha256=digest_file(sif),
                     runtime_pydeps=pydeps, runtime_pydeps_sha256=pydeps_tree_sha256(pydeps),
                     image=image, shift=root / "shift.json",
                     clean_tissue_mask=mask, classifier_cache=classifier)
    monkeypatch.setattr(supervisor.os, "sched_getaffinity", lambda *_: {0, 1}, raising=False)
    singularity = tmp_path / "singularity"
    singularity.write_bytes(b"synthetic launcher")
    monkeypatch.setattr(supervisor.shutil, "which", lambda name: str(singularity)
                        if name == "singularity" else "/usr/bin/taskset")
    monkeypatch.setattr(supervisor, "_contexts", lambda uuid: [])
    monkeypatch.setattr(supervisor, "_live_unit_state", lambda unit, memory_mb, tasks_max: {
        "unit": unit, "memory_max_bytes": memory_mb * 1024**2,
        "tasks_max": tasks_max, "control_group": "/user.slice/test", "cpu_affinity": [0, 1]})
    monkeypatch.setattr(supervisor, "_task_headroom", lambda *args, **kwargs: [{"pids_max": 512}])
    monkeypatch.setattr(supervisor, "_unit_state", lambda unit: {
        "active_state": "inactive", "control_group": "", "cgroup_processes_absent": True})
    monkeypatch.setattr(supervisor.time, "sleep", lambda *_: None)
    launched = []

    class Process:
        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    def launch(command, **kwargs):
        launched.append(command)
        return Process()

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    script = ROOT / "bin/run_cellvitpp.py"
    native = ["/opt/micromamba/envs/cellvit/bin/python", str(script),
              "--shard-id", shard["id"]]
    supervisor.supervise_case(args, root, plan, shard, native)
    command = launched[0]
    assert "--property=TasksMax=512" in command
    assert "--property=MemorySwapMax=0" in command
    assert [str(singularity), "exec", "--nv", "--cleanenv"] == command[
        command.index(str(singularity)):command.index(str(singularity)) + 4]
    assert command[-len(native):] == native
    assert f"PYTHONPATH={pydeps}" in command
    assert f"{pydeps}:{pydeps}" in command
    assert "OMP_NUM_THREADS=2" in command
    scope = json.loads((root / "cases" / shard["id"] / "CASE_SCOPE.json").read_text())
    assert scope["native_source_sha256"] == digest_file(script)
    assert scope["runtime"]["sif_sha256"] == digest_file(sif)
    assert scope["runtime"]["pydeps_sha256"] == pydeps_tree_sha256(pydeps)


def test_cpu_plan_and_storage_use_pinned_sif(tmp_path, monkeypatch):
    root, image, _ = _source(tmp_path)
    args = _options(root, image)
    sif = tmp_path / "cellvit.sif"
    sif.write_bytes(b"pinned-test-runtime")
    pydeps = tmp_path / "pydeps"
    pydeps.mkdir()
    (pydeps / "pyvips.py").write_text("# pinned fixture\n")
    args.runtime_sif = sif
    args.runtime_sif_sha256 = digest_file(sif)
    args.runtime_python = "/opt/micromamba/envs/cellvit/bin/python"
    args.runtime_executable = "/opt/micromamba/envs/cellvit/bin/cellvit-inference"
    args.runtime_pydeps = pydeps
    args.runtime_pydeps_sha256 = pydeps_tree_sha256(pydeps)
    monkeypatch.setattr(action.shutil, "which", lambda name: "/usr/bin/singularity")
    command = action._base_command(args, action="pack", shard_id="r0000c0000")
    assert command[:3] == ["/usr/bin/singularity", "exec", "--cleanenv"]
    assert str(sif) in command
    assert f"PYTHONPATH={pydeps}" in command
    assert [args.runtime_python, str(ROOT / "bin/run_cellvitpp.py")] == command[
        command.index(args.runtime_python):command.index(args.runtime_python) + 2]
    assert command[-2:] == ["--pack-features-shard-id", "r0000c0000"]
    cache = tmp_path / "model-cache"
    cache.mkdir()
    monkeypatch.setenv("CELLVIT_CACHE", str(cache))
    empty = action._base_command(args, action="cpu_empty", shard_id="r0000c0000")
    assert "--nv" not in empty
    assert f"CELLVIT_CACHE={cache}" in empty
    assert f"{cache}:{cache}" in empty
    assert empty[-3:] == ["--shard-id", "r0000c0000", "--cpu-empty-core"]
    sif.write_bytes(b"changed-runtime")
    with pytest.raises(RuntimeError, match="changed"):
        action._base_command(args, action="pack", shard_id="r0000c0000")


def test_runtime_pydeps_hash_includes_names_contents_and_rejects_links(tmp_path):
    folder = tmp_path / "pydeps"
    folder.mkdir()
    module = folder / "pyvips.py"
    module.write_text("first\n")
    original = pydeps_tree_sha256(folder)
    assert verified_pydeps(folder, original) == folder
    module.write_text("second\n")
    with pytest.raises(RuntimeError, match="changed"):
        verified_pydeps(folder, original)
    module.rename(folder / "renamed.py")
    assert pydeps_tree_sha256(folder) != original
    (folder / "link.py").symlink_to(folder / "renamed.py")
    with pytest.raises(ValueError, match="Unsupported"):
        pydeps_tree_sha256(folder)
