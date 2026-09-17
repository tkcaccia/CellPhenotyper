"""Visibility-aware admission with fake NVML/flock only; never probes a real GPU."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GPU0 = "GPU-aaaaaaaa-1111-1111-1111-111111111111"
GPU1 = "GPU-aaaaaaaa-2222-2222-2222-222222222222"
GPU2 = "GPU-bbbbbbbb-3333-3333-3333-333333333333"
INVENTORY = f"0, {GPU0}, 49152, 40960\n1, {GPU1}, 32768, 24576\n2, {GPU2}, 24576, 16384\n"
BASH = os.environ.get("CELLPHENOTYPER_TEST_BASH") or shutil.which("bash") or "/bin/bash"
BASH_VERSION = tuple(
    int(part)
    for part in subprocess.check_output(
        [BASH, "-c", 'printf "%s.%s" "${BASH_VERSINFO[0]}" "${BASH_VERSINFO[1]}"'], text=True
    ).split(".")
)
requires_dynamic_fds = pytest.mark.skipif(BASH_VERSION < (4, 1), reason="Full lease requires Bash >=4.1 dynamic file descriptors")


def run_allocator(tmp_path, visibility=None, inventory=INVENTORY, required_gb=2, inventory_status=0):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    inventory_path = tmp_path / "inventory.csv"
    inventory_path.write_text(inventory)
    query_log = tmp_path / "queries.txt"
    flock_log = tmp_path / "flock.txt"
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$GPU_TEST_QUERIES\"\n"
        "if [[ \"$*\" == '--query-gpu=index,uuid,memory.total,memory.free --format=csv,noheader,nounits' ]]; then\n"
        "  if (( GPU_TEST_INVENTORY_STATUS != 0 )); then exit \"$GPU_TEST_INVENTORY_STATUS\"; fi\n"
        "  while IFS= read -r row; do printf '%s\\n' \"$row\"; done < \"$GPU_TEST_INVENTORY\"\n"
        "elif [[ \"$*\" == --id=* ]]; then\n"
        "  printf 'Fake GPU, 49152, 40960\\n'\n"
        "else\n"
        "  printf 'Unexpected fake query: %s\\n' \"$*\" >&2; exit 99\n"
        "fi\n"
    )
    fake_flock = fake_bin / "flock"
    fake_flock.write_text(
        "#!/usr/bin/env bash\n"
        "[[ \"$1\" == -n && \"$2\" =~ ^[0-9]+$ ]] || exit 99\n"
        "printf '%s\\n' \"$*\" >> \"$GPU_TEST_FLOCK\"\n"
        "exit 0\n"
    )
    nvidia_smi.chmod(0o755)
    fake_flock.chmod(0o755)
    env = os.environ.copy()
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CELLPHENOTYPER_GPU_INDEX"):
        env.pop(name, None)
    env.update(visibility or {})
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        GPU_TEST_INVENTORY=str(inventory_path),
        GPU_TEST_QUERIES=str(query_log),
        GPU_TEST_FLOCK=str(flock_log),
        GPU_TEST_INVENTORY_STATUS=str(inventory_status),
    )
    command = (
        "set -uo pipefail\n"
        f"source {shlex.quote(str(ROOT / 'bin/acquire_gpu_slot.sh'))}\n"
        f"cellphenotyper_acquire_gpu_slot {shlex.quote(str(tmp_path / 'locks'))} {required_gb} 1 2 1 1 0\n"
        "result=$?\n"
        "printf 'visibility=%s\\nnvidia=%s\\nselected=%s\\n' "
        '"${CUDA_VISIBLE_DEVICES-__UNSET__}" "${NVIDIA_VISIBLE_DEVICES-__UNSET__}" '
        '"${CELLPHENOTYPER_GPU_INDEX-__UNSET__}"\n'
        'exit "$result"\n'
    )
    result = subprocess.run([BASH, "-c", command], env=env, text=True, capture_output=True, timeout=10)
    return result, query_log.read_text(), list((tmp_path / "locks").glob("*.lock")), flock_log


@pytest.mark.parametrize(
    "variable,is_set,restriction,expected",
    [
        ("CUDA_VISIBLE_DEVICES", "", "", [0, 1, 2]),
        ("NVIDIA_VISIBLE_DEVICES", "", "", [0, 1, 2]),
        ("NVIDIA_VISIBLE_DEVICES", "x", "all", [0, 1, 2]),
        ("CUDA_VISIBLE_DEVICES", "x", "2", [2]),
        ("NVIDIA_VISIBLE_DEVICES", "x", "1", [1]),
        ("CUDA_VISIBLE_DEVICES", "x", "2,1", [2, 1]),
        ("CUDA_VISIBLE_DEVICES", "x", GPU2, [2]),
        ("CUDA_VISIBLE_DEVICES", "x", "GPU-aaaaaaaa-2", [1]),
        ("NVIDIA_VISIBLE_DEVICES", "x", "GPU-bbbbbbbb", [2]),
        ("CUDA_VISIBLE_DEVICES", "x", f"0,{GPU2}", [0, 2]),
    ],
)
def test_visibility_resolution_runs_without_gpu_or_lease(tmp_path, variable, is_set, restriction, expected):
    # The restriction resolver is independently executable even on macOS Bash
    # 3.2; only the existing lease-descriptor code requires a newer Bash.
    normalized = INVENTORY.replace(" ", "").rstrip("\n")
    result = subprocess.run(
        [
            BASH, "-c",
            'set -euo pipefail; source "$1"; cellphenotyper_gpu_visibility_indices "$2" "$3" "$4" "$5"',
            "visibility-test", str(ROOT / "bin/acquire_gpu_slot.sh"), variable, is_set, restriction, normalized,
        ],
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert [int(value) for value in result.stdout.split()] == expected


@requires_dynamic_fds
@pytest.mark.parametrize(
    "visibility,index,uuid",
    [
        ({}, 0, GPU0),
        ({"NVIDIA_VISIBLE_DEVICES": "all"}, 0, GPU0),
        ({"CUDA_VISIBLE_DEVICES": "2"}, 2, GPU2),
        ({"NVIDIA_VISIBLE_DEVICES": "1"}, 1, GPU1),
        ({"CUDA_VISIBLE_DEVICES": "2,1"}, 1, GPU1),
        ({"CUDA_VISIBLE_DEVICES": GPU2}, 2, GPU2),
        ({"CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-2"}, 1, GPU1),
        ({"NVIDIA_VISIBLE_DEVICES": "GPU-bbbbbbbb"}, 2, GPU2),
        ({"CUDA_VISIBLE_DEVICES": "0,1", "NVIDIA_VISIBLE_DEVICES": "1,2"}, 1, GPU1),
        ({"CUDA_VISIBLE_DEVICES": GPU2, "NVIDIA_VISIBLE_DEVICES": "2"}, 2, GPU2),
        ({"CUDA_VISIBLE_DEVICES": "2", "NVIDIA_VISIBLE_DEVICES": "all"}, 2, GPU2),
    ],
)
def test_only_intersection_is_admitted_with_uuid_identity(tmp_path, visibility, index, uuid):
    result, queries, locks, flock_log = run_allocator(tmp_path, visibility)
    assert result.returncode == 0, result.stderr
    assert f"visibility={uuid}\n" in result.stdout
    assert f"selected={index}\n" in result.stdout
    assert f"nvidia={visibility.get('NVIDIA_VISIBLE_DEVICES', '__UNSET__')}\n" in result.stdout
    assert f"--id={index} " in queries
    assert len(queries.splitlines()) == 2
    assert locks and all(path.name.startswith(f"gpu-{index}-") for path in locks)
    assert any(path.name == f"gpu-{index}-task-0.lock" for path in locks)
    assert any(path.name == f"gpu-{index}-memory-0.lock" for path in locks)
    assert flock_log.exists()


@pytest.mark.parametrize("variable", ["CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"])
@pytest.mark.parametrize("disabled", ["", "none", "void", "-1"])
def test_explicit_no_gpu_never_becomes_unrestricted(tmp_path, variable, disabled):
    result, queries, locks, flock_log = run_allocator(tmp_path, {variable: disabled})
    assert result.returncode == 2
    assert "explicitly disables" in result.stderr
    assert "--id=" not in queries
    assert not locks and not flock_log.exists()
    assert "selected=__UNSET__\n" in result.stdout
    expected_cuda = disabled if variable == "CUDA_VISIBLE_DEVICES" else "__UNSET__"
    assert f"visibility={expected_cuda}\n" in result.stdout


@pytest.mark.parametrize(
    "visibility,reason",
    [
        ({"CUDA_VISIBLE_DEVICES": "0", "NVIDIA_VISIBLE_DEVICES": "2"}, "no common GPU"),
        ({"CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa"}, "ambiguous"),
        ({"NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa"}, "ambiguous"),
        ({"CUDA_VISIBLE_DEVICES": "42"}, "unknown"),
        ({"NVIDIA_VISIBLE_DEVICES": "42"}, "unknown"),
        ({"CUDA_VISIBLE_DEVICES": "GPU-cccccccc"}, "unknown"),
        ({"CUDA_VISIBLE_DEVICES": "MIG-GPU-aaaaaaaa-1111-1111-1111-111111111111/1/2"}, "MIG"),
        ({"NVIDIA_VISIBLE_DEVICES": "MIG-aaaaaaaa-1111-1111-1111-111111111111"}, "MIG"),
        ({"CUDA_VISIBLE_DEVICES": "all"}, "Unsupported"),
        ({"CUDA_VISIBLE_DEVICES": "0,2,-1,1"}, "truncation"),
        ({"CUDA_VISIBLE_DEVICES": "0,,1"}, "Unsupported"),
        ({"CUDA_VISIBLE_DEVICES": "0,"}, "Unsupported"),
        ({"NVIDIA_VISIBLE_DEVICES": ",1"}, "Unsupported"),
        ({"CUDA_VISIBLE_DEVICES": "0, 1"}, "Unsupported"),
        ({"CUDA_VISIBLE_DEVICES": "01"}, "Unsupported"),
        ({"CUDA_VISIBLE_DEVICES": "0,0"}, "duplicate"),
        ({"CUDA_VISIBLE_DEVICES": f"0,{GPU0}"}, "duplicate"),
    ],
)
def test_ambiguous_or_unsupported_restrictions_fail_before_lease(tmp_path, visibility, reason):
    result, queries, locks, flock_log = run_allocator(tmp_path, visibility)
    assert result.returncode == 2
    assert reason in result.stderr
    assert "--id=" not in queries
    assert not locks and not flock_log.exists()
    assert f"visibility={visibility.get('CUDA_VISIBLE_DEVICES', '__UNSET__')}\n" in result.stdout
    assert "selected=__UNSET__\n" in result.stdout


@pytest.mark.parametrize(
    "visibility",
    [
        {"CUDA_VISIBLE_DEVICES": "2"},
        {"CUDA_VISIBLE_DEVICES": "0,2", "NVIDIA_VISIBLE_DEVICES": "1,2"},
    ],
)
def test_allowed_device_without_capacity_does_not_fall_back_outside_allocation(tmp_path, visibility):
    result, queries, locks, flock_log = run_allocator(tmp_path, visibility, required_gb=20)
    assert result.returncode == 3
    assert "Timed out" in result.stderr
    assert "--id=" not in queries
    assert not locks and not flock_log.exists()
    assert f"visibility={visibility['CUDA_VISIBLE_DEVICES']}\n" in result.stdout


@requires_dynamic_fds
@pytest.mark.parametrize("uuid_field", ["", "N/A", "[N/A]", "[Not Supported]"])
def test_missing_uuid_fallback_retains_numeric_restriction(tmp_path, uuid_field):
    inventory = INVENTORY.replace(GPU2, uuid_field)
    result, queries, locks, _ = run_allocator(tmp_path, {"CUDA_VISIBLE_DEVICES": "2"}, inventory)
    assert result.returncode == 0, result.stderr
    assert "visibility=2\n" in result.stdout and "selected=2\n" in result.stdout
    assert "--id=2 " in queries
    assert all(path.name.startswith("gpu-2-") for path in locks)


def test_missing_uuid_never_widens_uuid_restriction(tmp_path):
    inventory = INVENTORY.replace(GPU2, "N/A")
    result, _, locks, flock_log = run_allocator(tmp_path, {"CUDA_VISIBLE_DEVICES": GPU0}, inventory)
    assert result.returncode == 2
    assert "incomplete GPU UUID inventory" in result.stderr
    assert not locks and not flock_log.exists()


@pytest.mark.parametrize(
    "inventory",
    [
        "",
        "0, 49152, 40960\n",
        INVENTORY.replace(GPU1, GPU0),
        INVENTORY.replace(f"1, {GPU1}", f"0, {GPU1}"),
        INVENTORY.replace("40960", "N/A"),
        INVENTORY.replace(GPU1, "MIG-aaaaaaaa-1111-1111-1111-111111111111"),
        INVENTORY.replace(GPU1, "GPU-unknown"),
    ],
)
def test_incomplete_or_ambiguous_inventory_fails_closed(tmp_path, inventory):
    result, queries, locks, flock_log = run_allocator(tmp_path, inventory=inventory)
    assert result.returncode == 2
    assert "inventory" in result.stderr
    assert "--id=" not in queries
    assert not locks and not flock_log.exists()


def test_inventory_command_failure_is_not_an_unrestricted_retry(tmp_path):
    result, queries, locks, flock_log = run_allocator(tmp_path, inventory_status=7)
    assert result.returncode == 2
    assert "Cannot obtain GPU inventory" in result.stderr
    assert len(queries.splitlines()) == 1
    assert not locks and not flock_log.exists()
