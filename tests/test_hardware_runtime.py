import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from hardware_runtime import (  # noqa: E402
    assigned_gpu_selector,
    auto_batch_from_free_vram,
    query_gpu_memory_mib,
)


def test_assigned_scheduler_gpu_takes_precedence() -> None:
    with patch.dict(
        os.environ,
        {"CELLPHENOTYPER_GPU_INDEX": "3", "CUDA_VISIBLE_DEVICES": "1"},
        clear=False,
    ):
        assert assigned_gpu_selector(0) == "3"


def test_visible_logical_gpu_maps_to_physical_selector() -> None:
    with patch.dict(
        os.environ,
        {"CELLPHENOTYPER_GPU_INDEX": "", "CUDA_VISIBLE_DEVICES": "5,2"},
        clear=False,
    ):
        assert assigned_gpu_selector(1) == "2"


@patch("subprocess.check_output", return_value="20000, 24576\n")
def test_memory_probe_uses_live_free_memory_and_reserve(check_output) -> None:
    with patch.dict(os.environ, {"CELLPHENOTYPER_GPU_INDEX": "4"}, clear=False):
        usable, total = query_gpu_memory_mib(0)
    assert total == 24576
    assert usable == 20000 - int(24576 * 0.08)
    assert check_output.call_args.args[0][2] == "4"
    assert "--query-gpu=memory.free,memory.total" in check_output.call_args.args[0]


@patch("subprocess.check_output", return_value="23000, 81920\n")
def test_auto_batch_uses_free_not_total_vram(_check_output) -> None:
    result = auto_batch_from_free_vram(
        0,
        0,
        tiers=((60_000, 64), (20_000, 16)),
        fallback=2,
    )
    assert result == 2


@patch("subprocess.check_output", side_effect=AssertionError("probe should not run"))
def test_explicit_batch_bypasses_probe_and_is_clamped(_check_output) -> None:
    result = auto_batch_from_free_vram(
        100,
        0,
        tiers=((20_000, 16),),
        fallback=2,
        minimum=2,
        maximum=48,
    )
    assert result == 48
