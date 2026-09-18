import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(shutil.which("flock") is None, reason="GPU admission uses util-linux flock")
def test_scheduler_selects_a_gpu_with_enough_free_memory(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *\"--query-gpu=index,uuid,memory.total,memory.free\"* ]]; then\n"
        "  printf '0, GPU-00000000-0000-0000-0000-000000000000, 16384, 8192\\n1, GPU-11111111-1111-1111-1111-111111111111, 24576, 20480\\n'\n"
        "else\n"
        "  printf 'Fake GPU, 24576, 20480\\n'\n"
        "fi\n"
    )
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    # Exercise unconstrained host scheduling even when the test itself runs
    # inside a GPU container whose base image defines an empty CUDA selector.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    command = (
        f"source '{ROOT / 'bin' / 'acquire_gpu_slot.sh'}'; "
        f"cellphenotyper_acquire_gpu_slot '{tmp_path / 'locks'}' 10 2 2 0 1 5; "
        "printf 'selected=%s required=%s\\n' \"$CELLPHENOTYPER_GPU_INDEX\" \"$CELLPHENOTYPER_GPU_REQUIRED_GB\""
    )
    completed = subprocess.run(
        ["bash", "-c", command], env=env, text=True, capture_output=True, check=True
    )
    assert "selected=1 required=10" in completed.stdout


def test_scheduler_rejects_missing_gpu_tooling(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    if shutil.which("nvidia-smi", path=env["PATH"]):
        pytest.skip("test requires a PATH without nvidia-smi")
    command = (
        f"source '{ROOT / 'bin' / 'acquire_gpu_slot.sh'}'; "
        f"cellphenotyper_acquire_gpu_slot '{tmp_path / 'locks'}' 1 0 1 1 1 1"
    )
    completed = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
    assert completed.returncode == 2
    assert "nvidia-smi is unavailable" in completed.stderr
