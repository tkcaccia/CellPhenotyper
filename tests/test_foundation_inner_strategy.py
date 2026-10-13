"""Runtime contract for the encoder-neutral inner-region API."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("encoder", ["uni2-h", "virchow2"])
def test_nextflow_resolves_identical_inner28_stride56_contract(
    tmp_path: Path, encoder: str
) -> None:
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    (tmp_path / "lib").mkdir()
    shutil.copy2(ROOT / "lib" / "PipelineHelpers.groovy", tmp_path / "lib")
    (tmp_path / "params.json").write_text(
        json.dumps(
            {
                "foundation_encoder": encoder,
                "foundation_inner_strategy": "inner28_stride56",
                "foundation_inner_extraction": "token_subset",
                "uni2_tile_size": 224,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "main.nf").write_text(
        """nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
  def resolved = PipelineHelpers.resolveFoundationRepresentation(params)
  println('FOUNDATION_CONTRACT=' + JsonOutput.toJson(resolved))
}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nextflow,
            "-log",
            str(tmp_path / "nextflow.log"),
            "run",
            str(tmp_path / "main.nf"),
            "-params-file",
            str(tmp_path / "params.json"),
            "-ansi-log",
            "false",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(
        value for value in result.stdout.splitlines() if value.startswith("FOUNDATION_CONTRACT=")
    )
    contract = json.loads(line.split("=", 1)[1])
    assert contract["foundation_encoder"] == encoder
    assert contract["foundation_inner_strategy_resolved"] == "inner28_stride56"
    assert contract["uni2_inner_square_fixed_px"] == 28
    assert contract["uni2_grid_stride_px"] == 56
    assert contract["uni2_paired_inner_square_mode"] == "token_subset"


def test_nextflow_legacy_aliases_override_new_defaults(tmp_path: Path) -> None:
    nextflow = shutil.which("nextflow")
    if not nextflow:
        pytest.skip("Nextflow unavailable")
    (tmp_path / "lib").mkdir()
    shutil.copy2(ROOT / "lib" / "PipelineHelpers.groovy", tmp_path / "lib")
    (tmp_path / "params.json").write_text(
        json.dumps(
            {
                "foundation_encoder": "virchow2",
                "foundation_inner_strategy": "inner28_stride56",
                "uni2_encoder": "uni2-h",
                "uni2_inner_square_fixed_px": 90,
                "uni2_grid_stride_px": 0,
                "uni2_paired_inner_square_mode": "masked_forward",
                "uni2_tile_size": 224,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "main.nf").write_text(
        """nextflow.enable.dsl=2
import groovy.json.JsonOutput
workflow {
  println('FOUNDATION_CONTRACT=' + JsonOutput.toJson(PipelineHelpers.resolveFoundationRepresentation(params)))
}
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [nextflow, "-log", str(tmp_path / "nextflow.log"), "run", str(tmp_path / "main.nf"),
         "-params-file", str(tmp_path / "params.json"), "-ansi-log", "false"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(value for value in result.stdout.splitlines() if value.startswith("FOUNDATION_CONTRACT="))
    contract = json.loads(line.split("=", 1)[1])
    assert contract["foundation_encoder"] == "uni2-h"
    assert contract["foundation_inner_strategy_resolved"] == "legacy_custom_inner90_stride90"
    assert contract["uni2_inner_square_fixed_px"] == 90
    assert contract["uni2_grid_stride_px"] == 90
    assert contract["uni2_paired_inner_square_mode"] == "masked_forward"
