"""One-slide CPU controller: source pins, full grid and durable failure state."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from bin import bora_continuous_fullslide_action as action
from bin.export_experiment import export_experiment, validate_experiment


BORA_SOURCE = Path(os.environ.get("BORA_CONTINUOUS_SOURCE_DIR",
    os.environ.get("BORA_SOURCE_DIR", str(Path(__file__).resolve().parents[2] / "Bora"))))
pytestmark = pytest.mark.skipif(
    not (BORA_SOURCE / "src/bora/wsi_continuous_chain.py").is_file(),
    reason="Bora integration source is unavailable; set BORA_SOURCE_DIR")


@pytest.fixture(autouse=True)
def isolated_bora_source_imports():
    # Independent source-pinned integrations must not reuse the contextual
    # package imported by a preceding test. Production still rejects a mixed
    # installation; isolate imports here instead of weakening that guard.
    import sys
    saved = {name: value for name, value in sys.modules.items()
             if name == "bora" or name.startswith("bora.")}
    saved_path = sys.path[:]
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "bora" or name.startswith("bora."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path


def _case(root: Path) -> dict:
    root.mkdir(parents=True)
    native_wh, working_wh = (171, 127), (85, 63)
    native = root / "native.tif"
    working = root / "working.tif"
    support = root / "grandqc.tif"
    Image.new("RGB", native_wh, "pink").save(native)
    Image.new("RGB", working_wh, "pink").save(working)
    Image.fromarray(np.full(native_wh[::-1], 255, np.uint8)).save(support)
    registration = root / "registration.json"
    registration.write_text(json.dumps({
        "format": "cellphenotyper-dual-mpp-registration-v1",
        "verified": True,
        "support_sampling": "absolute pixel centre, nearest, full-frame GrandQC mask",
        "image": {"frame_id": "working", "sha256": action.sha256(working),
                  "width": working_wh[0], "height": working_wh[1],
                  "mpp_x": .5, "mpp_y": .5},
        "cell_full_image": {"frame_id": "native", "sha256": action.sha256(native),
                            "width": native_wh[0], "height": native_wh[1],
                            "mpp_x": .25, "mpp_y": .25},
        "full_support_mask": {"sha256": action.sha256(support),
                              "width": native_wh[0], "height": native_wh[1]},
        "tissue_to_cell_full_affine": [
            [native_wh[0]/working_wh[0], 0, 0],
            [0, native_wh[1]/working_wh[1], 0], [0, 0, 1]],
    }))
    base = root / "base.cellphenotyper.json"
    export_experiment({"name": "Synthetic fullslide", "tissues": [{
        "name": "Synthetic native", "image": {"path": str(native),
            "frame_id": "native", "mpp_x": .25, "mpp_y": .25},
        "artifacts": [], "feature_sets": []}]}, base, base_dir=root)
    replay = root / "replay"
    replay.mkdir()
    grid = ((working_wh[1]+13)//14, (working_wh[0]+13)//14, 2)
    p9 = np.empty(grid, np.float32)
    p19 = np.empty(grid, np.float32)
    p9[..., 0], p9[..., 1] = .95, .05
    p19[..., 0], p19[..., 1] = .9, .1
    p9[:, 3:, 0], p9[:, 3:, 1] = .05, .95
    p19[:, 3:, 0], p19[:, 3:, 1] = .1, .9
    for name, array in (("p9", p9), ("p19", p19),
                        ("combined", (p9+p19)/2)):
        np.save(replay / f"{name}.npy", array, allow_pickle=False)
    (replay / "TOKEN_REPLAY_COMPLETE.json").write_text(json.dumps({
        "format": "bora-global-rgb-token-replay-v1",
        "method": "continuous-deployment-head-v1",
        "token_shape_yx": list(grid[:2]),
        "working_shape_yx": list(working_wh[::-1]),
        "class_ids": [3, 4], "whole_field_recipe": True,
        "independently_fitted_windows": False,
        "probability_regularization_applied": False,
        "recipe_sha256": "a"*64,
        "source_sha256": {name: "b"*64 for name in
                          ("base", "scale_1", "scale_2")},
        "output_sha256": {name: action.sha256(replay / f"{name}.npy")
                          for name in ("p9", "p19", "combined")},
    }))
    return {"replay_folder": replay, "registration_path": registration,
            "registered_support_tiff": support, "base_project": base,
            "bora_source": BORA_SOURCE,
            "expected_bora_sha256": action.bora_source_identity(BORA_SOURCE)["sha256"],
            "expected_runtime_sha256": action.identity_digest(action.runtime_identity()),
            "core_pixels": 85, "halo_pixels": 4,
            "disk_budget_bytes": 4 * action.GIB,
            "memory_budget_bytes": 512 * action.MIB}


def test_complete_synthetic_grid_publishes_one_review_project(tmp_path):
    params = _case(tmp_path / "inputs")
    output = tmp_path / "slide"
    result = action.run(**params, output_folder=output)
    assert result["complete"] is True
    assert result["shard_count"] == 6
    assert (output / "RUN_COMPLETE.json").is_file()
    assert not (output / "RUN_FAILED.json").exists()
    assert (output / "global/CONTINUOUS_CHAIN_COMPLETE.json").is_file()
    assert all((output / "shards" / f"{index:04d}" /
                "SHARD_COMPLETE.json").is_file() for index in range(6))
    project = validate_experiment(output / "experiment.cellphenotyper.json")
    artifacts = project["tissues"][0]["artifacts"]
    assert len(artifacts) == 1
    assert artifacts[0]["role"] == "bora_exploratory_boundary_review"
    assert artifacts[0]["review_only"] is True
    assert not any(row["role"] == "tissue_annotations" for row in artifacts)
    with pytest.raises(FileExistsError):
        action.run(**params, output_folder=output)


@pytest.mark.parametrize("pin", ["expected_bora_sha256", "expected_runtime_sha256"])
def test_wrong_source_or_runtime_pin_cannot_create_run(tmp_path, pin):
    params = _case(tmp_path / "inputs")
    params[pin] = "0" * 64
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="differs"):
        action.run(**params, output_folder=output)
    assert not output.exists()


def test_second_shard_failure_preserves_first_and_latches_run(tmp_path, monkeypatch):
    params = _case(tmp_path / "inputs")
    _, adapter = action._load_bora(BORA_SOURCE)
    original = adapter.produce_continuous_probability_shard
    calls = 0

    def stop_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled failure before second shard")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "produce_continuous_probability_shard", stop_second)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="controlled failure"):
        action.run(**params, output_folder=output)
    failure = json.loads((output / "RUN_FAILED.json").read_text())
    assert failure["complete"] is False
    assert failure["failed_stage"] == "shard_0001"
    assert failure["completed_shards"] == 1
    assert failure["retry_count"] == 0
    assert (output / "shards/0000/SHARD_COMPLETE.json").is_file()
    assert not (output / "RUN_COMPLETE.json").exists()
    assert not (output / "experiment.cellphenotyper.json").exists()
    with pytest.raises(FileExistsError):
        action.run(**params, output_folder=output)


def test_core_grid_rejects_excess_shards_and_padded_window():
    with pytest.raises(ValueError, match="4096 shards"):
        action.core_grid((1000, 1000), 10, 1, 2)
    with pytest.raises(ValueError, match="bounded"):
        action.core_grid((10000, 10000), 3073, 512, 2)
    assert len(action.core_grid((10000, 10000), 3072, 512, 2)) == 16


def test_nextflow_entrypoint_is_local_cpu_only_and_no_retry():
    source = (Path(__file__).resolve().parents[1] /
              "bora_continuous_fullslide.nf").read_text()
    assert "executor 'local'" in source
    assert "label 'bora_continuous_cpu'" in source
    assert "maxRetries 0" in source
    assert "errorStrategy 'terminate'" in source
    assert "export CUDA_VISIBLE_DEVICES=-1" in source
    assert "--compute_device cpu" in source
    assert '"complete":false' in source
    assert "gpu_capable" not in source
    assert not any(command in source for command in
                   ("sbatch", "qsub", "srun", "apptainer exec", "--nv"))
