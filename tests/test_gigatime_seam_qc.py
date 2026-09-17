import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "bin" / "gigatime_seam_qc.py"
SPEC = importlib.util.spec_from_file_location("gigatime_seam_qc", MODULE_PATH)
seam_qc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = seam_qc
SPEC.loader.exec_module(seam_qc)


def test_smooth_prediction_passes_block_boundary_qc():
    axis = np.linspace(0.1, 0.8, 64, dtype=np.float32)
    smooth = np.stack(
        [np.tile(axis, (64, 1)), np.tile(axis[:, None], (1, 64))],
        axis=0,
    )

    report = seam_qc.compute_seam_metrics(
        smooth,
        block_size=16,
        channel_names=["x-gradient", "y-gradient"],
        max_p95_excess=0.10,
        min_affected_fraction=0.05,
    )

    assert report["status"] == "pass"
    assert report["failed_channel_boundaries"] == 0


def test_rectangular_zero_fill_seam_fails_and_is_diagnosed():
    prediction = np.full((1, 64, 64), 0.4, dtype=np.float32)
    prediction[:, :16, :] = 0.0

    report = seam_qc.compute_seam_metrics(
        prediction,
        block_size=16,
        channel_names=["DAPI"],
        max_p95_excess=0.10,
        min_affected_fraction=0.05,
    )

    assert report["status"] == "fail"
    failed = [record for record in report["records"] if record["failed"]]
    assert failed[0]["orientation"] == "horizontal"
    assert failed[0]["boundary_px"] == 16
    assert failed[0]["zero_transition_fraction"] == 1.0


@pytest.mark.skipif(seam_qc.zarr is None, reason="zarr is supplied by the pipeline runtime")
def test_heatmap_and_json_are_written_for_zarr(tmp_path):
    prediction_path = tmp_path / "prediction.zarr"
    root = seam_qc.zarr.open(str(prediction_path), mode="w")
    creator = getattr(root, "create_array", None) or root.create_dataset
    array = creator(
        "0",
        shape=(1, 32, 32),
        dtype=np.uint8,
    )
    array[:] = np.zeros((1, 32, 32), dtype=np.uint8)

    report = seam_qc.assess_prediction_seams(
        prediction_path,
        outdir=tmp_path,
        block_size=16,
        channel_names=["DAPI"],
        max_p95_excess=0.10,
        min_affected_fraction=0.05,
        mode="warn",
    )

    assert report["status"] == "pass"
    assert (tmp_path / "gigatime_seam_qc.json").is_file()
    assert (tmp_path / "gigatime_seam_qc.png").is_file()
