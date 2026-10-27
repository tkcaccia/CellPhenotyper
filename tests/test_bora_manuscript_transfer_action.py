"""Safety gates for the separate review-only manuscript transfer route."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

from bora_manuscript_transfer_action import _core_grid, _materialize_native_support, run
from bora_fullslide_action import sha256
import export_bora_manuscript_transfer_project as exporter
from export_bora_manuscript_transfer_project import verify_result


def test_pm016_core_plan_is_bounded_and_covers_native_slide_once():
    shape = (14762, 23904)
    cores = _core_grid(shape, 2048, (64, 64), 3)
    assert len(cores) == 96
    assert sum(height * width for _, _, height, width in cores) == shape[0] * shape[1]
    assert cores[0] == (0, 0, 2048, 2048)
    assert cores[-1] == (14336, 22528, 426, 1376)
    with pytest.raises(ValueError, match="20M-pixel"):
        _core_grid((8000, 8000), 3072, (1000, 1000), 3)


def test_action_requires_explicit_exploratory_opt_in_before_any_inputs(tmp_path):
    with pytest.raises(ValueError, match="acknowledge-exploratory"):
        run(model_path=tmp_path / "missing-model", policy_path=tmp_path / "missing-policy",
            materialization_folder=tmp_path / "missing-materialization",
            registration_path=tmp_path / "missing-registration",
            registered_support_tiff=tmp_path / "missing-support",
            base_project=tmp_path / "missing-project", bora_source=tmp_path / "missing-bora",
            expected_bora_sha256="0" * 64, expected_runtime_sha256="0" * 64,
            output_folder=tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_cpu_action_refuses_visible_gpu_before_source_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="refuses an allocated GPU"):
        run(model_path=tmp_path / "missing-model", policy_path=tmp_path / "missing-policy",
            materialization_folder=tmp_path / "missing-materialization",
            registration_path=tmp_path / "missing-registration",
            registered_support_tiff=tmp_path / "missing-support",
            base_project=tmp_path / "missing-project", bora_source=tmp_path / "missing-bora",
            expected_bora_sha256="0" * 64, expected_runtime_sha256="0" * 64,
            output_folder=tmp_path / "result", acknowledge_exploratory=True)
    assert not (tmp_path / "result").exists()


def test_exporter_refuses_old_hard_group_chain_as_transfer(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    # A legacy stair-step result cannot be accepted by the new review route,
    # even if a caller names its directory as the requested result.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "CHAIN_COMPLETE.json").write_text('{"format":"bora-global-hard-group-v1"}')
    support = tmp_path / "support.tif"
    with rasterio.open(support, "w", driver="GTiff", width=8, height=8,
                       count=1, dtype="uint8") as dataset:
        dataset.write(np.ones((8, 8), np.uint8), 1)
    np.save(legacy / "support.npy", np.ones((8, 8), np.uint8))
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({
        "format": "cellphenotyper-dual-mpp-registration-v1", "verified": True,
        "support_sampling": "absolute pixel centre, nearest, full-frame GrandQC mask",
        "cell_full_image": {"sha256": "a" * 64, "frame_id": "native",
                            "height": 8, "width": 8, "mpp_x": .25, "mpp_y": .25},
        "full_support_mask": {"sha256": sha256(support), "height": 8, "width": 8},
    }))
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "MATERIALIZATION_COMPLETE.json").write_text(json.dumps({
        "format": "cellphenotyper-bora-semantic-token-materialization-v1",
        "complete": True, "registration_sha256": sha256(registration),
        "native_image_sha256": "a" * 64,
    }))
    with pytest.raises(ValueError, match="completion receipt"):
        verify_result(registration, materialized, support, legacy)


def test_coarse_grandqc_mask_maps_by_absolute_native_centres(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    coarse = np.array([[0, 255, 0, 0, 255],
                       [255, 0, 255, 0, 0],
                       [0, 0, 255, 255, 0]], dtype=np.uint8)
    source = tmp_path / "coarse-support.tif"
    with rasterio.open(source, "w", driver="GTiff", width=5, height=3,
                       count=1, dtype="uint8") as dataset:
        dataset.write(coarse, 1)
    registration = {
        "cell_full_image": {"height": 11, "width": 17},
        "full_support_mask": {"height": 3, "width": 5, "sha256": sha256(source)},
        "support_sampling": "absolute pixel centre, nearest, full-frame GrandQC mask",
    }
    target = tmp_path / "native-support.npy"
    count = _materialize_native_support(source, target, registration,
                                        memory_budget_bytes=64*1024**2)
    sy = ((2*np.arange(11)+1)*3)//(2*11)
    sx = ((2*np.arange(17)+1)*5)//(2*17)
    expected = (coarse[np.ix_(sy, sx)] > 0).astype(np.uint8)
    actual = np.load(target, allow_pickle=False)
    assert np.array_equal(actual, expected)
    assert count == np.count_nonzero(expected)
    assert exporter._verify_registered_support(registration, source, target) == sha256(target)
    np.save(target, np.roll(actual, 1, axis=1))
    with pytest.raises(ValueError, match="absolute-centre sampling"):
        exporter._verify_registered_support(registration, source, target)


def test_export_adds_only_locked_exploratory_review(tmp_path, monkeypatch):
    base = tmp_path / "base.cellphenotyper.json"
    base.write_text("original project unchanged")
    registration = tmp_path / "registration.json"
    registration.write_text("registered")
    support = tmp_path / "support.tif"
    support.write_bytes(b"supported")
    result = tmp_path / "result"
    result.mkdir()
    destination = tmp_path / "new.cellphenotyper.json"
    native = {"sha256": "a" * 64, "frame_id": "native",
              "height": 64, "width": 64, "mpp_x": .25, "mpp_y": .25}
    chain = {"token": {"model_sha256": "b" * 64,
                       "policy_sha256": "c" * 64,
                       "materialization_receipt_sha256": "d" * 64},
             "assembly": {"native_shape_yx": [64, 64], "native_mpp_xy": [.25, .25],
                          "registered_support_shape_yx": [8, 8],
                          "label_sha256": "e" * 64, "unknown_sha256": "f" * 64},
             "vector": {"output_sha256": {"review/mask.geojson": "1" * 64},
                        "selected_smoothing_tolerance_um": 1.5},
             "support_npy_sha256": "2" * 64,
             "vector_receipt_sha256": "3" * 64}
    tissue = {"tissue_id": "pm016", "image": native.copy(), "artifacts": []}
    monkeypatch.setattr(exporter, "verify_result", lambda *args: (native, chain))
    monkeypatch.setattr(exporter, "validate_experiment",
                        lambda path: {"tissues": [tissue]})

    def fake_export(project, path, *, base_dir):
        path.write_text(json.dumps(project))
        return project

    monkeypatch.setattr(exporter, "export_experiment", fake_export)
    receipt = exporter.export(base, registration, tmp_path, support, result, destination)
    assert receipt["review_only"] is True
    assert base.read_text() == "original project unchanged"
    artifact = json.loads(destination.read_text())["tissues"][0]["artifacts"][0]
    assert artifact["role"] == "bora_exploratory_boundary_review"
    assert artifact["review_only"] is True
    assert artifact["provenance"]["biological_accuracy_qualified"] is False
    assert artifact["provenance"]["unknown_semantics"] == \
        "abstention_not_background_or_tissue_class"
    assert not any(item["role"] == "tissue_annotations" for item in tissue["artifacts"])
