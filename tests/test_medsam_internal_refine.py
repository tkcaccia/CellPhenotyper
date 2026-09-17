import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile


pytest.importorskip("skimage")
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SPEC = importlib.util.spec_from_file_location("refine_grown_tissue_medsam", ROOT / "bin" / "refine_grown_tissue_medsam.py")
refine = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(refine)

import medsam_border_refine as medsam_core


def test_internal_boundary_snaps_to_native_image_edge() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, 36:, :] = 255
    labels = np.ones((80, 80), dtype=np.uint16)
    labels[:, 30:] = 2
    result, metadata = refine.refine_internal_cluster_boundaries(image, labels, radius=12)
    transition = np.argmax(result[40] == 2)
    assert 34 <= transition <= 38
    assert metadata["changed_pixels"] > 0
    np.testing.assert_array_equal(result[:, :17], labels[:, :17])
    np.testing.assert_array_equal(result[:, 43:], labels[:, 43:])


@pytest.mark.parametrize("space", ["luminance", "lab", "od", "lab_od"])
def test_internal_boundary_gradient_modes_preserve_support(space: str) -> None:
    image = np.zeros((64, 72, 3), dtype=np.uint8)
    image[:, :38] = (140, 65, 100)
    image[:, 38:] = (215, 180, 190)
    labels = np.ones((64, 72), dtype=np.uint16)
    labels[:, 30:] = 2
    labels[:4] = 0
    result, metadata = refine.refine_internal_cluster_boundaries(
        image, labels, radius=12, gradient_space=space,
        gradient_sigma_px=2.0, watershed_compactness=0.001,
    )
    np.testing.assert_array_equal(result > 0, labels > 0)
    assert metadata["gradient_space"] == space
    assert metadata["gradient_sigma_px"] == 2.0
    assert metadata["watershed_compactness"] == 0.001


def test_single_cluster_tile_is_unchanged() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    labels = np.ones((40, 40), dtype=np.uint16)
    result, metadata = refine.refine_internal_cluster_boundaries(image, labels, radius=12)
    np.testing.assert_array_equal(result, labels)
    assert metadata["changed_pixels"] == 0


def test_pre_medsam_competition_wrapper_preserves_protected_cores_and_support() -> None:
    image = np.zeros((80, 84, 3), dtype=np.uint8)
    image[:, :42] = (150, 65, 100)
    image[:, 42:] = (225, 185, 200)
    labels = np.ones((80, 84), dtype=np.uint16)
    labels[:, 32:] = 2
    tissue = np.ones(labels.shape, dtype=bool)
    tissue[:6] = False
    labels[~tissue] = 0
    protected = np.zeros_like(labels)
    protected[10:, :15] = 1
    protected[10:, 68:] = 2
    args = SimpleNamespace(
        pre_boundary_competition=True,
        pre_boundary_downsample=4,
        pre_boundary_radius=48,
        pre_boundary_iterations=16,
        pre_boundary_initial_temperature=2.0,
        pre_boundary_final_temperature=0.05,
        pre_boundary_data_weight=1.0,
        pre_boundary_smoothness_weight=0.3,
        pre_boundary_edge_beta=0.7,
    )

    result, metadata = refine.pre_medsam_boundary_competition(
        image, labels, tissue, args, protected_labels=protected
    )

    assert metadata["enabled"] is True
    assert metadata["foreground_footprint_invariant"] is True
    np.testing.assert_array_equal(result > 0, labels > 0)
    np.testing.assert_array_equal(result[protected > 0], protected[protected > 0])
    assert not np.any(result[~tissue])


def test_scaled_tissue_reader_has_identical_global_mapping_across_tile_seams(tmp_path: Path) -> None:
    source = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    mask_path = tmp_path / "tissue.tif"
    tifffile.imwrite(mask_path, source)
    reader = refine.ScaledBinaryMaskReader(str(mask_path), (4, 4))
    try:
        left = reader.read(0, 4, 0, 2)
        right = reader.read(0, 4, 2, 4)
    finally:
        reader.close()
    observed = np.concatenate([left, right], axis=1)
    expected = np.repeat(np.repeat(source.astype(bool), 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(observed, expected)


def test_scaled_tissue_reader_preview_preserves_empty_holes(tmp_path: Path) -> None:
    source = np.ones((8, 8), dtype=np.uint8)
    source[2:6, 3:5] = 0
    mask_path = tmp_path / "tissue.tif"
    tifffile.imwrite(mask_path, source)
    reader = refine.ScaledBinaryMaskReader(str(mask_path), (32, 32))
    try:
        preview = reader.read_stride(4)
    finally:
        reader.close()
    np.testing.assert_array_equal(preview, source.astype(bool))


def test_grandqc_empty_area_is_hard_negative_inside_medsam(monkeypatch) -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    seed = np.zeros((48, 48), dtype=np.uint16)
    seed[20:24, 6:10] = 1
    baseline_labels = np.ones((48, 48), dtype=np.uint16)
    support = np.ones((48, 48), dtype=bool)
    support[14:34, 18:30] = False

    def predict_everywhere(crop_rgb, _box, _config):
        shape = crop_rgb.shape[:2]
        return np.ones(shape, dtype=bool), np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(medsam_core, "_infer_box_prompt", predict_everywhere)
    config = medsam_core.MedSAMConfig(
        device="cpu",
        component_min_area=1,
        component_merge_distance=0,
        seed_dilation_radius=1,
        core_erosion_radius=2,
        outer_dilation_radius=8,
        min_object_size=1,
        smooth_radius=0,
        save_debug=True,
        cluster_tile_size=64,
        cluster_tile_overlap=0,
    )
    final_mask, _, _, metadata, artifacts = medsam_core.run_medsam_border_refine(
        image=image,
        seed_labels=seed,
        baseline_tissue_mask=baseline_labels > 0,
        baseline_label_map=baseline_labels,
        allowed_support_mask=support,
        config=config,
    )

    assert not np.any(final_mask[~support])
    assert not np.any(np.asarray(artifacts["label_map"])[~support])
    assert not np.any(np.asarray(artifacts["raw_medsam_label_map"])[~support])
    assert not np.any(np.asarray(artifacts["editable_band"])[~support])
    assert metadata["allowed_support_excluded_baseline_pixels"] == int((~support).sum())
    assert metadata["final_outside_allowed_support_pixels"] == 0
