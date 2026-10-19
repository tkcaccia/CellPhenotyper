import importlib.util
import sys
from pathlib import Path

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


def test_single_cluster_tile_is_unchanged() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    labels = np.ones((40, 40), dtype=np.uint16)
    result, metadata = refine.refine_internal_cluster_boundaries(image, labels, radius=12)
    np.testing.assert_array_equal(result, labels)
    assert metadata["changed_pixels"] == 0


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
