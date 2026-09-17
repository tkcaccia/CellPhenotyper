import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))

from tissue_appearance_refine import refine_tissue_domains_by_appearance


def test_appearance_refine_corrects_large_coherent_misassignment() -> None:
    height = width = 160
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :80] = (80, 40, 120)
    image[:, 80:] = (220, 160, 180)
    labels = np.ones((height, width), dtype=np.uint16)
    labels[:, 80:] = 2
    labels[10:70, 10:70] = 2  # deliberately wrong coherent interior region
    tissue = np.ones((height, width), dtype=bool)

    result, metadata = refine_tissue_domains_by_appearance(
        image, labels, tissue,
        clusters=24,
        core_erosion_px=3,
        smooth_sigma=1,
        min_region_area_px=500,
        sample_pixels=10_000,
        random_seed=1,
    )

    assert np.mean(result[:, :80] == 1) > 0.99
    assert np.mean(result[:, 80:] == 2) > 0.99
    assert metadata["applied"] is True
    assert metadata["changed_pixels"] >= 3_500


def test_appearance_refine_preserves_background() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :40] = (80, 40, 120)
    image[:, 40:] = (220, 160, 180)
    labels = np.ones((80, 80), dtype=np.uint8)
    labels[:, 40:] = 2
    tissue = np.ones((80, 80), dtype=bool)
    tissue[:10, :] = False
    labels[~tissue] = 0

    result, _ = refine_tissue_domains_by_appearance(
        image, labels, tissue,
        clusters=2,
        core_erosion_px=2,
        smooth_sigma=1,
        min_region_area_px=50,
        sample_pixels=4_000,
    )

    assert not np.any(result[~tissue])


def test_appearance_refine_abstains_for_more_than_two_domains() -> None:
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    labels = np.ones((60, 60), dtype=np.uint8)
    labels[:, 20:40] = 2
    labels[:, 40:] = 3
    tissue = np.ones((60, 60), dtype=bool)

    result, metadata = refine_tissue_domains_by_appearance(image, labels, tissue)

    np.testing.assert_array_equal(result, labels)
    assert metadata["applied"] is False
    assert metadata["reason"] == "requires_exactly_two_nonzero_tissue_labels"


def test_appearance_refinement_is_enabled_and_wired_to_medsam() -> None:
    config = (ROOT / "nextflow.config").read_text(encoding="utf-8")
    parameters = (ROOT / "pipeline_paramers.yml").read_text(encoding="utf-8")
    module = (ROOT / "modules" / "refine_grown_tissue_medsam.nf").read_text(encoding="utf-8")
    assert "medsam_appearance_refine         = true" in config
    assert "medsam_appearance_refine: true" in parameters
    assert "--appearance-refine" in module
    assert "tissue_appearance_refine.py" in module
