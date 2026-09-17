import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "bin"))
SPEC = importlib.util.spec_from_file_location(
    "annealed_wand_boundary", ROOT / "bin" / "annealed_wand_boundary.py"
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_annealed_wand_moves_boundary_toward_image_interface() -> None:
    image = np.zeros((72, 72, 3), dtype=np.uint8)
    image[:, :36] = np.array([165, 75, 105], dtype=np.uint8)
    image[:, 36:] = np.array([225, 190, 200], dtype=np.uint8)
    labels = np.ones((72, 72), dtype=np.uint16)
    labels[:, 27:] = 2
    tissue = np.ones(labels.shape, dtype=bool)
    protected = np.zeros_like(labels)
    protected[:, :12] = 1
    protected[:, 55:] = 2

    result, metadata = module.annealed_wand_boundary_competition(
        image,
        labels,
        tissue,
        protected_labels=protected,
        boundary_radius=16,
        iterations=16,
        initial_temperature=2.0,
        final_temperature=0.03,
    )

    transition = int(np.argmax(result[36] == 2))
    assert 34 <= transition <= 38
    assert metadata["changed_pixels"] > 0
    assert metadata["accepted_nonincreasing_energy"] is True
    assert metadata["neighbour_connectivity"] == 8
    assert np.isclose(metadata["diagonal_weight"], 1.0 / np.sqrt(2.0))
    assert metadata["final_energy"] <= metadata["initial_energy"]
    np.testing.assert_array_equal(result[:, :12], labels[:, :12])
    np.testing.assert_array_equal(result[:, 55:], labels[:, 55:])


def test_competition_preserves_foreground_grandqc_and_is_deterministic() -> None:
    image = np.full((48, 52, 3), 140, dtype=np.uint8)
    image[:, 28:] = 205
    labels = np.ones((48, 52), dtype=np.uint16)
    labels[:, 23:] = 2
    labels[:5] = 0
    tissue = np.ones(labels.shape, dtype=bool)
    tissue[:5] = False
    tissue[18:25, 17:21] = False
    labels[~tissue] = 0
    kwargs = dict(boundary_radius=12, iterations=10, initial_temperature=1.5, final_temperature=0.05)

    first, first_meta = module.annealed_wand_boundary_competition(image, labels, tissue, **kwargs)
    second, second_meta = module.annealed_wand_boundary_competition(image, labels, tissue, **kwargs)

    np.testing.assert_array_equal(first, second)
    assert first_meta["final_energy"] == second_meta["final_energy"]
    np.testing.assert_array_equal(first > 0, labels > 0)
    assert not np.any(first[~tissue])
    assert set(np.unique(first)) <= {0, 1, 2}


def test_single_label_abstains() -> None:
    labels = np.ones((24, 24), dtype=np.uint16)
    result, metadata = module.annealed_wand_boundary_competition(
        np.full((24, 24, 3), 127, dtype=np.uint8), labels, np.ones_like(labels, dtype=bool)
    )
    np.testing.assert_array_equal(result, labels)
    assert metadata["applied"] is False
    assert metadata["changed_pixels"] == 0


def test_grandqc_support_normalizes_diagnostic_sampling_edge() -> None:
    image = np.full((32, 36, 3), 150, dtype=np.uint8)
    image[:, 18:] = 205
    labels = np.ones((32, 36), dtype=np.uint16)
    labels[:, 17:] = 2
    tissue = np.ones(labels.shape, dtype=bool)
    tissue[0, :] = False
    tissue[:, -1] = False

    result, metadata = module.annealed_wand_boundary_competition(
        image,
        labels,
        tissue,
        boundary_radius=8,
        iterations=8,
        initial_temperature=1.5,
        final_temperature=0.05,
    )

    assert not np.any(result[~tissue])
    np.testing.assert_array_equal(result[tissue] > 0, labels[tissue] > 0)
    assert metadata["accepted_nonincreasing_energy"] is True
