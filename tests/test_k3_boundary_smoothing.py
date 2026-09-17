import importlib.util
import sys
from pathlib import Path

import numpy as np


AUDIT = Path(__file__).resolve().parents[1] / "audits" / "uni2_resolution_20260909"
sys.path.insert(0, str(AUDIT))
SCRIPT = AUDIT / "explore_k3_boundary_smoothing.py"
SPEC = importlib.util.spec_from_file_location("explore_k3_boundary_smoothing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_gaussian_diffusion_rounds_one_pixel_spur_without_changing_support():
    labels = np.ones((21, 21), dtype=np.int16)
    labels[:, 11:] = 2
    labels[10, 10] = 2
    result = MODULE.gaussian_label_diffusion(labels, sigma=1.5, boundary_radius=4)
    np.testing.assert_array_equal(result > 0, labels > 0)
    assert result[10, 10] == 1
    assert set(np.unique(result)) == {1, 2}


def test_majority_is_label_permutation_invariant():
    labels = np.ones((15, 17), dtype=np.int16)
    labels[:, 9:] = 2
    labels[7, 8] = 2
    first = MODULE.neighbour_majority(labels, boundary_radius=3, iterations=2, minimum_fraction=.625)
    permuted = np.where(labels == 1, 8, 3)
    second = MODULE.neighbour_majority(permuted, boundary_radius=3, iterations=2, minimum_fraction=.625)
    restored = np.where(second == 8, 1, 2)
    np.testing.assert_array_equal(first, restored)
