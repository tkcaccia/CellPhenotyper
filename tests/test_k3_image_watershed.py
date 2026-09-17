import importlib.util
import sys
from pathlib import Path

import numpy as np


AUDIT = Path(__file__).resolve().parents[1] / "audits" / "uni2_resolution_20260909"
sys.path.insert(0, str(AUDIT))
SCRIPT = AUDIT / "explore_k3_image_watershed.py"
SPEC = importlib.util.spec_from_file_location("explore_k3_image_watershed", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_watershed_keeps_support_and_all_component_markers():
    labels = np.ones((31, 37), dtype=np.int16)
    labels[:, 19:] = 2
    labels[4:7, 4:7] = 3
    labels[:2] = 0
    gradient = np.ones(labels.shape, dtype=np.float32)
    gradient[:, 17] = 0
    result = MODULE.image_watershed_refine(
        labels, gradient, boundary_radius=6, compactness=0,
    )
    np.testing.assert_array_equal(result > 0, labels > 0)
    assert set(np.unique(result)) == {0, 1, 2, 3}


def test_multichannel_gradient_is_finite_and_bounded():
    image = np.zeros((24, 27, 3), dtype=np.uint8)
    image[:, :13] = (80, 40, 100)
    image[:, 13:] = (220, 175, 190)
    tissue = np.ones(image.shape[:2], dtype=bool)
    for space in ("luminance", "lab", "od", "lab_od"):
        gradient = MODULE.robust_multichannel_gradient(
            image, tissue, space=space, sigma=1.0,
        )
        assert np.isfinite(gradient).all()
        assert gradient.min() >= 0 and gradient.max() <= 1
        assert gradient[:, 11:15].mean() > gradient[:, :5].mean()
