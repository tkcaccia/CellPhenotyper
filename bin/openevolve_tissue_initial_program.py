from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from skimage import morphology

BIN_DIR = Path(__file__).resolve().parent
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from grow_to_tissue_core import DEFAULT_CONFIG, cleanup_mask, keep_seeded_components
from tissue_mask_benchmark_lib import similarity_region_expand


# EVOLVE-BLOCK-START
"""Expand the cell mask into a tissue mask using only classical image cues."""


def run_refinement(image: np.ndarray, seed_labels: np.ndarray, base_labels: np.ndarray) -> np.ndarray:
    seed_labels = np.asarray(seed_labels)
    seed_binary = seed_labels > 0
    base_mask = np.asarray(base_labels) > 0 if base_labels is not None else seed_binary.copy()

    cfg = replace(
        DEFAULT_CONFIG,
        boundary_band=40,
        final_open_radius=1,
        final_close_radius=4,
        final_hole_area=8000,
        final_min_obj_area=1000,
    )

    grown = similarity_region_expand(image, seed_labels, base_mask, cfg)
    grown |= morphology.binary_dilation(seed_binary, morphology.disk(3))
    grown[seed_binary] = True
    grown = keep_seeded_components(grown, seed_binary)
    return cleanup_mask(grown, cfg, keep_largest=True)


# EVOLVE-BLOCK-END
