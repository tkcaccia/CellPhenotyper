#!/usr/bin/env python3
"""Automatic H&E-appearance correction for coherent tissue-domain errors.

This module never consumes expert annotations. It learns appearance groups from
the image and assigns them using conservative cores of the incoming labels.
Only large connected disagreements are accepted, preserving local MedSAM and
watershed boundaries elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = x @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4.0 / 29.0)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def _smooth(features: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack(
        [ndi.gaussian_filter(features[..., channel], sigma=sigma) for channel in range(features.shape[-1])],
        axis=-1,
    )


def multiscale_features(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    lab = _smooth(rgb_to_lab(rgb), 1.25)
    od = _smooth(-np.log((rgb.astype(np.float32) + 1.0) / 256.0), 1.25)
    channels = [lab, od]
    for sigma in (2.0, 6.0, 14.0):
        channels.append(_smooth(lab, sigma))
    luminance = lab[..., 0]
    channels.append(np.hypot(ndi.sobel(luminance, axis=0), ndi.sobel(luminance, axis=1))[..., None])
    return np.concatenate(channels, axis=-1).astype(np.float32)


def refine_tissue_domains_by_appearance(
    image: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    *,
    clusters: int = 24,
    core_erosion_px: int = 4,
    smooth_sigma: float = 3.0,
    min_region_area_px: int = 10_000,
    sample_pixels: int = 250_000,
    random_seed: int = 1,
    allow_multiclass: bool = False,
    editable_mask: np.ndarray | None = None,
    protected_mask: np.ndarray | None = None,
    core_support: np.ndarray | None = None,
    min_vote_fraction: float = 0.0,
    min_vote_margin: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Correct coherent errors using supported appearance votes, retaining ties.

    Multiclass mode is opt-in. Explicit editable/protected masks can constrain
    correction and core_support excludes uncertain labels from the vote source.
    Vote fractions and margins are descriptive, not calibrated probabilities.
    """
    source = np.asarray(labels)
    support = np.asarray(tissue, dtype=bool)
    if source.shape != support.shape or np.asarray(image).shape[:2] != source.shape:
        raise ValueError("Appearance image, labels and support shapes must match")
    active_labels = [int(value) for value in np.unique(source[support]) if value > 0]
    metadata: dict[str, Any] = {
        "enabled": True,
        "method": "multiscale_lab_od_kmeans_large_region_hybrid",
        "clusters": int(clusters),
        "core_erosion_px": int(core_erosion_px),
        "smooth_sigma": float(smooth_sigma),
        "min_region_area_px": int(min_region_area_px),
        "random_seed": int(random_seed),
        "input_labels": active_labels,
    }
    if (len(active_labels) != 2 and not allow_multiclass) or len(active_labels) < 2 or not support.any():
        metadata.update({"applied": False, "reason": "requires_exactly_two_nonzero_tissue_labels", "changed_pixels": 0})
        return source.copy(), metadata

    from sklearn.cluster import MiniBatchKMeans

    editable = support.copy() if editable_mask is None else np.asarray(editable_mask, bool) & support
    protected = np.zeros(source.shape, bool) if protected_mask is None else np.asarray(protected_mask, bool)
    trusted = support if core_support is None else np.asarray(core_support, bool) & support
    if any(mask.shape != source.shape for mask in (editable, protected, trusted)):
        raise ValueError("Appearance constraint shapes must match labels")
    if not 0 <= min_vote_fraction <= 1 or not 0 <= min_vote_margin <= 1:
        raise ValueError("Appearance vote thresholds must be in [0,1]")
    editable &= ~protected
    iterations = max(1, int(core_erosion_px))
    cores = {label: ndi.binary_erosion((source == label) & support, iterations=iterations) & trusted for label in active_labels}
    supported_labels = [label for label in active_labels if np.count_nonzero(cores[label]) >= 64]
    editable &= ~np.isin(source, [label for label in active_labels if label not in supported_labels])
    if allow_multiclass and len(active_labels) > 2:
        min_vote_fraction = max(.6, min_vote_fraction)
        min_vote_margin = max(.15, min_vote_margin)
    if len(supported_labels) < 2:
        metadata.update({"applied": False, "reason": "insufficient_conservative_core_pixels", "changed_pixels": 0})
        return source.copy(), metadata

    features = multiscale_features(image)
    matrix = features.reshape(-1, features.shape[-1])
    tissue_indices = np.flatnonzero(support)
    rng = np.random.default_rng(random_seed)
    chosen = rng.choice(tissue_indices, size=min(int(sample_pixels), len(tissue_indices)), replace=False)
    sample = matrix[chosen]
    center = np.median(sample, axis=0)
    q25, q75 = np.percentile(sample, [25, 75], axis=0)
    # A mostly-zero edge channel can have zero IQR while a few tissue edges
    # are very large. Do not let that channel dominate all color features.
    scale = np.maximum(np.maximum(q75 - q25, sample.std(axis=0) * .1), 1e-3)
    def normalize(values):
        return np.clip((values - center) / scale, -10.0, 10.0)
    model = MiniBatchKMeans(
        n_clusters=min(len(sample), max(2, int(clusters))), batch_size=8192, n_init=5,
        max_iter=200, random_state=int(random_seed),
    )
    model.fit(normalize(sample))
    appearance = np.full(source.shape, -1, dtype=np.int16)
    for start in range(0, len(tissue_indices), 250_000):
        indices = tissue_indices[start:start + 250_000]
        appearance.ravel()[indices] = model.predict(normalize(matrix[indices]))

    proposed = source.copy()
    confident_proposal = np.zeros(source.shape, bool)
    votes_metadata = []
    for cluster in range(model.n_clusters):
        region = appearance == cluster
        votes = np.array([np.count_nonzero(region & cores[label]) for label in supported_labels])
        ranked = np.argsort(votes, kind="stable")[::-1]
        total = int(votes.sum())
        top = int(votes[ranked[0]])
        runner_up = int(votes[ranked[1]])
        fraction = top / max(1, total)
        margin = (top - runner_up) / max(1, total)
        accepted = total > 0 and top > runner_up and fraction >= min_vote_fraction and margin >= min_vote_margin
        if accepted:
            proposed[region & support] = supported_labels[ranked[0]]
            confident_proposal[region & support] = True
        votes_metadata.append({"appearance_cluster": cluster, "core_votes": dict(zip(map(str, supported_labels), map(int, votes))), "fraction": fraction, "margin": margin, "accepted": bool(accepted)})

    if smooth_sigma > 0:
        best = np.full(source.shape, -np.inf, np.float32)
        second_best = np.full(source.shape, -np.inf, np.float32)
        smoothed = proposed.copy()
        denominator = ndi.gaussian_filter(support.astype(np.float32), sigma=float(smooth_sigma))
        for label in supported_labels:
            score = ndi.gaussian_filter((proposed == label).astype(np.float32), sigma=float(smooth_sigma)) / np.maximum(denominator, 1e-6)
            wins = score > best
            second_best = np.where(wins, best, np.maximum(second_best, score))
            best = np.maximum(best, score)
            smoothed[wins & support] = label
        confident_proposal &= (best - second_best) > 1e-6
        proposed[confident_proposal] = smoothed[confident_proposal]

    result = source.copy()
    accepted_regions = 0
    accepted_pixels = 0
    structure = np.ones((3, 3), dtype=np.uint8)
    for target in active_labels:
        disagreement = (proposed == target) & (source != target) & editable & confident_proposal
        disagreement = ndi.binary_closing(disagreement, iterations=2)
        disagreement &= editable & support & (proposed == target) & confident_proposal
        components, count = ndi.label(disagreement, structure=structure)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())
        accepted = np.flatnonzero(sizes >= max(1, int(min_region_area_px)))
        accepted = accepted[accepted > 0]
        mask = np.isin(components, accepted)
        result[mask] = target
        accepted_regions += int(len(accepted))
        accepted_pixels += int(np.count_nonzero(mask))

    result[~support] = 0
    result[protected] = source[protected]
    metadata.update({
        "applied": True,
        "accepted_regions": accepted_regions,
        "accepted_pixels": accepted_pixels,
        "changed_pixels": int(np.count_nonzero(result != source)),
        "feature_count": int(features.shape[-1]),
        "sample_pixels": int(len(chosen)),
        "multiclass": bool(allow_multiclass),
        "supported_labels": supported_labels,
        "unsupported_labels": [label for label in active_labels if label not in supported_labels],
        "appearance_votes": votes_metadata,
        "protected_pixels": int(protected.sum()),
        "editable_pixels": int(editable.sum()),
        "confidence_is_calibrated_probability": False,
        "feature_scaling": "median/IQR with 0.1*SD floor, absolute normalized value capped at 10",
    })
    return result.astype(source.dtype, copy=False), metadata
