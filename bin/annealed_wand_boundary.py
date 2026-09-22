#!/usr/bin/env python3
"""Deterministic image-aware competition between neighbouring tissue labels.

The routine implements a bounded multiclass Potts model with deterministic
mean-field annealing.  Only labels present on a pixel's local frontier may
compete, which gives the update its wand-like region-growing behaviour.
Distance-weighted diagonal neighbours can be included to reduce grid-aligned
staircase boundaries.  Protected cores, the foreground footprint, and pixels
outside the declared editable boundary band are invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from tissue_appearance_refine import rgb_to_lab


def _internal_boundary(labels: np.ndarray) -> np.ndarray:
    source = np.asarray(labels)
    foreground = source > 0
    boundary = np.zeros(source.shape, dtype=bool)
    vertical = (source[:-1] != source[1:]) & foreground[:-1] & foreground[1:]
    horizontal = (source[:, :-1] != source[:, 1:]) & foreground[:, :-1] & foreground[:, 1:]
    boundary[:-1] |= vertical
    boundary[1:] |= vertical
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    return boundary


def _raw_features(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    od = -np.log((rgb.astype(np.float32) + 1.0) / 256.0)
    return np.concatenate([lab, od], axis=-1).astype(np.float32)


def _features(
    image: np.ndarray,
    support: np.ndarray,
    calibration: dict[str, Any] | None = None,
) -> np.ndarray:
    values = _raw_features(image)
    selected = values[np.asarray(support, dtype=bool)]
    if selected.size == 0:
        raise ValueError("Annealed boundary competition requires foreground image pixels")
    if calibration is None:
        center = np.median(selected, axis=0)
        q25, q75 = np.percentile(selected, [25, 75], axis=0)
        scale = np.maximum(q75 - q25, 0.1 * np.std(selected, axis=0))
    else:
        center = np.asarray(calibration["feature_center"], dtype=np.float32)
        scale = np.asarray(calibration["feature_scale"], dtype=np.float32)
        if center.shape != (values.shape[-1],) or scale.shape != center.shape:
            raise ValueError("Boundary calibration has incompatible feature dimensions")
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.clip((values - center) / scale, -10.0, 10.0).astype(np.float32)


def fit_appearance_calibration(
    image: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    *,
    trusted_mask: np.ndarray | None = None,
    max_pixels: int = 250_000,
    random_seed: int = 1729,
) -> dict[str, Any]:
    """Fit one deterministic Lab/OD calibration for all streamed WSI tiles."""
    rgb = np.asarray(image, dtype=np.uint8)
    source = np.asarray(labels)
    support = np.asarray(tissue, dtype=bool) & (source > 0)
    if rgb.shape[:2] != source.shape or support.shape != source.shape:
        raise ValueError("Calibration image, labels and tissue must share one frame")
    trusted = support if trusted_mask is None else support & np.asarray(trusted_mask, dtype=bool)
    if not np.any(trusted):
        trusted = support
    if not np.any(trusted):
        raise ValueError("Boundary calibration requires labelled tissue pixels")
    max_pixels = max(1, int(max_pixels))
    rng = np.random.default_rng(int(random_seed))

    def sampled_indices(mask: np.ndarray, cap: int) -> np.ndarray:
        indices = np.flatnonzero(mask)
        if indices.size > cap:
            indices = np.sort(rng.choice(indices, size=cap, replace=False))
        return indices

    flat_rgb = rgb[..., :3].reshape(-1, 3)
    fit_indices = sampled_indices(trusted, max_pixels)
    fit_values = _raw_features(flat_rgb[fit_indices].reshape(-1, 1, 3)).reshape(-1, 6)
    center = np.median(fit_values, axis=0)
    q25, q75 = np.percentile(fit_values, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 0.1 * np.std(fit_values, axis=0))
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)

    prototypes: dict[int, np.ndarray] = {}
    prototype_counts: dict[int, int] = {}
    labels_flat = source.reshape(-1)
    per_label_cap = max(1_000, max_pixels // max(1, len(np.unique(labels_flat[trusted.reshape(-1)]))))
    for label in (int(v) for v in np.unique(source[trusted]) if int(v) > 0):
        ids = sampled_indices(trusted & (source == label), per_label_cap)
        if not ids.size:
            continue
        raw = _raw_features(flat_rgb[ids].reshape(-1, 1, 3)).reshape(-1, 6)
        standardized = np.clip((raw - center) / scale, -10.0, 10.0)
        prototypes[label] = np.median(standardized, axis=0).astype(np.float32)
        prototype_counts[label] = int(ids.size)

    return {
        "feature_center": center.astype(np.float32),
        "feature_scale": scale,
        "label_prototypes": prototypes,
        "metadata": {
            "scope": "single_slide_global_sample",
            "sample_pixels": int(fit_indices.size),
            "max_pixels": int(max_pixels),
            "random_seed": int(random_seed),
            "prototype_sample_pixels": prototype_counts,
            "feature_space": "Lab_plus_optical_density",
        },
    }


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(array)
    ys = slice(max(0, dy), array.shape[-2] + min(0, dy))
    xs = slice(max(0, dx), array.shape[-1] + min(0, dx))
    source_y = slice(max(0, -dy), array.shape[-2] - max(0, dy))
    source_x = slice(max(0, -dx), array.shape[-1] - max(0, dx))
    result[..., ys, xs] = array[..., source_y, source_x]
    return result


def _neighbour_steps(connectivity: int) -> tuple[tuple[int, int, float], ...]:
    if int(connectivity) == 4:
        return ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0))
    if int(connectivity) == 8:
        diagonal = float(1.0 / np.sqrt(2.0))
        return (
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, diagonal), (-1, 1, diagonal),
            (1, -1, diagonal), (1, 1, diagonal),
        )
    raise ValueError("Boundary competition connectivity must be 4 or 8")


def _edge_weights(
    features: np.ndarray,
    support: np.ndarray,
    edge_beta: float,
    connectivity: int,
) -> dict[tuple[int, int], np.ndarray]:
    weights: dict[tuple[int, int], np.ndarray] = {}
    support = np.asarray(support, dtype=bool)
    for dy, dx, distance_weight in _neighbour_steps(connectivity):
        shifted = _shift(np.moveaxis(features, -1, 0), dy, dx)
        diff = np.mean((np.moveaxis(features, -1, 0) - shifted) ** 2, axis=0)
        valid = support & _shift(support, dy, dx)
        weights[(dy, dx)] = (
            float(distance_weight) * np.exp(-float(edge_beta) * diff) * valid
        ).astype(np.float32)
    return weights


def _hard_energy(
    indices: np.ndarray,
    data_cost: np.ndarray,
    edge_weights: dict[tuple[int, int], np.ndarray],
    support: np.ndarray,
    data_weight: float,
    smoothness_weight: float,
    connectivity: int,
) -> float:
    rows, cols = np.indices(indices.shape)
    data = float(np.sum(data_cost[indices, rows, cols][support], dtype=np.float64)) * float(data_weight)
    pair = 0.0
    for dy, dx, _ in _neighbour_steps(connectivity):
        if dy < 0 or (dy == 0 and dx < 0):
            continue
        shifted = _shift(indices, dy, dx)
        valid = support & _shift(support, dy, dx)
        pair += float(np.sum(edge_weights[(dy, dx)][valid] * (indices[valid] != shifted[valid]), dtype=np.float64))
    return data + float(smoothness_weight) * pair


def annealed_wand_boundary_competition(
    image: np.ndarray,
    labels: np.ndarray,
    tissue: np.ndarray,
    *,
    editable_mask: np.ndarray | None = None,
    protected_labels: np.ndarray | None = None,
    boundary_radius: int = 16,
    iterations: int = 16,
    initial_temperature: float = 2.0,
    final_temperature: float = 0.05,
    data_weight: float = 1.0,
    smoothness_weight: float = 0.3,
    edge_beta: float = 0.7,
    connectivity: int = 8,
    appearance_calibration: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimize an image/Potts energy inside a bounded internal boundary band.

    Mean-field probabilities follow ``q(k) ∝ exp(-E(k)/T)``.  Cooling changes
    the neighbour term before the final hard assignment; the lowest hard-energy
    state seen during the schedule is returned, so the accepted result never
    has greater declared energy than the input state.
    """
    source = np.asarray(labels)
    support = np.asarray(tissue, dtype=bool)
    if source.ndim != 2 or source.shape != support.shape or np.asarray(image).shape[:2] != source.shape:
        raise ValueError("Image, labels and tissue support must share one two-dimensional frame")
    if not np.issubdtype(source.dtype, np.integer) or np.any(source < 0):
        raise ValueError("Boundary competition labels must be nonnegative integers")
    # GrandQC support is authoritative.  Diagnostic-scale sampling can retain
    # a foreground label at a pixel whose independently sampled support is
    # false; normalize that edge mismatch before declaring the footprint that
    # the competition itself must preserve.
    source = source.copy()
    source[~support] = 0
    if iterations < 1 or boundary_radius < 0:
        raise ValueError("Boundary radius must be nonnegative and iterations must be positive")
    if min(initial_temperature, final_temperature, data_weight, smoothness_weight, edge_beta) <= 0:
        raise ValueError("Temperatures and energy weights must be positive")
    _neighbour_steps(connectivity)
    active_labels = np.array([int(v) for v in np.unique(source[support]) if int(v) > 0], dtype=np.int64)
    metadata: dict[str, Any] = {
        "schema_version": "cellphenotyper.annealed_wand_boundary.v2",
        "enabled": True,
        "applied": False,
        "labels": active_labels.tolist(),
        "boundary_radius_px": int(boundary_radius),
        "iterations": int(iterations),
        "initial_temperature": float(initial_temperature),
        "final_temperature": float(final_temperature),
        "temperature_equation": "q(label) proportional to exp(-local_energy(label)/temperature)",
        "energy": f"robust_Lab_OD_data + edge_weighted_{int(connectivity)}_neighbour_Potts",
        "candidate_rule": f"current_or_{int(connectivity)}_connected_neighbour_label_only",
        "neighbour_connectivity": int(connectivity),
        "diagonal_weight": float(1.0 / np.sqrt(2.0)) if int(connectivity) == 8 else 0.0,
        "confidence_is_calibrated_probability": False,
    }
    if active_labels.size < 2 or boundary_radius == 0:
        metadata.update(editable_pixels=0, changed_pixels=0, reason="fewer_than_two_labels_or_zero_radius")
        return source.copy(), metadata

    foreground = (source > 0) & support
    boundary = _internal_boundary(source)
    editable = ndi.binary_dilation(boundary, iterations=int(boundary_radius)) & foreground
    if editable_mask is not None:
        editable &= np.asarray(editable_mask, dtype=bool)
    protected = np.zeros(source.shape, dtype=bool)
    protected_values = None
    if protected_labels is not None:
        protected_values = np.asarray(protected_labels)
        if protected_values.shape != source.shape:
            raise ValueError("Protected labels must match the competition frame")
        protected = protected_values > 0
        editable &= ~protected
    metadata["editable_pixels"] = int(np.count_nonzero(editable))
    metadata["protected_pixels"] = int(np.count_nonzero(protected))
    if not np.any(editable):
        metadata.update(changed_pixels=0, reason="empty_editable_boundary_band")
        return source.copy(), metadata

    feature_values = _features(image, foreground, appearance_calibration)
    label_to_index = {int(label): index for index, label in enumerate(active_labels)}
    index_labels = np.zeros(source.shape, dtype=np.int16)
    for label, index in label_to_index.items():
        index_labels[source == label] = index

    prototypes = []
    for label in active_labels:
        fitted = None
        if appearance_calibration is not None:
            fitted = appearance_calibration.get("label_prototypes", {}).get(int(label))
        if fitted is not None:
            prototypes.append(np.asarray(fitted, dtype=np.float32))
        else:
            candidate = (source == label) & foreground & ~editable
            if protected_values is not None:
                candidate |= protected_values == label
            if not np.any(candidate):
                candidate = (source == label) & foreground
            prototypes.append(np.median(feature_values[candidate], axis=0))
    prototypes = np.asarray(prototypes, dtype=np.float32)
    data_cost = np.mean((feature_values[None, ...] - prototypes[:, None, None, :]) ** 2, axis=-1).astype(np.float32)
    data_cost = np.minimum(data_cost, 100.0)
    edge_weights = _edge_weights(feature_values, foreground, edge_beta, connectivity)

    q = np.zeros((active_labels.size, *source.shape), dtype=np.float32)
    for index in range(active_labels.size):
        q[index] = index_labels == index
    best_indices = index_labels.copy()
    initial_energy = _hard_energy(
        index_labels, data_cost, edge_weights, foreground,
        data_weight, smoothness_weight, connectivity,
    )
    best_energy = initial_energy
    temperatures = np.geomspace(float(initial_temperature), float(final_temperature), int(iterations))
    hard = index_labels.copy()
    structure = ndi.generate_binary_structure(2, 1 if int(connectivity) == 4 else 2)
    for temperature in temperatures:
        local_energy = float(data_weight) * data_cost.copy()
        for index in range(active_labels.size):
            disagreement = np.zeros(source.shape, dtype=np.float32)
            for direction, weight in edge_weights.items():
                disagreement += weight * (1.0 - _shift(q[index], *direction))
            local_energy[index] += float(smoothness_weight) * disagreement
        candidates = np.stack(
            [ndi.binary_dilation(hard == index, structure=structure) for index in range(active_labels.size)],
            axis=0,
        )
        local_energy[:, ~editable] = np.inf
        local_energy[~candidates] = np.inf
        minimum = np.min(local_energy, axis=0)
        valid = editable & np.isfinite(minimum)
        logits = np.full_like(local_energy, -np.inf)
        logits[:, valid] = -(local_energy[:, valid] - minimum[valid]) / float(temperature)
        probabilities = np.zeros_like(q)
        probabilities[:, valid] = np.exp(np.clip(logits[:, valid], -80.0, 0.0))
        normalizer = probabilities.sum(axis=0)
        normalizer[~valid] = 1.0
        probabilities[:, valid] /= normalizer[valid]
        for index in range(active_labels.size):
            probabilities[index, ~editable] = index_labels[~editable] == index
        q = probabilities
        hard = np.argmax(q, axis=0).astype(np.int16)
        hard[~editable] = index_labels[~editable]
        energy = _hard_energy(
            hard, data_cost, edge_weights, foreground,
            data_weight, smoothness_weight, connectivity,
        )
        if energy < best_energy:
            best_energy = energy
            best_indices = hard.copy()

    result = source.copy()
    for index, label in enumerate(active_labels):
        result[editable & (best_indices == index)] = label
    if protected_values is not None:
        result[protected] = protected_values[protected]
    result[~support] = 0
    if np.any((result > 0) != (source > 0)):
        raise RuntimeError("Boundary competition changed the foreground footprint")
    changed = int(np.count_nonzero(result != source))
    metadata.update(
        applied=bool(changed),
        changed_pixels=changed,
        initial_energy=float(initial_energy),
        final_energy=float(best_energy),
        energy_change=float(best_energy - initial_energy),
        accepted_nonincreasing_energy=bool(best_energy <= initial_energy + 1e-8),
        data_weight=float(data_weight),
        smoothness_weight=float(smoothness_weight),
        edge_beta=float(edge_beta),
        appearance_calibration_scope=(
            appearance_calibration.get("metadata", {}).get("scope", "provided")
            if appearance_calibration is not None else "tile_local_legacy"
        ),
    )
    return result.astype(source.dtype, copy=False), metadata
