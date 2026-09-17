#!/usr/bin/env python3
"""Discover spatial subdomains within immutable, calibrated parent tissue domains.

Parent labels are an input, never re-estimated or filled. Local and wider-context
UNI-2 blocks are joined by grid ID and explicitly weighted. Discovery operates
in numerical feature space, never a two-dimensional display embedding. Region
outputs are four-connected components; background gaps cannot join regions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tifffile

from cell_profile_io import RasterReader, sha256_file, staged_files
from ome_tiff_metadata import create_tiff_memmap


SCHEMA_VERSION = "1.0.0"
STATUS = {0: "background", 1: "accepted_subdomain", 2: "parent_only_no_observation", 3: "mixed_parent_core", 4: "missing_features", 5: "insufficient_observations", 6: "no_stable_subdivision", 7: "seed_unstable", 8: "scale_disagreement", 9: "ambiguous_centroid_assignment", 10: "parent_assignment_uncertain", 11: "native_graph_isolate", 12: "ambiguous_graph_assignment"}
GRID_COLUMNS = ["label", "x", "y", "grid_row", "grid_col", "core_x0", "core_y0", "core_x1", "core_y1"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_geometry(grid_metadata, shift, resolution, shape):
    """Resolve original-slide coordinates without trusting an uncalibrated crop."""
    if resolution.get("status") != "pass":
        raise ValueError("Physical resolution report must have status=pass")
    mpp = np.array([resolution["mpp_x"], resolution["mpp_y"]], dtype=float)
    if not np.isfinite(mpp).all() or np.any(mpp <= 0):
        raise ValueError("Physical MPP must be finite and positive")
    if grid_metadata.get("observation_type") != "spatial_grid" or grid_metadata.get("coordinate_space") not in {"crop_roi_level0_pixels", "crop_level0_pixels"}:
        raise ValueError("Grid metadata must explicitly declare spatial_grid in crop level-zero pixels")
    expected = (int(grid_metadata["image_height_px"]), int(grid_metadata["image_width_px"]))
    size = shift.get("crop_size", {})
    if tuple(shape) != expected or expected != (int(size.get("height", 0)), int(size.get("width", 0))):
        raise ValueError("Parent mask, grid metadata and crop shift dimensions differ")
    declared = np.array([grid_metadata["source_mpp_x"], grid_metadata["source_mpp_y"]], float)
    if not np.allclose(declared, mpp, rtol=0.02, atol=0):
        raise ValueError(f"Grid calibration {declared.tolist()} differs from verified image MPP {mpp.tolist()}; regenerate or audit extraction provenance before hierarchy analysis")
    offset = shift.get("offset_crop_to_original")
    if not isinstance(offset, dict) or not {"dx", "dy"} <= set(offset):
        raise ValueError("Crop-to-original pixel offset must be explicit")
    origin_px = np.array([offset["dx"], offset["dy"]], float)
    if not np.isfinite(origin_px).all() or np.any(origin_px < 0):
        raise ValueError("Crop origin must be finite nonnegative level-zero pixels")
    if int(resolution.get("width_px", expected[1])) < origin_px[0] + expected[1] or int(resolution.get("height_px", expected[0])) < origin_px[1] + expected[0]:
        raise ValueError("Crop lies outside the calibrated original image")
    return {"mpp_xy": mpp.tolist(), "origin_px_xy": origin_px.tolist(), "origin_um_xy": (mpp * origin_px).tolist(), "shape_yx": list(expected), "coordinate_space": "original_slide_micrometres"}


def validate_grid(grid, shape):
    if not set(GRID_COLUMNS) <= set(grid):
        raise ValueError(f"Missing grid columns: {sorted(set(GRID_COLUMNS) - set(grid))}")
    grid = grid.copy()
    for name in GRID_COLUMNS:
        values = pd.to_numeric(grid[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Grid {name} must contain finite integers")
        grid[name] = values.astype(np.int64)
    if grid.empty or grid.label.duplicated().any() or (grid.label <= 0).any() or grid.duplicated(["grid_row", "grid_col"]).any():
        raise ValueError("Grid IDs and lattice sites must be unique and nonempty")
    h, w = shape
    if not ((grid.x >= 0) & (grid.x < w) & (grid.y >= 0) & (grid.y < h)).all():
        raise ValueError("Grid centers are outside the parent mask")
    if not ((grid.core_x1 > grid.core_x0) & (grid.core_y1 > grid.core_y0) & (grid.x >= grid.core_x0) & (grid.x < grid.core_x1) & (grid.y >= grid.core_y0) & (grid.y < grid.core_y1)).all():
        raise ValueError("Invalid grid core bounds or centers")
    if np.any((grid.core_x1 - grid.core_x0) * (grid.core_y1 - grid.core_y0) > 4194304):
        raise ValueError("Grid cores exceed the 4,194,304-pixel bounded-window limit")
    previous_y1 = -np.inf
    for _, rows in grid.groupby("grid_row", sort=True):
        if rows.core_y0.nunique() != 1 or rows.core_y1.nunique() != 1:
            raise ValueError("A grid row has inconsistent vertical core bounds")
        if rows.core_y0.iloc[0] < previous_y1:
            raise ValueError("Grid row cores overlap")
        previous_y1 = rows.core_y1.iloc[0]
        ordered = rows.sort_values("core_x0")
        if np.any(ordered.core_x0.to_numpy()[1:] < ordered.core_x1.to_numpy()[:-1]):
            raise ValueError("Grid column cores overlap")
    return grid.sort_values("label", kind="stable").reset_index(drop=True)


def validate_feature_definitions(metadata, geometry):
    """Require actual physical input fields and immutable model identities."""
    definitions = {}
    for name in ("local", "context"):
        definition = metadata.get(name)
        required = {"model_id", "model_revision", "pooling", "preprocessing", "field_width_source_px", "input_context_width_source_px"}
        if not isinstance(definition, dict) or not required <= set(definition):
            raise ValueError(f"{name} feature definition requires {sorted(required)}")
        definition = dict(definition)
        if any("path" in key.lower() for key in definition):
            raise ValueError("Feature definitions must not contain local filesystem paths")
        for field in ("model_id", "model_revision", "pooling", "preprocessing"):
            if not isinstance(definition[field], str) or not definition[field].strip():
                raise ValueError(f"{name} {field} must be a nonempty string")
            if definition[field].startswith(("/", "~/", "file:")):
                raise ValueError("Feature definitions must not contain local filesystem paths")
        if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-fA-F]{64})", definition["model_revision"]):
            raise ValueError("An immutable model revision or weights digest is required")
        for field in ("field_width_source_px", "input_context_width_source_px"):
            value = float(definition[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Feature field sizes must be positive source-pixel widths")
            definition[field] = value
        if definition["field_width_source_px"] > definition["input_context_width_source_px"]:
            raise ValueError("Selected feature field cannot exceed its input context")
        definition["field_width_um_xy"] = (np.array(geometry["mpp_xy"]) * definition["field_width_source_px"]).tolist()
        definition["input_context_width_um_xy"] = (np.array(geometry["mpp_xy"]) * definition["input_context_width_source_px"]).tolist()
        definition["source_mpp_xy"] = list(geometry["mpp_xy"])
        definitions[name] = definition
    if definitions["context"]["field_width_source_px"] <= definitions["local"]["field_width_source_px"]:
        raise ValueError("Context features must represent a wider field than local features")
    return definitions


def load_embedding_block(source, grid, destination, definition, *, chunk_rows=256):
    """Join actual CSV or binary shards into disk-backed rows without ID loss."""
    from uni2_embedding_io import discover_embedding_shards, iter_binary_blocks, binary_shard_paths
    from build_cell_profiles import verify_uni2_provenance
    source = Path(source)
    if source.is_dir() and (source / "embedding_manifest.json").is_file():
        if any(staged_files(source, pattern) for pattern in ('*embeddings_shard*.csv*', '*embeddings_shard*.embedding.json', '*embeddings_shard*.features.bin')):
            raise ValueError('Ambiguous mixture of hierarchy NPY bundle and UNI-2 shards')
        return load_binary_embedding_block(source, grid, destination, definition, chunk_rows=chunk_rows)
    files = discover_embedding_shards(source)
    if not files:
        raise ValueError(f"No embedding shards found in {source}")
    grid_index = {str(label): i for i, label in enumerate(grid.label)}
    seen = np.zeros(len(grid), dtype=bool)
    names = None
    output = None
    hashes = {}
    input_dtypes = set()
    binary_source = False
    shard_rows = {}
    for path in files:
        shard_rows[path.resolve()] = 0
        hashes[str(path.resolve())] = sha256_file(path)
        binary = path.name.endswith('.embedding.json')
        if binary:
            binary_source = True
            declared_names = json.loads(path.read_text())['feature_names']
            if not declared_names or not all(isinstance(c, str) and re.fullmatch(r'feat_[1-9][0-9]*', c) for c in declared_names):
                raise ValueError('Binary hierarchy feature schema requires feat_N names')
            feature_names = sorted(declared_names, key=lambda c: int(c[5:]))
            feature_order = [declared_names.index(name) for name in feature_names]
            for payload in binary_shard_paths(path):
                hashes[str(payload.resolve())] = sha256_file(payload)
            parts = iter_binary_blocks(path, block_rows=chunk_rows)
        else:
            parts = ((part, None) for part in pd.read_csv(path, chunksize=chunk_rows, dtype={"cell_id": str}, float_precision='round_trip', keep_default_na=False))
        for part, binary_values in parts:
            shard_rows[path.resolve()] += len(part)
            if not binary:
                feature_names = sorted((c for c in part if c.startswith("feat_") and c[5:].isdigit()), key=lambda c: int(c[5:]))
            if not feature_names or "cell_id" not in part:
                raise ValueError("Embedding shards need cell_id and feat_N columns")
            if names is None:
                names = feature_names
                output = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=(len(grid), len(names)))
                output[:] = np.nan
            elif names != feature_names:
                raise ValueError("Embedding feature schema changed between shards")
            if part.cell_id.isna().any() or part.cell_id.duplicated().any():
                raise ValueError("Embedding IDs are missing or duplicated")
            ids = part.cell_id.astype(str).tolist()
            if any(label not in grid_index for label in ids):
                raise ValueError("Embedding shard contains IDs absent from the declared grid")
            rows = np.array([grid_index[label] for label in ids])
            if seen[rows].any():
                raise ValueError("Embedding IDs are duplicated across shards")
            if "observation_type" in part and not part.observation_type.eq("grid").all():
                raise ValueError("Cell-centered embeddings cannot substitute for grid observations")
            for column, expected in (("cx", grid.x.iloc[rows].to_numpy()), ("cy", grid.y.iloc[rows].to_numpy())):
                if column in part and not np.allclose(pd.to_numeric(part[column]), expected, rtol=0, atol=0):
                    raise ValueError("Embedding coordinates disagree with grid coordinates")
            if "extraction_tile_size" in part and not np.allclose(pd.to_numeric(part.extraction_tile_size), definition["input_context_width_source_px"]):
                raise ValueError("Embedding input context size disagrees with feature definition")
            if "source_mpp" in part and "source_mpp_xy" in definition and not np.allclose(pd.to_numeric(part.source_mpp), np.mean(definition["source_mpp_xy"]), rtol=.02, atol=0):
                raise ValueError("Embedding extraction calibration disagrees with verified source MPP")
            if binary:
                input_dtypes.add(str(binary_values.dtype))
                values = np.asarray(binary_values[:, feature_order], dtype=np.float32)
            else:
                values = part[names].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
            if np.isinf(values).any():
                raise ValueError("Embedding block contains infinite values")
            output[rows] = values
            seen[rows] = True
    if output is None:
        raise ValueError("Embedding source contains no observations")
    output.flush()
    completions = []
    if source.is_dir():
        for path in staged_files(source, '.*_embedding_complete.json'):
            completions.append(json.loads(path.read_text()))
            hashes[str(path.resolve())] = sha256_file(path)
        for path in staged_files(source, '.*_grid_complete.json'):
            hashes[str(path.resolve())] = sha256_file(path)
    extraction, execution = verify_uni2_provenance(source, completions, files, None, observation_type='grid', shard_rows=shard_rows, expected_embedding_mode=definition.get('embedding_mode'))
    return output, names, {"files_sha256": hashes, "observations_present": int(seen.sum()), "observations_missing": int((~seen).sum()), "feature_definition": definition,
        "extraction_provenance": extraction, "verified_execution": execution,
        "storage": {"format": "cellphenotyper_uni2_binary" if binary_source else "legacy_csv", "binary_input_dtypes": sorted(input_dtypes),
                    "output_dtype": "float32", "conversion": "Existing hierarchy float32 semantics; storage format is not biological feature identity"}}


def load_binary_embedding_block(source, grid, destination, definition, *, chunk_rows=256):
    """Validate a binary feature bundle before zero-copy or chunked keyed access."""
    source = Path(source)
    manifest_path = source / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "cellphenotyper_grid_embeddings_npy":
        raise ValueError("Unknown binary grid feature format")
    if manifest.get("feature_definition") != definition:
        raise ValueError("Binary feature definition disagrees with declared physical/model metadata")
    paths, hashes = {}, {str(manifest_path.resolve()): sha256_file(manifest_path)}
    for key in ("matrix", "rows"):
        record = manifest[key]
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Binary feature bundle paths must be relative and contained")
        paths[key] = source / relative
        digest = sha256_file(paths[key])
        if digest != record["sha256"]:
            raise ValueError(f"Binary {key} checksum mismatch")
        hashes[str(paths[key].resolve())] = digest
    rows = pd.read_csv(paths["rows"], dtype={"cell_id": str}, keep_default_na=False, float_precision='round_trip')
    required = {"cell_id", "cx", "cy", "observation_type", "source_mpp", "extraction_tile_size"}
    if not required <= set(rows) or rows.cell_id.isna().any() or rows.cell_id.duplicated().any():
        raise ValueError("Binary feature rows need unique grid IDs and physical/coordinate metadata")
    if not rows.observation_type.eq("grid").all():
        raise ValueError("Binary hierarchy features must be grid observations")
    expected_ids = grid.label.astype(str).tolist()
    if len(rows) != len(grid) or set(rows.cell_id) != set(expected_ids) or manifest["rows"]["count"] != len(grid):
        raise ValueError("Binary feature IDs must exactly match the declared grid")
    lookup = pd.Series(np.arange(len(rows)), index=rows.cell_id)
    order = lookup.loc[expected_ids].to_numpy(int)
    ordered = rows.iloc[order]
    if not np.array_equal(pd.to_numeric(ordered.cx).to_numpy(), grid.x.to_numpy()) or not np.array_equal(pd.to_numeric(ordered.cy).to_numpy(), grid.y.to_numpy()):
        raise ValueError("Binary feature coordinates disagree with grid coordinates")
    if not np.allclose(pd.to_numeric(ordered.extraction_tile_size), definition["input_context_width_source_px"], rtol=0, atol=0):
        raise ValueError("Binary feature input context disagrees with its physical definition")
    if not np.allclose(pd.to_numeric(ordered.source_mpp), np.mean(definition["source_mpp_xy"]), rtol=.02, atol=0):
        raise ValueError("Binary feature calibration disagrees with verified MPP")
    values = np.load(paths["matrix"], mmap_mode="r", allow_pickle=False)
    names = manifest["feature_names"]
    if values.ndim != 2 or list(values.shape) != manifest["matrix"]["shape"] or values.dtype != np.dtype("float32") or values.shape != (len(grid), len(names)):
        raise ValueError("Binary feature matrix shape/dtype/schema mismatch")
    if len(set(names)) != len(names) or not names or not all(re.fullmatch(r"feat_[1-9][0-9]*", name) for name in names):
        raise ValueError("Binary feature names must be unique feat_N columns")
    needs_alignment = not np.array_equal(order, np.arange(len(grid)))
    output = np.lib.format.open_memmap(destination, mode="w+", dtype="float32", shape=values.shape) if needs_alignment else values
    present = 0
    for start in range(0, len(grid), max(1, chunk_rows)):
        selected = order[start:start+max(1, chunk_rows)]
        block = values[selected]
        finite = np.isfinite(block).all(axis=1)
        absent = np.isnan(block).all(axis=1)
        if not (finite | absent).all():
            raise ValueError("Binary features must be finite rows or explicit all-NaN missing rows")
        present += int(finite.sum())
        if needs_alignment:
            output[start:start+len(selected)] = block
    if needs_alignment:
        output.flush()
    return output, names, {"format": manifest["format"], "files_sha256": hashes, "observations_present": present,
                           "observations_missing": len(grid)-present, "feature_definition": definition,
                           "zero_copy_readonly": not needs_alignment,
                           "source_inputs": manifest.get("source_inputs", {}), "sample_id": manifest.get("sample_id")}


def validate_parent_support(parent, support=None, parent_uncertainty=None, tile_size=512):
    """Fail on support leaks, never edit the immutable parent to conceal them.

    Support is the same cropped field at potentially coarser resolution. The
    mapping floor(target_index * support_size / target_size) exactly matches
    the pipeline refinement reader; this is not a guessed physical transform.
    """
    if parent_uncertainty is not None and (parent_uncertainty.height, parent_uncertainty.width) != (parent.height, parent.width):
        raise ValueError("Parent uncertainty and parent mask dimensions differ")
    codes, leaks = {}, 0
    for y in range(0, parent.height, tile_size):
        for x in range(0, parent.width, tile_size):
            x1, y1 = min(x + tile_size, parent.width), min(y + tile_size, parent.height)
            block = parent.window(x, y, x1, y1)
            if block.ndim != 2 or not np.isfinite(block).all() or np.any(block < 0) or not np.equal(block, np.floor(block)).all():
                raise ValueError("Parent mask must contain nonnegative integer labels")
            if support is not None:
                sx = np.minimum((np.arange(x, x1, dtype=np.int64) * support.width) // parent.width, support.width - 1)
                sy = np.minimum((np.arange(y, y1, dtype=np.int64) * support.height) // parent.height, support.height - 1)
                source = support.window(int(sx[0]), int(sy[0]), int(sx[-1]) + 1, int(sy[-1]) + 1)
                if source.ndim != 2 or not np.isfinite(source).all() or np.any(source < 0):
                    raise ValueError("Support must be a finite nonnegative scalar mask")
                allowed = source[np.ix_(sy - sy[0], sx - sx[0])] > 0
                leaks += int(np.count_nonzero((block > 0) & ~allowed))
            if parent_uncertainty is not None:
                uncertainty = parent_uncertainty.window(x, y, x1, y1)
                if uncertainty.ndim != 2 or not np.isfinite(uncertainty).all() or np.any((uncertainty < 0) | (uncertainty > 254)) or not np.equal(uncertainty, np.floor(uncertainty)).all():
                    raise ValueError("Parent uncertainty must contain categorical integer codes 0..254")
                labels, counts = np.unique(uncertainty, return_counts=True)
                for label, count in zip(labels, counts):
                    codes[int(label)] = codes.get(int(label), 0) + int(count)
    if leaks:
        raise ValueError(f"Immutable parent mask leaks outside scaled tissue support at {leaks} pixels; correct upstream support/registration, not the hierarchy")
    return {"support_checked": support is not None, "support_mapping": "floor(target_index * support_size / parent_size); nearest-neighbor; same cropped field",
            "parent_pixels_outside_support": leaks, "parent_uncertainty_checked": parent_uncertainty is not None,
            "parent_uncertainty_code_counts": codes}


def assign_parent_domains(grid, parent, purity=0.8, parent_uncertainty=None):
    if not 0.5 < purity <= 1:
        raise ValueError("Parent purity must be in (0.5,1]")
    rows = grid.copy()
    parents, fractions, status, known_fractions = [], [], [], []
    for record in grid.itertuples():
        x0, y0, x1, y1 = max(0, record.core_x0), max(0, record.core_y0), min(parent.width, record.core_x1), min(parent.height, record.core_y1)
        block = parent.window(x0, y0, x1, y1)
        if block.ndim != 2 or not np.isfinite(block).all() or np.any(block < 0) or not np.equal(block, np.floor(block)).all():
            raise ValueError("Parent mask must contain nonnegative integer labels")
        labels, counts = np.unique(block[block > 0], return_counts=True)
        if len(labels) == 0:
            parents.append(0); fractions.append(0.); status.append(0); known_fractions.append(0.)
            continue
        winner = np.argmax(counts)
        parent_id = int(labels[winner])
        fraction = float(counts[winner] / counts.sum())
        center = int(block[record.y - y0, record.x - x0])
        parents.append(parent_id)
        fractions.append(fraction)
        code, known = (2 if fraction >= purity and center == parent_id else 3), 1.
        if parent_uncertainty is not None:
            uncertainty = parent_uncertainty.window(x0, y0, x1, y1)
            known = float(np.count_nonzero((block == parent_id) & (uncertainty == 0)) / counts[winner])
            if code == 2 and (known < purity or uncertainty[record.y - y0, record.x - x0] != 0):
                code = 10
        known_fractions.append(known)
        status.append(code)
    rows["parent_domain_id"], rows["parent_purity"], rows["status_code"] = parents, fractions, status
    rows["parent_assignment_known_fraction"] = known_fractions
    return rows


def _canonical(labels, centers):
    order = sorted(range(len(centers)), key=lambda i: tuple(centers[i]))
    return np.argsort(order)[labels], centers[order]


def _align(pred, reference, k):
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (pred, reference), 1)
    source, target = linear_sum_assignment(-contingency)
    mapping = np.empty(k, dtype=int)
    mapping[source] = target
    return mapping[pred]


def _transform_blocks(blocks, indices, training, weights, n_components, seed):
    from sklearn.decomposition import PCA
    transformed, metadata = {}, {}
    for name, block in blocks.items():
        fit = np.asarray(block[indices[training]], dtype=np.float64)
        mean, std = fit.mean(axis=0), fit.std(axis=0)
        keep = std > 1e-10
        if not keep.any():
            transformed[name] = np.zeros((len(indices), 1), np.float32)
            metadata[name] = {"constant": True, "dimensions": 1, "weight": weights[name]}
            continue
        fit = (fit[:, keep] - mean[keep]) / std[keep]
        dimensions = min(n_components, fit.shape[1], max(1, len(fit) - 1))
        reducer = None
        if fit.shape[1] > dimensions:
            reducer = PCA(n_components=dimensions, svd_solver="randomized", random_state=seed).fit(fit)
        projected = np.empty((len(indices), dimensions if reducer else fit.shape[1]), dtype=np.float32)
        for start in range(0, len(indices), 1024):
            data = (np.asarray(block[indices[start:start + 1024]], float)[:, keep] - mean[keep]) / std[keep]
            projected[start:start + len(data)] = reducer.transform(data) if reducer else data
        # Equal total variance per block, then user-declared group weights.
        rms = max(1e-12, float(np.sqrt(np.mean(np.sum(projected[training] ** 2, axis=1)))))
        transformed[name] = projected / rms * math.sqrt(weights[name])
        metadata[name] = {"constant": False, "dimensions": projected.shape[1], "source_dimensions": int(block.shape[1]), "retained_source_indices": np.flatnonzero(keep).tolist(), "means": mean[keep].tolist(), "std": std[keep].tolist(), "weight": weights[name], "normalization_rms": rms, "pca_explained_variance_fraction": float(reducer.explained_variance_ratio_.sum()) if reducer else 1.0, "pca_components": reducer.components_.tolist() if reducer else None}
    return transformed, metadata


def discover_subdomains(grid, blocks, *, weights=None, max_k=5, fixed_k=None, seed=17, repeats=5, fit_limit=5000, components=64, min_observations=20, min_seed_stability=.8, min_scale_agreement=1., min_margin=.1):
    """Legacy k-means comparison, not the default KODAMA-native hierarchy."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    if set(blocks) != {"local", "context"}:
        raise ValueError("Provide both local and wider-context feature blocks")
    weights = weights or {"local": 1., "context": 1.}
    if set(weights) != set(blocks) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Positive weights required for local and context blocks")
    if min(repeats, max_k, min_observations) < 2 or fit_limit < min_observations or components < 3:
        raise ValueError("Invalid clustering bounds; at least 3 feature components, 2 repeats and 2 observations are required")
    if fixed_k is not None and not 2 <= fixed_k <= max_k:
        raise ValueError("fixed_k must be between 2 and max_k")
    if any(not 0 <= value <= 1 for value in (min_seed_stability, min_scale_agreement, min_margin)):
        raise ValueError("Stability, scale-agreement and margin thresholds must be in [0,1]")
    rows = grid.copy()
    rows["local_subdomain_id"] = 0
    rows["subdomain_id"] = 0
    rows["seed_stability"] = np.nan
    rows["scale_agreement"] = np.nan
    rows["centroid_margin"] = np.nan
    complete = np.ones(len(rows), bool)
    for block in blocks.values():
        if block.ndim != 2 or block.shape[0] != len(rows):
            raise ValueError("Feature rows must match sorted grid rows")
        for begin in range(0, len(rows), 1024):
            complete[begin:begin + 1024] &= np.isfinite(block[begin:begin + 1024]).all(axis=1)
    rows.loc[(rows.status_code == 2) & ~complete, "status_code"] = 4
    diagnostics, next_subdomain = {}, 1
    rng = np.random.default_rng(seed)
    for parent_id in sorted(rows.parent_domain_id.unique()):
        if parent_id == 0:
            continue
        indices = np.flatnonzero((rows.parent_domain_id == parent_id) & (rows.status_code == 2))
        report = {"observations": len(indices), "candidates": [], "selected_k": 0}
        diagnostics[str(parent_id)] = report
        if len(indices) < min_observations:
            rows.loc[indices, "status_code"] = 5
            continue
        training = np.sort(rng.choice(len(indices), min(fit_limit, len(indices)), replace=False))
        data_blocks, normalization = _transform_blocks(blocks, indices, training, weights, components, seed)
        combined = np.column_stack([data_blocks[name] for name in sorted(data_blocks)])
        fit = combined[training]
        report.update({"normalization": normalization, "training_observations": len(training), "feature_dimensions": combined.shape[1]})
        distinct = len(np.unique(fit, axis=0))
        candidates = [fixed_k] if fixed_k else list(range(2, min(max_k, distinct, len(training) // max(2, min_observations // 2)) + 1))
        selected, best = None, -np.inf
        for k in candidates:
            if k > distinct:
                continue
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(fit)
            labels, centers = _canonical(model.predict(combined), model.cluster_centers_)
            distance = cdist(combined, centers)
            closest = distance[np.arange(len(labels)), labels]
            next_distance = np.partition(distance, 1, axis=1)[:, 1]
            margin = (next_distance - closest) / np.maximum(next_distance, 1e-12)
            seed_votes, aris = np.zeros(len(labels)), []
            for repeat in range(repeats):
                subset = np.sort(rng.choice(len(training), max(k, int(.8 * len(training))), replace=False))
                refit = KMeans(n_clusters=k, n_init=5, random_state=seed + repeat + 1).fit(fit[subset])
                aligned = _align(refit.predict(combined), labels, k)
                seed_votes += aligned == labels
                aris.append(float(adjusted_rand_score(labels[training], aligned[training])))
            scale_votes, scale_aris = np.zeros(len(labels)), {}
            for name, data in data_blocks.items():
                # A constant representation provides no independent scale evidence.
                if normalization[name]["constant"] or len(np.unique(data[training], axis=0)) < k:
                    scale_aris[name] = None
                    continue
                scale_model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(data[training])
                aligned = _align(scale_model.predict(data), labels, k)
                scale_votes += aligned == labels
                scale_aris[name] = float(adjusted_rand_score(labels[training], aligned[training]))
            seed_score = float(np.mean(aris))
            separation = float(margin[training].mean())
            size = int(np.bincount(labels[training], minlength=k).min())
            eligible = fixed_k is not None or (seed_score >= .75 and separation >= .4 and size >= max(2, min_observations // 2))
            score = seed_score * separation - .01 * k
            report["candidates"].append({"k": int(k), "seed_subsample_ari": seed_score, "centroid_silhouette": separation, "scale_ari": scale_aris, "smallest_training_cluster": size, "passes_auto_selection": bool(eligible), "score": score})
            if eligible and score > best:
                selected = labels, margin, seed_votes / repeats, scale_votes / len(data_blocks)
                report["selected_k"] = int(k)
                best = score
        if selected is None:
            rows.loc[indices, "status_code"] = 6
            continue
        labels, margin, seed_stability, scale_agreement = selected
        rows.loc[indices, "local_subdomain_id"] = labels + 1
        rows.loc[indices, "seed_stability"] = seed_stability
        rows.loc[indices, "scale_agreement"] = scale_agreement
        rows.loc[indices, "centroid_margin"] = margin
        status = np.ones(len(indices), int)
        status[margin < min_margin] = 9
        status[scale_agreement < min_scale_agreement] = 8
        status[seed_stability < min_seed_stability] = 7
        rows.loc[indices, "status_code"] = status
        for local_id in range(1, report["selected_k"] + 1):
            accepted = indices[(labels == local_id - 1) & (status == 1)]
            if len(accepted):
                rows.loc[accepted, "subdomain_id"] = next_subdomain
                next_subdomain += 1
        report["accepted_observations"] = int(np.sum(status == 1))
    rows["hierarchy_status"] = rows.status_code.map(STATUS)
    return rows, {"parent_domains": diagnostics, "seed": seed, "repeats": repeats, "weights": weights, "fixed_k": fixed_k, "thresholds": {"seed_stability": min_seed_stability, "scale_agreement": min_scale_agreement, "centroid_margin": min_margin}, "selection_rule": "bounded within-parent k-means; automatic K requires subsample ARI>=0.75 and mean centroid silhouette>=0.4, then maximizes ARI*separation-0.01*K; no stable split leaves parent-only", "confidence_is_calibrated_probability": False, "scale_evidence": "agreement of separate local/context partitions; contextualized token subsets are not independent inference scales"}


def rasterize(grid, parent, outdir, geometry, tile_size=512, parent_uncertainty=None):
    outdir = Path(outdir)
    shape = parent.height, parent.width
    mx, my = geometry["mpp_xy"]
    output = create_tiff_memmap(outdir / "subdomain_mask.ome.tif", shape=shape, dtype=np.dtype("uint32"), mpp_x=mx, mpp_y=my)
    uncertainty = create_tiff_memmap(outdir / "hierarchy_status.ome.tif", shape=shape, dtype=np.dtype("uint8"), mpp_x=mx, mpp_y=my)
    output[:] = 0
    parents = {}
    for y in range(0, shape[0], tile_size):
        for x in range(0, shape[1], tile_size):
            block = parent.window(x, y, min(x + tile_size, shape[1]), min(y + tile_size, shape[0]))
            if block.ndim != 2 or not np.isfinite(block).all() or np.any(block < 0) or not np.equal(block, np.floor(block)).all():
                raise ValueError("Invalid parent labels")
            uncertainty[y:y + block.shape[0], x:x + block.shape[1]] = np.where(block > 0, 2, 0)
            if parent_uncertainty is not None:
                uncertain = parent_uncertainty.window(x, y, x + block.shape[1], y + block.shape[0]) != 0
                uncertainty[y:y + block.shape[0], x:x + block.shape[1]][(block > 0) & uncertain] = 10
            labels, counts = np.unique(block, return_counts=True)
            for label, count in zip(labels, counts):
                if label > 0:
                    parents[int(label)] = parents.get(int(label), 0) + int(count)
    for row in grid.itertuples():
        x0, y0, x1, y1 = max(0, row.core_x0), max(0, row.core_y0), min(shape[1], row.core_x1), min(shape[0], row.core_y1)
        block = parent.window(x0, y0, x1, y1)
        tissue = block > 0
        known = np.ones(block.shape, dtype=bool) if parent_uncertainty is None else parent_uncertainty.window(x0, y0, x1, y1) == 0
        tissue &= known
        if row.status_code == 1:
            selected = (block == row.parent_domain_id) & known
            output[y0:y1, x0:x1][selected] = row.subdomain_id
            uncertainty[y0:y1, x0:x1][selected] = 1
            uncertainty[y0:y1, x0:x1][tissue & ~selected] = 3
        elif row.status_code > 0:
            uncertainty[y0:y1, x0:x1][tissue] = row.status_code
    output.flush(); uncertainty.flush()
    return output, uncertainty, parents


def connected_regions(subdomains, destination, geometry, tile_size=512, max_components=1000000):
    """Two-pass tiled four-connected labelling with disk-backed provisional IDs."""
    h, w = subdomains.shape
    mx, my = geometry["mpp_xy"]
    output = create_tiff_memmap(destination, shape=(h, w), dtype=np.dtype("uint32"), mpp_x=mx, mpp_y=my)
    output[:] = 0
    roots = [0]
    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            roots[max(a, b)] = min(a, b)
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1, x1 = min(h, y0 + tile_size), min(w, x0 + tile_size)
            block = np.asarray(subdomains[y0:y1, x0:x1])
            local = np.zeros(block.shape, np.uint32)
            for label in np.unique(block):
                if label == 0:
                    continue
                pieces, count = ndimage.label(block == label)
                if len(roots) + count > max_components:
                    raise ValueError("Connected region component bound exceeded; increase --max-components after reviewing fragmentation")
                start = len(roots) - 1
                roots.extend(range(len(roots), len(roots) + count))
                selected = pieces > 0
                local[selected] = pieces[selected] + start
            output[y0:y1, x0:x1] = local
            if y0 > 0:
                same = (block[0] > 0) & (block[0] == subdomains[y0 - 1, x0:x1])
                pairs = np.column_stack([local[0, same], output[y0 - 1, x0:x1][same]])
                for a, b in np.unique(pairs, axis=0):
                    union(int(a), int(b))
            if x0 > 0:
                same = (block[:, 0] > 0) & (block[:, 0] == subdomains[y0:y1, x0 - 1])
                pairs = np.column_stack([local[same, 0], output[y0:y1, x0 - 1][same]])
                for a, b in np.unique(pairs, axis=0):
                    union(int(a), int(b))
    resolved = np.array([find(i) for i in range(len(roots))], dtype=np.uint32)
    unique = np.unique(resolved)
    remap = np.searchsorted(unique, resolved).astype(np.uint32)
    stats = {int(i): {"pixel_count": 0, "sum_x": 0., "sum_y": 0., "subdomain_id": 0} for i in range(1, len(unique))}
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            block = remap[output[y0:y0 + tile_size, x0:x0 + tile_size]]
            output[y0:y0 + tile_size, x0:x0 + tile_size] = block
            yy, xx = np.indices(block.shape)
            ids, first, inverse, counts = np.unique(block.ravel(), return_index=True, return_inverse=True, return_counts=True)
            sum_x = np.bincount(inverse, weights=xx.ravel() + x0 + .5)
            sum_y = np.bincount(inverse, weights=yy.ravel() + y0 + .5)
            subdomain_values = np.asarray(subdomains[y0:y0 + tile_size, x0:x0 + tile_size]).ravel()[first]
            for region, count, sx, sy, subdomain in zip(ids, counts, sum_x, sum_y, subdomain_values):
                if region == 0:
                    continue
                stats[int(region)]["pixel_count"] += int(count)
                stats[int(region)]["sum_x"] += float(sx)
                stats[int(region)]["sum_y"] += float(sy)
                stats[int(region)]["subdomain_id"] = int(subdomain)
    output.flush()
    return output, stats


def export_region_profiles(grid, blocks, names, definitions, region_mask, stats, geometry, outdir, sample_id, hierarchy_id):
    """Atlas-ready raw feature means weighted by assigned grid-core tissue area."""
    outdir = Path(outdir)
    directory = outdir / "region_profiles"
    directory.mkdir()
    region_ids = sorted(stats)
    index = {region: i for i, region in enumerate(region_ids)}
    sums = {}
    for name, block in blocks.items():
        # Region count can be large on fragmented tissue; accumulation lives on
        # disk rather than allocating regions x 1536 features in RAM.
        shape = (len(region_ids), block.shape[1])
        if len(region_ids):
            sums[name] = np.lib.format.open_memmap(directory / f".{name}_sums.npy", mode="w+", dtype="float64", shape=shape)
            sums[name][:] = 0
        else:
            sums[name] = np.empty(shape)
    weights = np.zeros(len(region_ids))
    candidates = {region: [] for region in region_ids}
    for i, row in enumerate(grid.itertuples()):
        if row.status_code != 1:
            continue
        block = region_mask[max(0, row.core_y0):min(region_mask.shape[0], row.core_y1), max(0, row.core_x0):min(region_mask.shape[1], row.core_x1)]
        labels, counts = np.unique(block[block > 0], return_counts=True)
        for region, count in zip(labels, counts):
            ri = index[int(region)]
            weights[ri] += count
            for name, array in blocks.items():
                sums[name][ri] += np.asarray(array[i], float) * count
            candidates[int(region)].append(i)
    if len(weights) and np.any(weights == 0):
        raise RuntimeError("A rasterized region has no source feature observation")
    means = {}
    for name, values in sums.items():
        if len(region_ids):
            means[name] = np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+", dtype="float32", shape=values.shape)
            for begin in range(0, len(region_ids), 256):
                means[name][begin:begin + 256] = values[begin:begin + 256] / weights[begin:begin + 256, None]
            means[name].flush()
        else:
            means[name] = np.empty(values.shape, np.float32)
            np.save(directory / f"{name}.npy", means[name])
    subdomains = grid.loc[grid.status_code == 1].drop_duplicates("subdomain_id").set_index("subdomain_id")
    mx, my = geometry["mpp_xy"]
    ox, oy = geometry["origin_um_xy"]
    records = []
    for region in region_ids:
        ri, record = index[region], stats[region]
        ids = np.array(candidates[region], int)
        # The representative is a real contributing grid observation, not a synthetic mean.
        distance = np.zeros(len(ids))
        for name, array in blocks.items():
            norm = max(1e-12, float(np.linalg.norm(means[name][ri])))
            for begin in range(0, len(ids), 1024):
                values = np.asarray(array[ids[begin:begin + 1024]], float)
                distance[begin:begin + len(values)] += np.mean((values - means[name][ri]) ** 2, axis=1) / norm**2
        representative = int(ids[np.argmin(distance)])
        sd = int(record["subdomain_id"])
        count = record["pixel_count"]
        records.append({"region_uid": f"{quote(sample_id, safe='')}::{hierarchy_id}::region_{region}", "region_id": str(region), "sample_id": sample_id, "parent_domain_id": int(subdomains.loc[sd, "parent_domain_id"]), "subdomain_id": sd, "local_subdomain_id": int(subdomains.loc[sd, "local_subdomain_id"]), "area_um2": count * mx * my, "x_um": record["sum_x"] / count * mx + ox, "y_um": record["sum_y"] / count * my + oy, "representative_grid_id": int(grid.iloc[representative].label), "representative_x_um": float(grid.iloc[representative].x) * mx + ox, "representative_y_um": float(grid.iloc[representative].y) * my + oy, "grid_observations": len(ids), "label_status": "unsupervised_discovery"})
    columns = ["region_uid", "region_id", "sample_id", "parent_domain_id", "subdomain_id", "local_subdomain_id", "area_um2", "x_um", "y_um", "representative_grid_id", "representative_x_um", "representative_y_um", "grid_observations", "label_status"]
    regions = pd.DataFrame(records, columns=columns)
    regions.to_csv(directory / "region_profiles.csv", index=False)
    regions[["region_uid", "region_id", "sample_id"]].to_csv(directory / "feature_rows.csv", index=False)
    manifest = {"schema_version": SCHEMA_VERSION, "observation_unit": "tissue_region", "region_count": len(regions), "hierarchy_id": hierarchy_id, "coordinate_space": "original_slide_micrometres", "files": {}, "feature_blocks": {}}
    for name, values in means.items():
        path = directory / f"{name}.npy"
        manifest["feature_blocks"][name] = {"path": path.name, "shape": list(values.shape), "feature_names": names[name], "feature_definition": {**definitions[name], "aggregation": "assigned_grid_core_tissue_area_weighted_mean", "observation_unit": "tissue_region"}}
        manifest["files"][path.name] = sha256_file(path)
        temporary_sum = directory / f".{name}_sums.npy"
        if temporary_sum.exists():
            del sums[name]
            temporary_sum.unlink()
    for name in ("region_profiles.csv", "feature_rows.csv"):
        manifest["files"][name] = sha256_file(directory / name)
    write_json(directory / "region_profiles_manifest.json", manifest)
    return regions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-mask", type=Path, required=True)
    parser.add_argument("--parent-uncertainty", type=Path, help="Canonical categorical uncertainty: 0 accepted, 1..254 uncertain/inferred; copied without recoding")
    parser.add_argument("--support-mask", type=Path, help="Same cropped tissue field; lower-resolution masks use the pipeline nearest-neighbor index mapping")
    parser.add_argument("--image", type=Path, help="Bind hierarchy to the exact source image; requires binary feature provenance with matching image/grid/shift/resolution hashes")
    parser.add_argument("--grid-objects", type=Path, required=True)
    parser.add_argument("--grid-metadata", type=Path, required=True)
    parser.add_argument("--shift-json", type=Path, required=True)
    parser.add_argument("--resolution-json", type=Path, required=True)
    parser.add_argument("--context-embeddings", type=Path, required=True)
    parser.add_argument("--local-embeddings", type=Path, required=True)
    parser.add_argument("--embedding-metadata", type=Path, required=True, help="JSON local/context immutable feature definitions and source-pixel fields")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--discovery-method", choices=("kodama_graph", "legacy_kmeans"), default="kodama_graph",
                        help="KODAMA-native discovery is default; k-means is an explicitly labelled legacy comparison only")
    parser.add_argument("--kodama-ncomp", type=int, default=50, help="KODAMA.matrix input, separate from per-field PCA components")
    parser.add_argument("--kodama-m", type=int, default=100)
    parser.add_argument("--kodama-tcycle", type=int, default=20)
    parser.add_argument("--kodama-neighbors", type=int, default=100)
    parser.add_argument("--kodama-cpus", type=int, default=1)
    parser.add_argument("--kodama-r-library", type=Path, help="Explicit local native KODAMA R library; no installation or API fallback")
    parser.add_argument("--min-affinity-margin", type=float, default=.1, help="Native graph own-versus-strongest-other affinity margin, not a probability")
    parser.add_argument("--context-weight", type=float, default=1.)
    parser.add_argument("--local-weight", type=float, default=1.)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fit-limit", type=int, default=5000)
    parser.add_argument("--components-per-block", type=int, default=64)
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument("--parent-purity", type=float, default=.8)
    parser.add_argument("--min-seed-stability", type=float, default=.8)
    parser.add_argument("--min-scale-agreement", type=float, default=1.)
    parser.add_argument("--min-centroid-margin", type=float, help="Legacy k-means comparison only; default .1 there")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--max-components", type=int, default=1000000)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.sample_id.strip() or args.tile_size < 2 or args.max_components < 2:
        parser.error("Valid sample ID and positive processing bounds required")
    if args.discovery_method == "kodama_graph" and args.min_centroid_margin is not None:
        parser.error("--min-centroid-margin is legacy-only; use --min-affinity-margin for native KODAMA")
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error("Output directory must be new or empty; preserve previous analyses")
    with ExitStack() as stack:
        parent = stack.enter_context(RasterReader(args.parent_mask))
        support = stack.enter_context(RasterReader(args.support_mask)) if args.support_mask else None
        parent_uncertainty = stack.enter_context(RasterReader(args.parent_uncertainty)) if args.parent_uncertainty else None
        geometry = validate_geometry(json.loads(args.grid_metadata.read_text()), json.loads(args.shift_json.read_text()), json.loads(args.resolution_json.read_text()), (parent.height, parent.width))
        support_validation = validate_parent_support(parent, support, parent_uncertainty, args.tile_size)
        definitions = validate_feature_definitions(json.loads(args.embedding_metadata.read_text()), geometry)
        grid = validate_grid(pd.read_csv(args.grid_objects), (parent.height, parent.width))
        grid = assign_parent_domains(grid, parent, args.parent_purity, parent_uncertainty)
        args.outdir.mkdir(parents=True, exist_ok=True)
        inputs = {name: {"path": str(getattr(args, name).resolve()), "sha256": sha256_file(getattr(args, name))} for name in ("parent_mask", "grid_objects", "grid_metadata", "shift_json", "resolution_json", "embedding_metadata")}
        for name in ("image", "support_mask", "parent_uncertainty"):
            if getattr(args, name) is not None:
                inputs[name] = {"path": str(getattr(args, name).resolve()), "sha256": sha256_file(getattr(args, name))}
        with tempfile.TemporaryDirectory(prefix="hierarchy_features_", dir=args.outdir) as temporary:
            blocks, names, provenance = {}, {}, {}
            for name in ("local", "context"):
                blocks[name], names[name], provenance[name] = load_embedding_block(getattr(args, f"{name}_embeddings"), grid, Path(temporary) / f"{name}.npy", definitions[name])
                if args.image is not None:
                    if provenance[name].get("sample_id") != args.sample_id:
                        raise ValueError(f"{name} feature provenance specimen identity differs")
                    for source_name in ("image", "grid_objects", "grid_metadata", "shift_json", "resolution_json"):
                        recorded = provenance[name].get("source_inputs", {}).get(source_name, {}).get("sha256")
                        if recorded != inputs[source_name]["sha256"]:
                            raise ValueError(f"{name} feature provenance {source_name} checksum differs or is missing")
            source_hashes = {entry["path"]: entry["sha256"] for entry in inputs.values()}
            for entry in provenance.values():
                source_hashes.update(entry["files_sha256"])
            discovery_arguments = dict(weights={"local": args.local_weight, "context": args.context_weight},
                max_k=args.max_k, fixed_k=args.fixed_k, seed=args.seed, repeats=args.repeats,
                fit_limit=args.fit_limit, components=args.components_per_block,
                min_observations=args.min_observations, min_seed_stability=args.min_seed_stability,
                min_scale_agreement=args.min_scale_agreement)
            if args.discovery_method == "kodama_graph":
                from hierarchy_kodama import discover_kodama_subdomains, verify_discovery_artifacts
                grid, discovery = discover_kodama_subdomains(grid, blocks, outdir=args.outdir / "kodama_graphs",
                    transform_blocks=_transform_blocks, status_names=STATUS, source_hashes=source_hashes,
                    geometry=geometry, min_margin=args.min_affinity_margin, ncomp=args.kodama_ncomp,
                    m=args.kodama_m, tcycle=args.kodama_tcycle, neighbors=args.kodama_neighbors,
                    cores=args.kodama_cpus, library=args.kodama_r_library, **discovery_arguments)
            else:
                grid, discovery = discover_subdomains(grid, blocks,
                    min_margin=.1 if args.min_centroid_margin is None else args.min_centroid_margin,
                    **discovery_arguments)
                discovery["method"] = "legacy_kmeans_comparison_not_kodama_native"
            mx, my = geometry["mpp_xy"]
            ox, oy = geometry["origin_um_xy"]
            grid["sample_id"] = args.sample_id
            grid["x_um"], grid["y_um"] = grid.x * mx + ox, grid.y * my + oy
            grid.to_csv(args.outdir / "grid_subdomains.csv", index=False)
            subdomains, status, parent_counts = rasterize(grid, parent, args.outdir, geometry, args.tile_size, parent_uncertainty)
            region_mask, stats = connected_regions(subdomains, args.outdir / "region_mask.ome.tif", geometry, args.tile_size, args.max_components)
            identity = {"inputs": {key: value["sha256"] for key, value in inputs.items()}, "definitions": definitions, "parameters": {key: value for key, value in vars(args).items() if not isinstance(value, Path)}, "embedding_sha256": {name: sorted(v["files_sha256"].values()) for name, v in provenance.items()}}
            # Stochastic graph construction can differ even with the same
            # requested seed. A realized hierarchy must not share an identity
            # merely because its settings and source embeddings are equal.
            identity["realized_grid_assignment_sha256"] = sha256_file(args.outdir / "grid_subdomains.csv")
            identity["native_graph_sha256"] = {str(path.relative_to(args.outdir)): sha256_file(path)
                for path in sorted((args.outdir / "kodama_graphs").rglob("kodama_graph.rds"))} if args.discovery_method == "kodama_graph" else {}
            hierarchy_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            regions = export_region_profiles(grid, blocks, names, definitions, region_mask, stats, geometry, args.outdir, args.sample_id, hierarchy_id)
            shutil.copyfile(args.parent_mask, args.outdir / "parent_domains.ome.tif")
            if sha256_file(args.outdir / "parent_domains.ome.tif") != inputs["parent_mask"]["sha256"]:
                raise RuntimeError("Parent copy verification failed")
            if args.parent_uncertainty:
                shutil.copyfile(args.parent_uncertainty, args.outdir / "parent_uncertainty.ome.tif")
                if sha256_file(args.outdir / "parent_uncertainty.ome.tif") != inputs["parent_uncertainty"]["sha256"]:
                    raise RuntimeError("Parent uncertainty copy verification failed")
            status_counts = {}
            for y in range(0, parent.height, args.tile_size):
                for x in range(0, parent.width, args.tile_size):
                    labels, counts = np.unique(status[y:y + args.tile_size, x:x + args.tile_size], return_counts=True)
                    for label, count in zip(labels, counts):
                        status_counts[STATUS[int(label)]] = status_counts.get(STATUS[int(label)], 0) + int(count)
            summary = {"schema_version": SCHEMA_VERSION, "hierarchy_id": hierarchy_id, "sample_id": args.sample_id, "parent_labels_immutable": True, "parent_pixel_counts": parent_counts, "geometry": geometry, "feature_definitions": definitions, "embedding_provenance": provenance, "inputs": inputs, "discovery": discovery, "status_codes": STATUS, "status_pixel_counts": status_counts, "regions": len(regions), "subdomains": int(grid.subdomain_id.nunique() - int((grid.subdomain_id == 0).any())), "grid_observations": len(grid), "semantics": "0 subdomain/region with positive parent is explicitly unresolved parent tissue, not missing tissue; accepted subdomains never overwrite parent/background; regions use 4-connectivity; feature-field comparison is not scale-independent evidence when transformer tokens share context", "memory_bounds": {"raster_tile_size": args.tile_size, "fit_observations": args.fit_limit, "components_per_block": args.components_per_block, "max_provisional_components": args.max_components, "embeddings": "disk-backed; chunked input/transformation; no dense observation-by-observation distance matrix"}}
            summary["support_validation"] = support_validation
            summary["parent_uncertainty_semantics"] = "Input codes preserved byte-for-byte; 0 is accepted, any 1..254 is unresolved/inferred and excluded from accepted subdomain pixels. Input-specific reason codes retain upstream meaning; hierarchy status 10 does not recode the input."
            summary["realized_identity"] = {key: identity[key] for key in ("realized_grid_assignment_sha256", "native_graph_sha256")}
            if any(sha256_file(path) != expected for path, expected in source_hashes.items()):
                raise RuntimeError("Hierarchy source changed during raster/profile export")
            if args.discovery_method == "kodama_graph":
                verify_discovery_artifacts(discovery, args.outdir / "kodama_graphs")
                summary["native_artifacts_unchanged_after_export"] = True
                summary["memory_bounds"].update(
                    fit_limit_scope="PCA/normalization training rows and native landmark count; not a cap on graph observations",
                    native_graph_observations="all complete eligible grid observations in one parent at a time",
                    transformed_features="local/context and combined reduced matrices resident per parent; no dense N-by-N distance matrix")
            summary["sources_unchanged_after_export"] = True
            # tempfile returns an absolute path, whereas Nextflow invokes this
            # CLI with a relative output directory. Compare resolved paths so
            # temporary feature caches never enter the durable output receipt.
            temporary_root = Path(temporary).resolve()
            summary["outputs"] = {str(path.relative_to(args.outdir)): sha256_file(path) for path in sorted(args.outdir.rglob("*")) if path.is_file() and not path.resolve().is_relative_to(temporary_root)}
            write_json(args.outdir / "hierarchy_summary.json", summary)
    print(json.dumps({"parents": len(parent_counts), "regions": len(regions), "outdir": str(args.outdir.resolve())}))


if __name__ == "__main__":
    main()
