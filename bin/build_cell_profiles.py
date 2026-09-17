#!/usr/bin/env python3
"""Join canonical cells and independent feature blocks without dropping cells.

Coordinates in the master table are original-slide micrometres. Pixel columns
from resampled marker images are deliberately not used as spatial coordinates.
Missing feature blocks remain explicit NaN rows; they are never zero-imputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from cell_profile_io import RasterReader, calibration, read_keyed_csv, sha256_file, staged_files


SCHEMA_VERSION = "1.0.0"
BACKGROUND_CHANNELS = {"TRITC", "Cy5"}
COMPARTMENT_SEMANTICS = {"nuclei": "canonical_nuclear_mask", "cyto": "legacy_named_whole_cell_approximation_including_nucleus", "ring": "perinuclear_ring_excluding_all_nuclear_pixels"}


def canonical_cells(objects, sample_id, shift, segmentation_id=None, resolution_json=None):
    frame = read_keyed_csv(objects, "label")
    required = {"x", "y", "xmin", "ymin", "xmax", "ymax"}
    if not required <= set(frame):
        raise ValueError(f"Canonical objects missing columns {sorted(required - set(frame))}")
    numeric = frame[list(required)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite canonical cell geometry")
    if not str(sample_id).strip():
        raise ValueError("sample_id cannot be empty")
    cal = calibration(shift, resolution_json)
    xy = numeric[["x", "y"]].to_numpy(float)
    if ((xy < 0).any() or (xy[:, 0] >= cal["width"]).any() or (xy[:, 1] >= cal["height"]).any()):
        raise ValueError("Canonical centroids outside the declared crop")
    segmentation_id = segmentation_id or sha256_file(objects)
    frame = frame.rename(columns={"label": "cell_id", "x": "x_crop_px", "y": "y_crop_px"})
    # Additive detector provenance is numeric geometry, including when a blank
    # source mapping forces the CSV reader to retain the column as strings.
    # These coordinates never replace the canonical fused x/y above.
    for column in ("cellvitpp_x_px", "cellvitpp_y_px"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
    frame.insert(0, "sample_id", str(sample_id))
    frame.insert(2, "cell_uid", [f"{quote(str(sample_id), safe='')}:{segmentation_id}:{quote(label, safe='')}" for label in frame.cell_id])
    frame["segmentation_id"] = segmentation_id
    frame["source_mpp"] = cal["mpp"]
    frame[["x_um", "y_um"]] = (xy + cal["origin_px"]) * cal["mpp"]
    frame[["x_slide_px", "y_slide_px"]] = xy + cal["origin_px"]
    frame["crop_origin_um_x"], frame["crop_origin_um_y"] = cal["origin_px"] * cal["mpp"]
    # This is a detector taxonomy, not an adjudicated biological cell identity.
    frame["phenotype"] = frame.get("cellvitpp_type", pd.Series("unknown", index=frame.index)).replace("", "unknown")
    frame["phenotype_source"] = "CellViT_prediction" if "cellvitpp_type" in frame else "unavailable"
    return frame, cal


def join_columns(cells, path, key, prefix, include=None):
    other = read_keyed_csv(path, key)
    foreign = set(other[key]) - set(cells.cell_id)
    if foreign:
        raise ValueError(f"{path}: {len(foreign)} foreign cell IDs (example {sorted(foreign)[:3]})")
    fields = [c for c in other if c != key and (include is None or include(c))]
    names = {c: prefix + c for c in fields}
    if set(names.values()) & set(cells):
        raise ValueError(f"{path}: colliding profile columns")
    joined = cells.merge(other[[key] + fields].rename(columns={key: "cell_id", **names}), on="cell_id", how="left", validate="one_to_one", sort=False).copy()
    joined[prefix + "available"] = joined.cell_id.isin(set(other[key]))
    return joined, {"source": str(Path(path).resolve()), "sha256": sha256_file(path), "rows": len(other), "missing_cells": len(cells) - len(other), "columns": list(names.values())}


def verify_uni2_provenance(source, completions, files, expected_inputs, *, observation_type='cell', shard_rows=None, expected_embedding_mode=None):
    """Verify new extraction receipts without importing an encoder runtime."""
    from uni2_embedding_io import binary_shard_paths
    logical = {Path(path).resolve() for path in files}
    payloads = set(logical)
    binary = any(Path(path).name.endswith('.embedding.json') for path in files)
    for path in files:
        if Path(path).name.endswith('.embedding.json'):
            payloads.update(Path(p).resolve() for p in binary_shard_paths(path))
    markers = staged_files(source, '.*_grid_complete.json') if Path(source).is_dir() else []
    marker_records = [(path, json.loads(path.read_text())) for path in markers]
    actual_storage = 'binary' if binary else 'csv'
    for record in [*completions, *(marker for _, marker in marker_records)]:
        if 'embedding_storage' in record and record['embedding_storage'] != actual_storage:
            raise ValueError('UNI-2 completion/grid storage declaration disagrees with actual shard format')
    declared_modes = set()
    binary_contract = binary and any('cache_contract' in completion for completion in completions)

    def record_mode(record, description, *, required=False):
        mode = record.get('embedding_mode')
        if mode is None and not required:
            return
        if mode not in {'tile', 'inner_square', 'nuclei', 'cyto'}:
            raise ValueError(f'UNI-2 {description} requires a valid embedding_mode')
        declared_modes.add(mode)

    for path in files:
        if Path(path).name.endswith('.embedding.json'):
            record_mode(json.loads(Path(path).read_text()), 'binary shard', required=binary_contract)
    for completion in completions:
        record_mode(completion, 'completion', required=binary_contract)
    for _, marker in marker_records:
        record_mode(marker, 'grid receipt', required=binary_contract)
    if len(declared_modes) > 1:
        raise ValueError('UNI-2 embedding_mode conflicts between shard, grid receipt, or completion')
    embedding_mode = next(iter(declared_modes), None)
    if expected_embedding_mode is not None and embedding_mode is not None and embedding_mode != expected_embedding_mode:
        raise ValueError(f'UNI-2 embedding_mode {embedding_mode} cannot be assigned to expected {expected_embedding_mode} profile role')
    # A paired extraction shares the primary cache contract. Its secondary
    # inner-square mode is bound by its own receipts, not parameters.embedding_mode.
    mode_provenance = {'embedding_mode': embedding_mode,
        'status': 'declared_consistent' if embedding_mode is not None else 'not_declared',
        'expected_embedding_mode': expected_embedding_mode}
    expected_inventory = payloads | {p.resolve() for p in markers}
    root_inventory_verified = False
    for completion in completions:
        records = completion.get('payload_inventory')
        if records is None:
            if binary and completion.get('cache_contract', {}).get('parameters', {}).get('embedding_storage') == 'binary':
                raise ValueError('Binary UNI-2 extraction completion requires its exact root payload inventory')
            continue
        if not isinstance(records, list) or not records:
            raise ValueError('UNI-2 root payload inventory must be a nonempty record list')
        inventoried = set()
        for record in records:
            relative = record.get('path') if isinstance(record, dict) else None
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
                raise ValueError('UNI-2 root inventory paths must be safe relative paths')
            candidate = (Path(source) / relative).resolve()
            if candidate not in expected_inventory or candidate in inventoried:
                raise ValueError('UNI-2 root inventory contains an unknown or duplicate payload')
            size = record.get('size_bytes')
            if type(size) is not int or size < 0 or candidate.stat().st_size != size or sha256_file(candidate) != record.get('sha256'):
                raise ValueError('UNI-2 root payload inventory checksum/byte-size mismatch')
            inventoried.add(candidate)
        if inventoried != expected_inventory:
            raise ValueError('UNI-2 root inventory does not cover the exact shard/payload/grid-receipt set')
        root_inventory_verified = True
    storage_provenance = {'logical_shards': len(logical), 'payload_files': len(payloads),
        'root_payload_inventory': 'verified_exact_files' if root_inventory_verified else 'not_provided',
        'interpretation': 'Storage integrity is not encoder provenance or biological validation'}
    if not completions or not any('cache_contract' in c for c in completions):
        return {"status": "legacy_unverified_inputs", 'storage': storage_provenance, 'representation_binding': mode_provenance}, None
    if len(completions) != 1:
        raise ValueError("UNI-2 source must have exactly one extraction completion contract")
    completion = completions[0]
    contract = completion.get('cache_contract', {})
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if contract.get('schema_version') != '2.0.0' or digest != completion.get('cache_contract_sha256'):
        raise ValueError('UNI-2 extraction contract fingerprint/version mismatch')
    declared_storage = contract.get('parameters', {}).get('embedding_storage', 'csv')
    if declared_storage not in {'csv', 'binary'} or (declared_storage == 'binary') != binary:
        raise ValueError('UNI-2 shard storage disagrees with extraction contract; a conversion cannot invent encoder provenance')
    if completion.get('observation_type') != observation_type or contract.get('parameters', {}).get('observation_type') != observation_type:
        raise ValueError(f'UNI-2 extraction contract is not a {observation_type} observation product')
    state = contract.get('encoder_state_sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', state) or state != completion.get('model_provenance', {}).get('runtime_state_sha256'):
        raise ValueError('UNI-2 actual encoder state identity mismatch')
    declared = contract.get('inputs', {})
    verified = []
    for name, expected in (expected_inputs or {}).items():
        if expected and declared.get(name) and declared[name].get('sha256') != expected:
            raise ValueError(f'UNI-2 {name} source differs from canonical profile inputs; re-extract features')
        if expected and (declared.get(name) or {}).get('sha256') == expected:
            verified.append(name)
    inventory = {}
    total = 0
    for path, marker in marker_records:
        records = marker.get('shard_files', [])
        if marker.get('cache_contract_sha256') != digest or not records or len(records) != marker.get('shards'):
            raise ValueError('UNI-2 grid shard inventory/contract mismatch')
        if marker.get('rows_written') != marker.get('index_end', -1) - marker.get('index_start', -1):
            raise ValueError('UNI-2 grid row coverage mismatch')
        total += marker['rows_written']
        observed_rows = 0
        for record in records:
            if 'storage' in record and record['storage'] != actual_storage:
                raise ValueError('UNI-2 grid logical shard storage declaration mismatch')
            name = record.get('name', '')
            candidate = (path.parent / name).resolve()
            if not name or Path(name).name != name or candidate in inventory:
                raise ValueError('UNI-2 duplicate or unsafe shard inventory')
            if not candidate.is_file() or sha256_file(candidate) != record.get('sha256'):
                raise ValueError('UNI-2 shard checksum mismatch')
            if 'size_bytes' in record and (type(record['size_bytes']) is not int or candidate.stat().st_size != record['size_bytes']):
                raise ValueError('UNI-2 shard byte-size mismatch')
            if candidate.name.endswith('.embedding.json') and 'payload_files' in record:
                expected = {p.resolve() for p in binary_shard_paths(candidate)[1:]}
                actual = set()
                for payload in record['payload_files']:
                    payload_name = payload.get('name', '')
                    if not payload_name or Path(payload_name).name != payload_name or '\\' in payload_name:
                        raise ValueError('Unsafe UNI-2 grid binary payload path')
                    target = (candidate.parent / payload_name).resolve()
                    if target not in expected or target in actual or sha256_file(target) != payload.get('sha256') or target.stat().st_size != payload.get('size_bytes'):
                        raise ValueError('UNI-2 grid binary payload inventory mismatch')
                    actual.add(target)
                if actual != expected:
                    raise ValueError('UNI-2 grid binary payload inventory is incomplete')
            inventory[candidate] = record['sha256']
            if shard_rows is not None:
                if candidate not in shard_rows:
                    raise ValueError('UNI-2 receipt names an unread logical shard')
                observed_rows += shard_rows[candidate]
        if shard_rows is not None and observed_rows != marker['rows_written']:
            raise ValueError('UNI-2 grid receipt row count differs from actual shard rows')
    if set(inventory) != {p.resolve() for p in files} or len(markers) != completion.get('completed_grids'):
        raise ValueError('UNI-2 extraction receipt does not cover the exact supplied shards')
    if total != completion.get('rows_written') or total != completion.get('expected_observations'):
        raise ValueError('UNI-2 completion row coverage mismatch')
    complete = {'image', 'mask', 'resolution_json'} <= set(verified)
    reusable = {key: contract.get(key) for key in ('encoder_state_sha256', 'backend', 'pooling', 'preprocessing', 'code_sha256', 'software')}
    reusable['inference_parameters'] = {key: value for key, value in contract.get('parameters', {}).items()
                                       if key not in {'save_tiles', 'tiles_format', 'bucket_size', 'verbose', 'rows_per_csv', 'embedding_storage'}}
    return {'status': 'verified_exact_inputs_and_shards' if complete else 'verified_shards_partial_input_binding',
            'cache_contract_sha256': digest, 'verified_inputs': sorted(verified), 'rows_verified': total,
            'storage': storage_provenance, 'representation_binding': mode_provenance}, reusable


def export_uni2_block(cells, source, name, outdir, expected_inputs=None):
    from uni2_embedding_io import discover_embedding_shards, iter_binary_blocks, binary_shard_paths
    source = Path(source)
    files = discover_embedding_shards(source)
    if not files:
        raise ValueError(f"No embedding shards in requested {source}")
    index = {key: i for i, key in enumerate(cells.cell_id)}
    seen = np.zeros(len(cells), dtype=bool)
    matrix, columns = None, None
    definitions = set()
    outpath = Path(outdir) / f"{name}.npy"
    source_files = list(files)
    storage_formats = set()
    storage_dtypes = set()
    shard_rows = {}
    for file in files:
        shard_rows[file.resolve()] = 0
        binary = file.name.endswith('.embedding.json')
        if binary:
            binary_manifest = json.loads(file.read_text())
            binary_names = binary_manifest['feature_names']
            if not binary_names or not all(isinstance(c, str) and re.fullmatch(r'feat_[1-9][0-9]*', c) for c in binary_names):
                raise ValueError('Binary UNI-2 feature schema requires ordered feat_N names')
            source_files.extend(binary_shard_paths(file))
            blocks = iter_binary_blocks(file, block_rows=1024)
            storage_formats.add('cellphenotyper_uni2_binary')
        else:
            binary_names = None
            blocks = ((block, None) for block in pd.read_csv(file, dtype={"cell_id": str}, chunksize=1024, float_precision='round_trip', keep_default_na=False))
            storage_formats.add('legacy_csv')
        for block, binary_values in blocks:
            shard_rows[file.resolve()] += len(block)
            if "cell_id" not in block or "observation_type" not in block:
                raise ValueError(f"{file}: missing cell ID or observation type; grid/cell identity cannot be inferred")
            if not block.observation_type.eq("cell").all():
                raise ValueError(f"{file}: grid observations cannot be joined as cells")
            features = binary_names if binary else [c for c in block if c.startswith("feat_")]
            if not features or (columns is not None and features != columns):
                raise ValueError("Inconsistent or empty UNI-2 feature schema")
            if matrix is None:
                columns = features
                matrix = np.lib.format.open_memmap(outpath, mode="w+", dtype="float32", shape=(len(cells), len(columns)))
                matrix[:] = np.nan
            ids = block.cell_id.astype(str).tolist()
            if len(set(ids)) != len(ids) or any(key not in index for key in ids):
                raise ValueError(f"{file}: duplicate or foreign embedding IDs")
            rows = np.array([index[key] for key in ids], dtype=int)
            if seen[rows].any():
                raise ValueError(f"{file}: repeated embedding IDs across shards")
            if binary:
                storage_dtypes.add(str(binary_values.dtype))
                values = np.asarray(binary_values, dtype=np.float32)
            else:
                values = block[features].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{file}: nonfinite embedding vector")
            if {"cx", "cy"} <= set(block):
                expected = cells.iloc[rows][["x_crop_px", "y_crop_px"]].to_numpy(float)
                # Extraction centres are integer-rounded from raster centroids;
                # polygon and raster centroids can differ slightly.
                distance = np.linalg.norm(block[["cx", "cy"]].to_numpy(float) - expected, axis=1)
                if not np.isfinite(distance).all() or np.any(distance > 5.0):
                    raise ValueError(f"{file}: embedding/canonical coordinate mismatch >5 crop pixels")
            if "source_mpp" in block:
                recorded_mpp = pd.to_numeric(block.source_mpp, errors="raise").to_numpy(float)
                expected_mpp = cells.iloc[rows].source_mpp.to_numpy(float)
                if not np.isfinite(recorded_mpp).all() or not np.allclose(recorded_mpp, expected_mpp, rtol=1e-5, atol=1e-8):
                    raise ValueError(f"{file}: embedding source MPP conflicts with canonical calibration; re-extract features")
            else:
                definitions.add(("source_calibration_verified", ("false",)))
            for fields in ("model_tile_size", "target_mpp", "mask_context_mode", "effective_mpp"):
                if fields in block:
                    normalized = set()
                    for value in block[fields].drop_duplicates():
                        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                            normalized.add('unavailable')
                        elif fields == 'mask_context_mode':
                            normalized.add(str(value))
                        else:
                            number = float(value)
                            if not np.isfinite(number) or number <= 0:
                                raise ValueError(f'{file}: nonpositive/nonfinite UNI-2 physical setting {fields}')
                            normalized.add(str(int(number)) if number.is_integer() else str(number))
                    definitions.add((fields, tuple(sorted(normalized))))
            matrix[rows] = values
            seen[rows] = True
    if matrix is None:
        raise ValueError(f"Requested embedding source {source} has no rows")
    matrix.flush()
    completion_files = staged_files(source, ".*_embedding_complete.json") if source.is_dir() else []
    completions = [json.loads(p.read_text()) for p in completion_files]
    expected_mode = {'uni2_context': 'tile', 'uni2_local': 'inner_square'}.get(name)
    provenance, execution_definition = verify_uni2_provenance(source, completions, files, expected_inputs, shard_rows=shard_rows, expected_embedding_mode=expected_mode)
    source_files.extend(completion_files)
    source_files.extend(staged_files(source, '.*_grid_complete.json') if source.is_dir() else [])
    feature_definitions = [{"encoder": c.get("encoder"), "embedding_mode": c.get("embedding_mode"),
                            "paired_inner_square_mode": c.get("paired_inner_square_mode"),
                            "model": model_identity(c.get("model_provenance", {}))} for c in completions]
    unique_definitions = {json.dumps(d, sort_keys=True) for d in feature_definitions}
    if len(unique_definitions) > 1:
        raise ValueError("Embedding source contains incompatible completion/model definitions")
    preprocessing = {}
    for key, values in definitions:
        preprocessing.setdefault(key, set()).update(values)
    if any(len(values) != 1 for values in preprocessing.values()):
        raise ValueError("Embedding source mixes incompatible physical/context settings")
    definition = {"representation": feature_definitions[0] if feature_definitions else None,
                  "verified_execution": execution_definition,
                  "preprocessing": {k: sorted(v)[0] for k, v in sorted(preprocessing.items())}}
    compatible = bool(feature_definitions and feature_definitions[0]["model"]["verified"]
                      and {"model_tile_size", "target_mpp", "mask_context_mode", "effective_mpp"} <= set(preprocessing)
                      and "source_calibration_verified" not in preprocessing)
    compatible = compatible and provenance['status'] == 'verified_exact_inputs_and_shards'
    compatible = compatible and not any('unavailable' in values for values in preprocessing.values())
    compatible = compatible and (expected_mode is None or provenance['representation_binding']['embedding_mode'] == expected_mode)
    sources = sorted({Path(p).resolve() for p in source_files}, key=str)
    return seen, {"path": outpath.name, "sha256": sha256_file(outpath), "shape": list(matrix.shape), "dtype": "float32", "feature_names": columns, "missing_cells": int((~seen).sum()), "feature_definition": definition, "reference_compatible": compatible, "provenance": provenance,
        "storage": {"input_formats": sorted(storage_formats), "binary_input_dtypes": sorted(storage_dtypes),
                    "output_dtype": "float32", "conversion": "Existing profile float32 semantics; storage format is not biological feature identity"},
        "sources": [{"path": str(p), "sha256": sha256_file(p)} for p in sources]}


def model_identity(provenance):
    """A portable model definition: local cache paths are deliberately excluded."""
    checkpoints = sorted((str(c.get("logical_name", "")), str(c.get("sha256", ""))) for c in provenance.get("checkpoints", []))
    verified = bool(provenance.get("resolved_revision") and checkpoints and all(len(d) == 64 for _, d in checkpoints))
    return {"source_repository": provenance.get("source_repository"), "resolved_revision": provenance.get("resolved_revision"),
            "checkpoints": checkpoints, "verified": verified}


def export_cellvit_block(cells, source, outdir, expected_inputs=None, expected_geometry=None):
    """Attach detector vectors using source-bound correspondence, not fused XY.

    Source verification and canonical correspondence are separate requirements:
    an older objects table or vector bundle can remain inspectable, but cannot
    establish reference-compatible learned features from local IDs alone.
    """
    from cellvit_embedding_io import load_cellvit_embedding_bundle

    data, ids, binding, sourcefiles = load_cellvit_embedding_bundle(
        source, expected_inputs=expected_inputs, expected_geometry=expected_geometry)
    source_hashes = {name: sha256_file(path) for name, path in sourcefiles.items()}
    # Anchor the attachment snapshot to the hashes already verified by the
    # loader, so a replacement between its return and this snapshot cannot
    # become the new baseline for the post-attachment check.
    receipt = binding.get("receipt")
    if receipt is not None:
        if any(source_hashes.get(name) != artifact["sha256"]
               for name, artifact in receipt["files"].items()):
            raise ValueError("CellViT source artifacts differ from their validated completion receipt")
        if json.loads(sourcefiles["receipt"].read_bytes()) != receipt:
            raise ValueError("CellViT completion changed before canonical profile attachment")
    if "cellvitpp_id" not in cells:
        raise ValueError("Canonical cells have no CellViT source-ID mapping")
    if data.ndim != 2 or data.shape[0] != len(ids):
        raise ValueError("CellViT matrix and ID table dimensions differ")
    rows = pd.to_numeric(ids.embedding_row, errors="raise").to_numpy(int)
    if sorted(rows.tolist()) != list(range(len(ids))):
        raise ValueError("CellViT embedding_row must be a bijection to matrix rows")
    canonical_ids = cells.cellvitpp_id.astype(str)
    nonempty = canonical_ids[canonical_ids.ne("")]
    if not nonempty.eq(nonempty.str.strip()).all():
        raise ValueError("Canonical CellViT IDs contain leading/trailing whitespace")
    if nonempty.duplicated().any():
        raise ValueError("One CellViT source cell maps to multiple canonical cells")
    source_index = dict(zip(ids.cellvitpp_id, rows))
    fields = {"cellvitpp_x_px", "cellvitpp_y_px", "cellvitpp_source_sha256"}
    present = fields & set(cells)
    correspondence = {"status": "legacy_unverified_canonical_correspondence",
        "reason": "Canonical objects lack detector-centroid and source-population binding"}
    if present and present != fields:
        raise ValueError("Incomplete canonical CellViT correspondence fields")
    if present:
        mapped = canonical_ids.ne("")
        if not cells.loc[~mapped, sorted(fields)].replace("", np.nan).isna().all().all():
            raise ValueError("Canonical CellViT correspondence exists without a source ID")
        declared_hashes = cells.loc[mapped, "cellvitpp_source_sha256"].astype(str)
        if not declared_hashes.map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", value))).all():
            raise ValueError("Canonical CellViT source-population SHA256 is missing or malformed")
        retained_hash = binding.get("retained_population_sha256")
        if retained_hash is not None and not declared_hashes.eq(retained_hash).all():
            raise ValueError("Canonical CellViT source population differs from embedding population")
        if set(nonempty) - set(source_index):
            raise ValueError("Canonical CellViT source ID is absent from its declared retained population")
        if not {"x_px", "y_px"} <= set(ids):
            raise ValueError("CellViT vectors lack detector-centroid coordinates required by canonical objects")
        detector_xy = cells.loc[mapped, ["cellvitpp_x_px", "cellvitpp_y_px"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
        expected_xy = ids.set_index("cellvitpp_id").loc[canonical_ids[mapped], ["x_px", "y_px"]].to_numpy(float)
        # Both sides are the same detector coordinates serialized independently,
        # not raster/fusion centroids. Permit only decimal CSV round-trip noise.
        if (not np.isfinite(detector_xy).all() or not np.isfinite(expected_xy).all()
                or not np.allclose(detector_xy, expected_xy, rtol=0, atol=1e-6)):
            raise ValueError("Canonical CellViT detector centroids differ from source embedding rows")
        correspondence = {"status": "verified_source_ids_detector_centroids_and_population" if retained_hash is not None else "unverified_source_population",
            "source_population_sha256": retained_hash,
            "centroid_absolute_tolerance_px": 1e-6,
            "canonical_fused_centroid_used_for_matching": False}
    matrix = np.lib.format.open_memmap(Path(outdir) / "cellvit.npy", mode="w+", dtype="float32", shape=(len(cells), data.shape[1]))
    matrix[:] = np.nan
    found = np.zeros(len(cells), dtype=bool)
    for start in range(0, len(cells), 2048):
        indices = [i for i in range(start, min(start + 2048, len(cells))) if canonical_ids.iloc[i] in source_index]
        if indices:
            values = data[[source_index[canonical_ids.iloc[i]] for i in indices]]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite CellViT feature vector")
            matrix[indices] = values
            found[indices] = True
    matrix.flush()
    for name, path in sourcefiles.items():
        if sha256_file(path) != source_hashes[name]:
            raise ValueError("CellViT source artifacts changed during canonical profile attachment")
    verified = (binding.get("reference_compatible") is True
        and binding.get("status") == "verified_exact_inputs_and_population"
        and correspondence["status"] == "verified_source_ids_detector_centroids_and_population")
    return found, {"path": "cellvit.npy", "sha256": sha256_file(Path(outdir) / "cellvit.npy"),
        "shape": list(matrix.shape), "dtype": "float32", "feature_definition": binding.get("feature_definition"),
        "reference_compatible": bool(verified), "source_binding": binding,
        "canonical_correspondence": correspondence, "missing_cells": int((~found).sum()),
        "source_cells_not_canonical": len(set(ids.cellvitpp_id) - set(nonempty)),
        "sources": [{"role": name, "path": str(Path(path).resolve()), "sha256": source_hashes[name]}
                    for name, path in sourcefiles.items()]}


def export_table_block(cells, columns, name, arrays, definition, compatible=True):
    values = cells[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError(f"Infinite values in {name}")
    path = arrays / f"{name}.npy"
    np.save(path, values, allow_pickle=False)
    return {"path": f"feature_blocks/{path.name}", "sha256": sha256_file(path), "shape": list(values.shape),
            "dtype": "float32", "feature_names": columns, "feature_definition": definition,
            "reference_compatible": compatible, "missing_cells": int((~np.isfinite(values).all(axis=1)).sum())}


def marker_definition(table, compartment, semantic, cal, compartment_definition=None, mask_hashes=None):
    """Verify a quantification sidecar and separate reusable from specimen identity.

    Legacy tables stay inspectable but cannot silently become reference features.
    No image/model runtime is imported to validate this small JSON contract.
    """
    summary_path = Path(table).with_name(Path(table).name.replace("_quantification.csv", "_intensity_summary.json"))
    definition = {"compartment": semantic, "value_semantics": "uncalibrated_virtual_marker_score", "source_model": "unverified"}
    if not summary_path.is_file():
        return definition, False, {"authority_status": "legacy_unverified", "schema_available": False}
    summary = json.loads(summary_path.read_text())
    authority = summary.get("authority_status", "legacy_unverified")
    evidence = {"authority_status": authority, "summary_sha256": sha256_file(summary_path), "schema_available": bool(summary.get("marker_schema"))}
    schema = summary.get("marker_schema")
    if not schema:
        return definition, False, evidence
    digest = hashlib.sha256(json.dumps({k: v for k, v in schema.items() if k != "schema_sha256"}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if schema.get("schema_version") != "cellphenotyper.gigatime.v1" or schema.get("schema_sha256") != digest:
        raise ValueError(f"{table}: marker schema fingerprint/version mismatch")
    names = schema.get("marker_names", [])
    fields = pd.read_csv(table, nrows=0).columns
    observed = [c.removesuffix("__mean") for c in fields if c.endswith("__mean")]
    if observed != names or len(names) != 23 or len(set(names)) != len(names):
        raise ValueError(f"{table}: quantification marker order differs from versioned 23-channel schema")
    if set(schema.get("default_phenotype_excluded_channels", [])) != BACKGROUND_CHANNELS:
        raise ValueError(f"{table}: marker background exclusion schema mismatch")
    compartment_record = schema.get("compartments", {}).get(compartment)
    if not compartment_record or not re.fullmatch(r"[0-9a-f]{64}", str(compartment_record.get("mask_sha256", ""))):
        raise ValueError(f"{table}: missing compartment mask identity")
    if compartment_record.get("semantics") != COMPARTMENT_SEMANTICS[compartment]:
        raise ValueError(f"{table}: compartment semantics mismatch")
    expected_mask_hash = (mask_hashes or {}).get(compartment)
    if expected_mask_hash and compartment_record["mask_sha256"] != expected_mask_hash:
        raise ValueError(f"{table}: marker compartment mask differs from canonical profile masks")
    status = pd.read_csv(table, usecols=["quantification_status"], dtype=str, keep_default_na=False) if "quantification_status" in fields else None
    if status is None or not status.quantification_status.eq(authority).all():
        raise ValueError(f"{table}: row quantification status conflicts with authority summary")
    contract = schema.get("coordinate_contract", {})
    source_mpp = contract.get("source_mpp")
    if source_mpp is None or not np.isclose(float(source_mpp), cal["mpp"], rtol=1e-5, atol=1e-8):
        raise ValueError(f"{table}: marker source MPP conflicts with canonical calibration")
    if contract.get("original_shape_yx") != [cal["height"], cal["width"]]:
        raise ValueError(f"{table}: marker source dimensions conflict with canonical crop")
    evidence.update({"schema_sha256": digest, "compartment_mask_sha256": compartment_record["mask_sha256"]})
    keys = ("schema_version", "marker_names", "channel_roles", "default_phenotype_excluded_channels", "checkpoint_sha256", "prediction_precision", "reduction_precision", "model_arithmetic", "prediction_settings", "value_semantics")
    definition = {k: schema.get(k) for k in keys}
    definition.update({"compartment": semantic, "compartment_semantics": compartment_record["semantics"],
                       "effective_mpp": contract.get("effective_mpp"), "resolution_contract": contract.get("resolution_contract")})
    if compartment != "nuclei":
        definition["compartment_construction"] = compartment_definition
    verified = (authority in {"authoritative_full_precision_integrated", "equivalent_float32_restart"}
                and bool(re.fullmatch(r"[0-9a-f]{64}", str(schema.get("checkpoint_sha256", ""))))
                and schema.get("prediction_precision") == "float32" and schema.get("reduction_precision") == "float64"
                and schema.get("model_arithmetic") not in (None, "unspecified")
                and bool(schema.get("prediction_settings")) and contract.get("effective_mpp") is not None
                and bool(expected_mask_hash) and (compartment == "nuclei" or compartment_definition is not None))
    return definition, verified, evidence


def attach_domains(cells, mask_path, cal, uncertainty_path=None):
    xy = cells[["x_crop_px", "y_crop_px"]].to_numpy(float)
    with RasterReader(mask_path) as reader:
        x = np.floor(xy[:, 0] * reader.width / cal["width"]).astype(int)
        y = np.floor(xy[:, 1] * reader.height / cal["height"]).astype(int)
        cells["tissue_domain"] = reader.sample(x, y)
    cells["tissue_domain_status"] = np.where(cells.tissue_domain.eq(0), "unknown_or_outside_support", "assigned_uncertainty_unavailable")
    if uncertainty_path:
        with RasterReader(uncertainty_path) as reader:
            x = np.floor(xy[:, 0] * reader.width / cal["width"]).astype(int)
            y = np.floor(xy[:, 1] * reader.height / cal["height"]).astype(int)
            cells["tissue_uncertainty_code"] = reader.sample(x, y)
        cells.loc[cells.tissue_uncertainty_code.ne(0), "tissue_domain_status"] = "uncertain"
        cells.loc[cells.tissue_uncertainty_code.eq(0) & cells.tissue_domain.ne(0), "tissue_domain_status"] = "assigned"
    return cells


def build_profiles(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "cell_profiles_manifest.json").exists():
        raise FileExistsError("Output already contains a completed cell profile; use a separate output directory")
    arrays = outdir / "feature_blocks"
    arrays.mkdir(exist_ok=True)
    resolution_json = getattr(args, "resolution_json", None)
    labels = getattr(args, "labels", None)
    labels_sha256 = sha256_file(labels) if labels else None
    segmentation_id = args.segmentation_id
    if labels_sha256 and not segmentation_id:
        segmentation_id = hashlib.sha256((sha256_file(args.objects) + ":" + labels_sha256).encode()).hexdigest()
    cells, cal = canonical_cells(args.objects, args.sample_id, args.shift, segmentation_id, resolution_json)
    manifest = {"schema_version": SCHEMA_VERSION, "observation_unit": "cell", "sample_id": args.sample_id, "cell_count": len(cells), "coordinate_system": "original_slide_micrometres", "source_mpp": cal["mpp"], "calibration_source": cal["calibration_source"], "crop_origin_um": (cal["origin_px"] * cal["mpp"]).tolist(), "crop_size_px": [cal["width"], cal["height"]], "join_policy": "canonical_left_join_no_imputation", "tables": {}, "feature_blocks": {}, "inputs": {"objects_sha256": sha256_file(args.objects), "shift_sha256": sha256_file(args.shift)}, "interpretation": "H&E-derived features and virtual marker scores are predictions; detector agreement and clustering stability are not calibrated probabilities."}
    if resolution_json:
        manifest["inputs"]["resolution_json_sha256"] = sha256_file(resolution_json)
    if labels_sha256:
        manifest["inputs"]["labels_sha256"] = labels_sha256
    if getattr(args, "image", None):
        manifest["inputs"]["image_sha256"] = sha256_file(args.image)
    manifest["segmentation_identity_basis"] = "explicit_override" if args.segmentation_id else ("objects_and_label_raster_sha256" if labels_sha256 else "legacy_objects_table_sha256_only")
    if args.morphology:
        from cell_morphology_io import load_morphology_table, check_sources, GEOMETRY_FEATURES, TEXTURE_FEATURES
        morphology_inputs = {name: manifest["inputs"].get(field) for name, field in
            (("image", "image_sha256"), ("labels", "labels_sha256"), ("objects", "objects_sha256"),
             ("shift", "shift_sha256"), ("resolution_json", "resolution_json_sha256"))}
        morphology_geometry = {"mpp_xy": [cal["mpp"], cal["mpp"]], "crop_origin_px": cal["origin_px"].tolist(),
                              "crop_size_px": [cal["width"], cal["height"]]}
        measured, morphology_binding, morphology_sources = load_morphology_table(args.morphology,
            expected_inputs=morphology_inputs, expected_geometry=morphology_geometry, expected_ids=cells.cell_id.tolist())
        check_sources(morphology_sources, morphology_binding["source_sha256"])
        morphology_input_paths = {name: getattr(args, name) for name in
            morphology_binding.get("receipt", {}).get("sources", {})}
        if morphology_input_paths:
            check_sources(morphology_input_paths, morphology_binding["receipt"]["sources"])
        if set(measured.label) - set(cells.cell_id):
            raise ValueError("Morphology table contains foreign canonical cell IDs")
        names = {name: "morphology__" + name for name in measured if name != "label"}
        if set(names.values()) & set(cells):
            raise ValueError("Morphology table contains colliding profile columns")
        cells = cells.merge(measured.rename(columns={"label": "cell_id", **names}), on="cell_id",
                            how="left", validate="one_to_one", sort=False).copy()
        cells["morphology__available"] = cells.cell_id.isin(set(measured.label))
        manifest["tables"]["morphology"] = {"source": str(Path(args.morphology).resolve()),
            "sha256": morphology_binding["source_sha256"]["table"]["sha256"], "rows": len(measured),
            "missing_cells": len(cells) - len(measured), "columns": list(names.values()),
            "binding": morphology_binding}
        if "morphology__orientation_rad" in cells:
            cells["orientation_rad"] = pd.to_numeric(cells["morphology__orientation_rad"], errors="raise")
        morphology_columns = ["morphology__" + name for name in GEOMETRY_FEATURES if "morphology__" + name in cells]
        if morphology_columns:
            manifest["feature_blocks"]["morphology"] = export_table_block(cells, morphology_columns, "morphology", arrays,
                morphology_binding["feature_definitions"]["morphology"], morphology_binding["reference_compatible"])
        texture_columns = ["morphology__" + name for name in TEXTURE_FEATURES if "morphology__" + name in cells]
        if texture_columns:
            manifest["feature_blocks"]["nuclear_texture"] = export_table_block(cells, texture_columns, "nuclear_texture", arrays,
                morphology_binding["feature_definitions"]["nuclear_texture"], morphology_binding["reference_compatible"])
        check_sources(morphology_sources, morphology_binding["source_sha256"])
        if morphology_input_paths:
            check_sources(morphology_input_paths, morphology_binding["receipt"]["sources"])
    else:
        cells["morphology__available"] = False
    compartment_definition = None
    mask_hashes = {"nuclei": labels_sha256} if labels_sha256 else {}
    if getattr(args, "compartment_qc", None):
        cells, manifest["tables"]["compartment_qc"] = join_columns(cells, args.compartment_qc, "label", "compartment__")
        if manifest["tables"]["compartment_qc"]["missing_cells"]:
            raise ValueError("Compartment QC must account for every canonical cell")
        summary_path = Path(args.compartment_qc).with_name("compartment_summary.json")
        if not summary_path.is_file():
            raise ValueError("Compartment QC requires its construction summary")
        summary = json.loads(summary_path.read_text())
        if not all(np.isclose(float(summary.get(k, 0)), cal["mpp"], rtol=1e-5) for k in ("mpp_x", "mpp_y")):
            raise ValueError("Compartment QC calibration conflicts with canonical cells")
        if summary.get("label_frame") != "crop" or not np.allclose(summary.get("crop_offset_xy", []), cal["origin_px"]):
            raise ValueError("Compartment QC coordinate frame conflicts with canonical crop")
        construction = {k: summary[k] for k in ("schema_version", "expand_um", "distance_metric", "semantics")}
        verified = False
        if summary.get("inputs") and summary.get("output_artifacts"):
            expected_inputs = {"labels": labels_sha256, "shift": sha256_file(args.shift),
                               "resolution": sha256_file(resolution_json) if resolution_json else None}
            if getattr(args, "tissue_mask", None):
                expected_inputs["tissue_mask"] = sha256_file(args.tissue_mask)
            for name, expected in expected_inputs.items():
                if expected and summary["inputs"].get(name, {}).get("sha256") != expected:
                    raise ValueError(f"Compartment {name} provenance differs from canonical profile inputs")
            verified = all(expected_inputs.values()) and "tissue_mask" in expected_inputs
            bundle = Path(args.compartment_qc).parent
            for key, artifact, filename in (("cyto", "whole_cell_bundle", "labels_whole_cell.tif"), ("ring", "perinuclear_ring", "labels_perinuclear_ring.tif")):
                actual = sha256_file(bundle / filename)
                if actual != summary["output_artifacts"].get(artifact, {}).get("sha256"):
                    raise ValueError(f"Compartment {key} output mask fingerprint mismatch")
                if verified:
                    mask_hashes[key] = actual
            if sha256_file(args.compartment_qc) != summary["output_artifacts"].get("qc", {}).get("sha256"):
                raise ValueError("Compartment QC fingerprint mismatch")
            if verified:
                compartment_definition = construction
        manifest["tables"]["compartment_qc"].update({"definition": construction, "summary_sha256": sha256_file(summary_path), "mask_lineage_verified": bool(verified)})
    else:
        cells["compartment__available"] = False
    if args.marker_quant_dir:
        matched_tables = 0
        for compartment, semantic in (("nuclei", "nucleus"), ("cyto", "whole_cell_proxy"), ("ring", "perinuclear_ring")):
            files = staged_files(args.marker_quant_dir, f"*_{compartment}_gigatime_quantification.csv")
            if len(files) > 1:
                raise ValueError(f"Ambiguous {compartment} marker tables")
            if files:
                matched_tables += 1
                prefix = f"predicted__{semantic}__"
                cells, manifest["tables"][semantic] = join_columns(cells, files[0], "label_id", prefix, lambda c: c.endswith(("__mean", "__sum", "__max")) or c == "quantification_status")
                manifest["tables"][semantic]["value_semantics"] = "uncalibrated_virtual_marker_score"
                definition, compatible, evidence = marker_definition(files[0], compartment, semantic, cal, compartment_definition, mask_hashes)
                manifest["tables"][semantic]["provenance"] = evidence
                manifest["tables"][semantic]["background_columns"] = [c for c in manifest["tables"][semantic]["columns"] if c.removeprefix(prefix).split("__")[0] in BACKGROUND_CHANNELS]
                columns = [c for c in manifest["tables"][semantic]["columns"] if c.endswith("__mean") and c.removeprefix(prefix).split("__")[0] not in BACKGROUND_CHANNELS]
                numeric_columns = [c for c in manifest["tables"][semantic]["columns"] if c.endswith(("__mean", "__sum", "__max"))]
                cells[numeric_columns] = cells[numeric_columns].replace({"": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan}).apply(pd.to_numeric, errors="raise")
                if columns:
                    manifest["feature_blocks"][f"markers_{semantic}"] = export_table_block(cells, columns, f"markers_{semantic}", arrays,
                        definition, compatible=compatible)
        if not matched_tables:
            raise ValueError("Requested marker directory contains no canonical quantification tables")
    marker_columns = [c for c in cells if c.startswith("predicted__") and c.endswith("__mean") and c.split("__")[2] not in BACKGROUND_CHANNELS]
    manifest["biological_marker_features"] = marker_columns
    cells["markers_available"] = cells[marker_columns].notna().any(axis=1) if marker_columns else False
    for name, path in (("uni2_context", args.uni2_context), ("uni2_local", args.uni2_local)):
        cells[name + "_available"] = False
        if path:
            source_inputs = {key: manifest['inputs'].get(field) for key, field in
                             (('image', 'image_sha256'), ('mask', 'labels_sha256'),
                              ('objects_csv', 'objects_sha256'), ('resolution_json', 'resolution_json_sha256'))}
            found, record = export_uni2_block(cells, path, name, arrays, source_inputs)
            cells[name + "_available"] = found
            record["path"] = "feature_blocks/" + record["path"]
            manifest["feature_blocks"][name] = record
    cells["cellvit_available"] = False
    cells["cellvit_status"] = "not_requested"
    if args.cellvit_features:
        source_inputs = {key: manifest['inputs'].get(field) for key, field in
                         (('image', 'image_sha256'), ('shift', 'shift_sha256'), ('resolution_json', 'resolution_json_sha256'))}
        source_geometry = {"source_mpp": cal["mpp"], "crop_size_px": [cal["width"], cal["height"]],
                           "crop_origin_px": cal["origin_px"].tolist()}
        found, record = export_cellvit_block(cells, args.cellvit_features, arrays, source_inputs, source_geometry)
        cells["cellvit_available"] = found
        cells["cellvit_status"] = np.where(found,
            "available_verified_reference_definition" if record["reference_compatible"] else "available_unverified_for_reference",
            np.where(cells.cellvitpp_id.eq(""), "missing_no_cellvit_source", "missing_source_vector"))
        record["path"] = "feature_blocks/" + record["path"]
        manifest["feature_blocks"]["cellvit"] = record
    from cell_phenotype_io import phenotype_definition
    manifest["phenotype_definition"] = phenotype_definition(
        cells, manifest["feature_blocks"].get("cellvit"), source=args.cellvit_features)
    if args.domain_mask:
        cells = attach_domains(cells, args.domain_mask, cal, args.domain_uncertainty)
        manifest["inputs"]["domain_mask_sha256"] = sha256_file(args.domain_mask)
        if args.domain_uncertainty:
            manifest["inputs"]["domain_uncertainty_sha256"] = sha256_file(args.domain_uncertainty)
    else:
        cells["tissue_domain"] = 0
        cells["tissue_domain_status"] = "unavailable"
    cells.to_csv(outdir / "cell_profiles.csv", index=False)
    cells.to_parquet(outdir / "cell_profiles.parquet", index=False)
    cells[["sample_id", "cell_id", "cell_uid"]].to_csv(outdir / "feature_rows.csv", index=False)
    manifest["availability"] = {c: int(cells[c].sum()) for c in cells if c.endswith("_available") or c.endswith("__available")}
    manifest["files"] = {name: sha256_file(outdir / name) for name in ("cell_profiles.csv", "cell_profiles.parquet", "feature_rows.csv")}
    if args.morphology:
        check_sources(morphology_sources, morphology_binding["source_sha256"])
        if morphology_input_paths:
            check_sources(morphology_input_paths, morphology_binding["receipt"]["sources"])
    (outdir / "cell_profiles_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return cells, manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("objects", "sample-id", "shift", "outdir"):
        p.add_argument("--" + name, required=True)
    for name in ("segmentation-id", "image", "labels", "tissue-mask", "resolution-json", "morphology", "compartment-qc", "marker-quant-dir", "uni2-context", "uni2-local", "cellvit-features", "domain-mask", "domain-uncertainty"):
        p.add_argument("--" + name, default=None)
    return p


if __name__ == "__main__":
    options = parser().parse_args()
    _, result = build_profiles(options)
    print(f"Cell profiles: {result['cell_count']} canonical cells; blocks={list(result['feature_blocks'])}")
