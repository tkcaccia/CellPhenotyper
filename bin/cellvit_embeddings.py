#!/usr/bin/env python3
"""Validate and export CellViT 1.0.9 graph tokens without pickle in outputs.

The pinned cellvit-1.0.9 PyPI wheel (SHA256
0dc07a8cb5dd4318882f5200d60170d3326de6de8514cae6f6dafe0a8230b699)
supports --graph. Its inference.py applies the same keep_idx to cells, tokens,
and positions before writing cells.json and cells.pt. postprocessing_numpy.py
averages feature tokens within each detected nucleus bounding box. These are
cell-associated morphology features, not isolated-cell molecular measurements.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cellvit_embedding_io import (FORMAT, VERSION, FILES, RAW_COLUMNS, REPRESENTATION,
                                  file_record, population, read_json)


@dataclass
class CellGraphDataWSI:
    """Inert schema used by PyTorch's restricted weights-only unpickler."""

    x: object
    positions: object
    metadata: object
    nuclei_types: object


# Match the serialized type without importing CellViT's separate environment.
# No methods from the upstream class are executed during this conversion.
CellGraphDataWSI.__module__ = "cellvit.data.dataclass.cell_graph"


def validate_graph_option(help_text: str) -> None:
    if re.search(r"(?<![\w-])--graph(?![\w-])", help_text) is None:
        raise RuntimeError(
            "CellViT embedding export requested, but the executable does not advertise "
            "the required --graph option. Use the supported cellvit==1.0.9 runtime."
        )


def check_graph_support(executable: str, env: dict | None = None) -> dict:
    """Check the actual executable, including when it is in another environment."""
    try:
        result = subprocess.run(
            [str(executable), "--help"], capture_output=True, text=True,
            timeout=60, check=True, env=env,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("Unable to verify CellViT --graph support before inference") from exc
    help_text = result.stdout + result.stderr
    validate_graph_option(help_text)
    return {
        "graph_cli_flag": "--graph", "graph_cli_verified": True,
        "help_sha256": hashlib.sha256(help_text.encode()).hexdigest(),
        "required_graph_schema_contract": "cellvit==1.0.9",
        "upstream_source": "https://pypi.org/project/cellvit/1.0.9/",
    }


def cells_in_order(payload: object) -> list[tuple[str, np.ndarray]]:
    return population(payload)


def align_embeddings(
    raw_payload: object, retained_payload: object,
    features: np.ndarray, positions: np.ndarray,
) -> tuple[np.ndarray, list[dict], dict]:
    """Map filtered IDs to graph rows and verify upstream order using positions."""
    raw = cells_in_order(raw_payload)
    retained = cells_in_order(retained_payload)
    features = np.asarray(features)
    positions = np.asarray(positions, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("CellViT graph features must be a nonempty-dimensional N x D matrix")
    if features.dtype.kind not in "fiu":
        raise ValueError("CellViT graph features must be numeric")
    if len(features) != len(raw):
        raise ValueError(
            f"Missing or extra CellViT vectors: graph has {len(features)} rows, "
            f"cells.json has {len(raw)} cells"
        )
    for start in range(0, len(features), 4096):
        if not np.isfinite(features[start:start + 4096]).all():
            raise ValueError("CellViT raw embeddings contain nonfinite values")
    if positions.shape != (len(raw), 2) or not np.isfinite(positions).all():
        raise ValueError("CellViT graph positions must be finite with shape N x 2")
    raw_positions = np.asarray([xy for _, xy in raw], dtype=np.float64).reshape(-1, 2)
    if not np.allclose(positions, raw_positions, atol=0.05, rtol=0):
        raise ValueError("CellViT graph positions do not align with cells.json row order")
    lookup = {cell_id: i for i, (cell_id, _) in enumerate(raw)}
    selected = []
    rows = []
    for cell_id, xy in retained:
        if cell_id not in lookup:
            raise ValueError(f"Missing CellViT embedding for retained cell {cell_id}")
        index = lookup[cell_id]
        if not np.array_equal(xy, raw_positions[index]):
            raise ValueError(f"Retained cell {cell_id} centroid differs from upstream cell")
        selected.append(index)
        rows.append({
            "embedding_row": len(rows), "cellvitpp_id": cell_id,
            "x_px": float(xy[0]), "y_px": float(xy[1]), "source_graph_row": index,
        })
    selected_features = np.asarray(features[np.asarray(selected, dtype=np.int64)], dtype=np.float32)
    if not np.isfinite(selected_features).all():
        raise ValueError("CellViT retained embeddings contain missing or nonfinite values")
    summary = {
        "raw_cell_count": len(raw), "retained_cell_count": len(retained),
        "excluded_cell_count": len(raw) - len(retained),
        "embedding_dimension": int(features.shape[1]), "missing_embedding_count": 0,
        "dtype": "float32", "coordinate_system": "analysis_crop_level0_xy_pixels",
        "alignment": "upstream_row_order_verified_by_centroids_then_filtered_by_id",
        "centroid_absolute_tolerance_px": 0.05, "centroid_relative_tolerance": 0.0,
        "retained_centroid_comparison": "exact_raw_json_coordinates",
    }
    return selected_features, rows, summary


def require_graph_loader():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CellViT embedding conversion requires PyTorch in the wrapper runtime") from exc
    if not hasattr(torch.serialization, "safe_globals"):
        raise RuntimeError("CellViT embedding conversion requires PyTorch with safe_globals support")
    return torch


def load_graph_arrays(graph_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Allow tensors and one inert dataclass; never fall back to unsafe pickle."""
    torch = require_graph_loader()
    try:
        with torch.serialization.safe_globals([CellGraphDataWSI]):
            graph = torch.load(graph_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "CellViT cells.pt could not be decoded using the restricted 1.0.9 graph schema"
        ) from exc
    if not isinstance(graph, CellGraphDataWSI):
        raise ValueError("Unsupported CellViT graph object; expected CellGraphDataWSI")
    if not isinstance(graph.x, torch.Tensor) or not isinstance(graph.positions, torch.Tensor):
        raise ValueError("CellViT graph x and positions must be tensors")
    return graph.x.detach().float().cpu().numpy(), graph.positions.detach().double().cpu().numpy()


def export_embeddings(raw_json: Path, retained_json: Path, outdir: Path, provenance: dict, *, execution=None) -> dict:
    """Write payloads only; source-bound completion is finalized by the wrapper."""
    graph_path = raw_json.with_name("cells.pt")
    if not graph_path.is_file():
        raise FileNotFoundError(f"CellViT --graph produced no matching graph: {graph_path}")
    targets = [outdir / FILES[key] for key in ("embeddings", "ids", "metadata", "raw_population")]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("CellViT embedding artifacts already exist; choose a new output directory")
    before = {"raw_cells_json": file_record(raw_json), "retained_cells_json": file_record(retained_json),
              "raw_graph": file_record(graph_path)}
    if execution is not None and retained_json.resolve() != (outdir / FILES["retained_population"]).resolve():
        raise ValueError("Source-bound CellViT export requires its in-bundle retained population")
    raw_payload, retained_payload = read_json(raw_json), read_json(retained_json)
    features, positions = load_graph_arrays(graph_path)
    features, rows, summary = align_embeddings(
        raw_payload, retained_payload, features, positions,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        graph_location = graph_path.resolve().relative_to(outdir.resolve()).as_posix()
    except ValueError:
        graph_location = graph_path.name  # Diagnostic only; never a consumer input.
    with (outdir / "cellvit_embeddings.npy").open("xb") as handle:
        np.save(handle, features, allow_pickle=False)
    with (outdir / "cellvit_embedding_ids.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "embedding_row", "cellvitpp_id", "x_px", "y_px", "source_graph_row",
        ])
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / FILES["raw_population"]).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows({"source_graph_row": i, "cellvitpp_id": identity, "x_px": float(xy[0]), "y_px": float(xy[1])}
                         for i, (identity, xy) in enumerate(population(raw_payload)))
    summary.update({
        "schema_version": 1, "status": "exported", "provenance": provenance,
        "features_file": "cellvit_embeddings.npy", "ids_file": "cellvit_embedding_ids.csv",
        "source_graph": graph_location,
        "representation": REPRESENTATION,
        "upstream_artifacts": before,
        "interpretation": "cell-associated H&E morphology features; may include surrounding context",
    })
    if execution is not None:
        summary["binding_contract"] = f"{FORMAT}/{VERSION}"
    after = {"raw_cells_json": file_record(raw_json), "retained_cells_json": file_record(retained_json),
             "raw_graph": file_record(graph_path)}
    if before != after:
        raise ValueError("CellViT raw/retained population or graph changed during embedding export")
    with (outdir / "cellvit_embeddings_metadata.json").open("x") as handle:
        handle.write(json.dumps(summary, indent=2))
    return summary
