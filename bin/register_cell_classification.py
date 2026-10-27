#!/usr/bin/env python3
"""Register cached cell classifications in a new verified experiment index."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from export_experiment import (ExperimentError, _path, export_experiment, sha256,
                               validate_experiment)


def register_cell_classification(experiment, output, *, tissue_id, classification,
                                 feature_manifest=None, segmentation_review=None, tissue_context=None, cell_index=None):
    """Preserve source assets and all prior project data; never run a model."""
    source = _path(str(experiment), Path.cwd(), "source experiment")
    destination = Path(os.path.abspath(Path(output).expanduser()))
    if os.path.lexists(destination):
        raise ExperimentError(f"Output already exists: {destination}. Choose a NEW filename; source projects are never overwritten")
    before = sha256(source)
    spec = validate_experiment(source)
    if sha256(source) != before:
        raise ExperimentError("Source experiment changed during validation; no output was created")
    selected = [tissue for tissue in spec["tissues"] if tissue["tissue_id"] == tissue_id]
    if len(selected) != 1:
        available = ", ".join(tissue["tissue_id"] for tissue in spec["tissues"])
        raise ExperimentError(f"Unknown tissue_id '{tissue_id}'. Choose one of: {available}")
    tissue = selected[0]
    classification_path = _path(str(classification), Path.cwd(), "cell classification")
    inferred_manifest = classification_path.with_name(classification_path.name.replace('_cell_classification.json', '_cell_features.manifest.json'))
    manifest_path = _path(str(feature_manifest or inferred_manifest),
                          Path.cwd(), "cell feature manifest")
    reference = dict(path=str(classification_path), sha256=sha256(classification_path),
                     feature_manifest=dict(path=str(manifest_path), sha256=sha256(manifest_path)))
    if tissue.get("cell_classification") is not None:
        existing = tissue["cell_classification"]
        if existing.get("sha256") != reference["sha256"]:
            raise ExperimentError("This tissue already has a different cell classification. Preserve it in a separate project or choose another tissue; replacement is never implicit")
        existing_manifest = existing.get("feature_manifest")
        if existing_manifest is not None and existing_manifest.get("sha256") != reference["feature_manifest"]["sha256"]:
            raise ExperimentError("This tissue already has a different cell feature manifest")
        reference = dict(existing, **reference)
    tissue["cell_classification"] = reference
    if segmentation_review is not None:
        review_path = _path(str(segmentation_review), Path.cwd(), "segmentation review")
        reference["segmentation_review"] = dict(path=str(review_path), sha256=sha256(review_path))
    if tissue_context is not None:
        context_path = _path(str(tissue_context), Path.cwd(), "cell tissue context")
        reference["tissue_context"] = dict(path=str(context_path), sha256=sha256(context_path))
    if cell_index is not None:
        index_path = _path(str(cell_index), Path.cwd(), "cell SQLite index")
        reference["index"] = dict(path=str(index_path), sha256=sha256(index_path), format="sqlite")
    if sha256(source) != before:
        raise ExperimentError("Source experiment changed while preparing registration; no output was created")
    return export_experiment(spec, destination, base_dir=source.parent, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__ + ' No inference or model libraries are started.')
    parser.add_argument('--experiment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='NEW experiment index; prior projects are never overwritten')
    parser.add_argument('--tissue-id', required=True)
    parser.add_argument('--classification', required=True, type=Path)
    parser.add_argument('--feature-manifest', type=Path, help='Defaults to cellvit_cell_features.manifest.json beside classification JSON')
    parser.add_argument('--segmentation-review', type=Path, help='Optional existing segmentation_qc.json with verified source-cell lineage')
    parser.add_argument('--tissue-context', type=Path, help='Optional checked soft Bora cell-to-tissue context; never changes cell labels')
    parser.add_argument('--cell-index', type=Path, help='Optional checked SQLite index for bounded whole-slide cell queries')
    args = parser.parse_args(argv)
    try:
        result = register_cell_classification(args.experiment, args.output, tissue_id=args.tissue_id,
            classification=args.classification, feature_manifest=args.feature_manifest, segmentation_review=args.segmentation_review,
            tissue_context=args.tissue_context, cell_index=args.cell_index)
        tissue = next(t for t in result['tissues'] if t['tissue_id'] == args.tissue_id)
        reference = tissue['cell_classification']
        print(f'Ready to open in wsiTools: {args.output}\nRetained {len(result["tissues"])} tissues; cell review: {reference["review_status"]} ({reference["cell_count"]} cells). Source project and assets are unchanged; no inference was started.')
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        print(f'Cannot register cell classification: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
