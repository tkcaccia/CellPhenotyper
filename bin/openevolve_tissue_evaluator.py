from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

import numpy as np
from openevolve.evaluation_result import EvaluationResult

BIN_DIR = Path(__file__).resolve().parent
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from tissue_mask_benchmark_lib import (
    compute_unsupervised_metrics,
    discover_example_samples,
    grow_labels_within_mask,
    read_tiff_level0,
    score_binary_masks,
    segmentwise_reference_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = discover_example_samples(REPO_ROOT)


def _append_history(program_path: str, aggregate: dict[str, float]) -> None:
    history_path = os.environ.get('OPENEVOLVE_HISTORY_CSV', '').strip()
    if not history_path:
        return
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'candidate': Path(program_path).name, **aggregate}
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location('candidate_program', program_path)
    program = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(program)
    return program


def evaluate(program_path: str):
    try:
        warnings.filterwarnings('ignore', category=FutureWarning)
        program = _load_program(program_path)
        if not hasattr(program, 'run_refinement'):
            return EvaluationResult(
                metrics={'combined_score': 0.0, 'min_cell_retention_ratio': 0.0, 'mean_composite_score': -1e9, 'error': 1.0},
                artifacts={'error': 'Program must define run_refinement(image, seed_labels, base_labels).'},
            )

        per_sample = []
        for sample in SAMPLES:
            image = read_tiff_level0(sample.crop_roi)
            seed_labels = np.asarray(read_tiff_level0(sample.seed_mask))
            if seed_labels.ndim == 3:
                seed_labels = seed_labels[..., 0]
            base_labels = np.asarray(read_tiff_level0(sample.historical_mask)) if sample.historical_mask else seed_labels.copy()
            tissue_mask = np.asarray(program.run_refinement(image, seed_labels, base_labels)).astype(bool)
            tissue_mask[seed_labels > 0] = True
            pred_labels = grow_labels_within_mask(seed_labels, tissue_mask)
            metrics = compute_unsupervised_metrics(image, seed_labels, tissue_mask, runtime_sec=0.0)
            metrics['sample'] = sample.sample
            if sample.reference_mask:
                ref_labels = np.asarray(read_tiff_level0(sample.reference_mask))
                metrics.update({f'reference_{k}': v for k, v in score_binary_masks(pred_labels > 0, ref_labels > 0).items()})
                metrics.update(segmentwise_reference_scores(pred_labels, ref_labels))
            per_sample.append(metrics)

        min_retention = min(m['cell_retention_ratio'] for m in per_sample)
        mean_retention = float(np.mean([m['cell_retention_ratio'] for m in per_sample]))
        mean_composite = float(np.mean([m['composite_score'] for m in per_sample]))
        min_composite = float(np.min([m['composite_score'] for m in per_sample]))
        mean_leakage = float(np.mean([m['background_leakage_est'] for m in per_sample]))
        mean_fragmentation = float(np.mean([m['fragmentation_proxy'] for m in per_sample]))
        mean_continuity = float(np.mean([m['continuity_around_cells'] for m in per_sample]))
        mean_runtime = float(np.mean([m['runtime_sec'] for m in per_sample]))

        combined = (
            60.0 * min_retention
            + 8.0 * mean_retention
            + 1.5 * mean_composite
            + 0.6 * min_composite
            + 1.0 * mean_continuity
            - 6.0 * mean_leakage
            - 1.5 * mean_fragmentation
            - 0.05 * mean_runtime
        )
        if min_retention < 0.999999:
            combined -= 80.0 * (1.0 - min_retention)

        aggregate = {
            'combined_score': float(combined),
            'min_cell_retention_ratio': float(min_retention),
            'mean_cell_retention_ratio': float(mean_retention),
            'mean_composite_score': float(mean_composite),
            'min_composite_score': float(min_composite),
            'mean_background_leakage_est': float(mean_leakage),
            'mean_fragmentation_proxy': float(mean_fragmentation),
            'mean_continuity_around_cells': float(mean_continuity),
            'mean_runtime_sec': float(mean_runtime),
            'sample_count': float(len(per_sample)),
        }
        if all('reference_dice' in row for row in per_sample):
            aggregate['mean_reference_dice'] = float(np.mean([m['reference_dice'] for m in per_sample]))
            aggregate['mean_ref_segment_mean_dice'] = float(np.mean([m['ref_segment_mean_dice'] for m in per_sample]))

        _append_history(program_path, aggregate)
        return EvaluationResult(
            metrics=aggregate,
            artifacts={
                'summary': json.dumps({'aggregate': aggregate, 'per_sample': per_sample}, indent=2),
            },
        )
    except Exception as exc:
        return EvaluationResult(
            metrics={'combined_score': 0.0, 'min_cell_retention_ratio': 0.0, 'mean_composite_score': -1e9, 'error': 1.0},
            artifacts={'traceback': traceback.format_exc(), 'error': str(exc)},
        )
