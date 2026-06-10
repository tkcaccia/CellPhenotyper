#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tissue_mask_benchmark_lib import (
    BENCHMARK_METHODS,
    BENCHMARK_METHOD_MAP,
    SampleCase,
    colorize_labels,
    compute_unsupervised_metrics,
    config_dict,
    default_config_for_spec,
    discover_example_samples,
    read_tiff_level0,
    run_spec_on_sample,
    save_comparison_panel,
    save_method_artifacts,
    score_binary_masks,
    score_candidate_program,
    segmentwise_reference_scores,
    summarize_method_metrics,
    write_gallery_index,
    write_global_report,
    write_per_image_report,
    write_mask_tiff,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Benchmark classical tissue-mask expansion methods across all repo examples.')
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--methods', default='', help='Comma-separated subset of methods to run')
    ap.add_argument('--include-evolved-program', default='', help='Optional evolved candidate program to materialize across all images')
    ap.add_argument('--clean', action='store_true', help='Delete output directory before writing')
    return ap.parse_args()


def _load_labels(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    arr = np.asarray(read_tiff_level0(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _copy_code_snapshots(outdir: Path) -> None:
    snapshot_dir = outdir / 'code_snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    for rel in ['bin/grow_to_tissue.py', 'bin/grow_to_tissue_core.py', 'bin/grow_to_tissue_legacy.py', 'bin/tissue_mask_benchmark_lib.py', 'bin/benchmark_tissue_mask_methods.py', 'modules/grow_to_tissue.nf', 'nextflow.config']:
        src = repo_root / rel
        if src.exists():
            shutil.copy2(src, snapshot_dir / src.name)


def main() -> None:
    warnings.filterwarnings('ignore', category=FutureWarning)
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir).resolve()
    if args.clean and outdir.exists():
        shutil.rmtree(outdir)
    (outdir / 'images').mkdir(parents=True, exist_ok=True)
    (outdir / 'comparison_panels').mkdir(parents=True, exist_ok=True)
    (outdir / 'reports').mkdir(parents=True, exist_ok=True)
    (outdir / 'tables').mkdir(parents=True, exist_ok=True)
    (outdir / 'logs').mkdir(parents=True, exist_ok=True)
    (outdir / 'galleries').mkdir(parents=True, exist_ok=True)
    _copy_code_snapshots(outdir)

    samples = discover_example_samples(repo_root)
    inventory = [case.to_dict() for case in samples]
    (outdir / 'tables' / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    print(f'[INFO] discovered {len(samples)} benchmarkable example images', flush=True)
    if not samples:
        raise SystemExit('No benchmarkable examples found in repository.')

    method_specs = BENCHMARK_METHODS
    if args.methods.strip():
        requested = [m.strip() for m in args.methods.split(',') if m.strip()]
        missing = [m for m in requested if m not in BENCHMARK_METHOD_MAP]
        if missing:
            raise SystemExit(f'Unknown methods requested: {missing}')
        method_specs = [BENCHMARK_METHOD_MAP[m] for m in requested]

    rows: list[dict[str, object]] = []
    evolved_rows: list[dict[str, object]] = []

    for case in samples:
        print(f'[INFO] sample={case.sample}', flush=True)
        image = read_tiff_level0(case.crop_roi)
        seed_labels = _load_labels(case.seed_mask)
        assert seed_labels is not None
        reference_labels = _load_labels(case.reference_mask)
        historical_labels = _load_labels(case.historical_mask)
        panel_methods: dict[str, np.ndarray] = {}
        evolved_labels = None

        for spec in method_specs:
            cfg = default_config_for_spec(spec)
            print(f'[INFO]   method={spec.name}', flush=True)
            try:
                tissue_mask, tissue_labels, runtime_sec = run_spec_on_sample(spec, image, seed_labels, historical_labels, cfg)
                metrics = compute_unsupervised_metrics(image, seed_labels, tissue_mask, runtime_sec)
                if reference_labels is not None:
                    metrics.update({f'reference_{k}': v for k, v in score_binary_masks(tissue_labels > 0, reference_labels > 0).items()})
                    metrics.update(segmentwise_reference_scores(tissue_labels, reference_labels))
                metrics.update({
                    'sample': case.sample,
                    'method': spec.name,
                    'method_family': spec.family,
                    'selectable': spec.selectable,
                    'production_method': spec.production,
                    'notes': spec.notes,
                    'config': json.dumps(config_dict(cfg), sort_keys=True),
                    'status': 'ok',
                    'failure_reason': '',
                })
                rows.append(metrics)
                panel_methods[spec.name] = tissue_labels
                method_dir = outdir / 'images' / case.sample / spec.name
                save_method_artifacts(method_dir, image, seed_labels, tissue_mask, tissue_labels, reference_labels, spec.name, metrics)
                print(
                    f"[INFO]   done method={spec.name} score={metrics['composite_score']:.4f} retention={metrics['cell_retention_ratio']:.4f} leakage={metrics['background_leakage_est']:.4f} runtime={metrics['runtime_sec']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                failure_path = outdir / 'logs' / f'{case.sample}__{spec.name}.log'
                failure_path.write_text(tb)
                rows.append({
                    'sample': case.sample,
                    'method': spec.name,
                    'method_family': spec.family,
                    'selectable': spec.selectable,
                    'production_method': spec.production,
                    'notes': spec.notes,
                    'config': json.dumps(config_dict(cfg), sort_keys=True),
                    'status': 'failed',
                    'failure_reason': f'{type(exc).__name__}: {exc}',
                })
                print(f'[WARN]   failed method={spec.name}: {exc}', flush=True)

        if historical_labels is not None:
            hist_mask = historical_labels > 0
            hist_metrics = compute_unsupervised_metrics(image, seed_labels, hist_mask, runtime_sec=0.0)
            if reference_labels is not None:
                hist_metrics.update({f'reference_{k}': v for k, v in score_binary_masks(hist_mask, reference_labels > 0).items()})
                hist_metrics.update(segmentwise_reference_scores(historical_labels, reference_labels))
            hist_metrics.update({
                'sample': case.sample,
                'method': 'historical_16_grown_tissue',
                'method_family': 'historical_reference',
                'selectable': False,
                'production_method': False,
                'notes': 'Existing 16_grown_tissue output copied into benchmark for visual comparison only.',
                'config': '{}',
                'status': 'ok',
                'failure_reason': '',
            })
            rows.append(hist_metrics)
            save_method_artifacts(outdir / 'images' / case.sample / 'historical_16_grown_tissue', image, seed_labels, hist_mask, historical_labels, reference_labels, 'historical_16_grown_tissue', hist_metrics)

        if args.include_evolved_program:
            evo_metrics, evo_mask, evo_labels = score_candidate_program(args.include_evolved_program, case)
            evo_metrics.update({
                'sample': case.sample,
                'method': 'best_evolved_candidate',
                'method_family': 'openevolve',
                'selectable': False,
                'production_method': False,
                'notes': str(args.include_evolved_program),
                'config': '{}',
                'status': 'ok',
                'failure_reason': '',
            })
            evolved_rows.append(evo_metrics)
            evolved_labels = evo_labels
            save_method_artifacts(outdir / 'images' / case.sample / 'best_evolved_candidate', image, seed_labels, evo_mask, evo_labels, reference_labels, 'best_evolved_candidate', evo_metrics)

        save_comparison_panel(
            outdir / 'comparison_panels' / f'{case.sample}.png',
            image,
            seed_labels,
            panel_methods,
            historical_labels=historical_labels,
            evolved_labels=evolved_labels,
        )

    df = pd.DataFrame(rows)
    if evolved_rows:
        df = pd.concat([df, pd.DataFrame(evolved_rows)], ignore_index=True)

    per_image_metrics = outdir / 'tables' / 'per_image_metrics.csv'
    df.to_csv(per_image_metrics, index=False)
    df.to_csv(outdir / 'tables' / 'per_image_metrics.tsv', sep='\t', index=False)

    ok_df = df[df.get('status', 'ok') == 'ok'].copy() if 'status' in df.columns else df.copy()
    ranking_df = summarize_method_metrics(ok_df[ok_df['selectable'] == True].copy())
    ranking_df.to_csv(outdir / 'tables' / 'method_ranking.csv', index=False)
    summary_metrics = ranking_df.copy()
    summary_metrics.to_csv(outdir / 'tables' / 'summary_metrics.csv', index=False)
    best_method = ranking_df.iloc[0]['method'] if not ranking_df.empty else None

    runtime_summary = (
        ok_df[ok_df['selectable'] == True]
        .groupby('method', as_index=False)['runtime_sec']
        .agg(['mean', 'median', 'max', 'min'])
        .reset_index()
    )
    runtime_summary.columns = ['method', 'runtime_mean_sec', 'runtime_median_sec', 'runtime_max_sec', 'runtime_min_sec']
    runtime_summary.to_csv(outdir / 'tables' / 'runtime_summary.csv', index=False)

    failed_df = df[df.get('status', 'ok') == 'failed'].copy() if 'status' in df.columns else pd.DataFrame()
    failed_df.to_csv(outdir / 'tables' / 'method_failures.csv', index=False)

    best_candidates = pd.DataFrame(evolved_rows)
    best_candidates.to_csv(outdir / 'tables' / 'best_candidates.csv', index=False)
    if not (outdir / 'tables' / 'evolution_history.csv').exists():
        pd.DataFrame(columns=['iteration', 'candidate', 'combined_score']).to_csv(outdir / 'tables' / 'evolution_history.csv', index=False)

    write_global_report(outdir / 'reports' / 'global_report.md', repo_root, samples, ranking_df, best_method)
    for case in samples:
        sample_rows = df[df['sample'] == case.sample].copy()
        write_per_image_report(outdir / 'reports' / f'{case.sample}_per_image_report.md', case.sample, sample_rows)

    write_gallery_index(outdir / 'galleries' / 'index.html', ranking_df, samples, outdir)
    print(f'[OK] benchmark results written to {outdir}', flush=True)
    print(f'[OK] global report: {outdir / "reports" / "global_report.md"}', flush=True)
    print(f'[OK] comparison panels: {outdir / "comparison_panels"}', flush=True)
    print(f'[OK] gallery: {outdir / "galleries" / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
