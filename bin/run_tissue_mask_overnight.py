#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from openevolve import run_evolution

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / 'bin' / 'benchmark_tissue_mask_methods.py'
INITIAL_PROGRAM = REPO_ROOT / 'bin' / 'openevolve_tissue_initial_program.py'
EVALUATOR = REPO_ROOT / 'bin' / 'openevolve_tissue_evaluator.py'
KEY_FILE = REPO_ROOT / 'OpenEvolve.key'


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run the full overnight tissue-mask benchmark and OpenEvolve loop.')
    ap.add_argument('--repo-root', default=str(REPO_ROOT))
    ap.add_argument('--results-root', default=str(REPO_ROOT / 'results_tissue_mask_benchmark'))
    ap.add_argument('--iterations', type=int, default=220)
    ap.add_argument('--skip-benchmark', action='store_true')
    ap.add_argument('--skip-evolution', action='store_true')
    return ap.parse_args()


def ensure_api_key() -> None:
    if os.environ.get('OPENAI_API_KEY'):
        return
    if KEY_FILE.exists():
        os.environ['OPENAI_API_KEY'] = KEY_FILE.read_text().strip()
        return
    raise SystemExit('Missing OPENAI_API_KEY and OpenEvolve.key')


def run_benchmark(results_root: Path, include_evolved: Path | None = None, clean: bool = False) -> None:
    cmd = [sys.executable, str(BENCHMARK_SCRIPT), '--repo-root', str(REPO_ROOT), '--outdir', str(results_root)]
    if clean:
        cmd.append('--clean')
    if include_evolved is not None:
        cmd.extend(['--include-evolved-program', str(include_evolved)])
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def build_config(results_root: Path, iterations: int) -> Path:
    cfg = {
        'max_iterations': int(iterations),
        'checkpoint_interval': 10,
        'llm': {
            'primary_model': 'gpt-5-mini',
            'primary_model_weight': 1.0,
            'api_base': 'https://api.openai.com/v1',
            'api_key': '${OPENAI_API_KEY}',
            'temperature': 0.7,
            'max_tokens': 12000,
            'reasoning_effort': 'low',
            'timeout': 240,
            'retries': 2,
            'retry_delay': 5,
        },
        'prompt': {
            'system_message': (
                'Improve the classical tissue-mask refinement function for the full repository benchmark set. '
                'Primary objective: retain 100% of the cell_mask on every sample. '
                'Secondary objectives: maximize the unsupervised aggregate score, reduce background leakage, '
                'reduce fragmentation, preserve continuity around cell regions, and keep runtime acceptable. '
                'Use only non-deep-learning image processing. Modify only the EVOLVE-BLOCK function. '
                'Use image, seed_labels, and base_labels. Do not read files, network resources, or subprocess output.'
            )
        },
        'database': {
            'population_size': 16,
            'archive_size': 8,
            'num_islands': 3,
            'elite_selection_ratio': 0.25,
            'exploitation_ratio': 0.70,
            'exploration_ratio': 0.30,
            'similarity_threshold': 0.99,
        },
        'evaluator': {
            'timeout': 240,
            'parallel_evaluations': 1,
            'cascade_evaluation': False,
        },
        'diff_based_evolution': True,
        'max_code_length': 14000,
    }
    cfg_path = results_root / 'code_snapshots' / 'openevolve_config.yaml'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path


def run_evolution_loop(results_root: Path, iterations: int) -> Path:
    ensure_api_key()
    history_csv = results_root / 'tables' / 'evolution_history.csv'
    os.environ['OPENEVOLVE_HISTORY_CSV'] = str(history_csv)
    openevolve_dir = results_root / 'openevolve'
    if openevolve_dir.exists():
        shutil.rmtree(openevolve_dir)
    cfg_path = build_config(results_root, iterations)
    result = run_evolution(
        initial_program=str(INITIAL_PROGRAM),
        evaluator=str(EVALUATOR),
        config=str(cfg_path),
        iterations=iterations,
        output_dir=str(openevolve_dir),
        cleanup=False,
    )
    best_path = openevolve_dir / 'best' / 'best_program.py'
    if not best_path.exists():
        raise RuntimeError(f'OpenEvolve finished without a best program at {best_path}')
    snapshot_dir = results_root / 'code_snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, snapshot_dir / 'best_evolved_program.py')
    info_path = openevolve_dir / 'best' / 'best_program_info.json'
    if info_path.exists():
        shutil.copy2(info_path, snapshot_dir / 'best_evolved_program_info.json')
    return best_path


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    logs_dir = results_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_benchmark:
        print('[INFO] Running classical benchmark pass', flush=True)
        run_benchmark(results_root, include_evolved=None, clean=True)

    best_program = None
    if not args.skip_evolution:
        print(f'[INFO] Running OpenEvolve for {args.iterations} iterations', flush=True)
        best_program = run_evolution_loop(results_root, args.iterations)
        print(f'[INFO] Best evolved program: {best_program}', flush=True)

    if best_program is not None:
        print('[INFO] Re-materializing benchmark outputs with best evolved candidate', flush=True)
        run_benchmark(results_root, include_evolved=best_program, clean=False)

    print(f'[OK] Overnight run root: {results_root}', flush=True)
    print(f'[OK] Global report: {results_root / "reports" / "global_report.md"}', flush=True)
    print(f'[OK] Comparison panels: {results_root / "comparison_panels"}', flush=True)
    print(f'[OK] Gallery: {results_root / "galleries" / "index.html"}', flush=True)
    if best_program is not None:
        print(f'[OK] Best evolved program snapshot: {results_root / "code_snapshots" / "best_evolved_program.py"}', flush=True)


if __name__ == '__main__':
    main()
