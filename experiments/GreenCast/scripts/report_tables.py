"""Build paper tables from experiment outputs.

Input: `results/*/manifest.json` and optional `metrics.csv/json`.
Output: `paper/tables/RESULTS_main.csv`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

RESULTS_FIELDS: List[str] = [
    'dataset', 'domain', 'horizon', 'seed', 'model', 'backbone',
    'val_mae', 'val_rmse', 'test_mae', 'test_rmse',
    'phase_regret', 'delay_mae', 'support_f1', 'illegal_mass', 'residual_ratio',
    'dm_vs_tft_p', 'dm_vs_gcgnet_p', 'wilcoxon_p', 'bh_fdr_q', 'effect_size', 'run_id'
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--results-root', type=str, required=True)
    p.add_argument('--out-csv', type=str, required=True)
    return p.parse_args()


def read_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))


def read_metric_file(path: Path) -> list:
    if not path.exists():
        return []
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return [data]
        return data
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def empty_row(manifest: dict, horizon, seed) -> Dict[str, Any]:
    return {
        'dataset': manifest.get('dataset', 'n/a'),
        'domain': manifest.get('domain', manifest.get('dataset', 'general')),
        'horizon': horizon,
        'seed': seed,
        'model': manifest.get('model', 'GreenCast'),
        'backbone': manifest.get('backbone', 'StableGreenBank'),
        'val_mae': 'n/a',
        'val_rmse': 'n/a',
        'test_mae': 'n/a',
        'test_rmse': 'n/a',
        'phase_regret': 'n/a',
        'delay_mae': 'n/a',
        'support_f1': 'n/a',
        'illegal_mass': 'n/a',
        'residual_ratio': manifest.get('residual_ratio', 'n/a'),
        'dm_vs_tft_p': 'n/a',
        'dm_vs_gcgnet_p': 'n/a',
        'wilcoxon_p': 'n/a',
        'bh_fdr_q': 'n/a',
        'effect_size': 'n/a',
        'run_id': manifest.get('run_id', run_dir.name),
    }


def build_rows(manifest: dict, metrics: list, run_id: str) -> List[dict]:
    seeds = manifest.get('seeds', [manifest.get('seed', 0)])
    horizons = manifest.get('horizons', [manifest.get('horizon', 0)])
    if isinstance(seeds, tuple):
        seeds = list(seeds)
    if isinstance(horizons, tuple):
        horizons = list(horizons)
    if not seeds:
        seeds = [0]
    if not horizons:
        horizons = [0]

    rows = []
    metric_rows = metrics or []

    # deterministic row assignment: metric rows are assumed to correspond to cross-product
    cursor = 0
    for seed in seeds:
        for horizon in horizons:
            row = empty_row(manifest, horizon, seed)
            if metric_rows:
                m = metric_rows[cursor % len(metric_rows)]
                cursor += 1
                for k in row:
                    if k in m and m[k] not in (None, '', []):
                        row[k] = m[k]
                row['run_id'] = run_id
            rows.append(row)
    return rows


def aggregate(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(results_root.glob('*')):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / 'manifest.json'
        if not manifest_path.exists():
            continue
        manifest = read_manifest(run_dir)
        run_id = manifest.get('run_id', run_dir.name)

        # prefer explicit metric file naming convention, then fallback to csv
        metrics = []
        metrics_json = run_dir / f'metrics_{run_id}.json'
        if metrics_json.exists():
            metrics = read_metric_file(metrics_json)
        elif (run_dir / 'metrics.csv').exists():
            metrics = read_metric_file(run_dir / 'metrics.csv')
        elif (run_dir / f'metrics_{run_id}.csv').exists():
            metrics = read_metric_file(run_dir / f'metrics_{run_id}.csv')

        rows.extend(build_rows(manifest, metrics, run_id))
    return rows


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    rows = aggregate(results_root)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        w.writeheader()
        for row in rows:
            # fill fields not present in row
            for key in RESULTS_FIELDS:
                row.setdefault(key, 'n/a')
            w.writerow({k: row.get(k, 'n/a') for k in RESULTS_FIELDS})

    print(f'report built with {len(rows)} rows -> {out}')


if __name__ == '__main__':
    main()
