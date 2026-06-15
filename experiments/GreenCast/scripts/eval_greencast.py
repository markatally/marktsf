"""Evaluation template for GreenCast.

Reads run manifest and optional metric payload to append rows to
`paper/tables/RESULTS_main.csv` in a schema-compatible way.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


RESULTS_FIELDS: List[str] = [
    'dataset', 'domain', 'horizon', 'seed', 'model', 'backbone',
    'val_mae', 'val_rmse', 'test_mae', 'test_rmse',
    'phase_regret', 'delay_mae', 'support_f1', 'illegal_mass', 'residual_ratio',
    'dm_vs_tft_p', 'dm_vs_gcgnet_p', 'wilcoxon_p', 'bh_fdr_q',
    'effect_size', 'ci_95', 'run_id'
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', type=str, required=True)
    p.add_argument('--manifest', type=str, required=True)
    p.add_argument('--output', type=str, required=True)
    p.add_argument('--metrics', type=str, default='')
    return p.parse_args()


def parse_list(v: Any, default: List[int]) -> List[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if v is None:
        return default
    return default


def read_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_metrics(path: str) -> list:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")
    if p.suffix.lower() == '.json':
        metrics = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(metrics, dict):
            return [metrics]
        if isinstance(metrics, list):
            return metrics
        return []
    with p.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def rows_from_manifest(manifest: dict) -> List[Dict[str, Any]]:
    rows = []
    seeds = parse_list(manifest.get('seeds', []), [0])
    horizons = parse_list(manifest.get('horizons', []), [0])

    if not seeds:
        seeds = [manifest.get('seed', 0)]
    if not horizons:
        horizons = [manifest.get('horizon', 0)]

    if len(seeds) == 1 and len(horizons) == 1:
        rows.append(_empty_row(manifest, horizons[0], seeds[0]))
    else:
        for seed in seeds:
            for horizon in horizons:
                rows.append(_empty_row(manifest, horizon, seed))
    return rows


def _empty_row(manifest: dict, horizon: Any, seed: Any) -> Dict[str, Any]:
    row = {
        'dataset': manifest.get('dataset', 'n/a'),
        'domain': manifest.get('domain', 'general'),
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
        'ci_95': 'n/a',
        'run_id': manifest.get('run_id', 'n/a'),
    }
    return row


def merge_metric_rows(base_rows: List[Dict[str, Any]], metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not metrics:
        return base_rows

    merged: List[Dict[str, Any]] = []
    idx = 0
    for row in base_rows:
        metric = metrics[idx % len(metrics)] if metrics else {}
        idx += 1
        copy = row.copy()
        for k in copy:
            if k in metric and metric[k] not in (None, '', []):
                copy[k] = metric[k]
        merged.append(copy)
    return merged


def ensure_fields(row: dict) -> List[Any]:
    return [row.get(k, 'n/a') for k in RESULTS_FIELDS]


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    metric_rows = read_metrics(args.metrics)
    rows = rows_from_manifest(manifest)
    merged = merge_metric_rows(rows, metric_rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()

    with out.open('a', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(RESULTS_FIELDS)
        for row in merged:
            w.writerow(ensure_fields(row))

    print(f'eval placeholder finished; wrote {len(merged)} row(s) for run_id={args.run_id} to {out}')


if __name__ == '__main__':
    main()
