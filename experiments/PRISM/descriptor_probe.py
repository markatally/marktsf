"""Descriptor-to-winner probe for PRISM M1c step 12.

Given an oracle_drift output directory and the corresponding M1c frozen
prediction artifacts, this script computes simple regime descriptors from each
lookback context and tests whether they predict the per-window oracle winner in
a chronological train-on-past, predict-forward split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_losses(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        rows = [[float(x) for x in row[1:]] for row in reader]
    return names, np.asarray(rows, dtype=np.float64)


def find_context(results_root: Path, dataset: str, lookback: int, horizon: int, model: str) -> np.ndarray:
    prefix = f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model}_{dataset}_"
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    for p in matches:
        context = p / "context.npy"
        if context.exists():
            return np.load(context)
    raise FileNotFoundError(f"No context.npy found for {dataset} {model} under {results_root}")


def spectral_features(x: np.ndarray) -> np.ndarray:
    power = np.abs(np.fft.rfft(x, axis=1)) ** 2
    power = power[:, 1:]  # drop DC
    if power.shape[1] < 3:
        return np.zeros((x.shape[0], 4), dtype=np.float64)
    total = power.sum(axis=1) + 1e-12
    thirds = np.array_split(power, 3, axis=1)
    bands = np.stack([b.sum(axis=1) / total for b in thirds], axis=1)
    p = power / total[:, None]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1) / math.log(power.shape[1])
    return np.concatenate([bands, entropy[:, None]], axis=1)


def descriptors(context: np.ndarray) -> tuple[list[str], np.ndarray]:
    target = context[:, :, -1].astype(np.float64)
    W, L = target.shape
    t = np.arange(L, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    centered = target - target.mean(axis=1, keepdims=True)
    slope = (centered * t[None, :]).mean(axis=1)
    std = target.std(axis=1)
    first_half = target[:, : L // 2]
    second_half = target[:, L // 2 :]
    vol_ratio = second_half.std(axis=1) / (first_half.std(axis=1) + 1e-6)
    diff = np.diff(target, axis=1)
    diff_std = diff.std(axis=1)
    ac_num = (centered[:, 1:] * centered[:, :-1]).sum(axis=1)
    ac_den = np.sqrt((centered[:, 1:] ** 2).sum(axis=1) * (centered[:, :-1] ** 2).sum(axis=1)) + 1e-12
    ac1 = ac_num / ac_den
    spec = spectral_features(target)

    corr_mean = np.zeros(W, dtype=np.float64)
    corr_std = np.zeros(W, dtype=np.float64)
    abs_target_cov_corr = np.zeros(W, dtype=np.float64)
    if context.shape[2] > 1:
        vals = []
        target_corrs = []
        for i in range(W):
            c = np.corrcoef(context[i].astype(np.float64), rowvar=False)
            c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
            tri = c[np.triu_indices_from(c, k=1)]
            vals.append((float(tri.mean()), float(tri.std())))
            target_corrs.append(float(np.mean(np.abs(c[-1, :-1]))))
        corr_mean = np.asarray([v[0] for v in vals])
        corr_std = np.asarray([v[1] for v in vals])
        abs_target_cov_corr = np.asarray(target_corrs)

    names = [
        "target_mean",
        "target_std",
        "target_slope",
        "target_last_minus_first",
        "target_diff_std",
        "target_ac1",
        "target_vol_ratio",
        "spectral_low",
        "spectral_mid",
        "spectral_high",
        "spectral_entropy",
        "channel_corr_mean",
        "channel_corr_std",
        "target_cov_abs_corr",
    ]
    mat = np.column_stack(
        [
            target.mean(axis=1),
            std,
            slope,
            target[:, -1] - target[:, 0],
            diff_std,
            ac1,
            vol_ratio,
            spec,
            corr_mean,
            corr_std,
            abs_target_cov_corr,
        ]
    )
    return names, np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def ridge_probe(x: np.ndarray, y: np.ndarray, train_frac: float, alpha: float = 10.0) -> dict[str, object]:
    split = max(2, min(len(y) - 1, int(len(y) * train_frac)))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z_train = (x_train - mean) / std
    z_test = (x_test - mean) / std

    # Closed-form one-vs-rest ridge classifier.  Squared terms let the probe
    # capture threshold-like regime effects while keeping the fit deterministic.
    z_train = np.concatenate([z_train, z_train * z_train, np.ones((len(z_train), 1))], axis=1)
    z_test = np.concatenate([z_test, z_test * z_test, np.ones((len(z_test), 1))], axis=1)
    n_classes = int(y.max()) + 1
    y_onehot = np.eye(n_classes, dtype=np.float64)[y_train]
    eye = np.eye(z_train.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(z_train.T @ z_train + alpha * eye, z_train.T @ y_onehot)
    pred = (z_test @ coef).argmax(axis=1)

    counts = np.bincount(y_train, minlength=int(y.max()) + 1)
    marginal = int(counts.argmax())
    baseline = np.full_like(y_test, marginal)

    def acc(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(a == b)) if len(b) else math.nan

    def bal_acc(a: np.ndarray, b: np.ndarray) -> float:
        scores = []
        for c in np.unique(b):
            mask = b == c
            scores.append(float(np.mean(a[mask] == b[mask])))
        return float(np.mean(scores)) if scores else math.nan

    return {
        "train_windows": int(split),
        "test_windows": int(len(y_test)),
        "probe": "ridge_one_vs_rest_quadratic",
        "ridge_alpha": alpha,
        "probe_accuracy": acc(pred, y_test),
        "marginal_baseline_accuracy": acc(baseline, y_test),
        "probe_balanced_accuracy": bal_acc(pred, y_test),
        "marginal_balanced_accuracy": bal_acc(baseline, y_test),
        "accuracy_lift": acc(pred, y_test) - acc(baseline, y_test),
        "test_true_counts": {str(int(c)): int(np.count_nonzero(y_test == c)) for c in np.unique(y_test)},
        "train_true_counts": {str(int(c)): int(np.count_nonzero(y_train == c)) for c in np.unique(y_train)},
        "pred_counts": {str(int(c)): int(np.count_nonzero(pred == c)) for c in np.unique(pred)},
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    names, losses = load_losses(args.oracle_dir / "window_losses.csv")
    lower = args.lower_is_better
    winners = losses.argmin(axis=1) if lower else losses.argmax(axis=1)
    context = find_context(args.results_root, args.dataset, args.lookback, args.horizon, names[0])
    if context.shape[0] != losses.shape[0]:
        raise ValueError(f"Context/loss window mismatch: {context.shape[0]} vs {losses.shape[0]}")

    descriptor_names, x = descriptors(context)
    result = ridge_probe(x, winners, args.train_frac, alpha=args.ridge_alpha)
    result.update(
        {
            "oracle_dir": str(args.oracle_dir),
            "dataset": args.dataset,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "model_names": names,
            "descriptor_names": descriptor_names,
            "lower_is_better": lower,
            "num_windows": int(len(winners)),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "descriptor_probe_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M1c descriptor-to-winner probe.")
    p.add_argument("--oracle-dir", type=Path, required=True)
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--higher-is-better", action="store_true")
    args = p.parse_args()
    args.lower_is_better = not args.higher_is_better
    return args


def main() -> None:
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("dataset", "probe_accuracy", "marginal_baseline_accuracy", "accuracy_lift")}, sort_keys=True))


if __name__ == "__main__":
    main()
