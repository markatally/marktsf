"""
Online learning causal bounds for PRISM M1c (Step 11).

Implements Fixed-Share (and plain Hedge) over the frozen per-window loss
matrices produced by oracle_drift.py.  All algorithms are causal: they use
only losses up to time t to update weights for time t+1.

Usage
-----
python -m experiments.PRISM.online_learning \\
    --losses-csv experiments/PRISM/oracle_drift/ETTh1_L96_H96_target_last/window_losses.csv \\
    --output-dir experiments/PRISM/online_learning/ETTh1_L96_H96
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithms
# ──────────────────────────────────────────────────────────────────────────────

def hedge(
    losses: np.ndarray,
    lr: float | None = None,
    lower_is_better: bool = True,
) -> np.ndarray:
    """
    Hedge (Multiplicative Weights) algorithm.

    Parameters
    ----------
    losses  : [W, M] per-window per-model loss matrix
    lr      : learning rate η (if None, uses the Hedge minimax η = sqrt(ln M / W))
    lower_is_better : if False, negate losses before updating

    Returns
    -------
    mixture_losses : [W] per-window loss of the Hedge mixture (causal)
    """
    W, M = losses.shape
    if lr is None:
        lr = math.sqrt(math.log(M) / W) if W > 1 else 1.0

    w = np.ones(M) / M
    sign = 1.0 if lower_is_better else -1.0
    result = np.empty(W)

    for i in range(W):
        result[i] = float(np.dot(w, losses[i]))
        # Hedge update: w_j ∝ w_j * exp(-η * sign * loss_j).
        # Shift by min before exp to prevent underflow when lr is large.
        l = sign * losses[i]
        l = l - l.min()
        w = w * np.exp(-lr * l)
        w /= w.sum()

    return result


def fixed_share(
    losses: np.ndarray,
    alpha: float = 0.05,
    lr: float | None = None,
    lower_is_better: bool = True,
) -> np.ndarray:
    """
    Fixed-Share (FS) algorithm.

    After each Hedge update, each model transfers a fraction α of its weight
    uniformly to all other models.  This allows the mixture to track a
    switching sequence at rate α.

    Parameters
    ----------
    losses  : [W, M] per-window per-model loss matrix
    alpha   : sharing rate (fraction redistributed uniformly; default 0.05)
    lr      : Hedge learning rate (None → minimax)
    lower_is_better : if False, negate losses before updating

    Returns
    -------
    mixture_losses : [W] per-window loss of the FS mixture (causal)
    """
    W, M = losses.shape
    if lr is None:
        lr = math.sqrt(math.log(M) / W) if W > 1 else 1.0

    w = np.ones(M) / M
    sign = 1.0 if lower_is_better else -1.0
    result = np.empty(W)

    for i in range(W):
        result[i] = float(np.dot(w, losses[i]))
        # Hedge update; shift by min to prevent underflow at large lr.
        l = sign * losses[i]
        l = l - l.min()
        w = w * np.exp(-lr * l)
        w /= w.sum()
        # Fixed-Share mixing step.
        w = (1.0 - alpha) * w + alpha / M

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def load_window_losses(csv_path: Path) -> tuple[list[str], np.ndarray]:
    """Load window_losses.csv into (model_names, [W, M] float array)."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = header[1:]
        rows = []
        for row in reader:
            rows.append([float(x) for x in row[1:]])
    return model_names, np.array(rows, dtype=np.float64)


def analyze(
    losses: np.ndarray,
    model_names: list[str],
    lower_is_better: bool = True,
    alphas: list[float] | None = None,
    lrs: list[float] | None = None,
) -> dict[str, object]:
    """
    Compute oracle, Hedge, and Fixed-Share summary statistics.

    Returns a dict with per-algorithm mean losses and gap fractions relative
    to the oracle gap.
    """
    if alphas is None:
        alphas = [0.01, 0.05, 0.10, 0.20]
    if lrs is None:
        # Log-spaced grid including the minimax rate and practical values.
        minimax_lr = math.sqrt(math.log(losses.shape[1]) / losses.shape[0]) if losses.shape[0] > 1 else 1.0
        lrs = [minimax_lr, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    W, M = losses.shape
    agg = losses.mean(axis=0)
    best_single_idx = int(agg.argmin() if lower_is_better else agg.argmax())
    best_single_loss = float(agg[best_single_idx])

    if lower_is_better:
        oracle_loss = float(losses.min(axis=1).mean())
    else:
        oracle_loss = float(losses.max(axis=1).mean())

    oracle_gap = abs(best_single_loss - oracle_loss)

    # Directional gap: positive = online algo improves over best static.
    # For lower-is-better: improvement = best_single - algo_mean.
    # For higher-is-better: improvement = algo_mean - best_single.
    def _gap(algo_mean: float) -> float:
        return (best_single_loss - algo_mean) if lower_is_better else (algo_mean - best_single_loss)

    # Evaluate over all (lr, alpha) combinations; track best FS.
    fs_grid: list[dict[str, object]] = []
    best_fs_rec = -math.inf
    best_fs_entry: dict | None = None
    for lr in lrs:
        hedge_losses = hedge(losses, lr=lr, lower_is_better=lower_is_better)
        hedge_mean = float(hedge_losses.mean())
        hedge_gap = _gap(hedge_mean)
        hedge_rec = hedge_gap / oracle_gap if oracle_gap > 0 else math.nan
        for a in alphas:
            fs_losses = fixed_share(losses, alpha=a, lr=lr, lower_is_better=lower_is_better)
            fs_mean = float(fs_losses.mean())
            fs_gap = _gap(fs_mean)
            rec = fs_gap / oracle_gap if oracle_gap > 0 else math.nan
            entry = {
                "lr": lr,
                "alpha": a,
                "mean_loss": fs_mean,
                "gap_directional": float(fs_gap),
                "gap_recovered_frac": float(rec),
                "hedge_mean_loss": hedge_mean,
                "hedge_gap_recovered_frac": float(hedge_rec),
            }
            fs_grid.append(entry)
            if math.isfinite(rec) and rec > best_fs_rec:
                best_fs_rec = rec
                best_fs_entry = entry

    # Canonical "minimax" result for reference.
    minimax_lr = math.sqrt(math.log(M) / W) if W > 1 else 1.0
    canonical_fs = fixed_share(losses, alpha=0.05, lr=minimax_lr, lower_is_better=lower_is_better)
    canonical_mean = float(canonical_fs.mean())

    return {
        "num_windows": W,
        "num_models": M,
        "model_names": model_names,
        "lower_is_better": lower_is_better,
        "best_single_model": model_names[best_single_idx],
        "best_single_loss": best_single_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_abs": float(oracle_gap),
        "oracle_gap_rel": float(oracle_gap / abs(best_single_loss)) if best_single_loss != 0 else math.nan,
        # Best tuned FS result (note: tuned on full sequence = optimistic upper bound).
        "best_fixed_share": best_fs_entry,
        # Canonical minimax-lr FS (conservative, no tuning).
        "canonical_fs_alpha_0.05": {
            "lr": minimax_lr,
            "mean_loss": canonical_mean,
            "gap_recovered_frac": float(_gap(canonical_mean) / oracle_gap) if oracle_gap > 0 else math.nan,
        },
        # Full grid for analysis.
        "fs_grid": fs_grid,
    }


def run_analysis(
    losses_csv: Path,
    output_dir: Path,
    lower_is_better: bool = True,
    alphas: list[float] | None = None,
) -> dict[str, object]:
    model_names, losses = load_window_losses(losses_csv)
    result = analyze(losses, model_names, lower_is_better=lower_is_better, alphas=alphas)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "online_learning_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: batch over multiple oracle_drift run directories
# ──────────────────────────────────────────────────────────────────────────────

def batch_analyze(
    oracle_drift_dirs: list[Path],
    lower_is_better: bool = True,
    alphas: list[float] | None = None,
) -> list[dict[str, object]]:
    """
    Run online learning analysis over multiple oracle_drift output directories.

    Each directory must contain window_losses.csv (produced by oracle_drift).
    """
    results = []
    for d in oracle_drift_dirs:
        csv_path = d / "window_losses.csv"
        if not csv_path.exists():
            continue
        out = run_analysis(csv_path, d, lower_is_better=lower_is_better, alphas=alphas)
        out["run_dir"] = str(d)
        results.append(out)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed-Share / Hedge causal bound over oracle_drift window losses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--losses-csv", required=True, help="path to window_losses.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--higher-is-better", action="store_true", default=False,
                   help="Set for DA/IC losses (default: lower is better for MSE).")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20],
                   help="Fixed-Share alpha values to evaluate.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    lower_is_better = not args.higher_is_better
    result = run_analysis(
        losses_csv=Path(args.losses_csv),
        output_dir=Path(args.output_dir),
        lower_is_better=lower_is_better,
        alphas=args.alphas,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
