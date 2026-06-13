"""M3 dynamic-beta and drift-loop evaluation for PRISM.

After M2, the learned router is not the headline system.  This harness tests
the pivot: frozen heterogeneous experts tracked by Fixed-Share, with PRISM
descriptors used for a dynamic covariate-coupling proxy beta and a drift-aware
share rate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.router_viability import default_specs, load_losses, standardize


@dataclass(frozen=True)
class LoopParams:
    lr: float
    base_alpha: float
    drift_gain: float
    beta_gain: float
    beta_decay: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def dynamic_beta(x: np.ndarray, descriptor_names: list[str], train_end: int) -> np.ndarray:
    corr = x[:, descriptor_names.index("target_cov_abs_corr")]
    stability = -x[:, descriptor_names.index("channel_corr_std")]
    raw = 0.75 * corr + 0.25 * stability
    ref = raw[:train_end]
    loc = float(np.median(ref))
    scale = float(np.percentile(ref, 75) - np.percentile(ref, 25))
    if scale < 1e-8:
        scale = float(ref.std() + 1e-6)
    beta = sigmoid((raw - loc) / scale)
    return np.nan_to_num(beta, nan=0.5, posinf=1.0, neginf=0.0)


def drift_score(x: np.ndarray, train_end: int, decay: float = 0.97) -> np.ndarray:
    train = x[:train_end]
    _, z, mean, std = standardize(train, x)
    center = np.zeros(x.shape[1], dtype=np.float64)
    score = np.empty(len(x), dtype=np.float64)
    for i, row in enumerate(z):
        center = decay * center + (1.0 - decay) * row
        score[i] = float(np.linalg.norm(row - center) / math.sqrt(x.shape[1]))
    ref = score[:train_end]
    lo, hi = float(np.percentile(ref, 10)), float(np.percentile(ref, 90))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def run_plain_fs(
    train_losses: np.ndarray,
    eval_losses: np.ndarray,
    *,
    lr: float,
    alpha: float,
) -> np.ndarray:
    centered = train_losses.mean(axis=0)
    centered = centered - centered.min()
    weights = np.exp(-lr * centered)
    weights = weights / weights.sum()
    out = np.empty(len(eval_losses), dtype=np.float64)
    for i, row in enumerate(eval_losses):
        out[i] = float(weights @ row)
        shifted = row - row.min()
        weights = weights * np.exp(-lr * shifted)
        weights = weights / weights.sum()
        weights = (1.0 - alpha) * weights + alpha / len(weights)
    return out


def run_drift_loop(
    train_losses: np.ndarray,
    eval_losses: np.ndarray,
    eval_drift: np.ndarray,
    eval_beta: np.ndarray,
    model_names: list[str],
    params: LoopParams,
) -> np.ndarray:
    centered = train_losses.mean(axis=0)
    centered = centered - centered.min()
    weights = np.exp(-params.lr * centered)
    weights = weights / weights.sum()
    out = np.empty(len(eval_losses), dtype=np.float64)
    cov_idx = np.asarray([i for i, name in enumerate(model_names) if "RidgeCov" in name], dtype=np.int64)
    target_idx = np.asarray([i for i, name in enumerate(model_names) if name in {"TargetRidge", "Seasonal", "EWM", "HAR_EWM", "Persistence"}], dtype=np.int64)
    for i, row in enumerate(eval_losses):
        beta = float(eval_beta[i])
        routed = weights.copy()
        if len(cov_idx):
            routed[cov_idx] *= 1.0 + params.beta_gain * beta
        if len(target_idx):
            routed[target_idx] *= 1.0 + params.beta_gain * (1.0 - beta) * params.beta_decay
        routed = routed / routed.sum()
        out[i] = float(routed @ row)

        shifted = row - row.min()
        weights = weights * np.exp(-params.lr * shifted)
        weights = weights / weights.sum()
        alpha_t = min(0.5, max(0.001, params.base_alpha + params.drift_gain * float(eval_drift[i])))
        weights = (1.0 - alpha_t) * weights + alpha_t / len(weights)
    return out


def weighted_mean(losses: np.ndarray, drift: np.ndarray) -> float:
    weights = 1.0 + 2.0 * drift
    return float(np.average(losses, weights=weights))


def tune_params(
    train_losses: np.ndarray,
    val_losses: np.ndarray,
    val_drift: np.ndarray,
    val_beta: np.ndarray,
    model_names: list[str],
) -> tuple[LoopParams, float]:
    candidates = []
    for lr in [2.0, 5.0, 10.0, 20.0]:
        for base_alpha in [0.001, 0.005, 0.01, 0.05]:
            for drift_gain in [0.0, 0.01, 0.05, 0.10, 0.20]:
                for beta_gain in [0.0, 0.05, 0.10, 0.25]:
                    for beta_decay in [0.0, 0.5, 1.0]:
                        params = LoopParams(lr, base_alpha, drift_gain, beta_gain, beta_decay)
                        losses = run_drift_loop(train_losses, val_losses, val_drift, val_beta, model_names, params)
                        candidates.append((weighted_mean(losses, val_drift), params))
    return min(candidates, key=lambda item: item[0])


def analyze_one(spec, args: argparse.Namespace) -> dict[str, object]:
    model_names, losses = load_losses(spec.oracle_dir / "window_losses.csv")
    context = find_context(args.results_root, spec.artifact_tag, args.lookback, args.horizon, model_names[0])
    descriptor_names, x = descriptors(context)
    split = max(10, min(len(losses) - 10, int(len(losses) * args.train_frac)))
    beta = dynamic_beta(x, descriptor_names, split)
    drift = drift_score(x, split)

    val_start = int(split * 0.7)
    fit_losses = losses[:val_start]
    val_losses = losses[val_start:split]
    test_losses = losses[split:]
    val_drift, test_drift = drift[val_start:split], drift[split:]
    val_beta, test_beta = beta[val_start:split], beta[split:]

    _, params = tune_params(fit_losses, val_losses, val_drift, val_beta, model_names)
    plain_grid = []
    for lr in [2.0, 5.0, 10.0, 20.0]:
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.10]:
            val_plain = run_plain_fs(fit_losses, val_losses, lr=lr, alpha=alpha)
            plain_grid.append((weighted_mean(val_plain, val_drift), lr, alpha))
    _, plain_lr, plain_alpha = min(plain_grid, key=lambda item: item[0])

    plain = run_plain_fs(losses[:split], test_losses, lr=plain_lr, alpha=plain_alpha)
    loop = run_drift_loop(losses[:split], test_losses, test_drift, test_beta, model_names, params)

    high = test_drift >= np.quantile(test_drift, 0.75)
    beta_iqr = float(np.percentile(test_beta, 75) - np.percentile(test_beta, 25))
    beta_corr = float(np.corrcoef(test_beta, test_drift)[0, 1]) if test_beta.std() > 1e-8 and test_drift.std() > 1e-8 else 0.0
    return {
        "dataset": spec.dataset,
        "plain_fixed_share_loss": float(plain.mean()),
        "drift_loop_loss": float(loop.mean()),
        "plain_fixed_share_stress_loss": weighted_mean(plain, test_drift),
        "drift_loop_stress_loss": weighted_mean(loop, test_drift),
        "plain_high_drift_loss": float(plain[high].mean()),
        "drift_loop_high_drift_loss": float(loop[high].mean()),
        "stress_improvement_pct": float((weighted_mean(plain, test_drift) - weighted_mean(loop, test_drift)) / weighted_mean(plain, test_drift) * 100.0),
        "high_drift_improvement_pct": float((plain[high].mean() - loop[high].mean()) / plain[high].mean() * 100.0),
        "beta_mean": float(test_beta.mean()),
        "beta_iqr": beta_iqr,
        "beta_drift_corr": beta_corr,
        "beta_nontrivial": bool(beta_iqr >= 0.05),
        "plain_params": {"lr": plain_lr, "alpha": plain_alpha},
        "drift_loop_params": params.__dict__,
        "test_windows": int(len(test_losses)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = [analyze_one(spec, args) for spec in default_specs(args.oracle_root)]
    improved = sum(row["drift_loop_stress_loss"] < row["plain_fixed_share_stress_loss"] for row in rows)
    beta_ok = all(row["beta_nontrivial"] for row in rows)
    result = {
        "milestone": "M3",
        "goal": "Dynamic beta plus drift-aware Fixed-Share loop under drift-stress evaluation.",
        "gate": "Drift loop improves stressed loss on at least two battlefields and beta is nontrivial.",
        "rows": rows,
        "improved_datasets": int(improved),
        "beta_nontrivial_all": bool(beta_ok),
        "gate_pass": bool(improved >= 2 and beta_ok),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "drift_beta_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M3 dynamic beta/drift loop harness.")
    p.add_argument("--oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/drift_beta_loop"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
