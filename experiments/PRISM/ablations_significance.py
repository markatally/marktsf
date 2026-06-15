"""M4 ablations, significance, and identifiability checks for PRISM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.drift_beta_loop import (
    LoopParams,
    drift_score,
    dynamic_beta,
    run_drift_loop,
    run_plain_fs,
    tune_params,
    tune_plain_params,
    weighted_mean,
)
from experiments.PRISM.router_viability import default_specs, load_losses, validation_single_baseline


def normal_p_from_t(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def signflip_p_values(block_means: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    """
    Paired sign-flip p-values over block-level mean differences.

    Returns (two_sided, positive_directional).  The positive-directional p-value
    tests whether the mean difference is greater than zero, i.e. b has lower
    loss than a for diff = a - b.
    """
    observed = float(block_means.mean())
    n = len(block_means)
    if n == 0:
        return math.nan, math.nan
    if n <= 20:
        signs = np.array(
            [[1.0 if (mask >> j) & 1 else -1.0 for j in range(n)] for mask in range(1 << n)],
            dtype=np.float64,
        )
        null = signs @ block_means / n
        two_sided = float(np.mean(np.abs(null) >= abs(observed) - 1e-15))
        positive = float(np.mean(null >= observed - 1e-15))
        return two_sided, positive

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(9999, n))
    null = signs @ block_means / n
    two_sided = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (len(null) + 1))
    positive = float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    return two_sided, positive


def paired_test(a: np.ndarray, b: np.ndarray, block_size: int) -> dict[str, float]:
    """Test whether b has lower loss than a using horizon-block sign flips."""
    diff = a - b
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    se = sd / math.sqrt(len(diff)) if sd > 0 else math.inf
    t = mean / se if math.isfinite(se) and se > 0 else 0.0
    normal_p = normal_p_from_t(t)

    effective_block_size = min(block_size, len(diff))
    n_blocks = max(1, len(diff) // effective_block_size)
    trimmed = diff[: n_blocks * effective_block_size]
    block_means = trimmed.reshape(n_blocks, effective_block_size).mean(axis=1)
    block_two_sided, block_positive = signflip_p_values(block_means)
    return {
        "mean_improvement": mean,
        "improvement_pct": float(mean / a.mean() * 100.0) if a.mean() != 0 else math.nan,
        "t_stat": float(t),
        "normal_p_value_two_sided": normal_p,
        "p_value_two_sided": block_two_sided,
        "p_value_directional": block_positive if mean > 0.0 else 1.0,
        "test": "paired_block_signflip",
        "block_size": int(effective_block_size),
        "num_blocks": int(n_blocks),
        "wins_frac": float(np.mean(diff > 0.0)),
    }


def bh_fdr(p_values: list[float], alpha: float = 0.10) -> list[bool]:
    m = len(p_values)
    order = np.argsort(p_values)
    passed = [False] * m
    kmax = -1
    for rank, idx in enumerate(order, start=1):
        if p_values[idx] <= alpha * rank / m:
            kmax = rank
    if kmax >= 1:
        threshold = alpha * kmax / m
        passed = [p <= threshold for p in p_values]
    return passed


def synthetic_identifiability(seed: int = 2026, n: int = 900) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    states = np.repeat(np.arange(3), n // 3)
    x = np.zeros((len(states), 2), dtype=np.float64)
    centers = np.array([[-1.5, 0.0], [0.0, 1.5], [1.5, -0.5]])
    for state in range(3):
        mask = states == state
        x[mask] = centers[state] + 0.55 * rng.standard_normal((mask.sum(), 2))
    losses = 0.8 + 0.15 * rng.standard_normal((len(states), 3))
    for state in range(3):
        losses[states == state, state] -= 0.35
    pred = np.argmin(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    return {
        "state_accuracy": float(np.mean(pred == states)),
        "oracle_gap": float(losses.mean(axis=0).min() - losses.min(axis=1).mean()),
        "router_loss": float(losses[np.arange(len(losses)), pred].mean()),
        "oracle_loss": float(losses.min(axis=1).mean()),
        "best_single_loss": float(losses.mean(axis=0).min()),
    }


def analyze_one(spec, args: argparse.Namespace) -> dict[str, object]:
    model_names, losses = load_losses(spec.oracle_dir / "window_losses.csv")
    context = find_context(args.results_root, spec.artifact_tag, args.lookback, args.horizon, model_names[0])
    descriptor_names, x = descriptors(context)
    split = max(10, min(len(losses) - 10, int(len(losses) * args.train_frac)))
    beta = dynamic_beta(x, descriptor_names, split)
    drift = drift_score(x, split)
    test_losses = losses[split:]
    test_beta = beta[split:]
    test_drift = drift[split:]

    val_start = max(2, min(split - 1, int(split * 0.7)))
    fit_losses = losses[:val_start]
    val_losses = losses[val_start:split]
    val_drift = drift[val_start:split]
    val_beta = beta[val_start:split]
    val_single_loss, val_single_idx, val_single_validation_loss = validation_single_baseline(losses[:split], test_losses)
    validation_single = test_losses[:, val_single_idx]

    # Tune both the plain online baseline and full PRISM loop on the same
    # chronological validation slice. This prevents M4 significance from being
    # anchored to a weaker hand-picked Fixed-Share baseline.
    _, plain_params = tune_plain_params(fit_losses, val_losses, val_drift, feedback_delay=args.horizon)
    _, full_params = tune_params(fit_losses, val_losses, val_drift, val_beta, model_names, feedback_delay=args.horizon)
    plain = run_plain_fs(
        losses[:split],
        test_losses,
        lr=plain_params["lr"],
        alpha=plain_params["alpha"],
        feedback_delay=args.horizon,
    )
    beta_only_params = LoopParams(
        lr=full_params.lr,
        base_alpha=full_params.base_alpha,
        drift_gain=0.0,
        beta_gain=full_params.beta_gain,
        beta_decay=full_params.beta_decay,
    )
    drift_only_params = LoopParams(
        lr=full_params.lr,
        base_alpha=full_params.base_alpha,
        drift_gain=full_params.drift_gain,
        beta_gain=0.0,
        beta_decay=0.0,
    )
    beta_only = run_drift_loop(
        losses[:split],
        test_losses,
        test_drift,
        test_beta,
        model_names,
        beta_only_params,
        feedback_delay=args.horizon,
    )
    drift_only = run_drift_loop(
        losses[:split],
        test_losses,
        test_drift,
        test_beta,
        model_names,
        drift_only_params,
        feedback_delay=args.horizon,
    )
    full = run_drift_loop(
        losses[:split],
        test_losses,
        test_drift,
        test_beta,
        model_names,
        full_params,
        feedback_delay=args.horizon,
    )

    tests = {
        "beta_only_vs_plain": paired_test(plain, beta_only, args.block_size),
        "drift_only_vs_plain": paired_test(plain, drift_only, args.block_size),
        "full_vs_plain": paired_test(plain, full, args.block_size),
        "full_vs_validation_single": paired_test(validation_single, full, args.block_size),
    }
    return {
        "dataset": spec.dataset,
        "test_windows": int(len(test_losses)),
        "losses": {
            "plain_fixed_share": float(plain.mean()),
            "beta_only": float(beta_only.mean()),
            "drift_only": float(drift_only.mean()),
            "full": float(full.mean()),
            "validation_single": float(val_single_loss),
            "plain_stress": weighted_mean(plain, test_drift),
            "full_stress": weighted_mean(full, test_drift),
        },
        "validation_single": {
            "model": model_names[val_single_idx],
            "validation_loss": float(val_single_validation_loss),
            "test_loss": float(val_single_loss),
        },
        "plain_params": plain_params,
        "full_params": full_params.__dict__,
        "feedback_delay_windows": int(args.horizon),
        "tests": tests,
        "interpretability": {
            "beta_iqr": float(np.percentile(test_beta, 75) - np.percentile(test_beta, 25)),
            "beta_drift_corr": float(np.corrcoef(test_beta, test_drift)[0, 1]) if test_beta.std() > 1e-8 and test_drift.std() > 1e-8 else 0.0,
            "mean_drift_top_quartile_beta": float(test_beta[test_drift >= np.quantile(test_drift, 0.75)].mean()),
            "mean_drift_bottom_quartile_beta": float(test_beta[test_drift <= np.quantile(test_drift, 0.25)].mean()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = [
        analyze_one(spec, args)
        for spec in default_specs(args.oracle_root, lookback=args.lookback, horizon=args.horizon)
    ]
    hypotheses = []
    p_values = []
    for row in rows:
        for name, test in row["tests"].items():
            directional_p = test["p_value_directional"]
            hypotheses.append(
                {
                    "dataset": row["dataset"],
                    "comparison": name,
                    "p_value": test["p_value_two_sided"],
                    "directional_p_value": directional_p,
                    "mean_improvement": test["mean_improvement"],
                }
            )
            p_values.append(directional_p)
    passed = bh_fdr(p_values, alpha=args.fdr_alpha)
    for item, ok in zip(hypotheses, passed):
        item["fdr_pass"] = bool(ok)

    full_plain_passes = [
        item for item in hypotheses if item["comparison"] == "full_vs_plain" and item["fdr_pass"]
    ]
    full_validation_passes = [
        item for item in hypotheses if item["comparison"] == "full_vs_validation_single" and item["fdr_pass"]
    ]
    result = {
        "milestone": "M4",
        "goal": "Ablations, paired significance with BH/FDR, interpretability, identifiability.",
        "fdr_alpha": args.fdr_alpha,
        "rows": rows,
        "fdr_hypotheses": hypotheses,
        "synthetic_identifiability": synthetic_identifiability(),
        "gate": "At least two full_vs_plain and two full_vs_validation_single comparisons must survive BH/FDR; synthetic recovery accuracy must exceed 0.8.",
        "gate_pass": bool(
            len(full_plain_passes) >= 2
            and len(full_validation_passes) >= 2
            and synthetic_identifiability()["state_accuracy"] > 0.8
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ablations_significance_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M4 ablations/significance harness.")
    p.add_argument("--oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/ablations_significance"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--fdr-alpha", type=float, default=0.10)
    p.add_argument(
        "--block-size",
        type=int,
        default=96,
        help="Contiguous block size for paired sign-flip tests; default equals the M1c horizon.",
    )
    args = p.parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    return args


def main() -> None:
    result = run(parse_args())
    compact = {
        "gate_pass": result["gate_pass"],
        "fdr_hypotheses": result["fdr_hypotheses"],
        "synthetic_identifiability": result["synthetic_identifiability"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
