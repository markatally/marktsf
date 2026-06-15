"""M12 block/FDR significance tests for M10 calibrated stacking.

This script evaluates the narrowed M11 non-financial route.  It recomputes the
calibrated stack on the chronological past split, obtains per-window test
losses, and compares against:

* validation-selected single expert;
* delayed Fixed-Share;
* descriptor ridge expert selection.

Tests are paired horizon-block sign-flip tests, followed by BH/FDR correction
within each comparison family.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.PRISM.calibrated_stack_gate import (
    fit_selected_stacker,
    load_true_target,
    selected_loss,
    tune_stacker,
)
from experiments.PRISM.champion_risk_gate import json_safe, load_or_synthesize_predictions
from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.router_viability import (
    RunSpec,
    default_specs,
    fit_ridge_loss,
    load_losses,
    select_by_predicted_loss,
    standardize,
    tune_fixed_share_on_past,
    validation_single_baseline,
)


ETT_WEATHER = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
NONFINANCIAL_EXTRA = ("Electricity", "Traffic")


def fixed_share_loss_series(
    train_losses: np.ndarray,
    test_losses: np.ndarray,
    *,
    lr: float,
    alpha: float,
    feedback_delay: int,
) -> np.ndarray:
    centered = train_losses.mean(axis=0)
    centered = centered - centered.min()
    weights = np.exp(-lr * centered)
    weights = weights / weights.sum()
    out = np.empty(len(test_losses), dtype=np.float64)
    delay = max(0, int(feedback_delay))
    for i, row in enumerate(test_losses):
        out[i] = float(weights @ row)
        update_idx = i - delay
        if update_idx < 0:
            continue
        update_row = test_losses[update_idx]
        shifted = update_row - update_row.min()
        weights = weights * np.exp(-lr * shifted)
        weights = weights / weights.sum()
        weights = (1.0 - alpha) * weights + alpha / len(weights)
    return out


def stack_loss_series(preds: np.ndarray, true: np.ndarray, selected, coef: np.ndarray) -> np.ndarray:
    if selected.kind == "affine_ridge":
        x = preds.transpose(0, 2, 1)
        y_hat = np.tensordot(x, coef[:-1], axes=([2], [0])) + coef[-1]
    elif selected.kind == "horizon_affine_ridge":
        weights = coef[:, :-1]
        intercept = coef[:, -1]
        y_hat = np.einsum("nmh,hm->nh", preds, weights) + intercept[None, :]
    else:
        y_hat = np.tensordot(preds, coef, axes=([1], [0]))
    return ((y_hat - true) ** 2).mean(axis=1)


def block_means(x: np.ndarray, block_size: int) -> np.ndarray:
    block = max(1, int(block_size))
    n = len(x) // block
    if n == 0:
        return x.astype(np.float64)
    return x[: n * block].reshape(n, block).mean(axis=1)


def sign_flip_pvalue(delta: np.ndarray, seed: int, n_perm: int) -> float:
    """One-sided p-value for mean(delta) > 0 under paired sign flips."""
    delta = np.asarray(delta, dtype=np.float64)
    observed = float(delta.mean())
    if observed <= 0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(delta)))
    null = (signs * delta[None, :]).mean(axis=1)
    return float((np.count_nonzero(null >= observed) + 1) / (n_perm + 1))


def bh_fdr(pvalues: list[float], alpha: float) -> list[bool]:
    m = len(pvalues)
    order = np.argsort(np.asarray(pvalues))
    passes = [False] * m
    largest = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[int(idx)] <= alpha * rank / m:
            largest = rank
    if largest < 0:
        return passes
    threshold_idx = set(int(i) for i in order[:largest])
    return [i in threshold_idx for i in range(m)]


def analyze_one(
    spec: RunSpec,
    *,
    results_root: Path,
    lookback: int,
    horizon: int,
    train_frac: float,
    seed: int,
    n_perm: int,
) -> dict[str, object]:
    model_names, losses = load_losses(spec.oracle_dir / "window_losses.csv")
    context = find_context(results_root, spec.artifact_tag, lookback, horizon, model_names[0])
    preds = load_or_synthesize_predictions(
        results_root=results_root,
        dataset=spec.artifact_tag,
        lookback=lookback,
        horizon=horizon,
        model_names=model_names,
        context=context,
    )
    true = load_true_target(
        results_root=results_root,
        dataset=spec.artifact_tag,
        lookback=lookback,
        horizon=horizon,
        model_names=model_names,
    )
    split = max(10, min(len(losses) - 10, int(len(losses) * train_frac)))
    train_losses, test_losses = losses[:split], losses[split:]
    train_preds, test_preds = preds[:split], preds[split:]
    train_true, test_true = true[:split], true[split:]

    selected = tune_stacker(train_preds, train_true, train_losses)
    coef = fit_selected_stacker(train_preds, train_true, train_losses, selected)
    stack_series = stack_loss_series(test_preds, test_true, selected, coef)

    val_single_loss, val_single_idx, _ = validation_single_baseline(train_losses, test_losses)
    validation_series = test_losses[:, val_single_idx]

    fs_params, _ = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_series = fixed_share_loss_series(train_losses, test_losses, feedback_delay=horizon, **fs_params)

    _, x_raw = descriptors(context)
    train_x, test_x, _, _ = standardize(x_raw[:split], x_raw[split:])
    ridge_coef = fit_ridge_loss(train_x, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_x)
    ridge_series = test_losses[np.arange(len(test_losses)), ridge_picks]

    baselines = {
        "stack_vs_validation_single": validation_series,
        "stack_vs_fixed_share": fs_series,
        "stack_vs_descriptor_ridge": ridge_series,
    }
    tests = {}
    for offset, (name, baseline_series) in enumerate(baselines.items()):
        diff = baseline_series - stack_series
        blocks = block_means(diff, horizon)
        pvalue = sign_flip_pvalue(blocks, seed + offset, n_perm)
        tests[name] = {
            "mean_baseline_loss": float(np.mean(baseline_series)),
            "mean_stack_loss": float(np.mean(stack_series)),
            "improvement_abs": float(np.mean(diff)),
            "improvement_pct": float(100.0 * np.mean(diff) / max(np.mean(baseline_series), 1e-12)),
            "num_blocks": int(len(blocks)),
            "pvalue": pvalue,
        }
    return {
        "dataset": spec.dataset,
        "horizon": horizon,
        "model_names": model_names,
        "validation_single_model": model_names[val_single_idx],
        "stacker": {
            "kind": selected.kind,
            "alpha": selected.alpha,
            "prior_mode": selected.prior_mode,
            "validation_loss": selected.validation_loss,
        },
        "test_windows": int(len(test_losses)),
        "tests": tests,
        "aggregate": {
            "calibrated_stack_loss": float(np.mean(stack_series)),
            "validation_single_loss": float(val_single_loss),
            "fixed_share_loss": float(np.mean(fs_series)),
            "descriptor_ridge_loss": float(np.mean(ridge_series)),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    specs: list[tuple[int, RunSpec]] = []
    for horizon in (96, 192):
        specs.extend((horizon, spec) for spec in default_specs(args.base_oracle_root, horizon=horizon, datasets=ETT_WEATHER))
        specs.extend(
            (horizon, spec)
            for spec in default_specs(args.nonfinancial_oracle_root, horizon=horizon, datasets=NONFINANCIAL_EXTRA)
        )

    rows = []
    for idx, (horizon, spec) in enumerate(specs):
        root = args.base_results_root if spec.dataset in ETT_WEATHER else args.nonfinancial_results_root
        rows.append(
            analyze_one(
                spec,
                results_root=root,
                lookback=args.lookback,
                horizon=horizon,
                train_frac=args.train_frac,
                seed=args.seed + idx * 17,
                n_perm=args.n_perm,
            )
        )

    comparisons = ["stack_vs_validation_single", "stack_vs_fixed_share", "stack_vs_descriptor_ridge"]
    fdr_hypotheses = []
    for comparison in comparisons:
        pvals = [row["tests"][comparison]["pvalue"] for row in rows]
        passes = bh_fdr(pvals, args.fdr_alpha)
        for row, passed in zip(rows, passes):
            fdr_hypotheses.append(
                {
                    "dataset": row["dataset"],
                    "horizon": row["horizon"],
                    "comparison": comparison,
                    "pvalue": row["tests"][comparison]["pvalue"],
                    "fdr_pass": bool(passed),
                    "improvement_pct": row["tests"][comparison]["improvement_pct"],
                }
            )
    by_comparison = {
        comparison: sum(1 for item in fdr_hypotheses if item["comparison"] == comparison and item["fdr_pass"])
        for comparison in comparisons
    }
    result = {
        "milestone": "M12",
        "goal": "Block/FDR significance for M10 calibrated stacking on the narrowed M11 non-financial route.",
        "scope": {
            "datasets": list(ETT_WEATHER + NONFINANCIAL_EXTRA),
            "horizons": [96, 192],
        },
        "rows": rows,
        "fdr_alpha": args.fdr_alpha,
        "n_perm": args.n_perm,
        "fdr_hypotheses": fdr_hypotheses,
        "fdr_pass_counts": by_comparison,
        "gate_pass": all(count == len(rows) for count in by_comparison.values()),
        "gate": "All 14 dataset-horizon cells must pass BH/FDR against validation-single, delayed Fixed-Share, and descriptor ridge.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibrated_stack_significance_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M12 calibrated-stack block/FDR significance.")
    p.add_argument("--base-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--nonfinancial-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--base-results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--nonfinancial-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/calibrated_stack_significance"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20260615)
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "gate_pass": result["gate_pass"],
                "fdr_pass_counts": result["fdr_pass_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
