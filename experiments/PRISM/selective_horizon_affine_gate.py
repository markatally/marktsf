"""M16 selective horizon-affine gate with no-harm abstention.

This is an explicit selective-method audit rather than a relaxed superiority
test.  For each dataset-horizon cell, the method decides on the past split
whether horizon-wise affine calibration has enough evidence to be active.  If
not active, it abstains to the validation-selected single expert.

Gate:

* at least ``min_active`` cells must be active;
* active cells must pass BH/FDR against validation-single, delayed Fixed-Share,
  and descriptor ridge;
* inactive cells must be exactly no-worse than validation-single by construction
  because they abstain to that baseline;
* all decisions are made before the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.PRISM.calibrated_stack_significance import block_means, fixed_share_loss_series, sign_flip_pvalue
from experiments.PRISM.champion_risk_gate import json_safe, load_or_synthesize_predictions
from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.router_viability import (
    default_specs,
    fit_ridge_loss,
    load_losses,
    select_by_predicted_loss,
    standardize,
    tune_fixed_share_on_past,
    validation_single_baseline,
)
from experiments.PRISM.sensor_horizon_affine_significance import (
    best_alpha,
    predict_horizon_affine,
)
from experiments.PRISM.calibrated_stack_gate import horizon_affine_fit, load_true_target
from experiments.PRISM.sensor_stack_significance import INFRASTRUCTURE_DATASETS, TRAFFIC_SENSOR_DATASETS


ACTIVE_COMPARISONS = (
    "selective_active_vs_validation_single",
    "selective_active_vs_fixed_share",
    "selective_active_vs_descriptor_ridge",
)


def bh_fdr(pvalues: list[float], alpha: float) -> list[bool]:
    if not pvalues:
        return []
    order = np.argsort(np.asarray(pvalues))
    passes = [False] * len(pvalues)
    largest = -1
    for rank, idx in enumerate(order, start=1):
        if pvalues[int(idx)] <= alpha * rank / len(pvalues):
            largest = rank
    if largest < 0:
        return passes
    keep = set(int(i) for i in order[:largest])
    return [i in keep for i in range(len(pvalues))]


def active_decision(
    preds: np.ndarray,
    true: np.ndarray,
    losses: np.ndarray,
    *,
    horizon: int,
    alpha_threshold: float,
    seed: int,
) -> dict[str, object]:
    split = max(2, min(len(losses) - 1, int(len(losses) * 0.7)))
    fit_preds, val_preds = preds[:split], preds[split:]
    fit_true, val_true = true[:split], true[split:]
    fit_losses, val_losses = losses[:split], losses[split:]
    alpha = best_alpha(fit_preds, fit_true, fit_losses)
    method = predict_horizon_affine(val_preds, horizon_affine_fit(fit_preds, fit_true, alpha))
    method_series = ((method - val_true) ** 2).mean(axis=1)
    _, validation_idx, _ = validation_single_baseline(fit_losses, val_losses)
    baseline = val_losses[:, validation_idx]
    diff = baseline - method_series
    pvalue = sign_flip_pvalue(block_means(diff, horizon), seed, 9999)
    improvement_pct = float(100.0 * diff.mean() / max(float(baseline.mean()), 1e-12))
    return {
        "active": bool(improvement_pct > 0.0 and pvalue <= alpha_threshold),
        "past_improvement_pct": improvement_pct,
        "past_pvalue": float(pvalue),
        "past_blocks": int(len(block_means(diff, horizon))),
        "selected_alpha": float(alpha),
        "past_validation_single_index": int(validation_idx),
    }


def analyze_one(spec, *, results_root: Path, lookback: int, horizon: int, train_frac: float, active_alpha: float, seed: int, n_perm: int) -> dict[str, object]:
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
    true = load_true_target(results_root=results_root, dataset=spec.artifact_tag, lookback=lookback, horizon=horizon, model_names=model_names)
    split = max(10, min(len(losses) - 10, int(len(losses) * train_frac)))
    train_losses, test_losses = losses[:split], losses[split:]
    train_preds, test_preds = preds[:split], preds[split:]
    train_true, test_true = true[:split], true[split:]
    train_context = context[:split]
    test_context = context[split:]

    decision = active_decision(
        train_preds,
        train_true,
        train_losses,
        horizon=horizon,
        alpha_threshold=active_alpha,
        seed=seed,
    )
    method_alpha = best_alpha(train_preds, train_true, train_losses)
    method_forecast = predict_horizon_affine(test_preds, horizon_affine_fit(train_preds, train_true, method_alpha))
    method_series = ((method_forecast - test_true) ** 2).mean(axis=1)

    validation_loss, validation_idx, _ = validation_single_baseline(train_losses, test_losses)
    validation_series = test_losses[:, validation_idx]
    if not decision["active"]:
        method_series = validation_series.copy()

    fs_params, _ = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_series = fixed_share_loss_series(train_losses, test_losses, feedback_delay=horizon, **fs_params)
    _, x_raw = descriptors(context)
    train_x, test_x, _, _ = standardize(x_raw[:split], x_raw[split:])
    ridge_coef = fit_ridge_loss(train_x, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_x)
    ridge_series = test_losses[np.arange(len(test_losses)), ridge_picks]

    baselines = {
        "selective_active_vs_validation_single": validation_series,
        "selective_active_vs_fixed_share": fs_series,
        "selective_active_vs_descriptor_ridge": ridge_series,
    }
    tests = {}
    for offset, (name, baseline) in enumerate(baselines.items()):
        diff = baseline - method_series
        blocks = block_means(diff, horizon)
        tests[name] = {
            "mean_baseline_loss": float(baseline.mean()),
            "mean_method_loss": float(method_series.mean()),
            "improvement_abs": float(diff.mean()),
            "improvement_pct": float(100.0 * diff.mean() / max(float(baseline.mean()), 1e-12)),
            "num_blocks": int(len(blocks)),
            "pvalue": sign_flip_pvalue(blocks, seed + offset, n_perm),
        }
    return {
        "dataset": spec.dataset,
        "horizon": horizon,
        "active": bool(decision["active"]),
        "decision": decision,
        "validation_single_model": model_names[validation_idx],
        "test_windows": int(len(test_losses)),
        "tests": tests,
        "aggregate": {
            "selective_loss": float(method_series.mean()),
            "validation_single_loss": float(validation_loss),
            "fixed_share_loss": float(fs_series.mean()),
            "descriptor_ridge_loss": float(ridge_series.mean()),
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    specs = []
    for horizon in (96, 192):
        specs.extend(
            (horizon, spec, args.infrastructure_results_root)
            for spec in default_specs(args.infrastructure_oracle_root, horizon=horizon, datasets=INFRASTRUCTURE_DATASETS)
        )
        specs.extend(
            (horizon, spec, args.sensor_results_root)
            for spec in default_specs(args.sensor_oracle_root, horizon=horizon, datasets=TRAFFIC_SENSOR_DATASETS)
        )
    rows = [
        analyze_one(
            spec,
            results_root=results_root,
            lookback=args.lookback,
            horizon=horizon,
            train_frac=args.train_frac,
            active_alpha=args.active_alpha,
            seed=args.seed + idx * 17,
            n_perm=args.n_perm,
        )
        for idx, (horizon, spec, results_root) in enumerate(specs)
    ]
    active_rows = [row for row in rows if row["active"]]
    active_hypotheses = []
    active_pass_counts = {}
    for comparison in ACTIVE_COMPARISONS:
        pvals = [row["tests"][comparison]["pvalue"] for row in active_rows]
        passes = bh_fdr(pvals, args.fdr_alpha)
        active_pass_counts[comparison] = sum(1 for passed in passes if passed)
        for row, passed in zip(active_rows, passes):
            active_hypotheses.append(
                {
                    "dataset": row["dataset"],
                    "horizon": row["horizon"],
                    "comparison": comparison,
                    "pvalue": row["tests"][comparison]["pvalue"],
                    "fdr_pass": bool(passed),
                    "improvement_pct": row["tests"][comparison]["improvement_pct"],
                }
            )
    inactive_no_harm = all(
        abs(row["aggregate"]["selective_loss"] - row["aggregate"]["validation_single_loss"]) < 1e-12
        for row in rows
        if not row["active"]
    )
    active_gate = bool(active_rows) and all(count == len(active_rows) for count in active_pass_counts.values())
    result = {
        "milestone": "M16",
        "goal": "Selective horizon-wise affine calibration with validation-single no-harm abstention.",
        "scope": {
            "datasets": list(INFRASTRUCTURE_DATASETS + TRAFFIC_SENSOR_DATASETS),
            "horizons": [96, 192],
            "active_alpha": args.active_alpha,
            "min_active": args.min_active,
        },
        "rows": rows,
        "active_count": len(active_rows),
        "inactive_count": len(rows) - len(active_rows),
        "inactive_no_harm": inactive_no_harm,
        "fdr_alpha": args.fdr_alpha,
        "n_perm": args.n_perm,
        "active_fdr_hypotheses": active_hypotheses,
        "active_fdr_pass_counts": active_pass_counts,
        "gate_pass": bool(inactive_no_harm and len(active_rows) >= args.min_active and active_gate),
        "gate": "At least min_active past-selected active cells; all active cells pass BH/FDR vs validation-single, delayed Fixed-Share, and descriptor ridge; inactive cells exactly abstain to validation-single.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selective_horizon_affine_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M16 selective horizon-affine no-harm gate.")
    p.add_argument("--infrastructure-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--infrastructure-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--sensor-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor"))
    p.add_argument("--sensor-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/selective_horizon_affine"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--active-alpha", type=float, default=0.05)
    p.add_argument("--min-active", type=int, default=4)
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
                "active_count": result["active_count"],
                "inactive_no_harm": result["inactive_no_harm"],
                "active_fdr_pass_counts": result["active_fdr_pass_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
