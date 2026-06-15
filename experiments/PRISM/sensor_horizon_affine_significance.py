"""M15 high-dimensional sensor route with fixed horizon-wise affine stacking.

M14 showed that delayed online portfolioing improves Fixed-Share robustness but
does not improve every validation-single comparison.  M15 tests the simpler
candidate that the sensor route itself suggests: use horizon-wise affine
forecast calibration as the fixed method class for high-dimensional
sensor/infrastructure datasets, with ridge alpha selected only on the
chronological past split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.PRISM.calibrated_stack_gate import (
    HORIZON_AFFINE_ALPHA_GRID,
    horizon_affine_fit,
    horizon_affine_loss,
    load_true_target,
)
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
from experiments.PRISM.sensor_stack_significance import INFRASTRUCTURE_DATASETS, TRAFFIC_SENSOR_DATASETS


COMPARISONS = ("horizon_affine_vs_validation_single", "horizon_affine_vs_fixed_share", "horizon_affine_vs_descriptor_ridge")


def predict_horizon_affine(preds: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.einsum("nmh,hm->nh", preds, coef[:, :-1]) + coef[:, -1][None, :]


def bh_fdr(pvalues: list[float], alpha: float) -> list[bool]:
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


def best_alpha(train_preds: np.ndarray, train_true: np.ndarray, train_losses: np.ndarray) -> float:
    split = max(2, min(len(train_losses) - 1, int(len(train_losses) * 0.7)))
    best = (HORIZON_AFFINE_ALPHA_GRID[0], np.inf)
    for alpha in HORIZON_AFFINE_ALPHA_GRID:
        coef = horizon_affine_fit(train_preds[:split], train_true[:split], alpha)
        val = horizon_affine_loss(train_preds[split:], train_true[split:], coef)
        if val < best[1]:
            best = (alpha, val)
    return float(best[0])


def analyze_one(spec, *, results_root: Path, lookback: int, horizon: int, train_frac: float, seed: int, n_perm: int) -> dict[str, object]:
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

    alpha = best_alpha(train_preds, train_true, train_losses)
    coef = horizon_affine_fit(train_preds, train_true, alpha)
    forecast = predict_horizon_affine(test_preds, coef)
    method_series = ((forecast - test_true) ** 2).mean(axis=1)

    validation_loss, validation_idx, _ = validation_single_baseline(train_losses, test_losses)
    validation_series = test_losses[:, validation_idx]
    fs_params, _ = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_series = fixed_share_loss_series(train_losses, test_losses, feedback_delay=horizon, **fs_params)
    _, x_raw = descriptors(context)
    train_x, test_x, _, _ = standardize(x_raw[:split], x_raw[split:])
    ridge_coef = fit_ridge_loss(train_x, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_x)
    ridge_series = test_losses[np.arange(len(test_losses)), ridge_picks]

    baselines = {
        "horizon_affine_vs_validation_single": validation_series,
        "horizon_affine_vs_fixed_share": fs_series,
        "horizon_affine_vs_descriptor_ridge": ridge_series,
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
        "alpha": alpha,
        "validation_single_model": model_names[validation_idx],
        "test_windows": int(len(test_losses)),
        "tests": tests,
        "aggregate": {
            "horizon_affine_loss": float(method_series.mean()),
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
            seed=args.seed + idx * 17,
            n_perm=args.n_perm,
        )
        for idx, (horizon, spec, results_root) in enumerate(specs)
    ]
    fdr_hypotheses = []
    for comparison in COMPARISONS:
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
    pass_counts = {
        comparison: sum(1 for item in fdr_hypotheses if item["comparison"] == comparison and item["fdr_pass"])
        for comparison in COMPARISONS
    }
    result = {
        "milestone": "M15",
        "goal": "Fixed horizon-wise affine calibrated stacking for high-dimensional sensor/infrastructure forecasting.",
        "scope": {
            "datasets": list(INFRASTRUCTURE_DATASETS + TRAFFIC_SENSOR_DATASETS),
            "horizons": [96, 192],
        },
        "rows": rows,
        "fdr_alpha": args.fdr_alpha,
        "n_perm": args.n_perm,
        "fdr_hypotheses": fdr_hypotheses,
        "fdr_pass_counts": pass_counts,
        "gate_pass": all(count == len(rows) for count in pass_counts.values()),
        "gate": "All 8 sensor/infrastructure cells must pass BH/FDR against validation-single, delayed Fixed-Share, and descriptor ridge.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sensor_horizon_affine_significance_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M15 fixed horizon-affine sensor route significance.")
    p.add_argument("--infrastructure-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--infrastructure-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--sensor-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor"))
    p.add_argument("--sensor-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/sensor_horizon_affine_significance"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20260615)
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"gate_pass": result["gate_pass"], "fdr_pass_counts": result["fdr_pass_counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
