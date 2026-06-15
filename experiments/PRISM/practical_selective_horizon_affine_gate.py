"""M17 practical-effect selective horizon-affine gate.

M16 showed that a significance-only activation rule can admit cells whose
test-split advantage over delayed Fixed-Share is too fragile.  This audit adds
a pre-test practical-effect floor: activate horizon-wise affine calibration
only when the chronological past split shows both statistically significant and
at least ``min_effect_pct`` improvement over validation-single and delayed
Fixed-Share.  Inactive cells abstain to validation-single.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.PRISM.calibrated_stack_gate import horizon_affine_fit, load_true_target
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
from experiments.PRISM.selective_horizon_affine_gate import ACTIVE_COMPARISONS, bh_fdr
from experiments.PRISM.sensor_horizon_affine_significance import best_alpha, predict_horizon_affine
from experiments.PRISM.sensor_stack_significance import INFRASTRUCTURE_DATASETS, TRAFFIC_SENSOR_DATASETS


EXT_ENVIRONMENT_SENSOR_DATASETS = ("Wind", "AQShunyi", "AQWan", "METRLA")


def _comparison_stats(
    baseline: np.ndarray,
    method_series: np.ndarray,
    *,
    horizon: int,
    seed: int,
    n_perm: int,
) -> dict[str, float | int]:
    diff = baseline - method_series
    blocks = block_means(diff, horizon)
    return {
        "improvement_pct": float(100.0 * diff.mean() / max(float(baseline.mean()), 1e-12)),
        "pvalue": float(sign_flip_pvalue(blocks, seed, n_perm)),
        "num_blocks": int(len(blocks)),
    }


def practical_active_decision(
    preds: np.ndarray,
    true: np.ndarray,
    losses: np.ndarray,
    *,
    horizon: int,
    active_alpha: float,
    min_effect_pct: float,
    seed: int,
    n_perm: int,
) -> dict[str, object]:
    split = max(2, min(len(losses) - 1, int(len(losses) * 0.7)))
    fit_preds, val_preds = preds[:split], preds[split:]
    fit_true, val_true = true[:split], true[split:]
    fit_losses, val_losses = losses[:split], losses[split:]

    alpha = best_alpha(fit_preds, fit_true, fit_losses)
    method = predict_horizon_affine(val_preds, horizon_affine_fit(fit_preds, fit_true, alpha))
    method_series = ((method - val_true) ** 2).mean(axis=1)

    _, validation_idx, _ = validation_single_baseline(fit_losses, val_losses)
    validation_series = val_losses[:, validation_idx]
    fs_params, _ = tune_fixed_share_on_past(fit_losses, feedback_delay=horizon)
    fs_series = fixed_share_loss_series(fit_losses, val_losses, feedback_delay=horizon, **fs_params)

    validation_stats = _comparison_stats(
        validation_series,
        method_series,
        horizon=horizon,
        seed=seed,
        n_perm=n_perm,
    )
    fixed_share_stats = _comparison_stats(
        fs_series,
        method_series,
        horizon=horizon,
        seed=seed + 1,
        n_perm=n_perm,
    )
    active = all(
        stats["improvement_pct"] >= min_effect_pct and stats["pvalue"] <= active_alpha
        for stats in (validation_stats, fixed_share_stats)
    )
    return {
        "active": bool(active),
        "selected_alpha": float(alpha),
        "past_validation_single_index": int(validation_idx),
        "past_fixed_share_params": fs_params,
        "past_validation_single": validation_stats,
        "past_fixed_share": fixed_share_stats,
    }


def analyze_one(
    spec,
    *,
    results_root: Path,
    lookback: int,
    horizon: int,
    train_frac: float,
    active_alpha: float,
    min_effect_pct: float,
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
    true = load_true_target(results_root=results_root, dataset=spec.artifact_tag, lookback=lookback, horizon=horizon, model_names=model_names)
    split = max(10, min(len(losses) - 10, int(len(losses) * train_frac)))
    train_losses, test_losses = losses[:split], losses[split:]
    train_preds, test_preds = preds[:split], preds[split:]
    train_true, test_true = true[:split], true[split:]

    decision = practical_active_decision(
        train_preds,
        train_true,
        train_losses,
        horizon=horizon,
        active_alpha=active_alpha,
        min_effect_pct=min_effect_pct,
        seed=seed,
        n_perm=n_perm,
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


def _iter_specs(args: argparse.Namespace):
    for horizon in (96, 192):
        for spec in default_specs(args.infrastructure_oracle_root, horizon=horizon, datasets=INFRASTRUCTURE_DATASETS):
            yield horizon, spec, args.infrastructure_results_root, "core_infrastructure"
        for spec in default_specs(args.sensor_oracle_root, horizon=horizon, datasets=TRAFFIC_SENSOR_DATASETS):
            yield horizon, spec, args.sensor_results_root, "core_sensor"
        for spec in default_specs(args.ext_oracle_root, horizon=horizon, datasets=EXT_ENVIRONMENT_SENSOR_DATASETS):
            yield horizon, spec, args.ext_results_root, "ext_environment_sensor"


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = []
    for idx, (horizon, spec, results_root, group) in enumerate(_iter_specs(args)):
        row = analyze_one(
            spec,
            results_root=results_root,
            lookback=args.lookback,
            horizon=horizon,
            train_frac=args.train_frac,
            active_alpha=args.active_alpha,
            min_effect_pct=args.min_effect_pct,
            seed=args.seed + idx * 17,
            n_perm=args.n_perm,
        )
        row["group"] = group
        rows.append(row)

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
                    "group": row["group"],
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
        "milestone": "M17",
        "goal": "Practical-effect selective horizon-wise affine calibration with validation-single abstention.",
        "scope": {
            "datasets": list(INFRASTRUCTURE_DATASETS + TRAFFIC_SENSOR_DATASETS + EXT_ENVIRONMENT_SENSOR_DATASETS),
            "horizons": [96, 192],
            "active_alpha": args.active_alpha,
            "min_effect_pct": args.min_effect_pct,
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
        "gate": (
            "At least min_active cells pass a pre-test active rule requiring "
            "past-split improvement >= min_effect_pct and p<=active_alpha vs "
            "both validation-single and delayed Fixed-Share; all active cells "
            "then pass BH/FDR vs validation-single, delayed Fixed-Share, and "
            "descriptor ridge; inactive cells exactly abstain to validation-single."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "practical_selective_horizon_affine_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M17 practical-effect selective horizon-affine gate.")
    p.add_argument("--infrastructure-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--infrastructure-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--sensor-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor"))
    p.add_argument("--sensor-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor"))
    p.add_argument("--ext-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor_ext"))
    p.add_argument("--ext-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor_ext"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/practical_selective_horizon_affine"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--active-alpha", type=float, default=0.05)
    p.add_argument("--min-effect-pct", type=float, default=5.0)
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
