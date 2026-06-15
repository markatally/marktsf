"""M14 delayed online portfolio over calibrated stackers and strong baselines.

M12/M13 show that static validation-selected stacking is strong but not
FDR-stable on all long-horizon cells.  This script tests a stricter causal
alternative: treat several past-only forecasters as candidates, then combine
them online with delayed loss feedback.

The candidate set is fixed before test evaluation:

* validation-selected single expert;
* descriptor-ridge expert selector;
* delayed Fixed-Share over base experts;
* affine ridge stacker;
* horizon-wise affine ridge stacker;
* simplex stacker.

Portfolio hyperparameters are selected on an inner chronological validation
slice of the past split.  Test updates use only losses whose full horizon has
already elapsed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.PRISM.calibrated_stack_gate import (
    AFFINE_ALPHA_GRID,
    HORIZON_AFFINE_ALPHA_GRID,
    SIMPLEX_ALPHA_GRID,
    affine_fit,
    affine_loss,
    horizon_affine_fit,
    horizon_affine_loss,
    load_true_target,
    simplex_fit,
    simplex_loss,
    simplex_prior,
)
from experiments.PRISM.calibrated_stack_significance import (
    block_means,
    fixed_share_loss_series,
    sign_flip_pvalue,
)
from experiments.PRISM.champion_risk_gate import json_safe, load_or_synthesize_predictions, select_validation_champion
from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.router_viability import (
    RunSpec,
    default_specs,
    fit_ridge_loss,
    load_losses,
    select_by_predicted_loss,
    standardize,
    tune_fixed_share_on_past,
)
from experiments.PRISM.sensor_stack_significance import INFRASTRUCTURE_DATASETS, TRAFFIC_SENSOR_DATASETS


META_LR_GRID = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
META_ALPHA_GRID = (0.0, 0.001, 0.01, 0.05, 0.1)
COMPARISONS = ("portfolio_vs_validation_single", "portfolio_vs_fixed_share", "portfolio_vs_descriptor_ridge")


@dataclass(frozen=True)
class CandidateBundle:
    names: list[str]
    forecasts: np.ndarray  # [windows, candidates, horizon]
    losses: np.ndarray  # [windows, candidates]
    details: dict[str, object]


def mse_series(forecast: np.ndarray, true: np.ndarray) -> np.ndarray:
    return ((forecast - true) ** 2).mean(axis=1)


def predict_affine(preds: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x = preds.transpose(0, 2, 1)
    return np.tensordot(x, coef[:-1], axes=([2], [0])) + coef[-1]


def predict_horizon_affine(preds: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.einsum("nmh,hm->nh", preds, coef[:, :-1]) + coef[:, -1][None, :]


def predict_simplex(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(preds, weights, axes=([1], [0]))


def fixed_share_forecasts(
    warm_losses: np.ndarray,
    eval_losses: np.ndarray,
    eval_preds: np.ndarray,
    *,
    lr: float,
    alpha: float,
    feedback_delay: int,
) -> np.ndarray:
    centered = warm_losses.mean(axis=0)
    centered = centered - centered.min()
    weights = np.exp(-lr * centered)
    weights = weights / weights.sum()
    out = np.empty((len(eval_losses), eval_preds.shape[2]), dtype=np.float64)
    delay = max(0, int(feedback_delay))
    for i, row in enumerate(eval_losses):
        out[i] = weights @ eval_preds[i]
        update_idx = i - delay
        if update_idx < 0:
            continue
        update_row = eval_losses[update_idx]
        shifted = update_row - update_row.min()
        weights = weights * np.exp(-lr * shifted)
        weights = weights / weights.sum()
        weights = (1.0 - alpha) * weights + alpha / len(weights)
    return out


def best_affine_alpha(fit_preds: np.ndarray, fit_true: np.ndarray, val_preds: np.ndarray, val_true: np.ndarray) -> float:
    best_alpha, best_loss = AFFINE_ALPHA_GRID[0], np.inf
    for alpha in AFFINE_ALPHA_GRID:
        val = affine_loss(val_preds, val_true, affine_fit(fit_preds, fit_true, alpha))
        if val < best_loss:
            best_alpha, best_loss = alpha, val
    return float(best_alpha)


def best_horizon_affine_alpha(
    fit_preds: np.ndarray,
    fit_true: np.ndarray,
    val_preds: np.ndarray,
    val_true: np.ndarray,
) -> float:
    best_alpha, best_loss = HORIZON_AFFINE_ALPHA_GRID[0], np.inf
    for alpha in HORIZON_AFFINE_ALPHA_GRID:
        val = horizon_affine_loss(val_preds, val_true, horizon_affine_fit(fit_preds, fit_true, alpha))
        if val < best_loss:
            best_alpha, best_loss = alpha, val
    return float(best_alpha)


def best_simplex_params(
    fit_preds: np.ndarray,
    fit_true: np.ndarray,
    fit_losses: np.ndarray,
    val_preds: np.ndarray,
    val_true: np.ndarray,
) -> tuple[str, float]:
    best_mode, best_alpha, best_loss = "uniform", SIMPLEX_ALPHA_GRID[0], np.inf
    for mode in ("uniform", "validation_single", "inverse_train_loss"):
        prior = simplex_prior(fit_losses, mode)
        for alpha in SIMPLEX_ALPHA_GRID:
            weights = simplex_fit(fit_preds, fit_true, alpha=alpha, prior=prior)
            val = simplex_loss(val_preds, val_true, weights)
            if val < best_loss:
                best_mode, best_alpha, best_loss = mode, alpha, val
    return best_mode, float(best_alpha)


def descriptor_forecasts(
    fit_context: np.ndarray,
    fit_losses: np.ndarray,
    eval_context: np.ndarray,
    eval_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, fit_x_raw = descriptors(fit_context)
    _, eval_x_raw = descriptors(eval_context)
    fit_x, eval_x, _, _ = standardize(fit_x_raw, eval_x_raw)
    coef = fit_ridge_loss(fit_x, fit_losses, alpha=10.0)
    picks = select_by_predicted_loss(coef, eval_x)
    return eval_preds[np.arange(len(eval_preds)), picks], picks


def build_candidates(
    *,
    fit_preds: np.ndarray,
    fit_true: np.ndarray,
    fit_losses: np.ndarray,
    fit_context: np.ndarray,
    warm_preds: np.ndarray,
    warm_true: np.ndarray,
    warm_losses: np.ndarray,
    warm_context: np.ndarray,
    eval_preds: np.ndarray,
    eval_true: np.ndarray,
    eval_losses: np.ndarray,
    eval_context: np.ndarray,
    horizon: int,
) -> CandidateBundle:
    names: list[str] = []
    forecasts: list[np.ndarray] = []
    details: dict[str, object] = {}

    val_idx, val_loss = select_validation_champion(warm_losses)
    names.append("validation_single")
    forecasts.append(eval_preds[:, val_idx, :])
    details["validation_single"] = {"model_index": int(val_idx), "warm_validation_loss": float(val_loss)}

    descriptor_pred, descriptor_picks = descriptor_forecasts(warm_context, warm_losses, eval_context, eval_preds)
    names.append("descriptor_ridge")
    forecasts.append(descriptor_pred)
    details["descriptor_ridge"] = {"unique_picks": int(len(set(int(x) for x in descriptor_picks)))}

    fs_params, fs_val_loss = tune_fixed_share_on_past(warm_losses, feedback_delay=horizon)
    names.append("fixed_share")
    forecasts.append(
        fixed_share_forecasts(
            warm_losses,
            eval_losses,
            eval_preds,
            lr=fs_params["lr"],
            alpha=fs_params["alpha"],
            feedback_delay=horizon,
        )
    )
    details["fixed_share"] = {**fs_params, "warm_validation_loss": float(fs_val_loss)}

    affine_alpha = best_affine_alpha(fit_preds, fit_true, warm_preds, warm_true)
    names.append("affine_ridge")
    forecasts.append(predict_affine(eval_preds, affine_fit(warm_preds, warm_true, affine_alpha)))
    details["affine_ridge"] = {"alpha": affine_alpha}

    horizon_alpha = best_horizon_affine_alpha(fit_preds, fit_true, warm_preds, warm_true)
    names.append("horizon_affine_ridge")
    forecasts.append(predict_horizon_affine(eval_preds, horizon_affine_fit(warm_preds, warm_true, horizon_alpha)))
    details["horizon_affine_ridge"] = {"alpha": horizon_alpha}

    simplex_mode, simplex_alpha = best_simplex_params(fit_preds, fit_true, fit_losses, warm_preds, warm_true)
    simplex_weights = simplex_fit(
        warm_preds,
        warm_true,
        alpha=simplex_alpha,
        prior=simplex_prior(warm_losses, simplex_mode),
    )
    names.append("simplex")
    forecasts.append(predict_simplex(eval_preds, simplex_weights))
    details["simplex"] = {"prior_mode": simplex_mode, "alpha": simplex_alpha}

    forecast_cube = np.stack(forecasts, axis=1)
    loss_cube = np.stack([mse_series(forecast_cube[:, i, :], eval_true) for i in range(forecast_cube.shape[1])], axis=1)
    return CandidateBundle(names=names, forecasts=forecast_cube, losses=loss_cube, details=details)


def online_portfolio(
    candidate_forecasts: np.ndarray,
    candidate_losses: np.ndarray,
    true: np.ndarray,
    *,
    lr: float,
    alpha: float,
    feedback_delay: int,
    initial_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, c, h = candidate_forecasts.shape
    weights = np.ones(c, dtype=np.float64) / c if initial_weights is None else initial_weights.astype(np.float64).copy()
    weights = weights / weights.sum()
    pred = np.empty((n, h), dtype=np.float64)
    weight_trace = np.empty((n, c), dtype=np.float64)
    delay = max(0, int(feedback_delay))
    for i in range(n):
        weight_trace[i] = weights
        pred[i] = weights @ candidate_forecasts[i]
        update_idx = i - delay
        if update_idx < 0:
            continue
        row = candidate_losses[update_idx]
        shifted = row - row.min()
        weights = weights * np.exp(-lr * shifted)
        weights = weights / weights.sum()
        weights = (1.0 - alpha) * weights + alpha / c
    return pred, mse_series(pred, true), weight_trace


def tune_portfolio(val_bundle: CandidateBundle, val_true: np.ndarray, horizon: int) -> tuple[dict[str, float], np.ndarray, float]:
    best_params: dict[str, float] | None = None
    best_weights: np.ndarray | None = None
    best_loss = np.inf
    for lr in META_LR_GRID:
        for alpha in META_ALPHA_GRID:
            _, loss, trace = online_portfolio(
                val_bundle.forecasts,
                val_bundle.losses,
                val_true,
                lr=lr,
                alpha=alpha,
                feedback_delay=horizon,
            )
            mean_loss = float(loss.mean())
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_params = {"lr": float(lr), "alpha": float(alpha)}
                final = trace[-1].copy()
                for row in val_bundle.losses[max(0, len(val_bundle.losses) - horizon) :]:
                    shifted = row - row.min()
                    final = final * np.exp(-lr * shifted)
                    final = final / final.sum()
                    final = (1.0 - alpha) * final + alpha / len(final)
                best_weights = final / final.sum()
    assert best_params is not None and best_weights is not None
    return best_params, best_weights, best_loss


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
    split = max(20, min(len(losses) - 20, int(len(losses) * train_frac)))
    inner = max(10, min(split - 5, int(split * 0.7)))
    fit_slice = slice(0, inner)
    warm_slice = slice(inner, split)
    train_slice = slice(0, split)
    test_slice = slice(split, len(losses))

    val_bundle = build_candidates(
        fit_preds=preds[fit_slice],
        fit_true=true[fit_slice],
        fit_losses=losses[fit_slice],
        fit_context=context[fit_slice],
        warm_preds=preds[fit_slice],
        warm_true=true[fit_slice],
        warm_losses=losses[fit_slice],
        warm_context=context[fit_slice],
        eval_preds=preds[warm_slice],
        eval_true=true[warm_slice],
        eval_losses=losses[warm_slice],
        eval_context=context[warm_slice],
        horizon=horizon,
    )
    meta_params, initial_weights, meta_val_loss = tune_portfolio(val_bundle, true[warm_slice], horizon)

    test_bundle = build_candidates(
        fit_preds=preds[fit_slice],
        fit_true=true[fit_slice],
        fit_losses=losses[fit_slice],
        fit_context=context[fit_slice],
        warm_preds=preds[train_slice],
        warm_true=true[train_slice],
        warm_losses=losses[train_slice],
        warm_context=context[train_slice],
        eval_preds=preds[test_slice],
        eval_true=true[test_slice],
        eval_losses=losses[test_slice],
        eval_context=context[test_slice],
        horizon=horizon,
    )
    portfolio_pred, portfolio_series, trace = online_portfolio(
        test_bundle.forecasts,
        test_bundle.losses,
        true[test_slice],
        lr=meta_params["lr"],
        alpha=meta_params["alpha"],
        feedback_delay=horizon,
        initial_weights=initial_weights,
    )
    baseline_series = {
        "portfolio_vs_validation_single": test_bundle.losses[:, test_bundle.names.index("validation_single")],
        "portfolio_vs_fixed_share": test_bundle.losses[:, test_bundle.names.index("fixed_share")],
        "portfolio_vs_descriptor_ridge": test_bundle.losses[:, test_bundle.names.index("descriptor_ridge")],
    }
    tests = {}
    for offset, (name, baseline) in enumerate(baseline_series.items()):
        diff = baseline - portfolio_series
        blocks = block_means(diff, horizon)
        tests[name] = {
            "mean_baseline_loss": float(baseline.mean()),
            "mean_portfolio_loss": float(portfolio_series.mean()),
            "improvement_abs": float(diff.mean()),
            "improvement_pct": float(100.0 * diff.mean() / max(float(baseline.mean()), 1e-12)),
            "num_blocks": int(len(blocks)),
            "pvalue": sign_flip_pvalue(blocks, seed + offset, n_perm),
        }
    return {
        "dataset": spec.dataset,
        "horizon": horizon,
        "candidate_names": test_bundle.names,
        "candidate_details": test_bundle.details,
        "meta_params": meta_params,
        "meta_validation_loss": float(meta_val_loss),
        "initial_weights": {name: float(weight) for name, weight in zip(test_bundle.names, initial_weights)},
        "mean_weights": {name: float(weight) for name, weight in zip(test_bundle.names, trace.mean(axis=0))},
        "test_windows": int(len(portfolio_series)),
        "tests": tests,
        "aggregate": {
            "portfolio_loss": float(portfolio_series.mean()),
            "validation_single_loss": float(baseline_series["portfolio_vs_validation_single"].mean()),
            "fixed_share_loss": float(baseline_series["portfolio_vs_fixed_share"].mean()),
            "descriptor_ridge_loss": float(baseline_series["portfolio_vs_descriptor_ridge"].mean()),
        },
    }


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
        "milestone": "M14",
        "goal": "Delayed online portfolio over calibrated stackers and strong causal baselines.",
        "scope": {
            "datasets": list(INFRASTRUCTURE_DATASETS + TRAFFIC_SENSOR_DATASETS),
            "horizons": [96, 192],
        },
        "rows": rows,
        "fdr_alpha": args.fdr_alpha,
        "n_perm": args.n_perm,
        "meta_lr_grid": list(META_LR_GRID),
        "meta_alpha_grid": list(META_ALPHA_GRID),
        "fdr_hypotheses": fdr_hypotheses,
        "fdr_pass_counts": pass_counts,
        "gate_pass": all(count == len(rows) for count in pass_counts.values()),
        "gate": "All 8 sensor/infrastructure cells must pass BH/FDR against validation-single, delayed Fixed-Share, and descriptor ridge.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "online_stack_portfolio_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M14 delayed online stacker portfolio.")
    p.add_argument("--infrastructure-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--infrastructure-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--sensor-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor"))
    p.add_argument("--sensor-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/online_stack_portfolio"))
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
