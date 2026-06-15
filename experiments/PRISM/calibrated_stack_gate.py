"""M10 calibrated forecast stacking gate for PRISM.

M2/M9 showed that hard expert routing is brittle once a validation-selected
single expert is included.  M10 tests a different causal mechanism: learn a
forecast-level combination of experts on past realized windows, then evaluate it
once on the chronological future split.

Two stackers are tuned on an inner chronological validation slice:

* affine ridge stacking: unconstrained linear forecast calibration plus
  intercept;
* simplex stacking: nonnegative convex weights over expert forecasts.

Both operate on predictions and realized targets, not on future test losses.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from experiments.PRISM.champion_risk_gate import (
    _resolve_result_dir,
    json_safe,
    load_or_synthesize_predictions,
    select_validation_champion,
)
from experiments.PRISM.descriptor_probe import descriptors, find_context
from experiments.PRISM.router_viability import (
    DATASETS,
    FS_ALPHA_GRID,
    FS_LR_GRID,
    RunSpec,
    default_specs,
    fixed_share_with_prior,
    fit_ridge_loss,
    load_losses,
    mean_selected_loss,
    select_by_predicted_loss,
    standardize,
    tune_fixed_share_on_past,
    validation_single_baseline,
)


AFFINE_ALPHA_GRID = (0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
HORIZON_AFFINE_ALPHA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SIMPLEX_ALPHA_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
ANCHORS = {"ZeroPred", "Persistence", "HAR_EWM"}


@dataclass(frozen=True)
class StackCandidate:
    kind: str
    alpha: float
    prior_mode: str | None
    coef: np.ndarray
    validation_loss: float


def load_true_target(
    *,
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    model_names: list[str],
) -> np.ndarray:
    for name in model_names:
        if name not in ANCHORS:
            result_dir = _resolve_result_dir(results_root, dataset, lookback, horizon, name)
            true = np.load(result_dir / "true.npy")
            if true.shape[1] != horizon:
                raise ValueError(f"{name}: true.npy horizon mismatch {true.shape}")
            return true[:, :, -1].astype(np.float64)
    raise ValueError("Need at least one non-anchor model to load true.npy")


def flatten_predictions(preds: np.ndarray) -> np.ndarray:
    n, m, h = preds.shape
    return preds.transpose(0, 2, 1).reshape(n * h, m)


def affine_fit(preds: np.ndarray, true: np.ndarray, alpha: float) -> np.ndarray:
    x = flatten_predictions(preds)
    y = true.reshape(-1)
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    if alpha == 0.0:
        return np.linalg.lstsq(x, y, rcond=1e-8)[0]
    gram = x.T @ x / len(x)
    rhs = x.T @ y / len(x)
    penalty = np.eye(x.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + alpha * penalty, rhs)


def affine_loss(preds: np.ndarray, true: np.ndarray, coef: np.ndarray) -> float:
    x = flatten_predictions(preds)
    y_hat = x @ coef[:-1] + coef[-1]
    return float(((y_hat - true.reshape(-1)) ** 2).mean())


def horizon_affine_fit(preds: np.ndarray, true: np.ndarray, alpha: float) -> np.ndarray:
    n, m, h = preds.shape
    coefs = np.empty((h, m + 1), dtype=np.float64)
    eye = np.eye(m + 1, dtype=np.float64)
    eye[-1, -1] = 0.0
    for step in range(h):
        x = preds[:, :, step]
        y = true[:, step]
        x = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)
        gram = x.T @ x / n
        rhs = x.T @ y / n
        coefs[step] = np.linalg.solve(gram + alpha * eye, rhs)
    return coefs


def horizon_affine_loss(preds: np.ndarray, true: np.ndarray, coefs: np.ndarray) -> float:
    weights = coefs[:, :-1]
    intercept = coefs[:, -1]
    y_hat = np.einsum("nmh,hm->nh", preds, weights) + intercept[None, :]
    return float(((y_hat - true) ** 2).mean())


def simplex_prior(losses: np.ndarray, mode: str) -> np.ndarray:
    m = losses.shape[1]
    if mode == "uniform":
        return np.ones(m, dtype=np.float64) / m
    if mode == "validation_single":
        pick, _ = select_validation_champion(losses)
        prior = np.zeros(m, dtype=np.float64)
        prior[pick] = 1.0
        return prior
    if mode == "inverse_train_loss":
        inv = 1.0 / (losses.mean(axis=0) + 1e-12)
        inv = np.maximum(inv, 0.0)
        return inv / inv.sum()
    raise ValueError(f"Unknown prior mode: {mode}")


def simplex_fit(preds: np.ndarray, true: np.ndarray, *, alpha: float, prior: np.ndarray) -> np.ndarray:
    x = flatten_predictions(preds)
    y = true.reshape(-1)
    m = x.shape[1]
    gram = x.T @ x / len(x)
    rhs = x.T @ y / len(x)

    def objective(w: np.ndarray) -> float:
        resid = float(w @ gram @ w - 2.0 * rhs @ w)
        penalty = float(alpha * np.sum((w - prior) ** 2))
        return resid + penalty

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * (gram @ w - rhs) + 2.0 * alpha * (w - prior)

    result = minimize(
        objective,
        prior.copy(),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-12, "maxiter": 300, "disp": False},
    )
    if not result.success:
        # Fall back to the prior rather than silently emitting an invalid weight
        # vector.  The validation loss will expose that this candidate is weak.
        return prior.copy()
    w = np.maximum(result.x, 0.0)
    total = w.sum()
    return w / total if total > 0 else prior.copy()


def simplex_loss(preds: np.ndarray, true: np.ndarray, weights: np.ndarray) -> float:
    y_hat = np.tensordot(preds, weights, axes=([1], [0]))
    return float(((y_hat - true) ** 2).mean())


def tune_stacker(
    preds: np.ndarray,
    true: np.ndarray,
    losses: np.ndarray,
) -> StackCandidate:
    split = max(2, min(len(losses) - 1, int(len(losses) * 0.7)))
    fit_preds, val_preds = preds[:split], preds[split:]
    fit_true, val_true = true[:split], true[split:]
    fit_losses = losses[:split]
    best: StackCandidate | None = None

    for alpha in AFFINE_ALPHA_GRID:
        coef = affine_fit(fit_preds, fit_true, alpha)
        val = affine_loss(val_preds, val_true, coef)
        candidate = StackCandidate("affine_ridge", alpha, None, coef, val)
        if best is None or candidate.validation_loss < best.validation_loss:
            best = candidate

    for alpha in HORIZON_AFFINE_ALPHA_GRID:
        coef = horizon_affine_fit(fit_preds, fit_true, alpha)
        val = horizon_affine_loss(val_preds, val_true, coef)
        candidate = StackCandidate("horizon_affine_ridge", alpha, None, coef, val)
        if best is None or candidate.validation_loss < best.validation_loss:
            best = candidate

    for mode in ("uniform", "validation_single", "inverse_train_loss"):
        prior = simplex_prior(fit_losses, mode)
        for alpha in SIMPLEX_ALPHA_GRID:
            weights = simplex_fit(fit_preds, fit_true, alpha=alpha, prior=prior)
            val = simplex_loss(val_preds, val_true, weights)
            candidate = StackCandidate("simplex", alpha, mode, weights, val)
            if best is None or candidate.validation_loss < best.validation_loss:
                best = candidate

    assert best is not None
    return best


def fit_selected_stacker(
    preds: np.ndarray,
    true: np.ndarray,
    losses: np.ndarray,
    selected: StackCandidate,
) -> np.ndarray:
    if selected.kind == "affine_ridge":
        return affine_fit(preds, true, selected.alpha)
    if selected.kind == "horizon_affine_ridge":
        return horizon_affine_fit(preds, true, selected.alpha)
    if selected.kind == "simplex":
        assert selected.prior_mode is not None
        return simplex_fit(preds, true, alpha=selected.alpha, prior=simplex_prior(losses, selected.prior_mode))
    raise ValueError(f"Unknown stacker kind {selected.kind}")


def selected_loss(preds: np.ndarray, true: np.ndarray, selected: StackCandidate, coef: np.ndarray) -> float:
    if selected.kind == "affine_ridge":
        return affine_loss(preds, true, coef)
    if selected.kind == "horizon_affine_ridge":
        return horizon_affine_loss(preds, true, coef)
    return simplex_loss(preds, true, coef)


def summarize_coef(model_names: list[str], selected: StackCandidate, coef: np.ndarray) -> dict[str, object]:
    if selected.kind == "affine_ridge":
        weights = coef[:-1]
        top = sorted(
            [{"model": model_names[i], "weight": float(weights[i])} for i in range(len(model_names))],
            key=lambda item: abs(item["weight"]),
            reverse=True,
        )[:8]
        return {
            "kind": selected.kind,
            "alpha": float(selected.alpha),
            "intercept": float(coef[-1]),
            "top_weights": top,
            "l1_weight_sum": float(np.abs(weights).sum()),
        }
    if selected.kind == "horizon_affine_ridge":
        weights = coef[:, :-1]
        mean_abs = np.abs(weights).mean(axis=0)
        mean_weight = weights.mean(axis=0)
        top = sorted(
            [
                {"model": model_names[i], "mean_weight": float(mean_weight[i]), "mean_abs_weight": float(mean_abs[i])}
                for i in range(len(model_names))
            ],
            key=lambda item: item["mean_abs_weight"],
            reverse=True,
        )[:8]
        return {
            "kind": selected.kind,
            "alpha": float(selected.alpha),
            "top_mean_abs_weights": top,
            "mean_l1_weight_sum": float(np.abs(weights).sum(axis=1).mean()),
            "mean_intercept": float(coef[:, -1].mean()),
        }
    top = sorted(
        [{"model": model_names[i], "weight": float(coef[i])} for i in range(len(model_names)) if coef[i] > 1e-8],
        key=lambda item: item["weight"],
        reverse=True,
    )[:8]
    return {
        "kind": selected.kind,
        "alpha": float(selected.alpha),
        "prior_mode": selected.prior_mode,
        "top_weights": top,
        "weight_sum": float(coef.sum()),
    }


def analyze_one(
    spec: RunSpec,
    *,
    results_root: Path,
    lookback: int,
    horizon: int,
    train_frac: float,
) -> dict[str, object]:
    model_names, losses = load_losses(spec.oracle_dir / "window_losses.csv")
    context = find_context(results_root, spec.artifact_tag, lookback, horizon, model_names[0])
    if len(context) != len(losses):
        raise ValueError(f"{spec.dataset}: context/loss mismatch {len(context)} != {len(losses)}")
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
    train_preds, test_preds = preds[:split], preds[split:]
    train_true, test_true = true[:split], true[split:]
    train_losses, test_losses = losses[:split], losses[split:]

    val_single_loss, val_single_idx, val_single_validation_loss = validation_single_baseline(train_losses, test_losses)
    fs_params, fs_val_loss = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_loss = fixed_share_with_prior(train_losses, test_losses, feedback_delay=horizon, **fs_params)

    _, x_raw = descriptors(context)
    train_x, test_x, _, _ = standardize(x_raw[:split], x_raw[split:])
    ridge_coef = fit_ridge_loss(train_x, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_x)
    ridge_loss = mean_selected_loss(test_losses, ridge_picks)

    selected = tune_stacker(train_preds, train_true, train_losses)
    coef = fit_selected_stacker(train_preds, train_true, train_losses, selected)
    stack_loss = selected_loss(test_preds, test_true, selected, coef)

    return {
        "dataset": spec.dataset,
        "artifact_tag": spec.artifact_tag,
        "oracle_dir": str(spec.oracle_dir),
        "num_windows": int(len(losses)),
        "train_windows": int(split),
        "test_windows": int(len(test_losses)),
        "model_names": model_names,
        "validation_single_model": model_names[val_single_idx],
        "validation_single_loss": val_single_loss,
        "validation_single_validation_loss": val_single_validation_loss,
        "fixed_share_loss": fs_loss,
        "fixed_share_params": fs_params,
        "fixed_share_validation_loss": float(fs_val_loss),
        "fixed_share_grid": {
            "lr": list(FS_LR_GRID),
            "alpha": list(FS_ALPHA_GRID),
            "selection": "chronological validation slice of past split with delayed feedback",
        },
        "descriptor_ridge_loss": ridge_loss,
        "calibrated_stack_loss": stack_loss,
        "stacker_validation_loss": float(selected.validation_loss),
        "stacker": summarize_coef(model_names, selected, coef),
        "stack_beats_fixed_share": bool(stack_loss < fs_loss),
        "stack_beats_descriptor_ridge": bool(stack_loss < ridge_loss),
        "stack_beats_validation_single": bool(stack_loss < val_single_loss),
        "gate_pass": bool(stack_loss < fs_loss and stack_loss < ridge_loss and stack_loss < val_single_loss),
        "feedback_delay_windows": int(horizon),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = [
        analyze_one(spec, results_root=args.results_root, lookback=args.lookback, horizon=args.horizon, train_frac=args.train_frac)
        for spec in default_specs(args.oracle_root, lookback=args.lookback, horizon=args.horizon, datasets=tuple(args.datasets))
    ]
    result = {
        "milestone": "M10",
        "goal": "Causal forecast-level calibrated stacking after hard-router failure.",
        "gate": "Calibrated stack must beat delayed Fixed-Share, descriptor ridge, and validation-selected single expert on every battlefield.",
        "train_frac": args.train_frac,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "affine_alpha_grid": list(AFFINE_ALPHA_GRID),
        "horizon_affine_alpha_grid": list(HORIZON_AFFINE_ALPHA_GRID),
        "simplex_alpha_grid": list(SIMPLEX_ALPHA_GRID),
        "rows": rows,
        "gate_pass": all(row["gate_pass"] for row in rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibrated_stack_gate_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M10 calibrated forecast stacking gate.")
    p.add_argument("--oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/calibrated_stack_gate"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    compact = [
        {
            "dataset": row["dataset"],
            "validation_single": row["validation_single_loss"],
            "fixed_share": row["fixed_share_loss"],
            "descriptor_ridge": row["descriptor_ridge_loss"],
            "calibrated_stack": row["calibrated_stack_loss"],
            "stacker": row["stacker"]["kind"],
            "gate_pass": row["gate_pass"],
        }
        for row in result["rows"]
    ]
    print(json.dumps({"gate_pass": result["gate_pass"], "rows": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
