"""M9 champion-risk gate for PRISM.

The strengthened M2 audit exposed a simple failure mode: a causal single
expert selected on the most recent past validation slice is a very strong
baseline.  This harness therefore treats that validation champion as the
default action and asks a narrower question:

Can a causal pairwise risk model identify only the windows where switching
away from the validation champion is expected to reduce loss?

The gate uses no future loss at decision time.  Its features are computed from
the current lookback context and the already-available expert forecasts.  Its
targets and hyperparameters are fitted only on the chronological past split.
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


RIDGE_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
MARGIN_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9)
MARGIN_FRAC_GRID = (0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.0, math.inf)
ROBUST_IMPROVEMENT_FRAC_MIN = 0.02


@dataclass(frozen=True)
class PairwiseModel:
    coef: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    alpha: float
    margin: float
    champion_idx: int


def _resolve_result_dir(results_root: Path, dataset: str, lookback: int, horizon: int, model: str) -> Path:
    prefix = f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model}_{dataset}_"
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"No result directory for {dataset} {model} under {results_root}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous result directories for {dataset} {model}: {[p.name for p in matches]}")
    return matches[0]


def _ewm_state(target: np.ndarray, half_life: float = 24.0) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / half_life)
    state = target[:, 0].copy()
    for j in range(1, target.shape[1]):
        state = alpha * target[:, j] + (1.0 - alpha) * state
    return state


def load_or_synthesize_predictions(
    *,
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    model_names: list[str],
    context: np.ndarray,
) -> np.ndarray:
    """Return causal expert forecasts with shape [windows, models, horizon].

    Real model forecasts come from the frozen prediction artifacts.  Synthetic
    anchors are reconstructed from the lookback context so their features remain
    available at decision time.
    """
    windows = context.shape[0]
    target = context[:, :, -1].astype(np.float64)
    preds = np.empty((windows, len(model_names), horizon), dtype=np.float64)
    for m, name in enumerate(model_names):
        if name == "ZeroPred":
            pred = np.zeros((windows, horizon), dtype=np.float64)
        elif name == "Persistence":
            pred = np.repeat(target[:, -1:], horizon, axis=1)
        elif name == "HAR_EWM":
            pred = np.repeat(_ewm_state(target)[:, None], horizon, axis=1)
        else:
            result_dir = _resolve_result_dir(results_root, dataset, lookback, horizon, name)
            arr = np.load(result_dir / "pred.npy")
            if arr.shape[0] != windows or arr.shape[1] != horizon:
                raise ValueError(f"{name}: prediction shape {arr.shape} incompatible with W={windows}, H={horizon}")
            pred = arr[:, :, -1].astype(np.float64)
        preds[:, m] = pred
    return preds


def forecast_feature_names() -> list[str]:
    return [
        "pred_mean",
        "pred_std",
        "pred_slope",
        "pred_first_minus_ctx_last",
        "pred_last_minus_ctx_last",
        "pred_mean_minus_ctx_mean",
        "pred_std_over_ctx_std",
        "pred_slope_minus_ctx_slope",
        "pred_seasonal_gap",
        "pred_horizon_range",
    ]


def forecast_features(context: np.ndarray, preds: np.ndarray, *, season_lag: int | None = None) -> np.ndarray:
    """Compute per-window, per-expert causal forecast plausibility features."""
    target = context[:, :, -1].astype(np.float64)
    windows, models, horizon = preds.shape
    h = np.arange(horizon, dtype=np.float64)
    h = (h - h.mean()) / (h.std() + 1e-12)
    l = np.arange(target.shape[1], dtype=np.float64)
    l = (l - l.mean()) / (l.std() + 1e-12)
    ctx_centered = target - target.mean(axis=1, keepdims=True)
    ctx_slope = (ctx_centered * l[None, :]).mean(axis=1)
    ctx_mean = target.mean(axis=1)
    ctx_std = target.std(axis=1) + 1e-6
    pred_centered = preds - preds.mean(axis=2, keepdims=True)
    pred_slope = (pred_centered * h[None, None, :]).mean(axis=2)
    lag = min(season_lag or horizon, target.shape[1], horizon)
    if lag > 0:
        seasonal_ref = target[:, -lag:]
        seasonal_pred = preds[:, :, :lag]
        seasonal_gap = ((seasonal_pred - seasonal_ref[:, None, :]) ** 2).mean(axis=2)
    else:
        seasonal_gap = np.zeros((windows, models), dtype=np.float64)
    pieces = [
        preds.mean(axis=2),
        preds.std(axis=2),
        pred_slope,
        preds[:, :, 0] - target[:, -1, None],
        preds[:, :, -1] - target[:, -1, None],
        preds.mean(axis=2) - ctx_mean[:, None],
        preds.std(axis=2) / ctx_std[:, None],
        pred_slope - ctx_slope[:, None],
        seasonal_gap,
        preds.max(axis=2) - preds.min(axis=2),
    ]
    return np.stack(pieces, axis=2)


def _expand(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, x * x, np.ones((len(x), 1), dtype=np.float64)], axis=1)


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z = _expand((x - mean) / std)
    eye = np.eye(z.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(z.T @ z + alpha * eye, z.T @ y)
    return coef, mean, std


def _ridge_predict(model: PairwiseModel, x: np.ndarray) -> np.ndarray:
    return _expand((x - model.mean) / model.std) @ model.coef


def _pairwise_features(
    descriptor_x: np.ndarray,
    forecast_x: np.ndarray,
    *,
    champion_idx: int,
    candidate_indices: np.ndarray,
    model_count: int,
) -> np.ndarray:
    rows = len(candidate_indices)
    desc = descriptor_x
    champ = forecast_x[:, champion_idx, :]
    cand = forecast_x[np.arange(rows), candidate_indices, :]
    one_hot = np.zeros((rows, model_count), dtype=np.float64)
    one_hot[np.arange(rows), candidate_indices] = 1.0
    return np.concatenate([desc, champ, cand, cand - champ, np.abs(cand - champ), one_hot], axis=1)


def _make_pairwise_training(
    descriptor_x: np.ndarray,
    forecast_x: np.ndarray,
    losses: np.ndarray,
    *,
    champion_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows, model_count = losses.shape
    xs = []
    ys = []
    for cand in range(model_count):
        if cand == champion_idx:
            continue
        idx = np.full(windows, cand, dtype=np.int64)
        xs.append(
            _pairwise_features(
                descriptor_x,
                forecast_x,
                champion_idx=champion_idx,
                candidate_indices=idx,
                model_count=model_count,
            )
        )
        ys.append(losses[:, champion_idx] - losses[:, cand])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _margin_grid(predicted_delta: np.ndarray) -> list[float]:
    positive = predicted_delta[predicted_delta > 0]
    if len(positive) == 0:
        return [0.0, math.inf]
    margins = {0.0}
    for q in MARGIN_QUANTILES:
        margins.add(float(np.quantile(positive, q)))
    margins.add(math.inf)
    return sorted(margins)


def select_validation_champion(train_losses: np.ndarray) -> tuple[int, float]:
    """Select the best single expert on the most recent past validation slice."""
    split = max(2, min(len(train_losses) - 1, int(len(train_losses) * 0.7)))
    val_losses = train_losses[split:]
    pick = int(val_losses.mean(axis=0).argmin())
    return pick, float(val_losses[:, pick].mean())


def chronological_safety_folds(n: int) -> list[tuple[int, int]]:
    """Return fit/validation folds inside the past split.

    Each fold trains on [0:fit_end) and validates on [fit_end:val_end).  The
    folds intentionally cover several past regimes; a switch policy must be
    safe across all of them before it is allowed to beat the default champion.
    """
    folds: list[tuple[int, int]] = []
    val_len = max(20, int(n * 0.12))
    for frac in (0.45, 0.55, 0.65, 0.75):
        fit_end = int(n * frac)
        val_end = min(n, fit_end + val_len)
        if fit_end >= 30 and val_end - fit_end >= 20:
            folds.append((fit_end, val_end))
    if not folds:
        fit_end = max(2, min(n - 1, int(n * 0.7)))
        folds.append((fit_end, n))
    return folds


def margin_from_fraction(losses: np.ndarray, champion_idx: int, margin_frac: float) -> float:
    if math.isinf(margin_frac):
        return math.inf
    scale = float(np.mean(losses[:, champion_idx]))
    return float(max(0.0, margin_frac) * max(scale, 1e-12))


def json_safe(value: object) -> object:
    """Convert non-finite floats to explicit JSON-safe sentinels."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "never_switch"
    return value


def fit_pairwise_model(
    descriptor_x: np.ndarray,
    forecast_x: np.ndarray,
    losses: np.ndarray,
    *,
    champion_idx: int,
    alpha: float,
    margin: float,
) -> PairwiseModel:
    x_pair, y_pair = _make_pairwise_training(descriptor_x, forecast_x, losses, champion_idx=champion_idx)
    coef, mean, std = _ridge_fit(x_pair, y_pair, alpha)
    return PairwiseModel(coef=coef, mean=mean, std=std, alpha=alpha, margin=margin, champion_idx=champion_idx)


def choose_with_pairwise_gate(
    model: PairwiseModel,
    descriptor_x: np.ndarray,
    forecast_x: np.ndarray,
    *,
    model_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = len(descriptor_x)
    candidates = [idx for idx in range(model_count) if idx != model.champion_idx]
    deltas = np.full((windows, model_count), -math.inf, dtype=np.float64)
    for cand in candidates:
        cand_idx = np.full(windows, cand, dtype=np.int64)
        pair_x = _pairwise_features(
            descriptor_x,
            forecast_x,
            champion_idx=model.champion_idx,
            candidate_indices=cand_idx,
            model_count=model_count,
        )
        deltas[:, cand] = _ridge_predict(model, pair_x)
    best_candidate = deltas.argmax(axis=1)
    best_delta = deltas[np.arange(windows), best_candidate]
    picks = np.where(best_delta > model.margin, best_candidate, model.champion_idx).astype(np.int64)
    return picks, best_delta


def tune_safe_switch(
    descriptor_x: np.ndarray,
    forecast_x: np.ndarray,
    losses: np.ndarray,
    *,
    feedback_delay: int,
) -> tuple[PairwiseModel, dict[str, object]]:
    """Tune alpha/margin with multi-fold chronological safety validation."""
    folds = chronological_safety_folds(len(losses))
    diagnostics = []
    best_positive: tuple[float, float, float, float, float] | None = None
    for alpha in RIDGE_ALPHA_GRID:
        for margin_frac in MARGIN_FRAC_GRID:
            fold_losses = []
            fold_champion_losses = []
            fold_switch_rates = []
            fold_improvements = []
            fold_champions = []
            for fit_end, val_end in folds:
                fit_desc, val_desc = descriptor_x[:fit_end], descriptor_x[fit_end:val_end]
                fit_forecast, val_forecast = forecast_x[:fit_end], forecast_x[fit_end:val_end]
                fit_losses, val_losses = losses[:fit_end], losses[fit_end:val_end]
                champion_idx, champion_recent_val_loss = select_validation_champion(fit_losses)
                margin = margin_from_fraction(fit_losses, champion_idx, margin_frac)
                model = fit_pairwise_model(
                    fit_desc,
                    fit_forecast,
                    fit_losses,
                    champion_idx=champion_idx,
                    alpha=alpha,
                    margin=margin,
                )
                picks, _ = choose_with_pairwise_gate(model, val_desc, val_forecast, model_count=losses.shape[1])
                val_loss = mean_selected_loss(val_losses, picks)
                champion_loss = float(val_losses[:, champion_idx].mean())
                fold_losses.append(val_loss)
                fold_champion_losses.append(champion_loss)
                fold_switch_rates.append(float(np.mean(picks != champion_idx)))
                fold_improvements.append(champion_loss - val_loss)
                fold_champions.append(int(champion_idx))
                _ = champion_recent_val_loss
            mean_improvement = float(np.mean(fold_improvements))
            min_improvement = float(np.min(fold_improvements))
            max_regret = float(max(0.0, -min_improvement))
            mean_loss = float(np.mean(fold_losses))
            mean_champion_loss = float(np.mean(fold_champion_losses))
            mean_improvement_frac = mean_improvement / max(mean_champion_loss, 1e-12)
            mean_switch_rate = float(np.mean(fold_switch_rates))
            robust_positive = mean_improvement_frac >= ROBUST_IMPROVEMENT_FRAC_MIN and max_regret <= 0.0
            diagnostics.append(
                {
                    "alpha": alpha,
                    "margin_frac": margin_frac,
                    "mean_validation_loss": mean_loss,
                    "mean_champion_loss": mean_champion_loss,
                    "mean_improvement": mean_improvement,
                    "mean_improvement_frac": mean_improvement_frac,
                    "min_improvement": min_improvement,
                    "max_regret": max_regret,
                    "mean_switch_rate": mean_switch_rate,
                    "fold_champion_indices": fold_champions,
                    "robust_positive": robust_positive,
                }
            )
            if robust_positive:
                # Prefer larger robust improvement, then lower switch rate.
                score = (mean_improvement, -mean_switch_rate, -alpha)
                if best_positive is None or score > (best_positive[0], -best_positive[3], -best_positive[1]):
                    best_positive = (mean_improvement, alpha, margin_frac, mean_switch_rate, max_regret)

    selected_alpha = best_positive[1] if best_positive is not None else RIDGE_ALPHA_GRID[-1]
    selected_margin_frac = best_positive[2] if best_positive is not None else math.inf
    selected_mean_improvement = best_positive[0] if best_positive is not None else 0.0
    selected_mean_switch_rate = best_positive[3] if best_positive is not None else 0.0
    selected_max_regret = best_positive[4] if best_positive is not None else 0.0
    final_champion_idx, final_champion_validation_loss = select_validation_champion(losses)
    # Recompute final model on all past losses using the alpha/margin selected
    # by multi-fold chronological validation.
    final_margin = margin_from_fraction(losses, final_champion_idx, selected_margin_frac)
    final_model = fit_pairwise_model(
        descriptor_x,
        forecast_x,
        losses,
        champion_idx=final_champion_idx,
        alpha=selected_alpha,
        margin=final_margin,
    )
    tune_report = {
        "safety_folds": [{"fit_end": int(a), "val_end": int(b)} for a, b in folds],
        "selected_alpha": float(selected_alpha),
        "selected_margin_frac": float(selected_margin_frac),
        "selected_margin": float(final_margin),
        "selected_mean_improvement": float(selected_mean_improvement),
        "selected_mean_switch_rate": float(selected_mean_switch_rate),
        "selected_max_regret": float(selected_max_regret),
        "selected_by": "robust_positive_backtest" if best_positive is not None else "fallback_no_switch",
        "final_champion_idx": int(final_champion_idx),
        "final_champion_validation_loss": float(final_champion_validation_loss),
        "grid": diagnostics,
        "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
        "margin_frac_grid": list(MARGIN_FRAC_GRID),
        "robust_improvement_frac_min": ROBUST_IMPROVEMENT_FRAC_MIN,
        "selection": "multi-fold chronological safety validation on past split only",
        "feedback_delay_windows": int(feedback_delay),
    }
    return final_model, tune_report


def _season_lag_for(dataset: str) -> int:
    if dataset in {"ETTh1", "ETTh2"}:
        return 24
    if dataset in {"ETTm1", "ETTm2", "Weather"}:
        return 96
    if dataset == "Exchange":
        return 7
    return 24


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
    descriptor_names, descriptor_x_raw = descriptors(context)
    pred_cube = load_or_synthesize_predictions(
        results_root=results_root,
        dataset=spec.artifact_tag,
        lookback=lookback,
        horizon=horizon,
        model_names=model_names,
        context=context,
    )
    forecast_names = forecast_feature_names()
    forecast_x = forecast_features(context, pred_cube, season_lag=_season_lag_for(spec.dataset))

    split = max(10, min(len(losses) - 10, int(len(losses) * train_frac)))
    train_desc_raw, test_desc_raw = descriptor_x_raw[:split], descriptor_x_raw[split:]
    train_losses, test_losses = losses[:split], losses[split:]
    train_forecast, test_forecast = forecast_x[:split], forecast_x[split:]
    train_desc, test_desc, _, _ = standardize(train_desc_raw, test_desc_raw)

    train_mean = train_losses.mean(axis=0)
    best_single_idx = int(train_mean.argmin())
    best_single_loss = float(test_losses[:, best_single_idx].mean())
    val_single_loss, val_single_idx, val_single_validation_loss = validation_single_baseline(train_losses, test_losses)
    oracle_loss = float(test_losses.min(axis=1).mean())
    oracle_gap = best_single_loss - oracle_loss

    fs_params, fs_val_loss = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_loss = fixed_share_with_prior(train_losses, test_losses, feedback_delay=horizon, **fs_params)

    ridge_coef = fit_ridge_loss(train_desc, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_desc)
    ridge_loss = mean_selected_loss(test_losses, ridge_picks)

    pairwise_model, tune_report = tune_safe_switch(
        train_desc,
        train_forecast,
        train_losses,
        feedback_delay=horizon,
    )
    safe_picks, safe_delta = choose_with_pairwise_gate(
        pairwise_model,
        test_desc,
        test_forecast,
        model_count=len(model_names),
    )
    safe_loss = mean_selected_loss(test_losses, safe_picks)
    safe_rec = (best_single_loss - safe_loss) / oracle_gap if oracle_gap > 0 else math.nan

    return {
        "dataset": spec.dataset,
        "artifact_tag": spec.artifact_tag,
        "oracle_dir": str(spec.oracle_dir),
        "num_windows": int(len(losses)),
        "train_windows": int(split),
        "test_windows": int(len(test_losses)),
        "model_names": model_names,
        "descriptor_names": descriptor_names,
        "forecast_feature_names": forecast_names,
        "best_single_model_train_selected": model_names[best_single_idx],
        "best_single_loss": best_single_loss,
        "validation_single_model": model_names[val_single_idx],
        "validation_single_loss": val_single_loss,
        "validation_single_validation_loss": val_single_validation_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_abs": float(oracle_gap),
        "fixed_share_loss": fs_loss,
        "fixed_share_params": fs_params,
        "fixed_share_validation_loss": float(fs_val_loss),
        "fixed_share_grid": {
            "lr": list(FS_LR_GRID),
            "alpha": list(FS_ALPHA_GRID),
            "selection": "chronological validation slice of past split with delayed feedback",
        },
        "descriptor_ridge_loss": ridge_loss,
        "safe_switch_loss": safe_loss,
        "safe_switch_gap_recovered_frac": float(safe_rec),
        "safe_switch_beats_fixed_share": bool(safe_loss < fs_loss),
        "safe_switch_beats_descriptor_ridge": bool(safe_loss < ridge_loss),
        "safe_switch_beats_validation_single": bool(safe_loss < val_single_loss),
        "gate_pass": bool(safe_loss < fs_loss and safe_loss < ridge_loss and safe_loss < val_single_loss),
        "safe_switch_params": {
            "alpha": float(pairwise_model.alpha),
            "margin": float(pairwise_model.margin),
            "champion_model": model_names[pairwise_model.champion_idx],
            "champion_idx": int(pairwise_model.champion_idx),
            "tuning": tune_report,
        },
        "safe_switch_pick_counts": {
            model_names[int(i)]: int(np.count_nonzero(safe_picks == i)) for i in np.unique(safe_picks)
        },
        "safe_switch_rate": float(np.mean(safe_picks != pairwise_model.champion_idx)),
        "safe_switch_mean_predicted_delta": float(np.mean(safe_delta[np.isfinite(safe_delta)])),
        "feedback_delay_windows": int(horizon),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = [
        analyze_one(spec, results_root=args.results_root, lookback=args.lookback, horizon=args.horizon, train_frac=args.train_frac)
        for spec in default_specs(args.oracle_root, lookback=args.lookback, horizon=args.horizon, datasets=tuple(args.datasets))
    ]
    passed = all(row["gate_pass"] for row in rows)
    result = {
        "milestone": "M9",
        "goal": "Causal champion-risk gate over validation-selected champion.",
        "gate": "Safe-switch loss must beat delayed Fixed-Share, descriptor ridge, and validation-selected single expert on every battlefield.",
        "train_frac": args.train_frac,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "rows": rows,
        "gate_pass": bool(passed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "champion_risk_gate_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M9 causal champion-risk gate.")
    p.add_argument("--oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/champion_risk_gate"))
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
            "safe_switch": row["safe_switch_loss"],
            "safe_switch_rate": row["safe_switch_rate"],
            "gate_pass": row["gate_pass"],
        }
        for row in result["rows"]
    ]
    print(json.dumps({"gate_pass": result["gate_pass"], "rows": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
