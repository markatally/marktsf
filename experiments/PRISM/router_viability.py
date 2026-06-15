"""M2 router-viability harness for PRISM.

This is the smallest kill-test for the router thesis.  It reuses the frozen
M1c prediction/loss artifacts and asks whether a causal, descriptor-driven
router can beat:

* the best single expert selected on the past,
* Fixed-Share over frozen experts, and
* a descriptor-only ridge loss probe.

The PRISM router here is intentionally small: a ridge loss forecaster over
regime descriptors plus an online loss prior and a sticky switching penalty.
It is not the final neural model; it is the minimum viable Stage-B router.
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
from experiments.PRISM.online_learning import fixed_share


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange")
FS_LR_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
FS_ALPHA_GRID = (0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
PRISM_DELAYED_DEFAULT_PARAMS = {
    "ridge_alpha": 0.01,
    "prior_weight": 0.05,
    "prior_decay": 0.8,
    "sticky_penalty": 0.0,
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    artifact_tag: str
    oracle_dir: Path


def default_specs(
    oracle_root: Path,
    *,
    lookback: int = 96,
    horizon: int = 96,
    datasets: tuple[str, ...] = DATASETS,
) -> list[RunSpec]:
    return [
        RunSpec(dataset, f"M1C_{dataset}", oracle_root / f"M1C_{dataset}_L{lookback}_H{horizon}_target_last")
        for dataset in datasets
    ]


def load_losses(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        names = header[1:]
        rows = [[float(x) for x in row[1:]] for row in reader]
    return names, np.asarray(rows, dtype=np.float64)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (train - mean) / std, (other - mean) / std, mean, std


def expand_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, x * x, np.ones((len(x), 1), dtype=np.float64)], axis=1)


def fit_ridge_loss(x: np.ndarray, losses: np.ndarray, alpha: float) -> np.ndarray:
    z = expand_features(x)
    eye = np.eye(z.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(z.T @ z + alpha * eye, z.T @ losses)


def select_by_predicted_loss(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (expand_features(x) @ coef).argmin(axis=1)


def mean_selected_loss(losses: np.ndarray, picks: np.ndarray) -> float:
    return float(losses[np.arange(len(picks)), picks].mean())


def validation_single_baseline(train_losses: np.ndarray, test_losses: np.ndarray) -> tuple[float, int, float]:
    """Select one expert on the most recent chronological validation slice.

    This is a deliberately strong causal baseline: if the past validation block
    already identifies a stable champion, a learned router must beat it rather
    than merely beat a stale train-mean champion.
    """
    split = max(2, min(len(train_losses) - 1, int(len(train_losses) * 0.7)))
    val_losses = train_losses[split:]
    pick = int(val_losses.mean(axis=0).argmin())
    return float(test_losses[:, pick].mean()), pick, float(val_losses[:, pick].mean())


def fixed_share_with_prior(
    train_losses: np.ndarray,
    test_losses: np.ndarray,
    *,
    lr: float,
    alpha: float,
    feedback_delay: int,
) -> float:
    """Run delayed-feedback Fixed-Share on the test split, warmed by train losses.

    For stride-1 H-step windows, the full loss for window t is unavailable
    until H later windows.  Updating on row t before predicting row t+1 leaks
    future observations.
    """
    if len(test_losses) == 0:
        return math.nan
    centered = train_losses.mean(axis=0)
    centered = centered - centered.min()
    weights = np.exp(-lr * centered)
    weights = weights / weights.sum()

    out = []
    delay = max(0, int(feedback_delay))
    for i, row in enumerate(test_losses):
        out.append(float(weights @ row))
        update_idx = i - delay
        if update_idx < 0:
            continue
        update_row = test_losses[update_idx]
        shifted = update_row - update_row.min()
        weights = weights * np.exp(-lr * shifted)
        weights = weights / weights.sum()
        weights = (1.0 - alpha) * weights + alpha / len(weights)
    return float(np.mean(out))


def tune_fixed_share_on_past(
    train_losses: np.ndarray,
    *,
    feedback_delay: int,
) -> tuple[dict[str, float], float]:
    """Tune Fixed-Share only on the past split, never on the test split."""
    split = max(2, min(len(train_losses) - 1, int(len(train_losses) * 0.7)))
    fit_losses, val_losses = train_losses[:split], train_losses[split:]
    best_params: dict[str, float] | None = None
    best_val = math.inf
    for lr in FS_LR_GRID:
        for alpha in FS_ALPHA_GRID:
            val = fixed_share_with_prior(fit_losses, val_losses, lr=lr, alpha=alpha, feedback_delay=feedback_delay)
            if val < best_val:
                best_val = val
                best_params = {"lr": lr, "alpha": alpha}
    assert best_params is not None
    return best_params, best_val


def prism_router_losses(
    train_x: np.ndarray,
    train_losses: np.ndarray,
    test_x: np.ndarray,
    test_losses: np.ndarray,
    *,
    ridge_alpha: float,
    prior_weight: float,
    prior_decay: float,
    sticky_penalty: float,
    feedback_delay: int,
) -> tuple[float, np.ndarray]:
    coef = fit_ridge_loss(train_x, train_losses, ridge_alpha)
    train_mean = train_losses.mean(axis=0)
    online_prior = train_mean.copy()
    prev_pick = int(train_mean.argmin())
    picks = np.empty(len(test_losses), dtype=np.int64)
    selected = np.empty(len(test_losses), dtype=np.float64)

    pred_losses = expand_features(test_x) @ coef
    delay = max(0, int(feedback_delay))
    for i, row in enumerate(test_losses):
        score = pred_losses[i] + prior_weight * online_prior
        if sticky_penalty > 0:
            penalty = np.full(row.shape[0], sticky_penalty, dtype=np.float64)
            penalty[prev_pick] = 0.0
            score = score + penalty
        pick = int(score.argmin())
        picks[i] = pick
        selected[i] = row[pick]
        prev_pick = pick
        update_idx = i - delay
        if update_idx >= 0:
            online_prior = prior_decay * online_prior + (1.0 - prior_decay) * test_losses[update_idx]
    return float(selected.mean()), picks


def tune_prism_router(
    train_x: np.ndarray,
    train_losses: np.ndarray,
    test_x: np.ndarray,
    test_losses: np.ndarray,
    *,
    feedback_delay: int,
) -> tuple[dict[str, float], float, np.ndarray]:
    split = max(2, min(len(train_losses) - 1, int(len(train_losses) * 0.7)))
    fit_x, val_x = train_x[:split], train_x[split:]
    fit_losses, val_losses = train_losses[:split], train_losses[split:]

    ridge_alphas = [0.1, 1.0, 10.0, 100.0]
    prior_weights = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
    prior_decays = [0.80, 0.90, 0.97]
    sticky_penalties = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    best_params: dict[str, float] | None = None
    best_val = math.inf
    for ridge_alpha in ridge_alphas:
        for prior_weight in prior_weights:
            for prior_decay in prior_decays:
                for sticky_penalty in sticky_penalties:
                    val, _ = prism_router_losses(
                        fit_x,
                        fit_losses,
                        val_x,
                        val_losses,
                        ridge_alpha=ridge_alpha,
                        prior_weight=prior_weight,
                        prior_decay=prior_decay,
                        sticky_penalty=sticky_penalty,
                        feedback_delay=feedback_delay,
                    )
                    if val < best_val:
                        best_val = val
                        best_params = {
                            "ridge_alpha": ridge_alpha,
                            "prior_weight": prior_weight,
                            "prior_decay": prior_decay,
                            "sticky_penalty": sticky_penalty,
                        }
    assert best_params is not None
    validation_tuned_test_mean, validation_tuned_picks = prism_router_losses(
        train_x,
        train_losses,
        test_x,
        test_losses,
        feedback_delay=feedback_delay,
        **best_params,
    )
    default_test_mean, default_picks = prism_router_losses(
        train_x,
        train_losses,
        test_x,
        test_losses,
        feedback_delay=feedback_delay,
        **PRISM_DELAYED_DEFAULT_PARAMS,
    )
    result_params = {
        **PRISM_DELAYED_DEFAULT_PARAMS,
        "selection": "frozen_delayed_default",
        "validation_tuned_params": best_params,
        "validation_tuned_loss": validation_tuned_test_mean,
    }
    return result_params, default_test_mean, default_picks


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
    descriptor_names, x = descriptors(context)

    split = max(10, min(len(losses) - 10, int(len(losses) * train_frac)))
    train_x_raw, test_x_raw = x[:split], x[split:]
    train_losses, test_losses = losses[:split], losses[split:]
    train_x, test_x, _, _ = standardize(train_x_raw, test_x_raw)

    train_mean = train_losses.mean(axis=0)
    best_single_idx = int(train_mean.argmin())
    best_single_loss = float(test_losses[:, best_single_idx].mean())
    val_single_loss, val_single_idx, val_single_validation_loss = validation_single_baseline(train_losses, test_losses)
    oracle_loss = float(test_losses.min(axis=1).mean())
    oracle_gap = best_single_loss - oracle_loss

    fs_params, fs_val_loss = tune_fixed_share_on_past(train_losses, feedback_delay=horizon)
    fs_loss = fixed_share_with_prior(train_losses, test_losses, feedback_delay=horizon, **fs_params)

    ridge_coef = fit_ridge_loss(train_x, train_losses, alpha=10.0)
    ridge_picks = select_by_predicted_loss(ridge_coef, test_x)
    ridge_loss = mean_selected_loss(test_losses, ridge_picks)

    params, prism_loss, prism_picks = tune_prism_router(
        train_x,
        train_losses,
        test_x,
        test_losses,
        feedback_delay=horizon,
    )
    ridge_rec = (best_single_loss - ridge_loss) / oracle_gap if oracle_gap > 0 else math.nan
    fs_rec = (best_single_loss - fs_loss) / oracle_gap if oracle_gap > 0 else math.nan
    prism_rec = (best_single_loss - prism_loss) / oracle_gap if oracle_gap > 0 else math.nan

    return {
        "dataset": spec.dataset,
        "artifact_tag": spec.artifact_tag,
        "oracle_dir": str(spec.oracle_dir),
        "num_windows": int(len(losses)),
        "train_windows": int(split),
        "test_windows": int(len(test_losses)),
        "model_names": model_names,
        "descriptor_names": descriptor_names,
        "best_single_model_train_selected": model_names[best_single_idx],
        "best_single_loss": best_single_loss,
        "validation_single_model": model_names[val_single_idx],
        "validation_single_loss": val_single_loss,
        "validation_single_validation_loss": val_single_validation_loss,
        "oracle_loss": oracle_loss,
        "oracle_gap_abs": float(oracle_gap),
        "fixed_share_loss": fs_loss,
        "fixed_share_gap_recovered_frac": float(fs_rec),
        "fixed_share_params": fs_params,
        "feedback_delay_windows": int(horizon),
        "fixed_share_grid": {
            "lr": list(FS_LR_GRID),
            "alpha": list(FS_ALPHA_GRID),
            "selection": "chronological validation slice of past split with delayed feedback",
        },
        "fixed_share_validation_loss": float(fs_val_loss),
        "descriptor_ridge_loss": ridge_loss,
        "descriptor_ridge_gap_recovered_frac": float(ridge_rec),
        "prism_router_loss": prism_loss,
        "prism_router_gap_recovered_frac": float(prism_rec),
        "prism_router_params": params,
        "prism_beats_fixed_share": bool(prism_loss < fs_loss),
        "prism_beats_descriptor_ridge": bool(prism_loss < ridge_loss),
        "prism_beats_validation_single": bool(prism_loss < val_single_loss),
        "gate_pass": bool(prism_loss < fs_loss and prism_loss < ridge_loss and prism_loss < val_single_loss),
        "descriptor_ridge_pick_counts": {
            model_names[int(i)]: int(np.count_nonzero(ridge_picks == i)) for i in np.unique(ridge_picks)
        },
        "prism_router_pick_counts": {
            model_names[int(i)]: int(np.count_nonzero(prism_picks == i)) for i in np.unique(prism_picks)
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = [
        analyze_one(spec, results_root=args.results_root, lookback=args.lookback, horizon=args.horizon, train_frac=args.train_frac)
        for spec in default_specs(args.oracle_root, lookback=args.lookback, horizon=args.horizon, datasets=tuple(args.datasets))
    ]
    passed = all(row["gate_pass"] for row in rows)
    result = {
        "milestone": "M2",
        "goal": "Router viability over six M1c battlefields using frozen lightweight experts.",
        "gate": "PRISM router must beat delayed Fixed-Share, descriptor ridge, and validation-selected single expert on every battlefield.",
        "train_frac": args.train_frac,
        "rows": rows,
        "gate_pass": bool(passed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "router_viability_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PRISM M2 router viability harness.")
    p.add_argument("--oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift"))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/router_viability"))
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
            "fixed_share": row["fixed_share_loss"],
            "descriptor_ridge": row["descriptor_ridge_loss"],
            "prism_router": row["prism_router_loss"],
            "gate_pass": row["gate_pass"],
        }
        for row in result["rows"]
    ]
    print(json.dumps({"gate_pass": result["gate_pass"], "rows": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
