"""Oracle Drift Study, M1 first artifact.

This module consumes frozen prediction artifacts from baseline runs and asks the
first PRISM question: does the per-window best architecture change over time?
It intentionally does not train models. The M1 gate starts from measured
``pred.npy`` / ``true.npy`` files, converts them into per-window losses, and
writes the oracle trajectory that later PRISM routing must learn to approximate.

M1b additions
-------------
* ``--window-loss {mse,da,ic}``  — loss function choice (default: mse)
* ``--include-anchors``          — add ZeroPred/Persistence as pool members
* Anchor diagnostics (``anchor_mse``, ``best_vs_anchor_pct``) always emitted
* Streak statistics (``median_streak``, ``max_streak``)
* IID-argmin noise null (``switch_null``) and switch ratio (``switch_ratio``)
* Moving-block permutation p-value (``switch_pvalue``)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MODELS = ("DLinear", "PatchTST", "TiDE", "TimeXer")
LOWER_IS_BETTER: dict[str, bool] = {"mse": True, "da": False, "ic": False}


@dataclass(frozen=True)
class StudyConfig:
    results_root: str
    output_dir: str
    dataset: str
    lookback: int
    horizon: int
    models: tuple[str, ...]
    target_channel: int | None
    loss_type: str = "mse"
    include_anchors: bool = False
    seed: int | None = None  # if set, filters result dirs to those containing f"seed{seed}"


# ──────────────────────────────────────────────────────────────────────────────
# Directory resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_result_dir(
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    model: str,
    seed: int | None = None,
) -> Path:
    """Find the single TSLib result directory for one model/horizon cell."""
    prefix = f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model}_{dataset}_"
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if seed is not None:
        matches = [p for p in matches if f"seed{seed}" in p.name]
    if not matches:
        raise FileNotFoundError(
            f"No result directory found for {dataset} L={lookback} H={horizon} model={model} "
            f"(seed={seed}) under {results_root}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"Ambiguous result directories for {model}: {names}")
    return matches[0]


# ──────────────────────────────────────────────────────────────────────────────
# Per-window loss functions
# ──────────────────────────────────────────────────────────────────────────────

def _select_channel(arr: np.ndarray, target_channel: int | None) -> np.ndarray:
    """Select a single channel from [..., C]; if None, keep all."""
    if target_channel is None:
        return arr
    c = target_channel if target_channel >= 0 else arr.shape[-1] + target_channel
    if c < 0 or c >= arr.shape[-1]:
        raise ValueError(f"target_channel out of range: {target_channel} for C={arr.shape[-1]}")
    return arr[..., c : c + 1]


def _window_mse(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-window mean squared error. Arrays: [W, H, C]. Returns [W]."""
    err = (pred - true) ** 2
    return err.reshape(err.shape[0], -1).mean(axis=1)


def _window_da(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-window directional accuracy (higher = better). Arrays: [W, H, C]."""
    match = (np.sign(pred) == np.sign(true)).astype(np.float32)
    return match.reshape(match.shape[0], -1).mean(axis=1)


def _window_ic(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-window Pearson IC (higher = better). Arrays: [W, H, C]."""
    W = pred.shape[0]
    p_flat = pred.reshape(W, -1)
    t_flat = true.reshape(W, -1)
    p_c = p_flat - p_flat.mean(axis=1, keepdims=True)
    t_c = t_flat - t_flat.mean(axis=1, keepdims=True)
    num = (p_c * t_c).sum(axis=1)
    denom = np.sqrt((p_c ** 2).sum(axis=1)) * np.sqrt((t_c ** 2).sum(axis=1))
    with np.errstate(invalid="ignore"):
        return np.where(denom > 0, num / denom, 0.0).astype(np.float64)


_LOSS_FNS = {"mse": _window_mse, "da": _window_da, "ic": _window_ic}


def load_window_loss(
    result_dir: Path,
    target_channel: int | None,
    loss_type: str = "mse",
) -> np.ndarray:
    """Load one model's pred/true artifacts and return per-window loss [W]."""
    pred_path = result_dir / "pred.npy"
    true_path = result_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"{result_dir} must contain pred.npy and true.npy")
    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected [windows, horizon, channels], got {pred.shape}")
    pred = _select_channel(pred, target_channel)
    true = _select_channel(true, target_channel)
    return _LOSS_FNS[loss_type](pred, true)


# Backward-compatible alias for callers using the old name.
def load_window_mse(result_dir: Path, target_channel: int | None) -> np.ndarray:
    return load_window_loss(result_dir, target_channel, "mse")


# ──────────────────────────────────────────────────────────────────────────────
# Anchor losses  (synthetic; computed from true.npy only)
# ──────────────────────────────────────────────────────────────────────────────

def compute_anchor_losses(
    true: np.ndarray,
    target_channel: int | None,
    loss_type: str = "mse",
) -> dict[str, np.ndarray]:
    """
    Compute naive-anchor per-window losses from the ground-truth array.

    ZeroPred   – predict 0 for every step (= unconditional mean in
                 standardized space, i.e. the no-skill baseline).
    Persistence – predict the last observed input value (constant over the
                 horizon). With stride-1 sliding windows, the last input
                 value of window i equals true[i-1, 0, :] (the first
                 step of the previous window's horizon). Window 0 is
                 treated as its own first value (boundary approximation).

    Returns a dict mapping anchor name → per-window loss array [W].
    """
    true_tc = _select_channel(true, target_channel).astype(np.float32)
    W, H, C = true_tc.shape
    loss_fn = _LOSS_FNS[loss_type]

    # ZeroPred: constant 0 prediction.
    zero_pred = np.zeros_like(true_tc)
    anchors = {"ZeroPred": loss_fn(zero_pred, true_tc)}

    # Persistence: last-input value repeated over horizon.
    # last_input[i] = data[i+L-1] = true[i-1, 0, :] for i >= 1.
    last_vals = np.empty((W, C), dtype=np.float32)
    last_vals[0] = true_tc[0, 0, :]          # boundary: approximation
    last_vals[1:] = true_tc[:-1, 0, :]       # true[i-1, 0, :] for i >= 1
    persistence_pred = np.repeat(last_vals[:, np.newaxis, :], H, axis=1)  # [W, H, C]
    anchors["Persistence"] = loss_fn(persistence_pred, true_tc)

    # HAR-style anchor: exponentially weighted mean of past first-step true values.
    # Captures slow mean-reversion of log-RV (or any autocorrelated target).
    # Uses a 24-window half-life (≈ 1 day for hourly series).
    alpha_ewm = 1.0 - np.exp(-np.log(2) / 24.0)  # half-life = 24
    ewm = np.empty((W, C), dtype=np.float32)
    ewm[0] = true_tc[0, 0, :]
    for i in range(1, W):
        ewm[i] = alpha_ewm * true_tc[i - 1, 0, :] + (1.0 - alpha_ewm) * ewm[i - 1]
    har_pred = np.repeat(ewm[:, np.newaxis, :], H, axis=1)
    anchors["HAR_EWM"] = loss_fn(har_pred, true_tc)
    return anchors


# ──────────────────────────────────────────────────────────────────────────────
# Loss collection (models + optional anchors)
# ──────────────────────────────────────────────────────────────────────────────

def collect_losses(
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    models: Iterable[str],
    target_channel: int | None,
    loss_type: str = "mse",
    include_anchors: bool = False,
    seed: int | None = None,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """
    Load per-window losses for each model; optionally prepend anchor rows.

    Returns
    -------
    model_names : list of str (models + anchors if requested)
    losses      : [W, M] float64 array
    source_dirs : model → path string (anchors use "synthetic:<name>")
    """
    model_names: list[str] = []
    columns: list[np.ndarray] = []
    source_dirs: dict[str, str] = {}
    expected_windows: int | None = None
    first_true: np.ndarray | None = None

    for model in models:
        result_dir = resolve_result_dir(results_root, dataset, lookback, horizon, model, seed=seed)
        losses_col = load_window_loss(result_dir, target_channel, loss_type)
        if expected_windows is None:
            expected_windows = losses_col.shape[0]
            # Load true.npy once for anchor computation.
            first_true = np.load(result_dir / "true.npy")
        elif losses_col.shape[0] != expected_windows:
            raise ValueError(
                f"Window count mismatch for {model}: "
                f"got {losses_col.shape[0]}, expected {expected_windows}"
            )
        model_names.append(model)
        columns.append(losses_col)
        source_dirs[model] = str(result_dir)

    if not columns:
        raise ValueError("At least one model is required")

    if include_anchors and first_true is not None:
        anchor_losses = compute_anchor_losses(first_true, target_channel, loss_type)
        for anchor_name, anchor_col in anchor_losses.items():
            model_names.append(anchor_name)
            columns.append(anchor_col)
            source_dirs[anchor_name] = f"synthetic:{anchor_name}"

    return model_names, np.stack(columns, axis=1), source_dirs


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic statistics
# ──────────────────────────────────────────────────────────────────────────────

def _best_idx_from_losses(losses: np.ndarray, lower_is_better: bool) -> np.ndarray:
    return losses.argmin(axis=1) if lower_is_better else losses.argmax(axis=1)


def compute_streak_stats(best_idx: np.ndarray) -> tuple[float, int]:
    """Return (median_streak, max_streak) of consecutive same-model runs."""
    runs: list[int] = []
    current = best_idx[0]
    run_len = 1
    for v in best_idx[1:]:
        if v == current:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
            current = v
    runs.append(run_len)
    return float(np.median(runs)), int(max(runs))


def iid_switch_null(win_fractions: dict[str, float]) -> float:
    """IID-argmin switch rate null = 1 - Σ p_i²."""
    return float(1.0 - sum(p ** 2 for p in win_fractions.values()))


def permutation_pvalue(
    losses: np.ndarray,
    observed_switch_rate: float,
    lower_is_better: bool = True,
    n_perms: int = 999,
    block_size: int = 24,
    rng_seed: int = 42,
) -> float:
    """
    Moving-block permutation p-value for the persistence of oracle architecture.

    The null is that windows are exchangeable in blocks (no temporal structure).
    The test statistic is the oracle switch rate (lower = more persistent).
    A low p-value indicates significantly more persistence than expected under
    block-independence.

    p = P(null switch rate ≤ observed | H₀)
    """
    W, M = losses.shape
    n_blocks = math.ceil(W / block_size)
    pad = n_blocks * block_size - W

    losses_padded = np.concatenate([losses, losses[:pad]], axis=0) if pad > 0 else losses
    blocks = losses_padded.reshape(n_blocks, block_size, M)

    rng = np.random.default_rng(rng_seed)
    null_rates: list[float] = []
    for _ in range(n_perms):
        perm_idx = rng.permutation(n_blocks)
        perm_losses = blocks[perm_idx].reshape(-1, M)[:W]
        best = _best_idx_from_losses(perm_losses, lower_is_better)
        rate = float(np.count_nonzero(best[1:] != best[:-1])) / (W - 1)
        null_rates.append(rate)

    return float(np.mean(np.array(null_rates) <= observed_switch_rate))


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory and summary
# ──────────────────────────────────────────────────────────────────────────────

def build_trajectory(
    model_names: list[str],
    losses: np.ndarray,
    lower_is_better: bool = True,
) -> list[dict[str, object]]:
    """Build one row per window with oracle choice and margin diagnostics."""
    agg = losses.mean(axis=0)
    best_single_idx = int(agg.argmin() if lower_is_better else agg.argmax())
    best_idx = _best_idx_from_losses(losses, lower_is_better)
    order = np.argsort(losses, axis=1)
    if not lower_is_better:
        order = order[:, ::-1]

    rows: list[dict[str, object]] = []
    for i in range(losses.shape[0]):
        first = int(order[i, 0])
        second = int(order[i, 1]) if losses.shape[1] > 1 else first
        margin = (
            float(losses[i, first] - losses[i, second])  # higher-is-better: best > second
            if not lower_is_better
            else float(losses[i, second] - losses[i, first])  # lower-is-better: second > best
        )
        rows.append(
            {
                "window_index": i,
                "best_model": model_names[first],
                "best_loss": float(losses[i, first]),
                "second_model": model_names[second],
                "second_loss": float(losses[i, second]),
                "margin_vs_second": margin,
                "best_single_model": model_names[best_single_idx],
                "best_single_loss": float(losses[i, best_single_idx]),
                "oracle_gain_vs_best_single": float(
                    losses[i, best_single_idx] - losses[i, first]
                    if lower_is_better
                    else losses[i, first] - losses[i, best_single_idx]
                ),
            }
        )
    return rows


def summarize(
    cfg: StudyConfig,
    model_names: list[str],
    losses: np.ndarray,
    source_dirs: dict[str, str],
    true_for_anchors: np.ndarray | None = None,
) -> dict[str, object]:
    """
    Compute M1 headline statistics from a window-loss matrix.

    Always computes anchor-MSE diagnostics (from true_for_anchors or via
    ZeroPred=mean(true²) approximation when the array is unavailable).
    """
    lower = LOWER_IS_BETTER.get(cfg.loss_type, True)

    agg = losses.mean(axis=0)
    best_single_idx = int(agg.argmin() if lower else agg.argmax())
    best_idx = _best_idx_from_losses(losses, lower)

    oracle_loss = float(losses[np.arange(losses.shape[0]), best_idx].mean())
    best_single_loss = float(agg[best_single_idx])

    # Gap: positive = oracle better than best single.
    gap_abs = (best_single_loss - oracle_loss) if lower else (oracle_loss - best_single_loss)
    switch_count = int(np.count_nonzero(best_idx[1:] != best_idx[:-1])) if losses.shape[0] > 1 else 0
    win_counts = {m: int(np.count_nonzero(best_idx == i)) for i, m in enumerate(model_names)}
    win_fracs = {m: win_counts[m] / losses.shape[0] for m in model_names}

    margins = []
    if losses.shape[1] > 1:
        sorted_l = np.sort(losses, axis=1)
        if lower:
            margins = (sorted_l[:, 1] - sorted_l[:, 0]).tolist()
        else:
            margins = (sorted_l[:, -1] - sorted_l[:, -2]).tolist()

    # Streak statistics.
    median_streak, max_streak = compute_streak_stats(best_idx)

    # IID-argmin switch null and switch ratio.
    switch_rate = float(switch_count / (losses.shape[0] - 1)) if losses.shape[0] > 1 else 0.0
    null_rate = iid_switch_null(win_fracs)
    switch_ratio = switch_rate / null_rate if null_rate > 0 else math.nan

    # Permutation p-value (skip if too few windows for meaningful test).
    if losses.shape[0] >= 100:
        switch_pvalue = permutation_pvalue(losses, switch_rate, lower_is_better=lower)
    else:
        switch_pvalue = math.nan

    # Anchor diagnostics (always MSE-based for comparability across loss_type choices).
    anchor_mse: float = math.nan
    anchor_mse_by_type: dict[str, float] = {}
    best_vs_anchor_pct: float = math.nan
    if true_for_anchors is not None:
        anchor_losses_mse = compute_anchor_losses(true_for_anchors, cfg.target_channel, "mse")
        anchor_mse_by_type = {k: float(v.mean()) for k, v in anchor_losses_mse.items()}
        # Use the best (lowest MSE) anchor as the comparison baseline.
        anchor_mse = float(min(anchor_mse_by_type.values()))
        # best_single MSE for anchor comparison.
        # For MSE loss_type, best_single_loss IS MSE; for DA/IC we use the model's aggregate MSE.
        if lower:
            best_single_mse_for_anchor = best_single_loss
        else:
            # Recompute MSE for the model that won on DA/IC.
            best_m_name = model_names[best_single_idx]
            if best_m_name in source_dirs and not source_dirs[best_m_name].startswith("synthetic:"):
                result_dir = Path(source_dirs[best_m_name])
                ms_pred = _select_channel(np.load(result_dir / "pred.npy"), cfg.target_channel)
                ms_true = _select_channel(np.load(result_dir / "true.npy"), cfg.target_channel)
                best_single_mse_for_anchor = float(_window_mse(ms_pred, ms_true).mean())
            else:
                best_single_mse_for_anchor = math.nan
        if anchor_mse > 0 and math.isfinite(best_single_mse_for_anchor):
            best_vs_anchor_pct = float((best_single_mse_for_anchor - anchor_mse) / anchor_mse * 100)

    return {
        "config": asdict(cfg),
        "source_dirs": source_dirs,
        "num_windows": int(losses.shape[0]),
        "num_models": int(losses.shape[1]),
        "loss_type": cfg.loss_type,
        "lower_is_better": lower,
        # Aggregate model performance.
        "aggregate_model_loss": {m: float(agg[i]) for i, m in enumerate(model_names)},
        "best_single_model": model_names[best_single_idx],
        "best_single_loss": best_single_loss,
        # Oracle.
        "oracle_loss": oracle_loss,
        "oracle_gap_abs": float(gap_abs),
        "oracle_gap_rel": float(gap_abs / abs(best_single_loss)) if best_single_loss != 0 else math.nan,
        # Switching.
        "switch_count": switch_count,
        "switch_rate": switch_rate,
        "switch_null": null_rate,
        "switch_ratio": switch_ratio,
        "switch_pvalue": switch_pvalue,
        # Streak.
        "median_streak": median_streak,
        "max_streak": max_streak,
        # Win counts.
        "win_counts": win_counts,
        "win_fractions": win_fracs,
        # Margin.
        "mean_margin_vs_second": float(np.mean(margins)) if margins else 0.0,
        "median_margin_vs_second": float(np.median(margins)) if margins else 0.0,
        # Anchors (always MSE-based, best anchor = min over ZeroPred/Persistence/HAR_EWM).
        "anchor_mse": anchor_mse,
        "anchor_mse_by_type": anchor_mse_by_type,
        "best_vs_anchor_pct": best_vs_anchor_pct,
        # Back-compat aliases for MSE studies.
        "aggregate_model_mse": {m: float(agg[i]) for i, m in enumerate(model_names)}
        if cfg.loss_type == "mse"
        else {},
        "best_single_mse": best_single_loss if cfg.loss_type == "mse" else math.nan,
        "oracle_mse": oracle_loss if cfg.loss_type == "mse" else math.nan,
        "oracle_gap_rel": float(gap_abs / abs(best_single_loss)) if best_single_loss != 0 else math.nan,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def write_window_losses(path: Path, model_names: list[str], losses: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_index", *model_names])
        for i in range(losses.shape[0]):
            writer.writerow([i, *[f"{losses[i, j]:.10g}" for j in range(losses.shape[1])]])


def write_trajectory(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "window_index",
        "best_model",
        "best_loss",
        "second_model",
        "second_loss",
        "margin_vs_second",
        "best_single_model",
        "best_single_loss",
        "oracle_gain_vs_best_single",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, model_names: list[str], losses: np.ndarray, lower_is_better: bool = True) -> None:
    """Write a compact best-architecture trajectory plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_idx = _best_idx_from_losses(losses, lower_is_better)
    x = np.arange(losses.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, height_ratios=[1, 2])
    axes[0].scatter(x, best_idx, c=best_idx, cmap="tab10", s=7, linewidths=0)
    axes[0].set_yticks(range(len(model_names)), model_names)
    axes[0].set_ylabel("Oracle pick")
    axes[0].set_title("Oracle best architecture over test windows")
    axes[0].grid(axis="x", alpha=0.15)

    for j, model in enumerate(model_names):
        rolling = rolling_mean(losses[:, j], width=max(5, min(49, losses.shape[0] // 25)))
        axes[1].plot(x, rolling, label=model, linewidth=1.3)
    axes[1].set_xlabel("Test window index")
    axes[1].set_ylabel("Rolling loss")
    axes[1].legend(ncol=min(4, len(model_names)), fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values
    kernel = np.ones(width) / width
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ──────────────────────────────────────────────────────────────────────────────
# Main study runner
# ──────────────────────────────────────────────────────────────────────────────

def run_study(cfg: StudyConfig) -> dict[str, object]:
    results_root = Path(cfg.results_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lower = LOWER_IS_BETTER.get(cfg.loss_type, True)

    model_names, losses, source_dirs = collect_losses(
        results_root=results_root,
        dataset=cfg.dataset,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        models=cfg.models,
        target_channel=cfg.target_channel,
        loss_type=cfg.loss_type,
        include_anchors=cfg.include_anchors,
        seed=cfg.seed,
    )

    # Load true.npy once for anchor diagnostics (always MSE-based).
    first_model = [m for m in model_names if not m.startswith("synthetic") and m not in ("ZeroPred", "Persistence")][0]
    first_result_dir = resolve_result_dir(results_root, cfg.dataset, cfg.lookback, cfg.horizon, first_model, seed=cfg.seed)
    true_arr = np.load(first_result_dir / "true.npy")

    trajectory = build_trajectory(model_names, losses, lower_is_better=lower)
    summary = summarize(cfg, model_names, losses, source_dirs, true_for_anchors=true_arr)

    write_window_losses(output_dir / "window_losses.csv", model_names, losses)
    write_trajectory(output_dir / "best_architecture_trajectory.csv", trajectory)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_plot(output_dir / "best_architecture_trajectory.png", model_names, losses, lower_is_better=lower)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Oracle Drift Study artifacts from TSLib predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", default="external/TSLib/results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--target-channel",
        default="-1",
        help="Channel index for MISO target; 'all' to average all channels.",
    )
    parser.add_argument(
        "--window-loss",
        choices=["mse", "da", "ic"],
        default="mse",
        help="Per-window loss: mse (lower=better), da/ic (higher=better).",
    )
    parser.add_argument(
        "--include-anchors",
        action="store_true",
        default=False,
        help="Add ZeroPred and Persistence as pool members.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Filter result directories to those containing 'seed<N>'. Required when multiple seeds exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_channel = None if args.target_channel == "all" else int(args.target_channel)
    cfg = StudyConfig(
        results_root=args.results_root,
        output_dir=args.output_dir,
        dataset=args.dataset,
        lookback=args.lookback,
        horizon=args.horizon,
        models=tuple(args.models),
        target_channel=target_channel,
        loss_type=args.window_loss,
        include_anchors=args.include_anchors,
        seed=args.seed,
    )
    summary = run_study(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
