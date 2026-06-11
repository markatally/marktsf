"""Oracle Drift Study, M1 first artifact.

This module consumes frozen prediction artifacts from baseline runs and asks the
first PRISM question: does the per-window best architecture change over time?
It intentionally does not train models. The M1 gate starts from measured
``pred.npy`` / ``true.npy`` files, converts them into per-window losses, and
writes the oracle trajectory that later PRISM routing must learn to approximate.
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


@dataclass(frozen=True)
class StudyConfig:
    results_root: str
    output_dir: str
    dataset: str
    lookback: int
    horizon: int
    models: tuple[str, ...]
    target_channel: int | None


def resolve_result_dir(
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    model: str,
) -> Path:
    """Find the single TSLib result directory for one model/horizon cell."""
    prefix = f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model}_{dataset}_"
    matches = sorted(p for p in results_root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(
            f"No result directory found for {dataset} L={lookback} H={horizon} model={model} "
            f"under {results_root}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"Ambiguous result directories for {model}: {names}")
    return matches[0]


def load_window_mse(result_dir: Path, target_channel: int | None) -> np.ndarray:
    """Load one model's predictions and return per-window MSE."""
    pred_path = result_dir / "pred.npy"
    true_path = result_dir / "true.npy"
    if not pred_path.exists() or not true_path.exists():
        raise FileNotFoundError(f"{result_dir} must contain pred.npy and true.npy")

    pred = np.load(pred_path)
    true = np.load(true_path)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch in {result_dir}: pred={pred.shape}, true={true.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected pred/true shape [windows, horizon, channels], got {pred.shape}")

    err = (pred - true) ** 2
    if target_channel is not None:
        channel = target_channel
        if channel < 0:
            channel = pred.shape[-1] + channel
        if channel < 0 or channel >= pred.shape[-1]:
            raise ValueError(f"target_channel={target_channel} is outside [0, {pred.shape[-1] - 1}]")
        err = err[:, :, channel]
    return err.reshape(err.shape[0], -1).mean(axis=1)


def collect_losses(
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    models: Iterable[str],
    target_channel: int | None,
) -> tuple[list[str], np.ndarray, dict[str, str]]:
    """Return model names, [windows, models] loss matrix, and source dirs."""
    model_names: list[str] = []
    columns: list[np.ndarray] = []
    source_dirs: dict[str, str] = {}
    expected_windows: int | None = None

    for model in models:
        result_dir = resolve_result_dir(results_root, dataset, lookback, horizon, model)
        losses = load_window_mse(result_dir, target_channel)
        if expected_windows is None:
            expected_windows = losses.shape[0]
        elif losses.shape[0] != expected_windows:
            raise ValueError(
                f"Window count mismatch for {model}: got {losses.shape[0]}, expected {expected_windows}"
            )
        model_names.append(model)
        columns.append(losses)
        source_dirs[model] = str(result_dir)

    if not columns:
        raise ValueError("At least one model is required")
    return model_names, np.stack(columns, axis=1), source_dirs


def build_trajectory(model_names: list[str], losses: np.ndarray) -> list[dict[str, object]]:
    """Build one row per window with oracle choice and margin diagnostics."""
    best_idx = losses.argmin(axis=1)
    order = np.argsort(losses, axis=1)
    rows: list[dict[str, object]] = []
    best_single_idx = int(losses.mean(axis=0).argmin())

    for i in range(losses.shape[0]):
        first = int(order[i, 0])
        second = int(order[i, 1]) if losses.shape[1] > 1 else first
        rows.append(
            {
                "window_index": i,
                "best_model": model_names[first],
                "best_loss": float(losses[i, first]),
                "second_model": model_names[second],
                "second_loss": float(losses[i, second]),
                "margin_vs_second": float(losses[i, second] - losses[i, first]),
                "best_single_model": model_names[best_single_idx],
                "best_single_loss": float(losses[i, best_single_idx]),
                "oracle_gain_vs_best_single": float(losses[i, best_single_idx] - losses[i, first]),
            }
        )
    assert len(best_idx) == len(rows)
    return rows


def summarize(
    cfg: StudyConfig,
    model_names: list[str],
    losses: np.ndarray,
    source_dirs: dict[str, str],
) -> dict[str, object]:
    """Compute the M1 headline statistics from a window-loss matrix."""
    aggregate = losses.mean(axis=0)
    best_single_idx = int(aggregate.argmin())
    best_idx = losses.argmin(axis=1)
    oracle_mse = float(losses[np.arange(losses.shape[0]), best_idx].mean())
    best_single_mse = float(aggregate[best_single_idx])
    gap_abs = best_single_mse - oracle_mse
    switch_count = int(np.count_nonzero(best_idx[1:] != best_idx[:-1])) if losses.shape[0] > 1 else 0
    win_counts = {model: int(np.count_nonzero(best_idx == i)) for i, model in enumerate(model_names)}

    margins = []
    if losses.shape[1] > 1:
        sorted_losses = np.sort(losses, axis=1)
        margins = (sorted_losses[:, 1] - sorted_losses[:, 0]).tolist()

    return {
        "config": asdict(cfg),
        "source_dirs": source_dirs,
        "num_windows": int(losses.shape[0]),
        "num_models": int(losses.shape[1]),
        "aggregate_model_mse": {model: float(aggregate[i]) for i, model in enumerate(model_names)},
        "best_single_model": model_names[best_single_idx],
        "best_single_mse": best_single_mse,
        "oracle_mse": oracle_mse,
        "oracle_gap_abs": float(gap_abs),
        "oracle_gap_rel": float(gap_abs / best_single_mse) if best_single_mse > 0 else math.nan,
        "switch_count": switch_count,
        "switch_rate": float(switch_count / (losses.shape[0] - 1)) if losses.shape[0] > 1 else 0.0,
        "win_counts": win_counts,
        "win_fractions": {model: win_counts[model] / losses.shape[0] for model in model_names},
        "mean_margin_vs_second": float(np.mean(margins)) if margins else 0.0,
        "median_margin_vs_second": float(np.median(margins)) if margins else 0.0,
    }


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


def write_plot(path: Path, model_names: list[str], losses: np.ndarray) -> None:
    """Write a compact best-architecture trajectory plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best_idx = losses.argmin(axis=1)
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
    axes[1].set_ylabel("Rolling MSE")
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


def run_study(cfg: StudyConfig) -> dict[str, object]:
    results_root = Path(cfg.results_root)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names, losses, source_dirs = collect_losses(
        results_root=results_root,
        dataset=cfg.dataset,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        models=cfg.models,
        target_channel=cfg.target_channel,
    )
    trajectory = build_trajectory(model_names, losses)
    summary = summarize(cfg, model_names, losses, source_dirs)

    write_window_losses(output_dir / "window_losses.csv", model_names, losses)
    write_trajectory(output_dir / "best_architecture_trajectory.csv", trajectory)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_plot(output_dir / "best_architecture_trajectory.png", model_names, losses)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Oracle Drift Study artifacts from TSLib predictions.")
    parser.add_argument("--results-root", default="external/TSLib/results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--target-channel",
        default="-1",
        help="Channel index to score for MISO target; use 'all' to average all channels.",
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
    )
    summary = run_study(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
