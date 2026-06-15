"""Lightweight M1c breadth prediction producer.

This script creates frozen pred/true artifacts for the PRISM M1c breadth
screen.  It is intentionally cheap: the pool contains deterministic or
closed-form predictors with different inductive biases, so the oracle drift
pipeline can ask where switching structure exists before spending GPU time on
full TSLib retraining.

Artifacts follow the TSLib result naming convention consumed by
``experiments.PRISM.oracle_drift`` and additionally include ``context.npy``:
the scaled lookback panel for each test window, used by the M1c descriptor
probe.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_MODELS = ("RidgeCov", "TargetRidge", "Trend", "Seasonal", "EWM")
EXPANDED_MODELS = (
    *BASE_MODELS,
    "EWMFast",
    "EWMSlow",
    "SeasonalOffset",
    "SeasonalDrift",
    "DampedTrend",
    "MeanRevert",
    "MeanRevertSlow",
    "SeasonalEWM",
    "SeasonalTrend",
    "EWMTrend",
)
MODELS = BASE_MODELS


@dataclass(frozen=True)
class DatasetSpec:
    tag: str
    path: Path
    kind: str
    freq: str
    season_lag: int
    target: str = "OT"


SPECS = {
    "ETTh1": DatasetSpec("M1C_ETTh1", Path("input/ETT/ETTh1.csv"), "ett_hour", "h", 24),
    "ETTh2": DatasetSpec("M1C_ETTh2", Path("input/ETT/ETTh2.csv"), "ett_hour", "h", 24),
    "ETTm1": DatasetSpec("M1C_ETTm1", Path("input/ETT/ETTm1.csv"), "ett_minute", "t", 96),
    "ETTm2": DatasetSpec("M1C_ETTm2", Path("input/ETT/ETTm2.csv"), "ett_minute", "t", 96),
    "Weather": DatasetSpec("M1C_Weather", Path("input/Weather/weather.csv"), "custom", "t", 96),
    "Exchange": DatasetSpec("M1C_Exchange", Path("input/Exchange/exchange_rate.csv"), "custom", "d", 7),
    "Electricity": DatasetSpec("M1C_Electricity", Path("input/Electricity/electricity.csv"), "custom", "h", 24),
    "Traffic": DatasetSpec("M1C_Traffic", Path("input/Traffic/traffic.csv"), "custom", "h", 24),
    "Solar": DatasetSpec("M1C_Solar", Path("input/Solar/Solar_TFB.csv"), "tfb_long", "t", 144, "__auto_last__"),
    "METRLA": DatasetSpec("M1C_METRLA", Path("input/METR-LA/METR-LA_TFB.csv"), "tfb_long", "t", 288, "__auto_last__"),
    "PEMSBAY": DatasetSpec("M1C_PEMSBAY", Path("input/PEMS-BAY/PEMS-BAY_TFB.csv"), "tfb_long", "t", 288, "__auto_last__"),
    "PEMS04": DatasetSpec("M1C_PEMS04", Path("input/PEMS04/PEMS04_TFB.csv"), "tfb_long", "t", 288, "__auto_last__"),
    "PEMS08": DatasetSpec("M1C_PEMS08", Path("input/PEMS08/PEMS08_TFB.csv"), "tfb_long", "t", 288, "__auto_last__"),
    "Wind": DatasetSpec("M1C_Wind", Path("input/Wind/Wind.csv"), "tfb_long", "t", 96, "__auto_last__"),
    "AQShunyi": DatasetSpec("M1C_AQShunyi", Path("input/AQShunyi/AQShunyi.csv"), "tfb_long", "h", 24, "__auto_last__"),
    "AQWan": DatasetSpec("M1C_AQWan", Path("input/AQShunyi/AQWan.csv"), "tfb_long", "h", 24, "__auto_last__"),
}


def _split_borders(n: int, kind: str, seq_len: int) -> tuple[list[int], list[int]]:
    if kind == "ett_hour":
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif kind == "ett_minute":
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_val = n - num_train - num_test
        border1s = [0, num_train - seq_len, n - num_test - seq_len]
        border2s = [num_train, num_train + num_val, n]
    if border2s[-1] > n:
        raise ValueError(f"Dataset has {n} rows, but split requires {border2s[-1]}")
    return border1s, border2s


def select_covariates_by_train_corr(
    raw: np.ndarray,
    ordered: list[str],
    border1s: list[int],
    border2s: list[int],
    max_covariates: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Keep top train-correlated covariates plus the target.

    The selector uses only the train split, so it is safe for causal breadth
    experiments.  It is disabled by default and exists to keep high-dimensional
    Traffic/Electricity screens lightweight.
    """
    cov_count = raw.shape[1] - 1
    if max_covariates <= 0 or cov_count <= max_covariates:
        return raw, ordered, []
    train = raw[border1s[0] : border2s[0]]
    cov = train[:, :-1]
    target = train[:, -1]
    cov_centered = cov - cov.mean(axis=0, keepdims=True)
    target_centered = target - target.mean()
    denom = np.sqrt((cov_centered**2).sum(axis=0) * float((target_centered**2).sum())) + 1e-12
    corr = np.abs((cov_centered * target_centered[:, None]).sum(axis=0) / denom)
    top = np.argsort(corr)[::-1][:max_covariates]
    keep = list(sorted(int(i) for i in top)) + [raw.shape[1] - 1]
    selected_names = [ordered[i] for i in keep]
    dropped = [name for i, name in enumerate(ordered[:-1]) if i not in set(top)]
    return raw[:, keep], selected_names, dropped


def natural_key(value: object) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def load_raw_frame(spec: DatasetSpec) -> tuple[pd.DataFrame, str]:
    if spec.kind == "tfb_long":
        long = pd.read_csv(spec.path, usecols=["date", "data", "cols"])
        time_order = pd.to_datetime(long["date"], errors="coerce", utc=True)
        if time_order.isna().any():
            bad = int(time_order.isna().sum())
            raise ValueError(f"{spec.path} has {bad} unparsable timestamps")
        long = long.assign(__time_order=time_order)
        wide = long.pivot_table(index="__time_order", columns="cols", values="data", aggfunc="mean")
        wide = wide.sort_index()
        wide = wide.reindex(sorted(wide.columns, key=natural_key), axis=1)
        wide = wide.reset_index()
        return wide, "__time_order"
    df = pd.read_csv(spec.path)
    date_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else ""
    return df, date_col


def load_scaled_panel(
    spec: DatasetSpec,
    seq_len: int,
    *,
    max_covariates: int = 0,
) -> tuple[np.ndarray, tuple[list[int], list[int]], list[str], list[str]]:
    df, date_col = load_raw_frame(spec)
    cols = list(df.columns)
    if date_col:
        cols.remove(date_col)
    target = cols[-1] if spec.target == "__auto_last__" else spec.target
    if target not in cols:
        raise ValueError(f"{spec.path} has no target column {spec.target!r}")
    cols.remove(target)
    ordered = cols + [target]
    raw = df[ordered].astype("float64").to_numpy()

    border1s, border2s = _split_borders(len(raw), spec.kind, seq_len)
    raw, ordered, dropped = select_covariates_by_train_corr(raw, ordered, border1s, border2s, max_covariates)
    train = raw[border1s[0] : border2s[0]]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((raw - mean) / std).astype("float32"), (border1s, border2s), ordered, dropped


def make_windows(data: np.ndarray, start: int, end: int, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    n = end - start - seq_len - pred_len + 1
    if n <= 0:
        raise ValueError("Not enough rows for requested seq_len/pred_len")
    x = np.empty((n, seq_len, data.shape[1]), dtype=np.float32)
    y = np.empty((n, pred_len, 1), dtype=np.float32)
    target = data[:, -1]
    for i in range(n):
        s = start + i
        x[i] = data[s : s + seq_len]
        y[i, :, 0] = target[s + seq_len : s + seq_len + pred_len]
    return x, y


def feature_matrix(x: np.ndarray, target_only: bool) -> np.ndarray:
    target = x[:, :, -1]
    t = np.arange(x.shape[1], dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    slope = ((target - target.mean(axis=1, keepdims=True)) * t[None, :]).mean(axis=1)
    pieces = [
        target[:, -1:],
        target.mean(axis=1, keepdims=True),
        target.std(axis=1, keepdims=True),
        slope[:, None],
    ]
    if not target_only:
        pieces.extend([x[:, -1, :], x.mean(axis=1), x.std(axis=1)])
    return np.concatenate(pieces, axis=1).astype(np.float64)


def ridge_direct(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, *, target_only: bool, alpha: float) -> np.ndarray:
    xtr = feature_matrix(train_x, target_only=target_only)
    xte = feature_matrix(test_x, target_only=target_only)
    xtr = np.concatenate([xtr, np.ones((xtr.shape[0], 1))], axis=1)
    xte = np.concatenate([xte, np.ones((xte.shape[0], 1))], axis=1)
    ytr = train_y[:, :, 0].astype(np.float64)
    eye = np.eye(xtr.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    coef = np.linalg.solve(xtr.T @ xtr + alpha * eye, xtr.T @ ytr)
    return (xte @ coef).astype(np.float32)[:, :, None]


def trend_predict(x: np.ndarray, pred_len: int) -> np.ndarray:
    target = x[:, :, -1]
    t = np.arange(x.shape[1], dtype=np.float32)
    t = t - t.mean()
    denom = float((t * t).sum()) or 1.0
    slope = ((target - target.mean(axis=1, keepdims=True)) * t[None, :]).sum(axis=1) / denom
    steps = np.arange(1, pred_len + 1, dtype=np.float32)
    pred = target[:, -1, None] + slope[:, None] * steps[None, :]
    return pred[:, :, None].astype(np.float32)


def seasonal_predict(x: np.ndarray, pred_len: int, season_lag: int) -> np.ndarray:
    target = x[:, :, -1]
    lag = min(season_lag, x.shape[1])
    idx = (np.arange(pred_len) % lag) - lag
    return target[:, idx][:, :, None].astype(np.float32)


def ewm_predict(x: np.ndarray, pred_len: int, half_life: float = 24.0) -> np.ndarray:
    target = x[:, :, -1]
    alpha = 1.0 - np.exp(-np.log(2.0) / half_life)
    state = target[:, 0].copy()
    for j in range(1, target.shape[1]):
        state = alpha * target[:, j] + (1.0 - alpha) * state
    return np.repeat(state[:, None, None], pred_len, axis=1).astype(np.float32)


def damped_trend_predict(x: np.ndarray, pred_len: int, damping: float = 0.98) -> np.ndarray:
    target = x[:, :, -1]
    t = np.arange(x.shape[1], dtype=np.float32)
    t = t - t.mean()
    denom = float((t * t).sum()) or 1.0
    slope = ((target - target.mean(axis=1, keepdims=True)) * t[None, :]).sum(axis=1) / denom
    steps = np.arange(1, pred_len + 1, dtype=np.float32)
    damped_steps = (1.0 - np.power(damping, steps)) / (1.0 - damping)
    pred = target[:, -1, None] + slope[:, None] * damped_steps[None, :]
    return pred[:, :, None].astype(np.float32)


def mean_revert_predict(x: np.ndarray, pred_len: int, half_life: float = 24.0) -> np.ndarray:
    target = x[:, :, -1]
    mean = target.mean(axis=1)
    last = target[:, -1]
    decay = np.exp(-np.log(2.0) * np.arange(1, pred_len + 1, dtype=np.float32) / half_life)
    pred = mean[:, None] + (last - mean)[:, None] * decay[None, :]
    return pred[:, :, None].astype(np.float32)


def seasonal_offset_predict(x: np.ndarray, pred_len: int, season_lag: int) -> np.ndarray:
    target = x[:, :, -1]
    lag = min(season_lag, x.shape[1])
    idx = (np.arange(pred_len) % lag) - lag
    base = target[:, idx]
    offset = (target[:, -1] - target[:, -lag])[:, None]
    return (base + offset).astype(np.float32)[:, :, None]


def seasonal_drift_predict(x: np.ndarray, pred_len: int, season_lag: int) -> np.ndarray:
    target = x[:, :, -1]
    lag = min(season_lag, x.shape[1])
    idx = (np.arange(pred_len) % lag) - lag
    base = target[:, idx]
    if x.shape[1] >= 2 * lag:
        recent = target[:, -lag:]
        previous = target[:, -2 * lag : -lag]
        drift = (recent - previous).mean(axis=1, keepdims=True)
    else:
        drift = 0.5 * (target[:, -1] - target[:, -lag])[:, None]
    return (base + drift).astype(np.float32)[:, :, None]


def save_result(
    results_root: Path,
    dataset_tag: str,
    model: str,
    seq_len: int,
    pred_len: int,
    pred: np.ndarray,
    true: np.ndarray,
    context: np.ndarray,
    metadata: dict[str, object],
    context_source: Path | None = None,
) -> None:
    suffix = f"ftMS_sl{seq_len}_pl{pred_len}_m1c"
    out = results_root / f"long_term_forecast_{dataset_tag}_{seq_len}_{pred_len}_{model}_{dataset_tag}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred.npy", pred)
    np.save(out / "true.npy", true)
    context_path = out / "context.npy"
    if context_source is None:
        np.save(context_path, context)
    else:
        if context_path.exists() or context_path.is_symlink():
            context_path.unlink()
        context_path.symlink_to(Path("..") / context_source.relative_to(results_root))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def produce_one(args: argparse.Namespace, name: str) -> None:
    spec = SPECS[name]
    panel, (border1s, border2s), columns, dropped_columns = load_scaled_panel(
        spec,
        args.lookback,
        max_covariates=args.max_covariates,
    )
    train_x, train_y = make_windows(panel, border1s[0], border2s[0], args.lookback, args.horizon)
    test_x, test_y = make_windows(panel, border1s[2], border2s[2], args.lookback, args.horizon)

    if args.max_train_windows and len(train_x) > args.max_train_windows:
        train_x = train_x[-args.max_train_windows :]
        train_y = train_y[-args.max_train_windows :]

    preds = {
        "RidgeCov": ridge_direct(train_x, train_y, test_x, target_only=False, alpha=args.ridge_alpha),
        "TargetRidge": ridge_direct(train_x, train_y, test_x, target_only=True, alpha=args.ridge_alpha),
        "Trend": trend_predict(test_x, args.horizon),
        "Seasonal": seasonal_predict(test_x, args.horizon, spec.season_lag),
        "EWM": ewm_predict(test_x, args.horizon, half_life=args.ewm_half_life),
    }
    if args.pool == "expanded":
        preds.update(
            {
                "EWMFast": ewm_predict(test_x, args.horizon, half_life=6.0),
                "EWMSlow": ewm_predict(test_x, args.horizon, half_life=96.0),
                "SeasonalOffset": seasonal_offset_predict(test_x, args.horizon, spec.season_lag),
                "SeasonalDrift": seasonal_drift_predict(test_x, args.horizon, spec.season_lag),
                "DampedTrend": damped_trend_predict(test_x, args.horizon, damping=0.98),
                "MeanRevert": mean_revert_predict(test_x, args.horizon, half_life=24.0),
                "MeanRevertSlow": mean_revert_predict(test_x, args.horizon, half_life=96.0),
            }
        )
        preds.update(
            {
                "SeasonalEWM": 0.5 * preds["Seasonal"] + 0.5 * preds["EWM"],
                "SeasonalTrend": 0.5 * preds["Seasonal"] + 0.5 * preds["Trend"],
                "EWMTrend": 0.5 * preds["EWM"] + 0.5 * preds["Trend"],
            }
        )
    meta = {
        "dataset": name,
        "dataset_tag": spec.tag,
        "pool": args.pool,
        "source_csv": str(spec.path),
        "columns": columns,
        "dropped_columns": dropped_columns,
        "max_covariates": args.max_covariates,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "split_kind": spec.kind,
        "train_windows_used": int(len(train_x)),
        "test_windows": int(len(test_x)),
        "season_lag": spec.season_lag,
        "target_channel": -1,
        "target_column": columns[-1],
        "shared_context": bool(args.shared_context),
    }
    context_source = None
    if args.shared_context:
        shared_dir = args.results_root / "_shared_context"
        shared_dir.mkdir(parents=True, exist_ok=True)
        context_source = shared_dir / f"{spec.tag}_L{args.lookback}_H{args.horizon}_context.npy"
        np.save(context_source, test_x)
    for model, pred in preds.items():
        save_result(
            args.results_root,
            spec.tag,
            model,
            args.lookback,
            args.horizon,
            pred,
            test_y,
            test_x,
            {**meta, "model": model},
            context_source=context_source,
        )
    print(f"{name}: wrote {len(preds)} models, test_windows={len(test_x)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Produce lightweight PRISM M1c breadth predictions.")
    p.add_argument("--datasets", nargs="+", default=list(SPECS), choices=list(SPECS))
    p.add_argument("--results-root", type=Path, default=Path("external/TSLib/results"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--max-train-windows", type=int, default=5000)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--ewm-half-life", type=float, default=24.0)
    p.add_argument("--pool", choices=["base", "expanded"], default="base")
    p.add_argument("--max-covariates", type=int, default=0, help="Keep top train-correlated covariates plus target; 0 keeps all.")
    p.add_argument("--shared-context", action="store_true", help="Store one context.npy per dataset/horizon and symlink model artifacts to it.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        produce_one(args, name)


if __name__ == "__main__":
    main()
