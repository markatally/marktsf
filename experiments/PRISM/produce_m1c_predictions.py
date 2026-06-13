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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("RidgeCov", "TargetRidge", "Trend", "Seasonal", "EWM")


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


def load_scaled_panel(spec: DatasetSpec, seq_len: int) -> tuple[np.ndarray, tuple[list[int], list[int]], list[str]]:
    df = pd.read_csv(spec.path)
    cols = list(df.columns)
    if "date" in cols:
        cols.remove("date")
    if spec.target not in cols:
        raise ValueError(f"{spec.path} has no target column {spec.target!r}")
    cols.remove(spec.target)
    ordered = cols + [spec.target]
    raw = df[ordered].astype("float64").to_numpy()

    border1s, border2s = _split_borders(len(raw), spec.kind, seq_len)
    train = raw[border1s[0] : border2s[0]]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((raw - mean) / std).astype("float32"), (border1s, border2s), ordered


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
) -> None:
    suffix = f"ftMS_sl{seq_len}_pl{pred_len}_m1c"
    out = results_root / f"long_term_forecast_{dataset_tag}_{seq_len}_{pred_len}_{model}_{dataset_tag}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred.npy", pred)
    np.save(out / "true.npy", true)
    np.save(out / "context.npy", context)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def produce_one(args: argparse.Namespace, name: str) -> None:
    spec = SPECS[name]
    panel, (border1s, border2s), columns = load_scaled_panel(spec, args.lookback)
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
    meta = {
        "dataset": name,
        "dataset_tag": spec.tag,
        "source_csv": str(spec.path),
        "columns": columns,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "split_kind": spec.kind,
        "train_windows_used": int(len(train_x)),
        "test_windows": int(len(test_x)),
        "season_lag": spec.season_lag,
        "target_channel": -1,
    }
    for model, pred in preds.items():
        save_result(args.results_root, spec.tag, model, args.lookback, args.horizon, pred, test_y, test_x, {**meta, "model": model})
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        produce_one(args, name)


if __name__ == "__main__":
    main()
