"""Shared utilities for the MaskShift milestone scripts.

The implementation is intentionally lightweight: it tests the MaskShift thesis
with controlled mask generators and fast tabular forecasters before any costly
deep-backbone sweep.  All random choices are seeded and all masks are generated
from pre-origin input windows only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).parents[2]
EXP_DIR = Path(__file__).parent

DEFAULT_DATASETS = {
    "Weather": ROOT / "input" / "Weather" / "weather.csv",
    "Electricity": ROOT / "input" / "Electricity" / "electricity.csv",
    "Traffic": ROOT / "input" / "Traffic" / "traffic.csv",
    "AirConvection": ROOT / "input" / "AirConvection" / "AirConvection.csv",
}

MECHANISMS = ["mcar", "block", "value_high", "volatility", "blackout", "retirement"]
OPERATIONAL_MECHANISMS = ["value_high", "volatility", "blackout", "retirement"]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    max_rows: int = 7000
    max_channels: int = 24


@dataclass(frozen=True)
class ExperimentConfig:
    lookback: int = 48
    horizon: int = 12
    stride: int = 8
    target_rate: float = 0.35
    seed: int = 2026
    max_train_samples: int = 900
    max_test_samples: int = 420


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_numeric_frame(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path)
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError(f"{spec.path} has no numeric columns")

    nunique = numeric.nunique(dropna=True)
    usable_cols = [c for c in numeric.columns if nunique[c] > 8]
    numeric = numeric[usable_cols]
    if numeric.shape[1] > spec.max_channels:
        variances = numeric.var(skipna=True).sort_values(ascending=False)
        numeric = numeric[variances.index[: spec.max_channels]]
    if numeric.shape[0] > spec.max_rows:
        numeric = numeric.iloc[: spec.max_rows]

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill()
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    return numeric.astype(float)


def load_dataset(name: str, max_rows: int = 7000, max_channels: int = 24) -> tuple[np.ndarray, dict]:
    if name not in DEFAULT_DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choices={sorted(DEFAULT_DATASETS)}")
    spec = DatasetSpec(name=name, path=DEFAULT_DATASETS[name], max_rows=max_rows, max_channels=max_channels)
    raw = pd.read_csv(spec.path)
    natural_missing = raw.select_dtypes(include=[np.number]).isna().to_numpy().mean()
    frame = load_numeric_frame(spec)
    arr = frame.to_numpy(dtype=np.float64)
    meta = {
        "dataset": name,
        "path": str(spec.path),
        "rows": int(arr.shape[0]),
        "channels": int(arr.shape[1]),
        "natural_missing_rate_before_interpolation": float(natural_missing),
        "columns": list(frame.columns),
    }
    return arr, meta


def train_test_normalize(values: np.ndarray, split_idx: int) -> tuple[np.ndarray, dict]:
    train = values[:split_idx]
    mean = np.nanmean(train, axis=0)
    std = np.nanstd(train, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    norm = (values - mean) / std
    return norm, {"mean": mean.tolist(), "std": std.tolist()}


def sample_origins(
    n_time: int,
    lookback: int,
    horizon: int,
    start: int,
    stop: int,
    stride: int,
    max_samples: int,
) -> np.ndarray:
    lo = max(start, lookback)
    hi = min(stop, n_time - horizon)
    origins = np.arange(lo, hi, stride, dtype=int)
    if len(origins) > max_samples:
        idx = np.linspace(0, len(origins) - 1, max_samples).round().astype(int)
        origins = origins[idx]
    return origins


def _make_block_mask(shape: tuple[int, int], rate: float, rng: np.random.Generator, block_len: int = 8) -> np.ndarray:
    n_time, n_chan = shape
    mask = np.zeros(shape, dtype=bool)
    target_cells = int(rate * n_time * n_chan)
    attempts = 0
    while mask.sum() < target_cells and attempts < target_cells * 4 + 20:
        c = int(rng.integers(0, n_chan))
        start = int(rng.integers(0, max(1, n_time - block_len)))
        length = int(max(1, rng.poisson(block_len)))
        mask[start : min(n_time, start + length), c] = True
        attempts += 1
    return _match_rate(mask, rate, rng)


def _make_time_block_mask(shape: tuple[int, int], rate: float, rng: np.random.Generator, block_len: int = 6) -> np.ndarray:
    n_time, n_chan = shape
    mask = np.zeros(shape, dtype=bool)
    target_rows = max(1, int(rate * n_time))
    attempts = 0
    while mask.any(axis=1).sum() < target_rows and attempts < target_rows * 5 + 20:
        start = int(rng.integers(0, max(1, n_time - block_len)))
        length = int(max(1, rng.poisson(block_len)))
        mask[start : min(n_time, start + length), :] = True
        attempts += 1
    return _match_rate(mask, rate, rng)


def _match_rate(mask: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    n = out.size
    target = int(round(rate * n))
    flat = out.reshape(-1)
    current = int(flat.sum())
    if current > target:
        true_idx = np.flatnonzero(flat)
        drop = rng.choice(true_idx, size=current - target, replace=False)
        flat[drop] = False
    elif current < target:
        false_idx = np.flatnonzero(~flat)
        add = rng.choice(false_idx, size=target - current, replace=False)
        flat[add] = True
    return flat.reshape(out.shape)


def generate_mask(
    values: np.ndarray,
    mechanism: str,
    rate: float,
    seed: int,
    split_idx: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = values.shape
    if mechanism == "mcar":
        return rng.random(shape) < rate
    if mechanism == "block":
        return _make_block_mask(shape, rate, rng)
    if mechanism == "blackout":
        return _make_time_block_mask(shape, rate, rng)
    if mechanism == "value_high":
        score = np.nan_to_num(values, nan=0.0)
        score = (score - np.nanmedian(score, axis=0)) / (np.nanstd(score, axis=0) + 1e-6)
        probs = 1.0 / (1.0 + np.exp(-2.2 * (score - 0.75)))
        probs = probs / (probs.mean() + 1e-9) * rate
        probs = np.clip(probs, 0.0, 0.95)
        return _match_rate(rng.random(shape) < probs, rate, rng)
    if mechanism == "volatility":
        diffs = np.vstack([np.zeros((1, shape[1])), np.abs(np.diff(values, axis=0))])
        score = diffs / (np.nanmedian(diffs, axis=0) + np.nanstd(diffs, axis=0) + 1e-6)
        probs = 1.0 / (1.0 + np.exp(-2.0 * (score - 1.0)))
        probs = probs / (probs.mean() + 1e-9) * rate
        probs = np.clip(probs, 0.0, 0.95)
        return _match_rate(rng.random(shape) < probs, rate, rng)
    if mechanism == "retirement":
        n_time, n_chan = shape
        mask = np.zeros(shape, dtype=bool)
        # Drop a small number of sensors/channels near the end of the sequence.
        n_channels = max(1, min(n_chan, int(math.ceil(rate * n_chan * 2))))
        channels = rng.choice(np.arange(n_chan), size=n_channels, replace=False)
        for c in channels:
            start = int(rng.integers(max(1, n_time // 3), max(2, int(0.85 * n_time))))
            mask[start:, c] = True
        if split_idx is not None:
            # Keep training distribution legally available; deployment retirement
            # starts after the split for a subset of channels when possible.
            for c in channels:
                start = int(rng.integers(split_idx, max(split_idx + 1, n_time - 2)))
                mask[:, c] = False
                mask[start:, c] = True
        return _match_rate(mask, rate, rng)
    raise ValueError(f"unknown mechanism {mechanism}")


def mask_stats(mask: np.ndarray) -> dict:
    n_time, n_chan = mask.shape
    rates = mask.mean(axis=0)
    run_lengths = []
    for c in range(n_chan):
        run = 0
        for t in range(n_time):
            if mask[t, c]:
                run += 1
            elif run:
                run_lengths.append(run)
                run = 0
        if run:
            run_lengths.append(run)
    return {
        "missing_rate": float(mask.mean()),
        "channel_coverage_mean": float((~mask).mean(axis=0).mean()),
        "channel_coverage_min": float((~mask).mean(axis=0).min()),
        "time_outage_density_mean": float(mask.mean(axis=1).mean()),
        "time_outage_density_p95": float(np.quantile(mask.mean(axis=1), 0.95)),
        "max_gap": int(max(run_lengths) if run_lengths else 0),
        "mean_gap": float(np.mean(run_lengths) if run_lengths else 0.0),
        "channel_rate_std": float(np.std(rates)),
    }


def _forward_fill_window(window: np.ndarray, obs: np.ndarray) -> np.ndarray:
    out = window.copy()
    for c in range(out.shape[1]):
        last = 0.0
        for t in range(out.shape[0]):
            if obs[t, c]:
                last = out[t, c]
            else:
                out[t, c] = last
    return out


def topology_features(mask_window: np.ndarray) -> np.ndarray:
    lookback, n_chan = mask_window.shape
    obs = ~mask_window
    coverage = obs.mean(axis=0)
    final_age = np.zeros(n_chan, dtype=float)
    max_gap = np.zeros(n_chan, dtype=float)
    for c in range(n_chan):
        run = 0
        max_run = 0
        for t in range(lookback):
            if mask_window[t, c]:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        final_age[c] = run / lookback
        max_gap[c] = max_run / lookback
    time_density = mask_window.mean(axis=1)
    return np.concatenate(
        [
            np.array(
                [
                    mask_window.mean(),
                    coverage.mean(),
                    coverage.min(),
                    coverage.std(),
                    time_density.mean(),
                    time_density.std(),
                    time_density.max(),
                    final_age.mean(),
                    final_age.max(),
                    max_gap.mean(),
                    max_gap.max(),
                ],
                dtype=float,
            ),
            coverage,
            final_age,
            max_gap,
        ]
    )


def one_hot(mechanism: str, mechanisms: Iterable[str] = MECHANISMS) -> np.ndarray:
    mechs = list(mechanisms)
    out = np.zeros(len(mechs), dtype=float)
    out[mechs.index(mechanism)] = 1.0
    return out


def build_supervised(
    values: np.ndarray,
    mask: np.ndarray,
    origins: np.ndarray,
    lookback: int,
    horizon: int,
    target_channel: int = 0,
    mechanism: str = "mcar",
    include_mask: bool = False,
    include_topology: bool = False,
    include_mechanism: bool = False,
    fill: str = "zero",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    topo_rows = []
    target = []
    for origin in origins:
        x = values[origin - lookback : origin].copy()
        mw = mask[origin - lookback : origin].copy()
        obs = ~mw
        if fill == "zero":
            x_filled = np.where(obs, x, 0.0)
        elif fill == "ffill":
            x_filled = _forward_fill_window(np.where(obs, x, 0.0), obs)
        else:
            raise ValueError(fill)
        parts = [x_filled.reshape(-1)]
        if include_mask:
            parts.append(mw.astype(float).reshape(-1))
        topo = topology_features(mw)
        if include_topology:
            parts.append(topo)
        if include_mechanism:
            mech_hot = one_hot(mechanism)
            parts.append(mech_hot)
        rows.append(np.concatenate(parts))
        topo_rows.append(np.concatenate([topo, one_hot(mechanism)]))
        target.append(values[origin + horizon - 1, target_channel])
    return np.vstack(rows), np.asarray(target, dtype=float), np.vstack(topo_rows)


def make_model(alpha: float = 2.0):
    return make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=alpha, random_state=0))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(np.abs(y_true - y_pred) / np.maximum(denom, 1e-6))
    return {"mse": float(mse), "mae": float(mae), "smape": float(smape)}


def train_predict_ridge(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_origins: np.ndarray,
    test_origins: np.ndarray,
    cfg: ExperimentConfig,
    train_mechanism: str,
    test_mechanism: str,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    settings = {
        "zero": {"fill": "zero", "include_mask": False, "include_topology": False, "include_mechanism": False},
        "ffill": {"fill": "ffill", "include_mask": False, "include_topology": False, "include_mechanism": False},
        "mask": {"fill": "zero", "include_mask": True, "include_topology": False, "include_mechanism": False},
        "topology": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": False},
        "typed": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": True},
    }
    if variant not in settings:
        raise ValueError(f"unknown variant {variant}")
    kw = settings[variant]
    x_train, y_train, _ = build_supervised(
        train_values,
        train_mask,
        train_origins,
        cfg.lookback,
        cfg.horizon,
        mechanism=train_mechanism,
        **kw,
    )
    x_test, y_test, _ = build_supervised(
        test_values,
        test_mask,
        test_origins,
        cfg.lookback,
        cfg.horizon,
        mechanism=test_mechanism,
        **kw,
    )
    model = make_model()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return y_test, pred, evaluate_predictions(y_test, pred)


def make_dataset_splits(dataset: str, cfg: ExperimentConfig) -> dict:
    values_raw, meta = load_dataset(dataset)
    split_idx = int(values_raw.shape[0] * 0.7)
    values, norm_meta = train_test_normalize(values_raw, split_idx)
    train_origins = sample_origins(
        len(values),
        cfg.lookback,
        cfg.horizon,
        start=cfg.lookback,
        stop=split_idx,
        stride=cfg.stride,
        max_samples=cfg.max_train_samples,
    )
    test_origins = sample_origins(
        len(values),
        cfg.lookback,
        cfg.horizon,
        start=split_idx,
        stop=len(values) - cfg.horizon,
        stride=cfg.stride,
        max_samples=cfg.max_test_samples,
    )
    return {
        "values": values,
        "split_idx": split_idx,
        "train_origins": train_origins,
        "test_origins": test_origins,
        "meta": {**meta, "split_idx": split_idx, "normalization": norm_meta},
    }


def train_mixed_variant(
    values: np.ndarray,
    split_idx: int,
    train_origins: np.ndarray,
    test_origins: np.ndarray,
    cfg: ExperimentConfig,
    mechanisms: list[str],
    variant: str,
    seed: int,
    rate: float | None = None,
    randomize_labels: bool = False,
) -> list[dict]:
    rate = cfg.target_rate if rate is None else rate
    x_parts = []
    y_parts = []
    rng = np.random.default_rng(seed + 917)
    for i, mech in enumerate(mechanisms):
        mask = generate_mask(values, mech, rate, seed + i * 31, split_idx=split_idx)
        label = mech
        if randomize_labels:
            label = str(rng.choice(mechanisms))
        settings = {
            "topology": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": False},
            "typed": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": True},
        }[variant]
        x, y, _ = build_supervised(
            values,
            mask,
            train_origins,
            cfg.lookback,
            cfg.horizon,
            mechanism=label,
            **settings,
        )
        x_parts.append(x)
        y_parts.append(y)
    model = make_model()
    model.fit(np.vstack(x_parts), np.concatenate(y_parts))

    rows = []
    for i, mech in enumerate(mechanisms):
        mask = generate_mask(values, mech, rate, seed + 1000 + i * 37, split_idx=split_idx)
        settings = {
            "topology": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": False},
            "typed": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": True},
        }[variant]
        x_test, y_test, _ = build_supervised(
            values,
            mask,
            test_origins,
            cfg.lookback,
            cfg.horizon,
            mechanism=mech,
            **settings,
        )
        pred = model.predict(x_test)
        metrics = evaluate_predictions(y_test, pred)
        rows.append({"mechanism": mech, "variant": variant, **metrics})
    return rows


def rank_models(rows: list[dict], metric: str = "mse") -> dict[str, list[str]]:
    by_mech: dict[str, list[dict]] = {}
    for row in rows:
        by_mech.setdefault(row["test_mechanism"], []).append(row)
    ranks = {}
    for mech, mech_rows in by_mech.items():
        ordered = sorted(mech_rows, key=lambda r: r[metric])
        ranks[mech] = [r["variant"] for r in ordered]
    return ranks


def kendall_tau_between(a: list[str], b: list[str]) -> float:
    common = [x for x in a if x in b]
    if len(common) < 2:
        return float("nan")
    ra = [a.index(x) for x in common]
    rb = [b.index(x) for x in common]
    tau, _ = stats.kendalltau(ra, rb)
    return float(tau)


def paired_dm_like(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    loss_a = (y_true - pred_a) ** 2
    loss_b = (y_true - pred_b) ** 2
    diff = loss_a - loss_b
    t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0)
    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = float("nan"), 1.0
    return {
        "mean_loss_delta_a_minus_b": float(np.mean(diff)),
        "paired_t_stat": float(t_stat),
        "paired_t_p": float(p_val),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        denom_rank = n - rank + 1
        val = p[idx] * n / denom_rank
        running = min(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def degradation_auc(rows: list[dict], mechanisms: list[str] = OPERATIONAL_MECHANISMS) -> float:
    vals = [r["mse"] for r in rows if r["mechanism"] in mechanisms]
    return float(np.mean(vals)) if vals else float("nan")


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"
