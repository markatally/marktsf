"""Causal development data. Arrow is decoded fully; only prefixes enter analysis."""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc


def stable_hash(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:16], 16)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Series:
    dataset: str
    item: str
    values: np.ndarray
    train_end: int
    source: str
    digest: str
    dates: np.ndarray | None = None


def boom_development(repo_root, max_tasks=48, max_variates=4,
                     development_fraction=0.7, train_fraction=0.7):
    root = Path(repo_root) / "input/BOOM"
    paths = sorted((p for p in root.iterdir() if p.is_dir()),
                   key=lambda p: stable_hash(p.name))
    # Reserve 80% of task identities before inspecting target values.
    paths = [p for p in paths if stable_hash(p.name) % 10 < 2][:max_tasks]
    records, manifest = [], []
    for path in paths:
        files = sorted(path.glob("*.arrow"))
        hashes = {str(f.relative_to(repo_root)): sha256(f) for f in files}
        rows = []
        for f in files:
            with f.open("rb") as stream:
                rows.extend(ipc.open_stream(stream).read_all().to_pylist())
        channels = []
        for row in rows:
            target = np.asarray(row["target"], dtype=np.float64)
            if target.ndim == 1:
                channels.append((str(row["item_id"]), target))
            elif target.ndim == 2:
                channels.extend((f'{row["item_id"]}:v{v}', target[v]) for v in range(target.shape[0]))
            else:
                raise ValueError(f"Unsupported BOOM target shape {target.shape}")
        selected = sorted(channels, key=lambda pair: stable_hash(pair[0]))[:max_variates]
        for item, target in selected:
            end = int(len(target) * development_fraction)
            values = target[:end].copy()
            train_end = int(end * train_fraction)
            records.append(Series(path.name, item, values,
                                  train_end, str(path), str(hashes)))
        manifest.append({"dataset": path.name, "total_variates": len(channels),
                         "selected_variates": len(selected), "hashes": hashes})
    return records, manifest


def price_development(repo_root, datasets=("SP500", "CSI500"), max_assets=24,
                      development_fraction=0.7, train_fraction=0.7):
    records, manifest = [], []
    for name in datasets:
        path = Path(repo_root) / "input" / name / f"{name}.csv"
        frame = pd.read_csv(path)
        end = int(len(frame) * development_fraction)
        prefix = frame.iloc[:end]
        columns = sorted(frame.columns[1:], key=stable_hash)[:max_assets]
        digest = sha256(path)
        for col in columns:
            records.append(Series(name, col, prefix[col].to_numpy(dtype=float),
                                  int(end * train_fraction), str(path), digest,
                                  prefix.iloc[:, 0].to_numpy()))
        manifest.append({"dataset": name, "sha256": digest,
                         "columns": columns, "development_rows": end,
                         "train_end": int(end * train_fraction),
                         "development_last_date": str(prefix.iloc[-1, 0])})
    return records, manifest


def state_ages(state):
    ages = np.ones(len(state), dtype=np.int64)
    for t in range(1, len(state)):
        ages[t] = ages[t-1] + 1 if state[t] == state[t-1] else 1
    return ages


def calibration_atom(y, calibration_length=96):
    """An empirical repeated-value atom; physical zero semantics are unknown."""
    prefix = np.asarray(y[:calibration_length])
    if prefix.ndim != 1 or len(prefix) < calibration_length or not np.isfinite(prefix).all():
        return None
    values, counts = np.unique(prefix, return_counts=True)
    best = int(counts.argmax())
    return float(values[best]) if counts[best] >= max(8, .1*len(prefix)) else None


def probe_rows(series, domain, horizons=(1, 8, 32), max_origins=512):
    """Fixed causal covariates; age is a representation probe, not new information."""
    y = series.values
    if len(y) < 160 or not np.all(np.isfinite(y)):
        return None, "short_or_nonfinite"
    split = series.train_end
    if domain == "price":
        if np.any(y <= 0):
            return None, "nonpositive_price"
        x = np.r_[0.0, np.diff(np.log(y))]
        # The state at t compares r_t against volatility estimated through t-1.
        variance = pd.Series(x*x).ewm(alpha=0.06, adjust=False).mean().shift(1)
        scale = np.sqrt(variance.fillna(1e-8).to_numpy().clip(1e-10))
        state = (np.abs(x) > 1.5 * scale).astype(float)
        transformed = x / scale
    else:
        x = y
        atom = calibration_atom(y)
        if atom is None:
            return None, "no_atom_in_initial_calibration"
        state = (y == atom).astype(float)
        zero_rate = state[:split].mean()
        if not 0.01 <= zero_rate <= 0.99:
            return None, "not_intermittent_on_training_prefix"
        # Expanding nonzero scale is available at each origin, including training.
        scale = pd.Series(np.where(y != 0, np.abs(y), np.nan)).expanding().median().ffill().fillna(1.0).to_numpy().clip(1e-8)
        transformed = np.arcsinh(y / scale)
    ages = state_ages(state)
    s, z = pd.Series(transformed), pd.Series(state)
    features = {"state": state, "value": transformed,
                "historical_state_rate": z.expanding().mean()}
    for lag in (1, 2, 4, 8):
        features[f"value_lag_{lag}"] = s.shift(lag)
        features[f"state_lag_{lag}"] = z.shift(lag)
    for window in (8, 32, 96):
        features[f"mean_{window}"] = s.rolling(window).mean()
        features[f"std_{window}"] = s.rolling(window).std()
        features[f"state_rate_{window}"] = z.rolling(window).mean()
    frame = pd.DataFrame(features)
    frame["log_age"] = np.log1p(ages)
    frame["age_left_censored"] = (ages == np.arange(len(y)) + 1).astype(float)
    parts = []
    for horizon in horizons:
        # Train labels cannot overlap validation. No targets outside dev prefix.
        for partition, start, stop in (("train", 96, split-horizon),
                                       ("validation", split, len(y)-horizon)):
            if stop <= start:
                continue
            origins = np.unique(np.linspace(start, stop-1, min(max_origins, stop-start), dtype=int))
            out = frame.iloc[origins].copy()
            out["label"] = state[origins+horizon]
            out["horizon"] = horizon
            out["partition"] = partition
            out["dataset"] = series.dataset
            out["item"] = series.item
            out["origin"] = origins
            # Common calendar blocks account for correlated financial assets.
            out["cluster"] = (series.dataset if domain == "boom" else
                              np.asarray([f"{str(series.dates[t])[:7]}" for t in origins]))
            parts.append(out)
    if not parts:
        return None, "no_eligible_origins"
    return pd.concat(parts, ignore_index=True).dropna(), None
