"""
Crypto MISO dataset for the PRISM Oracle Drift Study.

Reads from input/Crypto/*.csv (1-minute Binance OHLCV per asset), resamples
to an hourly frequency, computes close log-returns (and optionally volume
log-returns), and aligns all 14 assets by timestamp.

Column layout (BTC target is always last = target_channel -1):
  target="ret", include_volume=False : [close_ret × 13 non-BTC, btc_close_ret]          (14 cols)
  target="ret", include_volume=True  : [close_ret × 13 non-BTC, vol_ret × 14, btc_close_ret] (28 cols)
  target="vol"                        : same non-target covariates, btc_log_rv as last col

No derived files are written to the repository.  The pipeline is fully
in-memory: input/ → pd.DataFrame → numpy → torch Tensors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# 14 Binance assets; BTC is last so it maps to target_channel = -1.
ASSETS: tuple[str, ...] = (
    "ADAUSDT",
    "ATOMUSDT",
    "AVAXUSDT",
    "BNBUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "MATICUSDT",
    "SOLUSDT",
    "UNIUSDT",
    "XRPUSDT",
    "BTCUSDT",  # target — always last
)


def _load_asset_close_returns(csv_path: Path, freq: str) -> pd.Series:
    """Resample to freq and return hourly close log-returns."""
    df = pd.read_csv(csv_path, usecols=["open_time", "close"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    close = df.set_index("date")["close"]
    close = close.resample(freq).last().ffill()
    ret = np.log(close).diff()
    ret.name = f"{csv_path.stem}_close"
    return ret


def _load_asset_vol_returns(csv_path: Path, freq: str) -> pd.Series:
    """Resample to freq and return hourly volume log-returns (log vol diff)."""
    df = pd.read_csv(csv_path, usecols=["open_time", "volume"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    vol = df.set_index("date")["volume"]
    vol = vol.resample(freq).sum()
    vol = vol.replace(0, np.nan).ffill()
    ret = np.log(vol).diff()
    ret.name = f"{csv_path.stem}_vol"
    return ret


def _load_btc_log_rv(csv_path: Path, hourly_freq: str = "1h") -> pd.Series:
    """
    Compute hourly log realized variance for BTC from 1-minute closes.

    log_RV[t] = log(Σ r_i² for 1-min returns within hour t)

    Log-RV is approximately Gaussian and strongly autocorrelated, making MSE
    a meaningful loss function (unlike raw returns).  A small floor (1e-10)
    is applied before the log to avoid -inf.

    Note: ``hourly_freq`` is only used for alignment; the realized variance is
    always computed from the native 1-minute bar resolution.
    """
    df = pd.read_csv(csv_path, usecols=["open_time", "close"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("date").sort_index()

    # 1-minute log returns.
    close_1m = df["close"].asfreq("1min").ffill()
    ret_1m = np.log(close_1m).diff()

    # Sum of squared 1-minute returns within each hour = realized variance.
    rv = (ret_1m ** 2).resample(hourly_freq).sum()
    rv = rv.replace(0, np.nan).ffill()
    log_rv = np.log(rv.clip(lower=1e-10))
    log_rv.name = "BTCUSDT_log_rv"
    return log_rv


def build_crypto_panel(
    input_dir: str | Path,
    assets: Sequence[str] = ASSETS,
    freq: str = "1h",
    include_volume: bool = False,
    target: str = "ret",
) -> pd.DataFrame:
    """
    Build an aligned DataFrame of MISO features for all assets.

    Parameters
    ----------
    input_dir      : path to input/Crypto/
    assets         : ordered sequence; last entry is the target asset
    freq           : pandas resample frequency string (default '1h')
    include_volume : if True, append volume log-returns for all assets
                     (ignored when target='vol')
    target         : 'ret'  → BTC close log-return (last col)
                     'vol'  → BTC hourly log realized variance (last col)

    Column layout (target always last = channel -1):
      target='ret', include_volume=False : [close_ret × 13 non-BTC, btc_close_ret]
      target='ret', include_volume=True  : [close_ret × 13 non-BTC, vol_ret × 14, btc_close_ret]
      target='vol'                        : [close_ret × 13 non-BTC, btc_log_rv]

    Returns
    -------
    pd.DataFrame with DatetimeIndex; last column is the target series.
    """
    if target not in ("ret", "vol"):
        raise ValueError(f"target must be 'ret' or 'vol', got {target!r}")

    input_dir = Path(input_dir)
    target_asset = assets[-1]

    close_series: list[pd.Series] = []
    vol_series: list[pd.Series] = []
    for asset in assets:
        path = input_dir / f"{asset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Crypto CSV: {path}")
        close_series.append(_load_asset_close_returns(path, freq))
        if include_volume and target == "ret":
            vol_series.append(_load_asset_vol_returns(path, freq))
        log.debug("loaded %s", asset)

    non_target_close = [s for s in close_series if s.name != f"{target_asset}_close"]

    if target == "ret":
        target_col = next(s for s in close_series if s.name == f"{target_asset}_close")
        parts: list[pd.Series] = list(non_target_close)
        if include_volume:
            parts += vol_series
        parts.append(target_col)
    else:  # target == "vol"
        btc_path = input_dir / f"{target_asset}.csv"
        target_col = _load_btc_log_rv(btc_path, hourly_freq=freq)
        parts = list(non_target_close) + [target_col]

    panel = pd.concat(parts, axis=1).dropna()
    log.info(
        "Crypto panel: %d rows × %d cols | target=%s | %s → %s",
        len(panel),
        panel.shape[1],
        target,
        panel.index[0].date(),
        panel.index[-1].date(),
    )
    return panel


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Encode [month, day, weekday, hour] normalised to [-0.5, 0.5]."""
    feat = np.column_stack(
        [
            index.month / 12.0 - 0.5,
            index.day / 31.0 - 0.5,
            index.dayofweek / 6.0 - 0.5,
            index.hour / 23.0 - 0.5,
        ]
    )
    return feat.astype(np.float32)


class CryptoDataset(Dataset):
    """
    Sliding-window dataset over the Crypto panel.

    Compatible with TSLib's (seq_x, seq_y, seq_x_mark, seq_y_mark) convention:
      seq_x      [L, C]              encoder input
      seq_y      [label_len + H, C]  decoder input (label + zero-padding)
      seq_x_mark [L, 4]              time features for encoder
      seq_y_mark [label_len + H, 4]  time features for decoder

    Parameters
    ----------
    panel       : output of build_crypto_panel
    split       : 'train' | 'val' | 'test'
    seq_len     : lookback L
    pred_len    : horizon H
    label_len   : decoder start-token length (typically seq_len // 2)
    scale       : fit StandardScaler on train partition, apply to all
    train_ratio : fraction of rows for training
    val_ratio   : fraction of rows for validation; remainder = test
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        split: str,
        seq_len: int,
        pred_len: int,
        label_len: int = 48,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ) -> None:
        assert split in ("train", "val", "test"), f"invalid split: {split!r}"

        n = len(panel)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        borders: dict[str, tuple[int, int]] = {
            "train": (0, n_train),
            "val": (n_train - seq_len, n_train + n_val),
            "test": (n_train + n_val - seq_len, n),
        }
        b1, b2 = borders[split]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        values = panel.values.astype(np.float32)
        if scale:
            scaler = StandardScaler()
            scaler.fit(values[:n_train])
            values = scaler.transform(values)

        self.data = values[b1:b2]
        self.time_feat = _time_features(panel.index)[b1:b2]

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int):
        s_end = idx + self.seq_len
        r_end = s_end + self.pred_len
        dec_start = s_end - self.label_len

        seq_x = self.data[idx:s_end]
        seq_y = self.data[dec_start:r_end]
        seq_x_mark = self.time_feat[idx:s_end]
        seq_y_mark = self.time_feat[dec_start:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
