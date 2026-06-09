"""Calendar features = canonical known-future covariates.

Calendar values are deterministic functions of the timestamp, hence available
over the forecast horizon — the legitimate ``known_future`` stream
(PROPOSAL.md §1). Encoded in [-0.5, 0.5] to keep them scale-friendly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["calendar_features", "CALENDAR_COLUMNS"]

# Stable, ordered names for the 4 calendar channels (kept in sync with the
# columns produced by calendar_features so adapters can advertise them).
CALENDAR_COLUMNS: tuple[str, ...] = ("cal_hour", "cal_dow", "cal_dom", "cal_month")


def calendar_features(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return [T, 4] calendar features (hour, day-of-week, day-of-month, month)."""
    # Normalize the input into a DatetimeIndex regardless of how it arrives.
    idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    # Each component is scaled by its max value then centered to [-0.5, 0.5]:
    #   hour ∈ 0..23, dow ∈ 0..6, day ∈ 1..31, month ∈ 1..12.
    hour = idx.hour.to_numpy() / 23.0 - 0.5
    dow = idx.dayofweek.to_numpy() / 6.0 - 0.5
    dom = (idx.day.to_numpy() - 1) / 30.0 - 0.5      # shift day to 0-based first
    month = (idx.month.to_numpy() - 1) / 11.0 - 0.5  # shift month to 0-based first
    # Stack the 4 channels column-wise → [T, 4].
    return np.stack([hour, dow, dom, month], axis=1).astype(np.float64)
