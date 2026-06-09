"""Sliding-window dataset + leak-free chronological splitting.

A ``DatasetBundle`` holds aligned arrays for one series; ``make_splits`` carves
it into train/val/test using the canonical TSF protocol (Informer/Autoformer/
iTransformer): windows are partitioned by **label** position, and a window's
lookback may reach back across the segment boundary into earlier data. This is
leak-free because no *future label* ever enters an earlier split, while still
using all available history for context — and it matches the protocol baselines
are evaluated under, which is required for fair SOTA claims (PROPOSAL.md §6.5).

For the financial scenario, an optional ``embargo`` purges train windows whose
label region ends within ``embargo`` steps of the validation boundary,
mirroring purged walk-forward backtesting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .contract import CovariateSpec, WindowSample

__all__ = ["DatasetBundle", "WindowDataset", "make_splits", "SplitIndices"]


@dataclass(frozen=True)
class DatasetBundle:
    """Aligned arrays for a single series (time on axis 0)."""

    endo: np.ndarray          # [T, C_endo] endogenous channels (input+predicted)
    past_cov: np.ndarray      # [T, C_p]    full past-only covariate matrix
    known_cov: np.ndarray     # [T, C_f]    full known-future covariate matrix
    static: np.ndarray        # [C_s]       static vector (shared across time)
    spec: CovariateSpec       # column roles + counts
    name: str = "unnamed"     # series id, propagated into WindowSample.meta

    def __post_init__(self) -> None:
        # Cross-check that every matrix is time-aligned to the endogenous block
        # and has exactly the channel counts the spec advertises — catches
        # adapter bugs at load time instead of deep in training.
        T = self.endo.shape[0]
        if self.endo.shape != (T, self.spec.c_endo):
            raise ValueError(
                f"endo shape {self.endo.shape} != ({T}, {self.spec.c_endo})"
            )
        if self.past_cov.shape != (T, self.spec.c_p):
            raise ValueError(
                f"past_cov shape {self.past_cov.shape} != ({T}, {self.spec.c_p})"
            )
        if self.known_cov.shape != (T, self.spec.c_f):
            raise ValueError(
                f"known_cov shape {self.known_cov.shape} != ({T}, {self.spec.c_f})"
            )

    @property
    def length(self) -> int:
        # Number of timesteps T.
        return int(self.endo.shape[0])


@dataclass(frozen=True)
class SplitIndices:
    """Window-start indices per split + label-segment boundaries (for tests).

    A window with start ``s`` has label region ``[s+L, s+L+H)``. ``bounds`` gives
    the [lo, hi) label boundaries of each split; starts may be < their bound's
    lo (lookback crosses the boundary) but label regions never do.
    """

    train: np.ndarray                       # valid start indices for train
    val: np.ndarray                         # valid start indices for val
    test: np.ndarray                        # valid start indices for test
    bounds: dict[str, tuple[int, int]]      # label [lo, hi) per split


def _starts_for_labels(label_lo: int, label_hi: int, L: int, H: int) -> np.ndarray:
    """Starts s with label region [s+L, s+L+H) inside [label_lo, label_hi)."""
    # Earliest start whose label begins at label_lo: lookback reaches L steps
    # back, possibly across the segment boundary (allowed — it's just history).
    first = label_lo - L
    # Latest start whose label END (s+L+H) still fits before label_hi.
    last = label_hi - L - H
    # Clamp: never index before t=0 (no data exists there).
    if last < first or first < 0:
        first = max(first, 0)
    # Empty segment (too short for one window) → return an empty index array.
    if last < first:
        return np.empty(0, dtype=np.int64)
    # Inclusive range [first, last].
    return np.arange(first, last + 1, dtype=np.int64)


def make_splits(
    T: int,
    L: int,
    H: int,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    embargo: int | None = None,
) -> SplitIndices:
    """Label-partitioned chronological split (canonical TSF protocol).

    embargo (default 0) purges train windows whose label end lies within
    ``embargo`` steps of the train/val boundary — use >0 for financial data.
    """
    # Fail loudly on misconfigured ratios rather than silently dropping data.
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    if embargo is None:
        embargo = 0
    # Label boundaries: train labels occupy [0, n_train), val [n_train, n_val),
    # test [n_val, T). Computed on the absolute timeline.
    n_train = int(T * ratios[0])
    n_val = int(T * (ratios[0] + ratios[1]))
    bounds = {"train": (0, n_train), "val": (n_train, n_val), "test": (n_val, T)}

    # Train starts whose labels fall entirely before the val boundary.
    train = _starts_for_labels(0, n_train, L, H)
    # Purge (finance): drop train windows whose label ends within `embargo` of
    # the val boundary, so autocorrelated samples can't leak across it.
    if embargo > 0 and train.size:
        train = train[(train + L + H) <= (n_train - embargo)]
    # Val / test starts — note their lookback may legally read earlier segments.
    val = _starts_for_labels(n_train, n_val, L, H)
    test = _starts_for_labels(n_val, T, L, H)
    return SplitIndices(train=train, val=val, test=test, bounds=bounds)


class WindowDataset(Dataset):
    """Yields ``WindowSample`` objects for a given set of start indices."""

    def __init__(
        self,
        bundle: DatasetBundle,
        starts: np.ndarray,
        L: int,
        H: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # Keep a reference to the shared arrays (no copy) — windows are sliced
        # lazily in __getitem__ to keep memory flat for long series.
        self._b = bundle
        self._starts = np.asarray(starts, dtype=np.int64)
        self.L = int(L)
        self.H = int(H)
        self._dtype = dtype
        # Static vector is identical for every window → convert once, reuse.
        self._static = torch.as_tensor(bundle.static, dtype=dtype)

    def __len__(self) -> int:
        # One sample per valid start index.
        return int(self._starts.shape[0])

    def __getitem__(self, idx: int) -> WindowSample:
        # Resolve the absolute start position s on the global timeline.
        s = int(self._starts[idx])
        L, H = self.L, self.H
        t = torch.as_tensor  # local alias for brevity
        # Slice the lookback window [s, s+L) and horizon window [s+L, s+L+H).
        # The WindowSample constructor re-validates every shape (cheap insurance).
        return WindowSample(
            x_endo=t(self._b.endo[s : s + L], dtype=self._dtype),             # [L, C_endo]
            x_past_cov=t(self._b.past_cov[s : s + L], dtype=self._dtype),      # [L, C_p]
            x_known_past=t(self._b.known_cov[s : s + L], dtype=self._dtype),   # [L, C_f]
            # Known-future segment lives AFTER the lookback — legitimately known.
            x_known_fut=t(self._b.known_cov[s + L : s + L + H], dtype=self._dtype),  # [H, C_f]
            static=self._static,                                              # [C_s]
            y=t(self._b.endo[s + L : s + L + H], dtype=self._dtype),          # [H, C_endo] label
            meta={"series": self._b.name, "start": s},
        )

    @property
    def starts(self) -> np.ndarray:
        # Expose the resolved start indices (used by tests / diagnostics).
        return self._starts
