"""Train-only normalization statistics.

Fitting on the training segment only (never on val/test) prevents statistical
leakage. ``transform`` returns new arrays and never mutates its input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Standardizer"]


@dataclass(frozen=True)
class Standardizer:
    """Per-channel z-score standardizer fit on a train index range."""

    mean: np.ndarray   # per-channel mean (scalar for 1-D input)
    std: np.ndarray    # per-channel std, floored away from zero

    @classmethod
    def fit(cls, arr: np.ndarray, train_end: int, eps: float = 1e-8) -> "Standardizer":
        """Fit on ``arr[:train_end]`` along axis 0. Accepts [T] or [T, C]."""
        # Work in float64 for numerically stable mean/std on long series.
        a = np.asarray(arr, dtype=np.float64)
        # CRITICAL: only the training segment contributes statistics — this is
        # what makes the normalization leak-free.
        seg = a[:train_end]
        mean = seg.mean(axis=0)
        std = seg.std(axis=0)
        # Replace ~zero std (constant channels) with 1.0 to avoid divide-by-zero.
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        # Returns a NEW array (immutability): (x - mean) / std.
        return (np.asarray(arr, dtype=np.float64) - self.mean) / self.std

    def inverse(self, arr: np.ndarray) -> np.ndarray:
        # Map standardized values back to the original scale (for reporting).
        return np.asarray(arr, dtype=np.float64) * self.std + self.mean
