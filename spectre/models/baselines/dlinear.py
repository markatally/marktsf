"""DLinear baseline (Zeng et al., AAAI 2023), channel-independent.

DLinear decomposes each endogenous channel's lookback into trend (moving
average) + remainder and maps each component to the horizon with a shared linear
layer. It ignores covariates by design, serving as the simple-but-strong
lower-bound baseline. Operates on the full endogenous block [B, L, C_endo].
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..base import ForecastModel

__all__ = ["DLinear"]


class _MovingAvg(nn.Module):
    """Per-channel sliding-window mean (trend extractor) with edge padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # AvgPool1d pools over the LAST dim independently per channel; we pad
        # manually so the output length equals the input length.
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:  # x: [B, C, L]
        pad = self.kernel_size - 1
        # Replicate edge values (ceil on the left, floor on the right) so the
        # trend isn't pulled toward zero at the series boundaries.
        front = x[..., :1].repeat(1, 1, pad - pad // 2)
        end = x[..., -1:].repeat(1, 1, pad // 2)
        padded = torch.cat([front, x, end], dim=-1)  # [B, C, L + pad]
        return self.pool(padded)                     # [B, C, L]


class DLinear(ForecastModel):
    def __init__(self, lookback: int, horizon: int, c_endo: int, kernel_size: int = 25) -> None:
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.c_endo = c_endo
        self.decomp = _MovingAvg(kernel_size)
        # Two shared linear maps L→H (weights reused across all channels).
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_season = nn.Linear(lookback, horizon)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = batch["x_endo"].transpose(1, 2)      # [B, C, L] — channel-major
        trend = self.decomp(x)                   # [B, C, L]
        season = x - trend                       # [B, C, L] residual
        # Project each component L→H and recombine, then back to [B, H, C].
        out = self.linear_trend(trend) + self.linear_season(season)  # [B, C, H]
        y_hat = out.transpose(1, 2)              # [B, H, C]
        return {"y_hat": y_hat, "x_hat": None}
