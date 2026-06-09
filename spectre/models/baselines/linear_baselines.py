"""Linear baselines: RLinear and NLinear (channel-independent).

Both are deliberately tiny yet competitive (Zeng et al. 2023; Li et al. 2023):
- NLinear: subtract the last lookback value, apply one Linear L→H, add it back —
  a cheap way to absorb a level shift between lookback and horizon.
- RLinear: RevIN normalize → one shared Linear L→H → RevIN denormalize.

Linear weights are SHARED across channels; non-stationarity is handled by the
subtraction (NLinear) or RevIN (RLinear).
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..base import ForecastModel
from ..layers.revin import RevIN

__all__ = ["NLinear", "RLinear"]


class NLinear(ForecastModel):
    def __init__(self, lookback: int, horizon: int, c_endo: int) -> None:
        super().__init__()
        # One shared linear map across all endogenous channels.
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = batch["x_endo"]            # [B, L, C]
        last = x[:, -1:, :]            # [B, 1, C] — level at the lookback end
        x = x - last                   # de-level so the linear sees deviations
        out = self.linear(x.transpose(1, 2)).transpose(1, 2)  # [B, H, C]
        y_hat = out + last             # re-add the level (broadcast over H)
        return {"y_hat": y_hat, "x_hat": None}


class RLinear(ForecastModel):
    def __init__(self, lookback: int, horizon: int, c_endo: int) -> None:
        super().__init__()
        # RevIN with a learnable per-channel affine handles the distribution shift.
        self.revin = RevIN(num_features=c_endo, affine=True)
        self.linear = nn.Linear(lookback, horizon)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = self.revin(batch["x_endo"], "norm")               # [B, L, C]
        out = self.linear(x.transpose(1, 2)).transpose(1, 2)  # [B, H, C]
        y_hat = self.revin(out, "denorm")                     # restore scale
        return {"y_hat": y_hat, "x_hat": None}
