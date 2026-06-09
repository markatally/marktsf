"""Model interface: every forecaster consumes a batch dict, returns outputs.

The output dict always carries ``y_hat`` [B, H] (target forecast) and may carry
``x_hat`` [B, H, C_p] (auxiliary covariate forecast, used only by the Version-A
loss; see PROPOSAL.md §3.5). Baselines return ``x_hat=None``.
"""

from __future__ import annotations

from typing import Any

from torch import nn

__all__ = ["ForecastModel"]


class ForecastModel(nn.Module):
    """Base class. Subclasses implement ``forward(batch) -> dict``.

    Required input keys (see contract.collate_samples):
        x_target [B, L], x_past_cov [B, L, C_p], x_known_past [B, L, C_f],
        x_known_fut [B, H, C_f], static [B, C_s], y [B, H]
    Required output keys:
        y_hat [B, H]; optional x_hat [B, H, C_p].
    """

    # Subclasses MUST override; the base raises to surface a missing impl early.
    # (Marked no-cover because it is never meant to execute.)
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
