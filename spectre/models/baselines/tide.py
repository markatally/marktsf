"""TiDE baseline (Das et al., TMLR 2024) — compact channel-independent variant.

TiDE is an MLP encoder-decoder built from residual blocks; it showed dense MLPs
rival Transformers on long-horizon forecasting. This compact version keeps the
residual-block backbone and processes each channel independently (channels
folded into the batch), with RevIN for non-stationarity. Covariate encoders from
the full TiDE are omitted here (added later for the MISO scenarios).
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from ..base import ForecastModel
from ..layers.revin import RevIN

__all__ = ["TiDE"]


class _ResBlock(nn.Module):
    """Residual MLP block: 2-layer MLP + linear skip + LayerNorm."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.skip = nn.Linear(d_in, d_out)   # projects input to match d_out
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor) -> Tensor:
        h = self.lin2(self.drop(F.relu(self.lin1(x))))
        # Residual add (with projection) then normalize.
        return self.norm(h + self.skip(x))


class TiDE(ForecastModel):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        c_endo: int,
        hidden: int = 256,
        n_enc: int = 2,
        n_dec: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.revin = RevIN(num_features=c_endo, affine=True)
        # Encoder: lookback vector → latent; first block maps L→hidden, rest keep.
        enc = [_ResBlock(lookback, hidden, hidden, dropout)]
        enc += [_ResBlock(hidden, hidden, hidden, dropout) for _ in range(n_enc - 1)]
        self.encoder = nn.Sequential(*enc)
        # Decoder: latent → horizon; last block maps hidden→horizon.
        dec = [_ResBlock(hidden, hidden, hidden, dropout) for _ in range(n_dec - 1)]
        dec += [_ResBlock(hidden, hidden, horizon, dropout)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = self.revin(batch["x_endo"], "norm")     # [B, L, C]
        B, L, C = x.shape
        # Channel independence: each channel's L-vector is one MLP input.
        seq = x.transpose(1, 2).reshape(B * C, L)    # [B*C, L]
        latent = self.encoder(seq)                   # [B*C, hidden]
        out = self.decoder(latent)                   # [B*C, H]
        out = out.reshape(B, C, -1).transpose(1, 2)  # [B, H, C]
        y_hat = self.revin(out, "denorm")
        return {"y_hat": y_hat, "x_hat": None}
