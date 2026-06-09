"""iTransformer baseline (Liu et al., ICLR 2024).

The "inverted" Transformer: instead of treating each time step as a token, it
treats each *variate's entire lookback series* as one token, so self-attention
models cross-variate (channel) dependencies directly. A linear head then maps
each variate token to the horizon. RevIN handles non-stationarity.

This is the headline baseline our asymmetric method is measured against, so its
multivariate (all-channel) behavior must be faithful (PROPOSAL.md §6.3).
"""

from __future__ import annotations

from typing import Any

from torch import nn

from ..base import ForecastModel
from ..layers.revin import RevIN

__all__ = ["ITransformer"]


class ITransformer(ForecastModel):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        c_endo: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.revin = RevIN(num_features=c_endo, affine=True)
        # Variate embedding: each channel's L-length series → one D-dim token.
        self.embed = nn.Linear(lookback, d_model)
        self.dropout = nn.Dropout(dropout)
        # Pre-norm encoder over the C variate tokens (GELU per the paper).
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # tokens shaped [B, C, D]
            norm_first=True,
        )
        # enable_nested_tensor=False: silences a no-op warning under norm_first.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=e_layers, enable_nested_tensor=False
        )
        # Project each variate token back to the horizon.
        self.head = nn.Linear(d_model, horizon)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = self.revin(batch["x_endo"], "norm")     # [B, L, C]
        # [B, L, C] -> [B, C, L] -> embed each variate series -> [B, C, D].
        tokens = self.dropout(self.embed(x.transpose(1, 2)))
        enc = self.encoder(tokens)                  # [B, C, D] cross-variate attn
        out = self.head(enc).transpose(1, 2)        # [B, H, C]
        y_hat = self.revin(out, "denorm")           # restore scale
        return {"y_hat": y_hat, "x_hat": None}
