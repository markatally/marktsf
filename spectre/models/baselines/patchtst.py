"""PatchTST baseline (Nie et al., ICLR 2023).

Two defining ideas:
1. Patching: each channel's lookback is split into overlapping patches, so the
   Transformer attends over ~L/stride patch tokens instead of L time steps.
2. Channel independence: every channel is processed by the SAME model with no
   cross-channel mixing (channels are folded into the batch dimension).

RevIN handles non-stationarity. A flatten+linear head maps patch tokens → horizon.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..base import ForecastModel
from ..layers.revin import RevIN

__all__ = ["PatchTST"]


class PatchTST(ForecastModel):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        c_endo: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback < patch_len:
            raise ValueError(f"lookback {lookback} < patch_len {patch_len}")
        self.revin = RevIN(num_features=c_endo, affine=True)
        self.patch_len = patch_len
        self.stride = stride
        # Number of patches from a length-L window via a sliding unfold.
        self.num_patches = (lookback - patch_len) // stride + 1
        # Per-patch value embedding + a learnable positional code per patch slot.
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # tokens shaped [B*C, num_patches, D]
            norm_first=True,
        )
        # enable_nested_tensor=False: silences a no-op warning under norm_first.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=e_layers, enable_nested_tensor=False
        )
        # Flatten all patch tokens, then map to the horizon.
        self.head = nn.Linear(self.num_patches * d_model, horizon)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = self.revin(batch["x_endo"], "norm")     # [B, L, C]
        B, L, C = x.shape
        # Channel independence: fold channels into the batch → [B*C, L].
        seq = x.transpose(1, 2).reshape(B * C, L)
        # Sliding patches → [B*C, num_patches, patch_len].
        patches = seq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Embed + add positional code, then attend over patch tokens.
        tok = self.dropout(self.embed(patches) + self.pos)   # [B*C, P, D]
        enc = self.encoder(tok)                              # [B*C, P, D]
        flat = enc.reshape(B * C, -1)                        # [B*C, P*D]
        out = self.head(flat).reshape(B, C, -1).transpose(1, 2)  # [B, H, C]
        y_hat = self.revin(out, "denorm")
        return {"y_hat": y_hat, "x_hat": None}
