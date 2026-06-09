"""RevIN — reversible instance normalization (Kim et al., 2022).

Normalizes each instance/channel by its own window statistics before encoding
and restores them after decoding, absorbing first-moment non-stationarity
(PROPOSAL.md §3.1). With affine disabled (or identity-initialized), denorm is an
exact inverse of norm.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RevIN"]


class RevIN(nn.Module):
    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps                 # variance floor for numerical stability
        self.affine = affine           # learnable per-channel scale/shift?
        if affine:
            # Identity init (weight=1, bias=0) → at start RevIN is a pure
            # normalize/denormalize pair with no learned distortion.
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        # Per-forward cache of the normalization stats, needed to invert later.
        # Stored on the module because norm and denorm are separate calls.
        self._mean: Tensor | None = None
        self._std: Tensor | None = None

    def forward(self, x: Tensor, mode: str) -> Tensor:
        """x: [B, L, C]. mode in {'norm', 'denorm'}."""
        # Dispatch on the requested direction; reject typos explicitly.
        if mode == "norm":
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got {mode!r}")

    def _normalize(self, x: Tensor) -> Tensor:
        # Compute per-instance, per-channel statistics over the TIME axis (dim=1).
        # .detach() stops gradients flowing through the statistics themselves —
        # they act as constants for this instance (the RevIN design choice).
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        # Standardize to zero-mean/unit-var per instance.
        out = (x - self._mean) / self._std
        if self.affine:
            # Apply the learnable affine AFTER standardization.
            out = out * self.weight + self.bias
        return out

    def _denormalize(self, x: Tensor) -> Tensor:
        # Denorm must run after a matching norm call populated the caches.
        if self._mean is None or self._std is None:
            raise RuntimeError("RevIN.denorm called before norm")
        out = x
        if self.affine:
            # Invert the affine first (mirror image of _normalize). The tiny
            # eps*eps guards against a learned weight collapsing to 0.
            out = (out - self.bias) / (self.weight + self.eps * self.eps)
        # Re-inject the original location/scale → back to data units.
        return out * self._std + self._mean
