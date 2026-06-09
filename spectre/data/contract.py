"""Unified sample contract for all SPECTRE datasets and models.

Every dataset adapter maps raw files into ``WindowSample`` objects; every model
consumes batches collated from them. This is the single interface boundary the
whole codebase is built around (see PROPOSAL.md §6.2).

The contract is mode-agnostic via the **endogenous block** ``x_endo`` / ``y`` of
width ``C_endo`` — the channels that are both observed and predicted:
- Symmetric multivariate (baseline reproduction): C_endo = all channels.
- Asymmetric MISO (the SPECTRE method): C_endo = 1 (the single target) plus
  exogenous covariates in ``x_past_cov`` / ``x_known_*``.

Design rules honored here:
- Immutability: ``WindowSample`` is a frozen dataclass; transforms return new
  objects, never mutate inputs.
- Boundary validation: shapes are checked at construction; malformed samples
  fail fast with a clear message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

# Public surface of this module — keeps `from contract import *` explicit.
__all__ = ["WindowSample", "collate_samples", "CovariateSpec"]


@dataclass(frozen=True)
class CovariateSpec:
    """Declares the role of every channel for a dataset.

    The tuples hold human-readable column names for traceability; the numeric
    counts (``c_endo``/``c_p``/``c_f``/``c_s``) are what models rely on.
    """

    # Endogenous channels: observed in the lookback AND predicted in the horizon.
    # MISO uses exactly one (the target); symmetric multivariate uses all.
    endogenous: tuple[str, ...]
    # past_only: covariates observed only up to `t` (e.g. order-flow proxy).
    past_only: tuple[str, ...] = ()
    # known_future: covariates whose future values are known (e.g. calendar).
    known_future: tuple[str, ...] = ()
    # static: time-invariant descriptors (asset id, store id); optional.
    static: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # There must be at least one channel to predict.
        if len(self.endogenous) == 0:
            raise ValueError("CovariateSpec needs ≥1 endogenous channel")

    @property
    def c_endo(self) -> int:
        # Number of predicted channels — drives the model output width.
        return len(self.endogenous)

    @property
    def c_p(self) -> int:
        # Channel count of the past-only block.
        return len(self.past_only)

    @property
    def c_f(self) -> int:
        # Channel count of the known-future block.
        return len(self.known_future)

    @property
    def c_s(self) -> int:
        # Channel count of the static block.
        return len(self.static)

    @property
    def target(self) -> str:
        # Convenience for the MISO case: the (first) endogenous channel.
        return self.endogenous[0]

    @property
    def is_miso(self) -> bool:
        # True when exactly one endogenous channel + at least one covariate.
        return self.c_endo == 1 and (self.c_p + self.c_f) > 0


@dataclass(frozen=True)
class WindowSample:
    """A single (lookback, horizon) training/eval example.

    Shapes (L = lookback, H = horizon):
        x_endo       : [L, C_endo]   endogenous history (input + predicted)
        x_past_cov   : [L, C_p]
        x_known_past : [L, C_f]
        x_known_fut  : [H, C_f]
        static       : [C_s]
        y            : [H, C_endo]   endogenous future (label)
    """

    x_endo: Tensor         # [L, C_endo]  endogenous history
    x_past_cov: Tensor     # [L, C_p]     past-only covariate history
    x_known_past: Tensor   # [L, C_f]     known-future covariates, past segment
    x_known_fut: Tensor    # [H, C_f]     known-future covariates, future segment
    static: Tensor         # [C_s]        static covariates
    y: Tensor              # [H, C_endo]  endogenous future (label)
    # meta carries provenance (series name, start index) for debugging/eval;
    # default_factory avoids the mutable-default-argument trap.
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate every tensor's rank/shape AT CONSTRUCTION so a malformed
        # sample can never silently propagate into the model.
        self._require(self.x_endo.ndim == 2, "x_endo must be 2-D [L, C_endo]")
        self._require(self.y.ndim == 2, "y must be 2-D [H, C_endo]")
        L = int(self.x_endo.shape[0])       # lookback from the endogenous block
        H = int(self.y.shape[0])            # horizon from the label
        c_endo = int(self.x_endo.shape[1])
        # Endogenous input and label must share the channel count C_endo.
        self._require(
            self.y.shape[1] == c_endo,
            f"y C_endo={self.y.shape[1]} != x_endo C_endo={c_endo}",
        )
        # Past covariates: 2-D and time-aligned to the lookback length L.
        self._require(
            self.x_past_cov.ndim == 2 and self.x_past_cov.shape[0] == L,
            f"x_past_cov must be [L, C_p] with L={L}, got {tuple(self.x_past_cov.shape)}",
        )
        # Known-future (past segment): 2-D and also length L.
        self._require(
            self.x_known_past.ndim == 2 and self.x_known_past.shape[0] == L,
            f"x_known_past must be [L, C_f] with L={L}, got {tuple(self.x_known_past.shape)}",
        )
        # The known-future channel count C_f is anchored on the past segment...
        c_f = int(self.x_known_past.shape[1])
        # ...and the future segment must match it on C_f and span exactly H steps.
        self._require(
            self.x_known_fut.ndim == 2
            and self.x_known_fut.shape[0] == H
            and self.x_known_fut.shape[1] == c_f,
            f"x_known_fut must be [H, C_f] with H={H}, C_f={c_f}, "
            f"got {tuple(self.x_known_fut.shape)}",
        )
        # Static covariates are a flat vector (possibly empty).
        self._require(self.static.ndim == 1, "static must be 1-D [C_s]")

    @staticmethod
    def _require(cond: bool, msg: str) -> None:
        # Tiny assertion helper that raises a typed, descriptive error.
        if not cond:
            raise ValueError(f"WindowSample validation failed: {msg}")

    @property
    def lookback(self) -> int:
        # Convenience accessor: L.
        return int(self.x_endo.shape[0])

    @property
    def horizon(self) -> int:
        # Convenience accessor: H.
        return int(self.y.shape[0])

    @property
    def c_endo(self) -> int:
        # Convenience accessor: number of endogenous channels.
        return int(self.x_endo.shape[1])


def collate_samples(samples: list[WindowSample]) -> dict[str, Any]:
    """Stack a list of samples into a batched dict consumed by models."""
    # Guard against empty batches early — torch.stack would raise a cryptic error.
    if not samples:
        raise ValueError("collate_samples received an empty batch")
    # Stack each field along a new leading batch dim B; meta stays a python list
    # (it is non-tensor, per-sample provenance).
    return {
        "x_endo": torch.stack([s.x_endo for s in samples]),           # [B, L, C_endo]
        "x_past_cov": torch.stack([s.x_past_cov for s in samples]),   # [B, L, C_p]
        "x_known_past": torch.stack([s.x_known_past for s in samples]),# [B, L, C_f]
        "x_known_fut": torch.stack([s.x_known_fut for s in samples]), # [B, H, C_f]
        "static": torch.stack([s.static for s in samples]),           # [B, C_s]
        "y": torch.stack([s.y for s in samples]),                     # [B, H, C_endo]
        # Copy each meta dict so downstream mutation can't corrupt the samples.
        "meta": [dict(s.meta) for s in samples],
    }
