"""Central model registry: config strings → model classes.

Every baseline shares the uniform constructor signature
``(lookback, horizon, c_endo, **hparams)`` so the runner can build any of them
the same way. New models are added here without touching runner logic.
"""

from __future__ import annotations

from typing import Any

from .base import ForecastModel
from .baselines.dlinear import DLinear
from .baselines.itransformer import ITransformer
from .baselines.linear_baselines import NLinear, RLinear
from .baselines.patchtst import PatchTST
from .baselines.tide import TiDE

__all__ = ["MODEL_REGISTRY", "build_model"]

MODEL_REGISTRY: dict[str, type[ForecastModel]] = {
    "dlinear": DLinear,
    "nlinear": NLinear,
    "rlinear": RLinear,
    "itransformer": ITransformer,
    "patchtst": PatchTST,
    "tide": TiDE,
}


def build_model(
    name: str, lookback: int, horizon: int, c_endo: int, **hparams: Any
) -> ForecastModel:
    # Resolve the model name; fail with the known list if unknown.
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](
        lookback=lookback, horizon=horizon, c_endo=c_endo, **hparams
    )
