"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "pick_device"]


def set_seed(seed: int) -> None:
    # Seed every RNG that can affect a run: python hash seed, stdlib random,
    # numpy, and torch (CPU + all CUDA devices) — so results are reproducible.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str = "auto") -> torch.device:
    # Explicit override wins (e.g. "cpu" for deterministic debugging).
    if prefer != "auto":
        return torch.device(prefer)
    # Otherwise prefer CUDA, then Apple-silicon MPS, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
