"""General forecasting metrics (MSE, MAE) on standardized targets."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["mse", "mae", "GENERAL_METRICS"]


def mse(y_hat: Tensor, y: Tensor) -> float:
    # Mean squared error over all elements; returned as a python float.
    return float(torch.mean((y_hat - y) ** 2).item())


def mae(y_hat: Tensor, y: Tensor) -> float:
    # Mean absolute error over all elements.
    return float(torch.mean(torch.abs(y_hat - y)).item())


# Name→fn registry so the trainer/evaluator can iterate metrics generically.
GENERAL_METRICS = {"mse": mse, "mae": mae}
