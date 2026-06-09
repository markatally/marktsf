"""Minimal, model-agnostic training loop with early stopping.

Loss = target MSE + (optional) auxiliary covariate MSE when the model returns
``x_hat`` and ``lambda_aux > 0`` (Version-A regularizer, PROPOSAL.md §3.6). The
auxiliary target is the future of the past-only covariates — known at train time
but never fed into the main head.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..metrics.general import GENERAL_METRICS

__all__ = ["TrainConfig", "Trainer"]


@dataclass(frozen=True)
class TrainConfig:
    """All training knobs in one immutable, serializable struct."""

    max_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5          # early-stop patience (epochs w/o val improvement)
    lambda_aux: float = 0.0    # weight of the Version-A auxiliary loss (0 = off)
    grad_clip: float = 5.0     # max global grad norm (0 disables clipping)
    log_every: int = 1


@dataclass
class _BestState:
    """Tracks the best checkpoint seen so far (for early stopping / restore)."""

    score: float = field(default=float("inf"))  # best val MSE
    weights: dict[str, Any] | None = None        # deep-copied state_dict
    epoch: int = -1


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, device: torch.device) -> None:
        # Move the model to the target device once, up front.
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        # AdamW = Adam with decoupled weight decay (a sane default for TSF).
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.loss_fn = nn.MSELoss()

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Move only tensor entries; leave `meta` (a python list) untouched.
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }

    def _step_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        out = self.model(batch)
        # Primary objective: target forecast error.
        loss = self.loss_fn(out["y_hat"], batch["y"])
        # Version-A auxiliary regularizer: only when enabled AND the model
        # actually produces a covariate forecast.
        if self.cfg.lambda_aux > 0 and out.get("x_hat") is not None:
            # Auxiliary target = the LAST H steps of the past covariates, i.e.
            # the covariates' own "future" relative to the window. Note this is
            # used only to shape the representation — it is never fed forward
            # into the main head (the whole point of Version A vs hard cascade).
            aux_target = batch["x_past_cov"][:, -out["x_hat"].shape[1] :, :]
            loss = loss + self.cfg.lambda_aux * self.loss_fn(out["x_hat"], aux_target)
        return loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        best = _BestState()
        history: list[dict[str, float]] = []
        for epoch in range(self.cfg.max_epochs):
            # ---- train one epoch ----
            self.model.train()
            for batch in train_loader:
                batch = self._to_device(batch)
                self.opt.zero_grad()
                loss = self._step_loss(batch)
                loss.backward()
                # Clip exploding gradients before the step (stabilizes training).
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()

            # ---- validate + early-stopping bookkeeping ----
            val_metrics = self.evaluate(val_loader)
            history.append({"epoch": epoch, **val_metrics})
            if val_metrics["mse"] < best.score - 1e-7:
                # New best → snapshot weights (deep copy so later epochs can't
                # mutate the saved checkpoint).
                best = _BestState(
                    score=val_metrics["mse"],
                    weights=copy.deepcopy(self.model.state_dict()),
                    epoch=epoch,
                )
            elif epoch - best.epoch >= self.cfg.patience:
                # No improvement for `patience` epochs → stop early.
                break

        # Restore the best checkpoint so downstream test eval uses it.
        if best.weights is not None:
            self.model.load_state_dict(best.weights)
        return {"best_val_mse": best.score, "best_epoch": best.epoch, "history": history}

    @torch.no_grad()  # eval needs no gradients → faster, less memory
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        preds, trues = [], []
        for batch in loader:
            batch = self._to_device(batch)
            out = self.model(batch)
            # Accumulate on CPU to avoid holding the whole test set on GPU/MPS.
            preds.append(out["y_hat"].cpu())
            trues.append(batch["y"].cpu())
        # Concatenate all batches, then compute every registered metric once.
        y_hat = torch.cat(preds, dim=0)
        y = torch.cat(trues, dim=0)
        return {name: fn(y_hat, y) for name, fn in GENERAL_METRICS.items()}
