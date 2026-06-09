"""Experiment entrypoint: ``python -m spectre.experiments.run --config <yaml>``.

P0 wires ETT -> windowing -> DLinear -> Trainer into one green pipeline. Later
phases register more adapters/models behind the same config schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from ..data.adapters.ett import ETTAdapter
from ..data.contract import collate_samples
from ..data.windowing import WindowDataset, make_splits
from ..engine.seed import pick_device, set_seed
from ..engine.trainer import Trainer, TrainConfig
from ..models.registry import build_model

# Adapter registry: config strings resolve to concrete dataset adapters.
_ADAPTERS = {"ett": ETTAdapter}


def load_config(path: str | Path) -> dict[str, Any]:
    # Parse a YAML experiment config into a plain dict.
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    # ---- reproducibility + device ----
    set_seed(int(cfg.get("seed", 2021)))
    device = pick_device(cfg.get("device", "auto"))

    # ---- core window geometry ----
    L, H = int(cfg["lookback"]), int(cfg["horizon"])
    ratios = tuple(cfg.get("split_ratios", (0.7, 0.1, 0.2)))

    # ---- data: two-pass load to keep normalization leak-free ----
    ds_cfg = cfg["dataset"]
    # Adapter-specific kwargs (e.g. ETT's mode) live under dataset.args.
    adapter = _ADAPTERS[ds_cfg["name"]](ds_cfg["path"], **ds_cfg.get("args", {}))
    # Pass 1: load raw (no standardization) only to learn the series length T.
    raw = adapter.load(train_end=None)
    # The train boundary is derived from T and the train ratio...
    train_end = int(raw.length * ratios[0])
    # Pass 2: reload, this time standardizing with train-only statistics.
    bundle = adapter.load(train_end=train_end)

    # ---- chronological, leak-free split into window-start indices ----
    splits = make_splits(bundle.length, L, H, ratios=ratios, embargo=cfg.get("embargo"))
    bs = int(cfg.get("batch_size", 64))

    # Local factory: build a DataLoader over a split's start indices. Train
    # shuffles; val/test keep chronological order.
    def loader(starts, shuffle):
        return DataLoader(
            WindowDataset(bundle, starts, L, H),
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=collate_samples,  # turns WindowSample list → batch dict
            drop_last=False,
        )

    train_loader = loader(splits.train, shuffle=True)
    val_loader = loader(splits.val, shuffle=False)
    test_loader = loader(splits.test, shuffle=False)

    # ---- model + trainer ----
    # c_endo (number of predicted channels) comes from the bundle's spec.
    model = build_model(
        cfg["model"]["name"], L, H, bundle.spec.c_endo, **cfg["model"].get("args", {})
    )
    tcfg = TrainConfig(**cfg.get("train", {}))
    trainer = Trainer(model, tcfg, device)

    # ---- fit (with early stopping) then evaluate on the held-out test split ----
    fit_info = trainer.fit(train_loader, val_loader)
    test_metrics = trainer.evaluate(test_loader)
    # Assemble a flat, loggable result record.
    result = {
        "dataset": bundle.name,
        "model": cfg["model"]["name"],
        "c_endo": bundle.spec.c_endo,
        "lookback": L,
        "horizon": H,
        "n_train": len(splits.train),
        "n_val": len(splits.val),
        "n_test": len(splits.test),
        "best_val_mse": fit_info["best_val_mse"],
        "best_epoch": fit_info["best_epoch"],
        "test": test_metrics,
    }
    return result


def main() -> None:
    # CLI: a single required --config path.
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    result = run(load_config(args.config))
    # Pretty-print the result record.
    print("=" * 60)
    for k, v in result.items():
        print(f"{k:>14}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
