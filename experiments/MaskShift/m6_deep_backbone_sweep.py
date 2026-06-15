"""M6 — Lightweight neural-backbone sweep for MaskShift.

This is a fast local proxy sweep, not a substitute for a final official
PatchTST/TimeXer/S4M reproduction.  It exists to test whether the M1 mechanism
effect survives beyond linear ridge forecasters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .maskshift_core import (
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ExperimentConfig,
    ensure_dir,
    generate_mask,
    kendall_tau_between,
    make_dataset_splits,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m6_deep_backbone_sweep")
DATASETS = ["Weather", "Electricity"]
BACKBONES = ["DLinearLite", "PatchTSTLite", "GRUDLite"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


class DLinearLite(nn.Module):
    def __init__(self, lookback: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(lookback * channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PatchTSTLite(nn.Module):
    def __init__(self, lookback: int, channels: int, patch_len: int = 8, d_model: int = 48):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len
        self.proj = nn.Linear(patch_len * channels * 2, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c2 = x.shape
        x = x[:, : self.n_patches * self.patch_len, :]
        x = x.reshape(b, self.n_patches, self.patch_len * c2)
        z = self.encoder(self.proj(x)).mean(dim=1)
        return self.head(z).squeeze(-1)


class GRUDLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gru = nn.GRU(input_size=channels * 2, hidden_size=64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def make_model(name: str, lookback: int, channels: int) -> nn.Module:
    if name == "DLinearLite":
        return DLinearLite(lookback, channels)
    if name == "PatchTSTLite":
        return PatchTSTLite(lookback, channels)
    if name == "GRUDLite":
        return GRUDLite(channels)
    raise ValueError(name)


def build_sequence_dataset(values: np.ndarray, mask: np.ndarray, origins: np.ndarray, cfg: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for origin in origins:
        x = values[origin - cfg.lookback : origin].copy()
        mw = mask[origin - cfg.lookback : origin].copy()
        obs = ~mw
        x = np.where(obs, x, 0.0)
        xs.append(np.concatenate([x, mw.astype(float)], axis=1))
        ys.append(values[origin + cfg.horizon - 1, 0])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int = 6) -> nn.Module:
    model = model.to(DEVICE)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    preds = []
    for start in range(0, len(x), 256):
        xb = torch.from_numpy(x[start : start + 256]).to(DEVICE)
        preds.append(model(xb).detach().cpu().numpy())
    pred = np.concatenate(preds)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    return {"mse": mse, "mae": mae}


def run_dataset(dataset: str, cfg: ExperimentConfig, seed_offset: int) -> dict:
    torch.manual_seed(cfg.seed + seed_offset)
    np.random.seed(cfg.seed + seed_offset)
    ds = make_dataset_splits(dataset, cfg)
    values = ds["values"]
    split_idx = ds["split_idx"]
    train_origins = ds["train_origins"][: min(len(ds["train_origins"]), 650)]
    test_origins = ds["test_origins"][: min(len(ds["test_origins"]), 260)]
    train_mask = generate_mask(values, "mcar", cfg.target_rate, cfg.seed + seed_offset, split_idx=split_idx)
    x_train, y_train = build_sequence_dataset(values, train_mask, train_origins, cfg)
    channels = values.shape[1]

    rows = []
    for backbone in BACKBONES:
        model = make_model(backbone, cfg.lookback, channels)
        model = train_model(model, x_train, y_train)
        for i, mechanism in enumerate(MECHANISMS):
            test_mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + seed_offset + 1000 + i * 17, split_idx=split_idx)
            x_test, y_test = build_sequence_dataset(values, test_mask, test_origins, cfg)
            metrics = evaluate_model(model, x_test, y_test)
            rows.append({"dataset": dataset, "backbone": backbone, "test_mechanism": mechanism, **metrics})

    ranks = {}
    for mech in MECHANISMS:
        mech_rows = [r for r in rows if r["test_mechanism"] == mech]
        ranks[mech] = [r["backbone"] for r in sorted(mech_rows, key=lambda r: r["mse"])]
    rank_taus = {mech: kendall_tau_between(ranks["mcar"], ranks[mech]) for mech in OPERATIONAL_MECHANISMS}
    mcar_mean = np.mean([r["mse"] for r in rows if r["test_mechanism"] == "mcar"])
    max_degradation = max(
        (np.mean([r["mse"] for r in rows if r["test_mechanism"] == mech]) - mcar_mean) / max(mcar_mean, 1e-9)
        for mech in OPERATIONAL_MECHANISMS
    )
    groups = [[r["mse"] for r in rows if r["test_mechanism"] == mech] for mech in MECHANISMS]
    f_stat, p_value = stats.f_oneway(*groups)
    pass_gate = any(t <= 0.5 for t in rank_taus.values()) and max_degradation > 0.05 and p_value <= 0.10
    return {
        "dataset": dataset,
        "rows": rows,
        "ranks": ranks,
        "rank_taus": rank_taus,
        "max_relative_degradation": float(max_degradation),
        "anova_p": float(p_value),
        "anova_f": float(f_stat),
        "gate_pass": bool(pass_gate),
    }


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=650, max_test_samples=260)
    datasets = [run_dataset(name, cfg, i * 1000) for i, name in enumerate(DATASETS)]
    complete = len(datasets) == len(DATASETS) and all(len({r["backbone"] for r in ds["rows"]}) == len(BACKBONES) for ds in datasets)
    gate_pass = complete and sum(ds["gate_pass"] for ds in datasets) >= 1
    summary = {
        "milestone": "M6",
        "status": "PASS_DEEP_LITE_SWEEP" if gate_pass else "HOLD_DEEP_LITE_SWEEP",
        "device": str(DEVICE),
        "config": cfg.__dict__,
        "backbones": BACKBONES,
        "datasets": datasets,
        "deep_lite_protocol_complete": bool(complete),
        "m6_gate": bool(gate_pass),
        "scope_note": "Lite proxy sweep; final submission still needs official PatchTST/TimeXer/S4M or ChannelTokenFormer-compatible reproduction.",
    }
    write_json(OUT_DIR / "deep_backbone_sweep_summary.json", summary)
    print(json.dumps({"milestone": "M6", "status": summary["status"], "device": str(DEVICE)}, indent=2))


if __name__ == "__main__":
    main()

