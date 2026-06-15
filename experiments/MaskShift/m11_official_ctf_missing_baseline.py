"""M11 — official ChannelTokenFormer-missing baseline adaptation.

This milestone imports the official ChannelTokenFormer_missing model class and
evaluates it under the MaskShift encoder-mask protocol.  It is an
official-architecture adaptation, not a full reproduction of the CTF paper's
practical/irregular benchmark pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .maskshift_core import (
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ROOT,
    ExperimentConfig,
    ensure_dir,
    generate_mask,
    make_dataset_splits,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m11_official_ctf_missing_baseline")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
CTF_DIR = ROOT / "external" / "ChannelTokenFormer"
DATASETS = ["Weather", "Electricity"]
SEED_OFFSETS = [0, 10_000, 20_000]
BACKBONE = "ChannelTokenFormer_missing_official"


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def git_revision(path: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def load_ctf_model():
    if not CTF_DIR.exists():
        raise FileNotFoundError(f"ChannelTokenFormer not found at {CTF_DIR}")
    sys.path.insert(0, str(CTF_DIR))
    from models import ChannelTokenFormer_missing  # type: ignore

    return ChannelTokenFormer_missing.Model


def make_config(channels: int, cfg: ExperimentConfig) -> SimpleNamespace:
    patch_len = 8 if cfg.lookback % 8 == 0 else 6
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=cfg.lookback,
        label_len=cfg.horizon,
        pred_len=cfg.horizon,
        enc_in=channels,
        dec_in=channels,
        c_out=channels,
        d_model=32,
        d_ff=64,
        dropout=0.05,
        factor=3,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        activation="gelu",
        use_norm=1,
        patch_lens=[patch_len] * channels,
        sampling_periods=[1.0] * channels,
        num_global_tokens=1,
        keep_prob=1.0,
        batch_size=48,
    )


def make_model(channels: int, cfg: ExperimentConfig) -> nn.Module:
    model_cls = load_ctf_model()
    return model_cls(make_config(channels, cfg))


def build_window_tensors(
    values: np.ndarray,
    mask: np.ndarray,
    origins: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    obs_flags = []
    for origin in origins:
        x = values[origin - cfg.lookback : origin].copy()
        m = mask[origin - cfg.lookback : origin].copy()
        observed = ~m
        xs.append(np.where(observed, x, 0.0))
        ys.append(values[origin : origin + cfg.horizon])
        obs_flags.append(observed.astype(np.float32))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(obs_flags, dtype=np.float32),
    )


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, obs: np.ndarray, epochs: int = 2) -> nn.Module:
    model = model.to(DEVICE)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(obs))
    loader = DataLoader(dataset, batch_size=48, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb, mb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            mb = mb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb, None, None, None, mb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray, obs: np.ndarray) -> dict:
    model.eval()
    preds = []
    for start in range(0, len(x), 96):
        xb = torch.from_numpy(x[start : start + 96]).to(DEVICE)
        mb = torch.from_numpy(obs[start : start + 96]).to(DEVICE)
        preds.append(model(xb, None, None, None, mb).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target_pred = pred[:, -1, 0]
    target_true = y[:, -1, 0]
    target_losses = (target_pred - target_true) ** 2
    return {
        "target_mse": float(np.mean(target_losses)),
        "target_mae": float(np.mean(np.abs(target_pred - target_true))),
        "full_horizon_mse": float(np.mean((pred - y) ** 2)),
        "target_losses": target_losses.astype(float).tolist(),
    }


def run_dataset(dataset: str, cfg: ExperimentConfig, seed_offset: int) -> dict:
    torch.manual_seed(cfg.seed + seed_offset)
    np.random.seed(cfg.seed + seed_offset)
    ds = make_dataset_splits(dataset, cfg)
    values = ds["values"]
    split_idx = ds["split_idx"]
    train_origins = ds["train_origins"][: min(len(ds["train_origins"]), cfg.max_train_samples)]
    test_origins = ds["test_origins"][: min(len(ds["test_origins"]), cfg.max_test_samples)]

    train_mask = generate_mask(values, "mcar", cfg.target_rate, cfg.seed + seed_offset, split_idx=split_idx)
    x_train, y_train, obs_train = build_window_tensors(values, train_mask, train_origins, cfg)
    model = make_model(values.shape[1], cfg)
    model = train_model(model, x_train, y_train, obs_train)

    rows = []
    loss_groups = []
    for i, mechanism in enumerate(MECHANISMS):
        test_mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + seed_offset + 7000 + i * 17, split_idx=split_idx)
        x_test, y_test, obs_test = build_window_tensors(values, test_mask, test_origins, cfg)
        metrics = evaluate_model(model, x_test, y_test, obs_test)
        loss_groups.append(metrics["target_losses"])
        rows.append(
            {
                "dataset": dataset,
                "backbone": BACKBONE,
                "test_mechanism": mechanism,
                "target_mse": metrics["target_mse"],
                "target_mae": metrics["target_mae"],
                "full_horizon_mse": metrics["full_horizon_mse"],
                "n_test_windows": len(metrics["target_losses"]),
            }
        )

    mcar_mse = next(row["target_mse"] for row in rows if row["test_mechanism"] == "mcar")
    mechanism_effect_rows = []
    for mechanism in OPERATIONAL_MECHANISMS:
        mech_mse = next(row["target_mse"] for row in rows if row["test_mechanism"] == mechanism)
        mechanism_effect_rows.append(
            {
                "mechanism": mechanism,
                "mean_target_mse": float(mech_mse),
                "absolute_delta_vs_mcar": float(mech_mse - mcar_mse),
                "relative_degradation_vs_mcar": float((mech_mse - mcar_mse) / max(mcar_mse, 1e-9)),
                "log_ratio_vs_mcar": float(np.log((mech_mse + 1e-6) / (mcar_mse + 1e-6))),
            }
        )
    try:
        h_stat, p_value = stats.kruskal(*loss_groups)
    except ValueError:
        h_stat, p_value = float("nan"), float("nan")
    max_degradation = max(row["relative_degradation_vs_mcar"] for row in mechanism_effect_rows)
    max_abs_delta = max(row["absolute_delta_vs_mcar"] for row in mechanism_effect_rows)
    strongest = max(mechanism_effect_rows, key=lambda row: row["absolute_delta_vs_mcar"])["mechanism"]
    gate_pass = max_degradation > 0.05 and (np.isnan(p_value) or p_value <= 0.05)
    return {
        "dataset": dataset,
        "rows": rows,
        "mechanism_effect_rows": mechanism_effect_rows,
        "kruskal_h": float(h_stat),
        "kruskal_p": float(p_value),
        "mcar_target_mse": float(mcar_mse),
        "max_relative_degradation": float(max_degradation),
        "max_absolute_delta": float(max_abs_delta),
        "strongest_mechanism": strongest,
        "gate_pass": bool(gate_pass),
    }


def mean_ci(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(len(arr))}
    sem = stats.sem(arr)
    half = float(stats.t.ppf(0.975, df=len(arr) - 1) * sem)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(len(arr))}


def fmt_ci(summary: dict, pct: bool = False, digits: int = 2) -> str:
    if pct:
        return f"{summary['mean'] * 100:.1f}% [{summary['ci_low'] * 100:.1f}, {summary['ci_high'] * 100:.1f}]"
    return f"{summary['mean']:.{digits}f} [{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"


def write_table(summary: dict) -> None:
    lines = [
        "# M11 official ChannelTokenFormer-missing adaptation table",
        "",
        "ChannelTokenFormer_missing is imported from the official repository and evaluated under the MaskShift encoder-mask protocol. This is not the full CTF practical/irregular benchmark pipeline.",
        "",
        "| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism | Gate seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["dataset_summaries"]:
        lines.append(
            "| {dataset} | {backbone} | {deg} | {abs_delta} | {strongest} | {gate}/{n} |".format(
                dataset=row["dataset"],
                backbone=BACKBONE,
                deg=fmt_ci(row["max_relative_degradation"], pct=True),
                abs_delta=fmt_ci(row["max_absolute_delta"], digits=3),
                strongest=row["strongest_mechanism_mode"],
                gate=row["gate_pass_count"],
                n=row["n_seeds"],
            )
        )
    (TABLE_DIR / "m11_ctf_missing_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=180, max_test_samples=80)
    raw = []
    dataset_summaries = []
    for dataset_index, dataset in enumerate(DATASETS):
        seed_rows = []
        for seed_index, offset in enumerate(SEED_OFFSETS):
            row = run_dataset(dataset, cfg, seed_offset=dataset_index * 1000 + offset)
            row["seed_index"] = seed_index
            row["seed_offset"] = offset
            raw.append(row)
            seed_rows.append(row)
        mechanisms = [row["strongest_mechanism"] for row in seed_rows]
        strongest_mode = max(sorted(set(mechanisms)), key=mechanisms.count)
        dataset_summaries.append(
            {
                "dataset": dataset,
                "n_seeds": len(seed_rows),
                "gate_pass_count": int(sum(row["gate_pass"] for row in seed_rows)),
                "max_relative_degradation": mean_ci([row["max_relative_degradation"] for row in seed_rows]),
                "max_absolute_delta": mean_ci([row["max_absolute_delta"] for row in seed_rows]),
                "kruskal_p": mean_ci([row["kruskal_p"] for row in seed_rows]),
                "strongest_mechanism_mode": strongest_mode,
            }
        )

    protocol_complete = len(raw) == len(DATASETS) * len(SEED_OFFSETS) and all(
        len(row["rows"]) == len(MECHANISMS) for row in raw
    )
    mechanism_shift_gate = any(row["gate_pass_count"] >= 1 for row in dataset_summaries)
    summary = {
        "milestone": "M11",
        "status": "PASS_OFFICIAL_CTF_MISSING_ADAPTATION" if protocol_complete else "HOLD_OFFICIAL_CTF_MISSING_ADAPTATION",
        "device": str(DEVICE),
        "ctf_dir": str(CTF_DIR),
        "ctf_revision": git_revision(CTF_DIR),
        "official_model_files": [
            "external/ChannelTokenFormer/models/ChannelTokenFormer_missing.py",
            "external/ChannelTokenFormer/layers/CTF_Embed.py",
        ],
        "config": cfg.__dict__,
        "backbone": BACKBONE,
        "raw": raw,
        "dataset_summaries": dataset_summaries,
        "protocol_note": "Imports official ChannelTokenFormer_missing model class; MaskShift loop masks encoder inputs only and keeps forecast targets clean.",
        "m11_gate": bool(protocol_complete),
        "mechanism_shift_gate": bool(mechanism_shift_gate),
    }
    write_json(OUT_DIR / "ctf_missing_baseline_summary.json", summary)
    write_table(summary)
    print(json.dumps({"milestone": "M11", "status": summary["status"], "device": str(DEVICE)}, indent=2))


if __name__ == "__main__":
    main()
