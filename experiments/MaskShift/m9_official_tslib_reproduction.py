"""M9 — TSLib official-architecture reproduction for MaskShift.

This milestone imports the pinned official Time-Series-Library model classes
instead of reimplementing PatchTST or TimeXer locally.  The training/evaluation
loop remains MaskShift-specific because the protocol masks encoder inputs only
while keeping forecast targets unmasked.
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
    kendall_tau_between,
    make_dataset_splits,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m9_official_tslib_reproduction")
TSLIB_DIR = ROOT / "external" / "TSLib"
DATASETS = ["Weather", "Electricity"]
BACKBONES = ["PatchTST_official", "TimeXer_official"]


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def load_tslib_models():
    if not TSLIB_DIR.exists():
        raise FileNotFoundError(f"TSLib not found at {TSLIB_DIR}")
    sys.path.insert(0, str(TSLIB_DIR))
    from models import PatchTST, TimeXer  # type: ignore

    return PatchTST, TimeXer


def git_revision(path: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def make_config(backbone: str, channels: int, cfg: ExperimentConfig) -> SimpleNamespace:
    common = {
        "task_name": "long_term_forecast",
        "features": "M",
        "seq_len": cfg.lookback,
        "label_len": cfg.horizon,
        "pred_len": cfg.horizon,
        "enc_in": channels,
        "dec_in": channels,
        "c_out": channels,
        "d_model": 32,
        "d_ff": 64,
        "dropout": 0.05,
        "factor": 3,
        "n_heads": 4,
        "e_layers": 1,
        "d_layers": 1,
        "activation": "gelu",
        "embed": "timeF",
        "freq": "h",
        "use_norm": True,
        "patch_len": 8,
        "moving_avg": 25,
    }
    if backbone == "TimeXer_official":
        common["patch_len"] = 8
    return SimpleNamespace(**common)


def make_model(backbone: str, channels: int, cfg: ExperimentConfig) -> nn.Module:
    PatchTST, TimeXer = load_tslib_models()
    config = make_config(backbone, channels, cfg)
    if backbone == "PatchTST_official":
        return PatchTST.Model(config, patch_len=8, stride=4)
    if backbone == "TimeXer_official":
        return TimeXer.Model(config)
    raise ValueError(backbone)


def build_window_tensors(
    values: np.ndarray,
    mask: np.ndarray,
    origins: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for origin in origins:
        x = values[origin - cfg.lookback : origin].copy()
        m = mask[origin - cfg.lookback : origin].copy()
        xs.append(np.where(~m, x, 0.0))
        ys.append(values[origin : origin + cfg.horizon])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_model(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int = 3) -> nn.Module:
    model = model.to(DEVICE)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=48, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(xb, None, None, None)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    model.eval()
    preds = []
    for start in range(0, len(x), 96):
        xb = torch.from_numpy(x[start : start + 96]).to(DEVICE)
        preds.append(model(xb, None, None, None).detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target_pred = pred[:, -1, 0]
    target_true = y[:, -1, 0]
    return {
        "target_mse": float(np.mean((target_pred - target_true) ** 2)),
        "target_mae": float(np.mean(np.abs(target_pred - target_true))),
        "full_horizon_mse": float(np.mean((pred - y) ** 2)),
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
    x_train, y_train = build_window_tensors(values, train_mask, train_origins, cfg)
    rows = []
    for backbone in BACKBONES:
        torch.manual_seed(cfg.seed + seed_offset + len(backbone))
        model = make_model(backbone, values.shape[1], cfg)
        model = train_model(model, x_train, y_train)
        for i, mechanism in enumerate(MECHANISMS):
            test_mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + seed_offset + 3000 + i * 17, split_idx=split_idx)
            x_test, y_test = build_window_tensors(values, test_mask, test_origins, cfg)
            metrics = evaluate_model(model, x_test, y_test)
            rows.append({"dataset": dataset, "backbone": backbone, "test_mechanism": mechanism, **metrics})

    ranks = {}
    for mech in MECHANISMS:
        mech_rows = [r for r in rows if r["test_mechanism"] == mech]
        ranks[mech] = [r["backbone"] for r in sorted(mech_rows, key=lambda r: r["target_mse"])]
    rank_taus = {mech: kendall_tau_between(ranks["mcar"], ranks[mech]) for mech in OPERATIONAL_MECHANISMS}
    mcar_mean = np.mean([r["target_mse"] for r in rows if r["test_mechanism"] == "mcar"])
    mechanism_effect_rows = []
    for mech in OPERATIONAL_MECHANISMS:
        mech_mse = np.mean([r["target_mse"] for r in rows if r["test_mechanism"] == mech])
        mechanism_effect_rows.append(
            {
                "mechanism": mech,
                "mean_target_mse": float(mech_mse),
                "relative_degradation_vs_mcar": float((mech_mse - mcar_mean) / max(mcar_mean, 1e-9)),
                "kendall_tau_vs_mcar_rank": rank_taus[mech],
            }
        )
    groups = [[r["target_mse"] for r in rows if r["test_mechanism"] == mech] for mech in MECHANISMS]
    f_stat, p_value = stats.f_oneway(*groups)
    max_degradation = max(row["relative_degradation_vs_mcar"] for row in mechanism_effect_rows)
    worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in mechanism_effect_rows)
    gate_pass = max_degradation > 0.05 and worst_tau <= 0.5
    return {
        "dataset": dataset,
        "rows": rows,
        "ranks": ranks,
        "rank_taus": rank_taus,
        "mechanism_effect_rows": mechanism_effect_rows,
        "anova_f": float(f_stat),
        "anova_p": float(p_value),
        "max_relative_degradation": float(max_degradation),
        "worst_rank_tau": float(worst_tau),
        "gate_pass": bool(gate_pass),
    }


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=360, max_test_samples=160)
    datasets = [run_dataset(name, cfg, i * 1000) for i, name in enumerate(DATASETS)]
    complete = len(datasets) == len(DATASETS) and all(
        len({row["backbone"] for row in ds["rows"]}) == len(BACKBONES) for ds in datasets
    )
    gate = complete and sum(ds["gate_pass"] for ds in datasets) >= 1
    summary = {
        "milestone": "M9",
        "status": "PASS_OFFICIAL_TSLIB_REPRODUCTION" if gate else "HOLD_OFFICIAL_TSLIB_REPRODUCTION",
        "device": str(DEVICE),
        "tslib_dir": str(TSLIB_DIR),
        "tslib_revision": git_revision(TSLIB_DIR),
        "official_model_files": [
            "external/TSLib/models/PatchTST.py",
            "external/TSLib/models/TimeXer.py",
        ],
        "config": cfg.__dict__,
        "backbones": BACKBONES,
        "datasets": datasets,
        "protocol_note": "Imports pinned TSLib official model classes; MaskShift loop masks only encoder inputs and keeps forecast targets clean.",
        "m9_gate": bool(gate),
    }
    write_json(OUT_DIR / "official_tslib_reproduction_summary.json", summary)

    lines = ["# MaskShift M9 — Official TSLib Architecture Reproduction", ""]
    lines.append(
        "M9 imports official PatchTST and TimeXer model classes from the pinned `external/TSLib` checkout and evaluates them under the MaskShift encoder-mask protocol."
    )
    lines.append("")
    lines.append("| Dataset | Max degradation | Worst tau | ANOVA p | Gate |")
    lines.append("|---|---:|---:|---:|---|")
    for ds in datasets:
        lines.append(
            f"| {ds['dataset']} | {ds['max_relative_degradation']:.1%} | {ds['worst_rank_tau']:.3f} | {ds['anova_p']:.3g} | {'PASS' if ds['gate_pass'] else 'FAIL'} |"
        )
    (EXP_DIR / "REPORT.md").write_text((EXP_DIR / "REPORT.md").read_text() + "\n\n" + "\n".join(lines) + "\n")
    print(json.dumps({"milestone": "M9", "status": summary["status"], "device": str(DEVICE)}, indent=2))


if __name__ == "__main__":
    main()
