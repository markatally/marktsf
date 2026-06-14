"""M6 — Deep-backbone D0/D1/D2 sweep.

This trains lightweight TSLib backbones on the DoCast semi-synthetic sample
shape under three heads. All three arms receive the same static item controls
in the backbone input; D1 and D2 also share item-specific scalar response
capacity. This keeps the M6 comparison focused on the orthogonalized objective
rather than an extra-control advantage.

D0: observational backbone + unconstrained horizon-wise treatment head.
D1: structural item-specific scalar treatment head without orthogonalization.
D2: DoCast-style R-learner using backbone nuisances m(V) and pi(V).

The goal is not to tune SOTA accuracy; it is to verify that the causal protocol
survives non-linear TSF backbones and that the full D0/D1/D2 pathway is runnable.
"""

from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
TSLIB = ROOT / "external" / "TSLib"
sys.path.insert(0, str(TSLIB))

import torch
from torch.utils.data import DataLoader, TensorDataset

from m2_docast import load_m5, generate_synthetic


OUT_DIR = Path(__file__).parent / "m6_backbone_sweep"
OUT_DIR.mkdir(exist_ok=True)

L = 56
H = 28
N_ITEMS = 24
TRAIN_ORIGINS = 220
TEST_ORIGINS = 40
GAMMA = 0.5
SEED = 2021


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def time_features(T: int, month: np.ndarray, wday: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.sin(2 * math.pi * month / 12),
        np.cos(2 * math.pi * month / 12),
        np.sin(2 * math.pi * wday / 7),
    ]).astype(np.float32)


def build_tensor_data(data: dict, gamma: float, seed: int) -> dict:
    base = {
        "Y": data["Y"][:N_ITEMS],
        "P": data["P"][:N_ITEMS],
        "snap": data["snap"],
        "month": data["month"],
        "wday": data["wday"],
        "n": N_ITEMS,
        "T": data["T"],
    }
    syn = generate_synthetic(base, gamma=gamma, seed=seed)
    log_y = np.log(syn["Y_syn"] + 1.0).astype(np.float32)
    phi = syn["log_pr"].astype(np.float32)
    theta_star = syn["theta_star"][:N_ITEMS].astype(np.float32)
    tf = time_features(base["T"], base["month"], base["wday"])

    train_end = min(L + TRAIN_ORIGINS, base["T"] - H - TEST_ORIGINS)
    test_start = train_end
    test_end = min(test_start + TEST_ORIGINS, base["T"] - H)

    def make_split(t0: int, t1: int):
        x_list, xm_list, xdm_list, y_list, phi_future_list, item_list = [], [], [], [], [], []
        for i in range(N_ITEMS):
            for t in range(max(L, t0), t1):
                y_lag = log_y[i, t - L:t]
                phi_lag = phi[i, t - L:t]
                snap_lag = np.repeat(base["snap"][t], L).astype(np.float32)
                x = np.column_stack([y_lag, phi_lag, snap_lag]).astype(np.float32)
                x_mark = tf[t - L:t]
                future_tf = tf[t:t + H]
                future_phi = phi[i, t:t + H]
                # Future mark includes calendar features plus the planned treatment.
                x_dec_mark = np.column_stack([future_tf, future_phi]).astype(np.float32)
                y = log_y[i, t:t + H]
                x_list.append(x)
                xm_list.append(x_mark)
                xdm_list.append(x_dec_mark)
                y_list.append(y)
                phi_future_list.append(future_phi)
                item_list.append(i)
        return {
            "x": torch.tensor(np.stack(x_list), dtype=torch.float32),
            "x_mark": torch.tensor(np.stack(xm_list), dtype=torch.float32),
            "x_dec_mark": torch.tensor(np.stack(xdm_list), dtype=torch.float32),
            "y": torch.tensor(np.stack(y_list), dtype=torch.float32),
            "phi_future": torch.tensor(np.stack(phi_future_list), dtype=torch.float32),
            "item_idx": torch.tensor(item_list, dtype=torch.long),
        }

    return {
        "train": make_split(L, train_end),
        "test": make_split(test_start, test_end),
        "theta_star": theta_star,
        "theta_star_mean": float(theta_star.mean()),
        "theta_star_ser_sign": -1,
    }


def model_config(name: str, enc_in: int = 3) -> Namespace:
    return Namespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=L,
        label_len=H,
        pred_len=H,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        factor=1,
        dropout=0.05,
        embed="timeF",
        freq="d",
        activation="gelu",
        use_norm=1,
        patch_len=8,
        individual=False,
        moving_avg=25,
    )


def load_model(name: str, enc_in: int = 3):
    mod = __import__(f"models.{name}", fromlist=["Model"])
    return mod.Model(model_config(name, enc_in=enc_in))


def forward_model(model, name: str, x, x_mark, x_dec_mark):
    b = x.shape[0]
    x_dec = torch.zeros((b, H, x.shape[-1]), device=x.device)
    if name == "TiDE":
        # TiDE expects future temporal features in batch_y_mark.
        y_mark = x_dec_mark[:, :, :3]
        out = model(x, x_mark, x_dec, y_mark)
    else:
        out = model(x, x_mark, x_dec, x_dec_mark[:, :, :3])
    return out[:, :, 0]


def wmape_log(pred: torch.Tensor, y: torch.Tensor) -> float:
    yh = torch.exp(pred) - 1.0
    yt = torch.exp(y) - 1.0
    return float((torch.abs(yh - yt).sum() / (yt.sum() + 1e-8)).detach().cpu())


def make_loader(split: dict, batch_size: int = 64, device: torch.device | None = None, include_item: bool = False):
    tensors = [split["x"], split["x_mark"], split["x_dec_mark"], split["y"], split["phi_future"]]
    if include_item:
        tensors.append(split["item_idx"])
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device is not None and device.type == "cuda"))


def zero_treatment_channel(x: torch.Tensor) -> torch.Tensor:
    x_v = x.clone()
    x_v[:, :, 1] = 0.0
    return x_v


def add_item_static_channels(split: dict, n_items: int = N_ITEMS) -> dict:
    """Append item one-hot channels to the observed V representation."""
    item_oh = torch.nn.functional.one_hot(split["item_idx"], num_classes=n_items).float()
    item_seq = item_oh[:, None, :].repeat(1, split["x"].shape[1], 1)
    out = dict(split)
    out["x"] = torch.cat([split["x"], item_seq], dim=-1)
    return out


def theta_rmse(theta_hat, theta_star: np.ndarray) -> float:
    if np.isscalar(theta_hat):
        theta_arr = np.full_like(theta_star, float(theta_hat), dtype=np.float32)
    else:
        theta_arr = np.asarray(theta_hat, dtype=np.float32)
    return float(np.sqrt(np.mean((theta_arr - theta_star) ** 2)))


def train_d0_or_d1(name: str, tensors: dict, device: torch.device, response_mode: str) -> dict:
    torch.manual_seed(SEED)
    train = tensors["train"]
    test = tensors["test"]
    model = load_model(name, enc_in=train["x"].shape[-1]).to(device)

    if response_mode == "item":
        theta = torch.nn.Parameter(torch.zeros(N_ITEMS, device=device))
        include_item = True
    elif response_mode == "horizon":
        theta = torch.nn.Parameter(torch.zeros(H, device=device))
        include_item = False
    else:
        raise ValueError(f"unknown response_mode={response_mode}")

    loader = make_loader(train, device=device, include_item=include_item)
    opt = torch.optim.AdamW(list(model.parameters()) + [theta], lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    model.train()
    losses = []
    for _epoch in range(3):
        epoch_losses = []
        for batch in loader:
            if include_item:
                xb, xmb, xdmb, yb, phib, itemb = batch
                itemb = itemb.to(device)
            else:
                xb, xmb, xdmb, yb, phib = batch
            xb = xb.to(device)
            xmb = xmb.to(device)
            xdmb = xdmb.to(device)
            yb = yb.to(device)
            phib = phib.to(device)
            opt.zero_grad(set_to_none=True)
            base = forward_model(model, name, xb, xmb, xdmb)
            if response_mode == "item":
                pred = base + phib * theta[itemb][:, None]
            else:
                pred = base + phib * theta
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        x = test["x"].to(device)
        xm = test["x_mark"].to(device)
        xdm = test["x_dec_mark"].to(device)
        y = test["y"].to(device)
        phi = test["phi_future"].to(device)
        if response_mode == "item":
            theta_eval = theta[test["item_idx"].to(device)][:, None]
            pred = forward_model(model, name, x, xm, xdm) + phi * theta_eval
            theta_for_rmse = theta.detach().cpu().numpy()
        else:
            pred = forward_model(model, name, x, xm, xdm) + phi * theta
            theta_for_rmse = float(theta.mean().detach().cpu())
        theta_hat = float(np.mean(theta.detach().cpu().numpy()))
        wmape = wmape_log(pred, y)

    if response_mode == "item":
        ser = float(np.mean(np.sign(theta_for_rmse) != np.sign(tensors["theta_star"])))
    else:
        ser = float(np.sign(theta_hat) != np.sign(tensors["theta_star_mean"]))
    rmse = theta_rmse(theta_for_rmse, tensors["theta_star"])
    return {
        "status": "complete",
        "response_mode": response_mode,
        "train_loss_first": round(losses[0], 5),
        "train_loss_last": round(losses[-1], 5),
        "obs_wmape": round(wmape, 4),
        "theta_hat": round(theta_hat, 4),
        "theta_star_mean": round(tensors["theta_star_mean"], 4),
        "theta_abs_error": round(rmse, 4),
        "ser": ser,
    }


def train_backbone_target(name: str, split: dict, device: torch.device, target: str, epochs: int = 5):
    torch.manual_seed(SEED)
    model = load_model(name, enc_in=split["x"].shape[-1]).to(device)
    loader = make_loader(split, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _epoch in range(epochs):
        for xb, xmb, xdmb, yb, phib in loader:
            xb = zero_treatment_channel(xb).to(device)
            xmb = xmb.to(device)
            xdmb = xdmb.to(device)
            y_target = (yb if target == "y" else phib).to(device)
            opt.zero_grad(set_to_none=True)
            pred = forward_model(model, name, xb, xmb, xdmb)
            loss = loss_fn(pred, y_target)
            loss.backward()
            opt.step()
    return model


def item_fe_adjust(train_pred: torch.Tensor, test_pred: torch.Tensor, train: dict, test: dict, target_key: str):
    """Calibrate deep nuisance residuals with static item fixed effects."""
    return item_fe_adjust_to_target(train_pred, test_pred, train, test, train[target_key])


def item_fe_adjust_to_target(
    train_pred: torch.Tensor,
    test_pred: torch.Tensor,
    train: dict,
    test: dict,
    train_target: torch.Tensor,
):
    """Calibrate arbitrary deep predictions to a train target using item fixed effects."""
    from sklearn.linear_model import Ridge

    y_train = train_target.numpy()
    train_raw = train_pred.numpy()
    test_raw = test_pred.numpy()
    n_items = int(max(train["item_idx"].max(), test["item_idx"].max()).item()) + 1
    fe_train = torch.nn.functional.one_hot(train["item_idx"], num_classes=n_items).numpy()
    fe_test = torch.nn.functional.one_hot(test["item_idx"], num_classes=n_items).numpy()
    reg = Ridge(alpha=1.0)
    reg.fit(fe_train, y_train - train_raw)
    train_adj = train_raw + reg.predict(fe_train)
    test_adj = test_raw + reg.predict(fe_test)
    raw_mse = float(np.mean((train_raw - y_train) ** 2))
    adj_mse = float(np.mean((train_adj - y_train) ** 2))
    return (
        torch.tensor(train_adj, dtype=torch.float32),
        torch.tensor(test_adj, dtype=torch.float32),
        {"raw_mse": round(raw_mse, 5), "item_fe_adjusted_mse": round(adj_mse, 5)},
    )


def v_feature_matrix(split: dict) -> np.ndarray:
    x = zero_treatment_channel(split["x"])
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    last = x[:, -1, :]
    return torch.cat([mean, std, last], dim=1).numpy()


def ridge_adjust_to_target(
    train_pred: torch.Tensor,
    test_pred: torch.Tensor,
    train: dict,
    test: dict,
    train_target: torch.Tensor,
):
    """Calibrate a nuisance with a lightweight V-only residual model."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = train_target.numpy()
    train_raw = train_pred.numpy()
    test_raw = test_pred.numpy()
    x_train = v_feature_matrix(train)
    x_test = v_feature_matrix(test)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    reg.fit(x_train, y_train - train_raw)
    train_adj = train_raw + reg.predict(x_train)
    test_adj = test_raw + reg.predict(x_test)
    raw_mse = float(np.mean((train_raw - y_train) ** 2))
    adj_mse = float(np.mean((train_adj - y_train) ** 2))
    return (
        torch.tensor(train_adj, dtype=torch.float32),
        torch.tensor(test_adj, dtype=torch.float32),
        {"raw_mse": round(raw_mse, 5), "v_ridge_adjusted_mse": round(adj_mse, 5)},
    )


def batched_predict(model, name: str, split: dict, device: torch.device, v_only: bool) -> torch.Tensor:
    preds = []
    ds = TensorDataset(split["x"], split["x_mark"], split["x_dec_mark"])
    loader = DataLoader(ds, batch_size=128, shuffle=False, pin_memory=(device.type == "cuda"))
    model.eval()
    with torch.no_grad():
        for xb, xmb, xdmb in loader:
            if v_only:
                xb = zero_treatment_channel(xb)
            preds.append(forward_model(model, name, xb.to(device), xmb.to(device), xdmb.to(device)).cpu())
    return torch.cat(preds, dim=0)


def train_d2(name: str, tensors: dict, device: torch.device) -> dict:
    train = add_item_static_channels(tensors["train"])
    test = add_item_static_channels(tensors["test"])

    m_model = train_backbone_target(name, train, device, target="y")
    pi_model = train_backbone_target(name, train, device, target="phi")

    m_tr_raw = batched_predict(m_model, name, train, device, v_only=True)
    m_te_raw = batched_predict(m_model, name, test, device, v_only=True)
    pi_tr_raw = batched_predict(pi_model, name, train, device, v_only=True)
    pi_te_raw = batched_predict(pi_model, name, test, device, v_only=True)

    m_tr, m_te, m_diag = item_fe_adjust(m_tr_raw, m_te_raw, train, test, target_key="y")
    pi_tr, pi_te, pi_diag = item_fe_adjust(pi_tr_raw, pi_te_raw, train, test, target_key="phi_future")

    y_tr = train["y"]
    phi_tr = train["phi_future"]
    y_res = y_tr - m_tr
    phi_res = phi_tr - pi_tr
    theta_global = float((phi_res * y_res).sum().item() / ((phi_res * phi_res).sum().item() + 1e-8))

    item_theta = np.full(N_ITEMS, theta_global, dtype=np.float32)
    item_idx = train["item_idx"].numpy()
    phi_res_np = phi_res.numpy()
    y_res_np = y_res.numpy()
    for item in range(N_ITEMS):
        mask = item_idx == item
        denom = float(np.sum(phi_res_np[mask] * phi_res_np[mask]))
        if denom > 1e-6:
            item_theta[item] = float(np.sum(phi_res_np[mask] * y_res_np[mask]) / denom)

    # Light shrinkage keeps item-level response estimates stable while still
    # allowing the structural head to represent heterogeneous effects.
    theta = 0.8 * item_theta + 0.2 * theta_global

    # Equivalent structural form to m(V) + theta * (phi - pi(V)), but more
    # stable for finite-sample forecasting because the final base learns
    # mu(V) = E[y - theta * phi | V] directly instead of subtracting two
    # separately estimated nuisances at test time.
    theta_tr = torch.tensor(theta[train["item_idx"].numpy()], dtype=torch.float32)[:, None]
    theta_te = torch.tensor(theta[test["item_idx"].numpy()], dtype=torch.float32)[:, None]
    mu_tr_raw = m_tr_raw - theta_tr * pi_tr_raw
    mu_te_raw = m_te_raw - theta_te * pi_te_raw
    mu_target = train["y"] - theta_tr * train["phi_future"]
    _, mu_te, mu_diag = ridge_adjust_to_target(mu_tr_raw, mu_te_raw, train, test, mu_target)
    pred = mu_te + theta_te * test["phi_future"]
    wmape = wmape_log(pred, test["y"])
    theta_mean = float(np.mean(theta))
    ser = float(np.mean(np.sign(theta) != np.sign(tensors["theta_star"])))
    rmse = theta_rmse(theta, tensors["theta_star"])
    return {
        "status": "complete",
        "obs_wmape": round(wmape, 4),
        "theta_hat": round(theta_mean, 4),
        "theta_star_mean": round(tensors["theta_star_mean"], 4),
        "theta_abs_error": round(rmse, 4),
        "ser": ser,
        "nuisance_diagnostics": {
            "m": m_diag,
            "pi": pi_diag,
            "mu": mu_diag,
            "residual_phi_variance": round(float(torch.var(phi_res).item()), 5),
            "theta_global": round(theta_global, 4),
            "theta_item_min": round(float(np.min(theta)), 4),
            "theta_item_max": round(float(np.max(theta)), 4),
        },
    }


def run_protocol(name: str, tensors: dict, device: torch.device) -> dict:
    fair_tensors = {
        **tensors,
        "train": add_item_static_channels(tensors["train"]),
        "test": add_item_static_channels(tensors["test"]),
    }
    d0 = train_d0_or_d1(name, fair_tensors, device, response_mode="horizon")
    d1 = train_d0_or_d1(name, fair_tensors, device, response_mode="item")
    d2 = train_d2(name, tensors, device)
    d2_vs_d0_ser_reduction = (d0["ser"] - d2["ser"]) / (d0["ser"] + 1e-8) if d0["ser"] > 0 else 0.0
    d2_vs_d0_theta_error_reduction = (d0["theta_abs_error"] - d2["theta_abs_error"]) / (d0["theta_abs_error"] + 1e-8)
    d2_vs_d1_theta_error_reduction = (d1["theta_abs_error"] - d2["theta_abs_error"]) / (d1["theta_abs_error"] + 1e-8)
    obs_loss_increase = (d2["obs_wmape"] - d0["obs_wmape"]) / (d0["obs_wmape"] + 1e-8)
    return {
        "backbone": name,
        "status": "complete",
        "fairness": "D0/D1/D2 share item static controls; D1/D2 share item-specific response capacity",
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "d2_vs_d0_ser_reduction": round(float(d2_vs_d0_ser_reduction), 4),
        "d2_vs_d0_theta_error_reduction": round(float(d2_vs_d0_theta_error_reduction), 4),
        "d2_vs_d1_theta_error_reduction": round(float(d2_vs_d1_theta_error_reduction), 4),
        "d2_vs_d0_obs_loss_increase": round(float(obs_loss_increase), 4),
        "protocol_pass": bool(
            d2["theta_abs_error"] <= d1["theta_abs_error"]
            and obs_loss_increase <= 0.05
        ),
    }


def main() -> None:
    device = select_device()
    print(f"Using device: {device}")
    data = load_m5()
    tensors = build_tensor_data(data, gamma=GAMMA, seed=SEED)
    rows = []
    for name in ["DLinear", "PatchTST", "TiDE", "TimeXer"]:
        print(f"Running D0/D1/D2 protocol for {name}...")
        try:
            rows.append(run_protocol(name, tensors, device))
        except Exception as exc:
            rows.append({"backbone": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    complete = [r for r in rows if r.get("status") == "complete"]
    deep_complete = [r for r in complete if r["backbone"] in {"PatchTST", "TiDE", "TimeXer"}]
    deep_protocol_pass = [r for r in deep_complete if r.get("protocol_pass")]
    summary = {
        "milestone": "M6",
        "status": "complete" if complete else "failed",
        "scope": "lightweight deep-backbone D0/D1/D2 protocol on DoCast semi-synthetic sample",
        "device": str(device),
        "gamma": GAMMA,
        "seed": SEED,
        "n_backbones_complete": len(complete),
        "n_deep_backbones_complete": len(deep_complete),
        "n_deep_protocol_pass": len(deep_protocol_pass),
        "full_docast_protocol_complete": bool(len(deep_complete) >= 3 and len(deep_protocol_pass) >= 3),
        "rows": rows,
    }
    out = OUT_DIR / "backbone_sweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"M6 backbone sweep → {out}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
