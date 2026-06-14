"""M3 — Real-data experiments.

Milestone M3 covers:
  - M5 SNAP natural experiment: exogenous c-type sanity check.
  - Favorita promotion NEE: real a-type validation against a matched within-unit ATT.
  - Semi-synthetic PRF: decision stress test, explicitly not real-data evidence.
  - BH-FDR significance pass over the real-data validation legs.

Gate (H5): Favorita a-type promotion NEE(D2) < NEE(D0), with paired
unit-level evidence. SNAP is a non-degradation check because SNAP is c-type.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parents[2]
M5_DIR = ROOT / "input" / "M5" / "m5" / "datasets"
FAV_DIR = ROOT / "input" / "Favorita"
OUT_DIR = Path(__file__).parent / "m3_real_data"
OUT_DIR.mkdir(exist_ok=True)

N_ITEMS = 40
H = 28
L = 56
SEEDS = [2021, 2022, 2023]

# ── DATA LOAD ─────────────────────────────────────────────────────────────────

def load_m5_all_stores() -> dict:
    """Load FOODS_1 across all stores for cross-store SNAP DiD."""
    cal = pd.read_csv(M5_DIR / "calendar.csv")
    sales = pd.read_csv(M5_DIR / "sales_train_evaluation.csv")
    mask = sales["dept_id"] == "FOODS_1"
    sub = sales[mask].reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    T_START, T_END = 200, 1941
    Y = sub[day_cols].values.astype(np.float64)[:, T_START:T_END]
    snap_CA = cal["snap_CA"].values[T_START:T_END].astype(np.float64)
    snap_TX = cal["snap_TX"].values[T_START:T_END].astype(np.float64)
    snap_WI = cal["snap_WI"].values[T_START:T_END].astype(np.float64)
    state_ids = sub["state_id"].tolist()
    store_ids = sub["store_id"].tolist()
    item_ids = sub["item_id"].tolist()

    wk = cal["wm_yr_wk"].values[T_START:T_END]
    prices_raw = pd.read_csv(M5_DIR / "sell_prices.csv")
    # Limit to FOODS_1 items
    prices_sub = prices_raw[prices_raw["item_id"].isin(sub["item_id"].unique())].copy()
    prices_sub = prices_sub.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"]
    n = len(sub)
    T = Y.shape[1]
    P = np.full((n, T), np.nan)
    for i in range(n):
        store, item = store_ids[i], item_ids[i]
        for t, w in enumerate(wk):
            key = (store, item, w)
            if key in prices_sub.index:
                P[i, t] = prices_sub[key]
        for t in range(1, T):
            if np.isnan(P[i, t]):
                P[i, t] = P[i, t - 1]
        med = np.nanmedian(P[i])
        P[i] = np.where(np.isnan(P[i]), med, P[i])

    month = cal["month"].values[T_START:T_END].astype(np.float64)
    wday = cal["wday"].values[T_START:T_END].astype(np.float64)

    # Build per-item SNAP flag based on state
    SNAP = np.zeros((n, T))
    for i, state in enumerate(state_ids):
        if state == "CA":
            SNAP[i] = snap_CA
        elif state == "TX":
            SNAP[i] = snap_TX
        elif state == "WI":
            SNAP[i] = snap_WI

    return {"Y": Y, "P": P, "SNAP": SNAP, "snap_CA": snap_CA, "snap_TX": snap_TX,
            "snap_WI": snap_WI, "month": month, "wday": wday,
            "state_ids": state_ids, "store_ids": store_ids, "item_ids": item_ids, "n": n, "T": T}


# ── M5 SNAP NATURAL EXPERIMENT ────────────────────────────────────────────────

def compute_snap_did(data: dict) -> dict:
    """
    Difference-in-Differences estimate of SNAP uplift using cross-state calendar variation.

    DiD design: CA and TX have different SNAP day allocations per month.
    Control group: item-days where neither CA nor TX has SNAP.
    Treated group: item-days where CA (or TX) has SNAP but not the other.
    Cross-state DiD eliminates common seasonality effects.
    """
    Y = data["Y"]
    SNAP = data["SNAP"]
    state_ids = data["state_ids"]
    snap_CA = data["snap_CA"]
    snap_TX = data["snap_TX"]
    n, T = Y.shape

    log_Y = np.log(Y + 1.0)

    # DiD per item: compare log_Y on SNAP days vs non-SNAP days
    # Control for day-of-week effects by within-weekday comparison
    wday_arr = data["wday"]

    did_estimates = []
    for i in range(n):
        snap_i = SNAP[i]
        y_i = log_Y[i]
        # Simple DiD: SNAP on vs SNAP off
        on_days = snap_i == 1
        off_days = snap_i == 0
        if on_days.sum() < 10 or off_days.sum() < 10:
            continue
        # Within-wday correction
        uplift_per_wday = []
        for wd in range(1, 8):
            wday_mask = wday_arr == wd
            on_wd = on_days & wday_mask
            off_wd = off_days & wday_mask
            if on_wd.sum() < 3 or off_wd.sum() < 3:
                continue
            uplift_per_wday.append(y_i[on_wd].mean() - y_i[off_wd].mean())
        if uplift_per_wday:
            did_estimates.append(np.mean(uplift_per_wday))

    did_mean = float(np.mean(did_estimates))
    did_std = float(np.std(did_estimates))
    did_n = len(did_estimates)
    did_se = did_std / math.sqrt(did_n)
    t_stat = did_mean / (did_se + 1e-8)
    p_val = 2 * (1 - st.norm.cdf(abs(t_stat)))

    return {
        "did_effect": round(did_mean, 4),
        "did_std": round(did_std, 4),
        "did_n_items": did_n,
        "did_t_stat": round(t_stat, 3),
        "did_p_value": round(p_val, 4),
        "did_significant": bool(p_val < 0.05),
        "interpretation": (
            "DiD estimate of log-sales uplift on SNAP days vs non-SNAP days, "
            "within-weekday controlled. Positive = SNAP days have higher sales."
        ),
    }


def compute_model_implied_snap_effect(
    data: dict, use_ortho: bool, seed: int
) -> dict:
    """
    Train D0 or D2 on the data and extract implied SNAP coefficient.
    D0: SNAP enters as ordinary feature → its coefficient is the implied effect.
    D2: SNAP is typed as exogenous c (not a), so it enters V_t directly;
        orthogonalization removes policy (price) confounding but SNAP itself
        is the treatment we measure here — so we report D0 SNAP coefficient as the
        model-implied effect both ways (SNAP is c-type, not a-type).
    For NEE: use D0 SNAP coefficient and D2 SNAP coefficient from two separate OLS regressions.
    """
    rng = np.random.default_rng(seed)
    Y = data["Y"]
    P = data["P"]
    SNAP = data["SNAP"]
    month = data["month"]
    wday = data["wday"]
    n, T = Y.shape

    log_Y = np.log(Y + 1.0)
    log_P_ref = np.log(np.nanmedian(P, axis=1, keepdims=True) + 1e-6)
    log_pr = np.log(P + 1e-6) - log_P_ref  # log price ratio (n, T)

    # Subset to first N_ITEMS items for speed
    n_use = min(N_ITEMS, n)
    log_Y = log_Y[:n_use]
    log_pr = log_pr[:n_use]
    SNAP_use = SNAP[:n_use]

    t_arr = np.arange(T).astype(float) / T
    sin_m = np.sin(2 * math.pi * month / 12)
    cos_m = np.cos(2 * math.pi * month / 12)
    sin_w = np.sin(2 * math.pi * wday / 7)
    cos_w = np.cos(2 * math.pi * wday / 7)

    train_end = T - H * 2

    snap_coefs = []
    for i in range(n_use):
        X_list, y_list = [], []
        for t in range(L, train_end):
            if t + H > T:
                break
            lag = log_Y[i, t - L:t]
            ctx = np.array([
                t_arr[t], sin_m[t], cos_m[t], sin_w[t], cos_w[t],
                SNAP_use[i, t], log_pr[i, t]
            ])
            feats = np.concatenate([lag, ctx])
            X_list.append(feats)
            y_list.append(log_Y[i, t:t + H].mean())
        if len(X_list) < 20:
            continue
        X = np.array(X_list)
        y_arr = np.array(y_list)

        sc = StandardScaler()
        X_s = sc.fit_transform(X)

        if use_ortho:
            # D2: residualize price first, then fit SNAP coefficient
            # Price is last column; SNAP is column L+5
            price_col = -1
            snap_col = L + 5
            # Stage 1: predict price from all other features
            X_no_price = np.delete(X_s, price_col, axis=1)
            reg_pi = Ridge(alpha=1.0)
            reg_pi.fit(X_no_price, X_s[:, price_col])
            price_resid = X_s[:, price_col] - reg_pi.predict(X_no_price)
            # Stage 1: predict y from features without price
            reg_m = Ridge(alpha=1.0)
            reg_m.fit(X_no_price, y_arr)
            y_resid = y_arr - reg_m.predict(X_no_price)
            # Stage 2: regress y_resid on price_resid and SNAP
            X2 = np.column_stack([price_resid, X_s[:, snap_col]])
            reg2 = Ridge(alpha=0.1)
            reg2.fit(X2, y_resid)
            # SNAP coefficient (column 1, unscaled)
            snap_coef_scaled = reg2.coef_[1]
            snap_scale = sc.scale_[snap_col]
            snap_coefs.append(snap_coef_scaled / snap_scale)
        else:
            # D0: plain OLS, SNAP coefficient
            reg = Ridge(alpha=1.0)
            reg.fit(X_s, y_arr)
            snap_col = L + 5
            snap_coef_scaled = reg.coef_[snap_col]
            snap_scale = sc.scale_[snap_col]
            snap_coefs.append(snap_coef_scaled / snap_scale)

    model_effect = float(np.mean(snap_coefs)) if snap_coefs else float("nan")
    return {"model_implied_snap_effect": round(model_effect, 4), "n_items": len(snap_coefs)}


def m5_snap_nee(data: dict) -> dict:
    did = compute_snap_did(data)
    did_effect = did["did_effect"]

    d0_effs, d2_effs = [], []
    for seed in SEEDS:
        res_d0 = compute_model_implied_snap_effect(data, use_ortho=False, seed=seed)
        res_d2 = compute_model_implied_snap_effect(data, use_ortho=True, seed=seed)
        d0_effs.append(res_d0["model_implied_snap_effect"])
        d2_effs.append(res_d2["model_implied_snap_effect"])

    d0_nee = float(np.mean([abs(e - did_effect) for e in d0_effs]))
    d2_nee = float(np.mean([abs(e - did_effect) for e in d2_effs]))

    # Paired t-test: is D2 NEE significantly smaller?
    d0_nee_per = [abs(e - did_effect) for e in d0_effs]
    d2_nee_per = [abs(e - did_effect) for e in d2_effs]
    t, p = st.ttest_rel(d0_nee_per, d2_nee_per, alternative="greater")

    h5_m5_pass = d2_nee < d0_nee and p < 0.10  # relaxed α for 3-seed comparison

    return {
        "did_estimate": did,
        "d0_nee_mean": round(d0_nee, 4),
        "d2_nee_mean": round(d2_nee, 4),
        "nee_reduction_frac": round((d0_nee - d2_nee) / (d0_nee + 1e-8), 4),
        "paired_t_stat": round(float(t), 3),
        "paired_p_value": round(float(p), 4),
        "h5_m5_pass": bool(h5_m5_pass),
        "d0_effects_per_seed": [round(float(e), 4) for e in d0_effs],
        "d2_effects_per_seed": [round(float(e), 4) for e in d2_effs],
    }


# ── M5 MARKDOWN REAL A-TYPE VALIDATION ───────────────────────────────────────

def compute_m5_markdown_nee(data: dict, unit_cap: int = 800, threshold: float = 0.05) -> dict:
    """Real M5 price/markdown validation using a binary discount-depth treatment."""
    Y = data["Y"]
    P = data["P"]
    units = [f"{s}_{it}" for s, it in zip(data["store_ids"], data["item_ids"])]
    n, T = Y.shape
    med = np.nanmedian(P, axis=1, keepdims=True)
    depth = np.maximum(0.0, np.log((med + 1e-6) / (P + 1e-6)))
    log_y = np.log(Y + 1.0)

    stats = []
    for i, unit in enumerate(units):
        treated = depth[i] > threshold
        n_treated = int(treated.sum())
        n_control = int((~treated).sum())
        if n_treated >= 15 and n_control >= 200:
            stats.append((n_treated, unit, i))
    stats = sorted(stats, reverse=True)[:unit_cap]
    if not stats:
        return {"status": "skipped", "reason": "insufficient M5 markdown overlap"}

    rows = []
    for _, unit, i in stats:
        for t in range(1, T):
            rows.append((
                unit,
                t,
                int(data["wday"][t]),
                float(log_y[i, t]),
                float(log_y[i, t - 1]),
                float(depth[i, t] > threshold),
            ))
    df = pd.DataFrame(rows, columns=["unit", "t", "weekday", "log_sales", "lag1_log_sales", "treated"])

    unit_effects = []
    for unit, g in df.groupby("unit"):
        treated = g[g["treated"] == 1]
        controls = g[g["treated"] == 0]
        if len(treated) < 10 or len(controls) < 50:
            continue
        control_by_wd = controls.groupby("weekday")["log_sales"].mean()
        fallback = float(controls["log_sales"].mean())
        diffs = []
        for _, row in treated.iterrows():
            base = float(control_by_wd.get(row["weekday"], fallback))
            diffs.append(float(row["log_sales"] - base))
        unit_effects.append({"unit": unit, "matched_att": float(np.mean(diffs)), "n_treated": len(diffs)})

    unit_eff_df = pd.DataFrame(unit_effects)
    if len(unit_eff_df) < 30:
        return {"status": "skipped", "reason": "insufficient M5 matched markdown units"}

    quasi_effect = float(np.average(unit_eff_df["matched_att"], weights=unit_eff_df["n_treated"]))
    t_stat, p_val = st.ttest_1samp(unit_eff_df["matched_att"], popmean=0.0)

    y = df["log_sales"].values.astype(float)
    treat = df["treated"].values.astype(float)

    # D0: pooled observational treatment effect.
    X_d0 = np.column_stack([
        treat,
        df["lag1_log_sales"].values,
        df["weekday"].values / 6.0,
        df["t"].values / T,
    ])
    sc0 = StandardScaler()
    X0 = sc0.fit_transform(X_d0)
    reg_d0 = Ridge(alpha=1.0)
    reg_d0.fit(X0, y)
    d0_implied = float(reg_d0.coef_[0] / sc0.scale_[0])

    # D2: residualize treatment and outcome on unit/date/weekday/lag controls.
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
    cats = enc.fit_transform(df[["unit", "t", "weekday"]])
    dense_ctrl = sp.csr_matrix(np.column_stack([df["lag1_log_sales"].values]))
    X_ctrl = sp.hstack([cats, dense_ctrl], format="csr")
    Xc = StandardScaler(with_mean=False).fit_transform(X_ctrl)

    reg_pi = Ridge(alpha=10.0)
    reg_pi.fit(Xc, treat)
    treat_resid = treat - reg_pi.predict(Xc)
    reg_m = Ridge(alpha=10.0)
    reg_m.fit(Xc, y)
    y_resid = y - reg_m.predict(Xc)
    reg_theta = Ridge(alpha=0.1, fit_intercept=False)
    reg_theta.fit(treat_resid.reshape(-1, 1), y_resid)
    d2_implied = float(reg_theta.coef_[0])

    d0_nee = abs(d0_implied - quasi_effect)
    d2_nee = abs(d2_implied - quasi_effect)
    d0_unit_abs = np.abs(unit_eff_df["matched_att"].values - d0_implied)
    d2_unit_abs = np.abs(unit_eff_df["matched_att"].values - d2_implied)
    try:
        w_stat, w_p = st.wilcoxon(d0_unit_abs, d2_unit_abs, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception:
        w_stat, w_p = float("nan"), 1.0

    return {
        "status": "complete",
        "validation_type": "real a-type M5 markdown matched-ATT",
        "threshold_log_discount": threshold,
        "n_rows": int(len(df)),
        "n_units_used": int(unit_eff_df.shape[0]),
        "treated_rate": round(float(df["treated"].mean()), 4),
        "matched_att_effect": round(quasi_effect, 4),
        "matched_att_t_stat": round(float(t_stat), 3),
        "matched_att_p_value": round(float(p_val), 4),
        "d0_implied_effect": round(d0_implied, 4),
        "d2_implied_effect": round(d2_implied, 4),
        "d0_nee": round(d0_nee, 4),
        "d2_nee": round(d2_nee, 4),
        "nee_reduction_frac": round(float((d0_nee - d2_nee) / (d0_nee + 1e-8)), 4),
        "unit_wilcoxon_stat": round(w_stat, 2) if not math.isnan(w_stat) else None,
        "unit_wilcoxon_p": round(w_p, 4),
        "h5_markdown_pass": bool(d2_nee < d0_nee and w_p < 0.05),
    }


# ── FAVORITA PROMO NEE: REAL A-TYPE VALIDATION ────────────────────────────────

def load_favorita_subset(
    unit_cap: int = 1200,
    min_count: int = 12,
    min_treated: int = 2,
    min_control: int = 6,
    rows_limit: int = 5_000_000,
) -> dict | None:
    chunks_dir = FAV_DIR / "chunks"
    if not chunks_dir.exists():
        return None
    # The local Favorita chunks are byte-split files named train.csv.part-*.
    # Nonzero parts may start with a partial row, so parse defensively below.
    chunk_files = sorted(chunks_dir.glob("train.csv.part-*"))
    if not chunk_files:
        return None

    names = ["id", "date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    dfs = []
    rows_left = rows_limit
    # Later chunks have actual promotion variation; early 2013 rows have almost none.
    for cf in reversed(chunk_files):
        if rows_left <= 0:
            break
        df_part = pd.read_csv(
            cf,
            nrows=rows_left,
            header=None,
            names=names,
            on_bad_lines="skip",
            low_memory=False,
        )
        df_part["date"] = pd.to_datetime(df_part["date"], errors="coerce")
        for col in ["store_nbr", "item_nbr", "unit_sales"]:
            df_part[col] = pd.to_numeric(df_part[col], errors="coerce")
        df_part = df_part.dropna(subset=["date", "store_nbr", "item_nbr", "unit_sales"])
        df_part["store_nbr"] = df_part["store_nbr"].astype(int)
        df_part["item_nbr"] = df_part["item_nbr"].astype(int)
        df_part["onpromotion"] = (
            df_part["onpromotion"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(["true", "1"])
            .astype(float)
        )
        dfs.append(df_part)
        rows_left -= len(df_part)

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["unit_sales"] >= 0].copy()

    hol = pd.read_csv(FAV_DIR / "holidays_events.csv", parse_dates=["date"])
    hol_dates = set(hol["date"].dt.normalize())

    df["is_holiday"] = df["date"].isin(hol_dates).astype(float)
    df["log_sales"] = np.log(df["unit_sales"].clip(lower=0) + 1.0)
    df["weekday"] = df["date"].dt.weekday.astype(int)
    df["unit"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
    df = df.sort_values(["unit", "date"])
    df["lag1_log_sales"] = df.groupby("unit")["log_sales"].shift(1)
    df["lag1_log_sales"] = df["lag1_log_sales"].fillna(df.groupby("unit")["log_sales"].transform("median"))

    # Keep units with enough within-unit promo/non-promo support; this is the
    # overlap condition for a defensible real-data a-type validation.
    unit_stats = df.groupby("unit")["onpromotion"].agg(["count", "sum"])
    unit_stats["nonpromo"] = unit_stats["count"] - unit_stats["sum"]
    eligible = unit_stats[
        (unit_stats["count"] >= min_count) &
        (unit_stats["sum"] >= min_treated) &
        (unit_stats["nonpromo"] >= min_control)
    ].copy()
    if eligible.empty:
        return None
    eligible["promo_rate"] = eligible["sum"] / eligible["count"]
    # Cap size for laptop reproducibility while keeping many independent units.
    keep_units = eligible.sort_values(["sum", "count"], ascending=False).head(unit_cap).index
    sub = df[df["unit"].isin(keep_units)].copy()

    return {
        "df": sub,
        "n_units_total": int(unit_stats.shape[0]),
        "n_units_eligible": int(eligible.shape[0]),
        "unit_cap": int(unit_cap),
        "min_count": int(min_count),
        "min_treated": int(min_treated),
        "min_control": int(min_control),
        "rows_limit": int(rows_limit),
        "date_min": str(sub["date"].min().date()),
        "date_max": str(sub["date"].max().date()),
    }


def compute_favorita_promo_nee(fav_data: dict | None) -> dict:
    """
    Real a-type validation on Favorita promotions.

    Quasi target: matched within-unit ATT. For every promoted row, compare
    log-sales with the same unit's non-promoted rows on the same weekday where
    possible, otherwise with all non-promoted rows for that unit. This target is
    not fitted by D0/D2 and gives a real-data external check.

    D0: pooled observational regression with promo as an ordinary feature.
    D2: R-learner residualization of promo and outcome on unit/date/weekday/lag
    controls. The controls encode the confounding story: promotions cluster by
    item/store/date and recent demand.
    """
    if fav_data is None:
        return {"status": "skipped", "reason": "Favorita chunk data not found or no overlap"}

    df = fav_data["df"]
    sub_use = df[[
        "date", "unit", "weekday", "log_sales", "lag1_log_sales", "is_holiday", "onpromotion"
    ]].dropna().copy()
    if len(sub_use) < 200:
        return {"status": "skipped", "reason": "insufficient usable rows after overlap filter"}

    # Matched within-unit ATT and unit-level paired evidence.
    unit_effects = []
    for unit, g in sub_use.groupby("unit"):
        treated = g[g["onpromotion"] == 1]
        controls = g[g["onpromotion"] == 0]
        if len(treated) < 2 or len(controls) < 4:
            continue
        control_by_wd = controls.groupby("weekday")["log_sales"].mean()
        fallback = float(controls["log_sales"].mean())
        diffs = []
        for _, row in treated.iterrows():
            base = float(control_by_wd.get(row["weekday"], fallback))
            diffs.append(float(row["log_sales"] - base))
        if diffs:
            unit_effects.append({"unit": unit, "matched_att": float(np.mean(diffs)), "n_treated": len(diffs)})

    unit_eff_df = pd.DataFrame(unit_effects)
    if len(unit_eff_df) < 30:
        return {"status": "skipped", "reason": "insufficient matched treated units"}
    quasi_effect = float(np.average(unit_eff_df["matched_att"], weights=unit_eff_df["n_treated"]))
    t_stat, p_val = st.ttest_1samp(unit_eff_df["matched_att"], popmean=0.0)

    y = sub_use["log_sales"].values.astype(float)
    promo = sub_use["onpromotion"].values.astype(float)

    # D0: intentionally observational pooled MISO-like treatment of promo.
    X_d0 = np.column_stack([
        promo,
        sub_use["lag1_log_sales"].values,
        sub_use["is_holiday"].values,
        sub_use["weekday"].values / 6.0,
        (sub_use["date"] - sub_use["date"].min()).dt.days.values / 60.0,
    ])
    sc0 = StandardScaler()
    X0 = sc0.fit_transform(X_d0)
    reg_d0 = Ridge(alpha=1.0)
    reg_d0.fit(X0, y)
    d0_implied = float(reg_d0.coef_[0] / sc0.scale_[0])

    # D2: orthogonalized real-data effect with unit/date FE and recent demand.
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn<1.2
        enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
    cats = enc.fit_transform(sub_use[["unit", "date", "weekday"]])
    dense_ctrl = sp.csr_matrix(np.column_stack([
        sub_use["lag1_log_sales"].values,
        sub_use["is_holiday"].values,
    ]))
    X_ctrl = sp.hstack([cats, dense_ctrl], format="csr")
    sc_ctrl = StandardScaler(with_mean=False)
    Xc = sc_ctrl.fit_transform(X_ctrl)

    reg_pi = Ridge(alpha=10.0)
    reg_pi.fit(Xc, promo)
    promo_resid = promo - reg_pi.predict(Xc)

    reg_m = Ridge(alpha=10.0)
    reg_m.fit(Xc, y)
    y_resid = y - reg_m.predict(Xc)

    reg_theta = Ridge(alpha=0.1, fit_intercept=False)
    reg_theta.fit(promo_resid.reshape(-1, 1), y_resid)
    d2_implied = float(reg_theta.coef_[0])

    d0_nee = abs(d0_implied - quasi_effect)
    d2_nee = abs(d2_implied - quasi_effect)
    h5_fav_pass = bool(d2_nee < d0_nee)

    # Unit-level NEE comparison: D2 and D0 are global effect estimates, but the
    # target has unit-level matched effects. This gives paired evidence that D2
    # is closer across units, not just to one aggregate number.
    d0_unit_abs = np.abs(unit_eff_df["matched_att"].values - d0_implied)
    d2_unit_abs = np.abs(unit_eff_df["matched_att"].values - d2_implied)
    try:
        w_stat, w_p = st.wilcoxon(d0_unit_abs, d2_unit_abs, alternative="greater")
        w_stat, w_p = float(w_stat), float(w_p)
    except Exception:
        w_stat, w_p = float("nan"), 1.0

    return {
        "status": "complete",
        "validation_type": "real a-type promotion matched-ATT",
        "date_min": fav_data["date_min"],
        "date_max": fav_data["date_max"],
        "n_rows": int(len(sub_use)),
        "n_units_total": fav_data["n_units_total"],
        "n_units_eligible": fav_data["n_units_eligible"],
        "n_units_used": int(unit_eff_df.shape[0]),
        "unit_cap": fav_data.get("unit_cap"),
        "min_count": fav_data.get("min_count"),
        "min_treated": fav_data.get("min_treated"),
        "min_control": fav_data.get("min_control"),
        "promo_rate": round(float(sub_use["onpromotion"].mean()), 4),
        "matched_att_effect": round(quasi_effect, 4),
        "matched_att_t_stat": round(float(t_stat), 3),
        "matched_att_p_value": round(float(p_val), 4),
        "d0_implied_effect": round(d0_implied, 4),
        "d2_implied_effect": round(d2_implied, 4),
        "d0_nee": round(d0_nee, 4),
        "d2_nee": round(d2_nee, 4),
        "nee_reduction_frac": round(float((d0_nee - d2_nee) / (d0_nee + 1e-8)), 4),
        "unit_wilcoxon_stat": round(w_stat, 2) if not math.isnan(w_stat) else None,
        "unit_wilcoxon_p": round(w_p, 4),
        "h5_fav_pass": bool(h5_fav_pass),
    }


def favorita_promo_robustness() -> dict:
    """Run a small reviewer-facing robustness grid for the real a-type gate."""
    configs = [
        {"name": "cap400_base", "unit_cap": 400, "min_count": 12, "min_treated": 2, "min_control": 6, "rows_limit": 5_000_000},
        {"name": "cap800_base", "unit_cap": 800, "min_count": 12, "min_treated": 2, "min_control": 6, "rows_limit": 5_000_000},
        {"name": "cap1200_base", "unit_cap": 1200, "min_count": 12, "min_treated": 2, "min_control": 6, "rows_limit": 5_000_000},
        {"name": "cap800_strict_overlap", "unit_cap": 800, "min_count": 16, "min_treated": 3, "min_control": 8, "rows_limit": 5_000_000},
    ]

    rows = []
    for cfg in configs:
        data = load_favorita_subset(
            unit_cap=cfg["unit_cap"],
            min_count=cfg["min_count"],
            min_treated=cfg["min_treated"],
            min_control=cfg["min_control"],
            rows_limit=cfg["rows_limit"],
        )
        res = compute_favorita_promo_nee(data)
        row = {"name": cfg["name"], **cfg, "status": res.get("status", "unknown")}
        if res.get("status") == "complete":
            row.update({
                "n_rows": res["n_rows"],
                "n_units_used": res["n_units_used"],
                "matched_att_effect": res["matched_att_effect"],
                "d0_nee": res["d0_nee"],
                "d2_nee": res["d2_nee"],
                "nee_reduction_frac": res["nee_reduction_frac"],
                "unit_wilcoxon_p": res["unit_wilcoxon_p"],
                "pass": bool(res["h5_fav_pass"]),
            })
        else:
            row["pass"] = False
            row["reason"] = res.get("reason")
        rows.append(row)

    complete = [r for r in rows if r.get("status") == "complete"]
    pass_rate = float(np.mean([r["pass"] for r in complete])) if complete else 0.0
    median_reduction = float(np.median([r["nee_reduction_frac"] for r in complete])) if complete else 0.0
    max_p = float(np.max([r["unit_wilcoxon_p"] for r in complete])) if complete else 1.0
    robust_pass = bool(len(complete) == len(configs) and pass_rate >= 0.75 and median_reduction >= 0.50 and max_p < 0.05)

    return {
        "status": "complete" if complete else "skipped",
        "robust_pass": robust_pass,
        "n_configs": len(configs),
        "n_complete": len(complete),
        "pass_rate": round(pass_rate, 4),
        "median_nee_reduction_frac": round(median_reduction, 4),
        "max_unit_wilcoxon_p": round(max_p, 4),
        "rows": rows,
        "gate": "all configs complete; >=75% D2<D0; median NEE reduction >=50%; max p<0.05",
    }


# ── POLICY-RANKING FIDELITY ───────────────────────────────────────────────────

def policy_ranking_fidelity(data: dict, gamma: float = 0.5, seed: int = 2021) -> dict:
    """
    Semi-synthetic Policy Ranking Fidelity using quality-confounding panel design.

    Generates K=10 price plans. Ground truth ranks them by theta_star * discount.
    D0 (no item FE) gets positive price coef (confounded) → inverted ranking → tau ≈ -1.
    D2 (item dummies in V) gets correct price coef → correct ranking → tau ≈ +1.
    This is a decision stress test, not real-data causal evidence.
    """
    rng = np.random.default_rng(seed)
    n = min(N_ITEMS, data["n"])
    Y = data["Y"][:n]
    T = Y.shape[1]
    month_arr = data["month"]
    wday_arr = data["wday"]

    # Quality-confounding synthetic data (same model as M1/M2)
    theta_star = rng.uniform(-0.8, -0.2, size=n)
    quality = rng.uniform(0.0, 3.0, size=n)
    log_demand = np.log(Y + 1.0)
    eps_pi = rng.normal(0, 0.5, (n, T))
    phi_mat = gamma * quality[:, None] + eps_pi        # (n, T)
    phi_mat = np.clip(phi_mat, -3.0, 3.0)
    beta_q = 2.0
    eps_y = rng.normal(0, 0.2, (n, T))
    log_y_syn = log_demand + theta_star[:, None] * phi_mat + beta_q * quality[:, None] + eps_y

    K = 10
    discount_levels = np.linspace(-0.30, 0.20, K)

    # Calendar features
    t_arr = np.arange(T).astype(float) / T
    sin_m = np.sin(2 * math.pi * month_arr / 12)
    cos_m = np.cos(2 * math.pi * month_arr / 12)
    sin_w = np.sin(2 * math.pi * wday_arr / 7)
    cos_w = np.cos(2 * math.pi * wday_arr / 7)

    train_end = T - H * 2
    item_dummies = np.eye(n)

    # Build pooled panel datasets
    rows_tr_d0, rows_tr_d2 = [], []
    y_tr_d0_list, y_tr_d2_list = [], []

    for i in range(n):
        for t in range(7, train_end):
            phi_t = phi_mat[i, t]
            y_t = log_y_syn[i, t]
            cal_feat = np.array([t_arr[t], sin_m[t], cos_m[t], sin_w[t], cos_w[t]])
            rows_tr_d0.append(np.concatenate([cal_feat, [phi_t]]))  # no item FE
            rows_tr_d2.append(np.concatenate([cal_feat, item_dummies[i], [phi_t]]))  # item FE
            y_tr_d0_list.append(y_t)
            y_tr_d2_list.append(y_t)

    X_d0 = np.array(rows_tr_d0);   y_d0_arr = np.array(y_tr_d0_list)
    X_d2 = np.array(rows_tr_d2);   y_d2_arr = np.array(y_tr_d2_list)

    sc0 = StandardScaler(); X0s = sc0.fit_transform(X_d0)
    sc2 = StandardScaler(); X2s = sc2.fit_transform(X_d2)

    reg_d0 = Ridge(alpha=0.5); reg_d0.fit(X0s, y_d0_arr)
    reg_d2 = Ridge(alpha=0.5); reg_d2.fit(X2s, y_d2_arr)

    d0_price_coef = float(reg_d0.coef_[-1] / sc0.scale_[-1])
    d2_price_coef = float(reg_d2.coef_[-1] / sc2.scale_[-1])

    # Item-level rankings
    tau_d0_list, tau_d2_list = [], []
    for i in range(n):
        gt_scores   = theta_star[i] * discount_levels  # true: best plan = deepest discount
        d0_scores   = d0_price_coef * discount_levels   # likely positive coef → wrong ranking
        d2_scores   = d2_price_coef * discount_levels   # negative coef → correct ranking

        gt_rank = np.argsort(-gt_scores)
        tau_d0, _ = st.kendalltau(gt_rank, np.argsort(-d0_scores))
        tau_d2, _ = st.kendalltau(gt_rank, np.argsort(-d2_scores))
        tau_d0_list.append(float(tau_d0))
        tau_d2_list.append(float(tau_d2))

    tau_d0_mean = float(np.mean(tau_d0_list))
    tau_d2_mean = float(np.mean(tau_d2_list))

    try:
        w, p = st.wilcoxon(tau_d2_list, tau_d0_list, alternative="greater")
        w, p = float(w), float(p)
    except Exception:
        w, p = float("nan"), 1.0

    prf_pass = bool(tau_d2_mean > tau_d0_mean and p < 0.10)

    return {
        "d0_kendall_tau_mean": round(tau_d0_mean, 4),
        "d2_kendall_tau_mean": round(tau_d2_mean, 4),
        "d0_price_coef": round(d0_price_coef, 4),
        "d2_price_coef": round(d2_price_coef, 4),
        "tau_improvement": round(tau_d2_mean - tau_d0_mean, 4),
        "wilcoxon_stat": round(w, 2) if not math.isnan(w) else None,
        "wilcoxon_p": round(p, 4),
        "prf_pass": prf_pass,
        "n_items": n,
        "K_plans": K,
        "evidence_role": "semi-synthetic decision stress test; not counted as real-data validation",
        "design": "quality-confounding panel; D0=no item FE, D2=item dummies in V",
    }


# ── BH-FDR CORRECTION ────────────────────────────────────────────────────────

def bh_fdr(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    rejected = [False] * n
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha * (rank + 1) / n:
            rejected[idx] = True
        else:
            break
    return rejected


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading M5 data (all stores, FOODS_1)...")
    m5_data = load_m5_all_stores()
    print(f"  {m5_data['n']} items × {m5_data['T']} days")

    print("Computing M5 SNAP DiD + NEE...")
    snap_result = m5_snap_nee(m5_data)
    print(f"  DiD SNAP effect: {snap_result['did_estimate']['did_effect']:.4f}")
    print(f"  D0 NEE: {snap_result['d0_nee_mean']:.4f}  D2 NEE: {snap_result['d2_nee_mean']:.4f}")
    print(f"  H5 M5 pass: {snap_result['h5_m5_pass']}")

    print("Computing M5 markdown real a-type NEE...")
    markdown_result = compute_m5_markdown_nee(m5_data)
    if markdown_result.get("status") == "complete":
        print(
            f"  M5 markdown D0 NEE: {markdown_result['d0_nee']:.4f} "
            f"D2 NEE: {markdown_result['d2_nee']:.4f} "
            f"pass: {markdown_result['h5_markdown_pass']}"
        )

    print("Loading Favorita promotion window...")
    fav_data = load_favorita_subset()
    if fav_data:
        print(
            f"  Loaded Favorita {fav_data['date_min']}..{fav_data['date_max']} "
            f"({len(fav_data['df'])} rows)"
        )
    else:
        print("  Favorita chunks not found or no overlap, will skip")

    print("Computing Favorita real promo NEE...")
    fav_result = compute_favorita_promo_nee(fav_data)
    if fav_result.get("status") == "complete":
        print(f"  Matched ATT promo effect: {fav_result['matched_att_effect']:.4f}")
        print(f"  D0 NEE: {fav_result['d0_nee']:.4f}  D2 NEE: {fav_result['d2_nee']:.4f}")
        print(f"  H5 Favorita pass: {fav_result['h5_fav_pass']}")

    print("Computing Favorita robustness grid...")
    fav_robust = favorita_promo_robustness()
    print(
        f"  Robust pass: {fav_robust['robust_pass']} "
        f"(median NEE reduction={fav_robust['median_nee_reduction_frac']:.1%})"
    )

    # Load M5 subset for PRF (CA_1 only for speed)
    print("Loading M5 CA_1 subset for policy ranking fidelity...")
    m5_ca1 = {
        "Y": m5_data["Y"][:N_ITEMS],
        "P": m5_data["P"][:N_ITEMS],
        "SNAP": m5_data["SNAP"][:N_ITEMS],
        "month": m5_data["month"],
        "wday": m5_data["wday"],
        "n": min(N_ITEMS, m5_data["n"]),
        "T": m5_data["T"],
    }
    print("Computing Policy Ranking Fidelity (PRF)...")
    prf_result = policy_ranking_fidelity(m5_ca1)
    print(f"  D0 Kendall-τ: {prf_result['d0_kendall_tau_mean']:.4f}")
    print(f"  D2 Kendall-τ: {prf_result['d2_kendall_tau_mean']:.4f}")
    print(f"  PRF pass: {prf_result['prf_pass']}")

    # ── BH-FDR across all tests ───────────────────────────────────────────────
    p_values = [
        snap_result["did_estimate"]["did_p_value"],
        snap_result["paired_p_value"],
    ]
    if "wilcoxon_p" in prf_result:
        p_values.append(prf_result["wilcoxon_p"])
    if fav_result.get("unit_wilcoxon_p") is not None:
        p_values.append(fav_result["unit_wilcoxon_p"])
    if markdown_result.get("unit_wilcoxon_p") is not None:
        p_values.append(markdown_result["unit_wilcoxon_p"])
    for row in fav_robust.get("rows", []):
        if row.get("unit_wilcoxon_p") is not None:
            p_values.append(row["unit_wilcoxon_p"])

    fdr_rejected = bh_fdr(p_values, alpha=0.05)
    labels = ["SNAP-DiD", "SNAP-NEE-paired-t", "PRF-Wilcoxon", "Favorita-unit-NEE", "M5-markdown-unit-NEE"]
    labels.extend([f"Favorita-robust-{row['name']}" for row in fav_robust.get("rows", []) if row.get("unit_wilcoxon_p") is not None])
    labels = labels[: len(p_values)]
    fdr_summary = {lbl: {"p": round(p, 4), "reject": bool(r)} for lbl, p, r in zip(labels, p_values, fdr_rejected)}

    claim_p_values = []
    claim_labels = []
    if fav_result.get("unit_wilcoxon_p") is not None:
        claim_p_values.append(fav_result["unit_wilcoxon_p"])
        claim_labels.append("Favorita-real-a-type-unit-NEE")
    if markdown_result.get("unit_wilcoxon_p") is not None:
        claim_p_values.append(markdown_result["unit_wilcoxon_p"])
        claim_labels.append("M5-markdown-real-a-type-unit-NEE")
    for row in fav_robust.get("rows", []):
        if row.get("unit_wilcoxon_p") is not None:
            claim_p_values.append(row["unit_wilcoxon_p"])
            claim_labels.append(f"Favorita-robust-{row['name']}")
    claim_rejected = bh_fdr(claim_p_values, alpha=0.05) if claim_p_values else []
    claim_fdr_summary = {
        lbl: {"p": round(p, 4), "reject": bool(r)}
        for lbl, p, r in zip(claim_labels, claim_p_values, claim_rejected)
    }

    # Overall H5 verdict
    # SNAP is c-type and therefore a non-degradation sanity check, not the
    # external-validity gate for controllable a-type covariates.
    snap_non_degrade = snap_result["d2_nee_mean"] <= snap_result["d0_nee_mean"] + 0.005
    h5_pass = bool(
        fav_result.get("h5_fav_pass", False)
        and markdown_result.get("h5_markdown_pass", False)
        and snap_non_degrade
        and fav_robust.get("robust_pass", False)
    )
    prf_pass = prf_result["prf_pass"]

    # Identification diagnostics (ported from M0)
    id_strength = {
        "note": (
            "M5 price overlap is weak in M0, so M5 price is not used for a real-data "
            "effect gate. SNAP is c-type and used only as a non-degradation check. "
            "Favorita promotion has real a-type overlap in the late-window subset and "
            "carries the H5 external-validity gate."
        ),
    }

    summary = {
        "milestone": "M3",
        "gate_label": "PASS" if (h5_pass and prf_pass) else ("PARTIAL-PASS" if (h5_pass or prf_pass) else "FAIL"),
        "h5_snap_m5": snap_result,
        "h5_m5_markdown": markdown_result,
        "h5_favorita_promo": fav_result,
        "favorita_promo_robustness": fav_robust,
        "prf_result": prf_result,
        "bh_fdr_summary": fdr_summary,
        "claim_family_fdr_summary": claim_fdr_summary,
        "h5_overall_pass": bool(h5_pass),
        "snap_non_degradation_check": bool(snap_non_degrade),
        "prf_overall_pass": bool(prf_pass),
        "identification_strength": id_strength,
    }

    out = OUT_DIR / "real_data_summary.json"
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, (np.floating, float)) and math.isnan(obj): return None
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)
    print(f"\nM3 real-data summary → {out}")


if __name__ == "__main__":
    main()
