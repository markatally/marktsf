"""Retrospective error attribution; never used to filter the benchmark."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


def diagnose_native(data_dir, comparison_dir, output_dir):
    root, dest = Path(data_dir), Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    groups = pd.read_parquet(root/"groups.parquet")
    scores = pd.read_parquet(Path(comparison_dir)/"case_scores.parquet")
    metadata = []
    for group in groups.itertuples():
        with np.load(root/"inputs"/group.file) as data:
            for v, key in enumerate(data["keys"]):
                y = data["target"][v]
                if not np.isfinite(y).all():
                    raise ValueError("This event-path attribution requires fully observed targets")
                atom, origin_state, age, recent_rate = data["features"][v]
                if np.isnan(atom):
                    event = "no_historical_atom"
                    first_change = None
                else:
                    state = y==atom
                    change = np.flatnonzero(state!=bool(origin_state))
                    first_change = int(change[0]+1) if len(change) else None
                    event = ("atom_exit" if len(change) else "atom_stay") if origin_state else ("active_enters_atom" if len(change) else "active_stays_active")
                x = data["context"][v].astype(float)
                dx = np.diff(x)
                scale = float(np.sqrt(np.mean(dx**2))) if len(dx) else 0.
                # Origin-available descriptors, separate from realized event labels.
                change_scale = float(abs(x[-1]-x[max(0,len(x)-128)])/max(scale,1e-12))
                metadata.append(dict(configuration=group.configuration,item=str(key[1]),origin=group.origin,
                    event=event,first_change=first_change,history_atom_age=age,
                    recent_atom_rate=recent_rate,history_displacement_in_diff_rms=change_scale,
                    age_bucket="no_atom" if np.isnan(atom) else "1_4" if age<=4 else "5_16" if age<=16 else "17_64" if age<=64 else "65_plus"))
    meta = pd.DataFrame(metadata)
    keys = ["configuration","item","origin"]
    if meta.duplicated(keys).any():
        raise ValueError("Ambiguous event attribution keys")
    enriched = scores.merge(meta,on=keys,validate="many_to_one",how="left")
    if enriched.event.isna().any():
        raise ValueError("Some baseline predictions lack event attribution")
    base = scores[scores.model.eq("timesfm3")]
    denominator = base.groupby("configuration").denominator.sum()
    if not (denominator>0).all():
        raise ValueError("Cannot attribute undefined configuration WQL")
    enriched["wql_contribution"] = enriched.numerator/enriched.configuration.map(denominator)/len(denominator)
    event_scores = enriched.groupby(["model","event"]).agg(
        macro_wql_contribution=("wql_contribution","sum"),variate_origins=("item","size"),
        datasets=("dataset","nunique"),points=("points","sum"),atom_points=("atom_points","sum"))
    event_scores["macro_error_share"] = event_scores.macro_wql_contribution/event_scores.groupby("model").macro_wql_contribution.transform("sum")
    terms = enriched.groupby(["model","term","event"]).wql_contribution.sum().rename("macro_wql_contribution").reset_index()
    age = enriched[enriched.model.eq("timesfm3") & enriched.event.isin(["atom_stay","atom_exit"])].copy()
    age["exit_observed"] = age.event.eq("atom_exit")
    age = age.groupby(["term","age_bucket"]).agg(variate_origins=("item","size"),
        datasets=("dataset","nunique"),exit_fraction=("exit_observed","mean"),
        macro_wql_contribution=("wql_contribution","sum")).reset_index()
    full = pd.read_parquet(Path(comparison_dir)/"configuration_scores.parquet")
    raw_mean = full.groupby("model").wql9.mean()
    for model, part in event_scores.groupby("model"):
        if not np.isclose(part.macro_wql_contribution.sum(),raw_mean[model],rtol=1e-10,atol=1e-12):
            raise ValueError("MECE event contributions do not conserve full configuration WQL")
    event_scores.reset_index().to_parquet(dest/"event_scores.parquet",index=False)
    terms.to_parquet(dest/"term_event_scores.parquet",index=False)
    age.to_parquet(dest/"origin_age_diagnostics.parquet",index=False)
    meta.to_parquet(dest/"event_metadata.parquet",index=False)
    result = dict(stage="native_error_attribution",configurations=len(denominator),
        variate_origins=len(meta),events=event_scores.reset_index().to_dict("records"),
        origin_age=age.to_dict("records"),sota=False,
        scope="All 131 development configurations retained. Contributions sum to unscaled arithmetic configuration WQL, not the official scaled geometric leaderboard. Events use future labels for retrospective diagnosis only; not model inputs or benchmark selection. Exit fractions mix tasks, channels and horizons and are not causal duration effects.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
