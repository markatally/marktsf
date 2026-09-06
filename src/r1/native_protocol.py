"""Full-variate BOOM development protocol with official horizon/window rules.

The Dataset rules are applied to a historical 70% prefix, never to the final
official test. Original Arrow rows remain native multivariate groups. No atom
or performance-based filtering is used. This is development, not a leaderboard.
"""
from pathlib import Path
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from r1.data import stable_hash, sha256, calibration_atom, state_ages


def prediction_length(freq):
    match = re.fullmatch(r"\d*(T|MIN|H|S|D|W|M)", str(freq).upper())
    if not match:
        raise ValueError(f"Unsupported frequency; must verify upstream mapping: {freq}")
    return {"T":48, "MIN":48, "H":48, "S":60, "D":30, "W":8, "M":12}[match[1]]


def historical_imputation(context):
    """Shared origin-available imputation: forward then back fill within context."""
    frame = pd.DataFrame(context.T).replace([np.inf, -np.inf], np.nan)
    filled = frame.ffill().bfill().fillna(0.)
    return filled.to_numpy().T.astype(np.float32)


def seasonal_lag(freq):
    """GluonTS 0.17 defaults for the explicitly supported BOOM cadences."""
    match = re.fullmatch(r"(\d*)(T|MIN|H|S|D|W|M)", str(freq).upper())
    if not match:
        raise ValueError(f"Unsupported seasonality: {freq}")
    base = {"T":1440,"MIN":1440,"H":24,"S":3600,"D":1,"W":1,"M":12}[match[2]]
    lag, remainder = divmod(base, int(match[1] or 1))
    return 1 if remainder else lag


def prepare_native(repo_root, output_dir, properties_path, max_tasks=48,
                   development_fraction=.7, train_fraction=.7, context_length=2048):
    dest = Path(output_dir)
    inputs = dest/"inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    properties = json.loads(Path(properties_path).read_text())
    paths = sorted((p for p in (Path(repo_root)/"input/BOOM").iterdir() if p.is_dir()), key=lambda p:stable_hash(p.name))
    paths = [p for p in paths if stable_hash(p.name)%10<2][:max_tasks]
    groups, skipped, audit, series_keys = [], [], [], set()
    all_sources = {}
    for path in paths:
        raw = []
        source_hashes = {}
        for f in sorted(path.glob("*.arrow")):
            source_hashes[str(f)] = sha256(f)
            with f.open("rb") as stream:
                raw.extend(ipc.open_stream(stream).read_all().to_pylist())
        all_sources.update(source_hashes)
        native = []
        for row in raw:
            y = np.asarray(row["target"], dtype=np.float64)
            if y.ndim==1:
                y = y[None, :]
            elif y.ndim!=2:
                raise ValueError("Unsupported native Arrow shape")
            native.append((str(row["item_id"]), str(row["freq"]), y))
        if len({item for item, freq, y in native})!=len(native):
            raise ValueError("Native row identities must be unique")
        frequencies = {freq for item, freq, y in native}
        if len(frequencies)!=1:
            raise ValueError("Task contains inconsistent frequencies")
        freq = next(iter(frequencies))
        short = prediction_length(freq)
        terms = ["short"] if properties[path.name]["term"]=="short" else ["short", "medium", "long"]
        lengths = [y.shape[-1] for item, f, y in native]
        minimum = min(lengths)
        dev_minimum = min(int(n*development_fraction) for n in lengths)
        for term in terms:
            horizon = short*{"short":1, "medium":10, "long":15}[term]
            official_windows = min(20, max(1, math.ceil(.1*minimum/horizon)))
            windows = min(20, max(1, math.ceil(.1*dev_minimum/horizon)))
            config = f"{path.name}/{freq}/{term}"
            for item, row_freq, complete in native:
                n = complete.shape[-1]
                end = int(n*development_fraction)
                train_end = int(end*train_fraction)
                official_start = n-horizon*official_windows
                if end>official_start:
                    raise ValueError(f"Development prefix intersects official test: {config}")
                first = end-horizon*windows
                audit.append(dict(dataset=path.name, item=item, term=term, n=n, dev_end=end,
                    train_end=train_end, official_start=official_start, dev_start=first, horizon=horizon, windows=windows))
                if first<train_end or first<2:
                    skipped.append(dict(config=config, item=item, reason="native_horizon_does_not_fit_development_validation",
                                        first_origin=first-1, train_end=train_end))
                    continue
                prefix = complete[:, :end]
                for window in range(windows):
                    start = first+window*horizon
                    origin = start-1
                    history = prefix[:, :start]
                    target = prefix[:, start:start+horizon]
                    context = history[:, -context_length:]
                    cleaned = historical_imputation(context)
                    lag = seasonal_lag(freq)
                    lag = lag if lag<=history.shape[-1] else 1
                    seasonal_difference = np.ma.masked_invalid(np.abs(history[:,lag:]-history[:,:-lag]))
                    seasonal_error = seasonal_difference.mean(axis=-1).filled(np.nan)
                    if not np.isfinite(cleaned).all():
                        raise ValueError("Shared native context imputation failed")
                    keys, hashes, features = [], [], []
                    for channel, y in enumerate(history):
                        name = item if complete.shape[0]==1 else f"{item}:v{channel}"
                        keys.append([path.name, name, str(origin)])
                        series_keys.add((path.name, name))
                        hashes.append(hashlib.sha256(np.ascontiguousarray(target[channel], dtype="<f8").tobytes()).hexdigest())
                        atom = calibration_atom(y)
                        state = y==atom if atom is not None else np.zeros(len(y), dtype=bool)
                        # Features are origin-available; realized constant targets
                        # are analyzed separately by the scorer, never selected here.
                        features.append([np.nan if atom is None else atom,
                            float(state[-1]), float(state_ages(state)[-1]) if atom is not None else 0.,
                            float(state[-min(128,len(state)):].mean()) if atom is not None else 0.])
                    filename = f"{len(groups):05d}.npz"
                    np.savez_compressed(inputs/filename, context=cleaned, target=target,
                        keys=np.asarray(keys), target_hashes=np.asarray(hashes), horizon=horizon,
                        configuration=config, freq=freq, features=np.asarray(features),
                        seasonal_error=seasonal_error,
                        imputed_context_values=int((~np.isfinite(context)).sum()))
                    groups.append(dict(file=filename, dataset=path.name, item=item, configuration=config,
                        freq=freq, term=term, horizon=horizon, origin=origin, window=window,
                        channels=complete.shape[0], input_sha=sha256(inputs/filename)))
        print(f"Native contract: {path.name}; groups={len(groups)}", flush=True)
    if not groups:
        raise ValueError("No native development cases")
    table = pd.DataFrame(groups)
    table.to_parquet(dest/"groups.parquet", index=False)
    result = dict(stage="native_development_contract", tasks=len(paths),
        evaluated_tasks=table.dataset.nunique(), configurations=table.configuration.nunique(), groups=len(groups),
        series=len(series_keys), variate_origins=int(table.channels.sum()),
        scalar_targets=int((table.channels*table.horizon).sum()), horizons=sorted(table.horizon.unique().tolist()),
        skipped=skipped, boundary_checks=audit, development_fraction=development_fraction,
        train_fraction=train_fraction, context_length=context_length,
        imputation="ffill then bfill using origin-available context; all-missing context set to zero; targets not imputed",
        groups_sha=sha256(dest/"groups.parquet"), sota=False,
        source_hashes=all_sources, properties_sha=sha256(properties_path),
        scope="Official horizon/window rule applied to a 70% historical prefix of fixed development task IDs. Full native variables, no atom filtering. Not final official test or full BOOM.")
    (dest/"manifest.json").write_text(json.dumps(result, indent=2))
    return result


def compare_native(data_dir, model_directories, output_dir, upstream_leaderboard):
    from r1.aggregation import official_low_variance, shifted_gmean
    from r1.evaluation import paired_cluster_interval
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    root = Path(data_dir)
    groups = pd.read_parquet(root/"groups.parquet")
    rows = []
    levels = np.arange(.1,1,.1)
    for group in groups.itertuples():
        with np.load(root/"inputs"/group.file) as data:
            y = data["target"]
            finite = np.isfinite(y)
            if not finite.any():
                raise ValueError("All targets missing; explicit benchmark handling required")
            predictions = {}
            for model, directory in model_directories.items():
                with np.load(Path(directory)/group.file) as f:
                    if not np.array_equal(f["target_hashes"], data["target_hashes"]) or not np.array_equal(f["keys"], data["keys"]) or str(f["input_sha"])!=group.input_sha:
                        raise ValueError("Native prediction identities differ")
                    predictions[model] = f["quantiles"].copy()
                    if model=="seasonalnaive":
                        predictions["last_value"] = np.repeat(data["context"][:, -1:, None], group.horizon, axis=1).repeat(9, axis=-1)
            for model, q in predictions.items():
                if q.shape!=y.shape+(9,) or not np.isfinite(q).all():
                    raise ValueError("Malformed native prediction")
                safe_y = np.where(finite, y, 0.)
                error = safe_y[..., None]-q
                pinball = (2*np.maximum(levels*error, (levels-1)*error)*finite[..., None]).mean(-1)
                for v, key in enumerate(data["keys"]):
                    mask = finite[v]
                    atom = data["features"][v, 0]
                    atom_mask = mask & (y[v]==atom)
                    valid_values = y[v, mask]
                    constant = bool(len(valid_values)>0 and np.all(valid_values==valid_values[0]))
                    rows.append(dict(dataset=group.dataset, configuration=group.configuration, term=group.term,
                        horizon=group.horizon, item=str(key[1]), origin=group.origin, model=model,
                        numerator=float(pinball[v].sum()), denominator=float(np.abs(valid_values).sum()),
                        absolute_error=float(np.abs(valid_values-q[v, mask, 4]).sum()),
                        squared_error=float(((valid_values-q[v, mask, 4])**2).sum()), points=int(mask.sum()),
                        mase=float(np.abs(valid_values-q[v,mask,4]).mean()/data["seasonal_error"][v]) if len(valid_values) and data["seasonal_error"][v]>0 else np.nan,
                        atom_loss=float(pinball[v, atom_mask].sum()), atom_points=int(atom_mask.sum()),
                        target_constant=constant, origin_atom_state=float(data["features"][v, 1]),
                        origin_age=float(data["features"][v, 2]),
                        above_q90=int((valid_values>q[v, mask, -1]).sum())))
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["dataset","configuration","term","model"]).sum(numeric_only=True)
    grouped["wql9"] = grouped.numerator/grouped.denominator
    grouped["mae"] = grouped.absolute_error/grouped.points
    grouped["mse"] = grouped.squared_error/grouped.points
    # No skip-NaN averaging: preserve invalid seasonal denominators explicitly.
    grouped["mase"] = frame.groupby(["dataset","configuration","term","model"]).mase.agg(lambda x: float(np.mean(x.to_numpy())))
    table = grouped.reset_index()
    naive = table[table.model.eq("seasonalnaive")].set_index("configuration")
    low = official_low_variance(upstream_leaderboard) | set(naive[naive.mase.eq(0)].dataset)
    table["partition"] = np.where(table.dataset.isin(low), "low_variance_unscaled", "nonzero_scaled")
    table["scaled_wql9"] = table.wql9/table.configuration.map(naive.wql9)
    undefined = table[~np.isfinite(table.wql9) | (~table.partition.eq("low_variance_unscaled") & ~np.isfinite(table.scaled_wql9))]
    summaries = {}
    for (partition, model), sub in table.groupby(["partition","model"]):
        metric = "wql9" if partition=="low_variance_unscaled" else "scaled_wql9"
        values = sub[metric]
        # No partial-subset leaderboard when any required configuration is invalid.
        aggregate = shifted_gmean(values) if np.isfinite(values).all() else None
        summaries.setdefault(partition,{})[model] = dict(configurations=len(sub),
            shifted_geometric_wql9=aggregate, task_wql9_mean=float(sub.wql9.mean()) if np.isfinite(sub.wql9).all() else None,
            invalid_configurations=int((~np.isfinite(values)).sum()))
    pair = table.pivot(index=["dataset","configuration"], columns="model", values="wql9")
    pair_difference = pair.timesfm3-pair.toto2_313m
    delta = pair_difference.groupby("dataset").agg(lambda x: float(np.mean(x.to_numpy())))
    # Diagnostic clustering is by task, not by the many overlapping origins.
    diagnostic = paired_cluster_interval(delta) if np.isfinite(delta).all() else None
    slices = frame.groupby(["model","term","target_constant"])[["numerator","denominator","atom_loss","points","above_q90"]].sum().reset_index()
    slices["wql9"] = slices.numerator/slices.denominator
    frame.to_parquet(dest/"case_scores.parquet",index=False)
    table.to_parquet(dest/"configuration_scores.parquet",index=False)
    slices.to_parquet(dest/"diagnostic_slices.parquet",index=False)
    result = dict(stage="native_development_comparison", tasks=table.dataset.nunique(),
        configurations=table.configuration.nunique(), variate_origins=len(frame)//frame.model.nunique(),
        scores=summaries, invalid_configurations=undefined[["configuration","model"]].to_dict("records"),
        timesfm_minus_toto_task_wql=diagnostic,
        undefined_naive_mase_configurations=naive.index[~np.isfinite(naive.mase)].tolist(),
        sota=False, scope="Development prefix with official horizon/window rules and full native variables. Foundation baselines only; no novel algorithm. No undefined-score imputation or partial leaderboard.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result
