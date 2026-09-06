"""Causal continuous-event representation probe; not a novel forecasting model."""
from collections import Counter
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from r1.data import boom_development, price_development
from r1.evaluation import paired_cluster_interval


def causal_features(values, domain, alpha=.06, threshold=1.5, floor_ratio=.001):
    y = np.asarray(values,dtype=float)
    if not np.isfinite(y).all() or (domain=="price" and np.any(y<=0)):
        raise ValueError("Nonfinite series or nonpositive price; no hidden imputation")
    x = np.log(y) if domain=="price" else y.copy()
    dx = np.r_[0.,np.diff(x)]
    variance = pd.Series(dx*dx).ewm(alpha=alpha,adjust=False).mean().to_numpy()
    expanding = np.cumsum(dx*dx)/np.maximum(1,np.arange(len(dx)))
    unconstrained = np.sqrt(variance)
    floor = floor_ratio*np.sqrt(expanding)
    scale = np.maximum(unconstrained,floor)
    previous_scale = np.r_[0.,scale[:-1]]
    state = np.abs(dx)>threshold*previous_scale
    indices = np.arange(len(x))
    changes = np.r_[True,state[1:]!=state[:-1]]
    age = indices-np.maximum.accumulate(np.where(changes,indices,0))+1
    z = np.divide(dx,scale,out=np.zeros_like(dx),where=scale>0)
    series, states = pd.Series(z),pd.Series(state.astype(float))
    frame = pd.DataFrame(dict(state=state.astype(float),increment=z,
        scale_zero=(scale==0).astype(float),floor_active=(unconstrained<floor).astype(float),
        historical_state_rate=states.expanding().mean()))
    for lag in (1,2,4,8,16):
        frame[f"increment_lag_{lag}"] = series.shift(lag)
        frame[f"state_lag_{lag}"] = states.shift(lag)
    for window in (16,32,96):
        frame[f"increment_mean_{window}"] = series.rolling(window).mean()
        frame[f"increment_std_{window}"] = series.rolling(window).std()
        frame[f"state_rate_{window}"] = states.rolling(window).mean()
    frame["log_age"] = np.log1p(age)
    frame["age_left_censored"] = (age==indices+1).astype(float)
    return frame,x,scale,state


def validate_continuous():
    rng = np.random.default_rng(19)
    y = np.r_[np.zeros(120),np.cumsum(rng.normal(size=180))]
    a,x,scale,state = causal_features(y,"boom")
    altered = y.copy(); altered[201:]+=rng.normal(size=len(y)-201)*100
    b,_,bs,bst = causal_features(altered,"boom")
    if not np.array_equal(a.iloc[:201].to_numpy(),b.iloc[:201].to_numpy(),equal_nan=True):
        raise AssertionError("Future perturbation changes past features")
    if not np.array_equal(scale[:201],bs[:201]) or not np.array_equal(state[:201],bst[:201]):
        raise AssertionError("Future perturbation changes past states/scales")
    c,_,cs,cst = causal_features(7*y,"boom")
    if not np.allclose(a.to_numpy(),c.to_numpy(),equal_nan=True) or not np.allclose(cs,7*scale) or not np.array_equal(state,cst):
        raise AssertionError("Unit scaling changes event representation")
    d,_,ds,dst = causal_features(y+31,"boom")
    if not np.allclose(a.to_numpy(),d.to_numpy(),equal_nan=True) or not np.allclose(ds,scale) or not np.array_equal(state,dst):
        raise AssertionError("Level translation changes increment representation")
    flat,_,fs,fst = causal_features(np.ones(150),"boom")
    if np.any(fs) or np.any(fst) or not np.isfinite(flat.iloc[96:].to_numpy()).all():
        raise AssertionError("Constant history contract failed")
    return dict(future_invariance=True,unit_equivariance=True,level_translation_invariance=True,
                flat_history_retained=True,event_scale="through t-1",target_scale="through t, frozen over horizon")


def prepare_continuous(repo_root,domain,horizons=(1,8,32),max_tasks=48,max_assets=24,
                       train_origins=64,validation_origins=32,alpha=.06,threshold=1.5,floor_ratio=.001):
    if domain=="boom":
        records,manifest = boom_development(repo_root,max_tasks=max_tasks,max_variates=1000000)
    elif domain=="price":
        records,manifest = price_development(repo_root,max_assets=max_assets)
    else:
        raise ValueError(domain)
    calendar_cutoff = None
    if domain=="price":
        calendar_cutoff = min(pd.Timestamp(s.dates[s.train_end]) for s in records)
        for s in records:
            dates = pd.to_datetime(s.dates)
            if not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError("Price dates must be unique and ordered")
            s.train_end = int(dates.searchsorted(calendar_cutoff))
    frames,skipped = [],[]
    maximum = max(horizons)
    for s in records:
        if not np.isfinite(s.values).all() or (domain=="price" and np.any(s.values<=0)):
            skipped.append(dict(dataset=s.dataset,item=s.item,reason="nonfinite_or_nonpositive_price"))
            continue
        features,x,scale,state = causal_features(s.values,domain,alpha,threshold,floor_ratio)
        added = False
        for partition,start,stop,limit in (("train",96,s.train_end-maximum,train_origins),
                ("validation",max(96,s.train_end),len(x)-maximum,validation_origins)):
            if stop<=start:
                continue
            origins = np.unique(np.linspace(start,stop-1,min(limit,stop-start),dtype=int))
            if partition=="train" and np.any(origins+maximum>=s.train_end):
                raise ValueError("Training label crossed cutoff")
            if partition=="validation" and np.any(origins<s.train_end):
                raise ValueError("Validation origin crossed cutoff")
            for horizon in horizons:
                out = features.iloc[origins].reset_index(drop=True).copy()
                target_scale = scale[origins]*np.sqrt(horizon)
                cumulative = x[origins+horizon]-x[origins]
                standardized = np.divide(cumulative,target_scale,out=np.zeros_like(cumulative),where=target_scale>0)
                # Undefined zero-scale transformed targets are excluded from
                # fitting this auxiliary regressor, not from event or raw scoring.
                out["target_transformed"] = np.where(target_scale>0,np.arcsinh(standardized),np.nan)
                out["event_label"] = state[origins+horizon].astype(float)
                out["raw_target"] = s.values[origins+horizon]
                out["origin_value"] = s.values[origins]
                out["origin_transformed"] = x[origins]
                out["target_scale"] = target_scale
                out["horizon"] = horizon
                out["partition"] = partition
                out["dataset"] = s.dataset
                out["item"] = s.item
                out["origin"] = origins
                out["cluster"] = s.dataset if domain=="boom" else [str(pd.Timestamp(s.dates[t]).to_period("M")) for t in origins]
                if domain=="price":
                    out["origin_date"] = [str(pd.Timestamp(s.dates[t]).date()) for t in origins]
                    out["label_date"] = [str(pd.Timestamp(s.dates[t+horizon]).date()) for t in origins]
                else:
                    out["origin_date"] = ""
                    out["label_date"] = ""
                frames.append(out); added = True
        if not added:
            skipped.append(dict(dataset=s.dataset,item=s.item,reason="no_origin_after_96_history_and_max_horizon_purge"))
    if not frames:
        raise ValueError("No eligible continuous-event cases")
    frame = pd.concat(frames,ignore_index=True)
    if domain=="price":
        latest = frame.loc[frame.partition.eq("train"),"label_date"].max()
        earliest = frame.loc[frame.partition.eq("validation"),"origin_date"].min()
        if latest>=earliest:
            raise ValueError("Pooled price training crossed common calendar cutoff")
    metadata = dict(sources=manifest,skipped=skipped,calendar_cutoff=str(calendar_cutoff) if calendar_cutoff is not None else None,
                    train_cases=int(frame.partition.eq("train").sum()),validation_cases=int(frame.partition.eq("validation").sum()),
                    tasks=frame.dataset.nunique(),series=frame.groupby(["dataset","item"]).ngroups,
                    zero_scale_cases=int(frame.target_scale.eq(0).sum()),floor_active_cases=int(frame.floor_active.sum()))
    return frame,metadata


def random_age_features(train,valid,seed):
    """Sample paired age/censor features from training-only state/task pools."""
    rng = np.random.default_rng(seed)
    age_columns = ["log_age","age_left_censored"]
    pools = {(dataset,state):sub[age_columns].to_numpy() for (dataset,state),sub in train.groupby(["dataset","state"])}
    fallback = {state:sub[age_columns].to_numpy() for state,sub in train.groupby("state")}
    all_pool = train[age_columns].to_numpy()
    outputs = []
    for frame in (train,valid):
        shuffled = frame.copy()
        for (dataset,state),index in frame.groupby(["dataset","state"]).groups.items():
            pool = pools.get((dataset,state),fallback.get(state,all_pool))
            shuffled.loc[index,age_columns] = pool[rng.integers(len(pool),size=len(index))]
        outputs.append(shuffled)
    return outputs


def fit_continuous(frame,domain,output_dir,seeds=(2021,2022,2023),max_iter=80,threads=4,
                   age_columns=("log_age","age_left_censored"),own_columns=None,
                   randomizer=None,deterministic_once=False):
    dest = Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    metadata = {"target_transformed","event_label","raw_target","origin_value","origin_transformed",
                "target_scale","partition","dataset","item","origin","cluster","origin_date","label_date","native_group"}
    full = [k for k in frame if k not in metadata]
    base = [k for k in full if k not in age_columns]
    if own_columns is not None and not set(own_columns).issubset(base):
        raise ValueError("Own-context features must be a subset of the context arm")
    train = frame[frame.partition.eq("train")].reset_index(drop=True)
    valid = frame[frame.partition.eq("validation")].reset_index(drop=True)
    if not np.isfinite(frame[full].to_numpy()).all():
        raise ValueError("Nonfinite predictor; no silent dropped rows")
    counts = train.groupby("dataset").size()
    weights = train.dataset.map(lambda d:1/counts[d]).to_numpy()
    weights = weights*(len(weights)/weights.sum())
    reg_mask = train.target_scale.gt(0).to_numpy()
    reg_weights = weights[reg_mask].copy()
    # Equal dataset fitting weight after the explicit undefined-scale exclusion.
    reg_counts = train.loc[reg_mask].groupby("dataset").size()
    reg_weights = train.loc[reg_mask,"dataset"].map(lambda d:1/reg_counts[d]).to_numpy()
    reg_weights = reg_weights*(len(reg_weights)/reg_weights.sum())
    levels = np.arange(.1,1,.1)
    rows,timings = [],[]
    for seed in seeds:
        random_train,random_valid = (randomizer or random_age_features)(train,valid,seed)
        arms = ["context","true_age","random_age"]
        if own_columns is not None:
            arms.append("own_context")
        if deterministic_once and seed!=seeds[0]:
            arms = ["random_age"]
        for arm in np.random.default_rng(seed).permutation(arms):
            features = own_columns if arm=="own_context" else base if arm=="context" else full
            tr,va = (random_train,random_valid) if arm=="random_age" else (train,valid)
            params = dict(max_iter=max_iter,max_leaf_nodes=15,learning_rate=.06,l2_regularization=1.,
                          min_samples_leaf=40,early_stopping=False,random_state=seed)
            tick = time.perf_counter()
            with threadpool_limits(limits=threads):
                classifier = HistGradientBoostingClassifier(**params).fit(tr[features],train.event_label,sample_weight=weights)
                probabilities = classifier.predict_proba(va[features])[:,1]
                transformed_quantiles = []
                for level in levels:
                    regressor = HistGradientBoostingRegressor(loss="quantile",quantile=float(level),**params)
                    regressor.fit(tr.loc[reg_mask,features],train.loc[reg_mask,"target_transformed"],sample_weight=reg_weights)
                    transformed_quantiles.append(regressor.predict(va[features]))
                    if round(float(level),1) in (.1,.5,.9):
                        print(f"Continuous {domain}: {arm} seed={seed}, quantile={level:.1f}",flush=True)
            q = np.stack(transformed_quantiles,-1)
            crossing = np.mean(np.diff(q,axis=-1)<0,axis=-1)
            # Identical monotone rearrangement for every arm, disclosed in scores.
            q = np.sort(q,axis=-1)
            delta = np.sinh(q)*valid.target_scale.to_numpy()[:,None]
            raw = valid.origin_transformed.to_numpy()[:,None]+delta
            raw = np.exp(raw) if domain=="price" else raw
            if not np.isfinite(raw).all():
                raise ValueError("Nonfinite inverse forecast; no hidden clipping or fallback")
            target = valid.raw_target.to_numpy()
            err = target[:,None]-raw
            loss = (2*np.maximum(levels*err,(levels-1)*err)).mean(-1)
            trans_error = valid.target_transformed.to_numpy()[:,None]-q
            transformed_loss = (2*np.maximum(levels*trans_error,(levels-1)*trans_error)).mean(-1)
            result = valid[["dataset","item","horizon","origin","cluster","target_scale","floor_active","state","origin_date","label_date"]].copy()
            for column in ("peer_count","peer_left_censored","native_group"):
                if column in valid:
                    result[column] = valid[column].to_numpy()
            result["seed"],result["arm"] = seed,arm
            result["brier"] = (probabilities-valid.event_label.to_numpy())**2
            result["event_probability"] = probabilities
            result["event_label"] = valid.event_label.to_numpy()
            result["transformed_pinball9"] = transformed_loss
            result["raw_pinball9"] = loss
            result["absolute_target"] = np.abs(target)
            result["median_squared_error"] = (target-raw[:,4])**2
            result["crossing_before_sort"] = crossing
            if domain=="price":
                result["cumulative_return_pinball9"] = loss/valid.origin_value.to_numpy()
                result["absolute_cumulative_return"] = np.abs(target/valid.origin_value.to_numpy()-1)
            rows.append(result)
            np.savez_compressed(dest/f"{arm}_{seed}_predictions.npz",keys=valid[["dataset","item","origin","horizon"]].to_numpy(dtype=str),quantiles=raw,event_probability=probabilities)
            seconds = time.perf_counter()-tick
            timings.append(dict(arm=arm,seed=seed,seconds=seconds))
            print(f"Continuous {domain}: {arm} seed={seed} completed in {seconds:.1f}s",flush=True)
    combined = pd.concat(rows,ignore_index=True)
    combined.to_parquet(dest/"validation_scores.parquet",index=False)
    summaries = []
    # Average optimization repeats inside each observational cluster first.
    for horizon,sub in combined.groupby("horizon"):
        for metric in ("brier","transformed_pinball9"):
            eligible = sub if metric=="brier" else sub[sub.target_scale.gt(0)]
            by_cluster = eligible.groupby(["cluster","arm"])[metric].mean().unstack("arm")
            if not np.isfinite(by_cluster.to_numpy()).all():
                raise ValueError("Incomplete paired cluster comparison")
            entry = dict(horizon=int(horizon),metric=metric,clusters=len(by_cluster),
                         scores=by_cluster.mean().to_dict(),age_minus_context_mean=float((by_cluster.true_age-by_cluster.context).mean()),
                         age_minus_random_mean=float((by_cluster.true_age-by_cluster.random_age).mean()))
            if domain=="boom":
                entry["age_minus_context_ci"] = paired_cluster_interval(by_cluster.true_age-by_cluster.context)
                entry["age_minus_random_ci"] = paired_cluster_interval(by_cluster.true_age-by_cluster.random_age)
            else:
                entry["ci"] = None
                entry["ci_reason"] = "Month clusters remain serially dependent; no iid-month interval. Need longer independent history or adequate common calendar blocks."
            if own_columns is not None:
                for other in ("context","true_age"):
                    difference = by_cluster[other]-by_cluster.own_context
                    entry[f"{other}_minus_own_mean"] = float(difference.mean())
                    if domain=="boom":
                        entry[f"{other}_minus_own_ci"] = paired_cluster_interval(difference)
            summaries.append(entry)
    sums = combined.groupby(["dataset","horizon","arm","seed"])[["raw_pinball9","absolute_target","median_squared_error"]].sum()
    sums["endpoint_wql9"] = sums.raw_pinball9/sums.absolute_target
    sums.to_parquet(dest/"task_endpoint_scores.parquet")
    undefined = sums[~np.isfinite(sums.endpoint_wql9)].reset_index()[["dataset","horizon","arm","seed"]]
    finite_by_arm = sums.groupby("arm").endpoint_wql9.agg(lambda x:float(x.mean()) if np.isfinite(x).all() else None)
    zero_cases = combined[combined.target_scale.eq(0)].groupby(["dataset","item","origin","horizon","arm"],as_index=False).raw_pinball9.mean()
    zero = zero_cases.groupby("arm").agg(cases=("raw_pinball9","size"),raw_pinball_sum=("raw_pinball9","sum"))
    endpoint_by_horizon = {int(h):group.groupby("arm").endpoint_wql9.agg(
        lambda x:float(x.mean()) if np.isfinite(x).all() else None).to_dict()
        for h,group in sums.reset_index().groupby("horizon")}
    result = dict(stage="continuous_event_representation_probe",domain=domain,train_cases=len(train),validation_cases=len(valid),
        tasks=frame.dataset.nunique(),series=frame.groupby(["dataset","item"]).ngroups,seeds=list(seeds),
        scores=summaries,raw_endpoint_wql9={k:None if pd.isna(v) else float(v) for k,v in finite_by_arm.items()},
        raw_endpoint_wql9_by_horizon=endpoint_by_horizon,
        undefined_raw_endpoint_configurations=undefined.to_dict("records"),
        zero_scale_validation_cases=int(valid.target_scale.eq(0).sum()),zero_scale_scores=zero.reset_index().to_dict("records"),
        seconds=timings,sota=False,
        feature_columns=dict(context=base,true_age=full,random_age=full,**({"own_context":list(own_columns)} if own_columns is not None else {})),
        deterministic_arms_fit_once=deterministic_once,
        scope="Development representation probe. Event uses future rolling threshold as its label; cumulative quantiles use origin-frozen scale and asinh training transform. Three horizons are not native full BOOM. Zero-scale cases remain in event/raw scoring with degenerate increment predictions, but their undefined transformed target is excluded from auxiliary regression/scoring. No novel algorithm or duration-amplitude identification claim.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def continuous_probe(repo_root,cache_dir,domain="boom",horizons=(1,8,32),max_tasks=48,max_assets=24,
                     train_origins=64,validation_origins=32,alpha=.06,threshold=1.5,floor_ratio=.001,
                     seeds=(2021,2022,2023),max_iter=80,threads=4):
    from r1.native_workflow import campaign
    from r1.workflow import emit,save_json
    checks = validate_continuous()
    frame,metadata = prepare_continuous(repo_root,domain,horizons,max_tasks,max_assets,train_origins,
                                         validation_origins,alpha,threshold,floor_ratio)
    config = dict(domain=domain,horizons=list(horizons),max_tasks=max_tasks,max_assets=max_assets,
                  train_origins=train_origins,validation_origins=validation_origins,alpha=alpha,
                  threshold=threshold,floor_ratio=floor_ratio,seeds=list(seeds),max_iter=max_iter,
                  threads=threads,data=metadata,validation=checks,
                  primary="H32 Brier plus nonzero-scale transformed pinball; both true-age vs context and true-age vs training-pool random-age must improve. Other horizons are sensitivity checks, not alternative success endpoints.")
    dest = campaign(repo_root,cache_dir,"continuous_probe_"+domain,config)
    save_json(dest/"data_contract.json",metadata)
    frame.to_parquet(dest/"cases.parquet",index=False)
    print(f"Continuous {domain}: {len(frame)} rows; train={metadata['train_cases']}; validation={metadata['validation_cases']}; tasks={metadata['tasks']}; series={metadata['series']}",flush=True)
    result = fit_continuous(frame,domain,dest,seeds,max_iter,threads)
    return emit(dict(cache=str(dest),data=metadata,validation=checks,**result))
