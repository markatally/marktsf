"""Frozen-case, leave-one-channel-out event-age probe; no new model claim."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from r1.continuous_probe import causal_features, fit_continuous
from r1.data import sha256


def peer_features(features, target, event_fraction=.25):
    """Exclude the target at every historical time before computing age/lags."""
    if not 0 < event_fraction <= 1:
        raise ValueError("Invalid common-event threshold")
    length = len(features[target])
    if any(len(f)!=length for f in features):
        raise ValueError("Native channels must have an identical time grid")
    others = [f for i,f in enumerate(features) if i!=target]
    count = len(others)
    if count:
        z = np.stack([f.increment.to_numpy() for f in others])
        events = np.stack([f.state.to_numpy(dtype=np.int64) for f in others])
        event_count = events.sum(axis=0,dtype=np.int64)
        rate = event_count/count
        # Integer counts prevent self-subtraction rounding from changing states.
        state = event_count>=int(np.ceil(event_fraction*count))
        mean,absolute,std = z.mean(0),np.abs(z).mean(0),z.std(0)
        floor = np.stack([f.floor_active.to_numpy() for f in others]).mean(0)
        zero = np.stack([f.scale_zero.to_numpy() for f in others]).mean(0)
    else:
        rate = mean = absolute = std = floor = zero = np.zeros(length)
        state = np.zeros(length,dtype=bool)
    indices = np.arange(length)
    changes = np.r_[True,state[1:]!=state[:-1]]
    age = indices-np.maximum.accumulate(np.where(changes,indices,0))+1
    out = pd.DataFrame(dict(peer_count=np.full(length,count),peer_available=float(count>0),
        peer_members_changed=np.zeros(length),history_log_length=np.log1p(indices+1),
        peer_increment_mean=mean,peer_increment_abs=absolute,peer_increment_std=std,
        peer_event_rate=rate,peer_floor_rate=floor,peer_zero_scale_rate=zero,
        state=state.astype(float),peer_left_censored=(age==indices+1).astype(float) if count else np.zeros(length),
        log_age=np.log1p(age) if count else np.zeros(length)))
    for lag in (1,2,4,8,16):
        out[f"peer_increment_lag_{lag}"] = pd.Series(mean).shift(lag)
        out[f"peer_event_rate_lag_{lag}"] = pd.Series(rate).shift(lag)
    for window in (16,32,96):
        out[f"peer_increment_mean_{window}"] = pd.Series(mean).rolling(window).mean()
        out[f"peer_event_rate_{window}"] = pd.Series(rate).rolling(window).mean()
    return out


def validate_peer_features():
    rng = np.random.default_rng(402)
    values = np.cumsum(rng.normal(size=(4,260)),axis=1)
    features = [causal_features(y,"boom")[0] for y in values]
    original = peer_features(features,0)
    changed = values.copy(); changed[:,181:]+=rng.normal(size=(4,79))*50
    future = peer_features([causal_features(y,"boom")[0] for y in changed],0)
    np.testing.assert_allclose(original.iloc[:181],future.iloc[:181],equal_nan=True)
    changed = values.copy(); changed[0] = rng.normal(size=260)*1000
    own = peer_features([causal_features(y,"boom")[0] for y in changed],0)
    np.testing.assert_allclose(original,own,equal_nan=True,rtol=0,atol=0)
    permuted = peer_features([features[i] for i in (2,0,3,1)],1)
    np.testing.assert_allclose(original,permuted,equal_nan=True,atol=1e-14)
    empty = peer_features([features[0]],0).iloc[96:]
    ignored = {"history_log_length"}
    if empty[[k for k in empty if k not in ignored]].to_numpy().any():
        raise AssertionError("No-peer group must have zero peer features and age")
    single = peer_features(features[:2],0)
    np.testing.assert_array_equal(single.state,features[1].state)
    train = pd.DataFrame(dict(native_group=["g","g","g","empty"],state=[0.,0.,1.,0.],
        peer_left_censored=[0.,0.,0.,0.],peer_count=[2.,2.,2.,0.],log_age=[1.,2.,3.,0.]))
    valid = train.copy(); valid.loc[0,"native_group"] = "unseen"
    a,b = random_peer_age(train,valid,71)
    changed = valid.copy(); changed["log_age"] = 1000000.
    _,other = random_peer_age(train,changed,71)
    pd.testing.assert_frame_equal(b,other)
    pd.testing.assert_frame_equal(a.drop(columns="log_age"),train.drop(columns="log_age"))
    pd.testing.assert_frame_equal(b.drop(columns="log_age"),valid.drop(columns="log_age"))
    if b.loc[b.peer_count.eq(0),"log_age"].any():
        raise AssertionError("Random age created phantom peers")
    return dict(future_invariance=True,complete_history_target_exclusion=True,
                peer_permutation_equivariance=True,no_peer_zero_representation=True,
                single_peer_state_identity=True,random_age_uses_training_only=True,
                placebo_changes_only_age=True)


def native_groups(repo_root,base,contract,domain):
    """Recover original row groups and freeze the previously selected universe."""
    root = Path(repo_root)
    wanted = set(zip(base.dataset,base.item))
    groups = []
    for source in contract["sources"]:
        dataset = source["dataset"]
        if domain=="boom":
            for relative,digest in source["hashes"].items():
                path = root/relative
                if sha256(path)!=digest:
                    raise ValueError("BOOM source changed since the fixed-case probe")
                with path.open("rb") as stream:
                    rows = ipc.open_stream(stream).read_all().to_pylist()
                for index,row in enumerate(rows):
                    target = np.asarray(row["target"],dtype=float)
                    keys = [str(row["item_id"])] if target.ndim==1 else [f'{row["item_id"]}:v{v}' for v in range(len(target))]
                    matrix = target[None,:] if target.ndim==1 else target
                    if matrix.ndim!=2:
                        raise ValueError("Unexpected native target shape")
                    active = [i for i,key in enumerate(keys) if (dataset,key) in wanted]
                    if not active:
                        continue
                    if len(active)!=len(keys):
                        raise ValueError("Partial native group requires an explicit missing-member protocol")
                    prefix = matrix[:,:int(matrix.shape[1]*.7)]
                    if not np.isfinite(prefix).all():
                        raise ValueError("Peer history contains missing values; no future-based filling")
                    groups.append((dataset,f"{relative}:row{index}",keys,prefix,None))
        elif domain=="price":
            path = root/"input"/dataset/f"{dataset}.csv"
            if sha256(path)!=source["sha256"]:
                raise ValueError("Price source changed since the fixed-case probe")
            frame = pd.read_csv(path).iloc[:source["development_rows"]]
            dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:,0]))
            if not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError("Dates must be unique and ordered")
            # Original exclusions are inherited, not reselected using this run.
            keys = [key for key in source["columns"] if (dataset,key) in wanted]
            prefix = frame.set_index(dates)[keys].reindex(dates).to_numpy(dtype=float).T
            if not np.isfinite(prefix).all() or np.any(prefix<=0):
                raise ValueError("Frozen price peers have invalid same-date observations")
            groups.append((dataset,f"{dataset}:same_close",keys,prefix,dates))
        else:
            raise ValueError(domain)
    recovered = [(dataset,key) for dataset,_,keys,_,_ in groups for key in keys]
    if len(recovered)!=len(set(recovered)) or set(recovered)!=wanted:
        raise ValueError("Original row groups did not recover exactly the fixed target universe")
    return groups


def enrich_cases(repo_root,base_dir,event_fraction=.25):
    directory = Path(base_dir)
    base = pd.read_parquet(directory/"cases.parquet")
    contract = json.loads((directory/"data_contract.json").read_text())
    config = json.loads((directory/"contract.json").read_text())["config"]
    domain = config["domain"]
    if tuple(config["horizons"])!=(1,8,32):
        raise ValueError("This probe requires the frozen 1/8/32 case contract")
    groups = native_groups(repo_root,base,contract,domain)
    result = base.rename(columns={"log_age":"own_log_age","age_left_censored":"own_age_left_censored","state":"own_state"}).copy()
    protected = list(result.columns)
    group_records = []
    for dataset,native,keys,matrix,dates in groups:
        decoded = [causal_features(y,domain,config["alpha"],config["threshold"],config["floor_ratio"]) for y in matrix]
        own_features = [d[0] for d in decoded]
        for v,key in enumerate(keys):
            index = base.index[base.dataset.eq(dataset)&base.item.eq(key)]
            sub = base.loc[index]
            origins,horizons = sub.origin.to_numpy(),sub.horizon.to_numpy()
            features,x,scale,state = decoded[v]
            np.testing.assert_allclose(sub[features.columns],features.iloc[origins],rtol=0,atol=0)
            np.testing.assert_array_equal(sub.raw_target,matrix[v,origins+horizons])
            np.testing.assert_array_equal(sub.origin_value,matrix[v,origins])
            np.testing.assert_array_equal(sub.event_label,state[origins+horizons])
            np.testing.assert_array_equal(sub.target_scale,scale[origins]*np.sqrt(horizons))
            if dates is not None:
                np.testing.assert_array_equal(sub.origin_date,dates[origins].strftime("%Y-%m-%d"))
                np.testing.assert_array_equal(sub.label_date,dates[origins+horizons].strftime("%Y-%m-%d"))
            peer = peer_features(own_features,v,event_fraction).iloc[origins]
            for col in peer:
                if col in protected:
                    raise ValueError(f"Peer column would overwrite original data: {col}")
                result.loc[index,col] = peer[col].to_numpy()
            result.loc[index,"native_group"] = native
        group_records.append(dict(dataset=dataset,native_group=native,channels=len(keys),items=keys,
                                  same_grid=True,member_changes=0))
    # Metadata strings never enter HGB. The own arm gets count/history controls.
    metadata = {"target_transformed","event_label","raw_target","origin_value","origin_transformed",
                "target_scale","partition","dataset","item","origin","cluster","origin_date","label_date"}
    own_columns = [k for k in protected if k not in metadata]+["peer_count","peer_available","peer_members_changed","history_log_length"]
    pd.testing.assert_frame_equal(result[protected],base.rename(columns={"log_age":"own_log_age","age_left_censored":"own_age_left_censored","state":"own_state"}))
    coverage = result[result.partition.eq("validation")].groupby("peer_count").agg(cases=("item","size"),tasks=("dataset","nunique"))
    record = dict(domain=domain,base_dir=str(directory),base_cases_sha=sha256(directory/"cases.parquet"),
        base_contract_sha=sha256(directory/"contract.json"),groups=group_records,
        own_columns=own_columns,validation_peer_coverage=coverage.reset_index().to_dict("records"),
        target_cases_unchanged=True,source_values_and_features_exactly_reproduced=True,
        calendar_cutoff=contract["calendar_cutoff"],inherited_exclusions=contract["skipped"],
        universe="All channels in each original BOOM row; original valid selected price assets within their own market. Original development-prefix missingness exclusions are inherited, not independent prospective cohort selection.")
    return result,record


def random_peer_age(train,valid,seed,coverage=None):
    """Randomize age only; common state and censor indicator stay in all peer arms."""
    rng = np.random.default_rng(seed)
    keys = ["native_group","state","peer_left_censored"]
    pools = {key:sub.log_age.to_numpy() for key,sub in train.groupby(keys)}
    # Match current state and censoring when the native group has no train pool.
    fallback = {key:sub.log_age.to_numpy() for key,sub in train[train.peer_count.gt(0)].groupby(keys[1:])}
    global_pool = train.loc[train.peer_count.gt(0),"log_age"].to_numpy()
    outputs = []
    for partition,frame in (("train",train),("validation",valid)):
        out = frame.copy()
        counts = dict(exact=0,state_censor_fallback=0,global_fallback=0,no_peer=0)
        for key,index in frame.groupby(keys).groups.items():
            if frame.loc[index,"peer_count"].eq(0).all():
                out.loc[index,"log_age"] = 0.
                counts["no_peer"]+=len(index)
                continue
            if key in pools:
                pool,source = pools[key],"exact"
            elif key[1:] in fallback:
                pool,source = fallback[key[1:]],"state_censor_fallback"
            else:
                pool,source = global_pool,"global_fallback"
            if not len(pool):
                raise ValueError("No eligible training-only age pool")
            out.loc[index,"log_age"] = pool[rng.integers(len(pool),size=len(index))]
            counts[source]+=len(index)
        if sum(counts.values())!=len(frame):
            raise AssertionError("Random-age pool did not cover every original case")
        if coverage is not None:
            coverage.append(dict(seed=seed,partition=partition,**counts))
        outputs.append(out)
    return outputs


def peer_probe(repo_root,cache_dir,base_dir,event_fraction=.25,seeds=(2021,2022,2023),max_iter=80,threads=4):
    from r1.native_workflow import campaign
    from r1.workflow import emit,save_json
    checks = validate_peer_features()
    frame,data = enrich_cases(repo_root,base_dir,event_fraction)
    coverage = []
    def randomizer(train,valid,seed):
        return random_peer_age(train,valid,seed,coverage)
    config = dict(data=data,validation=checks,event_fraction=event_fraction,seeds=list(seeds),max_iter=max_iter,threads=threads,
        arms={"own_context":"own history including own age plus peer count/availability and observed length",
              "context":"own context plus ordinary LOO peer moments, event rates, lags and common state/censor",
              "true_age":"context plus common state log age only",
              "random_age":"same dimensions as true_age; age from training-only native-group/state/censor pools"},
        primary="H32 Brier and transformed cumulative pinball: true age must improve over both peer context and random age. Raw endpoints must also not degrade relative to own context. H1/H8 secondary. All outcomes are development evidence, not independent confirmation or SOTA.",
        deterministic_policy="Fit own/context/true_age once; repeat only random_age for its three placebo seeds.")
    dest = campaign(repo_root,cache_dir,"peer_probe_"+data["domain"],config)
    frame.to_parquet(dest/"cases.parquet",index=False)
    save_json(dest/"data_contract.json",data)
    print(f"Peer {data['domain']}: {len(frame)} fixed cases, {len(data['groups'])} native groups; cache={dest}",flush=True)
    result = fit_continuous(frame,data["domain"],dest,seeds,max_iter,threads,age_columns=("log_age",),
                            own_columns=data["own_columns"],randomizer=randomizer,deterministic_once=True)
    result.update(stage="leave_one_out_common_event_age_probe",random_age_pool_coverage=coverage,
                  scope=config["primary"],arms=config["arms"])
    save_json(dest/"summary.json",result)
    return emit(dict(cache=str(dest),data={k:v for k,v in data.items() if k not in ("groups","own_columns")},validation=checks,**result))
