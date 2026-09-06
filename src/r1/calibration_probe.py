"""Causal calibration controls and explicitly hindsight-only path diagnostics."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from r1.aggregation import shifted_gmean
from r1.data import sha256
from r1.evaluation import paired_cluster_interval


def calibration_grid(biases,spreads):
    if not np.isfinite(list(biases)+list(spreads)).all() or any(s<0 for s in spreads) or 0 not in biases or 1 not in spreads:
        raise ValueError("Grid must contain identity and preserve quantile order")
    grid = sorted({(float(b),float(s)) for b in biases for s in spreads},
                  key=lambda p:(abs(p[0])+abs(p[1]-1),abs(p[0]),p[0],p[1]))
    if grid[0]!=(0.,1.):
        raise AssertionError("Identity must win exact loss ties")
    return grid


def causal_choices(history,origin,items,minimum_windows=2,recent_windows=3):
    """Only completed earlier paths may select the current calibration."""
    available = [r for r in history if r["last_label"]<=origin]
    pooled = np.sum([r["loss"].sum(axis=1) for r in available],axis=0) if available else None
    enough = len({r["origin"] for r in available})>=minimum_windows
    output = {"causal_pooled":np.full(len(items),int(np.argmin(pooled)) if enough else 0,dtype=int),
              "causal_channel":np.zeros(len(items),dtype=int),"causal_recent3":np.zeros(len(items),dtype=int)}
    support = {"causal_pooled":np.full(len(items),len({r["origin"] for r in available}),dtype=int),
               "causal_channel":np.zeros(len(items),dtype=int),"causal_recent3":np.zeros(len(items),dtype=int)}
    for v,item in enumerate(items):
        previous = [(r["origin"],r["loss"][:,r["items"].index(item)]) for r in available if item in r["items"]]
        previous.sort(key=lambda p:p[0])
        for arm,subset in (("causal_channel",previous),("causal_recent3",previous[-recent_windows:])):
            support[arm][v] = len(subset)
            if len(subset)>=minimum_windows:
                output[arm][v] = int(np.argmin(np.sum([p[1] for p in subset],axis=0)))
    return output,support


def validate_calibration():
    grid = calibration_grid([-.5,0,.5],[0.,1.,2.])
    g = len(grid)
    prior = dict(origin=10,last_label=20,items=["a"],loss=np.arange(g,dtype=float)[::-1,None])
    pending = dict(origin=19,last_label=29,items=["a"],loss=np.arange(g,dtype=float)[:,None]*1e6)
    first,_ = causal_choices([prior,pending],20,["a"],minimum_windows=1)
    pending["loss"] = -pending["loss"]
    second,_ = causal_choices([prior,pending],20,["a"],minimum_windows=1)
    for arm in first:
        np.testing.assert_array_equal(first[arm],second[arm])
        np.testing.assert_array_equal(first[arm],[g-1])
    cold,_ = causal_choices([prior],19,["a"],minimum_windows=1)
    if any(v.any() for v in cold.values()):
        raise AssertionError("Incomplete first path must not inform calibration")
    q = np.linspace(-1,1,9)
    for b,s in grid:
        altered = q[4]+b*(q[-1]-q[0])/2+s*(q-q[4])
        if np.any(np.diff(altered)<0):
            raise AssertionError("Calibration changed quantile order")
    return dict(pending_future_labels_cannot_select_parameters=True,
                labels_at_current_origin_are_available=True,cold_start_identity=True,
                all_grid_quantiles_monotone=True,identity_tie_priority=True)


def segment_oracles(losses,event_cut):
    """All returned choices see the target; never use them as predictive arms."""
    cumulative = losses.cumsum(axis=1)
    total = cumulative[:,-1]
    whole = int(np.argmin(total))
    h = losses.shape[1]
    def split_at(cut):
        left = cumulative[:,cut-1]
        right = total-left
        i,j = int(np.argmin(left)),int(np.argmin(right))
        return np.r_[losses[i,:cut],losses[j,cut:]],cut,i,j
    fixed = split_at(h//2)
    event = split_at(event_cut) if event_cut is not None else fixed
    split_losses = cumulative[:,:-1].min(axis=0)+(total[:,None]-cumulative[:,:-1]).min(axis=0)
    best = split_at(int(np.argmin(split_losses))+1)
    return {"oracle_path":(losses[whole],h,whole,whole),"oracle_fixed_half":fixed,
            "oracle_event_boundary":event,"oracle_best_cut":best}


def run_calibration(data_dir,prediction_dir,comparison_dir,output_dir,biases,spreads,
                    minimum_windows=2,recent_windows=3):
    root,pred,dest = Path(data_dir),Path(prediction_dir),Path(output_dir)
    groups = pd.read_parquet(root/"groups.parquet").sort_values(["configuration","origin","item"])
    if groups.groupby("configuration").item.nunique().gt(1).any():
        raise ValueError("Cross-row pooling requires a separately verified common calendar")
    if minimum_windows<1 or recent_windows<minimum_windows:
        raise ValueError("Require positive minimum windows and an adequate recent window pool")
    reference = pd.read_parquet(Path(comparison_dir)/"configuration_scores.parquet")
    baseline = reference[reference.model.eq("toto2_313m")].set_index("configuration")
    naive = reference[reference.model.eq("seasonalnaive")].set_index("configuration")
    grid = calibration_grid(biases,spreads)
    levels = np.arange(.1,1,.1)
    histories,rows = {},[]
    start = time.perf_counter()
    for number,group in enumerate(groups.itertuples(),1):
        with np.load(root/"inputs"/group.file) as data,np.load(pred/group.file) as saved:
            if sha256(root/"inputs"/group.file)!=group.input_sha or str(saved["input_sha"])!=group.input_sha:
                raise ValueError("Native input or prediction binding changed")
            np.testing.assert_array_equal(saved["keys"],data["keys"])
            np.testing.assert_array_equal(saved["target_hashes"],data["target_hashes"])
            y = data["target"].astype(float)
            q = saved["quantiles"].astype(float)
            if q.shape!=y.shape+(9,) or not np.isfinite(q).all() or not np.isfinite(y).all():
                raise ValueError("Invalid complete-path forecasts or targets")
            if np.any(np.diff(q,axis=-1)<0):
                raise ValueError("Baseline has crossing quantiles; no hidden sorting")
            items = list(data["keys"][:,1].astype(str))
            history = histories.setdefault(group.configuration,[])
            # No current labels, current losses, or future event cuts enter this call.
            choices,support = causal_choices(history,int(group.origin),items,minimum_windows,recent_windows)
            midpoint = q[...,4,None]
            halfwidth = (q[...,-1,None]-q[...,0,None])/2
            residual = q-midpoint
            losses = []
            for bias,spread in grid:
                altered = q if (bias,spread)==(0.,1.) else midpoint+bias*halfwidth+spread*residual
                error = y[...,None]-altered
                losses.append((2*np.maximum(levels*error,(levels-1)*error)).mean(-1))
            losses = np.stack(losses)
            for v,item in enumerate(items):
                atom,state,age,rate = data["features"][v]
                changed = np.flatnonzero((y[v]==atom)!=bool(state)) if np.isfinite(atom) else np.array([],dtype=int)
                first = int(changed[0]) if len(changed) else None
                internal = first if first is not None and 0<first<group.horizon else None
                event = "no_historical_atom" if not np.isfinite(atom) else (
                    ("atom_exit" if len(changed) else "atom_stay") if state else
                    ("active_enters_atom" if len(changed) else "active_stays_active"))
                paths = {"toto_raw":(losses[0,v],group.horizon,0,0)}
                paths.update({arm:(losses[index[v],v],group.horizon,int(index[v]),int(index[v])) for arm,index in choices.items()})
                paths.update(segment_oracles(losses[:,v],internal))
                for arm,(loss,cut,left,right) in paths.items():
                    rows.append(dict(dataset=group.dataset,configuration=group.configuration,term=group.term,
                        item=item,origin=int(group.origin),horizon=int(group.horizon),model=arm,
                        numerator=float(loss.sum()),denominator=float(np.abs(y[v]).sum()),points=len(y[v]),
                        event=event,internal_event_boundary=internal is not None,first_change=first,
                        past_complete_windows=int(support[arm][v]) if arm in support else 0,
                        warm=bool(support[arm][v]>=minimum_windows) if arm in support else False,
                        cut=int(cut),left_grid=left,right_grid=right,
                        uses_future_labels=arm.startswith("oracle_"),
                        baseline_below_q10=int((y[v]<q[v,:,0]).sum()),baseline_above_q90=int((y[v]>q[v,:,-1]).sum())))
            history.append(dict(origin=int(group.origin),last_label=int(group.origin+group.horizon),
                                items=items,loss=losses.sum(axis=-1)))
        if number%100==0 or number==len(groups):
            print(f"Native calibration: {number}/{len(groups)} groups",flush=True)
    frame = pd.DataFrame(rows)
    table = frame.groupby(["dataset","configuration","term","model"])[["numerator","denominator","points"]].sum().reset_index()
    table["wql9"] = table.numerator/table.denominator
    table["partition"] = table.configuration.map(baseline.partition)
    table["scaled_wql9"] = table.wql9/table.configuration.map(naive.wql9)
    reproduced = table[table.model.eq("toto_raw")].set_index("configuration")
    np.testing.assert_allclose(reproduced.wql9.reindex(baseline.index),baseline.wql9,rtol=1e-11,atol=1e-12)
    summaries = {}
    for (partition,arm),sub in table.groupby(["partition","model"]):
        metric = "wql9" if partition=="low_variance_unscaled" else "scaled_wql9"
        if not np.isfinite(sub[metric]).all():
            raise ValueError("Incomplete aggregate; no silent configuration removal")
        summaries.setdefault(partition,{})[arm] = dict(configurations=len(sub),shifted_geometric_wql9=shifted_gmean(sub[metric]),
            raw_configuration_mean=float(sub.wql9.mean()),uses_future_labels=arm.startswith("oracle_"))
    # Contributions conserve the arithmetic configuration score, not its geometric counterpart.
    denominators = reproduced.denominator
    frame["wql_contribution"] = frame.numerator/frame.configuration.map(denominators)/len(denominators)
    events = frame.groupby(["model","event","internal_event_boundary"]).agg(
        contribution=("wql_contribution","sum"),variate_origins=("item","size"),points=("points","sum"))
    for arm,sub in events.groupby("model"):
        np.testing.assert_allclose(sub.contribution.sum(),table[table.model.eq(arm)].wql9.mean(),rtol=1e-11,atol=1e-12)
    warm = frame[frame.model.str.startswith("causal_")].groupby(["model","term"]).agg(
        cases=("item","size"),warm_cases=("warm","sum"),points=("points","sum"))
    comparisons = {}
    pivot = table.pivot(index=["dataset","configuration"],columns="model",values="wql9")
    for arm in ("causal_pooled","causal_channel","causal_recent3"):
        delta = (pivot[arm]-pivot.toto_raw).groupby("dataset").mean()
        comparisons[arm] = paired_cluster_interval(delta)
    frame.to_parquet(dest/"case_scores.parquet",index=False)
    table.to_parquet(dest/"configuration_scores.parquet",index=False)
    events.reset_index().to_parquet(dest/"event_contributions.parquet",index=False)
    result = dict(stage="native_causal_calibration_and_hindsight_diagnostic",groups=len(groups),
        variate_origins=len(frame)//frame.model.nunique(),grid=grid,minimum_windows=minimum_windows,recent_windows=recent_windows,
        scores=summaries,warm_coverage=warm.reset_index().to_dict("records"),causal_task_differences=comparisons,
        event_contributions=events.reset_index().to_dict("records"),baseline_scores_reproduced=True,
        seconds=time.perf_counter()-start,sota=False,
        scope="131 fixed development configurations. Classical prequential affine quantile calibration uses only completed earlier paths; first two paths use identity. Oracle arms select parameters and/or cuts with current future targets and are not forecasting methods or achievable performance bounds. Event cuts use the existing empirical atom and only the first transition; absent/internal-edge cuts fall back to the fixed midpoint. Phase oracles test this affine family only. No new algorithm, independent test, or SOTA.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def calibration_probe(repo_root,cache_dir,data_dir,prediction_dir,comparison_dir,
                       biases=(-.5,-.25,0.,.25,.5),spreads=(0.,.5,.75,1.,1.25,1.5,2.),minimum_windows=2,recent_windows=3):
    from r1.native_workflow import campaign
    from r1.workflow import emit
    checks = validate_calibration()
    root,pred,ref = Path(data_dir),Path(prediction_dir),Path(comparison_dir)
    config = dict(data_dir=str(root),prediction_dir=str(pred),comparison_dir=str(ref),
        groups_sha=sha256(root/"groups.parquet"),reference_sha=sha256(ref/"configuration_scores.parquet"),
        predictions={p.name:sha256(p) for p in sorted(pred.glob("*.npz"))},
        biases=list(biases),spreads=list(spreads),minimum_windows=minimum_windows,recent_windows=recent_windows,
        validation=checks,primary="Existing native full-path WQL and fixed low-variance partitions. Only causal arms are predictions; all oracle scores are retrospectively selected diagnostics. No SOTA gate is relaxed.")
    dest = campaign(repo_root,cache_dir,"native_calibration",config)
    result = run_calibration(root,pred,ref,dest,biases,spreads,minimum_windows,recent_windows)
    return emit(dict(cache=str(dest),validation=checks,**result))
