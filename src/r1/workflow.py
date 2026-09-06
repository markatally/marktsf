"""Notebook steps; all persistent intermediate artifacts stay outside the repo."""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
import torch

from r1.data import boom_development, price_development, probe_rows, sha256, calibration_atom
from r1.models import fit_event_probe, select_device, timesfm3_forecaster, gaussian_random_walk
from r1.evaluation import event_scores, forecast_scores


def external_cache(repo_root, cache_dir):
    repo, cache = Path(repo_root).resolve(), Path(cache_dir).expanduser().resolve()
    if cache == repo or repo in cache.parents:
        raise ValueError("Cache must be outside repository")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return value


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def provenance(repo_root):
    root = Path(repo_root)
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "sources": {str(p.relative_to(root)): sha256(p) for p in sorted((root/"src/r1").glob("*.py"))},
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "torch", "pyarrow")}}


def preflight(repo_root, cache_dir, **kwargs):
    cache = external_cache(repo_root, cache_dir)
    result = {"stage": "preflight", "device": str(select_device()),
              "pin_memory": select_device().type == "cuda",
              "available_ram_gib": psutil.virtual_memory().available/2**30,
              "cache": str(cache), "provenance": provenance(repo_root),
              "sota": False, "reason": "No frozen full benchmark evaluation yet"}
    save_json(cache/"preflight.json", result)
    return emit(result)


def audit(repo_root, cache_dir, max_tasks=48, max_variates=4, **kwargs):
    cache = external_cache(repo_root, cache_dir)
    records, manifest = boom_development(repo_root, max_tasks=max_tasks, max_variates=max_variates)
    rows = []
    for s in records:
        prefix = s.values[:s.train_end]
        atom = calibration_atom(prefix)
        rows.append({"dataset": s.dataset, "item": s.item, "train_n": len(prefix),
                     "dev_n": len(s.values), "finite": bool(np.isfinite(prefix).all()),
                     "zero_rate": float(np.mean(prefix == 0)) if len(prefix) else None,
                     "calibration_atom": atom,
                     "atom_rate": float(np.mean(prefix == atom)) if atom is not None else None})
    result = {"stage": "data_audit", "selected_tasks": len(manifest),
              "selected_variates": len(records),
              "intermittent_training_variates": sum(r["zero_rate"] is not None and .01 <= r["zero_rate"] <= .99 for r in rows),
              "intermittent_atom_variates": sum(r["atom_rate"] is not None and .01 <= r["atom_rate"] <= .99 for r in rows),
              "empty_training_variates": sum(r["train_n"] == 0 for r in rows),
              "split": "hash(task)%10<2; first70% development, first70% of development training",
              "official_test_used": False, "sota": False}
    save_json(cache/"data_manifest.json", {"tasks": manifest, "series": rows, "summary": result})
    return emit(result)


def mechanism(repo_root, cache_dir, domain="boom", max_tasks=48, max_variates=4,
              max_assets=24, horizons=(1,8,32), max_origins=512, seed=2021,
              max_iter=100, threads=4, **kwargs):
    cache = external_cache(repo_root, cache_dir)
    config = dict(domain=domain,max_tasks=max_tasks,max_variates=max_variates,
                  max_assets=max_assets,horizons=list(horizons),max_origins=max_origins,
                  seed=seed,max_iter=max_iter,threads=threads)
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    campaign = cache/f"probe_{domain}_{signature}"
    campaign.mkdir(exist_ok=True)
    if domain == "boom":
        records, manifest = boom_development(repo_root,max_tasks=max_tasks,max_variates=max_variates)
    elif domain == "price":
        records, manifest = price_development(repo_root,max_assets=max_assets)
    else:
        raise ValueError(domain)
    frames, skipped = [], Counter()
    for s in records:
        f, reason = probe_rows(s,domain,horizons,max_origins)
        if reason:
            skipped[reason] += 1
        else:
            frames.append(f)
    if not frames:
        raise RuntimeError("No eligible development series; do not cherry-pick replacements")
    frame = pd.concat(frames,ignore_index=True)
    save_json(campaign/"contract.json", {"config": config,"provenance": provenance(repo_root),
              "data": manifest,"skipped": dict(skipped),"official_test_used": False})
    print(f"Fitting {domain} event probe: {len(frame)} rows, {frame.dataset.nunique()} datasets", flush=True)
    valid, predictions, timing, models = fit_event_probe(frame,seed,max_iter,threads=threads)
    scores, raw = event_scores(valid,predictions,seed)
    raw.to_parquet(campaign/"validation_predictions.parquet",index=False)
    result = {"stage": "development_event_probe", "domain": domain,"config": config,
              "rows": len(frame),"datasets": frame.dataset.nunique(),
              "items": frame.groupby(["dataset","item"]).ngroups,"skipped": dict(skipped),
              "seconds": timing,"scores": scores,"cache": str(campaign),
              "interpretation": "Age features are derived from history; this measures representation utility, not novel information or a new algorithm.",
              "cluster": "dataset" if domain=="boom" else "calendar month across both equity universes",
              "sota": False}
    save_json(campaign/"summary.json",result)
    return emit(result)


def foundation(repo_root,cache_dir,checkpoint_path,max_tasks=8,max_variates=2,
               horizon=32,context_length=512,batch_size=2,**kwargs):
    cache = external_cache(repo_root,cache_dir)
    if psutil.virtual_memory().available < 8*2**30:
        raise RuntimeError("Less than 8 GiB available; defer model load")
    records,manifest = boom_development(repo_root,max_tasks=max_tasks,max_variates=max_variates)
    cases=[]
    for s in records:
        origin=s.train_end
        if origin>=context_length and origin+horizon<=len(s.values) and np.isfinite(s.values).all():
            cases.append((s,s.values[origin-context_length:origin].astype(np.float32),s.values[origin:origin+horizon]))
    if not cases:
        raise RuntimeError("No forecast cases")
    tick=time.perf_counter()
    forecaster=timesfm3_forecaster(checkpoint_path,batch_size)
    contexts=[c[1] for c in cases]
    outputs=list(forecaster.predict_batch(contexts,horizon=horizon,return_quantiles=True,
                                         use_symmetric_averaging=False))
    rows=[]
    qs=np.arange(.1,1.,.1)
    for (s,context,target),out in zip(cases,outputs,strict=True):
        naive_p,naive_q=gaussian_random_walk([context],horizon,qs)
        rows.append({"dataset":s.dataset,"item":s.item,
                     "timesfm3":forecast_scores(target,out.forecast,out.quantiles,qs),
                     "gaussian_random_walk":forecast_scores(target,naive_p[0],naive_q[0],qs)})
    result={"stage":"development_foundation_baseline","device":str(select_device()),
            "use_symmetric_averaging":False,
            "checkpoint":str(checkpoint_path),"context":context_length,"horizon":horizon,
            "point_forecast_semantics":"TimesFM3 forecast is the median, not the mean; MSE is median-MSE diagnostic",
            "cases":len(rows),"seconds":time.perf_counter()-tick,"scores":rows,
            "sota":False,"reason":"Few development origins; official suite and competitive candidates still required"}
    save_json(cache/"foundation_development.json",result)
    return emit(result)


def validate(repo_root, cache_dir, **kwargs):
    """Material leakage and score checks; no test suite artifacts created."""
    from r1.data import Series, state_ages
    from r1.evaluation import paired_cluster_interval
    y=np.tile(np.r_[np.zeros(9),np.ones(7)],40)
    a=Series("synthetic","a",y.copy(),400,"generated","generated")
    b=Series("synthetic","a",y.copy(),400,"generated","generated")
    b.values[500:]=123.0
    fa,_=probe_rows(a,"boom",max_origins=2000)
    fb,_=probe_rows(b,"boom",max_origins=2000)
    columns=[c for c in fa if c not in ("label",)]
    pd.testing.assert_frame_equal(fa[fa.origin<500][columns].reset_index(drop=True),
                                  fb[fb.origin<500][columns].reset_index(drop=True))
    assert np.all(fa[fa.partition=="train"].origin+fa[fa.partition=="train"].horizon<400)
    assert state_ages([0,0,1,1,1,0]).tolist()==[1,2,1,2,3,1]
    q=np.arange(.1,1.,.1)
    perfect=forecast_scores(np.ones(3),np.ones(3),np.ones((3,9)),q)
    assert perfect["mse"]==perfect["wql9"]==0
    interval=paired_cluster_interval(np.zeros(10))
    assert interval["ci95"]==[0.,0.]
    from r1.duration import prepare_reward_series, EmpiricalDurationReward
    prepared,reason=prepare_reward_series(a,"boom")
    assert reason is None
    model=EmpiricalDurationReward(prepared,a.train_end)
    model.geometric[:]=[.2,.3]
    paths=model.sample(prepared,410,8,16000,2021)
    empirical=(paths==prepared["atom"]).mean(axis=0)
    expected=float(prepared["state"][410])
    for h in range(8):
        expected=(1-expected)*.2+expected*.7
        assert abs(empirical[h]-expected)<.02
    return emit({"stage":"validation","checks":["future perturbation leaves earlier features unchanged",
             "training labels do not cross split","state age recurrence","perfect forecast scores zero",
             "paired cluster resampling sanity","simulated two-state occupancy agrees with analytic Markov recurrence"],"passed":True})


def factorial(repo_root,cache_dir,domain="boom",max_tasks=48,max_variates=4,max_assets=24,
              horizon=32,samples=512,origins_per_series=6,seed=2021,smoothing=20.,age_cap=128,scale_feedback=True,**kwargs):
    from r1.duration import reward_factorial
    cache=external_cache(repo_root,cache_dir)
    records,manifest=(boom_development(repo_root,max_tasks,max_variates) if domain=="boom"
                      else price_development(repo_root,max_assets=max_assets))
    config=dict(protocol_version=2,domain=domain,max_tasks=max_tasks,max_variates=max_variates,max_assets=max_assets,
                horizon=horizon,samples=samples,origins_per_series=origins_per_series,
                seed=seed,smoothing=smoothing,age_cap=age_cap,scale_feedback=scale_feedback)
    source_provenance=provenance(repo_root)
    identity={"config":config,"source_hashes":source_provenance["sources"]}
    campaign=cache/("factorial_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    campaign.mkdir(exist_ok=True)
    save_json(campaign/"contract.json",{"config":config,"data":manifest,"provenance":source_provenance})
    predictions=campaign/"predictions"
    predictions.mkdir(exist_ok=True)
    tick=time.perf_counter()
    frame,result=reward_factorial(records,domain,horizon,samples,origins_per_series,seed,smoothing,age_cap,scale_feedback,predictions)
    frame.to_parquet(campaign/"case_scores.parquet",index=False)
    result.update(config=config,seconds=time.perf_counter()-tick,cache=str(campaign))
    save_json(campaign/"summary.json",result)
    return emit(result)


def matched_foundation(repo_root,cache_dir,checkpoint_path,case_scores_path,
                       domain="boom",max_tasks=48,max_variates=4,max_assets=24,
                       horizon=32,context_length=2048,batch_size=2,seed=2021,use_symmetric_averaging=True,**kwargs):
    """Native multivariate TimesFM on the exact pre-existing factorial cases."""
    from r1.evaluation import paired_cluster_interval
    cache=external_cache(repo_root,cache_dir)
    reference=pd.read_parquet(case_scores_path)
    if "horizon" not in reference or not reference.horizon.eq(horizon).all():
        raise ValueError("Reference horizon missing or mismatched")
    if "target_hash" not in reference or reference.wql9.isna().any():
        raise ValueError("Reference targets unauditable or WQL undefined")
    keys=reference[["dataset","item","origin","cluster"]].drop_duplicates()
    records,_=(boom_development(repo_root,max_tasks,max_variates) if domain=="boom"
                else price_development(repo_root,max_assets=max_assets))
    lookup={(s.dataset,s.item):s for s in records}
    batches,groups=[],[]
    for (dataset,origin),group in keys.groupby(["dataset","origin"],sort=True):
        contexts=[]
        members=[]
        for row in group.itertuples():
            s=lookup[(row.dataset,row.item)]
            if origin<s.train_end or origin+horizon>=len(s.values):
                raise ValueError("Requested case is outside frozen development validation")
            contexts.append(s.values[max(0,origin-context_length+1):origin+1].astype(np.float32))
            members.append((row,s))
        length=min(map(len,contexts))
        batches.append(np.stack([v[-length:] for v in contexts]))
        groups.append(members)
    tick=time.perf_counter()
    forecaster=timesfm3_forecaster(checkpoint_path,batch_size)
    print(f"TimesFM3 native multivariate inference: {len(batches)} groups / {len(keys)} cases",flush=True)
    def output_pairs():
        # Official API accepts variable time lengths but one channel count per call.
        for dimension in sorted({context.shape[0] for context in batches}):
            indexes=[i for i,context in enumerate(batches) if context.shape[0]==dimension]
            outputs=forecaster.predict_batch([batches[i] for i in indexes],horizon=horizon,
                            return_quantiles=True,use_symmetric_averaging=use_symmetric_averaging)
            for i,out in zip(indexes,outputs,strict=True):
                yield groups[i],out
    prediction_identity=dict(reference=sha256(case_scores_path),checkpoint=str(checkpoint_path),
                             context=context_length,symmetric=use_symmetric_averaging)
    prediction_directory=cache/("foundation_predictions_"+hashlib.sha256(json.dumps(prediction_identity,sort_keys=True).encode()).hexdigest()[:12])
    prediction_directory.mkdir(exist_ok=True)
    rows=[]
    for members,out in output_pairs():
        points=np.asarray(out.forecast).reshape(len(members),horizon)
        quantiles=np.asarray(out.quantiles).reshape(len(members),horizon,9)
        for v,(row,s) in enumerate(members):
            target=s.values[row.origin+1:row.origin+horizon+1]
            digest=hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest()
            prior=reference[(reference.dataset==row.dataset)&(reference.item==row.item)&(reference.origin==row.origin)]
            if not prior.target_hash.eq(digest).all():
                raise ValueError("Reference and foundation targets differ")
            scores=forecast_scores(target,points[v],quantiles[v],np.arange(.1,1.,.1))
            if scores["wql9"] is None:
                raise ValueError("Undefined case WQL; do not drop cases")
            np.savez_compressed(prediction_directory/f"{len(rows)}.npz",quantiles=quantiles[v],
                median=points[v],target=target,target_hash=digest,dataset=row.dataset,
                item=row.item,origin=row.origin,horizon=horizon)
            rows.append({"dataset":row.dataset,"item":row.item,"origin":row.origin,
                         "cluster":row.cluster,"arm":"timesfm3_native",
                         **{k:scores[k] for k in ("wql9","mse","mae")}})
    frame=pd.DataFrame(rows)
    combined=pd.concat([reference,frame],ignore_index=True)
    grouped=combined.groupby(["cluster","arm"])[["wql9","mse","mae"]].mean()
    summary={arm:{metric:float(grouped.xs(arm,level="arm")[metric].mean()) for metric in ("wql9","mse","mae")}
             for arm in combined.arm.unique()}
    pivot=grouped.wql9.unstack("arm")
    delta=paired_cluster_interval(pivot.age_coupled-pivot.timesfm3_native,seed)
    result={"stage":"matched_development_foundation","domain":domain,"cases":len(frame),
            "native_groups":len(groups),"context_max":context_length,"horizon":horizon,
            "checkpoint":str(checkpoint_path),"reference_sha256":sha256(case_scores_path),
            "use_symmetric_averaging":use_symmetric_averaging,
            "prediction_directory":str(prediction_directory),"provenance":provenance(repo_root),
            "point_semantics":"median for every arm", "scores":summary,
            "age_coupled_minus_timesfm3_wql9":delta,"seconds":time.perf_counter()-tick,
            "sota":False,"regimes":"empirical models adapted on historical prefix; TimesFM3 frozen checkpoint",
            "limitations":("Development subset selected for empirical atoms; not complete BOOM, not all strong baselines"
                            if domain=="boom" else "Fixed development price cases; limited shared calendar months; not a complete financial benchmark")}
    name=f"matched_foundation_{domain}_{sha256(case_scores_path)[:12]}"
    frame.to_parquet(cache/(name+".parquet"),index=False)
    save_json(cache/(name+".json"),result)
    return emit(result)


def refinement_data(repo_root,cache_dir,checkpoint_path,reference_path,foundation_predictions,
                    horizon=32,train_origins=24,context_length=2048,batch_size=2,**kwargs):
    from r1.refinement import build_refinement_data,validate_refinement
    cache=external_cache(repo_root,cache_dir)
    validation=validate_refinement()
    dest,manifest=build_refinement_data(repo_root,str(cache),checkpoint_path,reference_path,
                 foundation_predictions,horizon,train_origins,context_length,batch_size)
    return emit({"stage":"refinement_data","cache":str(dest),"validation":validation,**manifest})


def refinement_train(repo_root,cache_dir,data_dir,steps=600,batch_size=64,width=64,
                     learning_rate=.002,seed=2021,age_cap=128,**kwargs):
    from r1.refinement import fit_refiners
    cache=external_cache(repo_root,cache_dir)
    config=dict(data_sha=sha256(Path(data_dir)/"arrays.npz"),steps=steps,batch_size=batch_size,
                width=width,learning_rate=learning_rate,seed=seed,age_cap=age_cap)
    provenance_record=provenance(repo_root)
    identity=dict(config=config,source=provenance_record["sources"])
    dest=cache/("refinement_fit_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",{"config":config,"provenance":provenance_record})
    result=fit_refiners(data_dir,dest,steps,batch_size,width,learning_rate,seed,age_cap)
    return emit({"cache":str(dest),**result})


def matched_toto(repo_root,cache_dir,checkpoint_path,case_scores_path,
                 runtime_python="/Users/mark/.cache/marktsf-research/toto-venv/bin/python",
                 max_tasks=48,max_variates=4,horizon=32,context_length=2048,**kwargs):
    cache=external_cache(repo_root,cache_dir)
    reference=pd.read_parquet(case_scores_path)
    if not reference.horizon.eq(horizon).all() or reference.wql9.isna().any():
        raise ValueError("Reference horizon or score invalid")
    identity=dict(checkpoint=str(checkpoint_path),reference=sha256(case_scores_path),
                  context=context_length,horizon=horizon,worker=sha256(Path(repo_root)/"src/r1/toto_worker.py"))
    dest=cache/("matched_toto_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    inputs,outputs=dest/"inputs",dest/"predictions"
    inputs.mkdir(parents=True,exist_ok=True)
    outputs.mkdir(exist_ok=True)
    records,_=boom_development(repo_root,max_tasks,max_variates)
    lookup={(s.dataset,s.item):s for s in records}
    cases=reference[["dataset","item","origin","cluster","target_hash"]].drop_duplicates()
    for group_index,((dataset,origin),group) in enumerate(cases.groupby(["dataset","origin"],sort=True)):
        contexts,keys,hashes=[],[],[]
        for row in group.itertuples():
            s=lookup[(row.dataset,row.item)]
            if origin<s.train_end or origin+horizon>=len(s.values):
                raise ValueError("Toto case outside development validation")
            target=s.values[origin+1:origin+horizon+1]
            digest=hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest()
            if digest!=row.target_hash:
                raise ValueError("Toto reference target mismatch")
            contexts.append(s.values[max(0,origin-context_length+1):origin+1].astype(np.float32))
            keys.append([row.dataset,row.item,str(origin)])
            hashes.append(digest)
        length=min(map(len,contexts))
        np.savez_compressed(inputs/f"{group_index:04d}.npz",context=np.stack([a[-length:] for a in contexts]),
                            keys=np.asarray(keys),target_hashes=np.asarray(hashes))
    parameters=dict(checkpoint=str(checkpoint_path),input_dir=str(inputs),output_dir=str(outputs),horizon=horizon)
    save_json(dest/"contract.json",{"identity":identity,"provenance":provenance(repo_root)})
    tick=time.perf_counter()
    subprocess.run([runtime_python,str(Path(repo_root)/"src/r1/toto_worker.py"),"--parameters",json.dumps(parameters)],
                   check=True,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","MPLBACKEND":"Agg"})
    rows=[]
    for path in sorted(outputs.glob("*.npz")):
        data=np.load(path)
        for key,q,digest in zip(data["keys"],data["quantiles"],data["target_hashes"],strict=True):
            dataset,item,origin=key
            origin=int(origin)
            s=lookup[(dataset,item)]
            target=s.values[origin+1:origin+horizon+1]
            if hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest()!=digest:
                raise ValueError("Worker output target mismatch")
            score=forecast_scores(target,q[:,4],q,np.arange(.1,1,.1))
            rows.append(dict(dataset=dataset,item=item,origin=origin,arm="toto2_313m",**score))
    frame=pd.DataFrame(rows)
    if len(frame)!=len(cases) or frame.wql9.isna().any():
        raise ValueError("Incomplete Toto predictions")
    frame.to_parquet(dest/"case_scores.parquet",index=False)
    score=frame.groupby("dataset")[["wql9","mse","mae","crossing_fraction"]].mean().mean().to_dict()
    result={"stage":"matched_toto","cases":len(frame),"datasets":frame.dataset.nunique(),
            "scores":score,"seconds":time.perf_counter()-tick,"runtime":json.loads((outputs/"runtime.json").read_text()),
            "cache":str(dest),"sota":False,"scope":"313m baseline on fixed development cases; larger Toto2 sizes and complete suite remain"}
    save_json(dest/"summary.json",result)
    return emit(result)


def refinement_seeds(repo_root,cache_dir,data_dir,seeds=(2021,2022,2023,2024,2025),
                     steps=600,batch_size=64,width=64,learning_rate=.002,age_cap=128,
                     existing_runs=None,**kwargs):
    from r1.evaluation import paired_cluster_interval
    cache=external_cache(repo_root,cache_dir)
    existing_runs=existing_runs or {}
    tables,results=[],[]
    for seed in seeds:
        previous=existing_runs.get(str(seed))
        if previous:
            contract=json.loads((Path(previous)/"contract.json").read_text())
            expected=dict(data_sha=sha256(Path(data_dir)/"arrays.npz"),steps=steps,batch_size=batch_size,
                           width=width,learning_rate=learning_rate,seed=seed,age_cap=age_cap)
            if contract["config"]!=expected or contract["provenance"]["sources"]["src/r1/refinement.py"]!=sha256(Path(repo_root)/"src/r1/refinement.py"):
                raise ValueError("Existing seed run is not the same algorithm/configuration")
            result={"cache":previous,**json.loads((Path(previous)/"summary.json").read_text())}
        else:
            result=refinement_train(repo_root,cache_dir,data_dir,steps,batch_size,width,learning_rate,seed,age_cap)
        frame=pd.read_parquet(Path(result["cache"])/"case_scores.parquet")
        frame["seed"]=seed
        tables.append(frame)
        results.append({"seed":seed,"cache":result["cache"],"scores":result["scores"]})
    combined=pd.concat(tables,ignore_index=True)
    # Average optimization randomness within each dataset before resampling datasets.
    grouped=combined.groupby(["dataset","arm"])[["wql9","mse","mean_mixture_weight"]].mean()
    pivot=grouped.wql9.unstack("arm")
    result={"stage":"refinement_seed_replication","seeds":list(seeds),"datasets":len(pivot),
            "scores":{arm:grouped.xs(arm,level="arm").mean().to_dict() for arm in combined.arm.unique()},
            "semi_minus_direct":paired_cluster_interval(pivot.semi_markov-pivot.direct),
            "semi_minus_frozen":paired_cluster_interval(pivot.semi_markov-pivot.frozen_timesfm3),
            "runs":results,"sota":False,"scope":"Development only; seeds are not independent datasets"}
    signature=hashlib.sha256(json.dumps(results,sort_keys=True).encode()).hexdigest()[:12]
    combined.to_parquet(cache/f"refinement_seeds_{signature}.parquet",index=False)
    save_json(cache/f"refinement_seeds_{signature}.json",result)
    return emit(result)


def coupling(repo_root,cache_dir,data_dir,seeds=(2021,2022,2023),hazard_steps=400,
             emission_steps=400,batch_size=64,width=64,learning_rate=.002,
             support_count=32,age_cap=128,**kwargs):
    from r1.coupling import fit_coupling,validate_coupling
    cache=external_cache(repo_root,cache_dir)
    checks=validate_coupling()
    config=dict(data_sha=sha256(Path(data_dir)/"arrays.npz"),seeds=list(seeds),
                hazard_steps=hazard_steps,emission_steps=emission_steps,batch_size=batch_size,
                width=width,learning_rate=learning_rate,support_count=support_count,age_cap=age_cap)
    provenance_record=provenance(repo_root)
    identity=dict(config=config,source=provenance_record["sources"])
    dest=cache/("coupling_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",{"config":config,"provenance":provenance_record,"validation":checks})
    result=fit_coupling(data_dir,dest,seeds,hazard_steps,emission_steps,batch_size,width,
                        learning_rate,support_count,age_cap)
    return emit({"cache":str(dest),"validation":checks,**result})


def price_controls(repo_root,cache_dir,reference_path,foundation_predictions,reference_predictions,
                   max_assets=24,horizon=32,volatility_window=96,ewma_alpha=.06,**kwargs):
    from r1.financial import evaluate_price_controls
    cache=external_cache(repo_root,cache_dir)
    config=dict(reference_sha=sha256(reference_path),max_assets=max_assets,horizon=horizon,
                volatility_window=volatility_window,ewma_alpha=ewma_alpha,
                prediction_hashes={str(p):sha256(p) for directory in (foundation_predictions,reference_predictions)
                                   for p in sorted(Path(directory).glob("*.npz"))})
    provenance_record=provenance(repo_root)
    identity=dict(config=config,source=provenance_record["sources"])
    dest=cache/("financial_controls_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",{"config":config,"provenance":provenance_record})
    result=evaluate_price_controls(repo_root,dest,reference_path,foundation_predictions,
                     reference_predictions,max_assets,horizon,volatility_window,ewma_alpha)
    return emit({"cache":str(dest),**result})


def aggregation_audit(repo_root,cache_dir,data_dir,foundation_predictions,toto_predictions,
                       refinement_runs,coupling_run,upstream_leaderboard,
                       runtime_python="/Users/mark/.cache/marktsf-research/baseline-venv/bin/python",horizon=32,**kwargs):
    from r1.aggregation import audit_aggregation
    cache=external_cache(repo_root,cache_dir)
    directories=[Path(data_dir),Path(foundation_predictions),Path(toto_predictions),Path(coupling_run)]
    directories.extend(Path(p) for p in refinement_runs)
    files=sorted({p for directory in directories for p in directory.glob("*.npz")})
    config=dict(data_metadata_sha=sha256(Path(data_dir)/"cases.parquet"),horizon=horizon,
                inputs={str(p):sha256(p) for p in files},upstream_sha=sha256(upstream_leaderboard))
    provenance_record=provenance(repo_root)
    identity=dict(config=config,source=provenance_record["sources"])
    dest=cache/("aggregation_audit_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",{"config":config,"provenance":provenance_record})
    result=audit_aggregation(repo_root,dest,data_dir,foundation_predictions,toto_predictions,
                             refinement_runs,coupling_run,upstream_leaderboard,runtime_python,horizon)
    return emit({"cache":str(dest),**result})


def propagation_audit(repo_root,cache_dir,horizons=(8,32),tails=(8,32),tail_probabilities=(.05,.2),
                       query_probabilities=(.05,.4,.7),supports=(32,128),truth_probability=.2,
                       finite_difference_step=1e-5,grid_modes=("uniform","log"),**kwargs):
    from r1.propagation_audit import audit_propagation
    cache=external_cache(repo_root,cache_dir)
    config=dict(horizons=list(horizons),tails=list(tails),tail_probabilities=list(tail_probabilities),
                query_probabilities=list(query_probabilities),supports=list(supports),
                truth_probability=truth_probability,finite_difference_step=finite_difference_step,grid_modes=list(grid_modes))
    provenance_record=provenance(repo_root)
    identity=dict(config=config,source=provenance_record["sources"])
    dest=cache/("propagation_audit_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",{"config":config,"provenance":provenance_record})
    result=audit_propagation(dest,horizons,tails,tail_probabilities,query_probabilities,supports,
                              truth_probability,finite_difference_step,grid_modes)
    return emit({"cache":str(dest),**result})


from r1.native_workflow import native_prepare, native_forecast, native_compare, native_diagnose
from r1.continuous_probe import continuous_probe
from r1.continuous_audit import continuous_audit
from r1.peer_probe import peer_probe
from r1.peer_audit import peer_audit
from r1.calibration_probe import calibration_probe
from r1.calibration_null import calibration_null


STEPS={"preflight":preflight,"audit":audit,"mechanism":mechanism,
       "foundation":foundation,"validate":validate,"factorial":factorial,
       "matched_foundation":matched_foundation,"refinement_data":refinement_data,
       "refinement_train":refinement_train,"matched_toto":matched_toto,
       "refinement_seeds":refinement_seeds,"coupling":coupling,"price_controls":price_controls,
       "aggregation_audit":aggregation_audit,"propagation_audit":propagation_audit,
       "native_prepare":native_prepare,"native_forecast":native_forecast,"native_compare":native_compare,
       "native_diagnose":native_diagnose,"continuous_probe":continuous_probe,
       "continuous_audit":continuous_audit,"peer_probe":peer_probe,"peer_audit":peer_audit,
       "calibration_probe":calibration_probe,"calibration_null":calibration_null}


def run_step(step, **kwargs):
    if step not in STEPS:
        raise ValueError(f"Unknown step {step}")
    return STEPS[step](**kwargs)
