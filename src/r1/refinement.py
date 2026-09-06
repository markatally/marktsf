"""Experimental atom-mixture refinement of a frozen foundation distribution.

This is a candidate mechanism test, not an established novel algorithm.
Unstructured calibration and geometric-duration heads are mandatory controls.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from r1.data import boom_development, calibration_atom, state_ages, sha256
from r1.models import select_device, timesfm3_forecaster
from r1.evaluation import forecast_scores, paired_cluster_interval


def feature_case(series, origin, base, horizon):
    history=series.values[:origin+1]
    atom=calibration_atom(history)
    if atom is None:
        raise ValueError("Refinement pool requires a historical atom")
    scale=max(float(np.std(history[-128:])),float(np.mean(base[:,-1]-base[:,0])),1e-4)
    state=(history==atom).astype(int)
    age=int(state_ages(state)[-1])
    q=(base-atom)/scale
    features=np.r_[np.arcsinh((history[-64:]-atom)/scale),
                   np.arcsinh(q).flatten(),np.log1p(age),state[-1],state[-32:].mean(),state[-128:].mean()]
    return features.astype(np.float32),q.astype(np.float32),dict(atom=atom,scale=scale,
            state=int(state[-1]),age=age,target=(series.values[origin+1:origin+horizon+1]-atom)/scale)


def build_refinement_data(repo_root,cache_dir,checkpoint_path,reference_path,
                          foundation_predictions,horizon=32,train_origins=24,
                          context_length=2048,batch_size=2,max_tasks=48,max_variates=4):
    cache=Path(cache_dir)
    identity={"checkpoint":checkpoint_path,"reference_sha":sha256(reference_path),
              "horizon":horizon,"train_origins":train_origins,"context":context_length,
              "code_sha":sha256(__file__),"data_sha":sha256(Path(repo_root)/"src/r1/data.py")}
    dest=cache/("refinement_data_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12])
    dest.mkdir(parents=True,exist_ok=True)
    manifest_path=dest/"manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old["identity"]!=identity or not (dest/"arrays.npz").exists():
            raise ValueError("Incomplete or mismatched refinement cache")
        return dest,old
    records,_=boom_development(repo_root,max_tasks,max_variates)
    reference=pd.read_parquet(reference_path)
    if not reference.horizon.eq(horizon).all():
        raise ValueError("Horizon mismatch")
    names=set(zip(reference.dataset,reference.item))
    selected={(s.dataset,s.item):s for s in records if (s.dataset,s.item) in names}
    cases=[]
    for s in selected.values():
        origins=np.unique(np.linspace(192,s.train_end-horizon-1,train_origins,dtype=int))
        for origin in origins:
            if origin<192 or origin+horizon>=s.train_end:
                raise ValueError("Training origin overlaps validation")
            cases.append((s,int(origin)))
    groups={}
    for s,origin in cases:
        groups.setdefault((s.dataset,origin),[]).append((s,origin))
    buckets={}
    for members in groups.values():
        length=min(min(context_length,origin+1) for s,origin in members)
        contexts=np.stack([s.values[origin-length+1:origin+1].astype(np.float32) for s,origin in members])
        buckets.setdefault(len(members),[]).append((members,contexts))
    forecaster=timesfm3_forecaster(checkpoint_path,batch_size)
    built=[]
    print(f"Generating frozen-backbone training forecasts: {len(cases)} cases",flush=True)
    for count,grouped in sorted(buckets.items()):
        outputs=forecaster.predict_batch([g[1] for g in grouped],horizon=horizon,
                       return_quantiles=True,use_symmetric_averaging=True)
        for (members,context),out in zip(grouped,outputs,strict=True):
            qs=np.asarray(out.quantiles).reshape(count,horizon,9)
            for i,(s,origin) in enumerate(members):
                built.append((s,origin,qs[i],"train"))
    validation_keys=set()
    for path in sorted(Path(foundation_predictions).glob("*.npz")):
        with np.load(path) as f:
            name=(str(f["dataset"]),str(f["item"]))
            origin=int(f["origin"])
            s=selected[name]
            target=s.values[origin+1:origin+horizon+1]
            digest=hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest()
            if digest!=str(f["target_hash"]) or int(f["horizon"])!=horizon:
                raise ValueError("Cached validation prediction target mismatch")
            validation_keys.add((*name,origin))
            built.append((s,origin,f["quantiles"].copy(),"validation"))
    if validation_keys!=set(zip(reference.dataset,reference.item,reference.origin)):
        raise ValueError("Validation prediction set differs from reference")
    arrays={k:[] for k in ("features","base","target","state","age","scale","atom")}
    metadata=[]
    for s,origin,base,partition in built:
        features,q,extra=feature_case(s,origin,base,horizon)
        arrays["features"].append(features)
        arrays["base"].append(q)
        for k in ("target","state","age","scale","atom"):
            arrays[k].append(extra[k])
        metadata.append(dict(dataset=s.dataset,item=s.item,origin=origin,partition=partition))
    np.savez_compressed(dest/"arrays.npz",**{k:np.asarray(v) for k,v in arrays.items()})
    pd.DataFrame(metadata).to_parquet(dest/"cases.parquet",index=False)
    manifest={"identity":identity,"training_cases":len(cases),"validation_cases":len(validation_keys),
              "arrays_sha":sha256(dest/"arrays.npz"),"metadata_sha":sha256(dest/"cases.parquet"),
              "regime":"adapted on historical prefix; frozen TimesFM3 native multivariate features",
              "use_symmetric_averaging":True,"sota":False}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    return dest,manifest


def atom_mixture_quantiles(base,weight):
    """Exact quantiles of (1-w)F+w*delta_0 for the piecewise-linear base Q.

Virtual endpoints linearly extend the outer two quantiles to levels 0 and 1.
At w=0 this reproduces the original nine quantiles; all outputs stay ordered.
"""
    base=base.sort(dim=-1).values
    grid=torch.cat([2*base[...,:1]-base[...,1:2],base,
                    2*base[...,-1:]-base[...,-2:-1]],dim=-1)
    def at_zero(strict):
        count=((grid<0) if strict else (grid<=0)).sum(-1)
        index=(count-1).clamp(0,9)
        lo=grid.gather(-1,index[...,None])[...,0]
        hi=grid.gather(-1,(index+1)[...,None])[...,0]
        fraction=(-lo/(hi-lo).clamp_min(1e-8)).clamp(0,1)
        value=(index+fraction)/10
        return torch.where(count==0,0.,torch.where(count==11,1.,value))
    left,right=at_zero(True),at_zero(False)
    weight=weight.clamp(0,.95)
    low=((1-weight)*left)[...,None]
    high=((1-weight)*right+weight)[...,None]
    u=torch.arange(1,10,device=base.device,dtype=base.dtype)/10
    warped=torch.where(u<low,u/(1-weight[...,None]),
                       (u-weight[...,None])/(1-weight[...,None])).clamp(0,1)
    index=(warped*10).floor().long().clamp(0,9)
    frac=warped*10-index
    lo=grid.gather(-1,index)
    hi=grid.gather(-1,index+1)
    interpolated=lo+frac*(hi-lo)
    return torch.where((u>=low)&(u<=high),0.,interpolated)


class AtomRefiner(nn.Module):
    def __init__(self,features,horizon=32,width=64,arm="direct",age_cap=128):
        super().__init__()
        self.horizon,self.arm,self.age_cap=horizon,arm,age_cap
        self.encoder=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        outputs=horizon if arm=="direct" else 3 if arm=="geometric" else 17
        self.head=nn.Linear(width,outputs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            if arm=="direct":
                self.head.bias.fill_(-5.)
            else:
                self.head.bias[-1]=-5.

    def forward(self,features,base,state,age):
        output=self.head(self.encoder(features))
        if self.arm=="direct":
            weights=output.sigmoid()
        else:
            if self.arm=="geometric":
                hazards=output[:,:2].sigmoid()[...,None].expand(-1,-1,self.age_cap)
            else:
                bands=torch.log2(torch.arange(1,self.age_cap+1,device=base.device,dtype=base.dtype)).floor().long().clamp(max=7)
                hazards=output[:,:16].reshape(-1,2,8).sigmoid()[...,bands]
            location=state.long()*self.age_cap+(age.long().clamp(1,self.age_cap)-1)
            mass=torch.nn.functional.one_hot(location,num_classes=2*self.age_cap).to(base.dtype).reshape(-1,2,self.age_cap)
            occupancy=[]
            for h in range(self.horizon):
                stay=mass*(1-hazards)
                switch=(mass*hazards).sum(-1).flip(-1)[...,None]
                mass=torch.cat([switch,stay[...,:-2],stay[...,-2:].sum(-1,keepdim=True)],dim=-1)
                occupancy.append(mass[:,1].sum(-1))
            weights=torch.stack(occupancy,dim=-1)*output[:,-1:].sigmoid()
        return atom_mixture_quantiles(base,weights),weights


def fit_refiners(data_dir,output_dir,steps=600,batch_size=64,width=64,learning_rate=.002,
                 seed=2021,age_cap=128):
    root,dest=Path(data_dir),Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    arrays=np.load(root/"arrays.npz")
    meta=pd.read_parquet(root/"cases.parquet")
    device=select_device()
    train=np.flatnonzero(meta.partition.eq("train"))
    valid=np.flatnonzero(meta.partition.eq("validation"))
    counts=meta.iloc[train].groupby("dataset").size()
    probabilities=meta.iloc[train].dataset.map(lambda x:1/counts[x]).to_numpy()
    probabilities=probabilities/probabilities.sum()
    tensors={k:torch.as_tensor(np.asarray(arrays[k]),dtype=torch.float32,device=device)
             for k in ("features","base","target","state","age")}
    for k,tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite refinement input: {k}")
    levels=torch.arange(1,10,device=device,dtype=torch.float32)/10
    arms=np.random.default_rng(seed).permutation(["direct","geometric","semi_markov"])
    rows,timings,parameters=[],{},{}
    for arm in arms:
        torch.manual_seed(seed)
        model=AtomRefiner(tensors["features"].shape[-1],tensors["base"].shape[1],width,str(arm),age_cap).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.0001)
        rng=np.random.default_rng(seed)
        tick=time.perf_counter()
        losses=[]
        for step in range(steps):
            ids=torch.as_tensor(rng.choice(train,batch_size,replace=True,p=probabilities),device=device)
            prediction,weights=model(tensors["features"][ids],tensors["base"][ids],tensors["state"][ids],tensors["age"][ids])
            residual=tensors["target"][ids,...,None]-prediction
            loss=2*torch.maximum(residual*levels,residual*(levels-1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refinement loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step%100==0 or step==steps-1:
                losses.append([step,float(loss.detach().cpu())])
        model.eval()
        predictions,mixtures=[],[]
        with torch.no_grad():
            for start in range(0,len(valid),batch_size):
                ids=torch.as_tensor(valid[start:start+batch_size],device=device)
                p,w=model(tensors["features"][ids],tensors["base"][ids],tensors["state"][ids],tensors["age"][ids])
                predictions.append(p.cpu().numpy())
                mixtures.append(w.cpu().numpy())
        predictions=np.concatenate(predictions)
        mixtures=np.concatenate(mixtures)
        timings[str(arm)]=time.perf_counter()-tick
        parameters[str(arm)]=sum(p.numel() for p in model.parameters())
        torch.save(model.cpu().state_dict(),dest/f"{arm}.pt")
        np.savez_compressed(dest/f"{arm}_validation.npz",quantiles=predictions,weights=mixtures,indices=valid)
        (dest/f"{arm}_training.json").write_text(json.dumps(losses))
        for j,index in enumerate(valid):
            scale,atom=arrays["scale"][index],arrays["atom"][index]
            target=arrays["target"][index]*scale+atom
            q=predictions[j]*scale+atom
            score=forecast_scores(target,q[:,4],q,np.arange(.1,1,.1))
            rows.append({"dataset":meta.iloc[index]["dataset"],"item":meta.iloc[index]["item"],
                         "origin":int(meta.iloc[index].origin),"arm":str(arm),
                         "wql9":score["wql9"],"mse":score["mse"],"mean_mixture_weight":float(mixtures[j].mean())})
        print(f"Refiner {arm}: {timings[str(arm)]:.1f}s, {parameters[str(arm)]} parameters",flush=True)
    for index in valid:
        scale,atom=arrays["scale"][index],arrays["atom"][index]
        q=arrays["base"][index]*scale+atom
        score=forecast_scores(arrays["target"][index]*scale+atom,q[:,4],q,np.arange(.1,1,.1))
        rows.append({"dataset":meta.iloc[index]["dataset"],"item":meta.iloc[index]["item"],"origin":int(meta.iloc[index].origin),
                     "arm":"frozen_timesfm3","wql9":score["wql9"],"mse":score["mse"],"mean_mixture_weight":0.})
    frame=pd.DataFrame(rows)
    if frame.wql9.isna().any():
        raise ValueError("Undefined WQL; no silent exclusion")
    grouped=frame.groupby(["dataset","arm"])[["wql9","mse","mean_mixture_weight"]].mean()
    pivot=grouped.wql9.unstack("arm")
    result={"device":str(device),"steps":steps,"seed":seed,"parameters":parameters,"seconds":timings,
            "training_cases":len(train),"validation_cases":len(valid),
            "scores":{a:grouped.xs(a,level="arm").mean().to_dict() for a in frame.arm.unique()},
            "semi_minus_direct":paired_cluster_interval(pivot.semi_markov-pivot.direct,seed),
            "semi_minus_frozen":paired_cluster_interval(pivot.semi_markov-pivot.frozen_timesfm3,seed),
            "sota":False,"scope":"Frozen development atoms; single seed; no official full-suite result or novelty established"}
    frame.to_parquet(dest/"case_scores.parquet",index=False)
    (dest/"summary.json").write_text(json.dumps(result,indent=2))
    return result


def validate_refinement():
    torch.manual_seed(2021)
    base=torch.randn(16,8,9).sort(-1).values
    zero=atom_mixture_quantiles(base,torch.zeros(16,8))
    assert torch.allclose(zero,base,atol=2e-6)
    weights=torch.full((16,8),.25,requires_grad=True)
    output=atom_mixture_quantiles(base,weights)
    assert (output.diff(dim=-1)>=-1e-6).all()
    output.square().mean().backward()
    assert torch.isfinite(weights.grad).all()
    # Monte Carlo mixture check on Uniform(-1,1), whose quantile function is linear.
    uniform=(2*torch.arange(1,10)/10-1).reshape(1,1,9)
    predicted=atom_mixture_quantiles(uniform,torch.tensor([[.4]])).numpy()[0,0]
    rng=np.random.default_rng(2021)
    samples=rng.uniform(-1,1,100000)
    samples[rng.random(len(samples))<.4]=0
    assert np.max(np.abs(predicted-np.quantile(samples,np.arange(.1,1,.1))))<.015
    return {"passed":True,"checks":["identity at zero mixture weight","ordered output quantiles",
                  "finite gradients","mixture quantiles match independent Monte Carlo"]}
