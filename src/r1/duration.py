"""Classical empirical semi-Markov controls, not a claimed novel R1 algorithm.

Two factors: age-dependent exit hazard and age-conditioned emission amplitude.
Training uses observed transitions only; the last observation is right censored.
"""
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

from r1.data import calibration_atom, state_ages
from r1.evaluation import forecast_scores, paired_cluster_interval


def prepare_reward_series(series, domain):
    y = series.values
    if len(y) < 192 or not np.isfinite(y).all():
        return None, "short_or_nonfinite"
    if domain == "price":
        if np.any(y <= 0):
            return None, "nonpositive_price"
        increments = np.r_[0., np.diff(np.log(y))]
        # scale[t] is known after observing return[t] and scales return[t+1].
        scale = np.sqrt(pd.Series(increments**2).ewm(alpha=.06, adjust=False).mean().to_numpy().clip(1e-10))
        standardized = increments / np.r_[scale[0], scale[:-1]]
        state = (np.abs(standardized) > 1.5).astype(int)
        emission = standardized
        atom, floor = None, 1e-5
    else:
        atom = calibration_atom(y)
        if atom is None:
            return None, "no_atom_in_initial_calibration"
        state = (y == atom).astype(int)
        if not .01 <= state[:series.train_end].mean() <= .99:
            return None, "not_intermittent_on_training_prefix"
        residual = y-atom
        floor = max(float(np.sqrt(np.mean(residual[:96]**2)))*.1, 1e-8)
        scale = np.sqrt(pd.Series(residual**2).rolling(96,min_periods=1).mean().to_numpy()).clip(floor)
        emission = residual / np.r_[scale[0],scale[:-1]]
    return dict(y=y, state=state, ages=state_ages(state), emission=emission,
                scale=scale, atom=atom, floor=floor), None


class EmpiricalDurationReward:
    def __init__(self, prepared, train_end, smoothing=20., age_cap=128):
        self.age_cap, self.smoothing = age_cap, smoothing
        s, a, e = prepared["state"], prepared["ages"], prepared["emission"]
        t = np.arange(96,train_end-1)
        # Initial run has unknown pre-file age. A top-coded age is still identified.
        t = t[(a[t] < t+1) | (a[t] >= max(age_cap,128))]
        self.hazards, self.geometric = np.zeros((2,age_cap)), np.zeros(2)
        self.pools, self.local_pools = {}, {}
        for state in (0,1):
            indexes = t[s[t] == state]
            exits = s[indexes+1] != state
            self.geometric[state] = (exits.sum()+.5)/(len(indexes)+1)
            at = np.minimum(a[indexes],age_cap)-1
            counts = np.bincount(at,minlength=age_cap)
            failures = np.bincount(at,weights=exits,minlength=age_cap)
            self.hazards[state] = (failures+smoothing*self.geometric[state])/(counts+smoothing)
            incoming = t+1
            incoming = incoming[s[incoming] == state]
            values = e[incoming]
            self.pools[state] = values
            for band in range(8):
                mask = np.minimum(np.floor(np.log2(a[incoming])).astype(int),7)==band
                self.local_pools[(state,band)] = values[mask]
        if any(len(pool)==0 for pool in self.pools.values()):
            raise ValueError("Both states require observed training emissions")

    def sample(self, prepared, origin, horizon, samples, seed,
               age_hazard=False, coupled_amplitude=False, domain="boom", scale_feedback=True):
        rng=np.random.default_rng(seed)
        s=np.full(samples,prepared["state"][origin],dtype=int)
        a=np.full(samples,prepared["ages"][origin],dtype=int)
        scale=np.full(samples,prepared["scale"][origin])
        if domain=="price":
            level=np.full(samples,np.log(prepared["y"][origin]))
        else:
            residual=prepared["y"][origin-95:origin+1]-prepared["atom"]
            rolling=np.broadcast_to(residual,(samples,96)).copy()
        paths=np.empty((samples,horizon))
        for h in range(horizon):
            # Common random numbers across factorial arms, including emission draws.
            u_exit,u_pool,u_pick=rng.random((3,samples))
            hazard=self.hazards[s,np.minimum(a,self.age_cap)-1] if age_hazard else self.geometric[s]
            switch=u_exit<hazard
            s=np.where(switch,1-s,s)
            a=np.where(switch,1,a+1)
            band=np.minimum(np.floor(np.log2(a)).astype(int),7)
            amplitude=np.empty(samples)
            for state in (0,1):
                for b in range(8):
                    ids=np.flatnonzero((s==state)&(band==b))
                    if len(ids)==0:
                        continue
                    pool=self.pools[state]
                    local=self.local_pools[(state,b)]
                    use_local=(u_pool[ids]<len(local)/(len(local)+self.smoothing)) if coupled_amplitude else np.zeros(len(ids),dtype=bool)
                    amplitude[ids]=pool[np.minimum((u_pick[ids]*len(pool)).astype(int),len(pool)-1)]
                    if use_local.any():
                        chosen=ids[use_local]
                        amplitude[chosen]=local[np.minimum((u_pick[chosen]*len(local)).astype(int),len(local)-1)]
            if domain=="price":
                increment=scale*amplitude
                level=level+increment
                # Empirical finite-horizon draws have bounded support, finite moments.
                if np.any(np.abs(level)>700):
                    raise FloatingPointError("Exploding simulated log-price; no silent clipping")
                paths[:,h]=np.exp(level)
                if scale_feedback:
                    scale=np.sqrt(.94*scale**2+.06*increment**2).clip(prepared["floor"])
            else:
                residual=np.where(s==1,0.,scale*amplitude)
                paths[:,h]=prepared["atom"]+residual
                rolling[:,h%96]=residual
                if scale_feedback:
                    scale=np.sqrt(np.mean(rolling**2,axis=1)).clip(prepared["floor"])
        if not np.isfinite(paths).all():
            raise FloatingPointError("Non-finite simulated paths")
        return paths


def reward_factorial(records,domain,horizon=32,samples=512,origins_per_series=6,
                     seed=2021,smoothing=20.,age_cap=128,scale_feedback=True,prediction_dir=None):
    from collections import Counter
    arms={"geometric_independent":(False,False),"age_independent":(True,False),
          "geometric_coupled":(False,True),"age_coupled":(True,True)}
    rows,skipped=[],Counter()
    q=np.arange(.1,1.,.1)
    for index,series in enumerate(records):
        prepared,reason=prepare_reward_series(series,domain)
        if reason:
            skipped[reason]+=1
            continue
        if series.train_end<192 or len(series.values)-horizon-1<series.train_end:
            skipped["no_origins"]+=1
            continue
        try:
            model=EmpiricalDurationReward(prepared,series.train_end,smoothing,age_cap)
        except ValueError:
            skipped["missing_training_state"]+=1
            continue
        origins=np.unique(np.linspace(series.train_end,len(series.values)-horizon-1,origins_per_series,dtype=int))
        for j,origin in enumerate(origins):
            if prepared["ages"][origin] == origin+1 and prepared["ages"][origin] < max(age_cap,128):
                skipped["ambiguous_left_censored_origin"]+=1
                continue
            target=series.values[origin+1:origin+horizon+1]
            target_hash=hashlib.sha256(np.ascontiguousarray(target,dtype="<f8").tobytes()).hexdigest()
            order=np.random.default_rng(seed+index).permutation(list(arms))
            for arm in order:
                ah,ca=arms[arm]
                paths=model.sample(prepared,origin,horizon,samples,seed+index*100+j,
                                   ah,ca,domain,scale_feedback)
                qs=np.quantile(paths,q,axis=0).T
                scores=forecast_scores(target,np.median(paths,axis=0),qs,q)
                if scores["wql9"] is None:
                    raise ValueError("Zero target denominator: macro case-WQL is undefined; do not silently drop the case")
                if prediction_dir is not None:
                    np.savez_compressed(Path(prediction_dir)/f"{index}_{j}_{arm}.npz",
                        quantiles=qs,levels=q,median=np.median(paths,axis=0),mean=paths.mean(0),
                        target=target,target_hash=target_hash,dataset=series.dataset,item=series.item,
                        origin=origin,horizon=horizon)
                rows.append({"dataset":series.dataset,"item":series.item,"origin":int(origin),
                             "cluster":series.dataset if domain=="boom" else str(series.dates[origin])[:7],
                             "arm":arm,"max_abs_quantile":float(np.abs(qs).max()),
                             "mean_abs_target":float(np.abs(target).mean()),
                             "horizon":horizon,"target_hash":target_hash,
                             **{k:scores[k] for k in ("wql9","mse","mae","quantile_loss_numerator","absolute_target_denominator")}})
    if not rows:
        raise RuntimeError("No valid factorial cases")
    frame=pd.DataFrame(rows)
    grouped=frame.groupby(["cluster","arm"])[["wql9","mse","mae"]].mean()
    summary={arm:{metric:float(grouped.xs(arm,level="arm")[metric].mean()) for metric in ("wql9","mse","mae")} for arm in arms}
    pivot=grouped["wql9"].unstack("arm")
    effects={
        "age_effect_independent":pivot.age_independent-pivot.geometric_independent,
        "coupling_effect_geometric":pivot.geometric_coupled-pivot.geometric_independent,
        "interaction":pivot.age_coupled-pivot.age_independent-pivot.geometric_coupled+pivot.geometric_independent,
        "full_minus_geometric_independent":pivot.age_coupled-pivot.geometric_independent}
    return frame,{"domain":domain,"cases":len(frame)//4,"datasets":frame.dataset.nunique(),
                  "skipped":dict(skipped),"scores":summary,
                  "wql9_effects_negative_is_better":{k:paired_cluster_interval(v,seed) for k,v in effects.items()},
                  "sota":False,"interpretation":"Classical empirical semi-Markov factorial baseline; selected development cases only. No novel algorithm claim."}
