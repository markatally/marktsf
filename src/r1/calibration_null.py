"""Known-Bayes-quantile null: hindsight gains are not learnable signal."""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from r1.calibration_probe import calibration_grid
from r1.evaluation import paired_cluster_interval


def known_forecast(domain,horizon,replicates,rho,seed):
    rng = np.random.default_rng(seed)
    h = np.arange(1,horizon+1,dtype=float)
    levels = np.arange(.1,1,.1)
    if domain=="gaussian_ar":
        if not abs(rho)<1:
            raise ValueError("AR null requires |rho|<1")
        sd = np.sqrt((1-rho**(2*h))/(1-rho*rho))
        q = sd[:,None]*norm.ppf(levels)
        innovations = rng.normal(size=(2,replicates,horizon))
        values = innovations.copy()
        for t in range(1,horizon):
            values[...,t]+=rho*values[...,t-1]
    elif domain=="log_price_rw":
        sigma = .02
        # Known Gaussian increments with drift giving a martingale price.
        drift = -.5*sigma*sigma
        q = 100*np.exp(drift*h[:,None]+sigma*np.sqrt(h)[:,None]*norm.ppf(levels))
        increments = drift+sigma*rng.normal(size=(2,replicates,horizon))
        values = 100*np.exp(increments.cumsum(axis=-1))
    else:
        raise ValueError(domain)
    return values[0],values[1],q


def grid_losses(values,quantiles,grid):
    levels = np.arange(.1,1,.1)
    midpoint = quantiles[:,4,None]
    halfwidth = (quantiles[:,-1,None]-quantiles[:,0,None])/2
    residual = quantiles-midpoint
    losses = np.empty((len(grid),*values.shape),dtype=float)
    for i,(bias,spread) in enumerate(grid):
        q = quantiles if (bias,spread)==(0.,1.) else midpoint+bias*halfwidth+spread*residual
        error = values[...,None]-q
        losses[i] = (2*np.maximum(levels*error,(levels-1)*error)).mean(-1)
    return losses


def run_null(output_dir,horizons=(48,480),replicates=2048,rhos=(0.,.9),seed=873,
             biases=(-.5,-.25,0.,.25,.5),spreads=(0.,.5,.75,1.,1.25,1.5,2.)):
    dest = Path(output_dir)
    grid = calibration_grid(biases,spreads)
    if replicates<32 or any(h<2 for h in horizons):
        raise ValueError("Need independent path pairs and horizons permitting a split")
    rows,results = [],[]
    tick = time.perf_counter()
    settings = [("gaussian_ar",rho,h) for h in horizons for rho in rhos]+[("log_price_rw",None,h) for h in horizons]
    for number,(domain,rho,horizon) in enumerate(settings):
        # A and B are independent future paths conditional on the same known x0.
        a,b,q = known_forecast(domain,horizon,replicates,rho,seed+number)
        la,lb = grid_losses(a,q,grid),grid_losses(b,q,grid)
        total_a = la.mean(axis=-1)
        selected = total_a.argmin(axis=0)
        index = np.arange(replicates)
        baseline_a,baseline_b = total_a[0],lb[0].mean(axis=-1)
        oracle_a = total_a[selected,index]
        transfer_b = lb[selected,index].mean(axis=-1)
        half = horizon//2
        half_left = la[:,:,:half].sum(axis=-1).argmin(axis=0)
        half_right = la[:,:,half:].sum(axis=-1).argmin(axis=0)
        oracle_half = (la[half_left,index,:half].sum(axis=-1)+la[half_right,index,half:].sum(axis=-1))/horizon
        transfer_half = (lb[half_left,index,:half].sum(axis=-1)+lb[half_right,index,half:].sum(axis=-1))/horizon
        for r in range(replicates):
            rows.append(dict(domain=domain,rho=rho,horizon=horizon,replicate=r,
                baseline_a=baseline_a[r],oracle_path_a=oracle_a[r],oracle_half_a=oracle_half[r],
                baseline_b=baseline_b[r],transferred_path_b=transfer_b[r],transferred_half_b=transfer_half[r],
                path_grid=int(selected[r]),half_left_grid=int(half_left[r]),half_right_grid=int(half_right[r])))
        difference = transfer_b-baseline_b
        half_difference = transfer_half-baseline_b
        entry = dict(domain=domain,rho=rho,horizon=int(horizon),independent_path_pairs=replicates,
            baseline_mean_pinball=float(baseline_a.mean()),
            hindsight_path_improvement_fraction=float(1-oracle_a.mean()/baseline_a.mean()),
            hindsight_half_improvement_fraction=float(1-oracle_half.mean()/baseline_a.mean()),
            independent_transfer_path_degradation_fraction=float(difference.mean()/baseline_b.mean()),
            independent_transfer_half_degradation_fraction=float(half_difference.mean()/baseline_b.mean()),
            independent_transfer_path_difference_ci=paired_cluster_interval(pd.Series(difference)),
            independent_transfer_half_difference_ci=paired_cluster_interval(pd.Series(half_difference)))
        results.append(entry)
        print(f"Calibrated null {domain} rho={rho} H={horizon}: hindsight gain={entry['hindsight_path_improvement_fraction']:.3%}; independent transfer change={entry['independent_transfer_path_degradation_fraction']:.3%}",flush=True)
        del la,lb
    pd.DataFrame(rows).to_parquet(dest/"independent_path_pairs.parquet",index=False)
    result = dict(stage="known_distribution_hindsight_null",settings=results,grid=grid,seconds=time.perf_counter()-tick,
        sota=False,known_conditional_quantiles=True,
        scope="Synthetic diagnostic with exactly specified Bayes-optimal marginal quantiles. Oracle selection sees path A; transferred parameters are scored on independent path B from the same conditional distribution. Replicates are independent path pairs, not independent times within a path. Large hindsight improvements can occur without any forecast misspecification. This does not prove Toto is calibrated, model real BOOM dynamics, or establish a new forecasting method.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def calibration_null(repo_root,cache_dir,horizons=(48,480),replicates=2048,rhos=(0.,.9),seed=873,
                       biases=(-.5,-.25,0.,.25,.5),spreads=(0.,.5,.75,1.,1.25,1.5,2.)):
    from r1.native_workflow import campaign
    from r1.workflow import emit
    config = dict(horizons=list(horizons),replicates=replicates,rhos=list(rhos),seed=seed,
                  biases=list(biases),spreads=list(spreads),
                  claim="A known-correct forecast may exhibit substantial hindsight oracle gains that do not transfer to independent future paths.")
    dest = campaign(repo_root,cache_dir,"calibration_null",config)
    return emit(dict(cache=str(dest),**run_null(dest,horizons,replicates,rhos,seed,biases,spreads)))
