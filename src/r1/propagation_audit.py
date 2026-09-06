"""Falsify a proposed numerical gap before calling distribution propagation new.

No learned model or SOTA claim. FP64 CPU is an explicit reference computation
(MPS has no FP64), not a training-device fallback. The comparator is a classical
categorical projection on the full known reward bound, with no narrow-grid trick.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance


def reward_distribution(logit, horizon, tail, tail_probability, support_count=None, age_cap=4, grid_mode="uniform"):
    device, dtype = logit.device, logit.dtype
    count = horizon*tail+1 if support_count is None else support_count
    if grid_mode not in ("uniform", "log"):
        raise ValueError("Unknown categorical grid mode")
    if grid_mode=="log" and support_count is not None and support_count<3:
        raise ValueError("Log grid needs zero, minimum-positive and upper-bound knots")
    if support_count is not None and grid_mode=="log" and horizon*tail>1:
        # Classical nonuniform linear projection: keep the smallest positive
        # reward at the first positive knot, so it cannot smear into zero.
        grid = torch.cat([torch.zeros(1, dtype=dtype, device=device),
                          torch.logspace(0, np.log10(horizon*tail), count-1, dtype=dtype, device=device)])
    else:
        grid = torch.linspace(0, horizon*tail, count, dtype=dtype, device=device)
    ages = torch.arange(1, age_cap+1, dtype=dtype, device=device)
    rates = torch.cat([(logit+.3*ages.log()).sigmoid(), (-.7+.25*ages.log()).sigmoid()])
    states = torch.arange(2*age_cap, device=device)
    stay = (states//age_cap)*age_cap + ((states%age_cap)+1).clamp(max=age_cap-1)
    switch = (1-states//age_cap)*age_cap
    transition = ((1-rates[:, None])*torch.nn.functional.one_hot(stay, 2*age_cap)
                  + rates[:, None]*torch.nn.functional.one_hot(switch, 2*age_cap))
    mass = torch.zeros(2*age_cap, count, dtype=dtype, device=device)
    mass[0, 0] = 1
    clipped = torch.zeros((), dtype=dtype, device=device)
    def shift(source, reward):
        destination = grid+reward
        overflow = (source*(destination>grid[-1])[None, :]).sum()
        destination = destination.clamp(grid[0], grid[-1])
        lower = (torch.searchsorted(grid, destination, right=True)-1).clamp(0, count-2)
        upper = lower+1
        fraction = (destination-grid[lower])/(grid[upper]-grid[lower])
        out = torch.zeros_like(source).scatter_add(-1, lower[None, :].expand_as(source), source*(1-fraction))
        out = out.scatter_add(-1, upper[None, :].expand_as(source), source*fraction)
        return out, overflow
    for _ in range(horizon):
        moved = transition.T@mass
        small, clip_small = shift(moved[age_cap:], 1)
        large, clip_large = shift(moved[age_cap:], tail)
        mass = torch.cat([moved[:age_cap], (1-tail_probability)*small+tail_probability*large])
        clipped = clipped+(1-tail_probability)*clip_small+tail_probability*clip_large
    return grid, mass.sum(0), clipped


def expected_crps(grid, weights, truth_grid, truth_weights):
    """Expected CRPS against the entire exact target law, without label sampling."""
    x, p = truth_grid.detach().numpy(), truth_weights.detach().numpy()
    index = np.searchsorted(x, grid.detach().numpy(), side="right")
    cdf = np.r_[0., np.cumsum(p)]
    moments = np.r_[0., np.cumsum(x*p)]
    cross = grid*(2*torch.as_tensor(cdf[index], dtype=grid.dtype)-1)
    cross = cross+moments[-1]-2*torch.as_tensor(moments[index], dtype=grid.dtype)
    half_pair = (weights*grid*(2*weights.cumsum(0)-weights-1)).sum()
    return (weights*cross).sum()-half_pair


def audit_propagation(output_dir, horizons=(8, 32), tails=(8, 32), tail_probabilities=(.05, .2),
                       query_probabilities=(.05, .4, .7), supports=(32, 128), truth_probability=.2,
                       finite_difference_step=1e-5, grid_modes=("uniform", "log")):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not 0<truth_probability<1 or any(not 0<p<1 for p in query_probabilities):
        raise ValueError("Hazard probabilities must be interior")
    if min(horizons)<1 or min(tails)<1 or min(supports)<2:
        raise ValueError("Invalid propagation grid")
    rows = []
    torch.set_num_threads(2)
    truth_logit = torch.tensor(np.log(truth_probability/(1-truth_probability)), dtype=torch.float64)
    for horizon in horizons:
        for tail in tails:
            for tail_probability in tail_probabilities:
                truth_x, truth_p, _ = reward_distribution(truth_logit, horizon, tail, tail_probability)
                for query in query_probabilities:
                    theta = torch.tensor(np.log(query/(1-query)), dtype=torch.float64, requires_grad=True)
                    tick = time.perf_counter()
                    exact_x, exact_p, _ = reward_distribution(theta, horizon, tail, tail_probability)
                    exact_risk = expected_crps(exact_x, exact_p, truth_x, truth_p)
                    exact_grad = torch.autograd.grad(exact_risk, theta)[0].item()
                    reference_seconds = time.perf_counter()-tick
                    risks = []
                    for delta in (-finite_difference_step, finite_difference_step):
                        x, p, _ = reward_distribution(theta.detach()+delta, horizon, tail, tail_probability)
                        risks.append(float(expected_crps(x, p, truth_x, truth_p)))
                    finite_gradient = (risks[1]-risks[0])/(2*finite_difference_step)
                    if not np.isclose(exact_grad, finite_gradient, rtol=2e-5, atol=1e-7):
                        raise ValueError("Reference autograd does not match finite differences")
                    px = exact_x.detach().numpy()
                    pp = exact_p.detach().numpy()
                    for support_count, grid_mode in ((n, g) for n in supports for g in grid_modes):
                        theta_approx = theta.detach().clone().requires_grad_(True)
                        tick = time.perf_counter()
                        x, p, clipped = reward_distribution(theta_approx, horizon, tail, tail_probability, support_count, grid_mode=grid_mode)
                        risk = expected_crps(x, p, truth_x, truth_p)
                        gradient = torch.autograd.grad(risk, theta_approx)[0].item()
                        elapsed = time.perf_counter()-tick
                        if not torch.allclose(p.sum(), torch.tensor(1., dtype=p.dtype), atol=1e-10):
                            raise ValueError("Categorical probability not conserved")
                        w1 = wasserstein_distance(px, x.detach().numpy(), pp, p.detach().numpy())
                        # Final equal-weight quantile projection is a diagnostic,
                        # not a full iterative quantile-spline implementation.
                        midpoints = (np.arange(support_count)+.5)/support_count
                        indices = np.searchsorted(np.cumsum(pp), midpoints).clip(max=len(pp)-1)
                        hard_x = torch.as_tensor(px[indices])
                        hard_p = torch.full_like(hard_x, 1/support_count)
                        hard_risk = expected_crps(hard_x, hard_p, truth_x, truth_p)
                        rows.append(dict(horizon=horizon, tail=tail, tail_probability=tail_probability,
                            query_probability=query, supports=support_count, grid_mode=grid_mode, exact_risk=float(exact_risk.detach()),
                            exact_gradient=exact_grad, finite_difference_gradient=finite_gradient,
                            categorical_risk=float(risk.detach()), categorical_gradient=gradient,
                            absolute_gradient_error=abs(gradient-exact_grad),
                            relative_gradient_error=abs(gradient-exact_grad)/max(abs(exact_grad),1e-8),
                            gradient_same_sign=bool(gradient*exact_grad>0), wasserstein1=float(w1),
                            wasserstein1_relative_bound=float(w1/(horizon*tail)),
                            atom_mass_error=float(abs(p[0].detach()-exact_p[0].detach())),
                            clipped_mass_sum=float(clipped.detach()),
                            exact_seconds=reference_seconds, categorical_seconds=elapsed,
                            exact_state_values=8*len(px), categorical_state_values=8*support_count,
                            final_hard_quantile_risk=float(hard_risk),
                            final_hard_quantile_local_hazard_gradient=0.))
    frame = pd.DataFrame(rows)
    report = {}
    for (count, grid_mode), group in frame.groupby(["supports", "grid_mode"]):
        report[f"{grid_mode}_{count}"] = dict(scenarios=len(group),
            median_relative_gradient_error=float(group.relative_gradient_error.median()),
            worst_relative_gradient_error=float(group.relative_gradient_error.max()),
            same_sign_fraction=float(group.gradient_same_sign.mean()),
            median_wasserstein1=float(group.wasserstein1.median()),
            worst_atom_mass_error=float(group.atom_mass_error.max()),
            worst_clipped_mass_sum=float(group.clipped_mass_sum.max()),
            median_reference_seconds=float(group.exact_seconds.median()),
            median_categorical_seconds=float(group.categorical_seconds.median()))
    result = dict(stage="classical_distributional_projection_audit", reference_device="cpu_float64",
        support_results=report, scenarios=len(frame), finite_difference_checks=True,
        sota=False, novelty_established=False,
        scope="Finite known semi-Markov reward toy; classical full-range uniform/nonuniform categorical projection. Log grid is an exploratory follow-up to uniform-grid atom smearing. Hard quantile only at final output. Quantile-spline/OT not benchmarked. No real-data accuracy or training claim.")
    frame.to_parquet(dest/"scenarios.parquet", index=False)
    (dest/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
