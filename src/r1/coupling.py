"""Supervised duration--amplitude mechanism audit on a frozen development pool.

This is a neural semi-Markov control, not a claimed new algorithm. Hazard
supervision and emission fitting are separated so a calibration gate cannot
relabel arbitrary probabilities as durations. Continuous emissions can depend
on the propagated future age, rather than only on the observed origin age.
These are calibration experts, not identified non-atom conditional emissions:
the transformed frozen support can itself retain a point mass. Equal allocated
parameters do not remove the coupled arm's greater mixture capacity.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from r1.data import sha256
from r1.evaluation import forecast_scores, paired_cluster_interval
from r1.models import select_device


def age_bands(cap, device):
    return torch.log2(torch.arange(1, cap + 1, device=device).float()).floor().long().clamp(max=7)


def propagate(hazards, state, age, horizon, cap=128):
    """Exact finite, top-coded state-age recursion, aggregated into 8 age bands."""
    if cap < 2 or horizon < 1:
        raise ValueError("State-age recursion requires cap >= 2 and horizon >= 1")
    bands = age_bands(cap, hazards.device)
    expanded = hazards[..., bands]
    location = state.long() * cap + age.long().clamp(1, cap) - 1
    mass = torch.nn.functional.one_hot(location, 2 * cap).to(hazards.dtype).reshape(-1, 2, cap)
    band_map = torch.nn.functional.one_hot(bands, 8).to(hazards.dtype)
    result = []
    for _ in range(horizon):
        stay = mass * (1 - expanded)
        switch = (mass * expanded).sum(-1).flip(-1)[..., None]
        mass = torch.cat([switch, stay[..., :-2], stay[..., -2:].sum(-1, keepdim=True)], -1)
        result.append(mass @ band_map)
    return torch.stack(result, 1)


def transition_labels(target, state, age, cap=128):
    """Training labels only; do not pass these labels into a prediction head."""
    states, bands, exits = [], [], []
    current, elapsed = state.long(), age.long().clamp(1, cap)
    for h in range(target.shape[1]):
        following = (target[:, h] == 0).long()
        changed = following != current
        states.append(current)
        bands.append(elapsed.float().log2().floor().long().clamp(max=7))
        exits.append(changed.float())
        elapsed = torch.where(changed, 1, elapsed + 1).clamp(max=cap)
        current = following
    return torch.stack(states, 1), torch.stack(bands, 1), torch.stack(exits, 1)


def base_support(base, count=32):
    """Midpoint discretization of the frozen piecewise-linear quantile law."""
    q = base.sort(-1).values
    extended = torch.cat([2*q[..., :1]-q[..., 1:2], q, 2*q[..., -1:]-q[..., -2:-1]], -1)
    position = (torch.arange(count, device=q.device, dtype=q.dtype) + .5) * 10 / count
    lower = position.floor().long().clamp(0, 9)
    return extended[..., lower] + (position-lower) * (extended[..., lower+1]-extended[..., lower])


def weighted_crps(points, weights, target):
    """Exact CRPS for the represented discrete distribution; O(K log K)."""
    ordered, indices = points.sort(-1)
    w = weights.gather(-1, indices)
    first = (weights * (points-target[..., None]).abs()).sum(-1)
    half_pair_distance = (w * ordered * (2*w.cumsum(-1)-w-1)).sum(-1)
    return first - half_pair_distance


def distribution_quantiles(points, weights):
    ordered, indices = points.sort(-1)
    cumulative = weights.gather(-1, indices).cumsum(-1)
    levels = torch.arange(1, 10, device=points.device, dtype=points.dtype) / 10
    indices = (cumulative[..., None] < levels).sum(-2).clamp(max=points.shape[-1]-1)
    return ordered.gather(-1, indices)


class HazardHead(nn.Module):
    def __init__(self, inputs, width=64, duration="age"):
        super().__init__()
        self.duration = duration
        self.encoder = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        # Same parameter allocation in both arms; constant hazard averages the
        # eight logits. Origin-age information is shared and explicitly retained.
        self.head = nn.Linear(width, 16)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -2.)

    def forward(self, features):
        logits = self.head(self.encoder(features)).reshape(-1, 2, 8)
        if self.duration == "geometric":
            logits = logits.mean(-1, keepdim=True).expand(-1, -1, 8)
        return logits


class EmissionHead(nn.Module):
    def __init__(self, inputs, width=64, coupled=False):
        super().__init__()
        self.coupled = coupled
        self.encoder = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        # Equal parameter count; the independent arm averages age logits before
        # applying location/scale transforms. This alone is not equal capacity.
        self.head = nn.Linear(width, 17)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[-1] = -4.

    def forward(self, features, support, occupancy):
        out = self.head(self.encoder(features))
        location, logscale = out[:, :8], out[:, 8:16]
        if not self.coupled:
            location = location.mean(-1, keepdim=True).expand(-1, 8)
            logscale = logscale.mean(-1, keepdim=True).expand(-1, 8)
        # Bounded emission transforms prevent uncontrolled scale feedback.
        shift = 2 * location.tanh()
        scale = (.7 * logscale.tanh()).exp()
        emissions = support[:, :, None, :] * scale[:, None, :, None] + shift[:, None, :, None]
        batch, horizon, count = support.shape
        gate = out[:, -1].sigmoid()
        base_w = ((1-gate)[:, None, None] / count).expand(-1, horizon, count)
        nonatom_w = (gate[:, None, None] * occupancy[:, :, 0])[..., None].expand(-1, -1, -1, count) / count
        atom_w = gate[:, None] * occupancy[:, :, 1].sum(-1)
        points = torch.cat([support, emissions.reshape(batch, horizon, -1), torch.zeros_like(atom_w)[..., None]], -1)
        weights = torch.cat([base_w, nonatom_w.reshape(batch, horizon, -1), atom_w[..., None]], -1)
        return points, weights, gate


def validate_coupling():
    torch.manual_seed(2021)
    hazards = torch.full((3, 2, 8), .2)
    state = torch.tensor([0., 1., 0.])
    ages = torch.tensor([1., 128., 20.])
    occupancy = propagate(hazards, state, ages, 8)
    assert torch.allclose(occupancy.sum((-1, -2)), torch.ones(3, 8), atol=1e-6)
    # Known two-state Markov occupancy, valid at every initial age.
    expected = .5 + (state-.5)[:, None] * .6**torch.arange(1, 9)[None, :]
    assert torch.allclose(occupancy[:, :, 1].sum(-1), expected, atol=1e-6)
    x = torch.randn(3, 4, 11, requires_grad=True)
    w = torch.rand(3, 4, 11)
    w = w / w.sum(-1, keepdim=True)
    y = torch.randn(3, 4)
    slow = (w*(x-y[..., None]).abs()).sum(-1) - .5*(w[..., :, None]*w[..., None, :]*(x[..., :, None]-x[..., None, :]).abs()).sum((-1, -2))
    fast = weighted_crps(x, w, y)
    assert torch.allclose(fast, slow, atol=1e-6)
    fast.mean().backward()
    assert torch.isfinite(x.grad).all()
    q = distribution_quantiles(x.detach(), w)
    assert (q.diff(dim=-1) >= 0).all()
    # Independent slow path enumeration also exercises nonconstant hazards.
    h = torch.rand(1, 2, 8)*.6
    paths = [(0, 3, 1.)]
    for t in range(4):
        updated = []
        for s, a, p in paths:
            rate = float(h[0, s, min(int(np.log2(a)), 7)])
            updated.extend([(s, min(a+1, 128), p*(1-rate)), (1-s, 1, p*rate)])
        paths = updated
        exact = np.zeros((2, 8))
        for s, a, p in paths:
            exact[s, min(int(np.log2(a)), 7)] += p
        calculated = propagate(h, torch.tensor([0]), torch.tensor([3]), t+1)[0, -1]
        assert np.allclose(calculated.numpy(), exact, atol=1e-6)
    return {"passed": True, "checks": ["probability conservation", "Markov closed form", "nonconstant-hazard path enumeration", "CRPS versus pairwise definition", "finite gradients", "ordered quantiles"]}


def fit_coupling(data_dir, output_dir, seeds=(2021, 2022, 2023), hazard_steps=400,
                 emission_steps=400, batch_size=64, width=64, learning_rate=.002,
                 support_count=32, age_cap=128):
    root, dest = Path(data_dir), Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    if sha256(root/"arrays.npz") != manifest["arrays_sha"] or sha256(root/"cases.parquet") != manifest["metadata_sha"]:
        raise ValueError("Frozen refinement data changed")
    arrays = np.load(root/"arrays.npz")
    meta = pd.read_parquet(root/"cases.parquet")
    device = select_device()
    tensors = {k: torch.as_tensor(np.asarray(arrays[k]), dtype=torch.float32, device=device)
               for k in ("features", "base", "target", "state", "age")}
    train = np.flatnonzero(meta.partition.eq("train"))
    valid = np.flatnonzero(meta.partition.eq("validation"))
    counts = meta.iloc[train].groupby("dataset").size()
    probabilities = meta.iloc[train].dataset.map(lambda x: 1/counts[x]).to_numpy()
    probabilities = probabilities / probabilities.sum()
    support = base_support(tensors["base"], support_count)
    labels = transition_labels(tensors["target"], tensors["state"], tensors["age"], age_cap)
    rows, runtimes, event_rows = [], [], []
    for seed in seeds:
        for duration in np.random.default_rng(seed).permutation(["geometric", "age"]):
            torch.manual_seed(seed)
            hazard = HazardHead(tensors["features"].shape[-1], width, duration).to(device)
            optimizer = torch.optim.AdamW(hazard.parameters(), lr=learning_rate, weight_decay=.0001)
            rng = np.random.default_rng(seed)
            tick = time.perf_counter()
            for _ in range(hazard_steps):
                ids = torch.as_tensor(rng.choice(train, batch_size, p=probabilities), device=device)
                logits = hazard(tensors["features"][ids])
                selected = logits[torch.arange(batch_size, device=device)[:, None], labels[0][ids], labels[1][ids]]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(selected, labels[2][ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hazard.parameters(), 1.)
                optimizer.step()
            hazard.eval()
            occupancies = []
            with torch.no_grad():
                for start in range(0, len(meta), batch_size):
                    ids = slice(start, start+batch_size)
                    occupancies.append(propagate(hazard(tensors["features"][ids]).sigmoid(),
                        tensors["state"][ids], tensors["age"][ids], support.shape[1], age_cap))
            occupancy = torch.cat(occupancies)
            event_probability = occupancy[valid, :, 1].sum(-1).cpu().numpy()
            for j, index in enumerate(valid):
                event_rows.append(dict(dataset=meta.iloc[index]["dataset"], seed=int(seed), duration=str(duration),
                    brier=float(np.mean((event_probability[j]-(arrays["target"][index]==0))**2))))
            torch.save(hazard.cpu().state_dict(), dest/f"hazard_{duration}_{seed}.pt")
            hazard_seconds = time.perf_counter()-tick
            for coupled in np.random.default_rng(seed).permutation([False, True]):
                arm = str(duration) + ("_coupled" if coupled else "_independent")
                torch.manual_seed(seed)
                model = EmissionHead(tensors["features"].shape[-1], width, bool(coupled)).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.0001)
                rng = np.random.default_rng(seed)
                tick = time.perf_counter()
                for _ in range(emission_steps):
                    ids = torch.as_tensor(rng.choice(train, batch_size, p=probabilities), device=device)
                    points, weights, _ = model(tensors["features"][ids], support[ids], occupancy[ids])
                    loss = weighted_crps(points, weights, tensors["target"][ids]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite duration--amplitude CRPS")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                model.eval()
                allq, allgate = [], []
                with torch.no_grad():
                    for start in range(0, len(valid), batch_size):
                        ids = valid[start:start+batch_size]
                        points, weights, gate = model(tensors["features"][ids], support[ids], occupancy[ids])
                        if not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=2e-5):
                            raise ValueError("Prediction mass not conserved")
                        allq.append(distribution_quantiles(points, weights).cpu().numpy())
                        allgate.append(gate.cpu().numpy())
                allq, allgate = np.concatenate(allq), np.concatenate(allgate)
                torch.save(model.cpu().state_dict(), dest/f"{arm}_{seed}.pt")
                np.savez_compressed(dest/f"{arm}_{seed}_predictions.npz", indices=valid, quantiles=allq, gate=allgate)
                for j, index in enumerate(valid):
                    scale, atom = arrays["scale"][index], arrays["atom"][index]
                    q, target = allq[j]*scale+atom, arrays["target"][index]*scale+atom
                    score = forecast_scores(target, q[:, 4], q, np.arange(.1, 1, .1))
                    rows.append(dict(dataset=meta.iloc[index]["dataset"], item=meta.iloc[index]["item"],
                        origin=int(meta.iloc[index].origin), seed=int(seed), arm=arm, gate=float(allgate[j]), **score))
                elapsed = time.perf_counter()-tick
                runtimes.append(dict(seed=int(seed), arm=arm, emission_seconds=elapsed, hazard_seconds=hazard_seconds))
                print(f"Coupling {arm} seed={seed}: {elapsed:.1f}s", flush=True)
    with torch.no_grad():
        discrete = distribution_quantiles(support[valid], torch.ones_like(support[valid])/support_count).cpu().numpy()
    for j, index in enumerate(valid):
        for arm, normalized in (("frozen_timesfm3", arrays["base"][index]), ("discretized_timesfm3", discrete[j])):
            scale, atom = arrays["scale"][index], arrays["atom"][index]
            q, target = normalized*scale+atom, arrays["target"][index]*scale+atom
            rows.append(dict(dataset=meta.iloc[index]["dataset"], item=meta.iloc[index]["item"],
                origin=int(meta.iloc[index].origin), seed=-1, arm=arm, gate=0.,
                **forecast_scores(target, q[:, 4], q, np.arange(.1, 1, .1))))
    frame = pd.DataFrame(rows)
    if frame.wql9.isna().any():
        raise ValueError("Undefined WQL in coupling audit")
    grouped = frame.groupby(["dataset", "arm"])[["wql9", "mse", "gate"]].mean()
    pivot = grouped.wql9.unstack("arm")
    interaction = (pivot.age_coupled-pivot.age_independent) - (pivot.geometric_coupled-pivot.geometric_independent)
    events = pd.DataFrame(event_rows).groupby(["dataset", "duration"]).brier.mean().unstack()
    result = dict(stage="supervised_duration_amplitude", seeds=list(seeds), device=str(device),
        training_cases=len(train), validation_cases=len(valid), datasets=len(pivot), support_count=support_count,
        scores={arm: grouped.xs(arm, level="arm").mean().to_dict() for arm in frame.arm.unique()},
        interaction=paired_cluster_interval(interaction),
        age_coupled_minus_independent=paired_cluster_interval(pivot.age_coupled-pivot.age_independent),
        age_coupled_minus_geometric_coupled=paired_cluster_interval(pivot.age_coupled-pivot.geometric_coupled),
        age_coupled_minus_frozen=paired_cluster_interval(pivot.age_coupled-pivot.frozen_timesfm3),
        event_brier=events.mean().to_dict(), event_age_minus_geometric=paired_cluster_interval(events.age-events.geometric),
        runtimes=runtimes, sota=False,
        scope="Exploratory neural semi-Markov control; future age changes emission law; no novel algorithm established; not full-suite; marginal scores do not validate joint paths")
    frame.to_parquet(dest/"case_scores.parquet", index=False)
    pd.DataFrame(event_rows).to_parquet(dest/"event_scores.parquet", index=False)
    (dest/"summary.json").write_text(json.dumps(result, indent=2))
    return result
