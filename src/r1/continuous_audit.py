"""Post-fit controls for scaling effects and genuine age representation gains."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from r1.data import sha256


def audit_continuous(run_dir,output_dir):
    root,dest = Path(run_dir),Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    cases = pd.read_parquet(root/"cases.parquet")
    scores = pd.read_parquet(root/"validation_scores.parquet")
    summary = json.loads((root/"summary.json").read_text())
    domain,seeds = summary["domain"],summary["seeds"]
    valid = cases[cases.partition.eq("validation")].reset_index(drop=True)
    key = ["dataset","item","origin","horizon"]
    if valid.duplicated(key).any():
        raise ValueError("Duplicate validation keys")
    for (arm,seed),sub in scores.groupby(["arm","seed"]):
        if len(sub)!=len(valid) or sub.duplicated(key).any():
            raise ValueError("Incomplete paired arm/seed scores")
    seen = set(cases[cases.partition.eq("train")].dataset)
    valid["training_seen"] = valid.dataset.isin(seen)
    q_levels = np.arange(.1,1,.1)
    # A frozen origin-scale random walk: zero drift, normal increment law.
    random_delta = valid.target_scale.to_numpy()[:,None]*norm.ppf(q_levels)
    random_q = valid.origin_transformed.to_numpy()[:,None]+random_delta
    random_q = np.exp(random_q) if domain=="price" else random_q
    if not np.isfinite(random_q).all():
        raise ValueError("Nonfinite Gaussian control")
    err = valid.raw_target.to_numpy()[:,None]-random_q
    random_loss = (2*np.maximum(q_levels*err,(q_levels-1)*err)).mean(-1)
    random_trans = np.arcsinh(norm.ppf(q_levels))
    te = valid.target_transformed.to_numpy()[:,None]-random_trans
    transformed_loss = (2*np.maximum(q_levels*te,(q_levels-1)*te)).mean(-1)
    baseline = valid[key+["cluster","target_scale","floor_active"]].copy()
    baseline["arm"],baseline["seed"] = "gaussian_random_walk",-1
    baseline["raw_pinball9"] = random_loss
    baseline["transformed_pinball9"] = np.where(valid.target_scale.gt(0),transformed_loss,np.nan)
    baseline["absolute_target"] = np.abs(valid.raw_target.to_numpy())
    if domain=="price":
        baseline["cumulative_return_pinball9"] = random_loss/valid.origin_value.to_numpy()
        baseline["absolute_cumulative_return"] = np.abs(valid.raw_target.to_numpy()/valid.origin_value.to_numpy()-1)
    full = pd.concat([scores,baseline],ignore_index=True)
    # Seeds are averaged within the exact same origin before any grouping.
    metrics = ["brier","transformed_pinball9","raw_pinball9","absolute_target"]
    if domain=="price":
        metrics += ["cumulative_return_pinball9","absolute_cumulative_return"]
    per_origin = full.groupby(key+["arm"])[metrics].mean().reset_index()
    per_origin = per_origin.merge(valid[key+["floor_active","target_scale","training_seen"]],on=key,validate="many_to_one")
    effects = []
    for horizon,part in per_origin[per_origin.arm.ne("gaussian_random_walk")].groupby("horizon"):
        for metric in ("brier","transformed_pinball9"):
            if metric=="transformed_pinball9":
                part_metric = part[part.target_scale.gt(0)]
            else:
                part_metric = part
            pivot = part_metric.pivot(index=key,columns="arm",values=metric).reset_index()
            if not np.isfinite(pivot[["context","true_age","random_age"]].to_numpy()).all():
                raise ValueError("Missing paired metric")
            pivot = pivot.merge(valid[key+["floor_active","target_scale","training_seen","cluster"]],on=key,validate="one_to_one")
            # Match the primary probe: task clusters for BOOM; common calendar
            # month means (without iid confidence intervals) for finance.
            counts = pivot.groupby("cluster").size()
            pivot["denominator"] = pivot.cluster.map(counts)*len(counts)
            pivot["age_context_contribution"] = (pivot.true_age-pivot.context)/pivot.denominator
            pivot["age_random_contribution"] = (pivot.true_age-pivot.random_age)/pivot.denominator
            grouped = pivot.groupby(["floor_active","training_seen"]).agg(
                cases=("item","size"),tasks=("dataset","nunique"),
                age_minus_context_contribution=("age_context_contribution","sum"),
                age_minus_random_contribution=("age_random_contribution","sum"))
            total = (pivot.true_age-pivot.context).groupby(pivot.cluster).mean().mean()
            if not np.isclose(grouped.age_minus_context_contribution.sum(),total,rtol=1e-10,atol=1e-12):
                raise ValueError("Slice contributions do not conserve the declared task mean")
            for row in grouped.reset_index().to_dict("records"):
                effects.append(dict(horizon=int(horizon),metric=metric,**row))
    raw = per_origin.groupby(["dataset","horizon","arm"])[["raw_pinball9","absolute_target"]].sum()
    raw["wql9"] = raw.raw_pinball9/raw.absolute_target
    raw_summary = []
    for (horizon,arm),sub in raw.reset_index().groupby(["horizon","arm"]):
        raw_summary.append(dict(horizon=int(horizon),arm=arm,tasks=len(sub),
            endpoint_wql9=float(sub.wql9.mean()) if np.isfinite(sub.wql9).all() else None,
            undefined_tasks=sub.loc[~np.isfinite(sub.wql9),"dataset"].tolist()))
    returns = []
    if domain=="price":
        r = per_origin.groupby(["dataset","horizon","arm"])[["cumulative_return_pinball9","absolute_cumulative_return"]].sum()
        r["wql9"] = r.cumulative_return_pinball9/r.absolute_cumulative_return
        returns = r.reset_index()[["dataset","horizon","arm","wql9"]].to_dict("records")
    seed_equality = {}
    for arm in ("context","true_age","random_age"):
        with np.load(root/f"{arm}_{seeds[0]}_predictions.npz") as first:
            details = []
            for seed in seeds[1:]:
                with np.load(root/f"{arm}_{seed}_predictions.npz") as other:
                    if not np.array_equal(first["keys"],other["keys"]):
                        raise ValueError("Seed prediction keys changed")
                    details.append(dict(seed=seed,quantiles_identical=bool(np.array_equal(first["quantiles"],other["quantiles"])),
                        event_probabilities_identical=bool(np.array_equal(first["event_probability"],other["event_probability"]))))
            seed_equality[arm] = details
    pd.DataFrame(effects).to_parquet(dest/"slice_contributions.parquet",index=False)
    raw.reset_index().to_parquet(dest/"endpoint_scores.parquet",index=False)
    baseline.to_parquet(dest/"gaussian_control.parquet",index=False)
    result = dict(stage="continuous_probe_attribution",domain=domain,
        validation_only_tasks=sorted(set(valid.dataset)-seen),
        floor_active_fraction=float(valid.floor_active.mean()),zero_scale_cases=int(valid.target_scale.eq(0).sum()),
        h32_effects=[x for x in effects if x["horizon"]==32],raw_endpoint_scores=raw_summary,
        cumulative_return_scores=returns,seed_equality=seed_equality,sota=False,
        scope="Post-fit diagnostic; does not change H32 primary endpoints or remove tasks. Slice effects sum to the primary task-macro (BOOM) or calendar-month-macro (finance) contrast. Gaussian control uses frozen origin scale, not fitted GARCH. Raw endpoints at H1/8/32 are not the native full BOOM protocol. Financial scores are prediction errors, not trading alpha.")
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def continuous_audit(repo_root,cache_dir,run_dir):
    from r1.native_workflow import campaign
    from r1.workflow import emit
    root = Path(run_dir)
    config = dict(input_files={str(p):sha256(p) for p in root.glob("*") if p.is_file()})
    dest = campaign(repo_root,cache_dir,"continuous_audit",config)
    result = audit_continuous(root,dest)
    return emit(dict(cache=str(dest),**result))
