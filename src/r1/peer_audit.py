"""Recompute raw scores and conserve paired effects across peer-count strata."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from r1.data import sha256


def audit_peer(run_dir,output_dir):
    root,dest = Path(run_dir),Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    cases = pd.read_parquet(root/"cases.parquet")
    valid = cases[cases.partition.eq("validation")].reset_index(drop=True)
    scores = pd.read_parquet(root/"validation_scores.parquet")
    summary = json.loads((root/"summary.json").read_text())
    data_contract = json.loads((root/"data_contract.json").read_text())
    domain,seeds = summary["domain"],summary["seeds"]
    keys = ["dataset","item","origin","horizon"]
    if valid.duplicated(keys).any():
        raise ValueError("Duplicate reference cases")
    expected = {(arm,seeds[0]) for arm in ("own_context","context","true_age")}|{("random_age",s) for s in seeds}
    if set(scores.groupby(["arm","seed"]).groups)!=expected:
        raise ValueError("Missing or unexpected fitted arm/seed")
    columns = summary["feature_columns"]
    if set(columns["context"])!=set(columns["true_age"])-{"log_age"} or "log_age" not in columns["true_age"] or columns["true_age"]!=columns["random_age"]:
        raise ValueError("True and placebo age are not matched one-feature contrasts")
    if columns["own_context"]!=data_contract["own_columns"] or not set(columns["own_context"]).issubset(columns["context"]):
        raise ValueError("Own-context columns differ from the frozen contract")
    levels = np.arange(.1,1,.1)
    recomputed = []
    for arm,seed in sorted(expected):
        arm_scores = scores[scores.arm.eq(arm)&scores.seed.eq(seed)]
        if len(arm_scores)!=len(valid) or arm_scores.duplicated(keys).any():
            raise ValueError("Unexpected extra or duplicate scored cases")
        if set(map(tuple,arm_scores[keys].to_numpy()))!=set(map(tuple,valid[keys].to_numpy())):
            raise ValueError("Scored case keys differ from the frozen cases")
        recorded = arm_scores.set_index(keys)
        recorded = recorded.reindex(pd.MultiIndex.from_frame(valid[keys]))
        if len(recorded)!=len(valid) or recorded.brier.isna().any():
            raise ValueError("Incomplete scored cases")
        with np.load(root/f"{arm}_{seed}_predictions.npz") as saved:
            np.testing.assert_array_equal(saved["keys"],valid[keys].to_numpy(dtype=str))
            quantiles,probability = saved["quantiles"],saved["event_probability"]
            if quantiles.shape!=(len(valid),9) or probability.shape!=(len(valid),):
                raise ValueError("Saved prediction shape differs from the contract")
            if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles,axis=-1)<0):
                raise ValueError("Invalid saved quantiles")
            if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
                raise ValueError("Invalid saved probabilities")
            error = valid.raw_target.to_numpy()[:,None]-quantiles
            loss = (2*np.maximum(levels*error,(levels-1)*error)).mean(-1)
            brier = (probability-valid.event_label.to_numpy())**2
            np.testing.assert_allclose(loss,recorded.raw_pinball9,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(brier,recorded.brier,rtol=1e-12,atol=1e-12)
            if domain=="price":
                np.testing.assert_allclose(loss/valid.origin_value.to_numpy(),recorded.cumulative_return_pinball9,rtol=1e-12,atol=1e-12)
            zero = valid.target_scale.eq(0).to_numpy()
            np.testing.assert_allclose(quantiles[zero],np.repeat(valid.origin_value.to_numpy()[zero,None],9,axis=1),rtol=1e-12)
            recomputed.append(dict(arm=arm,seed=seed,cases=len(valid),raw_and_event_scores_match=True,zero_scale_forecast_verified=True))
    measures = ["brier","transformed_pinball9","raw_pinball9","absolute_target"]
    if domain=="price":
        measures += ["cumulative_return_pinball9","absolute_cumulative_return"]
    # Identical case weights for one deterministic fit and three placebo fits.
    per_case = scores.groupby(keys+["arm"])[measures].mean().reset_index()
    seen = set(cases.loc[cases.partition.eq("train"),"native_group"])
    valid["training_group_seen"] = valid.native_group.isin(seen)
    valid["peer_stratum"] = np.select([valid.peer_count.eq(0),valid.peer_count.eq(1)],["no_peer","single_peer"],default="multiple_peers")
    metadata = ["cluster","target_scale","floor_active","peer_stratum","training_group_seen"]
    main = per_case[per_case.horizon.eq(32)]
    effects,cluster_effects = [],[]
    for metric in ("brier","transformed_pinball9"):
        pivot = main.pivot(index=keys,columns="arm",values=metric).reset_index()
        pivot = pivot.merge(valid[keys+metadata],on=keys,validate="one_to_one")
        if metric=="transformed_pinball9":
            pivot = pivot[pivot.target_scale.gt(0)].copy()
        if not np.isfinite(pivot[["own_context","context","true_age","random_age"]]).all().all():
            raise ValueError("Incomplete matched metrics")
        counts = pivot.groupby("cluster").size()
        denominator = pivot.cluster.map(counts)*len(counts)
        for name,left,right in (("age_minus_peer","true_age","context"),("age_minus_random","true_age","random_age"),
                                ("peer_minus_own","context","own_context"),("age_minus_own","true_age","own_context")):
            delta = pivot[left]-pivot[right]
            pivot["contribution"] = delta/denominator
            grouped = pivot.groupby(["peer_stratum","floor_active","training_group_seen"]).agg(
                cases=("item","size"),tasks=("dataset","nunique"),contribution=("contribution","sum"))
            cluster = delta.groupby(pivot.cluster).mean()
            np.testing.assert_allclose(grouped.contribution.sum(),cluster.mean(),rtol=1e-10,atol=1e-12)
            effects.extend(dict(metric=metric,contrast=name,**r) for r in grouped.reset_index().to_dict("records"))
            cluster_effects.extend(dict(metric=metric,contrast=name,cluster=k,difference=float(v)) for k,v in cluster.items())
            reference = next(s for s in summary["scores"] if s["horizon"]==32 and s["metric"]==metric)
            np.testing.assert_allclose(cluster.mean(),reference["scores"][left]-reference["scores"][right],rtol=1e-9,atol=1e-12)
    raw = per_case.groupby(["dataset","horizon","arm"])[["raw_pinball9","absolute_target"]].sum()
    raw["wql9"] = raw.raw_pinball9/raw.absolute_target
    raw_records = []
    for (horizon,arm),sub in raw.reset_index().groupby(["horizon","arm"]):
        raw_records.append(dict(horizon=int(horizon),arm=arm,wql9=float(sub.wql9.mean()) if np.isfinite(sub.wql9).all() else None))
    gaussian = valid.origin_transformed.to_numpy()[:,None]+valid.target_scale.to_numpy()[:,None]*norm.ppf(levels)
    gaussian = np.exp(gaussian) if domain=="price" else gaussian
    error = valid.raw_target.to_numpy()[:,None]-gaussian
    gaussian_loss = (2*np.maximum(levels*error,(levels-1)*error)).mean(-1)
    baseline = valid[keys].copy()
    baseline["raw_pinball9"],baseline["absolute_target"] = gaussian_loss,valid.raw_target.abs()
    grouped = baseline.groupby(["dataset","horizon"])[["raw_pinball9","absolute_target"]].sum()
    grouped["wql9"] = grouped.raw_pinball9/grouped.absolute_target
    raw_records += [dict(horizon=int(h),arm="gaussian_random_walk",wql9=float(s.wql9.mean()) if np.isfinite(s.wql9).all() else None)
                    for h,s in grouped.reset_index().groupby("horizon")]
    returns = []
    if domain=="price":
        ret = per_case.groupby(["dataset","horizon","arm"])[["cumulative_return_pinball9","absolute_cumulative_return"]].sum()
        ret["wql9"] = ret.cumulative_return_pinball9/ret.absolute_cumulative_return
        returns = ret.reset_index()[["dataset","horizon","arm","wql9"]].to_dict("records")
        baseline["return_loss"] = gaussian_loss/valid.origin_value.to_numpy()
        baseline["absolute_return"] = np.abs(valid.raw_target.to_numpy()/valid.origin_value.to_numpy()-1)
        rr = baseline.groupby(["dataset","horizon"])[["return_loss","absolute_return"]].sum()
        rr["wql9"] = rr.return_loss/rr.absolute_return
        returns += [dict(arm="gaussian_random_walk",**r) for r in rr.reset_index()[["dataset","horizon","wql9"]].to_dict("records")]
    h32 = [r for r in summary["scores"] if r["horizon"]==32]
    if len(h32)!=2 or {r["metric"] for r in h32}!={"brier","transformed_pinball9"}:
        raise ValueError("Missing H32 primary endpoints")
    gate = all(r["age_minus_context_mean"]<0 and r["age_minus_random_mean"]<0 for r in h32)
    result = dict(stage="peer_age_attribution_audit",domain=domain,prediction_checks=recomputed,
        primary_h32_mean_gate_passed=gate,raw_endpoint_scores=raw_records,
        h32_slice_contributions=effects,cumulative_return_scores=returns,
        no_peer_tasks=valid.loc[valid.peer_count.eq(0),"dataset"].nunique(),
        single_peer_tasks=valid.loc[valid.peer_count.eq(1),"dataset"].nunique(),
        multiple_peer_tasks=valid.loc[valid.peer_count.ge(2),"dataset"].nunique(),
        transformed_quantiles_independently_recomputed=False,
        transformed_score_audit_limit="Transformed-space quantiles were not saved by this fit. Their scores are checked for paired completeness and aggregation, not independently reconstructed from raw values with potentially unstable inverse cancellation.",
        scope="Development diagnostic. Mean gate alone does not certify a mechanism; raw-scale guardrails and uncertainty remain separate. No peer strata are removed from the primary result. Shared model fitting can change no-peer predictions. No independent confirmation, new algorithm, or SOTA.",sota=False)
    pd.DataFrame(effects).to_parquet(dest/"slice_contributions.parquet",index=False)
    pd.DataFrame(cluster_effects).to_parquet(dest/"cluster_effects.parquet",index=False)
    raw.reset_index().to_parquet(dest/"endpoint_scores.parquet",index=False)
    (dest/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def peer_audit(repo_root,cache_dir,run_dir):
    from r1.native_workflow import campaign
    from r1.workflow import emit
    root = Path(run_dir)
    config = dict(input_files={str(p):sha256(p) for p in root.glob("*") if p.is_file()})
    dest = campaign(repo_root,cache_dir,"peer_audit",config)
    return emit(dict(cache=str(dest),**audit_peer(root,dest)))
