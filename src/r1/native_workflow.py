"""Notebook entry steps for the fixed full-variate development comparison."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from r1.data import sha256, stable_hash
from r1.native_protocol import prepare_native, compare_native


def campaign(repo_root, cache_dir, name, config):
    from r1.workflow import external_cache, provenance, save_json
    cache = external_cache(repo_root, cache_dir)
    record = provenance(repo_root)
    identity = dict(config=config, sources=record["sources"])
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    dest = cache/f"{name}_{signature}"
    dest.mkdir(exist_ok=True)
    save_json(dest/"contract.json",dict(config=config,provenance=record))
    snapshot = dest/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in (Path(repo_root)/"src/r1").glob("*.py"):
        (snapshot/path.name).write_bytes(path.read_bytes())
    return dest


def validate_inputs(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root/"manifest.json").read_text())
    if sha256(root/"groups.parquet")!=manifest["groups_sha"]:
        raise ValueError("Native group manifest changed")
    table = pd.read_parquet(root/"groups.parquet")
    expected = set(table.file)
    if {p.name for p in (root/"inputs").glob("*.npz")}!=expected:
        raise ValueError("Native inputs missing or stale")
    for row in table.itertuples():
        if sha256(root/"inputs"/row.file)!=row.input_sha:
            raise ValueError("Native input changed")
    return table


def native_prepare(repo_root, cache_dir, properties_path, max_tasks=48,
                   development_fraction=.7, train_fraction=.7, context_length=2048):
    from r1.workflow import emit
    paths = sorted((p for p in (Path(repo_root)/"input/BOOM").iterdir() if p.is_dir()),key=lambda p:stable_hash(p.name))
    paths = [p for p in paths if stable_hash(p.name)%10<2][:max_tasks]
    config = dict(max_tasks=max_tasks,development_fraction=development_fraction,
                  train_fraction=train_fraction,context_length=context_length,
                  properties_sha=sha256(properties_path),
                  source_hashes={str(f):sha256(f) for p in paths for f in sorted(p.glob("*.arrow"))})
    dest = campaign(repo_root,cache_dir,"native_data",config)
    if not (dest/"manifest.json").exists():
        prepare_native(repo_root,dest,properties_path,max_tasks,development_fraction,train_fraction,context_length)
    validate_inputs(dest)
    result = json.loads((dest/"manifest.json").read_text())
    return emit(dict(data_dir=str(dest),**{k:v for k,v in result.items() if k not in ("source_hashes","boundary_checks")}))


def native_forecast(repo_root,cache_dir,data_dir,model_name,checkpoint=None,
                    runtime_python=None,batch_size=2):
    from r1.workflow import emit,save_json
    table = validate_inputs(data_dir)
    runtime_python = runtime_python or sys.executable
    package_probe = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
    environment = json.loads(subprocess.check_output([runtime_python,"-B","-c",package_probe],text=True))
    weights = {str(p):sha256(p) for p in sorted(Path(checkpoint).rglob("*")) if p.is_file()} if checkpoint else {}
    config = dict(data_manifest_sha=sha256(Path(data_dir)/"manifest.json"),model_name=model_name,
                  checkpoint=checkpoint,weights=weights,runtime_python=runtime_python,
                  packages=environment,batch_size=batch_size)
    dest = campaign(repo_root,cache_dir,"native_"+model_name,config)
    output = dest/"predictions"
    parameters = dict(model_name=model_name,input_dir=str(Path(data_dir)/"inputs"),
                      output_dir=str(output),checkpoint=checkpoint,batch_size=batch_size)
    subprocess.run([runtime_python,"-B",str(dest/"source_snapshot/native_worker.py"),
                    "--parameters",json.dumps(parameters)],check=True,
                   env={**os.environ,"MPLBACKEND":"Agg","PYTHONDONTWRITEBYTECODE":"1"})
    if {p.name for p in output.glob("*.npz")}!=set(table.file):
        raise ValueError("Incomplete model predictions")
    for row in table.itertuples():
        with np.load(output/row.file) as pred, np.load(Path(data_dir)/"inputs"/row.file) as data:
            if str(pred["input_sha"])!=row.input_sha or not np.array_equal(pred["keys"],data["keys"]) or not np.array_equal(pred["target_hashes"],data["target_hashes"]):
                raise ValueError("Output target/input identity mismatch")
            if pred["quantiles"].shape!=(row.channels,row.horizon,9) or not np.isfinite(pred["quantiles"]).all():
                raise ValueError("Malformed output quantiles")
    runtime = json.loads((output/"runtime.json").read_text())
    result = dict(model=model_name,data_dir=str(data_dir),predictions=str(output),groups=len(table),
                  variate_origins=int(table.channels.sum()),runtime={k:v for k,v in runtime.items() if k!="packages"},sota=False)
    save_json(dest/"summary.json",result)
    return emit(result)


def native_compare(repo_root,cache_dir,data_dir,model_directories,upstream_leaderboard):
    from r1.workflow import emit
    validate_inputs(data_dir)
    if set(model_directories)!={"timesfm3","toto2_313m","seasonalnaive"}:
        raise ValueError("The three frozen baseline arms must all be present")
    config = dict(data_manifest_sha=sha256(Path(data_dir)/"manifest.json"),
                  predictions={str(p):sha256(p) for d in model_directories.values() for p in sorted(Path(d).glob("*.npz"))},
                  upstream_sha=sha256(upstream_leaderboard))
    dest = campaign(repo_root,cache_dir,"native_comparison",config)
    result = compare_native(data_dir,model_directories,dest,upstream_leaderboard)
    return emit(dict(cache=str(dest),**result))


def native_diagnose(repo_root,cache_dir,data_dir,comparison_dir):
    from r1.workflow import emit
    from r1.native_diagnostics import diagnose_native
    validate_inputs(data_dir)
    config = dict(data_manifest_sha=sha256(Path(data_dir)/"manifest.json"),
                  case_scores_sha=sha256(Path(comparison_dir)/"case_scores.parquet"),
                  configuration_scores_sha=sha256(Path(comparison_dir)/"configuration_scores.parquet"))
    dest = campaign(repo_root,cache_dir,"native_diagnostics",config)
    result = diagnose_native(data_dir,comparison_dir,dest)
    return emit(dict(cache=str(dest),**result))
