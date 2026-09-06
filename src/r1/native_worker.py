"""One isolated foundation or classical baseline runtime for native BOOM cases."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np


def run(model_name, input_dir, output_dir, checkpoint=None, batch_size=2):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(input_dir).glob("*.npz"))
    device = "cpu"
    if model_name in ("timesfm3", "toto2_313m"):
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        if model_name=="timesfm3":
            from timesfm3 import TimesFM3Evaluator, ModelConfig
            model = TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=batch_size, device=str(device)))
        else:
            from toto2 import Toto2Model
            from toto_worker import CPUStatistics
            model = Toto2Model.from_pretrained(checkpoint,map_location="cpu").to(device).eval()
            if device.type=="mps":
                model.scaler = CPUStatistics(model.scaler)
    elif model_name=="seasonalnaive":
        from gluonts.time_feature import get_seasonality
        from statsforecast.models import SeasonalNaive
    else:
        raise ValueError("Unknown native baseline")
    tick = time.perf_counter()
    for index, path in enumerate(paths):
        with np.load(path) as data:
            context, horizon = data["context"], int(data["horizon"])
            output_path = dest/path.name
            if output_path.exists():
                with np.load(output_path) as previous:
                    if np.array_equal(previous["target_hashes"],data["target_hashes"]) and np.array_equal(previous["keys"],data["keys"]) and str(previous["input_sha"])==hashlib.sha256(path.read_bytes()).hexdigest() and previous["quantiles"].shape==(context.shape[0],horizon,9) and np.isfinite(previous["quantiles"]).all():
                        continue
                raise ValueError("Corrupt completed native cache; do not silently reuse")
            if model_name=="timesfm3":
                out = list(model.predict_batch([context], horizon=horizon, return_quantiles=True, use_symmetric_averaging=True))
                if len(out)!=1:
                    raise ValueError("Native group was split unexpectedly")
                out = out[0]
                q = np.asarray(out.quantiles).reshape(context.shape[0],horizon,9)
            elif model_name=="toto2_313m":
                with torch.inference_mode():
                    x = torch.as_tensor(context,dtype=torch.float32,device=device)[None]
                    padding = (-x.shape[-1])%model.config.patch_size
                    target = torch.nn.functional.pad(x,(padding,0))
                    mask = torch.ones_like(target,dtype=torch.bool)
                    if padding:
                        mask[..., :padding] = False
                    ids = torch.zeros(target.shape[:2],dtype=torch.long,device=device)
                    out = model.forecast({"target":target,"target_mask":mask,"series_ids":ids},
                        horizon=horizon,decode_block_size=768,has_missing_values=bool(padding))
                    q = out[:,0].permute(1,2,0).float().cpu().numpy()
            else:
                season = get_seasonality(str(data["freq"]))
                qs = []
                for y in context:
                    out = SeasonalNaive(season).forecast(y=y,h=horizon,level=[0,20,40,60,80])
                    qs.append(np.stack([out[f'{"lo" if i<=5 else "hi"}-{abs(20*i-100)}'] for i in range(1,10)],-1))
                q = np.stack(qs)
            if q.shape!=(context.shape[0],horizon,9) or not np.isfinite(q).all():
                raise ValueError(f"Invalid {model_name} output at {path.name}; no silent model fallback")
            temporary = output_path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream,quantiles=q,keys=data["keys"],target_hashes=data["target_hashes"],input_sha=hashlib.sha256(path.read_bytes()).hexdigest())
            temporary.replace(output_path)
        if index%10==0 or index==len(paths)-1:
            print(f"{model_name}: {index+1}/{len(paths)} groups, H={horizon}, V={context.shape[0]}",flush=True)
    (dest/"runtime.json").write_text(json.dumps(dict(model=model_name,device=str(device),checkpoint=checkpoint,
        groups=len(paths),seconds=time.perf_counter()-tick,symmetric_averaging=model_name=="timesfm3",
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        timing_scope="processing loop, excludes model load; resumed outputs skipped",
        scaler="official CPU float64 statistics" if model_name=="toto2_313m" and str(device)=="mps" else "native"),indent=2))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters",required=True)
    run(**json.loads(parser.parse_args().parameters))
