"""Isolated Toto 2.0 runtime. Inputs/outputs are outside the repository."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True

import numpy as np
import torch
from toto2 import Toto2Model


class CPUStatistics(torch.nn.Module):
    """Preserve the official float64 statistics when MPS cannot represent float64."""
    def __init__(self,original):
        super().__init__()
        self.original=original

    def forward(self,data,mask=None):
        values=self.original(data.cpu(),None if mask is None else mask.cpu())
        return tuple(value.to(data.device) for value in values)


def run(checkpoint,input_dir,output_dir,horizon=32,decode_block_size=768):
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    else:
        device=torch.device("cpu")
    model=Toto2Model.from_pretrained(checkpoint,map_location="cpu").to(device).eval()
    if device.type=="mps":
        model.scaler=CPUStatistics(model.scaler)
    dest=Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    patch=model.config.patch_size
    paths=sorted(Path(input_dir).glob("*.npz"))
    print(f"Toto2 loaded on {device}; {len(paths)} native groups; patch={patch}",flush=True)
    with torch.inference_mode():
        for index,path in enumerate(paths):
            data=np.load(path)
            context=torch.as_tensor(data["context"],dtype=torch.float32,device=device)[None]
            padding=(-context.shape[-1])%patch
            target=torch.nn.functional.pad(context,(padding,0))
            mask=torch.ones_like(target,dtype=torch.bool)
            if padding:
                mask[...,:padding]=False
            ids=torch.zeros(target.shape[:2],dtype=torch.long,device=device)
            quantiles=model.forecast({"target":target,"target_mask":mask,"series_ids":ids},
                        horizon=horizon,decode_block_size=decode_block_size,has_missing_values=bool(padding))
            q=quantiles[:,0].permute(1,2,0).float().cpu().numpy()
            if q.shape!=(context.shape[1],horizon,9) or not np.isfinite(q).all():
                raise ValueError(f"Invalid Toto2 forecast: shape={q.shape}, finite={np.isfinite(q).sum()}/{q.size}")
            np.savez_compressed(dest/path.name,quantiles=q,keys=data["keys"],target_hashes=data["target_hashes"])
            if index%20==0 or index==len(paths)-1:
                print(f"Toto2 groups {index+1}/{len(paths)}",flush=True)
    (dest/"runtime.json").write_text(json.dumps({"device":str(device),"checkpoint":checkpoint,
          "horizon":horizon,"decode_block_size":decode_block_size,"groups":len(paths),
          "scaler":"official float64 statistics on CPU" if device.type=="mps" else "official device scaler",
          "torch":torch.__version__,"parameters":sum(p.numel() for p in model.parameters())},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--parameters",required=True)
    run(**json.loads(parser.parse_args().parameters))
