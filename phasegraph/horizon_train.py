"""Horizon-matched baseline training (NOT a claim of phase superiority).

Extends the original 1..16-rewrite PR1 curriculum to 1,4,16,64 with
intermediate supervision and truncated BPTT. This trains the ORIGINAL global
writer, deliberately separate from the new adjoint update. Outputs loss,
physical memory/parameter count, and evaluation at every tested horizon.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .benchmark import BenchmarkConfig, make_stream, read_all, _evaluate
from .cell import PhaseGraphFastWeightCell


def train_one(seed:int,kind:str,args):
    torch.manual_seed(seed)
    cell=PhaseGraphFastWeightCell(args.slots+args.value_size,args.value_size,
        query_size=args.slots,grid_size=args.grid_size,channels=args.channels,
        width=args.width,transport_steps=args.transport_steps,kind=kind).to(args.device)
    optimizer=torch.optim.Adam(cell.parameters(),lr=args.lr)
    generator=random.Random(seed+1717)
    trace=[]
    n_updates=0
    started=time.perf_counter()
    for step in range(args.steps):
        rewrites=generator.choice(args.train_horizons)
        batch=make_stream(args.batch_size,rewrites, seed*100_000+step+6_000_000,
                          slots=args.slots,value_size=args.value_size,device=args.device)
        state=None
        inputs=batch["inputs"]
        for begin in range(0,inputs.shape[1],args.segment):
            # Truncated BPTT: every segment sees the previously written state,
            # but gradients do not traverse the entire 64-step stream.
            if state is not None:
                state=state.detach()
            for t in range(begin,min(begin+args.segment,inputs.shape[1])):
                state=cell.write(inputs[:,t],state)
            last=min(begin+args.segment,inputs.shape[1])-1
            if last<args.slots-1:
                continue
            output=read_all(cell,state,args.slots)
            target=batch["targets"][:,last]
            loss=F.mse_loss(output,target)
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{seed} {kind} nonfinite at step {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cell.parameters(),1.0)
            optimizer.step()
            n_updates+=1
        if step==0 or (step+1)%max(1,args.steps//5)==0 or step+1==args.steps:
            trace.append({"step":step+1,"horizon":rewrites,"loss":float(loss.detach()),
                          "optimizer_updates":n_updates})
            print("TRAIN_PROGRESS_JSON="+json.dumps({"seed":seed,"kind":kind,**trace[-1]}),flush=True)
    cell.eval()
    cfg=BenchmarkConfig(steps=args.steps,seeds=(seed,),batch_size=args.batch_size,
        eval_count=args.eval_count,slots=args.slots,value_size=args.value_size,
        train_rewrites=max(args.train_horizons),horizons=tuple(args.eval_horizons),
        grid_size=args.grid_size,channels=args.channels,width=args.width,
        transport_steps=args.transport_steps,learning_rate=args.lr,device=args.device,
        threads=args.threads)
    rows,trajectory,elapsed,state=_evaluate(cfg,cell,seed)
    return cell,{"seed":seed,"kind":kind,"train_seconds":time.perf_counter()-started,
                 "eval_seconds":elapsed,"training_updates":n_updates,
                 "training_trace":trace,"horizons":rows,"state_bytes":cell.state_bytes(),
                 "parameter_bytes":cell.parameter_bytes(),
                 "parameter_count":sum(p.numel() for p in cell.parameters()),
                 "limits":"Horizon-matched ORIGINAL global writer, not local delta. Truncated BPTT and multiple optimizer steps per episode; original 600-step baseline not compute-matched."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--kinds",nargs="+",choices=["phase","signed","vector"],default=["phase","signed","vector"])
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--eval-count",type=int,default=128)
    p.add_argument("--train-horizons",type=int,nargs="+",default=[1,4,16,64])
    p.add_argument("--eval-horizons",type=int,nargs="+",default=[1,4,16,64])
    p.add_argument("--segment",type=int,default=8)
    p.add_argument("--slots",type=int,default=4)
    p.add_argument("--value-size",type=int,default=3)
    p.add_argument("--grid-size",type=int,default=4)
    p.add_argument("--channels",type=int,default=2)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--transport-steps",type=int,default=4)
    p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,default=Path("runs/horizon_matched"))
    a=p.parse_args()
    if (a.steps<1 or a.segment<1 or a.batch_size<1 or a.eval_count<2 or
        min(a.train_horizons)<1 or min(a.eval_horizons)<1 or
        not math.isfinite(a.lr) or a.lr<=0):
        p.error("invalid configuration")
    torch.set_num_threads(a.threads)
    a.output.mkdir(parents=True,exist_ok=True)
    summary={"protocol":"frozen-error-analysis follow-up; original global writer, full-horizon training",
             "args":vars(a).copy(),"runs":[]}
    summary["args"]["output"]=str(a.output)
    for seed in a.seeds:
        for kind in a.kinds:
            trained,receipt=train_one(seed,kind,a)
            (a.output/f"seed{seed}-{kind}.json").write_text(json.dumps(receipt,indent=2)+"\n")
            torch.save({"config":trained.config,"model":trained.state_dict(),
                        "seed":seed,"protocol":"horizon_matched_v2"},
                       a.output/f"seed{seed}-{kind}.pt")
            summary["runs"].append(receipt)
            (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print("EVAL_RESULT_JSON="+json.dumps({"seed":seed,"kind":kind,
                  "horizons":receipt["horizons"],"state_bytes":receipt["state_bytes"]}),flush=True)
    print("SAVED="+str(a.output/"summary.json"),flush=True)


if __name__=="__main__":
    main()
