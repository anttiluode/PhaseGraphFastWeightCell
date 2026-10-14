"""Gate 3: train only the reader through its own error-gradient delta writer.

Sample 1/4/16/64-rewrite episodes and train at segmented readouts.
The inner update uses quantized state and a true state Jacobian. There is
no separate learned MLP writer. The 'phase_first_order' control uses
identical forward arithmetic but stops outer gradients through J^T e.

CPU smoke: python -m phasegraph.meta_train --steps 8 --seeds 41 --batch-size 3
CUDA experiment: python -m phasegraph.meta_train --steps 600 --seeds 41 97 127 --device cuda

This is higher-order differentiation through a dissipative graph, not
efficient TTT inference or an algebraic proof of orthogonal readout space.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .benchmark import make_stream
from .meta_ttt import MetaDeltaReader, read_all, cross_talk_kernel


ARMS = {
    "phase_meta": ("phase",True),
    "phase_first_order": ("phase",False),
    "signed_meta": ("signed",True),
    "vector_meta": ("vector",True),
}


def make_model(args,arm,seed):
    kind,second_order = ARMS[arm]
    torch.manual_seed(seed)
    return MetaDeltaReader(slots=args.slots,value_size=args.value_size,
        grid_size=args.grid_size,channels=args.channels,width=args.width,
        transport_steps=args.transport_steps,kind=kind,
        step_bins=args.step_bins,topk=args.topk,
        tolerance=args.tolerance,second_order=second_order).to(args.device)


def train(args,seed,arm):
    model=make_model(args,arm,seed).train()
    optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    schedule=random.Random(seed+1717)
    traces=[]
    updates=0
    began=time.perf_counter()
    for episode in range(args.steps):
        horizon=schedule.choice(args.train_horizons)
        stream=make_stream(args.batch_size,horizon,
            seed*100000+episode+6000000,slots=args.slots,
            value_size=args.value_size,device=args.device)
        state=None
        inputs=stream["inputs"]
        for begin in range(0,inputs.shape[1],args.segment):
            if state is not None:
                state=state.detach()
            last=min(begin+args.segment,inputs.shape[1])
            for i in range(begin,last):
                state=model.write(inputs[:,i],state)
            if last<args.slots:continue
            predicted=read_all(model,state)
            loss=F.mse_loss(predicted,stream["targets"][:,last-1])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{arm} seed {seed} episode {episode+1} non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            updates+=1
        if episode==0 or (episode+1)%max(1,args.steps//4)==0 or episode+1==args.steps:
            entry={"episode":episode+1,"horizon":horizon,"loss":float(loss.detach()),
                   "optimizer_updates":updates}
            traces.append(entry)
            print("TRAIN_JSON="+json.dumps({"seed":seed,"arm":arm,**entry}),flush=True)
    model.eval()
    records,seconds=evaluate(model,args,seed)
    return model,{"seed":seed,"arm":arm,"train_seconds":time.perf_counter()-began,
        "eval_seconds":seconds,"training_trace":traces,"optimizer_updates":updates,
        "state_bytes":model.payload_bytes,"parameter_count":model.param_count(),
        "inner_jacobian_second_order":model.second_order,
        "horizons":records,
        "caveats":["The inner update uses known one-hot keys and the target new value.",
                   "Every inner write requires autograd and is not hardware-friendly.",
                   "Truncated BPTT, single inner step per observation, no learned writer.",
                   "Forward-STE quantization does not guarantee stable true byte-grid optimization."]}


def evaluate(model,args,seed):
    data=make_stream(args.eval_count,max(args.eval_horizons),1000000000+seed,
                     slots=args.slots,value_size=args.value_size,device=args.device)
    state=None
    old=None
    started=time.perf_counter()
    hist={}
    changed_sum=0
    sham_sum=0
    for t,obs in enumerate(data["inputs"].unbind(1)):
        prev=state
        # Even in eval, the writer temporarily enables state autograd.
        state=model.write(obs,state)
        if state.dtype!=torch.uint8:
            raise AssertionError("test-time state must be physical uint8")
        with torch.no_grad():
            predictions=read_all(model,state)
            if prev is not None:
                changed_sum+=float((state!=prev).float().mean())
            if t>=args.slots:
                latest=obs[:,:args.slots].argmax(-1)
                untouched=torch.arange(args.slots,device=state.device)[None]!=latest[:,None]
                drift=(predictions-old).square()[untouched].mean().item()
                rewrite=t-args.slots+1
                if rewrite in args.eval_horizons:
                    repeat=model.write(obs,state)
                    with torch.no_grad():
                        truth=data["targets"][:,t]
                        newest=(predictions[torch.arange(len(obs)),latest]-
                                truth[torch.arange(len(obs)),latest]).square().mean()
                        hist[str(rewrite)]={
                            "mse":float(F.mse_loss(predictions,truth)),
                            "updated_slot_mse":float(newest),
                            "untouched_drift_mse":float(drift),
                            "changed_byte_fraction_cumulative":changed_sum/t,
                            "sham_byte_fraction":float((repeat!=state).float().mean()),
                            "state_bytes":model.payload_bytes}
            old=predictions
    # Small fixed prefix, true J_query @ J_other^T diagnostic.
    with torch.no_grad():
        base=state[:min(args.kernel_cases,len(state))].clone()
    kernel=cross_talk_kernel(model,base)
    for row in hist.values():
        row.update(kernel)
    return hist,time.perf_counter()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps",type=int,default=600)
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--arms",nargs="+",choices=list(ARMS),default=list(ARMS))
    p.add_argument("--train-horizons",nargs="+",type=int,default=[1,4,16,64])
    p.add_argument("--eval-horizons",nargs="+",type=int,default=[1,4,16,64])
    p.add_argument("--segment",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--eval-count",type=int,default=64)
    p.add_argument("--kernel-cases",type=int,default=4)
    p.add_argument("--slots",type=int,default=4)
    p.add_argument("--value-size",type=int,default=3)
    p.add_argument("--grid-size",type=int,default=4)
    p.add_argument("--channels",type=int,default=2)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--transport-steps",type=int,default=4)
    p.add_argument("--step-bins",type=float,default=8.)
    p.add_argument("--topk",type=int,default=None)
    p.add_argument("--tolerance",type=float,default=.002)
    p.add_argument("--learning-rate",type=float,default=.002)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,default=Path("runs/meta_ttt"))
    args=p.parse_args()
    if (args.steps<1 or args.batch_size<1 or args.segment<1 or args.eval_count<2
            or args.kernel_cases<1 or min(args.train_horizons)<1
            or min(args.eval_horizons)<1 or not math.isfinite(args.learning_rate)
            or args.learning_rate<=0):
        p.error("invalid training arguments")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    summary={"protocol":"Gate 3: train reader THROUGH its own gradient write",
             "args":{**vars(args),"output":str(args.output)},
             "runs":[],"claims_allowed":["MSE on held-out keyed 4x3 random-value streams",
                 "Physical uint8 persistent state count and parameter budget"],
             "not_proven":["Query Jacobian orthogonality is not forced mathematically.",
                 "No superiority to exact keyed 48-byte delta control.",
                 "No speed or compute equivalence to modern recurrent layers.",
                 "No learned geometrical object identities or natural images."]}
    for seed in args.seeds:
        for arm in args.arms:
            model,receipt=train(args,seed,arm)
            file=args.output/f"seed{seed}-{arm}"
            file.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
            torch.save({"config":model.config(),"model":model.state_dict(),"seed":seed,
                       "protocol":"meta_ttt_v1"},file.with_suffix(".pt"))
            summary["runs"].append(receipt)
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
            print("EVAL_JSON="+json.dumps({"arm":arm,"seed":seed,"horizons":receipt["horizons"],
                "parameters":receipt["parameter_count"]}),flush=True)
    print("SAVED="+str(args.output/"summary.json"),flush=True)

if __name__=="__main__":
    main()
