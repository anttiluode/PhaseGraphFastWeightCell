"""Gate 2: independently audit local phase delta, known-address control, and frozen adjoint.

Run:
  python -m phasegraph.local_delta_benchmark --seeds 41 97 127 \
    --eval-count 128 --horizons 1 4 16 64 --output runs/local_delta

Known-key holonomy is a deliberately privileged comparator, not a learned
GraphNN or a replacement for a dense fast-weight cell. Adjoint experiments
reuse poor PR1 readers and therefore test local correction, NOT a new
trained state-of-the-art architecture.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .benchmark import make_stream
from .baselines import DeltaRuleMemory
from .cell import PhaseGraphFastWeightCell
from .delta import IndexedHolonomyDelta, AdjointPhaseDeltaWriter


def all_queries(slots, batch, device):
    return torch.eye(slots,device=device)[None].expand(batch,-1,-1).reshape(batch*slots,slots)


@torch.no_grad()
def read_all(memory,state,slots):
    B=len(state)
    q=all_queries(slots,B,state.device)
    repeated=state[:,None].expand(-1,slots,-1,-1).reshape(B*slots,*state.shape[1:])
    return memory.read(repeated,q).reshape(B,slots,-1)


@torch.no_grad()
def evaluate_keyed(model, data, horizons,kind):
    slots=data["targets"].shape[2]
    B=len(data["inputs"])
    state=None
    latest=None
    rows={}
    old=None
    changes=[]
    start=time.perf_counter()
    for t,obs in enumerate(data["inputs"].unbind(1)):
        state_before=state
        if kind=="delta":
            state=model.write(obs[:,:slots],obs[:,slots:],state)
            pred=state
        else:
            state=model.write(obs[:,:slots],obs[:,slots:],state)
            pred=read_all(model,state,slots)
        if state_before is not None:
            if kind!="delta":
                changes.append(float((state!=state_before).float().mean()))
            else:
                changes.append(float((state!=state_before).float().mean()))
        if t>=slots and t-slots+1 in horizons:
            rewrites=t-slots+1
            target=data["targets"][:,t]
            no_write_predict=latest
            ident=(obs[:,:slots],obs[:,slots:])
            if kind=="delta":
                repeat=model.write(*ident,state)
                sham_diff=float((repeat!=state).float().mean())
                budget=model.state_bytes()
            else:
                repeat=model.write(*ident,state)
                sham_diff=float((repeat!=state).float().mean())
                budget=model.state_bytes()
            keep=(torch.arange(slots)[None]!=data["slots"][:,t,None])
            before=old
            if before is None:
                before=pred
            drift=float((pred-before).square()[keep].mean())
            rows[str(rewrites)]={"mse":float(F.mse_loss(pred,target)),
                                "untouched_drift_mse":drift,
                                "sham_changed_byte_fraction":sham_diff,
                                "changed_byte_fraction_mean":sum(changes)/len(changes),
                                "state_bytes":budget,
                                "skip_rewrite_mse":float(F.mse_loss(no_write_predict,target))}
        old=pred
        latest=pred
    return rows,time.perf_counter()-start


def evaluate_adjoint(seed, cases, rewrites, iterations,topk):
    ck=Path(__file__).resolve().parents[1]/"results"/"sequential_v1"/f"seed{seed}"/"phase.pt"
    original=torch.load(ck,map_location="cpu",weights_only=True)
    reader=PhaseGraphFastWeightCell(**original["config"]).eval()
    reader.load_state_dict(original["model"])
    cell=AdjointPhaseDeltaWriter(reader,slots=4,topk=topk,iterations=iterations,
                                 protection=1.0,tolerance=.025)
    data=make_stream(cases,rewrites,1_200_000_000+seed)
    trained,adjoint=None,None
    report={"reader_is_frozen":True,"using_PR1_seed":seed,
            "iterations_per_write":iterations,"topk_edges":topk,"cases":cases,
            "rewrites":rewrites}
    t0=time.perf_counter()
    deviations=[]
    for step,obs in enumerate(data["inputs"].unbind(1)):
        trained=reader.write(obs,trained)
        adjoint=cell.write(obs,adjoint)
        deviations.append(float((adjoint!=reader.initial_state(cases)).float().mean()))
    target=data["targets"][:,-1]
    uncorrected=read_all(reader,trained,4)
    corrected=read_all(reader,adjoint,4)
    report.update({"global_writer_final_mse":float(F.mse_loss(uncorrected,target)),
                   "adjoint_final_mse":float(F.mse_loss(corrected,target)),
                   "changed_fraction_vs_init":float(deviations[-1]),
                   "bytes":reader.state_bytes(),
                   "seconds":time.perf_counter()-t0,
                   "caveat":"Frozen PR1 reader poorly trained on original benchmark. Success of one-step optimization cannot be extrapolated to 64 writes or learned spatial memory."})
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,nargs="+",default=[41,97,127])
    p.add_argument("--eval-count",type=int,default=128)
    p.add_argument("--horizons",type=int,nargs="+",default=[1,4,16,64])
    p.add_argument("--adjoint-cases",type=int,default=2)
    p.add_argument("--adjoint-rewrites",type=int,default=4)
    p.add_argument("--adjoint-iterations",type=int,default=2)
    p.add_argument("--adjoint-topk",type=int,default=8)
    p.add_argument("--skip-adjoint",action="store_true")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--output",type=Path,default=Path("runs/local_delta/summary.json"))
    a=p.parse_args()
    if min(a.horizons)<1 or a.eval_count<2 or a.adjoint_cases<1:
        p.error("invalid case counts/horizons")
    torch.set_num_threads(a.threads)
    result={"status":"measured","protocol":"known-key phase delta controls plus exploratory frozen-reader adjoint",
            "seeds":a.seeds,"eval_count_per_seed":a.eval_count,"horizons":a.horizons,
            "mean":{},"runs":[],
            "limitations":[
              "IndexedHolonomyDelta has PRIVILEGED fixed keys and uses direct loop-flux access.",
              "Only twelve of the 896 graph phase bytes are necessary for the 4x3 scalar benchmark. A 12-byte indexed scalar table would be even smaller.",
              "It does not train spatial/relational memory or show learned phase advantage.",
              "Benchmark values uniform [-1,1] quantized to ~8-bit, while DeltaRuleMemory retains float32 exact values.",
              "Generic adjoint updates use frozen PR1 decoders; test separately and do not claim improved learned reader."
            ]}
    for seed in a.seeds:
        data=make_stream(a.eval_count,max(a.horizons),1_000_000_000+seed)
        models={"indexed_loop_72B":IndexedHolonomyDelta(4,3,grid_size=4,channels=3),
                "indexed_loop_896B":IndexedHolonomyDelta(4,3,grid_size=8,channels=8),
                "orthogonal_delta_48B":DeltaRuleMemory(4,3)}
        seed_result={"seed":seed,"arms":{}}
        for name,m in models.items():
            rows,elapsed=evaluate_keyed(m,data,a.horizons,
                                         "delta" if isinstance(m,DeltaRuleMemory) else "loop")
            seed_result["arms"][name]={"horizons":rows,"seconds":elapsed}
        if not a.skip_adjoint:
            seed_result["adjoint"]=evaluate_adjoint(seed,a.adjoint_cases,
                                    a.adjoint_rewrites,a.adjoint_iterations,a.adjoint_topk)
        print("SEED_RESULT_JSON="+json.dumps(seed_result),flush=True)
        result["runs"].append(seed_result)
    for name in result["runs"][0]["arms"]:
        result["mean"][name]={}
        for n in a.horizons:
            result["mean"][name][str(n)]={
                key:sum(seed["arms"][name]["horizons"][str(n)][key] for seed in result["runs"])/len(result["runs"])
                for key in ("mse","untouched_drift_mse","sham_changed_byte_fraction","changed_byte_fraction_mean")}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print("MEAN_RESULT_JSON="+json.dumps(result["mean"]),flush=True)
    print("SAVED="+str(a.output),flush=True)


if __name__=="__main__":
    main()
