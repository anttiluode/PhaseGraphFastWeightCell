"""Smoke tests for horizon-matched training without modifying original PR1 checkpoints."""
from types import SimpleNamespace

import torch
from phasegraph.horizon_train import train_one


def test_horizon_trainer_backpropagates_and_evaluates_without_state_growth():
    args=SimpleNamespace(slots=4,value_size=3,grid_size=4,channels=2,width=16,
        transport_steps=2,device="cpu",lr=.003,steps=2,batch_size=3,
        eval_count=4,train_horizons=[1,4,16],eval_horizons=[1,4,16],
        segment=4,threads=1)
    torch.set_num_threads(1)
    for kind in ("phase","signed","vector"):
        cell,receipt=train_one(41,kind,args)
        assert receipt["training_updates"]>=2
        assert cell.state_bytes()==48
        assert receipt["horizons"]["16"]["mse"]>=0
        assert torch.isfinite(torch.tensor(receipt["horizons"]["16"]["mse"]))
        assert all(x["optimizer_updates"]>=1 for x in receipt["training_trace"])
