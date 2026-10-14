"""Gate 3: trained-through-adjoint differentiation, byte quantization, controls."""
import math

import pytest
import torch

from phasegraph.meta_ttt import MetaDeltaReader, read_all, cross_talk_kernel
from phasegraph.benchmark import make_stream
from phasegraph.meta_train import ARMS, make_model, train, evaluate
from types import SimpleNamespace


def tiny(**kw):
    return MetaDeltaReader(slots=4,value_size=3,grid_size=3,
          channels=2,width=16,transport_steps=2,**kw)


def key(ids):
    return torch.nn.functional.one_hot(torch.tensor(ids),4).float()


@pytest.mark.parametrize("kind",("phase","signed","vector"))
def test_actual_byte_state_and_independent_no_extra_writer(kind):
    m=tiny(kind=kind).eval()
    assert m.payload_bytes==24
    assert sum(p.numel() for p in m.reader.writer.parameters())==0
    state=m.initial_state(2)
    assert state.dtype==torch.uint8 and state.numel()==48
    with torch.no_grad():
        x=torch.cat((key([0,1]),torch.tensor([[.8,-.9,.5],[-.6,.2,.1]])),1)
        updated=m.write(x,state)
        assert updated.dtype==torch.uint8
        assert updated.shape==state.shape
        out=m.read(updated,key([0,1]))
        assert out.shape==(2,3) and bool(torch.isfinite(out).all())


def test_zero_error_rewrite_is_exact_byte_noop_and_packed_state_persists():
    torch.manual_seed(19)
    m=tiny(kind="phase",tolerance=0).eval()
    state=m.initial_state(2)
    q=key([0,1])
    with torch.no_grad():
        target=m.read(state,q)
        before=state.clone()
        revised=m.write(torch.cat((q,target),1),state)
        assert torch.equal(before,revised)
        assert torch.equal(revised,m.write(torch.cat((q,target),1),revised))


def test_topk_limits_modified_physical_edges_per_observation():
    torch.manual_seed(29)
    m=tiny(kind="phase",topk=3).eval()
    state=m.initial_state(2)
    with torch.no_grad():
        obs=torch.cat((key([0,1]),torch.tensor([[.8,-.9,.5],[-.6,.2,.1]])),1)
        revised=m.write(obs,state)
    assert torch.count_nonzero(revised!=state).item()<=6
    assert revised.dtype==torch.uint8


@pytest.mark.parametrize("kind",("phase","signed","vector"))
@pytest.mark.parametrize("second_order",(True,False))
def test_outer_gradient_through_quantized_delta_and_reader(kind,second_order):
    torch.manual_seed(41)
    m=tiny(kind=kind,second_order=second_order).train()
    batch=make_stream(2,2,seed=51)
    state=None
    for x in batch["inputs"].unbind(1):
        state=m.write(x,state)
    pred=read_all(m,state)
    loss=(pred-batch["targets"][:,-1]).square().mean()
    loss.backward()
    grads=[p.grad for p in m.parameters()]
    assert grads and any(x is not None and x.abs().sum().item()>1e-10 for x in grads)
    assert all(x is None or bool(torch.isfinite(x).all()) for x in grads)
    assert m.param_count()==sum(p.numel() for p in m.parameters())


def test_second_order_changes_outer_parameter_gradient():
    torch.manual_seed(13)
    a=tiny(second_order=True).train()
    b=tiny(second_order=False).train()
    b.load_state_dict(a.state_dict())
    data=make_stream(2,1,seed=26)
    def grad(m):
        state=None
        for obs in data["inputs"].unbind(1):
            state=m.write(obs,state)
        read_all(m,state).square().mean().backward()
        return m.reader.query_projection.weight.grad.clone()
    g1,g2=grad(a),grad(b)
    assert torch.isfinite(g1).all() and torch.isfinite(g2).all()
    assert (g1-g2).abs().max()>1e-8


def test_normalized_cross_kernel_is_measurable_not_assumed_zero():
    torch.manual_seed(11)
    m=tiny().eval()
    state=m.initial_state(2)
    metric=cross_talk_kernel(m,state)
    assert set(metric)=={"offdiag_cross_kernel","self_jacobian_frobenius"}
    assert all(math.isfinite(v) for v in metric.values())
    assert metric["offdiag_cross_kernel"]>=0


def test_smoke_meta_train_and_evaluate_four_arm_training_loop():
    torch.set_num_threads(1)
    args=SimpleNamespace(slots=4,value_size=3,grid_size=3,channels=2,width=16,
        transport_steps=2,step_bins=8.,topk=4,tolerance=.002,device="cpu",
        learning_rate=.002,steps=2,batch_size=2,train_horizons=[1,4],
        eval_horizons=[1,4],segment=4,eval_count=3,kernel_cases=1)
    for arm in ARMS:
        model,receipt=train(args,41,arm)
        assert receipt["state_bytes"]==24
        assert receipt["optimizer_updates"]>=2
        assert "4" in receipt["horizons"]
        row=receipt["horizons"]["4"]
        assert math.isfinite(row["mse"])
        assert "offdiag_cross_kernel" in row
        assert row["sham_byte_fraction"]>=0


def test_frozen_model_weights_unchanged_by_test_time_state_gradient():
    torch.manual_seed(7)
    m=tiny().eval()
    before={n:v.detach().clone() for n,v in m.named_parameters()}
    sample=torch.cat((key([0]),torch.ones(1,3)),1)
    with torch.inference_mode():
        state=m.write(sample)
        pred=m.read(state,key([0]))
    assert state.dtype==torch.uint8
    assert bool(torch.isfinite(pred).all())
    assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
