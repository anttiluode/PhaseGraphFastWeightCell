"""Gate 2: test actual byte-local error-driven writes and gauge-loop control."""
import pytest
import torch

from phasegraph.delta import IndexedHolonomyDelta, AdjointPhaseDeltaWriter
from phasegraph.cell import PhaseGraphFastWeightCell
from phasegraph.benchmark import make_stream


def key(n,idx):
    return torch.nn.functional.one_hot(torch.tensor(idx),n).float()


def test_physical_896_bytes_and_loop_capacity():
    x=IndexedHolonomyDelta(4,3,grid_size=8,channels=8)
    assert x.payload_bytes==896
    s=x.initial_state(2)
    assert s.dtype==torch.uint8 and s.shape==(2,8,112)
    assert s.numel()==1792
    assert len(x.top)==12
    assert x.state_bytes()==896
    with pytest.raises(ValueError):
        IndexedHolonomyDelta(4,3,grid_size=4,channels=2)


def test_exact_64_rewrites_and_unchanged_input_noop():
    memory=IndexedHolonomyDelta(4,3,grid_size=8,channels=8)
    stream=make_stream(3,64,seed=52)
    state=None
    fraction=[]
    for x in stream["inputs"].unbind(1):
        state=memory.write(x[:,:4],x[:,4:],state)
        repeated=memory.write(x[:,:4],x[:,4:],state)
        fraction.append(float((repeated!=state).float().mean()))
        assert torch.equal(repeated,state)
    assert max(fraction)==0
    queries=torch.eye(4)[None].expand(3,-1,-1).reshape(12,4)
    all_states=state[:,None].expand(-1,4,-1,-1).reshape(12,8,112)
    prediction=memory.read(all_states,queries).reshape(3,4,3)
    assert (prediction-stream["targets"][:,-1]).square().mean()<0.00004


def test_query_locality_and_wrap():
    cell=IndexedHolonomyDelta(4,3,grid_size=8,channels=1)
    state=cell.initial_state(1)
    a=key(4,[0]); b=key(4,[1])
    state=cell.write(a,torch.tensor([[.33,-.73,.47]]),state)
    answer_a=cell.read(state,a)
    before_b=cell.read(state,b)
    state_next=cell.write(b,torch.tensor([[-.37,.91,1.0]]),state)
    assert torch.equal(answer_a,cell.read(state_next,a))
    assert not torch.equal(before_b,cell.read(state_next,b))
    assert int((state_next!=state).sum())<=3
    assert torch.equal(state,cell.initial_state(1)) is False


def test_integer_gauge_exact_flux_after_rewrite():
    cell=IndexedHolonomyDelta(4,3,grid_size=8,channels=1)
    s=cell.initial_state(5)
    torch.manual_seed(19)
    gauge=torch.randint(0,256,(5,1,8,8),dtype=torch.int64)
    s=cell.gauge_rephase(s,gauge)
    assert torch.all(cell.flux(s)==0)
    key0=key(4,[0]*5)
    values=torch.rand(5,3)*2-1
    after=cell.write(key0,values,s)
    assert torch.equal(cell.flux(after)[:,1:],torch.zeros_like(cell.flux(after)[:,1:]))
    torch.testing.assert_close(cell.read(after,key0),values,atol=.008,rtol=0)
    gauge2=torch.randint(0,256,(5,1,8,8),dtype=torch.int64)
    twice=cell.gauge_rephase(after,gauge2)
    assert torch.equal(cell.flux(after),cell.flux(twice))


def test_reset_and_mask_without_hidden_state():
    c=IndexedHolonomyDelta(4,3,grid_size=8,channels=1)
    a=key(4,[0,0])
    s=c.write(a,torch.ones(2,3))
    t=c.write(a,-torch.ones(2,3),s,write_mask=torch.tensor([False,True]))
    assert torch.equal(s[0],t[0])
    assert not torch.equal(s[1],t[1])
    u=c.write(a,torch.ones(2,3),t,write_mask=torch.tensor([False,False]),
              reset_mask=torch.tensor([True,False]))
    assert torch.equal(u[0],c.initial_state(1)[0])
    assert torch.equal(u[1],t[1])


def test_reject_nonorthogonal_keys_and_out_of_bounds_values():
    c=IndexedHolonomyDelta(4,3)
    with pytest.raises(ValueError,match="one-hot"):
        c.write(torch.full((2,4),.25),torch.zeros(2,3))
    with pytest.raises(ValueError,match="values"):
        c.write(key(4,[0]),torch.full((1,3),2.0))
    with pytest.raises(ValueError):
        c.write(key(4,[0]),torch.ones(1,2))
    with pytest.raises(ValueError):
        c.read(torch.zeros((1,8,112),dtype=torch.float32),key(4,[0]))


def test_generic_adjoint_rejects_invalid_setup():
    reader=PhaseGraphFastWeightCell(7,3,query_size=4,grid_size=3,channels=2,width=16).eval()
    with pytest.raises(ValueError):
        AdjointPhaseDeltaWriter(reader,slots=3)
    with pytest.raises(ValueError):
        AdjointPhaseDeltaWriter(reader,slots=4,topk=0)
    assert AdjointPhaseDeltaWriter(reader,slots=4).payload_bytes==24


def test_adjoint_exact_match_is_strict_noop_and_preserves_uint8():
    torch.manual_seed(17)
    reader=PhaseGraphFastWeightCell(7,3,query_size=4,grid_size=3,channels=2,width=16).eval()
    opt=AdjointPhaseDeltaWriter(reader,slots=4,topk=4,iterations=2,tolerance=1e-4)
    state=reader.initial_state(2)
    query=key(4,[0,1])
    with torch.no_grad():
        target=reader.read(state,query).detach()
    new=opt.write(torch.cat((query,target),1),state)
    assert torch.equal(new,state) and new.dtype==torch.uint8


def test_adjoint_updates_only_topk_times_iterations_and_does_not_change_weights():
    torch.manual_seed(41)
    reader=PhaseGraphFastWeightCell(7,3,query_size=4,grid_size=3,channels=2,width=16).eval()
    delta=AdjointPhaseDeltaWriter(reader,slots=4,topk=2,iterations=2,tolerance=0,protection=0)
    s=reader.initial_state(2)
    query=key(4,[0,1])
    goal=torch.ones(2,3)
    weights={k:v.clone() for k,v in reader.state_dict().items()}
    with torch.no_grad():
        new=delta.write(torch.cat((query,goal),1),s)
    assert new.dtype==torch.uint8
    assert int((new!=s).sum())<=2*2*2
    assert all(torch.equal(v,weights[k]) for k,v in reader.state_dict().items())
    oldloss=(reader.read(s,query)-goal).square().mean()
    newloss=(reader.read(new,query)-goal).square().mean()
    assert newloss<=oldloss+1e-7


def test_adjoint_repeat_only_noops_when_prediction_is_good():
    torch.manual_seed(18)
    reader=PhaseGraphFastWeightCell(7,3,query_size=4,grid_size=3,channels=2,width=16).eval()
    delta=AdjointPhaseDeltaWriter(reader,slots=4,topk=4,iterations=1,tolerance=.01)
    s=delta.initial_state(1)
    q=key(4,[0])
    goal=reader.read(s,q).detach()
    assert torch.equal(delta.write(torch.cat((q,goal),1),s),s)
