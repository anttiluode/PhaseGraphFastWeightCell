"""Error-driven phase rewrites: generic adjoint update and exact holonomy control.

AdjointPhaseDeltaWriter uses an existing *frozen* graph reader to compute
J_query(state)^T (target - prediction), then proposes sparse byte rotations.
It has no learned writer and does not silently optimize shared model weights.
It is computationally expensive: autograd plus repeated full graph reads
at every update. It is an exploratory online optimizer, NOT a Mamba kernel.

IndexedHolonomyDelta is a deliberately privileged, exact-key *positive control*:
one fixed nonadjacent plaquette per scalar, updated by its measured loop-phase
error. It makes clear that high score with known key addresses is not proof
of learned graph interference or compressed associative memory.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from .cell import PhaseGraphFastWeightCell


class IndexedHolonomyDelta(nn.Module):
    """UInt8 gauge-loop memory with a sparse local delta write.

    Input: a one-hot key [batch, slots] and values in [-1,1] [batch,values].
    Repeated inputs that already decode to within half a quantization bin
    cause no state changes. Uses fixed, known, non-overlapping face indices.
    The graph is an 8x8 lattice by default; loop flux is measured directly,
    not through a learned trainable propagator.

    Precision is at best ~1/128 in the target value, and 896 physical state
    bytes may be used to store only twelve independently indexed byte values.
    """
    def __init__(self, slots: int = 4, value_size: int = 3,
                 grid_size: int = 8, channels: int = 8):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (slots,value_size,channels)) or type(grid_size) is not int or grid_size < 2:
            raise ValueError("invalid geometry or key/value counts")
        self.slots,self.value_size=slots,value_size
        self.grid_size,self.channels=grid_size,channels
        self.edges=2*grid_size*(grid_size-1)
        self.payload_bytes=channels*self.edges
        faces=[(channel,r,c) for channel in range(channels)
               for r in range(0,grid_size-1,2) for c in range(0,grid_size-1,2)]
        if len(faces)<slots*value_size:
            raise ValueError("not enough nonadjacent plaquettes for the requested values")
        tops,rights,bottoms,lefts=[],[],[],[]
        horizontal=grid_size*(grid_size-1)
        for ch,r,c in faces[:slots*value_size]:
            offset=ch*self.edges
            tops.append(offset+r*(grid_size-1)+c)
            rights.append(offset+horizontal+r*grid_size+(c+1))
            bottoms.append(offset+(r+1)*(grid_size-1)+c)
            lefts.append(offset+horizontal+r*grid_size+c)
        self.register_buffer("top",torch.tensor(tops,dtype=torch.long))
        self.register_buffer("right",torch.tensor(rights,dtype=torch.long))
        self.register_buffer("bottom",torch.tensor(bottoms,dtype=torch.long))
        self.register_buffer("left",torch.tensor(lefts,dtype=torch.long))

    def state_bytes(self,batch_size:int=1)->int:
        if type(batch_size) is not int or batch_size<1:
            raise ValueError("batch_size must be positive integer")
        return batch_size*self.payload_bytes

    def initial_state(self,batch_size:int,device=None)->Tensor:
        self.state_bytes(batch_size)
        return torch.zeros((batch_size,self.channels,self.edges),dtype=torch.uint8,device=device)

    def _valid(self,state:Tensor)->Tensor:
        if not isinstance(state,Tensor) or state.dtype!=torch.uint8 or state.ndim!=3 or tuple(state.shape[1:])!=(self.channels,self.edges):
            raise ValueError("expected uint8 [batch,channels,edges] state")
        return state.flatten(1).long()

    def _key(self,keys:Tensor)->Tensor:
        if not isinstance(keys,Tensor) or keys.ndim!=2 or keys.shape[1]!=self.slots or keys.shape[0]<1 or not keys.is_floating_point():
            raise ValueError("keys must be one-hot float [batch,slots]")
        if not bool(((keys==0)|(keys==1)).all()) or not bool((keys.sum(-1)==1).all()):
            raise ValueError("orthogonal one-hot keys required (privileged address control)")
        return keys.argmax(-1)

    def flux(self,state:Tensor)->Tensor:
        flat=self._valid(state)
        return (flat[:,self.top]+flat[:,self.right]-flat[:,self.bottom]-flat[:,self.left]).remainder(256).reshape(len(flat),self.slots,self.value_size)

    def read(self,state:Tensor,keys:Tensor)->Tensor:
        ids=self._key(keys)
        if len(ids)!=len(state):
            raise ValueError("batch mismatch")
        f=self.flux(state)
        # Values occupy the non-ambiguous [0, pi] portion of the circle.
        return (4.0/256)*f[torch.arange(len(f),device=f.device),ids].float()-1

    @torch.no_grad()
    def write(self,keys:Tensor,values:Tensor,state:Tensor|None=None,
              *,write_mask:Tensor|None=None,reset_mask:Tensor|None=None)->Tensor:
        ids=self._key(keys)
        if not isinstance(values,Tensor) or values.shape!=(len(keys),self.value_size) or not values.is_floating_point() or not bool(torch.isfinite(values).all()) or not bool(((values>=-1)&(values<=1)).all()):
            raise ValueError("values must be finite float [batch,value_size] in [-1,1]")
        if state is None:
            state=self.initial_state(len(keys),keys.device)
        flat=self._valid(state)
        if len(flat)!=len(keys) or state.device!=keys.device or values.device!=keys.device:
            raise ValueError("state/input batch or device mismatch")
        for mask in (write_mask,reset_mask):
            if mask is not None and (mask.shape!=(len(keys),) or mask.dtype!=torch.bool or mask.device!=keys.device):
                raise ValueError("masks must be bool [batch]")
        state=state.clone()
        if reset_mask is not None:
            state[reset_mask]=0
        old=self.read(state,keys)
        # Genuine discrepancy feedback; one quantized phase-edge correction per value.
        # Exact repeat after quantization is a no-op within half a value bin.
        target_code=torch.round((values+1)*64).long().clamp(0,128)
        current=self.flux(state)[torch.arange(len(state),device=state.device),ids]
        increments=target_code-current
        # Preserve exact bytes if target is closer to existing decoded code.
        increments=torch.where((values-old).abs()<=2/256+1e-7,0,increments)
        if write_mask is not None:
            increments=torch.where(write_mask[:,None],increments,0)
        flat=state.flatten(1)
        pos=self.top.reshape(self.slots,self.value_size)[ids]
        indices=torch.arange(len(state),device=state.device)[:,None].expand_as(pos)
        flat[indices,pos]=(flat[indices,pos].long()+increments).remainder(256).to(torch.uint8)
        return state

    def forward(self,x:Tensor,state:Tensor|None=None,query:Tensor|None=None):
        if x.ndim!=2 or x.shape[1]!=self.slots+self.value_size:
            raise ValueError("input shape")
        key=x[:,:self.slots]
        state=self.write(key,x[:,self.slots:],state)
        return self.read(state,key if query is None else query),state

    @torch.no_grad()
    def gauge_rephase(self,state:Tensor,node_phase:Tensor)->Tensor:
        """Apply integer U(1) vertex gauges, preserving all face fluxes exactly."""
        self._valid(state)
        g=self.grid_size
        if node_phase.shape!=(len(state),self.channels,g,g) or node_phase.dtype not in (torch.int16,torch.int32,torch.int64,torch.uint8) or node_phase.device!=state.device:
            raise ValueError("node gauge must be integer [B,C,G,G]")
        delta_h=(node_phase[:,:,:,:-1].long()-node_phase[:,:,:,1:].long()).flatten(2)
        delta_v=(node_phase[:,:,:-1,:].long()-node_phase[:,:,1:,:].long()).flatten(2)
        return (state.long()+torch.cat((delta_h,delta_v),dim=-1)).remainder(256).to(torch.uint8)


class AdjointPhaseDeltaWriter:
    """Sparse J^T-error updates on a *frozen*, optionally pretrained phase reader.

    Consumes the reader's current queried prediction and target. At each
    iteration the gradient supplies a query-sensitive edge direction. Tests
    quantized line-search proposals; commits only if query error improves and
    previously read keys do not drift too much.

    There is NO guaranteed exact no-op for a repeated observation when the
    reader remains inaccurate: the proper delta rule also keeps correcting
    the residual error in that case. Exact no-op is guaranteed only once
    that observation is already within tolerance.
    """
    def __init__(self,reader:PhaseGraphFastWeightCell,*,slots:int,
                 topk:int=8,iterations:int=3,tolerance:float=.015625,
                 protection:float=1.0,steps:tuple[int,...]=(1,2,4,8,16)):
        if reader.kind!="phase" or reader.query_size!=slots or reader.input_size!=slots+reader.output_size:
            raise ValueError("reader must be a keyed phase graph cell")
        if not 1<=topk<=reader.payload_bytes or iterations<1 or not math.isfinite(tolerance) or tolerance<0 or not math.isfinite(protection) or protection<0 or not steps or any(type(k) is not int or k<1 for k in steps):
            raise ValueError("invalid delta update hyperparameters")
        self.reader=reader.eval()
        self.slots,self.topk,self.iterations=slots,topk,iterations
        self.tolerance,self.protection=tolerance,protection
        self.steps=tuple(steps)
        self.payload_bytes=reader.payload_bytes

    def initial_state(self,batch:int)->Tensor:
        return self.reader.initial_state(batch,packed=True)

    def read(self,state:Tensor,query:Tensor)->Tensor:
        return self.reader.read(state,query)

    def write(self,x:Tensor,state:Tensor|None=None,*,protect:bool=True)->Tensor:
        self.reader._input(x,self.reader.input_size,"input")
        q,target=x[:,:self.slots],x[:,self.slots:]
        previous=self.initial_state(len(x)) if state is None else state
        if previous.dtype!=torch.uint8 or previous.shape!=(len(x),self.reader.channels,self.reader.edges):
            raise ValueError("requires physical packed uint8 state")
        # Convert inference tensor to ordinary autograd tensor even inside
        # a surrounding no_grad/inference_mode context.
        with torch.inference_mode(False),torch.enable_grad():
            current=previous.detach().clone()
            queries=torch.eye(self.slots,device=x.device,dtype=x.dtype)
            old_all=None
            if protect and self.protection:
                with torch.no_grad():
                    old_all=torch.stack([self.reader.read(current,queries[k].expand(len(x),-1))
                                         for k in range(self.slots)],1).detach()
            current_keys=q.argmax(-1)
            for _ in range(self.iterations):
                v=(current.float()/256).detach().requires_grad_(True)
                predictions=self.reader.read(v,q)
                sample_error=(predictions-target).square().mean(-1)
                if bool((sample_error<=self.tolerance**2).all()):
                    break
                grad=torch.autograd.grad(sample_error.sum(),v)[0].detach().flatten(1)
                active=sample_error.detach()>self.tolerance**2
                if not bool(active.any()):
                    break
                mag=grad.abs()
                index=mag.topk(self.topk,dim=-1).indices
                sparse=torch.zeros_like(grad).scatter_(1,index,1.)
                direction=-(grad/(mag.amax(-1,keepdim=True).clamp_min(1e-12)))*sparse
                trial_scores=sample_error.detach()
                if old_all is not None:
                    drift=0
                    with torch.no_grad():
                        for slot in range(self.slots):
                            out=self.reader.read(current,queries[slot].expand(len(x),-1))
                            keep=(current_keys!=slot).float()
                            drift=drift+((out-old_all[:,slot])**2).mean(-1)*keep
                    trial_scores=trial_scores+self.protection*drift/self.slots
                best=current
                best_cost=trial_scores.clone()
                for bins in self.steps:
                    increment=(direction*bins).round().long().reshape_as(current)
                    candidate=(current.long()+increment).remainder(256).to(torch.uint8)
                    with torch.no_grad():
                        cost=(self.reader.read(candidate,q)-target).square().mean(-1)
                        if old_all is not None:
                            for slot in range(self.slots):
                                other=self.reader.read(candidate,queries[slot].expand(len(x),-1))
                                keep=(current_keys!=slot).float()
                                cost=cost+self.protection*((other-old_all[:,slot])**2).mean(-1)*keep/self.slots
                    accept=active & (cost<best_cost-1e-10)
                    best=torch.where(accept[:,None,None],candidate,best)
                    best_cost=torch.where(accept,cost,best_cost)
                if torch.equal(best,current):
                    break
                current=best
            return current
