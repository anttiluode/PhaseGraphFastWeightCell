"""Learn a phase graph reader *through its own quantized gradient write rule*.

This is an experimental higher-order meta-learning algorithm, not the
previous frozen-reader backtracking optimizer and not a fast inference
kernel. The single persistent stream state is physical uint8 graph edges.
The old independent MLP writer is REMOVED entirely: nn.Identity() carries
no weights. Forward inner-loop writing is a gradient descent step on the
reader's prediction error, quantized after every observation.

'second_order' (second_order=True) differentiates THROUGH J^T e;
first_order=False is a stop-gradient-control where the same numerical
inner updates are made without differentiating their derivative.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn

from .cell import PhaseGraphFastWeightCell
from .quantization import quantize_phase, pack_phase


def interval_ste(x: Tensor) -> Tensor:
    bounded = x.clamp(0, 255 / 256)
    return (bounded * 256).round().detach() / 256 + (bounded - bounded.detach())


class MetaDeltaReader(nn.Module):
    """No external MLP writer; the reader's error Jacobian IS the writer."""

    def __init__(self, *, slots: int = 4, value_size: int = 3,
                 grid_size: int = 4, channels: int = 2, width: int = 64,
                 transport_steps: int = 4, kind: str = "phase",
                 step_bins: float = 8., topk: int | None = None,
                 tolerance: float = .002, second_order: bool = True):
        super().__init__()
        if type(slots) is not int or slots < 2 or type(value_size) is not int or value_size < 1:
            raise ValueError("invalid key/value dimensions")
        if not math.isfinite(step_bins) or not (0 < step_bins <= 128):
            raise ValueError("step_bins must be in (0,128]")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("invalid tolerance")
        if kind not in ("phase", "signed", "vector"):
            raise ValueError("kind must be phase, signed or vector")
        if topk is not None and (type(topk) is not int or topk < 1):
            raise ValueError("topk must be null or positive integer")
        self.slots,self.value_size,self.kind = slots,value_size,kind
        self.step_bins,self.topk,self.tolerance = float(step_bins),topk,float(tolerance)
        self.second_order = bool(second_order)
        self.reader = PhaseGraphFastWeightCell(
            slots + value_size, value_size, query_size=slots, grid_size=grid_size,
            channels=channels, width=width, transport_steps=transport_steps,
            kind=kind)
        # Removes the original redundant 2-layer global writer's weights.
        self.reader.writer = nn.Identity()
        self.payload_bytes = self.reader.payload_bytes
        if topk is not None and topk > self.payload_bytes:
            raise ValueError("topk exceeds physical byte count")

    def initial_state(self, batch: int, *, packed: bool | None = None) -> Tensor:
        return self.reader.initial_state(batch, packed=packed)

    def _gradient_step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        q,target = x[:, :self.slots],x[:, self.slots:]
        # When the first write starts from a constant, supply an independent
        # differentiable state to compute the input-gradient for the writer.
        work = state if state.requires_grad else state.detach().requires_grad_(True)
        pred = self.reader.read(work,q)
        residual = pred - target
        per_case = residual.square().mean(-1)
        gradient = torch.autograd.grad(
            per_case.sum(), work, create_graph=self.training and self.second_order,
            retain_graph=self.training)[0]
        flat = gradient.flatten(1)
        # Scale by detached RMS so the update has comparable size across
        # learned reader parameterizations; this is NOT natural gradient.
        rms = flat.square().mean(-1).sqrt().detach().clamp_min(1e-9)
        direction = flat / rms[:, None]
        if self.topk is not None:
            # Fixed selected indices in this iteration; gradient flows
            # through selected entries only (piecewise straight-through).
            indices = flat.detach().abs().topk(self.topk,dim=-1).indices
            selection = torch.zeros_like(flat).scatter_(1,indices,1.0)
            direction = direction * selection
        proposal = work - (self.step_bins / 256.) * direction.reshape_as(work)
        new = quantize_phase(proposal) if self.kind == "phase" else interval_ste(proposal)
        # If the target is already represented to tolerance, do not change
        # even a single byte. A repeated *inaccurate* target may still update.
        active = per_case.detach() > self.tolerance ** 2
        output = torch.where(active[:,None,None],new,work)
        return output,pred

    def write(self,x:Tensor,state:Tensor|None=None)->Tensor:
        # Validate key/value dtype, finiteness and dimensions exactly like PR1.
        self.reader._input(x,self.slots+self.value_size,"input")
        if state is None:
            state = self.initial_state(len(x),packed=False)
        if state.ndim!=3 or state.shape!=(len(x),self.reader.channels,self.reader.edges):
            raise ValueError("bad state dimensions")
        if self.training:
            normalized = self.reader._state(state,len(x))
            # Even when the caller disables gradients, training needs them
            # for the inner differentiated error write.
            with torch.enable_grad():
                output,_=self._gradient_step(x,normalized)
            return output
        # enable_grad and inference_mode(False) are essential: a test-time
        # 'write' takes gradients wrt fast state even with frozen model weights.
        # No differentiable/float state survives after packing.
        with torch.inference_mode(False), torch.enable_grad():
            normal = self.reader._state(state.detach().clone(),len(x))
            if state.dtype==torch.uint8:
                normal = normal.detach().clone()
            value = x.detach().clone()
            out,_=self._gradient_step(value,normal)
            if self.kind=="phase":
                packed=pack_phase(out)
            else:
                packed=(out.detach()*256).round().clamp(0,255).to(torch.uint8)
        return packed

    def read(self,state:Tensor,query:Tensor)->Tensor:
        return self.reader.read(state,query)

    def param_count(self)->int:
        return sum(p.numel() for p in self.parameters())

    def config(self):
        return dict(slots=self.slots,value_size=self.value_size,
                    grid_size=self.reader.grid_size,channels=self.reader.channels,
                    width=self.reader.width,transport_steps=self.reader.transport_steps,
                    kind=self.kind,step_bins=self.step_bins,topk=self.topk,
                    tolerance=self.tolerance,second_order=self.second_order)


def read_all(cell:MetaDeltaReader,state:Tensor)->Tensor:
    n=len(state)
    q=torch.eye(cell.slots,dtype=cell.reader.query_projection.weight.dtype,
                device=state.device)
    q=q[None].expand(n,-1,-1).reshape(n*cell.slots,cell.slots)
    old=state[:,None].expand(-1,cell.slots,-1,-1).reshape(
        n*cell.slots,*state.shape[1:])
    return cell.read(old,q).reshape(n,cell.slots,cell.value_size)


def cross_talk_kernel(cell:MetaDeltaReader,state:Tensor)->dict[str,float]:
    """Normalized off-key Jacobian Gram score; diag/self and off diag."""
    with torch.inference_mode(False),torch.enable_grad():
        st=cell.reader._state(state.detach().clone()).detach().requires_grad_(True)
        B=len(st)
        gradients=[]
        for key in range(cell.slots):
            q=torch.zeros(B,cell.slots,device=st.device,dtype=st.dtype)
            q[:,key]=1
            y=cell.read(st,q)
            for component in range(cell.value_size):
                derivative=torch.autograd.grad(y[:,component].sum(),st,
                         retain_graph=True)[0].detach().flatten(1)
                gradients.append(derivative)
        G=torch.stack(gradients,dim=1).reshape(B,cell.slots,cell.value_size,-1)
        # Frobenius of J_i J_j^T divided by geometric mean of self
        # Frobenius norms; not an exact proof of orthogonal subspaces.
        grams=torch.einsum("bsve,btwe->bstvw",G,G)
        frob=grams.square().sum((-1,-2)).sqrt()
        diag=frob.diagonal(dim1=1,dim2=2)
        denom=(diag[:,:,None]*diag[:,None,:]).sqrt().clamp_min(1e-9)
        relative=frob/denom
        mask=~torch.eye(cell.slots,device=st.device,dtype=torch.bool)
        return {"offdiag_cross_kernel":float(relative[:,mask].mean()),
                "self_jacobian_frobenius":float(diag.mean())}
