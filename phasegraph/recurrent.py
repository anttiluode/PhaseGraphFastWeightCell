"""Batch-first sequence wrapper; step and chunk execution share the same cell."""

import torch
from torch import Tensor, nn

from .cell import PhaseGraphFastWeightCell


class PhaseGraphRNN(nn.Module):
    def __init__(self, cell: PhaseGraphFastWeightCell):
        super().__init__()
        self.cell = cell

    def forward(self, x: Tensor, state: Tensor | None = None, *,
                queries: Tensor | None = None, write_mask: Tensor | None = None,
                reset_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if not isinstance(x, Tensor) or x.ndim != 3 or x.shape[0] < 1:
            raise ValueError("input sequence must be [batch,time,input_size]")
        batch, length, _ = x.shape
        weight = self.cell.query_projection.weight
        if (x.shape[-1] != self.cell.input_size or x.dtype != weight.dtype or x.device != weight.device
                or x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all())):
            raise ValueError("input sequence has incompatible shape, dtype, device or nonfinite values")
        if queries is None:
            if self.cell.query_size != self.cell.input_size:
                raise ValueError("queries are required when query_size != input_size")
            queries = x
        if (not isinstance(queries, Tensor) or queries.shape != (batch, length, self.cell.query_size)
                or queries.dtype != x.dtype or queries.device != x.device
                or not bool(torch.isfinite(queries).all())):
            raise ValueError("query sequence must be finite [batch,time,query_size] model-dtype floats")
        for name, mask in (('write', write_mask), ('reset', reset_mask)):
            if mask is not None and (not isinstance(mask, Tensor) or mask.shape != (batch, length)
                                     or mask.dtype != torch.bool or mask.device != x.device):
                raise ValueError(f"{name} mask must be boolean [batch,time] on the model device")
        if state is None:
            state = self.cell.initial_state(batch)
        else:
            self.cell._state(state, batch)
        outputs = []
        for step in range(length):
            output, state = self.cell(
                x[:, step], state, queries[:, step],
                None if write_mask is None else write_mask[:, step],
                None if reset_mask is None else reset_mask[:, step])
            outputs.append(output)
        result = torch.stack(outputs, 1) if outputs else x.new_empty((batch, 0, self.cell.output_size))
        return result, state

