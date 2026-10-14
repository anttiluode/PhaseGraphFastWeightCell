"""Analytic delta-rule positive control, not a reproduction of trained DeltaNet.

An orthonormal key for each slot makes its error-correcting rewrite exact.
This is deliberately included to avoid treating failure of weaker learned
controls as evidence that standard fast weights cannot solve the task.
"""

import math

import torch
from torch import Tensor
from torch.nn import functional as F


class DeltaRuleMemory:
    def __init__(self, key_size: int, value_size: int, beta: float = 1.):
        if type(key_size) is not int or key_size < 1 or type(value_size) is not int or value_size < 1:
            raise ValueError("key_size and value_size must be positive integers")
        if not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 < beta <= 1:
            raise ValueError("beta must be finite in (0,1]")
        self.key_size, self.value_size, self.beta = key_size, value_size, float(beta)

    def state_bytes(self, batch_size: int = 1) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be positive")
        return batch_size * self.key_size * self.value_size * 4

    def initial_state(self, batch_size: int, *, device=None) -> Tensor:
        self.state_bytes(batch_size)
        return torch.zeros(batch_size, self.key_size, self.value_size, device=device, dtype=torch.float32)

    def _keys(self, keys: Tensor) -> Tensor:
        if (not isinstance(keys, Tensor) or keys.ndim != 2 or keys.shape[0] < 1
                or keys.shape[1] != self.key_size or keys.dtype != torch.float32
                or not bool(torch.isfinite(keys).all())
                or bool((keys.square().sum(-1) == 0).any())):
            raise ValueError("keys must be finite nonzero float32 [batch,key_size]")
        return F.normalize(keys, dim=-1)

    def _state(self, state: Tensor, keys: Tensor) -> None:
        if (not isinstance(state, Tensor) or state.shape != (len(keys), self.key_size, self.value_size)
                or state.dtype != torch.float32 or state.device != keys.device
                or not bool(torch.isfinite(state).all())):
            raise ValueError("delta state shape, device or dtype is incompatible")

    def write(self, keys: Tensor, values: Tensor, state: Tensor | None = None) -> Tensor:
        keys = self._keys(keys)
        if (not isinstance(values, Tensor) or values.shape != (len(keys), self.value_size)
                or values.dtype != torch.float32 or values.device != keys.device
                or not bool(torch.isfinite(values).all())):
            raise ValueError("values must be finite float32 [batch,value_size] on the keys' device")
        if state is None:
            state = self.initial_state(len(keys), device=keys.device)
        self._state(state, keys)
        prediction = torch.einsum('bk,bkv->bv', keys, state)
        return state + self.beta * keys[:, :, None] * (values - prediction)[:, None, :]

    def read(self, state: Tensor, query: Tensor) -> Tensor:
        keys = self._keys(query)
        self._state(state, keys)
        return torch.einsum('bk,bkv->bv', keys, state)

