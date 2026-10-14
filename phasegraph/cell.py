"""A learned phase writer and fresh-query normalized Hermitian graph reader.

Inspired by GraphNN's half-memory phase experiment (MIT, anttiluode, 2026).
This implementation has a vector interface, circular updates and symmetric
degree normalization. It is a reference cell, not a fused Mamba kernel.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .quantization import pack_phase, quantize_phase, unpack_phase


class PhaseGraphFastWeightCell(nn.Module):
    """Write x into edge bytes; read memory with a separate query.

    Training returns normalized, byte-rounded float state so BPTT can use
    surrogate gradients. ``eval()`` returns actual uint8 state after each
    write. State never lives inside the module. Floating computation and
    autograd storage are additional to the packed per-stream byte budget.

    ``kind='signed'`` and ``kind='vector'`` are parameter-matched controls.
    They use interval rather than circular updates; neither has extra state.
    """

    def __init__(self, input_size: int, output_size: int, query_size: int | None = None,
                 grid_size: int = 4, channels: int = 2, width: int = 64,
                 transport_steps: int = 4, kind: str = "phase"):
        super().__init__()
        query_size = input_size if query_size is None else query_size
        for name, value in dict(input_size=input_size, output_size=output_size,
                                query_size=query_size, grid_size=grid_size,
                                channels=channels, width=width,
                                transport_steps=transport_steps).items():
            if type(value) is not int or value < (2 if name == 'grid_size' else 1):
                raise ValueError(f"{name} must be a positive integer (grid_size >= 2)")
        if kind not in ("phase", "signed", "vector"):
            raise ValueError("kind must be phase, signed or vector")
        self.input_size, self.output_size, self.query_size = input_size, output_size, query_size
        self.grid_size, self.channels, self.width = grid_size, channels, width
        self.transport_steps, self.kind = transport_steps, kind
        self.edges = 2 * grid_size * (grid_size - 1)
        self.payload_bytes = channels * self.edges
        nodes = channels * grid_size * grid_size
        # All variants have exactly these parameters, including the same
        # circular feature lifting of the stored byte code for the writer.
        self.writer = nn.Sequential(nn.Linear(input_size + 2 * self.payload_bytes, width),
                                    nn.SiLU(), nn.Linear(width, 2 * self.payload_bytes))
        self.query_projection = nn.Linear(query_size, 2 * nodes)
        self.decoder = nn.Sequential(nn.Linear(6 * nodes, width), nn.SiLU(),
                                     nn.Linear(width, output_size))
        degree = torch.zeros(grid_size, grid_size)
        degree[:, :-1] += 1
        degree[:, 1:] += 1
        degree[:-1, :] += 1
        degree[1:, :] += 1
        self.register_buffer('horizontal_norm', (degree[:, :-1] * degree[:, 1:]).rsqrt())
        self.register_buffer('vertical_norm', (degree[:-1, :] * degree[1:, :]).rsqrt())

    @property
    def config(self) -> dict:
        return {name: getattr(self, name) for name in
                ('input_size', 'output_size', 'query_size', 'grid_size', 'channels',
                 'width', 'transport_steps', 'kind')}

    def state_bytes(self, batch_size: int = 1) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return batch_size * self.payload_bytes

    def parameter_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def buffer_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.buffers())

    def initial_state(self, batch_size: int, *, packed: bool | None = None) -> Tensor:
        self.state_bytes(batch_size)
        packed = not self.training if packed is None else packed
        weight = self.query_projection.weight
        return torch.full((batch_size, self.channels, self.edges), 128 if packed else .5,
                          dtype=torch.uint8 if packed else weight.dtype, device=weight.device)

    def _state(self, state: Tensor, batch: int | None = None) -> Tensor:
        weight = self.query_projection.weight
        if (not isinstance(state, Tensor) or state.ndim != 3 or state.shape[0] < 1
                or tuple(state.shape[1:]) != (self.channels, self.edges)
                or (batch is not None and state.shape[0] != batch)
                or state.device != weight.device):
            raise ValueError("state shape, batch or device is incompatible")
        if state.dtype == torch.uint8:
            return unpack_phase(state, dtype=weight.dtype)
        if (state.dtype != weight.dtype or state.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(state).all())
                or bool((state < 0).any())
                or bool((state >= 1).any() if self.kind == 'phase' else (state > 255 / 256).any())):
            raise ValueError("state must be uint8 or finite normalized model-dtype floats in the code range")
        return state

    def _input(self, value: Tensor, size: int, name: str) -> None:
        weight = self.query_projection.weight
        if (not isinstance(value, Tensor) or value.ndim != 2 or value.shape[0] < 1
                or value.shape[1] != size or value.device != weight.device
                or value.dtype != weight.dtype or value.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} must be finite [batch,{size}] model-dtype floats on the model device")

    @staticmethod
    def _mask(mask: Tensor | None, batch: int, device: torch.device, name: str) -> None:
        if mask is not None and (not isinstance(mask, Tensor) or mask.shape != (batch,)
                                 or mask.dtype != torch.bool or mask.device != device):
            raise ValueError(f"{name} mask must be boolean [batch] on the model device")

    def pack_state(self, state: Tensor) -> Tensor:
        normalized = self._state(state)
        if self.kind == 'phase':
            return pack_phase(normalized)
        return (normalized.detach() * 256).round().clamp(0, 255).to(torch.uint8)

    def unpack_state(self, state: Tensor) -> Tensor:
        return self._state(state)

    def write(self, x: Tensor, state: Tensor | None = None, *,
              write_mask: Tensor | None = None, reset_mask: Tensor | None = None) -> Tensor:
        self._input(x, self.input_size, 'input')
        batch = len(x)
        self._mask(write_mask, batch, x.device, 'write')
        self._mask(reset_mask, batch, x.device, 'reset')
        prior = self.initial_state(batch, packed=False) if state is None else self._state(state, batch)
        if reset_mask is not None:
            prior = torch.where(reset_mask[:, None, None], .5, prior)
        flat = prior.flatten(1)
        context = torch.cat((x, (2 * math.pi * flat).sin(), (2 * math.pi * flat).cos()), -1)
        rotation, gate = self.writer(context).chunk(2, dim=-1)
        proposal = flat + .25 * rotation.tanh() * gate.sigmoid()
        if self.kind == 'phase':
            updated = quantize_phase(proposal)
        else:
            clipped = proposal.clamp(0, 255 / 256)
            updated = clipped + ((clipped * 256).round() / 256 - clipped).detach()
        updated = updated.reshape_as(prior)
        if write_mask is not None:
            updated = torch.where(write_mask[:, None, None], updated, prior)
        return updated if self.training else self.pack_state(updated)

    def _couplings(self, normalized: Tensor) -> tuple[Tensor, Tensor]:
        g = self.grid_size
        h, v = normalized.chunk(2, dim=-1)
        if self.kind == 'phase':
            theta_h, theta_v = 2 * math.pi * h, 2 * math.pi * v
            hc = torch.complex(theta_h.cos(), theta_h.sin())
            vc = torch.complex(theta_v.cos(), theta_v.sin())
        else:
            hc = torch.complex(2 * h * (256 / 255) - 1, torch.zeros_like(h))
            vc = torch.complex(2 * v * (256 / 255) - 1, torch.zeros_like(v))
        return (hc.reshape(-1, self.channels, g, g - 1) * self.horizontal_norm,
                vc.reshape(-1, self.channels, g - 1, g) * self.vertical_norm)

    @staticmethod
    def _hop(field: Tensor, hc: Tensor, vc: Tensor) -> Tensor:
        neighbors = (F.pad(hc * field[..., 1:], (0, 1, 0, 0))
                     + F.pad(hc.conj() * field[..., :-1], (1, 0, 0, 0))
                     + F.pad(vc * field[..., 1:, :], (0, 0, 0, 1))
                     + F.pad(vc.conj() * field[..., :-1, :], (0, 0, 1, 0)))
        return .5 * (field + neighbors)

    def transport(self, field: Tensor, state: Tensor, *, steps: int | None = None) -> Tensor:
        """Transport [batch,channels,nodes] complex probes (also useful for diagnostics)."""
        normalized = self._state(state)
        steps = self.transport_steps if steps is None else steps
        expected_dtype = torch.complex64 if normalized.dtype == torch.float32 else torch.complex128
        if (type(steps) is not int or steps < 1 or field.shape !=
                (len(normalized), self.channels, self.grid_size ** 2)
                or field.device != normalized.device or field.dtype != expected_dtype
                or not bool(torch.isfinite(field).all())):
            raise ValueError("field must be finite complex [batch,channels,nodes]; steps >= 1")
        hc, vc = self._couplings(normalized)
        work = field.reshape(-1, self.channels, self.grid_size, self.grid_size)
        for _ in range(steps):
            work = self._hop(work, hc, vc)
        return work.flatten(2)

    def read(self, state: Tensor, query: Tensor) -> Tensor:
        """No observation, source pixels, previous output or writer activation is read."""
        self._input(query, self.query_size, 'query')
        normalized = self._state(state, len(query))
        g, c = self.grid_size, self.channels
        real, imag = self.query_projection(query).tanh().chunk(2, dim=-1)
        field = torch.complex(real, imag).reshape(-1, c, g, g)
        features = [field.real.flatten(1), field.imag.flatten(1)]
        if self.kind == 'vector':
            h, v = normalized.chunk(2, dim=-1)
            h, v = h.reshape(-1, c, g, g - 1), v.reshape(-1, c, g - 1, g)
            features += [F.pad(fn(2 * math.pi * plane), padding).flatten(1)
                         for plane, padding in ((h, (0, 1, 0, 0)), (v, (0, 0, 0, 1)))
                         for fn in (torch.sin, torch.cos)]
        else:
            hc, vc = self._couplings(normalized)
            middle = None
            for depth in range(1, self.transport_steps + 1):
                field = self._hop(field, hc, vc)
                if depth == max(1, self.transport_steps // 2):
                    middle = field
            features += [middle.real.flatten(1), middle.imag.flatten(1),
                         field.real.flatten(1), field.imag.flatten(1)]
        return self.decoder(torch.cat(features, dim=-1))

    def forward(self, x: Tensor, state: Tensor | None = None, query: Tensor | None = None,
                write_mask: Tensor | None = None, reset_mask: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        if query is None:
            if self.query_size != self.input_size:
                raise ValueError("query is required when query_size != input_size")
            query = x
        self._input(query, self.query_size, 'query')
        if len(query) != len(x):
            raise ValueError("query batch must match input batch")
        updated = self.write(x, state, write_mask=write_mask, reset_mask=reset_mask)
        return self.read(updated, query), updated

