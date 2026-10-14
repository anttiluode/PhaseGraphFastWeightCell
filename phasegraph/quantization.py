"""256 unique circular bins. Training gradients use a straight-through estimator."""

import torch
from torch import Tensor


def _floating(value: Tensor) -> None:
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("phase values must be finite floating tensors")


def quantize_phase(turns: Tensor) -> Tensor:
    """Wrap and round in turns, with identity surrogate derivative at the seam."""
    _floating(turns)
    bins = torch.remainder(torch.round(torch.remainder(turns, 1) * 256), 256)
    # Add a numerically exact zero with derivative one. Subtracting the
    # unwrapped value from a wrapped bin before adding it back can lose a
    # few bits at negative seams and move the forward value off the grid.
    return bins.detach() / 256 + (turns - turns.detach())


def pack_phase(turns: Tensor) -> Tensor:
    """Detach and store one real byte per phase; no floating shadow state."""
    _floating(turns)
    return torch.remainder(torch.round(torch.remainder(turns.detach(), 1) * 256), 256).to(torch.uint8)


def unpack_phase(codes: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    if codes.dtype != torch.uint8:
        raise ValueError("phase codes must be uint8")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("unpacked phase dtype must be float32 or float64")
    return codes.to(dtype) / 256

