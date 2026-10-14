import math

import pytest
import torch

from phasegraph import PhaseGraphFastWeightCell
from phasegraph.quantization import pack_phase, quantize_phase, unpack_phase


def small(**kwargs):
    return PhaseGraphFastWeightCell(7, 3, query_size=4, grid_size=3,
                                    channels=2, width=16, **kwargs)


def test_all_256_bins_roundtrip_and_seam_wraps():
    codes = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(pack_phase(unpack_phase(codes)), codes)
    assert torch.unique(unpack_phase(codes)).numel() == 256
    turns = torch.tensor([-1 / 256, 255 / 256, 1., 1 + 1 / 256])
    assert torch.equal(pack_phase(turns), torch.tensor([255, 255, 0, 1], dtype=torch.uint8))
    turns.requires_grad_()
    quantize_phase(turns).sum().backward()
    assert torch.equal(turns.grad, torch.ones_like(turns))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_circular_quantization_is_exact_across_negative_and_positive_seams(dtype):
    generator = torch.Generator().manual_seed(41)
    turns = (torch.rand(8192, generator=generator, dtype=dtype) * 1.5 - .25).requires_grad_()
    rounded = unpack_phase(pack_phase(turns), dtype=dtype)
    output = quantize_phase(turns)
    assert torch.equal(output, rounded)
    output.sum().backward()
    assert torch.equal(turns.grad, torch.ones_like(turns))


def test_repeated_training_and_packed_writes_are_exactly_identical_across_seams():
    import copy
    torch.manual_seed(41)
    training = small()
    inference = copy.deepcopy(training).eval()
    codes = torch.randint(0, 256, (8, 2, 12), dtype=torch.uint8)
    trained, packed = training.unpack_state(codes), codes
    with torch.no_grad():
        for _ in range(128):
            x = torch.randn(8, 7)
            trained = training.write(x, trained)
            packed = inference.write(x, packed)
            assert torch.equal(trained, inference.unpack_state(packed))


def test_real_byte_storage_and_independent_initial_streams():
    cell = small().eval()
    state = cell.initial_state(3)
    assert state.dtype == torch.uint8
    assert state.shape == (3, 2, 12)
    assert state.numel() * state.element_size() == cell.state_bytes(3) == 72
    state[0].zero_()
    assert torch.all(state[1:] == 128)
    assert PhaseGraphFastWeightCell(7, 3, grid_size=8, channels=8).state_bytes() == 896
    assert cell.parameter_bytes() == sum(p.numel() * p.element_size() for p in cell.parameters())


def test_every_write_is_quantized_and_eval_matches_training_forward():
    torch.manual_seed(41)
    cell = small()
    x, query = torch.randn(2, 7), torch.randn(2, 4)
    before = cell.initial_state(2)
    output, state = cell(x, before, query)
    assert state.dtype == torch.float32
    assert torch.equal(state * 256, (state * 256).round())
    cell.eval()
    packed_output, packed = cell(x, cell.pack_state(before), query)
    assert packed.dtype == torch.uint8
    assert torch.equal(packed, cell.pack_state(state))
    torch.testing.assert_close(output, packed_output, rtol=1e-5, atol=1e-6)


def test_gradient_reaches_earliest_write_through_quantized_recurrence():
    torch.manual_seed(97)
    cell = small()
    first = torch.randn(2, 7, requires_grad=True)
    state = cell.write(first)
    for _ in range(3):
        state = cell.write(torch.randn(2, 7), state)
    cell.read(state, torch.randn(2, 4)).square().sum().backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
    assert cell.writer[0].weight.grad.abs().sum() > 0
    assert cell.query_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('kind', ['phase', 'signed'])
def test_normalized_transport_is_hermitian_and_contracts_field_norm(kind):
    torch.manual_seed(127)
    cell = small(kind=kind).double().eval()
    n = 9
    state = torch.randint(0, 256, (1, 2, 12), dtype=torch.uint8)
    basis = torch.eye(n, dtype=torch.complex128)[:, None].expand(-1, 2, -1)
    matrix = cell.transport(basis, state.expand(n, -1, -1), steps=1).permute(1, 2, 0)
    torch.testing.assert_close(matrix, matrix.mH)
    assert torch.linalg.eigvalsh(matrix).min() >= -1e-12
    assert torch.linalg.eigvalsh(matrix).max() <= 1 + 1e-12
    field = torch.randn(4, 2, n, dtype=torch.complex128)
    end = cell.transport(field, state.expand(4, -1, -1), steps=32)
    assert torch.all(end.abs().square().sum(-1) <= field.abs().square().sum(-1) + 1e-10)


def test_query_changes_spatial_excitation_and_read_does_not_modify_memory():
    torch.manual_seed(23)
    cell = small().eval()
    state = torch.randint(0, 256, (2, 2, 12), dtype=torch.uint8)
    copy = state.clone()
    query = torch.eye(4)[:2]
    out = cell.read(state, query)
    assert not torch.allclose(out, cell.read(state, query.flip(0)))
    assert torch.equal(state, copy)


def test_no_write_and_reset_are_per_stream_and_functional():
    cell = small().eval()
    state = torch.randint(0, 256, (3, 2, 12), dtype=torch.uint8)
    copy = state.clone()
    result = cell.write(torch.randn(3, 7), state,
                        write_mask=torch.tensor([False, True, False]),
                        reset_mask=torch.tensor([False, False, True]))
    assert torch.equal(result[0], state[0])
    assert torch.equal(result[2], cell.initial_state(1)[0])
    assert torch.equal(state, copy)


@pytest.mark.parametrize('bad', [torch.zeros(2, 6), torch.full((2, 7), math.nan),
                               torch.zeros(2, 7, dtype=torch.int64)])
def test_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        small().write(bad)


def test_rejects_invalid_state_query_and_masks():
    cell = small()
    x = torch.randn(2, 7)
    with pytest.raises(ValueError, match='query'):
        cell(x)
    with pytest.raises(ValueError, match='state'):
        cell.write(x, torch.zeros(2, 2, 13))
    with pytest.raises(ValueError, match='state'):
        cell.write(x, torch.full((2, 2, 12), float('inf')))
    with pytest.raises(ValueError, match='mask'):
        cell.write(x, write_mask=torch.tensor([1., 0.]))
    with pytest.raises(ValueError, match='query'):
        cell.read(cell.initial_state(2), torch.zeros(2, 3))


@pytest.mark.parametrize('kind', ['signed', 'vector'])
def test_interval_state_rejects_values_above_highest_physical_code(kind):
    cell = small(kind=kind)
    bad = torch.full((2, 2, 12), .999)
    with pytest.raises(ValueError, match='state'):
        cell.read(bad, torch.zeros(2, 4))
    # All physical byte codes, including the upper endpoint, remain valid.
    valid = torch.full((2, 2, 12), 255 / 256)
    assert torch.isfinite(cell.read(valid, torch.zeros(2, 4))).all()


@pytest.mark.parametrize('kw', [{'grid_size': 1}, {'channels': 0}, {'width': 0},
                               {'transport_steps': 0}, {'kind': 'unitary'}])
def test_rejects_invalid_configuration(kw):
    with pytest.raises(ValueError):
        PhaseGraphFastWeightCell(7, 3, **kw)
