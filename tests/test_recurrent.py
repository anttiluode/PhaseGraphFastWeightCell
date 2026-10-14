import io

import pytest
import torch

from phasegraph import PhaseGraphFastWeightCell, PhaseGraphRNN
from phasegraph.baselines import DeltaRuleMemory


def cell(kind='phase'):
    return PhaseGraphFastWeightCell(7, 3, query_size=4, width=16, kind=kind)


def test_sequence_steps_and_chunks_have_identical_outputs_and_state():
    torch.manual_seed(41)
    model = PhaseGraphRNN(cell().eval())
    x, q = torch.randn(3, 7, 7), torch.randn(3, 7, 4)
    writes = torch.rand(3, 7) > .3
    resets = torch.zeros(3, 7, dtype=torch.bool)
    resets[1, 4] = True
    full, state = model(x, queries=q, write_mask=writes, reset_mask=resets)
    left, first = model(x[:, :4], queries=q[:, :4], write_mask=writes[:, :4], reset_mask=resets[:, :4])
    right, last = model(x[:, 4:], first, queries=q[:, 4:], write_mask=writes[:, 4:], reset_mask=resets[:, 4:])
    torch.testing.assert_close(full, torch.cat((left, right), 1))
    assert torch.equal(state, last)
    assert state.dtype == torch.uint8
    pieces, prior = [], None
    for step in range(7):
        y, prior = model.cell(x[:, step], prior, q[:, step], writes[:, step], resets[:, step])
        pieces.append(y)
    torch.testing.assert_close(full, torch.stack(pieces, 1))
    assert torch.equal(prior, state)


def test_empty_sequence_returns_empty_output_and_original_state():
    model = PhaseGraphRNN(cell().eval())
    initial = torch.randint(0, 256, (2, 2, 24), dtype=torch.uint8)
    result, state = model(torch.empty(2, 0, 7), initial, queries=torch.empty(2, 0, 4))
    assert result.shape == (2, 0, 3)
    assert torch.equal(initial, state)


def test_checkpoint_roundtrip_needs_no_observations():
    original = cell().eval()
    x, q = torch.randn(2, 7), torch.randn(2, 4)
    out, state = original(x, query=q)
    stream = io.BytesIO()
    torch.save(dict(config=original.config, model=original.state_dict(), memory=state), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    loaded = PhaseGraphFastWeightCell(**saved['config']).eval()
    loaded.load_state_dict(saved['model'])
    torch.testing.assert_close(out, loaded.read(saved['memory'], q))


def test_control_parameters_and_storage_are_exactly_matched():
    models = [cell(kind) for kind in ('phase', 'signed', 'vector')]
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    assert len({m.state_bytes() for m in models}) == 1
    for model in models:
        model.eval()
        x, q = torch.randn(2, 7), torch.randn(2, 4)
        result, state = model(x, query=q)
        assert result.shape == (2, 3) and torch.isfinite(result).all()
        assert state.dtype == torch.uint8


def test_delta_rule_exactly_rewrites_orthogonal_keys_without_touching_others():
    memory = DeltaRuleMemory(4, 3)
    keys = torch.eye(4)
    state = memory.initial_state(1)
    values = torch.tensor([[-.5, .2, .9], [.1, -.7, .3], [.8, .4, .2], [-.1, .9, .6]])
    for slot in range(4):
        state = memory.write(keys[slot:slot + 1], values[slot:slot + 1], state)
    before = state.clone()
    update = torch.tensor([[.7, -.2, -.8]])
    state = memory.write(keys[:1], update, state)
    torch.testing.assert_close(memory.read(state, keys[:1]), update)
    assert torch.equal(state[:, 1:], before[:, 1:])
    assert state.numel() * state.element_size() == memory.state_bytes() == 48


def test_delta_state_is_float32_independent_of_global_default_dtype():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        memory = DeltaRuleMemory(4, 3)
        state = memory.initial_state(1)
        assert state.dtype == torch.float32
        assert state.numel() * state.element_size() == memory.state_bytes()
        key = torch.tensor([[1., 0., 0., 0.]], dtype=torch.float32)
        values = torch.tensor([[.2, -.3, .4]], dtype=torch.float32)
        updated = memory.write(key, values)
        torch.testing.assert_close(memory.read(updated, key), values)
    finally:
        torch.set_default_dtype(original_dtype)


def test_wrapper_uses_implicit_queries_and_preserves_bptt():
    model = PhaseGraphRNN(PhaseGraphFastWeightCell(7, 3, width=16))
    x = torch.randn(2, 3, 7, requires_grad=True)
    y, state = model(x)
    y[:, -1].square().sum().backward()
    assert x.grad[:, 0].abs().sum() > 0
    assert state.is_floating_point()


def test_wrapper_rejects_malformed_sequences_and_masks():
    model = PhaseGraphRNN(cell())
    with pytest.raises(ValueError):
        model(torch.randn(2, 7))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 7), queries=torch.randn(2, 2, 4))
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 7), queries=torch.randn(2, 3, 4), write_mask=torch.ones(2, 3))
