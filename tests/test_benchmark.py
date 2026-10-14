import json

import torch

from phasegraph import PhaseGraphFastWeightCell
from phasegraph.benchmark import BenchmarkConfig, make_stream, read_all, run_benchmark, untouched_drift


def test_stream_targets_are_causal_latest_values_without_replay():
    stream = make_stream(5, 12, seed=41)
    assert stream['inputs'].shape == (5, 16, 7)
    prior = torch.zeros(5, 4, 3)
    for time in range(16):
        x = stream['inputs'][:, time]
        assert torch.all(x[:, :4].sum(-1) == 1)
        prior[torch.arange(5), x[:, :4].argmax(-1)] = x[:, 4:]
        torch.testing.assert_close(prior, stream['targets'][:, time])
    assert torch.equal(stream['inputs'], make_stream(5, 12, seed=41)['inputs'])
    assert not torch.equal(stream['inputs'], make_stream(5, 12, seed=42)['inputs'])


def test_drift_excludes_written_slot():
    before = torch.zeros(2, 4, 3)
    after = before.clone()
    after[0, 1], after[1, 2] = 10, 10
    assert untouched_drift(before, after, torch.tensor([1, 2])) == 0
    after[0, 0] = 3
    assert untouched_drift(before, after, torch.tensor([1, 2])) == 1.5


def test_all_queries_read_memory_only():
    cell = PhaseGraphFastWeightCell(7, 3, query_size=4).eval()
    state = cell.write(torch.randn(3, 7))
    result = read_all(cell, state, 4)
    assert result.shape == (3, 4, 3)
    for slot in range(4):
        query = torch.eye(4)[slot].expand(3, -1)
        torch.testing.assert_close(result[:, slot], cell.read(state, query))


def test_benchmark_smoke_saves_all_controls_and_exact_bytes(tmp_path):
    cfg = BenchmarkConfig(steps=2, seeds=(41,), batch_size=4, eval_count=8,
                          train_rewrites=2, horizons=(1, 4), width=16)
    result = run_benchmark(cfg, tmp_path)
    loaded = json.loads((tmp_path / 'summary.json').read_text())
    assert loaded == result
    assert set(result['aggregate']) == {'phase', 'signed', 'vector', 'delta'}
    for name in ('phase', 'signed', 'vector'):
        row = result['runs'][0]['models'][name]
        assert row['state_bytes'] == 48
        assert set(row['horizons']) == {'1', '4'}
        assert row['horizons']['4']['mse'] >= 0
        assert 'shuffled_mse' in row['horizons']['4']
        assert row['checkpoint']
    delta = result['runs'][0]['models']['delta']
    assert delta['state_bytes'] == 48
    assert delta['horizons']['4']['mse'] < 1e-12
    params = [result['runs'][0]['models'][name]['parameters'] for name in ('phase', 'signed', 'vector')]
    assert len(set(params)) == 1


def test_configuration_rejects_invalid_protocol():
    import pytest
    with pytest.raises(ValueError):
        BenchmarkConfig(steps=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(horizons=())
