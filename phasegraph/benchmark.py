"""Frozen latest-value rewrite benchmark. Run: python -m phasegraph.benchmark.

Thresholds describe this synthetic task only. Parameters, bytes, seeds and
data are matched across the three learned arms; inference compute differs.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.nn import functional as F

from .baselines import DeltaRuleMemory
from .cell import PhaseGraphFastWeightCell


@dataclass(frozen=True)
class BenchmarkConfig:
    steps: int = 600
    seeds: tuple[int, ...] = (41, 97, 127)
    batch_size: int = 64
    eval_count: int = 128
    slots: int = 4
    value_size: int = 3
    train_rewrites: int = 16
    horizons: tuple[int, ...] = (1, 4, 16, 64)
    grid_size: int = 4
    channels: int = 2
    width: int = 64
    transport_steps: int = 4
    learning_rate: float = .003
    device: str = 'cpu'
    threads: int = 1

    def __post_init__(self):
        for name in ('steps', 'batch_size', 'eval_count', 'slots', 'value_size', 'train_rewrites',
                     'channels', 'width', 'transport_steps', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.slots < 2 or self.eval_count < 2 or type(self.grid_size) is not int or self.grid_size < 2:
            raise ValueError('slots/eval_count/grid_size must be >= 2')
        if (not self.horizons or any(type(h) is not int or h < 1 for h in self.horizons)
                or tuple(sorted(set(self.horizons))) != tuple(self.horizons)):
            raise ValueError('horizons must be nonempty, increasing unique positive integers')
        if not self.seeds or any(type(seed) is not int or not 0 <= seed < 2 ** 31 for seed in self.seeds):
            raise ValueError('seeds must be nonempty nonnegative integers below 2**31')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')


def make_stream(batch_size: int, rewrites: int, seed: int, *, slots: int = 4,
                value_size: int = 3, device='cpu') -> dict[str, torch.Tensor]:
    """Every input contains just one key and its new value, never retained slots."""
    if min(batch_size, slots, value_size) < 1 or rewrites < 0:
        raise ValueError('stream dimensions must be positive and rewrites nonnegative')
    generator = torch.Generator().manual_seed(seed)
    ids = torch.cat((torch.arange(slots).expand(batch_size, -1),
                     torch.randint(slots, (batch_size, rewrites), generator=generator)), 1)
    values = torch.rand(batch_size, slots + rewrites, value_size, generator=generator) * 2 - 1
    inputs = torch.cat((F.one_hot(ids, slots).float(), values), -1)
    latest = torch.zeros(batch_size, slots, value_size)
    targets = []
    for step in range(slots + rewrites):
        latest[torch.arange(batch_size), ids[:, step]] = values[:, step]
        targets.append(latest.clone())
    return {'inputs': inputs.to(device), 'targets': torch.stack(targets, 1).to(device),
            'slots': ids.to(device)}


def read_all(cell: PhaseGraphFastWeightCell, state: torch.Tensor, slots: int) -> torch.Tensor:
    batch = len(state)
    query = torch.eye(slots, device=state.device, dtype=cell.query_projection.weight.dtype)
    queries = query[None].expand(batch, -1, -1).reshape(batch * slots, slots)
    repeated = state[:, None].expand(-1, slots, -1, -1).reshape(batch * slots, *state.shape[1:])
    return cell.read(repeated, queries).reshape(batch, slots, cell.output_size)


def untouched_drift(before: torch.Tensor, after: torch.Tensor, written: torch.Tensor) -> float:
    keep = torch.arange(before.shape[1], device=written.device)[None] != written[:, None]
    return float((after - before).square()[keep].mean())


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _train(cfg: BenchmarkConfig, kind: str, seed: int):
    torch.manual_seed(seed)
    cell = PhaseGraphFastWeightCell(cfg.slots + cfg.value_size, cfg.value_size,
                                   query_size=cfg.slots, grid_size=cfg.grid_size,
                                   channels=cfg.channels, width=cfg.width,
                                   transport_steps=cfg.transport_steps, kind=kind).to(cfg.device)
    optimizer = torch.optim.Adam(cell.parameters(), lr=cfg.learning_rate)
    schedule = random.Random(seed + 17)
    trace = []
    _sync(cfg.device)
    started = time.perf_counter()
    for step in range(cfg.steps):
        rewrites = schedule.randint(1, cfg.train_rewrites)
        stream = make_stream(cfg.batch_size, rewrites, seed * 100_000 + step,
                             slots=cfg.slots, value_size=cfg.value_size, device=cfg.device)
        state = None
        for observation in stream['inputs'].unbind(1):
            state = cell.write(observation, state)
        prediction = read_all(cell, state, cfg.slots)
        loss = F.mse_loss(prediction, stream['targets'][:, -1])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'nonfinite training loss in {kind}, seed {seed}, step {step + 1}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            trace.append({'step': step + 1, 'mse': float(loss.detach())})
            print(f'seed={seed} kind={kind} step={step + 1}/{cfg.steps} mse={float(loss.detach()):.5f}', flush=True)
    _sync(cfg.device)
    return cell.eval(), trace, time.perf_counter() - started


@torch.inference_mode()
def _evaluate(cfg: BenchmarkConfig, memory, seed: int, *, delta=False):
    data = make_stream(cfg.eval_count, max(cfg.horizons), 1_000_000_000 + seed,
                       slots=cfg.slots, value_size=cfg.value_size, device=cfg.device)
    state = None
    def write(observation, old):
        if delta:
            return memory.write(observation[:, :cfg.slots], observation[:, cfg.slots:], old)
        return memory.write(observation, old)
    def read(old):
        if delta:
            # With explicit orthogonal keys the matrix's rows are the exact queried values.
            queries = torch.eye(cfg.slots, device=old.device)
            return torch.stack([memory.read(old, query.expand(len(old), -1)) for query in queries], 1)
        return read_all(memory, old, cfg.slots)
    _sync(cfg.device)
    started = time.perf_counter()
    for step in range(cfg.slots):
        state = write(data['inputs'][:, step], state)
    before = read(state)
    frozen_start = before.clone()
    drift_sum = 0.
    rows, trajectory = {}, []
    for rewrite in range(1, max(cfg.horizons) + 1):
        step = cfg.slots + rewrite - 1
        state = write(data['inputs'][:, step], state)
        after = read(state)
        target, written = data['targets'][:, step], data['slots'][:, step]
        drift = untouched_drift(before, after, written)
        mse = float(F.mse_loss(after, target))
        drift_sum += drift
        trajectory.append({'rewrites': rewrite, 'mse': mse, 'untouched_drift_mse': drift})
        if rewrite in cfg.horizons:
            row_indices = torch.arange(cfg.eval_count, device=state.device)
            rows[str(rewrite)] = {
                'mse': mse,
                'updated_slot_mse': float(F.mse_loss(after[row_indices, written], target[row_indices, written])),
                'untouched_drift_mse': drift_sum / rewrite,
                'shuffled_mse': float(F.mse_loss(read(state.roll(1, 0)), target)),
                'blank_mse': float(F.mse_loss(read(torch.zeros_like(state)), target)),
                'skip_all_rewrites_mse': float(F.mse_loss(frozen_start, target)),
            }
        before = after
    _sync(cfg.device)
    return rows, trajectory, time.perf_counter() - started, state


def _source_info():
    root = Path(__file__).resolve().parent
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.glob('*.py'))}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {'git_revision': revision, 'sha256': hashes}


def run_benchmark(cfg: BenchmarkConfig, output: str | Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(cfg.threads)
    try:
        result = {'protocol': 'latest-value-v1', 'config': json.loads(json.dumps(asdict(cfg))),
                  'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
                                  'platform': platform.platform(), 'device': cfg.device,
                                  'threads': cfg.threads}, 'source': _source_info(), 'runs': []}
        for seed in cfg.seeds:
            models = {}
            for kind in ('phase', 'signed', 'vector'):
                cell, trace, train_seconds = _train(cfg, kind, seed)
                rows, trajectory, eval_seconds, state = _evaluate(cfg, cell, seed)
                relative = f'seed{seed}/{kind}.pt'
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'config': cell.config, 'model': cell.cpu().state_dict(),
                            'protocol': result['protocol'], 'seed': seed}, path)
                models[kind] = {'state_bytes': state.numel() * state.element_size() // cfg.eval_count,
                                'parameters': sum(p.numel() for p in cell.parameters()),
                                'parameter_bytes': cell.parameter_bytes(), 'buffer_bytes': cell.buffer_bytes(),
                                'training_float_state_bytes': cell.state_bytes() * 4,
                                'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                                'training_trace': trace, 'checkpoint': relative,
                                'horizons': rows, 'trajectory': trajectory}
            delta = DeltaRuleMemory(cfg.slots, cfg.value_size)
            rows, trajectory, seconds, state = _evaluate(cfg, delta, seed, delta=True)
            models['delta'] = {'state_bytes': state.numel() * state.element_size() // cfg.eval_count,
                               'parameters': 0, 'parameter_bytes': 0, 'buffer_bytes': 0,
                               'eval_seconds': seconds, 'horizons': rows, 'trajectory': trajectory}
            result['runs'].append({'seed': seed, 'models': models})
            # Keep a recoverable checkpoint after each complete seed.
            (output / 'partial.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        result['aggregate'] = {}
        for kind in ('phase', 'signed', 'vector', 'delta'):
            result['aggregate'][kind] = {}
            for horizon in cfg.horizons:
                entries = [run['models'][kind]['horizons'][str(horizon)] for run in result['runs']]
                result['aggregate'][kind][str(horizon)] = {
                    key: sum(row[key] for row in entries) / len(entries) for key in entries[0]}
        result['gates'] = {
            '16_update_mse_at_most_0.05_all_seeds':
                all(run['models']['phase']['horizons']['16']['mse'] <= .05 for run in result['runs'])
                if 16 in cfg.horizons else None,
            '64_update_mse_at_most_0.10_all_seeds':
                all(run['models']['phase']['horizons']['64']['mse'] <= .10 for run in result['runs'])
                if 64 in cfg.horizons else None,
            'phase_beats_both_learned_controls_at_16_and_64_all_seeds':
                all(run['models']['phase']['horizons'][str(h)]['mse'] <
                    run['models'][other]['horizons'][str(h)]['mse']
                    for run in result['runs'] for h in (16, 64) for other in ('signed', 'vector'))
                if 16 in cfg.horizons and 64 in cfg.horizons else None,
        }
        (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        (output / 'partial.json').unlink(missing_ok=True)
        _report(result, output)
        return result
    finally:
        torch.set_num_threads(old_threads)


def _report(result, output):
    cfg = result['config']
    lines = ['# Frozen sequential rewrite results', '',
             f"{len(cfg['seeds'])} seeds, {cfg['steps']} optimizer steps per model, "
             f"{cfg['eval_count']} independent held-out streams per seed. "
             f"Training uses 1–{cfg['train_rewrites']} random rewrites after initializing all slots.", '',
             '| Model | State bytes | Parameters | ' + ' | '.join(f'MSE at {h}' for h in cfg['horizons']) + ' |',
             '| --- | ---: | ---: | ' + ' | '.join('---:' for _ in cfg['horizons']) + ' |']
    for kind in ('phase', 'signed', 'vector', 'delta'):
        row = result['runs'][0]['models'][kind]
        scores = ' | '.join(f"{result['aggregate'][kind][str(h)]['mse']:.6f}" for h in cfg['horizons'])
        lines.append(f"| {kind} | {row['state_bytes']} | {row['parameters']} | {scores} |")
    lines += ['', '## Predeclared gates', '']
    for gate, passed in result['gates'].items():
        lines.append(f"- {gate}: **{'not measured' if passed is None else 'PASS' if passed else 'FAIL'}**")
    lines += ['', '## Interpretation boundaries', '',
              'These are continuous synthetic latest-value streams, with explicit slot keys. '
              'The delta rule is an analytic orthogonal-key positive control, not a trained Mamba/DeltaNet model. '
              'All learned arms share parameter counts, state bytes, data, optimizer and step budget. '
              'Signed and vector arms use interval clipping; phase uses circular rotations. '
              'The writer sees sine/cosine features of the byte codes in every learned arm. '
              'The graph arms consume more read compute than the vector arm.', '',
              'MSE is averaged across all slots and value components. Untouched drift is the squared '
              'change in predictions for keys other than the latest written key, averaged over preceding rewrites. '
              'Low drift by itself is not useful retention: a constant reader also has zero drift. '
              'Shuffled, blank and skipped-update controls and full seed-level scores are in summary.json.', '',
              'State bytes count actual inference tensors, excluding shared weights and topology buffers '
              '(reported separately), temporary float/complex fields, training activations and file metadata. '
              'Timing includes diagnostic reads of every slot after every update and control reads; '
              'it is descriptive reference-code runtime, not an optimized kernel benchmark.', '',
              'Bounded bytes and normalized field transport do not guarantee task accuracy, orthogonal '
              'query isolation, bounded gradients, or superiority to modern recurrent models. '
              'This experiment does not establish a pretrained-model replacement or language-model quality.', '']
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/benchmark')
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seeds', type=int, nargs='+', default=[41, 97, 127])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--eval-count', type=int, default=128)
    parser.add_argument('--train-rewrites', type=int, default=16)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 16, 64])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--width', type=int, default=64)
    args = parser.parse_args()
    result = run_benchmark(BenchmarkConfig(steps=args.steps, seeds=tuple(args.seeds),
        batch_size=args.batch_size, eval_count=args.eval_count, train_rewrites=args.train_rewrites,
        horizons=tuple(args.horizons), device=args.device, threads=args.threads, width=args.width), args.output)
    print(json.dumps({'output': str(args.output), 'gates': result['gates']}, indent=2))


if __name__ == '__main__':
    main()
