"""Frozen latest-value updates; run after installing the package with pip -e ."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from phasegraph import PhaseGraphFastWeightCell
from phasegraph.baselines import DeltaRuleMemory
from phasegraph.benchmark import make_stream, read_all, untouched_drift


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(Path(__file__).resolve().parents[1] /
                         'results/sequential_v1/seed41/phase.pt'))
    parser.add_argument('--rewrites', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.rewrites < 1:
        parser.error('--rewrites must be positive')
    if not Path(args.checkpoint).is_file():
        parser.error('checkpoint missing; run the benchmark and pass --checkpoint path/to/phase.pt')
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cell = PhaseGraphFastWeightCell(**saved['config']).eval()
    cell.load_state_dict(saved['model'])
    slots, values = cell.query_size, cell.output_size
    if cell.input_size != slots + values:
        parser.error('this demo needs a latest-value benchmark checkpoint')
    delta = DeltaRuleMemory(slots, values)
    data = make_stream(1, args.rewrites, args.seed, slots=slots, value_size=values)
    phase_state, delta_state, before = None, None, None
    print(f"Frozen {cell.kind} cell: {cell.state_bytes()} persistent state bytes; "
          f"{cell.parameter_bytes():,} shared parameter bytes.")
    print(f"Analytic delta control: {delta.state_bytes()} state bytes, explicit orthogonal keys.")
    print('Training-result limitations are recorded in results/sequential_v1/REPORT.md.\n')
    print('rewrite  changed slot   cell MSE   untouched drift   exact-delta MSE')
    for time, observation in enumerate(data['inputs'].unbind(1)):
        phase_state = cell.write(observation, phase_state)
        delta_state = delta.write(observation[:, :slots], observation[:, slots:], delta_state)
        prediction = read_all(cell, phase_state, slots)
        target, written = data['targets'][:, time], data['slots'][:, time]
        if time >= slots:
            rewrite = time - slots + 1
            drift = untouched_drift(before, prediction, written)
            print(f'{rewrite:7d} {int(written[0]):13d} '
                  f'{float(F.mse_loss(prediction, target)):10.5f} {drift:17.6f} '
                  f'{float(F.mse_loss(delta_state, target)):17.8f}')
        before = prediction
    print('\nFinal packed shape:', tuple(phase_state.shape), 'dtype:', phase_state.dtype)
    print('Latest true values versus cell predictions:')
    for slot in range(slots):
        truth = ', '.join(f'{v:+.3f}' for v in target[0, slot].tolist())
        guessed = ', '.join(f'{v:+.3f}' for v in prediction[0, slot].tolist())
        print(f'  slot {slot}: true [{truth}]  read [{guessed}]')


if __name__ == '__main__':
    main()
