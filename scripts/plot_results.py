"""Generate a precise scientific figure from a completed benchmark JSON."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    summary = args.results if args.results.is_file() else args.results / 'summary.json'
    result = json.loads(summary.read_text(encoding='utf-8'))
    colors = dict(phase='#356bd8', signed='#dc7734', vector='#168d81', delta='#38434f')
    labels = dict(phase='Phase graph', signed='Signed graph', vector='Direct vector', delta='Keyed delta')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    fig.set_facecolor('#fafbfc')
    for kind in colors:
        trajectories = [run['models'][kind]['trajectory'] for run in result['runs']]
        x = np.array([point['rewrites'] for point in trajectories[0]])
        for ax, metric in zip(axes, ('mse', 'untouched_drift_mse')):
            values = np.array([[point[metric] for point in trajectory] for trajectory in trajectories])
            mean = values.mean(axis=0)
            # A running mean makes the drift panel match the reported horizon metric.
            if metric == 'untouched_drift_mse':
                values = values.cumsum(axis=1) / np.arange(1, len(x) + 1)
                mean = values.mean(axis=0)
            ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=colors[kind], alpha=.10)
            ax.plot(x, mean, label=labels[kind], color=colors[kind], linewidth=2,
                    linestyle='--' if kind == 'delta' else '-')
    cfg = result['config']
    for ax in axes:
        ax.axvline(cfg['train_rewrites'], color='#a0a6b0', linestyle=':', linewidth=1)
        ax.set_xlabel('Rewrites after initializing all four slots')
        ax.set_xlim(1, max(cfg['horizons']))
        ax.set_ylim(bottom=-.005 if ax is axes[0] else -.0005)
        ax.set_xticks(cfg['horizons'])
        ax.grid(axis='y', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('Latest-value recall (lower is better)', loc='left', fontweight='bold')
    axes[0].set_ylabel('Mean squared error')
    axes[0].axhline(1 / 3, color='#999999', linewidth=1, linestyle=':')
    axes[0].text(max(cfg['horizons']) - 1, 1 / 3 + .003, 'Zero predictor: expected 1/3',
                 ha='right', fontsize=8, color='#666666')
    axes[1].set_title('Untouched-slot drift (read with recall)', loc='left', fontweight='bold')
    axes[1].set_ylabel('Cumulative mean squared prediction change')
    axes[1].legend(frameon=False, fontsize=9, loc='upper right')
    state_bytes = result['runs'][0]['models']['phase']['state_bytes']
    fig.suptitle(f"{state_bytes}-byte sequential memory · {len(cfg['seeds'])} seeds · "
                 f"{cfg['steps']} training steps\nBands show seed range; dotted vertical line is training horizon",
                 fontsize=12, fontweight='bold')
    output = summary.parent / 'comparison.png'
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
