# Gate 2 — Error-driven local phase writes, not magic 896-byte memory

October 14, 2026. **Completed first falsification**, preserving the original PR #1 results.

## Why the original 48-byte learned cell failed

The original task repeatedly rewrites *four independently addressed 3-vectors*, with 1–16 rewrites seen during training and up to 64 at test time. The 48-byte learned phase/signed/vector state gave ~0.30 final MSE, but a 48-byte orthogonal-key delta matrix stored the 12 float32 values exactly. This does **not** imply a physical lower bound or prove that 48 bytes makes multi-path interference impossible.

A 4×4 lattice has **9 independent faces per channel**; the original two-channel 48-byte graph has 18 plaquette cycles. An 8×8 lattice has 49 faces per channel and 392 over eight channels, but those cycles were never shown to be the *learned* storage mechanism in GraphNN. Increasing graph size alone may improve expressivity or make optimization harder; it does not assign independent keys automatically.

The short/long-horizon mismatch is real, and the global writer has no explicit rule enforcing a no-op on an already-correct observation. Our new code tests each proposed repair independently rather than calling the original failure solved.

## Two different mechanisms in this PR

1. **`IndexedHolonomyDelta`** (positive control, *not learned*): 8-bit edge states on non-overlapping lattice plaquettes. Key+component indices are fixed. The writer measures the existing oriented loop-phase code and rotates one edge by the quantized discrepancy needed to install the new value. Repeat a value already matched to half a quantization bin → zero changed bytes. Neighboring indexed faces remain unchanged; integer vertex gauge rephasing leaves loop flux and memory intact. This is an exact addressed byte table in holonomy coordinates. It is not general memory compression, true wave propagation, nor evidence that 8×8 geometry is needed.

2. **`AdjointPhaseDeltaWriter`** (mechanism under test): wraps the existing learned `PhaseGraphFastWeightCell.read` without changing weights. It probes the current phase state and computes `gradient = J(q,state)^T (prediction-target)` by autograd. It picks the top-k phase coordinates by sensitivity, tests quantized phase rotations via a backtracking discrete line search, and commits a proposal *only if* current-key error plus drift penalty to other keys falls. It uses no persistent shadow memory, but each write needs autograd and multiple reader evaluations: this is an **expensive online optimization** experiment, not a fused fast-weight kernel. A repeated target cannot be an exact no-op unless the reader currently predicts it within tolerance; the original reader is often inaccurate. The failed weak-reader experiment below matters.

3. **`phasegraph/horizon_train.py`** (separate training fix): extends the original global-writer phase/signed/vector comparisons to a sampled 1,4,16,64-rewrite curriculum and truncated BPTT with supervision along the trajectory. The original phase/signed/vector reader must be retrained for this protocol. Different optimizer-update count and compute mean PR #1's 600-step run is not a rigorously compute-matched control. These are *training utilities*, **not** proven success at long horizons until training results are produced.

## Frozen three-seed measured control: 128 streams/seed; 64 rewrites

Source: [successful GitHub Actions test + 3-seed experiment](https://github.com/anttiluode/PhaseGraphFastWeightCell/actions/runs/38595055296) · [machine-readable result](pilot.json).

| Model | Persistent state | Mean 64-rewrite MSE | Unchanged-input bytes modified | Mean drift of other keys |
|---|---:|---:|---:|---:|
| Loop delta, 4×4, 3 channels | **72 bytes** | 0.00002016 | **0%** | 0 |
| Loop delta, 8×8, 8 channels | 896 bytes | 0.00002016 | **0%** | 0 |
| Explicit orthogonal float32 delta | **48 bytes** | 0 | 0% | 0 |
| Previous PR #1 learned phase writer, trained only through 16 rewrites | 48 bytes | 0.303904 | not comparable | not comparable |

The loop phase errors are exactly the expected 8-bit quantization error, about **2×10⁻⁵ MSE**. But using twelve independently addressed 8-bit values makes an even smaller **12-byte indexed table** possible, so there is **no phase-storage efficiency victory**. Only twelve of the phase graph's 72 or 896 bytes carry independent target values in this control; other bytes store zero or optional gauge rephasing.

**Frozen-reader adjoint pilot:** only two held-out episodes per seed and four rewrites (a smoke-sized falsification, not a power study). PR1 reader MSE → adjoint MSE:
- seed 41: 0.27803 → **0.28482** (worse)
- seed 97: 0.36327 → **0.36373** (slightly worse)
- seed 127: 0.31547 → **0.31054** (slightly better)

Thus **the proposed generic Jacobian delta rule has not repaired the original learned representation**; a poor reader sends a poor correction signal. Do not cite the exact-address positive control as evidence that the learned phase-graph has become good at remembering 64 unknown-key updates.

## Reproduction

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m phasegraph.local_delta_benchmark --seeds 41 97 127 --eval-count 128 --horizons 1 4 16 64 --output runs/local_delta/summary.json
python -m phasegraph.horizon_train --steps 600 --seeds 41 97 127 --kinds phase signed vector --train-horizons 1 4 16 64 --segment 8 --output runs/horizon_matched
```

For the spatial 896-byte experiment, use `--grid-size 8 --channels 8` in the horizon trainer. It has far more trainable parameters than the 48-byte version; compare parameters, FLOPs and runtime separately. CPU at 64 rewrites may be costly; a single RTX 3060 can train sequentially but results are not yet on record.

## What would validate the phase-graph hypothesis?

Not only achieving near-zero MSE with known-key slots. Need a **trained**, fixed-byte, locally updated graph where one observation determines a useful future response under changing and possibly ambiguous query contexts; success must beat same-state, same-compute signed and vector cells and a strong keyed memory baseline. Report consistency across independent training seeds, many rewrites, nontrivial relatings between objects, and held-out interventions where graph paths/loop phases causally matter.
