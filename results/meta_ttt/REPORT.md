# Gate 3 — learn the reader *through* an error-gradient phase writer

**Date:** October 14, 2026. **Code + CI unit/smoke verified.** [Predeclared protocol](../../docs/META_TTT_PROTOCOL.md) · [full 128-episode JSON](pilot128.json).

The algorithm deletes the old independent writer and computes each state update from a genuine state derivative of the **reader's own prediction error**. The 48-byte graph state is quantized to uint8 after each write, and only the top **8 / 48** edge bytes may change per observation. Reader parameters are updated through *second-order* differentiation of the state-gradient inner write. A first-order control makes the **same numerical inner writes** but stops higher-order differentiation through that Jacobian.

## First complete pilot: 128 training episodes / seed

Three independent seeds (41,97,127), training and held-out evaluation at 1/4/16/64 rewrites, 64 held-out streams/seed. Training batch=8, segment=8, reader-only parameters **12,867** per arm, the same underlying initialization and samples for each seed.

| After 64 rewrites | Second-order `phase_meta` | First-order `phase_first_order` |
| --- | ---: | ---: |
| Seed 41 MSE | **0.196211** | 0.234384 |
| Seed 97 MSE | **0.215583** | 0.244947 |
| Seed 127 MSE | **0.224201** | 0.236647 |
| Three-seed mean MSE | **0.211999** | 0.238659 |
| Newest-written-key MSE | **0.157728** | 0.215347 |
| Normalized off-diagonal `J_B J_A^T` overlap | **0.301755** | 0.384774 |
| **Untouched-key response drift** | **0.020787** | **0.011827** |
| Fraction of state bytes modified per write | 16.67% | 16.67% |
| Fraction changed after immediately rewriting same input | 16.67% | 16.67% |

[**Measured GitHub Actions run and per-seed checkpoints**](https://github.com/anttiluode/PhaseGraphFastWeightCell/actions/runs/38599831181).

## Interpretation

**Positive:** Differentiating through the actual `J_state^T e` byte writer helps the reader learn on this exploratory 128-episode sample. Overall recall MSE is lower across all three trained seeds and off-key state-Jacobian overlap is lower in every seed. The diagnostic supports gradient coordination, not *exact* orthogonal Jacobians.

**Important negatives:** This is weaker than the earlier user's **600-episode global-writer phase benchmark mean MSE 0.119708**. The training budget is not matched (128 versus 600 episodes, batch=8 versus 32). Also, although the Jacobian overlap gets lower, **actual unintended output drift for untouched keys gets WORSE**, from 0.011827 to 0.020787. That directly refutes the simplistic interpretation that lowering a normalized `J_BJ_A^T` score is enough to protect memory. Every immediately repeated input in the held-out pilot still changed all eight allowed bytes because the reader was not accurate enough to trigger the conditional no-op threshold. A generic error-gradient update is *not* automatically idempotent.

No exact key/value retrieval, loop-holonomy storage, phase computer superiority or Transformer replacement is demonstrated. The online autograd writer is much more costly than the original learned MLP writer, despite its small persistent state.

## Next controls

- A fully budgeted 600-episode meta vs first-order comparison at **identical hyperparameters** to each other, reported separately from the 128-episode exploratory pilot.
- Error-decreasing discrete line search / update acceptance versus the present fixed-eight-bin write (to avoid overshooting).
- Fair computational comparisons to the user's signed/vector global-writer benchmarks and analytic keyed delta matrix.
- Test whether the cross-key Gram score decrease survives outside the original one-hot key distribution.

### Reproduce locally

```bat
python3.13 -m pytest -q tests/test_meta_ttt.py
python3.13 -m phasegraph.meta_train --steps 600 --seeds 41 97 127 --arms phase_meta phase_first_order signed_meta vector_meta --train-horizons 1 4 16 64 --eval-horizons 1 4 16 64 --segment 8 --device cuda --batch-size 16 --output runs/meta_ttt
```

This uses higher-order derivatives and can require much more GPU memory/compute than the older global writer. If resource-limited, reduce `--batch-size` first. The 48-byte budget describes *persistent stream state*, not optimizer/autograd memory.
