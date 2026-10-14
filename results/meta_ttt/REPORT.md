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

## Completed 600-episode three-seed test — scientific result is negative

The [full GitHub Actions job succeeded across all seeds](https://github.com/anttiluode/PhaseGraphFastWeightCell/actions/runs/38600274002) and uploaded checkpoints. [Full machine-readable per-seed scores](full600.json).

The same physical **48-byte** state, **12,867 shared trained parameters**, **top-8 quantized write**, **600 episodes per seed**, **batch=8**, **1/4/16/64 training horizons** and **eight-step TBPTT** were used for meta-versus-first-order. Unlike the original global writer, both models deleted the independent writer MLP and need state autograd for every inference write.

| Held-out MSE (mean of seed 41,97,127) | Second-order phase meta | First-order phase control | Earlier global phase writer |
| --- | ---: | ---: | ---: |
| 1 rewrite | **0.851117** | **0.091192** | 0.130920 |
| 4 rewrites | 0.696384 | **0.120270** | 0.123657 |
| 16 rewrites | 0.237853 | 0.165776 | **0.122121** |
| 64 rewrites | 0.185918 | 0.192249 | **0.119708** |

**Seed-level 64-step meta vs first-order:** 41 0.189195 vs 0.192310; 97 0.199767 vs 0.203900; 127 0.168793 vs 0.180536. Thus the second-order write *slightly* improves long-horizon recall over its own control in all seeds, but **the original independent global writer remains better** on the same task. The comparison to that earlier 600-episode global writer is not parameter/compute matched: the global model has **25,763 shared parameters**, uses batch=32, and its CUDA optimization employed ~2,100 updates/seed. This new model has 12,867 and batch=8.

**Severe failure:** Meta-trained phase loses short-horizon calibration. Seed 41 one-rewrite MSE was **1.786638** and seed 127 **0.484247**, compared with **0.083922 / 0.097547** for the first-order control. Such collapse invalidates a claim of useful general-purpose fast-weight memory, regardless of the modest 64-step meta benefit.

**Cross-talk finding:** Final-state normalized off-key `J_B J_A^T` overlap was **0.189951** for meta and **0.208803** for first-order (only the final-64-state kernel was measured; the same number is currently copied into each stored horizon row). However, *actual untouched-key drift* at 64 rewrites is **0.012198 for meta vs 0.007378 for first-order**, again worse for the model with slightly more orthogonal reader derivatives. Its Jacobian self-norm is **49.84** versus **22.15**, indicating significant sensitivity scale changes not captured by normalized overlap.

**No-op failure:** The exact repeated-input test modified all eight allowed byte locations for both arms (`8/48 = 16.67%`). The error threshold is only reached when a value is already well predicted; the learned reader does not attain it. Even at 600 episodes, the feedback writer is not stable or idempotent under repeated values.

### Follow-up that can actually falsify a mechanism

Test a **quantized, loss-monotonic accepted inner update** (backtracking among 8/4/2/1 phase-bin steps and commit only if current-key error drops), trained with exactly the same reader and first-order controls. This distinguishes gradient overshoot/quantization from fundamentally entangled read geometry. Sweep step size separately, and audit the apparent increase in Jacobian self-norm and loss of short-horizon calibration. Repeat with actual matched batch and parameter budgets before any claim of phase advantage.

The original claim that meta-training *must* orthogonalize query Jacobians is refuted as a mathematical necessity. The observed reduction in normalized cross-kernel alone was not sufficient to protect old readouts, and training through the inner update did not solve the long-horizon task competitively.
