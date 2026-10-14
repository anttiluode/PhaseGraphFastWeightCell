# Gate 3 protocol — meta-train a phase reader through its own error-gradient writer

**Implementation:** [`phasegraph/meta_ttt.py`](../phasegraph/meta_ttt.py) and [`phasegraph/meta_train.py`](../phasegraph/meta_train.py). This protocol specifies the test before any full 600-episode Gate 3 model exists. The experiment follows the user's **completed CUDA PR2 horizon-matched 600-step run**.

## What we now know from the completed user CUDA run

User ran 600 episodes for three trained seeds **41, 97, 127** with **1, 4, 16, 64** rewrites in the curriculum and eight-step truncated BPTT, on 48 physical state bytes. This uses ~2120–2190 optimizer steps per model (not the original 600); longer histories **and training compute both increased** relative to PR1.

| Arm (48 physical bytes) | 64-rewrite MSE: seed 41 | 97 | 127 | mean |
| --- | ---: | ---: | ---: | ---: |
| Original global phase writer, trained through 64 | 0.176137 | 0.107905 | 0.075081 | **0.119708** |
| Signed-real graph, trained through 64 | 0.139769 | 0.142367 | 0.107328 | **0.129821** |
| Direct-byte vector, trained through 64 | 0.296215 | 0.295319 | 0.288157 | **0.293230** |
| PR1 original phase writer, trained only through 16 | — | — | — | 0.303904 |

Horizon-matched phase is now much better than PR1's mean-guessing-like result, and beats signed on **2/3** seeds. This does not show its phase representation is superior to signed real as a mechanism, or that the phase graph has met PR1's 0.10 gate in every seed. [Exact committed result](../results/horizon_matched/summary.json).

## New mechanism

The original independent **learned MLP writer is removed**, so the only model parameters live in the query projection and readout decoder. At each observation `key, target`:

1. Read `prediction = R_theta(state, key)`.
2. Compute the exact state gradient `g = d/dstate ||prediction-target||^2`.
3. Scale by a detached per-example gradient RMS and optionally retain the `topk` largest-magnitude coordinates.
4. Update `state := Q_uint8(state - (step_bins/256) * g / RMS(g))`, circular for complex phase, interval clipping for signed/vector.
5. If the reader already predicts the target within tolerance, *do not update the state*.

**This is not the analytic indexed plaquette writer**. The reader/geometry and its gradient are learned. The writer depends on the reader’s local derivative, so it cannot have an independently chosen write address. However, the reader itself can still be inaccurate, latent codes can collide, and gradient updates can cross-talk with other keys.

Outer-loop training differentiates through the **state gradient calculation** (`create_graph=True`) and quantization STE, then supervises the **readout of all keys after every truncated-BPTT segment** across 1,4,16,64 rewrite episodes.

## Required controls and budget

| Arm | Persistent bytes | Inner write | Reader training |
| --- | ---: | --- | --- |
| `phase_meta` | 48 | gradient via phase graph | second-order, through J^T e |
| `phase_first_order` | 48 | **same forward inner updates** | first-order stop-grad through J^T e |
| `signed_meta` | 48 | gradient via signed-real graph | second-order |
| `vector_meta` | 48 | gradient via direct-byte reader | second-order |
| `original_global_phase` | 48 | independently learned MLP writer | original user 600-episode checkpoint |
| `orthogonal_delta` | 48 | privileged exact one-hot keys | zero-parameter analytic lookup |

The models receive the **same one-hot key and new 3-vector target** for each write. Only the new models use internal **online autograd of the fast state at inference**. That imposes substantial *unmatched compute*. The older global-writer models also have extra shared writer parameters, so parameter count is explicitly reported.

Compare each model on held-out streams at **1/4/16/64** rewrite horizons, no other keys replayed. Log MSE, newest-key error, untouched-key output drift, byte changes including unchanged-input sham writes, and a measured normalized off-key `J_A J_B^T` Gram score on exactly the same memory snapshots. Do not call this score "orthogonality" merely because a loss decreases.

### Acceptance

Evidence that meta-training improves the proposed mechanism requires `phase_meta` beating **phase_first_order** on held-out 16- and 64-rewrite recall across independent training seeds, with no collapse in short-horizon performance and less interference than signed/vector at matched bytes. A compelling architectural advantage would also need fairer compute/parameter budgets, natural or relational data, and learned key assignment. A full gate failure should be reported plainly.

## Scientific nuances

- `J_B J_A^T=0` guarantees **zero first-order cross-talk** along an infinitesimal update `J_A^T e_A`; it does **not** guarantee zero finite-byte or 64-step cross-talk. Quantization, nonlinear state dependence, normalization, and discrete top-k selection matter.
- The off-key state-Jacobian Gram resembles a tangent kernel but is *not* automatically the conventional parameter neural tangent kernel (NTK).
- Minimizing multi-step retention loss **does not mathematically force** Jacobians to orthogonalize. Many other representations and nonlinear compensating mechanisms can satisfy the same training objective.
- The update is **not** a fast "test-time training" operator computationally: every inference write runs autograd and graph transport, including float/complex temporaries. Only the persistent stream state is 48 uint8 bytes.
- The phase graph's `_hop` is dissipative, not unitary. No gauge-invariance or holonomy mechanism is enforced in this reader.
- The original task's four keys are **explicit and orthogonal**, so the exact 48-byte delta rule is still an unbeatable positive control.

## Reproduction

Install PyTorch/pytest as in the README; CUDA on user's 3060:

```bat
python3.13 -m phasegraph.meta_train --steps 600 --seeds 41 97 127 --arms phase_meta phase_first_order signed_meta vector_meta --train-horizons 1 4 16 64 --eval-horizons 1 4 16 64 --segment 8 --device cuda --batch-size 16 --output runs/meta_ttt
```

Expect higher computation and memory cost than the original global writer because second-order differentiation is needed. For a small CPU diagnostic:

```bat
python3.13 -m pytest -q tests/test_meta_ttt.py
python3.13 -m phasegraph.meta_train --steps 8 --seeds 41 --train-horizons 1 4 16 --eval-horizons 1 4 16 --batch-size 4 --eval-count 8 --width 32 --output runs/meta_smoke
```

The `Meta-trained phase Jacobian writer` GitHub Actions workflow runs the smoke and exposes a manual **three-seed full experiment** with saved checkpoints and JSON receipts. Only claim trained performance from **completed** artifacts; do not present a smoke test as evidence of 64-step retention.
