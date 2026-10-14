# Frozen sequential rewrite results

3 seeds, 600 optimizer steps per model, 128 independent held-out streams per seed. Training uses 1–16 random rewrites after initializing all slots.

| Model | State bytes | Parameters | MSE at 1 | MSE at 4 | MSE at 16 | MSE at 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase | 48 | 25763 | 0.274162 | 0.285411 | 0.294348 | 0.303904 |
| signed | 48 | 25763 | 0.266109 | 0.282849 | 0.287949 | 0.293045 |
| vector | 48 | 25763 | 0.267593 | 0.283255 | 0.287519 | 0.298433 |
| delta | 48 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Predeclared gates

- 16_update_mse_at_most_0.05_all_seeds: **FAIL**
- 64_update_mse_at_most_0.10_all_seeds: **FAIL**
- phase_beats_both_learned_controls_at_16_and_64_all_seeds: **FAIL**

## Interpretation boundaries

These are continuous synthetic latest-value streams, with explicit slot keys. The delta rule is an analytic orthogonal-key positive control, not a trained Mamba/DeltaNet model. All learned arms share parameter counts, state bytes, data, optimizer and step budget. Signed and vector arms use interval clipping; phase uses circular rotations. The writer sees sine/cosine features of the byte codes in every learned arm. The graph arms consume more read compute than the vector arm.

MSE is averaged across all slots and value components. Untouched drift is the squared change in predictions for keys other than the latest written key, averaged over preceding rewrites. Low drift by itself is not useful retention: a constant reader also has zero drift. Shuffled, blank and skipped-update controls and full seed-level scores are in summary.json.

State bytes count actual inference tensors, excluding shared weights and topology buffers (reported separately), temporary float/complex fields, training activations and file metadata. Timing includes diagnostic reads of every slot after every update and control reads; it is descriptive reference-code runtime, not an optimized kernel benchmark.

Bounded bytes and normalized field transport do not guarantee task accuracy, orthogonal query isolation, bounded gradients, or superiority to modern recurrent models. This experiment does not establish a pretrained-model replacement or language-model quality.
