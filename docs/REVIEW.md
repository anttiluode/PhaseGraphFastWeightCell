# Verification and review

An independent whole-branch review checked the cell, normalized transport,
stream wrapper, controls, benchmark causality and recorded results. It
also reloaded the first seed's checkpoints and reproduced every horizon
metric. The negative benchmark result is retained.

The review found three concrete state-contract defects. All were fixed
with failing regression tests before the implementation changes:

1. A common straight-through expression lost floating-point bits when a
   negative phase wrapped around the circle. The new expression returns
   the exact rounded forward value plus an exactly zero residual whose
   surrogate derivative is one. Regression tests cover both float32 and
   float64 and exact 128-step training/packed recurrence agreement.
2. Externally supplied signed floating state could exceed the highest
   physical code, producing an edge magnitude above one. Signed/vector
   states now reject normalized values above 255/256. All actual byte
   codes remain supported.
3. The analytic delta control's initial state followed the process-wide
   default float dtype despite its float32 interface and byte accounting.
   It now explicitly creates float32 state, including when the global
   default is float64.

After correction, all **35 tests pass** on CPU. A million random wrapped
phase proposals produce zero off-grid training forwards. Installation,
checkpoint reload, both README API examples, the terminal demo, CLI smoke,
and a 10,000-write fixed-byte/finite-read check were exercised.

The full three-seed, 600-step experiment was rerun on the corrected source.
Its exact source hashes and local training-checkout revision are recorded
in `results/sequential_v1/summary.json`. The source hashes are the portable
way to identify the measured implementation; publication commits can have
different IDs from the local training checkout. Only the corrected run is
included in the published results.

CUDA and mixed precision were not tested; this reference explicitly
supports float32/float64, and does not claim AMP or optimized kernel
performance. The CI workflow also runs the suite and demo on Python 3.10
and 3.12.
