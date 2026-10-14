# Phase-graph fast-weight cell

Build a reusable PyTorch recurrent memory from GraphNN's phase-only idea. The user asked for an implementation in this repository, with evidence that it can serve as a stateful component. This is a new subsystem. Work proceeds under the user's instruction to build it.

## Scope and choices

A learned writer changes the phases on a fixed square lattice; a query launches fresh complex activity through the resulting Hermitian couplings. A shared decoder reads the transported field. Choose this sparse graph rather than a dense phase matrix (more work per state byte) or a direct copy of the image CNN (image-specific interface). Default grid 4, channels 2: 48 bytes per stream. Grid 8, channels 8 reproduces the upstream 896-byte phase budget. No source observations or activation history persist at inference.

Expose `PhaseGraphFastWeightCell(input_size, output_size, query_size=None, grid_size=4, channels=2, width=64, transport_steps=4, kind="phase")` with `write`, `read`, `forward`, `initial_state`, `pack_state`, `unpack_state`, `state_bytes`, and `parameter_bytes`. `forward(x, state=None, query=None, write_mask=None, reset_mask=None)` writes then reads, returning output and state. State is [batch, channels, 2*grid*(grid-1)]. Floating training state uses straight-through byte rounding; evaluation state is actual uint8. Explicit masks support no-op reads and per-stream reset. Calls are functional and do not mutate supplied state. Missing query uses x only if query_size == input_size; otherwise it is an error.

`PhaseGraphRNN` consumes batch-first sequences, including optional queries and masks, and returns all outputs plus final state. Its chunked and one-step executions must agree. The output decoder receives memory and query only. Models must be retrained around the cell; existing Mamba checkpoint weights are not interchangeable.

## Mathematical contract

Phase byte b is angle 2*pi*b/256, with 256 distinct circular bins. Updates are learned bounded rotations with circular wrapping, not clipping at the seam. Quantization happens at every write, including training forward passes. Fixed-magnitude edges alone do not bound propagation. Use symmetric degree normalization: A = D^(-1/2) W D^(-1/2); T = (I+A)/2. Unit-magnitude Hermitian edges imply spectrum(A) in [-1,1], so T is a contraction in the Euclidean field norm. This is dissipative, not unitary. Global phase rotation does not imply separate query addresses; learn spatial query excitation and measure selectivity.

Packed inference has one uint8 tensor, one byte per edge per channel. Floating training activations, parameters, topology buffers, unpacked temporary fields, and checkpoint metadata are counted separately. No hardware speedup is claimed from storage alone.

## Controls and measurements

Train phase, signed-real graph, and direct-vector variants with exactly matched parameter counts, byte budgets, batches, optimizer, and seeds. Signed edges have values in [-1,1] and the same normalized propagation. A keyed delta-rule matrix is an analytic equal-byte positive control on orthogonal keys, not a trained Mamba or DeltaNet implementation. Include that favorable baseline even if it wins.

Task: latest-value recall for four independently updated slots, each holding three continuous values. Inputs give the updated slot and its new value. Start with all slots initialized, train on repeated updates, freeze weights, then score 1/4/16/64 rewrites with independent data. Reader receives a slot query and state only. Report MSE, unchanged-slot readout drift, actual state and parameter bytes, shuffled-state and zero-state controls, per-seed results, runtime, and longer-horizon behavior. This replaces image reconstruction with a stricter recurrent test and does not inherit GraphNN's earlier scores. Predeclared descriptive thresholds: 16-update MSE <= .05, 64-update MSE <= .10; superiority requires better held-out MSE than both trained controls for every seed. Do not retune thresholds after seeing results. The analytic delta baseline is reported separately.

## Validation and delivery

Python >=3.10; torch >=2.5; CPU support required, CUDA optional. Tests cover the circular seam and every bin, actual byte storage, packed/fake-quantized equivalence, gradient flow through time, transport Hermiticity and contraction, learned spatial queries, no-write/reset behavior, checkpoint reload, invalid shapes/devices/nonfinite inputs, chunked execution, and baseline/protocol integrity. Provide an installable package, terminal demo, benchmark CLI, machine-readable results, a readable report, and CI. Preserve the existing MIT license and credit GraphNN at commit 0e3d4ca441aa6a549c5f596dc113eff79a8d21d8. No GUI or pretrained language-model integration is included.
