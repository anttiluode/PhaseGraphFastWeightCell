# Phase Graph Cell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver a byte-packed recurrent phase-graph cell with honest sequential-memory comparisons.
**Architecture:** A state-conditioned learned writer rotates quantized edge phases. A normalized Hermitian graph transports fresh query activity to a shared decoder; a sequence wrapper carries only the edge state.
**Tech Stack:** Python >=3.10, PyTorch >=2.5, pytest, NumPy, Matplotlib for recorded results.
**Spec:** `docs/superpowers/specs/2026-10-14-phase-cell-design.md`

## Global Constraints
- Python >=3.10; torch >=2.5; CPU support required, CUDA optional.
- Default grid 4, channels 2: 48 bytes per stream.
- No source observations or activation history persist at inference.
- Quantization happens at every write, including training forward passes.
- Preserve the existing MIT license and credit GraphNN.

## Review Focus
- Circular wraparound, distinct bins, and repacking cannot introduce an off-by-one angle.
- Per-stream masks and reset must survive chunk boundaries and preserve no-write bytes exactly.
- Nonfinite data, malformed state, wrong devices and unsupported dtypes fail clearly.
- Controls use the same writer/decoder budgets and held-out data without replaying forgotten values.
- Training-state graphs must preserve gradients while evaluation returns actual byte tensors.

### Task 1: State, transport, and recurrent cell
**Files:** `phasegraph/cell.py`, `phasegraph/quantization.py`, `phasegraph/__init__.py`, `tests/test_cell.py`, `pyproject.toml`.
**Interfaces:** produces the public cell API in the spec; inputs are floating tensors and optional [B,C,E] state.
- [x] Write tests for circular bins, gradients, byte accounting, Hermitian contraction and cell masks/validation.
- [x] Run pytest; observe missing implementation failures.
- [x] Implement circular STE quantization, symmetric lattice transport, learned write and query/read.
- [x] Run cell tests and commit passing core.

### Task 2: Sequence wrapper and controls
**Files:** `phasegraph/recurrent.py`, `phasegraph/baselines.py`, `tests/test_recurrent.py`.
**Interfaces:** consumes cell `forward`, `initial_state`; produces `PhaseGraphRNN.forward(x, state=None, queries=None, write_mask=None, reset_mask=None)` and keyed `DeltaRuleMemory`.
- [x] Write tests for step/chunk equality, empty sequences, checkpoint roundtrip, matched parameters and exact keyed delta retention.
- [x] Run tests and observe missing-interface failures.
- [x] Implement the wrapper, signed/vector variants through the common cell, and analytic delta memory.
- [x] Run the complete suite and commit.

### Task 3: Frozen sequential-memory experiment
**Files:** `phasegraph/benchmark.py`, `tests/test_benchmark.py`, `results/`.
**Interfaces:** consumes common cell and delta APIs; produces CLI JSON, plot and report with config, per-seed metrics, controls and bytes.
- [x] Write tests for causal stream targets, independent seeded data, metric drift and benchmark smoke output.
- [x] Observe failures, implement deterministic streams and training/evaluation CLIs.
- [x] Run the suite and a CLI smoke before the fixed multi-seed experiment.
- [x] Record results at 1/4/16/64 rewrites, including honest failures against predeclared thresholds.

### Task 4: Runnable delivery and verification
**Files:** `README.md`, `docs/MECHANISM.md`, `examples/streaming_demo.py`, `.github/workflows/tests.yml`, `.gitignore`.
**Interfaces:** consumes saved checkpoint and cell APIs; documents installation, reuse, reproduction and limits.
- [x] Run install, examples and full suite; fix defects with regression tests.
- [x] Review the complete branch against spec and numerical claims.
- [ ] Push verified code/results and check remote head.
