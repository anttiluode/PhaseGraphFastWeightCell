# Mechanism and contracts

For each channel, an undirected G × G lattice has E = 2G(G−1) stored
edges. The persistent tensor is `[batch, channels, edges]`; horizontal
edges precede vertical edges, both in row-major order. Channels do not
share edges. Shared writer/readout parameters mix their features.

## Circular update and quantization

Let u = b/256 be a normalized phase code. The writer sees the current
observation x and the old code's sine/cosine features:

```
(r, a) = MLP(concat(x, sin(2πu), cos(2πu)))
u_new  = Qcircle(u + 0.25 tanh(r) sigmoid(a))
Qcircle(z) = (round(256 (z mod 1)) mod 256) / 256
```

The forward state is rounded on every write. Training substitutes the
identity derivative for the discontinuous quantizer; this surrogate is
not the exact derivative of byte rounding. Inference returns `uint8`.
Signed and vector controls use the same parameters/features/step scale,
but clip to `[0,255/256]` and round instead of wrapping. Therefore the
comparison includes the update geometry as well as the read semantics;
it does not isolate complex arithmetic alone.

No auxiliary amplitude, residual-quantization accumulator, source input,
previous output or recurrent activation persists. Shared model weights
and fixed normalization buffers are additional storage. During training,
autograd retains intermediate graphs; training memory is not one byte
per edge.

## Fresh query and graph read

A query q creates fresh complex activity `z0 = tanh(Pq)_real + i*tanh(Pq)_imag`,
reshaped over all channel/node positions. This is a spatial query pattern.
It is not a single global rotation: in a linear complex graph a global
rotation commutes with propagation and cannot by itself produce protected
independent memory addresses.

An edge's forward coefficient is `exp(i*2πu)` and its reverse coefficient
is its complex conjugate. With D the fixed lattice degree matrix:

```
A(u) = D^(-1/2) W(u) D^(-1/2)
z(k+1) = (z(k) + A(u)z(k)) / 2
```

For every complex vector z, the absolute quadratic form of A is bounded
by `sum_edges (|z_i|²/d_i + |z_j|²/d_j) = ||z||²`, using the triangle
inequality and `2ab <= a²+b²`. A is Hermitian, so its eigenvalues lie
in [-1,1]; the hop's eigenvalues lie in [0,1]. Every hop contracts the
Euclidean field norm. The same bound holds for signed-real edge weights
with absolute values at most one.

The decoder receives real/imaginary features at depths 0, K//2 (at least
1), and K. It receives no image or observation bypass. The direct-vector
control receives the same-sized query features plus padded sine/cosine
features of the edge byte planes, with exactly the same decoder size.

This is a dissipative graph filter. Unit-magnitude edge coefficients do
not make the full operator unitary, and destructive interference can
attenuate a response. The field bound is not a bound on learned decoder
weights, output magnitudes or training Jacobians.

## Lifecycle

`initial_state` uses code 128 for every edge. `write` is functional:
reset selected streams to code 128, compute a proposed update, apply the
write mask, then return a new state. A read-only step can call `read`
directly or use a false write mask. `PhaseGraphRNN` threads the state
through steps with the exact cell implementation; no extra hidden state
is added by the wrapper. Per-step and chunked evaluations agree.

For truncated BPTT, detach a floating state at your chosen boundary.
For inference snapshots, save model config, model state_dict, and the
packed state. Reloading requires no observations. The model weights
must accompany the state because phases are a learned code.

## Integration limits

The module exposes recurrent-cell and batch-first mixer interfaces.
Training a network around it is required. Mamba's pretrained projections,
state layout and optimized selective-scan recurrence are not compatible
with this state representation. This version has no fused byte/complex
kernel, parallel associative scan, convolution cache, AMP support or
external-memory index. Vector I/O is generic; performance evidence is
limited to the recorded synthetic benchmark.
