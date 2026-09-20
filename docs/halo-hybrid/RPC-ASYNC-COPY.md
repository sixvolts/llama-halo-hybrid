# RPC async same-server cross-device copy (proto 7.1)

Commits: 773e1047c (hook + server handler), d52a65cea (server-minor capability gate), 5375a20c3 (drain the
owning backend before every host-side tensor access on the server).

## Why
The RPC backend had `.cpy_tensor_async = NULL`. Every scheduler input copy between two devices on the same
rpc-server hit `ggml_backend_tensor_copy_async`'s fallback: synchronize(src) + synchronize(dst) + a BLOCKING
`RPC_CMD_COPY_TENSOR`. The dispatcher is one socket / one queue / one worker per endpoint, so that blocking send
was head-of-line blocking for both devices. Measured 2026-09-20 on the four-device GLM layout.

## Standing review rule for ggml-rpc.cpp
`copy_tensor_async` is the first server command that replies with device work still in flight. Every other
reply had historically followed a blocking `graph_compute`, so `get_tensor`/`set_tensor`/`set_tensor_hash`/
`memset_tensor` touched device memory with no synchronization of their own - an unstated invariant. The async
copy rides the backend's `cudaStreamNonBlocking` stream; the CUDA buffer get/set/memset use `cudaStreamPerThread`
and sync only that. `rpc_server::sync_backend_for(buffer)` now precedes every host-side data access.
**Any future server command that touches device memory must call `sync_backend_for` or re-derive why it need
not. The code does not enforce this; only review does.** (Raised by the mainframe session.)

## Correctness gate for any RPC wire change
`~/bench/glm/greedy_ref.sh <tag> v3 4`: fixed 1210-token prompt, 160 greedy tokens at temp 0. Old (both hosts
pre-change) vs new (both post-change) must be TOKEN-IDENTICAL. Reference for this change: text_sha
a828e28289899da6. Neither draft acceptance nor a numerics check can see an RPC ordering fault.

## What it bought
+1% prefill, +2-3% decode on v3. Correct, small, kept. The blocking copy was ~8% of the per-crossing cost.

## Graph re-serialisation is NOT the residual (measured 2026-09-20, falsified)
`GRAPH_RECOMPUTE` can never fire (`ggml_backend_sched_split_graph` regenerates every split's uid on every
call), so every split re-sends its graph each ubatch. The candidate "fix #2" (stable split identity +
per-identity stored graphs, a second wire change) was gated on the server-side deserialise cost being ~17 ms.
Mainframe's rpc-server was instrumented locally (uncommitted, `GGML_RPC_TIMING=1`, per-call recv/deserialise/
compute/reply on 5375a20c3, wire untouched) and run under v3 KM=5 ub1024, 3275 graph_compute calls on the
R9700 device and 2889 on the APU device:

| dev | avg nodes | avg tensors | avg bytes | ms recv | ms deserialise | ms compute | ms reply |
|-----|-----------|-------------|-----------|---------|----------------|------------|----------|
| RPC0 (R9700) | 169 | 1828 | 542 K | 0.228 | 0.195 | 5.72 | 0.020 |
| RPC1 (APU)   |  21 |   28 |   8 K | 0.021 | 0.004 | 7.72 | 0.021 |

- deserialise is 0.16-0.21 ms across the whole run while compute swings 1.4-12.2 ms (RPC0) / 1.9-15.9 ms (RPC1)
  with the prefill/decode phases; the graph message is pure metadata (~300 B/tensor, constant), so it cannot
  scale with ub. Marshalling + transport on the server is < 0.45 ms per call. Fix #2 is dead; do not build it.
- recv is bandwidth-shaped: ~18 us fixed + payload at ~2.65 GB/s.
- Where the ~19 ms per-crossing cost (from the -ub two-point fit) actually lives, given the server accounts for
  ~4 ms of it at decode shape: the client side (serialize_graph, the single-worker dispatcher queue, the
  scheduler's per-split synchronize) and the structural serialisation - the dispatcher waits for each
  GRAPH_COMPUTE reply and the server is SINGLE-THREADED (`rpc_serve_client` runs inline from the accept loop,
  one client at a time), so dev1's graph cannot start until dev0's compute has finished. The per-crossing cost
  is largely "the other device's compute"; no transport change touches that. Next measurement is client-side.
