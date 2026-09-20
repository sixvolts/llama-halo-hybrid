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
The dominant cost is per-split graph re-serialisation: `GRAPH_RECOMPUTE` can never fire because
`ggml_backend_sched_split_graph` regenerates every split's uid on every call. See run_glm_two_host_v2.sh.
