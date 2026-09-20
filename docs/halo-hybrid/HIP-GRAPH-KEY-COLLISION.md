# HIP-graph cache key collision on the rpc-server (2026-09-20)

## Symptom
Any layout that sends structurally identical per-layer splits to one rpc-server device loses decode throughput with no
visible cause: v3 (dense of 21 remote layers on mainframe's R9700, ~21 splits per token to that device) decodes at 18.4 t/s
against v1's 21.0 with the card idle, while v2c (6 WHOLE layers on the card, one graph per token) decodes at 21.9.
Mainframe's per-call server timing showed dev0 spending 21 us per node at decode shape - launch-bound, not replay.

## Cause
`ggml_cuda_graph_get_key` (ggml-cuda.cu) keys the per-context HIP-graph cache on the ADDRESS of `cgraph->nodes[0]`,
`n_nodes`, and the `ne` of the first and last node. On a local backend the sched's split nodes have distinct addresses.
The rpc-server rebuilds every incoming graph in `stored_graphs[device].buffer`, one persistent arena per device, from
offset zero, so two splits with the same structure put nodes[0] at the same address and collide onto one cache entry.
With ~21 layers rotating through one entry, `ggml_cuda_graph_update_required` sees the previous layer's node properties
on every call, `warmup_complete` is never set, and every split executes directly. Both log messages ("warmup complete",
"warmup reset") fire only on TRANSITIONS, so a permanent collision logs nothing - event counts cannot detect it.

Two things that are NOT the cause, checked: KV views only change every 256 positions (`llama-kv-cache.cpp` pads n_kv
to 256), and the persistent arena makes the node_props memcmp stable for an unchanged graph (src pointers repeat).

## Fix
Mix the first and last node's `op` and `name` into the key. llama names nodes with the layer index (`attn_norm-25`), so
identical layers separate; local backends already had distinct addresses so it is a no-op there; a residual collision
degrades to today's behaviour. Server-local, no wire change, `git checkout` to revert. Gate: greedy token identity
(greedy_ref.sh) on gibson's local backend, then a v3 KM=5 ub1024 decode A/B against mainframe's server before/after.

## Standing (TCP, health-gated 3 reps, 12760-token prompt, before the fix)
| layout | prefill | decode |
|---|---|---|
| v1 | 523 | 20.99 |
| v2c RR=6 | 530 | 21.89 |
| v3 KM=5 | 373 | 18.43 |
v2c's earlier 13.3 t/s was a pre-async-copy-fix RDMA number that was never re-measured; never carry a pre-fix number for
a layout the fix touches. RR=6 is the VRAM ceiling for whole layers on a 32 GB card.
