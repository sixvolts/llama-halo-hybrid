# V3: the rpc-server runs its own ggml_backend_sched (plan, 2026-09-20)

## Intent (the user's, verbatim in spirit)
Mainframe's R9700 holds the KV cache and the dense parts (attention, norms, hyper-connections, router) of EVERY
remote layer plus as many routed-expert layers as fit; mainframe's APU holds ONLY routed experts. The two devices
split the work locally, the way gibson's R9700/APU pair already does. What runs today (whole layers on the card,
"v2c") is a ~v2.5 stand-in: it leaves the attention, KV and dense work of 15 layers on the APU.

## Why the intended placement loses today (measured 2026-09-20)
The client-side scheduler treats each rpc-server device as its own backend, so the intended placement becomes
~37 splits per token (21 dense on the card, 16 expert on the APU), each a GRAPH_COMPUTE round trip. Mainframe's
per-call server timing: recv 0.23 ms (542 KB graph payload) + deserialise 0.20 + reply 0.02 = ~0.45 ms, plus the
client's serialise and the dispatcher's wait per call. Per token at the observed 53/47 call split (mainframe's correction: the APU's 28-tensor
graphs cost 0.045 ms/call, not 0.444): 19.7 x 0.444 + 17.3 x 0.045 = 9.5 ms of SERVER-side marshalling, 53% of
v3c's measured 17.9 ms/step penalty (v3c KM=5 decodes at 18.4 t/s = 146.5 ms/step vs v1 21.0 = 128.6 vs v2c 21.9
= 123.3). The other 8.4 ms is expected to be the CLIENT-side per-call cost (serialise_graph of ~1800 records x 20,
the dispatcher's reply wait, TCP round trip) - Phase 0's gap_us measures it; if it sums to ~8 ms the budget closes
with no unknowns. Both halves are per-call, so 37 -> 2 calls removes both; step 2 removes the last 2. Fix #2
("stable split identity so GRAPH_RECOMPUTE fires") was dismissed this morning on a per-CALL cost of 0.2 ms; the
right denominator is per token (37 calls). Dispatch (HIP-graph boundaries, 2.87 vs 1.74 us) is ~0.5 ms of it.
Kernel-level expectation once the per-split cost is gone: dense+attention of 21 layers at n=3 move from ~1.1 ms
(APU) to ~0.4 ms (card) = ~15 ms/step, 5 expert layers at ~2.5x = ~3 ms -> step ~110 ms -> ~24.5 t/s at 2.7
tokens/step. That is the ~25 expected from the second card.

## Design
One composite device per endpoint on the client; one ggml_backend_sched per connection on the server.

### Client (ggml-rpc.cpp)
1. `RPC0[endpoint]` stays one device, but exposes TWO buffer types: the default (server device 0 = R9700) and an
   EXTRA buft (`get_extra_bufts`, server device 1 = APU), named so `-ot` can target it (e.g. `RPC0x[...]`). Weights,
   KV and recurrent state are placed by the client exactly as today (-ts puts layers 25-46 on RPC0, -ot sends
   `ffn_*_exps` of layers >= 25+KM to the extra buft). `supports_buft` returns true for both, so the client's sched
   assigns every op of layers 25-46 to ONE backend and emits ONE split per token for the remote range (two
   crossings per token, like v2c).
2. Scratch: the composite's compute buffer (the client sched's galloc buffer for that backend) is allocated on
   server device 0 as today, but the alloc carries a SCRATCH hint so the server knows which buffer holds
   client-placed intermediates.
3. Boundary outputs: `ggml_backend_sched_split_graph` knows which nodes of a split are read by later splits
   (`split->inputs` of later splits) or are graph outputs. Set a tensor flag (new bit, e.g.
   `GGML_TENSOR_FLAG_SCHED_OUTPUT`) on those nodes before `graph_compute`; `rpc_tensor.flags` already crosses the
   wire. Inputs copied in by the client (set_tensor_async into scratch) are leafs with data and need no marking.
4. GRAPH_COMPUTE for a composite backend carries a mode flag (or new cmd `GRAPH_COMPUTE_SCHED`); protocol minor
   bump, gated by the hello like the async copy was.

### Server (rpc_server)
5. Per connection: `sched = ggml_backend_sched_new(backends, NULL, n_backends, graph_size, false, true)`.
6. On a sched-mode graph: deserialise as today (tensors keep their buffers/data from the wire). Then UNPIN every
   tensor that lives in the SCRATCH buffer, is not a leaf (op != NONE), and is not flagged as a boundary output:
   buffer = NULL, data = NULL (views of an unpinned tensor: data = NULL, view_src kept; galloc handles views).
   Weights, KV, recurrent state, inputs and flagged outputs stay pinned where the client put them.
7. `ggml_backend_sched_graph_compute(sched, graph)`: the server's sched assigns each op by its pinned operands
   (expert ops to the APU, everything else to the card), allocates the unpinned intermediates in ITS OWN per-device
   compute buffers, and inserts the local device-to-device copies (~100 us each, same as gibson's local pair).
   Boundary outputs flagged in 3 stay pinned in scratch on device 0; if the sched wants their producer on the APU it
   would copy operands, so instead: unpin outputs too and copy them back to the client's location after compute
   (a few tensors x 60 KB at decode, ~20 MB at ub=1024). get_tensor/cpy from the client then works unchanged.
8. `sync_backend_for` already drains the owning backend before host access; the sched's compute is blocking per
   call, as today.

### Step 2 (after step 1 measures): stored graph + sched reuse
With graph reuse on the client the remote split is structurally identical token to token (it changes at prompt
boundaries and every 256 KV positions). Client computes a structural id of the split (node ops, shapes, names,
pinned data pointers); if it equals the last id sent, send RECOMPUTE(id) instead of the ~2 MB graph. Server keeps
the deserialised graph and its sched allocation per (device set, id) and re-runs compute without deserialise,
split or galloc (needs a sched entry point that skips alloc when the graph object is unchanged). This is fix #2
at the right layer; it removes the remaining per-token marshalling (~1 ms wire + ~1 ms deser + ~1-2 ms split/alloc).

## VRAM budget on the card (must be checked on paper before code)
Measured today (client-split v3c, KM=5, ctx 131072, ub 1024, two prefill lanes): weights 24420 MiB; client compute
buffer 1833 MiB x 2 lanes = 3666; KV 640 + 480 + RS 193 = 1313. New: the server sched's own compute buffer on
device 0, ~1800 MiB (one, sized for the largest graph). Total ~31.2 GB of 32.6 -> too tight at KM=5; KM=4 frees
~4.1 GB (one whole expert layer) and is the step-1 configuration. Mainframe can read exact free VRAM under v2c and
v3c before the code lands. Later recovery: a lazily-backed scratch buft so the client's 3.7 GB is not real memory.

## Attribution (so the measured win is credited to the right cause)
37 -> 2 calls recovers ~17 ms/token of v3c's penalty, i.e. it repairs v3c to roughly v1/v2c territory; it does NOT
by itself beat v2c, which already pays 2 crossings. The case for the design is the PLACEMENT: KV and the dense
parts of all 21 remote layers on the card instead of 6, worth the ~15 ms/step kernel arithmetic above. The A/B
that proves it is v3 (server sched) vs v2c at equal KM, not v3 vs v3c.

## Risks, named
- Server sched placing ops badly: `op_offload` / CPU fallback must be off; verify with GGML_SCHED_DEBUG on the
  server that expert ops land on the APU and everything else on the card, and that NO weight is ever copied.
- Views and in-place ops on unpinned tensors (view_src chains, the KDA conv-state concat writing into a view).
- VRAM: the client's scratch for the composite (client galloc sizes it for the whole split, ~1.8 GB at ub=1024)
  AND the server sched's own compute buffers both live on the card. KM=5 leaves ~3 GB; KM=4 if it does not fit.
  Later: a lazily-backed scratch buft so the client's allocation is not real.
- Prefill two-lane pipeline (LLAMA_PREFILL_LANES, remote fetch, "last remote split"): one remote split per ubatch
  simplifies it, but the code that walks RPC devices must be checked.
- HIP-graph cache keys on the server: fixed today (first/last node op+name), needed because the server sched now
  produces many per-layer splits itself.
- Draft/MTP: the draft runs on gibson's R9700 (-devd ROCm0), unaffected.

## Gates and measurements
- Correctness: greedy_ref.sh v3 KM=4, 1210-token prompt, 160 greedy tokens, block sha a828e28289899da6 (marker-
  inclusive scope), both hosts on the same commit.
- Baseline before code (Phase 0): client-split v3 KM=5 and v2c under mainframe's per-device uprobes + gibson sched
  trace: calls/token, per-call server wall, gap_us. Already known from the timing table: 0.45 ms/call server side.
- After step 1: v3 KM=5 ub1024 3 reps (harness), expect the ~37 crossings to become 2 and decode to land between
  v2c (21.9) and the kernel bound (~24.5). Server-side GGML_SCHED_DEBUG once to confirm placement.
- After step 2: same, expect ~24-25. Then a KM sweep and hybrid fill of the card's remaining VRAM.

## Sequencing
0. Phase 0 baseline (one v3 run, one v2c run; ~30 min).
1. Client composite device + extra buft + output flagging + wire flag; server sched mode. TCP only. ~2 days.
2. Stored graph + sched reuse (RECOMPUTE by id). ~1 day.
3. KM sweep / hybrid fill on the card. Then the kernel items from the hipfire review (mmvq scheduling barrier,
   scalar block headers, accumulator chains; router softmax+topk fusion; gfx12-native FA fragments), each A/B'd.
4. Later: host loop (GPU-side sampling/acceptance), retained PM4 replay (hipfire Redline, +6-7% over hipGraph on
   gfx1201 by their measurement).

Naming from here: "v3" = this (server sched, intended placement). The old client-split layout is "v3c"; what runs
today is v2c.
