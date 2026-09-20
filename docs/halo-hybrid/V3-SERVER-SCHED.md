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
Phase 0 (2026-09-20 19:38, mainframe uprobes on rpc_server::graph_compute per device + client sched trace,
v3c KM=5 probe-2 decode, 35.6 steps): per step dev0 (R9700) compute 25.90 ms over 18.2 calls (median 1.11 ms) =
dense of ALL 21 layers + experts of 5; dev1 (APU) compute 29.34 ms over 16.0 calls (median 1.86 ms) = experts of
16; inter-call gaps 17.64 ms (33 per step, median 416 us); one big gap per step 72.71 ms (gibson's layers 0-24 +
3 draft decodes + host loop). Total 145.58 vs measured 146.50 ms/step: 99.4% accounted. Two corrections to the
expectation that follow from it:
1. The per-call tax is ~27 ms/step (17.6 ms of gaps outside the calls + ~9.5 ms of marshalling inside them),
   larger than v3c's whole 17.9 ms deficit; 37 -> 2 calls is worth ~26 ms/step: v3c 146.5 -> ~120 ms -> ~22.5 t/s.
2. Dense work at n=3 is LAUNCH-bound, not bandwidth-bound: the card's dense-per-layer cost (~1.1-1.4 ms) is what
   the APU also spends, so card 25.9 + APU 29.3 = 55.2 ms against v1's 58 ms with everything on the APU. The
   "dense on the card is worth ~15 ms" arithmetic earlier in this section was wrong; the card buys on the
   bandwidth-bound EXPERT GEMVs only (~1.1 ms per expert layer moved, i.e. the KM knob), and within a layer dense
   and experts run sequentially, so the lane is the sum whatever the split. The dense number moves only with
   kernel COUNT per layer (fusion), on both hosts.
So the honest expectation for steps 1-2 is ~22.5 t/s at 2.7 tokens/step (v2c is 21.9); the road from there to
25 is kernel count on the dense path (both hosts), gibson's 72.7 ms share (host loop, draft calls, split gaps)
and tokens per step (draft depth).

## Topology this must serve: N hybrid nodes, not two
The head node (gibson) divides the model into contiguous per-node slices of layers. Each node is a hybrid pair
(or more devices); it loads the KV and dense parts of its slice onto its dGPU and the routed experts onto its APU,
and everything inside the slice runs locally between those devices through the node's own scheduler. Nodes
coordinate only where the computation crosses a slice boundary, and those crossings are tuned to be as few as
possible. For layer-sequential decode the minimum is one crossing per slice boundary per token (N for N remote
nodes, plus the return to the head); prefill can pipeline ubatches through the slices.

Consequences for the design below, so nothing is built two-node-shaped:
- One composite device per ENDPOINT (`RPC<k>[host:port]`), any number of endpoints; each composite exposes one
  extra buffer type per additional server device (`RPC<k>[..]#1`, `#2`, ...) so placement stays expressible with
  `-ts` (layers per node) and `-ot` (expert regexes per node's APU buft). The head's own pair is just the local
  backends. A default placement rule ("dense+KV -> device 0, `ffn_*_exps` -> device 1 for every remote node")
  should exist so N nodes do not need N hand-written regexes; the regex path stays as the override.
- Each rpc-server instance runs one ggml_backend_sched over ALL its local devices; a node with two dGPUs and an
  APU needs nothing new. Stored-graph reuse (step 2) is per node.
- Cross-node traffic per token is the hidden state at the slice boundary (n_tokens x n_embd x 4 B) in and out of
  each node, plus the draft/verify interplay which stays on the head. No node talks to another node; the head
  drives all of them, and each endpoint has its own dispatcher thread, so the N nodes' load phases and their
  per-token graph sends overlap naturally.
- Prefill: today's "rolling two-lane pipeline across the remote device" is written for ONE remote device
  (remote-fetch, "last remote split", lane count 2). For N nodes it becomes a chain of N+1 stages with ubatches
  in flight across them; this is a follow-on item after step 2, and the single-remote-device assumptions in
  src/llama-context.cpp must be found and listed before then.
- Per-node VRAM budgets (below) are per node; KM (expert layers on the card) becomes a per-node knob.

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

## GRAPH_RECOMPUTE is alive, and v2c is already using it (Phase 0, 2026-09-20)
Mainframe's uprobe on rpc_server::graph_compute saw, per v2c decode step, two dev0 calls (the 6-node KDA-state
split, ~60 us, and the 940-node whole-layer split, ~8.4 ms) and NO dev1 call at all, while dev1 did show 1-second
calls during prefill. The APU's decode graph is served by rpc_server::graph_recompute (a different function):
llama's graph reuse (src/llama-context.cpp:1489) skips ggml_backend_sched_alloc_graph, so the splits and their uids
are NOT regenerated for a reused graph, and the client's `last_graph_uid == cgraph->uid` check passes for any RPC
device that receives exactly ONE split per token. v2c's APU is that case; v1's APU and v3c's two devices alternate
2+ splits per token through a ONE-slot cache and never match. So "GRAPH_RECOMPUTE can never fire" (RPC-ASYNC-COPY.md,
memory) was wrong: the cache is one slot deep, not the uids unstable. It also explains part of v2c beating v1: its
APU lane pays no marshalling.
Cheap consequence (step 0.5, ~60 lines, protocol minor bump): a per-device uid cache of K slots on the client and
per-(device, uid) stored graphs on the server, with the uid in the RECOMPUTE request. Every split of every layout
then becomes a RECOMPUTE after its first token: no serialise, no 542 KB send, no deserialise - the ~9.5 ms of
marshalling and the serialise share of the 17.6 ms of gaps go, leaving the round trips (~37 x RTT + dispatcher wait,
est. 4-7 ms). It does not replace the server sched (which removes the round trips and is the N-node shape); it is
the same win at the wire level for ~1/10 of the work, and with it the server-sched design needs no step 2.
KM ceiling: moving one expert layer APU -> card is worth ~1.1 ms/step and only one more fits (KM=6, 4080 MiB), so
KM is a ~0.8% knob, not a tuning axis.

## Phase 0 closed (v2c re-run with both probes, 32 steps, 101.7% accounted) - and what it says about placement
Per v2c decode step: CALL dev0 (KDA-state split) 0.07 ms + CALL dev0 (6 whole layers) 8.65 ms + RECOMP dev1 (16
whole layers) 40.36 ms + two client round trips 4.35 ms + gibson's share 69.92 ms = 123.35 vs 121.25 measured.
Per-layer decode costs at n=3, solved EXACTLY from the two budgets (four equations, four unknowns, no assumed
bandwidth ratio; mainframe's solve): v3c card 21*dc + 5*ec = 25.90, APU 16*ea = 29.34; v2c card 6*(dc+ec) = 8.65,
APU 16*(da+ea) = 40.36:
  APU   dense 0.689   experts 1.834   whole 2.522 ms/layer
  CARD  dense 1.168   experts 0.273   whole 1.442 ms/layer
  -> the card is 1.70x SLOWER on dense and 6.7x FASTER on experts.
Every placement, per layer: dense APU + experts CARD 0.962 (best); whole on CARD 1.442; whole on APU 2.522; dense
CARD + experts APU 3.002 (the plan's placement: the WORST, 19% worse than not using the card).
So the card's decode value is expert bandwidth ONLY, and the right placement is the INVERSE of the plan's: experts on
the card, dense (and KV) on the APU. It is VRAM-capped: ~7 expert layers (4.08 GB each, ~28.6 GB + the server's
compute buffer). Reachable lane: 7 x 0.962 + 14 x 2.522 = 42.0 ms vs v2c's 8.65 + 40.36 = 49.0, i.e. ~7 ms/step
of compute saved, minus 14 local crossings at ~0.1 ms = ~5.6 ms net (~4.5%: 21.9 -> ~22.9 t/s). On today's
transport those 14 crossings are 14 client round trips at ~2.1 ms = 29 ms, a net LOSS, which is why v2c (whole
layers, 2 calls, one on RECOMPUTE) sits at the ceiling of what the current wire can extract from the card. The
server sched is what makes the inverse placement reachable; that, plus the N-node shape, is its case.
Prefill is different (compute-bound), but the card's dense prefill kernels are the weak ones (v3c 373 vs v2c 530
t/s), so dense-on-card loses there too until the gfx1201 prefill kernels improve.
What this does NOT change: the server-sched design is still the N-node architecture (local scheduling per node,
one crossing per slice boundary, placement expressible per node); it just should not be sold as a two-node decode
win. What moves decode toward 25 t/s on these numbers: kernel COUNT on the dense path on BOTH hosts (~60 kernels
per layer; gibson 25 layers + mainframe 21 layers of ~0.7-1.1 ms each = ~35 ms/step of launch-bound work; hipfire
runs ~15 launches per layer), gibson's 70 ms share (host loop ~9 ms, three draft decodes ~6 ms, 45 local splits ~10
ms), and tokens per step (draft depth). Step 0.5 stays: cheap, and any multi-split layout (including the server
sched's own composite graph) needs it.

## VRAM budget on the card (must be checked on paper before code)
Measured today (client-split v3c, KM=5, ctx 131072, ub 1024, two prefill lanes): weights 24420 MiB; client compute
buffer 1833 MiB x 2 lanes = 3666; KV 640 + 480 + RS 193 = 1313. New: the server sched's own compute buffer on
device 0, ~1800 MiB (one, sized for the largest graph). Total ~31.2 GB of 32.6 -> too tight at KM=5; KM=4 frees
~4.1 GB (one whole expert layer) and is the step-1 configuration. The client-scratch term scales with ub: ub=2048 doubles it to 7332 and overruns
the card by ~2.2 GB at KM=5 (and leaves 5.6% at KM=4), so under this design -ub is not a prefill knob until the
scratch is lazily backed; compute buffers allocate LAST at load, so an underestimate of the server-sched buffer
surfaces as a load-time OOM three minutes in, not as a planning error (1425 MiB of headroom at KM=5 absorbs at most
a 79% underestimate). Mainframe can read exact free VRAM under v2c and v3c before the code lands. Later recovery: a lazily-backed scratch buft so the client's 3.7 GB is not real memory.

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

Naming from here: "v3" = this (server sched, intended placement, N-node shape). The old client-split layout is
"v3c"; what runs today is v2c.
