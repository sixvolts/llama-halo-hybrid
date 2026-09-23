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
What the numbers mean, physics first: the card's DRAM floor for a layer's dense weights (170.7 MiB) is 0.28 ms
against the APU's 0.69, so the intended placement - KV and dense on the card, experts on the APU - is the right one
physically, by a margin of ~0.4 ms per layer. The card measuring 1.17 ms today is a SOFTWARE gap of ~0.9 ms per
layer: small GEMVs that mmvq's wave-per-row mapping cannot drive to bandwidth on a 64-CU part (shexp 8.5 MB at
35%, f_b/g_b 1 MB at 10%, hc_fn 0.4 MB at 2%) and ~40-50 tiny launches per layer. Both are fixable in the kernels
and the graph, and both are general to this card. They are optimisation work that follows the design, not a reason
to change it. The decision on the design is the user's; the plan below is that design, built end to end first.

## VRAM budget on the card (must be checked on paper before code)
Measured today (client-split v3c, KM=5, ctx 131072, ub 1024, two prefill lanes): weights 24420 MiB; client compute
buffer 1833 MiB x 2 lanes = 3666; KV 640 + 480 + RS 193 = 1313. New: the server sched's own compute buffer on
device 0, ~1800 MiB (one, sized for the largest graph). Total ~31.2 GB of 32.6 -> too tight at KM=5; KM=4 frees
~4.1 GB (one whole expert layer) and is the step-1 configuration. The client-scratch term scales with ub: ub=2048 doubles it to 7332 and overruns
the card by ~2.2 GB at KM=5 (and leaves 5.6% at KM=4), so under this design -ub is not a prefill knob until the
scratch is lazily backed; compute buffers allocate LAST at load, so an underestimate of the server-sched buffer
surfaces as a load-time OOM three minutes in, not as a planning error (1425 MiB of headroom at KM=5 absorbs at most
a 79% underestimate). Mainframe can read exact free VRAM under v2c and v3c before the code lands. Later recovery: a lazily-backed scratch buft so the client's 3.7 GB is not real memory.

## Measured facts the build is judged against (Phase 0, 2026-09-20)
- v3c step 146.5 ms: card 25.9 (dense x21 + experts x5), APU 29.3 (experts x16), inter-call gaps 17.6, gibson 72.7.
- v2c step 123.4 ms: card 8.7 (6 whole layers), APU 40.4 (16 whole layers, on RECOMPUTE), round trips 4.35, gibson 69.9.
- Per layer at n=3: APU dense 0.69 / experts 1.83; card dense 1.17 / experts 0.27; DRAM floors 0.28 (card) / 0.69 (APU).
- Card dense composition (in situ, no profiler): ~0.4 ms GEMVs at 70-92% of bandwidth, ~0.2 ms small GEMVs at
  2-35%, ~0.3-0.4 ms in ~40-50 tiny launches. Kernel work targets: small-GEMV mapping on gfx1201, launch fusion.
- The V3 build removes the 37-call chain (~26 ms/step in v3c). Its first measurement is v3 vs v3c at KM=4/5; its
  end state after the optimisation phase is judged against the physics: card dense toward 0.28 + launches.

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

## Phase 1 result (2026-09-20 23:04, commits 13742baf4 + 3c6600e04, TCP, health-gated 3 reps)
| layout, 12760-token prompt | prefill | decode | | 3148-token prompt | prefill | decode |
|---|---|---|---|---|---|---|
| v1 (card idle) | 523 | 20.99 | | | 446 | 20.17 |
| v2c RR=6 (whole layers) | 530 | 21.89 | | | 445 | 21.58 |
| v3c KM=5 (client-split) | 373 | 18.43 | | | 364 | 18.02 |
| **v3s KM=4 (server sched, intended placement)** | **492** | **21.87** | | | **394** | **22.16** |
Gates: greedy a828e28289899da6 (token-identical), placement proof 0 violations over 12,907 dump lines (65 splits per
graph: 33 card / 32 APU / 0 CPU), VRAM peak 20660 MiB on the card, no NIC events. v3c -> v3s recovers the ~26 ms/step
of per-call cost; the remaining gap to v2c on prefill is the card's dense prefill kernels, and the card's dense decode
software gap (Phase 0: 1.17 vs floor 0.28 ms/layer) is untouched - both Phase 3. Two server-side lessons from the build:
a fresh graph needs an explicit sched reset+alloc, a RECOMPUTE must reuse the plan (the sched mutates sources when it
splits), and every non-node deserialised tensor must be a leaf whatever its op (GLM reaches KV/state through views).
A small-model smoke test (docs/halo-hybrid/smoke_v3s.sh) proved the mechanics but not the GLM-size leaf case.
Per-step budget of the gate (mainframe uprobes, 700 calls): MODEL graph 46.4 ms (p25-p75 45.3-47.6) + TINY KDA-state
graph 0.1 ms + gaps 2.8 ms + gibson 80.9 ms = 130 ms at ~2.9 tokens/step (MTP n-max 2, acceptance 0.77) = 22 t/s;
server busy 36%. RECOMPUTE fired 0 of 700 times: the composite receives TWO splits per token (MODEL + TINY) that
strictly alternate (303 M->T / 304 T->M / 0 M->M) through the client's one-slot uid cache, so every token is a fresh
split on the server - step 0.5 (a 2+ slot cache, or merging the tiny split client-side) is the fix and its ceiling is
measured next (mainframe's gc-decomp.bt: alloc vs compute vs rest per MODEL call; client sched trace for serialise).
Step 0.5 SIZED AND DROPPED (2026-09-20 23:14, mainframe's nested uprobes on graph_compute / sched_alloc_graph /
sched_graph_compute, probe-2 decode, 39 MODEL calls): MODEL total med 46.22 ms = alloc (reset+split+galloc of 3182
nodes) 0.650 + compute 44.60 + rest (deserialise + boundary copies + sync) 0.82; TINY 0.097 total. Worst case for
what RECOMPUTE could recover: alloc + all of rest + TINY + the client's 0.7 ms submit = ~2.3 ms of a 130 ms step
(1.7%, ~+0.4 t/s) - below the harness noise floor (22.16 vs 22.34 between identical configs). Not worth building;
the 64-entry HIP-graph cache cap goes with it (replay measured at ~0.36 ms). Where the time is: gibson 80.9 ms (62%),
mainframe GPU compute 44.6 (34%, carrying the ~20 ms dense software gap of Phase 0), everything else 3.9.
A/B with GGML_CUDA_DISABLE_GRAPHS=1 on the server: 396 / 22.34 and 494 / 21.91 vs 394 / 22.16 and 492 / 21.87 - a
null: HIP-graph replay contributes nothing to the server's decode while every token re-splits (warmup never
completes); the 64-entry per-device graph cache (common.cuh max_cuda_graphs) is moot until RECOMPUTE fires and must
be re-checked then (33 + 32 live keys per token per device against a 64-entry LRU).

## KM / ubatch sweep (2026-09-20 23:46 - 00:07, v3s, TCP, 3 reps each, mainframe VRAM sampled every 2 s)
| config | prefill 3148 | decode 3148 | prefill 12760 | decode 12760 | card VRAM peak |
|---|---|---|---|---|---|
| KM=4 ub512 | 374 | 22.20 | 445 | 22.69 | (sampled, see mainframe) |
| KM=4 ub1024 (Phase 1 baseline) | 394 | 22.16 | 492 | 21.87 | 20660 MiB |
| KM=4 ub2048 | 356 | 22.97 | 479 | 23.21 | 30463 MiB (93%) |
| KM=5 ub1024 | 404 | 22.11 | 508 | 22.07 | 30435 MiB (93%) |
| KM=5 ub2048 | OOM at load: cudaMalloc of the 3667 MiB client lane buffer failed at 25059 MiB used (7.5 GB nominally free - a contiguous-block limit; the config needs ~34 GB in total anyway) | | | | |
Readings: prefill is best at KM=5 ub1024 (+3% over the baseline); ub2048 does not help prefill here (the two-lane
pipeline finding from 09-13 holds). Decode "varies with ub" (21.87 / 22.69 / 23.21 for ub 1024 / 512 / 2048) is NOT a decode
effect: mainframe's step period is flat across configs (111.6-114.1 ms, 2.3%) and MODEL compute 43.5-44.7 ms, while
per-rep draft acceptance tracks the t/s exactly (acc 0.77 -> 21.0-22.0, 0.82-0.86 -> 23.0-23.6, 0.88 -> 24.0). A
different prefill ubatch changes the f32 summation order of the router GEMM, near-tied experts flip, the generated
text differs, and MTP acceptance moves - the same mechanism recorded on 2026-09-13. Compare decode across configs by
step period or at equal acceptance, not by t/s alone; the harness should print tokens/step. KM's decode effect is ~+0.2 t/s (as
Phase 0's ~1.1 ms/layer predicted). VRAM: KM=5 and ub2048 each consume the card's headroom; both together do not fit
until the client's scratch stops being real memory on the card (the lazily-backed scratch buft from the design).

## With the scratch in host memory (28b6a3e5c, proto 7.3; 2026-09-21 01:10-01:21, 3 reps, tok/step and ms/step now printed)
| config | prefill 3148 | prefill 12760 | decode 12760 (t/s @ tok/step) | ms/step | card VRAM (weights) |
|---|---|---|---|---|---|
| KM=5 ub2048 (previously OOM) | 364 | 492 | 23.56 @ 2.74 | 114-116 | 24420 MiB + KV + server compute |
| **KM=6 ub1024** | 415 | **523** | 22.29 @ 2.59 | 112-115 | 28596 MiB + KV + server compute |
Readings: ub2048 still buys nothing (prefill -3% vs KM=5 ub1024's 508, ms/step +3%); KM=6 gives the best v3s prefill
(523, level with v2c's 530) and the card is now ~31 GB full with weights + KV + the server's own compute buffer; the
client scratch no longer costs the card anything. Step period is flat at ~112-116 ms across KM 4/5/6 - KM's decode
value is within noise, its value is prefill. Mainframe's samplers: card peak KM=5 ub2048 27422 MiB (5.2 GB headroom, previously could not allocate), KM=6 ub1024
31075 MiB (95.2%, 1.5 GB headroom - tighter than KM=4 ub2048 was); host-memory delta on mainframe 6.7 GB at KM=6
ub1024 (as predicted) but 14.8 GB at KM=5 ub2048 (double the two 3.67 GB lanes; RSS accounts for 9.8 GB of it -
unexplained, do not assume "2x the lane size" for larger ubatches). VmLck stayed 0 (hipHostMalloc pins through the
driver, not mlock), so "pinned" is by design, not measured. MODEL call: KM=6 compute 42.85 ms (lowest measured; more of
the slice on the card), alloc unchanged at 0.68. Prefill call KM=5 ub2048 3313 ms vs KM=6 ub1024 1636 ms.
Production candidate: v3s KM=6 ub1024 for prefill (523), with 1.5 GB of card headroom; KM=5 ub1024 (508) keeps ~5 GB.

## Phase 3a result: the card's "small GEMV" gap was one table entry (2026-09-21, commit dad9e8d71)
Root cause, from the Phase 0 rocprofv3 trace grouped by launch geometry: upstream mmvq.cu's RDNA4 table returns
nwarps=8 only at ncols_dst=1. With the MTP draft the verify width is n=3, so EVERY Q8_0 dense GEMV of a decode step
on the card launched ONE 32-thread wave per row - 24x16384 (hc_fn) as a 64-trip serial DRAM-latency chain on 24 of
the part's 2048 wave slots, 2048x4096 (shared expert) as 2048 waves each walking 16 trips. That is the "2-35% of
bandwidth" of Phase 0; it is not a general un-tuned-kernel problem.

Second finding on the way: **gfx1201 has a dispatch cliff** when a launch's total wave count lands within a few of
its 2048 wave slots (32 WGP x 4 SIMD32 x 16; HIP reports WGPs as multiProcessorCount). K-independent, 2-5x:
2048 rows x 1 wave 19.5 us vs 2032 rows 9.8; 256 blocks x 8 warps 19.7 us vs 252 blocks 3.7. The production
2048-row shared-expert GEMV sat exactly on it. The APU (1280 slots) shows no cliff. The mmvq launcher now steps any
shape that would land there to the other launch; the grouped-GEMV and MoE kernels are not guarded yet.

Change (`GGML_CUDA_MMVQ_NO_WIDE=1` restores upstream): on RDNA4 at n=2..4, 8-warp blocks (split-K over the block,
LDS reduction) when the 8-way split still gives each warp a full trip (K >= 2048 for q8_0) and the matrix is under
2^25 weights; upstream launch otherwise. Two variants measured and dropped: a rows-per-block launch for K=128
(f_b/g_b: no gain in situ, 15.9 -> 16.5 us) and wide for short K (256x512 / 512x256 lost 1.3-2.1x: idle warps plus
the reduction).

In situ, gibson's card, v3s KM=5 ub1024 decode, op timer (GGML_CUDA_TIME_OPS, graphs off), upstream -> wide:
| GEMV (K x rows, n=3) | per layer | upstream us | wide us |
|---|---|---|---|
| 16384x24 hc_fn | 1 | 30.8 | 14.9 |
| 4096x2048 shared-expert down | 1 | 39.1 | 28.6 |
| 2048x4096 shared-expert gate/up | 1 | 38.7 | 27.8 |
| 128x8192 f_b/g_b, 8192x4096 wo, 4096x12288 qkv, 12288x4096, 154880 head | - | unchanged (within 1%) | |
Card time per graph 7.37 -> 7.33 ms (mixed prefill+decode graphs). Step period, 3 reps x 2 ctx, gibson only on the
new kernel (mainframe still on 37cdac384): 115.7 -> 114.5 ms/step (first wide run 113.8); ~1 ms of the 130, i.e.
~38 us x 25 local layers, as the per-op numbers predict. Mainframe's 22 card layers should add about the same once
it rebuilds. Isolated (test-backend-ops perf, MALL-warm, grouping off): 24x16384 15.7 -> 4.5 us, 2048x4096
19.9 -> 11.8, 4096x2048 24.5 -> 14.8, 4096x4096 27.5 -> 21.2, 2048x1024 20.3 -> 7.8 (the cliff).
Greedy gate (v3s KM=4, greedy_ref.sh): text sha a828e28289899da6, identical to the reference even though the
8-way split changes the f32 summation order; 106/106 draft tokens accepted as before.

Both hosts on 1f8b0c194 (mainframe rebuilt 05:24, its own isolated A/B reproduced all four predicted shapes:
24x16384 17.1 -> 5.3 us, 2048x4096 22.8 -> 11.9, 2048x1024 23.2 -> 7.8, 12288x4096 unchanged), harness-gated
3 reps x 2 ctx, 05:27-05:38:
| config | ms/step before (7.3, upstream launch) | ms/step after | prefill 12760 tok |
|---|---|---|---|
| v3s KM=5 ub1024 | 115.7 (gibson-only A/B, same morning) | 113.0 | 505 (508 before) |
| v3s KM=6 ub1024 | 113.8 (01:10 batch) | 111.1 | 522 (523 before) |
-2.7 ms/step at both KMs, i.e. ~1.3 ms per host, in line with the per-op prediction (0.8-1.0 ms per host) plus
noise; prefill unchanged (ub=1024 GEMMs run on MMQ, not mmvq). Mainframe's decomp-trace MODEL compute median at
KM=6 was the pre-registered acceptance number (baseline 42.88 +/- 0.25 ms, prediction ~42.1): measured 41.81 ms
(n=250, +/-0.24), -1.07 ms, with alloc (0.684 -> 0.694) and rest (0.790 -> 0.796) unchanged, so the change is
isolated to the compute term. Its MODEL-to-MODEL step delta (-2.53 ms) matches the harness's (-2.70) to 0.17 ms.
The in-situ gain is ~0.3 ms larger than the isolated shape A/B predicted (22 layers x 35.3 us = 0.78): isolated
per-shape numbers UNDER-predict the in-situ effect of removing serial latency chains from a 3182-node graph, which
is the right direction to size 3b with. Best decode so far: KM=6 ub1024 at 111.1 ms/step, 23.08 t/s at 3148 ctx.

What this leaves of the 0.9 ms/layer software gap (Phase 0: card dense 1.17 vs floor 0.28): the small GEMVs were
~0.2 ms/layer and are now ~0.16 (the remaining shortfall on 2048-row shapes is 28 us vs a 14 us floor: 16384 one-trip
waves over 2048 slots is 8 rounds of DRAM latency plus a reduction each; a 4-warp or 2-rows-per-block variant is the
next thing to try there). The larger piece is the ~40-50 launches per layer (3b).

## Phase 3b, step 1: the decode launch inventory and two kernel fixes (2026-09-21, commit 5261729cc)
Inventory (op timer with decode/prefill-tagged keys, gibson's card, v3s KM=6, graphs off so the per-launch cost is
inflated; ~1514 launches and 37 ms per step on the card): MUL_MAT 300 launches / 17.8 ms; the remaining ~1200
launches are element-wise and small ops (RMS_NORM 163, MUL 153, CONT 98, GET_ROWS 74, ADD 67, UNARY 67, CPY 65, the
three DSV4 hyper-connection kernels 160, L2_NORM 41, SET_ROWS 38, GLU 30) at 8-13 us each under the timer. Under HIP
graph replay these cost far less than the timer shows (the ewchain pass measured ~1 ms per ~320 nodes removed on
2026-09-15), so launch fusion proper (3b) is worth ~2-3 ms per step on gibson, not the 12 ms the timer suggests.
Bigger single items the inventory exposed:
- The draft steps run the full 154880 x 4096 q8_0 head at n=1 twice per step (2.1 x 1.09 ms) plus once at n=3:
  3.4 ms of the step, bandwidth-bound (91%). A lower-precision head for the draft is a model-side lever (3c).
- CONCAT 6x24576 (the KDA conv-state concat, 19 per step) at 61.6 us: the generic kernel launched one 256-thread
  block per row for the 3 state columns. Row kernel: 10.0 us in situ (~1 ms per step). Same fix covers the
  single-token concat that hit per-element 64-bit index math (19 us).
- The shared-expert gate/up pair and the following add at n=3 went to the fused MMQ path because upstream fuses
  into mul_mat_vec_q only at n=1. Lifted to n <= 4. Trap found on the way: the fused ADD operand at n > 1 is a
  [rows, n] tensor (the shared-expert output added to the routed-expert sum), not a [rows] bias; indexing it per
  row passed every test (test biases are [rows,1]) and produced a different greedy text with draft acceptance
  0.44 in situ. The device fusion args now carry the operand's column stride and the test has bias_per_token
  cases. Gain is small (the fused down+add 33.9 us vs 27.8 + an 8 us add); the large 4096x8192 "+fused5" class
  (201 us, 19 per step) is the fork's grouped GEMV launch, not MMQ - what it covers is being checked with named
  timer keys before deciding whether it is at bandwidth.
Result, v3s KM=6 ub1024, gibson only on this build (mainframe on 1f8b0c194): 111.1 -> 110.1 ms/step, greedy text
identical (a828e28289899da6), prefill unchanged.

Step 1 also added a 32-warp "tall" mmvq launch for <= 64 rows over K >= 8192 (d09a3ac7e; hc_fn 24x16384: isolated
5.4 -> 4.4 us at n=3, 4.6 -> 3.4 at n=2; mainframe's card 4.56 -> 3.89, 32x32768 6.23 -> 4.61). Both hosts on
d09a3ac7e, KM=6 ub1024, 3 reps x 2 ctx (08:03-08:09): **109.7 ms/step** (111.1 after 3a, 113.8 before it),
prefill 530 t/s at 12760 tok. Mainframe's decomp trace (same method as 3a): MODEL compute median 41.81 -> 40.87 ms
(-0.94, CI +/-0.25, n=248), alloc unchanged, rest +0.05 (first non-compute move in the series; worth a glance if a
later step touches copies/sync again); cumulative Phase 3 on its compute term 42.88 -> 40.87 (-4.7%). Its
MODEL-to-MODEL step delta -1.98 vs the harness's -1.40. Two incidents worth the record: (1) with the op timer on (GGML_CUDA_TIME_OPS +
GGML_CUDA_DISABLE_GRAPHS), the tall build stalled at load once on gibson - main thread spinning in a synchronize,
GPU at 5%; graphs-off alone and the production configuration both load and run, and mainframe could not reproduce
any combination in isolation. Timer runs use GGML_CUDA_MMVQ_NO_TALL=1 until that is understood. (2) A build of the
next op that failed on ggml-rpc.h's op-count static assert had already relinked libggml-base/cpu with the new op
mid-enum while libggml-hip/rpc kept the old ids; a queued load ran on that mix and was killed before its warm-up
reached mainframe. The op now goes in appended last, as protocol 7.4 (minor is the field the handshake compares),
and the HELLO reply carries GGML_OP_COUNT so a mismatched pair is refused.

## Phase 3b, step 2: the hyper-connection prologue as one op (2026-09-21, 608cad103 + 919c2ade9, protocol 7.4)
GGML_OP_DSV4_HC_MIX replaces, per mixer at decode, rms_norm + the 24-row hc_fn GEMV + the two gate chains +
dsv4_hc_comb + dsv4_hc_pre (six launches; two mixers per layer, ~250 launches per step on gibson) with one
1024-thread block per token; post and comb are views of its result. v1's data path (one scalar q8 load per element
per row) ran 113 us per mixer in situ against ~60 for the pieces and regressed the step by 5 ms; v2 (16 consecutive
elements per thread, one 16-byte quant load per row, float4 activations) runs 31.5 us under the timer, 18.8 us
isolated. Greedy gate identical both times. A/B on one build, both hosts on 919c2ade9, KM=6, back to back:
| | ms/step (3 reps x 2 ctx) | prefill 12760 |
|---|---|---|
| LLAMA_NO_HC_MIX=1 (six launches) | 109.9 111.0 107.2 110.9 108.2 110.7 = **109.7** | 528 |
| merged op | 109.1 109.7 107.0 109.4 105.8 109.0 = **108.3** | 530 |
-1.3 ms/step. Mainframe's decomposition of the v2 op (its 22 layers): compute 40.86 -> 40.61, alloc 0.685 -> 0.581
(the smaller graph; first move of that term in the series), rest 0.848 -> 0.726, server busy -0.52 ms. A 3-rep
20 minutes earlier on the same build had read 116.5 ms/step with a 125.7 first probe; the back-to-back A/B and
mainframe's trace show that batch was an outlier of the host, not the op. Rule from it: judge a change by a
same-session A/B, never by two batches an hour apart.
Wire: the op is appended last in the enum, the protocol is 7.4 (minor is compared, patch is not) and the HELLO reply
carries GGML_OP_COUNT which the client refuses on mismatch. LLAMA_NO_HC_MIX=1 restores the six-launch path.

Phase 3b stands at 113.8 -> 108.3 ms/step (5%) across 3a + 3b on gibson's harness, and 42.88 -> 40.61 ms on
mainframe's compute term. The remaining launch-fusion items (hc_post into the next prologue, KDA concats, router
softmax+topk) are each worth well under 1 ms. Gibson's own share of the step (~65-70 ms of ~108 by mainframe's
client-idle measure) is 3c and is where the step is.

## Targets set 2026-09-21 (the user's, revised after the 3c decomposition): decode 27 tok/s, prefill 800 tok/s, on v3s KM=6 ub1024
Baseline at the time: decode 23-24.6 t/s (108 ms/step, 2.5-2.7 tok/step), prefill 530 t/s at 12.7K.
Decode budget (median step at 13K, sched trace 09:10): 96 ms blocked in the verify graph (mainframe busy 42;
gibson's 25 layers + APU experts + 45 split gaps 54), 1.2 ms launch work, 9.5 ms host chain (three sequential
draft graphs, each with the shared 634 MB q8_0 vocab head at n=1, plus sampling). GPU floor of the step with this
placement ~65-70 ms (APU expert reads at n=3 ~52, card dense ~13). 28 t/s at 2.6 tok/step = 93 ms/step, i.e.
~15 ms of the ~40 ms of overhead: the draft chain (head copy for the draft, fewer draft graphs), gibson's split
syncs, mainframe's launch gaps; n-max 4 helps at long contexts only (+17 ms/step per extra draft token from the
APU expert floor; sweep: n-max 2/3/4 = 108/143/143 ms/step, 2.6/3.1/3.5 tok/step).
Prefill: mainframe's 13K prefill is 94.6% GPU busy; expert GEMM 37% at ~17% of gfx1151's int8 peak, flash
attention 13.5%, dense GEMM 13%, MoE reduce/quantize 9%, element-wise remainder. 800 = 1.5x = roughly double the
expert-GEMM efficiency at 1024-token batches plus attention; kernel work on the APUs, the design unchanged.

## Phase 3c, measurements (2026-09-21, 09:10-10:48, v3s KM=6 both hosts on 919c2ade9+)
Same-session A/Bs, 2 reps x 2 ctx unless noted; ms/step at 3K / 13K:
| lever | result | verdict |
|---|---|---|
| draft depth n-max 3 / 4 | 108 -> 143 ms/step, 2.6 -> 3.1 / 3.5 tok/step; +17 ms per extra draft token (APU expert reads) | n-max 4 wins only past ~10K ctx (+9% there); policy item |
| target backend sampling (-bs) | 107-114 vs 107-110; prefill -5..7% | no |
| token_embd on the card (640 MB VRAM) | 107.1-110.2 vs 106.2-109.8 | no |
| draft head as Q4_0 in the MTP file (blk.45.nextn.shared_head_head, 348 MB, gguf-py Q4_0; Q4_K needs the C++ quantizer) | 105.4-108.1 vs 107.1-110.2, acceptance unchanged | **yes, -1.5 to -2 ms; launcher default** |
| HIP graphs off on gibson | +0.5 ms | replay is worth ~0.5 ms here; mainframe's eager splits cost the same, no launch-gap item (its interior GPU busy 97.7% by the amdgpu counter) |
Host chain per step (LLAMA_SPEC_TRACE, medians at 13K): ingest decode 1.09, draft-0 decode 0.67 + 2.37 wait, draft-1
decode 0.17 + ~2.4 wait, target sample+accept 0.45, bookkeeping ~2. Merging the ingest into the first draft graph is
blocked by the draft's recurrent-state checkpoint (saved after ingest, restored after drafting): the merged graph
would never materialise the post-ingest state. Mainframe's 40 ms MODEL compute is kernel floors, not gaps.

## Prefill scoping, step 1: the lane balance is a free lever (2026-09-21 11:04-11:18)
The scheduler trace of a 12.7K prefill shows, per 1024-token ubatch, gibson's lane thread blocked ~1.3 s in the
send to mainframe (its server still on the previous ubatch) plus ~0.3 s in the fetch wait; mainframe measures its
lane at 1.665 s per ubatch, 96.3% busy, cadence 1.73 s -> 592 t/s steady state, the observed 530 being head and
tail. Mainframe's call is 99.6% compute: nothing structural to remove there. So the split (25 gibson / 22
mainframe since V3) leaves gibson's lane short and the knob costs nothing: launcher `LOCAL=<n>` (layers 0..n-1 on
gibson, KM expert layers on mainframe's card counted from n).
| LOCAL | prefill 12.7K (cold first probe) | prefill 25.8K | decode ms/step | gibson card model MiB | mainframe card MiB |
|---|---|---|---|---|---|
| 25 | 482 | 521 | 109.6 / 107.8 | 13554 | 28596 |
| 26 | 502 | 547 | 111.1 / 110.5 | 13724 | 28426 |
| 27 | 526 | 577 | 109.7 / 106.4 | 13895 | 28255 |
| 28 | 506 | 551 | 111.0 / 108.5 | 14053 | 28097 |
| 29 | 505 | 542 | 110.0 / 108.7 | 14223 | 27926 |
| 30 | 104 (host paging: 108 of 122 GB used) | 474 | 183.6 / 115.1 | 14394 | 27756 |
Knee at 27: +11% prefill at 25.8K, +9% at 12.7K, decode unchanged, no quality effect (placement only). Mainframe's
per-ubatch call / idle gap at 25.8K (its uprobe trace): LOCAL 25 1691-1706 / 83-86 ms, 26 1612 / 85, 27 1507 / 81,
28 1393 / 289, 29 1295 / 433, 30 1199 / 614 - the crossover is sharp: at 27 mainframe is still binding by a hair, at
28 it idles a third of each ubatch waiting for gibson. Its decode MODEL compute fell 40.39 -> 30.26 ms with the two
layers moved while decode t/s stayed flat (gibson absorbed exactly that work). **LOCAL=27
is the production layout from here (launcher default), with KM=6 ub1024.** At 28+ gibson's lane is the long one;
30 pages the host. Mainframe's card headroom grows ~340 MiB at 27 (KM=7 there would need ~4 GB, so no).

**Production number, v3s LOCAL=27 KM=6 ub1024, both hosts on 919c2ade9 (gibson +client-only commits), harness-gated
3 reps (11:34-11:48):** 3K prefill 443 (480 warm) / decode 25.3 t/s at 103-106 ms/step; 12.7K prefill 577 (586
warm) / decode 25.9 t/s at 104-105 ms/step, 2.74 tok/step. From the day's start (113.8 ms/step, 23 t/s, 530
prefill): decode +12%, prefill +9%.

## Phase 3c wrap-up (2026-09-21)
Kept: Q4_0 draft head (-1.5..2 ms/step), lane balance 27/20 (prefill +9-11%, decode flat), draft-depth policy
(env-gated, off: +12% at 13K but -2..4% at 26K on the probe text). Ruled out with numbers: target backend sampling,
token_embd on the card, HIP-graph replay (0.5 ms), launch gaps on mainframe (97.7% interior busy), ingest/draft
merge (checkpoint order). Step budget now: ~106-108 ms of which ~65-70 is kernel floor (APU expert reads at n=3,
card dense), ~9 host chain, the rest crossings. Decode stands at 23-25 t/s at n-max 2 (26+ at 13K with n-max 4);
27 needs a physics change (fewer routed experts or lower expert quant on the APUs), which is the user's call.

## Prefill scoping for 800 t/s (2026-09-21)
Steady state after the rebalance: cadence set by the longer lane (~1.5 s per 1024-token ubatch at 27/20), i.e.
~650 t/s peak, 577 observed at 25.8K. 800 needs the long lane at ~1.2 s per ubatch. Mainframe's lane is 99.6%
kernel time; its 13K profile (rocprofv3 2026-09-13, shares sound, durations not): expert GEMM 37% (MMQ q4_K/q5_K
at J=32, ~17% of gfx1151's int8 peak at n=1024), flash attention 13.5% (dense (4,8) tiles for kv<10K, sparse
(1,32) above), dense q8_0 GEMM 13%, MoE reduce/ids/quantize 9% (quantize_mmq_q8_1 5%, moe_weighted_reduction
2.6% at ~60 GB/s), element-wise 19% (part removed since: conv-state concat, hc prologue), GDN 5%, indexer 2.6%.
Candidate work, in order of yield per effort:
  P1. Expert GEMM on gfx1151: MMQ q4_K/q5_K at J=32 is the 37%; the RDNA3.5 WMMA path is whitelisted for q4_K
      J=16/32/48 and q5_K J=32 with activation prefetch (8.47 ms per 1.36 GB call at n=1024). Targets: J=64 tiles
      with the register-staged prefetch (LDS 64 KiB per CU limits residency to 1-2 blocks), a q5_K weight-tile
      stage, and the moe reduce/quantize pass folded (9%). A 30% gain on this class is ~0.2 s per ubatch.
  P2. Attention: dense FA at (4,8) for kv<10K and sparse (1,32) above; the sparse tile's DKQ=512 spills (256 VGPRs);
      a 16-column config that does not spill or K/V tile sharing across a query group (see sparse-fa notes).
  P3. Element-wise remainder and the quantize_mmq_q8_1 layout kernel (5%): fusion into producers.
  P4. Dense q8_0 GEMM 13%: MMQ J=128 at 22-26 TFLOPS on gfx1151; modest headroom.
  P5. The card lanes: mainframe's card runs dense + KM expert layers; its dense GEMM efficiency (rocBLAS f16
      fallthrough at 6.7% of peak is 1.2% of time) is not the lever; the card is not the long lane.
Reaching 800 is P1 + P2 landing most of their estimates plus the balance re-tuned after each (the knee moves as
mainframe's lane shortens).

## P1 log: q4_K/q5_K expert GEMM on gfx1151 (2026-09-21, isolated on gibson's APU, test-backend-ops TBO_GLM)
Baseline (288 experts, 8 used, q4_K 2048x4096 / q5_K 4096x2048): n=128 5.85 / 7.2 ms, n=512 6.8 / 8.7, n=1024
8.48 / 10.07, n=2048 ~15 / 16.8, n=4096 28 / 29.8. The weight read is 1.36 / 1.66 GB per call regardless of n (each
expert's rows once per column tile), i.e. a 5.3 / 6.5 ms floor at 256 GB/s: at n=128 the kernel is at 91% of
bandwidth, at n=1024 at 63% (the column hint picks J=32 for ~29 tokens per expert; the extra 3.2 ms is compute at
that tile), at n=2048 (J=64) the per-token cost is 7.1-7.5 us vs 8.3 at 1024. So the class's software headroom at
ub=1024 is ~25-35% (8.5 -> 5.5-6.5), and ub=2048 halves the bandwidth term per token but pays a J=64 tile.
Experiments: I=128 tiles at J=32 (halve activation re-reads): n=1024 8.48 -> 9.15, no; J=64 and J=128 in the
RDNA3.5 prefetch whitelist: n=2048 15.4 -> 14.6 (q5_K 16.3 -> 16.0), n=4096 27 -> 61 (J=128 with the staged tiles
spills), so J=64 only; GGML_CUDA_MMQ_MOE_J_FACTOR=0.5 (J=32 at n=2048): no gain, and J=16 at n=1024 costs 20%.
In situ, ub=2048 vs 1024 at LOCAL=27 KM=5 with the J=64 whitelist: 12.7K 467 vs 509, 25.8K 536 vs 557 - still a loss,
and the arithmetic says why: with two lanes and N units the pipeline's tail costs about one unit, so 12.6 units of
2048 cost (12.6/2 + 1) x 2u = 14.6u against (25/2 + 1) x u = 13.5u for 1024-token units (+8%), which outweighs the
expert class's -14% per token at n=2048. A tapered ubatch schedule (large units first, the last one or two small)
would keep the tail at ~0.5u and let 2048 pay; that is a client-side change in the ubatch splitter, no wire change
(LLAMA_UBATCH_TAPER=1, in llama-memory-hybrid).
**8 waves per tile (256 threads at J=32/64, same 64-row tile) - INVALID, see below:** q4_K n=1024 8.45 -> 6.96 ms, q5_K 10.07 -> 8.78;
n=2048 q4_K 14.6 -> 7.97, q5_K 16.0 -> 10.16 - the first real kernel win of P1 (-18% / -13% at ub 1024, and at
ub 2048 the expert class costs 3.9 us per token against 8.3 before). Committed as the RDNA3.5 table default.
In situ with the tiles on gibson only (mainframe still on the old table), LOCAL=27 KM=5, 12.7K / 25.8K: ub 1024
plain 524 / 565 (509 / 557 before the tiles: gibson is not the long lane, so little shows yet); ub 1024 + taper mode 1
520 / 543; ub 2048 + taper 460 / 543. The half-unit-first clause of the taper costs more than it saves; mode 2
(equal final pair only) is in the tree for the re-test once mainframe has the tiles, which is when ub 2048 can be
judged at all (its expert class is 3.9 us/token there too only after the rebuild).
Extending 8 waves to the other Q4_K/Q5_K rows (J=16, 48, 80-128) and Q6_K J=32/64 FAILS correctness (TBO_GLM_EVAL:
sentinel mismatches, i.e. an out-of-bounds write, at q5_K n=128 and q4_K n=1024) although the perf looked good
(n=4096 27 -> 10 ms); those rows are not committed and the failing tile/wave combination is to be found before
any of them is used. Re-verification of the committed J=32/64 rows FAILED too (ROCm1: q4_K/q5_K at n=1024/2048
ERR ~3e3, sentinel mismatch; the standard suite's q4_K n=17 case as well; ROCm0's RDNA4 table untouched, 923/923):
the 8-wave timings above are a wrong kernel, not a win - the RDNA3.5 mma path maps waves to rows for 4 waves on a
64-row tile, and 8 waves need the split-J mapping the fork only has for Q8_0 J=128. The experiment scripts checked
perf without gating on the correctness line (a process failure: the t256 chain's decision read the timing only).
3abd6f26a is reverted in the tree; the in-situ p4b numbers were taken with the wrong kernel on gibson's APU and are
void. The lesson repeats 3b's: a kernel change is not a result until the correctness suite AND the greedy gate pass.
**Corrected 8 waves (split-J mapping ported to the K-quant mma dot, 923/923 on both parts, 2244/2244 standard):**
q4_K n=1024 9.20 ms (baseline 8.50), q5_K 10.75 (10.07), n=2048 15.0 / 16.8 (15.4 / 16.3 with J=64 whitelisted).
A loss at 1024, flat at 2048: the "win" was entirely the wrong kernel doing less work. The split-J generalisation
stays in the tree (inactive: no table row uses 8 waves at I=64 except Q8_0 J=128) and J=64 stays in the prefetch
whitelist (a real 5% at n=2048). Table-level knobs on the RDNA3.5 MMQ are exhausted: I=128, 8 waves, J=64/128 and the
column-hint factor all lose or tie at n=1024 against the J=32/4-wave config that sits at 63% of the weight-read
floor. What remains for P1 is profiler-guided kernel work (why the K loop stalls at occupancy 2 with 64 KiB LDS:
bank conflicts, the load->sync->compute serialisation, the ldmatrix pattern), days rather than hours.
Counters on the q4_K J=32 kernel at n=1024 (rocprofv3 --pmc, gfx1151): occupancy 31% (LDS-limited, 2 blocks x 4 waves
per CU), LDS bank-conflict stall 15.5% of GPU time, SQ_INSTS_LDS 8.3e7 vs VALU 5.3e8 per dispatch; VALUBusy /
MemUnitStalled / SQ_WAIT_INST_ANY do not collect on this part. First follow-up, a 78-int tile pitch with 8-byte tile
loads (16-byte-aligned pitches repeat banks every 8 rows): correct once the host sized the LDS from cc, and WORSE -
conflict stall 35.6%, kernel +1.4% (8.59 vs 8.47 ms); guarded by MMQ_RDNA35_STRIDE_78 and left off. Second lead from
the store side: load_tiles writes each row's nibble-split layout so that lanes 0-7 and 16-23 hit the same eight
banks in one instruction (ints 0-7 and 32-39); swapping the low/high store order for the upper half-wave keeps the
layout and removes that 2-way conflict. Measured: bank-conflict stall 15.5% -> 7.9%, kernel time unchanged (8.46 vs
8.47 ms at n=1024; 923/923 + 2244/2244 pass). Kept (harmless, and the conflicts were not on the critical path), and
it says the kernel is bound elsewhere: occupancy 31% with 4 syncs per K iteration is the next lead - an I=32 tile
(half the LDS per block, 4 blocks per CU) needs the split-J mapping generalised to I/16 row groups.
I=32 tiles (split-J generalised, 4 waves = 2 row groups x 2 J groups, 4 blocks per CU): correct, conflicts 5.6%, and
slower at n=1024 (q4_K 9.36 vs 8.46, q5_K 10.76 vs 10.07): the doubled activation re-reads cost more than the
occupancy buys. P1 table so far (q4_K 2048x4096 x288/8, n=1024, gfx1151, ms; all correctness-gated):
| variant | ms | note |
|---|---|---|
| baseline J=32 I=64 4 waves | 8.46 | 63% of the weight-read floor |
| I=128 | 9.15 | fewer activation re-reads, worse |
| 8 waves (correct, split-J) | 9.20 | more waves per tile, worse |
| J=64 whitelisted | 8.46 | +5% at n=2048 only, kept |
| column-hint factor 0.5 | 10.20 | J=16, worse |
| pitch 78 + 8-byte loads | 8.59 | conflicts 35.6%, worse |
| store-order swap | 8.46 | conflicts 15.5 -> 7.9%, kept |
| I=32 (4 blocks/CU) | 9.36 | occupancy up, worse |
The tile/wave/pitch space is exhausted at n=1024. Two structural experiments followed, both correct and both
neutral or worse, kept in the tree behind build macros that default off:
| variant | q4_K n=1024 ms | note |
|---|---|---|
| double-buffered activation tile, 2 barriers per iteration (GGML_CUDA_MMQ_Y2) | 8.68 | 28.8 KiB, still 2 blocks/CU; barriers are not the wait |
| two-deep weight prefetch (GGML_CUDA_MMQ_X_DEPTH2) | 8.52 | +20 VGPRs; the weight stream is not latency-bound |

**Ablation (timing-only builds with parts of the kernel deleted, never run against the model).** q4_K 2048x4096 x288/8 on
gfx1151, ms:
| build | n=128 | n=512 | n=1024 | n=2048 |
|---|---|---|---|---|
| real kernel | 5.85 | 6.80 | 8.44 | 14.5 |
| no scale epilogue (raw MMA sums) | - | 6.81 | 8.07 | (spills) |
| constant scales, math kept | - | - | 8.38 | - |
| no MMA | - | - | 7.07 | - |
| loads + LDS stores only, no dot products | - | 6.56 | 7.10 | 8.23 |
| no activation global loads | 5.82 | - | 7.89 | 14.2 |
| no activation loads or stores | 5.84 | - | 8.22 | 14.0 |
Loads-only at n=1024 is 7.09-7.11 for every forced J from 32 to 64 (7.76 at J=16, two tiles per expert), so neither
the column-tile count nor the activation volume sets it. The weight stream is 1.36 GB per call: 232 GB/s at n=128,
~190 GB/s at n >= 512, and the whole tuning surface of the day (barriers, tiles, bank conflicts, epilogue) is the last
1.3 ms above the skeleton. Registers are not the limit: q4_K J=32 uses 155 VGPRs with no spills (q8_0 J=32 spills
29-62 VGPRs on gfx1151, an APU-only lead). SQ counters at n=1024: VALU issue ~28% and WMMA ~15% of SIMD cycles,
LDS ~20%, occupancy 8 waves/CU. What separates n=128 from n=1024 in the skeleton is still open (the same loader,
the same block count at J=48); the next probe would be the ISA of the load issue pattern. At n=2048 (J=64, one
block per CU) the vec_dot is fully exposed: 14.5 vs an 8.2 skeleton, which is the number that decides whether ub=2048
can ever pay. Paused here to regroup (DFLASH for decode, prefill status).

## Sequencing (the user's order: build V3 end to end, then optimise)
Phase 1 - build V3, TCP only, KM=4 (VRAM), both hosts on one commit:
  1a. Client: composite device per endpoint (`RPC<k>[host]`), extra buffer type per additional server device,
      supports_buft true for all of them, scratch hint on the composite compute buffer, boundary-output flag from
      the scheduler, GRAPH_COMPUTE mode flag (protocol minor bump, hello-gated). Default placement rule
      (dense+KV -> device 0, `ffn_*_exps` -> device 1) with `-ot` as the override.
  1b. Server: per-connection ggml_backend_sched over all local backends; unpin scratch intermediates; sched
      compute; copy flagged outputs back. GGML_SCHED_DEBUG on the server once to prove no weight is ever copied.
  1c. Gates: builds on both parts; greedy_ref.sh token identity (a828e28289899da6); v3 KM=4 3-rep in the harness
      with mainframe's uprobes (expect 2 calls/token per node, the 17.6 + 9.5 ms of per-call cost gone).
Phase 2 - end to end:
  2a. Step 0.5 - DROPPED after measurement (see above): the per-token re-split is 0.65 ms on the server.
  2b. Prefill through the server sched (ub 1024; the two-lane pipeline's single-remote-device assumptions listed
      and fixed as needed), KM sweep within VRAM, production candidate decision.
  2c. N-node readiness: the placement rule and buft naming exercised with a second endpoint on paper (or gibson's
      own rpc-server as a fake third node) so nothing is two-node-shaped.
Phase 3 - optimisation, on the V3 layout, judged against the physics (card dense floor 0.28 ms/layer):
  3a. Small-GEMV mapping on gfx1201 (split-K / rows-per-wave for M or K under ~10 MB of weights; hc_fn, shexp,
      f_b/g_b first) - general to the card.
  3b. Launch fusion on the dense path (hc pre/comb/post chains, quantize-once for wo/down/f_b/g_b, KDA concats,
      router softmax+topk) - both hosts.
  3c. Gibson's share (host loop, draft decodes, local split gaps) and tokens per step (draft depth).
  3d. hipfire-derived micro-optimisations in mmvq (scheduling barrier, scalar block headers, accumulator chains).

Naming from here: "v3" = this (server sched, intended placement, N-node shape). The old client-split layout is
"v3c"; what runs today is v2c.

## Adversarial review of the week (2026-09-22): what the numbers were, what changed
**The probe was the problem first.** Every decode figure of Phase 3 (25.9 t/s production, 28 with four drafts, the
DFlash tables) came from probe_ctx/probe_tune: summaries of a synthetic ledger, formulaic text the drafter predicts
easily. The user's WebUI read ~21 t/s. `~/bench/glm/probe_real.py` (six everyday prompts, thinking on, the WebUI's
sampling, fixed seeds) reproduces the UI: production MTP 3 drafts = **21.0 t/s at T=0.7, 20.4 at T=1.0**, 2.53
tok/step, 120 ms/step. Seeded runs of one config repeat to 3 decimals; across configs whose sampling differs the
content variance is about +-1 t/s per 2400 tokens, so only 5%+ effects are judged on it. Judge decode on this probe.

**Decode findings, each measured on the real-content probe:**
| change | T=0.7 (2 seeds) | T=1.0 | step |
|---|---|---|---|
| production: MTP 3 drafts, compare | 21.0 / 21.7 | 20.4 | 120 ms |
| lossless rejection sampling (RS), 3 drafts | 22.1 / 22.4 | 22.2 | 120 ms |
| RS, 2 drafts | 24.2 / 24.7 | 24.2 | 95.6 ms |
| RS, 2 drafts, gfx1151 MoE GEMV -> MMQ from 3 tokens (gibson only) | 24.9 / 25.9 | - | 93.2 ms |
1. Verification was sample-and-compare against the draft's argmax: at T > 0 a draft token survived only with the
   target's probability of that one token. RS (Leviathan/Chen): the MTP draft samples from its top-10 at the target's
   temperature (top-p/min-p mirrored; measured neutral), the target accepts with min(1, p/q) and resamples the residual
   on rejection - exactly the target's distribution, more tokens per step. On by default (LLAMA_SPEC_RS=0 disables).
2. The verify graph cost 90.6 / 112.9 / 117.2 ms at 3 / 4 / 5 tokens (medians over ~2.5K steps each): a cliff at 4
   and almost nothing at 5. mul_mat_vec_q_moe gives each (token, expert slot) pair its own warp, so an expert picked by
   two tokens is streamed twice; MMQ (from 5 tokens on gfx1151) streams each distinct expert once, and real text
   shares experts between neighbouring tokens. Random-routing isolation shows no cliff (q4_K x288/8 on gfx1151:
   145/342/499/664/803 us at 1..5 tokens) - the upstream table was tuned on exactly that. The old "17.6 ms per
   verified token is physics" slope was this per-pair cost. RDNA3.5 now hands K-quant experts to MMQ from 3 tokens
   (gibson: 3 tokens 90.6 -> 87.8 ms, 4 tokens 112.9 -> 107.3); mainframe needs the same build for its half.
   The card's per-pair path is at bandwidth (q4_K n=4 236 us = 640 GB/s) and keeps its table.
3. At this acceptance two drafts beat three (the third token costs ~22 ms for +0.34 tok/step). Production: NMAX=2.

**Decode step anatomy (gibson kernel trace, 4-token verify, short context):** 22 x [card 0.85 ms (47 kernels: 0.58
ms of kernels, 0.27 ms of launch gaps) -> APU ~2.1 ms (6 kernels)], a ~42 ms hole while mainframe computes, then
the output head and the drafts (~16 ms of card activity). Gibson's card spends ~19-25 ms per step on small kernels
and their gaps; that and the same on mainframe's card is the next decode lever (fusion of the KDA decode chain).

**Prefill: the card and the APU never overlapped.** Kernel trace of gibson's 25.8K prefill (LOCAL=27, two lanes):
card busy 13.6 s, APU 19.0 s of 38.0 s, both busy **0 ms**; per 1024-token ubatch the card worked ~545 ms (MMQ 23%,
sparse FA 16%, GDN 13%, lightning indexer 12%) and the APU ~758 ms (expert MMQ 87%, weighted reduction 7%) strictly in
turn. The two-lane pipeline only ever overlapped gibson with mainframe. Four lanes (LLAMA_PREFILL_LANES=4): ubatches
go in pairs whose local heads are interleaved split by split while the previous pair is on mainframe. Greedy text
identical to two lanes; card and APU now both busy 41% of each pair; gibson ~1.1 s per ubatch instead of 1.4-1.6,
which makes mainframe the long lane - so layers move to gibson:
| config | 25.8K | 12.7K |
|---|---|---|
| 2 lanes, LOCAL=27 (production until today) | 576 | 522 |
| 4 lanes, LOCAL=27 | 552 | 497 |
| 4 lanes, LOCAL=28 | 589 | 522 |
| 4 lanes, LOCAL=29 | 622 | 548 |
LOCAL=30 pages gibson's host. The MoE weighted reduction ran at ~60 GB/s in situ (2.51 ms per 1024-token layer:
the column-chunk grid read 8 rows 32 KB apart per block in GTT memory) against cache speed in isolation; one block
per token with 16-byte loads runs 0.66 ms in situ, bit-identical (-44 ms per ubatch on gibson's 24 APU layers).
Mainframe (call median 1506 ms at LOCAL=27, 1374 -> 1583 over the prompt) is ~0.7 s longer per ubatch than gibson's
per-layer costs predict for its layers (card 18 x ~19 ms + 6 expert layers x ~12 ms, APU 12 x ~34.5 ms = ~0.83 s);
its rocprofv3 trace (2026-09-22 20:52) is what decides the next prefill step. Its busy counters cannot answer overlap
questions: the R9700's reads 100 whenever work is queued and the APU's is a slow average (retracted: "97.7% interior
busy" and "0% both busy").

## 2026-09-22 (late): server-side placement, prefill 799, and an open history-dependence bug
- **RPC weights flag (d86b94eca, proto 7.4.1).** Mainframe's server-side scheduler never saw WEIGHTS buffers, so for
  its APU-expert layers SwiGLU and the weighted reduction ran on the card: four PCIe crossings per layer of
  [n_ff, 8, n_tokens] intermediates. Fixed by carrying the usage in rpc_tensor.flags. Mainframe per 1024-token ubatch
  1377 -> 827 ms (-40%); server splits per prefill graph 41 -> 21; greedy text identical.
- **MTP prompt ingest inside the pipeline (4d6cb6c26)**, bit-transparent: removes ~2 s after a 25.8K prompt.
- **LOCAL sweep (all fixes, four lanes):** 25.8K prefill 29: 775, 27: 814, 26: 834, 25: 799 (mainframe long again).
  LOCAL=26 KM=6 fits mainframe's card with ~1.5 GB spare.
- **Production (LOCAL=26, 4 lanes, NMAX=2, no taper):** 25.8K 799 tok/s, 12.7K 690 (first request, one-time lane-2
  stall); real-content decode 24.7 / 25.2 tok/s at ~94 ms/step. NMAX=3 now loses (23.2 / 23.7, 111 ms/step); the
  grouped MoE GEMV (34ee0a814) gives nothing in situ.
- **Open bug: prefill depends on earlier requests.** Same 25.8K prompt, same server, first-token logprob moves
  ~0.01 between requests (deterministic given the history, reproduces across servers); with LLAMA_UBATCH_TAPER=3 it
  moves ~0.25 and the top token flips. Present with one lane. NOT in the memory contents (LLAMA_DBG_CLEAR_MEM=1 zeroes
  KV/indexer/recurrent of both contexts before every fresh prompt: identical results). Weights flag and early ingest
  verified transparent. Next suspects: compute-buffer read-before-write (poison buffers with NaN) or partially-set
  graph inputs. The partial-ubatch stall (2.4-2.9 s/prompt) needs another fix than the taper until this is found.

## 2026-09-23: the history-dependent prefill - found and fixed (bf3eb6e8d)
Cause: the rpc-server's own scheduler keeps unpinned intermediates in its buffers and writes back to the client only
BOUNDARY/OUTPUT tensors; the client flagged BOUNDARY only when it made a cross-backend copy. conv_states-26 (first
remote KDA layer, remote split -> small local split -> remote split) and the final `norm` crossed without a copy, so
their readers got an earlier graph's bytes. Elimination order: memory contents, graph reuse, HIP graphs, backend
caches, local and server compute buffers - all negative; GGML_SCHED_ZERO_BUFFERS=2 (client RPC buffers) positive.
After the fix, identical requests agree exactly without the MTP draft. With the draft, request 1 alone differs
slightly (lower-ranked logprobs); requests 2 and 3 are identical. Every zeroing test is negative for it; mainframe's
tracer shows request 1 alone runs decode-sized warm-up graphs first and then re-plans its allocation every ubatch,
so its intermediates sit at different offsets -> alignment-dependent kernel paths -> different f32 summation order.
Read as benign layout rounding, not a leak.
Diagnostics left in the tree: LLAMA_DBG_CLEAR_MEM/_PARTS, GGML_SCHED_ZERO_BUFFERS=1|2, LLAMA_DBG_NO_CKPT_SAVE.
- **2026-09-23 follow-ups.** LLAMA_UBATCH_TAPER=3 is production (post-fix it is exact; 12.7K 690 -> 724, 25.8K 799 -> 805
  tok/s, second-request acceptance 0.83). Splitting a one-unit batch (~1K tokens of a tool-call turn) into halves,
  paired or as two pipelined units, LOSES (987 tok: 374 -> 356 / 315 tok/s): the halves are new shapes every request,
  each paying the remote alloc-size round trip, and a pair's remote work starts only after both heads. Follow-up-turn
  acceptance (0.74 -> 0.64 -> 0.62 over three chat turns) is content, not state: replaying turn 3 as a fresh request
  gives the same 0.61.

## 2026-09-23 (afternoon): items 1-2 of the plan, and a "regression" that was mainframe's card
- **Item 1, `--checkpoint-min-new N` (c4e2f07c7):** no user-message checkpoint/prompt split for turns shorter than N new
  tokens (production 256); near-end checkpoints only at the tail offsets. Short tool turns 543-812 -> 440-573 ms TTFT.
- **Item 2a, view routing (9966d0805):** a full-size view of a crossing tensor reads a view of the root's copy; remote
  inputs per prefill ubatch 16 -> 12, 128 -> 64 MiB. Client-side it was measured at 800 t/s / 107.8 ms/step against
  mainframe on d86b94eca (vr1). GGML_SCHED_NO_VIEW_ROUTE=1 restores the old path; on the local Qwen composite the
  server-side path is neutral (dev 2620/2699, novr 2565/2689, host 2667/2706, both 2665/2684 t/s prefill; decode 63-64).
- **Item 2b, GGML_OP_KQ_MASK_BUILD (789f6d754 + 39c35d363, proto 7.5, GGML_OP_COUNT 103):** the causal mask and the
  indexer pool-select/candidate masks are built on each device from per-cell/per-query i32 position arrays; no mask
  bytes cross the host link or card<->APU. Correct (greedy hash 57fc9097aea5eef6 identical to host masks); mainframe's
  scheduler dump shows the three ops on its card next to their FLASH_ATTN consumers, split layout and per-call timing
  identical to host masks. Two fixes after the first cut: the graph callback must pin into the scheduler of the graph
  being built (`sched_build`; prefill lanes have their own, and the per-device instances had collapsed onto one that
  was then shipped as a graph input), and the hybrid memory input must delegate to the attention input's set_input
  (asserted on the unallocated host mask at warmup). LLAMA_NO_MASK_DEVICE=1 disables.
- **The regression that was not code.** With mainframe rebuilt to 789f6d754 the gate went 800 -> 768 t/s and 108 -> 115
  ms/step (mainframe compute +150-200 ms/ubatch, decode MODEL call 37.5 -> 53.4 ms; host masks equally slow). Mainframe
  re-ran the 09-21 isolated q8_0 GEMV shapes with the EXACT 09-21 binary and got the new, slower numbers (24x16384
  3.89 -> 4.66 us, 2048x1024 7.83 -> 9.95; the big 12288x4096 only +5%): its R9700 lost 17-27% on launch-bound
  kernels across the afternoon's reboot + 17-minute power-off, with clocks, temps, power, PCIe AER, ASPM, packages,
  firmware, host governor and its APU all unchanged. gibson's card, with the same amdgpu.lockup_timeout active, is at or
  below the 09-21 numbers (3.79 / 11.61 / 14.56 / 7.67 / 55.01 / 4.38 us), so the boot parameter is not it. Lesson:
  after any reboot, re-run the six-shape GEMV probe on both cards BEFORE reading a two-host number as a code change.
  The item-2 gate (expect >= 800 t/s, ~108 ms/step with masks gone) waits for mainframe's card.

## 2026-09-23 (evening): mainframe's card was pinned to perf level high; items 1-2 closed
- **The slowdown's cause (Mainframe session):** `gpu-perf-high.service` on mainframe wrote `high` into
  card1's power_dpm_force_performance_level at boot. It was written for the iGPU when that was card1; since the R9700
  went in, card1 is the R9700, and on the R9700 a forced `high` makes launch-bound kernels 20-27% slower than `auto`
  at the same clocks under load (24x16384 4.68 vs 3.86 us, reversible both ways; big SGEMM -3%; the APU gains nothing
  from `high`). Service disabled, both devices on auto; kernels back to 09-21 (3.85 / 11.87 / 14.68 / 7.79 / 56.52 /
  4.50 us). gibson runs nothing that pins a card by number (checked). Why 09-22 was fast with the service enabled is
  unknown (probably something had reset the card to auto during that boot).
- **Gate on the recovered mainframe, LOCAL=26 four lanes taper 3 NMAX=2 (25.8K prefill / ms/step / 12.7K greedy):**
  host masks 838 / 102.3 / 749; device masks 873 / 102.7 / 760; device masks + pinned flat norm (1d4e23b71) 892 / 103.3 / 776.
  Mainframe per-ubatch 1006 ms (ref 1011), decode MODEL 36.8 (ref 37.5); device masks save ~9 ms/ubatch on its side,
  the rest of the +4% is the mask bytes no longer crossing. Greedy hash 57fc9097aea5eef6 in every device-mask run.
- **Review fixes (Opus 5.5 review of items 1-2, 9a7b81a7f):** view routing only for pure view ops with element-size
  nb[0] (a transpose moving dim 0 would have been rebuilt with wrong strides; an in-place op's result would have
  reused a stale root copy); the first turn keeps its system/user checkpoint regardless of --checkpoint-min-new.
  Open from the review, not done: share pos_kv with the kpool pos_at array (same values, two uploads); the device
  path adds ~6 bytes per KV cell per decode step, measured neutral.
- **What crossed the link was not what the routing commit thought:** the second 64 MiB tensor at the layer-26
  boundary was the hyper-connection flat norm's RESULT, computed on gibson because the op has no weights. Pinned to the
  layer's device (hc_flat_norm) it reads the routed view of the residual instead: 128 -> 64 MiB per ubatch.

## 2026-09-23 (night): the overhead pass, and the first fusion wave from the Opus workflow
**Expert parallelism across hosts is refuted by the link RTT**, not by implementation: splitting each layer's experts
between the two APUs saves at best ~20 ms of the ~40 ms of expert streaming per step, but needs 90 crossings per
step; at the measured 0.17 ms ping RTT that is a 15 ms floor with a transport that does not exist, and 40-60 ms with
the RPC call tax as measured (0.4-0.7 ms per call). Intra-host splits do not fit VRAM. Shelved.

**Where a decode step actually goes (4.65K ctx, HIP graphs on, quiet box; LLAMA_SPEC_TRACE=1 timers):** loop
iteration 102.1 ms = target llama_decode 95.5 + draft path 3.5 (ingest 0.6, dec0 1.0, smp0 1.8 incl. the GPU wait,
dec1 0.1) + post_decode 0.9 (sample+accept 0.6) + speculative process 0.75 + ~1.4 other. Inside the 95.5: mainframe
MODEL 37.7, crossing ~2.7, gibson local ~54 = card ~28-33 ms of kernels across ~1150 launches (~1000 of them under
40 us: 14 ms with graphs off) + APU 33 ms of expert GEMVs at isolated speed (1.54 ms per layer at n=3). The host side
is ~7 ms, not 25: the earlier "25 ms of overhead" was the tiny-kernel time on the card plus the crossing.
Op-timer method: GGML_CUDA_TIME_OPS=1 GGML_CUDA_TIME_OPS_TOP=400 GGML_CUDA_TIME_OPS_NAMES=1 needs
GGML_CUDA_DISABLE_GRAPHS=1 (replayed graphs record no events) and GGML_CUDA_MMVQ_NO_TALL=1; decode-only figures are
the delta between two flushes (every 200 graphs) after the last prefill-sized batch. CPU builds on the same box
slow the APU 2.6x (shared LPDDR5X bandwidth) and inflate every wall number: measure on a quiet box.

**The crossing, split by mainframe's per-message tracer (2.7 ms per step):** 15 set_tensor messages (285 KiB),
handler time 0.02 ms, 1.1 ms of receive pacing between my messages, 1.0 ms of client work before the first send, a
0.5 ms second graph call, and 1.14 ms of deserialize + alloc because every call was a fresh graph_compute: two
graphs alternate per step (a 6-node recurrent-state gather for layer 26 that the node order emits before layer 25's
tail, then the model) through a one-slot uid cache. Fixes landed (ca77cbfa7): host-resident inputs are sent before
the produced residual (they now arrive while the local devices compute; only the residual is on the critical path);
kpool_pool_bias_f16 is a host input instead of a device cast that cost a synchronous device -> host -> remote copy
(110-360 us) per step. Parked: folding the state gather into the model graph via ggml_build_forward_expand at each
layer end (LLAMA_LAYER_EXPAND=1) makes one remote graph per step but the kpool select_device map then collapses
gibson's and mainframe's mask instances (a DUP of layer 3's device sel mask arrived in mainframe's graph as a
bufferless op leaf behind a VIEW; its allocator asserted). The rpc-server now refuses that with a named error
(fc6423eb6). Next: a 2-slot uid cache on both ends (~1.1 ms/step) or fixing the map key.

**Single-launch top-k (b9d45eb86):** the DSA indexer's pool select (k=128 of a few hundred to ~1.6K pool scores, 1-8
rows) went through argsort/radix + pool alloc + memcpy2D, ~85 us per call, 11 calls per step per host; one 512-thread
block per row with an in-block radix select is 48 us in situ (0.9 -> 0.5 ms per step per host; still fixed-cost
bound, a second pass could halve it again).

**Fusion wave 1 (Opus 5.5 workflow: survey + 4 implementers in worktrees + 4 adversarial reviewers):** survey
ground truth 46 launches per KDA layer, 59-70 per DSA layer on the card. Merged: KDA conv tail (c237ba96c: weight
concats + ssm_conv + silu + Q/K l2_norm, 39.5 -> 9.7 us isolated, 4 launches/layer x 34 layers), KDA conv-input
assembly (4e19f0360: q/k/v concats + state concat + rollback-slot copies, 61.6 -> 43.1 us, 5 launches/layer), and
the hyper-connection boundary norm + q8 copy (dad429ac7: bit-identical; its hc_post absorption phase never fires
under real allocation, harmless). Not merged: peer-copy-batch (never runs in production decode: the eager-copy path
is off with lanes at small ubatch). In situ on gibson's card (op timer, graphs off): 1153 -> 969 launches per step,
30.9 -> 28.8 ms; both KDA fusions fire in production (CONCAT +fused8 / +fused13 in the op table). Greedy 12.7K hash
57fc9097aea5eef6 unchanged. Survey flag for the next wave: the shared-expert FFN on the card runs AFTER the card's
~1.8 ms wait for the APU's routed experts; scheduling it before the wait would take ~2 ms per step off gibson's
critical path (a split-order change, not a fusion).
- **Gate on fc6423eb6, both hosts (2026-09-23 22:00):** 25.8K prefill 892 t/s, 102.2 ms/step (26.9 t/s at 0.86
  acceptance), greedy hash 57fc9097aea5eef6 unchanged; the afternoon gate on d0028207a was 892 / 103.3. The 4.65K
  gibson-only probe went 102.1 -> 98.6 ms/step; the smaller gain at 25.8K is mainframe's share (its per-step split
  pending from its tracer).
