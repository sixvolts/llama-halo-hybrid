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
pipeline finding from 09-13 holds). Decode varies with ub at fixed KM (21.87 / 22.69 / 23.21 for ub 1024 / 512 / 2048)
by more than the harness noise floor (~0.3) although decode should not depend on ubatch - unexplained, wants a
back-to-back ub1024 vs ub2048 A/B at regroup before anything is built on it. KM's decode effect is ~+0.2 t/s (as
Phase 0's ~1.1 ms/layer predicted). VRAM: KM=5 and ub2048 each consume the card's headroom; both together do not fit
until the client's scratch stops being real memory on the card (the lazily-backed scratch buft from the design).

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
