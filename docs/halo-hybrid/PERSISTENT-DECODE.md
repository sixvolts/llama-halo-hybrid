# Scoping: a persistent decode kernel for ggml on RDNA

Status: scoping, 2026-09-16. Nothing here is built. The measurements are real; the design is a proposal.

## The problem it solves

At decode the GPU on a Strix Halo is idle most of the time. Measured on this tree, Qwen3.8-Flash-Next on the iGPU
alone (rocprofv3, 4.9K prompt then 128 tokens, MTP head, HIP graphs on):

| | per token | busy |
|---|---|---|
| kernels launched | ~1,400 | |
| GPU busy in the decode window | | 37% |
| of which dense GEMVs (`mul_mat_vec_q`) | 9,000 per 3 s | 46% of busy time |
| separate `quantize_q8_1` launches | ~130 per token | 1.5% of busy time |

GLM-5.3-Flash across two hosts shows the same shape at a smaller scale: 60-87 kernels per layer, one hipGraphLaunch
per 20-layer graph on mainframe (confirmed with uprobes), and the replayed graph still spends ~10 ms of its 58 ms on
inter-node gaps. Fusing element-wise chains (`ewchain.cu`) removed 16 nodes per layer for ~1 ms per step: the nodes
that are cheap to fuse are the ones with the smallest gaps.

Upstream's multi-stream graph optimisation (`GGML_CUDA_GRAPH_OPT=1`, independent branches forked onto extra streams
inside the captured graph) was measured on this exact setup and changed nothing: decode 38-43 tok/s with the draft
head either way, prefill identical. The graph is a chain, not a tree; there is little to fork.

The gap is hardware dispatch between dependent kernels inside a replayed graph: wave drain, end-of-kernel release
(cache writeback), start-of-kernel acquire, dispatch. Fusion reduces the count; it cannot change the cost per boundary.

## What a boundary costs, measured

`scratchpad/persist/sync.hip`, cooperative launch, 2 blocks per CU, 2,000 iterations:

| boundary kind | gfx1151 (iGPU) | gfx1201 (R9700) |
|---|---|---|
| stream launch of an empty kernel | 1.79 us | 3.64 us |
| node inside a replayed hipGraph, empty kernel | 1.74 us | 3.36 us |
| node inside a replayed graph, real decode kernels (derived) | ~6 us | |
| cooperative-groups `grid.sync()` | 0.49 us | 0.85 us |
| device-scope atomic arrive/release hand-off (no full barrier) | 0.38-0.45 us | 0.74 us |

Cooperative launch works on both parts (`hipDeviceAttributeCooperativeLaunch` = 1, grid sync verified with 40 / 64
resident blocks). So a dependency step inside one resident kernel is 4-12x cheaper than a kernel boundary, and it
is a point-to-point wait rather than a global barrier, so independent work overlaps.

## The design

One cooperative kernel per decode step (or per graph split on the multi-device layouts), resident for the whole
step, executing a **static task list** derived from the ggml graph at capture time. This is the "megakernel"
pattern (Mirage Persistent Kernel, the Hazy Research low-latency Llama decode) done inside ggml's backend:

1. **Task list.** When the CUDA backend would capture a HIP graph for a split, it instead compiles the split's
   nodes into a device-side array of tasks: `{op, args, n_workgroups, deps[]}`. One ggml node becomes one task
   (a GEMV over N rows is one task with N/rows-per-block work items). Dependencies are the node's src edges, as
   counters: a task is runnable when its `deps_done` counter equals its dep count.
2. **Scheduler.** Every resident workgroup loops: claim the next work item of the lowest runnable task (one
   atomic on a per-task cursor), execute it, and when the task's last item retires, increment the dependents'
   counters and release with a device-scope fence. Waiting is a spin on an acquire load with `s_sleep`, the
   0.4 us hand-off above.
3. **Op bodies as device functions.** Each op the decode graph uses becomes a `__device__` function taking a
   `(task, item)` pair: the GEMV inner loops (MMVQ's dot4 path, or the activation-ingesting bf16 variant),
   rms_norm, the element-wise chain, rope, concat/cpy, get_rows, the attention vector kernel, the DeltaNet step,
   top-k/MoE routing, the MoE weighted sum. These already exist as `__global__` kernels; the work is refactoring
   their bodies to take an explicit block index instead of `blockIdx` and to share one launch configuration
   (256 threads, no static LDS beyond a common scratch).
4. **Same-input GEMVs become one task** with one weight table (the six KDA projections, the five DSA ones, the
   shared expert's gate and up), which also removes the separate quantize pass when the GEMV reads the activation
   directly.
5. **Fallback.** Any node type the persistent path does not implement ends the persistent region; the split is
   executed as today from that node on. So it can land op by op, starting with the ~10 ops that make up 95% of
   decode nodes, with correctness checked against the graph path per node type.

What stays outside: prefill (large batches are compute-bound and MMQ/WMMA kernels already saturate the GPU;
the persistent kernel is a decode and verify-batch device), the RPC hop, and sampling (a device-side sampler is a
separate, smaller item that removes the logits copy).

## What it is worth

Decode window on the iGPU: 1,400 boundaries per token at ~6 us = ~8 ms of a 26 ms token. Replacing them with
0.4 us hand-offs and letting independent tasks overlap recovers most of the 63% idle: the ceiling is the GPU's
busy time, ~10 ms per token, i.e. up to ~2.5x on serial decode for Qwen3.8 on the APU (26 -> 50-60 tok/s serial;
halogen, with ~500 launches per token, is at 37.6). On GLM two-host the same treatment applies per split on gibson
(45 splits, ~1,400 kernels per step) and, once mainframe has the same tree, to its 20-layer graph: the ~10 ms of
gaps there plus the split boundaries on gibson, roughly 15-20 ms of a 127 ms step.

The MTP draft calls (three `llama_decode`s per step) collapse naturally: a draft step is just more tasks in the
same resident kernel, if the draft context's graph is compiled into the same task list. That is a second-order
win of ~5 ms per step on GLM.

## Risks and unknowns

- **Cache coherence between tasks.** A kernel boundary implies a full L2 writeback/invalidate; inside one kernel,
  the producer's release and the consumer's acquire must cover the same data. On RDNA3/3.5 the L2 is the point of
  coherence for device scope and the acquire needs an L0/L1 invalidate (`buffer_gl0_inv`, `buffer_gl1_inv`), which
  the atomics with agent scope already emit. Verify with a producer/consumer test that writes a full activation and
  reads it from another workgroup, at size, before anything else.
- **Register and LDS budget is the union of all op bodies.** The resident kernel's occupancy is set by its worst
  op. If the attention or DeltaNet body needs 200 VGPRs, every task runs at that occupancy. Mitigation: two or
  three kernel "shapes" (a GEMV-class kernel and a heavy-op kernel) with a boundary between them, still far fewer
  than 1,400.
- **Forward progress.** Spinning workgroups must not starve the producers they wait on: with the grid sized to
  what is co-resident (`hipOccupancyMaxActiveBlocksPerMultiprocessor` x CUs: 160 on the iGPU, 256 on the R9700
  at 256 threads) and the task order topological, every task a workgroup waits on is already claimed by a running
  workgroup. Cooperative launch guarantees co-residency or fails the launch.
- **ggml's graph changes shape.** The verify batch alternates 1/2/3 tokens and the KV length grows. The task list
  is keyed like the HIP graphs are today (per split, per shape); a shape not seen before compiles a new list. The
  compile is a host-side pass over a few thousand nodes, well under a millisecond.
- **Multi-device splits.** The persistent region is per backend context; the split boundaries between ROCm0 and
  ROCm1 stay as they are. On gibson's GLM layout that is 45 persistent launches per step instead of 1,400 kernel
  launches, still a large reduction; whole-layer placement would take it to 3 once rocBLAS is out of the picture.
- **Debuggability.** A hang inside a resident kernel is a GPU hang. The task list, cursors and counters live in
  a host-visible buffer; a watchdog thread reads them on a timeout and prints which task stalled and on what.

## Staging

1. **Coherence and scheduler skeleton** (2-3 days): the task list format, the claim/retire/wait primitives, a test
   that runs a synthetic 1,000-task DAG of copy kernels and checks results and timing on both parts.
2. **GEMV class** (1 week): MMVQ's dot4 body and the activation-ingesting variant as device functions with the
   grouped weight table; rms_norm and the element-wise chain as prologue/epilogue tasks. Measured target on
   Qwen3.8 iGPU-only serial decode: 26 -> 35 tok/s with the trunk GEMVs in the persistent region and the rest
   falling back.
3. **MoE and attention** (1-2 weeks): top-k routing, expert GEMV with the ids table, weighted sum, the vector
   attention kernel, the DeltaNet recurrence step. Target: the whole decode step resident, 45-55 tok/s serial.
4. **Draft and sampling** (1 week): the MTP graph in the same list, argmax/top-k on device.
5. **GLM two-host** (few days): per-split persistence on gibson, then mainframe after its rebuild.

Four to six weeks of work with measurable gates at each stage, all in `ggml-cuda` behind an env switch, no model
changes. Stage 1 is cheap and decides whether the coherence and scheduling primitives behave on RDNA the way the
microbenchmark says; nothing after it should start until that test passes at size.

## Stage 1 result (2026-09-16): the primitives work, at one workgroup per WGP

`docs/halo-hybrid/persist/pk_test.hip` (build: `hipcc --offload-arch=gfx1151 --offload-arch=gfx1201 -O2 -DPK_SLEEP=8`;
run: `pk_test <floats per row> <rows per task> <layers> <blocks per CU, or -N blocks> <mode 1|2|3> <groups> <block size>`).
A synthetic decode-shaped DAG (per layer: norm, four independent GEMV-like tasks, a fan-in sum; 6 tasks per layer),
every task reading and writing activation-sized rows, checked against a CPU reference, run as one cooperative
kernel and as one kernel per task in a replayed hipGraph. Results, best of 20, `tick` = per task:

| DAG | gfx1151 persistent | gfx1151 hipGraph | gfx1201 persistent | gfx1201 hipGraph |
|---|---|---|---|---|
| 961 tasks, 32 rows x 16 KB (launch-bound) | 4.52 us | 5.20 us | 3.32 us | 6.43 us |
| 241 tasks, 256 rows x 16 KB, 4 MB per task (bandwidth-bound) | 26.2 us | 31.6 us | 10.3 us | 19.3 us |
| scheduling floor, no work | 1.84 us | (empty node 1.74) | 2.51 us | (empty node 3.36) |

Checksums match the CPU reference exactly (max rel err 1e-6, f32 summation order) on both parts, so release/acquire
at agent scope is sufficient for producer/consumer visibility inside one kernel on RDNA3.5 and RDNA4.

What it took to get there, each one measured as a wall before it was fixed:

1. **Every loop condition must be wave-uniform.** A `break` on a per-lane value (the abort flag read by all lanes,
   a shared-memory broadcast the compiler cannot prove uniform) makes the structurizer turn the loop into exec-mask
   bookkeeping and the schedule falls apart (-O0 worked, -O1/-O2 stalled). Read by lane 0, `readfirstlane`, then branch.
2. **The wave is the wrong scheduling unit; the workgroup is right, and it should be the whole WGP.** Per-wave
   claims cost one returning same-address atomic each (~45 ns serialized): 512 waves discovering an empty task cost
   23 us. Per-block static striding (item = blockIdx, += gridDim) needs no claim at all; one retire atomic per block.
3. **Pollers must be few.** Wake-up latency after a release grows with the number of polling blocks: 40 pollers see it
   within 0.3-7 us, 160 pollers within 12-40 us, regardless of sleep length, two-level polling, or acquire vs relaxed
   loads. One 1024-thread block per WGP gives full wave occupancy with one poller per WGP: that is the configuration
   in the table. More blocks per WGP are always slower.
4. **One cache line per task state** (padded to 128 B); adjacent tasks' atomics otherwise invalidate the polled word.
5. **Items must be balanced against the grid.** 64 rows over 20 blocks is 4 rounds with a 20%-full tail and the APU
   lost to the graph (36 vs 29 us); 256 rows is 13 rounds and it wins (26 vs 32). Real GEMVs have thousands of rows.
6. The body needs its own memory-level parallelism (8 loads in flight per lane); a wave-per-row loop was
   latency-bound at 14 us per 32 KB row.

Not needed after all: release/acquire variants (fence-based vs builtin atomics measure the same), spin backoff length,
group-level polling hierarchies (no effect once there is one poller per WGP).

Operational: a timed-out foreground test is backgrounded by the tool harness, not killed; the first stalled run spun
at full occupancy on the iGPU for 42 minutes and stalled production decode and every new process's first copy on
that device. The test now carries a host-settable abort flag and a 10 s watchdog on its first launch.

**Stage 2 go.** The scheduler is 1.2x (iGPU) to 1.9x (R9700) ahead of the graph on balanced bandwidth-bound tasks
and 1.15x / 1.9x on the launch-bound DAG, with a 1.8-2.5 us floor per task. Next: MMVQ's dot4 body and the
activation-ingesting variant as `(task, item)` device functions with the grouped weight table, rms_norm and the
element-wise chain as prologue/epilogue tasks, measured on Qwen3.8 iGPU-only serial decode against 26.2 tok/s.
