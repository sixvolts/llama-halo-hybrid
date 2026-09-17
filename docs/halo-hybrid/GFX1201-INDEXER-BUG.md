# GLM-5.3-Flash: f32 indexer projection aborts on gfx1201 under whole-layer placement

**Status 2026-09-17: reproducible, characterised, NOT fixed. Production is unaffected — see the workaround.**

## Symptom

```
ggml_cuda_compute_forward: MUL_MAT failed
  dst  node_70                     f32 ne=[32 9 1 1]    buft=ROCm0
  src0 blk.45.indexer.proj.weight  f32 ne=[4096 32 1 1] cont=1
  src1 mtp_attn_norm-45            f32 ne=[4096 9 1 1]  cont=1
ROCm error: no kernel image is available for execution on the device
```

## Trigger

Both conditions, together:
1. a gfx1201 device holds **whole layers** (`WHOLE_FROM` in run_glm_two_host.sh, or any layout v2 split), rather
   than layers whose experts have been moved off with `-ot`; and
2. the **MTP draft head** is attached (`DRAFT=1`, `--spec-type draft-mtp`).

Either alone is fine. It reproduces in the PRODUCTION launcher — this is not a layout-v2 bug:

```
WHOLE_FROM=6 DRAFT=1 REMOTE_APU=RPC1 run_glm_two_host.sh tag 25 131072 -b 32768    # aborts
                     DRAFT=1 REMOTE_APU=RPC1 run_glm_two_host.sh tag 25 131072 -b 32768    # 20.5 tok/s
```

## What it is NOT (all eliminated by measurement, do not re-derive)

- **Not a missing code object.** Both arch code objects were extracted from libggml-hip.so and their full symbol
  tables diffed: 6354 kernel symbols each, set difference 0 in both directions. mul_mat_q instantiations: 1419
  each, difference 0. Every .cu.o and all 75 template-instances objects carry gfx1201.
- **Not rocBLAS coverage.** gfx1201 ships the HH/BB contraction libraries (as `_fallback_` variants) and
  `TensileLibrary_lazy_gfx1201.dat`. A standalone hipblasSgemm at the exact failing shape (OP_T/OP_N, m=32, n=9,
  k=4096) succeeds on gfx1201.
- **Not a device/context mismatch.** `GGML_CUDA_TRACE_MM` shows ctx.device=0, current_device=0, cc=gfx1201 for the
  failing call. A hipBLAS handle created on one device and used on the other also works standalone.
- **Not HIP graphs** (`GGML_CUDA_DISABLE_GRAPHS=1` still aborts), **not the f32 tile kernel**
  (`GGML_CUDA_NO_F32_TILE=1` still aborts), **not q8_0 WMMA**, **not the device count** (three devices with the
  same layer split still aborts), **not the device order** (crashing order with the working split runs clean at
  18.73 tok/s), **not the cuBLAS compute type** (`compute_type = src0->type` is already f32 here).
- **Not the shape.** f32 x f32 m=32 k=4096 at n in {1,5,8,9,10,11,12,16,17} all pass in test-backend-ops on
  gfx1201 (TBO_GLM_EVAL=1 adds them).

## What is established

`ggml_cuda_should_use_mmf` admits f32 only on Ampere or MFMA hardware, so on RDNA **every** f32 x f32 MUL_MAT
falls through to `ggml_cuda_mul_mat_cublas` -> `cublasSgemm`. The trace confirms mmf=0 mmvq=0 mmq=0 for this op on
both devices. The identical op on gfx1151 survives that path; on gfx1201 it does not.

Admitting f32 to mmf on WMMA hardware (`|| amd_wmma_available(cc)`) reroutes the op — mmf=1 — but then fails with
`unspecified launch failure` on the same class of tensor (`blk.3.indexer.proj.weight`), which is a memory-fault
signature. So **two different dispatch paths both fail on this tensor only inside the real graph**, while both
succeed standalone at the same shape. That points at the tensor's buffer or its readiness at that point in the
split schedule, not at any kernel. Next suspect: inter-split event/copy handling when two local GPUs are adjacent
in the device list, where `00766acf5` on branch glm53-flash (gate split events on the async-copy capability) is
not present on main in the same form.

## The two hosts do not run the same validation logic (mainframe, 2026-09-17)

Worth knowing before reasoning from "it works on one side and not the other", because it is a confound in every
such comparison:

**`uid` is never transmitted over the RPC wire.** The graph message is
`| device | n_nodes | nodes[] | n_tensors | tensors[] |`; `uid` appears in ggml-rpc.cpp only on the client side.
The server rebuilds the graph with `ggml_new_graph_custom` into a fresh per-call context, and ggml.c initialises
`uid` to 0. So in `ggml_cuda_graph_update_required`:

```c
if (cgraph->uid != 0 && cgraph->uid == graph->uid) { /* reuse, skip revalidation */ return false; }
```

the fast path **can fire on the client and can never fire on the rpc-server**. The server always falls through to
the `node_props` comparison. Identical workload, different validation path per host.

**And `ggml_cuda_graph_get_key` is weak, most so on the server.** It seeds from `(uintptr_t) cgraph->nodes[0]` - a
raw pointer into a context that is `ggml_init`-ed and freed on every graph_compute call on the server, so the
allocator returns the same addresses repeatedly and the seed is close to a constant. It then mixes only
`n_nodes` and the **first and last** node's `ne`; interior structure is never hashed. Two structurally different
splits with equal n_nodes and equal endpoint shapes collide, returning a cached `ggml_cuda_graph` whose instance
was captured from a different graph. On the server the `node_props` comparison is the only remaining guard.

Both are latent bugs independent of this abort. NOTE the counter-evidence for this abort specifically: gibson
crashes locally with `GGML_CUDA_DISABLE_GRAPHS=1` set on gibson (v2nograph, RR=0), and also with graphs enabled
(v1whole, v1trace) - same tensor, same error - so gibson's own graph path is excluded for gibson-side failures.

## The cleanest isolation (2026-09-17, graphs disabled on BOTH hosts, verified binaries)

One trace, one run, the only variable is the device:

```
blk.7,11,15,19,23.indexer.proj  ctx.device=1 gfx1151  f32xf32 ne00=4096 ne01=32 ne11=9  mmf=0 mmvq=0 mmq=0  -> OK
blk.45.indexer.proj             ctx.device=0 gfx1201  f32xf32 ne00=4096 ne01=32 ne11=9  mmf=0 mmvq=0 mmq=0  -> ABORT
```

Same shape, same batch width, same dispatch path (all gates decline -> `ggml_cuda_mul_mat_cublas` ->
`cublasSgemm(OP_T, OP_N, 32, 9, 4096)`). It succeeds on gfx1151 and fails on gfx1201 **in the same process**, and
succeeds on gfx1201 **standalone**. So the fault is in process state, not in the call, the shape, or the kernel.

Standalone variations that all SUCCEED on gfx1201 and therefore do not reproduce it: plain sgemm at n in
{1,2,9,16}; with `hipblasSetMathMode(HIPBLAS_TF32_TENSOR_OP_MATH)` (which returns NOT_SUPPORTED, status 7, on both
devices); with the handle created on the other device and used across; with the APU's rocBLAS warmed first and
with the R9700's warmed first. Graphs disabled on both hosts simultaneously does not help either - that was run
with the drop-in live on the rpc-server and verified in its running process.

`ne11` is not the discriminator: the surviving gfx1151 calls are at ne11=9 too, the same as the failing one.
(`should_use_mmvf` returns `ne11 <= 8` for f32 on AMD, so at ne11=9 mmvf declines and the fall-through to cuBLAS
is real; at ne11<=8 the op goes to `mul_mat_vec_f` instead and the trace's mmf/mmvq/mmq zeros would be
misleading - TRACE_MM should print the mmvf result too, not yet added.)

## Next steps, in order

1. Catch it under rocgdb at the failing `cublasSgemm` and inspect the handle, its stream and its device binding
   against a surviving gfx1151 call in the same process.
2. Failing that, replicate llama.cpp's handle setup more faithfully standalone - multiple handles per device
   (GGML_CUDA_MAX_STREAMS), the fork's virtual-device indirection, and the exact stream each call uses.
3. `--spec-draft-n-max 1` to see whether the draft width at the failing site moves the op off this path at all.

## RESOLVED 2026-09-17 (52f240cf7) - the matmul was succeeding

An `AMD_LOG_LEVEL=3` trace, read in order, shows the whole thing:

```
Error: Cannot Find Global Var Sizes
Error: Cannot create kernels.
hipModuleLoadData: Returned hipErrorNoBinaryForGpu        <- lazy Tensile load FAILS
hipExtModuleLaunchKernel ... ShaderName : Cijk_Alik_Bljk_SB_MT128x64x12...
hipExtModuleLaunchKernel: Returned hipSuccess             <- fallback runs, GEMM SUCCEEDS
hipGetLastError ( )                                       <- ggml checks
-> ggml_cuda_compute_forward: MUL_MAT failed
```

rocBLAS loads Tensile modules lazily. On gfx1201 a first-choice load returns hipErrorNoBinaryForGpu, rocBLAS
HANDLES it by falling back to another solution, that kernel launches and returns success - and the unqualified
`cudaGetLastError()` at the end of `ggml_cuda_compute_forward` picks up the surviving sticky error and aborts an
op that computed correctly. gfx1201 hits it and gfx1151 does not because gfx1201's rocBLAS ships 56 Tensile files
against gfx1151's 96, so first-choice loads miss and fall back far more often.

Fix: `(void) cudaGetLastError();` immediately after each cuBLAS GEMM, where CUBLAS_CHECK has already validated the
outcome. **Clearing at the start of `ggml_cuda_compute_forward` does NOT work** - measured, it aborts identically,
because the load happens *during* the call. A blanket clear would also discard genuine async errors from the
preceding op, which HIP reports at the next synchronising call.

### Numbers in this document are NOT performance results

The verification run measured prefill 330 t/s and decode 15.52 t/s, and **that is a correctness result only**. It
ran with `HIP_LAUNCH_BLOCKING=1` (serialises every launch), `GGML_CUDA_TRACE_MM` (a log write per matching op) and
`GGML_CUDA_DISABLE_GRAPHS=1`. Comparing it with production's 20.06/20.50 tok/s would read as a 24% regression and
would be meaningless. The same applies to other figures recorded while chasing this: v1work 14.50 t/s and v1order
18.73 t/s were also taken under diagnostic env. The only clean baseline in this investigation is **v1 at
20.06 / 20.50 tok/s decode, 375 / 497 t/s prefill** at 3K / 13K prompts.

### Still open

Why the module load fails on gfx1201 at all. A fallback is a *different kernel* with different accumulation
order, so if gfx1201 falls back on a meaningful fraction of GEMMs that is a difference in what the two cards
compute, not only in how fast. Nothing measured so far compares numerics across them. Suggested shape: same
prompt at temp 0 on the R9700 and on an APU, diffing **logits** rather than text - text can agree while numerics
diverge, and divergence that only appears under sampling is the worst kind to find late.

### How this was found, and the instrument lesson

Nine call parameters were closed by measurement before the answer arrived: function (Sgemm vs Ex, settled by
ne12/ne13), shape, batch width, leading dimension, pointer alignment, device, architecture, handle binding,
VRAM pressure. None was causal. **When every call parameter is covered and nothing reproduces, the trigger is
process state; change instrument rather than write repro number eight.**

Two qualifiers on that rule, both learned the hard way here:
- **A negative is only as strong as the instrument's reach.** Several standalone repros could not have reached
  the mechanism - a fresh process has a different set of Tensile modules resident - so they produced negatives
  indistinguishable from real ones. "Closed by measurement" needs the instrument named beside it.
- **Summarise from the whole artifact, not the tail.** Two wrong summaries were published here from truncated
  output (`tail -4` hid blk.3; a trimmed trace hid the `hipSuccess` line), and both were recovered only because
  later evidence happened not to fit.

The 56-vs-96 Tensile file gap was the first instinct of the session and was dropped after an unsorted `comm`
produced contradictory output. It was the right signal, read wrongly.

## Workaround (no longer needed, kept for the record)

Do not combine `WHOLE_FROM` with the draft head. Production (`chain_hb.sh` -> no WHOLE_FROM) is unaffected and
measures 20.06 / 20.50 tok/s decode at 3K / 13K prompts.

## Tools added while chasing this

- The CUDA error path now prints the failing tensor: name, type, shape, contiguity and buffer for dst and every
  src, instead of only `MUL_MAT failed`.
- `GGML_CUDA_TRACE_MM=<substring>` logs the dispatch decision (ctx.device, current device, cc for both, and the
  mmf/mmvq/mmq gate results) for weights whose name matches.
- test-backend-ops `TBO_GLM_EVAL=1` gained the indexer shapes at the odd batch widths speculation produces.
