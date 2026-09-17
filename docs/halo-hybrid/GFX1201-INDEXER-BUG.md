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

## Workaround

Do not combine `WHOLE_FROM` with the draft head. Production (`chain_hb.sh` -> no WHOLE_FROM) is unaffected and
measures 20.06 / 20.50 tok/s decode at 3K / 13K prompts.

## Tools added while chasing this

- The CUDA error path now prints the failing tensor: name, type, shape, contiguity and buffer for dst and every
  src, instead of only `MUL_MAT failed`.
- `GGML_CUDA_TRACE_MM=<substring>` logs the dispatch decision (ctx.device, current device, cc for both, and the
  mmf/mmvq/mmq gate results) for weights whose name matches.
- test-backend-ops `TBO_GLM_EVAL=1` gained the indexer shapes at the odd batch widths speculation produces.
