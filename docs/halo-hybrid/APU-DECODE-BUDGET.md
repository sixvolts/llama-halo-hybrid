# Where a decode token goes on one Strix Halo (Qwen3.8-Flash-Next UD-Q4_K_XL)

Measured 2026-09-16 on gibson, iGPU only (`-dev ROCm1 -ngl 99 -fa on -c 4096 -ot per_layer_token_embd=CPU`,
llama-completion, greedy, no draft head). Serial decode: **26.0-26.3 tok/s = 38.3 ms per token**.

## The byte floor

Parsed from the GGUF tensor table (`scratchpad` helper, 1,224 tensors, 103.7 GiB, 48 layers, 512 experts of which
10 are used, expert width 640, n_embd 2560, vocab 248,320):

| read every token | GiB |
|---|---|
| dense weights (attention, SSM, hyper-connections, shared expert, router, LM head) | 4.493 |
| routed experts, 10 of 512 (gate/up q4_K, down q5_1) | 1.401 |
| **total** | **5.894 GiB = 6.33 GB** |

The DRAM ceiling on this part is **223 GB/s**, measured on the only matrix too large to cache: the 675 MB LM head
(`test-backend-ops perf`, `TBO_Q38=1`, 3,030 us/run). Everything smaller is served by the 32 MB Infinity Cache and
reports 300-775 GB/s in isolation, which is why op benchmarks flatter decode kernels; do not size decode work from
them. A streaming test with no math (`persist/bw.hip`) reaches 241 GB/s.

So the floor is 6.33 GB / 223 GB/s = **28.4 ms**, and we run at 165 GB/s = **74% of DRAM**. Halogen's 35.6 tok/s on
the same model is 225 GB/s, i.e. saturation: their whole advantage over this tree is the missing 26%.

## Halogen is not saturating this hardware either

Its weight file has a readable table (`HGN1`, 1,198 records of 160 bytes: name, dtype, dims, offset, size), so its
bytes per token can be computed the same way. Excluding the gathers (embed_tokens, the 47.7 GiB FP8 n-gram table) and
the MTP head:

| halogen w4b, read every token | GiB |
|---|---|
| dense trunk (attention, DeltaNet, hyper-connections, shared expert) | 1.95 |
| routed experts, 10 of 512 | 1.25 |
| LM head | 0.33 |
| router (f32) | 0.12 |
| **total** | **3.65 GiB = 3.92 GB**, up to ~4.4 GB with the activation-aware overlay that replaces trunk tensors |

At 35.6 tok/s that is **139-157 GB/s, 63-70% of the 223 GB/s ceiling**, against this tree's 166 GB/s (74%). The
trunk tensors both engines read every token are 8.50 bits per weight here (q8_0) and 4.50 there:

| tensor | this tree | halogen |
|---|---|---|
| linear-attention qkv, 36 layers | 8.50 bpw | 4.50 bpw |
| linear-attention out, 36 layers | 8.50 | 4.50 |
| LM head | 8.50 | 4.50 |

So the 1.37x is bytes, all of it: 6.33 GB per token against ~4 GB. This tree moves those bytes at least as
efficiently. The q8_0 trunk is 4.49 GiB of our 5.89 GiB and is the UD-Q4_K_XL choice for quality; at halogen's byte
count and our efficiency the same box would decode at ~42 tok/s. That is the actual size of what the format costs,
and it is a decision, not a kernel.

## Where the 26% goes

`rocm-smi --showuse` sampled through decode reports the iGPU busy **82.9%** of the time; the driver's own
`gpu_busy_percent` reports **99%** over the same kind of window. They count different things: the second says a
wave is alive, the first is closer to the shader engines being fed. Read together with the byte floor, the loss is
not empty time between kernels but the tails and ramps of ~2,250 kernels, during which waves exist and the memory
pipe is underfed. Clocks are not the cause: sclk sits at 2.9 GHz and mclk at its 1000 MHz maximum throughout.

| per token | ms | share |
|---|---|---|
| DRAM traffic at the ceiling | 28.4 | 74% |
| kernel time beyond the bytes (DeltaNet, attention, norms, topk, hc mixing) | ~3.3 | 9% |
| GPU idle between kernels | ~6.6 | 17% |

With 2,247 kernels per steady-state token (inventory below) that idle is ~2.9 us per kernel boundary, which matches the boundary cost measured
directly for the persistent-decode work (a real graph node costs ~6 us against 1.74 us for an empty one).

## The steady-state token, dispatch by dispatch

A hardware-counter pass (`rocprofv3 --pmc FETCH_SIZE`, two tokens, prompt "Hi") gives the exact launch list. The
first token of a sequence is **2,397** dispatches; every later token is **2,247**. The 150 extra are llama.cpp's
`build_rs`: on the first token of a sequence it zeroes and gathers the recurrent state of every SSM layer
(`scale_f32` + `k_get_rows_float` per layer, plus the conv-state equivalents). Steady-state decode never runs them,
so the earlier "~2,400" over-counted by 7%.

Steady-state, per token, with the bytes each class fetched (FETCH_SIZE calibrated on the LM head, which
under-reports 1.93x on this GPU):

| dispatches | class | MB | what |
|---|---|---|---|
| 437 | `mul_mat_vec_q` | 3,894 | trunk, expert and shared-expert projections |
| 48 | `mul_mat_vec_q_group` | 2,038 | the grouped qkv / qkv+gate launches |
| 300 | `mul_mat_vec_f` | 337 | router (48), hc inject (96), DeltaNet alpha/beta (72), indexer q/k (24, bf16), rest |
| 36 | `gated_delta_net` | 111 | reads the recurrent state once |
| 184 | `rms_norm_f32` | 10 | |
| 385 | `k_scale_silu` 97, `k_hc_mix` 97, `k_hc_combine` 95, `k_mul_sigmoid` 96 | ~20 | hyper-connection mixing and the output gate |
| 97 | `quantize_q8_1` | | the GLU outputs (expert and shared) |
| 125 | `__amd_rocclr_copyBuffer*` | | CPY nodes lowered to blits: conv/KV/state bookkeeping |
| 53 + 48 + 37 + 24 | `cpy_scalar`, `k_set_rows`, `concat_cont`, `fill` | | more bookkeeping |
| 48 × 3 | `topk_moe`, `moe_weighted_reduction`, `rope_multi` | | |
| 36 × 3 | `ssm_conv`, `l2_norm`, `k_gdn_gate` | | the DeltaNet chain around the recurrence |
| 64 + 51 + 28 + 27 | `k_bin_bcast`, `unary_op_kernel`, `k_ew_chain`, `k_get_rows_float` | | |
| 12 × 3 | `argsort`, `flash_attn_tile`, `flash_attn_combine` | | the 12 attention layers |
| **2,247** | | **6,453** | floor 6,330 |

The 785 GEMV-class dispatches carry **97%** of the bytes; the model fetches only 2% above its weight floor, so there
is no traffic to remove. The other 1,462 launches carry 3% of the bytes and all of the boundary cost: at the measured
4-5 us each that is the whole 6.6 ms. Which is why the program below is about removing launches, not bytes.

## What the idle is NOT (all measured, not argued)

| experiment | result |
|---|---|
| `GGML_CUDA_DISABLE_GRAPHS=1` (no HIP graph replay) | 26.32 vs 26.21 tok/s: **no effect** |
| per-layer token embedding table on the GPU instead of the CPU (26.8 GiB moved) | 26.16 vs 26.21: **no effect** |
| `GGML_CUDA_GRAPH_OPT=1` (multi-stream fork/join over independent branches) | 26.06 vs 26.12, and the pass finds **no forks** in this graph |
| persistent megakernel (12,500 fewer launches over 16 tokens) | slower, see PERSISTENT-DECODE.md |

So it is not CPU launch cost, not the host-side embedding gather, and not a missing chance to overlap branches. It is
the per-dispatch cost the hardware pays between dependent kernels, and the only lever on it is fewer dependent
kernels - without paying for that in register pressure, which is what killed the megakernel.

Two more candidates measured on the standalone cold GEMV (`persist/gemv_bench3.hip`, 8 matrices cycled so nothing
is cached): **rows per workgroup** 1/2/4/8/16 with one wave per row all give 225 GB/s on 9216x2560 q4_K, so
workgroup dispatch pressure is not a limiter; **non-temporal weight loads** are worse, 216 vs 225 on the large
shapes and 145 vs 237 on the 640-row shape. An isolated cold GEMV reaches 218-237 GB/s; the loss is around the
kernels, not in them.

One tuning knob looked mis-set and is not: gfx1151 falls through `get_device_table_id()` to the RDNA2 MMVQ table,
which has no branch in `calc_nwarps` and so gives **one wave per row**, where RDNA3 and RDNA4 parts get eight for the
simple quant types. Building with the RDNA3_0 table instead (`GGML_CUDA_MMVQ_RDNA35=1` in mmvq.cu) costs **16%**:
21.96 vs 26.12 tok/s, with the 10240x2560 q8_0 GEMV going from 35.9 to 57.8 us. On this part the row-per-wave form
wins; leave the fallthrough alone.

## How much is left in the obvious fusions

`GGML_CUDA_GEMV_GROUPS=1` reports every run of consecutive GEMVs that share an activation (the group the tree already
quantizes once for). Per token, on this model:

| group | count | launches a grouped kernel would save |
|---|---|---|
| attn_qkv + attn_gate (36 SSM layers) | 36 | 36 |
| attn_q + attn_k + attn_v (12 attention layers) | 12 | 24 |
| everything else | n=1, no group | 0 |

**60 of ~2,250 launches, under 1% of the token.** The remaining GEMVs each read a different activation: the
hyper-connection down-projections, the shared expert, ssm_out, attn_output and the LM head are genuinely serial.

## Fewer kernels: the two exchange rates

Both measured on this model, because the whole "get to ~500 kernels per token" plan depends on them.

**A bare dispatch costs 1.7 us.** `GGML_CUDA_PAD_KERNELS=<n>` appends n empty dependent kernels to every graph:

| injected kernels | ms per token |
|---|---|
| 0 | 38.26 |
| 200 | 38.60 |
| 400 | 38.99 |
| 800 | 39.61 |
| 0 (repeat) | 38.41 |

Linear, 1.7 us each. That is only the dispatch; an empty kernel has no ramp to pay for.

**Merging real work is worth 2-3x that.** Running several GEMVs that read the same activation as ONE launch removes
60 launches per token and buys 0.25 ms: **4.2 us per merged GEMV** (37.93 ms grouped against 38.18 ungrouped, three
repetitions each, spread under 0.02 ms, identical greedy output). The extra over 1.7 us is the drain and ramp of the
kernel that no longer exists. So the ceiling for this program is the 6.6 ms of idle, and the price list is ~4-5 us
per kernel that merging removes: reaching halogen's ~500 kernels per token would be worth roughly 3.5 ms, taking
26.2 tok/s to about 28.8.

## The grouped GEMV

`mul_mat_vec_q_group` in mmvq.cu: one launch for up to 8 matrices that read the same (already quantized) activation.
One wave per row, as gfx1151 wants; the row's matrix is found by a short block-uniform scan over the prefix sums.
Entries whose weights are **f32** ride along and dot the unquantized activation instead, which is what lets the
hyper-connection inject join its down-projection. `GGML_CUDA_NO_GEMV_GROUP=1` disables it; the shared-activation
detector that already quantized once for the group now makes one launch instead of one per matrix.

Groups this finds in Qwen3.8 decode, per token:

| group | count | launches removed |
|---|---|---|
| attn qkv + gate (36 SSM layers) | 36 | 36 |
| attn q + k + v (12 attention layers) | 12 | 24 |
| hyper-connection down + inject, attn + DeltaNet beta (f32 members, disabled) | 144 | 0 |
| **total** | | **60** (the f32 members are excluded, see below) |

**f32 members are off by default, and the measurement says they should be.** The kernel can dot an f32 matrix
against the unquantized activation, which is what would let the hyper-connection inject (10240x4 f32) join its
down-projection. It loses badly: `mul_mat_vec_f` gives an f32 matrix 8 warps per row, this kernel gives it one wave,
and the inject has 4 rows, so the group ran it on 4 waves. Merging it cost **7.6 ms per token** (45.5 ms against
38.3). The DeltaNet beta (48 rows, K=2560) is the mild version of the same thing: including it moved the result from
37.86 to 37.98 ms, i.e. it gave back half the win. `GGML_CUDA_GEMV_GROUP_F32ROWS=<n>` re-enables members with at
least n rows. It is only worth revisiting once an f32 member can use the whole block instead of one wave.

What is left needs more than adjacency: 256 single-matrix cases per token remain, including the per-layer embedding
key/value pair and the router/shared-expert/alpha trio, which all read one activation but are separated in the node
order by work that depends on them. Merging those needs a reorder pass in `graph_optimize` (hoist the consumers of a
tensor together, with the alias checks ggml-alloc's buffer reuse demands), not a bigger kernel.

## What this means for the "cheaper than halogen" list

- *Grouped GEMVs*: real but worth <1% here, because this architecture rarely runs two projections off one activation.
- *GEMVs that ingest the activation (no quantize pass)*: already done where it pays. The producer-side q8_1 side
  registry plus the shared-activation group cover it; the standalone quantize call in `ggml_cuda_mul_mat_vec_q` never
  fires during decode on this model (`GGML_CUDA_GEMV_GROUPS=1` counts 0).
- *N-gram / PLE table on the device*: dead, measured above.
- *Wider chain fusion*: the hyper-connection kernels (`k_hc_mix`, `k_hc_combine`, `k_scale_silu`) already replace
  ~1,000 ggml nodes with 277 launches per token, and `rms_norm` already writes its q8_1 copy.

The honest remaining program is the one halogen ran: get from ~2,250 kernels per token to ~500 by folding norms,
gates, the router and sampling into the projection kernels, at ~5 us of boundary each. Nothing smaller moves this
model, and the draft head (MTP) remains worth more than all of it, since it amortises the whole 6.33 GB over 2-3
tokens.

## 2026-09-24: Qwen3.8-Flash-Next APU-only decode, survey and first fusion wave (commits 01be81caf..82b8eaa32)
Baseline on the current tree (llama-completion, -dev ROCm1, -c 4096, 256 greedy tokens): 39.0 ms/token, the same as
the 09-16 build under the identical command (38.7 / 39.6), so no regression; the 37.2 ms of 09-16 was another workload.
Survey (Opus agent, rocprofv3 + op timer): ~2,067 dispatches per token, 1,380 under 20 us; GDN layer 34 launches,
attention layer 65 of which 28 were the QSA indexer's scoring; 39.0 ms = 28.4 bytes floor + ~1.6 GEMVs below
223 GB/s + ~3.5 non-GEMV kernel time + ~4.5 inter-kernel gaps + ~1.5 host gap at the token boundary.
- **QSA indexer skip (01be81caf):** with n_kv <= indexer_top_k + ratio - 1 (2051 cells) top-k returns every cell, so
  the scoring (336 launches per token) is skipped; keys are still cached. 38.96 -> 37.75 ms/token, identical text at
  short context; across a 1.9K prompt a near-tie flip, perplexity 7.1791 (scoring) vs 7.1818 (skip), +/- 0.198.
  LLAMA_QSA_ALWAYS_SCORE=1 restores.
- **Fusion wave (Opus 5.5 workflow, 4 implementers + 4 reviewers):** merged the hc boundary (combine + rms_norm*gamma
  + q8_1 in one launch, and the q8_1 side copy found through the hc_down reshape: 97 standalone quantizes per token
  gone; GGML_CUDA_NO_HC_BOUNDARY), the GDN conv front (concat + slot copy + ssm_conv/silu + Q/K l2_norm;
  GGML_CUDA_NO_GDN_CONV_FRONT) and the MoE tail (expert down + weighted sum + gated shared-expert down + add;
  GGML_CUDA_NO_MOE_TAIL; review fix: its alloc deps must run after the GEMV hoist). Rejected: GDN prologue/epilogue
  (no end-to-end gain; not bit-identical, the compiler contracts x*x into the first butterfly add; +3.7% on the R9700).
  Together: 37.78 / 37.87 -> 36.70 / 36.76 ms/token, identical greedy text, perplexity 7.1818 both.
- **Now:** 36.7 ms/token serial (27.2 t/s, from 25.6), MTP head 33.3-37.3 t/s (from 31.8-35.9). halogen: 37.6 / 44.8.
- **Open from the survey:** hc_down split-K on gfx1151 (~0.35 ms), topk_moe as one 512-thread block (~0.2 ms), the
  ~1.5 ms token-boundary host gap (needs a host trace), indexer fusion for contexts past 2K, prefill (700 vs 1,246).
- **Seen by a reviewer, pre-existing:** perplexity on one chunk varies ~2% with the ubatch width (8.087 at -ub 2,
  8.237 at 3, 8.241 at 4, 8.163 at 512): worth a look at the width-3/4 GEMV paths the MTP verify uses.

## 2026-09-24: Qwen3.8 APU-only prefill - sparse prefill attention and chunked gated DeltaNet (82b766555, fa1e1b122, fix commit after)
Profile before (server, -ub 2048, per prompt token at 4K / 16K / 32K): expert GEMM 0.39 / 0.38 / 0.38 ms, dense GEMM
0.35 / 0.37 / 0.37, element-wise 0.24 / 0.26 / 0.31, gated DeltaNet + conv 0.14, flash attention 0.04 / 0.18 / 0.39
(the only term growing with context: the QSA top-k was applied as a mask to DENSE attention at prefill; the gather
path only turns on above 65,536 cells). Measured with GGML_CUDA_TIME_OPS_EVERY=1 (b961f4fa5).
- **Sparse prefill attention (82b766555):** adjacent queries' top-k selections overlap poorly (for the dense kernel's
  16 x 64 tile, 58-84% of KV tiles still hold a selected cell at 32K, so tile skipping would save only 15-40%); the
  existing GLM sparse gather path now serves D=256 / GQA 12 with 16 queries of a tile walking the union of their
  selections, from 4x the 2051-cell bound (~8.2K cells). Also a gfx1151 D=256 config (Q in LDS, 32-cell K/V tiles)
  that removes a 1.1-1.6 KB/lane spill: dense FA per 2048-query call at 30K 118.9 -> 91.8 ms, sparse 41.8 ms.
  Switches: LLAMA_QSA_SPARSE_FA=0, GGML_CUDA_FA_SPARSE_D256=0; the config has none (compile-time).
- **Chunked gated DeltaNet (fa1e1b122):** WY/UT form, 32-token chunks, scalar-gate heads only (GLM's per-channel KDA
  keeps the token loop), MTP rollback tail still on the token loop, f64 gate cumsum (f32 was 10-1000x worse on strongly
  decaying heads): 6.7 -> 3.35 ms per 2048-token ubatch per layer. Perplexity +0.022 at 2K is within what a 1e-6
  output perturbation of the old kernel produces (KLD 0.0316 vs 0.0329); 16K 5.6042 -> 5.6003. GGML_CUDA_NO_GDN_CHUNKED=1.
- **Result:** 705 / 691 / 589 -> 747 / 756 / 701 t/s (-ub 2048), 769 / 783 / 724 with -ub 4096. halogen: 1,246 at 8K,
  1,424 at 32K. Left: expert GEMM (LDS-bound MMQ, the largest term), hc_down GEMM shape (7 TFLOPS), hc element-wise
  traffic at prefill widths, the QSA mask build (fill/set_rows/add over n_kv x 2048), a WMMA GDN scan.

## 2026-09-25/26: Qwen3.8 expert GEMMs (fb5ea41df, 5f9972fc0, b4ecb3a55)
Isolated (TBO_Q38_MOE, one call over 512 experts x 10 used): bandwidth-bound (205-212 GB/s) up to ~1K tokens, then
compute-bound at ~15-16 TOPS (q4_K gate/up) and ~12 TOPS (q5_1 down, K = 640) against ~59 TOPS peak.
- **Gate/up (fb5ea41df, GGML_CUDA_MMQ_GATEUP=0/1/2, default 2):** one quantize + expert sort for both; a fused gate+up+GLU
  kernel (two half-height MMQ blocks stacked, GLU in the epilogue). Bit-identical (per-chunk perplexity, KLD tables,
  greedy). Isolated MOE_GATE_UP graph -10 to -12% at 1-4K tokens; end to end ~+1%. Lesson: full-height halves (twice the
  weight LDS per activation tile) were 1.2-1.6x SLOWER - on gfx1151 fewer resident blocks cost more than halving
  activation-tile loads. Reviewer: the fused GEMM itself is +6% vs two unfused ones; the net gain is the removed
  launches; at GLM's J=16 on gfx1151 mode 1 is 1.5-2.5% faster than mode 2 (follow-up: route small J to mode 1).
- **Inner loop (5f9972fc0, GGML_CUDA_MMQ_SUBTILE_SKIP / GGML_CUDA_MMQ_KTAIL_SKIP, default on; Q5_1 RDNA3.5 tiles
  compile-time):** counters (gfx1151 offers no WMMA/wait counters): q4_K J=48 runs 4 waves/SIMD (LDS-limited), issue slots
  ~70% busy, LDS bank conflicts 8%, DRAM ~141 GB/s; ablations say load/barrier-bound at 2048 tokens, issue-bound at 4096.
  Exact fixes: skip 16-column subtiles past the last real column (MoE tiles sized to the mean are ~68% full), skip the
  K-tail half-iteration (K = 640 spent 1/6 of the down projection on padding), Q5_1 on 64-row tiles with prefetch.
  q4_K -16% and q5_1 -24% at 4096 tokens; GLM q4_K/q5_K -8/-9% at 2048 on gfx1151. Dropped: dot2 min-term (lost
  dual issue, spills), larger J (spills). Plan for the real redesign: min term as one f16 WMMA per C tile, q4_K nibbles
  packed in LDS (half the A bytes, 5+ blocks/WGP), per-type MoE column tile - compare with gufo's RoutedF16GEMMKernel.
  Measurements: ~/bench/results/mmq-inner-loop-0925/.
- **Merged result:** greedy identical, perplexity per chunk identical (7.1541 / 5.5909). Server prefill 747/756/701 ->
  754/794/735 t/s (4K noisy). GLM two-host unchanged (887 t/s, 94.4 ms/step, same hash) once the TTM pool is drained -
  a run started with the pool full read 742 t/s and 169.5 ms/step; sweep_0922.sh now drains before each server start.

## 2026-09-26: F16-WMMA routed expert GEMM ported from gufo (8defd8fb4)
gufo (github.com/gufo-org/gufo, MIT) runs this model's prefill at 2x ours on the same box; its routed expert GEMM keeps
the codes packed in LDS and dequantizes to F16 per wave (no per-32 scale VALU), with F16 activations. Ported as
ggml/src/ggml-cuda/mmid-f16.cu behind MUL_MAT_ID (attribution in the header). Isolated vs our MMQ: q4_K -16% / -19%
at 2048 / 4096 tokens, slower below ~1K tokens and on GLM's 2048 x 4096 experts, so an auto rule (small experts,
>= 40 rows per expert, RDNA3.5) decides. KLD 0.0295 vs MMQ (below the ~0.032 of a 1e-6 perturbation). Prefill
754/792/733 -> 787/817/752 t/s; 804/847/777 with -ub 4096 (recommended for Qwen3.8 on the APU). GLM unchanged.
Not ported yet from gufo: the paired gate/up variant with the SwiGLU written as F16 for the down projection (removes the
f32 intermediate and a conversion), the dense Q8->F16 WMMA with fused HC/conv/attention epilogues, HC combine kernels.
