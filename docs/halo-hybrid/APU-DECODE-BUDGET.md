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
them. A streaming test with no math reaches 241 GB/s.

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

Two more candidates measured on the standalone cold GEMV (8 matrices cycled so nothing
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

## 2026-09-26: dense GEMMs and hyper-connection passes at prefill (ae85dfdb3)
- **Dense q8_0 GEMMs:** our WMMA kernel already reaches gufo's 30-36 TFLOPS on the wide shapes (attn_qkv 35.9, attn_q
  36.0); a port was not needed. The slow shapes were ssm_out / attn_output (2560 x 6144) and attn_gate (6144 x 2560) at
  17.7 / 28 TFLOPS: at 15.7M weights they fell just under the 16M cutoff for the f16 activation pre-pass and re-read
  f32 activations per 128-row tile. Cutoff 8M (ae85dfdb3): 3.45 -> 2.49 and 2.40 -> 2.09 ms per 2048 tokens,
  bit-identical. Prefill 787/817/752 -> 813/824/761 t/s (852/857/788 at -ub 4096).
- **Split-K for the 320 x 10240 HC down-projection: measured, rejected.** 1 split 1.46 ms, 2-8 splits 1.56-1.77 ms. The
  shape is bound by re-reading its 84 MB f32 input (10240 per token) once per 128-row tile (~250 MB for 13 GFLOP), not
  by grid parallelism. Its fix is fusion with the producer (rms_norm) or reading the input as f16 once.
- **Hyper-connection element-wise work (25% of a 16K prefill) is already fused and bandwidth-bound:** k_hc_combine_norm,
  k_hc_mix and k_mul_sigmoid run at ~210 GB/s (k_hc_mix: 189 MB in 0.89 ms per 2048 tokens). What remains are the
  84 MB f32 intermediates between kernels (the up-projection's gate, the normalized streams). gufo removes them by
  computing the mix in the up-projection GEMM's epilogue (DenseF16GEMMKernel kHcMix); for us that needs the w_up rows
  permuted so the four streams of one channel land in one tile, plus a GEMM epilogue - a multi-day change.
- **Note:** the F16 routed expert path's auto rule needs >= 40 rows per expert: a 1.9K-token prompt (18.6K rows over
  512 experts) stays on MMQ.

## 2026-09-26: Qwen3.8 APU + R9700 vs gufo APU-only (the parity bar)
Target set by the user: gufo owns single-APU; our APU + R9700 configuration must at least match gufo's APU-only numbers.
Measured on gibson, same model files and prompts (gufo via ~/bench/gufo_probe.sh, ours via ~/bench/q38_hybrid_srv.sh):

| | prefill 4K / 16K / 32K (t/s) | MTP decode (t/s) |
|---|---|---|
| gufo, APU only | 1,359 / 1,431 / 1,410 | 34.8 |
| ours, APU only (-ub 2048) | 813 / 824 / 761 | 33-37 |
| ours, APU + R9700, 1 lane | 1,059 / 1,218 / 1,191 | 47.3 |
| ours, APU + R9700, 2 prefill lanes | **1,491 / 1,781 / 1,719** | **47.8** |

Layout: `-dev ROCm0,ROCm1 -ts 1,0`, experts of layers 11-47 on ROCm1 (`-ot 'blk\.(11|...|47)\.ffn_(gate|up|down)_exps=ROCm1'`),
dense trunk, layers 0-10 experts and the MTP draft on the R9700, `-b 4096 -ub 2048`, `LLAMA_PREFILL_LANES=2`. Greedy text
identical with 1 and 2 lanes (bit-exact only with -ctxcp 0: the server picks different checkpoint boundaries per lane
count). Corrections (review workflow 2026-09-26, ~/bench/results/q38rev-0926/PLAN.md): the probe drains the page cache
before each server start, so these are COLD numbers (warm 4K is 1,829-1,950; steady-state MTP decode ~51 t/s, 47.8 was
one cold request); LLAMA_PREFILL_LANES=4 only adds lanes with a remote device - locally it runs 2 lanes, so the
"4 lanes at -ub 1024" run (1,315 / 1,545 / 1,520) was 2 lanes at -ub 1024; the 12+ expert-layer and -ub 3072/4096
failures were the MTP draft's KV / compute buffers (the draft inherits the target's n_ubatch), not the target layout.

## 2026-09-27: review pass 0926, first items (bb282bd3e, 609f0944c, 9375f16ec)

- **Peer copy with the source device current** (bb282bd3e). A card -> APU copy issued while the APU was current
  returned zeros without an error.
- **Pair-lane guard** (609f0944c). The paired two-lane walk now checks that both lanes have the same split backends
  and the same first and last node per split; on a mismatch it logs and runs the lanes one after the other. Node
  counts are allowed to differ: the first ubatch skips QSA indexer scoring (58 nodes per QSA layer).
- **Scheduler events for the APU plus a 256-entry graph cache** (9375f16ec). Card graph replays went from 4% to 85%.

Hybrid (EXP_FROM=11, 2 lanes, -ctxcp 0, cold, 4K / 16K / 32K prefill, 256-token MTP decode):

| build | prefill t/s | decode t/s |
|---|---|---|
| before (2 runs) | 1554-1558 / 1800-1801 / 1729-1730 | 48.0-48.5 |
| D1 on (3 runs incl. defaults) | 1559-1571 / 1797-1810 / 1725-1735 | 51.2-52.0 |

Greedy text identical in every run (5741ae8aed8d short, 28e9e3446f1f long), with and without the overlap cut.
Perplexity (8 chunks, ub 512) is bit-identical with D1 on and off, APU-only (7.1541) and hybrid (7.1907). APU-only
at -b 4096 -ub 2048: 814/832/768 off, 813/827/762 on, decode 26.3 both (neutral). Opt-outs:
GGML_SCHED_IGPU_EVENTS=0, GGML_CUDA_MAX_GRAPHS=64.

### P1 and P2 (f18e30ce3, c9bb9635f, fca6d3183)

Hybrid, 2 lanes, -ctxcp 0, cold, repeated runs (min-max). "q4k draft" is the MTP head with its experts
requantised to q4_K/q5_1 (models/qwen38-flash-next/MTP/mtp-Qwen3.8-Flash-Next-shared-exps-q4k.gguf, 1050 MiB less VRAM,
acceptance 0.63).

| config | 4K | 16K | 32K | decode |
|---|---|---|---|---|
| before, q4k draft | 1525-1544 | 1795-1799 | 1728 | 52.5 |
| + PLE gather read-ahead + threads | 1912-1940 | 1870-1872 | 1743-1745 | 56.1 |
| + 55/45 split of short batches (default now) | 1933 | 1919 | 1778 | 56.0 |
| same, -ub 2560 -b 5120 | 1892-1942 | 2008-2017 | 1844-1854 | 56.0-56.1 |

- **PLE gather** is the biggest item: the cold first ubatch spent 389 ms in set_inputs faulting table rows in one
  at a time. Output identical; APU-only 4K 818 -> 915, decode 26.6 -> 27.0.
- **Short-batch split:** 50/50 was slower than the lopsided 2048 + 1648 pair (1735 vs 1668 ms): a smaller second
  ubatch shortens the pipeline drain. 55/45 keeps 4K and gains 2% at 16K/32K.
- **Draft n_ubatch cap** (LLAMA_SPEC_DRAFT_UB) costs 1.5-3% prefill at 512 or 1024: opt-in for VRAM only.
- **2D eh_proj** (LLAMA_MTP_EH_PROJ_2D=1): prefill +0.5-1%, acceptance 0.63 -> 0.62, decode -1%: opt-in.
- Perplexity (APU-only, 8 chunks) unchanged at 7.1541.

**OPEN: one APU page fault + hang (2026-09-27 00:57).** First request (3696 tokens) of a hybrid two-lane server
with all of the above: amdgpu c5:00.0 (Strix Halo) gfxhub page fault at address 0, SQC instruction fetch
(PERMISSION_FAULTS 0xb), no GPU reset, and llama-server never aborted: it waited until the client's 2 h timeout. First
fault of the boot across ~40 runs. APU scheduler events (9375f16ec) are the prime suspect but unproven. Soak of the exact first
request (fresh server, cold caches, 2 lanes, same build and flags): 0 hangs in 50, plus 0 in 11 at a 1873-token first
request, so the rate is below ~1/60 and it stays open as an intermittent. Bisection order if it reproduces: GGML_SCHED_IGPU_EVENTS=0, GGML_CUDA_MAX_GRAPHS=64,
LLAMA_PLE_WILLNEED=0 LLAMA_PLE_THREADS=1, LLAMA_LANES_SPLIT_MIN=0.

## 2026-09-27: Swift 1.5 Qwen3.8-Flash-Next (ukisai fine-tune) bring-up (bc65067c5, 70b145d40)

**Files** (~/models/swift15-flash-next): ukisai's Q4_K_M GGUF (3 shards, 111.4 GiB, sha256 checked against HF), plus
an MTP head extracted from the fine-tune's bf16 safetensors. The GGUF has no MTP tensors; fetch_mtp.py range-reads only
the `mtp.*` tensors (4.97 GB of the 330 GB checkpoint) and `convert_hf_to_gguf.py --mtp --mtp-shared-embd` makes a
shared head. The same pipeline on base Qwen3.8 reproduces unsloth's shared-Q8_0 head byte for byte except the two
indexer projections, which unsloth keeps bf16 (we now do too: `llama-quantize --tensor-type indexer=bf16`).
Heads: shared-Q8_0 (2.79 GB) and shared-exps-q4k (1.69 GB, experts q4_K/q5_1): same acceptance, use the q4k one.

**Quant mix differs from unsloth's UD-Q4_K_XL** and that decides speed: dense trunk q4_K/q5_K/q6_K (not q8_0),
expert down q5_0/q8_0 (not q5_1), hc inject / output_hc bf16, PLE table q5_0 (32.8 GiB, lazy on disk).
Two code fixes, both general:
- q5_0 in the F16 routed expert GEMM (-17% at 2048/4096 tokens on gfx1151).
- thin bf16/f16 weights at prefill widths (4 x 10240 hc_*_inject) take the swapped-MMVF path: 1243 -> 147 us per
  call on the card. Exact f32 dots: perplexity 7.6218 -> 7.5703 and real-content acceptance slightly up.

Hybrid (EXP_FROM=11, 2 lanes, -ctxcp 0, cold), Swift with its own q4k draft:

| build | 4K | 16K | 32K | decode (launcher) |
|---|---|---|---|---|
| first load | 1730 | 1714 | 1603 | 58.6 |
| + q5_0 F16 experts | 1774 | 1719 | 1607 | 58.6 |
| + thin bf16 fix (2 runs) | 1866-1873 | 1828-1834 | 1701-1704 | 56.8 (text changed; see acceptance) |
| same, -ub 2560 -b 5120 | 1872 | 1904 | 1759 | 56.7 |

Real-content acceptance (probe_real, T=0.7, 400 tokens, seeds 1/2, n-max 2): Swift head 0.65/0.67 (57.1/58.2 t/s),
base head on Swift 0.62/0.63 (56.6/56.8). APU-only (-b 4096 -ub 2048, no draft): Swift 851/807/739, decode 31.1 vs
base 897/848/773, 27.4: the k-quant trunk misses the q8_0 WMMA prefill path but moves fewer bytes per token.

**What is left of the gap to base (1933/1919/1778):** the card's dense trunk. 16K op timer, card: attn_qkv q6_K
2.9 ms per 2048-token call vs 1.35 ms q8_0 (+480 ms), attn_q/attn_gate/hc_down in q4_K/q6_K (+~270 ms). The APU's
expert time is unchanged. Options: a q8_0 trunk (needs the bf16 trunk tensors, range-fetchable like the MTP head, and
a re-assembled GGUF = disk: 27 GB free), or faster k-quant MMQ on gfx1201.

### k-quant dense GEMM on WMMA (43d424a53) and a q8_0-trunk Swift build

**Kernel:** mmq-wmma.cu now takes q4_K / q5_K / q6_K (16-weight staging units, (q - c) * s + b with one rounding).
At 2048 tokens: gfx1201 q4_K 62-68 -> 81-92, q5_K 71-78 -> 75-89, q6_K 36-38 -> 74-83 TFLOPS (q8_0 MMQ: 84-95);
gfx1151 q4_K 20-23 -> 23-32, q5_K 21-24 -> 22-30, q6_K 17-28 -> 20-29. Default on for RDNA3/4, GGML_CUDA_KQ_WMMA=0
reverts. Swift Q4_K_M perplexity 7.5703 -> 7.5247 (f16 activations instead of q8_1).

**q8_0-trunk Swift** (~/models/swift15-flash-next/Q8T, 121 GB): trunk tensors range-fetched from the bf16 checkpoint
(9.4 GB, fetch_mtp.py with an exclude pattern), converted with convert_hf_to_gguf.py --no-mtp --outtype f32, merged by
tools/merge_trunk.py: every tensor from ukisai's Q4_K_M except those whose type in unsloth's UD-Q4_K_XL differs
(359 matrices -> q8_0, inject bf16 -> f32); experts and the PLE table unchanged. f32 tensors both files share are
bit-identical except 21 ssm_a with a 1-ulp exp() difference in one of 48 values (kept ukisai's).

| Swift build (hybrid 2 lanes, cold) | 4K | 16K | 32K | real-content decode | ppl (8 chunks, hybrid / APU) |
|---|---|---|---|---|---|
| Q4_K_M + k-quant WMMA | 1887 | 1907 | 1769 | 57.1 / 58.2 (40 ms/step) | 7.5247 / 7.5320 |
| q8_0 trunk (2 runs) | 1877-1907 | 1902-1903 | 1764 | 55.3 / 55.5 (41 ms/step) | 7.2907 / 7.3107 |
| q8_0 trunk, -ub 2560 -b 5120 | 1919 | 1994 | 1836 | | |

(The Q4_K_M decode row was measured just before 43d424a53; decode never enters the N >= 64 WMMA path, so it stands.
Base-model gate after 43d424a53: greedy hashes unchanged (5741ae8aed8d / 1b969dc886b0, APU-only text), prefill
1891/1915/1776 and APU-only 922/853/776. GLM-5.3's dense weights are all q8_0/f32, so the kernel does not touch it.)
APU-only q8_0 trunk: 902/847/772, decode 26.8 (Q4_K_M: 871/808/739, 31.3). The q8_0 trunk costs ~3% hybrid decode and
~14% APU-only decode (more trunk bytes per token) for a 3% lower perplexity.

### Draft-only LM head (a5fef593a)

The shared-embedding MTP head drafted through the target's q8_0 output.weight (680 MB, read twice per step). A draft
file can now carry its own nextn.shared_head_head (add_draft_head.py + llama-quantize), used only for drafting.
Greedy output is byte-identical (verification is unchanged): Q8T 841b89707654 / 07091463e3ea, base 5741ae8aed8d /
1b969dc886b0. Real-content decode, 2 seeds (x2 runs):

| model, draft head | ms/step | acceptance | t/s |
|---|---|---|---|
| Swift Q8T, shared q8_0 | 40.5-40.8 | 0.64 | 55.6-56.2 |
| Swift Q8T, own q6_K | 40.2 | 0.64-0.65 | 56.4-57.0 |
| Swift Q8T, own q4_K | 39.5-39.7 | 0.64-0.66 | 57.3-58.1 |
| Swift Q8T, 64K-token subset (d2t) | 39.7 | 0.63-0.65 | 56.4-57.7 |
| base, shared q8_0 | 40.2-40.3 | 0.66 | 57.2-57.5 |
| base, own q4_K (from bf16) | 39.2-39.3 | 0.62-0.64 | 56.9-57.9 |

The subset head works (loader logs "MTP head over a reduced vocabulary") but self-generated held-out coverage is
only 87% at 32K and 94% at 64K tokens (English prose 15-20% misses at 32K: the corpus is code-heavy and 59K
generated tokens are too few), and the full-vocabulary scatter eats the smaller read. Needs a large chat corpus.

### Decode items D2, D3, D6, D5 (bd8ad77a7 .. 68d8b7f80)

Real-content decode (probe_real, T=0.7, 400 tokens, n-max 2), ms/step, hybrid two lanes:

| change | Swift Q8T + q4_K draft head | base + shared head |
|---|---|---|
| start of this round (shared q8_0 head for Swift) | 40.5-40.8 | 40.2-40.3 |
| own q4_K draft head (Swift) | 39.5-39.7 | - |
| kernel peer copies take byte tails (120 B ids / weights) | 39.4-39.5 | 40.0 |
| eager copies for decode graphs (local only) | 38.1-38.4 | 38.8-39.0 |
| batched eager copies (one kernel + event per split boundary) | 37.8-38.0 | 38.5-38.7 |
| grouped MoE GEMV on by default, + q5_0 / q5_1 | 36.6-36.9 | 37.3-37.7 |
| GEMV groups < 512 rows -> per-matrix (hc_down 18 -> 4 us) | 36.1-36.2 | 37.0-37.1 |
| MMVF steps around the gfx1201 cliff (router 23.5 -> 7.1 us) | 36.0 | 36.5-36.7 |

Swift Q8T decode is now ~62 t/s real content (was 55.6-56.2). Greedy text is unchanged by the copy changes; the
grouped GEMV and the router block size change summation order: KLD at ub 3 vs the old path 0.042 / 0.038 (the
model's perturbation floor is ~0.032), perplexity within error, acceptance within seed noise over 3-5 seeds.
New greedy references (launcher prompt): Swift Q8T d1a9781806fd after the grouped GEMV; base cdfc2aa3aa56 /
62b45bb3f58a after the router change. One transient slow window (15:18-15:22, both models, prefill included, no
kernel fault) coincided with R9700 runtime-PM resumes; later runs with ~10 resumes were normal.
Deferred: folding the shared-expert gate into the router as a 513th row (~0.15 ms/step).

**D7 tried and dropped:** a one-block HC boundary kernel that stages b / inj in LDS (so xn may reuse their memory)
is bit-identical (greedy hashes equal with and without GGML_SCHED_ZERO_BUFFERS) and removes ~2,860 of 3,856 decode
declines, but runs 12 rows serially in one block: decode 36.0 -> 37.8 ms/step. Patch kept in
bench/results/d7-hc-1b/d7_one_block.patch. A parallel version changes the reduction order; not worth ~0.4-0.8 ms.

### Prefill items P6 and P4(a) (180b9325a, a819ecb91)

Cold prefill 4K / 16K / 32K (launcher probe), hybrid = 2 lanes EXP_FROM=11 ub 2048:

| build | Swift Q8T APU-only | Swift Q8T hybrid | base APU-only | base hybrid |
|---|---|---|---|---|
| before P6 | 904 / 839 / 763 | 1915 / 1901 / 1764 | 922 / 853 / 776 | 1901 / 1915 / 1775 |
| P6 paired gate/up + down on F16 | 936 / 884 / 799 | 1994-2012 / 1985-1993 / 1833-1840 | 949 / 894 / 812 | 1986 / 1951 / 1801 |
| + P4(a) f16 expert-input crossing | (not taken) | 1807-2002 / 2019-2041 / 1877-1882 | (not taken) | 1948 / 2084 / 1936 |

P6: KLD vs unfused at ub 2048 0.0185 (base) / 0.0146 (Swift). P4(a): bit-identical where the F16 expert path takes
the ubatch, KLD 0.0275 at ub 1024. APU-only Swift Q8T prefill vs gufo (1359/1431/1410): 69% / 62% / 57%.

### State at the end of the 2026-09-27 round (reference for the next gate)

Current greedy references (q38_hybrid_srv.sh launcher, LLAMA_PREFILL_LANES=2 EXP_FROM=11 -ctxcp 0):
- Swift Q8T + mtp-Swift1.5-shared-exps-q4k-head-q4_K.gguf: short 0e61acbea328, long-prompt c30330dde10b
- base UD-Q4_K_XL + mtp-...-shared-exps-q4k.gguf: short cdfc2aa3aa56, long-prompt 34bb40f4d6f1

APU-only real-content MTP decode (sw_accept_apu.sh, seeds 1-2): Swift Q8T + q4_K head 56.7 ms/step, 39.3-39.5 t/s
(gufo APU-only 34.8); base 57.8-58.2 ms/step, 38.5-38.9 t/s (with GGML_CUDA_MMVQ_GROUPED=0 and
GGML_CUDA_GEMV_GROUP_MINROWS=0: 60.2-60.4).

Tried and reverted: MTP eh_proj as a 2D GEMM at ingest widths only (> 8 tokens): prefill +0.5%, within noise.
Draft ingest cost: without the draft head hybrid Swift Q8T prefill is 2274/2283/2122 vs ~2000/2050/1910, so the MTP
ingest (dense draft attention over the whole context + the draft's expert GEMMs on the card, per ubatch) is ~11-12%
of prefill. Open options: a windowed ingest (constant cost; acceptance risk on long prompts; needs the total prompt
length from the server) or an asynchronous ingest overlapping the next pair (no quality change; capped by how busy
the card is).
Not re-tested this round: the two-host GLM path. Eager decode copies are gated on !prefill_pipeline (off with a
remote device), but mainframe needs a smoke run before any of this is pushed.

### Async (deferred) MTP ingest (see the commit after cbbc6bf9e)

Per pair of 2048-token ubatches the draft ingest cost: a full-card synchronize (the whole pair) and 45-80 ms of host
blocking in the draft's decode per ubatch (grows with depth: dense draft attention), during which nothing else was
submitted. Deferring the hook past the next submission helps only when one llama_decode holds several pairs, and
even then only ~1.8% (-b 16384), because the ingest is card work and the card is the busy device in a pair. With
the server's 4096-token batches: +0.5%. A q8_0-expert draft does not make the ingest cheaper (prefill same, decode
-1%). Remaining ingest levers change what the draft computes: sparse (QSA) draft attention at depth, or a windowed
ingest - both need an acceptance gate.

### QSA for the MTP draft (tried; opt-in LLAMA_MTP_QSA=1)

The draft block now can attend through the trunk's QSA path (its own indexer, an indexer cache for its layer). It is
not a win on this box: the per-ubatch ingest is slower on the R9700 at 16-32K (126 vs 110 ms per 2048-token ubatch at
~30K; dense FA on gfx1201 is tuned, sparse FA only reached parity at 13K in the GLM work), prefill unchanged, and at
19-23K-token contexts (long_accept.sh: 3 prompts x 2 seeds, T=0.7) acceptance drops 0.759 -> 0.732 and decode 57.4 ->
54.0 t/s. The head looks trained dense; the converter's ratio 0 for the MTP block was right. The draft ingest's ~11%
of hybrid prefill stays; a windowed ingest is the remaining lever and trades long-prompt acceptance.

## 2026-09-28: final round, step 1 (review 0927 plan, ~/bench/results/q38rev-0927/PLAN.md)

**Zero-code wins, confirmed on the base model at 6ed31ea3c..e6bb60de4 (gibson, two runs / two seeds each):**

| config | change | before | after |
|---|---|---|---|
| hybrid | `-ub 2560 -b 5120` + draft with its own q4_K head | 2074-2080 / 2097 / 1944-1946 tok/s, 36.5-36.6 ms/step | 2086-2088 / 2211-2227 / 2025-2031, 35.4-35.5 ms/step |
| APU-only | `-ub 4096` (draft loaded) | 895-896 / 846-847 / 764-767 | 937-946 / 865-871 / 778-782 |
| APU-only | draft with its own q4_K head | 58.3-58.7 ms/step, 38.3-38.5 t/s | 55.2-56.0 ms/step, 39.9-41.4 t/s |

The base head is the target's q8_0 `output.weight` requantized to q4_K (COOKBOOK recipe 6); acceptance 0.62-0.65 vs
0.63-0.64. The launchers now default to these (q38_hybrid_srv.sh: lanes 2, ub 2560, b 5120, q4_K-head draft;
q38_prefill_srv.sh / sw_accept_apu.sh: ub 4096). Without `LLAMA_PREFILL_LANES=2` the hybrid is 1420-1490 at 16K.
The review's "APU decode 34 t/s at ub 2048" lead was the one-request sanity check: 55.2-55.8 ms/step at ub 2048.

**Correctness items (A2-A5), no perf change intended:**
- P4 crossing (6ed31ea3c): f16 only where the F16 expert GEMM takes the ubatch (>= 40 rows per expert), experts in
  GPU memory only; the cast pair stays (as f32) on smaller prefill ubatches so both lanes keep one split structure -
  without that a 4K prompt's 2037 + 1666 lanes ran sequentially (2013 -> 1374 t/s).
- f16 saturation in mmid-f16 / mmq-wmma (fc210e5b9); scheduler: input events cleared per compute, copy-event ring
  wrap guard, P6 output/view checks, grouped-GEMV width bound, small-eager keyed on has_remote_backend, ingest
  events freed after synchronize (e83948ad4); tests for the whole MoE block and the round's edge shapes (fde611e5f),
  all pass on gfx1201 and gfx1151.
- `GGML_CUDA_GDN_PERTURB=<eps>` (e6bb60de4) for KLD floors.

**References (Swift Q8T + q4_K head, hybrid, lanes 2, -ctxcp 0, lanes_greedy.py, 48 tokens):** the 09-27 greedy
references in this document (0e61acbea328 / c30330dde10b / cdfc2aa3aa56 / 34bb40f4d6f1) are stale. At ub 2560
after A2: 4K a18289237479 (sum lp -0.021129), 9K 5fa5e1bdc18e, 16K and 32K 0c4db8b0c917 (bit-identical to
27ad2f27a: -0.061230 / -0.220081). Files: ~/bench/results/q38r3-gate/runs/.

**Default-on switches added since 09-24 (set to disable):**

| switch | default | what |
|---|---|---|
| GGML_SCHED_IGPU_EVENTS=0 | on | scheduler events on the APU |
| GGML_CUDA_MAX_GRAPHS | 256 | graph cache entries |
| LLAMA_PLE_WILLNEED=0, LLAMA_PLE_THREADS | on, 8 | PLE gather read-ahead + threads |
| LLAMA_LANES_SPLIT_FRAC / _MIN | 0.55 / 1024 | short-batch lane split |
| GGML_CUDA_KQ_WMMA=0 | on (RDNA3/4) | k-quant dense GEMM on dequant-once WMMA |
| GGML_CUDA_MMVQ_GROUPED=0 | on | grouped MoE GEMV |
| GGML_CUDA_GEMV_GROUP_MINROWS | 512 | per-matrix kernels below it |
| GGML_CUDA_MMVF_NO_CLIFF=0 | on | MMVF dispatch-cliff guard |
| GGML_SCHED_NO_COPY_BATCH=1 | off (batching on) | batched eager copies |
| LLAMA_LANES_SMALL_EAGER=0/1 | = !has_remote_backend | eager copies for decode graphs |
| GGML_CUDA_MMID_F16_PAIR=0 | on (gfx11) | P6 paired gate/up + GLU + down |
| LLAMA_MOE_F16_CROSS=0 | on | P4 f16 expert-input crossing |
| LLAMA_MTP_INGEST_DEFER=0 | on (local lanes) | deferred MTP early ingest |

Opt-in: LLAMA_SPEC_DRAFT_UB, LLAMA_MTP_EH_PROJ_2D, LLAMA_MTP_QSA. Presence-tested switches (any value, including 0,
turns them on): LLAMA_QSA_ALWAYS_SCORE, GGML_CUDA_NO_GDN_CHUNKED.

### 09-28, later: GLM re-gate, soak, calibration, producer-side MMQ copies

- **GLM two-host (A1):** passes on 880d979de with all defaults - 25.8K prefill 898 t/s, 92.9 ms/step, 30.1 t/s, acc 0.90,
  greedy aa0b8bd157cde3d8 (new reference). The first attempts crashed mainframe's rpc-server at the first decode: the
  grouped MoE GEMV with APU scheduler events in the SERVER's own ggml_backend_sched. Bisected on the server's env;
  a race (gone under AMD_SERIALIZE_KERNEL). 585b06b4e (source waits on an APU destination before a copy) is correct but
  not the fix; 880d979de keeps the server's scheduler on host syncs for the APU (step unchanged). Root cause open.
- **Soak (A7):** 120/120 fresh hybrid servers served a 4.4K first request + the 1.9K long prompt, no hang (3.1-4.0 s).
- **Kernel boundary cost:** GGML_CUDA_PAD_KERNELS=4 (~460 empty kernels per verify step) costs +0.6-0.7 ms/step, i.e.
  ~1.5-2 us per boundary on the decode critical path. B8 (more q8_1 side copies) measured nothing; a fused hc
  up+mix kernel (B7) was 4 ms/step slower APU-only (80 blocks of serial work lose to the tuned GEMV) and its f32
  activations moved KLD above the floor (Swift C 0.031 vs 0.023) - dropped.
- **Producer-side MMQ copies (ea92b2d17):** combine_norm and hc_mix write the block_q8_1_mmq copy their q8_0 MMQ
  consumers would quantize (bit-identical); flat q8_1 side copies capped at decode widths (prefill wrote ~30 MB per
  combine_norm for nothing). Hybrid 16K 2208-2217 -> 2230-2235, 32K 2020-2023 -> 2032-2039; refs identical.

### 09-28, C6 + C4: the QSA indexer at depth (782542e4e)

- **C6 block-key cache:** the finished indexer key of every complete block persists per QSA layer; decode recomputes
  only the trailing blocks its tokens touch (fixed count, so graphs stay reusable). ~20K contexts, T=0.7: hybrid
  42.9 -> 40.9 ms/step, APU-only 67.5 -> 64.4. Bit-identical on the APU; on the card within the hybrid's run-to-run
  noise (KLD 0.0065 vs 0.0070 for the unchanged build). Traps met: Qwen3.8's text positions are 4-section mrope rows, so
  ubatch.is_pos_2d() is always true (the plan tests "no repeated position" via the hole-free check instead); inputs the
  new graph no longer reads (blk_cells, blk_pos, cell_blk) are unallocated and set_input fills host shadows.
- **C4 block-level top-k:** GGML_OP_QSA_TOP_K replaces the per-cell expansion + radix top-k when cell j = position j.
  Not bit-identical (the old top-k broke ties inside the cut block arbitrarily): KLD 0.0014 / 0.0013 at 16K, floors
  0.0096 / 0.0089. APU-only prefill 16K 852-860 -> 891, 32K 762-768 -> 820-822; hybrid 32K 2041 -> 2107-2116.
- RPC proto 7.7 (the op): both GLM hosts must run it.

### End of the 09-28 round: full sweep on f05fe5f29 (gibson, ~/bench/results/q38-final/sweep.txt)

| config / model | prefill 4K / 16K / 32K (t/s, cold) | decode short (ms/step, t/s) | decode ~20K (ms/step, t/s) |
|---|---|---|---|
| hybrid, base | 2078 / 2300 / 2141 | 35.3-35.4, 63.0-63.1 | 40.0, 61.9 |
| hybrid, Swift Q8T | 2052 / 2285 / 2129 | 35.8-35.9, 60.5-63.8 | 40.4, 60.3 |
| APU-only, base | 954 / 892 / 821 | 55.7-56.2, 40.1-40.6 | 62.8, 39.1 |
| APU-only, Swift Q8T | 964 / 893 / 823 | 56.5-56.8, 39.4 | 63.7, 38.8 |
| GLM-5.3 two-host (gate) | 12.7K 783, 25.8K 896 | - | 92.2 at 25.8K, 30.3 |
