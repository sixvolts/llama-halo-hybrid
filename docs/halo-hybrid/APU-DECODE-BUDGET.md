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

## Where the 26% goes

`rocm-smi --showuse` sampled through decode: the iGPU is busy **82.9%** of the time (mean of 39 samples, max 99%).

| per token | ms | share |
|---|---|---|
| DRAM traffic at the ceiling | 28.4 | 74% |
| kernel time beyond the bytes (DeltaNet, attention, norms, topk, hc mixing) | ~3.3 | 9% |
| GPU idle between kernels | ~6.6 | 17% |

With ~1,333 kernels per token that idle is ~5 us per kernel boundary, which matches the boundary cost measured
directly for the persistent-decode work (a real graph node costs ~6 us against 1.74 us for an empty one).

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

**60 of ~1,333 launches, under 1% of the token.** The remaining GEMVs each read a different activation: the
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

The honest remaining program is the one halogen ran: get from ~1,333 kernels per token to ~500 by folding norms,
gates, the router and sampling into the projection kernels, at ~5 us of boundary each. Nothing smaller moves this
model, and the draft head (MTP) remains worth more than all of it, since it amortises the whole 6.33 GB over 2-3
tokens.
