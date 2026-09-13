# llama.cpp — Strix Halo + R9700 hybrid fork

This tree is upstream `ggml-org/llama.cpp` (plus the open `qwen4exp` PR #27742 for
Qwen3.8-Flash-Next) with the changes made while getting Qwen3.5-122B-A10B to run *across*
a Ryzen AI MAX+ 395 "Strix Halo" APU (128 GB unified LPDDR5X, gfx1151) and a Radeon AI PRO
R9700 (32 GB GDDR6, gfx1201) on a PCIe Gen4 x4 link, built with `GGML_HIP=ON`.

Headline result (Qwen3.5-122B-A10B, UD-Q4_K_XL, batch 1): 24.3 t/s APU-only →
36.3 t/s plain / 49.3 t/s with the MTP draft head, from placement flags plus the patches
below. The full investigation is in the project report; this file only lists what the
tree changes and how to switch each piece off.

## The layout ("hybrid")

Every layer is *assigned* to the R9700 (keeps the fused Gated-DeltaNet path enabled), and a
tensor override moves only the routed experts of the higher layers to the APU:

```
llama-server -m model.gguf -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 \
  -ot 'blk\.(1[6-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1'          # hybrid-16
# with the MTP draft on the card: hybrid-14 + -md <draft> --spec-type draft-mtp -devd ROCm0
#   --spec-draft-n-max 4 --spec-draft-p-min 0.5
```

Repeated `-ot` flags accumulate and patterns may be comma-joined in one flag (`common/arg.cpp`); the first pattern that matches a tensor wins (`src/llama-model-loader.cpp`). The launchers comma-join into a single flag.
For Qwen3.8-Flash-Next (unsloth GGUF: three expert tensors per layer) keep the 28.8 GB PLE n-gram
table in host memory: `-ot 'blk\.(1[6-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1,per_layer_token_embd=CPU'`
(hybrid-16: 4.6 GB dense + 16 expert layers on the R9700, 26.9 t/s before the fusions below; 37.4 t/s with everything in this document, hybrid-14 36.7).
The table's 16-row gather then runs on the host inside `set_input` (no CPU split, see below).
It can also be placed on the APU (`per_layer_token_embd=ROCm1`) now that `get_rows` on IQ4_NL accepts
160-wide rows — upstream's HIP backend declined those and silently fell back to a CPU gather of a
device-resident 28.8 GB tensor, which drained host memory and page-faulted — but a device gather
is a split either way and measured ~0.7 t/s slower than the host gather.

## Changes, on by default

| Where | What | Off switch |
|---|---|---|
| `ggml/src/ggml-backend.cpp` | scheduler events are created regardless of `n_copies` for backends that implement `cpy_tensor_async_nowait` (local CUDA/HIP; not RPC or CPU), so split boundaries are GPU-side waits instead of host syncs (+5–7% on this layout); other backends keep upstream's host synchronize (see the GLM two-host table). Eager-copied input copies are allocated at their producer split (`GGML_SCHED_NO_EAGER_PRODUCER_ALLOC=1` reverts): the copy lands right after the producer, on the source stream, and a copy allocated at the consumer could share memory with destination splits in between; the ordinary consumer-time copies are ordered by the alternating layout's data dependencies. Measured and rejected: a two-hop staging copy (a pull from the destination reads wrong bytes on R9700 ↔ APU; a push on a copy stream is correct but 25% slower) and dedicated input-copy memory via `ggml_set_output` (nondeterministic greedy output) | `GGML_SCHED_NO_EVENTS=1` |
| `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmvq.cu` | grouped `mul_mat_vec_q`: quantise a shared activation once for adjacent matmuls; `qwen35moe.cpp` reorders projections so they are adjacent (bit-exact) | `GGML_CUDA_NO_MMVQ_GROUP=1` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | cross-device copies ≤ 256 KB as a push kernel on the source device into peer-mapped memory (+1.5%) | `GGML_CUDA_KERNEL_COPY_MAX=0` (SDMA), or a byte limit |
| `src/llama-graph.cpp` (`qwen35moe.cpp` part not yet re-applied on this base) | recurrent-state gathers become views when the copy map is the identity; one `l2_norm` over q and k instead of two (bit-exact, +3.7%) | `LLAMA_NO_GRAPH_FUSE=1` |
| `ggml/src/ggml-cuda/hc.cu`, `ggml-cuda.cu` (`ggml_cuda_try_fuse_hc`) | Qwen3.8-Flash-Next (`qwen4exp`) launch-count fusions, matched by op pattern with data-flow, use-count and aliasing checks: hyper-connection mix (sigmoid·mul·stream-sum·scale → 1 kernel), hyper-connection combine (repeat·scale·sigmoid·scale·mul·add → 1), scale+silu, x·sigmoid(g) (GDN output gate, attention gate, shared-expert gate), MoE weighted expert sum (mul + fused adds → 1), GDN gate chain (add·softplus·mul → 1). Bit-identical to the unfused graph (`__fmul_rn`/`__fadd_rn`, same summation order); a matcher declines when ggml-alloc placed the output over an input it reads non-elementwise | `GGML_CUDA_NO_HC_FUSE=1` |
| `ggml/src/ggml-cuda/hc.cu`, `mmvq.cu`, `norm.cu` | **q8_1 side copies**: a fused kernel that produces a single-row f32 activation also writes its q8_1 quantisation (same `warp_reduce` arithmetic as `quantize_q8_1`, so identical bits) into a per-graph arena; `mul_mat_vec_q` consumers (plain, GLU-fused, grouped) pick it up by data pointer, validated against the producing tensor, instead of launching a quantise kernel | `GGML_CUDA_NO_Q8_SIDE=1` |
| `src/models/qwen4exp.cpp` | graph tidy-ups for the same model: no copy of stream 0 in the hyper-connection sum, strided `cpy` for the conv-state tail, same-input projections pinned adjacent (grouped `mul_mat_vec_q`), norm-weight view hoisted so `rms_norm+mul` fuses, one `l2_norm` over q and k, one QSA/KV-index input set shared by all attention layers (was one synchronous host→device upload per layer per token; upstream master now does the same, and the branch uses upstream's) | `LLAMA_NO_GRAPH_FUSE=1` (l2_norm) |
| `src/models/qwen4exp.cpp`, `ggml/src/ggml-cuda/getrows.cu`, `dequantize.cuh`, `ggml-cuda.cu` | **PLE n-gram table**: with the table in a host buffer the 16-row gather + dequant runs on the host in `set_input` (same `to_float` as the CPU `get_rows`, bit-identical) and is fed as an F32 input, instead of a `get_rows` node that forced a CPU split with a synchronising host→device copy each token (72 → 70 splits, 33.9 → 33.0 ms/token cold decode). `get_rows` on IQ4_NL uses the generic 32-block gather, so the table may also live on a GPU | `LLAMA_PLE_GET_ROWS=1` (in-graph gather) |
| *(superseded)* `src/models/qwen4exp.cpp` | **graph reuse** for the QSA/PLE inputs (`can_reuse`): implemented here first (cold decode 33.0 → 29.3 ms/token, +10% on the server); upstream master now carries its own, so the branch uses upstream's. Kept in the table because the numbers above include it | `LLAMA_GRAPH_REUSE_DISABLE=1` |
| `ggml/src/ggml-cuda/mmvq.cu` | **small-K matvec mode on RDNA4**: upstream disables `mul_mat_vec_q`'s rows-per-block ("small_k") mode on every RDNA part, so a short-K row (Qwen3.8's 96 hyper-connection up-projections/token are Q8_0 [10240 × 320]) ran as one 256-thread block per row doing a single loop trip: 23 µs, 150 GB/s. Enabled for RDNA4: 7 µs, 490 GB/s; K=640 shared-expert down 6.3 → 3.5 µs; large-K shapes unchanged within 1%. Bit-identical; cold decode 29.1 → 27.3 ms/token (−6%) | `GGML_CUDA_MMVQ_NO_SMALLK=1` |
| `ggml/src/ggml-cuda/mmvf.cu` | **load pipelining in the f32/f16/bf16 matvec**: the per-row loop issued one dependent load per trip, which is fine for wide launches but latency-bound for the few-row shapes here (48 f32 routers [512 × 2560] at 25 µs / 210 GB/s, the 4-row hyper-connection inject at 6 µs, ssm alpha/beta). Four loads in flight per thread with the same per-thread accumulation order (bit-identical); cold decode 27.3 → 26.8 ms/token (−2%). Batched `test-backend-ops` cannot see this shape effect (it hides launch latency), only the real run does | none (unfused path only) |
| `src/llama-graph.cpp`, `src/models/qwen4exp.cpp` | **QSA gather at depth** (ported from ucicelos/flashnext-hybrid, their fix 4): above `LLAMA_QSA_GATHER` cells (default 65536) a decode step gathers the ~2k selected K/V rows out of the cache and runs flash attention over exactly those instead of masking the whole cache. Below the threshold the graph is unchanged by construction. Measured here at 68.5K tokens (hybrid-12, no draft): needles at positions 200 and 1400 of 2300 records retrieved with it on and off, identical answer text, decode 23.9 → 25.7 t/s (+7%); prefill unchanged. The padded variant of the original was not ported (it lost needles for them) | `LLAMA_QSA_GATHER=0` (off) or `=<n_kv>` |
| `tools/server/server-context.cpp` | speculative recurrent-state checkpoints are saved and restored with `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` (the `spec_ckpt` update/load pairs); on both shipped archs the `n_rs_seq` rollback path makes them unreachable, they are the fallback for recurrent targets outside `llm_arch_supports_rs_rollback` | none |
| `src/llama-context.cpp`, `src/models/glm5next.cpp` | fused lightning-indexer scoring for the DSA layers (the fused kernel sums heads in a different order, so a near-tied top-k can differ from the unfused path) | `LLAMA_FUSED_LID_DISABLE=1` |
| `src/models/dflash.cpp`, `src/llama-hparams.h`, `src/llama-graph.cpp` | DFlash drafts run their sliding-window layers **causally**, as the z-lab reference does (`is_causal = layer_type == "sliding_attention"`); upstream ran every draft layer bidirectionally. Verified token-for-token against the reference PyTorch draft | `LLAMA_DFLASH_SWA_BIDIR=1` |

## Changes, off by default (experiments kept behind env vars)

The expert-parallel prototype (`LLAMA_EP`) and the DFlash dump/tap diagnostics were dropped when the branch moved to the merged base; they are in the history up to tag `main-pre-mtp`.

| Env | What | Outcome |
|---|---|---|
| `GGML_SCHED_LAZY_INPUTS=1` | cut a split at the first consumer of a cross-backend activation; skip host syncs for no-input splits; source stream waits for the destination's progress before a peer copy | net loss on ROCm |
| `LLAMA_PIPELINE_PARALLEL_FORCE=1` | keep pipeline parallelism on despite `-ot` (`src/llama-context.cpp`; needs `n_batch > n_ubatch`) | 0–2% single-host; cannot overlap the interleaved-split layout; the same 308 / 345 as the lane pair two-host |
| `GGML_CUDA_GRAPH_OPT_MULTI=1` | lift the multi-device gate on the CUDA graph-optimisation pass | −3% |
| `mm_ids_helper_wide` (default for ≥ 512 tokens, `GGML_CUDA_MMID_WIDE_MIN=0` restores the warp kernel) | the MoE expert-id compaction ran one wave per expert walking the tokens two at a time with a ubatch-sized shared-memory store: 182 µs per launch at 1024 tokens, 628 at 2048, 2453 at 4096 (15% of the APU's prefill time). One 256-thread block per expert, two passes (count, stable scan), same order, no shared-memory dependence on the ubatch | prefill at `-ub 2048` +2–10% (1169 / 951 / 846 → 1209 / 1050 / 867 tok/s with two lanes), none at 1024 |
| `GGML_CUDA_MMQ_MOE_J_FACTOR` (default 1, `0` restores upstream) | MMQ picks the column tile `J` that minimizes the tile count for `ncols_max`, which for MUL_MAT_ID is the whole ubatch, so the expert GEMMs always ran `J = 128` while an average expert has 20 tokens at `-ub 1024` (10 of 512 experts per token): 15–30% of each tile's columns were live. The ids path now sizes `J` for `factor × expected tokens per expert` (`ncols_hint`, `J = 32` at `-ub 1024`, 48 at 2048); the grid still covers every column | factor 2 alone (before the tile list): warm two-lane prefill, hybrid-12, 3.7K / 15K / 30K prompts, `-ub 2048` 1178 / 1036 / 853 → 1368 / 1152 / 935, `-ub 1024` 891 / 832 / 723 → 1231 / 1058 / 886; with the tile list factor 1 is best (isolated expert GEMMs at 1024 tokens: gate/up 4.6 → 3.4 ms, down 6.0 → 6.0; at 2048: 6.8 → 5.1, 8.5 → 6.9); output identical |
| compact MoE tile list (default, `GGML_CUDA_MMQ_MOE_TILE_LIST=0` restores the full grid) | the MUL_MAT_ID tiling grid is row tiles × ceil(ubatch/J) × experts and every (expert, column tile) beyond the expert's token count exits at the top: at `-ub 1024`, `J = 48` that is 95% of 56K (gate/up) / 225K (down) blocks. `mmq_moe_tile_list` (one block, a scan over `expert_bounds`) writes the live (expert, tile) pairs and the GEMM grid is sized to `ncols_dst/J + n_expert` | isolated expert GEMMs at 1024 tokens: gate/up −12%, down −24%; with factor 1, warm two-lane prefill 1.9K / 3.7K / 5.6K / 7.5K / 11K / 15K / 22K / 30K: `-ub 1024` 1171 / 1217 / 1194 / 1150 / 1088 / 1061 / 973 / 882 → **1443 / 1470 / 1433 / 1381 / 1277 / 1225 / 1107 / 997**, `-ub 2048` 910 / 1368 / 1213 / 1314 / 1218 / 1152 / 1020 / 935 → **1025 / 1603 / 1380 / 1483 / 1374 / 1284 / 1124 / 1018**; output identical. Profile after (single lane, 3.7K, `-ub 1024`): APU kernel time is 82% `mul_mat_q`, 8.8% `moe_weighted_reduction`, 2.9% `mm_ids_helper_wide`, 2.5% quantize; per layer the three expert GEMMs read 1.85 GB of weights (down is q8_0: 890 MB), so at `-ub 1024` they sit within ~30% of the APU's ~215 GB/s floor |
| rocBLAS instead of hipBLASLt for f32 GEMMs on HIP (default; `ROCBLAS_USE_HIPBLASLT=1` restores) | the R9700's f32 weight matmuls (router `[2560 → 512]`, `hc_*_inject [10240 → 4]`, `ssm_alpha/beta [2560 → 48]`, 5 per layer) went to hipBLASLt, whose gfx1201 sgemm solutions are 8x8 macro-tile fallbacks: the router ran 1.0 ms per layer at 1024 tokens (2.5 TFLOPS), 19% of the R9700's prefill kernel time. `ggml_cuda_init` sets `ROCBLAS_USE_HIPBLASLT=0` unless the user set it | isolated on the R9700 at 1024 tokens: router 1083 → 229 µs, `[2560 → 48]` 124 → 139 µs; f16 GEMMs are slower under rocBLAS (router shape 56 → 298 µs) but this model has no f16/bf16 weights |
| swapped MMVF for thin f32 matrices (default; `GGML_CUDA_NO_MMVF_SWAP=1` restores cuBLAS) | a `[K → n]` f32 matmul with `1 < n ≤ 8` rows and many tokens is a batched dot product over the activations; like the existing `ne01 == 1` case it runs `mul_mat_vec_f` with the operands swapped (activations as the "weights", the `n` rows as the batch) into a `[tokens, n]` scratch and transposes it into `dst` with the copy kernel | `hc_*_inject [10240 → 4]` at 1024 tokens: 214 → 32 µs (activations in the infinity cache), 235 → 163 at 2048. Both together, warm two-lane prefill 1.9K … 30K: `-ub 1024` 1443 … 997 → **1467 / 1503 / 1471 / 1417 / 1322 / 1258 / 1137 / 1025**, `-ub 2048` 1025 … 1018 → **1069 / 1626 / 1425 / 1540 / 1416 / 1324 / 1159 / 1049**; R9700 prefill kernel time −12% (3323 → 2912 ms per profiled 2×3.7K), the two lanes are now equal (2912 / 2930 ms). The f32 summation order changes, so greedy text can differ from the hipBLASLt build (it did at `-ub 2048`, not at 1024) |
| `LLAMA_PREFILL_LANES=2` | two-lane prefill: consecutive prefill ubatches on two schedulers with interleaved splits, eager cross-device copies, stream-ordered input copies, a lane-aware server tail chunk (see "Two-lane prefill") | prefill +30–45% at `-ub 1024`/`2048`; decode unchanged; a second set of compute buffers |
| `LLAMA_LANES_DEBUG=1` | log every computed pair (tokens, outputs, splits, wall, tok/s) after a host sync | diagnostic |
| `GGML_SCHED_TRACE_WAITS=1` | per graph: count of the scheduler's host-blocking points (no-input splits, synchronous input copies, sync-copy fallbacks) and the host submit time split into copies / compute; names the first 48 input copies | diagnostic |

## Qwen3.8-Flash-Next with the MTP draft head (on `main`; the pre-MTP history ends at tag `main-pre-mtp`)

This branch is ggml-org `master` (3466812d1, the merged `qwen4exp`) + unslothai/llama.cpp#144
(NextN/MTP draft head, draft-only exports, borrowing the target's embeddings and head, conv-state
rollback, CUDA graph key by shape) + ggml-org#28118 (speculative recurrent-state checkpoints on-device; on HEAD
both shipped archs take the `n_rs_seq` rollback path instead, so the on-device checkpoints are only the fallback for
recurrent targets that cannot roll back) + the patch set above ported onto it. Upstream now has graph
reuse for the QSA/PLE inputs, QSA input sharing and a MoE weighted-reduction fusion of its own, so those
parts of the old branch were dropped; so were the expert-parallel prototype and the DFlash dump
diagnostics.

Draft head: `unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (2.6 GB). The
`shared-` file borrows `token_embd` and `output` from the target, so the draft must live on the device
that holds them (the R9700):

```
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 8192 --no-mmap -np 1 \
  -ot 'blk\.(1[4-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1,per_layer_token_embd=CPU' \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

Hybrid-14 leaves room for the head on the R9700; hybrid-16 does not. Measured (8k context, 256-token
completions, sampler temperature 1.0 / top-p 0.95 / top-k 20 as the model card recommends; greedy in
brackets):

| build | no draft | MTP n-max 2 | acceptance (sampled / greedy) |
|---|---|---|---|
| stock new base | 27–28 (27–28) | 35–38 (38–41) | 0.50 / 0.60 |
| this branch | 35–37 (35–36) | **42–45 (47–52)** | 0.51 / 0.70 |

Full table on the merged base (model-card sampler, 4K prompts, 256-token completions, `-b 4096 -ub 1024`;
`bench/streams_q38.py` drives N concurrent chat requests, "agg" sums the streams):

| streams | layout | prefill agg tok/s | decode, no draft | decode, MTP n-max 2 |
|---|---|---|---|---|
| 1 | hybrid-12 | 610 (16K prompt: 623) | 34.3 | 45.4 (greedy ~52) |
| 2 | hybrid-12 | 627 | 49.0 agg / 26.0 each | 53.0 agg / 29.8 each |
| 4 | hybrid-12 | 642 | 66.3 agg / 17.9 each | does not fit with the head |
| 4 | hybrid-10 | 585 | 47.8 agg / 12.7 each | 51.5 agg / 14.0 each |
| 1 @ 68K ctx | hybrid-12 / hybrid-10 | 413 (`-ub 1024`) / 306 (`-ub 512`) | 25.7 | 43.6 |

Four-stream decode moves by up to 30% between sessions (a hybrid-1 run gave 61.6 agg plain / 49.3 MTP), so the
multi-stream rows say "the head is about break-even at four streams", nothing finer. The flashnext-hybrid fix 5
(uniform draft lengths across streams) is not ported; it is what makes the head pay at four streams there.

`--spec-draft-n-max 3` is equal within noise, 4 collapses (acceptance 0.32), and `--spec-draft-p-min`
0.5–0.7 raises acceptance but not throughput. The hyper-connection fusions and q8 side copies still
pay under speculation (all switches off: 41–43 sampled).

Two things to know:

* **Greedy output with the draft is not identical to greedy output without it**, and this is not a
  speculative-decoding bug: the model's own logits depend on how tokens were batched. With no draft,
  `-ub 512` puts `<think>` at 0.53 vs `Unified` at 0.36 for the first token of a test prompt, `-ub 8`
  0.48/0.39, `-ub 2` 0.47/0.44, `-ub 1` 0.38/0.50. The chunked (multi-token) and recurrent
  (single-token) delta-net paths of upstream `qwen4exp` disagree by ~0.1 in probability; the verify
  batch simply lands on the multi-token side. Worth an upstream report; it affects chunked prefill too.
* Where an MTP round goes (rocprofv3, n-max 2, greedy; shares, since profiling inflates absolute
  times ~2.7×): the 3-token verify forward is ~92% of the round, the two draft steps ~6%, host gaps
  ~1–3%. Inside the verify forward the dense matvecs cost about what they cost for one token (the
  multi-column `mul_mat_vec_q` config on RDNA4 already streams at 530–545 GB/s for 2–4 columns), the
  extra is the APU expert traffic: three tokens select up to 30 experts per layer, so APU busy goes
  5.5 → 13.5 ms. That is inherent to MoE speculation. Extending the q8 side copies to ≤4-row
  activations was exact but measured neutral (the verify batch is not launch-bound), so it was not
  kept; the remaining host overhead (~85 stream syncs per round from input uploads, logits/embedding
  readbacks and per-backend scheduler syncs) is worth ~3% and has no single owner.
* This branch's greedy output differs from the stock new base's, but the port itself is exact: with
  `GGML_CUDA_DISABLE_FUSION=1` (plus this branch's own switches off) the stock graph and the ported graph
  produce byte-identical text, whereas the stock graph with upstream's fusions on differs from itself
  with them off. Upstream's rms_norm+mul+rope / MoE-reduction fusions are not bit-exact, and the graph
  edits here (hyper-connection order, host PLE gather, projection adjacency) change which of them fire.

Prefill ubatch sweep (hybrid-12, no draft, 4K / 16K / 32K prompts): `-ub 1024` 593 / 618 / 543 tok/s, `-ub 2048`
705 / 731 / 641, `-ub 4096` 781 / 804 / 689; decode unchanged (33–34 t/s). Compute buffers: 0.75 / 1.5 / 3.0 GB on
the R9700. Pipeline parallelism across ubatches (`LLAMA_PIPELINE_PARALLEL_FORCE=1` keeps it on despite `-ot`;
the server logs "pipeline parallelism enabled") measured 0–2% at every ubatch: a kernel trace of a 3.7K prefill at
`-ub 4096` shows the R9700 busy 42% of the wall and the APU 42%, summing to 100%, so the devices run strictly in
turn (resolved below: two-lane prefill). On the APU the expert GEMM runs at ~6 TFLOPS at `-ub 1024` and ~13.5 at
4096 (90% / 69% of APU kernel time); `mm_ids_helper` is 15% and the MoE reduction 8% at 4096, which is the cheap
part to fix.

### Two-lane prefill (`LLAMA_PREFILL_LANES=2`)

Why the devices never overlapped: with the experts on the APU every ubatch's graph is a chain of 74 splits that
alternate R9700 / APU / R9700 ..., and the scheduler submits one graph at a time onto one in-order stream per
device. Ubatch k+1's first R9700 split therefore queues behind ubatch k's last R9700 split, which is itself
waiting on the APU, so pipeline parallelism (`LLAMA_PIPELINE_PARALLEL_FORCE=1`) cannot overlap anything here;
it only works for layer splits where the two devices' work is not interleaved. Kernel traces of every variant
showed the R9700 busy 36%, the APU 46%, and both busy 0 ms.

What was needed (all in `ggml/src/ggml-backend.cpp`, `ggml-cuda.cu`, `src/llama-context.cpp`):

1. **Two schedulers, interleaved splits.** `llama_context` builds consecutive prefill ubatches on two
   `ggml_backend_sched` instances (own compute buffers, shared backends and streams) and
   `ggml_backend_sched_graph_compute_async_pair` submits their splits alternately (a0 b0 a1 b1 ...). Lane b's
   attention then sits in the R9700 queue right behind lane a's, ahead of lane a's wait for the APU.
   Cross-lane ordering (b reads the KV/recurrent state a wrote at the same layer) follows from the stream order.
2. **Stream-ordered input copies** (`ggml_backend_sched_set_async_inputs`). Graph inputs live in host memory and
   the scheduler copied each one to its device with a host wait for that device's previous split plus a
   synchronous copy. Ten inputs are consumed at the first split, but the output-row selector is first used at
   split 71 of 74, so the host blocked there until almost the whole graph had run and could not submit the other
   lane. `GGML_SCHED_TRACE_WAITS=1` prints these waits per graph. Now `ggml_backend_tensor_set_async` on the
   destination stream; `prepare_ubatch` synchronizes a lane before rewriting its host inputs. Zero-byte inputs
   (the selector on ubatches without outputs) are skipped instead of waited for.
3. **A pool of copy events** (`ggml_backend_cuda_context::next_copy_event`, 512 per device). On ROCm a queued
   `hipStreamWaitEvent` resolves against the event's newest record when the queue reaches it, not the record at
   call time as on CUDA (`docs/halo-hybrid/lane_pattern.hip`: the two-lane pattern takes 599 ms with distinct
   events, 873 ms with one re-recorded event per device, 780 ms if fully serialized). `cpy_tensor_async`
   re-recorded one `copy_event` per device on every cross-device copy, so every earlier wait became a wait for a
   later copy. This alone also serializes upstream's pipeline parallelism on ROCm.
4. **Eager copies** (`ggml_backend_sched_set_eager_copies`, backend iface `cpy_tensor_async_nowait`). The
   scheduler enqueued a cross-device copy on the producer's stream only when the consumer split was submitted,
   i.e. in the interleaved order a0 b0 a1 the copy of a0's output landed on the R9700 stream behind b0 and the
   APU started a1 only after b0 (the trace showed the first expert block starting after both trunks). Now the
   copy is pushed right behind its producer and the consumer waits on a pooled event. Same caller contract as 2:
   synchronize before the next graph on that scheduler (`graph_compute` does it when lanes are on).

5. **The server's prompt chunking** (`tools/server/server-context.cpp`). For models with recurrent state the
   server ends the prompt batch `4 + n_ubatch` tokens before the end (a checkpoint is taken there), processes that
   ubatch alone, then the last 4 tokens. So the last `n_ubatch` tokens of every prompt ran single-lane and prompts
   under ~2·n_ubatch never paired at all (a 3.7K prompt at `-ub 2048` showed no gain; per-pair timing with
   `LLAMA_LANES_DEBUG=1` showed the pairs that did form at 1230–1270 tok/s). With `LLAMA_PREFILL_LANES=2` the
   break is now `4 + 2·n_ubatch` before the end, so the tail is a pair; a checkpoint restore then re-processes up
   to two ubatches instead of one. `-b` must be at least `2 × -ub` for pairs to form at all.

Items 1–3 each measured zero gain on their own; 4 made the pairs overlap and 5 made the pairs form. Output is byte-identical to single-lane
(greedy text compared at 4K). `rocprofv3 --kernel-trace` serializes dispatch across the two agents and shows
0 ms overlap even when throughput says otherwise, so it cannot measure this.

Cost: a second set of compute buffers (R9700 0.75 GB at `-ub 1024`, 1.5 GB at 2048; APU 0.29 / 0.57 GB).
Decode is untouched (single-token ubatches never pair; 33–34 t/s no draft, 45 t/s with the MTP head as before).

Measured (hybrid-12, warm, 4K / 16K / 32K prompts, tok/s; `-b 4096`):

| | `-ub 1024` | `-ub 2048` | `-ub 4096` |
|---|---|---|---|
| single lane | 676 / 616 / 540 | 830 / 730 / 640 | 781 / 804 / 689 |
| `LLAMA_PREFILL_LANES=2` | 891 / 835 / 722 | **1179 / 1041 / 858** | does not fit (2 × 3 GB) |

The launch line with the MTP head (hybrid-12, `-ub 1024`): 644 / 592 / 510 → 833 / 784 / 678, decode unchanged
(46.6 greedy, 44.9 sampled). Two and four streams at `-ub 1024`, measured before the last two fixes: aggregate
prefill 627 → 736 and 634 → 747, decode 48.9 → 50.2 and 66.6 → 63.1 (that row's usual noise band). Prompts are
processed at one to two decode calls per 4K, so the gain grows with prompt length up to the point where the
32K attention cost dominates.

## GLM-5.3-Flash across two hosts (on `main`; the pre-rewrite history is on branch `glm53-flash`)

GLM-5.3-Flash (unsloth UD-Q4_K_XL, 200 GB; 45 layers + 1 MTP block, 288 experts / 8 used, 34 KDA linear-attention
layers, 11 DSA sparse-attention layers, mHC hyper-connections) does not fit one box, so it runs across gibson and a
second Strix Halo (`mainframe`, 128 GB, no dGPU) over a direct 100G Intel E810 link with llama.cpp's RPC backend.
This work is on `main`: upstream PR #27754's `glm5next` plus the fixes below. Launcher and MTP export:
`docs/halo-hybrid/run_glm_two_host.sh`, `docs/halo-hybrid/export_mtp.py`. Both hosts must run `ggml-rpc-server` and `llama-server`
from the same commit of this tree: the RPC wire format is fork-local (graph compute replies, protocol major 7); a mismatched
pair is refused at the handshake ("RPC server version mismatch" on the client) instead of hanging with no error on either end. With a remote device `LLAMA_PREFILL_LANES=2` runs the rolling prefill pipeline below (`LLAMA_PREFILL_LANES_RPC=1` keeps the older pair schedule); launch with `-b 16384` so the pipeline is not drained at every logical batch, and with the experts of layer 5 on the APU (`APU_FROM=5` in the launcher): two lanes plus the draft head sit at 30.3 of the R9700's 32 GB otherwise and the draft's compute buffers fail to allocate on one launch in three. If gibson goes down, restart mainframe's
`ggml-rpc-server` too (it spins on the dead connection and keeps the model resident). The R9700's runtime power
management is pinned on (`/etc/udev/rules.d/90-r9700-no-runtime-pm.rules`) after the card dropped off the PCIe bus
following a resume from D3 during an interactive session (`device lost from bus`, host power-cycled).

**Layout.** Contiguous by layer so the link is crossed twice per token: layers 0–24 on gibson (dense trunk, KV and
the experts of layers 3–5 on the R9700, experts of 6–24 on the iGPU), layers 25–44 on mainframe's iGPU exposed as
`RPC0` by `ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0`, output head pulled back to the R9700. Per token the
residual (64 KB) goes out and the last hidden state comes back; the split is expressed as
`-dev ROCm0,RPC0,ROCm1 -ts 25,22,0` (layer `il` goes to the device whose cumulative split exceeds `il/47`, so the
denominator is `n_layer_all + 1`). RDMA over RoCE is negotiated automatically once both builds have `GGML_RPC_RDMA`;
its bytes do not show in the netdev counters, only in `/sys/class/infiniband/irdma0/ports/1/hw_counters`. The 86 GB
remote load takes ~3 minutes and is bound by the loader's read-then-send loop, not the link (TCP 561 MB/s, RDMA
~500 MB/s, iperf3 27.6 Gbit/s).

**Fixes it needed** (each found by bisecting two-host runs with the mainframe session; a run costs 4 minutes):

| Change | Symptom | Effect |
|---|---|---|
| `ggml-rpc/transport.cpp`: retry `ibv_create_qp` with inline 256 → 64 → 0, log why a connection stays on TCP | irdma caps `max_inline_data` at 101, the QP creation failed and both peers dropped to TCP without a word | RDMA active (`RDMA probed ... inline=64`, `RDMA activated`) |
| `ggml-backend.cpp`: the always-on split events (this branch's two-GPU optimisation) only for backends with `cpy_tensor_async_nowait` | the asynchronous RPC backend implements events, so `event_synchronize` replaced the full synchronize the split loop relied on: garbage from token one, same op counts as a correct run | correct output |
| `ggml-backend.cpp`: the "run the op where its weights are" rule looks through views (`src->view_src->buffer`) | hyper-connection scale/base weights are used through views, which have no buffer at split time; the ops fell to the priority passes and ran on the R9700 with weights and activations copied over the link twice per layer per token | sync copies per token 136 → 56, decode 10.8 → 11.7 |
| `glm5next.cpp`: `ssm_a` as `LLM_TENSOR_SSM_A_NOSCAN` (a MUL operand, like every other KDA/GDN model) | the loader's support probe used `GGML_OP_SSM_SCAN`, HIP declined it, the 256-byte tensor landed on the host and every split re-copied it with a device synchronize | 19 sync copies per token |
| launch line: `^output\.weight$` | an unanchored `output\.weight` override also matched every `attn_output.weight` and pinned all 45 output projections to the R9700 | splits per token 80 → 40, decode 11.7 → 13.9, prefill 206 → 296 |
| `LLAMA_ASYNC_INPUTS=1`: the two-lane prefill's stream-ordered host-input copies without the second lane | the sparse-attention pool inputs are host-resident and read by many splits | sync copies 16 → 4 (the inherent boundary), decode 13.9 → 14.2 |
| `ggml-rpc/transport.cpp`: `rdma_poll` spins 5 ms, then polls with 50 µs sleeps (e77a9df94) | an idle `rpc-server` sat at 100% CPU on one core between requests and looked exactly like a hung process (last log line `get_tensor`, GPU idle, spinning); the client burned a core for every remote `graph_compute`. Twice diagnosed as a deadlock before a backtrace of the client showed it idle in the server queue | no idle spin, no measurable decode/prefill change |
| gfx1151 kernels from [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) (ffd4afb5e..a6db5f49e): 16 waves per block for the Gated DeltaNet recurrence, RDNA3.5 MMQ tile configs (128 threads / 64 rows for J ≥ 48), MMQ register prefetch, split-J Q8_0 J=128 tile | the KDA recurrence ran 4 waves per block (21.6 ms per 1024-token chunk per layer on the iGPU; their tiled KDA kernel is H=16/32 only, GLM has 64 heads); the dense q8_0 prefill GEMMs used 256-thread / 128-row tiles, 1-2 blocks per CU with 64 KiB of LDS | KDA 21.6 → 8.3 ms, dense q8_0 GEMMs at 1024 tokens −11-17%, expert GEMMs −4-7% at ≥ 2K tokens and neutral below (their routed-compact MoE path measures at parity with the tile list and was not ported; their MMVQ is faster at batch 1 but slower at the n=3 verify batch, not ported). Prefill 254 → 277 tok/s at 3K, 222 → 236 at 13K, decode unchanged. The 8-wave / 64-row Q8_0 config without the split-J vec-dot and write-back returns wrong output at a fake 5x; `TBO_GLM_EVAL=1 test-backend-ops test` carries the GLM shapes that catch it, `TBO_GLM=1 … perf` the timing cases |
| WMMA flash attention for head size 512 on RDNA (c23469e3e): device-code gate, the tile_Q reuse barrier from halo-box, a retuned RDNA (512,512,32) config (8 waves, occupancy 1, nbatch_fa 64, K/V sub-batches 64/32, Q in shared memory), selected above `GGML_CUDA_FA_MMA512_MIN` columns (default 8) | rocprofv3 on both hosts (13K prompt, ub 1024): the FMA-bound `flash_attn_tile<512,512>` grew linearly with context, 7 ms per call at 1K → 130 ms at 13K on the R9700 (36% of its kernel time over the request); mainframe spent 26% of its prefill and 34% of its decode kernel time in it and sat idle 41% of the prefill waiting for gibson. Upstream's RDNA config (128 threads, occupancy 2) spills the 128-VGPR VKQ accumulator and runs 2x slower than tile; K/V batches above 64/64 overflow the 64 KiB of LDS next to the 32 KiB Q tile | per call at 13K, 1024 tokens: R9700 128 → 70 ms, gfx1151 220 → 181; 3-token verify batch: R9700 790 → 284 µs, gfx1151 871 → 745. Prefill 277 → 283 tok/s at 3K, 236 → 264 at 13K, decode 19.7 / 18.4 → 20.0 / 19.7. KDA at 32 waves: 8.3 → 6.6 ms per 1024-token call |
| WMMA flash attention for head size 256 on RDNA (de1a7e78a): same enablement, RDNA (256,256,64) config 8 waves / occupancy 1 / nbatch_fa 64 / K,V 128,64 / Q in registers, above `GGML_CUDA_FA_MMA256_MIN` columns (default 64, so the draft head's 3-token verify batch keeps the faster tile kernel) | Qwen3.8-Flash-Next's 12 attention layers (24 q heads over 2 kv heads, QSA caps the selected cells near 8K) ran the tile kernel on the R9700: 16 ms per 1024-token call at the cap, ~20% of prefill at depth; upstream's RDNA config is 13% slower than tile there | per call on the R9700: 7.9 → 3.8 ms at 4K cells, 16.0 → 7.5 at 8K. Service launch line, warm: prefill 1283 → 1319 tok/s at 4.8K, 1084 → 1274 at 19.5K, decode 47 unchanged (attention is ~3% of a decode step). `TBO_Q38_FA=1` perf cases |
| MTP draft head from `blk.45`, exported into a draft-only GGUF (`export_mtp.py`; no such file is published), on the R9700 | | decode 14.2 → **20.2** tok/s at 4K, 19.0 at 16K; acceptance 0.85 greedy / 0.82 sampled; `--spec-draft-n-max 2` (3: 18.8, 4: 18.3, 6: 18.4–19.2 on prose; code at temperature 0.7 prefers 4: 23.5 tok/s vs 20.1; `--spec-draft-p-min` 0.5/0.7 with n-max 2/3 all within ±5% of the default; `-ub 2048` does not fit on the R9700 next to the draft) |
| Rolling prefill pipeline across the remote device (b3f940011; `LLAMA_PREFILL_LANES=2`, `-b 16384`): ubatch k+1 is prepared and submitted up to its last remote split while mainframe computes k, then k's tail runs. Three waits on the remote's whole command queue serialized it, each invisible on the wire (mainframe's timestamped rpc-server log showed dead air): the synchronous copies at the boundary (now queue-ordered sends and eager fetches with an RPC event marker, `ggml_backend_sched_set_remote_fetch`), the cut at the *first* remote split (GLM emits a six-node recurrent-state split for layer 25 before layer 24's expert tail; cut at the last), and the synchronize before a graph re-allocation, which the growing sparse-attention shapes trigger every ubatch (`ggml_backend_sched_set_local_sync`). Small graphs bypass the lanes and their eager copies, which is what halved decode with the draft before | per 1024-token ubatch at 13K gibson works ~1.4–1.8 s and mainframe 1.9–2.2 s, in strict alternation; `LLAMA_LANES_DEBUG=1` prints per-ubatch pipeline and preparation phases | prefill 283 / 266 → 328 / 395 tok/s at 3K / 13K, decode unchanged, greedy output identical |
| RPC transport: `rdma_send` posts six 256 KiB chunks before reaping the oldest completion; the server receives `SET_TENSOR` payloads into a persistent, never-zeroed buffer; `set_tensor_async` hashes only weight buffers (b3f940011) | the 64 MB residual streams took 95–110 ms each (650 MB/s): one ~70 ms completion stall per message, the rest at line rate. Cause: the sender filled all 24 pre-posted receive slots in ~2 ms while the server was still allocating and zero-filling a fresh payload vector (8 ms per 64 MB, 200 ms per 1.6 GB weight tensor); the next send got a receiver-not-ready NAK and irdma retried on a ~70 ms timer (gibson's `RetransSegs` grew by ~2,700 per 13K request, mainframe's `ip4InDiscards` by ~0.5 per 64 MB). Pause frames on both NICs did not help and quadrupled idle RTT | residuals 26 ms at 2.56 GB/s, retransmits +11 per run, model load 230 → 208 s; prefill **345 / 430** tok/s |

**Two-lane prefill across the hosts** (2026-09-07). The first attempt died on RDMA with `RDMA CQ wc error: status=12
(transport retry counter exceeded)` and nothing logged on the server. Cause: `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` had no
reply, so the client's dispatcher kept sending while the single-threaded server was inside a graph and not reposting
its 24 x 256 KB receive ring; the second lane's inputs and graph overflowed it and the Intel NIC gave up within
seconds (TCP survived on kernel buffering; a longer queue-pair ACK timeout did nothing). Fix: the server answers
graph compute with an empty reply and the dispatcher waits for it before the next command, without blocking the
caller (wire change: both peers must run this tree). With the lanes stable, the alternating split order gave nothing
(both lanes reach the remote split together and serialize there), so with a remote backend the pair now runs lane a
through its remote submit, then lane b's local splits, then the tails (`GGML_SCHED_PAIR_LOCKSTEP=1` restores):
prefill 288 / 326 → 316 / 347 tok/s at 4K / 16K over TCP, 296 / 275 over RDMA (the 16K figure is unexplained);
upstream's pipeline parallelism forced on gives the same 308 / 345. Both are capped by the host-blocking copy of the
remote result. The lanes are opt-in for this layout: with the draft head the target's verify graphs wait ~2.3x longer
in the copy path (257 vs 113 ms per graph, same splits) and decode falls from 20 to 9 tok/s, not yet understood; the
per-graph synchronize the lanes used to do was removed (one-time after a pair) and a draft-only head keeps one lane.

**The link** (2026-09-13). Both E810s run at PCIe Gen3 x4: mainframe's root port 00:03.2 offers nothing more, gibson's NIC sits on a Gen4-capable x4 port but trained down with correctable errors (a retrain is a live hardware action; not tried). perftest between the hosts (`ib_send_bw`/`ib_write_bw`; the one-shot server dies if the port is probed with a bare connect): one queue pair caps at 2.45 GiB/s for 256 KiB–4 MiB messages at any depth, four queue pairs reach 3.2 GiB/s, the same 27 Gbit/s iperf3 measures over TCP. The design number for transfers is 3.4 GB/s; the 100G nameplate is not available on this bus. Per 1024-token ubatch two 64 MB residual streams go out and 16 MB of hidden states come back, ~30 ms at the current rate, on mainframe's serial timeline between graphs.

**Where the time goes now** (single stream, 4K prompt): ~66 ms per token submit, of which ~60 waits for the remote
graph (20 layers on the 215 GB/s iGPU) behind the local 25 layers; both GPUs are ~50% busy. `GGML_SCHED_TRACE_WAITS=1`
names every synchronous copy (cap 256), which is how each of the fixes above was found; the server-side kernel
profile (`rocprofv3`) could not see any of them, since a blocking host copy consumes no kernel time.

**Numbers** (warm, single stream, `-c 131072`, RDMA; prefill / decode in tok/s at 4K and 16K prompts):

| Run | 4K | 16K |
|---|---|---|
| first correct run (upstream code both ends, TCP) | 206 / 10.3 | 184 / 10.1 |
| + RDMA | 206 / 10.7 | 184 / 10.1 |
| + view-aware weight rule, `ssm_a`, anchored override | 296 / 13.9 | 329 / 13.6 |
| + `LLAMA_ASYNC_INPUTS=1` | 288 / 14.2 | 326 / 13.6 |
| + MTP draft head, n-max 2 | 254 / **20.2** | 222 / **19.0** |
| + halo-box gfx1151 kernels (a6db5f49e) | **277** / 19.7 | **236** / 18.4 |
| + 32-wave KDA, WMMA flash attention for D=512 on both parts (c23469e3e) | **283** / 20.0 | **264** / 19.7 |
| + review fixes: eager-copy allocation, kpool graph reuse (graphs reused 0 → 36-80 per request), RPC major 7 + keepalive (5d6d07f2a) | 282 / 20.4 | 264 / 19.6 |
| + rolling prefill pipeline, `-b 16384`, layer 5's experts on the APU (b3f940011; 3K / 13K prompts from here) | 328 / 20.5 | 395 / 20.2 |
| + RDMA send pipelining and the server's persistent receive buffer (b3f940011) | **345** / 20.5 | **430** / 20.1 |

Context: 128K costs ~1.4 GB of KV per side (11 DSA layers with MLA-compressed KV; the KDA layers keep a fixed
state), so the limit is the dense-attention scratch, not the cache. Mainframe peaks at 92 GB of its 120 GB GTT.

## For upstream (facts to report; not filed)

0. *(historical, resolved before the merge-base)* the un-merged `pr-28118` branch restored prompt checkpoints with `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` although they were saved without it; upstream master and this tree restore with `PARTIAL_ONLY` only. Kept so the fix history is findable; nothing to report.

1. **`qwen4exp`: chunked and recurrent delta-net paths disagree.** Same prompt, greedy, no draft,
   ROCm build, `-c 4096`; first generated token after the prompt's `\n\n`:

   | `-ub` | P(`<think>`) | P(`Unified`) |
   |---|---|---|
   | 512 (one ubatch) | 0.534 | 0.358 |
   | 8 | 0.476 | 0.393 |
   | 2 | 0.467 | 0.443 |
   | 1 (pure recurrence) | 0.378 | 0.496 |

   The prompt: `Explain in about 300 words why unified memory changes the tradeoffs for running large
   MoE models locally, then list three practical tips.` Repeatable run-to-run. Speculative verify
   batches (2–3 tokens) land on the multi-token side, so greedy text differs with and without the draft.
2. **CUDA fusions are not bit-exact.** Same build and prompt: `GGML_CUDA_DISABLE_FUSION=1` changes the
   greedy text (829 vs 950 chars in the test above), and two graphs that are byte-identical unfused
   diverge once fused because node order decides which fusions fire (rms_norm+mul+rope, MoE reduction).
3. **`get_rows` on IQ4_NL declines rows that are not a multiple of 256** in `supports_op`, so a
   device-resident IQ4_NL table falls back to a CPU gather (28.8 GB host copy for the PLE table here);
   the 32-block gather in this branch's `getrows.cu` handles any multiple of 32.
4. **`mul_mat_vec_q` small-K mode is disabled for all RDNA;** enabling it on RDNA4 takes a Q8_0
   [10240 × 320] matvec from 150 to 490 GB/s with large shapes unchanged (`mmvq.cu` here).

5. **ROCm event semantics vs. `ggml-cuda`'s single `copy_event`.** On ROCm 7.2.2 a queued
   `hipStreamWaitEvent` resolves against the event's newest `hipEventRecord` when the queue reaches it (CUDA
   snapshots the record at call time). `ggml_backend_cuda_cpy_tensor_async` re-records one `copy_event` per
   device on every cross-device copy, so every earlier queued wait becomes a wait for a later copy and any
   cross-device overlap (pipeline parallelism included) collapses to a total order. Repro:
   `docs/halo-hybrid/lane_pattern.hip` (599 ms with distinct events, 873 ms with one re-recorded event, 780 ms if
   fully serialized). Fix in this branch: a per-context pool of copy events.
6. **The scheduler's cross-backend copies are issued too late to overlap.** A split input is copied on the
   producer's stream only when the consumer split is submitted, so anything queued on the producer's stream in
   between (another graph's split) delays the copy and the consumer. Graph inputs in host memory are worse:
   each consuming split does a host wait for the destination's previous split plus a synchronous copy
   (`ggml_backend_sched_compute_splits`, the `GGML_TENSOR_FLAG_INPUT` branch), and a *view* of an input (the
   output-row selector slice) takes the sync-copy fallback. With splits alternating between two devices this
   blocks the host mid-graph. This branch adds eager producer-side copies with deferred consumer waits
   (`cpy_tensor_async_nowait`) and stream-ordered input copies, both opt-in.
7. **Server prompt chunking for recurrent models** (`checkpoint_offsets = {4 + n_ubatch, 4}`, PR #20288) means
   the last `n_ubatch` tokens of every prompt are always a separate `llama_decode`, which defeats any scheme that
   overlaps consecutive ubatches; an option to size that tail chunk would help.
8. **Thin f32 matrices at prefill go to cuBLAS.** `ggml_cuda_mul_mat` swaps a `[K → 1]` f32 matrix into
   `mul_mat_vec_f` but a `[K → 2..8]` one (hyper-connection inject matrices, `[10240 → 4]` at 1024 tokens)
   goes to cuBLAS/hipBLASLt; the swapped MMVF plus a transpose is 7× faster on gfx1201 (this branch).
9. **ROCm 7.2.2: rocBLAS routes f32 GEMMs to hipBLASLt on gfx1201**, whose sgemm solutions are 8x8 macro-tile
   fallbacks (`[2560 → 512] × 1024` at 2.5 TFLOPS); `ROCBLAS_USE_HIPBLASLT=0` is 4× faster for that shape but
   slower for f16 GEMMs. A ROCm issue rather than a llama.cpp one; ggml could pick per data type if hipBLASLt
   were called directly.
10. **`ggml-rpc-server` never notices a dead client.** A client crash or host reboot leaves the server spinning in its This tree now sets `SO_KEEPALIVE` (30 s idle, 10 s × 3 probes) on both ends of the control socket.
   RDMA poll loop (or waiting on a TCP socket with no keepalive) with every buffer still allocated (92 GB here) until
   it is restarted by hand (SIGTERM suffices). A TCP keepalive on the control socket is the safe fix: a poll-loop deadline
   alone would also fire during a legitimately long remote graph (long-context prefill graphs run for seconds).
   Related: upstream's `rdma_poll` busy-loops on `ibv_poll_cq` with no backoff, so an *idle* server also pins a core
   at 100% and is indistinguishable from a hung one from outside (this branch spins 5 ms, then sleeps 50 µs per poll).

## Notes

Long-context items from the flashnext-hybrid report: their v3 pair (indexer heads summed by slices, #28023, and
the sequence-position index for the n-gram predecessor lookup, #28040) is already in this base; their fix 4
(the QSA gather) is ported above.

Validation of the merged base (2026-09-02): `test-backend-ops test` passes 14571/14571 on both devices
(ROCm0 = R9700, ROCm1 = APU), which covers the small-K and pipelined matvecs, `MUL_MAT_ID`, the IQ4_NL
`get_rows` and `rms_norm` against the CPU reference; the fused hyper-connection kernels and q8 side copies
are covered by the byte-identical greedy check with fusions disabled. Server under MTP: back-to-back short
requests, a 1k-token prompt, a 2000-token generation to the context limit, and a 6k-token chunked prompt
at `-c 8192` all complete without an assert (`bench/results/chain_solid.sh`).

Benchmarking note: `test-backend-ops perf` replays one weight tensor, so anything under the R9700's 64 MB
infinity cache reports cache bandwidth (1.4 TB/s for a 62 MB Q8_0 matrix). `TBO_Q38_SHAPES=1` adds this
model's decode shapes as 64-matrix batches (working sets of 0.1–3.5 GB) to `tests/test-backend-ops.cpp`;
even then, per-launch latency effects only show up in the real decode (rocprofv3 kernel trace, grid size
identifies the tensor). HIP graphs must stay on (`GGML_CUDA_DISABLE_GRAPHS=1` costs 8–40%); the DPM
`high` perf level caps the R9700 core clock at 2.3 GHz and is 9% slower than `auto`; `profile_peak` is neutral.

* Measurement drift on this box is ~3% within a session and larger across memory-state
  changes (page cache, TTM pool, fragmentation); always A/B by alternating in one session.
  Drain the TTM page pool (`echo 2 > /proc/sys/vm/drop_caches`, twice, 2 s apart) before a
  large host allocation or `earlyoom` kills the process.
* The current z-lab `Qwen3.5-122B-A10B-DFlash` checkpoint (Jun 19 retrain) drafts
  garbage from this target's features in both llama.cpp and the reference implementation;
  the Apr 26 checkpoint works but does not pay back on this platform. The built-in MTP head
  is the production drafter.

- **Other Strix Halo trees (surveyed 2026-09-08).** halo-box/strix-llama.cpp is the one with HIP work for gfx1151 and it
  accepts agent-authored PRs (its AGENTS.md overrides upstream's); halo-box/llama.cpp stages upstream submissions.
  peonist-ai/halogen-flash-server is a closed engine (container only) whose Qwen3.8-Flash-Next numbers on the iGPU alone
  (prefill ~1,200-1,300 tok/s flat to 131K, decode ~42 tok/s with MTP) are the comparison target for the hybrid endpoint.
  myhacsint's `production/strix-halo-qwen4exp-b10685` is a Vulkan snapshot, nothing HIP-side. halo-box's warning about a
  gfx1151 HIP async bug (`HIP_LAUNCH_BLOCKING=1`) did not reproduce here: identical perplexity with and without it.
- **Open: TOPK_MOE fails one random case per full test-backend-ops run** at the 1e-7 tolerance (ERR 1e-5..4e-4, a different
  small-expert-count shape each time: 31, 40, 129, 256 experts; the 288-expert GLM shapes pass). Seen on the R9700 at 36a0f18
  before any of the halo-box ports and on mainframe's gfx1151 at c23469e3e; a TOPK_MOE-only run passes 416/416. It predates
  the kernel ports and is not in any file they touch, but it is expert routing and 4e-4 is not a marginal miss: reproduce
  deterministically (fixed seed, the failing shape, GPU vs CPU probabilities) before touching `topk-moe.cu`.
