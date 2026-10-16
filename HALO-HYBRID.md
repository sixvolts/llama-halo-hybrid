# llama-halo-hybrid: what this fork changes

This tree is upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (last synced at `ec7630a64`,
2026-10-01) plus the changes made to run large mixture-of-experts models on an AMD Strix Halo APU (gfx1151, 128 GB
unified memory), alone, with one Radeon card next to it, or across two machines over RPC. Setup, kernel parameters,
model files and launch lines are in the [cookbook](halo-cookbook/README.md). This page lists what the tree changes,
what is on by default and how to switch each piece off; [configuration.md](halo-cookbook/configuration.md) lists
every flag and environment switch. The measurements behind each change are in its commit message
(`git log -S<switch name>` finds it).

Everything here is developed and tested with the HIP (ROCm) backend on gfx1151 plus gfx1201, gfx1101 or gfx1030.
The CUDA and Vulkan backends compile but are not tested by this fork, and some upstream NVIDIA changes were not
taken at the last sync (see [Known differences from upstream](#known-differences-from-upstream)).

## The hybrid layout

Every layer is *assigned* to the card (`-dev ROCm0,ROCm1 -ts 1,0`), which keeps the fused attention and
Gated-DeltaNet paths there, and a tensor override moves only the routed experts of chosen layers to the APU:

```
-ot 'blk\.(1[1-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1'
```

Repeated `-ot` flags accumulate, patterns may be comma-joined in one flag, and the first pattern that matches a
tensor wins. Anchor patterns that name one tensor (`^output\.weight$`): an unanchored `output\.weight` also matches
every `attn_output.weight`. Qwen3.8's 26.8 GB per-layer n-gram table stays in the page cache and is read on demand;
do not pin it with `-ot per_layer_token_embd=CPU` or `--load-mode none`.

## Changes, on by default

Every switch reads unset as its default, `0`, `false`, `off`, `no` or an empty value as off, and anything else as on
(numeric settings take an integer; a malformed one logs a warning and keeps the default). At startup the log lists
the `GGML_*` and `LLAMA_*` variables that are set, so a bug report shows them. Until 2026-10-15 about 40 of the
`..._NO_...` switches reacted to the variable being set at all, so `NAME=0` also switched the feature off; now it
leaves it on.

### Scheduler and cross-device copies

| What | Off switch |
|---|---|
| Split boundaries between local GPUs are GPU-side event waits instead of host syncs (+5-7% hybrid). RPC and CPU backends keep upstream's host synchronize. | `GGML_SCHED_NO_EVENTS` |
| The APU (an integrated GPU) gets the same events instead of a host stream sync per split (hybrid decode +6%). The rpc-server turns them off for its own scheduler. | `GGML_SCHED_IGPU_EVENTS=0` |
| Cross-device copies are pushed right behind their producer (eager copies) from a pool of copy events, because on ROCm a queued event wait binds to the event's newest record; eager input copies are allocated at their producer split. | `GGML_SCHED_NO_EAGER_PRODUCER_ALLOC` for the allocation |
| Small cross-device copies (up to 256 KB) run as a push kernel into peer-mapped memory; several copies to one split go out as one batched kernel with one event. | `GGML_CUDA_KERNEL_COPY_MAX=0`, `GGML_CUDA_NO_PEER_COPY_BATCH`, `GGML_SCHED_NO_COPY_BATCH` |
| A full-size view of a tensor that crosses devices is copied once and viewed, not copied once per view. | `GGML_SCHED_NO_VIEW_ROUTE` |
| A split with enough independent work (e.g. the shared expert) is cut so it starts before a cross-device input arrives. | `GGML_SCHED_OVERLAP_CUT=0` |
| Output reads re-record the split events after async reads, so the next ubatch's inputs cannot overwrite data still being read. | `GGML_SCHED_NO_OUTPUT_EVENTS` (unsafe, for bisecting only) |
| For a remote split, host-resident inputs are sent before tensors produced by earlier splits. | `GGML_SCHED_NO_INPUT_ORDER` |
| rocBLAS instead of hipBLASLt for f32 GEMMs (hipBLASLt's gfx1201 sgemm kernels are fallbacks). Set by `ggml_cuda_init` unless the variable is already set. | `ROCBLAS_USE_HIPBLASLT=1` |

### Matrix-vector (decode) kernels

| What | Off switch |
|---|---|
| Grouped GEMV: adjacent matmuls that read the same activation quantize it once and run as one launch. | `GGML_CUDA_NO_MMVQ_GROUP`, `GGML_CUDA_NO_GEMV_GROUP=1` |
| Fused kernels that produce a one-row activation also write its q8_1 copy, which the GEMVs pick up (bit-identical). | `GGML_CUDA_NO_Q8_SIDE=1` |
| Decode GEMVs that read the same activation are hoisted next to each other; gate/up/GLU triples join as one paired launch. | `GGML_CUDA_NO_GEMV_HOIST`, `GGML_CUDA_NO_GEMV_PAIR` |
| `mul_mat_vec_q` small-K mode on RDNA4 (short rows such as Qwen3.8's [10240 x 320] hyper-connection projections: 23 -> 7 us). | `GGML_CUDA_MMVQ_NO_SMALLK=1` |
| RDNA3/RDNA4 GEMV launch planner for 2-4 tokens (verify batches): multi-wave blocks, and launches kept clear of gfx1201's dispatch cliff near its 2048 wave slots. The f32 GEMV has the same cliff guard. | `GGML_CUDA_MMVQ_NO_WIDE=1`, `GGML_CUDA_MMVQ_NO_TALL`, `GGML_CUDA_MMVF_NO_CLIFF` |
| Grouped MoE GEMV for verify batches: each distinct expert is read once per batch. | `GGML_CUDA_MMVQ_GROUPED=0` |
| MoE tail at decode (RDNA3/RDNA4): down-projection of the used experts, router weighting, the gated shared expert and the residual add in one launch. | `GGML_CUDA_NO_MOE_TAIL=1` |
| Load pipelining in the f32/f16/bf16 GEMV (four loads in flight per thread, same accumulation order). | none |

### Matrix-matrix (prefill) kernels

| What | Off switch |
|---|---|
| MMQ for MoE sizes its column tile for the expected tokens per expert and runs only the live (expert, tile) pairs. | `GGML_CUDA_MMQ_MOE_J_FACTOR=0`, `GGML_CUDA_MMQ_MOE_TILE_LIST=0` |
| Expert-id compaction (`mm_ids_helper`) as one histogram block (from asf0/strix-llama.cpp); the older per-expert kernel stays as the fallback. | `GGML_CUDA_MMID_HIST_MIN=0`, `GGML_CUDA_MMID_WIDE_MIN=0` |
| Gate and up expert GEMMs share one activation quantize and expert sort; by default a fused gate+up+GLU kernel. | `GGML_CUDA_MMQ_GATEUP=0` (or `1` for the shared quantize only) |
| RDNA3.5 MMQ tile configs and the split-J Q8_0 tile (from halo-box/strix-llama.cpp), register prefetch for Q4_K. | none |
| Dense q8_0 GEMM on RDNA3 WMMA with the weights dequantized once per K tile and no activation pre-pass; RDNA4 keeps MMQ. | `GGML_CUDA_Q8_WMMA=0` |
| k-quant (q4_K/q5_K/q6_K) GEMMs on WMMA for RDNA3/RDNA4. | `GGML_CUDA_KQ_WMMA=0` |
| Routed expert GEMM in F16 WMMA for RDNA3 / RDNA3.5 prefill (adapted from gufo), at the shapes where it measured faster. | `GGML_CUDA_MMID_F16=0` |
| Thin f32 matrices: `[K -> 2..8]` at prefill widths run as a swapped GEMV plus a transpose; a tiled f32 GEMM takes the MoE router shape instead of rocBLAS. | `GGML_CUDA_NO_MMVF_SWAP=1`, `GGML_CUDA_NO_F32_TILE=1` |
| The KDA/Mamba conv-state concat with a transposed operand runs as a tiled kernel; small dim-0 concats at decode as a row kernel. | `GGML_CUDA_NO_CONCAT_ROWS` for the row kernel |

### Attention

| What | Off switch |
|---|---|
| WMMA flash attention for head sizes 256 and 512 on RDNA, above a column threshold (D=256 dense on gfx1151 uses 64-key tiles). | thresholds `GGML_CUDA_FA_MMA256_MIN` (default 64), `GGML_CUDA_FA_MMA512_MIN` (default 8) |
| Sparse flash attention on RDNA: GLM-5.3's DSA layers (D=512) and Qwen3.8's QSA layers (D=256) walk only the selected keys, above a per-architecture context ratio. | `GGML_CUDA_FA_NO_SPARSE=1`, `GGML_CUDA_FA_SPARSE_D256=0`; tuning `GGML_CUDA_FA_SPARSE_MIN_RATIO`, `GGML_CUDA_FA_SPARSE_PAIRS`, `GGML_CUDA_FA_SPARSE_HEADS` |
| QSA (Qwen3.8 sparse attention) at depth: above a cell threshold a decode step gathers the ~2K selected K/V rows (from ucicelos/flashnext-hybrid). | `LLAMA_QSA_GATHER=0`, or the threshold in cells |
| QSA block-key cache and block top-k per KV stream; top-k from registers; fused block scoring at decode. | `LLAMA_QSA_BLK_CACHE=0`, `LLAMA_QSA_BLOCK_TOPK=0`, `LLAMA_QSA_INPUT_FAST=0`, `GGML_CUDA_QSA_TK_REG=0`, `GGML_CUDA_NO_QSA_SCORE=1` |
| KQ masks built on the device from per-cell / per-query indices instead of uploading `[n_kv, n_tokens]`; the host fill of the rest runs on 8 threads for large masks. | `LLAMA_NO_MASK_DEVICE`, `LLAMA_KQ_MASK_THREADS=1` |
| Fused lightning-indexer scoring for GLM-5.3's DSA layers. | `LLAMA_FUSED_LID_DISABLE=1` |
| GLM-5.3 DSA indexer pool compressor in one launch. | `GGML_CUDA_NO_KPOOL_COMPRESS=1` |

### Gated DeltaNet / KDA

| What | Off switch |
|---|---|
| Chunked prefill scan instead of the token loop. | `GGML_CUDA_NO_GDN_CHUNKED`; `GGML_CUDA_GDN_CHUNK_MIN` (default 64 tokens) |
| bf16 WMMA chunked prefill for S_v = 128 on RDNA3/RDNA4 (from stew675/llama-cpp-rdna-boosts and libr4d; near-lossless, not bit-exact). | `GGML_CUDA_GDN_BF16=0` |
| More waves per block for the recurrence on RDNA3.5. | none |

### Model graphs and fusions

| What | Off switch |
|---|---|
| Qwen3.8 (`qwen4exp`) hyper-connection mix and combine, gate chains and the MoE weighted sum fused by op pattern (bit-identical). | `GGML_CUDA_NO_HC_FUSE=1` |
| Generic element-wise chain fusion (mul/add/scale/activation chains with single-use intermediates). | `GGML_CUDA_NO_EWCHAIN=1` |
| DeepSeek-V4 hyper-connection prologue as one op (`DSV4_HC_MIX`). | `LLAMA_NO_HC_MIX` |
| Graph tidy-ups: identity recurrent-state gathers become views, one `l2_norm` over q and k. | `LLAMA_NO_GRAPH_FUSE` |
| Qwen3.8's per-layer n-gram rows gathered on the host in `set_input` (no CPU split), started early for the next ubatch. | `LLAMA_PLE_GET_ROWS`, `LLAMA_PLE_PREFETCH=0`, `LLAMA_PREFETCH_HINT=0` |
| DFlash drafts run their sliding-window layers causally, as the reference does. | `LLAMA_DFLASH_SWA_BIDIR` |

### KV cache and server

| What | Off switch |
|---|---|
| Shared KV pool (`-kvu`): each ubatch attends only its own sequences' window, per-sequence QSA groups, per-sequence dense attention spans, and cell moves when one conversation grows into another's cells. | `LLAMA_KV_NO_WINDOW`, `LLAMA_QSA_SHARED=0`, `LLAMA_KV_NO_SPANS`, `LLAMA_MTP_NO_WINDOW`, `LLAMA_KV_NO_MOVE=1` |
| Idle-slot purge with a unified cache saves the least recently used idle slot to the RAM prompt cache. Run a shared pool with `--no-cache-idle-slots`. | none |
| New tasks prefer idle slots that keep busy slots contiguous (one ubatch only joins consecutive KV streams). | `LLAMA_SERVER_SLOT_CONTIG=0` |
| `--checkpoint-min-new N`: no checkpoint or prompt split at the last user message when fewer than N new tokens follow it. | default 0 (always split) |

### Speculative decoding (MTP draft heads)

| What | Off switch |
|---|---|
| Sampled requests (T > 0) verify MTP drafts with lossless rejection sampling, so sampled output follows the target's distribution. Greedy requests compare tokens. | `LLAMA_SPEC_RS=0` |
| Sampled MTP requests draft at most 2 tokens; greedy ones use `--spec-draft-n-max`. | `--spec-draft-n-max-sampled 0` (no cap) |
| The draft ingests a long prompt per target ubatch, inside the target's prefill, instead of once after it. | `LLAMA_MTP_EARLY_MIN=0`, `LLAMA_MTP_INGEST_DEFER=0` |

## Opt-in settings

| Setting | What |
|---|---|
| `--prefill-lanes 2` (`LLAMA_PREFILL_LANES=2`) | Two-lane prefill (below). Needs two GPU devices and `-b` at least twice `-ub`. Unset or 1 is off; 2 or more gives two local lanes; with a remote RPC device 2 runs the rolling pipeline and 4 the paired one. |
| `--qsa-chunk-mb N` (`LLAMA_QSA_CHUNK_MB=N`) | Build the QSA indexer's prefill scratch in chunks of N MiB (default 2048; ubatches of one sequence). Needed for 256K context on a 32 GB card. |
| `--spec-draft-ubatch-size N` (`LLAMA_SPEC_DRAFT_UB=N`) | Cap the draft's ubatch (0 = the target's `-ub`). |
| `--spec-draft-n-max-sampled N` (`LLAMA_SPEC_NMAX_SAMPLED=N`) | Sampled MTP requests draft at most N tokens (default 2, 0 = no cap). |
| `LLAMA_ASYNC_INPUTS=1` | Stream-ordered host input copies without the second lane (helps GLM-5.3 over RPC). |
| `GGML_RPC_COMPOSITE=1` | Let a multi-device rpc-server schedule its half of the graph itself ([multi-machine.md](halo-cookbook/multi-machine.md)). |
| `LLAMA_UBATCH_TAPER=N` | Make the first ubatch of a prompt partial so a pipeline fills sooner. |

## Two-lane prefill (`LLAMA_PREFILL_LANES=2`)

With the experts on the APU, every ubatch's graph alternates card / APU / card ... dozens of times, and the
scheduler submits one graph at a time onto one in-order stream per device, so the two devices ran strictly in turn
(upstream's pipeline parallelism cannot overlap this interleaved layout). Two lanes fix that:

1. Consecutive prefill ubatches are built on two schedulers (own compute buffers, shared backends and streams) and
   their splits are submitted alternately, so lane b's attention sits on the card right behind lane a's, ahead of
   lane a's wait for the APU. A pair whose splits could touch the same cache rows is serialised.
2. Host inputs are copied on the destination stream instead of with a host wait plus a synchronous copy.
3. Cross-device copies come from a pool of events (on ROCm a queued event wait binds to the newest record, so one
   re-recorded event per device serialised everything) and are pushed right behind their producer.
4. The server ends the prompt so that its tail is still a pair, not one ubatch alone.

Output is byte-identical to one lane. The cost is a second set of compute buffers; decode is unchanged. On the
Strix Halo + R9700 it is the whole hybrid prefill win (about 1,450 -> 2,200 tok/s at 16K for Qwen3.8).

## MTP draft heads: things to know

* **Greedy output with the draft is not identical to greedy output without it,** and that is not a speculative
  decoding bug: the verify batch takes the multi-token (chunked) Gated-DeltaNet path, whose probabilities differ
  from the single-token recurrence by ~0.1 on some tokens. Upstream llama.cpp behaves the same way.
* The 3-token verify forward is about 90% of an MTP step. Each extra draft token pulls in more experts, so the
  APU's expert traffic, not the dense matvecs, sets the step cost.
* Upstream's fusions (rms_norm+mul+rope, the MoE reduction) are not bit-exact, and this tree's graph edits change
  which of them fire, so greedy text differs from upstream's at the same seed; with all fusions off
  (`GGML_CUDA_DISABLE_FUSION=1` and this tree's switches) the graphs match.

## Two machines

Both hosts must run `llama-server` / `ggml-rpc-server` built from the same commit of this tree: the RPC protocol is
fork-local (major 107), and the handshake refuses a peer with a different protocol version or op table, upstream
builds included. Layouts and launch lines:
[multi-machine.md](halo-cookbook/multi-machine.md).

## Diagnostics

The full list, with the unsafe switches that re-open fixed bugs, is in
[configuration.md](halo-cookbook/configuration.md#diagnostics).

| Switch | What |
|---|---|
| `GGML_CUDA_TIME_OPS=1` (with `GGML_CUDA_DISABLE_GRAPHS=1`) | per-op GPU time table; `_EVERY`, `_TOP`, `_NAMES` tune it |
| `GGML_SCHED_TRACE_WAITS=1` | per graph, the scheduler's host-blocking points and the submit time |
| `LLAMA_LANES_DEBUG=1` | per prefill pair: tokens, splits, wall time (2 and 3 add more detail) |
| `GGML_SCHED_PAIR_DEBUG=1` | the cache hazards found for every lane pair |

Benchmarking: `test-backend-ops perf` replays one weight tensor, so on the R9700 anything under its 64 MB infinity
cache reports cache bandwidth; per-launch effects only show in a real run. Keep HIP graphs on and the card's power
profile on `auto`. Measurements drift by a few percent within a session, so compare by alternating runs (ABBA).

## Known differences from upstream

* The Blackwell FP4 activation-precision path (#24364, W4A4/W4A8) was not taken at the last sync:
  `GGML_CUDA_MMQ_PREC` in [docs/build.md](docs/build.md) has no effect in this tree.
* The NVIDIA flash-attention changes #28536, #29152, #28770 and #29298 were not taken; this tree keeps its own
  `fattn-mma-f16.cuh` (and the `fattn-swizzle.cuh` that upstream removed).
* Upstream's Qwen4Exp k-pool / MTP rewrite (#29751, #29761) and its `glm5-next` architecture are carried beside this
  tree's own `qwen4exp` and `glm5next`; Unsloth's GLM-5.3 GGUFs load as `glm5next`.
* Upstream's probabilistic draft sampling (#27694) landed after the base; this tree has its own rejection sampling
  (above).
