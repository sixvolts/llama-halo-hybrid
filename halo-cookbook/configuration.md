# Configuration: flags and environment switches

Everything this tree adds on top of upstream llama.cpp's options. Upstream's own variables (`GGML_CUDA_DISABLE_FUSION`,
`GGML_CUDA_DISABLE_GRAPHS`, `LLAMA_ARG_*`, ...) work as upstream documents them. The defaults are what the cookbook
numbers were measured with: a normal run sets nothing from this page except the production settings.

**Values.** A switch (on/off) is off for `0`, `false`, `off`, `no` or an empty value, and on for anything else; unset
means its default. A number must be a whole integer (or a float where noted); anything else is ignored with a
warning and the default is used. Each process reads a switch once, at first use.

**Startup line.** `llama-server` prints every `GGML_*` and `LLAMA_*` variable set in its environment
(`environment: ...`), so a bug report carries them and a typo shows up as an unknown name. `LLAMA_ARG_*` variables
are left out, and names containing KEY, TOKEN, SECRET or PASS show `<redacted>`.

**Where a switch acts.** Library switches act in the process that runs the code: `GGML_CUDA_*` on the host that owns
the GPU, `GGML_SCHED_*` in the process whose scheduler splits the graph (`llama-server`, or an `rpc-server` with
`GGML_RPC_COMPOSITE=1`), `GGML_RPC_*` on the client unless marked "server". With two machines, set a kernel switch on
the host whose device runs the kernel.

## Production settings

| Flag | Environment | Default | What |
|---|---|---|---|
| `--prefill-lanes N` | `LLAMA_PREFILL_LANES` | off | Two-lane prefill: consecutive prompt ubatches on two schedulers so the card and the APU overlap ([HALO-HYBRID.md](../HALO-HYBRID.md#two-lane-prefill-llama_prefill_lanes2)). Needs two GPU devices and `-b` at least twice `-ub`. 1 = off, 2 = two local lanes; with an RPC device 2 runs the rolling pipeline and 4 the paired one. |
| `--qsa-chunk-mb N` | `LLAMA_QSA_CHUNK_MB` | 2048 | Build the QSA indexer's prefill scratch (Qwen3.8 and other `qwen4exp` models) in chunks of at most N MiB per tensor, for ubatches holding one sequence. 256 fits 256K of context on a 32 GB card. 0 = no chunking. |
| `--spec-draft-ubatch-size N` | `LLAMA_SPEC_DRAFT_UB` | 0 | Physical batch size of the draft context (0 = the target's `-ub`); the draft otherwise sizes its compute buffer for the target's ubatch although it runs one layer. Not for block drafts (DFlash, DSpark). |
| `--spec-draft-n-max-sampled N` | `LLAMA_SPEC_NMAX_SAMPLED` | 2 | Sampled (T > 0) requests with an MTP draft draft at most N tokens; greedy requests use `--spec-draft-n-max`. 0 = no cap. |
| `--checkpoint-min-new N` | `LLAMA_ARG_CHECKPOINT_MIN_NEW` | see `--help` | Skip the checkpoint (and the prompt split) at the last user message unless at least N new tokens follow it, so a short chat turn runs as one batch with one checkpoint. Models that use context checkpoints only. |

The flags set the environment variable of the same row, so either form works and the startup line shows it.

| Environment | Default | What |
|---|---|---|
| `LLAMA_ASYNC_INPUTS=1` | off | Stream-ordered host input copies without the second lane (GLM-5.3 over RPC: its sparse-attention pool inputs are read by dozens of splits per token). |
| `GGML_RPC_COMPOSITE=1` | off | A multi-device `rpc-server` schedules its half of the graph itself ([multi-machine.md](multi-machine.md)). Set on the client and the server. |
| `LLAMA_UBATCH_TAPER=N` | 0 | Shape the first and last ubatches of a prompt so a pipeline fills sooner: 1 = half a ubatch first and an equal final pair, 2 = equal final pair only, 3 = partial first ubatch. |
| `LLAMA_MTP_EARLY_MIN=N` | 256 | The MTP draft ingests a prompt per target ubatch, inside the target's prefill, for prompts of at least N tokens. 0 = after the prefill, as upstream. |
| `GGML_CUDA_ALLOW_MIXED_GENERATIONS=1` | off | Run with discrete GPUs of different generations visible (refused by default: untested). |

## Kill switches

Each row turns off one change that is on by default, to compare against upstream's path or to isolate a problem.
The tables of [HALO-HYBRID.md](../HALO-HYBRID.md#changes-on-by-default) say what each change does and what it gained.

### Scheduler and copies

| Switch | What it turns off |
|---|---|
| `GGML_SCHED_NO_EVENTS=1` | GPU-side event waits at split boundaries between local GPUs (back to host synchronizes) |
| `GGML_SCHED_IGPU_EVENTS=0` | the same events for an integrated GPU (the APU); an `rpc-server` sets 0 itself |
| `GGML_SCHED_NO_COPY_BATCH=1` | one event for all eager copies of a consumer split (one record/wait pair per copy instead) |
| `GGML_SCHED_NO_EAGER_PRODUCER_ALLOC=1` | allocating eager copies at their producer's position (back to the consumer's) |
| `GGML_SCHED_NO_INPUT_ORDER=1` | sending a remote split's host inputs before the tensors produced by earlier splits |
| `GGML_SCHED_NO_VIEW_ROUTE=1` | copying a crossing tensor once and reading its views from the copy (one copy per view instead) |
| `GGML_SCHED_OVERLAP_CUT=0` | cutting a split where it consumes the previous split's output, so the head overlaps it (a number: the fewest nodes before the cut, default 4) |
| `GGML_SCHED_PAIR_HAZARD=0` | interleaving lane pairs whose split names differ but have no cache hazards (the name guard alone decides) |
| `GGML_CUDA_NO_PEER_COPY_BATCH=1` | one peer-copy kernel for a consumer's small cross-device inputs |
| `LLAMA_REMOTE_FETCH=0` | fetching remote boundary tensors without draining the remote's queue (two-host pipeline) |
| `LLAMA_LANES_SMALL_EAGER=0` | eager copies for decode-size batches without a remote device (default: on without a remote, off with one) |
| `LLAMA_MTP_INGEST_DEFER=0` | running the MTP draft's ingest behind the next lane pair instead of between pairs |
| `LLAMA_PREFETCH_HINT=0` | telling the memory module which tokens come next, so the next ubatch's inputs start early |
| `GGML_RPC_NO_GRAPH_SLOTS=1` | a second graph slot on the `rpc-server` for small graphs, so a decode step's two graphs both stay cached |

### Matrix-vector products (decode)

| Switch | What it turns off |
|---|---|
| `GGML_CUDA_MMVQ_NO_WIDE=1` | 8-warp blocks for 2-4 columns on RDNA4 and RDNA3 (back to one wave per row) |
| `GGML_CUDA_MMVQ_NO_SMALLK=1` | the small-K rows-per-block GEMV mode on RDNA4 (upstream disables it on RDNA) |
| `GGML_CUDA_MMVQ_NO_TALL=1` | the launch that steps clear of gfx1201's / gfx1101's wave-slot dispatch cliff |
| `GGML_CUDA_MMVQ_GROUPED=0` | reading each distinct expert once per verify batch (back to one pass per token-expert pair) |
| `GGML_CUDA_NO_MMVQ_GROUP=1` | grouping GEMVs that read the same activation (one quantize for the group), and the hoist below |
| `GGML_CUDA_NO_GEMV_GROUP=1` | the one-launch kernel for such a group (back to one launch per matrix) |
| `GGML_CUDA_NO_GEMV_HOIST=1` | moving such GEMVs next to each other in the graph so they group |
| `GGML_CUDA_NO_GEMV_PAIR=1` | gate, up and GLU as one member of such a group |
| `GGML_CUDA_MMVQ_FUSE_N1=1` | the fused gate/up/GLU GEMV at 2-4 columns (upstream fuses at 1 column only) |
| `GGML_CUDA_NO_MMVF_SWAP=1` | thin f32 matrices at prefill widths as a swapped GEMV plus a transpose |
| `GGML_CUDA_MMVF_NO_CLIFF=1` | stepping f32 GEMV blocks clear of the gfx1201 dispatch cliff |
| `GGML_CUDA_NO_MOE_TAIL=1` | the MoE tail at decode in one launch (expert down projections, router weighting, shared expert, residual) |
| `GGML_CUDA_NO_Q8_SIDE=1` | q8_1 copies written by fused producers for the GEMVs after them |
| `GGML_CUDA_NO_MMQ_SIDE=1` | the same for MMQ consumers |
| `GGML_CUDA_NO_TOPK_SMALL=1` | the single-block top-k for short rows on HIP (the DSA indexer's pool selection at decode) |
| `GGML_CUDA_QSA_TK_REG=0` | the register variant of the QSA cell top-k (back to one row per block) |

### Matrix-matrix products (prefill)

| Switch | What it turns off |
|---|---|
| `GGML_CUDA_Q8_WMMA=0` | dense q8_0 GEMMs on WMMA with weights dequantized once per K tile (on for RDNA3 by default; any other value turns it on for RDNA4 too) |
| `GGML_CUDA_KQ_WMMA=0` | q4_K / q5_K / q6_K GEMMs on WMMA (RDNA3 and RDNA4) |
| `GGML_CUDA_MMID_F16=0` | the f16 routed expert GEMM where it measured faster (`=1` uses it for every eligible shape) |
| `GGML_CUDA_MMID_F16_PAIR=0` | gate and up of the f16 expert GEMM as one paired kernel with the GLU in its epilogue |
| `GGML_CUDA_MMQ_GATEUP=0` | sharing gate/up's quantize and expert sort (1) and the fused gate+up+GLU MMQ kernel (2, default) |
| `GGML_CUDA_MMQ_MOE_TILE_LIST=0` | the compact list of non-empty expert tiles (back to launching empty blocks) |
| `GGML_CUDA_MMQ_SUBTILE_SKIP=0` | skipping 16-column subtiles past the last token |
| `GGML_CUDA_MMQ_KTAIL_SKIP=0` | skipping the half K iteration that lies past the end of K |
| `GGML_CUDA_NO_F32_TILE=1` | the tiled f32 GEMM for the MoE router shape (back to rocBLAS) |
| `LLAMA_MOE_F16_CROSS=0` | sending the expert input across devices as f16 where the f16 expert GEMM takes it |

### Attention and the KV cache

| Switch | What it turns off |
|---|---|
| `GGML_CUDA_FA_NO_SPARSE=1` | the sparse flash-attention walk on RDNA for top-k-selected attention (GLM-5.3 DSA, D=512) |
| `GGML_CUDA_FA_SPARSE_D256=0` | the same for Qwen3.8's QSA layers (D=256) at prefill widths |
| `GGML_CUDA_FA_SPARSE_PAIRS=0` | two queries sharing one gathered tile at prefill |
| `GGML_CUDA_FA_NCOLS16=0` | 16 heads of one query per tile for one-query-per-sequence batches (back to 8) |
| `GGML_CUDA_FA_TILE_SPARSE=0` | the sparse walk in the tile kernel on RDNA2 |
| `LLAMA_QSA_SPARSE_FA=0` | passing the per-query bound to flash attention (back to the dense walk under a mask) |
| `LLAMA_QSA_HEAD_SUM=0` | the indexer's relu + head sum + bias as one op |
| `LLAMA_QSA_BLOCK_TOPK=0` | the block-level top-k of the QSA indexer |
| `LLAMA_QSA_SHARED=0` | per-sequence QSA groups in a shared (`-kvu`) pool |
| `LLAMA_QSA_BLK_CACHE=0` | the block-key cache |
| `LLAMA_QSA_INPUT_FAST=0` | the O(n_kv) QSA input fill for the decode layout |
| `GGML_CUDA_NO_QSA_SCORE=1` | the fused decode block scoring kernel |
| `LLAMA_NO_MASK_DEVICE=1` | building attention masks on the device (back to uploading them) |
| `LLAMA_FUSED_LID_DISABLE=1` | the fused lightning-indexer op of the DSA layers (it sums heads in another order, so near-tied top-k picks can differ) |
| `LLAMA_KV_NO_WINDOW=1` | the per-sequence KV window of a unified (`-kvu`) cache |
| `LLAMA_KV_NO_SPANS=1` | per-sequence spans in that window (each sequence's attention covers only its own cells) |
| `LLAMA_KV_NO_MOVE=1` | moving cells to make a sequence's cells contiguous |
| `LLAMA_MTP_NO_WINDOW=1` | the KV window for the qwen4exp MTP head |

### Gated DeltaNet / KDA

| Switch | What it turns off |
|---|---|
| `GGML_CUDA_GDN_BF16=0` | the bf16 WMMA chunked GDN prefill on RDNA3/RDNA4 (back to the fp32 chunked path) |
| `GGML_CUDA_NO_GDN_CHUNKED=1` | the chunked GDN prefill altogether (back to the token loop) |
| `GGML_CUDA_NO_GDN_CONV_FRONT=1` | the qwen4exp GDN conv front as one launch at decode |
| `GGML_CUDA_NO_KDA_CONV_L2=1` | conv + SiLU + Q/K l2 norms as one launch at decode |
| `GGML_CUDA_NO_KDA_CONV_ROWS=1` | the KDA conv-input assembly (concats and state copies) as one launch at decode |
| `GGML_CUDA_NO_KDA_GATE_PROLOGUE=1` | the gate GEMV with its sigmoid prologue as one launch |
| `GGML_CUDA_NO_KDA_STATE_GATHER=1` | reading recurrent states in place instead of a get_rows copy |
| `GGML_CUDA_NO_CONCAT_ROWS=1` | the row kernel for few-column concats (the conv-state concat) |

### Model graphs and fusions

| Switch | What it turns off |
|---|---|
| `LLAMA_NO_GRAPH_FUSE=1` | graph shortcuts in the model builders (the identity recurrent-state gather as a view, one l2_norm over q and k) |
| `LLAMA_NO_HC_MIX=1` | the hyper-connection prologue as one op (DeepSeek-V4 style models) |
| `GGML_CUDA_NO_HC_FUSE=1` | the qwen4exp hyper-connection mix/combine, gate chains and MoE weighted sum fused by pattern |
| `GGML_CUDA_NO_HC_NORM_FUSE=1` | hc_post + rms_norm as one launch |
| `GGML_CUDA_NO_HC_POST_FUSE=1` | only the hc_post part of that fusion |
| `GGML_CUDA_NO_HC_BOUNDARY=1` | the qwen4exp hc boundary (=1 both halves, =2 only the fused combine + norm kernel) |
| `GGML_CUDA_NO_HC_MIX_SPLIT=1` | the multi-block hc mix kernel (back to one block per token) |
| `GGML_CUDA_NO_EWCHAIN=1` | generic element-wise chain fusion |
| `GGML_CUDA_NO_KPOOL_COMPRESS=1` | the GLM-5.3 indexer pool compressor as one launch |
| `GGML_CUDA_NO_MLA_V_PERMUTE=1` | the MLA attention tail's GEMV writing the permuted layout directly |
| `LLAMA_PLE_GET_ROWS=1` | the host-side gather of Qwen3.8's per-layer embeddings (back to a get_rows split) |
| `LLAMA_PLE_PREFETCH=0` | starting that gather early for the next ubatch |
| `LLAMA_PLE_WILLNEED=0` | asking the kernel to read the mmapped table's pages ahead |

### Speculative decoding and the server

| Switch | What it turns off |
|---|---|
| `LLAMA_SPEC_RS=0` | lossless rejection sampling for sampled MTP requests (back to comparing tokens) |
| `LLAMA_MTP_INGEST_FULL=1` | the K/V-only draft ingest (back to full attention for ingest batches) |
| `LLAMA_SERVER_SLOT_CONTIG=0` | choosing idle slots that keep the busy ones contiguous (back to similarity / LRU) |
| `LLAMA_DFLASH_SWA_BIDIR=1` | causal sliding-window layers in the DFlash drafter (back to all-bidirectional) |
| `LLAMA_TAIL_CKPT=2` | the single tail checkpoint under the prefill lanes (restores the second one) |

## Tunables

| Switch | Default | What |
|---|---|---|
| `GGML_SCHED_OVERLAP_CUT_BACKEND` | 0 | the backend whose splits the overlap cut applies to (0 = the card) |
| `GGML_CUDA_MAX_GRAPHS` | 256 | HIP/CUDA graph instances cached per device (at least 1) |
| `GGML_CUDA_KERNEL_COPY_MAX` | 262144 | largest copy (bytes) the batched peer-copy kernel takes |
| `GGML_CUDA_FA_MMA256_MIN` | 64 | columns from which D=256 attention takes the MMA kernel (-1 = never) |
| `GGML_CUDA_FA_MMA512_MIN` | 8 | the same for D=512 |
| `GGML_CUDA_FA_SPARSE_MIN_RATIO` | 3 (RDNA4), 5 (other) | cache / selection ratio from which the D=512 sparse walk is used at prefill |
| `GGML_CUDA_FA_SPARSE_D256_MIN_RATIO` | 4 | the same for D=256 |
| `GGML_CUDA_FA_SPARSE_HEADS` | 32 | heads of one query per sparse tile (16 = the older tiles) |
| `GGML_CUDA_FA_SPARSE_NCOLS1` | 16 | queries per D=256 sparse tile (8 = 32-column tiles) |
| `GGML_CUDA_FA_TILE_SPARSE_COLS` | 32 | columns per RDNA2 sparse tile |
| `GGML_CUDA_FA_TILE_SPARSE_MIN_RATIO` | 3 | ratio from which the RDNA2 sparse walk is used |
| `GGML_CUDA_GDN_CHUNK_MIN` | 64 | fewest tokens for the chunked GDN path |
| `GGML_CUDA_GEMV_GROUP_MINROWS` | 512 | fewest total rows for a GEMV group (smaller groups run per matrix) |
| `GGML_CUDA_GEMV_GROUP_F32ROWS` | 0 | fewest rows of an f32 member of a GEMV group |
| `GGML_CUDA_MMID_F16_MIN` | 64 | fewest tokens for the f16 expert GEMM |
| `GGML_CUDA_MMID_HIST_MIN` | 1 | fewest tokens for the expert histogram kernel (0 = off) |
| `GGML_CUDA_MMID_WIDE_MIN` | 512 | fewest tokens for the wide expert-sort kernel (0 = off) |
| `GGML_CUDA_MMID_MMVQ_MAX_RDNA3` | 0 (built-in table) | cap the expert GEMV path at N tokens for the k-quants on RDNA3 (lowering only) |
| `GGML_CUDA_MMID_MMVQ_MAX_RDNA4` | 0 (built-in table) | the same on RDNA4 |
| `GGML_CUDA_MMQ_GATEUP_MIN_J` | 24 (RDNA3.5), 8 | smallest column tile of the fused gate+up MMQ kernel |
| `GGML_CUDA_MMQ_MOE_J_FACTOR` | 1.0 (float) | size the MoE column tile for this many times the expected tokens per expert (0 = the whole ubatch) |
| `GGML_CUDA_MOE_TAIL_ROWS` | 0 (built-in) | rows per wave of the MoE tail kernel: 1, 2, 4 or 8 |
| `GGML_CUDA_Q8_WMMA_PREPASS_MIN` | 8388608 | M x K above which the q8_0 WMMA GEMM converts the activations to f16 first |
| `LLAMA_KQ_MASK_THREADS` | 8 | host threads that fill a large attention mask |
| `LLAMA_KQ_MASK_MIN_CELLS` | 4194304 | mask size from which those threads are used |
| `LLAMA_LANES_SPLIT_MIN` | 1024 | batches between this and 2 x `-ub` tokens are split into two ubatches so both lanes work (0 = off) |
| `LLAMA_LANES_SPLIT_FRAC` | 0.55 (float, 0.1-0.9) | the first ubatch's share of such a split |
| `LLAMA_PIPELINE_MIN_TOKENS` | 32 | smallest ubatch the two-host rolling pipeline takes |
| `LLAMA_MTP_EH_PROJ_2D_MIN` | 256 | ingest batches of at least this many tokens take one 2D GEMM for the draft's eh projection (0 = never) |
| `LLAMA_PLE_THREADS` | 8 | host threads for the per-layer embedding gather |
| `LLAMA_QSA_GATHER` | 16384 (card), 24576 (APU) | context from which QSA decode gathers the selected cells instead of a dense walk (0 = never) |
| `LLAMA_DFLASH_PMIN_FROM` | 4 | the DFlash `--spec-draft-p-min` cut applies from this block position on |

## Experiments

Measured, kept for comparison, off by default. They may be removed.

| Switch | What |
|---|---|
| `GGML_CUDA_MMVQ_BIG_WIDE=1` | keep the wide GEMV launch for matrices of 2^25 weights and more |
| `GGML_CUDA_MOE_WR_COLS=1` | the column-wise MoE weighted-reduction kernel |
| `GGML_CUDA_MMQ_GATEUP_DECODE=1` | the fused gate+up MMQ kernel at verify widths too |
| `GGML_SCHED_PAIR_LOCKSTEP=1` | with a remote backend, run the two lanes in lockstep instead of one through its remote split first |
| `LLAMA_LAYER_EXPAND=1` | expand each GLM-5.3 layer's output early (two-host split shape) |
| `LLAMA_MTP_EH_PROJ_2D=1` | the 2D eh projection for every draft batch |
| `LLAMA_QSA_ALWAYS_SCORE=1` | compute the QSA block scores even when the window fits the top-k |
| `LLAMA_PREFILL_LANES_RPC=1` | with a remote device, the older pair schedule instead of the rolling pipeline |

## Diagnostics

They log at the library's INFO level, which `llama-server` and `llama-cli` show with `-lv 4`; the op timer
(`GGML_CUDA_TIME_OPS`) and the GEMV planner trace print to stderr directly.

| Switch | What |
|---|---|
| `GGML_CUDA_TIME_OPS=1` | per-op GPU time table; needs `GGML_CUDA_DISABLE_GRAPHS=1` |
| `GGML_CUDA_TIME_OPS_EVERY=N` | graphs between two tables (200) |
| `GGML_CUDA_TIME_OPS_TOP=N` | rows per table (40) |
| `GGML_CUDA_TIME_OPS_NAMES=1` | split the table's rows by weight name |
| `GGML_CUDA_DUMP_GRAPH=N` | print the N-th graph's node list once |
| `GGML_CUDA_GRAPH_STATS=N` | every N graphs, how many were replayed, captured or run directly |
| `GGML_CUDA_FA_UNION_STATS=1` | sparse attention: mean / max union length per call (with `GGML_CUDA_DISABLE_GRAPHS=1`) |
| `GGML_CUDA_GEMV_GROUPS=1` | report GEMVs that share an activation but run separately |
| `GGML_CUDA_MMVQ_DEBUG=1` | the GEMV planner's launch decisions |
| `GGML_SCHED_TRACE_WAITS=1` | per graph, the scheduler's host-blocking points and submit time |
| `GGML_SCHED_TRACE_SPLITS=N` | the split structure of the first N large graphs |
| `GGML_SCHED_TIME=1` | split and allocation time of graphs that take over 1 ms |
| `GGML_SCHED_PAIR_DEBUG=1` | the cache hazards of every lane pair (2 adds a self-test; the test suites use it) |
| `GGML_SCHED_ZERO_BUFFERS=1` | zero the compute buffers after every allocation (2: the remote ones too); a changed result means a read before a write |
| `LLAMA_LANES_DEBUG=1` | per prefill pair: tokens, splits, wall time (2-4 add detail) |
| `LLAMA_KQ_MASK_VERIFY=1` | fill every mask on one thread as well and abort if the threaded fill differs |
| `LLAMA_QSA_LAYOUT_CHECK=1` | verify every cell of the shared-pool QSA layout |
| `GGML_RPC_SEND_TRACE=1` | phase timing of bulk RPC sends (8 MiB and more) |

## Unsafe diagnostic switches

Each restores a behaviour that produced wrong output or a crash, and logs a warning when set. Use them only to
reproduce the original problem.

| Switch | What it re-opens |
|---|---|
| `GGML_SCHED_NO_OUTPUT_EVENTS=1` | the next graph can overwrite output tensors whose reads are still in flight (MTP drafts with several sequences) |
| `GGML_SCHED_NO_EAGER_FENCE=1` | eager copies onto the APU can overwrite memory its pending output reads still use |
| `GGML_RPC_NO_SUPPORTS_OP=1` | assumes the `rpc-server` supports every op, so an op its kernels cannot run aborts the server |
| `GGML_SCHED_IGPU_EVENTS=1` on an `rpc-server` | the APU split events that faulted the server's scheduler (GLM-5.3 two-host, 2026-09-28) |
| `GGML_CUDA_GDN_PERTURB=x` (float) | scales every GDN output by 1 + x to measure the KLD noise floor; output is wrong on purpose |

## Adding a switch

A new switch needs, in the same commit: a row on this page (the `tests/halo` base suite fails on a name the code
reads but this page does not list), a default, and a read through `ggml_env_flag` / `ggml_env_int` /
`ggml_env_float` (`ggml/include/ggml-env.h`) cached in a function-local static. Names:

- `GGML_<BACKEND>_NO_<FEATURE>=1` turns off a default-on feature; `LLAMA_NO_<FEATURE>` / `LLAMA_<FEATURE>=0` in libllama.
- `<PREFIX>_<FEATURE>_<PARAM>=N` tunes one; say the unit (tokens, MiB, columns, a ratio).
- `*_DEBUG`, `*_TRACE`, `*_STATS`, `*_CHECK`, `*_VERIFY` are diagnostics; they only log or abort.
- An experiment that loses its measurement is deleted, not left switched off; one that is kept gets a dated
  "measured, kept" line in its comment and a row under Experiments.
- A per-device default is resolved per call, not cached in the static (the static holds only the environment value).
