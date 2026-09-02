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

`-ot` may be given once only (llama.cpp keeps the last); join patterns with commas.
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
| `ggml/src/ggml-backend.cpp` | scheduler events are created regardless of `n_copies`, so split boundaries are GPU-side waits instead of host syncs (+5–7% on this layout) | `GGML_SCHED_NO_EVENTS=1` |
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
| `tools/server/server-context.cpp` | prompt-history checkpoints are saved host-side but PR #28118 restored them with the on-device flag, which asserts (`mem_storage.find(seq_id_read)`) on the first request that reuses a prompt longer than the checkpoint spacing (8K tokens). Restore with the same flags as the save | none |
| `src/models/dflash.cpp`, `src/llama-hparams.h`, `src/llama-graph.cpp` | DFlash drafts run their sliding-window layers **causally**, as the z-lab reference does (`is_causal = layer_type == "sliding_attention"`); upstream ran every draft layer bidirectionally. Verified token-for-token against the reference PyTorch draft | `LLAMA_DFLASH_SWA_BIDIR=1` |

## Changes, off by default (experiments kept behind env vars)

The expert-parallel prototype (`LLAMA_EP`), `LLAMA_PIPELINE_PARALLEL_FORCE` and the DFlash dump/tap diagnostics were dropped when the branch moved to the merged base; they are in the history up to tag `main-pre-mtp`.

| Env | What | Outcome |
|---|---|---|
| `GGML_SCHED_LAZY_INPUTS=1` | cut a split at the first consumer of a cross-backend activation; skip host syncs for no-input splits; source stream waits for the destination's progress before a peer copy | net loss on ROCm |
| `GGML_CUDA_GRAPH_OPT_MULTI=1` | lift the multi-device gate on the CUDA graph-optimisation pass | −3% |

## Qwen3.8-Flash-Next with the MTP draft head (branch `hetero-qwen38-mtp`)

This branch is ggml-org `master` (3466812d1, the merged `qwen4exp`) + unslothai/llama.cpp#144
(NextN/MTP draft head, draft-only exports, borrowing the target's embeddings and head, conv-state
rollback, CUDA graph key by shape) + ggml-org#28118 (the server keeps speculative recurrent-state
checkpoints on-device; without it every round serialises the recurrent state to host memory and
speculation is a net loss on Strix Halo) + the patch set above ported onto it. Upstream now has graph
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
`-ub 4096` shows the R9700 busy 42% of the wall and the APU 42%, summing to 100%, so the devices still run
strictly in turn — why the scheduler does not overlap them is the open question. On the APU the expert GEMM runs
at ~6 TFLOPS at `-ub 1024` and ~13.5 at 4096 (90% / 69% of APU kernel time); `mm_ids_helper` is 15% and the MoE
reduction 8% at 4096, which is the cheap part to fix.

## For upstream (facts to report; not filed)

0. **#28118's prompt-checkpoint restore** uses `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` for checkpoints that were saved without it (`tools/server/server-context.cpp`, the `it->load_tgt/load_dft` pair after `checking checkpoint`); any prompt over the 8K checkpoint spacing then aborts on its second request. Fixed in this branch by restoring host-side.

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
