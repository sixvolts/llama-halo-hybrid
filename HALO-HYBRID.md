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
For Qwen3.8-Flash-Next the expert tensors are `ffn_(gate_up|down)_exps` and the PLE n-gram
table should stay in host memory: `-ot 'blk\.(...)\.ffn_(gate_up|down)_exps=ROCm1,per_layer_token_embd=CPU'`.

## Changes, on by default

| Where | What | Off switch |
|---|---|---|
| `ggml/src/ggml-backend.cpp` | scheduler events are created regardless of `n_copies`, so split boundaries are GPU-side waits instead of host syncs (+5–7% on this layout) | `GGML_SCHED_NO_EVENTS=1` |
| `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmvq.cu` | grouped `mul_mat_vec_q`: quantise a shared activation once for adjacent matmuls; `qwen35moe.cpp` reorders projections so they are adjacent (bit-exact) | `GGML_CUDA_NO_MMVQ_GROUP=1` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | cross-device copies ≤ 256 KB as a push kernel on the source device into peer-mapped memory (+1.5%) | `GGML_CUDA_KERNEL_COPY_MAX=0` (SDMA), or a byte limit |
| `src/llama-graph.cpp`, `src/models/qwen35moe.cpp` | recurrent-state gathers become views when the copy map is the identity; one `l2_norm` over q and k instead of two (bit-exact, +3.7%) | `LLAMA_NO_GRAPH_FUSE=1` |
| `src/models/dflash.cpp`, `src/llama-hparams.h`, `src/llama-graph.cpp` | DFlash drafts run their sliding-window layers **causally**, as the z-lab reference does (`is_causal = layer_type == "sliding_attention"`); upstream ran every draft layer bidirectionally. Verified token-for-token against the reference PyTorch draft | `LLAMA_DFLASH_SWA_BIDIR=1` |

## Changes, off by default (experiments kept behind env vars)

| Env | What | Outcome |
|---|---|---|
| `LLAMA_EP=<n_a>:<dev>[:first-last]` | expert parallelism: each expert tensor is sliced by index into two device buffers (`create_tensor_expert_slice`), the graph runs gate/up/down on both slices with `-1` sentinel ids (CUDA `mmvq`/`mmq`/`mmid` skip and zero-fill), LUT id remap | correct; slower at every batch size on ROCm (split/launch cost > overlap) |
| `LLAMA_EP_DEV_A=<dev>`, `GGML_CUDA_DEVICES=<n>` | virtual device so slice A gets its own stream | same |
| `GGML_SCHED_LAZY_INPUTS=1` | cut a split at the first consumer of a cross-backend activation; skip host syncs for no-input splits; source stream waits for the destination's progress before a peer copy | net loss on ROCm |
| `GGML_CUDA_GRAPH_OPT_MULTI=1` | lift the multi-device gate on the CUDA graph-optimisation pass | −3% |
| `LLAMA_PIPELINE_PARALLEL_FORCE=1` | keep pipeline-parallel on despite `-ot` | superseded by the events change |
| `LLAMA_DFLASH_DUMP=1`, `LLAMA_DFLASH_DUMP_FILE=<prefix>` | per-tap statistics / raw dump of the target features and drafts (`common/speculative.cpp`) | diagnostics |
| `LLAMA_DFLASH_TAP=norm\|attnres\|postnorm\|ffnout` | expose an alternative tensor as the DFlash "layer input" in `qwen35moe.cpp` | diagnostics; all collapse |

## Notes

* Measurement drift on this box is ~3% within a session and larger across memory-state
  changes (page cache, TTM pool, fragmentation); always A/B by alternating in one session.
  Drain the TTM page pool (`echo 2 > /proc/sys/vm/drop_caches`, twice, 2 s apart) before a
  large host allocation or `earlyoom` kills the process.
* The current z-lab `Qwen3.5-122B-A10B-DFlash` checkpoint (Jun 19 retrain) drafts
  garbage from this target's features in both llama.cpp and the reference implementation;
  the Apr 26 checkpoint works but does not pay back on this platform. The built-in MTP head
  is the production drafter.
