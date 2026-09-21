# DFlash2 on GLM-5.3-Flash (two-host v3s) — findings and plan

2026-09-21. Drafter: `incoai/GLM-5.3-Flash-DFlash2` (5-layer Qwen3-backbone block-diffusion drafter, block 8, two-tap
dynamic convolutions, candidate selector top-16 rank 256; conditions on the target's layer inputs [6,15,25,34,43] through
`fc` (5x4096 -> 4096) injected into every draft layer's KV; shares the target's token embeddings and LM head; CC BY-NC-ND,
training code not released). GGUF: `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` (standard llama.cpp `dflash` arch, no embeddings;
the fork shares the target's). Files under `/home/sixvolts/models/glm-5.3-flash/DFlash2/` (Q8_0, bf16, incoai safetensors).
Launcher: `SPEC=dflash DRAFT=1 NMAX<=7 ~/bench/glm/run_glm_v3s.sh` (`-devd ROCm0`).

## It runs unmodified

The fork's `draft-dflash` path (upstream #22105/#27342 plus the causal-SWA fix) loads it against the `glm5next` target and
drafts correctly across both hosts: `glm5next.cpp` records `layer_inp` (the hyper-connection mean) for the requested
layers, layers 34 and 43 come back from the mainframe composite device through `extract_layer_inputs` (async GET), and the
drafter runs on the R9700 next to the target's dense layers (1.16 GiB Q8_0 + draft KV; the card had 8 GiB free).

## Decode: the verify cost line and what each drafter buys

Step cost is linear in verified tokens: the MTP chain at NMAX 1/2/4/8 (n = 1 + draft = 2/3/5/9) measures 83/101/136/206 ms
per step, i.e. **~48 ms fixed + 17.6 ms per verified token**, the same on three probes. The slope is MoE physics (each
verified token widens the expert union the APU has to read). DFlash sits on the same line plus 13-26 ms of its own per step
(127-144 ms at n=4, 174 at n=8).

Per-position acceptance (unconditional; 1 + sum = tokens per step), probe_tune set:

| drafter | 4K prose | code (sampled) | 16K prose |
|---|---|---|---|
| MTP chain, 4 drafts | .81 .76 .46 .41 = 3.4 | .86 .71 .60 .46 = 3.6 | .91 .77 .62 .44 = 3.7 |
| DFlash2, 3 drafts | .81 .69 .52 = 3.0 | .91 .84 .72 = 3.5 | .87 .77 .62 = 3.3 |
| DFlash2, 7 drafts | .87 .77 .45 .39 .23 .23 .16 = 4.1 | .97 .78 .75 .56 .50 .50 .47 = 5.5 | .90 .79 .62 .41 .35 .17 .14 = 4.4 |

Decode t/s (4K prose / code / 16K prose): MTP2 26.7 / 26.5 / 26.2; MTP4 25.3 / 27.3 / 27.4; DFlash3 23.8 / 27.8 / 25.7;
DFlash7 23.5 / 32.5 / 25.1. On the probe_ctx ledger-summary probe DFlash3 falls to (.74 .54 .40) = 2.8 tok/step and
19-21 t/s (identical at -b 4096 and -b 32768: task dependence, not a batch bug) while MTP2 keeps (.97 .79) and 25.4-26.2.

Reading: the built-in MTP head is as good as DFlash over the first three positions on prose and better on the ledger
summaries; DFlash wins only in the tail, which code rewards (32.5 vs 27.3). Folding the profiles into the cost line, prose
tops out at ~27-28 t/s for either drafter at 3-4 drafts (the marginal token beyond that is accepted ~40% of the time and
costs 17.6 ms); code keeps paying to the full block. The 27 target is reachable on prose with MTP at NMAX=3-4 today; past
that the lever is the 48 ms fixed cost per step (two-host serial chain, host work, gaps), not the drafter.

## Prefill: no cost at the production batch

With the two-lane pipeline on (`LLAMA_PREFILL_LANES=2`, now the launcher default) and `-b 32768`: MTP 524/576 t/s at
12.7K/25.8K, DFlash3 524/598, DFlash7 521/597. The drafter's prompt injection (whole prompt through the 5 layers, 1.16 s for
12.7K tokens) is hidden. At `-b 4096` it costs 11-15% because the injection runs between target batches (1.6-1.7 s per
4096 tokens). Injecting only the drafter's 2048-token window would remove even that; not needed at the production batch.
(The evening's "310 t/s for everything" was the lanes running serially for want of the env var; the serial run measured
gibson's lane at 1710 ms/ubatch against mainframe's 1506 at LOCAL=27, so the balance knee has moved: LOCAL=26 re-check queued.)

## Drafter weight format on the R9700 (gfx1201)

Full drafter pass (5 layers + fc, test-backend-ops perf, MALL-warm, no grouping/hoist) at the drafter's batch sizes:

| format | n=8 (draft pass) | n=4 (inject) | n=1 |
|---|---|---|---|
| iq4_xs | 1.74 ms | 1.09 | 0.70 |
| q4_1 | 1.90 | 1.16 | 0.66 |
| iq4_nl | 1.94 | 1.22 | 0.75 |
| q4_0 | 2.03 | 1.19 | 0.68 |
| q8_0 | 2.20 | 1.46 | 1.04 |
| q4_K | 2.86 | 1.72 | 0.88 |
| q5_K | 2.95 | 1.79 | 0.97 |
| q6_K | 3.39 | 2.17 | 1.32 |
| f16 | 5.04 | 4.61 | 3.68 |

The simple 4-bit formats are the fastest kernels on this card at these batch sizes and the K-quants are 30-40% slower, but
the whole pass is ~2 ms of a 130 ms step: the format is worth under 0.5 ms per step here. The 13-26 ms of DFlash overhead
per step is not its GEMMs; it is the step path (five layer-input extractions including two synchronous remote GETs, the
injection decode of the accepted tokens, the selector, host-side batch work) and is the thing to instrument. On the APU
(4x less bandwidth) the format would matter 4x more, which is where a re-quant pays.

## Plan

1. **Instrument the DFlash step** (LLAMA_SPEC_TRACE: extract / inject / draft / select ms) and take the overhead down: batch
   the remote layer-input GETs into the decode graph's existing remote fetch, skip the injection pass for rejected
   positions, keep the selector on the backend. Target: DFlash step = verify line + ~5 ms.
2. **Adaptive block length** with the selector confidence (`--spec-draft-p-min`, already implemented for DFlash2): sweep
   0.3/0.5/0.7 at block 7 so code keeps long blocks and prose stops at 3-4. Same sweep for the MTP chain is a policy
   (LLAMA_SPEC_NMAX_SHORT) not a confidence.
3. **Decide per workload**: MTP at NMAX=3-4 for prose (27 today), DFlash for code (32). If the fork's multi-impl
   `--spec-type` list can hold both, try MTP-first with DFlash as the long-block fallback on high-confidence steps.
4. **The fixed 48 ms** is the remaining decode lever for both drafters: re-decompose the step (op timer + mainframe tracer)
   into host, RPC round trips, gaps, dense layers; that budget was 15 + 10 ms of non-kernel time at the last measurement.
5. **Re-quant / re-train** (the user's release goal): with no training code released and an ND license on the only GLM
   drafter, a redistributable drafter means training one (the paper's recipe: ~800K Nemotron-v2 + CodeAlpaca samples,
   target-generated responses, 6 epochs AdamW 6e-4, frozen target, block 8 gamma 4; the drafter arch is in z-lab/dflash).
   That is a multi-day, multi-GPU project; on this hardware a light fine-tune of the drafter against the *quantized*
   target's features is the realistic version and would also close the acceptance gap the Q4_K_XL target induces. Park
   until 1-4 are done and the ceiling is known.
6. Not for this setup: DFlash for Qwen3.8-Flash-Next (no drafter exists; the 27B one is a different model).
