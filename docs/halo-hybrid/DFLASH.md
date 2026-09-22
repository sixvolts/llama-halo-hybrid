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

## Adaptive block length (2026-09-21 evening, greedy 256-320 token probes: 4K prose / code / 16K prose)

| config | 4K prose | code | 16K prose |
|---|---|---|---|
| MTP, 2 drafts | 26.8 | 24.4 | 24.9 |
| MTP, 4 drafts | 28.2 | 22.5 | 26.2 |
| DFlash block 4 | 28.1 | 25.7 | 22.3 |
| DFlash block 7, uncut | 26.0-27.3 | 20.5 | 20.7 |
| DFlash block 7, decoded whole, truncated to 4 below 8K ctx | 27.7-29.1 | 25.5 | 20.2 |
| DFlash block 7, selector cut p 0.7 from position 4 | 26.9-28.2 | **31.3-31.5** | 19.6 |
| DFlash block 7, cut p 0.85 from position 3 | 26.7 | 24.5 | 23.2 |
| feedback controller (acceptance EMA x cost line), best variant | 24.7-25.2 | 21-25 | 21-22 |

Findings, each reproduced in an alternating repeat (greedy probes agree to 0.05 t/s):

* `--spec-draft-p-min` applied from block position 1 leaves empty drafts (a full ~83 ms step for one token):
  p 0.85 gave 19.0 t/s against 22.4 uncut on 4K prose while the acceptance ratio rose 0.45 -> 0.74. The cut now
  starts at `LLAMA_DFLASH_PMIN_FROM` (default 3): the first two drafts are kept unconditionally (accepted 0.8-0.97
  of the time on every probe) and confidence only decides the extension.
* The cut from position 4 raises acceptance at EVERY position on the code probe (0.93/0.82/0.69 vs 0.88/0.63/0.41
  uncut), positions the cut never touches. Greedy output is identical, so it is where steps end: an uncut block ends
  its accepted run exactly at a token the drafter got wrong, so the next step starts at a hard spot; a cut block ends
  on the drafter's own terms. Rolling the rejected drafts' injected features out of the draft KV (the other candidate
  explanation) measured identical to the third decimal on/off, so it is not a cache defect. The corollary for any
  length policy: the marginal value of a draft position is lower than its acceptance rate suggests, because a
  rejection also costs the next step.
* A per-step feedback controller (per-position acceptance EMA, cost line T(n) = 48 + 17.6 n, argmax tokens/ms;
  `LLAMA_SPEC_ADAPT=1`, env-gated, default off) lost to every fixed setting in four variants: periodic full-block
  exploration costs ~7% by itself; burn-in exploration freezes the tail estimates on the hardest text (the start of a
  response) and locks the depth at 2; optimistic drift recovers the estimates but the objective is flat between 3
  and 5 drafts on prose, so the controller adds variance without gain, and on code it never learns the long-block
  payoff because that payoff is the rejection-reset effect above, which a per-position model does not see.
  Varying the verify size costs nothing per step (the graph cache is shape-keyed, 64 entries): decode by n is the
  same in adaptive and fixed runs (122-128 ms at n=4, 166-175 at n=8).
* DFlash truncated after a whole-block decode equals a natively smaller block (27.7-29.1 vs 28.1), so any policy can
  set the length per step through `dp.n_max` without retraining concerns.
* DFlash stays weak on the 16K narrative (19.6-23 vs MTP's 26): its 2048-token sliding window cannot see the records
  the answer draws on, and neither the selector confidence nor a feedback controller can fix a drafter that lacks the
  context.
* Drafter format on the R9700: at a shallow cut (p 0.5 from position 1, sampled code) IQ4_XS matched Q8_0, but in the
  config that pays (block 7, cut 0.7 from position 4, greedy code) it does not: Q8_0 31.1-31.5 t/s (acceptance 0.78),
  IQ4_XS 23.3 (0.56), IQ4_XS with the selector, fc and conv projections kept at Q8_0 22.5 (0.53), so it is the 4-bit
  backbone, not the selector, that loses the block's later positions. Q4_1 loses drafts everywhere (no imatrix). The
  drafter's weights cost ~2 ms of a 130+ ms step on this card, so Q8_0 is the build here; a 4-bit drafter only makes
  sense where its bandwidth matters (APU-only) and then with an imatrix and a re-measured cut.

Standing at the regroup: the 27 target is met on prose by MTP at 4 drafts (28.2 / 26.2) and on code by DFlash block 7
with the cut (31.4); no single drafter config reaches both, and the working adaptive mechanism is the selector cut
(default from position 4, `LLAMA_DFLASH_PMIN_FROM`), not the feedback controller. Drafter file: Q8_0. The next real lever for either drafter is the 48 ms fixed cost per step.

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

## Real content (2026-09-22): DFlash is out for chat, the MTP head wins with rejection sampling
On the WebUI-shaped probe (probe_real.py: six everyday prompts, thinking on, T=0.7) the ledger-summary numbers above
did not hold: DFlash block 7 with the cut from position 4 accepted 0.66 / 0.31 / 0.19 per position on technical prose
and ran 15.5-16.6 t/s (the user saw 13), against 19-21 for the MTP head. With lossless rejection sampling for the MTP
drafts, two drafts, and the gfx1151 MoE GEMV fix (V3-SERVER-SCHED.md, 2026-09-22 section) the MTP path runs
24.9-25.9 t/s on the same probe. DFlash remains a code-only option (31 t/s on the greedy code probe); it would need
rejection sampling too (its selector already has per-position candidate scores) before any re-test on real content.
