# Cookbook: Strix Halo + an RDNA4 card (Radeon AI Pro R9700, RX 9070 / 9070 XT)

The production setup: one Strix Halo plus a **Radeon AI Pro R9700** (gfx1201, 32 GB GDDR6, ~640 GB/s) on a PCIe 4.0
x4 link. The RX 9070 XT is the same die with 16 GB; its layouts are [below](#radeon-rx-9070-xt-16-gb-simulated).
Start with [getting started](README.md) for the build (`GPU_TARGETS="gfx1151;gfx1201"`), kernel setup and placement
rules.

ROCm0 is the card and ROCm1 the iGPU in every line here; check the device order in the startup log.

## Card setup

* Leave the card's power profile on `auto`. Forcing `high` (`rocm-smi --setperflevel high`) made launch-bound decode
  kernels 20-27% slower.
* The R9700 does not need `amdgpu.ras_enable=0` (the V620 does); keeping it from another card is harmless.

## Qwen3.8-Flash-Next

Dense trunk, KV cache and the draft head on the R9700, the routed experts of layers 11-47 on the iGPU, the n-gram
table read on demand from the page cache. Single user, 40K context:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 40960 -b 5120 -ub 2560 -np 1 \
  -ot 'blk\.(1[1-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

Measured 2026-09-28 (prefill = fresh server, 4K/16K/32K record prompts, two servers; decode = six real-content
prompts, T=0.7, 400 tokens, two seeds):

| | prefill 4K / 16K / 32K tok/s | decode |
|---|---|---|
| UD-Q4_K_XL, this line | 2086-2088 / 2211-2227 / 2025-2031 | 35.4-35.5 ms/step, 62.8-62.9 tok/s |
| same with `-b 4096 -ub 2048` and the draft sharing the q8_0 output layer | 2074-2080 / 2097 / 1944-1946 | 36.5-36.6 ms/step, 61.7-62.2 tok/s |
| Swift 1.5 Q8T ([getting started](README.md#swift-15-qwen38-flash-next-with-a-q8_0-trunk)), this line | 1895-1936 / 2174-2181 / 1980-1981 | ~36.0 ms/step |

* **Draft head:** the MTP head from the model repo's `MTP/` folder with its own q4_K LM head
  ([getting started](README.md#draft-head-with-its-own-lm-head)). It borrows the target's embeddings, so
  `-devd ROCm0` is required.
* **`-ub 2560 -b 5120`:** every ubatch of a long prompt is >= 2048 tokens, where the iGPU's F16 expert GEMM takes
  over from MMQ (+6% at 16K, +4% at 32K over `-ub 2048`). `-ub 4096` does not fit on the R9700 with two lanes.
  Moving the expert split (layers 9-12) is a wash.
* **Prefill lanes:** `LLAMA_PREFILL_LANES=2` is the whole win (one lane: 1,420-1,490 tok/s). `-b` must be at least
  twice `-ub`.
* **Decode:** `--spec-draft-n-max 2` (3 loses 2.5%). The head only pays at one or two streams; for more users leave
  the `-md`/`--spec-*` lines out.
* **Memory:** ~51 GB of experts on the iGPU, the n-gram table in page cache, 23-26 GB plus the head on the R9700.

### 128K context

For long prompts, put every routed expert on the iGPU and raise the ubatch (the line the [DGX Spark
comparison](../README.md#against-a-dgx-spark-2026-09-30) uses):

```
LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 8192 -ub 4096 -np 1 \
  -ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

On that benchmark (40 prompts at T=0, needle-in-filler prefill): 65.5-67.3 tok/s decode median and 2,265-2,322 /
2,190-2,195 / 1,707-1,790 tok/s prefill at 7K / 28K / 113K over two runs (67.4 and 2,354 / 2,232 / 1,815 in the
9070 XT session below). Two other presets from the same comparison:

* **More streams:** experts of layers 5-47 on the iGPU, `-ub 2560 -b 5120`: 70.6 tok/s on one stream, 48.1 / 32.2 /
  21.5 per stream at 2 / 4 / 6 streams.
* **262K context:** experts of layers 5-47 on the iGPU, `-ub 2560`, `LLAMA_QSA_CHUNK_MB=256 LLAMA_SPEC_DRAFT_UB=512`:
  67.8 tok/s, 2,081 / 2,108 / 1,671 tok/s prefill.

### Several streams

Each slot (`-np N`) gets its own share of `-c` and its own KV stream. Two 128K slots, one for the WebUI and one for
an agent, is the line this fork runs in production:

```
LLAMA_PREFILL_LANES=2 \
llama-server -m Swift-1.5-Qwen3.8-Flash-Next-Q8trunk-00001-of-00003.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 262144 -b 8192 -ub 4096 -np 2 \
  -ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Swift1.5-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

What fits on the R9700 (32 GB; Swift 1.5 Q8T with the draft head, every routed expert on the iGPU). The card holds
4.5 GB of weights plus 1.9 GB of draft head, 3.6 GB of KV and indexer cache per 128K of total context, 0.33 GB of
recurrent state per slot, and two prefill compute buffers that grow with the context per slot (6.9 GB each at 128K
and `-ub 4096`, 3.5 GB at `-ub 2048`, ~6.6 GB at 256K for either ubatch):

| slots x context | `-ub` | card |
|---|---|---|
| 1 x 128K | 4096 | 27.1 GB |
| 2 x 128K | 4096 | 31.3 GB |
| 4 x 128K | 2048 | 31.8 GB |
| 1 x 256K | 4096, with `LLAMA_QSA_CHUNK_MB=256 LLAMA_SPEC_DRAFT_UB=512` | 28.6 GB |
| 2 x 256K, 4 x 128K at `-ub 4096`, 3 or 4 x 256K | | does not fit |

Long context on several streams (2026-10-06; no draft, `-ub 2048`, each slot at 59K): 36.4 tok/s on one stream,
26.4 each on two (51.4 agg), 18.0 each on four (62.9 agg). Busy slots should be neighbours, since one ubatch only
joins consecutive slots; the server now hands new requests the idle slot that keeps them so
(`LLAMA_SERVER_SLOT_CONTIG=0` restores the old choice).

Older measurements (2026-09-2x, `-b 4096 -ub 1024`, experts of layers 12-47 on the iGPU, draft sharing the q8_0
output layer; not re-measured since; 4K prompts, 256-token completions; "agg" is the sum over streams):

| streams | prefill, agg tok/s | decode, no draft | decode, MTP n-max 2 |
|---|---|---|---|
| 1 | 1227 (16K prompt: 1264) | 35.4 | **45.6** (greedy ~52) |
| 2 | 1157 | 50.7 agg, 26.8 each | 56.1 agg, 30.4 each |
| 4 | 1262 | 67.6 agg, 18.0 each | does not fit with the head |
| 4, experts of 10-47 on the iGPU | 585 (single lane) | 47.8 agg, 12.7 each | 51.5 agg, 14.0 each |

## GLM-5.3-Flash at UD-Q2_K_XL

GLM-5.3-Flash at Unsloth's UD-Q2_K_XL quant (108.7 GB in four shards) fits on one Strix Halo + R9700 (the Q4 quant
needs two machines, [multi-machine.md](multi-machine.md)).
- **Card:** the dense trunk, attention, KV cache and output head, plus the MTP draft head.
- **iGPU:** the routed experts.

The draft head comes from the same download. Shard 4 carries the MTP block (`blk.45`), so export it into a draft-only
file once (pure Python, about a second):

```
python3 scripts/halo-hybrid/export_mtp.py <dir with the four UD-Q2_K_XL shards> GLM-5.3-Flash-mtp-UD-Q2_K_XL.gguf   # 2.8 GB
```

Launch at 128K context:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 \
llama-server -m GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 4096 -ub 2048 -np 1 -t 16 \
  -ot 'blk\.([3-9]|[1-3][0-9]|4[0-6])\.ffn_(gate|up|down)_exps=ROCm1,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q2_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --jinja --host 0.0.0.0 --port 8080
```

Measured with the DGX Spark comparison's scripts: 40 prompts at T=0, needle-in-filler prefill at matched token counts,
MTP n-max 2, one run each. The first row is from after the 2026-10-01 upstream sync; the 64K rows are from before it.

| Layout | Decode median (prose) | Prefill 7K / 28K / 113K | Draft acceptance | Quality auto-score |
|---|---|---|---|---|
| `-c 131072`, all routed experts on the iGPU, `-ub 2048` (the line above) | 43.8 tok/s (32.8) | 619 / 623 / 498 tok/s | 0.73 | 0.88 |
| `-c 65536`, experts of layers 3-9 on the card, `-ub 1024` | 36.4 tok/s (27.5) | 447 / 496 / - | 0.74 | 0.83 |
| `-c 65536`, experts of layers 3-7 on the card, `-ub 2048` (prefill only) | - | 581 / 596 / - | - | - |

* **Thinking:** GLM's chat template always opens a `<think>` block and has no `enable_thinking` switch. The numbers above
  use `--reasoning-budget 0`, the equivalent of the Spark harness's "thinking off". With thinking on, answers take
  longer but per-token speed is the same.
* **Card memory:** at 128K context the attention buffers grow, so every routed expert has to live on the iGPU. That
  costs 2-3 tok/s of decode against the 64K layout. At 64K, `-ub 2048` with experts of layers 3-9 on the card does not
  fit next to the draft; experts of 3-7 do.
* **Prefill:** `-ub 2048` is the sweet spot, about 30% faster than 1024; 4096 is no better (571 / 593).
* **Draft head:** the Q2 export and the Q4_K_XL export measure the same: 34.1 vs 33.8 tok/s, acceptance 0.738 vs
  0.734.
* **Quality:** the auto-score is the only quality check run on Q2. A KLD comparison against Q4_K_XL has not been done.

## Radeon RX 9070 XT (16 GB, simulated)

The RX 9070 XT is the same gfx1201 die as the R9700, with the same 64 CUs and ~640 GB/s of memory bandwidth, but
16 GB of VRAM instead of 32. It runs the same kernels; the only question is what fits on the card. These numbers are
**simulated** on the R9700: a ballast allocation holds 16 GiB of the card for the whole run, which leaves 15.65 GiB
free. A headless 9070 XT should have ~15.8 GiB, so the simulation is about 150 MB on the tight side. Not yet checked
on a real 9070 XT; the 16 GB RX 7800 XT ([rdna3.md](rdna3.md)) ran the same layouts for real.

What has to fit, Qwen3.8-Flash-Next at 128K context:

| On the card | Size |
|---|---|
| Trunk weights | 4.6 GB |
| MTP draft head | 1.9 GB |
| KV cache (128K, plus the draft's) | 3.7 GB |
| Recurrent state | 0.3 GB |
| Compute buffers | what is left (~5 GB) |

The 32 GB preset spends 14 GB on compute (`-ub 4096`, two prefill lanes), so on 16 GB the ubatch has to come down:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 2048 -ub 1024 -np 1 \
  -ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

Same benchmark as above, one run each. "Spare" is the card's free memory after load; no layout logged an
out-of-memory error during the 113K prefill.

| Layout | Decode median | Prefill 7K / 28K / 113K | Spare |
|---|---|---|---|
| Two lanes, `-ub 1024 -b 2048` (the line above) | 67.7 tok/s | 1,563 / 1,643 / 1,378 | 0.6 GiB |
| Two lanes, `-ub 1024`, `LLAMA_QSA_CHUNK_MB=256 LLAMA_SPEC_DRAFT_UB=512` | 62.0 tok/s | 1,317 / 1,434 / 1,272 | 1.5 GiB |
| One lane, `-ub 4096 -b 8192`, chunk 256, draft ubatch 512 | 64.2 tok/s | 1,281 / 1,307 / 1,163 | 0.2 GiB |
| Two lanes, `-ub 2048 -b 4096`, chunk 256, draft ubatch 512 | 63.6 tok/s | 1,986 / 2,137 / 1,700 | 0.04 GiB |
| For comparison: R9700 (32 GB), 128K line above | 67.4 tok/s | 2,354 / 2,232 / 1,815 | - |

* **Decode is unchanged:** 67.7 tok/s against 67.4 on the 32 GB card. Only prefill pays for the smaller card,
  about 30% at 7K and 25% at 113K.
* **The `-ub 2048` row** keeps nearly all of the R9700's prefill but leaves 40 MB spare. Do not use it on a card that
  also drives a display; on a headless card try it, and fall back to the line above if it fails to load.
* **Driving a display:** the desktop takes a few hundred MB. Use the second row (1.5 GiB spare).
* **`LLAMA_QSA_CHUNK_MB=256`** splits the sparse-attention indexer's score buffer. It is what lets the bigger ubatches
  fit at 128K; one lane at `-ub 2048` without it does not load.
* **GLM-5.3-Flash Q2 on 16 GB** needs one prefill lane: see [rdna3.md](rdna3.md#glm-53-flash-ud-q2_k_xl-128k-context),
  measured on the 16 GB 7800 XT.

## Qwen3.5-122B-A10B (the original run)

The layout was worked out on Qwen3.5-122B before 3.8 existed (24 tok/s stock -> 49 with a grafted MTP head, 682 tok/s
prefill at 32K); the model with the head is at
https://huggingface.co/SixVolts/Qwen3.5-122B-A10B-Opus-Reasoning-MTP-GGUF.

```
llama-server -m Qwen3.5-122B-A10B-Opus-Reasoning-Q4_K_XL.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -ngl 999 -fa on --jinja --load-mode none \
  -ot 'blk\.(1[4-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -c 32768 -ub 4096 -b 4096 \
  -md mtp-draft-out-q4_K.gguf --spec-type draft-mtp -devd ROCm0 \
  --spec-draft-n-max 4 --spec-draft-p-min 0.5
```
