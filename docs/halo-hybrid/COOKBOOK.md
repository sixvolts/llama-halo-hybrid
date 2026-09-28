# Cookbook: the configurations this fork runs

Five hardware shapes, one tree. Every recipe below is a `llama-server` invocation for a big MoE model with the
routed experts in unified memory and the dense parts where the compute is. The placement rules are the same
everywhere; only the device list changes.

| # | Hardware | Status | Best numbers here | Recipe |
|---|---|---|---|---|
| 1 | One Strix Halo, nothing else | measured (2026-09-28) | Qwen3.8-Flash-Next: 40-41 tok/s with the MTP head (55-56 ms/step, real content, T=0.7); prefill 940 / 870 / 780 tok/s at 4K / 16K / 32K | [1](#1-one-strix-halo-by-itself) |
| 2 | One Strix Halo + one R9700 | production (Qwen3.8-Flash-Next, 2026-09-28) | 63 tok/s decode (35.4 ms/step, real content, T=0.7); prefill 2,090 / 2,220 / 2,030 tok/s at 4K / 16K / 32K | [2](#2-one-strix-halo--one-r9700) |
| 3 | Two Strix Halos over RDMA, no dGPU | derived, not measured | see 4 minus the R9700 | [3](#3-two-strix-halos-over-rdma) |
| 4 | Two Strix Halos + one R9700 on the head node | production (GLM-5.3-Flash, 200 GB) | 517 tok/s prefill / 20.5 tok/s decode at 13K, 503 / 20.7 at 26K | [4](#4-two-strix-halos--one-r9700-on-the-head-node) |
| 5 | Two Strix Halos + one R9700 on each | planned (card ordered) | estimate 23-25 tok/s decode | [5](#5-two-strix-halos--one-r9700-on-each) |

Device names in the recipes are what ROCm enumerates on the box: on the head node `ROCm0` is the R9700 and `ROCm1`
the iGPU (gfx1151); check the order in the startup log before copying a line. `RPC0`/`RPC1` are the remote
devices exposed by `ggml-rpc-server`.

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON "-DGPU_TARGETS=gfx1151;gfx1201" -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_RPC=ON -DGGML_RPC_RDMA=ON -DLLAMA_CURL=ON
ninja -C build
```

`GGML_RPC_RDMA` needs `libibverbs-dev`; drop it (and `GGML_RPC`) on a single box. Drop `gfx1201` from `GPU_TARGETS`
if there is no R9700. ROCm 7.2 is what this tree is built and tested with.

## Placement rules (apply to every recipe)

* **Experts in unified memory, everything else where the compute is.** A tensor override moves only the routed
  experts (`ffn_(gate|up|down)_exps`) of the higher layers to the iGPU; every layer stays *assigned* to the fast
  device (`-ts`), which keeps the fused attention paths on it. Repeated `-ot` flags accumulate, patterns may be
  comma-joined in one flag, and the first pattern that matches a tensor wins.
* **Anchor overrides that name a single tensor** (`^output\.weight$`): an unanchored `output\.weight` also matches
  every `attn_output.weight`.
* **Drain the iGPU's TTM pool before a big load**: `sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'`. Without it
  the kernel's OOM killer takes the loader on the second run.
* **`--fit off -fa on -ngl 999`** on every line: no automatic fitting, flash attention, everything offloaded. For
  Qwen3.8 leave the load mode alone and do NOT override `per_layer_token_embd` to CPU: the 26.8 GiB n-gram table is
  read on demand from the page cache by default (lazy mode, a846a1e01); the old `-ot per_layer_token_embd=CPU` /
  `--load-mode none` lines forced a resident 26.8 GiB copy.
* **The MTP draft head** (`-md <draft.gguf> --spec-type draft-mtp --spec-draft-n-max 2 -devd <device>`) goes on the
  device that holds the dense trunk; it borrows the target's embeddings, so `-devd` must name a device that has them.
  Give it its own q4_K LM head (recipe 6 shows how): drafting then reads 341 MB instead of the target's 680 MB q8_0
  output layer twice per step (-1.1 ms/step hybrid, -3 ms/step APU-only, same acceptance); verification keeps the
  target's output layer. n-max 2 measured best (3 loses 2.5%).
* **`LLAMA_PREFILL_LANES=2`** runs consecutive ubatches on two schedulers so the iGPU's expert GEMMs overlap the
  other device's attention (`-b` at least twice `-ub`). It needs two GPU devices: on the iGPU alone it is ignored.
  With a remote device it becomes the rolling prefill pipeline. It is the whole hybrid prefill win (one lane:
  ~1,450 tok/s at 16K instead of ~2,200).

## 1. One Strix Halo by itself

No dGPU: one device, no placement overrides. The tree's gfx1151 work (RDNA3.5 MMQ tile configs, the 32-wave
Gated-DeltaNet recurrence, WMMA flash attention for head sizes 256 and 512, the sparse DSA attention path, the
dequantize-once q8_0 WMMA GEMM, per-layer launch fusion) all applies to this box; every change in the tree is
measured on the iGPU too.

Qwen3.8-Flash-Next, single stream, 40K context, draft head (own q4_K LM head, recipe 6) on the same device:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0 --fit off -fa on -ngl 999 -c 40960 -b 4096 -ub 4096 -np 1 \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

(`ROCm0` is the iGPU when it is the only ROCm device; on gibson it is `ROCm1`.) `-ub 4096` beats 2048 by 5% / 2.5% /
2% at 4K / 16K / 32K; decode is the same at either. Measured 2026-09-28 at e6bb60de4 (gibson, R9700 idle; prefill =
fresh server, 4K/16K/32K record prompts; decode = six real-content prompts, T=0.7, 400 tokens, two seeds):

| | prefill 4K / 16K / 32K tok/s | decode |
|---|---|---|
| base UD-Q4_K_XL, draft with own q4_K head | 937-946 / 865-871 / 778-782 | 55.2-56.0 ms/step, 39.9-41.4 tok/s |
| same, draft sharing the q8_0 output layer | - | 58.3-58.7 ms/step, 38.3-38.5 tok/s |
| Swift 1.5 Q8T, draft with own q4_K head | 935-940 / 866-867 / 780-781 | - |

gufo (APU-only engine) on the same box: prefill 1,359 / 1,431 / 1,410, MTP decode 34.8 tok/s on its own harness
(temperature 0, not like for like). The prefill gap is hyper-connection traffic and the sparse-attention pipeline;
experts, dense GEMMs and the GDN scan are at parity (see APU-DECODE-BUDGET.md).

Older comparison (2026-09-16, before the 09-24..09-28 rounds):

Same box, same HTTP bench tool (halogen's `halogen-bench.py`, tg128 = mean over its ten prompt shapes, pp = cold
prefill), this tree on the iGPU against halogen-flash-server 0.11.1 on its own 4-bit checkpoint (2026-09-16):

| | this tree, iGPU only | halogen 0.11.1, iGPU only |
|---|---|---|
| decode, serial | 26.2 tok/s | 35.6 |
| decode, MTP head | ~40 (36.8-41.8) | 43.5 (35.8-53.4) |
| prefill 2K / 8K | 714-757 / 684-722 tok/s | 932 / 1,361 |
| prefill, 4.9K record prompt, cold | 700 | 1,110 |

Halogen's engine is gfx1151-only and runs ~500 kernels per token against our ~1,400; its GEMVs read bf16 activations
directly (no quantize pass), its sampler and MTP acceptance run on the GPU, and the n-gram table is on the device.
Its GGUF mode does not accept K-quants, so this is engine-plus-format against engine-plus-format, not the same weights.
Upstream's multi-stream graph optimisation (`GGML_CUDA_GRAPH_OPT=1`) measured no difference here. The launch-count
analysis and the persistent-kernel plan that follows from it: [PERSISTENT-DECODE.md](PERSISTENT-DECODE.md).

The 111 GB model plus the 2.6 GB head leaves room for 8-16K of context on a 128 GB box; Qwen3.5-122B (71 GB) fits
with room to spare; GLM-5.3-Flash (200 GB) does not, which is what recipes 3-5 are for.

## 2. One Strix Halo + one R9700

Dense trunk, KV cache and the draft head on the R9700, routed experts of most layers on the Strix, a few whole
expert layers on the card as VRAM allows. The launch line is in the README; the details:

### Qwen3.8-Flash-Next

Dense trunk, KV cache and the draft head on the R9700, the routed experts of layers 11-47 on the Strix, the n-gram
table read on demand from the page cache. This repo's `main` is upstream master plus the MTP work from
unslothai/llama.cpp#144 and ggml-org#28118, plus the kernel and scheduler changes in
[HALO-HYBRID.md](../../HALO-HYBRID.md) and [APU-DECODE-BUDGET.md](APU-DECODE-BUDGET.md).

Launch (single user, 40K context; ROCm0 is the R9700, ROCm1 the iGPU - check the device order in the startup log):

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

This is `~/bench/q38_hybrid_srv.sh`'s default line. Measured 2026-09-28 at e6bb60de4 (gibson; prefill = fresh
server, 4K/16K/32K record prompts, two servers; decode = six real-content prompts, T=0.7, 400 tokens, two seeds):

| | prefill 4K / 16K / 32K tok/s | decode |
|---|---|---|
| base UD-Q4_K_XL, this line | 2086-2088 / 2211-2227 / 2025-2031 | 35.4-35.5 ms/step, 62.8-62.9 tok/s |
| same with `-b 4096 -ub 2048` and the draft sharing the q8_0 output layer (the 09-27 line) | 2074-2080 / 2097 / 1944-1946 | 36.5-36.6 ms/step, 61.7-62.2 tok/s |
| Swift 1.5 Q8T (recipe 6), this line | 1895-1936 / 2174-2181 / 1980-1981 | ~36.0 ms/step |

* **Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL (four shards, 111 GB). Draft: the MTP head from that repo's
  `MTP/` folder with q4_K experts and its own q4_K LM head (recipe 6's commands, with the base model's
  `output.weight` from shard 2 as the head source). It borrows the target's embeddings, so `-devd ROCm0` is required.
* **`-ub 2560 -b 5120`:** every ubatch of a long prompt is >= 2048 tokens, where the iGPU's F16 expert GEMM takes
  over from MMQ (+6% at 16K, +4% at 32K over `-ub 2048`). `-ub 4096` does not fit on the R9700 with two lanes.
  Moving the expert split (layers 9-12) is a wash.
* **Prefill lanes:** `LLAMA_PREFILL_LANES=2` is the whole win (one lane: 1,420-1,490 tok/s). `-b` must be at least
  twice `-ub`.
* **Decode:** `--spec-draft-n-max 2` (3 loses 2.5%). The head only pays at one or two streams; for more users leave
  the `-md`/`--spec-*` lines out.
* **API:** use `/v1/chat/completions` (a bare prompt on `/completion` stops after one token with this model). The
  model thinks by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off per request.
* **Memory:** ~51 GB of experts on the iGPU, the n-gram table in page cache, 23-26 GB plus the head on the R9700.
  Drain caches before launching after big file activity.

Older multi-stream measurements (2026-09-2x, `-b 4096 -ub 1024`, hybrid-12, shared-Q8_0 draft, model-card sampler;
not re-measured since; 4K prompts, 256-token completions; `LLAMA_PREFILL_LANES=2`; "agg" is the sum over streams, single-stream rows are the per-stream number; the prefill
column's first request of a fresh server is cold, warm numbers are 10–20% higher):

| streams | layout | prefill, agg tok/s | decode, no draft | decode, MTP n-max 2 |
|---|---|---|---|---|
| 1 | hybrid-12 | 1227 (16K prompt: 1264) | 35.4 | **45.6** (greedy ~52) |
| 2 | hybrid-12 | 1157 | 50.7 agg, 26.8 each | 56.1 agg, 30.4 each |
| 4 | hybrid-12 | 1262 | 67.6 agg, 18.0 each | does not fit with the head |
| 4 | hybrid-10 | 585 (single lane) | 47.8 agg, 12.7 each | 51.5 agg, 14.0 each |
| 1 at 68K context | hybrid-12 / hybrid-10 | 413 (`-ub 1024`) / 306 (`-ub 512`) | 25.7 | **43.6** |

### The original run: Qwen3.5-122B-A10B

The layout was worked out on Qwen3.5-122B before 3.8 existed (24 tok/s stock → 49 with the grafted MTP head,
682 tok/s prefill at 32K); the model with the head is at
https://huggingface.co/SixVolts/Qwen3.5-122B-A10B-Opus-Reasoning-MTP-GGUF.

```
llama-server -m Qwen3.5-122B-A10B-Opus-Reasoning-Q4_K_XL.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -ngl 999 -fa on --jinja --load-mode none \
  -ot 'blk\.(1[4-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -c 32768 -ub 4096 -b 4096 \
  -md mtp-draft-out-q4_K.gguf --spec-type draft-mtp -devd ROCm0 \
  --spec-draft-n-max 4 --spec-draft-p-min 0.5
```

## 3. Two Strix Halos over RDMA

Derived from recipe 4 by taking the R9700 out; it has not been run on this tree yet (the head node here has always
had the card), so treat the line as the starting point, not a measurement. Layers 0..24 on the head node's iGPU,
25..44 on the second box, both halves keep their experts with their layers, KV and the draft head on the head
node's iGPU.

Second box (the model half it serves is uploaded by the client; keep the server under systemd so it survives the
client, see recipe 4):

```
ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0 -t 16
```

Head node:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
LLAMA_PREFILL_LANES=2 LLAMA_ASYNC_INPUTS=1 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --rpc 10.100.100.2:50052 \
  -dev ROCm0,RPC0 -ts 25,22 --fit off -fa on -ngl 999 \
  -c 131072 -b 32768 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot '^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8081
```

`-ts 25,22`: layer `il` goes to the device whose cumulative split exceeds `il/47` (the denominator is
`n_layer_all + 1`, 45 layers + the MTP block + the output), so 25 puts layers 0..24 local and the output lands
remote, pulled back by the override. The head node's iGPU then holds 25 layers of everything plus KV at 128K
(about 110 GB + the draft), which is the memory budget to check first.

## 4. Two Strix Halos + one R9700 on the head node

The production layout for GLM-5.3-Flash: layers 0..24 on the head node (dense trunk, KV, the experts of layers
3..4 and the draft head on the R9700, experts of 5..24 on the iGPU), layers 25..44 plus the MTP block on the
second box's iGPU, output head pulled back to the R9700. The link is crossed twice per token (64 KB out, one
hidden state back), so the 100G E810 pair runs RoCE RDMA (negotiated automatically once both builds have
`GGML_RPC_RDMA`; `GGML_RPC_NO_RDMA=1` forces TCP, which costs ~20% prefill and ~30% decode).

The launcher is [`run_glm_two_host.sh`](run_glm_two_host.sh) (`LLAMA_PREFILL_LANES=2 APU_FROM=5 DRAFT=1
run_glm_two_host.sh <tag> 25 131072 -b 32768`); the line it runs is:

```
LLAMA_PREFILL_LANES=2 LLAMA_ASYNC_INPUTS=1 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --rpc 10.100.100.2:50052 \
  -dev ROCm0,RPC0,ROCm1 -ts 25,22,0 --fit off -fa on -ngl 999 \
  -c 131072 -b 32768 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot 'blk\.([5-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps=ROCm1,^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8081
```

Second box, as a systemd unit so it comes back by itself after a reboot and outlives a crashed client
(the binary flushes on SIGTERM but does not exit, hence `KillMode=mixed`; `LimitMEMLOCK=infinity` is what lets
RDMA register the receive ring):

```
[Unit]
Description=ggml-rpc-server (GLM-5.3-Flash layers 25-44)
After=network-online.target

[Service]
User=sixvolts
SupplementaryGroups=render video
LimitMEMLOCK=infinity
ExecStartPre=/bin/sh -c 'echo 2 > /proc/sys/vm/drop_caches'
ExecStart=/home/sixvolts/llama-halo-hybrid/build-rpc/bin/ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0 -t 16
Restart=on-failure
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

Rules that cost a day each to learn:

* **Both hosts on the same commit.** The RPC wire format is fork-local (graph compute replies, protocol major 7);
  a mismatched pair is refused at the handshake. After a rebuild, `readlink /proc/<pid>/exe` on the server must
  not end in "(deleted)".
* **Load is ~3.5 minutes** for the 86 GB remote half, bound by the loader's read-then-send loop, not the link.
* **A peer rebooting looks like a bad link** from the other side (five "drops" in 25 minutes were the second box
  OOM-rebooting under a competing service). Check the peer's uptime before blaming the NIC.
* **E810 firmware**: NVM 3.00 threw "HMC Error" PF resets that kill an RDMA session mid-generation; NVM 5.01 has
  been clean. Rings 8160/8160, adaptive coalescing off, 10 us; MTU 9000.
* **Two-host numbers on the current tree** (single stream, MTP n-max 2, `-ub 1024`): 3K prompt 375 tok/s prefill /
  19.6 tok/s decode, 13K 517 / 20.5, 26K 503 / 20.7; acceptance 0.75-0.84; 14 tok/s without the head. The decode
  step is 127 ms for ~2.7 tokens: 58 ms on the second box (its 20 layers at a 3-token batch, 80% expert weight
  streaming), ~35 ms on the head node's two GPUs, the rest the draft chain, sync points and the link.

## 5. Two Strix Halos + one R9700 on each

Planned; the second card is ordered. The intent is to mirror recipe 4 on the second box: its dense trunk, KV
and attention on its R9700, its experts on its iGPU, both devices behind one `ggml-rpc-server`:

```
ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0,ROCm1 -t 16        # exposes RPC0 (R9700) and RPC1 (iGPU)
```

and on the head node `-dev ROCm0,RPC0,ROCm1,RPC1` with a split that assigns layers 25..44 to RPC0 and an override
that moves their experts to RPC1. Expected: the second box's lane is the critical path at 13K prefill and 80% of
its decode share is expert streaming, so the estimate is 23-25 tok/s decode and a noticeably shorter prefill lane.

Two things are known before the card arrives:

* **rocBLAS serves one GPU architecture per process.** Its lazily loaded Tensile library is cached for the first
  architecture that calls it, and a later GEMM on the other architecture fails with "no kernel image is available
  for execution on the device". The head node only works because its iGPU never calls rocBLAS (experts are MMQ);
  the same must hold inside the second box's rpc-server: every dense and MLA tensor on the R9700, experts only on
  the iGPU. The fork's whole-layer-on-iGPU experiment died on exactly this.
* **The rpc multi-device path is untested here**; it can be dry-run on the head node alone by pointing a local
  `ggml-rpc-server -d ROCm0,ROCm1` at its own two GPUs before the hardware lands.

## 6. Swift 1.5 Qwen3.8-Flash-Next on Strix Halo + R9700 (2026-09-27)

Two builds, both with the fine-tune's own MTP head (the published GGUF has none):

| file | trunk | hybrid prefill 4K/16K/32K | real-content decode | ppl (8 chunks) |
|---|---|---|---|---|
| ukisai Q4_K_M | q4_K/q5_K/q6_K | 1887 / 1907 / 1769 (09-27, `-ub 2048`) | 57-58 t/s (09-27) | 7.52 |
| Q8T (this recipe) | q8_0 (unsloth UD types) | 1895-1936 / 2174-2181 / 1980-1981 (09-28, recipe 2's line) | ~36.0 ms/step, ~62 t/s with the q4_K draft head | 7.29 |

Q8T is the better model (3% lower perplexity) at the same hybrid decode once the draft has its own q4_K head. APU-only
its q8_0 trunk costs decode bandwidth (26.8 vs 31.3 t/s for the Q4_K_M without a draft, 09-27), so APU-only users may
prefer the Q4_K_M; prefill is the same (935-940 / 866-867 / 780-781 with `-ub 4096` and the draft).

```bash
S=docs/halo-hybrid; R=ukisai/Swift1.5-Qwen3.8-Flash-Next
# MTP head (5 GB of range requests), shared embeddings, indexer kept bf16 like unsloth's heads
python3 $S/fetch_hf_tensors.py $R mtp-src
python convert_hf_to_gguf.py mtp-src --mtp --mtp-shared-embd --outtype bf16 --outfile mtp-bf16.gguf
llama-quantize --tensor-type indexer=bf16 --tensor-type ffn_gate_exps=q4_K --tensor-type ffn_up_exps=q4_K \
  --tensor-type ffn_down_exps=q5_1 mtp-bf16.gguf mtp-shared-exps-q4k.gguf Q8_0
# recommended: give the draft its own q4_K LM head (drafting reads 341 MB instead of the 680 MB q8_0 output layer
# twice per step; verification keeps output.weight). Swift Q8T: 40.6 -> 39.6 ms/step at unchanged acceptance.
# (lm_head.weight comes from the bf16 trunk conversion below. For base Qwen3.8 use the target's q8_0 output.weight:
#  add_draft_head.py mtp-Qwen3.8-Flash-Next-shared-exps-q4k.gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
#    output.weight mtp-head-q8.gguf, then the llama-quantize line below with --allow-requantize. Measured 09-28:
#  hybrid 36.5 -> 35.4 ms/step, APU-only 58.5 -> 55.6, acceptance 0.63-0.64 -> 0.62-0.65.)
python3 $S/add_draft_head.py mtp-bf16.gguf trunk-f32.gguf output.weight mtp-head-f32.gguf
llama-quantize --tensor-type indexer=bf16 --tensor-type ffn_gate_exps=q4_K --tensor-type ffn_up_exps=q4_K \
  --tensor-type ffn_down_exps=q5_1 --tensor-type shared_head_head=q4_K mtp-head-f32.gguf mtp-shared-exps-q4k-head-q4_K.gguf Q8_0
# q8_0 trunk (9.4 GB fetch, 20 GB f32 intermediate, 121 GB output)
python3 $S/fetch_hf_tensors.py $R trunk-src . 'mlp\.experts\.|ngram_embedding\.shard_|^model\.visual\.|^mtp\.'
python convert_hf_to_gguf.py trunk-src --no-mtp --outtype f32 --outfile trunk-f32.gguf
python3 $S/merge_trunk.py Swift-...-Q4_K_M-00001-of-00003.gguf trunk-f32.gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  Q8T/Swift-1.5-Qwen3.8-Flash-Next-Q8trunk.gguf --name="Swift 1.5 Qwen3.8-Flash-Next (q8_0 trunk, ukisai Q4_K_M experts)"
# run: q38_hybrid_srv.sh with M=<Q8T shard 1> MD=<mtp-shared-exps-q4k-head-q4_K.gguf> (its defaults: lanes 2, -ub 2560 -b 5120, experts 11-47 on the iGPU)
```
