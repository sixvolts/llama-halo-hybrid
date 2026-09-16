# Cookbook: the configurations this fork runs

Five hardware shapes, one tree. Every recipe below is a `llama-server` invocation for a big MoE model with the
routed experts in unified memory and the dense parts where the compute is. The placement rules are the same
everywhere; only the device list changes.

| # | Hardware | Status | Best numbers here | Recipe |
|---|---|---|---|---|
| 1 | One Strix Halo, nothing else | measured | Qwen3.8-Flash-Next: 26 tok/s serial, 34 greedy / 30 sampled with the MTP head; prefill 700 tok/s at 4.9K, 590 at 20K | [1](#1-one-strix-halo-by-itself) |
| 2 | One Strix Halo + one R9700 | production (Qwen3.8-Flash-Next) | 45 tok/s decode (52 greedy), ~1,500 tok/s prefill | [2](#2-one-strix-halo--one-r9700) |
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
* **`--fit off -fa on -ngl 999 --load-mode none`** on every line: no automatic fitting, flash attention, everything
  offloaded, no mmap.
* **The MTP draft head** (`-md <draft.gguf> --spec-type draft-mtp --spec-draft-n-max 2 -devd <device>`) goes on the
  device that holds the dense trunk; it borrows the target's embeddings and lm head, so `-devd` must name a device
  that has them. n-max 2 measured best on both models here; 4 only for sampled code.
* **`LLAMA_PREFILL_LANES=2`** runs consecutive ubatches on two schedulers so the iGPU's expert GEMMs overlap the
  other device's attention (`-b` at least twice `-ub`). With a remote device it becomes the rolling prefill pipeline.

## 1. One Strix Halo by itself

No dGPU: one device, no placement overrides. The tree's gfx1151 work (RDNA3.5 MMQ tile configs, the 32-wave
Gated-DeltaNet recurrence, WMMA flash attention for head sizes 256 and 512, the sparse DSA attention path, the
dequantize-once q8_0 WMMA GEMM, per-layer launch fusion) all applies to this box; every change in the tree is
measured on the iGPU too.

Qwen3.8-Flash-Next, single stream, 8K context, draft head on the same device:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0 --fit off -fa on -ngl 999 -c 8192 -b 4096 -ub 1024 --load-mode none -np 1 \
  -ot 'per_layer_token_embd=CPU' \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

(`ROCm0` is the iGPU when it is the only ROCm device.) Measured on this tree (same prompt and sampler as recipe 2's
table; the R9700 was present but idle):

| | prefill tok/s | decode greedy | decode, model-card sampler |
|---|---|---|---|
| no draft head | (short-prompt harness) | 26.3 | 26.2 |
| MTP head, n-max 2 | (short-prompt harness) | 33.9 | 29.9 |
| MTP head + `LLAMA_PREFILL_LANES=2` | 700 at 4.9K, 590 at 20K (cold and warm alike) | 34.7 | 30.4 |

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

New model, who dis. Same idea as above:
dense trunk, KV cache and the draft head on the R9700, the routed experts of most layers on the Strix, the n-gram
table in host RAM. This repo's `main` is upstream master plus the MTP work from unslothai/llama.cpp#144 and
ggml-org#28118, plus the kernel and scheduler changes in [HALO-HYBRID.md](HALO-HYBRID.md). On this layout stock
llama.cpp decodes at 27–28 tok/s; this branch does ~45 (52 greedy), and prefills at ~1,500 tok/s.

Launch (single user, 8K context; ROCm0 is the R9700, ROCm1 the iGPU — check the device order in the startup log):

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 8192 -b 4096 -ub 1024 --load-mode none -np 1 \
  -ot 'blk\.(1[4-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1,per_layer_token_embd=CPU' \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

* **Model:** `unsloth/Qwen3.8-Flash-Next-GGUF` UD-Q4_K_XL (four shards, 111 GB). Draft head: the `shared-Q8_0` file
  in that repo's `MTP/` folder (2.6 GB); it borrows the target's embeddings and lm head, so `-devd ROCm0` is required.
* **Layout:** `-ot` sends the routed experts of layers N–47 to the iGPU ("hybrid-N") and keeps the n-gram table in
  host RAM. Each layer kept on the R9700 costs it ~1.4 GB, so pick N by what has to fit next to the 2.6 GB head:

  | context | layout | `-ot` pattern |
  |---|---|---|
  | 1 slot, 8K | hybrid-14 | `blk\.(1[4-9]\|[2-4][0-9])` |
  | 1–2 slots, 16K each | hybrid-12 | `blk\.(1[2-9]\|[2-4][0-9])` |
  | 2 slots, 32K each | hybrid-11 | `blk\.(1[1-9]\|[2-4][0-9])` |
  | 64K+ (sparse-attention gather turns on by itself) | hybrid-10 | `blk\.(1[0-9]\|[2-4][0-9])` |

  Without the head, 4 slots at 32K fit at hybrid-12. `-ctk q8_0 -ctv q8_0` halves the KV cost (1.1 GB per 32K slot).
* **Prefill:** `LLAMA_PREFILL_LANES=2` runs consecutive ubatches on two schedulers so the iGPU's expert GEMMs
  overlap the R9700's attention; with the MoE GEMM tile fixes and the R9700's f32 matmul paths (all on by
  default) warm prefill at 3.7K / 15K / 30K is **1503 / 1258 / 1025** tok/s at `-ub 1024` and 1626 / 1324 / 1049
  at `-ub 2048`, from 676 / 616 / 540 on the single-lane branch. The second lane costs a second set of compute
  buffers on the R9700 (0.75 GB at `-ub 1024`, 1.5 GB at 2048), so with the head use `-ub 1024`; `-b` must be at
  least twice `-ub`. Details and the scheduler fixes it needed: HALO-HYBRID.md, "Two-lane prefill".
* **Decode:** `--spec-draft-n-max 2` (3 is the same within noise, 4 is worse). Acceptance is ~0.70 greedy and
  ~0.51 with the model-card sampler (temperature 1.0, top-p 0.95, top-k 20), which is the whole difference between
  52 and 45 tok/s. The head only pays at one or two streams; for more users leave the `-md`/`--spec-*` lines out.
* **API:** use `/v1/chat/completions` (a bare prompt on `/completion` stops after one token with this model). The
  model thinks by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off per request.
* **Memory:**  ~51 GB of experts on the iGPU, the 28.8 GB table plus page cache in
  host RAM, 23–26 GB plus the head on the R9700. Drain caches before launching after big file activity.

Measured on this build (model-card sampler, 4K prompts, 256-token completions; `-b 4096 -ub 1024`,
`LLAMA_PREFILL_LANES=2`; "agg" is the sum over streams, single-stream rows are the per-stream number; the prefill
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
