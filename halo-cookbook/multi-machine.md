# Cookbook: two Strix Halos over a 100G link

GLM-5.3-Flash at UD-Q4_K_XL is 200 GB: more than one 128 GB Strix Halo holds. This page splits it across two boxes
with llama.cpp's RPC backend over a direct 100G link (an Intel E810 in each box, no switch). The head node runs
`llama-server`; the second box runs `ggml-rpc-server` and holds the second half of the layers.

Start with [getting started](README.md) for the kernel setup and placement rules.

| Layout | Status | Numbers (single stream, 128K context, MTP head) |
|---|---|---|
| [Two Strix Halos + an R9700 on each](#two-strix-halos--an-r9700-on-each) | production since 2026-09-21 | 896 tok/s prefill / 30 tok/s decode at 25.8K |
| [Two Strix Halos + an R9700 on the head node](#two-strix-halos--an-r9700-on-the-head-node) | superseded | 517 tok/s prefill / 20.5 tok/s decode at 13K |
| [Two Strix Halos, no cards](#two-strix-halos-no-cards) | derived, not measured | - |

## Build and network

Both boxes build the same commit with RPC on:

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON "-DGPU_TARGETS=gfx1151;gfx1201" -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DGGML_RPC=ON
ninja -C build llama-server ggml-rpc-server
```

`-DGGML_RPC_RDMA=ON` (needs `libibverbs-dev`) adds RoCE RDMA, negotiated automatically when both ends have it;
`GGML_RPC_NO_RDMA=1` forces TCP. The production layout runs over TCP (see the rules below).

The link: the two E810s cabled directly, static addresses (10.100.100.1 head node, 10.100.100.2 second box), MTU
9000, rings 8160/8160, adaptive interrupt coalescing off at 10 us.

## Two Strix Halos + an R9700 on each

Measured 2026-09-28: **896 tok/s prefill and 30 tok/s decode at a 25.8K prompt** (92 ms/step, acceptance 0.90), 783
tok/s at 12.7K; single stream, 128K context, MTP head with two drafts.

| | R9700 | iGPU |
|---|---|---|
| head node | dense trunk + KV of layers 0-25, experts of 0-4, MTP draft head, output head | experts of layers 5-25 |
| second box | dense trunk + KV of layers 26-46, experts of 26-31 | experts of layers 32-45 |

The second box runs one rpc-server over both of its GPUs and schedules its half itself (composite mode: the client
sends the graph, the server's own scheduler places it across its card and iGPU):

```
# second box
GGML_RPC_NO_RDMA=1 ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0,ROCm1 -t 16

# head node (layer lists abbreviated: 5|6|...|25 and 32|33|...|45)
GGML_RPC_NO_RDMA=1 GGML_RPC_COMPOSITE=1 LLAMA_PREFILL_LANES=4 LLAMA_UBATCH_TAPER=3 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --rpc 10.100.100.2:50052 \
  -dev ROCm0,ROCm1,RPC0 -ts 26,0,21 --fit off -fa on -ngl 999 -c 131072 -b 32768 -ub 1024 \
  -ot 'blk\.(5|...|25)\.ffn_(gate|up|down)_exps=ROCm1,blk\.(32|...|45)\.ffn_(gate|up|down)_exps=RPC1[10.100.100.2:50052],^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q4_K_XL-q40head.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2
```

The draft head is exported from the model's own shards with `scripts/halo-hybrid/export_mtp.py`
([rdna4.md](rdna4.md#glm-53-flash-at-ud-q2_k_xl) shows the command for the Q2 quant; the same works for Q4_K_XL).

* **Four prefill lanes** run as a rolling pipeline across the link: the remote half works on ubatch k while the head
  node runs k+1. `LLAMA_UBATCH_TAPER=3` makes the first ubatch partial so the pipeline fills sooner.
* **The server's scheduler keeps host syncs for its iGPU:** with scheduler events on the APU, the composite server
  faulted at the first decode step under the grouped MoE GEMV. The client keeps the events.
* **rocBLAS serves one GPU architecture per process:** every dense tensor on each box's R9700, experts only on the
  iGPUs; a dense GEMM on the iGPU in the same process as one on the card fails with "no kernel image".

## Two Strix Halos + an R9700 on the head node

The layout before the second card: layers 0..24 on the head node (dense trunk, KV, the experts of layers 3..4 and the
draft head on the R9700, experts of 5..24 on the iGPU), layers 25..44 plus the MTP block on the second box's iGPU,
output head pulled back to the R9700. The link is crossed twice per token (64 KB out, one hidden state back).

```
LLAMA_PREFILL_LANES=2 LLAMA_ASYNC_INPUTS=1 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --rpc 10.100.100.2:50052 \
  -dev ROCm0,RPC0,ROCm1 -ts 25,22,0 --fit off -fa on -ngl 999 \
  -c 131072 -b 32768 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot 'blk\.([5-9]|1[0-9]|2[0-4])\.ffn_(gate|up|down)_exps=ROCm1,^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8081
```

`-ts 25,22,0`: layer `il` goes to the device whose cumulative split exceeds `il/47` (the denominator is
`n_layer_all + 1`: 45 layers, the MTP block and the output), so 25 puts layers 0..24 local and the output lands
remote, pulled back by the override.

Numbers (single stream, MTP n-max 2, `-ub 1024`, over RDMA): 3K prompt 375 tok/s prefill / 19.6 tok/s decode, 13K
517 / 20.5, 26K 503 / 20.7; acceptance 0.75-0.84; 14 tok/s without the head. The decode step is 127 ms for ~2.7
tokens: 58 ms on the second box (its 20 layers at a 3-token batch, mostly expert weight streaming), ~35 ms on the head
node's two GPUs, the rest the draft chain, sync points and the link. RDMA was worth ~20% prefill and ~30% decode
over TCP on this layout.

## Two Strix Halos, no cards

Derived from the layout above by taking the R9700 out; not run on this tree (the head node here has always had a
card), so treat the line as a starting point. Layers 0..24 on the head node's iGPU, 25..44 on the second box, both
halves keep their experts with their layers, KV and the draft head on the head node's iGPU.

```
# second box
ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0 -t 16

# head node
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
LLAMA_PREFILL_LANES=2 LLAMA_ASYNC_INPUTS=1 \
llama-server -m GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf --rpc 10.100.100.2:50052 \
  -dev ROCm0,RPC0 -ts 25,22 --fit off -fa on -ngl 999 \
  -c 131072 -b 32768 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot '^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8081
```

The head node's iGPU then holds 25 layers of everything plus KV at 128K (about 110 GB plus the draft), which is the
memory budget to check first.

## Running the second box

Run the rpc-server as a systemd unit so it comes back after a reboot and outlives a crashed client. The binary
flushes on SIGTERM but does not exit, hence `KillMode=mixed`; `LimitMEMLOCK=infinity` lets RDMA register its receive
ring.

```
[Unit]
Description=ggml-rpc-server (GLM-5.3-Flash, second half)
After=network-online.target

[Service]
User=<user>
SupplementaryGroups=render video
LimitMEMLOCK=infinity
ExecStartPre=/bin/sh -c 'echo 2 > /proc/sys/vm/drop_caches'
ExecStart=<build>/bin/ggml-rpc-server -H 10.100.100.2 -p 50052 -d ROCm0,ROCm1 -t 16
Restart=on-failure
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

(`-d ROCm0` alone on a box without a card.)

## Rules that cost a day each to learn

* **Both hosts on the same commit.** The RPC wire format is fork-local (the handshake compares the protocol version,
  which includes the op table); a mismatched pair is refused. After a rebuild, `readlink /proc/<pid>/exe` on the
  server must not end in "(deleted)".
* **Loading takes ~3.5 minutes** for the 86 GB remote half, bound by the loader's read-then-send loop, not the link.
* **A peer rebooting looks like a bad link** from the other side (five "drops" in 25 minutes were the second box
  OOM-rebooting under a competing service). Check the peer's uptime before blaming the NIC.
* **E810 firmware and RDMA:** NVM 3.00 threw "HMC Error" PF resets that kill an RDMA session mid-generation; after the
  update to NVM 5.01 the E810s still hit RDMA faults under load, so RDMA has been off on both hosts since 2026-09-23
  and the production numbers above are over TCP.
