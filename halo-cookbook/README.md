# Halo cookbook: getting started

This fork runs big mixture-of-experts models on an AMD Strix Halo (Ryzen AI Max+ 395, 128 GB unified memory), alone
or with a Radeon card next to it. The routed experts stay in unified memory on the iGPU; the dense parts (trunk,
attention, KV cache, the MTP draft head) go where the compute is. The placement rules are the same everywhere; only
the device list changes.

This page covers what every setup shares (build, kernel setup, placement rules, model preparation) and the Strix
Halo on its own. Each other page is one hardware shape:

| Page | Hardware | Status | Best numbers (Qwen3.8-Flash-Next unless noted) |
|---|---|---|---|
| [This page](#one-strix-halo-by-itself) | One Strix Halo, nothing else | measured (2026-09-28) | 40-41 tok/s decode; prefill 940 / 870 / 780 tok/s at 4K / 16K / 32K |
| [rdna4.md](rdna4.md) | + Radeon AI Pro R9700 (32 GB) or RX 9070 XT (16 GB) | production | 63 tok/s decode, 2,090 / 2,220 / 2,030 tok/s prefill at 4K / 16K / 32K; GLM-5.3-Flash Q2 44 tok/s |
| [rdna3.md](rdna3.md) | + RX 7800 XT (16 GB); 7900 XT / XTX untested | measured (2026-10-04) | 60 tok/s decode, 1,538 / 1,327 / 989 tok/s prefill at 7K / 28K / 113K |
| [rdna2.md](rdna2.md) | + Radeon Pro V620 (32 GB); RX 6800 / 6900 XT | measured (2026-10-02) | 58 tok/s decode, 1,517 / 1,324 / 1,023 tok/s prefill at 7K / 28K / 113K |
| [usb4-thunderbolt.md](usb4-thunderbolt.md) | Any of the cards above in a USB4 / Thunderbolt eGPU dock | measured with the RX 7800 XT (2026-10-04) | same decode as a PCIe x4 slot; 3.8 GB/s card <-> iGPU |
| [multi-machine.md](multi-machine.md) | Two Strix Halos over a 100G link, with or without cards | production (GLM-5.3-Flash, 200 GB) | 896 tok/s prefill / 30 tok/s decode at 25.8K |

All numbers come from one test box (gibson: Framework Strix Halo board, 128 GB, ROCm 7.2, Linux 7.0) unless a page
says otherwise. The kernel and scheduler changes behind them are listed in [HALO-HYBRID.md](../HALO-HYBRID.md).

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON "-DGPU_TARGETS=gfx1151;gfx1201" -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DGGML_HIP_NO_VMM=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_MMQ_MFMA=ON -DGGML_HIP_ROCWMMA_FATTN=OFF
ninja -C build llama-server
```

`GPU_TARGETS` is `gfx1151` (the Strix Halo iGPU) plus the card's architecture:

| Card | Target |
|---|---|
| none | `gfx1151` alone |
| R9700, RX 9070 / 9070 XT (RDNA4) | `gfx1201` |
| RX 7900 XT / XTX (RDNA3) | `gfx1100` |
| RX 7800 XT / 7700 XT (RDNA3) | `gfx1101` |
| Radeon Pro V620, W6800, RX 6800 / 6900 XT (RDNA2) | `gfx1030` |

For two machines add `-DGGML_RPC=ON` (see [multi-machine.md](multi-machine.md)). ROCm 7.2 is what this tree is built
and tested with.

## Kernel setup

The test box boots with:

```
GRUB_CMDLINE_LINUX_DEFAULT="... amd_iommu=off ttm.pages_limit=28311552 ttm.page_pool_size=28311552 amdgpu.lockup_timeout=10000,20000,10000,10000"
GRUB_CMDLINE_LINUX="amdgpu.ras_enable=0 amdgpu.runpm=0"
```

* **`ttm.pages_limit` / `ttm.page_pool_size`** (28,311,552 pages of 4 KiB = 108 GiB) let the iGPU map most of the
  128 GB. The 111 GB Qwen3.8 models need this.
* **`amdgpu.lockup_timeout`** gives long kernels (prefill on a 100K+ prompt) more time before the driver calls a hang.
* **`amdgpu.runpm=0`** with a card: an RX 7800 XT failed to wake from runtime power-down after ~9 minutes idle, and
  the stuck card took ROCm down for the iGPU as well. The cost is that the card never powers down at idle.
* **`amdgpu.ras_enable=0`** is needed for the V620 ([rdna2.md](rdna2.md)); it does no harm on the other cards.
* An eGPU dock also needs `thunderbolt.host_reset=false` ([usb4-thunderbolt.md](usb4-thunderbolt.md)).

Run `sudo update-grub` and reboot after changing these.

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
  read on demand from the page cache by default; `-ot per_layer_token_embd=CPU` or `--load-mode none` force a resident
  26.8 GiB copy.
* **The MTP draft head** (`-md <draft.gguf> --spec-type draft-mtp --spec-draft-n-max 2 -devd <device>`) goes on the
  device that holds the dense trunk; it borrows the target's embeddings, so `-devd` must name a device that has them.
  Give it its own q4_K LM head ([below](#draft-head-with-its-own-lm-head)): drafting then reads 341 MB instead of
  the target's 680 MB q8_0 output layer twice per step. n-max 2 measured best (3 loses 2.5%).
* **`LLAMA_PREFILL_LANES=2`** runs consecutive ubatches on two schedulers so the iGPU's expert GEMMs overlap the
  card's attention (`-b` at least twice `-ub`). It needs two GPU devices; on the iGPU alone it is ignored. It is the
  whole hybrid prefill win (one lane: ~1,450 tok/s at 16K instead of ~2,200 on the R9700).
* **Device names** are what ROCm enumerates: with a card, `ROCm0` is usually the card and `ROCm1` the iGPU. Check the
  startup log before copying a line.

## Models and draft heads

* **Qwen3.8-Flash-Next:** `unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q4_K_XL (four shards, 111 GB). The MTP draft head is
  in that repo's `MTP/` folder.
* **GLM-5.3-Flash:** unsloth UD-Q4_K_XL (200 GB, two machines) or UD-Q2_K_XL (108.7 GB, one machine). The draft head
  is exported from the model's own shards ([rdna4.md](rdna4.md#glm-53-flash-at-ud-q2_k_xl)).
* **API:** use `/v1/chat/completions` (a bare prompt on `/completion` stops after one token with Qwen3.8). Qwen3.8
  thinks by default; `"chat_template_kwargs": {"enable_thinking": false}` turns it off per request.

### Draft head with its own LM head

The published Qwen3.8 MTP head drafts through the target's q8_0 output layer. A q4_K copy of that layer inside the
head file makes each draft step cheaper (hybrid 36.5 -> 35.4 ms/step, APU-only 58.5 -> 55.6, same acceptance);
verification still uses the target's layer. The tools are in [`scripts/halo-hybrid/`](../scripts/halo-hybrid):

```
python3 scripts/halo-hybrid/add_draft_head.py mtp-Qwen3.8-Flash-Next-shared-exps-q4k.gguf \
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf output.weight mtp-head-q8.gguf
llama-quantize --allow-requantize --tensor-type indexer=bf16 --tensor-type ffn_gate_exps=q4_K \
  --tensor-type ffn_up_exps=q4_K --tensor-type ffn_down_exps=q5_1 --tensor-type shared_head_head=q4_K \
  mtp-head-q8.gguf mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf Q8_0
```

### Swift 1.5 Qwen3.8-Flash-Next with a q8_0 trunk

A fine-tune of Qwen3.8 whose published GGUF has no MTP head. This builds the head and a q8_0-trunk model ("Q8T",
3% lower perplexity than ukisai's Q4_K_M, the same decode speed once the draft has its own head):

| file | trunk | hybrid prefill 4K/16K/32K (R9700) | decode | ppl (8 chunks) |
|---|---|---|---|---|
| ukisai Q4_K_M | q4_K/q5_K/q6_K | 1887 / 1907 / 1769 | 57-58 t/s | 7.52 |
| Q8T (this recipe) | q8_0 (unsloth UD types) | 1895-1936 / 2174-2181 / 1980-1981 | ~62 t/s | 7.29 |

On the iGPU alone the q8_0 trunk costs decode bandwidth (26.8 vs 31.3 t/s without a draft), so APU-only users may
prefer the Q4_K_M; prefill is the same.

```bash
S=scripts/halo-hybrid; R=ukisai/Swift1.5-Qwen3.8-Flash-Next
# MTP head (5 GB of range requests), shared embeddings, indexer kept bf16 like unsloth's heads
python3 $S/fetch_hf_tensors.py $R mtp-src
python convert_hf_to_gguf.py mtp-src --mtp --mtp-shared-embd --outtype bf16 --outfile mtp-bf16.gguf
# q8_0 trunk (9.4 GB fetch, 20 GB f32 intermediate, 121 GB output)
python3 $S/fetch_hf_tensors.py $R trunk-src . 'mlp\.experts\.|ngram_embedding\.shard_|^model\.visual\.|^mtp\.'
python convert_hf_to_gguf.py trunk-src --no-mtp --outtype f32 --outfile trunk-f32.gguf
python3 $S/merge_trunk.py Swift-...-Q4_K_M-00001-of-00003.gguf trunk-f32.gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  Q8T/Swift-1.5-Qwen3.8-Flash-Next-Q8trunk.gguf --name="Swift 1.5 Qwen3.8-Flash-Next (q8_0 trunk, ukisai Q4_K_M experts)"
# draft head with its own q4_K LM head (lm_head.weight from the trunk conversion)
python3 $S/add_draft_head.py mtp-bf16.gguf trunk-f32.gguf output.weight mtp-head-f32.gguf
llama-quantize --tensor-type indexer=bf16 --tensor-type ffn_gate_exps=q4_K --tensor-type ffn_up_exps=q4_K \
  --tensor-type ffn_down_exps=q5_1 --tensor-type shared_head_head=q4_K mtp-head-f32.gguf mtp-shared-exps-q4k-head-q4_K.gguf Q8_0
```

Run it with the same lines as the base model, swapping `-m` and `-md`.

## One Strix Halo by itself

No card: one device, no placement overrides. The tree's gfx1151 work (RDNA3.5 MMQ tile configs, the 32-wave
Gated-DeltaNet recurrence, WMMA flash attention for head sizes 256 and 512, the sparse attention path, the
dequantize-once q8_0 WMMA GEMM, per-layer launch fusion) all applies; every change in the tree is measured on the iGPU
too.

Qwen3.8-Flash-Next, single stream, 40K context, draft head with its own q4_K LM head on the same device:

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0 --fit off -fa on -ngl 999 -c 40960 -b 4096 -ub 4096 -np 1 \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

(`ROCm0` is the iGPU when it is the only ROCm device; with a card installed it is usually `ROCm1`.) `-ub 4096` beats
2048 by 5% / 2.5% / 2% at 4K / 16K / 32K; decode is the same at either. Measured 2026-09-28 (prefill = fresh server,
4K/16K/32K record prompts; decode = six real-content prompts, T=0.7, 400 tokens, two seeds):

| | prefill 4K / 16K / 32K tok/s | decode |
|---|---|---|
| UD-Q4_K_XL, draft with own q4_K head | 937-946 / 865-871 / 778-782 | 55.2-56.0 ms/step, 39.9-41.4 tok/s |
| same, draft sharing the q8_0 output layer | - | 58.3-58.7 ms/step, 38.3-38.5 tok/s |
| Swift 1.5 Q8T, draft with own q4_K head | 935-940 / 866-867 / 780-781 | - |

For reference on the same box: gufo (an APU-only engine) runs prefill 1,359 / 1,431 / 1,410 and MTP decode 34.8
tok/s on its own harness (temperature 0, not like for like); halogen-flash-server 0.11.1 (gfx1151-only, its own 4-bit
format) 1,246 at 8K / 1,424 at 32K prefill and 44.8 tok/s MTP decode on prose. The prefill gap is hyper-connection
traffic and the sparse-attention pipeline; experts, dense GEMMs and the GDN scan are at parity.

The 111 GB model plus the 2.6 GB head runs at 40K context as above; Qwen3.5-122B (71 GB) fits with room to spare. GLM-5.3-Flash at Q4 (200 GB) does not: that needs a second
machine ([multi-machine.md](multi-machine.md)); its Q2 quant fits one machine with a card ([rdna4.md](rdna4.md)).
