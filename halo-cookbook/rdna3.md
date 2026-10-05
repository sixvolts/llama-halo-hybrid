# Cookbook: Strix Halo + an RDNA3 card (Navi 32: RX 7800 XT / 7700 XT; Navi 31: RX 7900 XT / XTX)

Measured 2026-10-04 on gibson: one Strix Halo (Ryzen AI Max+ 395) plus a **Radeon RX 7800 XT** (gfx1101, 16 GB GDDR6,
~624 GB/s) in a **USB4 eGPU dock** ([usb4-thunderbolt.md](usb4-thunderbolt.md)). The RX 7900 XT / XTX (gfx1100, 20 / 24 GB) and RX 7700 XT (gfx1101, 12 GB) run the same kernels but were not
measured; with 20 GB or more, start from these layouts and raise `-ub`.

RDNA3 has the same f16 matrix (WMMA) instructions as the Strix Halo's own iGPU (RDNA3.5), so the card runs the same
kernels as the iGPU. One change in this tree was needed: **sparse attention for Qwen3.8's D=256 layers on RDNA3**.
The sparse walk (~2,051 selected cells per query instead of the whole context) was gated to RDNA3.5 and RDNA4; on the
7800 XT a 2,560-query batch against 28K of context takes 38.5 ms sparse against 97 ms dense. Below 8K of context it
stays dense, as on the other cards.

## Card setup

* **`amdgpu.runpm=0`** in `GRUB_CMDLINE_LINUX` ([getting started](README.md#kernel-setup)). The 7800 XT failed to
  wake from runtime power-down after ~9 minutes idle, and the stuck card took ROCm down for the iGPU as well.
* **In a USB4 / Thunderbolt dock**, as measured here, follow [usb4-thunderbolt.md](usb4-thunderbolt.md) first
  (`thunderbolt.host_reset=false`, or the card gets a 256 MB BAR and no peer access).
* **Device order:** ROCm0 is the card and ROCm1 the iGPU (check the startup log).
* **This card on a riser cable:** it logged PCIe link errors even at idle on gibson's Gen4 x4 riser, hung
  llama-server twice and dropped off the bus when its link speed was changed, while an R9700 and a V620 on the same
  cable were clean. In the dock it logged none. If a card hangs, check the link's error counters first
  ([usb4-thunderbolt.md](usb4-thunderbolt.md#troubleshooting)).

## Build

[Getting started](README.md#build) with `GPU_TARGETS="gfx1151;gfx1101"` (`gfx1100` for the RX 7900 XT / XTX).

## Qwen3.8-Flash-Next (UD-Q4_K_XL), 128K context

The 16 GB layout of the [RX 9070 XT recipe](rdna4.md#radeon-rx-9070-xt-16-gb-simulated):
dense trunk, KV cache and the MTP draft head on the card, every routed expert on the iGPU.

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 LLAMA_QSA_CHUNK_MB=256 LLAMA_SPEC_DRAFT_UB=512 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 4096 -ub 2048 -np 1 \
  -ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

Measured with the DGX Spark comparison's scripts ([README, "Against a DGX Spark"](../README.md#against-a-dgx-spark-2026-09-30)): 40 prompts at T=0, needle-in-filler
prefill at matched token counts, one run each, build a7fca0639. "Spare" is the card's free memory after load; no
layout logged an out-of-memory error during the 113K prefill.

| Layout | Decode median (prose) | Prefill 7K / 28K / 113K tok/s | Spare |
|---|---|---|---|
| Two lanes, `-ub 2048 -b 4096`, chunk 256, draft ubatch 512 (the line above) | 60.0 tok/s (49.6) | 1,538 / 1,327 / 989 | 0.04 GiB |
| Two lanes, `-ub 1024 -b 2048` (the 9070 XT line, no env) | 59.9 tok/s (49.6) | 1,304 / 1,181 / 914 | 0.7 GiB |
| Two lanes, `-ub 1024 -b 2048`, chunk 256, draft ubatch 512 | 59.9 tok/s (49.5) | 1,301 / 1,171 / 896 | 1.6 GiB |
| Strix Halo alone (same boot) | 47.8 tok/s (40.5) | ~950 / ~865 / ~700 | - |
| For comparison: V620 (32 GB, RDNA2, Gen4 x4) | 58.2 tok/s (46.9) | 1,517 / 1,324 / 1,023 | - |
| For comparison: R9700 (32 GB, RDNA4), 128K prefill preset | 67.4 tok/s | 2,354 / 2,232 / 1,815 | - |

* **The first row leaves 40 MB spare.** Use it only on a headless card. On a card that also drives a display, use
  the third row (1.6 GiB spare).
* **Quality:** KLD against the Strix Halo alone is 0.020 / 0.009 / 0.023 on the three standard setups (V620 0.043 /
  0.039 / 0.021, R9700 0.041 / 0.039 / 0.022). The large-batch setups come out lower because the card runs the same
  RDNA3 WMMA kernels as the iGPU the reference was computed on.
* The Strix Halo-alone prefill numbers come from the server log (the client-side figures were lost); server-side
  figures run ~0.5% above client-side ones.

## GLM-5.3-Flash (UD-Q2_K_XL), 128K context

The [one-box GLM recipe](rdna4.md#glm-53-flash-at-ud-q2_k_xl) with all routed experts on the
iGPU, but **one prefill lane at `-ub 1024`** and a smaller draft ubatch: with two lanes the card has 2.0 GB left after
the trunk (6.3 GB), KV and compute buffers, and the Q2 MTP head needs 2.7 GB. The head cannot move to the iGPU
because it shares the trunk's output weights (the server aborts with "pre-allocated tensor (output.weight) in a
buffer (ROCm0) that cannot run the operation").

```
LLAMA_PREFILL_LANES=1 LLAMA_SPEC_DRAFT_UB=512 \
llama-server -m GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 2048 -ub 1024 -np 1 -t 16 \
  -ot 'blk\.([3-9]|[1-3][0-9]|4[0-6])\.ffn_(gate|up|down)_exps=ROCm1,^token_embd\.weight$=CPU' \
  -md GLM-5.3-Flash-mtp-UD-Q2_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

| | Decode median (prose) | Prefill 7K / 28K / 113K | Draft acceptance |
|---|---|---|---|
| 7800 XT, one lane `-ub 1024` | 38.6 tok/s (30.5) | 284 / 276 / 242 | 0.74 |
| V620 (32 GB), two lanes `-ub 2048` | 39.1 tok/s (30.9) | 436 / 469 / 386 | 0.74 |
| R9700 (32 GB), two lanes `-ub 2048` | 43.8 tok/s (32.8) | 619 / 623 / 498 | 0.73 |

Decode holds up; prefill pays for the single lane and the small ubatch.

## Not measured

* RX 7900 XT / XTX: with 24 GB the V620 line (`-ub 2560 -b 5120`, ~10 GB of compute buffers) should fit; with 20 GB
  start from the first row above. Neither is tested.
* A 7800 XT in a PCIe slot: this card's results above are from the dock; in a healthy Gen4 x4 or wider slot expect
  the same decode and slightly faster prefill.
