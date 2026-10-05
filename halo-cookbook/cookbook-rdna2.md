# Cookbook: Strix Halo + an RDNA2 card (Navi 21: Radeon Pro V620, W6800, RX 6800 / 6900 XT)

Measured 2026-10-02 on gibson: one Strix Halo (Ryzen AI Max+ 395) plus a **Radeon Pro V620** (gfx1030, 32 GB GDDR6,
~512 GB/s, 250 W VBIOS cap) on a PCIe Gen4 x4 link. The RX 6800 / 6800 XT / 6900 XT are the same chip with 16 GB; for
those use the 16 GB layouts of the [RX 9070 XT recipe](COOKBOOK.md#radeon-rx-9070-xt-16-gb-instead-of-the-r9700-simulated-2026-10-01)
(not measured on RDNA2).

RDNA2 has no matrix (WMMA) units, so this card runs different kernels from the R9700: the tile flash-attention kernel
and dp4a integer matmuls. Three changes in this tree make it work well (all gated to RDNA2, other cards unchanged):

* **Flash attention on gfx1030** (from [llama-navi21-furnace](https://github.com/sixvolts/llama-navi21-furnace)):
  occupancy computed for RDNA2's register file, D=128/256/512/576 tile configs and dispatch caps. Without them
  `-fa on` aborts on Qwen3.8 and GLM.
* **A memory fault in the tile kernel's partial KV batch** (also present upstream): the last batch of an unaligned
  KV length loaded through a pointer to a local zero array, which faults on gfx1030. Qwen3.8 hit it on the first
  decode step after a ~113K prompt.
* **Sparse attention on the tile kernel.** Qwen3.8 (QSA) and GLM-5.3 (DSA) attend to ~2,051 selected cells per
  query; the sparse walk existed only for WMMA cards, so RDNA2 walked the whole context. At 113K that made the V620
  hybrid prefill *slower* than the Strix Halo alone (591 vs ~700 tok/s); with the sparse walk it is 1,023.

## Card setup (V620)

* **CSM off** in the BIOS, or the 32 GB BAR is not placed and ROCm sees no card.
* **Turn off GPU memory ECC (GECC).** The V620 ships with it on: the card reports 30,704 MiB instead of 32,752 and
  runs ~10% slower. Add `amdgpu.ras_enable=0` to `GRUB_CMDLINE_LINUX`, `update-grub`, reboot twice (the first boot
  schedules the change, the second applies it; `dmesg | grep GECC` says "GECC is disabled").
  **Keep the flag.** Without it the driver turns GECC back on at the next boot, and with the RAS framework active the
  card's writes into the APU's memory crawl (0.85 GB/s, 172 us for a 20 KB copy, against 6.4 GB/s and 24 us): hybrid
  decode fell from ~60 to 14-20 tok/s. This applies to the V620 on this kernel (7.0.0-34); the R9700 does not need it.
* The "VF BAR ... can't assign" lines in dmesg are the card's SR-IOV virtual functions and are harmless.
* The V620 is passively cooled: it needs a blower or forced airflow.

## Build

```
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS="gfx1151;gfx1030" -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server
```

## Qwen3.8-Flash-Next (UD-Q4_K_XL), 128K context

Same layout as the R9700 recipe: dense trunk, KV cache and the MTP draft head on the card, every routed expert on the
iGPU. ROCm0 is the V620, ROCm1 the iGPU (check the startup log).

```
sudo sh -c 'echo 2 > /proc/sys/vm/drop_caches'   # drain the iGPU's TTM pool before a big load

LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 131072 -b 5120 -ub 2560 -np 1 \
  -ot 'blk\.([0-9]|[1-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1' \
  -md mtp-Qwen3.8-Flash-Next-shared-exps-q4k-head-q4_K.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

Measured with the DGX Spark comparison's scripts (README, "Against a DGX Spark"): 40 prompts at T=0, needle-in-filler
prefill at matched token counts, one run each.

| Layout | Decode median (prose) | Prefill 7K / 28K / 113K tok/s |
|---|---|---|
| V620, two lanes, `-ub 2560 -b 5120` (the line above) | 58.2 tok/s (46.9) | 1,517 / 1,324 / 1,023 |
| V620, two lanes, `-ub 4096 -b 8192` | 59.5 tok/s (46.1) | 1,499 / 1,294 / 1,015 |
| V620, one lane, `-ub 4096` | 53.4 tok/s (42.8) | 1,008 / 931 / - |
| Strix Halo alone (same boot) | 47.8 tok/s (40.5) | 956 / 870 / 712 |
| For comparison: R9700 (32 GB, RDNA4) | 67.4 tok/s | 2,354 / 2,232 / 1,815 |

Several streams (`-np 6`, routed experts of layers 0-4 on the card, `-ub 2560`): 61.0 tok/s on one stream, 44.1 /
29.8 / 17.9 per stream at 2 / 4 / 6 streams.

* **Card memory:** 4.6 GB trunk + 1.9 GB draft + 3.7 GB KV + ~10 GB compute buffers at `-ub 2560`; 32 GB has room.
* **Quality:** KLD against the Strix Halo alone is 0.043 / 0.039 / 0.021 on the three standard setups (the R9700:
  0.041 / 0.039 / 0.022). Greedy outputs are identical between the two V620 layouts.
* **Swift 1.5 (Q8 trunk)** with the same line and its own head: 58.2 tok/s, prefill 1,482 / 1,290 / 1,012.

## GLM-5.3-Flash (UD-Q2_K_XL), 128K context

The [one-box GLM recipe](COOKBOOK.md#glm-53-flash-at-ud-q2_k_xl-one-box-2026-10-01) unchanged (all routed experts on
the iGPU, `-ub 2048 -b 4096`, the Q2 MTP export, `--reasoning-budget 0` for the benchmark):

| | Decode median (prose) | Prefill 7K / 28K / 113K | Draft acceptance |
|---|---|---|---|
| V620 | 39.1 tok/s (30.9) | 436 / 469 / 386 | 0.74 |
| R9700 | 43.8 tok/s (32.8) | 619 / 623 / 498 | 0.73 |

## Where the V620 is slower than the R9700, and why

At 28K the card is busy ~19 s of a 30 s prefill and the iGPU ~8 s, so the card sets the pace. Its time goes to the
trunk's q8_0 matmuls (~35%, at ~32 of ~80 int8 TOPS: the dp4a path's limit, not a tuning gap), attention (~30%,
already sparse), the gated delta net (~6%) and many small element-wise kernels (most near memory bandwidth).
Moving dense layers to the iGPU (`-ts 7,1`) does not help: 28K prefill 1,276 -> 1,224, decode 57 -> 51.
