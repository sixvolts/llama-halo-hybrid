# Cookbook: a card in a USB4 / Thunderbolt eGPU dock

The Strix Halo has two 40 Gb/s USB4 ports, so a card can sit in an eGPU dock instead of a PCIe slot. Measured
2026-10-04 on gibson with an **RX 7800 XT** in a Thunderbolt 5 dock (Intel JHL9480 hub, its own power supply, a PCIe
x16 slot) on one of the USB4 ports. The recipes are the card's own page ([rdna3.md](rdna3.md) for this card); this
page is the setup and what the link costs.

## What you need

* A dock with a full-size PCIe slot and its own power for the card (a plain Thunderbolt hub will not do).
* The dock plugged in and powered **before** the box boots (see the next section for why).
* The kernel setup from [getting started](README.md#kernel-setup), including `amdgpu.runpm=0`.

## Setup

1. Add **`thunderbolt.host_reset=false`** to `GRUB_CMDLINE_LINUX`, `sudo update-grub`, reboot with the dock connected.

   The BIOS builds the PCIe tunnel to the dock at power-on and gives the card its full BAR (16 GB on the 7800 XT). By
   default the Linux Thunderbolt driver resets the host router at load, which tears that tunnel down; the card then
   comes back by hot-plug with a 256 MB BAR (dmesg: "Not enough PCI address space for a large BAR"). With the small
   BAR the iGPU cannot reach the card's memory: ROCm reports no peer access from the iGPU to the card, and HIP peer
   copies crash inside `libamdhip64` instead of returning an error. With the parameter, the driver keeps the BIOS
   tunnel.

2. Check after boot:

   ```
   boltctl list                                  # the dock: status authorized, authflags boot
   sudo lspci -vv -d 1002: | grep "Region 0"      # the card's BAR 0: size=16G (not 256M)
   llama-server --list-devices                   # both GPUs; note which is ROCm0
   ```

3. If the dock shows `connected` but not `authorized` (security level "user" and no BIOS tunnel, for example after
   plugging it in while running), authorize it once: `sudo boltctl authorize <uuid>` (`boltctl list` shows the uuid).
   That brings the card up by hot-plug, with the small BAR; reboot with the dock connected to get the full one.

The device order can differ from a PCIe slot: with the BIOS tunnel the card was ROCm0 and the iGPU ROCm1; after a
hot-plug it was the other way round. Check `--list-devices` before copying a line.

## What the link costs

The card still trains its own PCIe link inside the dock (Gen4 x4 here); the bottleneck is the 40 Gb/s USB4 tunnel.

| | USB4 dock (7800 XT) | PCIe Gen4 x4 slot (V620, R9700) |
|---|---|---|
| Card <-> iGPU peer copy, 64 MB | 3.8 GB/s each way | ~6.4 GB/s each way |
| Both directions at once | 3.8 GB/s total | - |
| 20 KB copy (one decode-step crossing) | 45 us | 24 us |
| Pinned host <-> card | 3.85 GB/s one way, 3.06 + 3.06 both ways | - |

In practice this costs little. Qwen3.8-Flash-Next at 128K context on the 7800 XT in the dock decodes at 60.0 tok/s
with 1,538 / 1,327 / 989 tok/s prefill at 7K / 28K / 113K ([rdna3.md](rdna3.md)); the V620 in a Gen4 x4 slot
decodes at 58.2 with 1,517 / 1,324 / 1,023. A decode step crosses the link a few times
with small tensors, so the latency matters more than the bandwidth, and 45 us per crossing is a small part of a
step that takes tens of milliseconds.

Not measured: an R9700 or another 32 GB card in the dock, where prefill moves more data per ubatch across the link.

## Troubleshooting

* **Card hangs, `llama-server` waits forever on a GPU event, or the box's network card stalls under load:** check the
  PCIe link's error counters.

  ```
  sudo lspci -vv | grep -E "^[0-9a-f]{2}:|CESta|UESta"    # look for BadTLP+, BadDLLP+, Timeout+, CmpltTO+
  ```

  BadTLP, BadDLLP and replay timeouts on the card's link mean signal trouble. The 7800 XT measured here showed them
  even at idle on a Gen4 x4 riser cable (where an R9700 and a V620 were clean), hung `llama-server` twice and
  dropped off the bus when its link speed was changed; in the dock it ran every benchmark with zero errors. Clear the
  counters with `sudo setpci -s <bdf> ECAP_AER+0x10.L=0xffffffff` and watch whether they come back.
* **Do not change a card's link speed with `setpci` while amdgpu is bound to it.** On gibson that dropped the card
  off the bus and hard-reset the box.
* **Card gone after the box idles for a while:** runtime power management; set `amdgpu.runpm=0`.
* **Peer access missing or HIP crashes on a cross-device copy:** the small-BAR case above. Check `Region 0` and boot
  with the dock connected and `thunderbolt.host_reset=false`.
