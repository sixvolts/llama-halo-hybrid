# Machine files

The suites read two per-machine files from `$HALO_MACHINE_DIR` (default `~/.config/halo-test`), named after
`hostname -s`. They are kept out of the tree because they describe one machine.

- `<host>.env`: optional shell variables, see [example.env](example.env). `PROD_LAUNCHER`, `PROD_PIDFILE` and
  `PROD_HEALTH` name a server the suites stop before using the GPUs and restart afterwards; without them nothing is
  stopped.
- `<host>.json`: the pre-flight references, written by the first `gpu <gen>` run on the machine and rewritten by
  `--bless`. Per generation it holds the card's PCIe link (`link`), VRAM in MiB (`vram_mib`), the GEMV probe times in
  microseconds (`gemv_us`) and the peer-copy bandwidth in GB/s (`p2p_gbs`). Later runs stop with `HOST-STATE` when the
  machine no longer matches them.

[reference.json](reference.json) is the maintainer's machine, for comparison only (it is never loaded): Strix Halo
(Framework Desktop) with a Radeon AI PRO R9700 behind a PCIe 4.0 x4 extender (`rdna4`) and an RX 7800 XT in a USB4
dock (`rdna3`).
