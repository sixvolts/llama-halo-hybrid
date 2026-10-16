# Halo regression suites

On-demand correctness and host-state checks for the halo-hybrid configurations: a Strix Halo APU alone, or the APU
plus one discrete card. Every GPU step takes the production server down (by its pidfile) under the bench lock and
restores it on exit.

```
tests/halo/run.sh base [smoke|full]     # infra + APU; builds build-halo (all cookbook GPU targets) + build-halo-cpu
tests/halo/run.sh gpu rdna4             # R9700 / RX 9070 + APU, using the base build
tests/halo/run.sh gpu rdna3             # RX 7800 XT / 7900 + APU (on the test box: the USB4 dock)
tests/halo/run.sh gpu rdna4 --bless     # record this machine's pre-flight references (first run on a machine)
tests/halo/run.sh last                  # recent runs from the ledger
```

Results go to `$HALO_RESULTS/<date>-<commit>-<host>-<suite>/` (default `~/.local/state/halo-test`; `<host>.env` may set
`HALO_RESULTS`). Each run writes `summary.md`, `verdict.json`
and one log per step, plus one line in `ledger.jsonl`. The exit code is 0 only on `PASS`.

| Suite | Steps |
|---|---|
| base smoke | HIP build (gfx1151, gfx1201, gfx1100, gfx1101, gfx1030), CPU-only build, ctest `main` on the CPU build, every environment switch the code reads listed in [configuration.md](../../halo-cookbook/configuration.md) and none it no longer reads (`lib/env_check.py`), test-backend-ops on the APU (`ops.txt` + Qwen3.8 / GLM model shapes) |
| base full | + random-weight architectures vs CPU on the APU, recurrent-state rollback and save/load, scheduler exactness through an RPC loopback (local vs RPC vs split, 1 vs 2 lanes, and a composite rpc-server with the four-lane pipeline) |
| gpu &lt;gen&gt; | pre-flight (host state, GEMV probe, peer copies; a failure stops the run with `HOST-STATE`), test-backend-ops and architectures on the card, batched peer copies, scheduler exactness over card + APU (production layout and layer split, 1 vs 2 lanes, lane-pair hazard model and its self-test, the production layout behind a composite rpc-server that owns both devices) |
| gpu rdna3 | + `gpu/rdna3.sh`: the matmul, MoE and attention cases again with RDNA3's default fast paths off (q8_0 and k-quant WMMA, D=256 sparse attention), so the MMQ and dense fallbacks are checked on the card too |

Verdicts: `PASS`, `FAIL` (a step failed), `HOST-STATE` (the machine is not in its reference state, so no result
after it would mean anything), `SETUP` (a device or tool is missing).

Devices are found by gfx name (`lib/devices.py`) and exposed with `ROCR_VISIBLE_DEVICES`, one card plus the APU, so
a second card in the machine (a dock) never joins a run. Mixed-generation setups are not supported.

Per-machine files live in `$HALO_MACHINE_DIR` (default `~/.config/halo-test`), not in the tree; see
[machines/README.md](machines/README.md). `<host>.env` names the production launcher, pidfile and health URL; without
it nothing is stopped or restarted. `<host>.json` holds the pre-flight references per generation (the card's link, VRAM, GEMV probe times, peer-copy bandwidth). The first run
of a generation on a machine records them; `--bless` re-records them. The link is the card's own one, between its
topmost AMD port and the port above it: the root port in a slot, the dock's bridge behind USB4 (the USB4 root port
only shows the tunnel). A dock card also needs `thunderbolt.host_reset=false` for its full BAR.

`gpu/<gen>.sh`, when present, adds the generation's own steps (`gen_steps`) after the card's op and architecture
steps.

## Running on your own machine

Prerequisites: a Strix Halo APU (gfx1151) with ROCm (`$ROCM_PATH`, default `/opt/rocm`; `rocminfo` and `hipcc` on the
path), cmake, ninja, python3, and for `gpu <gen>` one card of that generation. `base` builds what it needs; the `gpu`
suites use the base suite's build (`--build <dir>` to name another, `--no-build` to use one as is).

| Variable | Default | Meaning |
|---|---|---|
| `HALO_MACHINE_DIR` | `~/.config/halo-test` | per-machine files (above) |
| `HALO_RESULTS` | `~/.local/state/halo-test` | run directories and `ledger.jsonl` |
| `HALO_LOCK` | `/run/lock/llamabench.lock` | lock every GPU step holds (shared with your own benchmarks) |
| `HALO_GPU_TARGETS` | `gfx1151;gfx1201;gfx1100;gfx1101;gfx1030` | GPU_TARGETS of a new `build-halo` (keep gfx1151) |
| `CPU_BUILD` | `<repo>/build-halo-cpu` | the CPU-only build |
| `ROCM_PATH` | `/opt/rocm` | ROCm install |

Production handling is opt-in: only with `PROD_PIDFILE`, `PROD_LAUNCHER` and `PROD_HEALTH` all set in `<host>.env` is
a server stopped and restarted; with `PROD_PIDFILE` alone the run stops with `SETUP` rather than stop something it
cannot restart. RDNA2 cards run as `gpu rdna2` (a Radeon Pro V620 needs `amdgpu.ras_enable=0` on the kernel command
line). The `devices` step checks that the build's GPU_TARGETS include the card and the APU.

Each step's log is `<step>.log` in the run directory; a step's last `detail:` line is its summary in `summary.md`.
A step exits 0 for `PASS`, 77 for `SKIP`, anything else for `FAIL`. The run's verdict is the worst step:
`FAIL` over `HOST-STATE` over `SETUP` over `PASS`. Gate steps (lock, devices, pre-flight) stop the run on failure;
a checker's own error (exit 2) counts as `SETUP`. `gpu rdna3` repeats its matmul, MoE and attention cases with
`GGML_CUDA_Q8_WMMA=0 GGML_CUDA_KQ_WMMA=0 GGML_CUDA_FA_SPARSE_D256=0`; the scheduler checks set `GGML_SCHED_PAIR_DEBUG`
and `LLAMA_PREFILL_LANES` themselves.
