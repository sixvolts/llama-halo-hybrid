# Halo regression suites

On-demand correctness and host-state checks for the halo-hybrid configurations: a Strix Halo APU alone, or the APU
plus one discrete card. Every GPU step takes the production server down (by its pidfile) under the bench lock and
restores it on exit.

```
tests/halo/run.sh base [smoke|full]     # infra + APU; builds build-halo (all cookbook GPU targets) + build-halo-cpu
tests/halo/run.sh gpu rdna4             # R9700 / RX 9070 + APU, using the base build
tests/halo/run.sh gpu rdna4 --bless     # record this machine's pre-flight references (first run on a machine)
tests/halo/run.sh last                  # recent runs from the ledger
```

Results go to `~/bench/results/halo-test/<date>-<commit>-<host>-<suite>/`. Each run writes `summary.md`, `verdict.json`
and one log per step, plus one line in `ledger.jsonl`. The exit code is 0 only on `PASS`.

| Suite | Steps |
|---|---|
| base smoke | HIP build (gfx1151, gfx1201, gfx1100, gfx1101, gfx1030), CPU-only build, ctest `main` on the CPU build, test-backend-ops on the APU (`ops.txt` + Qwen3.8 / GLM model shapes) |
| base full | + random-weight architectures vs CPU on the APU, recurrent-state rollback and save/load, scheduler exactness through an RPC loopback (local vs RPC vs split, 1 vs 2 lanes) |
| gpu &lt;gen&gt; | pre-flight (host state, GEMV probe, peer copies; a failure stops the run with `HOST-STATE`), test-backend-ops and architectures on the card, batched peer copies, scheduler exactness over card + APU (production layout and layer split, 1 vs 2 lanes, lane-pair hazard model and its self-test) |

Verdicts: `PASS`, `FAIL` (a step failed), `HOST-STATE` (the machine is not in its reference state, so no result
after it would mean anything), `SETUP` (a device or tool is missing).

Devices are found by gfx name (`lib/devices.py`) and exposed with `ROCR_VISIBLE_DEVICES`, one card plus the APU, so
a second card in the machine (a dock) never joins a run. Mixed-generation setups are not supported.

`machines/<host>.env` names the production launcher, pidfile and health URL. `machines/<host>.json` holds the
pre-flight references per generation (root-port link, VRAM, GEMV probe times, peer-copy bandwidth). The first
`--bless` run records the measured values.
