#!/bin/bash
# Halo regression suites. Runs on demand; every GPU step takes production down and restores it on exit.
#
#   tests/halo/run.sh base  [smoke|full]          infra + APU (builds first unless --no-build)
#   tests/halo/run.sh gpu   <rdna2|rdna3|rdna4>   one card generation + the APU (uses the base build)
#   tests/halo/run.sh last  [n]                   the last n ledger lines
#
# options: --build <dir>   HIP build to build/use (default <repo>/build-halo, every cookbook GPU target)
#          --no-build      use the existing build as is
#          --bless         gpu: record this machine's pre-flight references (GEMV probe, peer copies)
# Results: $HALO_RESULTS (default ~/.local/state/halo-test)/<date>-<commit>-<host>-<suite>/ (summary.md, verdict.json,
# per-step logs), one line per run in $HALO_RESULTS/ledger.jsonl. Exit 0 only on PASS.
set -u
. "$(dirname "$0")/lib/common.sh"

BUILD=$REPO/build-halo; NO_BUILD=0; BLESS=0; args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --build) BUILD=$(realpath "$2"); shift 2;;
        --no-build) NO_BUILD=1; shift;;
        --bless) BLESS=1; shift;;
        -h|--help) sed -n '2,13p' "$0"; exit 0;;
        *) args+=("$1"); shift;;
    esac
done
set -- "${args[@]}"

case "${1:-}" in
    base)
        . "$HALO_DIR/suites/base.sh"
        halo_init base "${2:-smoke}"
        suite_base;;
    gpu)
        GEN=${2:?gpu needs a generation: rdna2, rdna3 or rdna4}
        . "$HALO_DIR/suites/base.sh"   # shared helpers (gen_models, archs_dev)
        . "$HALO_DIR/suites/gpu.sh"
        CPU_BUILD=${CPU_BUILD:-$REPO/build-halo-cpu}
        halo_init "gpu-$GEN" full
        suite_gpu;;
    last)
        python3 "$HALO_DIR/lib/ledger.py" last "$HALO_RESULTS/ledger.jsonl" "${2:-10}";;
    *)
        sed -n '2,13p' "$0"; exit 2;;
esac
