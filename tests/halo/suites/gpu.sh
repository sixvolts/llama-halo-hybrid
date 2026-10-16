# gpu-<gen>: one card generation + the APU (ROCm0 = card, ROCm1 = APU). Sourced by run.sh; $GEN is rdna2|rdna3|rdna4.
# Uses the build from the base suite ($BUILD); run base first on a new commit.
#   pre-flight (stops the run with HOST-STATE): host state, GEMV probe, peer copies, against $HALO_MACHINE_DIR/<host>.json
#   card: op cases + model shapes, random-weight architectures vs CPU, then gpu/<gen>.sh's gen_steps if it exists
#         (a generation's own cases, e.g. rdna3's fallback paths)
#   two devices: batched peer copies, scheduler exactness over card + APU (lanes, hazard model)

MACHINE_JSON="$HALO_MACHINE_DIR/$HOST.json"
mkdir -p "$HALO_MACHINE_DIR"

build_p2p() {   # the peer-copy probe (host code only, so one binary for every card)
    local exe="$BUILD/bin/halo-p2p"
    if [ ! -x "$exe" ] || [ "$HALO_DIR/gpu/p2p.cpp" -nt "$exe" ]; then
        hipcc -O2 -o "$exe" "$HALO_DIR/gpu/p2p.cpp" || return 1
    fi
}

build_present() {   # the gpu suites use the base suite's build
    [ -x "$BUILD/bin/test-backend-ops" ] || { echo "detail: no build in $BUILD: run the base suite first"; return 1; }
    echo "detail: $BUILD"
}

suite_gpu() {
    step_gate build SETUP build_present
    local bless=; [ "$BLESS" = 1 ] && bless=--bless
    step_gate prod-stop    SETUP      prod_stop
    step_gate devices      SETUP      python3 "$HALO_DIR/lib/devices.py" check "$GEN" --build "$BUILD"
    export ROCR_VISIBLE_DEVICES=$(python3 "$HALO_DIR/lib/devices.py" visible "$GEN")
    step_gate p2p-build    SETUP      build_p2p
    step_gate host-state   HOST-STATE python3 "$HALO_DIR/lib/preflight.py" host "$GEN" "$MACHINE_JSON" $bless
    step_gate gemv-probe   HOST-STATE python3 "$HALO_DIR/lib/preflight.py" gemv "$GEN" "$MACHINE_JSON" "$BUILD/bin" $bless
    step_gate peer-copy    HOST-STATE python3 "$HALO_DIR/lib/preflight.py" p2p "$GEN" "$MACHINE_JSON" "$BUILD/bin/halo-p2p" $bless

    step ops-card     run_ops ROCm0 "$HALO_DIR/ops.txt" TBO_Q38_EVAL=1 TBO_GLM_EVAL=1
    step shapes-card  run_shapes ROCm0
    step archs-card   archs_dev ROCm0
    if [ -f "$HALO_DIR/gpu/$GEN.sh" ]; then . "$HALO_DIR/gpu/$GEN.sh"; gen_steps; fi
    step peer-batch   "$BUILD/bin/test-peer-copy-batch"
    [ -d "$RUN_DIR/models" ] || step models gen_models
    step sched-card   python3 "$HALO_DIR/lib/sched_check.py" card --bin "$BUILD/bin" --models "$RUN_DIR/models" \
        --out "$RUN_DIR/sched-card" --known "$HALO_DIR/known-failures.txt"
    halo_finish
}
