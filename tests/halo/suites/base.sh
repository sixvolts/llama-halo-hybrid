# base: infra + APU. Needs only the Strix Halo. Sourced by run.sh after halo_init; $BUILD, $LEVEL, $NO_BUILD set.
#   smoke: HIP build (every cookbook target) + CPU build + CPU unit tests, then on the APU: op cases + model shapes
#   full:  + random-weight architectures vs CPU, state rollback / save-load, scheduler + RPC loopback exactness

CPU_BUILD=${CPU_BUILD:-$REPO/build-halo-cpu}
GPU_TARGETS_ALL=${HALO_GPU_TARGETS:-"gfx1151;gfx1201;gfx1100;gfx1101;gfx1030"}   # every cookbook target by default
# the architectures whose graphs or kernels this tree changes (glm5-next, bailingmoe3, dots3note: upstream's, through the
# shared MoE and attention paths)
ARCHS='^(qwen4exp|qwen35|qwen35moe|glm5next|glm5-next|glm-dsa|deepseek4|bailingmoe3|dots3note)$'

build_jobs() {   # ~3 GiB per HIP compile job: production runs during builds and memguard kills it below 10 GiB
    local avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    local j=$(( (avail - 16) / 3 ))
    [ $j -gt "$(nproc)" ] && j=$(nproc)
    [ $j -lt 2 ] && j=2
    echo $j
}

build_hip() {
    if [ ! -f "$BUILD/CMakeCache.txt" ]; then
        cmake -S "$REPO" -B "$BUILD" -DGGML_HIP=ON "-DGPU_TARGETS=$GPU_TARGETS_ALL" \
            -DCMAKE_HIP_COMPILER="$ROCM_PATH/lib/llvm/bin/clang" -DGGML_RPC=ON \
            -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON || return 1
    else
        cmake "$BUILD" > /dev/null || return 1   # re-glob: new .cu files are picked up only at configure time
    fi
    grep -q "GPU_TARGETS.*=$GPU_TARGETS_ALL" "$BUILD/CMakeCache.txt" || { echo "detail: $BUILD lacks GPU_TARGETS=$GPU_TARGETS_ALL"; return 1; }
    nice -n 10 cmake --build "$BUILD" -j "$(build_jobs)" || return 1
    echo "detail: $(basename "$BUILD") built for $GPU_TARGETS_ALL"
}

build_cpu() {
    if [ ! -f "$CPU_BUILD/CMakeCache.txt" ]; then
        cmake -S "$REPO" -B "$CPU_BUILD" -DGGML_HIP=OFF -DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON || return 1
    else
        cmake "$CPU_BUILD" > /dev/null || return 1
    fi
    nice -n 10 cmake --build "$CPU_BUILD" -j "$(build_jobs)" || return 1
    echo "detail: CPU-only build ok"
}

unit_cpu() {   # upstream's ctest "main" label on the CPU build. The state tests run per model in the state-* steps
    # (one abort would hide every later architecture); the ggml-vocabs tokenizer test needs git-lfs for its downloads.
    local skip='^(test-save-load-state|test-recurrent-state-rollback)$'
    command -v git-lfs > /dev/null || skip='^(test-save-load-state|test-recurrent-state-rollback|test-tokenizers-ggml-vocabs)$'
    (cd "$CPU_BUILD" && ctest -L main -E "$skip" --timeout 900 -j 8 --output-on-failure) > "$RUN_DIR/ctest.out" 2>&1
    local rc=$?
    cat "$RUN_DIR/ctest.out"
    echo "detail: $(grep -E 'tests passed|tests failed' "$RUN_DIR/ctest.out" | tail -1)"
    return $rc
}

server_cpu() {   # the server tests of this tree's behaviour (tools/server/tests/unit/test_halo.py) and upstream's
    # speculative ones, on the CPU build. Needs pytest and the packages of tools/server/tests/requirements.txt (point
    # HALO_PYTHON at a venv's python); downloads two tiny models on first use. A free port: never a running server's.
    local py=${HALO_PYTHON:-python3}
    "$py" -c 'import pytest, requests, openai' 2> /dev/null || {
        echo "detail: $py lacks pytest/requests/openai (set HALO_PYTHON to a venv with tools/server/tests/requirements.txt)"
        return 77
    }
    local port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    local out="$RUN_DIR/server-cpu.out"
    (cd "$REPO/tools/server/tests" && mkdir -p tmp && PORT=$port LLAMA_SERVER_BIN_PATH="$CPU_BUILD/bin/llama-server" \
        "$py" -m pytest -q unit/test_halo.py unit/test_speculative.py) > "$out" 2>&1
    local rc=$?
    tail -40 "$out"
    echo "detail: $(grep -E '[0-9]+ (passed|failed)' "$out" | tail -1)"
    return $rc
}

gen_models() {   # tiny random-weight models of every architecture (seeded, so every run uses the same weights)
    mkdir -p "$RUN_DIR/models"
    "$CPU_BUILD/bin/test-llama-archs" -s 42 -o "$RUN_DIR/models/" || return 1
    echo "detail: $(ls "$RUN_DIR/models" | wc -l) models"
}

archs_dev() {   # random-weight architectures on one device vs the CPU backend (NMSE) and a save/load roundtrip
    local out="$RUN_DIR/archs-$1.txt"
    "$BUILD/bin/test-llama-archs" -s 42 -a "$ARCHS" -b "$1" 2>&1 | tee "$out"
    local rc=${PIPESTATUS[0]}
    # a glm5-next file loads on glm5next; upstream's port is what a file the alias declines runs
    LLAMA_GLM5_NEXT_UPSTREAM=1 "$BUILD/bin/test-llama-archs" -s 42 -a '^glm5-next$' -b "$1" 2>&1 | tee -a "$out"
    local rc_up=${PIPESTATUS[0]}
    echo "detail: $(grep -o 'Summary: .*' "$out" | cut -c10- | paste -sd ';' | sed 's/;/; upstream glm5-next: /')"
    [ $rc = 0 ] && return $rc_up
    return $rc
}

state_check() {   # state_check <bin dir> <tag>: rollback + save/load per generated model, vs known-failures.txt
    python3 "$HALO_DIR/lib/state_check.py" "$1" "$RUN_DIR/models" "$RUN_DIR/state-$2" "$HALO_DIR/known-failures.txt"
}

suite_base() {
    unset ROCR_VISIBLE_DEVICES
    step_gate apu SETUP python3 "$HALO_DIR/lib/devices.py" visible apu
    if [ "$NO_BUILD" != 1 ]; then
        step build-hip build_hip || { halo_finish; return 1; }
        step build-cpu build_cpu || { halo_finish; return 1; }
    fi
    step unit-cpu unit_cpu
    step env-doc python3 "$HALO_DIR/lib/env_check.py" "$REPO"
    step server-cpu server_cpu
    step models gen_models
    step state-cpu state_check "$CPU_BUILD/bin" cpu
    step fuse-cpu  python3 "$HALO_DIR/lib/sched_check.py" fuse --bin "$CPU_BUILD/bin" --models "$RUN_DIR/models" \
        --out "$RUN_DIR/fuse-cpu" --known "$HALO_DIR/known-failures.txt"

    step_gate prod-stop SETUP prod_stop
    step_gate devices SETUP python3 "$HALO_DIR/lib/devices.py" check apu --build "$BUILD"
    export ROCR_VISIBLE_DEVICES=$(python3 "$HALO_DIR/lib/devices.py" visible apu)
    step ops-apu      run_ops ROCm0 "$HALO_DIR/ops.txt" TBO_Q38_EVAL=1 TBO_GLM_EVAL=1
    step shapes-apu   run_shapes ROCm0
    if [ "$LEVEL" = full ]; then
        step archs-apu archs_dev ROCm0
        step state-apu state_check "$BUILD/bin" apu
        step sched-rpc python3 "$HALO_DIR/lib/sched_check.py" base --bin "$BUILD/bin" --models "$RUN_DIR/models" \
            --out "$RUN_DIR/sched-rpc" --known "$HALO_DIR/known-failures.txt"
    fi
    halo_finish
}
