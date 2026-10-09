# Shared helpers for the halo regression suites (sourced by run.sh and the suite scripts).
#
# A suite is a list of steps. Each step runs one command with its output in $RUN_DIR/<step>.log and records
# PASS / FAIL / SKIP from the exit code. A step that must stop the run on failure (pre-flight) uses step_gate.
# The run's verdict, step table and ledger line are written by halo_finish.

HALO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HALO_DIR/../.." && pwd)
HOST=$(hostname -s)
HALO_RESULTS=${HALO_RESULTS:-$HOME/bench/results/halo-test}
HALO_LOCK=${HALO_LOCK:-/run/lock/llamabench.lock}
export PATH=/opt/rocm/bin:$PATH

# machine settings: production launcher, pidfile, health URL, link references (see machines/README.md)
PROD_LAUNCHER=; PROD_PIDFILE=; PROD_HEALTH=
[ -f "$HALO_DIR/machines/$HOST.env" ] && . "$HALO_DIR/machines/$HOST.env"

halo_init() {   # halo_init <suite> <level>
    SUITE=$1; LEVEL=$2
    COMMIT=$(git -C "$REPO" rev-parse --short=9 HEAD)
    DIRTY=$(git -C "$REPO" status --porcelain --untracked-files=no | grep -q . && echo 1 || echo 0)
    RUN_DIR=$HALO_RESULTS/$(date +%Y%m%d-%H%M%S)-$COMMIT-$HOST-$SUITE
    mkdir -p "$RUN_DIR"
    : > "$RUN_DIR/steps.tsv"
    T0=$(date +%s)
    halo_log "suite $SUITE ($LEVEL) on $HOST, commit $COMMIT$( [ "$DIRTY" = 1 ] && echo ' (dirty)'), results $RUN_DIR"
}

halo_log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUN_DIR/run.log" >&2; }

_record() {   # _record <name> <verdict> <seconds> <detail>
    printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RUN_DIR/steps.tsv"
    halo_log "$(printf '%-28s %-10s %5ss  %s' "$1" "$2" "$3" "$4")"
}

# step <name> <command...>: exit 0 = PASS, 77 = SKIP, anything else = FAIL. The command's last line of output that
# starts with "detail:" becomes the step's detail column.
step() {
    local name=$1; shift
    local t=$(date +%s) rc=0
    "$@" > "$RUN_DIR/$name.log" 2>&1 || rc=$?
    local detail=$(grep '^detail:' "$RUN_DIR/$name.log" | tail -1 | cut -c9-)
    local v=PASS
    [ $rc = 77 ] && v=SKIP
    [ $rc != 0 ] && [ $rc != 77 ] && v=FAIL && [ -z "$detail" ] && detail="exit $rc, see $name.log"
    _record "$name" "$v" $(( $(date +%s) - t )) "$detail"
    [ $v != FAIL ]
}

# step_gate <name> <verdict-on-failure> <command...>: like step, but a failure ends the run with that verdict
# (HOST-STATE for pre-flight, SETUP for a missing device or tool)
step_gate() {
    local name=$1 fail_v=$2; shift 2
    if ! step "$name" "$@"; then
        sed -i '$ s/\tFAIL\t/\t'"$fail_v"'\t/' "$RUN_DIR/steps.tsv"
        halo_log "$name failed: stopping with $fail_v"
        halo_finish
        exit 1
    fi
}

# --- production: down for every GPU run, restored on exit (user decision 2026-10-09) ---
PROD_WAS_UP=0
prod_stop() {
    if [ -n "$PROD_PIDFILE" ] && [ -f "$PROD_PIDFILE" ]; then
        local p=$(cat "$PROD_PIDFILE")
        if kill -0 "$p" 2>/dev/null && [ "$(ps -o comm= -p "$p")" = llama-server ]; then
            halo_log "stopping production (pid $p)"
            PROD_WAS_UP=1
            kill "$p"
            for i in $(seq 1 90); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
            kill -0 "$p" 2>/dev/null && { halo_log "production pid $p did not exit"; return 1; }
            sleep 5
        fi
    fi
    halo_log "waiting for the bench lock"
    exec 9>"$HALO_LOCK"
    flock 9
    halo_log "bench lock held"
    trap prod_restore EXIT
}

prod_restore() {
    exec 9>&-
    [ $PROD_WAS_UP = 1 ] || return 0
    PROD_WAS_UP=0
    halo_log "restoring production ($PROD_LAUNCHER)"
    sleep 3
    setsid nohup "$PROD_LAUNCHER" > /dev/null 2>&1 < /dev/null &
    for i in $(seq 1 600); do
        curl -s -m 2 "$PROD_HEALTH" 2>/dev/null | grep -q '"ok"' && { halo_log "production healthy after ${i}s"; return 0; }
        sleep 1
    done
    halo_log "WARNING: production not healthy after 600 s"
}

# --- results ---
halo_finish() {
    local verdict=PASS
    grep -q $'\tSETUP\t' "$RUN_DIR/steps.tsv" && verdict=SETUP
    grep -q $'\tHOST-STATE\t' "$RUN_DIR/steps.tsv" && verdict=HOST-STATE
    grep -q $'\tFAIL\t' "$RUN_DIR/steps.tsv" && verdict=FAIL
    python3 "$HALO_DIR/lib/ledger.py" finish "$RUN_DIR" "$HALO_RESULTS/ledger.jsonl" \
        suite="$SUITE" level="$LEVEL" host="$HOST" commit="$COMMIT" dirty="$DIRTY" verdict="$verdict" \
        seconds=$(( $(date +%s) - T0 )) setup="${SETUP:-}"
    halo_log "VERDICT $verdict ($SUITE $LEVEL, $(( ($(date +%s) - T0) / 60 )) min) - $RUN_DIR"
    [ $verdict = PASS ]
}

# run_ops <backend> <op list file | ops | -> [VAR=value...]: test-backend-ops in test mode ("-": no -o filter, for
# the env-selected shape sets that replace the case list); FAIL on any failed case or a missing "tests passed" line
run_ops() {
    local backend=$1 ops=$2; shift 2
    [ -f "$ops" ] && ops=$(grep -v '^#' "$ops" | tr -s ' \n' ',' | sed 's/^,//; s/,$//')
    local filter=(); [ "$ops" != - ] && filter=(-o "$ops")
    # one worker: with -j N a worker's hipMemset on the legacy stream breaks another's HIP graph capture (2026-10-09)
    env "$@" "$BUILD/bin/test-backend-ops" test -b "$backend" "${filter[@]}" 2>&1 | tee /dev/stderr \
        | python3 "$HALO_DIR/lib/tbo_check.py"
}

# run_shapes <backend>: the real-model shape sets that replace the case list (Qwen3.8 expert GEMMs + MoE blocks,
# Swift's thin bf16 weights)
run_shapes() {
    local a b rc=0
    a=$(run_ops "$1" - TBO_Q38_MOE_EVAL=1) || rc=1
    b=$(run_ops "$1" - TBO_THIN_W=1) || rc=1
    printf '%s\n%s\n' "$a" "$b" | grep -v '^detail:'
    echo "detail: q38 moe $(echo "$a" | grep '^detail:' | cut -c9-); thin w $(echo "$b" | grep '^detail:' | cut -c9-)"
    return $rc
}
