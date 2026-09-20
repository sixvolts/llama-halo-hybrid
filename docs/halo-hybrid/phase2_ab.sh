#!/bin/bash
# Phase 2: v1 vs v2c vs v3, 3 reps each, decode + prefill separately, on HEALTH-GATED hardware.
# The whole point vs last time: a degraded card marks its layout INVALID instead of producing a number.
# Mainframe-side GPU snapshots are requested out-of-band from its session before/after the batch.
# usage: phase2_ab.sh [layouts...]   default: v1 v2c v3     env: REMOTE_APU (v1, default RPC1), RR, KM, EXPECT_HEAD
G=/home/sixvolts/bench/glm; LOG=$G/phase2.log; SRC=/home/sixvolts/llama.cpp
LAYOUTS=${@:-v1 v2c v3}; export LLAMA_PREFILL_LANES=2 DRAFT=1
REMOTE_APU=${REMOTE_APU:-RPC1}; RR=${RR:-6}; KM=${KM:-5}; EXPECT_HEAD=${EXPECT_HEAD:-8cfdf914b}
: > $LOG; say(){ echo "$*" | tee -a $LOG; }
say "=== PHASE 2  $(date '+%F %T')  layouts=[$LAYOUTS]  REMOTE_APU=$REMOTE_APU RR=$RR KM=$KM"
# ---- artifact gate: refuse to measure a stale or wrong build
head=$(git -C $SRC log --format=%h -1); stale=$(find $SRC/ggml/src $SRC/src -newer $SRC/build-hip/bin/libggml-hip.so.0 \( -name '*.cu' -o -name '*.cpp' -o -name '*.cuh' \) 2>/dev/null | wc -l)
say "artifact: HEAD=$head stale_sources=$stale dirty=$(git -C $SRC status --short | grep -vc rocprof)"
[[ "$head" == "$EXPECT_HEAD"* ]] || { say "ABORT: HEAD $head != $EXPECT_HEAD"; exit 1; }
[ "$stale" -eq 0 ] || { say "ABORT: $stale sources newer than binary (rebuild first)"; exit 1; }
# ---- diagnostic env must be absent for any quoted number
for v in HIP_LAUNCH_BLOCKING AMD_LOG_LEVEL GGML_CUDA_TRACE_MM GGML_CUDA_DISABLE_GRAPHS; do [ -n "${!v}" ] && { say "ABORT: diagnostic env $v set"; exit 1; }; done
exec 9>/run/lock/llamabench.lock; flock 9
stop(){ pgrep -x llama-server | while read p; do kill $p; done; sleep 4; pgrep -x llama-server | while read p; do kill -9 $p; done; sleep 2; }
HW=$(echo /sys/bus/pci/devices/0000:c4:00.0/hwmon/hwmon*)
gib_health(){ # gibson R9700: temps (tolerate D3 EBUSY), AER, fault signatures — one line
  j=$(cat $HW/temp2_input 2>/dev/null); j=${j:+$((j/1000))C}; j=${j:-D3}
  aer="$(grep -cvE ' 0$' /sys/bus/pci/devices/0000:c4:00.0/aer_dev_correctable 2>/dev/null)/$(grep -cvE ' 0$' /sys/bus/pci/devices/0000:c4:00.0/aer_dev_fatal 2>/dev/null)"
  faults=$((sudo -n dmesg 2>/dev/null||dmesg) | grep -cE 'sync flood|device lost from bus|SMU is in hanged|GPU reset')
  echo "junction=$j aer=$aer faults=$faults up=$(cut -d. -f1 /proc/uptime)s"
}
wait_health(){ # $1 tag $2 log ; returns 1 on load failure
  for i in $(seq 1 900); do
    curl -s -m2 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status":"ok"' && { say "  loaded ${i}s"; return 0; }
    grep -qaE 'failed to allocate|out of memory|cudaMalloc failed|error loading|no kernel image|MUL_MAT failed|Remote RPC server crashed' "$2" 2>/dev/null && { say "  LOAD FAILED: $(grep -aoE 'failed to allocate[^\n]{0,60}|out of memory|no kernel image|MUL_MAT failed|Remote RPC server crashed' "$2" | head -1)"; return 1; }
    sleep 1
  done; say "  LOAD TIMEOUT"; return 1
}
launch(){ # $1 layout $2 tag
  case $1 in
    v1)  APU_FROM=5 REMOTE_APU=$REMOTE_APU $G/run_glm.sh     $2 25 131072 -b 32768 9>&- & ;;
    v2c) RR=$RR                          $G/run_glm_v2c.sh $2 131072    -b 32768 9>&- & ;;
    v3)  KM=$KM                          $G/run_glm_v3.sh  $2 131072    -b 32768 9>&- & ;;
    *) say "unknown layout $1"; return 1 ;;
  esac
}
stop
for L in $LAYOUTS; do
  tag=p2_$L; log=$G/$tag.log
  say ""; say "##### $L  $(date '+%T')"
  say "  gibson before: $(gib_health)"; f0=$((sudo -n dmesg 2>/dev/null||dmesg) | grep -cE 'sync flood|device lost from bus|SMU is in hanged')
  /home/sixvolts/bench/drain_pool.sh >/dev/null
  launch $L $tag || continue
  if ! wait_health $L $log; then say "  $L: INVALID (load)"; stop; continue; fi
  say "  split (verify it landed as designed):"; grep -aE 'model buffer size|KV buffer size' $log | grep -E 'RPC|ROCm' | sed -E 's/^[0-9.]+ [A-Z] +/    /' | head -8 | tee -a $LOG >/dev/null
  for r in 1 2 3; do
    python3 $G/probe_ctx.py 135,540 2>&1 | grep -E 'records ->' | sed "s/^ */  $L r$r /" | tee -a $LOG
  done
  stop
  f1=$((sudo -n dmesg 2>/dev/null||dmesg) | grep -cE 'sync flood|device lost from bus|SMU is in hanged')
  say "  gibson after:  $(gib_health)"
  [ "$f1" -gt "$f0" ] && say "  $L: INVALID — gibson fault signature appeared during run" 
  grep -qa 'Remote RPC server crashed' $log && say "  $L: INVALID — remote RPC crashed (check mainframe card)"
done
say ""; say "=== SUMMARY (mean of reps; INVALID layouts excluded by inspection)"
for L in $LAYOUTS; do
  grep -E "^  $L r[123] " $LOG | awk -v L=$L '{
    tok=$0; sub(/.*prompt +/,"",tok); split(tok,a," "); n=a[1];
    match($0,/prefill +[0-9]+/); pf=substr($0,RSTART+8,RLENGTH-8);
    match($0,/decode +[0-9.]+/); dc=substr($0,RSTART+7,RLENGTH-7);
    P[n]+=pf; D[n]+=dc; C[n]++ }
    END{ for(n in C) printf "  %-4s %6s tok: prefill %4.0f t/s  decode %5.2f t/s  (%d reps)\n", L, n, P[n]/C[n], D[n]/C[n], C[n] }' | sort -k2 -n | tee -a $LOG
done
say "PHASE2 DONE $(date '+%T')"
