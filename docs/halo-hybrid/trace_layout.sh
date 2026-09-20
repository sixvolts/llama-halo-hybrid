#!/bin/bash
# trace_layout.sh <v1|v2c|v3> [tag]: one layout with GGML_SCHED_TRACE_WAITS=1 + GGML_RPC_SEND_TRACE=1, one 3148-token
# probe (prefill + ~96 decode tokens), then a per-label aggregate of the sched-trace summary lines. Diagnostic only:
# never quote its throughput. Env: RR, KM, UB as in phase2_ab.sh.
G=/home/sixvolts/bench/glm; L=$1; TAG=trace_${L}${2:-}; LOG=$G/$TAG.log; OUT=$G/$TAG.out
export LLAMA_PREFILL_LANES=2 DRAFT=1 GGML_SCHED_TRACE_WAITS=1 GGML_RPC_SEND_TRACE=1
exec 9>/run/lock/llamabench.lock; flock 9
stop(){ pgrep -x llama-server | while read p; do kill $p; done; sleep 4; pgrep -x llama-server | while read p; do kill -9 $p; done; sleep 2; }
stop; : > $OUT
case $L in
  v1)  APU_FROM=5 REMOTE_APU=${REMOTE_APU:-RPC1} $G/run_glm.sh $TAG 25 131072 -b 32768 9>&- & ;;
  v2c) RR=${RR:-6} $G/run_glm_v2c.sh $TAG 131072 -b 32768 9>&- & ;;
  v3)  KM=${KM:-5} $G/run_glm_v3.sh $TAG 131072 -b 32768 -ub ${UB:-1024} 9>&- & ;;
esac
for i in $(seq 1 900); do
  curl -s -m2 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status":"ok"' && { echo "loaded ${i}s" | tee -a $OUT; break; }
  grep -qaE 'failed to allocate|out of memory|Remote RPC server crashed|no kernel image' $LOG && { echo "LOAD FAILED" | tee -a $OUT; stop; exit 1; }
  sleep 1
done
n0=$(grep -ac 'sched-trace' $LOG)
python3 $G/probe_ctx.py 135 2>&1 | tee -a $OUT
python3 $G/probe_ctx.py 135 2>&1 | tee -a $OUT     # second pass: warm, same prompt shape -> the decode steps to read
stop
echo "=== sched-trace summary lines (pass 2 only), mean per label ===" | tee -a $OUT
grep -a 'sched-trace' $LOG | tail -n +$((n0+1)) | grep -aE 'sched-trace [a-z0-9_-]+: splits' | \
  sed -E 's/^[0-9.]+ I //' | awk '
  { lab=$2; sub(":","",lab); n[lab]++
    for(i=1;i<=NF;i++){ if($i=="splits") sp[lab]+=$(i+1)+0; if($i=="no-input") ni[lab]+=$(i+1)+0; if($i=="input-copy") ic[lab]+=$(i+1)+0; if($i=="sync-copy") sc[lab]+=$(i+1)+0;
      if($i=="submit") sub_[lab]+=$(i+1)+0; if($i=="(copies") cp[lab]+=$(i+1)+0; if($i=="compute") cm[lab]+=$(i+1)+0;
      if($i=="send") { rs[lab]+=$(i+1)+0; rsm[lab]+=$(i+2)+0 } if($i=="fetch") f[lab]+=$(i+1)+0; if($i=="fetch-wait") { fw[lab]+=$(i+1)+0; fwm[lab]+=$(i+2)+0 } } }
  END { for(l in n) printf "  %-10s n=%4d splits %5.1f | waits: no-input %4.1f input-copy %4.1f sync-copy %4.1f | submit %6.1f ms (copies %5.1f, compute %6.1f) | remote send %4.1f (%6.1f ms) fetch %4.1f fetch-wait %4.1f (%6.1f ms)\n",
        l, n[l], sp[l]/n[l], ni[l]/n[l], ic[l]/n[l], sc[l]/n[l], sub_[l]/n[l], cp[l]/n[l], cm[l]/n[l], rs[l]/n[l], rsm[l]/n[l], f[l]/n[l], fw[l]/n[l], fwm[l]/n[l] }' | tee -a $OUT
echo "=== blocking rpc sends > 2 ms (pass 2), top by count ===" | tee -a $OUT
grep -a 'rpc blocking send' $LOG | tail -n +1 | sed -E 's/^[0-9.]+ I //' | awk '{c[$4" "$5" "$6" "$7" "$8]++; m[$4" "$5" "$6" "$7" "$8]+=$(NF-1)} END{for(k in c) printf "  %5d x %-40s mean %.1f ms\n", c[k], k, m[k]/c[k]}' | sort -rn | head -8 | tee -a $OUT
echo "TRACE DONE $(date +%T)" | tee -a $OUT
