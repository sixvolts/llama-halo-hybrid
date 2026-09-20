#!/bin/bash
# Fixed-prompt greedy generation for wire-change correctness. usage: greedy_ref.sh <tag> <layout v1|v3> [KM]
# Output goes to ~/bench/glm/greedy_<tag>.txt for diff against another build. Deterministic prompt, temp 0.
G=/home/sixvolts/bench/glm; TAG=$1; L=$2; KM=${3:-4}
exec 9>/run/lock/llamabench.lock; flock 9
pgrep -x llama-server | while read p; do kill $p; done; sleep 4
/home/sixvolts/bench/drain_pool.sh >/dev/null
export LLAMA_PREFILL_LANES=2 DRAFT=1
case $L in v1) APU_FROM=5 REMOTE_APU=RPC1 $G/run_glm.sh g_$TAG 25 131072 -b 32768 9>&- & ;;
           v3) KM=$KM $G/run_glm_v3.sh g_$TAG 131072 -b 32768 -ub 1024 9>&- & ;; esac
for i in $(seq 1 900); do curl -s -m2 http://127.0.0.1:8081/health 2>/dev/null | grep -q '"status":"ok"' && break; sleep 1; done
echo "loaded ${i}s" > $G/greedy_$TAG.txt
P=$(python3 -c "print(('The ledger records that in each year the clerk noted the harvest, the tithe, the names of the newly born and the names of the dead. ' * 40) + 'Summarise the ledger in exactly three sentences.')")
python3 - "$P" "$G/greedy_$TAG.txt" <<'PY'
import sys,json,urllib.request
p,out=sys.argv[1],sys.argv[2]
req=urllib.request.Request('http://127.0.0.1:8081/completion',data=json.dumps({"prompt":p,"n_predict":160,"temperature":0,"top_k":1,"seed":1,"n_probs":0}).encode(),headers={'Content-Type':'application/json'})
d=json.loads(urllib.request.urlopen(req,timeout=1800).read())
t=d["timings"]
with open(out,'a') as f:
    f.write(f'prompt_n={t["prompt_n"]} predicted_n={t["predicted_n"]} draft_n={t.get("draft_n",0)} draft_acc={t.get("draft_n_accepted",0)}\n')
    f.write('---TEXT---\n'+d["content"]+'\n---END---\n')
print("greedy captured:", t["predicted_n"], "tokens")
PY
pgrep -x llama-server | while read p; do kill $p; done; sleep 3; pgrep -x llama-server | while read p; do kill -9 $p; done
echo GREEDY DONE >> $G/greedy_$TAG.txt
