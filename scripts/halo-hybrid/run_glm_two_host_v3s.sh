#!/bin/bash
# V3 (server-side scheduling): mainframe is ONE composite device RPC0 to gibson's scheduler. Its R9700 (server
# device 0 = the composite's default buffer type) holds KV and the dense parts of every remote layer plus the
# experts of KM layers; its APU (extra buffer type RPC1[endpoint]) holds only routed experts. The rpc-server
# runs its own ggml_backend_sched across both. See docs/halo-hybrid/V3-SERVER-SCHED.md.
export PATH=/opt/rocm/bin:$PATH
export GGML_RPC_NO_RDMA=${GGML_RPC_NO_RDMA-1}
export GGML_RPC_COMPOSITE=1
TAG=${1:-glmv3s}; CTX=${2:-131072}; shift 2 2>/dev/null; shift $(( $# < 0 ? 0 : 0 ))
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
KM=${KM:-4}                       # expert layers 25..(24+KM) stay on the card; experts of the rest go to the APU
MF_APU_FROM=$((25+KM))
MF_APU="($(seq $MF_APU_FROM 45 | paste -sd'|'))"
DEVS="ROCm0,ROCm1,RPC0"
TS="25,0,22"                       # 0-24 gibson; 25-46 mainframe (one composite device)
GIBSON_APU="([5-9]|1[0-9]|2[0-4])"
OT="blk\.${GIBSON_APU}\.ffn_(gate|up|down)_exps=ROCm1,blk\.${MF_APU}\.ffn_(gate|up|down)_exps=RPC1[$RPC],^output\.weight\$=ROCm0,^output_norm\.weight\$=ROCm0,^token_embd\.weight\$=CPU"
LOG=/home/sixvolts/bench/glm/$TAG.log
echo "layout v3s: gibson R9700 0-24(exp 0-4) | gibson APU exp 5-24 | mainframe [server sched] card: KV+dense 25-46 + exp 25-$((24+KM)) | APU exp ${MF_APU_FROM}-45" | tee $LOG
echo "  dev $DEVS ts $TS  KM=$KM  MF_APU=$MF_APU  ctx $CTX  composite=1" | tee -a $LOG
/home/sixvolts/bench/drain_pool.sh | tee -a $LOG
DRAFTARGS=""
if [ "${DRAFT:-0}" = 1 ]; then DRAFTARGS="-md /home/sixvolts/models/glm-5.3-flash/MTP/GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2}"; fi
exec env GGML_RPC_DEBUG=1 LLAMA_ASYNC_INPUTS=${LLAMA_ASYNC_INPUTS:-1} $BIN/llama-server -m "$M" --rpc $RPC $DRAFTARGS \
  -dev "$DEVS" -ts $TS --fit off -fa on -ngl 999 \
  -c $CTX -b 4096 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot "$OT" \
  --alias glm-5.3-flash --metrics --host 0.0.0.0 --port 8081 -lv 4 "$@" >> $LOG 2>&1
