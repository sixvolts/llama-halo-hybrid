#!/bin/bash
# GLM-5.3-Flash UD-Q4_K_XL across gibson + mainframe (RPC over the 100G E810 link, RDMA auto-negotiated).
# Layout v1: layers 0..LOCAL-1 on gibson (dense trunk + KV on the R9700, routed experts of layers R9700_EXPS on the
# R9700 and the rest on the APU), layers LOCAL..44 (+ the MTP block 45) on mainframe's APU, output head on the R9700.
# usage: run_glm.sh <tag> [LOCAL=25] [ctx=131072] [extra llama-server args...]
export PATH=/opt/rocm/bin:$PATH
TAG=${1:-glm}; LOCAL=${2:-25}; CTX=${3:-131072}; shift 3 2>/dev/null
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
# layer il goes to the first device whose cumulative split exceeds il/(n_layer_all+1) = il/47, so a
# denominator of 47 puts exactly layers 0..LOCAL-1 here (output lands remote and is pulled back by -ot)
REMOTE=$((47 - LOCAL))
# experts of layers 3..5 stay on the R9700 (3 x 4.4 GB), 6..LOCAL-1 go to the APU; the MTP draft block gets the rest
APU_LAYERS="([6-9]|1[0-9]|2[0-$((LOCAL-1 > 29 ? 9 : LOCAL-1-20))])"
if [ "$LOCAL" -le 20 ]; then APU_LAYERS="([6-9]|1[0-$((LOCAL-1-10))])"; fi
# ssm_a (256-byte KDA decay per layer) is mapped to GGML_OP_SSM_SCAN by the loader's support check, which the HIP
# backend declines for this shape, so it lands on the host and every split re-copies it with a device sync
# (143 synchronous copies per token); pin it next to its layer
LOCAL_LAYERS="([0-9]|1[0-9]|2[0-$((LOCAL-1-20))])"
if [ "$LOCAL" -le 20 ]; then LOCAL_LAYERS="([0-9]|1[0-$((LOCAL-1-10))])"; fi
REMOTE_LAYERS="(2[$((LOCAL-20))-9]|[34][0-9])"
SSMA=""
if [ "${PIN_SSM_A:-0}" = 1 ]; then SSMA="blk\.${LOCAL_LAYERS}\.ssm_a=ROCm0,blk\.${REMOTE_LAYERS}\.ssm_a=RPC0[$RPC],"; fi
OT="blk\.${APU_LAYERS}\.ffn_(gate|up|down)_exps=ROCm1,${SSMA}^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU"
LOG=/home/sixvolts/bench/glm/$TAG.log
echo "layout: local layers 0..$((LOCAL-1)) (ts $LOCAL,$REMOTE,0), APU experts $APU_LAYERS, ctx $CTX" | tee $LOG
/home/sixvolts/bench/drain_pool.sh | tee -a $LOG
# DRAFT=1: the exported MTP block as the draft head, on the R9700 (n-max 2 measured best)
DRAFTARGS=""
if [ "${DRAFT:-0}" = 1 ]; then DRAFTARGS="-md /home/sixvolts/models/glm-5.3-flash/MTP/GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2}"; fi
exec env GGML_RPC_DEBUG=1 LLAMA_ASYNC_INPUTS=${LLAMA_ASYNC_INPUTS:-1} $BIN/llama-server -m "$M" --rpc $RPC $DRAFTARGS \
  -dev "ROCm0,RPC0,ROCm1" -ts $LOCAL,$REMOTE,0 --fit off -fa on -ngl 999 \
  -c $CTX -b 4096 -ub 1024 --no-mmap -np 1 -t 16 \
  -ot "$OT" \
  --alias glm-5.3-flash --metrics --host 0.0.0.0 --port 8081 -lv 4 "$@" >> $LOG 2>&1
