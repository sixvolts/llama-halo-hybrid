#!/bin/bash
# GLM-5.3-Flash layout v2c (CLEAN v2): gibson identical to v1/v3; mainframe holds WHOLE layers, no expert split.
# Derived from v3 - the ONLY variable vs v3 is mainframe placement (whole-layer -ts vs dense/expert -ot split).
# was: layout v3: mirror gibson's dense-on-fast/experts-on-slow split onto mainframe too,
# now that mainframe has an R9700. gibson is IDENTICAL to production v1 (single variable vs v1).
#
# Per layer: dense/attn/KV ~170 MiB (read every token), experts ~4176 MiB (sparse, 8/512 active). So the
# per-token hot state is tiny and belongs on the fast card; experts are big and belong on the APU except for as
# many as fill the leftover VRAM.
#
#   gibson R9700 (ROCm0): dense/attn/KV of 0-24 + experts of 0-4
#   gibson APU   (ROCm1): experts of 5-24
#   mainframe R9700 (RPC0): dense/attn/KV of 25-44 (all, ~3.4 GiB) + experts of 25..(24+KM)
#   mainframe APU   (RPC1): experts of the rest of 25-44
#
# KM=5: R9700 holds ~3.4 (dense) + 5*4.08 (experts) = ~23.7 GiB weights, ~25.7 steady, ~27.6 peak vs 31.86 usable.
# This does NOT add a crossing vs v1 - v1 already crosses to mainframe once; v3 keeps that and only swaps the
# remote dense from APU (~225 GB/s) to R9700 (~3x). Decode reads resident weights from R9700 VRAM, so the Gen3 x4
# host link is not in the decode path. UNTESTED until this loads and the A/B runs.
# usage: run_glm_v3.sh <tag> [ctx=131072] [extra args...]
export PATH=/opt/rocm/bin:$PATH
TAG=${1:-glmv3}; CTX=${2:-131072}; shift 2 2>/dev/null; shift $(( $# < 0 ? 0 : 0 ))
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
RR=${RR:-6}                       # WHOLE layers 25..(24+RR) on mainframe R9700; the rest whole on its APU
MF_FIRST_APU=$((25+RR))           # first whole layer on the mainframe APU
DEVS="ROCm0,ROCm1,RPC0,RPC1"
TS="25,0,$RR,$((22-RR))"           # 0-24 gibson R9700; 25..24+RR mf R9700 WHOLE; 25+RR..46 mf APU WHOLE (MTP blk45 lands on APU)
# gibson experts 5-24 -> gibson APU (production APU_FROM=5)
GIBSON_APU="([5-9]|1[0-9]|2[0-4])"
OT="blk\.${GIBSON_APU}\.ffn_(gate|up|down)_exps=ROCm1,^output\.weight\$=ROCm0,^output_norm\.weight\$=ROCm0,^token_embd\.weight\$=CPU"
LOG=/home/sixvolts/bench/glm/$TAG.log
echo "layout v2c: gibson R9700 0-24(exp 0-4) | gibson APU exp 5-24 | mainframe R9700 WHOLE 25-$((24+RR)) | mainframe APU WHOLE ${MF_FIRST_APU}-46" | tee $LOG
echo "  dev $DEVS ts $TS  RR=$RR  ctx $CTX" | tee -a $LOG
/home/sixvolts/bench/drain_pool.sh | tee -a $LOG
DRAFTARGS=""
if [ "${DRAFT:-0}" = 1 ]; then DRAFTARGS="-md /home/sixvolts/models/glm-5.3-flash/MTP/GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2}"; fi
MCPARGS=""
exec env GGML_RPC_DEBUG=1 LLAMA_ASYNC_INPUTS=${LLAMA_ASYNC_INPUTS:-1} $BIN/llama-server -m "$M" --rpc $RPC $DRAFTARGS $MCPARGS \
  -dev "$DEVS" -ts $TS --fit off -fa on -ngl 999 \
  -c $CTX -b 4096 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot "$OT" \
  --alias glm-5.3-flash --metrics --host 0.0.0.0 --port 8081 -lv 4 "$@" >> $LOG 2>&1
