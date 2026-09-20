#!/bin/bash
# GLM-5.3-Flash two-host layout v3: mirror gibson's dense-on-fast/experts-on-slow split onto mainframe too,
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
# 2026-09-20: TCP by default. Mainframe's E810 RoCE engine faults (LCE_QP_CATASTROPHIC) under RDMA; TCP measured equal to RDMA
# on v1 and v3 (health-gated 3-rep A/B). Set GGML_RPC_NO_RDMA= (empty) to try RDMA again.
export GGML_RPC_NO_RDMA=${GGML_RPC_NO_RDMA-1}
TAG=${1:-glmv3}; CTX=${2:-131072}; shift 2 2>/dev/null; shift $(( $# < 0 ? 0 : 0 ))
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
KM=${KM:-5}                       # mainframe layers 25..(24+KM) keep experts on its R9700
MF_APU_FROM=$((25+KM))            # experts of MF_APU_FROM..45 go to mainframe APU (45 = MTP block: dead weight in the
                                 # main forward pass, so its experts must NOT sit on the R9700 - that was a 4 GiB leak)
DEVS="ROCm0,ROCm1,RPC0,RPC1"
TS="25,0,22,0"                    # 0-24 -> gibson R9700; 25-46 -> mainframe R9700; APUs get experts via -ot only
# gibson experts 5-24 -> gibson APU (production APU_FROM=5)
GIBSON_APU="([5-9]|1[0-9]|2[0-4])"
# mainframe experts MF_APU_FROM..45 -> mainframe APU (RPC1). KM=5 => 30..45. MUST include 45 or the MTP block's
# experts stay resident on the R9700 as dead weight (~4 GiB).
# explicit alternation - correct for ANY start layer 25..45. The previous bracket-range builder only worked for
# a start of 30..39: at KM=4 (start 29) it produced "3[-1-9]" which silently kept experts of 29 AND 30 on the
# R9700 (6 layers, not 4) and OOMed - and with 4 GiB more VRAM it would have run mislabelled. Caught by the split check.
MF_APU="($(seq $MF_APU_FROM 45 | paste -sd'|'))"
OT="blk\.${GIBSON_APU}\.ffn_(gate|up|down)_exps=ROCm1,blk\.${MF_APU}\.ffn_(gate|up|down)_exps=RPC1[$RPC],^output\.weight\$=ROCm0,^output_norm\.weight\$=ROCm0,^token_embd\.weight\$=CPU"
LOG=/home/sixvolts/bench/glm/$TAG.log
echo "layout v3: gibson R9700 0-24(exp 0-4) | gibson APU exp 5-24 | mainframe R9700 dense 25-45 + exp 25-$((24+KM)) | mainframe APU exp ${MF_APU_FROM}-45" | tee $LOG
echo "  dev $DEVS ts $TS  KM=$KM  MF_APU=$MF_APU  ctx $CTX" | tee -a $LOG
/home/sixvolts/bench/drain_pool.sh | tee -a $LOG
DRAFTARGS=""
if [ "${DRAFT:-0}" = 1 ]; then DRAFTARGS="-md /home/sixvolts/models/glm-5.3-flash/MTP/GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2}"; fi
MCPARGS=""
exec env GGML_RPC_DEBUG=1 LLAMA_ASYNC_INPUTS=${LLAMA_ASYNC_INPUTS:-1} $BIN/llama-server -m "$M" --rpc $RPC $DRAFTARGS $MCPARGS \
  -dev "$DEVS" -ts $TS --fit off -fa on -ngl 999 \
  -c $CTX -b 4096 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot "$OT" \
  --alias glm-5.3-flash --metrics --host 0.0.0.0 --port 8081 -lv 4 "$@" >> $LOG 2>&1
