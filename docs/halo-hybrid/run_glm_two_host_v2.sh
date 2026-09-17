#!/bin/bash
# GLM-5.3-Flash UD-Q4_K_XL across gibson + mainframe, layout v2: FOUR devices, now that mainframe has an R9700.
# UNTESTED - written 2026-09-17 before either host ran it. v1 (run_glm.sh) is untouched and remains production.
#
# Why this is not "mirror gibson on the far side":
#   The obvious v2 is dense trunk + KV on each R9700 with routed experts on each APU. That is wrong here. Per
#   token the activations are tens of KB, so bytes are irrelevant; what costs is the NUMBER of device crossings,
#   each of which is a PCIe round trip (and across hosts, an RPC serialization). Splitting every remote layer
#   into "dense here, experts there" would add ~2 crossings x 21 layers = ~42 per token on the remote side.
#   So the remote side gets WHOLE layers per device, exactly the way WHOLE_FROM already collapses gibson's side.
# Crossings per token, by construction: ROCm0 -> ROCm1 -> RPC0 -> RPC1, three, plus the output-head pull-back.
#
# Device notes measured 2026-09-17 (see memory: mainframe-r9700, two-host-link-ceiling):
#   ROCm0 gibson R9700  30.5 GiB, host link Gen4 x4, 6.90 GB/s
#   ROCm1 gibson APU   108   GiB, on package
#   RPC0  mainframe R9700 32.6 GiB, host link Gen3 x4, 3.61 GB/s  <- half of gibson's; load is ~2x slower, expected
#   RPC1  mainframe APU 120   GiB, on package
#   inter-host 3.46 GB/s (Gen3 x4, does not retrain)
# usage: run_glm_v2.sh <tag> [LOCAL=25] [ctx=131072] [extra llama-server args...]
export PATH=/opt/rocm/bin:$PATH
TAG=${1:-glmv2}; LOCAL=${2:-25}; CTX=${3:-131072}; shift 3 2>/dev/null
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
REMOTE=$((47 - LOCAL))                 # denominator 47 = n_layer_all + 1, as v1
# gibson: WHOLE_FROM whole layers on the R9700, the rest of the local layers whole on the APU
WHOLE_FROM=${WHOLE_FROM:-6}
# mainframe: RR whole layers on its R9700, the rest whole on its APU. RR is a GUESS pending the first run --
# GLM averages ~2.4 GB/layer, so 12 layers is ~29 GB against 32.6 GiB, before this layer range's KV. Read the
# per-device buffer sizes llama-server prints on load and retune; too high fails to allocate, too low wastes the card.
RR=${RR:-12}
[ "$RR" -gt "$REMOTE" ] && RR=$REMOTE
DEVS="ROCm0,ROCm1,RPC0,RPC1"
TS="$WHOLE_FROM,$((LOCAL-WHOLE_FROM)),$RR,$((REMOTE-RR))"
# ssm_a: the loader maps it to GGML_OP_SSM_SCAN, the HIP backend declines that shape, so it lands on the host and
# every split re-copies it with a device sync. Pin it next to its layer (v1 keeps this behind PIN_SSM_A=1).
# -ot device names: plain RPCn in -dev, bracketed RPCn[endpoint] in -ot (ggml-rpc names devices at two sites and
# they disagree: line 1121 bracketed, line 2375 bare).
OT="^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU"
LOG=/home/sixvolts/bench/glm/$TAG.log
echo "layout v2: gibson 0..$((WHOLE_FROM-1)) R9700 | $WHOLE_FROM..$((LOCAL-1)) APU | mainframe $LOCAL..$((LOCAL+RR-1)) R9700 | $((LOCAL+RR))..45 APU" | tee $LOG
echo "  dev $DEVS  ts $TS  ctx $CTX" | tee -a $LOG
/home/sixvolts/bench/drain_pool.sh | tee -a $LOG
MCPARGS=""
if [ "${MCP:-1}" = 1 ] && [ -f /home/sixvolts/llama-tools/mcp-servers.json ]; then
  MCPARGS="--mcp-servers-config /home/sixvolts/llama-tools/mcp-servers.json"
fi
DRAFTARGS=""
if [ "${DRAFT:-0}" = 1 ]; then DRAFTARGS="-md /home/sixvolts/models/glm-5.3-flash/MTP/GLM-5.3-Flash-mtp-UD-Q4_K_XL.gguf -devd ROCm0 -ngld 999 --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2}"; fi
exec env GGML_RPC_DEBUG=1 LLAMA_ASYNC_INPUTS=${LLAMA_ASYNC_INPUTS:-1} $BIN/llama-server -m "$M" --rpc $RPC $DRAFTARGS $MCPARGS \
  -dev "$DEVS" -ts $TS --fit off -fa on -ngl 999 \
  -c $CTX -b 4096 -ub 1024 --load-mode none -np 1 -t 16 \
  -ot "$OT" \
  --alias glm-5.3-flash --metrics --host 0.0.0.0 --port 8081 -lv 4 "$@" >> $LOG 2>&1
