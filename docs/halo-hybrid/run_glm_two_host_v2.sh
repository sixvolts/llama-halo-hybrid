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
# Per-device buffers actually observed at RR=6, ctx 131072 (gibson's view, all four devices):
#   ROCm0 model 14551 MiB   ROCm1 model 83711 MiB   RPC0 model 26067 MiB   RPC1 model 61108 MiB
#   compute 1577 / 1835 / 1545 / 1826 MiB, KV 2464 MiB total, load 210 s
#   ROCm0 gibson R9700  30.5 GiB, host link Gen4 x4, 6.90 GB/s
#   ROCm1 gibson APU   108   GiB, on package
#   RPC0  mainframe R9700 32.6 GiB, host link Gen3 x4, 3.61 GB/s  <- half of gibson's; load is ~2x slower, expected
#   RPC1  mainframe APU 120   GiB, on package
#   inter-host 3.46 GB/s (Gen3 x4, does not retrain)
# usage: run_glm_v2.sh <tag> [LOCAL=25] [ctx=131072] [extra llama-server args...]
export PATH=/opt/rocm/bin:$PATH
TAG=${1:-glmv2}; LOCAL=${2:-25}; CTX=${3:-131072}
# a failed `shift 3` shifts NOTHING, so with fewer than three arguments the tag leaks into llama-server's argv
# (measured: `run_glm_two_host_v2.sh mytag` passed a stray positional "mytag" to the server). v1 has the same
# line and is only safe because chain_hb.sh always passes three.
shift $(( $# < 3 ? $# : 3 ))
# gibson-side paths: this script drives BOTH hosts from gibson. On mainframe the binaries are elsewhere
# (~/llama-halo-hybrid/build-rpc/bin), so override BIN if you ever run it from there.
BIN=${BIN:-/home/sixvolts/llama.cpp/build-hip/bin}
M=/home/sixvolts/models/glm-5.3-flash/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
RPC=10.100.100.2:50052
REMOTE=$((47 - LOCAL))                 # denominator 47 = n_layer_all + 1, as v1
# gibson: WHOLE_FROM whole layers on the R9700, the rest of the local layers whole on the APU
WHOLE_FROM=${WHOLE_FROM:-6}
# mainframe: RR whole layers on its R9700, the rest whole on its APU.
# MEASURED 2026-09-17, do not re-derive this by dividing the model size by the layer count (that was the original
# error here, and it was off by ~1.8x): the cost is the routed experts and nothing else. From the GGUF, per MoE
# layer, ffn_down_exps 1.568 + ffn_gate_exps 1.272 + ffn_up_exps 1.272 = **4.11 GiB of expert weights**, and the
# whole model is 186.0 GiB over 46 blocks of which 3 are leading dense. KV is negligible by comparison: 2.4 GiB
# TOTAL at ctx 131072 across every device (only 11 layers carry a cache, and it is K-only), so trading context
# for layers buys nothing -- RR is bounded by weights.
# Measured on mainframe's R9700 (31.86 GiB usable, NOT the 32624 MiB --list-devices advertises):
#   RR=12  asked 50.90 GiB  -> cudaMalloc failed, no spill, clean refusal
#   RR=6   27.59 GiB PEAK committed, 4.27 GiB spare   <- default, the operating point
#   RR=7   31.78 GiB peak against a 31.86 ceiling = 80 MiB of headroom: do not ship it
# SAMPLE THE PEAK, NOT THE STEADY STATE. Weight residency at RR=6 settles at 25.74 GiB, but graph warmup
# transiently adds ~1.85 GiB (measured on mainframe: peak 27.59, steady 25.74, gone after teardown). Sampling
# after load looks like 6.1 GiB spare when the run actually had 4.3. A two-point fit over RR=6 and the failed
# RR=12 gives slope 4.193 GiB/layer, intercept 0.58 GiB, which matches the GGUF's 4.11 GiB/layer of routed
# experts; the warmup 1.85 GiB is a separate, roughly fixed cost on top and is what rules RR=7 out. That margin
# is also what a draft head or a larger batch would eat, so keep it.
RR=${RR:-6}
[ "$RR" -gt "$REMOTE" ] && RR=$REMOTE
DEVS="ROCm0,ROCm1,RPC0,RPC1"
TS="$WHOLE_FROM,$((LOCAL-WHOLE_FROM)),$RR,$((REMOTE-RR))"
# ssm_a: the loader maps it to GGML_OP_SSM_SCAN, the HIP backend declines that shape, so it lands on the host and
# every split re-copies it with a device sync. Pin it next to its layer. PIN_SSM_A defaults to 0 to match v1's
# production behaviour, but v2 has four devices and therefore more splits than v1, so this is MORE likely to pay
# here, not less - worth trying 1 early. Explicit layer alternations rather than character-class ranges, because
# ranges like 2[0-4] are what make these regexes wrong when a boundary moves.
# -ot device names: plain RPCn in -dev, bracketed RPCn[endpoint] in -ot (ggml-rpc names devices at two sites and
# they disagree: line 1121 bracketed, line 2375 bare).
range_re() {   # range_re lo hi -> (lo|lo+1|...|hi), empty string if the range is empty
    local lo=$1 hi=$2 out="" i
    [ "$lo" -gt "$hi" ] && return 0
    for ((i=lo; i<=hi; i++)); do out="$out|$i"; done
    printf '(%s)' "${out#|}"
}
LAST=45          # layers 0..44 plus the MTP block at 45
SSMA=""
if [ "${PIN_SSM_A:-0}" = 1 ]; then
    for spec in "0:$((WHOLE_FROM-1)):ROCm0" "$WHOLE_FROM:$((LOCAL-1)):ROCm1" \
                "$LOCAL:$((LOCAL+RR-1)):RPC0[$RPC]" "$((LOCAL+RR)):$LAST:RPC1[$RPC]"; do
        lo=${spec%%:*}; rest=${spec#*:}; hi=${rest%%:*}; dev=${rest#*:}
        re=$(range_re "$lo" "$hi")
        [ -n "$re" ] && SSMA="${SSMA}blk\.${re}\.ssm_a=${dev},"
    done
fi
OT="${SSMA}^output\.weight$=ROCm0,^output_norm\.weight$=ROCm0,^token_embd\.weight$=CPU"
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
