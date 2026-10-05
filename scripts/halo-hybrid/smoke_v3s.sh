#!/bin/bash
# Local end-to-end smoke test of the V3 composite path: gibson's own rpc-server (both local devices) as a fake node.
# 1) reference: plain local run on ROCm0.  2) composite: -dev RPC0 with layers 10-27's FFN on the extra buft (APU).
# Greedy text must be identical; the server's GGML_SCHED_DEBUG=2 dump shows the split/placement.
export PATH=/opt/rocm/bin:$PATH
B=/home/sixvolts/llama.cpp/build-hip/bin; M=/home/sixvolts/models/drafters/Qwen3.5-0.8B-Q8_0.gguf
S=/tmp/claude-1000/-home-sixvolts/14be11ce-3917-4928-9848-657e1a4c0fbb/scratchpad/smoke
mkdir -p $S; cd $S
P="The quick brown fox jumps over the lazy dog. In the beginning, the universe was"
ARGS="-m $M -n 48 --temp 0 --top-k 1 -c 2048 -no-cnv -p"
echo "=== reference (local ROCm0) ==="
GGML_RPC_NO_RDMA=1 $B/llama-completion $ARGS "$P" -dev ROCm0 -ngl 99 2>ref.err | tail -c 700 | tee ref.txt
echo; echo "=== fake node: rpc-server on 127.0.0.1:50053 ==="
env GGML_RPC_NO_RDMA=1 ${SERVER_ENV} $B/ggml-rpc-server -H 127.0.0.1 -p 50053 > server.log 2>&1 &
SP=$!; sleep 4; grep -aE 'Starting|device|backend' server.log | head -5
echo "=== composite client: -dev RPC0, FFN of layers 10-27 on RPC1[...] (the extra buft) ==="
env GGML_RPC_NO_RDMA=1 GGML_RPC_COMPOSITE=1 ${CLIENT_ENV} $B/llama-completion $ARGS "$P" --rpc 127.0.0.1:50053 -dev RPC0 -ngl 99 \
  -ot 'blk\.(1[0-9]|2[0-7])\.ffn_.*=RPC1[127.0.0.1:50053]' 2>comp.err | tail -c 700 | tee comp.txt
echo; echo "=== client log: composite + buffer placement lines ==="
grep -aE 'composite|model buffer size|RPC[01]\[' comp.err | head -8
echo "=== server log: scheduler + split summary ==="
grep -aE 'server-side scheduler|## SPLIT' server.log | head -6; echo "splits logged: $(grep -ac '## SPLIT' server.log)"
kill $SP 2>/dev/null; wait $SP 2>/dev/null
echo "=== token identity ==="
if cmp -s <(tail -c 600 ref.txt) <(tail -c 600 comp.txt); then echo "SMOKE PASS: identical text"; else echo "SMOKE DIFF"; diff <(tail -c 600 ref.txt) <(tail -c 600 comp.txt) | head -10; fi
grep -aE 'error|abort|Assert|failed' comp.err server.log | head -5
