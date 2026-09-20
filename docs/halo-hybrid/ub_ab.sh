#!/bin/bash
# Single-variable -ub test for v3 prefill sync cost. KM=4 both runs (KM=5 + -ub 2048 + 2 lanes exceeds 32.6 GiB).
G=/home/sixvolts/bench/glm
KM=4 UB=1024 SUFFIX=_km4ub1024 $G/phase2_ab.sh v3
KM=4 UB=2048 SUFFIX=_km4ub2048 $G/phase2_ab.sh v3
echo UB_AB DONE >> $G/ub_ab.done
