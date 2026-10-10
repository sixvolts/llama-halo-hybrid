# gpu-rdna3 extra steps (sourced by suites/gpu.sh, card = ROCm0). RDNA3 (gfx1101, RX 7800 XT) runs three fast paths
# by default that RDNA4 does not, or not for the same types:
#   - q8_0 matmuls on f16 WMMA (mmq-wmma.cu, shared with gfx1151; RDNA4 keeps MMQ)   GGML_CUDA_Q8_WMMA=0 turns it off
#   - k-quant matmuls on WMMA                                                       GGML_CUDA_KQ_WMMA=0
#   - Qwen3.8's D=256 sparse flash-attention walk for prefill-width batches         GGML_CUDA_FA_SPARSE_D256=0
# ops-card checks those paths against the CPU. This step runs the same matmul, MoE and attention cases with all three
# off, so the MMQ and dense fallbacks stay correct on this card as well.
GEN_FALLBACK_OPS=MUL_MAT,MUL_MAT_ID,MUL_MAT_ID_FUSION,MUL_MAT_VEC_FUSION,MOE_BLOCK,MOE_GATE_UP,MOE_TAIL,FLASH_ATTN_EXT

gen_steps() {
    step ops-card-fallback run_ops ROCm0 "$GEN_FALLBACK_OPS" TBO_Q38_EVAL=1 TBO_GLM_EVAL=1 \
        GGML_CUDA_Q8_WMMA=0 GGML_CUDA_KQ_WMMA=0 GGML_CUDA_FA_SPARSE_D256=0
}
