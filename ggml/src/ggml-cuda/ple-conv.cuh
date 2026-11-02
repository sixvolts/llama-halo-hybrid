#pragma once
#include "common.cuh"

// halo-hybrid: qwen4exp's PLE conv (llama_model_qwen4exp::graph's ple block) as one launch: a depthwise causal conv over
// the tokens of the padded input p = concat(history, x^T) ([hist + T tokens, C channels, n_seqs], tokens on ne[0]),
// each tap k reading p at token start[k] + t with one weight per channel, then SiLU, written token-major
// out[s][t][c]. The graph builds it as per tap CONT(TRANSPOSE(VIEW(p))) * w[k] (a full transposed copy per tap), a
// left-to-right ADD chain, SILU and a CONT; the kernel keeps their f32 arithmetic (each product rounded, then the adds
// in tap order, then x / (1 + expf(-x))), so it is bit-identical to that chain. GGML_CUDA_NO_PLE_CONV=1 disables.

#define PLE_CONV_MAX_TAPS 8

struct ggml_cuda_ple_conv_args {
    const float * p;                      // [L][C][S] tokens contiguous
    int64_t       L;                      // tokens per channel row of p (hist + T)
    int64_t       p_nb1, p_nb2;           // bytes
    int64_t       C, T, S;
    int           n_taps;
    int64_t       start[PLE_CONV_MAX_TAPS];   // first token tap k reads
    const char *  w[PLE_CONV_MAX_TAPS];       // weight of channel c at w[k] + c * w_nb[k]
    int64_t       w_nb[PLE_CONV_MAX_TAPS];
    float *       out;                    // [S][T][C] contiguous
};

void ggml_cuda_op_ple_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_args & a);
