#pragma once
#include "common.cuh"

// dense q8_0 / q4_K / q5_K / q6_K x f32 GEMM on RDNA WMMA with the weights dequantized to f16 once per K tile
// (mmq-wmma.cu); returns false when the shape or device is not covered and the caller falls through to MMQ
bool ggml_cuda_mul_mat_q_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
// the WMMA GEMM would take this call and convert src1 to f16 in a pre-pass (a producer may write that copy instead)
bool ggml_cuda_mul_mat_q_wmma_prepass(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
// ... or reads src1 through a producer-side f16 copy when one exists (pre-pass or the narrow tiles)
bool ggml_cuda_mul_mat_q_wmma_f16_input(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);

// qwen4exp's hc up projection fused with the hc mix (k_hc_mix) as the GEMM epilogue, bit-identical to the two launches:
// out[n][i] = scale * sum_c xn[n][c][i] * sigmoid((w x)[n][c*n_embd + i]) + bias; false when not covered (RDNA3, q8_0, hc 4)
// xs: xn is not in memory; the epilogue rebuilds it from the boundary's res, rms scales and gamma (bit-identical)
struct ggml_cuda_hc_xn_src {
    const float * res;
    const float * rstd;
    const float * gamma;
    int           gamma_rows;
};
bool ggml_cuda_mul_mat_q_wmma_hc_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * xn, int64_t n_embd, int64_t hc, float scale, float bias, ggml_tensor * out, float * g_debug = nullptr,
        const ggml_cuda_hc_xn_src * xs = nullptr, half * out16 = nullptr);

// qwen4exp's qkv projection with the GDN conv front (ggml_cuda_op_gdn_conv_front's conv + SiLU + l2 norm) as the GEMM
// epilogue, plus the boundary launch for the tokens each tile cannot see, bit-identical to the GEMM and the conv front;
// dst (x) is written only where the boundary kernel and the rollback slots read it. false when not covered (RDNA3, q8_0,
// the 128x128 pre-pass tile, prefill widths)
struct ggml_cuda_gdn_conv_front_args;
bool ggml_cuda_mul_mat_q_wmma_gdn_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
        ggml_tensor * dst, const ggml_cuda_gdn_conv_front_args & a);
