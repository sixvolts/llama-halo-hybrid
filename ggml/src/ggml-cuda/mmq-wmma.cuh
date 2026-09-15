#pragma once
#include "common.cuh"

// dense q8_0 x f32 GEMM on RDNA WMMA with the weights dequantized to f16 once per K tile (mmq-wmma.cu);
// returns false when the shape or device is not covered and the caller falls through to MMQ
bool ggml_cuda_mul_mat_q8_0_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
