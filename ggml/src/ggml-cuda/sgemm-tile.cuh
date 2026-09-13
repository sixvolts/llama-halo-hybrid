#pragma once

#include "common.cuh"

// halo-hybrid: f32 x f32 GEMM with a small number of output rows (the MoE router gate, ffn_gate_inp [K x 288]) at
//     prefill widths. On RDNA rocBLAS picks a 32x32x8 macro tile for this shape and runs it at ~2.6 TFLOPS on
//     gfx1151 (0.94 ms per 1024-token ubatch and layer for K = 4096); a plain LDS-tiled FMA kernel keeps the exact
//     f32 arithmetic and runs several times faster. Returns false if the shape is not handled (caller falls back).
bool ggml_cuda_mul_mat_f32_tile(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
