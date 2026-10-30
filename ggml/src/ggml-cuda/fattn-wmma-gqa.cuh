#pragma once

#include "common.cuh"

// halo-hybrid: Qwen3.8 QSA prefill attention (D = 256, GQA 12, per-cell selection mask) on RDNA3 WMMA, 4 queries x 12
// heads per block; returns false when the call does not fit (the caller then takes the generic kernels)
bool ggml_cuda_flash_attn_ext_wmma_gqa12(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
