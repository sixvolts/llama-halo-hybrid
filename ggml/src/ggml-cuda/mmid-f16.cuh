#pragma once
#include "common.cuh"

// halo-hybrid: F16-WMMA routed expert GEMM (adapted from gufo, MIT) - see mmid-f16.cu. Returns false when not taken.
bool ggml_cuda_mmid_f16_enabled();
// would ggml_cuda_mmid_f16 take this MUL_MAT_ID node (no launch)
bool ggml_cuda_mmid_f16_takes(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
bool ggml_cuda_mmid_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                        const ggml_tensor * ids, ggml_tensor * dst);
