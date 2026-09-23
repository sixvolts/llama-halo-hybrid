#pragma once

#include "common.cuh"

void ggml_cuda_op_kq_mask_build(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
