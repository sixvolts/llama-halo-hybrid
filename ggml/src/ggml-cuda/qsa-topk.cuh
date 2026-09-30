#pragma once

#include "common.cuh"

// halo-hybrid: GGML_OP_QSA_TOP_K (see ggml.h)
void ggml_cuda_op_qsa_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// halo-hybrid: GGML_OP_QSA_HEAD_SUM (see ggml.h)
void ggml_cuda_op_qsa_head_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
