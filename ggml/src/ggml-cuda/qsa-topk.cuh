#pragma once

#include "common.cuh"

// halo-hybrid: GGML_OP_QSA_TOP_K (see ggml.h)
void ggml_cuda_op_qsa_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// halo-hybrid: GGML_OP_QSA_HEAD_SUM (see ggml.h)
void ggml_cuda_op_qsa_head_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// halo-hybrid: MUL_MAT(f32 block keys [128, n_blocks, ns], f32 queries [128, NC, ns]) -> QSA_HEAD_SUM in one pass,
// bit-identical to the two ops (see qsa-topk.cu). mm: the MUL_MAT node, hs: the QSA_HEAD_SUM over its reshaped
// result. Returns false (nothing launched) for shapes it does not cover.
bool ggml_cuda_qsa_score_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * hs);

// halo-hybrid: the same pair at prefill widths (at least 16 queries per stream), f32 with a different summation order
// than the unfused MUL_MAT (see qsa-topk.cu). GGML_CUDA_NO_QSA_SCORE_WIDE=1 disables it.
// out: where to write the head sum in hs's (contiguous) layout, nullptr = hs itself
bool ggml_cuda_qsa_score_fused_wide(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, ggml_tensor * hs, float * out = nullptr);
