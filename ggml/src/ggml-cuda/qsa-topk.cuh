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
