#pragma once
#include "common.cuh"

// halo-hybrid: qwen4exp's sparse KQ mask (llama_model_qwen4exp::graph::build_attn_qsa) as one launch. The graph builds it
// as FILL(dup(kq_mask), -inf), FILL(zeros, 0), SET_ROWS(the filled mask viewed as one-element rows, zeros, top_k) and
// ADD_inplace(that, kq_mask): three full passes over [n_kv, n_tokens] halves plus the scattered zeros. The kernel
// marks each row's top-k cells in an LDS bitmap and writes (half) ((selected ? 0 : -inf) + (float) kq_mask) once,
// the ADD's arithmetic, so it is bit-identical. GGML_CUDA_NO_QSA_MASK=1 disables.

struct ggml_cuda_qsa_mask_args {
    const half *  mask;                 // kq_mask [n_kv, R, 1, S]
    half *        out;                  // same layout as mask
    int64_t       n_kv, R, S;
    int64_t       nb1, nb3;             // bytes, mask and out
    const char *  idx;                  // top-k cells of row (r, s) at idx + (r % idx_ne1)*idx_nb1 + (s % idx_ne2)*idx_nb2
    bool          idx_i64;
    int64_t       n_top;
    int64_t       idx_ne1, idx_ne2;
    int64_t       idx_nb1, idx_nb2;
};

// the largest n_kv the LDS bitmap holds
int64_t ggml_cuda_qsa_mask_max_kv();

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_mask_args & a);
