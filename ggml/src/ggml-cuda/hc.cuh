#pragma once

#include "common.cuh"

// Fused kernels for the hyper-connection (multi-stream residual) blocks of qwen4exp-style
// graphs. Each replaces a chain of tiny element-wise ops that would otherwise be one launch
// each on a [hc * n_embd, n_tokens] activation. Matched by op pattern in ggml_cuda_try_fuse.

// dst = silu(scale * x + bias)                                  (SCALE -> UNARY(SILU))
void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node, ggml_tensor * dst);

// dst[i, t] = scale * sum_c xn[c*n_embd + i, t] * sigmoid(g[c*n_embd + i, t]) + bias
//   (UNARY(SIGMOID) -> MUL -> ADD x (hc-1) over stream views -> SCALE); the sum runs in stream order
void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * xn, const ggml_tensor * g,
        int64_t n_embd, int64_t hc, int64_t nt, float scale, float bias, ggml_tensor * dst);

// dst[i, c, t] = x[i, c, t] + b[i, t] * (s2 * sigmoid(s1 * inj[c, t] + b1) + b2)
//   (REPEAT(b) ; SCALE -> UNARY(SIGMOID) -> SCALE on inj ; MUL ; ADD)
void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * inj, int64_t n_embd, int64_t hc, int64_t nt,
        float s1, float b1, float s2, float b2, ggml_tensor * dst);

// dst = x * sigmoid(g); g is either the same shape as x or a per-column scalar [1, ne1, ne2, ne3]
void ggml_cuda_op_mul_sigmoid(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * g, ggml_tensor * dst);

// dst[i, t] = sum_c e[i, c, t] * w[c, t]   (MUL(e [ne0, m, nt], w [1, m, nt]) -> ADD over the m expert views)
void ggml_cuda_op_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        int64_t ne0, int64_t m, int64_t nt, ggml_tensor * dst);

// dst = softplus(x + b[i % ne0]) * a[i % ne0]    (ADD -> UNARY(SOFTPLUS) -> MUL with per-row vectors)
void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * a, ggml_tensor * dst);

// q8_1 side-copy registry (see ggml_backend_cuda_context::q8_side)
void         ggml_cuda_q8_side_reset(ggml_backend_cuda_context & ctx);   // call at graph-compute start, before capture
const char * ggml_cuda_q8_side_find (ggml_backend_cuda_context & ctx, const ggml_tensor * src1);
