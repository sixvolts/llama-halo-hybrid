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

// res = hc_combine(x, b, inj) as above, then xn = rms_norm(res rows of n_embd, eps) * gamma[n_embd, 1 or hc] with a
//   q8_1 side copy of xn  (hc_combine ; RMS_NORM(res) -> MUL(gamma)); bit-identical to the separate launches
void ggml_cuda_op_hc_combine_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * inj, const ggml_tensor * gamma, int64_t n_embd, int64_t hc, int64_t nt,
        float s1, float b1, float s2, float b2, float eps, ggml_tensor * res, ggml_tensor * xn);
// the same with the next mix's inject projection (w_inj [hc*n_embd, hc] f32) folded in at prefill widths: inj_out =
//   w_inj^T xn per token, bit-identical to the swapped MMVF GEMV; false when the call does not fit (nothing launched)
bool ggml_cuda_op_hc_combine_norm_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * inj, const ggml_tensor * gamma, const ggml_tensor * w_inj, int64_t n_embd, int64_t hc, int64_t nt,
        float s1, float b1, float s2, float b2, float eps, ggml_tensor * res, ggml_tensor * xn, ggml_tensor * inj_out,
        half * xn16 = nullptr, float * rstd = nullptr);
// the same boundary on a MoE layer, with the block output built in the kernel: the routed experts' weighted sum
// (experts [n_embd, n_used, nt] * weights [n_used, nt], optional expert_scale) plus shexp * sigmoid(sgate [nt]).
// false when the shape is not supported (nothing launched). GGML_CUDA_NO_HC_MOE_FUSE: 1 = all of it, 2 = the kernel
// only (graph_optimize's reorder and alloc deps stay: the baseline with the same allocation for an exactness check)
bool ggml_cuda_op_hc_moe_combine_norm_inject(ggml_backend_cuda_context & ctx, const ggml_tensor * experts,
        const ggml_tensor * expert_scale, const ggml_tensor * weights, const ggml_tensor * shexp, const ggml_tensor * sgate,
        const ggml_tensor * x, const ggml_tensor * inj, const ggml_tensor * gamma, const ggml_tensor * w_inj, int64_t n_embd,
        int64_t hc, int64_t nt, float s1, float b1, float s2, float b2, float eps, ggml_tensor * res, ggml_tensor * xn,
        ggml_tensor * inj_out, half * xn16 = nullptr, float * rstd = nullptr);
// (rstd != nullptr: xn is not written; the kernel stores the per-stream rms scales [nt, hc] instead)
bool ggml_cuda_hc_moe_fuse_disabled(int level);
// xn = (rstd * res) * gamma as the boundary kernel computes it, for a boundary that stored rstd in place of xn
void ggml_cuda_op_hc_xn_materialize(ggml_backend_cuda_context & ctx, const float * res, const float * rstd, const float * gamma,
        int gamma_rows, int64_t n_embd, int64_t hc, int64_t nt, float * xn);
// producer-side f16 copy of `prod` for the next q8_0 / k-quant WMMA GEMM's pre-pass; nullptr when not available
half *       ggml_cuda_f16_side_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * prod);
const half * ggml_cuda_f16_side_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1);
// RMS_NORM(x, 128-wide rows) -> MUL(w) -> MUL(., sigmoid(z)) into dst, and its f16 copy into d16 when not null
void ggml_cuda_op_gdn_out_norm_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        const ggml_tensor * z, float eps, ggml_tensor * dst, half * d16);
bool ggml_cuda_hc_boundary_disabled(int level);   // GGML_CUDA_NO_HC_BOUNDARY: 1 = all of it, 2 = the fused kernel only

// dst = x * sigmoid(g) (+ y); g is either the same shape as x or a per-column scalar [1, ne1, ne2, ne3];
// y, when given, is the same shape as x (the residual add that follows the shared-expert gate)
void ggml_cuda_op_mul_sigmoid(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * g, const ggml_tensor * y, ggml_tensor * dst);

// dst[i, t] = sum_c e[i, c, t] * w[c, t]   (MUL(e [ne0, m, nt], w [1, m, nt]) -> ADD over the m expert views)
void ggml_cuda_op_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        int64_t ne0, int64_t m, int64_t nt, ggml_tensor * dst);

// dst = softplus(x + b[i % ne0]) * a[i % ne0]    (ADD -> UNARY(SOFTPLUS) -> MUL with per-row vectors)
void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * a, ggml_tensor * dst);

// q8_1 side-copy registry (see ggml_backend_cuda_context::q8_side)
void         ggml_cuda_q8_side_reset(ggml_backend_cuda_context & ctx);   // call at graph-compute start, before capture
const char * ggml_cuda_q8_side_find (ggml_backend_cuda_context & ctx, const ggml_tensor * src1);

// reserve a q8_1 side buffer for `dst` (a single row of n elements); nullptr if not applicable
block_q8_1 * ggml_cuda_q8_side_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, int64_t n);
// consumer-side variant: `t` is the f32 activation a GEMV is about to quantize itself ([ne0, ne1], ne1 <= 8, rows
// padded to ne0_padded); the next GEMV on the same tensor finds the copy instead of re-quantizing it
block_q8_1 * ggml_cuda_q8_side_reserve_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne0_padded);

// quantise the value this thread just produced, exactly as quantize_q8_1 does: 32 consecutive
// elements form a block; requires n % 32 == 0 and every lane of the 32-lane group active
static __device__ __forceinline__ void q8_side_store(block_q8_1 * q8, const int64_t idx, const float v) {
    float amax = fabsf(v);
    amax = warp_reduce_max<QK8_1>(amax);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(v / d);
    const int sumq = warp_reduce_sum<QK8_1>((int) q);   // s = d*sum(q), as quantize_q8_1 (quantize.cu)
    const int64_t ib  = idx / QK8_1;
    const int     iqs = idx % QK8_1;
    q8[ib].qs[iqs] = q;
    if (iqs == 0) {
        q8[ib].ds = make_half2(d, d*sumq);
    }
}


// halo-hybrid: producer-side MMQ q8_1 copies of prefill activations (one slot per context, reset per graph)
struct block_q8_1_mmq;
block_q8_1_mmq * ggml_cuda_mmq_side_reserve(ggml_backend_cuda_context & ctx, const ggml_tensor * prod, int64_t ne0, int64_t ne1);
const char * ggml_cuda_mmq_side_find(ggml_backend_cuda_context & ctx, const ggml_tensor * src1, ggml_type type_src0);
