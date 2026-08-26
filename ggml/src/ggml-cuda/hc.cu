#include "hc.cuh"

#define HC_BLOCK 256

// The separate kernels these replace round every intermediate to f32; __fmul_rn/__fadd_rn keep the
// compiler from contracting products and sums into fma, so the fused results are bit-identical.

static __device__ __forceinline__ float hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __global__ void k_scale_silu(const float * x, float * dst, const float scale, const float bias, const int64_t n) {
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = scale * x[i] + bias;
        dst[i] = v / (1.0f + expf(-v));
    }
}

static __global__ void k_hc_mix(const float * xn, const float * g, float * dst,
        const int64_t n_embd, const int64_t hc, const int64_t nt, const float scale, const float bias) {
    const int64_t n      = n_embd * nt;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        const int64_t t = idx / n_embd;
        const int64_t i = idx - t * n_embd;
        const float * xr = xn + t * hc * n_embd + i;
        const float * gr = g  + t * hc * n_embd + i;
        float acc = __fmul_rn(xr[0], hc_sigmoid(gr[0]));
        for (int64_t c = 1; c < hc; ++c) {
            acc = __fadd_rn(acc, __fmul_rn(xr[c * n_embd], hc_sigmoid(gr[c * n_embd])));
        }
        dst[idx] = scale * acc + bias;
    }
}

static __global__ void k_hc_combine(const float * x, const float * b, const float * inj, float * dst,
        const int64_t n_embd, const int64_t hc, const int64_t nt,
        const float s1, const float b1, const float s2, const float b2) {
    const int64_t n      = n_embd * hc * nt;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        const int64_t t  = idx / (n_embd * hc);
        const int64_t r  = idx - t * n_embd * hc;
        const int64_t c  = r / n_embd;
        const int64_t i  = r - c * n_embd;
        const float w    = s2 * hc_sigmoid(s1 * inj[t * hc + c] + b1) + b2;
        dst[idx] = __fadd_rn(x[idx], __fmul_rn(b[t * n_embd + i], w));
    }
}

static int64_t hc_grid(int64_t n) {
    const int64_t blocks = (n + HC_BLOCK - 1) / HC_BLOCK;
    return blocks < 65535 ? blocks : 65535;
}

void ggml_cuda_op_scale_silu(ggml_backend_cuda_context & ctx, const ggml_tensor * scale_node, ggml_tensor * dst) {
    const ggml_tensor * src0 = scale_node->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));
    float scale, bias;
    memcpy(&scale, (const float *) scale_node->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) scale_node->op_params + 1, sizeof(float));
    const int64_t n = ggml_nelements(dst);
    k_scale_silu<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) src0->data, (float *) dst->data, scale, bias, n);
}

void ggml_cuda_op_hc_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * xn, const ggml_tensor * g,
        int64_t n_embd, int64_t hc, int64_t nt, float scale, float bias, ggml_tensor * dst) {
    GGML_ASSERT(xn->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(g) && ggml_is_contiguous(dst));
    const int64_t n = n_embd * nt;
    k_hc_mix<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) xn->data, (const float *) g->data, (float *) dst->data,
            n_embd, hc, nt, scale, bias);
}

void ggml_cuda_op_hc_combine(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * inj, int64_t n_embd, int64_t hc, int64_t nt,
        float s1, float b1, float s2, float b2, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && inj->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(b) && ggml_is_contiguous(inj) && ggml_is_contiguous(dst));
    const int64_t n = n_embd * hc * nt;
    k_hc_combine<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) x->data, (const float *) b->data, (const float *) inj->data,
            (float *) dst->data, n_embd, hc, nt, s1, b1, s2, b2);
}

static __global__ void k_mul_sigmoid(const float * x, const float * g, float * dst, const int64_t n, const int64_t ne0, const bool g_per_col) {
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float gv = g_per_col ? g[i / ne0] : g[i];
        dst[i] = __fmul_rn(x[i], hc_sigmoid(gv));
    }
}

static __global__ void k_weighted_sum(const float * e, const float * w, float * dst, const int64_t ne0, const int64_t m, const int64_t nt) {
    const int64_t n      = ne0 * nt;
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        const int64_t t = idx / ne0;
        const int64_t i = idx - t * ne0;
        const float * er = e + t * m * ne0 + i;
        const float * wr = w + t * m;
        float acc = __fmul_rn(er[0], wr[0]);
        for (int64_t c = 1; c < m; ++c) {
            acc = __fadd_rn(acc, __fmul_rn(er[c * ne0], wr[c]));
        }
        dst[idx] = acc;
    }
}

static __device__ __forceinline__ float hc_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));   // identical to unary.cu op_softplus
}

static __global__ void k_gdn_gate(const float * x, const float * b, const float * a, float * dst, const int64_t n, const int64_t ne0) {
    const int64_t stride = (int64_t) blockDim.x * gridDim.x;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int64_t r = i % ne0;
        dst[i] = __fmul_rn(hc_softplus(__fadd_rn(x[i], b[r])), a[r]);
    }
}

void ggml_cuda_op_mul_sigmoid(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * g, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && g->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(g) && ggml_is_contiguous(dst));
    const bool per_col = g->ne[0] == 1 && x->ne[0] != 1;
    const int64_t n = ggml_nelements(dst);
    k_mul_sigmoid<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) x->data, (const float *) g->data, (float *) dst->data, n, x->ne[0], per_col);
}

void ggml_cuda_op_weighted_sum(ggml_backend_cuda_context & ctx, const ggml_tensor * e, const ggml_tensor * w,
        int64_t ne0, int64_t m, int64_t nt, ggml_tensor * dst) {
    GGML_ASSERT(e->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(e) && ggml_is_contiguous(w) && ggml_is_contiguous(dst));
    const int64_t n = ne0 * nt;
    k_weighted_sum<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) e->data, (const float *) w->data, (float *) dst->data, ne0, m, nt);
}

void ggml_cuda_op_gdn_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * b,
        const ggml_tensor * a, ggml_tensor * dst) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && a->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(b) && ggml_is_contiguous(a) && ggml_is_contiguous(dst));
    const int64_t n = ggml_nelements(dst);
    k_gdn_gate<<<hc_grid(n), HC_BLOCK, 0, ctx.stream()>>>((const float *) x->data, (const float *) b->data, (const float *) a->data, (float *) dst->data, n, x->ne[0]);
}
