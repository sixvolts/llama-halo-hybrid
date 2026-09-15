#include "ewchain.cuh"

// halo-hybrid: one launch for a chain of tiny element-wise ops (see ewchain.cuh). The chain is evaluated per
// element in graph order with the same f32 arithmetic the separate kernels use.

static __device__ __forceinline__ float ew_unary(const int u, const float x) {
    switch (u) {
        case GGML_UNARY_OP_SIGMOID: return 1.0f / (1.0f + expf(-x));
        case GGML_UNARY_OP_SILU:    return x / (1.0f + expf(-x));
        case GGML_UNARY_OP_EXP:     return expf(x);
        case GGML_UNARY_OP_NEG:     return -x;
        case GGML_UNARY_OP_RELU:    return fmaxf(x, 0.0f);
        case GGML_UNARY_OP_TANH:    return tanhf(x);
        case GGML_UNARY_OP_ABS:     return fabsf(x);
        default:                    return x;
    }
}

// 32-bit index arithmetic (the chains are small: hc gates are hc x n_tokens, the KDA gate d_inner x n_tokens) and
// per-dimension broadcast flags instead of a modulo per op per dimension: the first version at 64-bit with four
// modulos per op took ~7 us per launch on gfx1151, more than the ~1.5 us kernels it replaced
static __global__ void __launch_bounds__(256) k_ew_chain(const ggml_cuda_ew_chain c) {
    const uint32_t ne0 = c.ne[0], ne1 = c.ne[1], ne2 = c.ne[2];
    const uint32_t n   = ne0 * ne1 * ne2 * (uint32_t) c.ne[3];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const uint32_t i0 = i % ne0;
        const uint32_t t1 = i / ne0;
        const uint32_t i1 = t1 % ne1;
        const uint32_t t2 = t1 / ne1;
        const uint32_t i2 = t2 % ne2;
        const uint32_t i3 = t2 / ne2;

        float v = *(const float *) (c.src0 + i0*c.nb0[0] + i1*c.nb0[1] + i2*c.nb0[2] + i3*c.nb0[3]);

#pragma unroll
        for (int k = 0; k < GGML_CUDA_EW_MAX_OPS; ++k) {
            if (k >= c.n) {
                break;
            }
            const ggml_cuda_ew_op & o = c.ops[k];
            switch (o.op) {
                case GGML_OP_MUL:
                case GGML_OP_ADD:
                case GGML_OP_SUB:
                case GGML_OP_DIV: {
                    // broadcast: a dimension of size 1 in src1 contributes nothing, a full one indexes directly
                    const uint32_t j0 = o.ne1[0] == 1 ? 0 : (o.ne1[0] == ne0 ? i0 : i0 % (uint32_t) o.ne1[0]);
                    const uint32_t j1 = o.ne1[1] == 1 ? 0 : (o.ne1[1] == ne1 ? i1 : i1 % (uint32_t) o.ne1[1]);
                    const uint32_t j2 = o.ne1[2] == 1 ? 0 : (o.ne1[2] == ne2 ? i2 : i2 % (uint32_t) o.ne1[2]);
                    const uint32_t j3 = o.ne1[3] == 1 ? 0 : i3 % (uint32_t) o.ne1[3];
                    const float y = *(const float *) (o.src1 + j0*o.nb1[0] + j1*o.nb1[1] + j2*o.nb1[2] + j3*o.nb1[3]);
                    v = o.op == GGML_OP_MUL ? v * y : o.op == GGML_OP_ADD ? v + y : o.op == GGML_OP_SUB ? v - y : v / y;
                } break;
                case GGML_OP_SCALE: v = v * o.s + o.b;   break;
                case GGML_OP_UNARY: v = ew_unary(o.unary, v); break;
                case GGML_OP_SQR:   v = v * v;           break;
                case GGML_OP_SQRT:  v = sqrtf(v);        break;
                default: break;
            }
        }
        c.dst[i] = v;
    }
}

void ggml_cuda_op_ew_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ew_chain & c) {
    const int64_t n = c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3];
    if (n == 0) {
        return;
    }
    GGML_ASSERT(n < (int64_t) 1 << 31);
    const int64_t nblk = std::min<int64_t>((n + 255) / 256, 65535);
    k_ew_chain<<<(unsigned) nblk, 256, 0, ctx.stream()>>>(c);
    CUDA_CHECK(cudaGetLastError());
}
