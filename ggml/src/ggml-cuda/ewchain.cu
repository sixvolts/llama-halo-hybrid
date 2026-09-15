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

static __global__ void k_ew_chain(const ggml_cuda_ew_chain c) {
    const int64_t n = c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3];
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t i0 = i % c.ne[0];
        const int64_t t1 = i / c.ne[0];
        const int64_t i1 = t1 % c.ne[1];
        const int64_t t2 = t1 / c.ne[1];
        const int64_t i2 = t2 % c.ne[2];
        const int64_t i3 = t2 / c.ne[2];

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
                    const float y = *(const float *) (o.src1 + (i0 % o.ne1[0])*o.nb1[0] + (i1 % o.ne1[1])*o.nb1[1]
                                                             + (i2 % o.ne1[2])*o.nb1[2] + (i3 % o.ne1[3])*o.nb1[3]);
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
    const int64_t nblk = std::min<int64_t>((n + 255) / 256, 65535);
    k_ew_chain<<<(unsigned) nblk, 256, 0, ctx.stream()>>>(c);
    CUDA_CHECK(cudaGetLastError());
}
