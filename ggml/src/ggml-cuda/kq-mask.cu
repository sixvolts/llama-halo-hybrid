// halo-hybrid: see ggml_kq_mask_build (ggml.h)
#include "kq-mask.cuh"

template <typename T, int mode>
static __global__ void k_kq_mask_build(
        const int32_t * __restrict__ pos_kv, const int32_t * __restrict__ pos_q, const int32_t * __restrict__ pool_of,
        const int32_t * __restrict__ tail_start, const int32_t * __restrict__ bo_vis,
        T * __restrict__ dst, const int64_t n_kv, const int64_t nb1) {
    const int64_t j = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i = blockIdx.y;
    if (j >= n_kv) {
        return;
    }
    const int32_t  p  = pos_kv[j];
    const uint32_t q  = (uint32_t) pos_q[i];
    bool keep = (uint32_t) p <= q;
    if (mode == 1) {
        keep = keep && p >= tail_start[i];
    } else if (mode == 2) {
        keep = keep && (((uint32_t) pool_of[j] < (uint32_t) bo_vis[i]) || p >= tail_start[i]);
    }
    const float v = keep ? 0.0f : -INFINITY;
    T * row = (T *) ((char *) dst + i*nb1);
    if constexpr (std::is_same<T, half>::value) {
        row[j] = __float2half(v);
    } else {
        row[j] = v;
    }
}

template <typename T>
static void kq_mask_build_launch(int mode, const int32_t * pos_kv, const int32_t * pos_q, const int32_t * pool_of,
        const int32_t * tail_start, const int32_t * bo_vis, T * dst, int64_t n_kv, int64_t n_q, int64_t nb1, cudaStream_t stream) {
    const dim3 grid((n_kv + 255)/256, n_q);
    switch (mode) {
        case 0: k_kq_mask_build<T, 0><<<grid, 256, 0, stream>>>(pos_kv, pos_q, pool_of, tail_start, bo_vis, dst, n_kv, nb1); break;
        case 1: k_kq_mask_build<T, 1><<<grid, 256, 0, stream>>>(pos_kv, pos_q, pool_of, tail_start, bo_vis, dst, n_kv, nb1); break;
        default: k_kq_mask_build<T, 2><<<grid, 256, 0, stream>>>(pos_kv, pos_q, pool_of, tail_start, bo_vis, dst, n_kv, nb1); break;
    }
}

void ggml_cuda_op_kq_mask_build(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int32_t * pos_kv     = (const int32_t *) dst->src[0]->data;
    const int32_t * pos_q      = (const int32_t *) dst->src[1]->data;
    const int32_t * pool_of    = dst->src[2] ? (const int32_t *) dst->src[2]->data : nullptr;
    const int32_t * tail_start = dst->src[3] ? (const int32_t *) dst->src[3]->data : nullptr;
    const int32_t * bo_vis     = dst->src[4] ? (const int32_t *) dst->src[4]->data : nullptr;
    const int32_t mode = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(dst->ne[1] <= 65535);

    cudaStream_t stream = ctx.stream();
    if (dst->type == GGML_TYPE_F16) {
        kq_mask_build_launch<half>(mode, pos_kv, pos_q, pool_of, tail_start, bo_vis, (half *) dst->data, dst->ne[0], dst->ne[1], dst->nb[1], stream);
    } else {
        kq_mask_build_launch<float>(mode, pos_kv, pos_q, pool_of, tail_start, bo_vis, (float *) dst->data, dst->ne[0], dst->ne[1], dst->nb[1], stream);
    }
}
