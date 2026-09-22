#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

// halo-hybrid: one block per token reading its n_expert_used rows front to back with 16-byte loads. The column-chunk
//     grid above has the blocks that run together (consecutive tokens, one 1 KB chunk each) touch 8 rows 32 KB apart
//     per token, which on gfx1151 GTT memory ran at ~60 GB/s in situ (2.5 ms per 1024-token GLM layer) while the same
//     launch ran at cache speed in isolation. Same arithmetic per element and the same expert order, so the result is
//     bit-identical. GGML_CUDA_MOE_WR_COLS=1 restores the column-chunk kernel.
static __global__ void moe_weighted_reduction_f32_rows(const float * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       float * __restrict__ dst,
                                                       const int64_t n_embd,
                                                       const int     n_expert_used) {
    const int64_t  token     = blockIdx.x;
    const uint64_t first_row = (uint64_t) token * n_expert_used;
    const int64_t  n4        = n_embd / 4;

    for (int64_t c4 = threadIdx.x; c4 < n4; c4 += blockDim.x) {
        const float  s0 = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
        const float  w0 = weights[first_row];
        const float4 v0 = reinterpret_cast<const float4 *>(experts + first_row * n_embd)[c4];
        float4 sum = make_float4((v0.x * s0) * w0, (v0.y * s0) * w0, (v0.z * s0) * w0, (v0.w * s0) * w0);

        for (int expert = 1; expert < n_expert_used; ++expert) {
            const uint64_t row = first_row + expert;
            const float    sc  = expert_scale != nullptr ? expert_scale[row] : 1.0f;
            const float    w   = weights[row];
            const float4   v   = reinterpret_cast<const float4 *>(experts + row * n_embd)[c4];
            sum.x += (v.x * sc) * w;
            sum.y += (v.y * sc) * w;
            sum.z += (v.z * sc) * w;
            sum.w += (v.w * sc) * w;
        }
        reinterpret_cast<float4 *>(dst + token * n_embd)[c4] = sum;
    }
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    static const bool cols = getenv("GGML_CUDA_MOE_WR_COLS") != nullptr;
    const bool aligned = n_embd % 4 == 0 && ((uintptr_t) experts % 16) == 0 && ((uintptr_t) dst % 16) == 0;
    if (!cols && aligned) {
        moe_weighted_reduction_f32_rows
            <<<n_tokens, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
        return;
    }
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    launch_moe_weighted_reduction((const float *) experts->data,
                                  expert_scale ? (const float *) expert_scale->data : nullptr,
                                  (const float *) weights->data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
