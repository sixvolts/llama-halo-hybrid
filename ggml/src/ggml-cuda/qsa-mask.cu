#include "qsa-mask.cuh"

// One block per mask row: clear the row's bitmap, set the bits of its top-k cells, then stream the row (8 halves per
// thread and step when the rows are 16-byte aligned).
#define QSA_MASK_NTHR      256
#define QSA_MASK_MAX_WORDS 8192   // 32 KB of LDS: n_kv <= 262144

static __device__ __forceinline__ half qsa_mask_cell(const bool sel, const half m) {
    // the graph's ADD: (float) filled + (float) kq_mask, rounded to half once
    return __float2half((sel ? 0.0f : -INFINITY) + __half2float(m));
}

template <typename idx_t, bool vec>
static __global__ void __launch_bounds__(QSA_MASK_NTHR) k_qsa_mask(const half * __restrict__ mask, half * __restrict__ out,
        const int64_t n_kv, const int64_t nb1, const int64_t nb3, const char * __restrict__ idx, const int64_t n_top,
        const int64_t idx_ne1, const int64_t idx_ne2, const int64_t idx_nb1, const int64_t idx_nb2) {
    extern __shared__ uint32_t s_sel[];
    const int64_t r   = blockIdx.x;
    const int64_t s   = blockIdx.y;
    const int     tid = threadIdx.x;

    const int nw = (int) ((n_kv + 31) / 32);
    for (int w = tid; w < nw; w += QSA_MASK_NTHR) {
        s_sel[w] = 0;
    }
    __syncthreads();
    // SET_ROWS writes only in-range cells (the CPU asserts it); duplicates write the same zero
    const idx_t * ir = (const idx_t *) (idx + (r % idx_ne1) * idx_nb1 + (s % idx_ne2) * idx_nb2);
    for (int64_t k = tid; k < n_top; k += QSA_MASK_NTHR) {
        const int64_t c = (int64_t) ir[k];
        if (c >= 0 && c < n_kv) {
            atomicOr(&s_sel[c >> 5], 1u << (c & 31));
        }
    }
    __syncthreads();

    const half * mr = (const half *) ((const char *) mask + r * nb1 + s * nb3);
    half *       orow = (half *) ((char *) out + r * nb1 + s * nb3);
    if constexpr (vec) {
        for (int64_t j8 = tid; j8 < n_kv / 8; j8 += QSA_MASK_NTHR) {
            const uint32_t bits = (s_sel[j8 >> 2] >> ((j8 & 3) * 8)) & 0xffu;
            uint4 v = ((const uint4 *) mr)[j8];
            half * h = (half *) &v;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                h[e] = qsa_mask_cell((bits >> e) & 1u, h[e]);
            }
            ((uint4 *) orow)[j8] = v;
        }
    } else {
        for (int64_t j = tid; j < n_kv; j += QSA_MASK_NTHR) {
            orow[j] = qsa_mask_cell((s_sel[j >> 5] >> (j & 31)) & 1u, mr[j]);
        }
    }
}

int64_t ggml_cuda_qsa_mask_max_kv() {
    return (int64_t) QSA_MASK_MAX_WORDS * 32;
}

template <typename idx_t>
static void qsa_mask_launch(const ggml_cuda_qsa_mask_args & a, cudaStream_t stream) {
    const dim3   grid((unsigned) a.R, (unsigned) a.S);
    const size_t smem = (size_t) ((a.n_kv + 31) / 32) * sizeof(uint32_t);
    const bool   vec  = a.n_kv % 8 == 0 && a.nb1 % 16 == 0 && a.nb3 % 16 == 0 && (uintptr_t) a.mask % 16 == 0 &&
                        (uintptr_t) a.out % 16 == 0;
    if (vec) {
        k_qsa_mask<idx_t, true><<<grid, QSA_MASK_NTHR, smem, stream>>>(a.mask, a.out, a.n_kv, a.nb1, a.nb3, a.idx, a.n_top,
                a.idx_ne1, a.idx_ne2, a.idx_nb1, a.idx_nb2);
    } else {
        k_qsa_mask<idx_t, false><<<grid, QSA_MASK_NTHR, smem, stream>>>(a.mask, a.out, a.n_kv, a.nb1, a.nb3, a.idx, a.n_top,
                a.idx_ne1, a.idx_ne2, a.idx_nb1, a.idx_nb2);
    }
}

void ggml_cuda_op_qsa_mask(ggml_backend_cuda_context & ctx, const ggml_cuda_qsa_mask_args & a) {
    GGML_ASSERT(a.n_kv <= ggml_cuda_qsa_mask_max_kv());
    GGML_ASSERT(a.S <= 65535 && a.R <= INT32_MAX);
    if (a.R == 0 || a.S == 0 || a.n_kv == 0) {
        return;
    }
    if (a.idx_i64) {
        qsa_mask_launch<int64_t>(a, ctx.stream());
    } else {
        qsa_mask_launch<int32_t>(a, ctx.stream());
    }
    CUDA_CHECK(cudaGetLastError());
}
