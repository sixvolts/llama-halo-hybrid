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

// one block per group of G rows. LDS: the union bitmap, one membership byte per union entry (bit r: row r selected
// the cell, bit 4: the cell survives), per-thread word ranges with their exclusive popcount bases.
#define QSA_LIST_NTHR 256
template <typename idx_t>
static __global__ void __launch_bounds__(QSA_LIST_NTHR) k_qsa_mask_lists(const half * __restrict__ kq, const int64_t s1,
        const int64_t n_kv, const int64_t R, const char * __restrict__ idx, const int64_t n_top, const int64_t idx_ne1,
        const int64_t idx_nb1, int32_t * __restrict__ lists, half * __restrict__ vals, const int cap, const int G,
        const int n_mem_words) {
    extern __shared__ uint32_t s_dyn[];
    __shared__ int s_base[QSA_LIST_NTHR];
    __shared__ int s_scan[QSA_LIST_NTHR];
    const int tid = threadIdx.x;
    const int64_t g  = blockIdx.x;
    const int64_t r0 = g*G;
    const int nw  = (int) ((n_kv + 31) / 32);
    uint32_t * s_bits = s_dyn;
    uint32_t * s_mem  = s_dyn + nw;
    for (int w = tid; w < nw; w += QSA_LIST_NTHR) { s_bits[w] = 0; }
    for (int w = tid; w < n_mem_words; w += QSA_LIST_NTHR) { s_mem[w] = 0; }
    __syncthreads();
    const int nrows = (int) min((int64_t) G, R - r0);
    for (int r = 0; r < nrows; ++r) {
        const idx_t * ir = (const idx_t *) (idx + ((r0 + r) % idx_ne1) * idx_nb1);
        for (int64_t k = tid; k < n_top; k += QSA_LIST_NTHR) {
            const int64_t c = (int64_t) ir[k];
            if (c >= 0 && c < n_kv) {
                atomicOr(&s_bits[c >> 5], 1u << (c & 31));
            }
        }
    }
    __syncthreads();

    const auto scan = [&](int v) {   // exclusive prefix over the threads, s_scan[NT - 1] + last v = total
        s_scan[tid] = v;
        __syncthreads();
        for (int off = 1; off < QSA_LIST_NTHR; off <<= 1) {
            const int u = tid >= off ? s_scan[tid - off] : 0;
            __syncthreads();
            s_scan[tid] += u;
            __syncthreads();
        }
        return s_scan[tid] - v;
    };
    const int per = (nw + QSA_LIST_NTHR - 1) / QSA_LIST_NTHR;
    const int w0  = min(nw, tid*per), w1 = min(nw, w0 + per);
    int local = 0;
    for (int w = w0; w < w1; ++w) { local += __popc(s_bits[w]); }
    s_base[tid] = scan(local);
    __syncthreads();

    // membership: the union rank of every selected cell
    for (int r = 0; r < nrows; ++r) {
        const idx_t * ir = (const idx_t *) (idx + ((r0 + r) % idx_ne1) * idx_nb1);
        for (int64_t k = tid; k < n_top; k += QSA_LIST_NTHR) {
            const int64_t c = (int64_t) ir[k];
            if (c >= 0 && c < n_kv) {
                const int w = (int) (c >> 5), o = w / per;
                int rank = s_base[o];
                for (int u = o*per; u < w; ++u) { rank += __popc(s_bits[u]); }
                rank += __popc(s_bits[w] & ((1u << (c & 31)) - 1u));
                atomicOr(&s_mem[rank >> 2], 1u << ((rank & 3)*8 + r));
            }
        }
    }
    __syncthreads();

    // survivors: some member row's value (half) (0.0f + kq_mask) is finite, as the written mask's compaction decides
    int keep = 0;
    {
        int p = s_base[tid];
        for (int w = w0; w < w1; ++w) {
            uint32_t b = s_bits[w];
            while (b) {
                const int c = w*32 + __ffs(b) - 1;
                b &= b - 1;
                const uint32_t m = (s_mem[p >> 2] >> ((p & 3)*8)) & 0xfu;
                bool fin = false;
                for (int r = 0; r < nrows; ++r) {
                    if ((m >> r) & 1u) {
                        fin = fin || isfinite(__half2float(__float2half(0.0f + __half2float(kq[(r0 + r)*s1 + c]))));
                    }
                }
                if (fin) {
                    atomicOr(&s_mem[p >> 2], 0x10u << ((p & 3)*8));
                    ++keep;
                }
                ++p;
            }
        }
    }
    __syncthreads();
    const int out0 = scan(keep);
    int32_t * list = lists + g*(int64_t) (cap + 1);
    if (tid == QSA_LIST_NTHR - 1) {
        list[0] = min(out0 + keep, cap);
    }
    {
        half * gv = vals + g*(int64_t) cap*G;
        int p = s_base[tid], q = out0;
        for (int w = w0; w < w1 && q < cap; ++w) {
            uint32_t b = s_bits[w];
            while (b) {
                const int c = w*32 + __ffs(b) - 1;
                b &= b - 1;
                const uint32_t m = (s_mem[p >> 2] >> ((p & 3)*8)) & 0xffu;
                if (m & 0x10u) {
                    if (q < cap) {
                        list[1 + q] = c;
                        for (int r = 0; r < G; ++r) {
                            gv[(int64_t) q*G + r] = r < nrows && ((m >> r) & 1u) ?
                                __float2half(0.0f + __half2float(kq[(r0 + r)*s1 + c])) : __float2half(-INFINITY);
                        }
                    }
                    ++q;
                }
                ++p;
            }
        }
    }
}

void ggml_cuda_qsa_mask_lists(const ggml_cuda_qsa_mask_args & a, int32_t * lists, half * vals, int cap, int group, cudaStream_t stream) {
    GGML_ASSERT(a.S == 1 && group >= 1 && group <= 4 && a.n_kv <= INT32_MAX);
    const int     groups = (int) ((a.R + group - 1) / group);
    const int     nw     = (int) ((a.n_kv + 31) / 32);
    const int     n_mem  = (int) ((group*a.n_top + 3) / 4);
    const size_t  smem   = size_t(nw + n_mem) * sizeof(uint32_t);
    if (groups == 0) {
        return;
    }
    const int64_t s1 = a.nb1 / (int64_t) sizeof(half);
    if (a.idx_i64) {
        k_qsa_mask_lists<int64_t><<<groups, QSA_LIST_NTHR, smem, stream>>>(a.mask, s1, a.n_kv, a.R, a.idx, a.n_top,
            a.idx_ne1, a.idx_nb1, lists, vals, cap, group, n_mem);
    } else {
        k_qsa_mask_lists<int32_t><<<groups, QSA_LIST_NTHR, smem, stream>>>(a.mask, s1, a.n_kv, a.R, a.idx, a.n_top,
            a.idx_ne1, a.idx_nb1, lists, vals, cap, group, n_mem);
    }
    CUDA_CHECK(cudaGetLastError());
}
