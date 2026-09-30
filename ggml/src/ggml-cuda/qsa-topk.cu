#include "qsa-topk.cuh"

// halo-hybrid: the top-`width` cells of qwen4exp's QSA indexer from its BLOCK scores (GGML_OP_QSA_TOP_K, ggml.h).
// Cells [0, F) of query row i are scored with their block's value (F = min(n_bid*ratio, q+1)), all later cells are
// -inf, so the cell-level top-k (ties in ascending cell index, as ggml_top_k's CUDA path gathers them) is: every block
// above a threshold key T, then the cells at T in ascending index until `width`, then -inf cells from F up. One
// workgroup per row: a weighted radix select over the block keys (each block weighs its scored cells), then two
// scans place the selected cells in ascending order. Replaces expanding the scores to every cell ([n_kv, n_q] f32)
// and a radix top-k over them.

#define QSA_TK_NT 256

static __device__ __forceinline__ uint32_t qsa_tk_key(const float x) {
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}

// exclusive scan of v over the workgroup; returns the prefix, *total the sum
static __device__ int qsa_tk_scan(int v, int * sh, int * total) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int off = 1; off < QSA_TK_NT; off *= 2) {
        const int t = tid >= off ? sh[tid - off] : 0;
        __syncthreads();
        sh[tid] += t;
        __syncthreads();
    }
    const int incl = sh[tid];
    *total = sh[QSA_TK_NT - 1];
    __syncthreads();
    return incl - v;
}

// one radix pass over the digit bits [shift, shift + BITS) of the keys that match the resolved prefix: the weighted
// histogram, then a parallel search (descending bins) for the bin holding the need-th heaviest cell. Resolves BITS
// more bits of the threshold key T into *prefix / *pmask and leaves in *need the cells still to take at or below it.
template <int BITS>
static __device__ void qsa_tk_pass(const float * __restrict__ srow, const int nbq, const int r, const int F, const int shift,
        uint32_t * hist, int * sh, uint32_t * prefix, uint32_t * pmask, int * need) {
    constexpr int NBINS = 1 << BITS;
    constexpr int PER   = NBINS / QSA_TK_NT;   // bins per thread in the search
    static_assert(NBINS % QSA_TK_NT == 0, "bins must split evenly over the threads");
    const int tid = threadIdx.x;

    for (int k = tid; k < NBINS; k += QSA_TK_NT) { hist[k] = 0; }
    __syncthreads();
    const uint32_t pf = *prefix, pm = *pmask;
    for (int b = tid; b < nbq; b += QSA_TK_NT) {
        const uint32_t key = qsa_tk_key(srow[b]);
        if ((key & pm) == pf) {
            atomicAdd(&hist[(key >> shift) & (NBINS - 1)], (uint32_t) min(r, F - b*r));
        }
    }
    __syncthreads();

    // thread t owns bins [NBINS - PER*(t+1), NBINS - PER*t), scanned from the top: the exclusive scan over threads
    // gives the weight of every higher bin
    const int top = NBINS - PER*tid - 1;
    int own = 0;
    for (int j = 0; j < PER; ++j) { own += (int) hist[top - j]; }
    int tot;
    const int above = qsa_tk_scan(own, sh, &tot);
    const int nd = *need;
    __syncthreads();
    if (above < nd && above + own >= nd) {
        int acc = above;
        int bin = top - PER + 1;
        for (int j = 0; j < PER; ++j) {
            if (acc + (int) hist[top - j] >= nd) { bin = top - j; break; }
            acc += (int) hist[top - j];
        }
        *prefix = pf | ((uint32_t) bin << shift);
        *pmask  = pm | ((uint32_t) (NBINS - 1) << shift);
        *need   = nd - acc;
    }
    __syncthreads();
}

// halo-hybrid: the threshold search in three passes of 12, 12 and 8 bits (was four of 8). The indexer scores are
// ReLU'd sums, so the top 8 key bits (sign and exponent) put nearly every block in a handful of bins and all 256
// threads' atomics serialized on them; 12-bit digits spread them over the mantissa as well, and one row read less.
// Same T and tie count, so the same cells in the same order. GGML_CUDA_QSA_TK_V1=1 restores the old search.
template <bool V2>
static __global__ void k_qsa_top_k(const float * __restrict__ score, const int64_t s1, const int32_t * __restrict__ q_pos,
        const int32_t * __restrict__ n_bid, int32_t * __restrict__ dst, const int64_t d1,
        const int n_blocks, const int width, const int r) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int q   = q_pos[i];
    const int nb  = min(n_bid[0], n_blocks);
    const int F   = max(0, min(nb * r, q + 1));
    int32_t * out = dst + (int64_t) i * d1;

    if (F <= width) {   // every scored cell, then the -inf cells F.. in index order: 0 .. width-1
        for (int k = tid; k < width; k += QSA_TK_NT) {
            out[k] = k;
        }
        return;
    }

    const float * srow = score + (int64_t) i * s1;
    const int nbq = (F + r - 1) / r;   // blocks with scored cells; block b weighs min(r, F - b*r)

    __shared__ uint32_t hist[V2 ? 4096 : 256];
    __shared__ uint32_t prefix, pmask;
    __shared__ int      need;
    __shared__ int      sh[QSA_TK_NT];
    if (tid == 0) { prefix = 0; pmask = 0; need = width; }
    __syncthreads();

    if constexpr (V2) {
        qsa_tk_pass<12>(srow, nbq, r, F, 20, hist, sh, &prefix, &pmask, &need);
        qsa_tk_pass<12>(srow, nbq, r, F,  8, hist, sh, &prefix, &pmask, &need);
        qsa_tk_pass< 8>(srow, nbq, r, F,  0, hist, sh, &prefix, &pmask, &need);
    } else
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int k = tid; k < 256; k += QSA_TK_NT) { hist[k] = 0; }
        __syncthreads();
        const uint32_t pf = prefix, pm = pmask;
        for (int b = tid; b < nbq; b += QSA_TK_NT) {
            const uint32_t key = qsa_tk_key(srow[b]);
            if ((key & pm) == pf) {
                atomicAdd(&hist[(key >> shift) & 255], (uint32_t) min(r, F - b*r));
            }
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0;
            int bin = 255;
            for (; bin > 0; --bin) {
                if (acc + (int) hist[bin] >= need) { break; }
                acc += (int) hist[bin];
            }
            prefix |= (uint32_t) bin << shift;
            pmask  |= 255u << shift;
            need   -= acc;   // cells still to take at or below this bin
        }
        __syncthreads();
    }
    const uint32_t T     = prefix;
    const int      limit = need;   // cells taken at key T, in ascending index

    // contiguous chunk of blocks per thread, ascending
    const int chunk = (nbq + QSA_TK_NT - 1) / QSA_TK_NT;
    const int b0 = min(nbq, tid * chunk);
    const int b1 = min(nbq, b0 + chunk);

    int eq_w = 0;
    for (int b = b0; b < b1; ++b) {
        if (qsa_tk_key(srow[b]) == T) { eq_w += min(r, F - b*r); }
    }
    int tot;
    int eq_before = qsa_tk_scan(eq_w, sh, &tot);

    int sel_w = 0;
    {
        int eb = eq_before;
        for (int b = b0; b < b1; ++b) {
            const uint32_t key = qsa_tk_key(srow[b]);
            const int w = min(r, F - b*r);
            if (key > T) {
                sel_w += w;
            } else if (key == T) {
                sel_w += max(0, min(w, limit - eb));
                eb += w;
            }
        }
    }
    int o = qsa_tk_scan(sel_w, sh, &tot);

    int eb = eq_before;
    for (int b = b0; b < b1; ++b) {
        const uint32_t key = qsa_tk_key(srow[b]);
        const int w = min(r, F - b*r);
        int take = 0;
        if (key > T) {
            take = w;
        } else if (key == T) {
            take = max(0, min(w, limit - eb));
            eb += w;
        }
        for (int m = 0; m < take; ++m) {
            out[o++] = b*r + m;
        }
    }
}

void ggml_cuda_op_qsa_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * score = dst->src[0];
    const ggml_tensor * q_pos = dst->src[1];
    const ggml_tensor * n_bid = dst->src[2];
    GGML_ASSERT(score->type == GGML_TYPE_F32 && q_pos->type == GGML_TYPE_I32 && n_bid->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(q_pos) && ggml_is_contiguous(dst));
    const int width = ggml_get_op_params_i32(dst, 0);
    const int r     = ggml_get_op_params_i32(dst, 1);
    const int64_t n_q = score->ne[1];
    static const bool v1 = getenv("GGML_CUDA_QSA_TK_V1") != nullptr && atoi(getenv("GGML_CUDA_QSA_TK_V1")) != 0;
    if (v1) {
        k_qsa_top_k<false><<<(int) n_q, QSA_TK_NT, 0, ctx.stream()>>>((const float *) score->data, score->nb[1] / sizeof(float),
                (const int32_t *) q_pos->data, (const int32_t *) n_bid->data, (int32_t *) dst->data, dst->nb[1] / sizeof(int32_t),
                (int) score->ne[0], width, r);
    } else {
        k_qsa_top_k<true><<<(int) n_q, QSA_TK_NT, 0, ctx.stream()>>>((const float *) score->data, score->nb[1] / sizeof(float),
                (const int32_t *) q_pos->data, (const int32_t *) n_bid->data, (int32_t *) dst->data, dst->nb[1] / sizeof(int32_t),
                (int) score->ne[0], width, r);
    }
    CUDA_CHECK(cudaGetLastError());
}

// halo-hybrid: GGML_OP_QSA_HEAD_SUM - relu per head, summed in head order, plus the optional bias; one read of each
// head row and one write, where relu + a chain of adds + the bias add made ~4x the passes over [n_blocks, n_tok]
template <int NH>
static __global__ void k_qsa_head_sum(const float * __restrict__ x, const float * __restrict__ bias, float * __restrict__ dst,
        const int64_t nb, const int nh_rt, const int64_t n_tok,
        const int64_t sx1, const int64_t sx2, const int64_t sx3, const int64_t sb1, const int64_t sb2,
        const int64_t sd1, const int64_t sd2) {
    const int64_t b = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t t = blockIdx.y;
    const int64_t s = blockIdx.z;
    if (b >= nb) {
        return;
    }
    const float * xr = x + t*sx2 + s*sx3 + b;
    const int nh = NH > 0 ? NH : nh_rt;
    float acc = fmaxf(xr[0], 0.0f);
#pragma unroll
    for (int h = 1; h < (NH > 0 ? NH : 64); ++h) {
        if (NH == 0 && h >= nh) { break; }
        acc = acc + fmaxf(xr[h*sx1], 0.0f);
    }
    if (bias) {
        acc = acc + bias[t*sb1 + s*sb2 + b];
    }
    dst[t*sd1 + s*sd2 + b] = acc;
}

void ggml_cuda_op_qsa_head_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x    = dst->src[0];
    const ggml_tensor * bias = dst->src[1];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    const int64_t nb = x->ne[0];
    const int     nh = (int) x->ne[1];
    const dim3 grid((unsigned) ((nb + 255) / 256), (unsigned) x->ne[2], (unsigned) x->ne[3]);
    const int64_t sx1 = x->nb[1]/sizeof(float), sx2 = x->nb[2]/sizeof(float), sx3 = x->nb[3]/sizeof(float);
    const int64_t sb1 = bias ? bias->nb[1]/sizeof(float) : 0, sb2 = bias ? bias->nb[2]/sizeof(float) : 0;
    const int64_t sd1 = dst->nb[1]/sizeof(float), sd2 = dst->nb[2]/sizeof(float);
    const float * bp = bias ? (const float *) bias->data : nullptr;
    if (nh == 4) {
        k_qsa_head_sum<4><<<grid, 256, 0, ctx.stream()>>>((const float *) x->data, bp, (float *) dst->data, nb, nh, x->ne[2],
            sx1, sx2, sx3, sb1, sb2, sd1, sd2);
    } else {
        GGML_ASSERT(nh >= 1 && nh <= 64);
        k_qsa_head_sum<0><<<grid, 256, 0, ctx.stream()>>>((const float *) x->data, bp, (float *) dst->data, nb, nh, x->ne[2],
            sx1, sx2, sx3, sb1, sb2, sd1, sd2);
    }
    CUDA_CHECK(cudaGetLastError());
}
