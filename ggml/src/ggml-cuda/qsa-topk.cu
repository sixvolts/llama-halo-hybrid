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

    __shared__ uint32_t hist[256];
    __shared__ uint32_t prefix, pmask;
    __shared__ int      need;
    __shared__ int      sh[QSA_TK_NT];
    if (tid == 0) { prefix = 0; pmask = 0; need = width; }
    __syncthreads();

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
    k_qsa_top_k<<<(int) n_q, QSA_TK_NT, 0, ctx.stream()>>>((const float *) score->data, score->nb[1] / sizeof(float),
            (const int32_t *) q_pos->data, (const int32_t *) n_bid->data, (int32_t *) dst->data, dst->nb[1] / sizeof(int32_t),
            (int) score->ne[0], width, r);
    CUDA_CHECK(cudaGetLastError());
}
