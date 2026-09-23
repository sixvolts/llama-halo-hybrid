#include "common.cuh"
#include "dsv4-hc.cuh"
#include "hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}

// halo-hybrid: the whole hyper-connection prologue of a sublayer in one launch, one 1024-thread block per token.
//     Replaces rms_norm + the 24-row hc_fn GEMV + two gate chains + dsv4_hc_comb + dsv4_hc_pre (6 launches per
//     mixer, 2 mixers per layer). The token's hc*n_embd activation stays in registers (16 per thread at n_embd
//     4096); the 24 dot products are block-reduced; the 4x4 sinkhorn runs on one thread; the pre-mix accumulates
//     the 4 streams through LDS in 4 ordered passes (no atomics, deterministic).
#define DSV4_HC_MIX_THREADS 1024
#define DSV4_HC_MIX_PER_THREAD 16   // hc*n_embd == 16 * 1024 (host checks); each thread owns 16 CONSECUTIVE elements

// halo-hybrid: optional work at the two ends of the prologue (dsv4_hc_mix_fused)
//     POST: the preceding dsv4_hc_post is computed in the load stage: this thread's 16 elements of the new residual
//           are post*x + sum_src comb*res, in dsv4_hc_post_f32's order, written to the hc_post dst (it has a second
//           consumer, the next hc_post's residual) and used as the prologue's input without a reload
//     NORM: the following RMS_NORM + MUL(w) runs on the finished pre-mix row in LDS in rms_norm_f32<1024>'s order and
//           also writes the q8_1 copy of the result for the GEMVs that consume it
struct dsv4_hc_mix_ext {
    const float * px;   int64_t spx1;                   // hc_post x [n_embd, nt]
    const float * pres; int64_t spr1, spr2;             // hc_post residual [n_embd, hc, nt]
    const float * ppost; int64_t spp0, spp1;            // hc_post post [hc, nt]
    const float * pcomb; int64_t spc0, spc1, spc2;      // hc_post comb [hc, hc, nt]
    float * pdst;                                       // hc_post dst == the prologue's x (same strides)
    const float * nw;                                   // norm weight [n_embd]
    float * ndst; int64_t snd1;                         // MUL dst [n_embd, nt]
    block_q8_1 * q8;                                    // q8_1 copy of ndst, [nt][n_embd/32] blocks; may be null
    float eps_rms;
    int write_out;                                      // also write the un-normed pre-mix row into dst
};

template <bool WQ8, bool POST, bool NORM>
static __global__ void __launch_bounds__(DSV4_HC_MIX_THREADS) dsv4_hc_mix_f32(
        const float * __restrict__ x, const void * __restrict__ w, const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ dst,
        const int n_embd, const int64_t sx1, const int64_t sx2, const int64_t ss0, const int64_t sb0, const int64_t sd1,
        const float eps_norm, const float eps_hc, const int n_iter, const dsv4_hc_mix_ext ext) {
    constexpr int hc      = DSV4_HC;
    constexpr int mix_dim = (2 + hc)*hc;
    constexpr int NL      = DSV4_HC_MIX_PER_THREAD;
    const int hc_dim = hc*n_embd;
    const int tid    = threadIdx.x;
    const int lane   = tid % 32;
    const int warp   = tid / 32;
    const int it     = blockIdx.x;

    extern __shared__ float smem[];
    float * acc     = smem;                        // [n_embd]  pre-mix accumulator
    float * red     = acc + n_embd;                // [32][mix_dim] per-warp partial dots
    float * mixes   = red + 32*mix_dim;            // [mix_dim]
    float * pre     = mixes + mix_dim;             // [hc]
    float * redss   = pre + hc;                    // [32]

    // this thread's 16 consecutive elements: flat index i0 = 16*tid, all in stream h (n_embd % 16 == 0)
    const int i0 = NL*tid;
    const int h  = i0 / n_embd;
    const int j0 = i0 - h*n_embd;
    // 1. load (4 x float4, consecutive threads read consecutive 64-byte runs)
    float xv[NL];
    float ss = 0.0f;
    if constexpr (POST) {
        // new_residual[h][j] = x[j]*post[h] + sum_src res[src][j]*comb[h][src], as dsv4_hc_post_f32 does it
        const float ph = ext.ppost[h*ext.spp0 + it*ext.spp1];
        float ch[hc];
#pragma unroll
        for (int isrc = 0; isrc < hc; ++isrc) { ch[isrc] = ext.pcomb[h*ext.spc0 + isrc*ext.spc1 + it*ext.spc2]; }
        const float4 * pxt = (const float4 *) (ext.px + it*ext.spx1 + j0);
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            const float4 v = pxt[q];
            xv[4*q + 0] = __fmul_rn(v.x, ph); xv[4*q + 1] = __fmul_rn(v.y, ph);
            xv[4*q + 2] = __fmul_rn(v.z, ph); xv[4*q + 3] = __fmul_rn(v.w, ph);
        }
#pragma unroll
        for (int isrc = 0; isrc < hc; ++isrc) {
            const float4 * rt = (const float4 *) (ext.pres + it*ext.spr2 + isrc*ext.spr1 + j0);
#pragma unroll
            for (int q = 0; q < NL/4; ++q) {
                const float4 r = rt[q];
                xv[4*q + 0] = __fmaf_rn(r.x, ch[isrc], xv[4*q + 0]); xv[4*q + 1] = __fmaf_rn(r.y, ch[isrc], xv[4*q + 1]);
                xv[4*q + 2] = __fmaf_rn(r.z, ch[isrc], xv[4*q + 2]); xv[4*q + 3] = __fmaf_rn(r.w, ch[isrc], xv[4*q + 3]);
            }
        }
        float4 * ot = (float4 *) (ext.pdst + it*sx2 + h*sx1 + j0);
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            ot[q] = make_float4(xv[4*q + 0], xv[4*q + 1], xv[4*q + 2], xv[4*q + 3]);
            ss += xv[4*q + 0]*xv[4*q + 0] + xv[4*q + 1]*xv[4*q + 1] + xv[4*q + 2]*xv[4*q + 2] + xv[4*q + 3]*xv[4*q + 3];
        }
    } else {
        const float4 * xt = (const float4 *) (x + it*sx2 + h*sx1 + j0);
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            const float4 v = xt[q];
            xv[4*q + 0] = v.x; xv[4*q + 1] = v.y; xv[4*q + 2] = v.z; xv[4*q + 3] = v.w;
            ss += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }
    // 2. rms
    ss = warp_reduce_sum(ss);
    if (lane == 0) { redss[warp] = ss; }
    __syncthreads();
    if (warp == 0) {
        float v = redss[lane];
        v = warp_reduce_sum(v);
        if (lane == 0) { redss[0] = v; }
    }
    __syncthreads();
    const float rms_inv = rsqrtf(redss[0] / (float) hc_dim + eps_norm);

    // 3. mixes = W . x (scaled after the reduction): per row this thread touches half a q8_0 block
    //    (one 16-byte load of quants + the block scale), or 4 float4 of an f32 row
    const int blk  = i0 / QK8_0;          // = tid/2
    const int half = (i0 % QK8_0) / 16;   // = tid%2
#pragma unroll 4
    for (int r = 0; r < mix_dim; ++r) {
        float part;
        if constexpr (WQ8) {
            const block_q8_0 * b = (const block_q8_0 *) w + (int64_t) r * (hc_dim / QK8_0) + blk;
            const int4 qv = *(const int4 *) (b->qs + 16*half);
            const int8_t * q = (const int8_t *) &qv;
            float acc_q = 0.0f;
#pragma unroll
            for (int k = 0; k < NL; ++k) { acc_q += (float) q[k] * xv[k]; }
            part = __half2float(b->d) * acc_q;
        } else {
            const float4 * wr = (const float4 *) ((const float *) w + (int64_t) r * hc_dim + i0);
            part = 0.0f;
#pragma unroll
            for (int q = 0; q < NL/4; ++q) {
                const float4 v = wr[q];
                part += v.x*xv[4*q] + v.y*xv[4*q + 1] + v.z*xv[4*q + 2] + v.w*xv[4*q + 3];
            }
        }
        part = warp_reduce_sum(part);
        if (lane == 0) { red[warp*mix_dim + r] = part; }
    }
    __syncthreads();
    if (tid < mix_dim) {
        float v = 0.0f;
#pragma unroll 8
        for (int wi = 0; wi < 32; ++wi) { v += red[wi*mix_dim + tid]; }
        mixes[tid] = v * rms_inv;
    }
    __syncthreads();

    // 4. gates and the 4x4 sinkhorn on one thread; post and comb go straight to dst
    float * drow = dst + it*sd1;
    if (tid == 0) {
        const float scale_pre = scale[0], scale_post = scale[ss0], scale_comb = scale[2*ss0];
        for (int hh = 0; hh < hc; ++hh) {
            pre[hh] = 1.0f / (1.0f + expf(-(mixes[hh]*scale_pre + base[hh*sb0]))) + eps_hc;
            drow[n_embd + hh] = 2.0f / (1.0f + expf(-(mixes[hc + hh]*scale_post + base[(hc + hh)*sb0])));
        }
        float comb[hc*hc];
        for (int isrc = 0; isrc < hc; ++isrc) {
            float max = -INFINITY;
            for (int idst = 0; idst < hc; ++idst) {
                const int idx = idst + hc*isrc;
                const float v = mixes[2*hc + idx]*scale_comb + base[(2*hc + idx)*sb0];
                comb[idx] = v; max = fmaxf(max, v);
            }
            float sum = 0.0f;
            for (int idst = 0; idst < hc; ++idst) { const int idx = idst + hc*isrc; const float v = expf(comb[idx] - max); comb[idx] = v; sum += v; }
            const float inv_sum = 1.0f / sum;
            for (int idst = 0; idst < hc; ++idst) { const int idx = idst + hc*isrc; comb[idx] = comb[idx]*inv_sum + eps_hc; }
        }
        dsv4_hc_comb_norm_cols(comb, eps_hc);
        for (int i = 1; i < n_iter; ++i) { dsv4_hc_comb_norm_rows(comb, eps_hc); dsv4_hc_comb_norm_cols(comb, eps_hc); }
        for (int idx = 0; idx < hc*hc; ++idx) { drow[n_embd + hc + idx] = comb[idx]; }
    }
    for (int j = tid; j < n_embd; j += DSV4_HC_MIX_THREADS) { acc[j] = 0.0f; }
    __syncthreads();

    // 5. out = sum_h pre[h] * x[:, h]: 4 ordered passes; in pass hh only the threads of stream hh write, each its
    //    own 16 consecutive slots
    for (int hh = 0; hh < hc; ++hh) {
        if (h == hh) {
            const float ph = pre[hh];
#pragma unroll
            for (int k = 0; k < NL; ++k) { acc[j0 + k] += ph * xv[k]; }
        }
        __syncthreads();
    }
    if constexpr (NORM) {
        // 6. RMS_NORM + MUL(w) on the finished row, rms_norm_f32<1024, true>'s loop and reduction order
        float tmp = 0.0f;
        for (int col = tid; col < n_embd; col += DSV4_HC_MIX_THREADS) {
            const float xi = acc[col];
            tmp += xi * xi;
        }
        tmp = block_reduce<block_reduce_method::SUM, DSV4_HC_MIX_THREADS>(tmp, redss);
        const float mean  = tmp / n_embd;
        const float scl   = rsqrtf(mean + ext.eps_rms);
        float * nrow = ext.ndst + it*ext.snd1;
        for (int col = tid; col < n_embd; col += DSV4_HC_MIX_THREADS) {
            const float v = scl * acc[col] * ext.nw[col];
            nrow[col] = v;
            if (ext.q8) {
                q8_side_store(ext.q8, (int64_t) it*n_embd + col, v);
            }
            if (ext.write_out) { drow[col] = acc[col]; }
        }
    } else {
        for (int j = tid; j < n_embd; j += DSV4_HC_MIX_THREADS) { drow[j] = acc[j]; }
    }
}

template <bool WQ8, bool POST, bool NORM>
static void dsv4_hc_mix_launch(const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream,
        const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * scale, const ggml_tensor * base, ggml_tensor * dst,
        const float eps_norm, const float eps_hc, const int32_t n_iter, const dsv4_hc_mix_ext & ext) {
    dsv4_hc_mix_f32<WQ8, POST, NORM><<<grid, block, smem, stream>>>(
        (const float *) x->data, w->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
        (int) x->ne[0], x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), scale->nb[0] / sizeof(float), base->nb[0] / sizeof(float),
        dst->nb[1] / sizeof(float), eps_norm, eps_hc, n_iter, ext);
}

template <bool WQ8>
static void dsv4_hc_mix_dispatch(bool post, bool norm, const dim3 grid, const dim3 block, const size_t smem, cudaStream_t stream,
        const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * scale, const ggml_tensor * base, ggml_tensor * dst,
        const float eps_norm, const float eps_hc, const int32_t n_iter, const dsv4_hc_mix_ext & ext) {
    if (post && norm) {
        dsv4_hc_mix_launch<WQ8, true,  true >(grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else if (post) {
        dsv4_hc_mix_launch<WQ8, true,  false>(grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else if (norm) {
        dsv4_hc_mix_launch<WQ8, false, true >(grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else {
        dsv4_hc_mix_launch<WQ8, false, false>(grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    }
}

void ggml_cuda_op_dsv4_hc_mix_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * hc_post, const ggml_tensor * rms_norm, ggml_tensor * mul, bool write_out) {
    const ggml_tensor * x     = dst->src[0];
    const ggml_tensor * w     = dst->src[1];
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * base  = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_Q8_0 || w->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(x->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float));

    const int     n_embd   = (int) x->ne[0];
    const int     n_tokens = (int) x->ne[2];
    const int     hc_dim   = DSV4_HC*n_embd;
    GGML_ASSERT(x->ne[1] == DSV4_HC);
    GGML_ASSERT(hc_dim == DSV4_HC_MIX_THREADS * DSV4_HC_MIX_PER_THREAD);   // 4 x 4096 today; other widths need a second instance
    GGML_ASSERT(x->nb[1] % 16 == 0 && x->nb[2] % 16 == 0 && ((uintptr_t) x->data) % 16 == 0);

    const float   eps_norm = ((const float *) dst->op_params)[0];
    const float   eps_hc   = ((const float *) dst->op_params)[1];
    const int32_t n_iter   = ggml_get_op_params_i32(dst, 2);

    dsv4_hc_mix_ext ext = {};
    ext.write_out = 1;
    if (hc_post) {
        // the matcher (ggml_cuda_dsv4_hc_mix_fusable) has checked shapes, contiguity and alignment
        GGML_ASSERT(hc_post == x || hc_post == x->view_src);
        const ggml_tensor * px = hc_post->src[0], * pr = hc_post->src[1], * pp = hc_post->src[2], * pc = hc_post->src[3];
        ext.px    = (const float *) px->data; ext.spx1 = px->nb[1] / sizeof(float);
        ext.pres  = (const float *) pr->data; ext.spr1 = pr->nb[1] / sizeof(float); ext.spr2 = pr->nb[2] / sizeof(float);
        ext.ppost = (const float *) pp->data; ext.spp0 = pp->nb[0] / sizeof(float); ext.spp1 = pp->nb[1] / sizeof(float);
        ext.pcomb = (const float *) pc->data; ext.spc0 = pc->nb[0] / sizeof(float); ext.spc1 = pc->nb[1] / sizeof(float); ext.spc2 = pc->nb[2] / sizeof(float);
        ext.pdst  = (float *) x->data;
    }
    if (mul) {
        const ggml_tensor * nw = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
        memcpy(&ext.eps_rms, rms_norm->op_params, sizeof(float));
        ext.nw   = (const float *) nw->data;
        ext.ndst = (float *) mul->data;
        ext.snd1 = mul->nb[1] / sizeof(float);
        ext.q8   = ggml_is_contiguous(mul) ? ggml_cuda_q8_side_reserve_rows(ctx, mul, n_embd, n_tokens, n_embd) : nullptr;   // nullptr above 8 tokens
        ext.write_out = write_out ? 1 : 0;
    }

    constexpr int mix_dim = (2 + DSV4_HC)*DSV4_HC;
    const size_t smem = (n_embd + 32*mix_dim + mix_dim + DSV4_HC + 32) * sizeof(float);

    cudaStream_t stream = ctx.stream();
    const dim3 grid(n_tokens, 1, 1);
    const dim3 block(DSV4_HC_MIX_THREADS, 1, 1);
    if (w->type == GGML_TYPE_Q8_0) {
        dsv4_hc_mix_dispatch<true >(hc_post != nullptr, mul != nullptr, grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else {
        dsv4_hc_mix_dispatch<false>(hc_post != nullptr, mul != nullptr, grid, block, smem, stream, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    }
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_dsv4_hc_mix_fused(ctx, dst, nullptr, nullptr, nullptr, true);
}
