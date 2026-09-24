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

// halo-hybrid: the same prologue at decode widths (<= 8 tokens) with the K dimension spread over the GPU. The
//     one-block-per-token kernel above puts the whole 417 KB hc_fn read (24 rows, 16384 wide) behind one WGP's
//     memory-level parallelism, ~19 us isolated on gfx1201 for what is a ~1 us read. Here grid = (hc_dim/KS, nt):
//     each 256-thread block reads a KS-wide slice of the token's activation (running the fused hc_post for that
//     slice when POST) and of all 24 rows, and writes 24 partial dots + a partial sum of squares; the block that
//     arrives last for a token (per-token counter, reset by that block) reduces the partials in fixed block order
//     (deterministic whichever block finishes last), runs the gates and the 4x4 sinkhorn on 16 lanes (DPP lane
//     exchanges, the serial loop's summation order, v_rcp_f32 for the 40 iteration reciprocals), the pre-mix and the
//     optional RMS_NORM*w + q8_1 tail from registers. The weight quants are loaded before the activation so the two
//     latencies overlap. Scratch: ctx.hc_mix_scratch (counters + partials, allocated and zeroed before any capture).
//     GGML_CUDA_NO_HC_MIX_SPLIT=1 keeps the one-block-per-token kernel.
#define DSV4_HC_SPLIT_T    256
#define DSV4_HC_SPLIT_KS   512   // flat elements per block: one wave's 32 lanes x 16 (half a q8_0 block each)
#define DSV4_HC_SPLIT_MAXT 8
#define DSV4_HC_SPLIT_PS   32    // partial records per token: 24 dots + the sum of squares, padded
#define DSV4_HC_SPLIT_MAXB 64    // blocks per token the scratch holds
#define DSV4_HC_SPLIT_SCRATCH (256 + DSV4_HC_SPLIT_MAXT*DSV4_HC_SPLIT_PS*DSV4_HC_SPLIT_MAXB*sizeof(float))

// lane exchanges inside a 16-lane row for the lane-parallel sinkhorn: DPP (no LDS round trip) on AMD
static __device__ __forceinline__ float dsv4_quad_bcast(const float v, const int k) {   // lane k of this quad; k constant
#if defined(GGML_USE_HIP)
    switch (k) {
        case 0:  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x00, 0xF, 0xF, true));
        case 1:  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x55, 0xF, 0xF, true));
        case 2:  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xAA, 0xF, 0xF, true));
        default: return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xFF, 0xF, 0xF, true));
    }
#else
    return __shfl_sync(0xffffffff, v, (threadIdx.x & ~3) + k, WARP_SIZE);
#endif
}

// the sinkhorn's 40 reciprocals are its critical path: v_rcp_f32 (1 ulp) instead of the ~11-instruction IEEE division
static __device__ __forceinline__ float dsv4_rcp(const float x) {
#if defined(GGML_USE_HIP)
    return __builtin_amdgcn_rcpf(x);
#else
    return 1.0f / x;
#endif
}

template <int M>
static __device__ __forceinline__ float dsv4_row_xmask(const float v) {   // lane ^ M, M < 16
// DPP row_xmask exists from gfx10 on (RDNA); gfx9 (Vega, CDNA) takes the shuffle
#if defined(GGML_USE_HIP) && defined(RDNA)
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + M, 0xF, 0xF, true));
#else
    return __shfl_xor_sync(0xffffffff, v, M, WARP_SIZE);
#endif
}

template <bool WQ8, bool POST, bool NORM>
static __global__ void __launch_bounds__(DSV4_HC_SPLIT_T) dsv4_hc_mix_split_f32(
        const float * __restrict__ x, const void * __restrict__ w, const float * __restrict__ scale, const float * __restrict__ base,
        float * __restrict__ dst,
        const int n_embd, const int64_t sx1, const int64_t sx2, const int64_t ss0, const int64_t sb0, const int64_t sd1,
        const float eps_norm, const float eps_hc, const int n_iter, const dsv4_hc_mix_ext ext,
        float * part, unsigned int * counters) {
    constexpr int hc      = DSV4_HC;
    constexpr int mix_dim = (2 + hc)*hc;
    constexpr int T       = DSV4_HC_SPLIT_T;
    constexpr int KS      = DSV4_HC_SPLIT_KS;
    constexpr int PS      = DSV4_HC_SPLIT_PS;
    constexpr int NW      = T/WARP_SIZE;          // waves; wave wv owns rows wv, wv + NW, wv + 2*NW
    constexpr int NL      = 16;                   // elements per lane in the dot stage and per thread in the tail
    static_assert(KS == WARP_SIZE*NL, "one wave spans the slice");
    static_assert(mix_dim % NW == 0, "rows split evenly over the waves");
    static_assert(mix_dim + 1 <= PS, "partial record");
    static_assert(2*NL == QK8_1, "a q8_1 block is two threads' columns");

    const int hc_dim = hc*n_embd;
    const int nb     = gridDim.x;                 // hc_dim / KS
    const int kb     = blockIdx.x;
    const int it     = blockIdx.y;
    const int tid    = threadIdx.x;
    const int lane   = tid % WARP_SIZE;
    const int wv     = tid / WARP_SIZE;

    __shared__ float xs[KS];
    __shared__ float red[NW];
    __shared__ float sums[mix_dim + 1];
    __shared__ float pre[hc];
    __shared__ int   s_last;

    // partial records of token it, [PS][nb]: the tail's lanes read one record's nb values contiguously
    float * tpart = part + (int64_t) it*PS*nb;

    const int k0 = kb*KS;
    const int h  = k0 / n_embd;                   // KS divides n_embd: the slice lies in one stream
    const int jb = k0 - h*n_embd;

    // 0. the q8_0 quants of this lane's rows first: their latency overlaps the activation load below
    constexpr int NR = mix_dim/NW;
    const int i0   = k0 + NL*lane;
    const int blk  = i0 / QK8_0;
    const int half = (i0 % QK8_0) / 16;
    int4  wq[NR];
    float wd[NR];
    if constexpr (WQ8) {
#pragma unroll
        for (int rr = 0; rr < NR; ++rr) {
            const block_q8_0 * b = (const block_q8_0 *) w + (int64_t) (wv + NW*rr) * (hc_dim / QK8_0) + blk;
            wq[rr] = *(const int4 *) (b->qs + 16*half);
            wd[rr] = __half2float(b->d);
        }
    }

    // 1. the slice of this token's activation, 2 elements per thread (the hc_post for them when POST)
    {
        const int e = 2*tid;
        float2 v;
        if constexpr (POST) {
            const float ph = ext.ppost[h*ext.spp0 + it*ext.spp1];
            const float2 p = *(const float2 *) (ext.px + it*ext.spx1 + jb + e);
            v.x = __fmul_rn(p.x, ph); v.y = __fmul_rn(p.y, ph);
#pragma unroll
            for (int isrc = 0; isrc < hc; ++isrc) {
                const float c  = ext.pcomb[h*ext.spc0 + isrc*ext.spc1 + it*ext.spc2];
                const float2 r = *(const float2 *) (ext.pres + it*ext.spr2 + isrc*ext.spr1 + jb + e);
                v.x = __fmaf_rn(r.x, c, v.x); v.y = __fmaf_rn(r.y, c, v.y);
            }
            *(float2 *) (ext.pdst + it*sx2 + h*sx1 + jb + e) = v;
        } else {
            v = *(const float2 *) (x + it*sx2 + h*sx1 + jb + e);
        }
        xs[e] = v.x; xs[e + 1] = v.y;
        float ss = v.x*v.x + v.y*v.y;
        ss = warp_reduce_sum(ss);
        if (lane == 0) { red[wv] = ss; }
    }
    __syncthreads();

    // 2. partial dots: lane = half a q8_0 block of the slice (16 elements), one 16-byte quant load per row
    {
        float xv[NL];
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            const float4 v = *(const float4 *) (xs + NL*lane + 4*q);
            xv[4*q + 0] = v.x; xv[4*q + 1] = v.y; xv[4*q + 2] = v.z; xv[4*q + 3] = v.w;
        }
        float p[NR];
#pragma unroll
        for (int rr = 0; rr < NR; ++rr) {
            const int r = wv + NW*rr;
            if constexpr (WQ8) {
                const int8_t * q = (const int8_t *) &wq[rr];
                float acc_q = 0.0f;
#pragma unroll
                for (int k = 0; k < NL; ++k) { acc_q += (float) q[k] * xv[k]; }
                p[rr] = wd[rr] * acc_q;
            } else {
                const float4 * wr = (const float4 *) ((const float *) w + (int64_t) r * hc_dim + i0);
                float s = 0.0f;
#pragma unroll
                for (int q = 0; q < NL/4; ++q) {
                    const float4 v = wr[q];
                    s += v.x*xv[4*q] + v.y*xv[4*q + 1] + v.z*xv[4*q + 2] + v.w*xv[4*q + 3];
                }
                p[rr] = s;
            }
        }
#pragma unroll
        for (int rr = 0; rr < NR; ++rr) {
            const float s = warp_reduce_sum(p[rr]);
            if (lane == 0) { tpart[(wv + NW*rr)*nb + kb] = s; }
        }
        if (tid == 0) {
            float s = 0.0f;
#pragma unroll
            for (int i = 0; i < NW; ++i) { s += red[i]; }
            tpart[mix_dim*nb + kb] = s;
        }
    }

    // 3. publish the partials; the last block of this token to arrive runs the tail
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        const unsigned int prev = atomicAdd(&counters[it], 1u);
        const int last = prev == (unsigned int) (nb - 1);
        if (last) { counters[it] = 0; }   // all nb blocks have arrived: ready for the next launch
        s_last = last;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();   // acquire: the other blocks' partials (and hc_post rows) through a clean L0

    // 4. this thread's 16 consecutive columns of all 4 streams, loaded before the reduction needs them
    const float * xrow = (POST ? (const float *) ext.pdst : x) + it*sx2;
    const int j0 = NL*tid;
    float xr[hc][NL];
#pragma unroll
    for (int hh = 0; hh < hc; ++hh) {
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            const float4 v = *(const float4 *) (xrow + hh*sx1 + j0 + 4*q);
            xr[hh][4*q + 0] = v.x; xr[hh][4*q + 1] = v.y; xr[hh][4*q + 2] = v.z; xr[hh][4*q + 3] = v.w;
        }
    }

    // 5. reduce the records in block order; wave wv takes records wv, wv + NW, ...
    for (int r = wv; r < mix_dim + 1; r += NW) {
        float s = 0.0f;
        for (int b = lane; b < nb; b += WARP_SIZE) { s += tpart[r*nb + b]; }
        s = warp_reduce_sum(s);
        if (lane == 0) { sums[r] = s; }
    }
    __syncthreads();

    // 6. gates and the 4x4 sinkhorn on wave 0, one comb element per lane (lanes 16..31 mirror 0..15); every sum is
    //    taken in the serial loop's order, so all lanes of a row/column agree on its normaliser
    float * drow = dst + it*sd1;
    if (wv == 0) {
        const float rms_inv = rsqrtf(sums[mix_dim] / (float) hc_dim + eps_norm);
        const float scale_pre = scale[0], scale_post = scale[ss0], scale_comb = scale[2*ss0];
        if (lane < hc) {
            pre[lane] = 1.0f / (1.0f + expf(-((sums[lane]*rms_inv)*scale_pre + base[lane*sb0]))) + eps_hc;
        } else if (lane < 2*hc) {
            drow[n_embd + lane - hc] = 2.0f / (1.0f + expf(-((sums[lane]*rms_inv)*scale_post + base[lane*sb0])));
        }
        const int idx  = lane & (hc*hc - 1);
        const int isrc = idx / hc;
        float c = (sums[2*hc + idx]*rms_inv)*scale_comb + base[(2*hc + idx)*sb0];
        // row isrc = the 4 lanes of a quad: element k of the row is a quad broadcast of lane k. column idst = lanes
        // idst + 4*s' of the 16-lane row: element k is at lane ^ 4*(k ^ isrc)
        float mx = -INFINITY;
#pragma unroll
        for (int k = 0; k < hc; ++k) { mx = fmaxf(mx, dsv4_quad_bcast(c, k)); }
        c = expf(c - mx);
        {
            float sum = 0.0f;
#pragma unroll
            for (int k = 0; k < hc; ++k) { sum += dsv4_quad_bcast(c, k); }
            const float inv_sum = 1.0f / sum;
            c = c*inv_sum + eps_hc;
        }
        auto norm_cols = [&]() {
            const float x1 = dsv4_row_xmask<4>(c), x2 = dsv4_row_xmask<8>(c), x3 = dsv4_row_xmask<12>(c);
            float sum = eps_hc;
#pragma unroll
            for (int k = 0; k < hc; ++k) {
                const int j = k ^ isrc;
                sum += j == 0 ? c : (j == 1 ? x1 : (j == 2 ? x2 : x3));
            }
            const float inv_sum = dsv4_rcp(sum);
            c *= inv_sum;
        };
        auto norm_rows = [&]() {
            float sum = eps_hc;
#pragma unroll
            for (int k = 0; k < hc; ++k) { sum += dsv4_quad_bcast(c, k); }
            const float inv_sum = dsv4_rcp(sum);
            c *= inv_sum;
        };
        norm_cols();
        for (int i = 1; i < n_iter; ++i) { norm_rows(); norm_cols(); }
        if (lane < hc*hc) { drow[n_embd + hc + idx] = c; }
    }
    __syncthreads();

    // 7. out = sum_h pre[h] * x[:, h], the streams accumulated in order
    float o[NL];
#pragma unroll
    for (int k = 0; k < NL; ++k) { o[k] = 0.0f; }
#pragma unroll
    for (int hh = 0; hh < hc; ++hh) {
        const float ph = pre[hh];
#pragma unroll
        for (int k = 0; k < NL; ++k) { o[k] += ph * xr[hh][k]; }
    }
    if (!NORM || ext.write_out) {
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            *(float4 *) (drow + j0 + 4*q) = make_float4(o[4*q + 0], o[4*q + 1], o[4*q + 2], o[4*q + 3]);
        }
    }
    if constexpr (NORM) {
        // 8. RMS_NORM + MUL(w) on the finished row, from registers; a q8_1 block (32 columns) is this thread's 16 and
        //    its neighbour's, quantized as q8_side_store does it (amax/127, roundf, the block sum in ds.y)
        float tmp = 0.0f;
#pragma unroll
        for (int k = 0; k < NL; ++k) { tmp += o[k] * o[k]; }
        tmp = block_reduce<block_reduce_method::SUM, T>(tmp, red);
        const float mean = tmp / n_embd;
        const float scl  = rsqrtf(mean + ext.eps_rms);
        float v[NL];
        float amax = 0.0f, sum = 0.0f;
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            const float4 nw4 = *(const float4 *) (ext.nw + j0 + 4*q);
            v[4*q + 0] = scl * o[4*q + 0] * nw4.x; v[4*q + 1] = scl * o[4*q + 1] * nw4.y;
            v[4*q + 2] = scl * o[4*q + 2] * nw4.z; v[4*q + 3] = scl * o[4*q + 3] * nw4.w;
        }
        float * nrow = ext.ndst + it*ext.snd1;
#pragma unroll
        for (int q = 0; q < NL/4; ++q) {
            *(float4 *) (nrow + j0 + 4*q) = make_float4(v[4*q + 0], v[4*q + 1], v[4*q + 2], v[4*q + 3]);
        }
        if (ext.q8) {
#pragma unroll
            for (int k = 0; k < NL; ++k) { amax = fmaxf(amax, fabsf(v[k])); sum += v[k]; }
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1, WARP_SIZE));
            sum += __shfl_xor_sync(0xffffffff, sum, 1, WARP_SIZE);
            const float d = amax / 127.0f;
            const int64_t col0 = (int64_t) it*n_embd + j0;
            block_q8_1 * b = ext.q8 + col0 / QK8_1;
            int8_t qs[NL];
#pragma unroll
            for (int k = 0; k < NL; ++k) { qs[k] = amax == 0.0f ? 0 : (int8_t) roundf(v[k] / d); }
#pragma unroll
            for (int k = 0; k < NL/4; ++k) {   // qs is only 4-byte aligned in block_q8_1
                *(int *) (b->qs + col0 % QK8_1 + 4*k) = *(const int *) (qs + 4*k);
            }
            if (col0 % QK8_1 == 0) {
                b->ds = make_half2(d, sum);
            }
        }
    }
}

template <bool WQ8, bool POST, bool NORM>
static void dsv4_hc_mix_split_launch(const int nt, cudaStream_t stream, float * part, unsigned int * counters,
        const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * scale, const ggml_tensor * base, ggml_tensor * dst,
        const float eps_norm, const float eps_hc, const int32_t n_iter, const dsv4_hc_mix_ext & ext) {
    const int n_embd = (int) x->ne[0];
    const dim3 grid(DSV4_HC*n_embd / DSV4_HC_SPLIT_KS, nt, 1);
    dsv4_hc_mix_split_f32<WQ8, POST, NORM><<<grid, DSV4_HC_SPLIT_T, 0, stream>>>(
        (const float *) x->data, w->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
        n_embd, x->nb[1] / sizeof(float), x->nb[2] / sizeof(float), scale->nb[0] / sizeof(float), base->nb[0] / sizeof(float),
        dst->nb[1] / sizeof(float), eps_norm, eps_hc, n_iter, ext, part, counters);
}

template <bool WQ8>
static void dsv4_hc_mix_split_dispatch(bool post, bool norm, const int nt, cudaStream_t stream, float * part, unsigned int * counters,
        const ggml_tensor * x, const ggml_tensor * w, const ggml_tensor * scale, const ggml_tensor * base, ggml_tensor * dst,
        const float eps_norm, const float eps_hc, const int32_t n_iter, const dsv4_hc_mix_ext & ext) {
    if (post && norm) {
        dsv4_hc_mix_split_launch<WQ8, true,  true >(nt, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else if (post) {
        dsv4_hc_mix_split_launch<WQ8, true,  false>(nt, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else if (norm) {
        dsv4_hc_mix_split_launch<WQ8, false, true >(nt, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    } else {
        dsv4_hc_mix_split_launch<WQ8, false, false>(nt, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
    }
}

void ggml_cuda_dsv4_hc_mix_scratch_init(ggml_backend_cuda_context & ctx) {
    static const bool disabled = getenv("GGML_CUDA_NO_HC_MIX_SPLIT") != nullptr && atoi(getenv("GGML_CUDA_NO_HC_MIX_SPLIT"));
    if (disabled || ctx.hc_mix_scratch != nullptr) {
        return;
    }
    void * p = nullptr;
    if (cudaMalloc(&p, DSV4_HC_SPLIT_SCRATCH) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(p, 0, DSV4_HC_SPLIT_SCRATCH, ctx.stream()));   // ordered before every kernel on this stream
    ctx.hc_mix_scratch = p;
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
    if (ctx.hc_mix_scratch != nullptr && n_tokens <= DSV4_HC_SPLIT_MAXT && n_embd == DSV4_HC_SPLIT_T*16 &&
            n_embd % DSV4_HC_SPLIT_KS == 0 && hc_dim / DSV4_HC_SPLIT_KS <= DSV4_HC_SPLIT_MAXB &&
            dst->nb[1] % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0 &&
            (!mul || (((uintptr_t) ext.nw) % 16 == 0 && ((uintptr_t) ext.ndst) % 16 == 0 && ext.snd1 % 4 == 0))) {
        unsigned int * counters = (unsigned int *) ctx.hc_mix_scratch;
        float * part = (float *) ((char *) ctx.hc_mix_scratch + 256);
        if (w->type == GGML_TYPE_Q8_0) {
            dsv4_hc_mix_split_dispatch<true >(hc_post != nullptr, mul != nullptr, n_tokens, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
        } else {
            dsv4_hc_mix_split_dispatch<false>(hc_post != nullptr, mul != nullptr, n_tokens, stream, part, counters, x, w, scale, base, dst, eps_norm, eps_hc, n_iter, ext);
        }
        return;
    }
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
