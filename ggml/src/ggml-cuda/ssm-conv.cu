#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// halo-hybrid: the KDA conv tail at decode as one launch (ggml_cuda_op_ssm_conv_kda_l2). The graph builds the conv
// weight as concat(concat(w_q, w_k), w_v) every step and L2-normalises the Q and K heads of the SiLU output in two
// more launches. Here each 128-channel block picks its rows straight from w_q / w_k / w_v, writes the full SiLU
// output (V is read through a view of it) and, for Q and K blocks (one block = one 128-wide head), writes the L2
// normalised head. The norm reproduces l2_norm_f32<WARP_SIZE> exactly: one 32-lane warp per token row, each lane
// summing columns lane, lane+32, lane+64, lane+96 in that order, then warp_reduce_sum.
#define KDA_L2_HEAD      128
#define KDA_L2_MAX_TOKENS 8

template <size_t d_conv>
static __global__ void __launch_bounds__(KDA_L2_HEAD) ssm_conv_kda_l2_f32(
        const float * __restrict__ src0, const float * __restrict__ w_q, const float * __restrict__ w_k,
        const float * __restrict__ w_v, const int src0_nb1, const int src0_nb2, const int d_inner,
        float * __restrict__ dst, const int dst_nb1, const int dst_nb2,
        float * __restrict__ q_out, float * __restrict__ k_out, const int qk_nb2, const int qk_nb3,
        const int n_t, const float eps) {
    static_assert(KDA_L2_HEAD == 4*WARP_SIZE, "one head = four warps");
    __shared__ float s_y[KDA_L2_MAX_TOKENS][KDA_L2_HEAD];

    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;   // sequence
    const int bidy = blockIdx.y;   // 128-channel block of the 3*d_inner conv channels

    const int c0    = bidy * KDA_L2_HEAD;
    const int which = c0 / d_inner;           // 0 = q, 1 = k, 2 = v
    const int row0  = c0 - which * d_inner;   // first channel within that weight
    const float * w_block = (which == 0 ? w_q : which == 1 ? w_k : w_v) + (int64_t) row0 * d_conv;

    const float * x_block = (const float *) ((const char *) src0 + (int64_t) bidx * src0_nb2 + (int64_t) c0 * src0_nb1);
    float       * y_block = (float *) ((char *) dst + (int64_t) bidx * dst_nb2) + c0;

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * d_conv + j];
    }

    for (int i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += 0.0f;   // the unfused kernel's (zero) bias term, kept so -0.0 rounds the same way
        const float y = ggml_cuda_op_silu_single(sumf);
        y_block[i * stride_y + tid] = y;
        if (which < 2) {
            s_y[i][tid] = y;
        }
    }

    if (which >= 2) {
        return;
    }
    __syncthreads();

    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    float * out = (which == 0 ? q_out : k_out) + (int64_t) bidx * qk_nb3 + row0;
    for (int i = warp; i < n_t; i += KDA_L2_HEAD / WARP_SIZE) {
        // l2_norm_f32 runs this as a rolled loop of fmac into a zeroed accumulator: x0*x0, then fma per column.
        // Written out with explicit fmaf, because unrolled `tmp += xi*xi` lets the compiler fuse the first two
        // terms the other way round (fma(x0, x0, x32*x32)), which moves the scale by an ulp on some heads.
        float tmp = s_y[i][lane] * s_y[i][lane];
#pragma unroll
        for (int col = lane + WARP_SIZE; col < KDA_L2_HEAD; col += WARP_SIZE) {
            const float xi = s_y[i][col];
            tmp = fmaf(xi, xi, tmp);
        }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
        for (int col = lane; col < KDA_L2_HEAD; col += WARP_SIZE) {
            out[(int64_t) i * qk_nb2 + col] = scale * s_y[i][col];
        }
    }
}

bool ggml_cuda_ssm_conv_kda_l2_supported(int64_t d_conv, int64_t d_inner, int64_t n_t) {
    return (d_conv == 3 || d_conv == 4) && d_inner % KDA_L2_HEAD == 0 && n_t >= 1 && n_t <= KDA_L2_MAX_TOKENS;
}

void ggml_cuda_op_ssm_conv_kda_l2(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_in,
        const ggml_tensor * w_q, const ggml_tensor * w_k, const ggml_tensor * w_v,
        ggml_tensor * silu_dst, ggml_tensor * q_out, ggml_tensor * k_out, float eps) {
    const int64_t d_conv  = w_q->ne[0];
    const int64_t d_inner = w_q->ne[1];
    const int64_t n_t     = silu_dst->ne[1];
    const int64_t n_s     = silu_dst->ne[2];

    GGML_ASSERT(ggml_cuda_ssm_conv_kda_l2_supported(d_conv, d_inner, n_t));
    GGML_ASSERT(silu_dst->ne[0] == 3*d_inner && conv_in->ne[1] == 3*d_inner);
    GGML_ASSERT(conv_in->nb[0] == sizeof(float) && conv_in->nb[1] == conv_in->ne[0]*sizeof(float));
    GGML_ASSERT(silu_dst->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(q_out) && ggml_is_contiguous(k_out));

    const float * src0_d = (const float *) conv_in->data;
    const float * wq_d   = (const float *) w_q->data;
    const float * wk_d   = (const float *) w_k->data;
    const float * wv_d   = (const float *) w_v->data;
    float * dst_d = (float *) silu_dst->data;
    float * q_d   = (float *) q_out->data;
    float * k_d   = (float *) k_out->data;

    const int qk_nb2 = q_out->nb[2] / sizeof(float);   // token stride of the normed heads
    const int qk_nb3 = q_out->nb[3] / sizeof(float);   // sequence stride

    const dim3 blocks(n_s, 3*d_inner / KDA_L2_HEAD, 1);
    cudaStream_t stream = ctx.stream();
    if (d_conv == 4) {
        ssm_conv_kda_l2_f32<4><<<blocks, KDA_L2_HEAD, 0, stream>>>(src0_d, wq_d, wk_d, wv_d, conv_in->nb[1], conv_in->nb[2],
            d_inner, dst_d, silu_dst->nb[1], silu_dst->nb[2], q_d, k_d, qk_nb2, qk_nb3, n_t, eps);
    } else {
        ssm_conv_kda_l2_f32<3><<<blocks, KDA_L2_HEAD, 0, stream>>>(src0_d, wq_d, wk_d, wv_d, conv_in->nb[1], conv_in->nb[2],
            d_inner, dst_d, silu_dst->nb[1], silu_dst->nb[2], q_d, k_d, qk_nb2, qk_nb3, n_t, eps);
    }
}

// halo-hybrid: the GDN conv front at decode as one launch (ggml_cuda_try_fuse_gdn_conv_front). qwen4exp emits, per
// DeltaNet layer and step: conv_input = concat(states, transpose(qkv_mixed), 0), K rollback-slot copies of its last
// d_conv-1 columns into the conv cache, ssm_conv + silu (one launch already) and one l2_norm over the Q and K heads
// (a view of the SiLU output at offset 0). Here one 128-thread block owns 128 channels (= one 128-wide head): each
// thread assembles its channel's window in registers (state columns, then the nt new values), writes the slot
// windows, runs the conv + SiLU and, when the block is one of the H normalised heads, the norm. conv_input itself is
// never materialised (the matcher requires the copies and the conv to be its only readers).
// Arithmetic mirrors ssm_conv_f32 (sum from 0 in tap order, plus the zero bias) and l2_norm_f32<WARP_SIZE> (one warp
// per token row, lane sums columns lane, +32, +64, +96 as x0*x0 then fmaf, then warp_reduce_sum), as in
// ssm_conv_kda_l2_f32.
struct gdn_conv_front_dst {
    float * d[GDN_CONV_FRONT_MAX_DST];
    int     s_idx[GDN_CONV_FRONT_MAX_DST];
};

template <int d_conv>
static __global__ void __launch_bounds__(GDN_CONV_FRONT_HEAD) gdn_conv_front_f32(
        const char * st, const int64_t nbs0, const int64_t nbs1,
        const char * __restrict__ x, const int64_t nbx_t,
        const float * __restrict__ w, const int64_t stride_w,
        float * __restrict__ y, const int64_t stride_y,
        float * __restrict__ l2, const int64_t l2_stride_t, const int n_heads,
        const int nt, const float eps, const int n_dst, const gdn_conv_front_dst out) {
    static_assert(GDN_CONV_FRONT_HEAD == 4*WARP_SIZE, "one head = four warps");
    constexpr int ns   = d_conv - 1;
    constexpr int MAXT = GDN_CONV_FRONT_MAX_TOKENS;
    constexpr int MAXW = ns + MAXT;
    __shared__ float s_y[MAXT][GDN_CONV_FRONT_HEAD];

    const int tid  = threadIdx.x;
    const int head = blockIdx.x;
    const int64_t c = (int64_t) head*GDN_CONV_FRONT_HEAD + tid;

    // the channel's window: its conv history, then the new columns. Every read of the history happens before any
    // slot write below; a slot may be the history's own memory (build_rs's single-slot view), channel for channel.
    float win[MAXW];
#pragma unroll
    for (int j = 0; j < ns; ++j) {
        win[j] = *(const float *) (st + c*nbs1 + j*nbs0);
    }
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        win[ns + t] = t < nt ? *(const float *) (x + t*nbx_t + c*(int64_t) sizeof(float)) : 0.0f;
    }

    float wr[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        wr[j] = w[c*stride_w + j];
    }

    for (int k = 0; k < n_dst; ++k) {
        float * d = out.d[k] + c*ns;
        const int s = out.s_idx[k];
#pragma unroll
        for (int p = 0; p < MAXW; ++p) {
            if (p >= s && p < s + ns) {
                d[p - s] = win[p];
            }
        }
    }

    const bool norm = head < n_heads;
#pragma unroll
    for (int i = 0; i < MAXT; ++i) {
        if (i < nt) {
            float sumf = 0.0f;
#pragma unroll
            for (int j = 0; j < d_conv; ++j) {
                sumf += win[i + j] * wr[j];
            }
            sumf += 0.0f;   // the unfused kernel's (zero) bias term
            const float v = ggml_cuda_op_silu_single(sumf);
            y[i*stride_y + c] = v;
            if (norm) {
                s_y[i][tid] = v;
            }
        }
    }

    if (!norm) {
        return;
    }
    __syncthreads();

    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    float * o = l2 + (int64_t) head*GDN_CONV_FRONT_HEAD;
    for (int i = warp; i < nt; i += GDN_CONV_FRONT_HEAD / WARP_SIZE) {
        float tmp = s_y[i][lane] * s_y[i][lane];
#pragma unroll
        for (int col = lane + WARP_SIZE; col < GDN_CONV_FRONT_HEAD; col += WARP_SIZE) {
            const float xi = s_y[i][col];
            tmp = fmaf(xi, xi, tmp);
        }
        tmp = warp_reduce_sum(tmp);
        const float scale = rsqrtf(fmaxf(tmp, eps * eps));
#pragma unroll
        for (int col = lane; col < GDN_CONV_FRONT_HEAD; col += WARP_SIZE) {
            o[(int64_t) i*l2_stride_t + col] = scale * s_y[i][col];
        }
    }
}

bool ggml_cuda_gdn_conv_front_supported(int64_t d_conv, int64_t C, int64_t nt) {
    return (d_conv == 3 || d_conv == 4) && C > 0 && C % GDN_CONV_FRONT_HEAD == 0 && C <= INT_MAX / 16 &&
           nt >= 1 && nt <= GDN_CONV_FRONT_MAX_TOKENS;
}

void ggml_cuda_op_gdn_conv_front(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_front_args & a) {
    const int64_t d_conv = a.w->ne[0];
    const int64_t C      = a.w->ne[1];
    const int64_t nt     = a.y->ne[1];
    GGML_ASSERT(ggml_cuda_gdn_conv_front_supported(d_conv, C, nt));
    GGML_ASSERT(a.xt->nb[1] == sizeof(float) && a.w->nb[0] == sizeof(float) && a.y->nb[0] == sizeof(float));
    GGML_ASSERT(ggml_is_contiguous(a.l2) && a.l2->ne[0] == GDN_CONV_FRONT_HEAD && a.l2->ne[1]*GDN_CONV_FRONT_HEAD <= C);
    GGML_ASSERT(a.n_dst >= 0 && a.n_dst <= GDN_CONV_FRONT_MAX_DST);

    gdn_conv_front_dst out = {};
    for (int k = 0; k < a.n_dst; ++k) {
        out.d[k]     = a.dst[k];
        out.s_idx[k] = a.s_idx[k];
    }
    const int n_heads = (int) a.l2->ne[1];
    const dim3 blocks(C / GDN_CONV_FRONT_HEAD, 1, 1);
    cudaStream_t stream = ctx.stream();
#define GDN_CONV_FRONT_LAUNCH(DC) \
    gdn_conv_front_f32<DC><<<blocks, GDN_CONV_FRONT_HEAD, 0, stream>>>( \
        (const char *) a.states->data, a.states->nb[0], a.states->nb[1], \
        (const char *) a.xt->data, a.xt->nb[0], \
        (const float *) a.w->data, a.w->nb[1] / sizeof(float), \
        (float *) a.y->data, a.y->nb[1] / sizeof(float), \
        (float *) a.l2->data, a.l2->nb[2] / sizeof(float), n_heads, \
        (int) nt, a.eps, a.n_dst, out)
    if (d_conv == 4) {
        GDN_CONV_FRONT_LAUNCH(4);
    } else {
        GDN_CONV_FRONT_LAUNCH(3);
    }
#undef GDN_CONV_FRONT_LAUNCH
}
