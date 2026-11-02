#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// halo-hybrid: concat(w_q, w_k, w_v) -> ssm_conv -> silu -> l2_norm(Q heads), l2_norm(K heads) as one launch (decode)
bool ggml_cuda_ssm_conv_kda_l2_supported(int64_t d_conv, int64_t d_inner, int64_t n_t);
void ggml_cuda_op_ssm_conv_kda_l2(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_in,
        const ggml_tensor * w_q, const ggml_tensor * w_k, const ggml_tensor * w_v,
        ggml_tensor * silu_dst, ggml_tensor * q_out, ggml_tensor * k_out, float eps);

// halo-hybrid: the GDN conv front at decode and prefill (qwen4exp build_conv_state_at + ssm_conv + silu + one l2_norm over the
// leading 128-wide heads of the SiLU output) as one launch. See ggml_cuda_try_fuse_gdn_conv_front in ggml-cuda.cu.
#define GDN_CONV_FRONT_MAX_DST    8
#define GDN_CONV_FRONT_MAX_TOKENS 8
#define GDN_CONV_FRONT_HEAD       128
#define GDN_CONV_FRONT_LONG_T     32   // tokens per block of the prefill kernel (nt > GDN_CONV_FRONT_MAX_TOKENS)

struct ggml_cuda_gdn_conv_front_args {
    const ggml_tensor * states;   // [d_conv-1, C, 1] conv history, any element/row strides
    const ggml_tensor * xt;       // transposed view [nt, C, 1] of the new columns: x(t, c) at data + t*nb[0] + c*4
    const ggml_tensor * w;        // [d_conv, C] conv weight
    ggml_tensor       * y;        // SiLU output [C, nt, 1]
    ggml_tensor       * l2;       // L2-normalised leading heads, contiguous [128, H, nt, 1]
    float               eps;
    bool                y_skip_l2;   // the l2-normalised heads' SiLU columns have no reader besides the norm: not written
    bool                head_major;  // y_skip_l2 and the GDN scan is the only reader: the SiLU V heads and the l2 output are
                                     // written head-major ([head][nt][128]) in their own memory (ggml_cuda_gdn_head_major)
    int                 n_dst;    // rollback slots
    float             * dst[GDN_CONV_FRONT_MAX_DST];    // contiguous [(d_conv-1)*C] each
    int                 s_idx[GDN_CONV_FRONT_MAX_DST];  // first conv_input column copied into dst[k]
};

// where the conv front writes: token / head strides in floats; y is offset so that y + h*y_hs + t*y_ts is V head h
struct ggml_cuda_gdn_conv_front_out {
    float * y;
    int64_t y_ts, y_hs;
    float * l2;
    int64_t l2_ts, l2_hs;
};
ggml_cuda_gdn_conv_front_out ggml_cuda_gdn_conv_front_out_layout(const ggml_cuda_gdn_conv_front_args & a);
bool ggml_cuda_gdn_conv_front_supported(int64_t d_conv, int64_t C, int64_t nt);
void ggml_cuda_op_gdn_conv_front(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_front_args & args);
// the same front for only the first d_conv-1 tokens of every `tile`-token block (the rest computed by the qkv GEMM's conv
// epilogue, mmq-wmma.cu); chunk 0 still writes the slots. x must hold those tokens, the d_conv-1 before each block and the
// last d_conv-1 of the batch.
void ggml_cuda_op_gdn_conv_front_tiles(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_front_args & args, int tile);

#include "unary.cuh"

// conv taps + SiLU and the l2 scale of one 128-wide head row, shared by the conv front kernels and the GEMM epilogue so
// every path rounds the same way: ssm_conv_f32's sum from 0 in tap order plus its (zero) bias, and l2_norm_f32<WARP_SIZE>'s
// order (lane sums columns lane, +32, +64, +96 as x0*x0 then fmaf, then warp_reduce_sum)
template <int d_conv>
static __device__ __forceinline__ float gdn_conv_silu(const float * win, const float * wr) {
    float sumf = 0.0f;
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        sumf += win[j] * wr[j];
    }
    sumf += 0.0f;   // the unfused kernel's (zero) bias term
    return ggml_cuda_op_silu_single(sumf);
}

static __device__ __forceinline__ float gdn_conv_l2_scale(const float * row, const int lane, const float eps) {
    float tmp = row[lane] * row[lane];
#pragma unroll
    for (int col = lane + WARP_SIZE; col < GDN_CONV_FRONT_HEAD; col += WARP_SIZE) {
        const float xi = row[col];
        tmp = fmaf(xi, xi, tmp);
    }
    tmp = warp_reduce_sum(tmp);
    return rsqrtf(fmaxf(tmp, eps * eps));
}
