#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// halo-hybrid: concat(w_q, w_k, w_v) -> ssm_conv -> silu -> l2_norm(Q heads), l2_norm(K heads) as one launch (decode)
bool ggml_cuda_ssm_conv_kda_l2_supported(int64_t d_conv, int64_t d_inner, int64_t n_t);
void ggml_cuda_op_ssm_conv_kda_l2(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_in,
        const ggml_tensor * w_q, const ggml_tensor * w_k, const ggml_tensor * w_v,
        ggml_tensor * silu_dst, ggml_tensor * q_out, ggml_tensor * k_out, float eps);

// halo-hybrid: the GDN conv front at decode (qwen4exp build_conv_state_at + ssm_conv + silu + one l2_norm over the
// leading 128-wide heads of the SiLU output) as one launch. See ggml_cuda_try_fuse_gdn_conv_front in ggml-cuda.cu.
#define GDN_CONV_FRONT_MAX_DST    8
#define GDN_CONV_FRONT_MAX_TOKENS 8
#define GDN_CONV_FRONT_HEAD       128

struct ggml_cuda_gdn_conv_front_args {
    const ggml_tensor * states;   // [d_conv-1, C, 1] conv history, any element/row strides
    const ggml_tensor * xt;       // transposed view [nt, C, 1] of the new columns: x(t, c) at data + t*nb[0] + c*4
    const ggml_tensor * w;        // [d_conv, C] conv weight
    ggml_tensor       * y;        // SiLU output [C, nt, 1]
    ggml_tensor       * l2;       // L2-normalised leading heads, contiguous [128, H, nt, 1]
    float               eps;
    int                 n_dst;    // rollback slots
    float             * dst[GDN_CONV_FRONT_MAX_DST];    // contiguous [(d_conv-1)*C] each
    int                 s_idx[GDN_CONV_FRONT_MAX_DST];  // first conv_input column copied into dst[k]
};

bool ggml_cuda_gdn_conv_front_supported(int64_t d_conv, int64_t C, int64_t nt);
void ggml_cuda_op_gdn_conv_front(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_front_args & args);
