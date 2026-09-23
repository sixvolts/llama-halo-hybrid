#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// halo-hybrid: concat(w_q, w_k, w_v) -> ssm_conv -> silu -> l2_norm(Q heads), l2_norm(K heads) as one launch (decode)
bool ggml_cuda_ssm_conv_kda_l2_supported(int64_t d_conv, int64_t d_inner, int64_t n_t);
void ggml_cuda_op_ssm_conv_kda_l2(ggml_backend_cuda_context & ctx, const ggml_tensor * conv_in,
        const ggml_tensor * w_q, const ggml_tensor * w_k, const ggml_tensor * w_v,
        ggml_tensor * silu_dst, ggml_tensor * q_out, ggml_tensor * k_out, float eps);
