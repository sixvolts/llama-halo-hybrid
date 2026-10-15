#pragma once
// bf16 WMMA chunked GATED_DELTA_NET prefill (S_v == 128), ported from stew675/llama-cpp-rdna-boosts block 02 (block 03 there since):
// gated_delta_net_chunked_bf16.cu (gfx12) and gated_delta_net_chunked_bf16_gfx11.cu (gfx11). Plain layouts only.
// Derived from libr4d (Apache-2.0) via rdna-boosts (MIT): licenses/LICENSE-radiance-libr4d, LICENSE-llama-cpp-rdna-boosts.
#include "common.cuh"

template<typename Kernel, typename... Args>
static __inline__ bool ggml_cuda_kernel_launch_try(Kernel kernel, const ggml_cuda_kernel_launch_params & launch_params, Args&&... args) {
    kernel<<<launch_params.block_nums, launch_params.block_dims, launch_params.shmem, launch_params.stream>>>(std::forward<Args>(args)... );
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_gated_delta_net_chunked_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext = nullptr, int64_t n_tokens_limit = 0, const float * s_d_ext = nullptr);

bool ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_d_ext = nullptr, int64_t n_tokens_limit = 0, const float * s_d_ext = nullptr);
