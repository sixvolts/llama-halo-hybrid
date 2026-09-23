#include "common.cuh"

#define CUDA_CONCAT_BLOCK_SIZE 256

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// halo-hybrid: KDA conv-input assembly at decode as one launch (see ggml_cuda_try_fuse_kda_conv_rows in ggml-cuda.cu):
//     conv_input[c] = [ states[0..ns-1, c], x[c, 0..nt-1] ] with x = concat(q, k, v) along channels, plus the K
//     rollback-slot copies dst_k[c*ns + j] = conv_input[c][s_idx_k + j]. f32 only.
#define KDA_CONV_ROWS_MAX_DST 8
struct ggml_cuda_kda_conv_rows_args {
    const ggml_tensor * q;
    const ggml_tensor * k;
    const ggml_tensor * v;
    const ggml_tensor * states;      // [ns, C] any strides
    ggml_tensor       * conv_input;  // [ns + nt, C] contiguous
    int                 n_dst;
    float             * dst[KDA_CONV_ROWS_MAX_DST];    // contiguous [ns*C] each
    int                 s_idx[KDA_CONV_ROWS_MAX_DST];  // first conv_input column copied into dst[k]
};

void ggml_cuda_op_kda_conv_rows(ggml_backend_cuda_context & ctx, const ggml_cuda_kda_conv_rows_args & args);
