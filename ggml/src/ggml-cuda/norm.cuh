#include "common.cuh"

// halo-hybrid: the one order every RMS norm here sums its squares in (upstream's float4 rms_norm, 623c16393, plus the
// fused kernels that replace RMS_NORM: hc.cu, dsv4-hc.cu, rope.cu). A row with ncols % 4 == 0 and at most 16K columns
// is read as float4 groups by rms_norm_vec_block(ncols) threads, rms_norm_vec_per_thread(ncols) groups each, every
// group summed by rms_norm_sumsq4 (explicit fmaf: no contraction choice left to the compiler); other rows keep the
// strided scalar loop. Threads at or past the block width add nothing, and block_reduce is a fixed butterfly in which
// absent warps count as zero, so a fused kernel with a wider block reduces to the same bits: fused and unfused launches
// of the same row agree exactly, which the scheduler checks rely on (a split can stop a fusion on one side).
static __host__ __device__ constexpr bool rms_norm_vec_order(const int ncols) {
    return ncols > 0 && ncols % 4 == 0 && ncols / 4 <= 4096;
}

static __host__ __device__ constexpr int rms_norm_vec_block(const int ncols) {
    return ncols / 4 <= 32 ? 32 : ncols / 4 <= 128 ? 128 : ncols / 4 <= 256 ? 256 : ncols / 4 <= 512 ? 512 : 1024;
}

static __host__ __device__ constexpr int rms_norm_vec_per_thread(const int ncols) {
    return ncols / 4 <= 1024 ? 1 : 4;
}

static __device__ __forceinline__ float rms_norm_sumsq4(const float4 v) {
    float s = v.x*v.x;
    s = fmaf(v.y, v.y, s);
    s = fmaf(v.z, v.z, s);
    return fmaf(v.w, v.w, s);
}

// the per-thread partial of the order above, for a row read with scalar loads (global or shared memory, any alignment)
static __device__ __forceinline__ float rms_norm_vec_partial(const float * row, const int ncols, const int tid) {
    const int n4  = ncols / 4;
    const int vb  = rms_norm_vec_block(ncols);
    const int vpt = rms_norm_vec_per_thread(ncols);
    float tmp = 0.0f;
    if (tid < vb) {
        for (int k = 0; k < vpt; k++) {
            const int i = tid + k*vb;
            if (i < n4) {
                tmp += rms_norm_sumsq4(make_float4(row[4*i + 0], row[4*i + 1], row[4*i + 2], row[4*i + 3]));
            }
        }
    }
    return tmp;
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
