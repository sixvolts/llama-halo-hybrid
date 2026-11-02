#include "ple-conv.cuh"
#include "unary.cuh"

// One block per (64 channels, 64 tokens, seq): the tile's channel rows of p, with the taps' token span, are staged in
// LDS by coalesced reads along the tokens and read back per (token, channel) so the output is written channel-fastest.
#define PLE_CONV_TC   64
#define PLE_CONV_TT   64
#define PLE_CONV_NTHR 256

struct ple_conv_dev_args {
    int64_t       start[PLE_CONV_MAX_TAPS];
    const char *  w[PLE_CONV_MAX_TAPS];
    int64_t       w_nb[PLE_CONV_MAX_TAPS];
};

static __global__ void __launch_bounds__(PLE_CONV_NTHR) k_ple_conv(const float * __restrict__ p, const int64_t L,
        const int64_t p_nb1, const int64_t p_nb2, const int C, const int T, const int n_taps, const int span,
        const ple_conv_dev_args da, float * __restrict__ out) {
    extern __shared__ float s_x[];   // [PLE_CONV_TC][pitch]
    const int pitch = PLE_CONV_TT + span + 1;
    const int c0  = blockIdx.x * PLE_CONV_TC;
    const int t0  = blockIdx.y * PLE_CONV_TT;
    const int s   = blockIdx.z;
    const int tid = threadIdx.x;

    // the tile reads tokens [base, base + PLE_CONV_TT + span) of each channel row
    const int64_t base = da.start[0] + t0;
    const char * ps = (const char *) p + (int64_t) s * p_nb2;
    const int nload = PLE_CONV_TT + span;
    const int nrows = min(PLE_CONV_TC, C - c0);
    for (int idx = tid; idx < nrows * nload; idx += PLE_CONV_NTHR) {
        const int r = idx / nload;
        const int j = idx - r * nload;
        const int64_t q = base + j;
        s_x[r * pitch + j] = q < L ? ((const float *) (ps + (int64_t) (c0 + r) * p_nb1))[q] : 0.0f;
    }
    __syncthreads();

    const int cl = tid % PLE_CONV_TC;
    const int c  = c0 + cl;
    if (c >= C) {
        return;
    }
    float w[PLE_CONV_MAX_TAPS];
    int   off[PLE_CONV_MAX_TAPS];
#pragma unroll
    for (int k = 0; k < PLE_CONV_MAX_TAPS; ++k) {
        if (k < n_taps) {
            w[k]   = *(const float *) (da.w[k] + (int64_t) c * da.w_nb[k]);
            off[k] = (int) (da.start[k] - da.start[0]);
        }
    }
    float * os = out + (int64_t) s * T * C;
    for (int tl = tid / PLE_CONV_TC; tl < PLE_CONV_TT; tl += PLE_CONV_NTHR / PLE_CONV_TC) {
        const int t = t0 + tl;
        if (t >= T) {
            break;
        }
        const float * xr = s_x + cl * pitch + tl;
        // MUL per tap (rounded), the ADD chain in tap order, then SILU: the graph's k_bin_bcast / k_ew_chain arithmetic.
        // The product must be rounded before the add (separate kernels in the graph): HIP's __fmul_rn / __fadd_rn are
        // plain operators that -ffp-contract=fast merges into an fma, so ggml_cuda_add_mul_rn turns contraction off
        float acc = xr[off[0]] * w[0];
#pragma unroll
        for (int k = 1; k < PLE_CONV_MAX_TAPS; ++k) {
            if (k < n_taps) {
                acc = ggml_cuda_add_mul_rn(acc, xr[off[k]], w[k]);
            }
        }
        os[(int64_t) t * C + c] = ggml_cuda_op_silu_single(acc);
    }
}

void ggml_cuda_op_ple_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_ple_conv_args & a) {
    GGML_ASSERT(a.n_taps >= 1 && a.n_taps <= PLE_CONV_MAX_TAPS);
    ple_conv_dev_args da = {};
    for (int k = 0; k < a.n_taps; ++k) {
        GGML_ASSERT(a.start[k] >= a.start[0]);
        da.start[k] = a.start[k];
        da.w[k]     = a.w[k];
        da.w_nb[k]  = a.w_nb[k];
    }
    const int span = (int) (a.start[a.n_taps - 1] - a.start[0]);
    const size_t smem = (size_t) PLE_CONV_TC * (PLE_CONV_TT + span + 1) * sizeof(float);
    GGML_ASSERT(smem <= 64 * 1024);
    const dim3 grid((unsigned) ((a.C + PLE_CONV_TC - 1) / PLE_CONV_TC), (unsigned) ((a.T + PLE_CONV_TT - 1) / PLE_CONV_TT),
            (unsigned) a.S);
    k_ple_conv<<<grid, PLE_CONV_NTHR, smem, ctx.stream()>>>(a.p, a.L, a.p_nb1, a.p_nb2, (int) a.C, (int) a.T, a.n_taps, span,
            da, a.out);
    CUDA_CHECK(cudaGetLastError());
}
