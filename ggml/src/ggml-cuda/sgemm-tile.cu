#include "sgemm-tile.cuh"

// dst[n][m] = sum_k A[m][k] * B[n][k]; A = src0 [K x M] and B = src1 [K x N], both K-contiguous rows.
// Block tile 32 (m) x 128 (n) x 32 (k), 64 threads (4 x 16), 8 x 8 outputs per thread: four float4 LDS reads per
// 64 FMAs per lane, which is what keeps the LDS side (128 B/clk per CU, shared by two SIMDs) level with the FMAs.
#define SGT_TM 32
#define SGT_TN 128
#define SGT_TK 32
#define SGT_THREADS 64
#define SGT_PAD 4

static __global__ void __launch_bounds__(SGT_THREADS) mul_mat_f32_tile(
        const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
        const int M, const int N, const int K, const int64_t stride_a, const int64_t stride_b, const int64_t stride_c) {
    __shared__ float As[SGT_TK][SGT_TM + SGT_PAD];
    __shared__ float Bs[SGT_TK][SGT_TN + SGT_PAD];

    const int tid = threadIdx.x;
    const int tm  = tid % 4;   // 4 thread columns along m, 8 outputs each
    const int tn  = tid / 4;   // 16 thread rows along n, 8 outputs each
    const int m0  = blockIdx.y * SGT_TM;
    const int n0  = blockIdx.x * SGT_TN;

    float acc[8][8] = {};

    // A tile: 32 rows (m) x 32 k -> 16 floats per thread: row = tid/2, k = (tid%2)*16
    // B tile: 128 rows (n) x 32 k -> 64 floats per thread: rows tid/2 + 32*j, k = (tid%2)*16
    const int lr = tid / 2;
    const int lk = (tid % 2) * 16;

    const bool a_ok = m0 + lr < M;
    const float * Ap = A + (int64_t)(m0 + (a_ok ? lr : 0)) * stride_a + lk;
    const float * Bp[4];
    bool b_ok[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        b_ok[j] = n0 + lr + 32*j < N;
        Bp[j]   = B + (int64_t)(n0 + (b_ok[j] ? lr + 32*j : 0)) * stride_b + lk;
    }

    float4 ar[4], br[4][4];
    auto load_tile = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            ar[i] = a_ok ? *(const float4 *)(Ap + k0 + 4*i) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                br[j][i] = b_ok[j] ? *(const float4 *)(Bp[j] + k0 + 4*i) : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    };
    load_tile(0);

    for (int k0 = 0; k0 < K; k0 += SGT_TK) {
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            As[lk + 4*i + 0][lr] = ar[i].x; As[lk + 4*i + 1][lr] = ar[i].y; As[lk + 4*i + 2][lr] = ar[i].z; As[lk + 4*i + 3][lr] = ar[i].w;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                Bs[lk + 4*i + 0][lr + 32*j] = br[j][i].x; Bs[lk + 4*i + 1][lr + 32*j] = br[j][i].y;
                Bs[lk + 4*i + 2][lr + 32*j] = br[j][i].z; Bs[lk + 4*i + 3][lr + 32*j] = br[j][i].w;
            }
        }
        __syncthreads();

        if (k0 + SGT_TK < K) {
            load_tile(k0 + SGT_TK);
        }

#pragma unroll 8
        for (int k = 0; k < SGT_TK; ++k) {
            const float4 a0 = *(const float4 *)&As[k][tm*8];
            const float4 a1 = *(const float4 *)&As[k][tm*8 + 4];
            const float4 b0 = *(const float4 *)&Bs[k][tn*8];
            const float4 b1 = *(const float4 *)&Bs[k][tn*8 + 4];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[j][i] += a[i]*b[j];
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int n = n0 + tn*8 + j;
        if (n >= N) {
            continue;
        }
        float * Cp = C + (int64_t) n * stride_c + m0 + tm*8;
        if (m0 + tm*8 + 7 < M) {
            *(float4 *) Cp       = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
            *(float4 *)(Cp + 4)  = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (m0 + tm*8 + i < M) {
                    Cp[i] = acc[j][i];
                }
            }
        }
    }
}

bool ggml_cuda_mul_mat_f32_tile(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_F32_TILE") != nullptr && atoi(getenv("GGML_CUDA_NO_F32_TILE")) != 0;
    if (disabled) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->nb[0] != sizeof(float) || src1->nb[0] != sizeof(float) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (K % SGT_TK != 0 || (src0->nb[1] % 16) != 0 || (src1->nb[1] % 16) != 0 || M > (1 << 30) || N > (1 << 30)) {
        return false;
    }
    // the shape it is for: many tokens against a thin weight; wider weights keep the vendor GEMM
    if (N <= 16 || M > 1024) {
        return false;
    }
    const dim3 grid((N + SGT_TN - 1) / SGT_TN, (M + SGT_TM - 1) / SGT_TM, 1);
    mul_mat_f32_tile<<<grid, SGT_THREADS, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        (int) M, (int) N, (int) K, src0->nb[1] / sizeof(float), src1->nb[1] / sizeof(float), dst->nb[1] / sizeof(float));
    CUDA_CHECK(cudaGetLastError());
    return true;
}
