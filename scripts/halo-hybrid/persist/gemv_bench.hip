// standalone GEMV body benchmark: q4_K rows x K, resident grid of 20x1024 with static striding (pk_run's shape)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#define GGML_USE_HIP 1
#include "vecdotq.cuh"
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
#define WAVES 32
// A: ggml vec_dot per block, RPW rows per wave, KU blocks in flight (compiler decides the batching)
template <int RPW, int KU>
__global__ void __launch_bounds__(1024) gemv_A(const block_q4_K * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int nrows, int bpr, int n_items) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int row0 = (item * WAVES + wave) * RPW;
        if (row0 >= nrows) continue;
        float tmp[RPW]; for (int r = 0; r < RPW; ++r) tmp[r] = 0.f;
        const int kqs = 2 * (lane % 16);
        for (int kbx0 = lane / 16; kbx0 < bpr; kbx0 += 2 * KU) {
#pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kbx_u = kbx0 + u * 2; const float m = (KU == 1 || kbx_u < bpr) ? 1.f : 0.f; const int kbx = KU == 1 ? kbx_u : min(kbx_u, bpr - 1);
#pragma unroll
                for (int r = 0; r < RPW; ++r) { const int row = RPW == 1 ? row0 : min(row0 + r, nrows - 1); tmp[r] += m * vec_dot_q4_K_q8_1(x, &y[kbx * 8], row * bpr + kbx, kqs); }
            }
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) { tmp[r] = warp_reduce_sum<32>(tmp[r]); if (lane == 0 && row0 + r < nrows) dst[row0 + r] = tmp[r]; }
    }
}
// B: explicit load phase (all weight words of the RPW x KU group into registers), then compute (same math as ggml)
struct q4k_regs { int v0, v1; uint32_t s0, s2, s4; uint32_t dm; };
__device__ __forceinline__ void q4k_load(const block_q4_K * b, int iqs, q4k_regs & R) {
    const int bq8_offset = 2 * ((iqs/2) / 4);
    const int * q4 = (const int *)(b->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    R.v0 = q4[0]; R.v1 = q4[4];
    const uint16_t * scales = (const uint16_t *) b->scales; const int j = bq8_offset/2, jm = j & 1;
    R.s0 = scales[jm]; R.s2 = scales[jm + 2]; R.s4 = scales[jm + 4];
    R.dm = *(const uint32_t *) &b->dm;
}
__device__ __forceinline__ float q4k_dot(const q4k_regs & R, const block_q8_1 * bq8_1, int iqs) {
    const int bq8_offset = 2 * ((iqs/2) / 4); const int j = bq8_offset/2;
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((R.s0 & 0x3f3f) & ~hi) | ((((R.s4 >> 0) & 0x0f0f) | ((R.s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((R.s2 & 0x3f3f) & ~hi) | ((((R.s4 >> 4) & 0x0f0f) | ((R.s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *) aux; const uint8_t * m = sc + 2;
    int v[2] = { R.v0, R.v1 }; int u[4]; float d8[2];
    for (int i = 0; i < 2; ++i) { const block_q8_1 * bq8i = bq8_1 + bq8_offset + i; d8[i] = __low2float(bq8i->ds); const int * q8 = (const int *) bq8i->qs + ((iqs/2)%4); u[2*i] = q8[0]; u[2*i+1] = q8[4]; }
    half2 dm; *(uint32_t *) &dm = R.dm;
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}
template <int RPW, int KU>
__global__ void __launch_bounds__(1024) gemv_B(const block_q4_K * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int nrows, int bpr, int n_items) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        const int row0 = (item * WAVES + wave) * RPW;
        if (row0 >= nrows) continue;
        float tmp[RPW]; for (int r = 0; r < RPW; ++r) tmp[r] = 0.f;
        const int kqs = 2 * (lane % 16);
        for (int kbx0 = lane / 16; kbx0 < bpr; kbx0 += 2 * KU) {
            q4k_regs R[RPW][KU];
#pragma unroll
            for (int u = 0; u < KU; ++u) { const int kbx = min(kbx0 + u * 2, bpr - 1);
#pragma unroll
                for (int r = 0; r < RPW; ++r) { const int row = min(row0 + r, nrows - 1); q4k_load(x + row * bpr + kbx, kqs, R[r][u]); } }
#pragma unroll
            for (int u = 0; u < KU; ++u) { const int kbx_u = kbx0 + u * 2; const float m = (kbx_u < bpr) ? 1.f : 0.f; const int kbx = min(kbx_u, bpr - 1);
#pragma unroll
                for (int r = 0; r < RPW; ++r) tmp[r] += m * q4k_dot(R[r][u], &y[kbx * 8], kqs); }
        }
#pragma unroll
        for (int r = 0; r < RPW; ++r) { tmp[r] = warp_reduce_sum<32>(tmp[r]); if (lane == 0 && row0 + r < nrows) dst[row0 + r] = tmp[r]; }
    }
}
// C: MMVQ-like normal launch: 4 waves per block, one row per wave (same loop as A<1,1>), grid covers all rows
__global__ void __launch_bounds__(128) gemv_C(const block_q4_K * __restrict__ x, const block_q8_1 * __restrict__ y, float * __restrict__ dst, int nrows, int bpr) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32; const int row = blockIdx.x * 4 + wave; if (row >= nrows) return;
    float tmp = 0.f; const int kqs = 2 * (lane % 16);
    for (int kbx = lane / 16; kbx < bpr; kbx += 2) tmp += vec_dot_q4_K_q8_1(x, &y[kbx * 8], row * bpr + kbx, kqs);
    tmp = warp_reduce_sum<32>(tmp); if (lane == 0) dst[row] = tmp;
}
int main(int argc, char ** argv) {
    const int nrows = argc > 1 ? atoi(argv[1]) : 9216, K = argc > 2 ? atoi(argv[2]) : 2560, nmat = 8; const int bpr = K / 256;
    const size_t mat_bytes = (size_t) nrows * bpr * sizeof(block_q4_K);
    std::vector<uint8_t> h(mat_bytes * nmat); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
    // sane halves for d/dmin: 1.0 / 0.0
    for (int mI = 0; mI < nmat; ++mI) for (int b = 0; b < nrows * bpr; ++b) { block_q4_K * B = (block_q4_K *) (h.data() + mI * mat_bytes) + b; uint16_t one = 0x3c00, zero = 0; memcpy(&B->dm, &one, 2); memcpy((char *) &B->dm + 2, &zero, 2); }
    std::vector<block_q8_1> hy(bpr * 8); for (auto & q : hy) { for (int i = 0; i < 32; ++i) q.qs[i] = (int8_t) (i - 16); uint16_t one = 0x3c00; memcpy(&q.ds, &one, 2); memcpy((char *) &q.ds + 2, &one, 2); }
    block_q4_K * x; block_q8_1 * y; float * dst; CHECK(hipMalloc(&x, h.size())); CHECK(hipMalloc(&y, hy.size() * sizeof(block_q8_1))); CHECK(hipMalloc(&dst, nrows * 4 * nmat));
    CHECK(hipMemcpy(x, h.data(), h.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(y, hy.data(), hy.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    hipDeviceProp_t pr; CHECK(hipGetDeviceProperties(&pr, 0)); const int grid = pr.multiProcessorCount;
    printf("%s: %d rows x K=%d q4_K (%.1f MB), %d matrices cycled, grid %d x 1024\n", pr.gcnArchName, nrows, K, mat_bytes / 1e6, nmat, grid);
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    std::vector<float> ref(nrows);
    auto time = [&](const char * tag, auto launch) {
        for (int i = 0; i < nmat; ++i) launch(i);
        CHECK(hipDeviceSynchronize()); float best = 1e9;
        for (int rep = 0; rep < 3; ++rep) { CHECK(hipEventRecord(e0)); for (int i = 0; i < nmat * 4; ++i) launch(i % nmat); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= nmat * 4; if (ms < best) best = ms; }
        std::vector<float> out(nrows); CHECK(hipMemcpy(out.data(), dst, nrows * 4, hipMemcpyDeviceToHost));
        double maxd = 0; if (ref[0] == 0 && ref[1] == 0) ref = out; else for (int r = 0; r < nrows; ++r) maxd = fmax(maxd, fabs(out[r] - ref[r]) / (fabs(ref[r]) + 1e-3));
        printf("  %-22s %8.1f us  %6.1f GB/s  maxrel %.1e\n", tag, best * 1000, mat_bytes / best / 1e6, maxd);
    };
    #define MAT(i) (x + (size_t) (i) * nrows * bpr)
    time("C normal 4w/128", [&](int i) { gemv_C<<<(nrows + 3) / 4, 128>>>(MAT(i), y, dst, nrows, bpr); });
    time("A resident 1x1", [&](int i) { gemv_A<1,1><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 31) / 32); });
    time("A resident 1x2", [&](int i) { gemv_A<1,2><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 31) / 32); });
    time("A resident 2x2", [&](int i) { gemv_A<2,2><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 63) / 64); });
    time("A resident 4x1", [&](int i) { gemv_A<4,1><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 127) / 128); });
    time("B regs 1x1", [&](int i) { gemv_B<1,1><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 31) / 32); });
    time("B regs 1x2", [&](int i) { gemv_B<1,2><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 31) / 32); });
    time("B regs 2x2", [&](int i) { gemv_B<2,2><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 63) / 64); });
    time("B regs 4x1", [&](int i) { gemv_B<4,1><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 127) / 128); });
    time("B regs 4x2", [&](int i) { gemv_B<4,2><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 127) / 128); });
    time("B regs 2x4", [&](int i) { gemv_B<2,4><<<grid, 1024>>>(MAT(i), y, dst, nrows, bpr, (nrows + 63) / 64); });
    return 0;
}
