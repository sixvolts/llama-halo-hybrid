// the real pk_run (device part of persist.cu) on a hand-built [quant, mmvq] region, standalone
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#define GGML_USE_HIP 1
#include "common.cuh"
#include "pk_device.inc"   // device part of ggml/src/ggml-cuda/persist.cu: sed -n '1,/^\/\/ ---- host: classify/p' persist.cu
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
int main(int argc, char ** argv) {
    const int nrows = argc > 1 ? atoi(argv[1]) : 9216, K = argc > 2 ? atoi(argv[2]) : 2560, nmat = 8; const int bpr = K / 256;
    const int rpw = argc > 3 ? atoi(argv[3]) : 1, ku = argc > 4 ? atoi(argv[4]) : 1, w = argc > 5 ? atoi(argv[5]) : 1;
    const size_t mat_bytes = (size_t) nrows * bpr * sizeof(block_q4_K);
    std::vector<uint8_t> h(mat_bytes * nmat); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
    for (int mI = 0; mI < nmat; ++mI) for (int b = 0; b < nrows * bpr; ++b) { block_q4_K * B = (block_q4_K *) (h.data() + mI * mat_bytes) + b; uint16_t one = 0x3c00, zero = 0; memcpy(&B->dm, &one, 2); memcpy((char *) &B->dm + 2, &zero, 2); }
    std::vector<float> hy(K); for (int i = 0; i < K; ++i) hy[i] = (float) ((i % 32) - 16);
    block_q4_K * x; float * y; block_q8_1 * q8; float * dst; CHECK(hipMalloc(&x, h.size())); CHECK(hipMalloc(&y, K * 4)); CHECK(hipMalloc(&q8, (K / 32 + 8) * sizeof(block_q8_1))); CHECK(hipMalloc(&dst, nrows * 4));
    CHECK(hipMemcpy(x, h.data(), h.size(), hipMemcpyHostToDevice)); CHECK(hipMemcpy(y, hy.data(), K * 4, hipMemcpyHostToDevice));
    hipDeviceProp_t pr; CHECK(hipGetDeviceProperties(&pr, 0)); const int grid = pr.multiProcessorCount;
    // region: task 0 quant (f32 y -> q8_1), task 1 mmvq
    std::vector<pk_task> tasks(2); memset(tasks.data(), 0, sizeof(pk_task) * 2);
    pk_task & q = tasks[0]; q.op = PK_QUANT; q.i[0] = K; q.i[1] = K; q.i[2] = 1; q.p[1] = y; q.ne[1][0] = K; q.ne[1][1] = 1; q.ne[1][2] = 1; q.ne[1][3] = 1; q.nb[1][0] = 4; q.nb[1][1] = K * 4; q.nb[1][2] = K * 4; q.nb[1][3] = K * 4; q.p[0] = q8; q.n_items = (K + PK_QUANT_ITEM - 1) / PK_QUANT_ITEM; q.dep_first = 0; q.dep_count = 1; q.n_deps = 0;
    pk_task & m = tasks[1]; m.op = PK_MMVQ; m.type = GGML_TYPE_Q4_K; m.ncols = 1; m.i[0] = K; m.i[1] = nrows; m.i[2] = K / 32; m.i[3] = bpr; m.p[0] = dst; m.p[2] = q8; m.ne[0][0] = nrows; m.x[0] = rpw; m.x[1] = ku; m.x[2] = w; m.n_items = (nrows + (PK_WAVES / w) * rpw - 1) / ((PK_WAVES / w) * rpw); m.n_deps = 1; m.dep_first = 1; m.dep_count = 0;
    pk_task * d_tasks; pk_state * d_state; int * d_deps; CHECK(hipMalloc(&d_tasks, sizeof(pk_task) * 2 * nmat)); CHECK(hipMalloc(&d_state, sizeof(pk_state) * 3)); CHECK(hipMalloc(&d_deps, 16)); int deps[2] = { 1, 0 }; CHECK(hipMemcpy(d_deps, deps, 8, hipMemcpyHostToDevice)); CHECK(hipMemset(d_state, 0, sizeof(pk_state) * 3));
    for (int i = 0; i < nmat; ++i) { m.p[1] = x + (size_t) i * nrows * bpr; CHECK(hipMemcpy(d_tasks + 2 * i, tasks.data(), sizeof(pk_task) * 2, hipMemcpyHostToDevice)); }
    printf("%s: pk_run on [quant, mmvq q4_K %d x %d] (%.1f MB) cfg rpw %d ku %d w %d, n_items %d, grid %d\n", pr.gcnArchName, nrows, K, mat_bytes / 1e6, rpw, ku, w, m.n_items, grid);
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    auto launch = [&](int i) { pk_region_dev D = { d_tasks + 2 * i, d_state, d_deps, &d_state[2].done, 2, 0 }; pk_run<<<grid, PK_BLOCK>>>(D); };
    for (int i = 0; i < nmat; ++i) launch(i); CHECK(hipDeviceSynchronize());
    float best = 1e9; for (int rep = 0; rep < 3; ++rep) { CHECK(hipEventRecord(e0)); for (int i = 0; i < nmat * 4; ++i) launch(i % nmat); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); ms /= nmat * 4; if (ms < best) best = ms; }
    std::vector<pk_state> st(3); CHECK(hipMemcpy(st.data(), d_state, sizeof(pk_state) * 3, hipMemcpyDeviceToHost));
    printf("  pk_run region: %8.1f us  %6.1f GB/s  (marker %lld, arrivals %lld)\n", best * 1000, mat_bytes / best / 1e6, st[0].pad[0], st[2].done);
    return 0;
}
