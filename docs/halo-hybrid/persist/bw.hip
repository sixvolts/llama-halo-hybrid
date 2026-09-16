#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s: %s\n", #x, hipGetErrorString(e)); exit(1); } } while (0)
template <int U>
__global__ void __launch_bounds__(1024) stream_k(const int4 * __restrict__ p, size_t n, int * out) {
    const size_t tid = (size_t) blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t) gridDim.x * blockDim.x;
    int acc = 0;
    for (size_t i = tid; i < n; i += stride * U) {
        int4 v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) { const size_t k = i + (size_t) u * stride; v[u] = k < n ? p[k] : make_int4(0,0,0,0); }
#pragma unroll
        for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].w;
    }
    if (acc == 0x7fffffff) out[tid] = acc;
}
template <int U> void run(const int4 * p, size_t n, int * out, int grid, int bs, const char * tag) {
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    stream_k<U><<<grid, bs>>>(p, n, out); CHECK(hipDeviceSynchronize());
    float best = 1e9;
    for (int r = 0; r < 5; ++r) { CHECK(hipEventRecord(e0)); stream_k<U><<<grid, bs>>>(p, n, out); CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1)); float ms; CHECK(hipEventElapsedTime(&ms, e0, e1)); if (ms < best) best = ms; }
    printf("  %-8s grid %5d x %4d  U=%2d : %7.3f ms  %6.1f GB/s\n", tag, grid, bs, U, best, n * 16.0 / best / 1e6);
}
int main(int argc, char ** argv) {
    const size_t bytes = (size_t) (argc > 1 ? atoi(argv[1]) : 1024) << 20; const size_t n = bytes / 16;
    int4 * p; int * out; CHECK(hipMalloc(&p, bytes)); CHECK(hipMalloc(&out, 1 << 24)); CHECK(hipMemset(p, 1, bytes));
    hipDeviceProp_t pr; CHECK(hipGetDeviceProperties(&pr, 0)); printf("%s: %d SMs, %zu MB\n", pr.gcnArchName, pr.multiProcessorCount, bytes >> 20);
    const int nsm = pr.multiProcessorCount;
    run<1>(p, n, out, nsm, 1024, "resident"); run<2>(p, n, out, nsm, 1024, "resident"); run<4>(p, n, out, nsm, 1024, "resident"); run<8>(p, n, out, nsm, 1024, "resident"); run<16>(p, n, out, nsm, 1024, "resident");
    run<1>(p, n, out, 2*nsm, 1024, "2/WGP"); run<4>(p, n, out, 2*nsm, 1024, "2/WGP"); run<8>(p, n, out, 2*nsm, 1024, "2/WGP");
    run<1>(p, n, out, 4*nsm, 512, "4x512"); run<4>(p, n, out, 4*nsm, 512, "4x512");
    run<1>(p, n, out, 64*nsm, 256, "normal"); run<4>(p, n, out, 64*nsm, 256, "normal"); run<8>(p, n, out, 16*nsm, 1024, "normal");
    return 0;
}
