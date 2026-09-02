// Reproduce ggml's two-lane, two-device split pattern and measure whether the devices overlap.
// per lane l, per layer k:  R (dev0): attn_k (spin ~12 ms) ; copy R->A (16 MB) ; record eR[l]
//                           A (dev1): wait eR[l] ; exp_k (spin ~27 ms) ; copy A->R ; record eA[l]
//                           R (dev0): wait eA[l] ; next layer ...
// submission interleaved a,b per split as ggml does. Ideal overlap ~ N*2*27 ms; serialized ~ N*2*(12+27+copies).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t err_ = (x); if (err_ != hipSuccess) { printf("%s failed: %s\n", #x, hipGetErrorString(err_)); exit(1); } } while (0)
__global__ void spin(long long iters, int * out) { float x = threadIdx.x; for (long long i = 0; i < iters; i++) { x = x * 0.999f + 0.001f; } if (x == 12345.f) *out = 1; }
__global__ void copy16(const uint4 * __restrict__ src, uint4 * __restrict__ dst, size_t n16) { size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (i < n16) dst[i] = src[i]; }
static double ms(std::chrono::steady_clock::time_point a) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count(); }
int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const int N = 10, L = 2; const size_t bytes = 16u << 20;
    const char * mode = argc > 1 ? argv[1] : "kernelcopy";   // kernelcopy | memcpypeer | nocopy | onelane | sharedevent
    const bool shared = strcmp(mode, "sharedevent") == 0;   // one event per device re-recorded by every lane, as ggml does
    const bool selfwait = strcmp(mode, "selfwait") == 0;
    const bool selfrec = strcmp(mode, "selfrecord") == 0;    // record only, no wait
    const bool h2d = strcmp(mode, "h2d") == 0;                // plus an async pinned H2D copy on the R stream before each R split    // per-lane copy events, plus ggml-sched-like same-stream record/wait pairs on a shared per-stream event
    hipEvent_t eSelf[2][L]; void * hbuf = nullptr; CK(hipHostMalloc(&hbuf, 1 << 20));
    const int lanes = strcmp(mode, "onelane") == 0 ? 1 : L;
    hipStream_t s[2]; int * d[2]; void * buf[2][L]; hipEvent_t eR[L], eA[L];
    for (int i = 0; i < 2; i++) { CK(hipSetDevice(i)); CK(hipStreamCreateWithFlags(&s[i], hipStreamNonBlocking)); CK(hipMalloc(&d[i], 4));
        for (int l = 0; l < L; l++) CK(hipMalloc(&buf[i][l], bytes)); }
    CK(hipSetDevice(0)); CK(hipDeviceEnablePeerAccess(1, 0)); for (int l = 0; l < L; l++) { CK(hipEventCreateWithFlags(&eR[l], hipEventDisableTiming)); CK(hipEventCreateWithFlags(&eSelf[0][l], hipEventDisableTiming)); }
    CK(hipSetDevice(1)); CK(hipDeviceEnablePeerAccess(0, 0)); for (int l = 0; l < L; l++) { CK(hipEventCreateWithFlags(&eA[l], hipEventDisableTiming)); CK(hipEventCreateWithFlags(&eSelf[1][l], hipEventDisableTiming)); }
    long long it[2];
    for (int i = 0; i < 2; i++) { CK(hipSetDevice(i)); auto t = std::chrono::steady_clock::now(); spin<<<1,64,0,s[i]>>>(20000000LL, d[i]); CK(hipStreamSynchronize(s[i])); it[i] = (long long)(20000000.0 / ms(t)); }
    const long long itR = it[0] * 12, itA = it[1] * 27;   // 12 ms attention, 27 ms experts
    auto copy = [&](int src, int dst, void * from, void * to) {
        CK(hipSetDevice(src));
        if (strcmp(mode, "nocopy") == 0) return;
        if (strcmp(mode, "memcpypeer") == 0) { CK(hipMemcpyPeerAsync(to, dst, from, src, bytes, s[src])); return; }
        const size_t n16 = bytes / 16; copy16<<<(unsigned)((n16 + 255) / 256), 256, 0, s[src]>>>((const uint4 *) from, (uint4 *) to, n16); CK(hipGetLastError());
    };
    for (int rep = 0; rep < 2; rep++) {
        for (int i = 0; i < 2; i++) { CK(hipSetDevice(i)); CK(hipStreamSynchronize(s[i])); }
        auto t = std::chrono::steady_clock::now();
        for (int k = 0; k < N; k++) {
            // split 2k (R) for each lane, then split 2k+1 (A) for each lane
            for (int l = 0; l < lanes; l++) {
                CK(hipSetDevice(0));
                if (selfwait && k > 0) CK(hipStreamWaitEvent(s[0], eSelf[0][l], 0));
                if (h2d) CK(hipMemcpyAsync(buf[0][l], hbuf, 1 << 20, hipMemcpyHostToDevice, s[0]));
                if (k > 0) CK(hipStreamWaitEvent(s[0], eA[shared ? 0 : l], 0));
                spin<<<1,64,0,s[0]>>>(itR, d[0]);
                copy(0, 1, buf[0][l], buf[1][l]);                  // on the src (R) stream, like ggml
                CK(hipEventRecord(eR[shared ? 0 : l], s[0]));
                if (selfwait || selfrec) CK(hipEventRecord(eSelf[0][l], s[0]));
            }
            for (int l = 0; l < lanes; l++) {
                CK(hipSetDevice(1));
                if (selfwait && k > 0) CK(hipStreamWaitEvent(s[1], eSelf[1][l], 0));
                CK(hipStreamWaitEvent(s[1], eR[shared ? 0 : l], 0));
                spin<<<1,64,0,s[1]>>>(itA, d[1]);
                copy(1, 0, buf[1][l], buf[0][l]);                  // on the src (A) stream
                CK(hipEventRecord(eA[shared ? 0 : l], s[1]));
                if (selfwait || selfrec) CK(hipEventRecord(eSelf[1][l], s[1]));
            }
        }
        double t_submit = ms(t);
        for (int i = 0; i < 2; i++) { CK(hipSetDevice(i)); CK(hipStreamSynchronize(s[i])); }
        printf("%-11s lanes=%d rep %d: submit %.1f ms, total %.0f ms  (serialized ~%d, overlapped ~%d)\n", mode, lanes, rep, t_submit, ms(t), N*lanes*(12+27), N*lanes*27);
    }
    return 0;
}
