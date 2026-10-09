#include <hip/hip_runtime.h>
#include <cstdio>
#include <chrono>
int main() {
    int can01=0, can10=0; hipDeviceCanAccessPeer(&can01, 0, 1); hipDeviceCanAccessPeer(&can10, 1, 0);
    printf("canAccessPeer 0->1 %d, 1->0 %d\n", can01, can10);
    hipSetDevice(0); hipDeviceEnablePeerAccess(1, 0); hipSetDevice(1); hipDeviceEnablePeerAccess(0, 0);
    for (size_t sz : {(size_t)20480, (size_t)262144, (size_t)8<<20, (size_t)64<<20}) {
        void *a, *b; hipSetDevice(0); hipMalloc(&a, sz); hipSetDevice(1); hipMalloc(&b, sz);
        hipStream_t s; hipSetDevice(0); hipStreamCreate(&s);
        for (int dir = 0; dir < 2; ++dir) {
            int n = sz < (1<<20) ? 2000 : 50;
            hipMemcpyPeerAsync(dir ? a : b, dir ? 0 : 1, dir ? b : a, dir ? 1 : 0, sz, s); hipStreamSynchronize(s);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < n; ++i) { hipMemcpyPeerAsync(dir ? a : b, dir ? 0 : 1, dir ? b : a, dir ? 1 : 0, sz, s); hipStreamSynchronize(s); }
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
            printf("%s %8zu B: %8.1f us  %6.2f GB/s\n", dir ? "APU->card" : "card->APU", sz, us, sz / us / 1e3);
        }
        hipFree(a); hipFree(b); hipStreamDestroy(s);
    }
}
