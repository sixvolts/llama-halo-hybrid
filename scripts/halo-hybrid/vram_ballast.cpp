// Holds N bytes of VRAM on a HIP device until killed, so a 32 GB R9700 looks like a 16 GB card (RX 9070 XT) to the
// next process. usage: vram_ballast <device> <gib>   (prints free memory after allocating, then holds it until killed)
// build: hipcc -O2 --offload-arch=gfx1201 vram_ballast.cpp -o vram_ballast
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    const double gib = argc > 2 ? atof(argv[2]) : 16.0;
    hipDeviceProp_t p; hipGetDeviceProperties(&p, dev); hipSetDevice(dev);
    const size_t n = (size_t) (gib * 1024.0 * 1024.0 * 1024.0);
    void * buf = nullptr;
    if (hipMalloc(&buf, n) != hipSuccess) { fprintf(stderr, "hipMalloc %zu failed\n", n); return 1; }
    hipMemset(buf, 0, n); hipDeviceSynchronize();   // commit every page
    size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot);
    printf("ballast: %s (%s) holding %.2f GiB; free %.2f / %.2f GiB\n", p.name, p.gcnArchName, n / 1073741824.0, fr / 1073741824.0, tot / 1073741824.0);
    fflush(stdout);
    pause();
    return 0;
}
