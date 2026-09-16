#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;
// (a) N grid-wide syncs back to back; (b) N "task" hand-offs through a device-scope counter: block b waits until
// the counter reaches i before doing step i (a dependency chain, the persistent-kernel primitive), no full barrier
__global__ void k_grid(int iters) { for (int i = 0; i < iters; ++i) cg::this_grid().sync(); }
__global__ void k_chain(int iters, unsigned * ctr) {
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x == 0) {
            // block 0 advances the step after all blocks arrived at it: arrival counter + release counter
            unsigned arrived = atomicAdd(&ctr[0], 1u) + 1;            // device-scope (HIP atomics on global memory)
            if (arrived == (unsigned) gridDim.x * (i + 1)) { __threadfence(); atomicExch(&ctr[1], (unsigned) (i + 1)); }
            while (__hip_atomic_load(&ctr[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < (unsigned) (i + 1)) { __builtin_amdgcn_s_sleep(1); }
        }
        __syncthreads();
    }
}
int main() {
    int d = 0; hipGetDevice(&d); hipDeviceProp_t p; hipGetDeviceProperties(&p, d);
    const int iters = 2000; int blocks = p.multiProcessorCount * 2;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); float ms;
    void * args[] = { (void *) &iters };
    hipLaunchCooperativeKernel((const void *) k_grid, dim3(blocks), dim3(256), args, 0, 0); hipDeviceSynchronize();
    hipEventRecord(a); hipLaunchCooperativeKernel((const void *) k_grid, dim3(blocks), dim3(256), args, 0, 0); hipEventRecord(b); hipDeviceSynchronize();
    hipEventElapsedTime(&ms, a, b); printf("%s: %d blocks, cg grid.sync: %.2f us per barrier\n", p.gcnArchName, blocks, 1000 * ms / iters);
    unsigned * ctr; hipMalloc(&ctr, 8);
    for (int bl : { blocks, p.multiProcessorCount, 8 }) {
        hipMemset(ctr, 0, 8);
        void * args2[] = { (void *) &iters, (void *) &ctr };
        hipLaunchCooperativeKernel((const void *) k_chain, dim3(bl), dim3(256), args2, 0, 0); hipDeviceSynchronize();
        hipMemset(ctr, 0, 8);
        hipEventRecord(a); hipLaunchCooperativeKernel((const void *) k_chain, dim3(bl), dim3(256), args2, 0, 0); hipEventRecord(b); hipDeviceSynchronize();
        hipEventElapsedTime(&ms, a, b); printf("%s: %d blocks, atomic arrive/release step: %.2f us per step\n", p.gcnArchName, bl, 1000 * ms / iters);
    }
    // reference: back-to-back empty kernel launches (the boundary cost a persistent kernel replaces), plain and in a graph
    hipEventRecord(a); for (int i = 0; i < iters; ++i) hipLaunchKernelGGL(k_grid, dim3(1), dim3(64), 0, 0, 0); hipEventRecord(b); hipDeviceSynchronize();
    hipEventElapsedTime(&ms, a, b); printf("%s: %d empty kernel launches, stream: %.2f us each\n", p.gcnArchName, iters, 1000 * ms / iters);
    hipStream_t s; hipStreamCreate(&s); hipGraph_t g; hipGraphExec_t ge;
    hipStreamBeginCapture(s, hipStreamCaptureModeGlobal); for (int i = 0; i < 500; ++i) hipLaunchKernelGGL(k_grid, dim3(1), dim3(64), 0, s, 0); hipStreamEndCapture(s, &g);
    hipGraphInstantiate(&ge, g, nullptr, nullptr, 0); hipGraphLaunch(ge, s); hipStreamSynchronize(s);
    hipEventRecord(a, s); hipGraphLaunch(ge, s); hipEventRecord(b, s); hipStreamSynchronize(s);
    hipEventElapsedTime(&ms, a, b); printf("%s: 500 empty kernels inside one hipGraph: %.2f us per node\n", p.gcnArchName, 1000 * ms / 500);
    return 0;
}
