// Stage 1 of the persistent decode kernel: the scheduler primitives at size.
//
// A task list in topological order; every resident workgroup walks it in order, claims work items of the
// current task with one atomic, executes them, and retires them; the last retiring item of a task releases
// the task's dependents by bumping their counters (device-scope release); a workgroup arriving at a task whose
// dependencies are not all done spins on an acquire load. No grid barrier anywhere.
//
// The synthetic DAG mimics a decode layer: per "layer" a norm-like task, four independent GEMV-like tasks
// reading its output (fan-out), an add-like task reading all four (fan-in), repeated. Every task writes and
// reads activation-sized rows so the release/acquire path is exercised on real memory traffic, and the final
// checksum is compared with a CPU reference. The same DAG is also run as one kernel per task inside a replayed
// hipGraph, which is what the persistent kernel replaces.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <unistd.h>

#ifndef PK_SLEEP
#define PK_SLEEP 1
#endif
#ifdef PK_RELAXED_DONE
#define PK_DONE_ADD add_relaxed
#else
#define PK_DONE_ADD add_release
#endif
#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

enum pk_op : int { PK_SCALE = 0, PK_SUM4 = 1, PK_NOP = 2 };

struct pk_task {
    int      op;
    int      n_items;      // work items (one item = one row here)
    int      n_deps;       // number of predecessor tasks
    int      dep_first;    // first entry in the dependents CSR list
    int      dep_count;    // number of dependent tasks
    int      row0;         // output rows [row0, row0 + n_items)
    int      src[4];       // input row bases (op-specific)
    float    a, b;
};

struct __attribute__((aligned(128))) pk_state {   // one per task, own cache line: no false sharing between tasks
    int cursor;            // unused by the static scheduler
    int done;              // retired items
    int deps_done;         // completed predecessors
    int pad[29];
};

struct pk_list {
    int             chunks;       // items per row
    unsigned long long * tl;      // timeline [4 tasks][2048 blocks][4]
    int           * group_ready;  // [n_groups][32] ints, one line per group: the task index the leader has released
    int             n_groups;
    volatile int  * abort_flag;   // host-coherent: the host sets it to stop a stalled kernel
    int           * progress;     // host-coherent: progress[block] = task index * 4 + phase
    const pk_task * tasks;
    pk_state      * state;
    const int     * dependents;   // CSR: dependents of task t are dependents[dep_first .. dep_first + dep_count)
    int             n_tasks;
    float         * buf;          // rows of N floats
    int             N;
};

__device__ __forceinline__ int ld_acquire(int * p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
// poll with a RELAXED agent-scope load (bypasses L0/L1 by cache policy, invalidates nothing) and fence once after
__device__ __forceinline__ int ld_relaxed(int * p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void fence_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
__device__ __forceinline__ int add_release(int * p, int v) { return __hip_atomic_fetch_add(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int add_relaxed(int * p, int v) { return __hip_atomic_fetch_add(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }

// op bodies: one work item = one row of N floats, 256 threads
__device__ __forceinline__ void op_scale(const pk_list & L, const pk_task & t, int item) {
    const float * src = L.buf + (size_t) (t.src[0] + item) * L.N;
    float       * dst = L.buf + (size_t) (t.row0   + item) * L.N;
    for (int i = threadIdx.x; i < L.N; i += blockDim.x) dst[i] = src[i] * t.a + t.b;
}
__device__ __forceinline__ void op_sum4(const pk_list & L, const pk_task & t, int item) {
    float * dst = L.buf + (size_t) (t.row0 + item) * L.N;
    for (int i = threadIdx.x; i < L.N; i += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) s += L.buf[(size_t) (t.src[k] + item) * L.N + i];
        dst[i] = s * t.a;
    }
}
__device__ __forceinline__ void run_item(const pk_list & L, const pk_task & t, int item) {
    switch (t.op) {
        case PK_SCALE: op_scale(L, t, item); break;
        case PK_SUM4:  op_sum4(L, t, item);  break;
        default: break;
    }
}

// ---- the persistent scheduler ---------------------------------------------------------------------------------
// The scheduling unit is a WAVE (32 lanes): no workgroup barriers anywhere in the loop, every control decision is
// taken from a wave-uniform value (readfirstlane), so the loop is provably convergent. A work item here is one row,
// processed by the 32 lanes of the wave.
__device__ __forceinline__ void run_item_wave(const pk_list & L, const pk_task & t, int item, int lane) {
    // item = (row, chunk): a wave processes CH = N/chunks floats of one row with 8 loads in flight per lane
    const int N = L.N, CH = N / L.chunks;
    const int row = item / L.chunks, c0 = (item % L.chunks) * CH;
    if (t.op == PK_SCALE) {
        const float * __restrict__ src = L.buf + (size_t) (t.src[0] + row) * N + c0;
        float       * __restrict__ dst = L.buf + (size_t) (t.row0   + row) * N + c0;
        for (int i0 = lane; i0 < CH; i0 += 32 * 8) {
            float v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) { const int i = i0 + 32 * u; v[u] = i < CH ? src[i] : 0.f; }
#pragma unroll
            for (int u = 0; u < 8; ++u) { const int i = i0 + 32 * u; if (i < CH) dst[i] = v[u] * t.a + t.b; }
        }
    } else if (t.op == PK_SUM4) {
        float * __restrict__ dst = L.buf + (size_t) (t.row0 + row) * N + c0;
        const float * __restrict__ s0 = L.buf + (size_t) (t.src[0] + row) * N + c0;
        const float * __restrict__ s1 = L.buf + (size_t) (t.src[1] + row) * N + c0;
        const float * __restrict__ s2 = L.buf + (size_t) (t.src[2] + row) * N + c0;
        const float * __restrict__ s3 = L.buf + (size_t) (t.src[3] + row) * N + c0;
        for (int i0 = lane; i0 < CH; i0 += 32 * 4) {
            float v[4][4];
#pragma unroll
            for (int u = 0; u < 4; ++u) { const int i = i0 + 32 * u; const bool ok = i < CH; v[u][0] = ok ? s0[i] : 0.f; v[u][1] = ok ? s1[i] : 0.f; v[u][2] = ok ? s2[i] : 0.f; v[u][3] = ok ? s3[i] : 0.f; }
#pragma unroll
            for (int u = 0; u < 4; ++u) { const int i = i0 + 32 * u; if (i < CH) dst[i] = (v[u][0] + v[u][1] + v[u][2] + v[u][3]) * t.a; }
        }
    }
}

// Block-level scheduler: the workgroup is the unit. Thread 0 polls the dependency counter and broadcasts through
// LDS behind a barrier (block-uniform control flow); items are assigned statically (item = block, += n_blocks) and a
// block's 8 waves split each row; one retire atomic per block per task. Idle blocks poll with a long sleep.
__global__ void __launch_bounds__(1024) pk_run(pk_list L) {
    __shared__ int s_flag;
    const int lane = threadIdx.x % 32, wave_in_block = threadIdx.x / 32;
    const int n_blocks = gridDim.x;
    for (int ti = 0; ti < L.n_tasks; ++ti) {
        const pk_task * t = &L.tasks[ti];
        pk_state * st = &L.state[ti];
        const int n_items = __builtin_amdgcn_readfirstlane(t->n_items);
        const int n_deps  = __builtin_amdgcn_readfirstlane(t->n_deps);
        // wait for the predecessors: two-level polling. Leaders (one per group) poll the task's counter and publish
        // the task index into the group's own cache line; followers poll that line. Pollers per line drop from the
        // whole grid to the group size.
        if (threadIdx.x == 0) {
            int f = 0;
            const int g = blockIdx.x % L.n_groups;
            const bool leader = blockIdx.x < (unsigned) L.n_groups;
            while (true) {
                if (leader) {
                    if (ld_relaxed(&st->deps_done) >= n_deps) { fence_acquire(); __hip_atomic_store(&L.group_ready[g * 32], ti, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); f = 1; break; }
                } else {
                    if (ld_relaxed(&L.group_ready[g * 32]) >= ti) { fence_acquire(); f = 1; break; }
                }
                if (*L.abort_flag) { f = -1; break; }
                __builtin_amdgcn_s_sleep(PK_SLEEP);
            }
            s_flag = f;
            L.progress[blockIdx.x] = ti * 4 + (f > 0 ? 1 : 3);
        }
        __syncthreads();
        const int flag = s_flag;
        __syncthreads();
        if (flag < 0) return;
        const int tlt = ti - 100;
        if (tlt >= 0 && tlt < 4 && threadIdx.x == 0) L.tl[(tlt * 2048 + blockIdx.x) * 4 + 0] = wall_clock64();
        // static striding over the rows; the 8 waves of the block split each row (chunks == 8 in this layout)
        int mine = 0;
        for (int item = blockIdx.x; item < n_items; item += n_blocks) {
#ifndef PK_NOWORK
            run_item_wave(L, *t, item * L.chunks + wave_in_block, lane);
#endif
            ++mine;
        }
        __syncthreads();                                                        // all waves' stores issued
        if (tlt >= 0 && tlt < 4 && threadIdx.x == 0) L.tl[(tlt * 2048 + blockIdx.x) * 4 + 1] = wall_clock64();
        if (threadIdx.x == 0 && mine > 0) {
            const int d = PK_DONE_ADD(&st->done, mine) + mine;                    // release: stores ordered before the count
            if (d == n_items) {
                const int dep_first = t->dep_first, dep_count = t->dep_count;
                for (int k = 0; k < dep_count; ++k) add_release(&L.state[L.dependents[dep_first + k]].deps_done, 1);
                if (tlt >= 0 && tlt < 4) L.tl[(tlt * 2048 + 2047) * 4 + 3] = wall_clock64();   // release moment
            }
        }
        if (tlt >= 0 && tlt < 4 && threadIdx.x == 0) L.tl[(tlt * 2048 + blockIdx.x) * 4 + 2] = wall_clock64();
    }
}

// ---- the same ops as one kernel per task (the hipGraph baseline) ------------------------------------------------
__global__ void __launch_bounds__(256) pk_one(pk_list L, int ti) {
    const pk_task t = L.tasks[ti];
    if ((int) blockIdx.x < t.n_items) run_item(L, t, blockIdx.x);
}

// ---- host: build the DAG ----------------------------------------------------------------------------------------
struct dag {
    std::vector<pk_task> tasks; std::vector<int> deps_of_task; // flattened later
    std::vector<std::vector<int>> preds;
    int rows = 0;
    int add(pk_op op, int n_items, std::vector<int> pred, int s0, int s1, int s2, int s3, float a, float b) {
        pk_task t{}; t.op = op; t.n_items = n_items; t.row0 = rows; rows += n_items;
        t.src[0] = s0; t.src[1] = s1; t.src[2] = s2; t.src[3] = s3; t.a = a; t.b = b;
        t.n_deps = (int) pred.size();
        tasks.push_back(t); preds.push_back(pred);
        return (int) tasks.size() - 1;
    }
};

int main(int argc, char ** argv) {
    const int N       = argc > 1 ? atoi(argv[1]) : 4096;   // floats per row (an activation)
    const int items   = argc > 2 ? atoi(argv[2]) : 32;     // rows per task
    const int layers  = argc > 3 ? atoi(argv[3]) : 160;    // 6 tasks per layer -> ~1000 tasks
    setvbuf(stdout, NULL, _IONBF, 0);
    hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, 0));
    printf("dag build...\n");

    dag g;
    int in = g.add(PK_NOP, items, {}, 0, 0, 0, 0, 0.f, 0.f);   // rows [0, items) are the input, filled by the host
    int cur = in;
    for (int l = 0; l < layers; ++l) {
        int norm = g.add(PK_SCALE, items, {cur}, g.tasks[cur].row0, 0, 0, 0, 0.999f, 0.001f);
        int b[4];
        for (int k = 0; k < 4; ++k) b[k] = g.add(PK_SCALE, items, {norm}, g.tasks[norm].row0, 0, 0, 0, 1.0f + 0.01f * k, 0.5f * k);
        cur = g.add(PK_SUM4, items, {b[0], b[1], b[2], b[3]}, g.tasks[b[0]].row0, g.tasks[b[1]].row0, g.tasks[b[2]].row0, g.tasks[b[3]].row0, 0.25f, 0.f);
    }
    const int n_tasks = (int) g.tasks.size();
    // dependents CSR
    std::vector<std::vector<int>> dependents(n_tasks);
    for (int t = 0; t < n_tasks; ++t) for (int pr : g.preds[t]) dependents[pr].push_back(t);
    std::vector<int> csr; for (int t = 0; t < n_tasks; ++t) { g.tasks[t].dep_first = (int) csr.size(); g.tasks[t].dep_count = (int) dependents[t].size(); for (int d : dependents[t]) csr.push_back(d); }

    printf("cpu reference...\n");
    // CPU reference
    std::vector<float> h_buf((size_t) g.rows * N);
    for (int i = 0; i < items * N; ++i) h_buf[i] = (float) ((i * 7919) % 1000) / 1000.f - 0.5f;
    std::vector<float> ref = h_buf;
    for (int t = 1; t < n_tasks; ++t) {
        const pk_task & tk = g.tasks[t];
        for (int it = 0; it < tk.n_items; ++it) for (int i = 0; i < N; ++i) {
            float * dst = &ref[(size_t) (tk.row0 + it) * N + i];
            if (tk.op == PK_SCALE) *dst = ref[(size_t) (tk.src[0] + it) * N + i] * tk.a + tk.b;
            else if (tk.op == PK_SUM4) { float s = 0; for (int k = 0; k < 4; ++k) s += ref[(size_t) (tk.src[k] + it) * N + i]; *dst = s * tk.a; }
        }
    }
    const pk_task & last = g.tasks[n_tasks - 1];
    double ref_sum = 0; for (int i = 0; i < last.n_items * N; ++i) ref_sum += ref[(size_t) last.row0 * N + i];

    printf("device setup...\n");
    // device buffers
    const int bs = argc > 7 ? atoi(argv[7]) : 256;
    pk_list L{}; L.n_tasks = n_tasks; L.N = N; L.chunks = bs / 32; L.n_groups = argc > 6 ? atoi(argv[6]) : 20;
    CHECK(hipMalloc((void **) &L.group_ready, 4096 * 32 * 4));
    CHECK(hipMalloc((void **) &L.tl, 4 * 2048 * 4 * 8)); CHECK(hipMemset(L.tl, 0, 4 * 2048 * 4 * 8));
    CHECK(hipHostMalloc((void **) &L.abort_flag, 4096, hipHostMallocCoherent)); *L.abort_flag = 0;
    CHECK(hipMalloc((void **) &L.progress, 65536 * 4)); CHECK(hipMemset(L.progress, 0, 65536 * 4));
    printf("  a1\n"); CHECK(hipMalloc((void **) &L.tasks, n_tasks * sizeof(pk_task)));
    printf("  a2\n"); CHECK(hipMalloc((void **) &L.state, n_tasks * sizeof(pk_state)));
    CHECK(hipMalloc((void **) &L.dependents, csr.size() * sizeof(int) + 4));
    printf("  a4\n"); CHECK(hipMalloc((void **) &L.buf, h_buf.size() * sizeof(float)));
    printf("  c1\n"); CHECK(hipMemcpy((void *) L.tasks, g.tasks.data(), n_tasks * sizeof(pk_task), hipMemcpyHostToDevice));
    printf("  c2\n"); CHECK(hipMemcpy((void *) L.dependents, csr.data(), csr.size() * sizeof(int), hipMemcpyHostToDevice));
    printf("  allocs done, copying %zu MB...\n", h_buf.size() * 4 >> 20); CHECK(hipMemcpy(L.buf, h_buf.data(), h_buf.size() * sizeof(float), hipMemcpyHostToDevice)); printf("  copy done\n");

    int max_blocks = 0; CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, pk_run, bs, 0));
    const int per_cu = argc > 4 ? atoi(argv[4]) : 2;
    const int grid = per_cu < 0 ? -per_cu : (per_cu > 0 ? per_cu : max_blocks) * p.multiProcessorCount;
    printf("%s: %d tasks (%d layers x 6), %d rows x %d chunks of %d floats per task, %.1f MB of activations; persistent grid %d blocks (%d/CU x %d, occupancy says %d/CU)\n",
           p.gcnArchName, n_tasks, layers, items, L.chunks, N, h_buf.size() * 4 / 1e6, grid, grid / p.multiProcessorCount, p.multiProcessorCount, max_blocks); fflush(stdout);

    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1)); float ms;
    auto reset = [&]() { CHECK(hipMemset(L.group_ready, 0xff, 4096 * 32 * 4)); CHECK(hipMemset(L.state, 0, n_tasks * sizeof(pk_state))); CHECK(hipMemset(&L.state[0], 0, sizeof(pk_state))); };
    auto check = [&](const char * what) {
        std::vector<float> out((size_t) last.n_items * N);
        CHECK(hipMemcpy(out.data(), L.buf + (size_t) last.row0 * N, out.size() * 4, hipMemcpyDeviceToHost));
        double s = 0, maxrel = 0; for (size_t i = 0; i < out.size(); ++i) { s += out[i]; double r = fabs(out[i] - ref[(size_t) last.row0 * N + i]) / (fabs(ref[(size_t) last.row0 * N + i]) + 1e-3); if (r > maxrel) maxrel = r; }
        printf("   %-12s checksum %.6e vs ref %.6e, max rel err %.2e -> %s\n", what, s, ref_sum, maxrel, maxrel < 1e-4 ? "OK" : "MISMATCH");
    };
    // task 0 is the input: mark it done up front
    auto prime = [&]() { reset(); pk_state s0{}; s0.done = items; CHECK(hipMemcpy(&L.state[0], &s0, sizeof(s0), hipMemcpyHostToDevice));
                         // and its dependents' counters
                         std::vector<pk_state> st(n_tasks, pk_state{}); st[0] = s0; for (int d : dependents[0]) st[d].deps_done = 1;
                         CHECK(hipMemcpy(L.state, st.data(), n_tasks * sizeof(pk_state), hipMemcpyHostToDevice)); };

    const int mode = argc > 5 ? atoi(argv[5]) : 3;   // 1 persistent, 2 graph, 3 both
    const int chunks = argc > 6 ? atoi(argv[6]) : 8;
    void * args[] = { &L };
    const int reps = 20; float best = 1e9, total = 0;
    printf("launch persistent...\n");
    if (mode & 1) {
    // 1. persistent kernel: correctness, then timing over reps
    prime(); CHECK(hipLaunchCooperativeKernel((const void *) pk_run, dim3(grid), dim3(bs), args, 0, 0)); CHECK(hipEventRecord(e1));
    for (int w = 0; w < 100 && hipEventQuery(e1) != hipSuccess; ++w) { usleep(100000); }
    if (hipEventQuery(e1) != hipSuccess) {
        printf("STALL after 10 s: setting abort. progress per block (task*4+phase; phase 0 waiting deps, 1 running, 3 aborted while waiting):\n");
        *L.abort_flag = 1; CHECK(hipDeviceSynchronize());
        std::vector<int> prog(65536); CHECK(hipMemcpy(prog.data(), L.progress, 65536 * 4, hipMemcpyDeviceToHost));
        for (int b = 0; b < grid && b < 16; ++b) printf("  block %3d: task %d phase %d\n", b, prog[b] / 4, prog[b] % 4);
        std::vector<pk_state> st(n_tasks); CHECK(hipMemcpy(st.data(), L.state, n_tasks * sizeof(pk_state), hipMemcpyDeviceToHost));
        for (int t = 0; t < n_tasks && t < 16; ++t) printf("  task %2d op %d n_items %d n_deps %d: cursor %d done %d deps_done %d\n", t, g.tasks[t].op, g.tasks[t].n_items, g.tasks[t].n_deps, st[t].cursor, st[t].done, st[t].deps_done);
        return 1;
    }
    check("persistent");
    best = 1e9; total = 0;
    for (int r = 0; r < reps; ++r) {
        prime(); CHECK(hipEventRecord(e0)); CHECK(hipLaunchCooperativeKernel((const void *) pk_run, dim3(grid), dim3(bs), args, 0, 0)); CHECK(hipEventRecord(e1)); CHECK(hipDeviceSynchronize());
        CHECK(hipEventElapsedTime(&ms, e0, e1)); total += ms; if (ms < best) best = ms;
    }
    printf("   persistent: %.3f ms best, %.3f ms mean per DAG (%.2f us per task)\n", best, total / reps, 1000 * best / n_tasks);
    {
        std::vector<unsigned long long> tl(4 * 2048 * 4); CHECK(hipMemcpy(tl.data(), L.tl, tl.size() * 8, hipMemcpyDeviceToHost));
        const double tick_us = 0.01;   // wall_clock64 runs at 100 MHz on these parts
        for (int k = 0; k < 4; ++k) {
            unsigned long long rel_prev = k > 0 ? tl[((k - 1) * 2048 + 2047) * 4 + 3] : 0, rel = tl[(k * 2048 + 2047) * 4 + 3];
            unsigned long long seen_min = ~0ull, seen_max = 0, done_max = 0, ret_max = 0; int nb = 0;
            for (int b = 0; b < grid; ++b) { const unsigned long long * e = &tl[(k * 2048 + b) * 4]; if (!e[0]) continue; ++nb; seen_min = std::min(seen_min, e[0]); seen_max = std::max(seen_max, e[0]); done_max = std::max(done_max, e[1]); ret_max = std::max(ret_max, e[2]); }
            printf("   task %d (%d blocks): released by predecessor at +0, first block saw it +%.2f us, last block +%.2f, last work done +%.2f, last retire +%.2f, this task released +%.2f us\n",
                   100 + k, nb, (seen_min - rel_prev) * tick_us, (seen_max - rel_prev) * tick_us, (done_max - rel_prev) * tick_us, (ret_max - rel_prev) * tick_us, (rel - rel_prev) * tick_us);
        }
    }
    }
    if (mode & 2) {

    // 2. one kernel per task, captured once into a hipGraph and replayed
    hipStream_t s; CHECK(hipStreamCreate(&s)); hipGraph_t gr; hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int t = 1; t < n_tasks; ++t) hipLaunchKernelGGL(pk_one, dim3(g.tasks[t].n_items), dim3(256), 0, s, L, t);
    CHECK(hipStreamEndCapture(s, &gr)); CHECK(hipGraphInstantiate(&ge, gr, nullptr, nullptr, 0));
    CHECK(hipMemcpy(L.buf, h_buf.data(), h_buf.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipGraphLaunch(ge, s)); CHECK(hipStreamSynchronize(s)); check("hipGraph");
    best = 1e9; total = 0;
    for (int r = 0; r < reps; ++r) {
        CHECK(hipEventRecord(e0, s)); CHECK(hipGraphLaunch(ge, s)); CHECK(hipEventRecord(e1, s)); CHECK(hipStreamSynchronize(s));
        CHECK(hipEventElapsedTime(&ms, e0, e1)); total += ms; if (ms < best) best = ms;
    }
    printf("   hipGraph:   %.3f ms best, %.3f ms mean per DAG (%.2f us per node)\n", best, total / reps, 1000 * best / (n_tasks - 1));
    }
    return 0;
}
