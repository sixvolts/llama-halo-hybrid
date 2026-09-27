// Batched eager copies at a split boundary (ggml_backend_i::cpy_tensors_async_nowait, CUDA/HIP k_peer_copy_multi).
//
// 1. iface: n tensors (16 B .. 300 KB, one misaligned size, more than 8 so the kernel is launched twice) copied from
//    device 0 to device 1 and back in one call, compared byte for byte.
// 2. scheduler: an L-layer MoE-shaped graph whose router half (norm-ish add, router GEMV, softmax, top-k, weights) runs
//    on one GPU and whose expert half (mul_mat_id with the weights and a sum over the k experts) on the other, so every
//    layer has a 3-input boundary (activation, expert weights, ids) that the eager copies loop pushes as one batch.
//    Run at widths n = 1, 3, 8, 64, 512 in both directions, with eager copies on (batched) and off (consumer-side
//    copies), compared bit for bit with each other and against the CPU backend (NMSE).
// 3. perf (argv[1] == "perf"): decode-shaped (n = 3) graph, host submit time and wall time per graph compute.
//    Run twice, with and without GGML_SCHED_NO_COPY_BATCH=1.
//
// Needs two GPU devices; exits 0 with a message otherwise.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<ggml_backend_dev_t> gpu_devices() {
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            devs.push_back(d);
        }
    }
    return devs;
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. iface test

static bool test_iface(ggml_backend_t b_src, ggml_backend_t b_dst) {
    if (!b_src->iface.cpy_tensors_async_nowait) {
        printf("  iface: %s has no cpy_tensors_async_nowait, skipped\n", ggml_backend_name(b_src));
        return true;
    }
    const std::vector<int64_t> n_bytes = { 16, 96, 96, 4096*3*4, 6144*3*4, 1024, 300000 - 300000 % 16, 100, 48, 64*1024, 16*1024, 262144 };
    const int n = (int) n_bytes.size();

    ggml_init_params ip = { ggml_tensor_overhead() * (2*n + 4), NULL, true };
    ggml_context * ctx_s = ggml_init(ip);
    ggml_context * ctx_d = ggml_init(ip);
    std::vector<const ggml_tensor *> srcs;
    std::vector<ggml_tensor *> dsts, srcs_m;
    for (int i = 0; i < n; i++) {
        ggml_tensor * s = ggml_new_tensor_1d(ctx_s, GGML_TYPE_I8, n_bytes[i]);
        ggml_tensor * d = ggml_new_tensor_1d(ctx_d, GGML_TYPE_I8, n_bytes[i]);
        srcs.push_back(s); srcs_m.push_back(s); dsts.push_back(d);
    }
    ggml_backend_buffer_t buf_s = ggml_backend_alloc_ctx_tensors(ctx_s, b_src);
    ggml_backend_buffer_t buf_d = ggml_backend_alloc_ctx_tensors(ctx_d, b_dst);

    std::mt19937 rng(1234);
    std::vector<std::vector<uint8_t>> data(n);
    for (int i = 0; i < n; i++) {
        data[i].resize(n_bytes[i]);
        for (auto & v : data[i]) { v = (uint8_t) rng(); }
        ggml_backend_tensor_set(srcs_m[i], data[i].data(), 0, n_bytes[i]);
        std::vector<uint8_t> z(n_bytes[i], 0xAB);
        ggml_backend_tensor_set(dsts[i], z.data(), 0, n_bytes[i]);
    }
    ggml_backend_synchronize(b_dst);
    bool ok = b_src->iface.cpy_tensors_async_nowait(b_src, b_dst, n, srcs.data(), dsts.data());
    if (!ok) {
        printf("  iface %s -> %s: returned false (no peer path), nothing to check\n", ggml_backend_name(b_src), ggml_backend_name(b_dst));
        ok = true;
    } else {
        ggml_backend_synchronize(b_src);
        for (int i = 0; i < n; i++) {
            std::vector<uint8_t> got(n_bytes[i]);
            ggml_backend_tensor_get(dsts[i], got.data(), 0, n_bytes[i]);
            if (memcmp(got.data(), data[i].data(), n_bytes[i]) != 0) {
                printf("  iface %s -> %s: segment %d (%" PRId64 " bytes) MISMATCH\n", ggml_backend_name(b_src), ggml_backend_name(b_dst), i, n_bytes[i]);
                ok = false;
            }
        }
        printf("  iface %s -> %s: %d segments %s\n", ggml_backend_name(b_src), ggml_backend_name(b_dst), n, ok ? "OK" : "FAIL");
    }
    ggml_backend_buffer_free(buf_s);
    ggml_backend_buffer_free(buf_d);
    ggml_free(ctx_s);
    ggml_free(ctx_d);
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------
// 2./3. scheduler test

struct moe_cfg {
    int H = 512;   // hidden
    int E = 16;    // experts
    int K = 4;     // experts per token
    int L = 4;     // layers
    int F = 64;    // expert output rows (added back to h by broadcast; the boundary, not the expert GEMV, is the subject)
};

struct moe_weights {
    std::vector<std::vector<float>> bias, wr, we; // per layer
};

static moe_weights make_weights(const moe_cfg & c) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    moe_weights w;
    for (int l = 0; l < c.L; l++) {
        std::vector<float> b(c.H), r((size_t) c.H*c.E), e((size_t) c.H*c.F*c.E);
        for (auto & v : b) { v = 0.1f*u(rng); }
        for (auto & v : r) { v = u(rng) / sqrtf((float) c.H) * 4.0f; }
        for (auto & v : e) { v = u(rng) / sqrtf((float) c.H); }
        w.bias.push_back(b); w.wr.push_back(r); w.we.push_back(e);
    }
    return w;
}

// b_router / b_expert: backends holding the router-side / expert-side weights and running those ops.
// eager: ggml_backend_sched_set_eager_copies. Returns the output [H, n].
static std::vector<float> run_moe(const moe_cfg & c, const moe_weights & w, int n, ggml_backend_t b_router, ggml_backend_t b_expert,
        ggml_backend_t b_cpu, bool eager, int n_iter = 1, double * t_submit_us = nullptr, double * t_wall_us = nullptr) {
    ggml_init_params wp = { ggml_tensor_overhead() * (4*c.L + 4), NULL, true };
    ggml_context * ctx_wr = ggml_init(wp);
    ggml_context * ctx_we = ggml_init(wp);
    std::vector<ggml_tensor *> t_bias, t_wr, t_we;
    for (int l = 0; l < c.L; l++) {
        t_bias.push_back(ggml_new_tensor_1d(ctx_wr, GGML_TYPE_F32, c.H));
        t_wr  .push_back(ggml_new_tensor_2d(ctx_wr, GGML_TYPE_F32, c.H, c.E));
        t_we  .push_back(ggml_new_tensor_3d(ctx_we, GGML_TYPE_F32, c.H, c.F, c.E));
    }
    ggml_backend_buffer_t buf_wr = ggml_backend_alloc_ctx_tensors(ctx_wr, b_router);
    ggml_backend_buffer_t buf_we = ggml_backend_alloc_ctx_tensors(ctx_we, b_expert);
    ggml_backend_buffer_set_usage(buf_wr, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(buf_we, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int l = 0; l < c.L; l++) {
        ggml_backend_tensor_set(t_bias[l], w.bias[l].data(), 0, ggml_nbytes(t_bias[l]));
        ggml_backend_tensor_set(t_wr[l],   w.wr[l].data(),   0, ggml_nbytes(t_wr[l]));
        ggml_backend_tensor_set(t_we[l],   w.we[l].data(),   0, ggml_nbytes(t_we[l]));
    }

    const size_t graph_size = 64*c.L + 64;
    ggml_init_params gp = { ggml_tensor_overhead()*graph_size + ggml_graph_overhead_custom(graph_size, false), NULL, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    std::vector<std::pair<ggml_tensor *, ggml_backend_t>> placement;
    auto on = [&](ggml_tensor * t, ggml_backend_t b) { placement.push_back({t, b}); return t; };

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.H, n);
    ggml_set_input(x);
    ggml_set_name(x, "x");
    ggml_tensor * h = x;
    for (int l = 0; l < c.L; l++) {
        ggml_tensor * a      = on(ggml_add(ctx, h, t_bias[l]), b_router);                                   // "ffn_norm" [H, n]
        ggml_tensor * logits = on(ggml_mul_mat(ctx, t_wr[l], a), b_router);                                 // [E, n]
        ggml_tensor * probs  = on(ggml_soft_max(ctx, logits), b_router);                                    // [E, n]
        ggml_tensor * ids    = on(ggml_argsort_top_k(ctx, probs, c.K), b_router);                           // [K, n] i32
        ggml_tensor * p3     = on(ggml_reshape_3d(ctx, probs, 1, c.E, n), b_router);
        ggml_tensor * wts    = on(ggml_get_rows(ctx, p3, ids), b_router);                                   // [1, K, n]
        ggml_build_forward_expand(gf, wts); // the router split ends with all three expert inputs (as after topk_moe)
        ggml_tensor * y      = on(ggml_mul_mat_id(ctx, t_we[l], ggml_reshape_3d(ctx, a, c.H, 1, n), ids), b_expert);                         // [F, K, n]
        y                    = on(ggml_mul(ctx, y, wts), b_expert);
        y                    = on(ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3)), b_expert);              // [K, F, n]
        y                    = on(ggml_sum_rows(ctx, y), b_expert);                                         // [1, F, n]
        y                    = on(ggml_reshape_2d(ctx, y, c.F, n), b_expert);
        h                    = on(ggml_add(ctx, h, y), b_router);
    }
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);

    std::vector<ggml_backend_t> backends;
    backends.push_back(b_router);
    if (b_expert != b_router) {
        backends.push_back(b_expert);
    }
    if (b_cpu != b_router) {
        backends.push_back(b_cpu);
    }
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), NULL, (int) backends.size(), graph_size, false, false);
    ggml_backend_sched_set_eager_copies(sched, eager);
    for (auto & p : placement) {
        if (p.first->op != GGML_OP_RESHAPE) {
            ggml_backend_sched_set_tensor_backend(sched, p.first, p.second);
        }
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, gf));

    std::vector<float> xd((size_t) c.H*n);
    std::mt19937 rng(7 + n);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto & v : xd) { v = u(rng); }
    ggml_backend_tensor_set(x, xd.data(), 0, ggml_nbytes(x));

    double sub = 0.0, wall = 0.0;
    for (int it = 0; it < n_iter; it++) {
        const int64_t t0 = ggml_time_us();
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched, gf) == GGML_STATUS_SUCCESS);
        const int64_t t1 = ggml_time_us();
        ggml_backend_sched_synchronize(sched);
        const int64_t t2 = ggml_time_us();
        if (it >= n_iter/5) { // warmup
            sub += t1 - t0; wall += t2 - t0;
        }
    }
    const int n_meas = n_iter - n_iter/5;
    if (t_submit_us) { *t_submit_us = sub / n_meas; }
    if (t_wall_us)   { *t_wall_us   = wall / n_meas; }

    std::vector<float> out((size_t) c.H*n);
    ggml_backend_tensor_get(h, out.data(), 0, ggml_nbytes(h));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf_wr);
    ggml_backend_buffer_free(buf_we);
    ggml_free(ctx_wr);
    ggml_free(ctx_we);
    return out;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - ref[i];
        num += d*d; den += (double) ref[i]*ref[i];
    }
    return num / (den > 0 ? den : 1.0);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const bool perf = argc > 1 && std::string(argv[1]) == "perf";

    std::vector<ggml_backend_dev_t> gpus = gpu_devices();
    if (gpus.size() < 2) {
        printf("test-peer-copy-batch: needs two GPU devices, found %zu: skipped\n", gpus.size());
        return 0;
    }
    ggml_backend_t g0  = ggml_backend_dev_init(gpus[0], NULL);
    ggml_backend_t g1  = ggml_backend_dev_init(gpus[1], NULL);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    printf("devices: %s, %s\n", ggml_backend_name(g0), ggml_backend_name(g1));

    if (perf) {
        // decode shape: n = 3, production-like boundary sizes (hidden 2560 f32 x 3 = 30 KB, 8 of 64 experts)
        moe_cfg c; c.H = 2560; c.E = 64; c.K = 8; c.L = 36; c.F = 64;
        const moe_weights w = make_weights(c);
        const int n_iter = argc > 2 ? atoi(argv[2]) : 500;
        double sub, wall;
        run_moe(c, w, 3, g0, g1, cpu, true, n_iter, &sub, &wall);
        printf("perf %s -> %s n=3 L=%d: submit %.1f us/graph, wall %.1f us/graph (%.1f us/layer)\n",
            ggml_backend_name(g0), ggml_backend_name(g1), c.L, sub, wall, wall / c.L);
        return 0;
    }

    bool ok = true;
    printf("iface:\n");
    ok &= test_iface(g0, g1);
    ok &= test_iface(g1, g0);

    printf("scheduler (MoE boundary, 3 inputs per layer):\n");
    moe_cfg c;
    const moe_weights w = make_weights(c);
    for (int n : { 1, 3, 8, 64, 512 }) {
        const std::vector<float> ref = run_moe(c, w, n, cpu, cpu, cpu, false);
        for (int dir = 0; dir < 2; dir++) {
            ggml_backend_t br = dir == 0 ? g0 : g1;
            ggml_backend_t be = dir == 0 ? g1 : g0;
            const std::vector<float> on  = run_moe(c, w, n, br, be, cpu, true);
            const std::vector<float> off = run_moe(c, w, n, br, be, cpu, false);
            const bool bit = memcmp(on.data(), off.data(), on.size()*sizeof(float)) == 0;
            const double e = nmse(on, ref);
            const bool pass = bit && e < 1e-6;
            printf("  n=%3d %s -> %s: eager(batched) vs consumer copies %s, NMSE vs CPU %.2e  %s\n", n,
                ggml_backend_name(br), ggml_backend_name(be), bit ? "bit-identical" : "DIFFER", e, pass ? "OK" : "FAIL");
            ok &= pass;
        }
    }

    ggml_backend_free(g0);
    ggml_backend_free(g1);
    ggml_backend_free(cpu);
    printf("%s\n", ok ? "ALL OK" : "FAILED");
    return ok ? 0 : 1;
}
