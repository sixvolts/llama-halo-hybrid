// Standalone check of the KDA state-gather deferral under REAL ggml-alloc (gallocr) allocation, GLM KDA-layer node
// order (glm5next build_kda_layer + build_conv_state + build_rs + build_recurrent_attn), n_seqs = 1, rollback K slots.
// Runs the same step sequence on a GPU backend and on CPU, compares output + both recurrent caches after each step.
// usage: sg <backend> <T> [neg]   (neg: a cpy into the ssm cache between the gather and the gdn -> gate must decline)
// build: g++ -O2 -std=c++17 -I ggml/include docs/halo-hybrid/kda_state_gather_gallocr.cpp -o sg \
//            -L build-hip/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build-hip/bin
//         SG_SCHED=1 allocates through ggml_backend_sched (graph_optimize alloc deps) like llama.cpp
// proof:  rocprofv3 --kernel-trace --output-format csv -d p -o run -- env SG_NO_REF=1 ./sg ROCm0 3
//         -> gated_delta_net_cuda<128, true, true, true> per layer, no k_get_rows_float_vec<float>
//         (T=1: the last layer keeps its get_rows -- ggml-alloc hands the freed s_copy bytes to a later node)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int64_t n_embd = 256, H = 64, S = 128, d_inner = H*S, d_conv = 4, D = S*S*H;
static const int64_t mem_size = 1, Kslots = 3, n_layer = 3;

struct Model {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    struct L { ggml_tensor *wq,*wk,*wv,*conv_w,*fa,*fb,*dtb,*a,*wbeta,*wo,*onorm,*s_l,*r_l; } l[n_layer];
};

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(ggml_nelements(t));
    for (auto & x : v) x = d(rng);
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

static Model make_model(ggml_backend_t be) {
    Model m;
    ggml_init_params p = { ggml_tensor_overhead()*256, nullptr, true };
    m.ctx = ggml_init(p);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = m.l[il];
        L.wq = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, n_embd, d_inner);
        L.wk = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, n_embd, d_inner);
        L.wv = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, n_embd, d_inner);
        L.conv_w = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, d_conv, 3*d_inner);
        L.fa = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, n_embd, 32);
        L.fb = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, 32, d_inner);
        L.dtb = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, d_inner);
        L.a = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, H);
        L.wbeta = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, n_embd, H);
        L.wo = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, d_inner, n_embd);
        L.onorm = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, S);
        L.s_l = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, D, mem_size*Kslots);
        L.r_l = ggml_new_tensor_2d(m.ctx, GGML_TYPE_F32, (d_conv-1)*3*d_inner, mem_size*Kslots);
    }
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, be);
    std::mt19937 rng(42);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = m.l[il];
        fill(L.wq, rng, -0.06f, 0.06f); fill(L.wk, rng, -0.06f, 0.06f); fill(L.wv, rng, -0.06f, 0.06f);
        fill(L.conv_w, rng, -0.5f, 0.5f); fill(L.fa, rng, -0.1f, 0.1f); fill(L.fb, rng, -0.2f, 0.2f);
        fill(L.dtb, rng, -1.0f, 1.0f); fill(L.a, rng, 0.2f, 1.5f); fill(L.wbeta, rng, -0.1f, 0.1f);
        fill(L.wo, rng, -0.01f, 0.01f); fill(L.onorm, rng, 0.5f, 1.5f);
        fill(L.s_l, rng, -0.05f, 0.05f); fill(L.r_l, rng, -1.0f, 1.0f);
    }
    return m;
}

struct Graph { ggml_context * ctx; ggml_cgraph * gf; ggml_tensor * x, * s_copy, * junk, * out; };

// build_rs as in llama-graph.cpp (zero view + main gather + extra gather/cpy), n_rs == n_seqs == 1
static ggml_tensor * build_rs(ggml_context * ctx, ggml_cgraph * gf, ggml_tensor * s, int64_t state_size,
                              ggml_tensor * s_copy_main, ggml_tensor * s_copy_extra, int64_t head) {
    ggml_tensor * states = ggml_reshape_2d(ctx, s, state_size, s->ne[1]);
    ggml_tensor * state_zero = ggml_view_1d(ctx, states, 0, 0);
    ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, state_zero, 0));
    ggml_tensor * out = ggml_get_rows(ctx, states, s_copy_main);
    ggml_build_forward_expand(gf, out);
    ggml_tensor * extra = ggml_get_rows(ctx, states, s_copy_extra);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, extra, ggml_view_2d(ctx, s, state_size, 0, s->nb[1], (head + 1)*s->nb[1])));
    return out;
}

static Graph build(Model & m, int64_t T, bool neg) {
    Graph g;
    ggml_init_params p = { ggml_tensor_overhead()*4096 + ggml_graph_overhead_custom(4096, false), nullptr, true };
    g.ctx = ggml_init(p);
    ggml_context * ctx = g.ctx;
    g.gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_cgraph * gf = g.gf;
    const int64_t head = 0;
    g.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T); ggml_set_input(g.x);
    g.s_copy = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(g.s_copy);
    g.junk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D); ggml_set_input(g.junk);
    ggml_tensor * s_copy_main  = ggml_view_1d(ctx, g.s_copy, 1, 0);
    ggml_tensor * s_copy_extra = ggml_view_1d(ctx, g.s_copy, 0, g.s_copy->nb[0]);
    ggml_tensor * cur = g.x;
    for (int il = 0; il < n_layer; ++il) {
        auto & L = m.l[il];
        ggml_tensor * inp = cur;
        ggml_tensor * Q = ggml_mul_mat(ctx, L.wq, inp), * K = ggml_mul_mat(ctx, L.wk, inp), * V = ggml_mul_mat(ctx, L.wv, inp);
        ggml_tensor * qkv = ggml_reshape_3d(ctx, ggml_concat(ctx, ggml_concat(ctx, Q, K, 0), V, 0), 3*d_inner, T, 1);
        // build_conv_state
        ggml_tensor * cs = build_rs(ctx, gf, L.r_l, (d_conv-1)*3*d_inner, s_copy_main, s_copy_extra, head);
        cs = ggml_reshape_3d(ctx, cs, d_conv - 1, 3*d_inner, 1);
        ggml_tensor * conv_input = ggml_concat(ctx, cs, ggml_transpose(ctx, qkv), 0);
        const size_t row_size = ggml_row_size(GGML_TYPE_F32, (d_conv-1)*3*d_inner);
        for (int64_t t = 1; t <= Kslots; ++t) {
            const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - cs->ne[0] - Kslots + t);
            ggml_tensor * last = ggml_view_3d(ctx, conv_input, d_conv - 1, 3*d_inner, 1, conv_input->nb[1], conv_input->nb[2],
                                              ggml_row_size(GGML_TYPE_F32, s_idx));
            ggml_tensor * upd = ggml_view_2d(ctx, L.r_l, (d_conv-1)*3*d_inner, 1, L.r_l->nb[1], ((Kslots - t)*mem_size + head)*row_size);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, last, upd));
        }
        ggml_tensor * conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, L.conv_w));
        const size_t nb_qkv = ggml_row_size(GGML_TYPE_F32, 3*d_inner), nb_head = ggml_row_size(GGML_TYPE_F32, S);
        Q = ggml_view_4d(ctx, conv_out, S, H, T, 1, nb_head, nb_qkv, nb_qkv*T, 0);
        K = ggml_view_4d(ctx, conv_out, S, H, T, 1, nb_head, nb_qkv, nb_qkv*T, ggml_row_size(GGML_TYPE_F32, d_inner));
        V = ggml_view_4d(ctx, conv_out, S, H, T, 1, nb_head, nb_qkv, nb_qkv*T, ggml_row_size(GGML_TYPE_F32, 2*d_inner));
        Q = ggml_l2_norm(ctx, Q, 1e-6f);
        K = ggml_l2_norm(ctx, K, 1e-6f);
        ggml_tensor * gg = ggml_mul_mat(ctx, L.fb, ggml_mul_mat(ctx, L.fa, inp));
        gg = ggml_add(ctx, gg, L.dtb);
        gg = ggml_reshape_3d(ctx, gg, S, H, T);
        gg = ggml_mul(ctx, gg, ggml_reshape_3d(ctx, L.a, 1, H, 1));
        gg = ggml_sigmoid(ctx, ggml_scale(ctx, gg, -1.0f));
        gg = ggml_scale(ctx, gg, -5.0f);
        gg = ggml_reshape_4d(ctx, gg, S, H, T, 1);
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_4d(ctx, ggml_mul_mat(ctx, L.wbeta, inp), 1, H, T, 1));
        // state gather
        ggml_tensor * state = build_rs(ctx, gf, L.s_l, D, s_copy_main, s_copy_extra, head);
        if (neg && il == 1) {
            // a writer into the ssm cache between the gather and the gdn: the deferral must decline
            ggml_build_forward_expand(gf, ggml_cpy(ctx, g.junk, ggml_view_1d(ctx, L.s_l, D, 1*mem_size*L.s_l->nb[1])));
        }
        state = ggml_reshape_4d(ctx, state, S, S, H, 1);
        ggml_tensor * gdn = ggml_gated_delta_net(ctx, Q, K, V, gg, beta, state, Kslots);
        const int64_t n_written = std::min<int64_t>(T, Kslots);
        ggml_tensor * outv = ggml_view_4d(ctx, gdn, S, H, T, 1, ggml_row_size(GGML_TYPE_F32, S), ggml_row_size(GGML_TYPE_F32, S*H),
                                          ggml_row_size(GGML_TYPE_F32, S*H*T), 0);
        const size_t rs = ggml_row_size(GGML_TYPE_F32, D);
        ggml_tensor * src = ggml_view_3d(ctx, gdn, D, 1, n_written, rs, rs, ggml_row_size(GGML_TYPE_F32, S*H*T));
        ggml_tensor * dst = ggml_view_3d(ctx, L.s_l, D, 1, n_written, L.s_l->nb[1], (size_t) mem_size*rs, (size_t) head*rs);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        ggml_tensor * o = ggml_cont_3d(ctx, outv, S, H, T);
        o = ggml_mul(ctx, ggml_rms_norm(ctx, o, 1e-5f), L.onorm);
        cur = ggml_add(ctx, cur, ggml_mul_mat(ctx, L.wo, ggml_cont_2d(ctx, o, d_inner, T)));
    }
    g.out = cur;
    ggml_set_output(g.out);
    ggml_build_forward_expand(gf, g.out);
    return g;
}

struct Run {
    std::vector<std::vector<float>> steps; // per step: out, then s_l/r_l of every layer
};

static Run run(const char * be_name, int64_t T, bool neg, int n_steps) {
    ggml_backend_t be = strcmp(be_name, "CPU") == 0 ? ggml_backend_cpu_init() : ggml_backend_init_by_name(be_name, nullptr);
    if (!be) { fprintf(stderr, "no backend %s\n", be_name); exit(1); }
    Model m = make_model(be);
    Graph g = build(m, T, neg);
    // SG_SCHED=1: allocate through ggml_backend_sched (backend graph_optimize + its alloc deps, as llama.cpp does)
    // instead of a bare gallocr
    const bool use_sched = getenv("SG_SCHED") != nullptr && strcmp(be_name, "CPU") != 0;
    ggml_gallocr_t ga = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t cpu = nullptr;
    if (use_sched) {
        cpu = ggml_backend_cpu_init();
        ggml_backend_t bes[2] = { be, cpu };
        sched = ggml_backend_sched_new(bes, nullptr, 2, 4096, false, true);
        if (!ggml_backend_sched_alloc_graph(sched, g.gf)) { fprintf(stderr, "sched alloc failed\n"); exit(1); }
    } else {
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        if (!ggml_gallocr_alloc_graph(ga, g.gf)) { fprintf(stderr, "alloc failed\n"); exit(1); }
    }
    Run r;
    std::mt19937 rng(7);
    const int planes[] = { 1, 0, 2, 1, 2, 0 };
    for (int st = 0; st < n_steps; ++st) {
        fill(g.x, rng, -1.0f, 1.0f);
        if (g.junk->buffer) { fill(g.junk, rng, -0.05f, 0.05f); }
        const int32_t idx = planes[st % 6]*mem_size + 0;
        ggml_backend_tensor_set(g.s_copy, &idx, 0, sizeof(idx));
        if (sched) { ggml_backend_sched_graph_compute(sched, g.gf); } else { ggml_backend_graph_compute(be, g.gf); }
        std::vector<float> v(ggml_nelements(g.out));
        ggml_backend_tensor_get(g.out, v.data(), 0, ggml_nbytes(g.out));
        for (int il = 0; il < n_layer; ++il) {
            for (ggml_tensor * t : { m.l[il].s_l, m.l[il].r_l }) {
                std::vector<float> c(ggml_nelements(t));
                ggml_backend_tensor_get(t, c.data(), 0, ggml_nbytes(t));
                v.insert(v.end(), c.begin(), c.end());
            }
        }
        r.steps.push_back(std::move(v));
    }
    if (sched) {
        fprintf(stderr, "%s (sched): graph nodes %d, splits %d\n", be_name, ggml_graph_n_nodes(g.gf), ggml_backend_sched_get_n_splits(sched));
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
    } else {
        fprintf(stderr, "%s: graph nodes %d, compute buffer %zu bytes\n", be_name, ggml_graph_n_nodes(g.gf), ggml_gallocr_get_buffer_size(ga, 0));
        ggml_gallocr_free(ga);
    }
    ggml_backend_buffer_free(m.buf);
    ggml_free(m.ctx); ggml_free(g.ctx);
    ggml_backend_free(be);
    return r;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * be = argc > 1 ? argv[1] : "ROCm0";
    const int64_t T = argc > 2 ? atoll(argv[2]) : 3;
    const bool neg = argc > 3 && strcmp(argv[3], "neg") == 0;
    const int n_steps = 6;
    Run a = run(be, T, neg, n_steps);
    if (getenv("SG_NO_REF")) { printf("done (no reference)\n"); return 0; }
    Run b = run("CPU", T, neg, n_steps);
    bool ok = true;
    for (int st = 0; st < n_steps; ++st) {
        double num = 0, den = 0, maxd = 0;
        const auto & x = a.steps[st], & y = b.steps[st];
        for (size_t i = 0; i < x.size(); ++i) {
            const double d = x[i] - y[i];
            num += d*d; den += (double) y[i]*y[i]; maxd = std::max(maxd, std::fabs(d));
            if (!std::isfinite(x[i])) { num = INFINITY; }
        }
        const double nmse = num/den;
        printf("%s T=%lld%s step %d: nmse %.3e maxdiff %.3e %s\n", be, (long long) T, neg ? " neg" : "", st, nmse, maxd,
               nmse < 1e-9 ? "OK" : "FAIL");
        ok = ok && nmse < 1e-9;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
