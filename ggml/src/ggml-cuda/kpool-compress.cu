#include "kpool-compress.cuh"
#include "convert.cuh"

#include <cstdlib>
#include <cstring>

#define KPOOL_R_MAX 8

// one block per new pool; one thread per column (d <= 1024 loops). r slots live in registers.
template <typename T>
static __global__ void kpool_compress_kernel(
        const char * __restrict__ kg, const int64_t kg_nb1, const int64_t kg_nb2,
        const int32_t * __restrict__ cells, const int64_t cells_s1,
        const float * __restrict__ ape, const int64_t ape_s1,
        const int64_t * __restrict__ reps,
        char * __restrict__ dst, const int64_t dst_nb1,
        const int d, const int r, const int n_new) {
    const int p = blockIdx.x;
    const int s = blockIdx.y;

    const int32_t * pc   = cells + s*cells_s1 + (int64_t) p*r;
    const char    * base = kg + s*kg_nb2;

    int64_t row[KPOOL_R_MAX];
#pragma unroll
    for (int j = 0; j < KPOOL_R_MAX; ++j) {
        row[j] = j < r ? pc[j] : 0;
    }
    T * out = (T *) (dst + reps[p + (int64_t) s*n_new]*dst_nb1);

    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float k[KPOOL_R_MAX];
        float g[KPOOL_R_MAX];
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < KPOOL_R_MAX; ++j) {
            if (j < r) {
                const T * rp = (const T *) (base + row[j]*kg_nb1);
                k[j] = ggml_cuda_cast<float>(rp[c]);
                g[j] = ggml_cuda_cast<float>(rp[d + c]) + ape[c + j*ape_s1];
                mx = fmaxf(mx, g[j]);
            }
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < KPOOL_R_MAX; ++j) {
            if (j < r) {
                g[j] = expf(g[j] - mx);
                sum += g[j];
            }
        }
        const float inv_sum = 1.0f / sum;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < KPOOL_R_MAX; ++j) {
            if (j < r) {
                acc += k[j] * (g[j] * inv_sum);
            }
        }
        out[c] = ggml_cuda_cast<T>(acc);
    }
}

void ggml_cuda_op_kpool_compress(ggml_backend_cuda_context & ctx, const ggml_cuda_kpool_compress_match & m) {
    cudaStream_t stream = ctx.stream();
    const int d = (int) m.d;
    const dim3 grid((unsigned) m.n_new, (unsigned) m.n_stream, 1);
    const int  nth = d < 1024 ? ((d + 31)/32)*32 : 1024;

    const int64_t cells_s1 = m.cells->nb[1]/sizeof(int32_t);
    const int64_t ape_s1   = m.ape->nb[1]/sizeof(float);

    if (m.kg->type == GGML_TYPE_F16) {
        kpool_compress_kernel<half><<<grid, nth, 0, stream>>>(
            (const char *) m.kg->data, m.kg->nb[1], m.kg->nb[2], (const int32_t *) m.cells->data, cells_s1,
            (const float *) m.ape->data, ape_s1, (const int64_t *) m.reps->data,
            (char *) m.dst->data, m.dst->nb[1], d, (int) m.r, (int) m.n_new);
    } else {
        kpool_compress_kernel<float><<<grid, nth, 0, stream>>>(
            (const char *) m.kg->data, m.kg->nb[1], m.kg->nb[2], (const int32_t *) m.cells->data, cells_s1,
            (const float *) m.ape->data, ape_s1, (const int64_t *) m.reps->data,
            (char *) m.dst->data, m.dst->nb[1], d, (int) m.r, (int) m.n_new);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---- matcher ----------------------------------------------------------------------------------

static bool kp_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}
static int kp_next(const ggml_cgraph * g, int j) {
    while (j < g->n_nodes && kp_is_view_op(g->nodes[j])) { j++; }
    return j;
}
static const ggml_tensor * kp_root(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// PERMUTE(VIEW(members, d x r x n_new x ns at byte offset off), 1, 0, 2, 3) as build_indexer makes it
static bool kp_is_slot_major(const ggml_tensor * p, const ggml_tensor * G, int64_t d, int64_t r, int64_t n_new, int64_t ns, size_t off) {
    return p->op == GGML_OP_PERMUTE && p->view_src == G && p->view_offs == off &&
        p->ne[0] == r && p->ne[1] == d && p->ne[2] == n_new && p->ne[3] == ns &&
        p->nb[0] == G->nb[1] && p->nb[1] == sizeof(float) && p->nb[2] == (size_t) r*G->nb[1] && p->nb[3] == G->nb[2];
}

bool ggml_cuda_kpool_compress_match_graph(const ggml_cgraph * cgraph, int i, ggml_cuda_kpool_compress_match & m) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * G = cgraph->nodes[i];
    if (G->op != GGML_OP_GET_ROWS || G->type != GGML_TYPE_F32 || !ggml_is_contiguous(G)) {
        return false;
    }
    const ggml_tensor * kg    = G->src[0];
    const ggml_tensor * cells = G->src[1];
    if ((kg->type != GGML_TYPE_F16 && kg->type != GGML_TYPE_F32) || kg->nb[0] != ggml_type_size(kg->type) ||
            kg->ne[3] != 1 || cells->type != GGML_TYPE_I32 || !ggml_is_contiguous(cells) ||
            cells->ne[2] != 1 || cells->ne[3] != 1 || cells->ne[1] != kg->ne[2] || G->ne[3] != 1) {
        return false;
    }
    if (kg->ne[0] % 2 != 0) {
        return false;
    }
    const int64_t d  = kg->ne[0]/2;
    const int64_t ns = kg->ne[2];

    // (1) CONT(key) (2) CONT(gate), both slot-major permutes of the members
    const int jk = kp_next(cgraph, i + 1);
    if (jk >= n || cgraph->nodes[jk]->op != GGML_OP_CONT) { return false; }
    const ggml_tensor * ck = cgraph->nodes[jk];
    const int64_t r = ck->ne[0];
    if (r < 1 || r > KPOOL_R_MAX || G->ne[1] % r != 0) { return false; }
    const int64_t n_new = G->ne[1]/r;
    if (!kp_is_slot_major(ck->src[0], G, d, r, n_new, ns, 0) || ck->type != GGML_TYPE_F32) { return false; }

    const int jg = kp_next(cgraph, jk + 1);
    if (jg >= n || cgraph->nodes[jg]->op != GGML_OP_CONT) { return false; }
    const ggml_tensor * cg = cgraph->nodes[jg];
    if (!kp_is_slot_major(cg->src[0], G, d, r, n_new, ns, d*sizeof(float)) || cg->type != GGML_TYPE_F32) { return false; }

    // (3) CONT(TRANSPOSE(ape)) of the constant [d, r] weight
    const int ja = kp_next(cgraph, jg + 1);
    if (ja >= n || cgraph->nodes[ja]->op != GGML_OP_CONT) { return false; }
    const ggml_tensor * ca = cgraph->nodes[ja];
    const ggml_tensor * tr = ca->src[0];
    if (tr->op != GGML_OP_TRANSPOSE || tr->view_offs != 0 || ca->type != GGML_TYPE_F32) { return false; }
    const ggml_tensor * ape = tr->view_src;
    if (ape == nullptr || tr->src[0] != ape || ape->type != GGML_TYPE_F32 || !ggml_is_contiguous(ape) ||
            ape->ne[0] != d || ape->ne[1] != r || ape->ne[2] != 1 || ape->ne[3] != 1) { return false; }

    // (4) ADD(gate, RESHAPE(ape^T)) (5) SOFT_MAX (6) MUL(key, probs) (7) SUM_ROWS
    const int jadd = kp_next(cgraph, ja + 1);
    if (jadd >= n || cgraph->nodes[jadd]->op != GGML_OP_ADD) { return false; }
    const ggml_tensor * add = cgraph->nodes[jadd];
    const ggml_tensor * ar  = add->src[1];
    if (add->src[0] != cg || add->type != GGML_TYPE_F32 || kp_root(ar) != ca || ar->view_offs != 0 ||
            ar->ne[0] != r || ar->ne[1] != d || ar->ne[2] != 1 || ar->ne[3] != 1) { return false; }

    const int jsm = kp_next(cgraph, jadd + 1);
    if (jsm >= n || cgraph->nodes[jsm]->op != GGML_OP_SOFT_MAX) { return false; }
    const ggml_tensor * sm = cgraph->nodes[jsm];
    float sm_scale, sm_bias;
    memcpy(&sm_scale, (const float *) sm->op_params + 0, sizeof(float));
    memcpy(&sm_bias,  (const float *) sm->op_params + 1, sizeof(float));
    if (sm->src[0] != add || sm->src[1] != nullptr || sm->src[2] != nullptr || sm->type != GGML_TYPE_F32 ||
            sm_scale != 1.0f || sm_bias != 0.0f) { return false; }

    const int jmul = kp_next(cgraph, jsm + 1);
    if (jmul >= n || cgraph->nodes[jmul]->op != GGML_OP_MUL) { return false; }
    const ggml_tensor * mul = cgraph->nodes[jmul];
    if (!((mul->src[0] == ck && mul->src[1] == sm) || (mul->src[0] == sm && mul->src[1] == ck)) || mul->type != GGML_TYPE_F32) { return false; }

    const int jsum = kp_next(cgraph, jmul + 1);
    if (jsum >= n || cgraph->nodes[jsum]->op != GGML_OP_SUM_ROWS) { return false; }
    const ggml_tensor * sum = cgraph->nodes[jsum];
    if (sum->src[0] != mul || sum->type != GGML_TYPE_F32) { return false; }

    // (8) SET_ROWS(RESHAPE(sum) [d, n_new*ns], reps I64) into the pooled-head view of the same cache
    const int jset = kp_next(cgraph, jsum + 1);
    if (jset >= n || cgraph->nodes[jset]->op != GGML_OP_SET_ROWS) { return false; }
    ggml_tensor * set = cgraph->nodes[jset];
    const ggml_tensor * sv   = set->src[0];   // set_rows keeps (values, ids, dst) in src[0..2]
    const ggml_tensor * reps = set->src[1];
    const ggml_tensor * dv   = set->src[2];
    if (kp_root(sv) != sum || sv->view_offs != 0 || sv->ne[0] != d || sv->ne[1] != n_new*ns || sv->ne[2] != 1 || sv->ne[3] != 1) { return false; }
    if (reps == nullptr || reps->type != GGML_TYPE_I64 || !ggml_is_contiguous(reps) || reps->ne[0] != n_new*ns ||
            reps->ne[1] != 1 || reps->ne[2] != 1 || reps->ne[3] != 1) { return false; }
    if (set->type != kg->type || dv->type != kg->type || dv->ne[0] != d || dv->nb[0] != ggml_type_size(kg->type) ||
            dv->ne[2] != 1 || dv->ne[3] != 1 || set->data != dv->data || set->nb[1] != dv->nb[1]) { return false; }

    // the write (pooled head) must not overlap the columns the kernel reads (key + gate) in any row:
    // same cache tensor, same row stride, disjoint column ranges within a row. The stream stride is the
    // cache's (kv_size rows), not n_kv rows: get_k views the first n_kv cells of every stream.
    const ggml_tensor * root = kp_root(kg);
    if (kp_root(dv) != root || kg->view_src == nullptr || dv->nb[1] != kg->nb[1] || kg->nb[2] % kg->nb[1] != 0) { return false; }
    {
        const size_t ts   = ggml_type_size(kg->type);
        const size_t row  = kg->nb[1];
        const size_t roff = kg->view_offs % row;
        const size_t woff = dv->view_offs % row;
        if (kg->view_offs / row != 0 || dv->view_offs / row != 0) { return false; }
        const bool disjoint = woff >= roff + 2*d*ts || woff + d*ts <= roff;
        if (!disjoint || roff + 2*d*ts > row || woff + d*ts > row) { return false; }
    }

    // nothing outside the chain may read an elided intermediate (or a view of one)
    const ggml_tensor * elided[] = { G, ck, cg, ca, add, sm, mul, sum };
    auto is_elided_root = [&](const ggml_tensor * t) {
        const ggml_tensor * rt = kp_root(t);
        for (const ggml_tensor * e : elided) { if (rt == e) { return true; } }
        return false;
    };
    for (int q = i; q < jset; ++q) {
        const ggml_tensor * t = cgraph->nodes[q];
        if (!kp_is_view_op(t) && t != G && t != ck && t != cg && t != ca && t != add && t != sm && t != mul && t != sum) {
            return false;   // a foreign compute node inside the range would be skipped
        }
        if (!is_elided_root(t)) {
            continue;
        }
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
        int uses = 0;
        for (int u = q + 1; u <= jset; ++u) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (cgraph->nodes[u]->src[s] == t) { uses++; }
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, q)) { return false; }
    }

    m.kg = kg; m.cells = cells; m.ape = ape; m.reps = reps; m.dst = set;
    m.d = d; m.r = r; m.n_new = n_new; m.n_stream = ns; m.last = jset;
    return true;
}
