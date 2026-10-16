#pragma once

// halo-hybrid: helpers shared by this tree's graph matchers in ggml-cuda.cu. Matcher naming:
//   ggml_cuda_<pattern>_find (structure only, safe before allocation), ggml_cuda_<pattern>_disjoint (alias checks on
//   the allocated tensors), ggml_cuda_try_fuse_<pattern> (find, check, launch; returns the nodes consumed),
//   ggml_cuda_op_<pattern> (the launch), kernels k_<pattern>, kill switch GGML_CUDA_NO_<PATTERN>.

#include "common.cuh"

#include <cstring>
#include <initializer_list>

// a node that only reinterprets its source (no kernel)
static inline bool ggml_cuda_fuse_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_NONE;
}

// the first node at or after j that is not a view
static inline int ggml_cuda_fuse_next(const ggml_cgraph * g, int j) {
    while (j < g->n_nodes && ggml_cuda_fuse_is_view(g->nodes[j])) {
        j++;
    }
    return j;
}

static inline bool ggml_cuda_fuse_unary(const ggml_tensor * t, ggml_unary_op u) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == u;
}

// node j has exactly one reader, so a fused kernel may skip writing it
static inline bool ggml_cuda_fuse_uses1(const ggml_cgraph * g, int j) {
    return ggml_node_get_use_count(g, j) == 1;
}

static inline const ggml_tensor * ggml_cuda_fuse_root(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

// The byte ranges of a and b do not overlap (nullptr overlaps nothing). A fused kernel reads its inputs at indices
// other than the one it writes; if ggml-alloc placed the output over memory an input occupied (legal in the unfused
// graph, where the input is dead by then), the fused kernel would race with itself, so the matcher declines.
static inline bool ggml_cuda_fuse_disjoint(const ggml_tensor * a, const ggml_tensor * b) {
    if (a == nullptr || b == nullptr) {
        return true;
    }
    const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
    return a1 <= b0 || b1 <= a0;
}

static inline float ggml_cuda_fuse_param_f32(const ggml_tensor * t, int k) {
    float v;
    memcpy(&v, (const float *) t->op_params + k, sizeof(float));
    return v;
}

// all view nodes strictly between lo and hi are views of one of the given roots
static inline bool ggml_cuda_fuse_views_belong(const ggml_cgraph * g, int lo, int hi, std::initializer_list<const ggml_tensor *> roots) {
    for (int q = lo + 1; q < hi; ++q) {
        const ggml_tensor * t = g->nodes[q];
        if (!ggml_cuda_fuse_is_view(t)) {
            continue;
        }
        bool ok = false;
        for (const ggml_tensor * r : roots) {
            ok = ok || ggml_cuda_fuse_root(t) == r;
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}
