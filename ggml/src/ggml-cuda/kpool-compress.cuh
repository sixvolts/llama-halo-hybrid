#pragma once

#include "common.cuh"

// halo-hybrid: the glm5next DSA indexer pool compressor (build_indexer, the new-pool branch) in one launch.
// Replaces GET_ROWS(members) -> CONT(PERMUTE key) , CONT(PERMUTE gate) , CONT(TRANSPOSE ape) -> ADD ->
// SOFT_MAX -> MUL -> SUM_ROWS -> SET_ROWS(pooled head). For every new pool (p, s) and column c:
//   g[j] = gate[cells[p*r + j, s], c] + ape[c, j]
//   out  = sum_j key[cells[p*r + j, s], c] * softmax_j(g)[j]
//   dst[reps[p + s*n_new], c] = out
// The constant ape weight is read in place (no per-step transpose), the members / probs / sums are
// never materialised. Matched in ggml_cuda_try_fuse; GGML_CUDA_NO_KPOOL_COMPRESS=1 disables.

struct ggml_cuda_kpool_compress_match {
    const ggml_tensor * kg   = nullptr;   // view of the indexer cache: [2*d, n_kv, n_stream], key then gate
    const ggml_tensor * cells = nullptr;  // I32 [r*n_new, n_stream]
    const ggml_tensor * ape  = nullptr;   // F32 [d, r]
    const ggml_tensor * reps = nullptr;   // I64 [n_new*n_stream]
    ggml_tensor       * dst  = nullptr;   // SET_ROWS node: view of the pooled head, [d, rows]
    int64_t d = 0, r = 0, n_new = 0, n_stream = 0;
    int last = 0;                         // index of the SET_ROWS node
};

bool ggml_cuda_kpool_compress_match_graph(const ggml_cgraph * cgraph, int i, ggml_cuda_kpool_compress_match & m);
void ggml_cuda_op_kpool_compress(ggml_backend_cuda_context & ctx, const ggml_cuda_kpool_compress_match & m);
