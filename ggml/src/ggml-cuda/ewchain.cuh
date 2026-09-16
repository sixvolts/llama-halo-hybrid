#pragma once

#include "common.cuh"

// halo-hybrid: generic element-wise chain fusion. A run of MUL/ADD/SUB/DIV (with a broadcast src1), SCALE, UNARY
// (sigmoid, silu, exp, neg, relu, tanh, abs), SQR and SQRT nodes where each node consumes the previous one and
// nothing else reads the intermediates becomes one launch. On the GLM-5.3-Flash graph this covers the two
// hyper-connection gate chains per sublayer (mul, add, sigmoid, scale: 8 launches per hc mixer), the KDA gate
// (add, mul, scale, sigmoid, scale) and the hc mean (add x3, scale). Matched in ggml_cuda_try_fuse_ewchain
// (ggml-cuda.cu); GGML_CUDA_NO_EWCHAIN=1 disables.

#define GGML_CUDA_EW_MAX_OPS 6

struct ggml_cuda_ew_op {
    int          op;        // ggml_op
    int          unary;     // ggml_unary_op when op == GGML_OP_UNARY
    float        s, b;      // SCALE
    const char * src1;      // binary ops: broadcast operand
    int64_t      ne1[4];
    size_t       nb1[4];
};

struct ggml_cuda_ew_chain {
    int              n;
    ggml_cuda_ew_op  ops[GGML_CUDA_EW_MAX_OPS];
    const char *     src0;  // first node's input (any strides)
    size_t           nb0[4];
    float *          dst;   // last node's output (contiguous)
    int64_t          ne[4];
};

// what the matcher found: the chain itself plus the tensors it touches (persist.cu needs them for hazard ranges)
struct ggml_cuda_ew_match {
    ggml_cuda_ew_chain  ch;
    int                 last;                          // last graph node index the chain consumes
    const ggml_tensor * in0;
    const ggml_tensor * out;
    const ggml_tensor * others[GGML_CUDA_EW_MAX_OPS];  // broadcast operands, in chain order
    int                 n_others;
};

// true when cgraph->nodes[i] starts a chain of at least two element-wise nodes (ggml-cuda.cu)
bool ggml_cuda_ewchain_match(const ggml_cgraph * cgraph, int i, ggml_cuda_ew_match & m);

void ggml_cuda_op_ew_chain(ggml_backend_cuda_context & ctx, const ggml_cuda_ew_chain & c);
