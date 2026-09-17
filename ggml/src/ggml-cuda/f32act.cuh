#pragma once
#include "common.cuh"

// halo-hybrid: q8_0 weight x f32 activation GEMV for the short-K projections of decode, with an optional
// scale+silu prologue on the activation. It deletes launches: the consumers of the hyper-connection
// down-projection (K=320) and of the GLU outputs (K=640) otherwise need a q8_1 copy of their activation, which is
// a quantize launch each (or the k_scale_silu launch that writes one).
//
// OFF by default (GGML_CUDA_F32ACT_K=<max K> enables it), because deleting those launches does not pay:
// Qwen3.8 decode on gfx1151, 128 tokens, 3 reps, against the same build with it disabled --
//   8 rows/block, LDS stage per block:            37.40 vs 37.20 ms  (-145 launches, LOST 0.20 ms)
//   4 rows/wave, 32 rows/block, + the ids form:   37.18 vs 37.22 ms  (-193 launches, a wash)
//   + first tile prefetched before the stage, pipelined K loop: 37.66 vs 37.20 ms (LOST 0.46 ms)
// A kernel trace explains it: this kernel matches MMVQ per call (19.7 vs 19.9 us on the hc_up class) and removes
// 0.29 ms/token of quantize + k_scale_silu kernel time, yet wall-clock does not move. A merged GEMV is worth
// ~4 us per removed launch (see docs/halo-hybrid/APU-DECODE-BUDGET.md), but a ~1.7 us element-wise kernel next to
// a GEMV costs nearly nothing to keep -- its work overlaps. The rule this leaves: a replacement kernel has to beat
// MMVQ outright; matching it and deleting a small neighbour buys zero, and the register pressure of the pipelined
// version then costs occupancy.
struct ggml_cuda_f32act_prologue {
    float scale;
    float bias;
    bool  silu;     // v = silu(scale*x + bias); false: v = x
};

// returns false (nothing launched) when the shapes do not qualify
bool ggml_cuda_mul_mat_vec_q8_f32act(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
                                     ggml_tensor * dst, const ggml_cuda_f32act_prologue * pro);

// MUL_MAT_ID form (experts selected by ids, one activation row per slot); same gate, no prologue
bool ggml_cuda_mul_mat_id_vec_q8_f32act(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
                                        const ggml_tensor * ids, ggml_tensor * dst);
