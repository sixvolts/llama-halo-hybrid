#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

#include <algorithm>

// RDNA3.5 (gfx1151): 32 waves per block, measured 3.3x faster on the KDA recurrence at H=64, S=128 (16 waves: 2.6x)
//     (GLM-5.3-Flash prefill; from halo-box/strix-llama.cpp). Other devices keep the upstream 4.
#ifndef GDN_RDNA35_WARPS
#define GDN_RDNA35_WARPS 32
#endif
static constexpr int gdn_num_warps(int cc) {
    return GGML_CUDA_CC_IS_RDNA3_5(cc) ? GDN_RDNA35_WARPS : 4;
}
#if defined(RDNA3_5)
static constexpr int gdn_num_warps_dev = gdn_num_warps(GGML_CUDA_CC_RDNA3_5);
#else
static constexpr int gdn_num_warps_dev = gdn_num_warps(GGML_CUDA_CC_RDNA3);
#endif

// gather_t: the input state is row state_idx[sequence] of the cache (row stride state_row_stride floats) instead of
// a gathered copy -- the get_rows that built that copy is left out (ggml_cuda_try_defer_gdn_state_gather)
template <int S_v, bool KDA, bool keep_rs_t, bool gather_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * gdn_num_warps_dev, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const int32_t * state_idx,
                                     int64_t       state_row_stride,
                                     int64_t       attn_seq_stride) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    attn_data += sequence * attn_seq_stride + h_idx * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
    // the index is written by the host before the graph runs; read it after the PDL sync all the same
    const int64_t state_in_offset = (gather_t ? (int64_t) state_idx[sequence] * state_row_stride : sequence * H * S_v * S_v) +
        h_idx * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}


template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K,
        const int32_t * state_idx, int64_t state_row_stride, int64_t attn_seq_stride, cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    // one column per wave: never more waves than columns (S_v = 16 with 32 waves indexed past the state)
    const int num_warps = std::min<int>(gdn_num_warps(cc), (int) S_v);
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
#define GDN_LAUNCH(S)                                                                                        \
    if (state_idx != nullptr) {                                                                              \
        ggml_cuda_kernel_launch(gated_delta_net_cuda<S, KDA, keep_rs_t, true>, launch_params,                \
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,                                                 \
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                                  \
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride); \
    } else {                                                                                                 \
        ggml_cuda_kernel_launch(gated_delta_net_cuda<S, KDA, keep_rs_t, false>, launch_params,               \
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,                                                 \
            n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                                                  \
            sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride); \
    }
    switch (S_v) {
        case 16:  GDN_LAUNCH(16);  break;
        case 32:  GDN_LAUNCH(32);  break;
        case 64:  GDN_LAUNCH(64);  break;
        case 128: GDN_LAUNCH(128); break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
#undef GDN_LAUNCH
}

// ---------------------------------------------------------------------------------------------------------------------
// Chunked prefill for the scalar gate (Qwen3-Next / Qwen3.8 GDN; the KDA per-channel gate keeps the token loop).
//
// The token loop above carries a dependent S_v x S_v update per token. Over a chunk of C tokens with gcum[t] the
// in-chunk cumulative log-gate and S0 the state at the chunk start, the per-token deltas
//     d_t = beta_t (v_t - e^{g_t} S_{t-1}^T k_t)
// solve the unit lower-triangular system (I + L) d = Y (the WY / UT form of the delta rule):
//     L[t][s] = beta_t (k_t . k_s) e^{gcum_t - gcum_s}   (s < t)        T = (I + L)^-1
//     Y_t     = beta_t (v_t - e^{gcum_t} S0^T k_t)                       d = T Y
//     o_t     = scale (e^{gcum_t} S0^T q_t + sum_{s<=t} (q_t . k_s) e^{gcum_t - gcum_s} d_s)
//     S_C     = e^{gcum_C-1} S0 + sum_s e^{gcum_C-1 - gcum_s} k_s d_s^T
// Every exponent is <= 0 (g <= 0 and s <= t), so nothing overflows however strongly a head decays.
//
// gdn_chunk_prep: parallel over (k-head, chunk, seq): K K^T and Q K^T once per k-head, then per v-head T (forward
//     substitution, one thread per column), P (the decayed, causal Q K^T) and the e^{gcum} factors into a workspace.
//     gcum is summed in f64: e^{gcum_t - gcum_s} from f32 sums loses ~|gcum| ulp, which put the chunked output 100x
//     further from the CPU reference than the token loop on strongly decaying heads (now the same, ~1e-14 NMSE).
// gdn_chunk_scan: one block per (value-column block of NC, v-head, seq); the state slice S[:, cols] lives in LDS and
//     is carried across the chunks in order (columns of the state are independent, so the split is exact); the next
//     chunk's q / k / v / T / P are loaded into registers while the current one computes.
// fp32 throughout (no WMMA): the chunk form only reorders the sums.
// ---------------------------------------------------------------------------------------------------------------------

static constexpr int GDN_CH_C       = 32;  // chunk length
static constexpr int GDN_CH_THREADS = 256;
// per (seq, chunk, head): T, P, e^{gcum_t}, e^{gcum_last - gcum_t}
static constexpr int GDN_CH_WS      = 2 * GDN_CH_C * GDN_CH_C + 2 * GDN_CH_C;
static constexpr int GDN_CH_HGRP    = 8;   // v-heads per k-head whose gates the prep loads at once

static __device__ __forceinline__ float gdn_dot4(const float4 a, const float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// a C x S_v tile of q or k rows (row t at base + (t0 + t) * stride, zero past n_tokens) through registers: every load
// of the tile is issued before the first LDS store, so a chunk pays one memory latency (a plain element loop waits for
// each load before its store)
template <int S_v>
struct gdn_ch_rows {
    static constexpr int R4 = S_v / 4;
    static constexpr int N4 = GDN_CH_C * R4;
    static constexpr int NL = (N4 + GDN_CH_THREADS - 1) / GDN_CH_THREADS;
    float4 r[NL];

    __device__ __forceinline__ void load(const float * base, int64_t stride, int t0, int n_tokens, int tid) {
#pragma unroll
        for (int j = 0; j < NL; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            const int t   = idx / R4;
            const int d   = idx % R4;
            r[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (idx < N4 && t0 + t < n_tokens) {
                r[j] = *(const float4 *) (base + (int64_t) (t0 + t) * stride + 4 * d);
            }
        }
    }
    __device__ __forceinline__ void store(float * sh, int tid) const { // sh: [C][S_v + 4]
#pragma unroll
        for (int j = 0; j < NL; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            if (idx < N4) {
                *(float4 *) &sh[(idx / R4) * (S_v + 4) + 4 * (idx % R4)] = r[j];
            }
        }
    }
};

// the prep's LDS: K and Q rows while K K^T and Q K^T are formed, then L of up to GDN_CH_HGRP v-heads
template <int S_v>
static constexpr int gdn_chunk_prep_union() {
    return 2 * GDN_CH_C * (S_v + 4) > GDN_CH_HGRP * GDN_CH_C * (GDN_CH_C + 4) ?
        2 * GDN_CH_C * (S_v + 4) : GDN_CH_HGRP * GDN_CH_C * (GDN_CH_C + 4);
}

template <int S_v>
__global__ void __launch_bounds__(GDN_CH_THREADS)
gdn_chunk_prep(const float * q, const float * k, const float * g, const float * beta, float * ws,
               int H, int n_tokens, int n_chunks, int neqk1, int rq3,
               int64_t sq1, int64_t sq2, int64_t sq3, int64_t sb1, int64_t sb2, int64_t sb3) {
    constexpr int C  = GDN_CH_C;
    constexpr int KS = S_v + 4;
    constexpr int LS = C + 4;     // L row stride (16-byte rows)
    static_assert(GDN_CH_HGRP * C <= GDN_CH_THREADS, "one solve thread per (head, column)");
    // heads fastest: the blocks in flight read the same token rows (a chunk-major grid spread them over thousands of
    // pages and ran 6x slower at 2048 tokens than at 512 per token)
    const int hk    = blockIdx.x;
    const int chunk = blockIdx.y;
    const int seq   = blockIdx.z;
    const int tid   = threadIdx.x;
    const int t0    = chunk * C;

    __shared__ __align__(16) float U[gdn_chunk_prep_union<S_v>()];
    __shared__ float KK[C][C + 1];
    __shared__ float QK[C][C + 1];
    __shared__ double gsh[GDN_CH_HGRP][C]; // gcum in f64: e^{gcum_t - gcum_s} from f32 sums lost ~|gcum| ulp
    __shared__ float bsh[GDN_CH_HGRP][C];
    float * Ksh = U;
    float * Qsh = U + C * KS;
    float * Lsh = U;              // [GDN_CH_HGRP][C][LS], after K K^T / Q K^T

    ggml_cuda_pdl_sync();

    {
        const int64_t qk_off = (int64_t) (seq / rq3) * sq3 + (int64_t) hk * sq1;
        gdn_ch_rows<S_v> kr;
        gdn_ch_rows<S_v> qr;
        kr.load(k + qk_off, sq2, t0, n_tokens, tid);
        qr.load(q + qk_off, sq2, t0, n_tokens, tid);
        kr.store(Ksh, tid);
        qr.store(Qsh, tid);
    }
    __syncthreads();

    // K K^T and Q K^T: thread (t, sg) covers columns s = sg + 8 j
    {
        const int t  = tid >> 3;
        const int sg = tid & 7;
        float kk[C / 8] = { 0.0f };
        float qk[C / 8] = { 0.0f };
#pragma unroll 4
        for (int d = 0; d < S_v; d += 4) {
            const float4 kt = *(const float4 *) &Ksh[t * KS + d];
            const float4 qt = *(const float4 *) &Qsh[t * KS + d];
#pragma unroll
            for (int j = 0; j < C / 8; j++) {
                const float4 ks = *(const float4 *) &Ksh[(sg + 8 * j) * KS + d];
                kk[j] += gdn_dot4(kt, ks);
                qk[j] += gdn_dot4(qt, ks);
            }
        }
#pragma unroll
        for (int j = 0; j < C / 8; j++) {
            KK[t][sg + 8 * j] = kk[j];
            QK[t][sg + 8 * j] = qk[j];
        }
    }

    // the v-heads that read this k-head (h % neqk1 == hk, as in the token loop), GDN_CH_HGRP at a time
    const int n_rep = (H - hk + neqk1 - 1) / neqk1;
    for (int r0 = 0; r0 < n_rep; r0 += GDN_CH_HGRP) {
        const int nr = min(GDN_CH_HGRP, n_rep - r0);
        __syncthreads(); // K / Q (or the previous group's L) consumed, KK / QK written
        if (tid < nr * C) {
            const int     r   = tid / C;
            const int     t   = tid % C;
            const bool    ok  = t0 + t < n_tokens;
            const int64_t off = (int64_t) seq * sb3 + (int64_t) (t0 + t) * sb2 + (int64_t) (hk + (r0 + r) * neqk1) * sb1;
            gsh[r][t] = ok ? (double) g[off] : 0.0;
            bsh[r][t] = ok ? beta[off] : 0.0f;
        }
        __syncthreads();
        if (tid < nr) {
            double acc = 0.0;
            for (int t = 0; t < C; t++) {
                acc += gsh[tid][t];
                gsh[tid][t] = acc;
            }
        }
        __syncthreads();

        // L (to LDS), P and gcum (to the workspace) of every head of the group
        for (int idx = tid; idx < nr * C * C; idx += GDN_CH_THREADS) {
            const int   r   = idx / (C * C);
            const int   t   = (idx / C) % C;
            const int   s   = idx % C;
            const float dec = s <= t ? expf((float) (gsh[r][t] - gsh[r][s])) : 0.0f;
            Lsh[(r * C + t) * LS + s] = s < t ? bsh[r][t] * KK[t][s] * dec : 0.0f;
            float * wsb = ws + (((int64_t) seq * n_chunks + chunk) * H + hk + (r0 + r) * neqk1) * GDN_CH_WS;
            wsb[C * C + t * C + s] = QK[t][s] * dec; // P
            if (s == 0) {
                wsb[2 * C * C + t]     = expf((float) gsh[r][t]);
                wsb[2 * C * C + C + t] = expf((float) (gsh[r][C - 1] - gsh[r][t]));
            }
        }
        __syncthreads();

        // T = (I + L)^-1 by forward substitution, one thread per (head r, column c) with the column in registers:
        // T[t][c] = [t == c] - sum_{s < t} L[t][s] T[s][c]
        if (tid < nr * C) {
            const int r = tid / C;
            const int c = tid % C;
            const float * Lr = Lsh + r * C * LS;
            float Tc[C];
#pragma unroll
            for (int t = 0; t < C; t++) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
#pragma unroll
                for (int s = 0; s < (t & ~3); s += 4) {
                    const float4 l = *(const float4 *) &Lr[t * LS + s];
                    acc0 += l.x * Tc[s + 0];
                    acc1 += l.y * Tc[s + 1];
                    acc0 += l.z * Tc[s + 2];
                    acc1 += l.w * Tc[s + 3];
                }
#pragma unroll
                for (int s = t & ~3; s < t; s++) {
                    acc0 += Lr[t * LS + s] * Tc[s];
                }
                Tc[t] = (t == c ? 1.0f : 0.0f) - (acc0 + acc1);
            }
            float * wsb = ws + (((int64_t) seq * n_chunks + chunk) * H + hk + (r0 + r) * neqk1) * GDN_CH_WS;
#pragma unroll
            for (int t = 0; t < C; t++) {
                wsb[t * C + c] = Tc[t];
            }
        }
    }
}

// the Q rows of gdn_chunk_scan, reused for T and P once Q S0 is done
template <int S_v>
static constexpr int gdn_chunk_scan_qrows() {
    return GDN_CH_C * (S_v + 4) > 2 * GDN_CH_C * (GDN_CH_C + 4) ? GDN_CH_C * (S_v + 4) : 2 * GDN_CH_C * (GDN_CH_C + 4);
}

// dynamic LDS of gdn_chunk_scan in floats
template <int S_v, int NC>
static constexpr int gdn_chunk_scan_lds() {
    return GDN_CH_C * (S_v + 4) + gdn_chunk_scan_qrows<S_v>() + NC * (S_v + 4) + 2 * GDN_CH_C * (NC + 4) + 3 * GDN_CH_C;
}

template <int S_v, int NC, bool gather_t>
__global__ void __launch_bounds__(GDN_CH_THREADS)
gdn_chunk_scan(const float * q, const float * k, const float * v, const float * beta, const float * ws,
               const float * curr_state, float * dst, float * state,
               int H, int n_tokens, int n_chunks, int neqk1, int rq3,
               int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
               int64_t sb1, int64_t sb2, int64_t sb3, float scale, int64_t attn_seq_stride,
               const int32_t * state_idx, int64_t state_row_stride) {
    constexpr int C  = GDN_CH_C;
    constexpr int KS = S_v + 4;
    constexpr int YS = NC + 4;
    constexpr int TS = C + 4;     // row strides = 4 mod 32 floats: rows read by consecutive lanes hit distinct banks
    constexpr int CG = NC / 4;    // 4-column groups
    static_assert(C * CG <= GDN_CH_THREADS, "one (token, column group) item per thread");
    constexpr int NTP = (2 * C * C / 4 + GDN_CH_THREADS - 1) / GDN_CH_THREADS; // float4 of T and P per thread
    constexpr int NS4 = (NC * S_v / 4 + GDN_CH_THREADS - 1) / GDN_CH_THREADS;  // float4 of the state slice

    const int col0 = blockIdx.x * NC;
    const int h    = blockIdx.y;
    const int seq  = blockIdx.z;
    const int tid  = threadIdx.x;

    extern __shared__ float4 gdn_ch_smem4[];
    float * Ksh = (float *) gdn_ch_smem4;              // [C][KS]
    float * Qsh = Ksh + C * KS;                        // [C][KS], then T [C][TS] and P [C][TS]
    float * Ssh = Qsh + gdn_chunk_scan_qrows<S_v>();   // [NC][KS]: Ssh[c][i] = S[i][col0 + c] (global layout)
    float * Ysh = Ssh + NC * KS;                       // [C][YS]
    float * Nsh = Ysh + C * YS;                        // [C][YS]: the deltas d
    float * esh = Nsh + C * YS;                        // e^{gcum_t}
    float * wsh = esh + C;                             // e^{gcum_last - gcum_t}
    float * bsh = wsh + C;                             // beta
    float * Tsh = Qsh;
    float * Psh = Qsh + C * TS;

    ggml_cuda_pdl_sync();

    {
        const int64_t state_in_offset = (gather_t ? (int64_t) state_idx[seq] * state_row_stride : (int64_t) seq * H * S_v * S_v) +
            (int64_t) h * S_v * S_v + (int64_t) col0 * S_v;
        float4 sr[NS4];
#pragma unroll
        for (int j = 0; j < NS4; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            if (idx < NC * S_v / 4) {
                sr[j] = *(const float4 *) (curr_state + state_in_offset + 4 * idx);
            }
        }
#pragma unroll
        for (int j = 0; j < NS4; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            if (idx < NC * S_v / 4) {
                *(float4 *) &Ssh[(idx / (S_v / 4)) * KS + 4 * (idx % (S_v / 4))] = sr[j];
            }
        }
    }

    const int64_t qk_off = (int64_t) (seq / rq3) * sq3 + (int64_t) (h % neqk1) * sq1;
    const int64_t v_off  = (int64_t) seq * sv3 + (int64_t) h * sv1 + col0;
    const int64_t b_off  = (int64_t) seq * sb3 + (int64_t) h * sb1;
    const float * wsh0   = ws + ((int64_t) seq * n_chunks * H + h) * GDN_CH_WS; // workspace: [seq][chunk][head]
    float * attn = dst + (int64_t) seq * attn_seq_stride + (int64_t) h * S_v + col0;

    // (token, column group) item of the C x NC phases: lanes run over tokens, so the rows of K / Q / T / P a wave reads
    // are KS / TS apart (distinct banks) and the S / Y / N reads are broadcasts
    const int  it  = tid % C;
    const int  c0  = (tid / C) * 4;
    const bool own = tid < C * CG;

    // next chunk's inputs, loaded while the current one computes
    gdn_ch_rows<S_v> kr;
    gdn_ch_rows<S_v> qr;
    float4 tp[NTP];
    float  epre = 0.0f;
    float  wpre = 0.0f;
    float  bpre = 0.0f;
    float4 vpre = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    auto prefetch = [&](int chunk) {
        const int t0 = chunk * C;
        const float * wsb = wsh0 + (int64_t) chunk * H * GDN_CH_WS;
        kr.load(k + qk_off, sq2, t0, n_tokens, tid);
        qr.load(q + qk_off, sq2, t0, n_tokens, tid);
#pragma unroll
        for (int j = 0; j < NTP; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            if (idx < 2 * C * C / 4) {
                tp[j] = *(const float4 *) (wsb + 4 * idx);
            }
        }
        if (tid < C) {
            epre = wsb[2 * C * C + tid];
            wpre = wsb[2 * C * C + C + tid];
            bpre = t0 + tid < n_tokens ? beta[b_off + (int64_t) (t0 + tid) * sb2] : 0.0f;
        }
        vpre = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (own && t0 + it < n_tokens) {
            vpre = *(const float4 *) (v + v_off + (int64_t) (t0 + it) * sv2 + c0);
        }
    };
    prefetch(0);

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        const int t0 = chunk * C;

        __syncthreads(); // previous chunk's K / T / P / N reads done
        kr.store(Ksh, tid);
        qr.store(Qsh, tid);
        if (tid < C) {
            esh[tid] = epre;
            wsh[tid] = wpre;
            bsh[tid] = bpre;
        }
        float4 tpc[NTP];
#pragma unroll
        for (int j = 0; j < NTP; j++) {
            tpc[j] = tp[j];
        }
        const float4 vc = vpre;
        __syncthreads();
        if (chunk + 1 < n_chunks) {
            prefetch(chunk + 1);
        }

        // phase A: X = K S0 and Qs = Q S0 for row it, columns c0..c0+3; Y = beta (v - e^g X)
        float oacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        if (own) {
            float x[4]  = { 0.0f, 0.0f, 0.0f, 0.0f };
            float qs[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
            for (int i = 0; i < S_v; i += 4) {
                const float4 kt = *(const float4 *) &Ksh[it * KS + i];
                const float4 qt = *(const float4 *) &Qsh[it * KS + i];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const float4 sc = *(const float4 *) &Ssh[(c0 + j) * KS + i];
                    x[j]  += gdn_dot4(kt, sc);
                    qs[j] += gdn_dot4(qt, sc);
                }
            }
            const float eg = esh[it];
            const float bt = bsh[it];
            *(float4 *) &Ysh[it * YS + c0] = make_float4(bt * (vc.x - eg * x[0]), bt * (vc.y - eg * x[1]),
                                                         bt * (vc.z - eg * x[2]), bt * (vc.w - eg * x[3]));
#pragma unroll
            for (int j = 0; j < 4; j++) {
                oacc[j] = eg * qs[j];
            }
        }
        __syncthreads(); // Q reads done: T and P go into its rows
#pragma unroll
        for (int j = 0; j < NTP; j++) {
            const int idx = tid + j * GDN_CH_THREADS;
            if (idx < 2 * C * C / 4) {
                const int e = 4 * idx;            // T then P, row-major [C][C]
                float * dsth = e < C * C ? Tsh : Psh;
                const int ee = e % (C * C);
                float * row = dsth + (ee / C) * TS + ee % C;
                row[0] = tpc[j].x;
                row[1] = tpc[j].y;
                row[2] = tpc[j].z;
                row[3] = tpc[j].w;
            }
        }
        __syncthreads();

        // phase B: d = T Y (T is zero above the diagonal)
        if (own) {
            float nv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 2
            for (int s = 0; s < C; s += 4) {
                const float4 tv = *(const float4 *) &Tsh[it * TS + s];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float  tu = u == 0 ? tv.x : u == 1 ? tv.y : u == 2 ? tv.z : tv.w;
                    const float4 y  = *(const float4 *) &Ysh[(s + u) * YS + c0];
                    nv[0] += tu * y.x;
                    nv[1] += tu * y.y;
                    nv[2] += tu * y.z;
                    nv[3] += tu * y.w;
                }
            }
            *(float4 *) &Nsh[it * YS + c0] = make_float4(nv[0], nv[1], nv[2], nv[3]);
        }
        __syncthreads();

        // phase B2: o = e^g Qs + P d (P is zero above the diagonal)
        if (own) {
#pragma unroll 2
            for (int s = 0; s < C; s += 4) {
                const float4 pv = *(const float4 *) &Psh[it * TS + s];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const float  pu = u == 0 ? pv.x : u == 1 ? pv.y : u == 2 ? pv.z : pv.w;
                    const float4 d  = *(const float4 *) &Nsh[(s + u) * YS + c0];
                    oacc[0] += pu * d.x;
                    oacc[1] += pu * d.y;
                    oacc[2] += pu * d.z;
                    oacc[3] += pu * d.w;
                }
            }
            if (t0 + it < n_tokens) {
                *(float4 *) (attn + (int64_t) (t0 + it) * H * S_v + c0) =
                    make_float4(oacc[0] * scale, oacc[1] * scale, oacc[2] * scale, oacc[3] * scale);
            }
        }

        // phase C: S = e^{gcum_last} S + sum_s (e^{gcum_last - gcum_s} k_s) d_s^T; item (4 rows i0.., 4 columns c)
        const float el = esh[C - 1];
        for (int item = tid; item < (S_v / 4) * CG; item += GDN_CH_THREADS) {
            const int i0 = (item % (S_v / 4)) * 4; // lanes over rows: contiguous K / S reads, broadcast N reads
            const int cs = (item / (S_v / 4)) * 4;
            float4 acc[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 sv = *(const float4 *) &Ssh[(cs + j) * KS + i0];
                acc[j] = make_float4(el * sv.x, el * sv.y, el * sv.z, el * sv.w);
            }
#pragma unroll 4
            for (int s = 0; s < C; s++) {
                const float  w  = wsh[s];
                float4       ks = *(const float4 *) &Ksh[s * KS + i0];
                ks.x *= w; ks.y *= w; ks.z *= w; ks.w *= w;
                const float4 d  = *(const float4 *) &Nsh[s * YS + cs];
                acc[0].x += ks.x * d.x; acc[0].y += ks.y * d.x; acc[0].z += ks.z * d.x; acc[0].w += ks.w * d.x;
                acc[1].x += ks.x * d.y; acc[1].y += ks.y * d.y; acc[1].z += ks.z * d.y; acc[1].w += ks.w * d.y;
                acc[2].x += ks.x * d.z; acc[2].y += ks.y * d.z; acc[2].z += ks.z * d.z; acc[2].w += ks.w * d.z;
                acc[3].x += ks.x * d.w; acc[3].y += ks.y * d.w; acc[3].z += ks.z * d.w; acc[3].w += ks.w * d.w;
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                *(float4 *) &Ssh[(cs + j) * KS + i0] = acc[j];
            }
        }
    }
    __syncthreads();

    float * sout = state + (int64_t) (seq * H + h) * S_v * S_v + (int64_t) col0 * S_v;
    for (int idx = tid; idx < NC * S_v / 4; idx += GDN_CH_THREADS) {
        *(float4 *) (sout + 4 * idx) = *(const float4 *) &Ssh[(idx / (S_v / 4)) * KS + 4 * (idx % (S_v / 4))];
    }
}

// GGML_CUDA_NO_GDN_CHUNKED=1: prefill keeps the token loop. GGML_CUDA_GDN_CHUNK_MIN: fewest tokens for the chunk path
static bool gdn_chunked_disabled() {
    static const bool d = getenv("GGML_CUDA_NO_GDN_CHUNKED") != nullptr;
    return d;
}
static int64_t gdn_chunked_min_tokens() {
    static const int64_t n = getenv("GGML_CUDA_GDN_CHUNK_MIN") ? atoll(getenv("GGML_CUDA_GDN_CHUNK_MIN")) : 64;
    return n;
}

template <int S_v>
static void launch_gdn_chunked(ggml_backend_cuda_context & ctx,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1, int64_t rq3, float scale, int64_t attn_seq_stride,
        const int32_t * state_idx, int64_t state_row_stride) {
    constexpr int NC = S_v < 32 ? S_v : 32;
    cudaStream_t stream = ctx.stream();
    const int n_chunks = (int) ((n_tokens + GDN_CH_C - 1) / GDN_CH_C);

    ggml_cuda_pool_alloc<float> ws(ctx.pool(), (size_t) n_seqs * H * n_chunks * GDN_CH_WS);

    gdn_chunk_prep<S_v><<<dim3(neqk1, n_chunks, n_seqs), GDN_CH_THREADS, 0, stream>>>(
        q_d, k_d, g_d, b_d, ws.get(), (int) H, (int) n_tokens, n_chunks, (int) neqk1, (int) rq3,
        sq1, sq2, sq3, sb1, sb2, sb3);
    CUDA_CHECK(cudaGetLastError());

    const size_t lds = gdn_chunk_scan_lds<S_v, NC>() * sizeof(float);
    const dim3 grid(S_v / NC, H, n_seqs);
    if (state_idx != nullptr) {
        CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_scan<S_v, NC, true>), lds);
        gdn_chunk_scan<S_v, NC, true><<<grid, GDN_CH_THREADS, lds, stream>>>(
            q_d, k_d, v_d, b_d, ws.get(), s_d, dst_d, state_d, (int) H, (int) n_tokens, n_chunks, (int) neqk1, (int) rq3,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale, attn_seq_stride, state_idx, state_row_stride);
    } else {
        CUDA_SET_SHARED_MEMORY_LIMIT((gdn_chunk_scan<S_v, NC, false>), lds);
        gdn_chunk_scan<S_v, NC, false><<<grid, GDN_CH_THREADS, lds, stream>>>(
            q_d, k_d, v_d, b_d, ws.get(), s_d, dst_d, state_d, (int) H, (int) n_tokens, n_chunks, (int) neqk1, (int) rq3,
            sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, scale, attn_seq_stride, state_idx, state_row_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    // a deferred state gather: read the cache row through the index instead of the (never written) gathered copy
    const int32_t * state_idx        = nullptr;
    int64_t         state_row_stride = 0;
    for (auto & e : ctx.gdn_state_gather) {
        if (e.gdn == dst) {
            s_d              = (const float *) e.rows->src[0]->data;
            state_idx        = (const int32_t *) e.rows->src[1]->data;
            state_row_stride = e.rows->src[0]->nb[1] / sizeof(float);
            e = {};
            break;
        }
    }

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int64_t attn_seq_stride = S_v * H * n_tokens;

    // prefill, scalar gate: chunked scan over all but the last K tokens (K > 1: their per-token snapshots come from
    // the token loop, which then continues from the chunked state)
    const int64_t n_tail = keep_rs ? std::min<int64_t>(K, n_tokens) : 0;
    // float4 loads: rows of q / k / v and the state 16-byte aligned
    auto al16 = [](const void * p) { return ((uintptr_t) p) % 16 == 0; };
    const bool aligned = al16(q_d) && al16(k_d) && al16(v_d) && al16(s_d) && al16(dst_d) && al16(state_d) &&
        sq1 % 4 == 0 && sq2 % 4 == 0 && sq3 % 4 == 0 && sv1 % 4 == 0 && sv2 % 4 == 0 && sv3 % 4 == 0 &&
        state_row_stride % 4 == 0 && state_slot_stride % 4 == 0;
    // the scan keeps a state slice in dynamic LDS (~60 KB at S_v = 128): parts whose opt-in limit is smaller (NVIDIA
    // Pascal, 48 KB) keep the token loop
    auto chunk_lds = [](int64_t sv) -> size_t {
        switch (sv) {
            case 16:  return gdn_chunk_scan_lds<16,  16>() * sizeof(float);
            case 32:  return gdn_chunk_scan_lds<32,  32>() * sizeof(float);
            case 64:  return gdn_chunk_scan_lds<64,  32>() * sizeof(float);
            case 128: return gdn_chunk_scan_lds<128, 32>() * sizeof(float);
            default:  return SIZE_MAX;
        }
    };
    const bool lds_ok = chunk_lds(S_v) <= ggml_cuda_info().devices[ctx.device].smpbo;
    if (!kda && aligned && lds_ok && !gdn_chunked_disabled() && n_tokens >= gdn_chunked_min_tokens() && n_tokens - n_tail >= GDN_CH_C &&
            (S_v == 16 || S_v == 32 || S_v == 64 || S_v == 128)) {
        const int64_t n_main = n_tokens - n_tail;
        ggml_cuda_pool_alloc<float> state_mid(ctx.pool());
        float * state_main = state_d;
        if (n_tail > 0) {
            state_main = state_mid.alloc((size_t) S_v * S_v * H * n_seqs);
        }
        switch (S_v) {
#define GDN_CH_LAUNCH(S) \
            case S: launch_gdn_chunked<S>(ctx, q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_main, H, n_main, n_seqs, \
                sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, attn_seq_stride, state_idx, \
                state_row_stride); break;
            GDN_CH_LAUNCH(16)
            GDN_CH_LAUNCH(32)
            GDN_CH_LAUNCH(64)
            GDN_CH_LAUNCH(128)
#undef GDN_CH_LAUNCH
            default: GGML_ABORT("fatal error");
        }
        if (n_tail == 0) {
            return;
        }
        // the last n_tail tokens through the token loop from the chunked state (plain layout, no gather)
        const int64_t t0 = n_main;
        launch_gated_delta_net<false, true>(q_d + t0 * sq2, k_d + t0 * sq2, v_d + t0 * sv2, g_d + t0 * sb2, b_d + t0 * sb2,
            state_main, dst_d + t0 * S_v * H, state_d,
            S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, nullptr, 0, attn_seq_stride, stream);
        return;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, state_idx, state_row_stride, attn_seq_stride, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
