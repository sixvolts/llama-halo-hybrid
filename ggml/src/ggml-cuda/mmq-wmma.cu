#include "mmq-wmma.cuh"
#include "convert.cuh"
#include "hc.cuh"

// halo-hybrid: dense q8_0 GEMM for RDNA WMMA at prefill widths.
//
// MMQ keeps the weights as int8 in LDS and pays the block-scale arithmetic per output element every 32 K
// (about 24 VALU per WMMA pair per lane), which is what bounds it on gfx1151 at 17-22% of the matrix peak.
// This kernel dequantizes each q8_0 block once, when it is staged into LDS (one v_perm + one packed fma per
// pair of weights, shared by every activation row the block computes), and then runs a plain f16 x f16 -> f32
// WMMA loop over the tile. The activations are converted to f16 once per call (same as the vendor GEMM path).
//
//   dst[n][m] = sum_k W[m][k] * X[n][k]      W = src0 (q8_0, M rows of K), X = src1 (f32 -> f16, N rows of K)
//
// WMMA operands: A = activations (16 n x 16 k), B = weights (16 k x 16 m), C = 16 n x 16 m. With the
// activations as the A operand the accumulator's 16 lanes hold 16 consecutive m for one n, so the epilogue
// writes 64-byte runs of the (m-contiguous) dst row.
//
// Block tile: BN activation rows x BM weight rows x KT of K, warps WN x WM, warp tile TN x TM fragments.
// LDS: [BN][KT + 8] + [BM][KT + 8] halves per buffer (the +8 halves make the row pitch 4 mod 8 dwords, which
// is conflict-free for the 16-byte fragment reads); NBUF buffers.

// the host side is compiled unconditionally on HIP (the RDNA macros exist only in the device passes); the
// device code is real only where AMD_WMMA_AVAILABLE (gfx11 / gfx12), NO_DEVICE_CODE elsewhere
#if defined(GGML_USE_HIP)
#define MMQ_WMMA_ENABLED
#endif

#ifdef MMQ_WMMA_ENABLED

typedef __attribute__((ext_vector_type(8)))  float   floatx8_t;
typedef __attribute__((ext_vector_type(16))) _Float16 halfx16_t;
typedef __attribute__((ext_vector_type(8)))  _Float16 halfx8_t;

// halo-hybrid (A4): saturating f32 -> f16 for activations narrowed inside these kernels (|x| > 65504 would become inf and
// poison a whole output row); NaN passes through
static __device__ __forceinline__ float f16_sat(const float x) {
    return x != x ? x : fminf(fmaxf(x, -65504.0f), 65504.0f);
}

static __global__ void mmqw_f32_to_f16_sat(const float * __restrict__ x, half * __restrict__ y, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half(f16_sat(x[i]));
    }
}

#if defined(RDNA4)
typedef halfx8_t  mmqw_frag_t;   // gfx12: lane l holds 8 k of row l%16, k offset 8*(l/16)
#else
typedef halfx16_t mmqw_frag_t;   // gfx11: lane l holds all 16 k of row l%16 (both half-waves identical)
#endif

static __device__ __forceinline__ void mmqw_wmma(floatx8_t & acc, const mmqw_frag_t & a, const mmqw_frag_t & b) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA4)
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
#elif defined(AMD_WMMA_AVAILABLE)
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
#else
    GGML_UNUSED_VARS(acc, a, b);
    NO_DEVICE_CODE;
#endif
}

// fragment for row (lane % 16), k-slice starting at column k0 of an LDS tile with row pitch `pitch` halves
static __device__ __forceinline__ mmqw_frag_t mmqw_load_frag(const half * __restrict__ tile, const int pitch, const int row0, const int k0) {
    const int lane = threadIdx.x % 32;
    const half * p = tile + (row0 + lane % 16) * pitch + k0;
#if defined(RDNA4)
    const uint4 v = *(const uint4 *) (p + 8 * (lane / 16));
    return *(const mmqw_frag_t *) &v;
#else
    uint4 v[2];
    v[0] = *(const uint4 *) (p);
    v[1] = *(const uint4 *) (p + 8);
    return *(const mmqw_frag_t *) v;
#endif
}

// int8 quad (already XORed with 0x80 so each byte is q + 128) -> two f16 values q*d, via the exponent trick:
// f16 bits 0x6400 | u encode 1024 + u exactly for u < 256, so (1024 + q + 128) - 1152 = q, folded into one fma.
static __device__ __forceinline__ uint32_t mmqw_dq2(const uint32_t qx, const uint32_t sel, const half2 d2, const half2 nd2) {
#if defined(AMD_WMMA_AVAILABLE)
    const uint32_t bits = __builtin_amdgcn_perm(0x64006400u, qx, sel);
#else
    const uint32_t bits = 0; GGML_UNUSED_VARS(qx, sel);
#endif
    const half2    h    = *(const half2 *) &bits;
    const half2    r    = __hfma2(h, d2, nd2);
    return *(const uint32_t *) &r;
}

// k-quant staging (q4_K / q5_K / q6_K): a unit is 16 consecutive weights of one row, 16-aligned in K, so it lies in one
// 16-element scale group of q6_K and one 32-element sub-block of q4_K / q5_K. The raw bytes are fetched in load_stage
// (with a small word of unit geometry), decoded at the LDS store. A code byte q (< 64) becomes a half through the
// exponent trick (0x6400 | q = 1024 + q exactly); one exact packed subtract and one packed fma then give
// w = (q - c) * s + b, rounded once to f16.
static __device__ __forceinline__ uint32_t mmqw_kq2(const uint32_t qx, const uint32_t sel, const half2 c2, const half2 s2, const half2 b2) {
#if defined(AMD_WMMA_AVAILABLE)
    const uint32_t bits = __builtin_amdgcn_perm(0x64006400u, qx, sel);
#else
    const uint32_t bits = 0; GGML_UNUSED_VARS(qx, sel);
#endif
    const half2 h = __hsub2(*(const half2 *) &bits, c2);
    const half2 r = __hfma2(h, s2, b2);
    return *(const uint32_t *) &r;
}

// byte i of a 16-byte q4_K / q5_K header (d, dmin, scales[12])
static __device__ __forceinline__ uint32_t mmqw_hb(const uint4 & h, const int i) {
    const uint32_t w = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
    return (w >> (8 * (i & 3))) & 0xFFu;
}

template <ggml_type WT> struct mmqw_wtraits;
template <> struct mmqw_wtraits<GGML_TYPE_Q8_0> { static constexpr int QK = 32;  static constexpr int BYTES = 34;  };
template <> struct mmqw_wtraits<GGML_TYPE_Q4_K> { static constexpr int QK = 256; static constexpr int BYTES = 144; };
template <> struct mmqw_wtraits<GGML_TYPE_Q5_K> { static constexpr int QK = 256; static constexpr int BYTES = 176; };
template <> struct mmqw_wtraits<GGML_TYPE_Q6_K> { static constexpr int QK = 256; static constexpr int BYTES = 210; };

// halo-hybrid: the hyper-connection mix epilogue (HC > 0). The GEMM is qwen4exp's hc up projection g = W x (M = HC*n_embd
// rows, row c*n_embd + i = stream c of channel i), and instead of storing g the block computes what k_hc_mix (hc.cu) reads
// it for: out[n][i] = scale * sum_c xn[n][c][i] * sigmoid(g[n][c*n_embd + i]) + bias, with the c-sum in k_hc_mix's order.
// The weight rows are read through a remap that gives each warp's HC consecutive 16-row fragments the HC streams of the same
// 16 channels (logical m: group of 16*HC rows -> 16 channels, fragment c = stream c, lane = channel), so every lane holds
// all streams of its channel: no shuffles, and the xn reads and out writes are 64-byte runs. Each g is the same
// accumulator value the plain GEMM stores, so out is bit-identical.
struct mmqw_mix_args {
    const float * xn;     // [n][HC][n_embd]
    float *       out;    // [n][n_embd]
    float         scale;
    float         bias;
    int           n_embd;
    // xn left unwritten by the hc boundary (ggml_cuda_hc_xn_lazy): rebuilt here as (rstd * res) * gamma, its arithmetic
    const float * res        = nullptr;   // [n][HC][n_embd]
    const float * rstd       = nullptr;   // [n][HC]
    const float * gamma      = nullptr;   // [gamma_rows][n_embd]
    int           gamma_rows = 1;
    half *        out16      = nullptr;   // O16: the f16 copy of out its consumers' WMMA / routed GEMMs read (f16 side copy)
};

static __device__ __forceinline__ float mmqw_hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));   // identical to hc.cu's hc_sigmoid
}

template <ggml_type WT, int BN, int BM, int KT, int WN, int WM, int NBUF, int PF, bool XF32, int HC = 0, bool LZ = false, bool O16 = false>
static __global__ void __launch_bounds__(32 * WN * WM) mul_mat_q_wmma(
        const void * __restrict__ W, const void * __restrict__ Xv, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t ldx, const int64_t ldd, const mmqw_mix_args mix) {
    static_assert(KT % 32 == 0, "K tile must cover whole q8_0 blocks");
    constexpr bool KQ = WT != GGML_TYPE_Q8_0;
    static_assert(HC == 0 || (!KQ && (BM / (16 * WM)) % HC == 0), "the mix epilogue covers q8_0 with a multiple of HC fragments per warp");
    static_assert(!KQ || 256 % KT == 0, "a K tile must not straddle k-quant superblocks");
    static_assert(BN % (16 * WN) == 0 && BM % (16 * WM) == 0, "warp grid must tile the block");
    static_assert(PF == 1 || PF == 2, "register prefetch depth");
    constexpr int TN      = BN / (16 * WN);   // fragments per warp along n
    constexpr int TM      = BM / (16 * WM);   // fragments per warp along m
    constexpr int NT      = 32 * WN * WM;     // threads
    constexpr int PITCH   = KT + 8;           // halves per LDS row
    constexpr int KB      = KT / 32;          // q8_0 blocks per row per K tile
    constexpr int NSLICE  = KT / 16;
    constexpr int SU      = NSLICE < 2 ? NSLICE : 2;   // slices unrolled together (bounds the live fragments)

    // global -> register staging per K tile:
    //   activations: BN rows x KT halves = BN*KT*2 bytes -> XA uint4 per thread
    //   weights:     BM rows x KB blocks (34 bytes each), staged as half-blocks (16 int8 + the block's d) -> XW per thread
    static_assert((BN * KT * 2) % (16 * NT) == 0, "activation tile must split into uint4 per thread");
    static_assert((BM * KB * 2) % NT == 0, "weight half-blocks must split evenly over the threads");
    constexpr int XA = (BN * KT * 2) / (16 * NT);
    constexpr int XW = (BM * KB * 2) / NT;
    constexpr int A_PER_ROW = KT * 2 / 16;    // uint4 (of f16) per activation row
    constexpr int XAR = XF32 ? 2 * XA : XA;   // raw staging registers: f32 input is twice the bytes, converted at the LDS store
    const half  * __restrict__ X16 = (const half  *) Xv;
    const float * __restrict__ X32 = (const float *) Xv;

    extern __shared__ char mmqw_smem[];
    half * As = (half *) mmqw_smem;                       // [NBUF][BN][PITCH]
    half * Bs = As + NBUF * BN * PITCH;                   // [NBUF][BM][PITCH]

    const int tid  = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int wn   = warp % WN;
    const int wm   = warp / WN;
    const int n0   = blockIdx.y * BN;
    const int m0   = blockIdx.x * BM;

    const block_q8_0 * __restrict__ Wq = (const block_q8_0 *) W;
    const int nbk = K / 32;                               // blocks per weight row
    const char * __restrict__ Wb = (const char *) W;
    const int64_t row_bytes = (int64_t) (K / mmqw_wtraits<WT>::QK) * mmqw_wtraits<WT>::BYTES;

    floatx8_t acc[TN][TM];
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                acc[i][j][e] = 0.0f;
            }
        }
    }

    // staging registers, PF tiles deep (tile t lives in stage t % PF)
    uint4    ra[PF][XAR];
    uint4    rw[PF][XW];              // q8_0: 16 int8 of a half-block; k-quants: 16 code bytes (ql / qs)
    half     rd[PF][XW];
    uint4    rh[PF][KQ ? XW : 1];     // q4_K / q5_K: header; q6_K: .x = d | scale << 16
    uint4    rq[PF][KQ ? XW : 1];     // q5_K / q6_K: 16 high-bit bytes (qh)
    int      rm[PF][KQ ? XW : 1];     // unit geometry: q4/q5_K: sub-block j | nibble << 3 | qh bit << 4; q6_K: quarter qd

    auto load_stage = [&](const int k0, const int st) {
#pragma unroll
        for (int x = 0; x < XA; ++x) {
            const int idx = tid + x * NT;                 // uint4 index within the BN x KT tile
            const int row = idx / A_PER_ROW;
            const int col = (idx % A_PER_ROW) * 8;        // halves
            const int n   = n0 + row;
            if constexpr (XF32) {
                const float * src = X32 + (int64_t) n * ldx + k0 + col;
                ra[st][2*x + 0] = n < N ? *(const uint4 *) (src)     : make_uint4(0, 0, 0, 0);
                ra[st][2*x + 1] = n < N ? *(const uint4 *) (src + 4) : make_uint4(0, 0, 0, 0);
            } else {
                ra[st][x] = n < N ? *(const uint4 *) (X16 + (int64_t) n * ldx + k0 + col) : make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int x = 0; x < XW; ++x) {
            const int idx = tid + x * NT;                 // half-block index within the BM x KB tile
            const int blk = idx / 2;
            const int hf  = idx % 2;
            const int row = blk / KB;
            const int kb  = blk % KB;
            int m = min(m0 + row, M - 1);
            if constexpr (HC > 0) {
                // logical row m: group g = m / (16*HC), stream c = (m / 16) % HC, channel 16*g + m % 16 -> weight row c*n_embd + ch
                m = ((m / 16) % HC) * (M / HC) + (m / (16 * HC)) * 16 + m % 16;
            }
            if constexpr (!KQ) {
            const block_q8_0 * b = Wq + (int64_t) m * nbk + k0 / 32 + kb;
            rd[st][x] = b->d;
            rw[st][x] = *(const uint4 *) (b->qs + 16 * hf);   // 2-byte aligned: AMD runs unaligned dwordx4 loads
            } else {
            // unit x: row = idx / (KT/16), 16-weight slot u = idx % (KT/16) (same count as the q8_0 half-blocks)
            const int urow = idx / (KT / 16);
            const int u    = idx % (KT / 16);
            const int mm   = min(m0 + urow, M - 1);
            const int k    = k0 + 16 * u;
            const char * sb = Wb + (int64_t) mm * row_bytes + (int64_t) (k / 256) * mmqw_wtraits<WT>::BYTES;
            const int o    = k % 256;
            if constexpr (WT == GGML_TYPE_Q4_K || WT == GGML_TYPE_Q5_K) {
                constexpr int QS = WT == GGML_TYPE_Q4_K ? 16 : 48;
                const int g = o / 64, w = o % 64;
                rh[st][x] = *(const uint4 *) sb;
                rw[st][x] = *(const uint4 *) (sb + QS + 32 * g + (w % 32));
                if constexpr (WT == GGML_TYPE_Q5_K) {
                    rq[st][x] = *(const uint4 *) (sb + 16 + (w % 32));
                }
                rm[st][x] = (o / 32) | ((w >= 32) << 3) | ((2 * g + (w >= 32)) << 4);
            } else {
                const int n = o / 128, r = o % 128, qd = r / 32, l0 = r % 32;
                rw[st][x] = *(const uint4 *) (sb + 64 * n + 32 * (qd % 2) + l0);   // unaligned (210-byte blocks)
                rq[st][x] = *(const uint4 *) (sb + 128 + 32 * n + l0);
                const uint32_t d  = *(const uint16_t *) (sb + 208);
                const uint32_t sc = (uint8_t) sb[192 + 8 * n + l0 / 16 + 2 * qd];
                rh[st][x].x = d | (sc << 16);
                rm[st][x] = qd;
            }
            GGML_UNUSED_VARS(blk, hf, row, kb, m);
            }
        }
    };

    auto store_stage = [&](const int buf, const int st) {
        half * as = As + buf * BN * PITCH;
        half * bs = Bs + buf * BM * PITCH;
#pragma unroll
        for (int x = 0; x < XA; ++x) {
            const int idx = tid + x * NT;
            const int row = idx / A_PER_ROW;
            const int col = (idx % A_PER_ROW) * 8;
            if constexpr (XF32) {
                const float4 f0 = *(const float4 *) &ra[st][2*x + 0];
                const float4 f1 = *(const float4 *) &ra[st][2*x + 1];
                half2 h[4] = {__floats2half2_rn(f16_sat(f0.x), f16_sat(f0.y)), __floats2half2_rn(f16_sat(f0.z), f16_sat(f0.w)), __floats2half2_rn(f16_sat(f1.x), f16_sat(f1.y)), __floats2half2_rn(f16_sat(f1.z), f16_sat(f1.w))};
                *(uint4 *) (as + row * PITCH + col) = *(const uint4 *) h;
            } else {
                *(uint4 *) (as + row * PITCH + col) = ra[st][x];
            }
        }
#pragma unroll
        for (int x = 0; x < XW; ++x) {
            const int idx = tid + x * NT;
            const int blk = idx / 2;
            const int hf  = idx % 2;
            const int row = blk / KB;
            const int kb  = blk % KB;
            uint4 out[2];
            if constexpr (KQ) {
                const int urow = idx / (KT / 16);
                const int u    = idx % (KT / 16);
                const uint32_t w4[4] = {rw[st][x].x, rw[st][x].y, rw[st][x].z, rw[st][x].w};
                uint32_t q[4];
                float sc_f, b_f, c_f;
                if constexpr (WT == GGML_TYPE_Q4_K || WT == GGML_TYPE_Q5_K) {
                    const int j = rm[st][x] & 7, hi = (rm[st][x] >> 3) & 1, bit = rm[st][x] >> 4;
                    const uint4 h = rh[st][x];
                    uint32_t sc, mn;
                    if (j < 4) {
                        sc = mmqw_hb(h, 4 + j) & 63;
                        mn = mmqw_hb(h, 8 + j) & 63;
                    } else {
                        sc = (mmqw_hb(h, 8 + j) & 0xF) | ((mmqw_hb(h, j) >> 6) << 4);
                        mn = (mmqw_hb(h, 8 + j) >> 4)  | ((mmqw_hb(h, 4 + j) >> 6) << 4);
                    }
                    const float d = __half2float(__ushort_as_half((unsigned short) (h.x & 0xFFFF)));
                    const float dm = __half2float(__ushort_as_half((unsigned short) (h.x >> 16)));
                    sc_f = d * (float) sc; b_f = -dm * (float) mn; c_f = 1024.0f;
                    const uint32_t qh4[4] = {rq[st][x].x, rq[st][x].y, rq[st][x].z, rq[st][x].w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        q[i] = (w4[i] >> (4 * hi)) & 0x0F0F0F0Fu;
                        if constexpr (WT == GGML_TYPE_Q5_K) {
                            q[i] |= ((qh4[i] >> bit) & 0x01010101u) << 4;
                        } else {
                            GGML_UNUSED(qh4); GGML_UNUSED(bit);
                        }
                    }
                } else {
                    const int qd = rm[st][x];
                    const float d = __half2float(__ushort_as_half((unsigned short) (rh[st][x].x & 0xFFFF)));
                    const int8_t sc = (int8_t) (rh[st][x].x >> 16);
                    sc_f = d * (float) sc; b_f = 0.0f; c_f = 1056.0f;
                    const uint32_t qh4[4] = {rq[st][x].x, rq[st][x].y, rq[st][x].z, rq[st][x].w};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        q[i] = ((w4[i] >> (4 * (qd / 2))) & 0x0F0F0F0Fu) | (((qh4[i] >> (2 * qd)) & 0x03030303u) << 4);
                    }
                }
                const half2 c2 = __float2half2_rn(c_f);
                const half2 s2 = __float2half2_rn(sc_f);
                const half2 b2 = __float2half2_rn(b_f);
                out[0].x = mmqw_kq2(q[0], 0x07010500u, c2, s2, b2);
                out[0].y = mmqw_kq2(q[0], 0x07030502u, c2, s2, b2);
                out[0].z = mmqw_kq2(q[1], 0x07010500u, c2, s2, b2);
                out[0].w = mmqw_kq2(q[1], 0x07030502u, c2, s2, b2);
                out[1].x = mmqw_kq2(q[2], 0x07010500u, c2, s2, b2);
                out[1].y = mmqw_kq2(q[2], 0x07030502u, c2, s2, b2);
                out[1].z = mmqw_kq2(q[3], 0x07010500u, c2, s2, b2);
                out[1].w = mmqw_kq2(q[3], 0x07030502u, c2, s2, b2);
                half * p = bs + urow * PITCH + 16 * u;
                *(uint4 *) (p)     = out[0];
                *(uint4 *) (p + 8) = out[1];
                GGML_UNUSED_VARS(blk, hf, row, kb);
                continue;
            }
            const float df = __half2float(rd[st][x]);
            const half2 d2  = __float2half2_rn(df);
            const half2 nd2 = __float2half2_rn(-1152.0f * df);
            const uint32_t q[4] = {rw[st][x].x ^ 0x80808080u, rw[st][x].y ^ 0x80808080u, rw[st][x].z ^ 0x80808080u, rw[st][x].w ^ 0x80808080u};
            out[0].x = mmqw_dq2(q[0], 0x07010500u, d2, nd2);   // bytes 0,1 -> halves 0,1
            out[0].y = mmqw_dq2(q[0], 0x07030502u, d2, nd2);   // bytes 2,3
            out[0].z = mmqw_dq2(q[1], 0x07010500u, d2, nd2);
            out[0].w = mmqw_dq2(q[1], 0x07030502u, d2, nd2);
            out[1].x = mmqw_dq2(q[2], 0x07010500u, d2, nd2);
            out[1].y = mmqw_dq2(q[2], 0x07030502u, d2, nd2);
            out[1].z = mmqw_dq2(q[3], 0x07010500u, d2, nd2);
            out[1].w = mmqw_dq2(q[3], 0x07030502u, d2, nd2);
            half * p = bs + row * PITCH + kb * 32 + 16 * hf;
            *(uint4 *) (p)     = out[0];
            *(uint4 *) (p + 8) = out[1];
        }
    };

    auto compute_tile = [&](const int buf) {
        const half * as = As + buf * BN * PITCH + (wn * TN * 16) * PITCH;
        const half * bs = Bs + buf * BM * PITCH + (wm * TM * 16) * PITCH;
#pragma unroll 1
        for (int s0 = 0; s0 < NSLICE; s0 += SU) {
#pragma unroll
            for (int si = 0; si < SU; ++si) {
                const int s = s0 + si;
                mmqw_frag_t a[TN];
                mmqw_frag_t b[TM];
#pragma unroll
                for (int i = 0; i < TN; ++i) {
                    a[i] = mmqw_load_frag(as, PITCH, i * 16, s * 16);
                }
#pragma unroll
                for (int j = 0; j < TM; ++j) {
                    b[j] = mmqw_load_frag(bs, PITCH, j * 16, s * 16);
                }
#pragma unroll
                for (int i = 0; i < TN; ++i) {
#pragma unroll
                    for (int j = 0; j < TM; ++j) {
                        mmqw_wmma(acc[i][j], a[i], b[j]);
                    }
                }
            }
        }
    };

    const int nkt = K / KT;

    // prologue: tiles 0..PF-1 into the register stages, tile 0 into LDS buffer 0
#pragma unroll
    for (int p = 0; p < PF; ++p) {
        if (p < nkt) {
            load_stage(p * KT, p);
        }
    }
    store_stage(0, 0);
    __syncthreads();

    for (int kt0 = 0; kt0 < nkt; kt0 += PF) {
#pragma unroll
        for (int p = 0; p < PF; ++p) {
            const int kt = kt0 + p;
            if (kt >= nkt) {
                break;
            }
            // tile kt is in LDS buffer kt % NBUF and its register stage (p) is free: fetch tile kt + PF into it
            if (kt + PF < nkt) {
                load_stage((kt + PF) * KT, p);
            }
            compute_tile(kt % NBUF);
            if (kt + 1 < nkt) {
                if constexpr (NBUF == 1) {
                    __syncthreads();                      // everyone is done reading before the overwrite
                }
                store_stage((kt + 1) % NBUF, (p + 1) % PF);
                __syncthreads();
            }
        }
    }

    // epilogue: element e of lane l holds C[n = 2e + l/16 (gfx11) | 8*(l/16) + e (gfx12)][m = l % 16]
    if constexpr (HC > 0) {
#pragma unroll
        for (int i = 0; i < TN; ++i) {
#pragma unroll
            for (int jg = 0; jg < TM; jg += HC) {
                const int mg = m0 + wm * TM * 16 + jg * 16;                 // first logical row of this channel group
                const int ie = (mg / (16 * HC)) * 16 + lane % 16;           // the lane's channel
                float gv[HC];   // LZ: the channel's gamma per stream
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    gv[c] = LZ && mg < M ? mix.gamma[(c % mix.gamma_rows) * mix.n_embd + ie] : 0.0f;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                    const int n = n0 + wn * TN * 16 + i * 16 + 8 * (lane / 16) + e;
#else
                    const int n = n0 + wn * TN * 16 + i * 16 + 2 * e + lane / 16;
#endif
                    if (n >= N || mg >= M) {
                        continue;
                    }
                    const int64_t xo = (int64_t) n * HC * mix.n_embd + ie;
                    const float * xr = mix.xn + xo;
                    float xv[HC];
#pragma unroll
                    for (int c = 0; c < HC; ++c) {
                        if constexpr (LZ) {
                            xv[c] = mix.rstd[(int64_t) n * HC + c] * mix.res[xo + c * mix.n_embd] * gv[c];
                        } else {
                            xv[c] = xr[c * mix.n_embd];
                        }
                    }
                    if (dst != nullptr) {   // debug (GGML_CUDA_HC_MIX_GEMM_CHECK): the raw g at its original rows
#pragma unroll
                        for (int c = 0; c < HC; ++c) {
                            dst[(int64_t) n * ldd + c * mix.n_embd + ie] = acc[i][jg + c][e];
                        }
                    }
                    // k_hc_mix's loop: acc = x0*s0, then acc + x_c*s_c rounded twice, as the unfused MUL and ADD kernels
                    float sum = xv[0] * mmqw_hc_sigmoid(acc[i][jg][e]);
#pragma unroll
                    for (int c = 1; c < HC; ++c) {
                        sum = ggml_cuda_add_mul_rn(sum, xv[c], mmqw_hc_sigmoid(acc[i][jg + c][e]));
                    }
                    const float r = mix.scale * sum + mix.bias;
                    mix.out[(int64_t) n * mix.n_embd + ie] = r;
                    if constexpr (O16) {
                        mix.out16[(int64_t) n * mix.n_embd + ie] = __float2half(f16_sat(r));
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < TN; ++i) {
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int m = m0 + wm * TM * 16 + j * 16 + lane % 16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
#if defined(RDNA4)
                const int n = n0 + wn * TN * 16 + i * 16 + 8 * (lane / 16) + e;
#else
                const int n = n0 + wn * TN * 16 + i * 16 + 2 * e + lane / 16;
#endif
                if (n < N && m < M) {
                    dst[(int64_t) n * ldd + m] = acc[i][j][e];
                }
            }
        }
    }
}

template <ggml_type WT, int BN, int BM, int KT, int WN, int WM, int NBUF, int PF, bool XF32, int HC = 0, bool LZ = false, bool O16 = false>
static void launch_mul_mat_q_wmma(const void * W, const void * X, float * dst, int M, int N, int K, int64_t ldx, int64_t ldd, cudaStream_t stream,
        const mmqw_mix_args mix = {}) {
    constexpr int PITCH = KT + 8;
    constexpr size_t smem = (size_t) NBUF * (BN + BM) * PITCH * sizeof(half);
    const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    mul_mat_q_wmma<WT, BN, BM, KT, WN, WM, NBUF, PF, XF32, HC, LZ, O16><<<grid, 32 * WN * WM, smem, stream>>>(W, X, dst, M, N, K, ldx, ldd, mix);
    CUDA_CHECK(cudaGetLastError());
}
template <ggml_type WT>
static bool mmqw_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    cudaStream_t stream = ctx.stream();
    const int64_t ldd = dst->nb[1] / sizeof(float);
    const float * x32 = (const float *) src1->data;

    // auto: 128x128 tiles when the grid fills the CUs twice over, 64x128 otherwise. The activations are read as f32
    // in-kernel unless the re-read volume is large (every m-tile re-reads its n-tile; at M*K > 16M the 2 MB f32
    // tile thrashes gfx1151's L2 and the kernel drops to 22 TFLOPS), then a f16 pre-pass (0.1 ms at 1024x4096).
    // 2-way split-K for the sub-4-wave grids was tried: the memset + atomics cost more than the tail it recovers.
#define MMQW_LAUNCH16(BN, BM, KT, WN, WM, NBUF, PF) launch_mul_mat_q_wmma<WT, BN, BM, KT, WN, WM, NBUF, PF, false>(src0->data, x16, (float *) dst->data, M, N, K, K, ldd, stream)
#define MMQW_LAUNCH32(BN, BM, KT, WN, WM, NBUF, PF) launch_mul_mat_q_wmma<WT, BN, BM, KT, WN, WM, NBUF, PF, true>(src0->data, x32, (float *) dst->data, M, N, K, K, ldd, stream)
    const int64_t blocks128 = ((M + 127) / 128) * ((N + 127) / 128);
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const bool big     = blocks128 >= 2 * nsm;
    // (halo-hybrid, 2026-09-26: 16M -> 8M. Qwen3.8's 2560 x 6144 ssm_out / attn_output and 6144 x 2560 attn_gate
    //  (15.7M each) read f32 in-kernel at 17.7 / 28 TFLOPS; with the pre-pass 2.49 vs 3.45 ms and 2.09 vs 2.40 ms per
    //  2048 tokens on gfx1151. The f16 rounding of the activations is the same either way.)
    static const int64_t prepass_min = ggml_env_int("GGML_CUDA_Q8_WMMA_PREPASS_MIN", (int64_t) 8 * 1024 * 1024);
    const bool prepass = M * K > prepass_min;
    if (prepass) {
        // the producer may have written the f16 copy already (hc.cu, f16 side copies): the same conversion
        const half * pre = ggml_cuda_f16_side_find(ctx, src1);
        ggml_cuda_pool_alloc<half> x16_buf(ctx.pool());
        if (pre == nullptr) {
            x16_buf.alloc((size_t) N * K);
            mmqw_f32_to_f16_sat<<<(N * K + 255) / 256, 256, 0, stream>>>((const float *) src1->data, x16_buf.get(), N * K);
        }
        const half * x16 = pre ? pre : x16_buf.get();
        if (big) { MMQW_LAUNCH16(128, 128, 64, 2, 4, 1, 1); } else { MMQW_LAUNCH16(64, 128, 64, 2, 2, 1, 1); }
        return true;
    }
    // halo-hybrid: a narrow weight matrix (Qwen3.8's hc down projection 10240 -> 320, the 640-row shared expert) gives the
    // 128x128 tile 3-5 m-tiles, i.e. 96-160 long-K blocks on 40 CUs: the hc down ran at ~7 TFLOPS. 32-token tiles with
    // 1x4 warps quadruple the blocks and keep each block's activation tile in cache: 1.07 -> 0.70 s of hc down per 11K
    // prompt on gfx1151 (16x128 ties, 64x64 / 128x64 / whole-M 320 tiles are slower). Every tile shape accumulates each
    // output over the same 16-wide K slices in the same order, so this is bit-identical. RDNA3 only (not measured on RDNA4).
    // GGML_CUDA_Q8_WMMA_NARROW=0 = off
    static const bool narrow = ggml_env_flag("GGML_CUDA_Q8_WMMA_NARROW", true);
    if (narrow && M <= 640 && GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc)) {
        // a producer-side f16 copy (the hc boundary's xn) is the conversion the f32 load does at the LDS store
        const half * x16 = ggml_cuda_f16_side_find(ctx, src1);
        if (x16 != nullptr) {
            MMQW_LAUNCH16(32, 128, 64, 1, 4, 1, 1);
        } else {
            MMQW_LAUNCH32(32, 128, 64, 1, 4, 1, 1);
        }
        return true;
    }
    if (big) { MMQW_LAUNCH32(128, 128, 64, 2, 4, 1, 1); } else { MMQW_LAUNCH32(64, 128, 64, 2, 2, 1, 1); }
#undef MMQW_LAUNCH16
#undef MMQW_LAUNCH32
    return true;
}

#endif // MMQ_WMMA_ENABLED

bool ggml_cuda_mul_mat_q_wmma_hc_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x,
        const ggml_tensor * xn, int64_t n_embd, int64_t hc, float scale, float bias, ggml_tensor * out, float * g_debug,
        const ggml_cuda_hc_xn_src * xs, half * out16) {
#ifndef MMQ_WMMA_ENABLED
    GGML_UNUSED_VARS(ctx, w, x, xn, n_embd, hc, scale, bias, out, g_debug, xs, out16);
    return false;
#else
    static const bool off = ggml_env_flag("GGML_CUDA_NO_HC_MIX_GEMM", false);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (off || !GGML_CUDA_CC_IS_RDNA3(cc) || w->type != GGML_TYPE_Q8_0 || hc != 4 || x->type != GGML_TYPE_F32 ||
            xn->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(x) || !ggml_is_contiguous(xn) || !ggml_is_contiguous(out) ||
            w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) {
        return false;
    }
    const int64_t K = w->ne[0];
    const int64_t M = w->ne[1];
    const int64_t N = x->ne[1];
    // the same path the plain GEMM would take for this shape (f32 activations read in-kernel), see mmqw_launch
    static const int64_t prepass_min = ggml_env_int("GGML_CUDA_Q8_WMMA_PREPASS_MIN", (int64_t) 8 * 1024 * 1024);
    if (K % 64 != 0 || M != hc * n_embd || M % 128 != 0 || n_embd % 16 != 0 || N < 64 || x->ne[0] != K || M * K > prepass_min ||
            xn->ne[0] != M || xn->ne[1] != N || out->ne[0] != n_embd || out->ne[1] != N) {
        return false;
    }
    mmqw_mix_args mix = { (const float *) xn->data, (float *) out->data, scale, bias, (int) n_embd };
    if (xs != nullptr) {
        mix.res = xs->res; mix.rstd = xs->rstd; mix.gamma = xs->gamma; mix.gamma_rows = xs->gamma_rows;
    }
    mix.out16 = out16;
#define MMQW_MIX(LZ, O16) launch_mul_mat_q_wmma<GGML_TYPE_Q8_0, 128, 128, 64, 4, 2, 1, 1, true, 4, LZ, O16>(w->data, x->data, g_debug, \
        (int) M, (int) N, (int) K, K, M, ctx.stream(), mix)
    if (xs != nullptr) {
        if (out16 != nullptr) { MMQW_MIX(true, true); } else { MMQW_MIX(true, false); }
    } else {
        if (out16 != nullptr) { MMQW_MIX(false, true); } else { MMQW_MIX(false, false); }
    }
#undef MMQW_MIX
    return true;
#endif
}

#ifdef MMQ_WMMA_ENABLED
// the checks ggml_cuda_mul_mat_q_wmma takes a call on (dst aside): an f32 dst of the same contiguous shape is assumed
static bool mmqw_accepts(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1);
#endif

bool ggml_cuda_mul_mat_q_wmma_prepass(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifndef MMQ_WMMA_ENABLED
    GGML_UNUSED_VARS(ctx, src0, src1);
    return false;
#else
    static const int64_t prepass_min = ggml_env_int("GGML_CUDA_Q8_WMMA_PREPASS_MIN", (int64_t) 8 * 1024 * 1024);
    return mmqw_accepts(ctx, src0, src1) && src0->ne[0] * src0->ne[1] > prepass_min;
#endif
}

bool ggml_cuda_mul_mat_q_wmma_f16_input(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifndef MMQ_WMMA_ENABLED
    GGML_UNUSED_VARS(ctx, src0, src1);
    return false;
#else
    // mmqw_launch's pre-pass and narrow branches look for a side copy; the other f32 tiles read src1 as f32
    static const bool narrow = ggml_env_flag("GGML_CUDA_Q8_WMMA_NARROW", true);
    return ggml_cuda_mul_mat_q_wmma_prepass(ctx, src0, src1) ||
           (mmqw_accepts(ctx, src0, src1) && narrow && src0->ne[1] <= 640 && GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[ctx.device].cc));
#endif
}

#ifdef MMQ_WMMA_ENABLED
static bool mmqw_accepts(const ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
    // GGML_CUDA_Q8_WMMA: 0 = off (MMQ), any other value = on; unset: on for RDNA3, off for RDNA4 (its int8 WMMA rate
    // keeps MMQ ahead there)
    static const int mode = (int) ggml_env_int("GGML_CUDA_Q8_WMMA", -1);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))) {
        return false;
    }
    // k-quants (q4_K / q5_K / q6_K): GGML_CUDA_KQ_WMMA = 0 off (MMQ), unset / 1 on for RDNA3 / RDNA4. MMQ applies a float
    // scale per 16 (q6_K) or 32 (q4_K / q5_K) K per output element; on gfx1201 that leaves q6_K at 37 and q4_K at 67 TFLOPS
    // against 88 for q8_0, while this kernel runs every type at the q8_0 rate.
    static const int kq_mode = (int) ggml_env_int("GGML_CUDA_KQ_WMMA", -1);
    const bool kq = src0->type == GGML_TYPE_Q4_K || src0->type == GGML_TYPE_Q5_K || src0->type == GGML_TYPE_Q6_K;
    if (kq) {
        if (kq_mode == 0 || !(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))) {
            return false;
        }
    } else if (src0->type != GGML_TYPE_Q8_0 || mode == 0 || (mode < 0 && !GGML_CUDA_CC_IS_RDNA3(cc))) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (K % (kq ? 256 : 64) != 0 || N < 64 || M < 128 || M > (1 << 30) || N > (1 << 30)) {
        return false;
    }
    return true;
}
#endif // MMQ_WMMA_ENABLED

bool ggml_cuda_mul_mat_q_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifndef MMQ_WMMA_ENABLED
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#else
    if (dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) || !mmqw_accepts(ctx, src0, src1)) {
        return false;
    }

    switch (src0->type) {
        case GGML_TYPE_Q4_K: return mmqw_launch<GGML_TYPE_Q4_K>(ctx, src0, src1, dst);
        case GGML_TYPE_Q5_K: return mmqw_launch<GGML_TYPE_Q5_K>(ctx, src0, src1, dst);
        case GGML_TYPE_Q6_K: return mmqw_launch<GGML_TYPE_Q6_K>(ctx, src0, src1, dst);
        default:             return mmqw_launch<GGML_TYPE_Q8_0>(ctx, src0, src1, dst);
    }
#endif
}
