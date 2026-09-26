#include "mmq-wmma.cuh"
#include "convert.cuh"

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

template <int BN, int BM, int KT, int WN, int WM, int NBUF, int PF, bool XF32>
static __global__ void __launch_bounds__(32 * WN * WM) mul_mat_q8_0_wmma(
        const void * __restrict__ W, const void * __restrict__ Xv, float * __restrict__ dst,
        const int M, const int N, const int K, const int64_t ldx, const int64_t ldd) {
    static_assert(KT % 32 == 0, "K tile must cover whole q8_0 blocks");
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
    uint4    rw[PF][XW];
    half     rd[PF][XW];

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
            const int m   = min(m0 + row, M - 1);
            const block_q8_0 * b = Wq + (int64_t) m * nbk + k0 / 32 + kb;
            rd[st][x] = b->d;
            rw[st][x] = *(const uint4 *) (b->qs + 16 * hf);   // 2-byte aligned: AMD runs unaligned dwordx4 loads
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
                half2 h[4] = {__floats2half2_rn(f0.x, f0.y), __floats2half2_rn(f0.z, f0.w), __floats2half2_rn(f1.x, f1.y), __floats2half2_rn(f1.z, f1.w)};
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
            const float df = __half2float(rd[st][x]);
            const half2 d2  = __float2half2_rn(df);
            const half2 nd2 = __float2half2_rn(-1152.0f * df);
            const uint32_t q[4] = {rw[st][x].x ^ 0x80808080u, rw[st][x].y ^ 0x80808080u, rw[st][x].z ^ 0x80808080u, rw[st][x].w ^ 0x80808080u};
            uint4 out[2];
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

template <int BN, int BM, int KT, int WN, int WM, int NBUF, int PF, bool XF32>
static void launch_mul_mat_q8_0_wmma(const void * W, const void * X, float * dst, int M, int N, int K, int64_t ldx, int64_t ldd, cudaStream_t stream) {
    constexpr int PITCH = KT + 8;
    constexpr size_t smem = (size_t) NBUF * (BN + BM) * PITCH * sizeof(half);
    const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
    mul_mat_q8_0_wmma<BN, BM, KT, WN, WM, NBUF, PF, XF32><<<grid, 32 * WN * WM, smem, stream>>>(W, X, dst, M, N, K, ldx, ldd);
    CUDA_CHECK(cudaGetLastError());
}
#endif // MMQ_WMMA_ENABLED

bool ggml_cuda_mul_mat_q8_0_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
#ifndef MMQ_WMMA_ENABLED
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
#else
    // GGML_CUDA_Q8_WMMA: 0 = off (MMQ), 1 = auto (default on RDNA3; RDNA4's int8 WMMA rate keeps MMQ ahead there, so
    // off unless asked), 2-10 fixed configs on an f16 pre-pass, 11-13 fixed configs reading f32 (tuning knobs)
    static const int mode = getenv("GGML_CUDA_Q8_WMMA") ? atoi(getenv("GGML_CUDA_Q8_WMMA")) : -1;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (mode == 0 || !(GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) || (mode < 0 && !GGML_CUDA_CC_IS_RDNA3(cc))) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    if (K % 64 != 0 || N < 64 || M < 128 || M > (1 << 30) || N > (1 << 30)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    const int64_t ldd = dst->nb[1] / sizeof(float);
    const float * x32 = (const float *) src1->data;

    // auto: 128x128 tiles when the grid fills the CUs twice over, 64x128 otherwise. The activations are read as f32
    // in-kernel unless the re-read volume is large (every m-tile re-reads its n-tile; at M*K > 16M the 2 MB f32
    // tile thrashes gfx1151's L2 and the kernel drops to 22 TFLOPS), then a f16 pre-pass (0.1 ms at 1024x4096).
    // 2-way split-K for the sub-4-wave grids was tried: the memset + atomics cost more than the tail it recovers.
#define MMQW_LAUNCH16(BN, BM, KT, WN, WM, NBUF, PF) launch_mul_mat_q8_0_wmma<BN, BM, KT, WN, WM, NBUF, PF, false>(src0->data, x16.get(), (float *) dst->data, M, N, K, K, ldd, stream)
#define MMQW_LAUNCH32(BN, BM, KT, WN, WM, NBUF, PF) launch_mul_mat_q8_0_wmma<BN, BM, KT, WN, WM, NBUF, PF, true>(src0->data, x32, (float *) dst->data, M, N, K, K, ldd, stream)
    const int64_t blocks128 = ((M + 127) / 128) * ((N + 127) / 128);
    const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
    const bool big     = blocks128 >= 2 * nsm;
    // (halo-hybrid, 2026-09-26: 16M -> 8M. Qwen3.8's 2560 x 6144 ssm_out / attn_output and 6144 x 2560 attn_gate
    //  (15.7M each) read f32 in-kernel at 17.7 / 28 TFLOPS; with the pre-pass 2.49 vs 3.45 ms and 2.09 vs 2.40 ms per
    //  2048 tokens on gfx1151. The f16 rounding of the activations is the same either way.)
    static const int64_t prepass_min = getenv("GGML_CUDA_Q8_WMMA_PREPASS_MIN") ? atoll(getenv("GGML_CUDA_Q8_WMMA_PREPASS_MIN")) : (int64_t) 8 * 1024 * 1024;
    const bool prepass = M * K > prepass_min;
    if ((mode >= 2 && mode <= 10) || (mode <= 1 && prepass)) {
        ggml_cuda_pool_alloc<half> x16(ctx.pool(), (size_t) N * K);
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        to_fp16(src1->data, x16.get(), N * K, stream);
        switch (mode) {
            case 2:  MMQW_LAUNCH16( 64, 128, 64, 2, 2, 1, 1); break;
            case 3:  MMQW_LAUNCH16( 64,  64, 32, 2, 2, 2, 1); break;
            case 4:  MMQW_LAUNCH16(128, 128, 32, 2, 4, 2, 1); break;
            case 5:  MMQW_LAUNCH16( 96, 128, 64, 2, 4, 1, 1); break;
            case 6:  MMQW_LAUNCH16(128, 128, 32, 2, 4, 2, 2); break;
            case 7:  MMQW_LAUNCH16(128, 128, 64, 2, 4, 1, 1); break;
            case 8:  MMQW_LAUNCH16(128, 128, 64, 2, 4, 1, 2); break;
            case 9:  MMQW_LAUNCH16( 64, 128, 32, 2, 2, 2, 2); break;
            case 10: MMQW_LAUNCH16( 64, 128, 64, 2, 2, 1, 1); break;
            default: if (big) { MMQW_LAUNCH16(128, 128, 64, 2, 4, 1, 1); } else { MMQW_LAUNCH16(64, 128, 64, 2, 2, 1, 1); } break;
        }
        return true;
    }
    switch (mode) {
        case 11: MMQW_LAUNCH32(128, 128, 64, 2, 4, 1, 1); break;
        case 12: MMQW_LAUNCH32( 64, 128, 64, 2, 2, 1, 1); break;
        case 13: MMQW_LAUNCH32( 64,  64, 64, 2, 2, 1, 1); break;
        default: if (big) { MMQW_LAUNCH32(128, 128, 64, 2, 4, 1, 1); } else { MMQW_LAUNCH32(64, 128, 64, 2, 2, 1, 1); } break;
    }
#undef MMQW_LAUNCH16
#undef MMQW_LAUNCH32
    return true;
#endif
}
