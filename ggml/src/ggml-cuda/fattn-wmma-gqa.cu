#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-wmma-gqa.cuh"

// halo-hybrid: Qwen3.8's QSA attention at prefill widths on RDNA3 (gfx11) WMMA: D = 256, 12 query heads per KV head,
// every query selecting up to 2051 cells (op param 4) through the mask.
//
// The MMA kernel's sparse tile is 16 queries x 4 heads, and those 16 queries share one gathered key list, the union
// of their selections: about 6x a single query's cells at depth, so most of its work multiplies keys a row has masked
// out. Here a block owns 4 consecutive queries x all 12 heads of one KV head (48 rows = 3 WMMA row blocks): the
// union of 4 selections is ~1.6x one, and every gathered K/V row feeds 48 rows instead of 64 of which only a few
// see it. The layout and the software pipeline are a port of gufo's WmmaCausalAttentionKernel (MIT, gufo
// contributors; src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp), on ggml's per-cell f16 mask and the
// group-of-4 index lists of ggml_cuda_flash_attn_ext_compact_mask.
//
// Numerics are the MMA kernel's: Q * scale rounded to f16, f16 x f16 -> f32 scores, f32 online softmax, P rounded to
// f16, f16 x f16 -> f32 PV, each row with its own mask values over the same cells. Only the key-tile boundaries
// (16 vs 32 cells) and thus the accumulation and rescale order differ.

#if defined(GGML_USE_HIP)

typedef __attribute__((ext_vector_type(8)))  float    fgqa_v8f;
typedef __attribute__((ext_vector_type(16))) _Float16 fgqa_v16h;

namespace {

constexpr int kD       = 256;
constexpr int kGqa     = 12;               // query heads per KV head
constexpr int kQueries = 4;                // queries per block
constexpr int kRows    = kQueries * kGqa;  // 48
constexpr int kRB      = kRows / 16;       // 3 row blocks
constexpr int kKeys    = 16;               // cells per key tile
constexpr int kKSteps  = kD / 16;          // 16
constexpr int kKStepsW = kKSteps / 2;      // S waves split D in two halves
constexpr int kKStride = kD + 8;           // k_lds row pitch (halves)
constexpr int kVtStride = kKeys + 8;       // v_lds (V^T) row pitch (halves)
constexpr int kSStride = 17;               // s_lds row pitch (floats)
constexpr int kKRegs   = (kKeys * (kD / 8)) / 256;   // 2 uint4 of K per thread per tile
constexpr int kVRegs   = (kKeys * kD) / (256 * 8);   // 2 uint4 of V per thread per tile
static_assert(kRows % 16 == 0, "row blocks");
static_assert(kKSteps == 2 * 8, "8 waves own two 16-dim O tiles each");

static __device__ __forceinline__ fgqa_v8f fgqa_wmma(const fgqa_v16h a, const fgqa_v16h b, const fgqa_v8f c) {
#if defined(AMD_WMMA_AVAILABLE) && !defined(RDNA4)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    GGML_UNUSED_VARS(a, b);
    return c;
#endif
}

static __device__ __forceinline__ fgqa_v16h fgqa_frag(const half * p) {
    return *reinterpret_cast<const fgqa_v16h *>(p);
}

// lane i gets lane i ^ off (off < 16) as a DPP row_xmask move instead of an LDS crossbar round trip
static __device__ __forceinline__ float fgqa_xor(const float v, const int off) {
#if defined(RDNA3)
    const int i = __builtin_bit_cast(int, v);
    switch (off) {
        case 1: return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, i, 0x161, 0xF, 0xF, false));
        case 2: return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, i, 0x162, 0xF, 0xF, false));
        default: break;
    }
#endif // defined(RDNA3)
    return __shfl_xor(v, off, 32);
}

// a barrier for LDS traffic only: __syncthreads also waits for the outstanding global loads, which would serialize the
// next tile's register prefetch behind it
static __device__ __forceinline__ void fgqa_sync_lds() {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_s_barrier();
    asm volatile("" ::: "memory");
}

} // namespace

// grid: (ceil(n_q / 4), n_kv_heads); 256 threads = 8 waves
__launch_bounds__(256, 2)
static __global__ void flash_attn_wmma_gqa12(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const half * __restrict__ mask, const int32_t * __restrict__ lists, float * __restrict__ dst,
        const float scale, const int n_q, const int list_stride,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2,
        const int64_t nbv1, const int64_t nbv2, const int64_t s31, const int64_t nbd1, const int64_t nbd2,
        const float * __restrict__ gate, const int64_t gate_s, const int64_t gate_hs, half * __restrict__ d16) {
#if defined(AMD_WMMA_AVAILABLE) && !defined(RDNA4)
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int wave    = tid >> 5;
    const int sub     = lane & 15;
    const int half_id = lane >> 4;

    const int group   = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int q0      = group * kQueries;

    const int32_t * list  = lists + int64_t(group) * list_stride;
    const int       count = list[0];
    list += 1;

    // row (rb, r) -> query rr / 12, head rr % 12 of this KV head, rr = rb*16 + r
    const auto row_query = [&](int rb, int r) { return q0 + (rb * 16 + r) / kGqa; };
    const auto row_head  = [&](int rb, int r) { return kv_head * kGqa + (rb * 16 + r) % kGqa; };

    __shared__ __attribute__((aligned(16))) half  k_lds[kKeys * kKStride];
    __shared__ __attribute__((aligned(16))) half  v_lds[kD * kVtStride];
    __shared__ __attribute__((aligned(16))) float s_lds[2][kRB][16][kSStride];
    __shared__ __attribute__((aligned(16))) half  p_lds[kRows][kKeys + 8];
    __shared__ float row_sum[kRows];
    __shared__ float row_scale[kRows];

    // S: waves 0..5 = 3 row blocks x 2 halves of D
    const int s_rb = wave % kRB;
    const int s_kh = wave / kRB;

    fgqa_v16h q_frag[kKStepsW];
    if (wave < 2 * kRB) {
        const int  qi   = row_query(s_rb, sub);
        const bool live = qi < n_q;
        const float * q_row = (const float *) (Q + int64_t(live ? qi : 0) * nbq1 + int64_t(row_head(s_rb, sub)) * nbq2);
#pragma unroll
        for (int ks = 0; ks < kKStepsW; ++ks) {
            const int d0 = (s_kh * kKStepsW + ks) * 16;
            const float4 * qp = reinterpret_cast<const float4 *>(q_row + d0);
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const float4 f = live ? qp[v] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                q_frag[ks][4*v + 0] = (_Float16) (f.x * scale);
                q_frag[ks][4*v + 1] = (_Float16) (f.y * scale);
                q_frag[ks][4*v + 2] = (_Float16) (f.z * scale);
                q_frag[ks][4*v + 3] = (_Float16) (f.w * scale);
            }
        }
    }

    // wave w owns dim tiles w and w + 8 of every row block
    fgqa_v8f o_acc[kRB][2] = {};
    // the same four threads own a softmax row for the whole sweep
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // a tile is 16 consecutive list entries starting at t0; entries past the count are dead
    const auto tile_cell = [&](int t0, int r) -> int {
        const int i = t0 + r;
        return i < count ? list[i] : -1;
    };

    const int  v_key   = lane % kKeys;
    const int  v_slice = (tid / kKeys) * (kVRegs * 8);
    const char * k_base = K + int64_t(kv_head) * nbk2;
    const char * v_base = V + int64_t(kv_head) * nbv2 + v_slice * sizeof(half);

    // V: one cell per lane, a 16-dim slice per thread, so the transposed LDS writes stay conflict-free
    // the cells a thread loads are themselves read from the list a step ahead (idx_*), so that each K/V/mask load
    // is one global round trip instead of a list read followed by a dependent load
    const auto idx_v = [&](int t0) { return tile_cell(t0, v_key); };
    const auto idx_k = [&](int t0, int * ci) {
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            ci[n] = tile_cell(t0, (tid + n * 256) / (kD / 8));
        }
    };
    const auto load_v = [&](const int c, uint4 * regs) {
        const uint4 * src = reinterpret_cast<const uint4 *>(v_base + int64_t(c < 0 ? 0 : c) * nbv1);
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            regs[j] = c >= 0 ? src[j] : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    // K: a wave reads one cell's 512 contiguous bytes
    const auto load_k = [&](const int * ci, uint4 * regs) {
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n * 256;
            const int c   = ci[n];
            const int d8  = (idx % (kD / 8)) * 8;
            regs[n] = c >= 0 ? *reinterpret_cast<const uint4 *>(k_base + int64_t(c) * nbk1 + d8 * sizeof(half))
                             : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    const auto stage_k = [&](const uint4 * regs) {
#pragma unroll
        for (int n = 0; n < kKRegs; ++n) {
            const int idx = tid + n * 256;
            *reinterpret_cast<uint4 *>(&k_lds[(idx / (kD / 8)) * kKStride + (idx % (kD / 8)) * 8]) = regs[n];
        }
    };
    const auto stage_v = [&](const uint4 * regs) {
        // neighbouring lanes hold neighbouring cells of the same dims: swapping halves with the partner lane lets each
        // lane store a cell pair of one dim as one word, the even lane the even dim and the odd lane the odd dim
        const bool odd = (v_key & 1) != 0;
#pragma unroll
        for (int j = 0; j < kVRegs; ++j) {
            const uint32_t words[4] = {regs[j].x, regs[j].y, regs[j].z, regs[j].w};
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t own     = words[w];
                const uint32_t partner = (uint32_t) __builtin_amdgcn_mov_dpp((int) own, 0xB1, 0xF, 0xF, false);
                const uint32_t pair    = odd ? __builtin_amdgcn_perm(partner, own, 0x03020706u)
                                             : __builtin_amdgcn_perm(partner, own, 0x05040100u);
                const int dim = v_slice + j * 8 + 2 * w + (odd ? 1 : 0);
                *reinterpret_cast<uint32_t *>(&v_lds[dim * kVtStride + (v_key & ~1)]) = pair;
            }
        }
    };

    // S = Q K^T of the tile in k_lds; each D half writes its own slot, the softmax sums them
    const auto scores = [&]() {
        if (wave < 2 * kRB) {
            fgqa_v8f s_acc = {};
#pragma unroll
            for (int ks = 0; ks < kKStepsW; ++ks) {
                const int d0 = (s_kh * kKStepsW + ks) * 16;
                s_acc = fgqa_wmma(q_frag[ks], fgqa_frag(&k_lds[sub * kKStride + d0]), s_acc);
                if ((ks + 1) % 2 == 0) {
                    // bound the live K fragments; hoisting all eight raises register use
                    __builtin_amdgcn_sched_barrier(0);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                s_lds[s_kh][s_rb][2*i + half_id][sub] = s_acc[i];
            }
        }
    };

    // the softmax thread's mask values for its row's four cells of a tile (-inf for a dead cell or row), loaded a
    // phase ahead of the softmax so the list and mask reads stay off its critical path
    const int   sm_rg   = tid / 4;
    const int   sm_seg  = tid % 4;
    const int   sm_rb   = sm_rg / 16;
    const int   sm_row  = sm_rg % 16;
    const int   sm_q    = sm_rg < kRows ? row_query(sm_rb, sm_row) : n_q;
    const half * sm_mrow = mask + int64_t(sm_q < n_q ? sm_q : 0) * s31;
    const auto idx_m = [&](int t0, int * ci) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            ci[m] = sm_q < n_q ? tile_cell(t0, sm_seg * 4 + m) : -1;
        }
    };
    const auto load_m = [&](const int * ci, half * mv) {
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            mv[m] = ci[m] >= 0 ? sm_mrow[ci[m]] : __float2half(-INFINITY);
        }
    };

    // online softmax over the tile, four threads per row, four cells each
    const auto softmax = [&](const half * mv) {
        const int rg = sm_rg;
        if (rg < kRows) {
            const int  seg  = sm_seg;
            const int  rb   = sm_rb;
            const int  row  = sm_row;
            float vals[4];
            float part_max = -INFINITY;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int col = seg * 4 + m;
                vals[m]  = s_lds[0][rb][row][col] + s_lds[1][rb][row][col] + __half2float(mv[m]);
                part_max = fmaxf(part_max, vals[m]);
            }
            part_max = fmaxf(part_max, fgqa_xor(part_max, 1));
            part_max = fmaxf(part_max, fgqa_xor(part_max, 2));
            const float prev_max    = running_max;
            const float next_max    = fmaxf(prev_max, part_max);
            const float prior_scale = isfinite(prev_max) ? expf(prev_max - next_max) : 0.0f;
            float part_sum = 0.0f;
            half p[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const float w = isfinite(vals[m]) ? expf(vals[m] - next_max) : 0.0f;
                part_sum += w;
                p[m] = __float2half(w);
            }
            __builtin_memcpy(&p_lds[rg][seg * 4], p, sizeof(p));
            part_sum += fgqa_xor(part_sum, 1);
            part_sum += fgqa_xor(part_sum, 2);
            running_max = next_max;
            running_sum = __fmaf_rn(prior_scale, running_sum, part_sum);
            if (seg == 0) {
                row_scale[rg] = prior_scale;
            }
        }
    };

    // rescale the running O by the tile's new maximum, then O += P V
    const auto accumulate = [&]() {
#pragma unroll
        for (int rb = 0; rb < kRB; ++rb) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float s = row_scale[rb * 16 + 2*i + half_id];
                o_acc[rb][0][i] *= s;
                o_acc[rb][1][i] *= s;
            }
        }
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int dim_tile = wave + t * 8;
            const fgqa_v16h v_frag = fgqa_frag(&v_lds[(dim_tile * 16 + sub) * kVtStride]);
#pragma unroll
            for (int rb = 0; rb < kRB; ++rb) {
                o_acc[rb][t] = fgqa_wmma(fgqa_frag(&p_lds[rb * 16 + sub][0]), v_frag, o_acc[rb][t]);
            }
        }
    };

    // Software-pipelined sweep (gufo's): phase A accumulates tile j's PV while the score waves compute tile j + 1's S;
    // phase B runs tile j + 1's softmax while staging K for tile j + 2 and V for tile j + 1, whose registers were
    // loaded one step earlier. Every row sees the same S, P, rescale and accumulation sequence as a plain sweep.
    uint4 k_regs[kKRegs] = {};
    uint4 v_regs[kVRegs] = {};
    half  m_regs[4];
    int   ki[kKRegs];        // cells of K(k_t + 16), V and mask of tile k_t: loaded a step before their use
    int   vi;
    int   mi[4];
    int cur  = 0;            // tile j: P in p_lds, V in v_lds
    int next = kKeys;        // tile j + 1: K in k_lds
    int k_t  = 2 * kKeys;    // tile j + 2: K in k_regs
    if (cur < count) {
        idx_k(cur, ki);
        idx_m(cur, mi);
        load_k(ki, k_regs);
        load_v(idx_v(cur), v_regs);
        load_m(mi, m_regs);
    }
    stage_k(k_regs);
    stage_v(v_regs);
    if (next < count) {
        idx_k(next, ki);
        load_k(ki, k_regs);
        load_v(idx_v(next), v_regs);
    }
    idx_m(next, mi);
    fgqa_sync_lds();
    if (cur < count) {
        scores();
    }
    fgqa_sync_lds();
    if (cur < count) {
        softmax(m_regs);
    }
    stage_k(k_regs);
    if (k_t < count) {
        idx_k(k_t, ki);
        load_k(ki, k_regs);
    }
    if (next < count) {
        load_m(mi, m_regs);
    }
    idx_k(k_t + kKeys, ki);
    vi = idx_v(k_t);
    idx_m(k_t, mi);
    fgqa_sync_lds();

    while (cur < count) {
        // phase A: k_lds = K(j + 1), v_lds = V(j), p_lds = P(j)
        if (next < count) {
            scores();
        }
        accumulate();
        // finish PV before the barrier; left alone the compiler sinks these WMMAs behind the softmax
#pragma unroll
        for (int rb = 0; rb < kRB; ++rb) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                asm volatile("" : "+v"(o_acc[rb][t]));
            }
        }
        fgqa_sync_lds();
        // phase B
        if (next < count) {
            softmax(m_regs);
        }
        stage_v(v_regs);
        stage_k(k_regs);
        cur  = next;
        next = k_t;
        k_t += kKeys;
        if (k_t < count) {
            load_k(ki, k_regs);
        }
        if (next < count) {
            load_v(vi, v_regs);
            load_m(mi, m_regs);
        }
        idx_k(k_t + kKeys, ki);
        vi = idx_v(k_t);
        idx_m(k_t, mi);
        fgqa_sync_lds();
    }
    if (tid < kRows * 4 && tid % 4 == 0) {
        row_sum[tid / 4] = running_sum;
    }
    fgqa_sync_lds();

#pragma unroll
    for (int rb = 0; rb < kRB; ++rb) {
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int dim_tile = wave + t * 8;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 2*i + half_id;
                const int qi  = row_query(rb, row);
                if (qi >= n_q) {
                    continue;
                }
                const float den = row_sum[rb * 16 + row];
                const int   h   = row_head(rb, row);
                const int   d   = dim_tile * 16 + sub;
                float v = den > 0.0f ? o_acc[rb][t][i] / den : 0.0f;
                if (gate) {
                    // the output gate (qwen4exp attn_gated, k_mul_sigmoid's x * sigmoid(g))
                    v = v * (1.0f / (1.0f + expf(-gate[int64_t(qi) * gate_s + int64_t(h) * gate_hs + d])));
                }
                float * out = (float *) ((char *) dst + int64_t(qi) * nbd2 + int64_t(h) * nbd1);
                out[d] = v;
                if (d16) {
                    // mmqw_f32_to_f16_sat's conversion, for the output projection's WMMA pre-pass
                    d16[(int64_t(qi) * (kGqa * gridDim.y) + h) * kD + d] = __float2half(v != v ? v : fminf(fmaxf(v, -65504.0f), 65504.0f));
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, lists, dst, scale, n_q, list_stride, nbq1, nbq2, nbk1, nbk2, nbv1, nbv2, s31, nbd1, nbd2,
        gate, gate_s, gate_hs, d16);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_AVAILABLE) && !defined(RDNA4)
}

#endif // defined(GGML_USE_HIP)

#if defined(GGML_USE_HIP)
static bool fa_gqa12_fits(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    // GGML_CUDA_FA_GQA12=0 = off (the MMA kernel's 16 x 4 union walk)
    static const bool enabled = ggml_env_flag("GGML_CUDA_FA_GQA12", true);
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || !GGML_CUDA_CC_IS_RDNA3(cc) || mask == nullptr || sinks != nullptr) {
        return false;
    }
    float max_bias, logit_softcap;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            dst->type != GGML_TYPE_F32 || Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 ||
            K->ne[2] == 0 || Q->ne[2] != kGqa * K->ne[2] || V->ne[2] != K->ne[2] || Q->ne[1] < 64 ||
            Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1 || mask->ne[3] != 1 || mask->ne[2] != 1 ||
            mask->ne[0] != K->ne[1] || mask->ne[1] < Q->ne[1] || n_kv_max <= 0 ||
            max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    // 16-byte rows for the vector loads; dst is [D, heads, queries]
    return !(Q->nb[0] != sizeof(float) || Q->nb[1] % 16 || Q->nb[2] % 16 || K->nb[0] != sizeof(half) || K->nb[1] % 16 ||
            K->nb[2] % 16 || V->nb[0] != sizeof(half) || V->nb[1] % 16 || V->nb[2] % 16 || dst->nb[0] != sizeof(float) ||
            dst->ne[0] != 256 || dst->ne[1] != Q->ne[2] || dst->ne[2] != Q->ne[1] ||
            (uintptr_t) Q->data % 16 || (uintptr_t) K->data % 16 || (uintptr_t) V->data % 16);
}

// out: dst itself, or (gated) a contiguous [D*heads, queries] tensor with the same element order
static void fa_gqa12_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * dst, float * out, const int64_t nbo1,
        const int64_t nbo2, const float * gate, const int64_t gate_s, const int64_t gate_hs, half * d16) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);

    cudaStream_t stream = ctx.stream();
    const int n_q    = (int) Q->ne[1];
    const int cap    = (int) std::min<int64_t>(int64_t(kQueries) * n_kv_max, K->ne[1]);
    const int groups = (int) ((mask->ne[1] + kQueries - 1) / kQueries);
    ggml_cuda_pool_alloc<int32_t> lists(ctx.pool(), size_t(groups) * (cap + 1));
    ggml_cuda_flash_attn_ext_compact_mask(mask, lists.get(), cap, kQueries, stream);

    const dim3 grid((n_q + kQueries - 1) / kQueries, (unsigned) K->ne[2], 1);
    flash_attn_wmma_gqa12<<<grid, 256, 0, stream>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data, (const half *) mask->data, lists.get(),
        out, scale, n_q, cap + 1, Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        mask->nb[1] / (int64_t) sizeof(half), nbo1, nbo2, gate, gate_s, gate_hs, d16);
    CUDA_CHECK(cudaGetLastError());
}
#endif // defined(GGML_USE_HIP)

bool ggml_cuda_flash_attn_ext_wmma_gqa12(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED_VARS(ctx, dst);
    return false;
#else
    if (!fa_gqa12_fits(ctx, dst)) {
        return false;
    }
    fa_gqa12_launch(ctx, dst, (float *) dst->data, dst->nb[1], dst->nb[2], nullptr, 0, 0, nullptr);
    return true;
#endif // !defined(GGML_USE_HIP)
}

bool ggml_cuda_flash_attn_ext_wmma_gqa12_gated(ggml_backend_cuda_context & ctx, const ggml_tensor * fa, const ggml_tensor * gate,
        ggml_tensor * out, half * d16) {
#if !defined(GGML_USE_HIP)
    GGML_UNUSED_VARS(ctx, fa, gate, out, d16);
    return false;
#else
    // gate: [D*heads, queries] contiguous rows, or the strided [D, heads, queries] view it is copied from
    const int64_t width = fa->ne[0] * fa->ne[1];
    if (!fa_gqa12_fits(ctx, fa) || !ggml_is_contiguous(fa) || gate->type != GGML_TYPE_F32 || out->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(out) || ggml_nelements(out) != ggml_nelements(fa) || out->ne[0] != width ||
            gate->nb[0] != sizeof(float) || gate->ne[3] != 1 || ggml_nelements(gate) != ggml_nelements(fa)) {
        return false;
    }
    int64_t gs, ghs;
    if (gate->ne[0] == width && gate->ne[1] == fa->ne[2] && gate->ne[2] == 1) {
        gs = gate->nb[1]; ghs = fa->ne[0] * sizeof(float);
    } else if (gate->ne[0] == fa->ne[0] && gate->ne[1] == fa->ne[1] && gate->ne[2] == fa->ne[2]) {
        gs = gate->nb[2]; ghs = gate->nb[1];
    } else {
        return false;
    }
    if (gs % sizeof(float) || ghs % sizeof(float)) {
        return false;
    }
    fa_gqa12_launch(ctx, fa, (float *) out->data, fa->ne[0] * sizeof(float), out->nb[1], (const float *) gate->data,
            gs / sizeof(float), ghs / sizeof(float), d16);
    return true;
#endif // !defined(GGML_USE_HIP)
}
