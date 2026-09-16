#include "persist.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// ---- task list ---------------------------------------------------------------------------------------------------

enum pk_op : int32_t { PK_QUANT = 0, PK_MMVQ, PK_RMS_NORM, PK_BIN, PK_SCALE, PK_UNARY, PK_SQR, PK_SQRT, PK_CPY, PK_GLU, PK_GET_ROWS };

struct pk_task {
    int32_t      op, n_items, n_deps, dep_first, dep_count, type, ncols, aux;
    const void * p[4];      // [0] dst, [1] src0, [2] src1, [3] src2 / ids
    int64_t      ne[3][4];  // dst, src0, src1
    int64_t      nb[3][4];  // byte strides
    float        f[4];
    int32_t      i[4];
    int32_t      x[4];      // body tuning: MMVQ x[0] = rows per wave (1/2/4); RMS_NORM x[0] = 1 -> whole block per row
};

// counters are monotonic across launches (no reset: a memset node inside a replayed hipGraph is not reliably
// ordered before the kernel node on ROCm); every launch works at an epoch and the thresholds scale with it
struct __attribute__((aligned(128))) pk_state {
    long long done;
    long long deps_done;
    long long pad[14];
};

struct pk_region_dev {
    const pk_task * tasks;
    pk_state      * state;
    const int     * dependents;
    long long     * arrivals;     // launch tickets: blocks of one launch share (ticket / gridDim.x)
    int             n_tasks;
    int             trace;        // GGML_CUDA_PERSIST_TRACE: block 0 stamps wait/work times per task into state pads
};

#define PK_BLOCK 1024
#define PK_WAVES (PK_BLOCK / 32)
#define PK_EW_ITEM 1024           // elements per element-wise work item (one per thread: an item is one latency, and idle blocks take the rest)
#define PK_QUANT_ITEM 1024        // elements per quantize work item (32 q8_1 blocks)
#define PK_SLEEP 8

__device__ __forceinline__ long long pk_ld_relaxed(long long * p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ long long pk_add_release(long long * p, long long v) { return __hip_atomic_fetch_add(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void pk_fence_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }

// ---- op bodies: every body is called by the whole block for one work item -----------------------------------------

// flat dst index -> 4D coordinates (32-bit)
__device__ __forceinline__ void pk_coords(uint32_t e, const int64_t * ne, uint32_t & i0, uint32_t & i1, uint32_t & i2, uint32_t & i3) {
    const uint32_t n0 = ne[0], n1 = ne[1], n2 = ne[2];
    i0 = e % n0; e /= n0; i1 = e % n1; e /= n1; i2 = e % n2; i3 = e / n2;
}
// pointers read out of the task struct are generic: without this the bodies compile to flat_load/flat_store (both
// wait counters, no address-space knowledge) and the GEMV ran at 120-150 GB/s instead of 210; the cast tells the
// address-space inference that everything under it is global memory
typedef __attribute__((address_space(1))) const char pk_gchar;
__device__ __forceinline__ const void * pk_g(const void * p) {
    pk_gchar * g = (pk_gchar *) p;
    asm("" : "+v"(g));   // opaque: otherwise the generic->global->generic round trip folds away and nothing is learned
    return (const void *) g;
}
__device__ __forceinline__ const char * pk_at(const void * base, const int64_t * nb, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    return (const char *) pk_g(base) + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3];
}
// broadcast index of src1 for a dst coordinate
__device__ __forceinline__ const char * pk_at_bcast(const void * base, const int64_t * ne, const int64_t * nb, uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) {
    return (const char *) base + (i0 % (uint32_t) ne[0])*nb[0] + (i1 % (uint32_t) ne[1])*nb[1] + (i2 % (uint32_t) ne[2])*nb[2] + (i3 % (uint32_t) ne[3])*nb[3];
}

__device__ __forceinline__ float pk_unary(int op, float x) {
    switch (op) {
        case GGML_UNARY_OP_SIGMOID:    return 1.0f / (1.0f + expf(-x));
        case GGML_UNARY_OP_SILU:       return x / (1.0f + expf(-x));
        case GGML_UNARY_OP_GELU:       return 0.5f*x*(1.0f + tanhf(0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x)));
        case GGML_UNARY_OP_GELU_QUICK: return x * (1.0f / (1.0f + expf(-1.702f*x)));
        case GGML_UNARY_OP_RELU:       return fmaxf(x, 0.0f);
        case GGML_UNARY_OP_EXP:        return expf(x);
        case GGML_UNARY_OP_NEG:        return -x;
        case GGML_UNARY_OP_ABS:        return fabsf(x);
        case GGML_UNARY_OP_TANH:       return tanhf(x);
        default:                       return x;
    }
}

// element-wise family: item = PK_EW_ITEM consecutive dst elements
__device__ __forceinline__ void pk_body_ew(const pk_task & t, int item) {
    const uint32_t n = (uint32_t) (t.ne[0][0] * t.ne[0][1] * t.ne[0][2] * t.ne[0][3]);
    float * dst = (float *) t.p[0];
    for (uint32_t e = item * PK_EW_ITEM + threadIdx.x; e < n && e < (uint32_t) (item + 1) * PK_EW_ITEM; e += PK_BLOCK) {
        uint32_t i0, i1, i2, i3; pk_coords(e, t.ne[0], i0, i1, i2, i3);
        float v;
        switch (t.op) {
            case PK_BIN: {
                const float a = *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3);
                const float b = *(const float *) pk_at_bcast(t.p[2], t.ne[2], t.nb[2], i0, i1, i2, i3);
                v = t.aux == GGML_OP_MUL ? a * b : t.aux == GGML_OP_ADD ? a + b : t.aux == GGML_OP_SUB ? a - b : a / b;
            } break;
            case PK_SCALE: v = *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3) * t.f[0] + t.f[1]; break;
            case PK_UNARY: v = pk_unary(t.aux, *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3)); break;
            case PK_SQR:   { const float a = *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3); v = a * a; } break;
            case PK_SQRT:  v = sqrtf(*(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3)); break;
            case PK_CPY:   v = *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3); break;
            case PK_GLU: {
                // src1 == nullptr: src0 rows hold [a | b] (or [b | a] when swapped); else a = src0, b = src1
                float a, b;
                if (t.p[2]) {
                    a = *(const float *) pk_at(t.p[1], t.nb[1], i0, i1, i2, i3);
                    b = *(const float *) pk_at(t.p[2], t.nb[2], i0, i1, i2, i3);
                } else {
                    const int64_t half = t.ne[0][0];
                    const float * row = (const float *) pk_at(t.p[1], t.nb[1], 0, i1, i2, i3);
                    a = row[i0 + (t.i[0] ? half : 0)];
                    b = row[i0 + (t.i[0] ? 0 : half)];
                }
                const float g = t.aux == GGML_GLU_OP_SWIGLU ? a / (1.0f + expf(-a)) : t.aux == GGML_GLU_OP_GEGLU ? pk_unary(GGML_UNARY_OP_GELU, a) : fmaxf(a, 0.0f);
                v = g * b;
            } break;
            default: v = 0.0f; break;
        }
        // dst may be a strided view (CPY into a view): write through the dst strides
        *(float *) pk_at(t.p[0], t.nb[0], i0, i1, i2, i3) = v;
    }
    GGML_UNUSED(dst);
}

// rms_norm over rows of ne00: item = PK_WAVES rows, one wave per row
__device__ __forceinline__ void pk_body_rms_norm(const pk_task & t, int item) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int64_t nrows = t.ne[0][1] * t.ne[0][2] * t.ne[0][3];
    if (t.x[0] == 1) {
        // whole block per row (decode: one or a few rows of thousands of columns; a single wave was 12 us per row)
        __shared__ float s_part[PK_WAVES];
        const int64_t r = item;
        if (r >= nrows) return;
        const uint32_t i1 = r % t.ne[0][1], i2 = (r / t.ne[0][1]) % t.ne[0][2], i3 = r / (t.ne[0][1] * t.ne[0][2]);
        const float * x   = (const float *) pk_at(t.p[1], t.nb[1], 0, i1, i2, i3);
        float       * dst = (float *)       pk_at(t.p[0], t.nb[0], 0, i1, i2, i3);
        const int ncols = t.ne[0][0];
        float s = 0.0f;
        for (int c = threadIdx.x; c < ncols; c += PK_BLOCK) { const float v = x[c]; s += v * v; }
        s = warp_reduce_sum<32>(s);
        if (lane == 0) s_part[wave] = s;
        __syncthreads();
        s = lane < PK_WAVES ? s_part[lane] : 0.0f;
        s = warp_reduce_sum<32>(s);
        const float scale = rsqrtf(s / ncols + t.f[0]);
        for (int c = threadIdx.x; c < ncols; c += PK_BLOCK) dst[c] = x[c] * scale;
        __syncthreads();   // s_part is reused by the next item
        return;
    }
    const int64_t r = (int64_t) item * PK_WAVES + wave;
    if (r >= nrows) return;
    const uint32_t i1 = r % t.ne[0][1], i2 = (r / t.ne[0][1]) % t.ne[0][2], i3 = r / (t.ne[0][1] * t.ne[0][2]);
    const float * x   = (const float *) pk_at(t.p[1], t.nb[1], 0, i1, i2, i3);
    float       * dst = (float *)       pk_at(t.p[0], t.nb[0], 0, i1, i2, i3);
    const int ncols = t.ne[0][0];
    float s = 0.0f;
    for (int c = lane; c < ncols; c += 32) { const float v = x[c]; s += v * v; }
    s = warp_reduce_sum<32>(s);
    const float scale = rsqrtf(s / ncols + t.f[0]);
    for (int c = lane; c < ncols; c += 32) dst[c] = x[c] * scale;
}

// get_rows (f32/f16 source rows -> f32): item = PK_WAVES output rows, one wave per row
__device__ __forceinline__ void pk_body_get_rows(const pk_task & t, int item) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int64_t nrows = t.ne[0][1] * t.ne[0][2] * t.ne[0][3];    // dst rows: ne10 x ne11 x ne12
    const int64_t r = (int64_t) item * PK_WAVES + wave;
    if (r >= nrows) return;
    const uint32_t i10 = r % t.ne[0][1], i11 = (r / t.ne[0][1]) % t.ne[0][2], i12 = r / (t.ne[0][1] * t.ne[0][2]);
    const int32_t i01 = *(const int32_t *) pk_at(t.p[3], t.nb[2], i10, i11, i12, 0);   // ids strides in nb[2]
    const char  * src = pk_at(t.p[1], t.nb[1], 0, i01, i11, i12);
    float * dst = (float *) pk_at(t.p[0], t.nb[0], 0, i10, i11, i12);
    const int ncols = t.ne[0][0];
    if (t.type == GGML_TYPE_F16) {
        for (int c = lane; c < ncols; c += 32) dst[c] = __half2float(((const half *) src)[c]);
    } else {
        for (int c = lane; c < ncols; c += 32) dst[c] = ((const float *) src)[c];
    }
}

// quantize f32 columns to q8_1 for MMVQ: item = PK_QUANT_ITEM elements of the padded [ne10_padded x ncols] layout
__device__ __forceinline__ void pk_body_quant(const pk_task & t, int item) {
    const int64_t ne10 = t.i[0], ne10p = t.i[1], ncols = t.i[2];
    const int64_t e = (int64_t) item * PK_QUANT_ITEM + threadIdx.x;
    if (e >= ne10p * ncols) return;
    const int64_t col = e / ne10p, k = e % ne10p;
    const float xi = k < ne10 ? ((const float *) ((const char *) t.p[1] + col * t.nb[1][1]))[k] : 0.0f;
    float amax = fabsf(xi), sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    block_q8_1 * y = (block_q8_1 *) t.p[0];
    const int64_t ib = e / QK8_1, iqs = e % QK8_1;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}

typedef float (*pk_vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);
static constexpr __device__ pk_vec_dot_t pk_get_vec_dot(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:   return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:   return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:   return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:   return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:   return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q4_K:   return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:   return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:   return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ4_NL: return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS: return vec_dot_iq4_xs_q8_1;
        default:               return nullptr;
    }
}
static constexpr __host__ __device__ int pk_block_bytes(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return sizeof(block_q4_0);   case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
        case GGML_TYPE_Q5_0: return sizeof(block_q5_0);   case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
        case GGML_TYPE_Q8_0: return sizeof(block_q8_0);   case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return sizeof(block_q5_K);   case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
        case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl); case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
        default: return 0;
    }
}

static constexpr __host__ __device__ int pk_get_vdr(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:   return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:   return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:   return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:   return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:   return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:   return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:   return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:   return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL: return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS: return VDR_IQ4_XS_Q8_1_MMVQ;
        default:               return 1;
    }
}

// quantized GEMV against a q8_1 activation: item = PK_WAVES rows, one wave per row (MMVQ's single-warp form)
// RPW rows per wave and KU k-blocks per lane are processed together so that RPW*KU independent weight loads are in
// flight per lane (a wave-per-row loop with one load per lane was latency-bound at ~5 us per row on the APU)
// RPW rows per wave, KU k-blocks per lane in flight, W waves per row (split K, partials reduced through LDS): the
// same three levers MMVQ uses (rows per block, unroll, nwarps); a wave-per-row loop with one load per lane was
// latency-bound at ~5 us per row on the APU
template <ggml_type type, int ncols, int RPW, int KU, int W>
__device__ __forceinline__ void pk_body_mmvq(const pk_task & t, int item) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = pk_get_vdr(type);
    constexpr pk_vec_dot_t vec_dot = pk_get_vec_dot(type);
    constexpr int rows_per_item = (PK_WAVES / W) * RPW;
    __shared__ float s_part[PK_WAVES][RPW][ncols];
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32;
    const int row0 = (item * (PK_WAVES / W) + wave / W) * RPW;
    const int nrows = t.i[1];
    const int blocks_per_row = t.i[0] / qk;
    const int stride_row_x   = t.i[3];                    // blocks per weight row
    const int stride_col_y   = t.i[2];                    // q8_1 blocks per activation column
    const block_q8_1 * y = (const block_q8_1 *) pk_g(t.p[2]);
    const void * x = pk_g(t.p[1]);
    float * dst = (float *) pk_g(t.p[0]);
    constexpr int blocks_per_iter = vdr * 32 / qi;
    float tmp[RPW][ncols];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) tmp[r][j] = 0.0f;
    }
    if (row0 < nrows) {
        const int kqs = vdr * (lane % (qi/vdr));
        // tails are clamped, not branched (a clamped block is read again and multiplied by 0; a clamped row is
        // recomputed and not stored), so every load of the RPW x KU group is issued before any is consumed
        for (int kbx0 = lane / (qi/vdr) + (wave % W) * blocks_per_iter; kbx0 < blocks_per_row; kbx0 += blocks_per_iter * W * KU) {
#pragma unroll
            for (int u = 0; u < KU; ++u) {
                const int kbx_u = kbx0 + u * blocks_per_iter * W;
                const float m   = (KU == 1 || kbx_u < blocks_per_row) ? 1.0f : 0.0f;
                const int kbx   = KU == 1 ? kbx_u : min(kbx_u, blocks_per_row - 1);
                const int kby   = kbx * (qk/QK8_1);
#pragma unroll
                for (int r = 0; r < RPW; ++r) {
                    const int row = RPW == 1 ? row0 : min(row0 + r, nrows - 1);
#pragma unroll
                    for (int j = 0; j < ncols; ++j) tmp[r][j] += m * vec_dot(x, &y[j*stride_col_y + kby], row*stride_row_x + kbx, kqs);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            tmp[r][j] = warp_reduce_sum<32>(tmp[r][j]);
            if (W > 1 && lane == 0) s_part[wave][r][j] = tmp[r][j];
        }
    }
    if (W > 1) {
        __syncthreads();
        if (wave % W == 0 && lane == 0 && row0 < nrows) {
#pragma unroll
            for (int r = 0; r < RPW; ++r) {
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    float v = 0.0f;
#pragma unroll
                    for (int w = 0; w < W; ++w) v += s_part[wave + w][r][j];
                    if (row0 + r < nrows) dst[(int64_t) j * t.ne[0][0] + row0 + r] = v;
                }
            }
        }
        __syncthreads();   // s_part is reused by the next item
    } else if (lane == 0 && row0 < nrows) {
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) if (row0 + r < nrows) dst[(int64_t) j * t.ne[0][0] + row0 + r] = tmp[r][j];   // dst [nrows, ncols] contiguous
        }
    }
}

template <ggml_type type, int ncols>
__device__ __forceinline__ void pk_mmvq_cfg(const pk_task & t, int item) {
#ifdef PK_MMVQ_LEAN
    pk_body_mmvq<type, ncols, 1, 1, 1>(t, item); return;
#endif
    switch (t.x[0] * 256 + t.x[1] * 16 + t.x[2]) {          // rows per wave, k-blocks per lane in flight, waves per row
        case 4*256+1*16+1: pk_body_mmvq<type, ncols, 4, 1, 1>(t, item); break;
        case 2*256+2*16+1: pk_body_mmvq<type, ncols, 2, 2, 1>(t, item); break;
        case 1*256+2*16+1: pk_body_mmvq<type, ncols, 1, 2, 1>(t, item); break;
        case 1*256+4*16+1: pk_body_mmvq<type, ncols, 1, 4, 1>(t, item); break;
        case 1*256+1*16+2: pk_body_mmvq<type, ncols, 1, 1, 2>(t, item); break;
        default:           pk_body_mmvq<type, ncols, 1, 1, 1>(t, item); break;
    }
}

template <ggml_type type>
__device__ __forceinline__ void pk_mmvq_ncols(const pk_task & t, int item) {
    switch (t.ncols) {
        case 1:  pk_mmvq_cfg<type, 1>(t, item); break;
        case 2:  pk_mmvq_cfg<type, 2>(t, item); break;
#ifdef PK_MMVQ_LEAN
        case 3:  pk_body_mmvq<type, 3, 1, 1, 1>(t, item); break;
        default: pk_body_mmvq<type, 4, 1, 1, 1>(t, item); break;
#else
        case 3:  if (t.x[1] >= 2) pk_body_mmvq<type, 3, 1, 2, 1>(t, item); else pk_body_mmvq<type, 3, 1, 1, 1>(t, item); break;
        default: if (t.x[1] >= 2) pk_body_mmvq<type, 4, 1, 2, 1>(t, item); else pk_body_mmvq<type, 4, 1, 1, 1>(t, item); break;
#endif
    }
}

#ifdef PK_NOINLINE_BODIES
#define PK_BODY_ATTR __attribute__((noinline))
#else
#define PK_BODY_ATTR __forceinline__
#endif
__device__ PK_BODY_ATTR void pk_body_mmvq_dispatch(const pk_task & t, int item) {
    switch ((ggml_type) t.type) {
        case GGML_TYPE_Q4_0:   pk_mmvq_ncols<GGML_TYPE_Q4_0>(t, item);   break;
        case GGML_TYPE_Q4_1:   pk_mmvq_ncols<GGML_TYPE_Q4_1>(t, item);   break;
        case GGML_TYPE_Q5_0:   pk_mmvq_ncols<GGML_TYPE_Q5_0>(t, item);   break;
        case GGML_TYPE_Q5_1:   pk_mmvq_ncols<GGML_TYPE_Q5_1>(t, item);   break;
        case GGML_TYPE_Q8_0:   pk_mmvq_ncols<GGML_TYPE_Q8_0>(t, item);   break;
        case GGML_TYPE_Q4_K:   pk_mmvq_ncols<GGML_TYPE_Q4_K>(t, item);   break;
        case GGML_TYPE_Q5_K:   pk_mmvq_ncols<GGML_TYPE_Q5_K>(t, item);   break;
        case GGML_TYPE_Q6_K:   pk_mmvq_ncols<GGML_TYPE_Q6_K>(t, item);   break;
        case GGML_TYPE_IQ4_NL: pk_mmvq_ncols<GGML_TYPE_IQ4_NL>(t, item); break;
        case GGML_TYPE_IQ4_XS: pk_mmvq_ncols<GGML_TYPE_IQ4_XS>(t, item); break;
        default: break;
    }
}

__device__ __forceinline__ void pk_run_item(const pk_task & t, int item) {
    switch (t.op) {
        case PK_QUANT:    pk_body_quant(t, item);         break;
        case PK_MMVQ:     pk_body_mmvq_dispatch(t, item); break;
#ifndef PK_ONLY_MMVQ
        case PK_RMS_NORM: pk_body_rms_norm(t, item);      break;
        case PK_GET_ROWS: pk_body_get_rows(t, item);      break;
        default:          pk_body_ew(t, item);            break;
#else
        default: break;
#endif
    }
}

// ---- the resident scheduler ---------------------------------------------------------------------------------------
// one 1024-thread block per WGP; all loop conditions are block-uniform (thread 0 decides, LDS broadcast)
__global__ void __launch_bounds__(PK_BLOCK) pk_run(const pk_region_dev R) {
    __shared__ int s_flag;
    __shared__ long long s_epoch;
    if (threadIdx.x == 0) {
        const long long ticket = __hip_atomic_fetch_add(R.arrivals, 1ll, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        s_epoch = ticket / gridDim.x + 1;
    }
    __syncthreads();
    const long long epoch = s_epoch;
    int skip = 0;   // set after a bounded wait expires: stop waiting and computing, but keep retiring so the
                    // monotonic counters stay consistent and the next launch of the region is healthy again
#ifndef PK_TASK_IN_GLOBAL
#define PK_TASK_IN_LDS 1
#endif
#ifdef PK_TASK_IN_LDS
    __shared__ pk_task s_task;
#endif
    for (int ti = 0; ti < R.n_tasks; ++ti) {
#ifdef PK_TASK_IN_LDS
        __syncthreads();   // previous item loop may still read s_task
        if (threadIdx.x < sizeof(pk_task) / 4) { ((int *) &s_task)[threadIdx.x] = ((const int *) &R.tasks[ti])[threadIdx.x]; }
        __syncthreads();
        const pk_task & t = s_task;
#else
        const pk_task & t = R.tasks[ti];
#endif
        pk_state * st = &R.state[ti];
        const int n_items = __builtin_amdgcn_readfirstlane(t.n_items);
        const int n_deps  = __builtin_amdgcn_readfirstlane(t.n_deps);
        if (threadIdx.x == 0) {
            long spins = 0; int f = 1;
            const long long need = (long long) n_deps * epoch;
            if (R.trace && blockIdx.x == 0) { st->pad[4] = (long long) wall_clock64(); }
            if (!skip) {
                while (pk_ld_relaxed(&st->deps_done) < need) {
                    __builtin_amdgcn_s_sleep(PK_SLEEP);
                    if (++spins > 4000000L) { f = 0; break; }      // ~1 s: give up rather than hang the GPU
                }
            }
            pk_fence_acquire();
            s_flag = f;
            if (!f) { R.state[0].pad[0] = ti + 1; }                  // failure marker for the host
            if (R.trace && blockIdx.x == 0) { st->pad[5] = (long long) wall_clock64(); }
        }
        __syncthreads();
        if (__builtin_amdgcn_readfirstlane(s_flag) == 0) { skip = 1; }
        int mine = 0;
        for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
            if (!skip) { pk_run_item(t, item); }
            ++mine;
        }
        __syncthreads();
        if (R.trace && blockIdx.x == 0 && threadIdx.x == 0) { st->pad[6] = (long long) wall_clock64(); st->pad[7] = mine; }
        if (threadIdx.x == 0 && mine > 0) {
            const long long d = pk_add_release(&st->done, mine) + mine;
            if (d == (long long) n_items * epoch) {
                const int f = t.dep_first, c = t.dep_count;
                for (int k = 0; k < c; ++k) pk_add_release(&R.state[R.dependents[f + k]].deps_done, 1);
            }
        }
    }
}

// ---- host: classify nodes, compile a region, cache it, launch ----------------------------------------------------

static bool pk_f32_4d(const ggml_tensor * t) { return t->type == GGML_TYPE_F32; }

static int pk_blocks_per_iter(ggml_type type) {   // host mirror of vdr * 32 / qi
    switch (type) {
        case GGML_TYPE_Q4_0: return pk_get_vdr(GGML_TYPE_Q4_0) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q4_0>::qi;
        case GGML_TYPE_Q4_1: return pk_get_vdr(GGML_TYPE_Q4_1) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q4_1>::qi;
        case GGML_TYPE_Q5_0: return pk_get_vdr(GGML_TYPE_Q5_0) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q5_0>::qi;
        case GGML_TYPE_Q5_1: return pk_get_vdr(GGML_TYPE_Q5_1) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q5_1>::qi;
        case GGML_TYPE_Q8_0: return pk_get_vdr(GGML_TYPE_Q8_0) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
        case GGML_TYPE_Q4_K: return pk_get_vdr(GGML_TYPE_Q4_K) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q4_K>::qi;
        case GGML_TYPE_Q5_K: return pk_get_vdr(GGML_TYPE_Q5_K) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q5_K>::qi;
        case GGML_TYPE_Q6_K: return pk_get_vdr(GGML_TYPE_Q6_K) * 32 / ggml_cuda_type_traits<GGML_TYPE_Q6_K>::qi;
        case GGML_TYPE_IQ4_NL: return pk_get_vdr(GGML_TYPE_IQ4_NL) * 32 / ggml_cuda_type_traits<GGML_TYPE_IQ4_NL>::qi;
        case GGML_TYPE_IQ4_XS: return pk_get_vdr(GGML_TYPE_IQ4_XS) * 32 / ggml_cuda_type_traits<GGML_TYPE_IQ4_XS>::qi;
        default: return 8;
    }
}

static bool pk_mmvq_type_ok(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

// which persistent op implements this node, or -1
static int pk_classify(const ggml_tensor * node) {
    const ggml_tensor * s0 = node->src[0];
    const ggml_tensor * s1 = node->src[1];
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            if (!pk_mmvq_type_ok(s0->type) || s1->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return -1;
            if (s0->ne[2] != 1 || s0->ne[3] != 1 || s1->ne[2] != 1 || s1->ne[3] != 1) return -1;
            if (s1->ne[1] < 1 || s1->ne[1] > 4 || s1->nb[0] != sizeof(float) || !ggml_is_contiguous(node)) return -1;
            if (s0->ne[0] % QK8_1 != 0 || !ggml_is_contiguous(s0)) return -1;
            return PK_MMVQ;
        case GGML_OP_RMS_NORM:
            if (!pk_f32_4d(node) || !pk_f32_4d(s0) || s0->nb[0] != sizeof(float) || node->nb[0] != sizeof(float)) return -1;
            return PK_RMS_NORM;
        case GGML_OP_MUL: case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_DIV:
            if (!pk_f32_4d(node) || !pk_f32_4d(s0) || !pk_f32_4d(s1) || !ggml_are_same_shape(s0, node) || !ggml_can_repeat(s1, node)) return -1;
            return PK_BIN;
        case GGML_OP_SCALE: return (pk_f32_4d(node) && pk_f32_4d(s0)) ? PK_SCALE : -1;
        case GGML_OP_UNARY:
            if (!pk_f32_4d(node) || !pk_f32_4d(s0)) return -1;
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_SIGMOID: case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_GELU: case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_RELU: case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_NEG: case GGML_UNARY_OP_ABS: case GGML_UNARY_OP_TANH:
                    return PK_UNARY;
                default: return -1;
            }
        case GGML_OP_SQR:  return (pk_f32_4d(node) && pk_f32_4d(s0)) ? PK_SQR : -1;
        case GGML_OP_SQRT: return (pk_f32_4d(node) && pk_f32_4d(s0)) ? PK_SQRT : -1;
        case GGML_OP_CPY: case GGML_OP_CONT: case GGML_OP_DUP:
            if (!pk_f32_4d(node) || !pk_f32_4d(s0) || ggml_nelements(node) != ggml_nelements(s0)) return -1;
            if (!ggml_are_same_shape(s0, node)) return -1;
            return PK_CPY;
        case GGML_OP_GLU: {
            if (!pk_f32_4d(node) || !pk_f32_4d(s0) || (s1 && !pk_f32_4d(s1))) return -1;
            const ggml_glu_op g = ggml_get_glu_op(node);
            if (g != GGML_GLU_OP_SWIGLU && g != GGML_GLU_OP_GEGLU && g != GGML_GLU_OP_REGLU) return -1;
            if (s1 && !ggml_are_same_shape(s1, node)) return -1;
            return PK_GLU;
        }
        case GGML_OP_GET_ROWS:
            if (node->type != GGML_TYPE_F32 || (s0->type != GGML_TYPE_F32 && s0->type != GGML_TYPE_F16) || s1->type != GGML_TYPE_I32) return -1;
            if (s0->nb[0] != ggml_type_size(s0->type)) return -1;
            return PK_GET_ROWS;
        default:
            return -1;
    }
}

struct pk_range { const char * lo; const char * hi; };
static pk_range pk_range_of(const ggml_tensor * t) { return { (const char *) t->data, (const char *) t->data + ggml_nbytes(t) }; }
static bool pk_overlap(const pk_range & a, const pk_range & b) { return a.lo < b.hi && b.lo < a.hi; }

struct pk_task_host {
    pk_task t;
    std::vector<pk_range> reads;
    pk_range write;
};

struct pk_region_cache {
    std::vector<pk_task> host;
    std::vector<int>     deps_host;
    pk_task            * d_tasks = nullptr;
    pk_state           * d_state = nullptr;
    int                * d_deps  = nullptr;
    size_t               d_tasks_cap = 0, d_deps_cap = 0, d_state_cap = 0;
    char               * scratch = nullptr;
    size_t               scratch_size = 0;
    int                  n_nodes = 0;
};

struct pk_ctx_state {
    std::map<std::pair<const ggml_tensor *, int>, pk_region_cache> regions;   // keyed by (first node, node index)
    char * arena = nullptr;
    size_t arena_size = 0, arena_used = 0;
};

static pk_ctx_state & pk_ctx(ggml_backend_cuda_context & ctx) {
    static std::unordered_map<ggml_backend_cuda_context *, pk_ctx_state> m;
    return m[&ctx];
}

static void pk_fill_tensor(pk_task & t, int slot, const ggml_tensor * x) {
    for (int d = 0; d < 4; ++d) { t.ne[slot][d] = x->ne[d]; t.nb[slot][d] = x->nb[d]; }
}

int ggml_cuda_persist_region(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i0) {
    static const int enable  = getenv("GGML_CUDA_PERSIST")     ? atoi(getenv("GGML_CUDA_PERSIST"))     : 0;
    static const int min_len = getenv("GGML_CUDA_PERSIST_MIN") ? atoi(getenv("GGML_CUDA_PERSIST_MIN")) : 2;
    if (!enable) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) {
        return 0;
    }

    // 1. scan the run of supported nodes
    std::vector<int> nodes;
    int j = i0;
    for (; j < cgraph->n_nodes && nodes.size() < 512; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
            continue;
        }
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        static const int ops_mask = getenv("GGML_CUDA_PERSIST_OPS") ? (int) strtol(getenv("GGML_CUDA_PERSIST_OPS"), nullptr, 0) : 0x7ff;   // debug: bitmask of pk_op classes
        static const int max_len  = getenv("GGML_CUDA_PERSIST_MAX") ? atoi(getenv("GGML_CUDA_PERSIST_MAX")) : 512;
        const int cls = pk_classify(n);
        if (cls < 0 || !(ops_mask & (1 << cls)) || (cls == PK_MMVQ && !(ops_mask & (1 << PK_QUANT))) || (int) nodes.size() >= max_len) {
            break;
        }
        nodes.push_back(j);
    }
    if ((int) nodes.size() < min_len) {
        return 0;
    }
    const int consumed = nodes.empty() ? 0 : nodes.back() - i0 + 1;

    // 2. build tasks
    pk_ctx_state & S = pk_ctx(ctx);
    if (S.arena == nullptr) {
        S.arena_size = (size_t) 256 << 20;
        ggml_cuda_set_device(ctx.device);
        if (cudaMalloc((void **) &S.arena, S.arena_size) != cudaSuccess) { S.arena = nullptr; S.arena_size = 0; return 0; }
    }
    pk_region_cache & R = S.regions[std::make_pair((const ggml_tensor *) cgraph->nodes[i0], i0)];
    size_t scratch_need = 0;
    std::vector<pk_task_host> tasks;
    std::unordered_map<const void *, int> quant_of;   // src1 data -> quant task index (shared across GEMVs)
    for (int idx : nodes) {
        const ggml_tensor * n = cgraph->nodes[idx];
        const ggml_tensor * s0 = n->src[0];
        const ggml_tensor * s1 = n->src[1];
        const int op = pk_classify(n);
        pk_task_host h{}; pk_task & t = h.t;
        t.op = op; t.p[0] = n->data; pk_fill_tensor(t, 0, n);
        if (s0) { t.p[1] = s0->data; pk_fill_tensor(t, 1, s0); }
        if (s1) { t.p[2] = s1->data; pk_fill_tensor(t, 2, s1); }
        h.write = pk_range_of(n);
        if (s0) h.reads.push_back(pk_range_of(s0));
        if (s1) h.reads.push_back(pk_range_of(s1));
        const int64_t nel = ggml_nelements(n);
        switch (op) {
            case PK_MMVQ: {
                const int64_t ne10 = s1->ne[0], ncols = s1->ne[1];
                const int64_t ne10p = GGML_PAD(ne10, MATRIX_ROW_PADDING);
                const size_t  qbytes = (size_t) (ne10p / QK8_1) * ncols * sizeof(block_q8_1);
                int qi;
                auto it = quant_of.find(s1->data);
                bool fresh = it != quant_of.end() && tasks[it->second].t.i[0] == ne10 && tasks[it->second].t.i[2] == ncols;
                // ggml-alloc recycles addresses inside one graph: the q8_1 copy is only reusable if nothing written
                // since the quant task overlaps src1 (the same pointer may hold a different tensor by now)
                if (fresh) {
                    const pk_range r1 = pk_range_of(s1);
                    for (size_t a = it->second + 1; a < tasks.size() && fresh; ++a) { if (tasks[a].t.op != PK_QUANT && pk_overlap(tasks[a].write, r1)) fresh = false; }
                }
                if (fresh) {
                    qi = it->second;
                } else {
                    pk_task_host q{}; pk_task & qt = q.t;
                    qt.op = PK_QUANT; qt.i[0] = ne10; qt.i[1] = ne10p; qt.i[2] = ncols;
                    qt.p[1] = s1->data; pk_fill_tensor(qt, 1, s1);
                    const size_t off = (scratch_need + 255) & ~(size_t) 255;
                    qt.p[0] = (const void *) (uintptr_t) off;           // resolved against the scratch base below
                    scratch_need = off + qbytes;
                    qt.n_items = (int) ((ne10p * ncols + PK_QUANT_ITEM - 1) / PK_QUANT_ITEM);
                    q.reads.push_back(pk_range_of(s1));
                    q.write = { (const char *) (uintptr_t) off, (const char *) (uintptr_t) (off + qbytes) };   // scratch-relative
                    qt.aux = 1;   // marks a scratch-relative dst
                    tasks.push_back(q);
                    qi = (int) tasks.size() - 1;
                    quant_of[s1->data] = qi;
                }
                t.type  = s0->type;
                t.ncols = (int) ncols;
                t.i[0]  = (int) s0->ne[0];                       // K
                t.i[1]  = (int) s0->ne[1];                       // rows
                t.i[2]  = (int) (ne10p / QK8_1);                 // q8 blocks per activation column
                t.i[3]  = (int) (s0->nb[1] / ggml_type_size(s0->type));   // weight row stride in blocks
                t.p[2]  = tasks[qi].t.p[0];                      // scratch-relative, resolved below
                t.aux   = 2;                                     // marks a scratch-relative src1
                {
                    // rows per wave: short rows (one k-iteration per lane) get 4 rows in flight, medium 2, long rows
                    // rely on the 2-deep k unroll instead; keep enough items to cover the grid
                    const int bpi = pk_blocks_per_iter(s0->type);
                    const int iters = (int) ((s0->ne[0] / ggml_blck_size(s0->type) + bpi - 1) / bpi);
                    // waves per row: enough that each wave sees at most ~2 k-iterations (MMVQ's nwarps), rows per
                    // wave only for one-iteration rows
                    int w   = 1;
                    int rpw = iters <= 1 ? 4 : 1;
                    int ku  = iters <= 1 ? 1 : 2;
                    static const char * cfg = getenv("GGML_CUDA_PERSIST_MMVQ");   // debug override "rpw,ku,w"
                    if (cfg) { int a = 0, b = 0, c = 0; if (sscanf(cfg, "%d,%d,%d", &a, &b, &c) == 3) { rpw = a; ku = b; w = c; } }
                    if (ncols >= 3) { rpw = 1; w = 1; if (ku > 2) ku = 2; }
                    if (w > 2) w = 2;
                    while (rpw > 1 && s0->ne[1] / (PK_WAVES * rpw) < 20) rpw /= 2;
                    const int rows_per_item = (PK_WAVES / w) * rpw;
                    t.x[0] = rpw; t.x[1] = ku; t.x[2] = w;
                    t.n_items = (int) ((s0->ne[1] + rows_per_item - 1) / rows_per_item);
                }
                h.reads.clear(); h.reads.push_back(pk_range_of(s0)); h.reads.push_back(tasks[qi].write);
            } break;
            case PK_RMS_NORM: {
                t.f[0] = ggml_get_op_params_f32(n, 0);
                const int64_t rows = n->ne[1] * n->ne[2] * n->ne[3];
                if (rows <= 64 && n->ne[0] >= 512) { t.x[0] = 1; t.n_items = (int) rows; }   // block per row
                else { t.x[0] = 0; t.n_items = (int) ((rows + PK_WAVES - 1) / PK_WAVES); }
            } break;
            case PK_GET_ROWS:
                t.type = s0->type;
                t.p[3] = s1->data; for (int d = 0; d < 4; ++d) { t.nb[2][d] = s1->nb[d]; }
                t.n_items = (int) ((n->ne[1] * n->ne[2] * n->ne[3] + PK_WAVES - 1) / PK_WAVES);
                break;
            case PK_BIN:   t.aux = n->op; t.n_items = (int) ((nel + PK_EW_ITEM - 1) / PK_EW_ITEM); break;
            case PK_SCALE: t.f[0] = ggml_get_op_params_f32(n, 0); t.f[1] = ggml_get_op_params_f32(n, 1); t.n_items = (int) ((nel + PK_EW_ITEM - 1) / PK_EW_ITEM); break;
            case PK_UNARY: t.aux = ggml_get_unary_op(n); t.n_items = (int) ((nel + PK_EW_ITEM - 1) / PK_EW_ITEM); break;
            case PK_GLU:   t.aux = ggml_get_glu_op(n); t.i[0] = ggml_get_op_params_i32(n, 1); t.n_items = (int) ((nel + PK_EW_ITEM - 1) / PK_EW_ITEM); break;
            default:       t.n_items = (int) ((nel + PK_EW_ITEM - 1) / PK_EW_ITEM); break;   // SQR, SQRT, CPY
        }
        tasks.push_back(h);
    }

    // 3. scratch: keep the region's slice across recompiles when it still fits
    if (R.scratch == nullptr || R.scratch_size < scratch_need) {
        const size_t off = (S.arena_used + 255) & ~(size_t) 255;
        if (off + scratch_need > S.arena_size) { return 0; }
        R.scratch = S.arena + off; R.scratch_size = scratch_need; S.arena_used = off + scratch_need;
    }
    for (auto & h : tasks) {
        if (h.t.op == PK_QUANT && h.t.aux == 1) { h.t.p[0] = R.scratch + (uintptr_t) h.t.p[0]; h.write = { R.scratch + (uintptr_t) h.write.lo, R.scratch + (uintptr_t) h.write.hi }; }
        if (h.t.op == PK_MMVQ && h.t.aux == 2) { h.t.p[2] = R.scratch + (uintptr_t) h.t.p[2]; h.reads[1] = { R.scratch + (uintptr_t) h.reads[1].lo, R.scratch + (uintptr_t) h.reads[1].hi }; }
    }

    // 4. dependencies: data flow and memory hazards (RAW, WAW, WAR) against every earlier task
    const int nt = (int) tasks.size();
    std::vector<std::vector<int>> dependents(nt);
    // reach[b] = every task b transitively depends on; a hazard edge a -> b is kept only if a is not already
    // reached through a later dependency (transitive reduction: the perf harness's chain of 500 identical nodes
    // would otherwise release 500 dependents per task)
    std::vector<std::vector<uint64_t>> reach(nt, std::vector<uint64_t>((nt + 63) / 64, 0));
    static const int serial = getenv("GGML_CUDA_PERSIST_SERIAL") ? atoi(getenv("GGML_CUDA_PERSIST_SERIAL")) : 0;   // debug: chain every task
    for (int b = 0; b < nt; ++b) {
        int nd = 0;
        for (int a = b - 1; a >= 0; --a) {
            bool dep = serial ? (a == b - 1) : pk_overlap(tasks[a].write, tasks[b].write);
            for (const pk_range & r : tasks[b].reads) dep = dep || pk_overlap(tasks[a].write, r);
            for (const pk_range & r : tasks[a].reads) dep = dep || pk_overlap(r, tasks[b].write);
            if (!dep) continue;
            if (reach[b][a / 64] & (1ull << (a % 64))) continue;      // already ordered after a through another edge
            dependents[a].push_back(b); ++nd;
            for (size_t w = 0; w < reach[b].size(); ++w) reach[b][w] |= reach[a][w];
            reach[b][a / 64] |= 1ull << (a % 64);
        }
        tasks[b].t.n_deps = nd;
    }
    std::vector<pk_task> host(nt);
    std::vector<int> csr;
    for (int a = 0; a < nt; ++a) {
        tasks[a].t.dep_first = (int) csr.size();
        tasks[a].t.dep_count = (int) dependents[a].size();
        for (int d : dependents[a]) csr.push_back(d);
        host[a] = tasks[a].t;
    }
    if (csr.empty()) csr.push_back(0);

    static const int dbg_compile = getenv("GGML_CUDA_PERSIST_DEBUG") ? atoi(getenv("GGML_CUDA_PERSIST_DEBUG")) : 0;
    if (dbg_compile >= 2) {
        std::string line;
        for (int a = 0; a < nt; ++a) { char b[96]; snprintf(b, sizeof(b), " %d[%lldx%lld,i%d,d%d]", host[a].op, (long long) host[a].ne[0][0], (long long) host[a].ne[0][1], host[a].n_items, host[a].n_deps); line += b; }
        GGML_LOG_WARN("persist: region @%d (%d nodes, %d tasks):%s\n", i0, consumed, nt, line.c_str());
    }
    // 5. cache: reuse the device copy when nothing changed; a changed region cannot be uploaded during graph capture
    cudaStream_t stream = ctx.stream();
    const bool same = R.host.size() == host.size() && R.deps_host.size() == csr.size() &&
                      memcmp(R.host.data(), host.data(), host.size() * sizeof(pk_task)) == 0 &&
                      memcmp(R.deps_host.data(), csr.data(), csr.size() * sizeof(int)) == 0;
    if (!same) {
        hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
        (void) hipStreamIsCapturing(stream, &cap);
        if (cap != hipStreamCaptureStatusNone) { return 0; }
        if (dbg_compile >= 2) { GGML_LOG_WARN("persist: RECOMPILE region @%d (%d tasks, was %zu)\n", i0, nt, R.host.size()); }
        ggml_cuda_set_device(ctx.device);
        // the uploads below are synchronous on the null stream; ggml's streams are non-blocking, so a previous launch of
        // this region could still be reading the old task list -> drain the stream first (recompiles are rare)
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (R.d_tasks_cap < host.size()) { if (R.d_tasks) { (void) cudaFree(R.d_tasks); } CUDA_CHECK(cudaMalloc((void **) &R.d_tasks, host.size() * sizeof(pk_task))); R.d_tasks_cap = host.size(); }
        if (R.d_deps_cap  < csr.size())  { if (R.d_deps)  { (void) cudaFree(R.d_deps); }  CUDA_CHECK(cudaMalloc((void **) &R.d_deps,  csr.size() * sizeof(int)));      R.d_deps_cap  = csr.size(); }
        if (R.d_state_cap < host.size()) { if (R.d_state) { (void) cudaFree(R.d_state); } CUDA_CHECK(cudaMalloc((void **) &R.d_state, (host.size() + 1) * sizeof(pk_state))); R.d_state_cap = host.size(); }
        CUDA_CHECK(cudaMemset(R.d_state, 0, (host.size() + 1) * sizeof(pk_state)));   // a recompile restarts the epochs
        CUDA_CHECK(cudaMemcpy(R.d_tasks, host.data(), host.size() * sizeof(pk_task), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(R.d_deps,  csr.data(),  csr.size()  * sizeof(int),     cudaMemcpyHostToDevice));
        R.host = host; R.deps_host = csr; R.n_nodes = consumed;
    }

    // 6. launch: reset the counters, one block per WGP
    // every block must be resident at once (blocks wait on each other): one block per WGP, capped by what the
    // occupancy calculator says fits (pk_run holds ~125 VGPRs, so a second 1024-thread block per WGP does not fit)
    static const int grid_env = getenv("GGML_CUDA_PERSIST_GRID") ? atoi(getenv("GGML_CUDA_PERSIST_GRID")) : 0;
    static int grid_dev[GGML_CUDA_MAX_DEVICES] = { 0 };
    if (grid_dev[ctx.device] == 0) {
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, pk_run, PK_BLOCK, 0));
        const int nsm = ggml_cuda_info().devices[ctx.device].nsm;
        grid_dev[ctx.device] = std::max(1, std::min(nsm, per_sm * nsm));
        if (getenv("GGML_CUDA_PERSIST_DEBUG")) {
            GGML_LOG_INFO("persist: device %d: %d SMs, %d resident blocks of %d per SM -> grid %d\n", ctx.device, nsm, per_sm, PK_BLOCK, grid_dev[ctx.device]);
        }
    }
    const int grid = grid_env > 0 ? grid_env : grid_dev[ctx.device];
    static const int trace = getenv("GGML_CUDA_PERSIST_TRACE") ? atoi(getenv("GGML_CUDA_PERSIST_TRACE")) : 0;
    pk_region_dev D = { R.d_tasks, R.d_state, R.d_deps, &R.d_state[nt].done, nt, trace };
    static const int verify = getenv("GGML_CUDA_PERSIST_VERIFY") ? atoi(getenv("GGML_CUDA_PERSIST_VERIFY")) : 0;
    hipStreamCaptureStatus cap_pre = hipStreamCaptureStatusNone;
    (void) hipStreamIsCapturing(stream, &cap_pre);
    const bool verifying = verify && cap_pre == hipStreamCaptureStatusNone;
    // verifier: every source that some node of the region overwrites (in-place chains) is saved before the region
    // runs, so the reference re-run below starts from the same inputs
    std::vector<std::pair<const ggml_tensor *, std::vector<char>>> saved;
    if (verifying) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int idx : nodes) {
            const ggml_tensor * n = cgraph->nodes[idx];
            for (int sI = 0; sI < GGML_MAX_SRC; ++sI) {
                const ggml_tensor * sx = n->src[sI];
                if (!sx || !sx->data) continue;
                bool hit = false, dup = false;
                for (int idx2 : nodes) { const ggml_tensor * w = cgraph->nodes[idx2]; hit = hit || pk_overlap({ (const char *) sx->data, (const char *) sx->data + ggml_nbytes(sx) }, { (const char *) w->data, (const char *) w->data + ggml_nbytes(w) }); }
                for (auto & sv : saved) dup = dup || sv.first->data == sx->data;
                if (!hit || dup) continue;
                std::vector<char> buf(ggml_nbytes(sx));
                CUDA_CHECK(cudaMemcpy(buf.data(), sx->data, buf.size(), cudaMemcpyDeviceToHost));
                saved.emplace_back(sx, std::move(buf));
            }
        }
    }
    pk_run<<<grid, PK_BLOCK, 0, stream>>>(D);
    CUDA_CHECK(cudaGetLastError());
    static const int dbg = getenv("GGML_CUDA_PERSIST_DEBUG") ? atoi(getenv("GGML_CUDA_PERSIST_DEBUG")) : 0;
    hipStreamCaptureStatus cap_now = hipStreamCaptureStatusNone;
    (void) hipStreamIsCapturing(stream, &cap_now);
    if (verifying) {
        // re-run every node of the region on the normal path and compare its output with what the region wrote;
        // the normal path's result stays in place, so the run continues correctly after a mismatch is reported
        static int n_reports = 0;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<std::vector<char>> got(nodes.size());
        for (size_t k = 0; k < nodes.size(); ++k) {
            const ggml_tensor * n = cgraph->nodes[nodes[k]];
            got[k].resize(ggml_nbytes(n));
            CUDA_CHECK(cudaMemcpy(got[k].data(), n->data, got[k].size(), cudaMemcpyDeviceToHost));
        }
        for (auto & sv : saved) { CUDA_CHECK(cudaMemcpy((void *) sv.first->data, sv.second.data(), sv.second.size(), cudaMemcpyHostToDevice)); }
        for (size_t k = 0; k < nodes.size(); ++k) {
            ggml_tensor * n = cgraph->nodes[nodes[k]];
            (void) ggml_cuda_compute_forward_node(ctx, n);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<char> ref(ggml_nbytes(n));
            CUDA_CHECK(cudaMemcpy(ref.data(), n->data, ref.size(), cudaMemcpyDeviceToHost));
            if (n->type != GGML_TYPE_F32) continue;
            bool overwritten = false;   // a later node of the region writes this buffer: only the last writer is compared
            for (size_t k2 = k + 1; k2 < nodes.size(); ++k2) { const ggml_tensor * w = cgraph->nodes[nodes[k2]]; overwritten = overwritten || pk_overlap({ (const char *) n->data, (const char *) n->data + ggml_nbytes(n) }, { (const char *) w->data, (const char *) w->data + ggml_nbytes(w) }); }
            if (overwritten) continue;
            const float * a = (const float *) got[k].data(); const float * b = (const float *) ref.data();
            const int64_t ne = ggml_nelements(n);
            double worst = 0; int64_t wi = -1; int64_t nbad = 0;
            for (int64_t e = 0; e < ne; ++e) {
                // element e of a contiguous-by-nbytes view: walk in memory order using the tensor strides
                const int64_t i0 = e % n->ne[0], i1 = (e / n->ne[0]) % n->ne[1], i2 = (e / (n->ne[0]*n->ne[1])) % n->ne[2], i3 = e / (n->ne[0]*n->ne[1]*n->ne[2]);
                const size_t off = i0*n->nb[0] + i1*n->nb[1] + i2*n->nb[2] + i3*n->nb[3];
                if (off + sizeof(float) > got[k].size()) continue;
                const float x = *(const float *) (got[k].data() + off), y = *(const float *) (ref.data() + off);
                const double d = fabs((double) x - y), tol = 1e-3 * fabs((double) y) + 1e-4;
                if (!(d <= tol) ) { ++nbad; if (d > worst || wi < 0) { worst = d; wi = e; } }
            }
            if (nbad > 0 && n_reports < 20) {
                ++n_reports;
                const ggml_tensor * s0 = n->src[0]; const ggml_tensor * s1 = n->src[1];
                const bool inplace = (s0 && n->data == s0->data) || (s1 && n->data == s1->data);
                GGML_LOG_ERROR("persist VERIFY: region @%d node %d/%zu %s (%s) %s: %lld of %lld elements differ, worst %.3g at %lld; dst ne %lldx%lldx%lldx%lld nb %zu,%zu,%zu,%zu%s\n",
                    i0, (int) k, nodes.size(), ggml_op_name(n->op), n->name, ggml_op_desc(n), (long long) nbad, (long long) ne, worst, (long long) wi,
                    (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3], n->nb[0], n->nb[1], n->nb[2], n->nb[3], inplace ? " IN-PLACE" : "");
                if (s0) GGML_LOG_ERROR("   src0 %s %s ne %lldx%lldx%lldx%lld nb %zu,%zu,%zu,%zu contiguous %d\n", s0->name, ggml_type_name(s0->type), (long long) s0->ne[0], (long long) s0->ne[1], (long long) s0->ne[2], (long long) s0->ne[3], s0->nb[0], s0->nb[1], s0->nb[2], s0->nb[3], (int) ggml_is_contiguous(s0));
                if (s1) GGML_LOG_ERROR("   src1 %s %s ne %lldx%lldx%lldx%lld nb %zu,%zu,%zu,%zu contiguous %d\n", s1->name, ggml_type_name(s1->type), (long long) s1->ne[0], (long long) s1->ne[1], (long long) s1->ne[2], (long long) s1->ne[3], s1->nb[0], s1->nb[1], s1->nb[2], s1->nb[3], (int) ggml_is_contiguous(s1));
                if (wi >= 0) { const size_t off = (wi % n->ne[0])*n->nb[0] + ((wi / n->ne[0]) % n->ne[1])*n->nb[1] + ((wi / (n->ne[0]*n->ne[1])) % n->ne[2])*n->nb[2] + (wi / (n->ne[0]*n->ne[1]*n->ne[2]))*n->nb[3];
                    GGML_LOG_ERROR("   got %g ref %g (task %zu op %d)\n", *(const float *) (got[k].data() + off), *(const float *) (ref.data() + off), k, pk_classify(n)); }
            }
        }
    }
    if (trace && cap_now == hipStreamCaptureStatusNone) {
        static int n_traced = 0;
        static std::map<std::pair<const ggml_tensor *, int>, int> seen;
        int & cnt = seen[std::make_pair((const ggml_tensor *) cgraph->nodes[i0], i0)];
        ++cnt;
        if (cnt == trace && n_traced < 4 && nt >= 6) {     // the trace-th execution of a region (skips the cold first one)
            ++n_traced;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<pk_state> st(nt);
            CUDA_CHECK(cudaMemcpy(st.data(), R.d_state, nt * sizeof(pk_state), cudaMemcpyDeviceToHost));
            int khz = 0; CUDA_CHECK(cudaDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, ctx.device));
            const double us = 1000.0 / (khz > 0 ? khz : 100000);
            const long long t0 = st[0].pad[4];
            double wait_sum = 0, work_sum = 0;
            GGML_LOG_WARN("persist TRACE: region @%d, %d tasks, block 0 timeline (wall clock %d kHz)\n", i0, nt, khz);
            for (int a = 0; a < nt; ++a) {
                const double w = (st[a].pad[5] - st[a].pad[4]) * us, k = (st[a].pad[6] - st[a].pad[5]) * us;
                wait_sum += w; work_sum += k;
                GGML_LOG_WARN("   %3d op %2d n_items %4d n_deps %d mine %lld  t %8.1f  wait %6.1f  work %6.1f us  ne %lldx%lld\n", a, R.host[a].op, R.host[a].n_items, R.host[a].n_deps, st[a].pad[7], (st[a].pad[4] - t0) * us, w, k, (long long) R.host[a].ne[0][0], (long long) R.host[a].ne[0][1]);
            }
            GGML_LOG_WARN("   total %.1f us: wait %.1f, work %.1f (block 0)\n", (st[nt-1].pad[6] - t0) * us, wait_sum, work_sum);
        }
    }
    if (dbg && cap_now == hipStreamCaptureStatusNone) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<pk_state> st(nt);
        CUDA_CHECK(cudaMemcpy(st.data(), R.d_state, nt * sizeof(pk_state), cudaMemcpyDeviceToHost));
        if (st[0].pad[0] != 0) {
            const int bad = st[0].pad[0] - 1;
            GGML_LOG_ERROR("persist: region of %d tasks timed out at task %d (op %d, n_items %d, n_deps %d, done %lld, deps_done %lld)\n",
                nt, bad, host[bad].op, host[bad].n_items, host[bad].n_deps, st[bad].done, st[bad].deps_done);
            for (int a = 0; a < nt && a < 12; ++a) {
                GGML_LOG_ERROR("   task %2d op %d n_items %4d n_deps %d deps[%d..+%d] done %4lld deps_done %lld\n", a, host[a].op, host[a].n_items, host[a].n_deps, host[a].dep_first, host[a].dep_count, st[a].done, st[a].deps_done);
            }
        }
    }
    return consumed;
}

// called after every graph compute (direct or replayed) when GGML_CUDA_PERSIST_DEBUG=1: reports the first region whose
// last launch hit the bounded wait, with the counters around the stuck task; reported once per region
void ggml_cuda_persist_debug_after(ggml_backend_cuda_context & ctx) {
    static const int dbg = getenv("GGML_CUDA_PERSIST_DEBUG") ? atoi(getenv("GGML_CUDA_PERSIST_DEBUG")) : 0;
    if (!dbg) return;
    static std::unordered_map<const void *, bool> reported;
    pk_ctx_state & S = pk_ctx(ctx);
    if (S.regions.empty()) return;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    for (auto & kv : S.regions) {
        pk_region_cache & R = kv.second;
        const int nt = (int) R.host.size();
        if (nt == 0 || R.d_state == nullptr || reported[R.d_state]) continue;
        std::vector<pk_state> st(nt + 1);
        CUDA_CHECK(cudaMemcpy(st.data(), R.d_state, (nt + 1) * sizeof(pk_state), cudaMemcpyDeviceToHost));
        if (st[0].pad[0] == 0) continue;
        reported[R.d_state] = true;
        const int bad = (int) st[0].pad[0] - 1;
        GGML_LOG_ERROR("persist: region %p@%d of %d tasks timed out at task %d (op %d, n_items %d, n_deps %d); arrivals %lld\n",
            (const void *) kv.first.first, kv.first.second, nt, bad, R.host[bad].op, R.host[bad].n_items, R.host[bad].n_deps, st[nt].done);
        for (int a = 0; a < nt && a <= bad + 2; ++a) {
            const pk_task & t = R.host[a];
            GGML_LOG_ERROR("   task %3d op %d aux %d type %d ncols %d ne %lldx%lldx%lldx%lld src0 %lldx%lldx%lld n_items %4d n_deps %d deps[%d..+%d] done %lld deps_done %lld\n", a, t.op, t.aux, t.type, t.ncols,
                (long long) t.ne[0][0], (long long) t.ne[0][1], (long long) t.ne[0][2], (long long) t.ne[0][3], (long long) t.ne[1][0], (long long) t.ne[1][1], (long long) t.ne[1][2],
                t.n_items, t.n_deps, t.dep_first, t.dep_count, st[a].done, st[a].deps_done);
        }
    }
}
