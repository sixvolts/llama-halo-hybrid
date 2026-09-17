#include "f32act.cuh"
#include "vecdotq.cuh"

#define F32ACT_WAVES 8

#define F32ACT_RPW 4    // rows per wave: their loads are issued together, 4x the bytes in flight per wave, and the
                        // LDS staging of the activation is amortised over 32 rows per block instead of 8

template <int ncols_dst>
static __global__ void __launch_bounds__(F32ACT_WAVES * ggml_cuda_get_physical_warp_size(), 1)
k_gemv_q8_0_f32act(const block_q8_0 * __restrict__ w, const float * __restrict__ y, float * __restrict__ dst,
                   const int ncols_x, const int stride_row_w, const int stride_col_y, const int stride_col_dst,
                   const int nrows, const ggml_cuda_f32act_prologue pro) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int block_threads = F32ACT_WAVES * warp_size;
    extern __shared__ float s_y[];   // [ncols_dst][ncols_x], the activation after the prologue

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size, wave = tid / warp_size;
    const int row0 = (blockIdx.x * F32ACT_WAVES + wave) * F32ACT_RPW;
    const int nblk = ncols_x / QK8_0;
    const int sub  = lane % 4;                 // 4 lanes per block, 8 weights each
    constexpr int kstep = warp_size / 4;

    // the first weight tile is requested BEFORE the activation is staged, so the two latencies overlap; a wave
    // past the last row keeps a harmless clamped row (it still has to reach the barrier)
    auto load_tile = [&](int kb, int q0[F32ACT_RPW], int q1[F32ACT_RPW], float d[F32ACT_RPW]) {
#pragma unroll
        for (int r = 0; r < F32ACT_RPW; ++r) {
            const int row = min(row0 + r, nrows - 1);   // clamp: a tail row is computed twice, never out of bounds
            const block_q8_0 & b = w[(int64_t) row * stride_row_w + kb];
            d[r]  = __half2float(b.d);
            q0[r] = get_int_b2(b.qs, 2*sub + 0);
            q1[r] = get_int_b2(b.qs, 2*sub + 1);
        }
    };
    int   q0[F32ACT_RPW], q1[F32ACT_RPW];
    float d[F32ACT_RPW];
    int kb = lane / 4;
    if (kb < nblk) {
        load_tile(kb, q0, q1, d);
    }

    for (int i = tid; i < ncols_dst * ncols_x; i += block_threads) {
        const int j = i / ncols_x, c = i - j * ncols_x;
        float v = y[j * stride_col_y + c];
        if (pro.silu) {
            v = pro.scale * v + pro.bias;
            v = v / (1.0f + expf(-v));
        }
        s_y[i] = v;
    }
    __syncthreads();
    if (row0 >= nrows) {
        return;
    }

    float acc[F32ACT_RPW][ncols_dst];
#pragma unroll
    for (int r = 0; r < F32ACT_RPW; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[r][j] = 0.0f;
        }
    }
    for (; kb < nblk; kb += kstep) {
        int   n0[F32ACT_RPW], n1[F32ACT_RPW];
        float nd[F32ACT_RPW];
        if (kb + kstep < nblk) {
            load_tile(kb + kstep, n0, n1, nd);   // next tile in flight while this one is used
        }
        const float * sy = s_y + kb * QK8_0 + sub * 8;
#pragma unroll
        for (int r = 0; r < F32ACT_RPW; ++r) {
            float part[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                part[j] = 0.0f;
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float w0 = (float) (int8_t) ((q0[r] >> (8*e)) & 0xFF);
                const float w1 = (float) (int8_t) ((q1[r] >> (8*e)) & 0xFF);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    part[j] += w0 * sy[j * ncols_x + e] + w1 * sy[j * ncols_x + 4 + e];
                }
            }
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                acc[r][j] += d[r] * part[j];
            }
        }
        if (kb + kstep < nblk) {
#pragma unroll
            for (int r = 0; r < F32ACT_RPW; ++r) {
                q0[r] = n0[r]; q1[r] = n1[r]; d[r] = nd[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < F32ACT_RPW; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[r][j] = warp_reduce_sum<warp_size>(acc[r][j]);
            if (lane == 0 && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = acc[r][j];
            }
        }
    }
}

// MUL_MAT_ID at decode width: blockIdx.y is the (expert slot, token) pair; the weight is the expert ids names,
// the activation is that slot's row, the output that slot's column. One column per slot, so ncols_dst is 1.
static __global__ void __launch_bounds__(F32ACT_WAVES * ggml_cuda_get_physical_warp_size(), 1)
k_gemv_q8_0_f32act_id(const block_q8_0 * __restrict__ w, const float * __restrict__ y, float * __restrict__ dst,
                      const int32_t * __restrict__ ids, const int ncols_x, const int stride_row_w, const int64_t stride_expert_w,
                      const int n_slots, const int64_t stride_slot_y, const int64_t stride_tok_y,
                      const int64_t stride_slot_ids, const int64_t stride_tok_ids,
                      const int64_t stride_slot_dst, const int64_t stride_tok_dst, const int nrows) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int block_threads = F32ACT_WAVES * warp_size;
    extern __shared__ float s_y[];

    const int slot = blockIdx.y % n_slots, tok = blockIdx.y / n_slots;
    const int expert = ids[tok * stride_tok_ids + slot * stride_slot_ids];
    const float * yr = y + tok * stride_tok_y + slot * stride_slot_y;
    const block_q8_0 * we = w + expert * stride_expert_w;
    float * dr = dst + tok * stride_tok_dst + slot * stride_slot_dst;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size, wave = tid / warp_size;
    const int row0 = (blockIdx.x * F32ACT_WAVES + wave) * F32ACT_RPW;
    const int nblk = ncols_x / QK8_0;
    const int sub  = lane % 4;
    constexpr int kstep = warp_size / 4;
    auto load_tile = [&](int kb, int q0[F32ACT_RPW], int q1[F32ACT_RPW], float d[F32ACT_RPW]) {
#pragma unroll
        for (int r = 0; r < F32ACT_RPW; ++r) {
            const int row = min(row0 + r, nrows - 1);
            const block_q8_0 & b = we[(int64_t) row * stride_row_w + kb];
            d[r]  = __half2float(b.d);
            q0[r] = get_int_b2(b.qs, 2*sub + 0);
            q1[r] = get_int_b2(b.qs, 2*sub + 1);
        }
    };
    int   q0[F32ACT_RPW], q1[F32ACT_RPW];
    float d[F32ACT_RPW];
    int kb = lane / 4;
    if (kb < nblk) {
        load_tile(kb, q0, q1, d);
    }
    for (int i = tid; i < ncols_x; i += block_threads) {
        s_y[i] = yr[i];
    }
    __syncthreads();
    if (row0 >= nrows) {
        return;
    }
    float acc[F32ACT_RPW] = {};
    for (; kb < nblk; kb += kstep) {
        int   n0[F32ACT_RPW], n1[F32ACT_RPW];
        float nd[F32ACT_RPW];
        if (kb + kstep < nblk) {
            load_tile(kb + kstep, n0, n1, nd);
        }
        const float * sy = s_y + kb * QK8_0 + sub * 8;
#pragma unroll
        for (int r = 0; r < F32ACT_RPW; ++r) {
            float part = 0.0f;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                part += (float) (int8_t) ((q0[r] >> (8*e)) & 0xFF) * sy[e] + (float) (int8_t) ((q1[r] >> (8*e)) & 0xFF) * sy[4 + e];
            }
            acc[r] += d[r] * part;
        }
        if (kb + kstep < nblk) {
#pragma unroll
            for (int r = 0; r < F32ACT_RPW; ++r) {
                q0[r] = n0[r]; q1[r] = n1[r]; d[r] = nd[r];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < F32ACT_RPW; ++r) {
        acc[r] = warp_reduce_sum<warp_size>(acc[r]);
        if (lane == 0 && row0 + r < nrows) {
            dr[row0 + r] = acc[r];
        }
    }
}

bool ggml_cuda_mul_mat_id_vec_q8_f32act(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
                                        const ggml_tensor * ids, ggml_tensor * dst) {
    static const int64_t max_k = getenv("GGML_CUDA_F32ACT_K") ? atoll(getenv("GGML_CUDA_F32ACT_K")) : 0;   // OFF: see the header
    if (max_k <= 0 || w->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(w) || w->ne[3] != 1) {
        return false;
    }
    const int64_t K = w->ne[0];
    if (K > max_k || K % QK8_0 != 0 || K > INT32_MAX) {
        return false;
    }
    // src1 [K, n_expert_used, n_tokens], ids [n_expert_used, n_tokens], dst [M, n_expert_used, n_tokens]
    if (y->type != GGML_TYPE_F32 || y->nb[0] != sizeof(float) || y->ne[0] != K || y->ne[3] != 1 || y->ne[2] > 4 ||
        ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[0] != y->ne[1] || ids->ne[1] != y->ne[2] ||
        dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) || dst->ne[0] != w->ne[1] || dst->ne[1] != y->ne[1] || dst->ne[2] != y->ne[2]) {
        return false;
    }
    ggml_cuda_set_device(ctx.device);
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int n_slots = (int) y->ne[1];
    const dim3 grid((w->ne[1] + F32ACT_WAVES*F32ACT_RPW - 1) / (F32ACT_WAVES*F32ACT_RPW), n_slots * y->ne[2], 1);
    const dim3 block(F32ACT_WAVES * warp_size, 1, 1);
    k_gemv_q8_0_f32act_id<<<grid, block, K * sizeof(float), ctx.stream()>>>(
        (const block_q8_0 *) w->data, (const float *) y->data, (float *) dst->data, (const int32_t *) ids->data,
        (int) K, (int) (w->nb[1] / sizeof(block_q8_0)), (int64_t) (w->nb[2] / sizeof(block_q8_0)),
        n_slots, (int64_t) (y->nb[1] / sizeof(float)), (int64_t) (y->nb[2] / sizeof(float)),
        (int64_t) (ids->nb[0] / sizeof(int32_t)), (int64_t) (ids->nb[1] / sizeof(int32_t)),
        (int64_t) (dst->nb[1] / sizeof(float)), (int64_t) (dst->nb[2] / sizeof(float)), (int) w->ne[1]);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mul_mat_vec_q8_f32act(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * y,
                                     ggml_tensor * dst, const ggml_cuda_f32act_prologue * pro) {
    static const int64_t max_k = getenv("GGML_CUDA_F32ACT_K") ? atoll(getenv("GGML_CUDA_F32ACT_K")) : 0;   // OFF: see the header
    if (max_k <= 0 || w->type != GGML_TYPE_Q8_0 || !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (y->type != GGML_TYPE_F32 || y->nb[0] != sizeof(float) || y->ne[0] != w->ne[0] || y->ne[1] < 1 || y->ne[1] > 4 ||
        y->ne[2] != 1 || y->ne[3] != 1 || dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t K = w->ne[0];
    if (K > max_k || K % QK8_0 != 0 || K > INT32_MAX) {
        return false;
    }
    if (dst->ne[0] != w->ne[1] || dst->ne[1] != y->ne[1]) {
        return false;
    }
    ggml_cuda_set_device(ctx.device);
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t ncols_dst = y->ne[1];
    const ggml_cuda_f32act_prologue p = pro ? *pro : ggml_cuda_f32act_prologue{1.0f, 0.0f, false};
    const dim3 grid((w->ne[1] + F32ACT_WAVES*F32ACT_RPW - 1) / (F32ACT_WAVES*F32ACT_RPW), 1, 1);
    const dim3 block(F32ACT_WAVES * warp_size, 1, 1);
    const size_t smem = ncols_dst * K * sizeof(float);
    const block_q8_0 * wd = (const block_q8_0 *) w->data;
    const float * yd = (const float *) y->data;
    float * dd = (float *) dst->data;
    const int srw = (int) (w->nb[1] / sizeof(block_q8_0));
    const int scy = (int) (y->nb[1] / sizeof(float));
    const int scd = (int) (dst->nb[1] / sizeof(float));
    switch (ncols_dst) {
        case 1: k_gemv_q8_0_f32act<1><<<grid, block, smem, ctx.stream()>>>(wd, yd, dd, (int) K, srw, scy, scd, (int) w->ne[1], p); break;
        case 2: k_gemv_q8_0_f32act<2><<<grid, block, smem, ctx.stream()>>>(wd, yd, dd, (int) K, srw, scy, scd, (int) w->ne[1], p); break;
        case 3: k_gemv_q8_0_f32act<3><<<grid, block, smem, ctx.stream()>>>(wd, yd, dd, (int) K, srw, scy, scd, (int) w->ne[1], p); break;
        case 4: k_gemv_q8_0_f32act<4><<<grid, block, smem, ctx.stream()>>>(wd, yd, dd, (int) K, srw, scy, scd, (int) w->ne[1], p); break;
        default: return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}
