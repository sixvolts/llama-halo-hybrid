// halo-hybrid: fused MoE gate+up MMQ with the GLU in the epilogue.
//
// The expert FFN of a MoE layer is up = MUL_MAT_ID(up_exps, x, ids), gate = MUL_MAT_ID(gate_exps, x, ids),
// act = GLU(gate, up). Unfused, that is two MMQ launches (each with its own activation quantize and expert sort, see
// ggml_cuda_mul_mat_q_gate_up in mmq.cu for the shared preparation), two [n_ff, n_used, n_tokens] f32 outputs written
// to memory and read back by a separate GLU launch.
//
// This kernel runs one block per (row tile, expert column tile) for BOTH matrices: the block is two half-height MMQ
// blocks (GGML_CUDA_MMQ_VARIANT_HALF_I: half the rows, half the waves) stacked along threadIdx.z (z = 0 gate, z = 1 up),
// so it has the threads, rows and LDS of one regular block and the same number of blocks as the two unfused launches. Each half loads its own weight tile with the
// unchanged per-type loader and runs the unchanged vec_dot over one shared activation tile, so every output value is
// accumulated exactly as in mul_mat_q (bit-identical sums). After the K loop the up half passes its accumulators to
// the gate half through LDS (register order is the same in both halves, so element k of lane (x, y) is the same
// (row, column) in both), the gate half applies the GLU with the same float ops as the GLU kernels (unary.cu) and
// writes the single [n_ff, n_used, n_tokens] output through the regular write-back.
//
// A first version stacked two full-height blocks (8 waves, ~44-51 KiB LDS): one block per CU instead of two, all waves
// in barrier lockstep, 1.2-1.6x slower than two unfused launches at 1024-4096 tokens on gfx1151. Only the non-stream-k MoE tiling path with the compact tile list,
// the MMA data layout on AMD WMMA and full row tiles (rows % 128 == 0) are supported; everything else keeps the
// unfused path.

#include "common.cuh"
#include "mmq.cuh"
#include "unary.cuh"

template <ggml_type type, int J>
__launch_bounds__(2*ggml_cuda_mmq_get_nthreads(type, J, GGML_CUDA_MMQ_VARIANT_HALF_I), ggml_cuda_mmq_get_occupancy(type, J, GGML_CUDA_MMQ_VARIANT_HALF_I))
static __global__ void mul_mat_q_gate_up(
        const char * __restrict__ x_gate, const char * __restrict__ x_up, const int * __restrict__ y,
        const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ expert_bounds, float * __restrict__ dst,
        const int nblocks_k, const int nrows_x, const int stride_row_x, const int ncols_y, const int stride_col_dst,
        const int stride_channel_x, const int x_tile_ints, const int32_t * __restrict__ tile_list,
        const int glu_op, const float glu_alpha, const float glu_limit) {
    constexpr int fallback = GGML_CUDA_MMQ_VARIANT_HALF_I;

    if constexpr (ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT ||
                  !ggml_cuda_mmq_get_config(type, J, fallback).use_mma_data_layout() ||
                  ggml_cuda_mmq_get_stream_k(type, J, fallback)) {
        GGML_UNUSED_VARS(x_gate, x_up, y, ids_dst, expert_bounds, dst, nblocks_k, nrows_x, stride_row_x, ncols_y,
            stride_col_dst, stride_channel_x, x_tile_ints, tile_list, glu_op, glu_alpha, glu_limit);
        NO_DEVICE_CODE;
    } else {
        constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
        constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
        constexpr int nthreads        = nwarps*warp_size;   // per half
        constexpr int qk              = ggml_cuda_type_traits<type>::qk;
        constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
        constexpr int ITER_K          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
        constexpr int blocks_per_iter = ITER_K / qk;
        constexpr int sz              = sizeof(block_q8_1_mmq) / sizeof(int);
        constexpr int ne_block        = QK8_1_MMQ;
        constexpr int ny              = J*MMQ_TILE_Y_K;     // ints of one activation half-tile
        constexpr int nsum            = J*I / nthreads;

        constexpr ggml_cuda_mmq_load_tiles_t load_tiles = ggml_cuda_mmq_get_load_tiles<type, J, fallback>();
        constexpr ggml_cuda_mmq_vec_dot_t    vec_dot    = ggml_cuda_mmq_get_vec_dot<type, J, fallback>();
        constexpr ggml_cuda_mmq_write_back_t write_back = ggml_cuda_mmq_get_write_back<type, J, fallback>();

        extern __shared__ int data_mul_mat_q_gate_up[];
        int * ids_dst_shared = data_mul_mat_q_gate_up;
        // padded like mul_mat_q's tile (2*nthreads == the regular block's threads): the stage reads up to tile_y_n ints
        // per half, past the last column into the next K block / the q8_1 buffer padding, exactly as mul_mat_q does
        constexpr int tile_y_n = GGML_PAD(ny, 2*nthreads);
        int * tile_y  = data_mul_mat_q_gate_up + J;
        int * tile_x0 = tile_y + tile_y_n;

#if defined(GGML_USE_HIP)
        const int zsel = __builtin_amdgcn_readfirstlane(threadIdx.z);  // 0: gate, 1: up (wave-uniform: keep x and tile_x scalar)
#else
        const int zsel = threadIdx.z;                                  // 0: gate, 1: up
#endif // defined(GGML_USE_HIP)
        const int tid  = threadIdx.y*warp_size + threadIdx.x;    // thread index within the half
        const int tid2 = zsel*nthreads + tid;                    // thread index within the block
        int * tile_x = tile_x0 + zsel*x_tile_ints;

        // the compact (expert, column tile) list, see mmq_moe_tile_list; the pair count is stored behind the list
        if ((int) blockIdx.y >= tile_list[gridDim.y]) {
            return;
        }
        const int packed = tile_list[blockIdx.y];
        const int zt = packed & 0xFFFF;
        const int jt = packed >> 16;
        const int it = blockIdx.x;

        const int col_low  = expert_bounds[zt + 0];
        const int col_diff = expert_bounds[zt + 1] - col_low;
        if (jt*J >= col_diff) {
            return;
        }

        const int tile_x_max_i = nrows_x  - it*I - 1;
        const int tile_y_max_j = col_diff - jt*J - 1;

        for (int j = tid2; j < J; j += 2*nthreads) {
            ids_dst_shared[j] = j <= tile_y_max_j ? ids_dst[col_low + jt*J + j] : 0;
        }
        // (the first __syncthreads of the K loop orders these stores before the write-back reads them)

        const int  * yb       = y + (col_low + jt*J)*sz;
        const char * x        = zsel ? x_up : x_gate;
        const int    offset_x = zt*stride_channel_x + it*I*stride_row_x;

        float sum[nsum] = {0.0f};

        if constexpr (ggml_cuda_mmq_use_prefetch<type, J, fallback>()) {
            // RDNA3.5 register prefetch, as in mul_mat_q_process_tile; the activation stage is split over both halves
            constexpr int  y_regs_n  = (ny + 2*nthreads - 1) / (2*nthreads);
            constexpr bool x_regs_ok = ggml_cuda_mmq_x_regs<type, J, fallback>::ok;

            int yr0[y_regs_n];
            int yr1[y_regs_n];
            ggml_cuda_mmq_x_regs<type, J, fallback> xr;
            GGML_UNUSED(xr);

            auto load_y_regs = [&](int (&yr)[y_regs_n], const int kb0, const int half) {
                const int * by0 = yb + ncols_y * ((kb0 * qk / ne_block) * sz + half*sz);
#pragma unroll
                for (int r = 0; r < y_regs_n; ++r) {
                    yr[r] = by0[r*2*nthreads + tid2];
                }
            };
            auto store_y_regs = [&](const int (&yr)[y_regs_n]) {
#pragma unroll
                for (int r = 0; r < y_regs_n; ++r) {
                    tile_y[r*2*nthreads + tid2] = yr[r];
                }
            };

            if constexpr (x_regs_ok) {
                xr.load(x, offset_x, tile_x_max_i, stride_row_x);
            }
            load_y_regs(yr0, 0, 0);
            load_y_regs(yr1, 0, 1);

            for (int kb0 = 0; kb0 < nblocks_k; kb0 += blocks_per_iter) {
                const bool has_next = kb0 + blocks_per_iter < nblocks_k;

                if constexpr (x_regs_ok) {
                    xr.store(tile_x, tile_x_max_i);
                } else {
                    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
                }
                store_y_regs(yr0);

                __syncthreads();

                if constexpr (x_regs_ok) {
                    if (has_next) {
                        xr.load(x, offset_x + kb0 + blocks_per_iter, tile_x_max_i, stride_row_x);
                    }
                }
                if (has_next) {
                    load_y_regs(yr0, kb0 + blocks_per_iter, 0);
                }

                vec_dot(tile_x, tile_y, sum, 0, J);

                __syncthreads();

                store_y_regs(yr1);

                __syncthreads();

                if (has_next) {
                    load_y_regs(yr1, kb0 + blocks_per_iter, 1);
                }

                vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K, J);

                __syncthreads();
            }
        } else {
            auto load_y = [&](const int kb0, const int half) {
                const int * by0 = yb + ncols_y * ((kb0 * qk / ne_block) * sz + half*sz);
#pragma unroll
                for (int l0 = 0; l0 < ny; l0 += 2*nthreads) {
                    tile_y[l0 + tid2] = by0[l0 + tid2];
                }
            };

            for (int kb0 = 0; kb0 < nblocks_k; kb0 += blocks_per_iter) {
                load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
                load_y(kb0, 0);

                __syncthreads();

                vec_dot(tile_x, tile_y, sum, 0, J);

                __syncthreads();

                load_y(kb0, 1);

                __syncthreads();

                vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K, J);

                __syncthreads();
            }
        }

        // The K loop ended on a barrier: the weight tiles are free. The up half hands its accumulators to the gate half.
        float * xchg = (float *) tile_x0;
        if (zsel == 1) {
#pragma unroll
            for (int k = 0; k < nsum; ++k) {
                xchg[k*nthreads + tid] = sum[k];
            }
        }
        __syncthreads();
        if (zsel == 1) {
            return;
        }
        switch (glu_op) {
            case GGML_GLU_OP_SWIGLU:
#pragma unroll
                for (int k = 0; k < nsum; ++k) {
                    sum[k] = ggml_cuda_op_silu_single(sum[k]) * xchg[k*nthreads + tid];
                }
                break;
            case GGML_GLU_OP_GEGLU:
#pragma unroll
                for (int k = 0; k < nsum; ++k) {
                    sum[k] = ggml_cuda_op_gelu_single(sum[k]) * xchg[k*nthreads + tid];
                }
                break;
            case GGML_GLU_OP_SWIGLU_OAI:
#pragma unroll
                for (int k = 0; k < nsum; ++k) {
                    sum[k] = ggml_cuda_op_swiglu_oai_single(sum[k], xchg[k*nthreads + tid], glu_alpha, glu_limit);
                }
                break;
            default: // GGML_GLU_OP_SWIGLU_CLAMP
#pragma unroll
                for (int k = 0; k < nsum; ++k) {
                    sum[k] = ggml_cuda_op_swiglu_clamp_single(sum[k], xchg[k*nthreads + tid], glu_limit);
                }
                break;
        }
        write_back(sum, ids_dst_shared, dst + it*I, nullptr, stride_col_dst, tile_x_max_i, tile_y_max_j);
    }
}

static int mmq_gate_up_nbytes_shared(const ggml_cuda_mmq_config & config, const int cc) {
    const int nbs_ids = config.J*sizeof(int);
    const int nbs_x   = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    const int nbs_y   = config.J*sizeof(block_q8_1_mmq);
    return nbs_ids + 2*nbs_x + GGML_PAD(nbs_y, 2*config.nthreads*(int) sizeof(int));
}

// Whether the fused kernel can run this configuration at all.
static bool mmq_gate_up_config_ok(const ggml_cuda_mmq_config & config, const int cc, const size_t smpbo, const int warp_size) {
    if (config.type == GGML_TYPE_COUNT || config.stream_k || config.fallback || !config.use_mma_data_layout(cc)) {
        return false;
    }
    if (2*config.nthreads > 1024 || config.nthreads % warp_size != 0) {
        return false;
    }
    // the up half's accumulators (J*I floats) are handed over in the two weight tiles
    const int nbs_x = ggml_cuda_mmq_get_nbytes_shared_x(config, cc);
    if (config.J*config.I*(int) sizeof(float) > 2*nbs_x) {
        return false;
    }
    return (size_t) mmq_gate_up_nbytes_shared(config, cc) <= smpbo;
}

template <ggml_type type, int J>
static void launch_mul_mat_q_gate_up(ggml_backend_cuda_context & ctx, const mmq_args & args, const char * x_up,
        const int glu_op, const float alpha, const float limit, cudaStream_t stream) {
    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;

    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, GGML_CUDA_MMQ_VARIANT_HALF_I, cc);
    const int nwarps        = config.nthreads / warp_size;
    const int nbytes_shared = mmq_gate_up_nbytes_shared(config, cc);
    const int x_tile_ints   = ggml_cuda_mmq_get_nbytes_shared_x(config, cc) / sizeof(int);

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q_gate_up<type, J>), nbytes_shared);

    const int nty = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx = (args.ncols_max + config.J - 1) / config.J;

    // sum_e ceil(tokens_e/J) <= ncols_dst/J + n_expert, and never more than the full grid
    const int64_t cap = std::min<int64_t>((int64_t) ntx * args.nchannels_y, (args.ncols_dst + config.J - 1) / config.J + args.nchannels_y);
    ggml_cuda_pool_alloc<int32_t> tile_list(ctx.pool(id), cap + 1);
    mmq_moe_tile_list<<<1, MMQ_MOE_TILE_LIST_THREADS, 0, stream>>>
        (args.expert_bounds, args.nchannels_y, config.J, ntx, tile_list.ptr, cap);
    CUDA_CHECK(cudaGetLastError());

    const dim3 block_nums(nty, cap, 1);
    const dim3 block_dims(warp_size, nwarps, 2);
    mul_mat_q_gate_up<type, J><<<block_nums, block_dims, nbytes_shared, stream>>>
        (args.x, x_up, args.y, args.ids_dst, args.expert_bounds, args.dst,
         args.ncols_x / ggml_cuda_type_traits<type>::qk, args.nrows_x, args.stride_row_x, args.ncols_y, args.nrows_dst,
         args.stride_channel_x, x_tile_ints, tile_list.ptr, glu_op, alpha, limit);
    CUDA_CHECK(cudaGetLastError());
}

template <ggml_type type>
static bool mul_mat_q_gate_up_case(ggml_backend_cuda_context & ctx, const mmq_args & args, const char * x_up,
        const int glu_op, const float alpha, const float limit, cudaStream_t stream) {
    const int    id        = ggml_cuda_get_device();
    const int    cc        = ggml_cuda_info().devices[id].cc;
    const size_t smpbo     = ggml_cuda_info().devices[id].smpbo;
    const int    warp_size = ggml_cuda_info().devices[id].warp_size;

    if (args.nrows_x % 128 != 0) {
        return false;   // fallback tiles are not supported
    }

    // the unfused column tile (mul_mat_q_switch_J) and the best one the fused kernel can run
    int J_unfused = 0, nt_unfused = INT_MAX;
    int J_fused   = 0, nt_fused   = INT_MAX;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, cc) > smpbo) {
            continue;
        }
        const int nt = (args.ncols_opt + config.J - 1) / config.J;
        if (nt < nt_unfused) {
            J_unfused  = J;
            nt_unfused = nt;
        }
        const ggml_cuda_mmq_config config_half = ggml_cuda_mmq_get_config(type, J, GGML_CUDA_MMQ_VARIANT_HALF_I, cc);
        if (nt < nt_fused && mmq_gate_up_config_ok(config_half, cc, smpbo, warp_size)) {
            J_fused  = J;
            nt_fused = nt;
        }
    }
    GGML_UNUSED(J_unfused);
    // do not trade a wider unfused tile for more fused column tiles (RDNA4 q4_K J >= 80 uses 128-row tiles whose
    // doubled weight LDS does not fit)
    if (J_fused == 0 || nt_fused > nt_unfused) {
        return false;
    }
    // on RDNA3.5 narrow tiles are bandwidth-bound and the fused GEMM itself is ~6% slower than the two it replaces, so
    // only the shared preparation pays there (GLM q4_K 2048 x 4096 at J = 16, 512 tokens: fused 1.5-2.5% slower than
    // mode 1). GGML_CUDA_MMQ_GATEUP_MIN_J overrides the smallest fused tile (default 24 on RDNA3.5, 8 elsewhere)
    static const int min_j_env = getenv("GGML_CUDA_MMQ_GATEUP_MIN_J") ? atoi(getenv("GGML_CUDA_MMQ_GATEUP_MIN_J")) : -1;
    const int min_j = min_j_env >= 0 ? min_j_env : (GGML_CUDA_CC_IS_RDNA3_5(cc) ? 24 : 8);
    if (J_fused < min_j) {
        return false;
    }

    switch (J_fused) {
        case   8: launch_mul_mat_q_gate_up<type,   8>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  16: launch_mul_mat_q_gate_up<type,  16>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  24: launch_mul_mat_q_gate_up<type,  24>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  32: launch_mul_mat_q_gate_up<type,  32>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  40: launch_mul_mat_q_gate_up<type,  40>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  48: launch_mul_mat_q_gate_up<type,  48>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  56: launch_mul_mat_q_gate_up<type,  56>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  64: launch_mul_mat_q_gate_up<type,  64>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  72: launch_mul_mat_q_gate_up<type,  72>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  80: launch_mul_mat_q_gate_up<type,  80>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  88: launch_mul_mat_q_gate_up<type,  88>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case  96: launch_mul_mat_q_gate_up<type,  96>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case 104: launch_mul_mat_q_gate_up<type, 104>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case 112: launch_mul_mat_q_gate_up<type, 112>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case 120: launch_mul_mat_q_gate_up<type, 120>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        case 128: launch_mul_mat_q_gate_up<type, 128>(ctx, args, x_up, glu_op, alpha, limit, stream); break;
        default: return false;
    }
    return true;
}

bool ggml_cuda_mul_mat_q_gate_up_fused(ggml_backend_cuda_context & ctx, const mmq_args & args, const char * x_up,
        const int glu_op, const float alpha, const float limit, cudaStream_t stream) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_AMD(cc) || !amd_wmma_available(cc) || args.y_scale != nullptr) {
        return false;
    }
    switch (args.type_x) {
        case GGML_TYPE_Q4_K:
            return mul_mat_q_gate_up_case<GGML_TYPE_Q4_K>(ctx, args, x_up, glu_op, alpha, limit, stream);
        default:
            return false;
    }
}
