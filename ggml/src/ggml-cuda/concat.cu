#include <cstdlib>
#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

// halo-hybrid: dim-0 concat whose second operand is a transposed 2D view (ne10 strided, ne11 contiguous), the
//     recurrent conv-state concat of the Mamba/KDA layers (conv_states [d_conv-1, C] ++ transpose(x [C, n_tokens])).
//     The generic kernel reads one element per lane down the strided dimension, a separate cache line each, at
//     ~9 GB/s on gfx1151 (5.6 ms per KDA layer at 1024 tokens); a 32x32 tile through LDS reads and writes rows.
#define CONCAT_TRANSPOSE_TILE 32
template <typename T>
static __global__ void __launch_bounds__(CONCAT_TRANSPOSE_TILE*8) concat_transpose_dim0(
        const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
        const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const int64_t ne10, const int64_t ne11, const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int64_t ne00, const int64_t ne2, const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3) {
    __shared__ T tile[CONCAT_TRANSPOSE_TILE][CONCAT_TRANSPOSE_TILE + 1];

    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i2 = blockIdx.z % ne2;
    // blockIdx.x walks ne11, the dimension that is contiguous in src1: consecutive blocks then stream the same
    //     32 source rows chunk by chunk (open DRAM pages, full lines) instead of touching 32 new rows each.
    //     blockIdx.y walks the destination columns from 0, tile-aligned, so each warp writes full lines. Columns
    //     below ne00 come from src0 (the d_conv-1 = 3 state columns in front; contiguous rows of ne00 elements), the
    //     rest from the transposed src1. Covering src0 here is what removes the generic kernel's launch of ne11
    //     blocks that each copied ne00 elements (24576 blocks for 3 elements each: 62 us per KDA layer at decode).
    const int64_t i1_0 = int64_t(blockIdx.x) * CONCAT_TRANSPOSE_TILE;   // along ne11 (rows of dst)
    const int64_t c_0  = int64_t(blockIdx.y) * CONCAT_TRANSPOSE_TILE;   // dst column
    const int64_t ne0  = ne00 + ne10;

    const char * s0 = src0 + i3*nb03 + i2*nb02;
    const char * s1 = src1 + i3*nb13 + i2*nb12;
    // read: for each column (strided in src1), ne11 consecutive elements are contiguous
#pragma unroll
    for (int r = threadIdx.y; r < CONCAT_TRANSPOSE_TILE; r += blockDim.y) {
        const int64_t c  = c_0 + r;
        const int64_t i1 = i1_0 + threadIdx.x;
        if (c < ne0 && i1 < ne11) {
            tile[r][threadIdx.x] = c < ne00 ? *(const T *)(s0 + c*nb00 + i1*nb01)
                                            : *(const T *)(s1 + (c - ne00)*nb10 + i1*nb11);
        }
    }
    __syncthreads();
    // write: dst rows i1, columns contiguous
    char * d = dst + i3*nb3 + i2*nb2;
#pragma unroll
    for (int r = threadIdx.y; r < CONCAT_TRANSPOSE_TILE; r += blockDim.y) {
        const int64_t i1 = i1_0 + r;
        const int64_t c  = c_0 + threadIdx.x;
        if (c < ne0 && i1 < ne11) {
            *(T *)(d + i1*nb1 + c*nb0) = tile[threadIdx.x][r];
        }
    }
}

// halo-hybrid: the same concat at decode (ne0 = d_conv-1 + n_tokens <= 16 columns): one thread per destination row
//     writes its ne0 contiguous elements; neighbouring threads are neighbouring rows, so every src1 column read and
//     every dst row write coalesces across the warp. 24576 rows x 6 columns: one launch of 96 blocks.
template <typename T>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_dim0_rows(
        const char * __restrict__ src0, const char * __restrict__ src1, char * __restrict__ dst,
        const int64_t ne00, const uint64_t nb00, const uint64_t nb01, const uint64_t nb02, const uint64_t nb03,
        const int64_t ne10, const int64_t ne11, const uint64_t nb10, const uint64_t nb11, const uint64_t nb12, const uint64_t nb13,
        const int64_t ne2, const uint64_t nb0, const uint64_t nb1, const uint64_t nb2, const uint64_t nb3) {
    const int64_t i1 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i1 >= ne11) {
        return;
    }
    const int64_t i3 = blockIdx.y / ne2;
    const int64_t i2 = blockIdx.y % ne2;
    const char * s0 = src0 + i3*nb03 + i2*nb02 + i1*nb01;
    const char * s1 = src1 + i3*nb13 + i2*nb12 + i1*nb11;
    char       * d  = dst  + i3*nb3  + i2*nb2  + i1*nb1;
    for (int64_t c = 0; c < ne00; c++) {
        *(T *)(d + c*nb0) = *(const T *)(s0 + c*nb00);
    }
    for (int64_t c = 0; c < ne10; c++) {
        *(T *)(d + (ne00 + c)*nb0) = *(const T *)(s1 + c*nb10);
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    // halo-hybrid: dim-0 concat with few columns (the KDA conv-state concat at decode: 3 state columns + n_tokens),
    //     any strides: one thread per destination row. The contiguous-path kernel below is one thread per element
    //     with 64-bit index math (19 us for 4 x 16384 on gfx1201); the row kernel is 3.5 us.
    static const bool no_rows = getenv("GGML_CUDA_NO_CONCAT_ROWS") != nullptr;
    if (!no_rows && dim == 0 && dst->ne[0] <= 32 && dst->nb[0] == ggml_type_size(dst->type) && src0->ne[1] == src1->ne[1] &&
            src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3]) {
        dim3 grid((src1->ne[1] + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE, dst->ne[2] * dst->ne[3], 1);
        concat_dim0_rows<T><<<grid, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            src0->ne[0], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        return;
    }
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        // transposed second operand along dim 0 (the conv-state concat): one tiled kernel covers both operands
        const bool src1_transposed = dim == 0 && src1->nb[1] == ggml_type_size(src1->type) && src1->nb[0] > src1->nb[1] &&
            dst->nb[0] == ggml_type_size(dst->type) && src1->ne[1] > 1 && src0->ne[1] == src1->ne[1];
        if (src1_transposed) {
            dim3 block(CONCAT_TRANSPOSE_TILE, 8, 1);
            dim3 grid((src1->ne[1] + CONCAT_TRANSPOSE_TILE - 1) / CONCAT_TRANSPOSE_TILE,
                      (src0->ne[0] + src1->ne[0] + CONCAT_TRANSPOSE_TILE - 1) / CONCAT_TRANSPOSE_TILE,
                      dst->ne[2] * dst->ne[3]);
            concat_transpose_dim0<T><<<grid, block, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                src0->ne[0], dst->ne[2], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
            return;
        }

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}

// halo-hybrid: the KDA conv-input assembly at decode in one launch. Replaces concat(q,k), concat(+v),
//     concat(states, transpose(qkv)) and the K rollback-slot copies of the last ns columns (build_conv_state):
//     2 + 1 + K launches -> 1. One thread per channel c of the 3*d_inner channels; neighbouring threads read
//     neighbouring elements of q/k/v at each token (coalesced) and write neighbouring rows. The slot copies re-read
//     the row the thread has just written (same thread, so ordered), which keeps the row out of a dynamically
//     indexed register array. The states may be the exact memory of dst[0] (single-slot fast path of build_rs):
//     every state value of channel c is read before any write of channel c, and no other channel touches it.
struct kda_conv_rows_dst {
    float * d[KDA_CONV_ROWS_MAX_DST];
    int     s_idx[KDA_CONV_ROWS_MAX_DST];
};

static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) kda_conv_state_rows_f32(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const uint64_t nbq1, const uint64_t nbk1, const uint64_t nbv1,
        const int d0, const int d1, const int C, const int nt,
        const char * st, const uint64_t nbs0, const uint64_t nbs1, const int ns,
        float * __restrict__ ci, const int n_dst, const kda_conv_rows_dst out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    const char * src;
    uint64_t     nb1;
    if (c < d0) {
        src = q + (int64_t) c*sizeof(float);        nb1 = nbq1;
    } else if (c < d1) {
        src = k + (int64_t) (c - d0)*sizeof(float); nb1 = nbk1;
    } else {
        src = v + (int64_t) (c - d1)*sizeof(float); nb1 = nbv1;
    }
    const int ne0 = ns + nt;
    float * row = ci + (int64_t) c*ne0;

    float sv[KDA_CONV_ROWS_MAX_DST];   // ns <= KDA_CONV_ROWS_MAX_DST (checked by the matcher)
#pragma unroll
    for (int j = 0; j < KDA_CONV_ROWS_MAX_DST; ++j) {
        if (j < ns) {
            sv[j] = *(const float *) (st + (int64_t) c*nbs1 + (int64_t) j*nbs0);
        }
    }
#pragma unroll
    for (int j = 0; j < KDA_CONV_ROWS_MAX_DST; ++j) {
        if (j < ns) {
            row[j] = sv[j];
        }
    }
    for (int t = 0; t < nt; ++t) {
        row[ns + t] = *(const float *) (src + (int64_t) t*nb1);
    }
    for (int kk = 0; kk < n_dst; ++kk) {
        float *   d = out.d[kk] + (int64_t) c*ns;
        const int s = out.s_idx[kk];
        for (int j = 0; j < ns; ++j) {
            d[j] = row[s + j];
        }
    }
}

void ggml_cuda_op_kda_conv_rows(ggml_backend_cuda_context & ctx, const ggml_cuda_kda_conv_rows_args & a) {
    kda_conv_rows_dst out = {};
    for (int kk = 0; kk < a.n_dst; ++kk) {
        out.d[kk]     = a.dst[kk];
        out.s_idx[kk] = a.s_idx[kk];
    }
    const int d0 = (int) a.q->ne[0];
    const int d1 = d0 + (int) a.k->ne[0];
    const int C  = (int) a.states->ne[1];
    const int nt = (int) a.q->ne[1];
    const int ns = (int) a.states->ne[0];
    const int nblocks = (C + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    kda_conv_state_rows_f32<<<nblocks, CUDA_CONCAT_BLOCK_SIZE, 0, ctx.stream()>>>(
        (const char *) a.q->data, (const char *) a.k->data, (const char *) a.v->data,
        a.q->nb[1], a.k->nb[1], a.v->nb[1], d0, d1, C, nt,
        (const char *) a.states->data, a.states->nb[0], a.states->nb[1], ns,
        (float *) a.conv_input->data, a.n_dst, out);
}
