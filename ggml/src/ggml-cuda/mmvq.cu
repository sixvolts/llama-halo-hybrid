#define GGML_CUDA_MMVQ_RDNA35 0
#include "mmvq.cuh"
#include "hc.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static __device__ __forceinline__ void mmvq_prefetch_l2(const void * p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA3_5)
    // halo-hybrid: gfx1151 fell through the RDNA2 table to nwarps = 1 (one wave per row). GGML_CUDA_MMVQ_RDNA35
    // picks the table at build time so the choice can be measured on the real model: 0 = RDNA2 (upstream), 1 = RDNA3_0.
#if defined(GGML_CUDA_MMVQ_RDNA35) && GGML_CUDA_MMVQ_RDNA35 == 1
    return MMVQ_PARAMETERS_RDNA3_0;
#else
    return MMVQ_PARAMETERS_RDNA2;
#endif
#elif defined(RDNA2)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
#if defined(GGML_CUDA_MMVQ_RDNA35) && GGML_CUDA_MMVQ_RDNA35 == 1
        return MMVQ_PARAMETERS_RDNA3_0;
#else
        return MMVQ_PARAMETERS_RDNA2;
#endif
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}
// halo-hybrid: Q4_K/Q5_K/Q6_K cap at 2 on RDNA3.5 (RDNA3 keeps 4). mul_mat_vec_q_moe streams an expert once per (token, slot)
//     pair, MMQ once per distinct expert; on real text the tokens of a speculative verify batch share experts, and on
//     gfx1151 (GLM-5.3-Flash, two-host) MMQ took the verify graph from 90.6 to 87.8 ms at 3 tokens and 112.9 to 107.3
//     at 4 while random-routing isolation showed a tie. GGML_CUDA_MMID_MMVQ_MAX_RDNA3=<n> (RDNA3 and 3.5) still lowers it further.
static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3_5(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 2;
        case GGML_TYPE_Q5_K:    return 2;
        case GGML_TYPE_Q6_K:    return 2;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// halo-hybrid: the multi-token MoE GEMV (mul_mat_vec_q_moe) gives every (token, expert slot) pair its own warp, so
//     an expert chosen by two tokens of a verify batch is streamed twice, while MMQ streams each distinct expert once.
//     On real text neighbouring tokens share experts, so the crossover is lower than the random-routing tables
//     assume. GGML_CUDA_MMID_MMVQ_MAX_RDNA3=<n> / _RDNA4=<n> caps the vector path at n tokens for the K-quants on
//     that family (lowering only; the compiled kernels cover every smaller width).
static int get_mmvq_mmid_max_batch_env_cap(int cc) {
    static const int cap3 = getenv("GGML_CUDA_MMID_MMVQ_MAX_RDNA3") ? atoi(getenv("GGML_CUDA_MMID_MMVQ_MAX_RDNA3")) : 0;
    static const int cap4 = getenv("GGML_CUDA_MMID_MMVQ_MAX_RDNA4") ? atoi(getenv("GGML_CUDA_MMID_MMVQ_MAX_RDNA4")) : 0;
    if (GGML_CUDA_CC_IS_RDNA4(cc)) { return cap4; }
    if (GGML_CUDA_CC_IS_RDNA3(cc)) { return cap3; }
    return 0;
}

static int get_mmvq_mmid_max_batch_uncapped(ggml_type type, int cc);

// Host function: returns the max batch size for the current arch+type at runtime.
bool ggml_cuda_mmvq_moe_grouped_enabled(ggml_type type);

int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    if (ggml_cuda_mmvq_moe_grouped_enabled(type)) {
        // halo-hybrid: the grouped MoE GEMV reads each distinct expert once, like MMQ, without MMQ's tile overhead
        return MMVQ_MAX_BATCH_SIZE;
    }
    const int n = get_mmvq_mmid_max_batch_uncapped(type, cc);
    const int cap = get_mmvq_mmid_max_batch_env_cap(cc);
    const bool kquant = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
    return (cap > 0 && kquant) ? std::min(n, cap) : n;
}

static int get_mmvq_mmid_max_batch_uncapped(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
            return get_mmvq_mmid_max_batch_rdna3_5(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3_5)
    return get_mmvq_mmid_max_batch_rdna3_5(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        // halo-hybrid: the same 8-warp block for ncols_dst 2..4 when the host asks for it (small_k or
        // halve_iters tag). Upstream launches ONE wave per row there, which on a 128-SIMD part leaves
        // short-M GEMVs (24..2048 rows) as serial latency chains at 2-35% of DRAM bandwidth; decode with
        // an MTP draft runs at n=3, so every dense GEMV of a decode step took that path.
        // The small_k tag at ncols_dst 2..4 is the TALL launch instead: 32 warps per block for a handful of rows
        // over a long K (hc_fn 24 x 16384: 24 blocks x 8 warps = 192 waves, 8 dependent DRAM trips each, 15-18 us
        // in situ; 32 warps make it 2 trips).
        if (ncols_dst != 1 && ncols_dst <= 4 && small_k) {
            switch (type) {
                case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_NL:
                    return 32;
                default:
                    return 1;
            }
        }
        if (ncols_dst == 1 || (ncols_dst <= 4 && halve_iters)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // halo-hybrid: small-K rows-per-block mode on RDNA4 (K=2560 rows took 2 loop trips + an 8-warp reduction)
        return (ncols_dst == 1 && small_k) ? nwarps : 1;
    }
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

// halo-hybrid: upstream disables the small-K rows-per-block mode on every RDNA part. On RDNA4
// (R9700, gfx1201) it is what fixes short-K matvecs: a Q8_0 [10240 x 320] row went from 23 us
// (150 GB/s, one 256-thread block per row doing one loop trip) to 7 us (490 GB/s), and the
// large-K shapes are unchanged within 1%. Measured -6% per decoded token on Qwen3.8-Flash-Next.
// GGML_CUDA_MMVQ_NO_SMALLK=1 restores the upstream behaviour.
// halo-hybrid: 8-warp blocks for ncols_dst 2..4 on RDNA4 (see calc_nwarps). GGML_CUDA_MMVQ_NO_WIDE=1 restores
// the upstream one-wave-per-row launch for A/B.
static bool ggml_cuda_mmvq_rdna4_wide() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_MMVQ_NO_WIDE");
        return env == nullptr || atoi(env) == 0;
    }();
    return enabled;
}

static bool ggml_cuda_mmvq_rdna4_small_k() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_MMVQ_NO_SMALLK");
        return env == nullptr || atoi(env) == 0;
    }();
    return enabled;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;


    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    if constexpr (has_fusion) {
        // halo-hybrid: the KDA beta sigmoid rides on block 0 (see ggml_cuda_mm_fusion_args_device::aux_dst); an
        // in-place alias of aux_src is fine, each element is read and written by the same thread
        if (fusion.aux_dst && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            for (uint32_t e = tid; e < fusion.aux_n; e += nwarps*warp_size) {
                fusion.aux_dst[e] = 1.0f / (1.0f + expf(-fusion.aux_src[e]));
            }
        }
    }

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    [[maybe_unused]] float x_mul_v = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (fusion.x_mul) {   // halo-hybrid: KDA gate tail multiplier, same thread as the row's epilogue
                x_mul_v = fusion.x_mul[(row0 + threadIdx.x) / fusion.x_mul_div];
            }
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * fusion.x_bias_stride_col + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * fusion.gate_bias_stride_col + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            constexpr int pf_dist = 2; // loop iterations, not blocks
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    mmvq_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            mmvq_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                    if (fusion.tail_act) {   // halo-hybrid: KDA gate tail, same op order as the unfused chain
                        result *= x_mul_v;
                        result = fusion.tail_s0 * result + fusion.tail_b0;
                        if (fusion.tail_act == 1) {
                            result = 1.0f / (1.0f + expf(-result));
                        }
                        result = fusion.tail_s1 * result + fusion.tail_b1;
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    uint32_t channel_x = ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);


    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr ||
                            fusion.tail_act != 0 || fusion.aux_dst != nullptr;
    // halo-hybrid: fused gate/up + GLU up to 4 columns (the verify batch of an MTP draft); see
    // ggml_cuda_should_fuse_mul_mat_vec_q for why
    if constexpr (c_ncols_dst <= 4) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst<=4");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

// halo-hybrid: grouped MoE GEMV (GGML_CUDA_MMVQ_GROUPED=1). mul_mat_vec_q_moe gives every (token, expert slot) pair its
// own warp, so an expert chosen by several tokens of a speculative verify batch is streamed from memory once per token:
// on gfx1151 the per-pair kernel runs at DRAM speed (234-257 GB/s at 4 tokens) on largely redundant bytes, and MMQ,
// which reads each distinct expert once, loses most of that back to its tile overhead at 3-4 tokens. Here one tiny
// kernel groups the pairs by expert and the GEMV runs one block per (row block, distinct expert), reading the expert's
// rows once and applying them to every token that routed to it.
#define MMVQ_GRP_MAX_PAIRS 256 // n_tokens * n_expert_used
#define MMVQ_GRP_MAX_COLS  MMVQ_MAX_BATCH_SIZE

// grp: [0] = number of distinct experts D, then D records of (2 + max_cols) ints: expert, count, count x (token << 16 | slot)
// in token-major pair order (the first token's experts first). A token never routes to one expert twice, so count <=
// n_tokens <= max_cols; malformed ids are clamped rather than written out of bounds.
static __global__ void mmvq_moe_group_ids(const int32_t * __restrict__ ids, int32_t * __restrict__ grp,
        const int n_tokens, const int n_slots, const int ids_stride, const int max_cols) {
    __shared__ int32_t e_sh[MMVQ_GRP_MAX_PAIRS];
    __shared__ int32_t lead_sh[MMVQ_GRP_MAX_PAIRS];

    const int p = threadIdx.x;
    const int P = n_tokens*n_slots;

    int e = -1;
    if (p < P) {
        e = ids[(p % n_slots) + (p / n_slots)*ids_stride];
    }
    e_sh[p] = e;
    __syncthreads();

    bool leader = p < P;
    for (int q = 0; q < p && leader; ++q) {
        leader = e_sh[q] != e;
    }
    lead_sh[p] = leader;
    __syncthreads();

    if (leader) {
        int d = 0;
        for (int q = 0; q < p; ++q) {
            d += lead_sh[q];
        }
        int32_t * rec = grp + 1 + d*(2 + max_cols);
        int cnt = 0;
        for (int q = p; q < P && cnt < max_cols; ++q) {
            if (e_sh[q] == e) {
                rec[2 + cnt++] = ((q / n_slots) << 16) | (q % n_slots);
            }
        }
        rec[0] = e;
        rec[1] = cnt;
    }
    if (p == 0) {
        int D = 0;
        for (int q = 0; q < P; ++q) {
            D += lead_sh[q];
        }
        grp[0] = D;
    }
}

// Grid: (ceil(nrows_x / c_rows_per_block), n_tokens*n_slots) - blockIdx.y = distinct expert d (blocks past D exit).
// Block: one warp; it reads c_rows_per_block rows of the expert once per K step and dots them with up to c_max_cols
// activations. Same epilogue (fused gate, biases, GLU) as mul_mat_vec_q_moe.
template <ggml_type type, int c_rows_per_block, int c_max_cols, bool has_fusion>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe_grouped(
        const void * vx_ptr, const void * vy_ptr, const int32_t * grp_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT grp = grp_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
    }

    ggml_cuda_pdl_sync();

    const int d = blockIdx.y;
    if (d >= grp[0]) {
        return;
    }
    const int32_t * rec = grp + 1 + d*(2 + c_max_cols);
    const uint32_t channel_x = rec[0];
    const int      cnt       = rec[1];

    const int row0             = c_rows_per_block*blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * warp_size / qi;

    const block_q8_1 * y[c_max_cols];
    uint32_t           dst_off[c_max_cols];
#pragma unroll
    for (int j = 0; j < c_max_cols; ++j) {
        const int      pk  = rec[2 + (j < cnt ? j : 0)];
        const uint32_t tok = pk >> 16;
        const uint32_t slt = pk & 0xFFFF;
        y[j]       = ((const block_q8_1 *) vy) + fastmodulo(slt, nchannels_y)*stride_channel_y + tok*stride_col_y;
        dst_off[j] = slt*stride_channel_dst + tok*stride_col_dst;
    }

    const int kbx_offset = channel_x*stride_channel_x + row0*stride_row_x;

    float tmp[c_max_cols][c_rows_per_block]      = {{0.0f}};
    float tmp_gate[c_max_cols][c_rows_per_block] = {{0.0f}};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int j = 0; j < c_max_cols; ++j) {
            if (j < cnt) {
#pragma unroll
                for (int i = 0; i < c_rows_per_block; ++i) {
                    tmp[j][i] += vec_dot_q_cuda(vx, &y[j][kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            tmp_gate[j][i] += vec_dot_q_cuda(vgate, &y[j][kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

#pragma unroll
    for (int j = 0; j < c_max_cols; ++j) {
        if (j >= cnt) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }
        }

        if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
            float result = tmp[j][threadIdx.x];
            if constexpr (has_fusion) {
                const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;
                if (x_bias) {
                    result += x_bias[bias_idx];
                }
                if (use_gate) {
                    float gate_value = tmp_gate[j][threadIdx.x];
                    if (gate_bias) {
                        gate_value += gate_bias[bias_idx];
                    }
                    switch (active_glu) {
                        case GGML_GLU_OP_SWIGLU:
                            result *= ggml_cuda_op_silu_single(gate_value);
                            break;
                        case GGML_GLU_OP_GEGLU:
                            result *= ggml_cuda_op_gelu_single(gate_value);
                            break;
                        case GGML_GLU_OP_SWIGLU_OAI:
                            result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                            break;
                        case GGML_GLU_OP_SWIGLU_CLAMP:
                            result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                            break;
                        default:
                            result = result * gate_value;
                            break;
                    }
                }
            }
            dst[dst_off[j] + row0 + threadIdx.x] = result;
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit);
    }
}

template <ggml_type type, int c_max_cols>
static void mul_mat_vec_q_moe_grouped_launch_cols(
        const void * vx, const void * vy, const int32_t * grp, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const int n_pairs, const int warp_size, cudaStream_t stream) {
    constexpr int rows_per_block = 2;
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, n_pairs);
    const dim3 block_dims(warp_size, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_grouped<type, rows_per_block, c_max_cols, true>, launch_params,
            vx, vy, grp, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe_grouped<type, rows_per_block, c_max_cols, false>, launch_params,
            vx, vy, grp, fusion, dst, ncols_x, nchannels_y, nrows_x, stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst);
    }
}

bool ggml_cuda_mmvq_moe_grouped_enabled(ggml_type type) {
    static const bool enabled = getenv("GGML_CUDA_MMVQ_GROUPED") != nullptr && atoi(getenv("GGML_CUDA_MMVQ_GROUPED")) != 0;
    if (!enabled) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// grouping (n_tokens*n_slots <= MMVQ_GRP_MAX_PAIRS) + grouped GEMV; the caller checked ggml_cuda_mmvq_moe_grouped_enabled.
// With the env set, get_mmvq_mmid_max_batch admits up to MMVQ_MAX_BATCH_SIZE tokens for these types, which the per-pair
// kernel's launch bounds do not cover: n_used <= MMVQ_GRP_MAX_PAIRS / MMVQ_MAX_BATCH_SIZE (32) keeps every such call here.
static void ggml_cuda_mul_mat_vec_q_moe_grouped(ggml_backend_cuda_context & ctx,
        const void * vx, const ggml_type type, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst, const int ncols_x, const int nrows_x, const int n_tokens, const int n_slots, const int ids_stride,
        const int nchannels_y, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst, cudaStream_t stream) {
    const int n_pairs  = n_tokens*n_slots;
    const int max_cols = n_tokens <= 4 ? 4 : MMVQ_GRP_MAX_COLS;
    GGML_ASSERT(n_pairs <= MMVQ_GRP_MAX_PAIRS && n_tokens <= MMVQ_GRP_MAX_COLS);

    ggml_cuda_pool_alloc<int32_t> grp(ctx.pool(), 1 + (size_t) n_pairs*(2 + max_cols));
    mmvq_moe_group_ids<<<1, MMVQ_GRP_MAX_PAIRS, 0, stream>>>(ids, grp.get(), n_tokens, n_slots, ids_stride, max_cols);
    CUDA_CHECK(cudaGetLastError());

    const int device    = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const uint3 nchannels_y_fd = init_fastdiv_values(nchannels_y);

#define MMVQ_GRP_CASE(T) \
    case T: \
        if (max_cols == 4) { \
            mul_mat_vec_q_moe_grouped_launch_cols<T, 4>(vx, vy, grp.get(), fusion, dst, ncols_x, nchannels_y_fd, nrows_x, \
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, \
                n_pairs, warp_size, stream); \
        } else { \
            mul_mat_vec_q_moe_grouped_launch_cols<T, MMVQ_GRP_MAX_COLS>(vx, vy, grp.get(), fusion, dst, ncols_x, nchannels_y_fd, nrows_x, \
                stride_row_x, stride_col_y, stride_col_dst, stride_channel_x, stride_channel_y, stride_channel_dst, \
                n_pairs, warp_size, stream); \
        } \
        break;

    switch (type) {
        MMVQ_GRP_CASE(GGML_TYPE_Q4_0)
        MMVQ_GRP_CASE(GGML_TYPE_Q8_0)
        MMVQ_GRP_CASE(GGML_TYPE_Q4_K)
        MMVQ_GRP_CASE(GGML_TYPE_Q5_K)
        MMVQ_GRP_CASE(GGML_TYPE_Q6_K)
        default:
            GGML_ABORT("grouped MoE GEMV: unsupported type %s", ggml_type_name(type));
    }
#undef MMVQ_GRP_CASE
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                (GGML_CUDA_CC_IS_RDNA(cc) && !(GGML_CUDA_CC_IS_RDNA4(cc) && ggml_cuda_mmvq_rdna4_small_k()))) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2:
        case 3:
        case 4: {
            // halo-hybrid: RDNA4 gets the 8-warp block here too (split-K over the block, or rows-per-block for
            // short K); every other table keeps upstream's launch. Tag types keep the flags compile-time.
            const bool rdna4_wide = table_id == MMVQ_PARAMETERS_RDNA4 && ggml_cuda_mmvq_rdna4_wide();
            const auto launch_n = [&](auto ncols_tag) {
                static constexpr int c_ncols_dst = decltype(ncols_tag)::value;
                const auto launch = [&](auto small_k_tag, auto wide_tag) {
                    // Types the RDNA4 table does not promote would compile a second, identical kernel.
                    constexpr bool c_promoted = calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_RDNA4, false, true) > 1;
                    constexpr bool c_small_k  = decltype(small_k_tag)::value && c_promoted;   // = tall on RDNA4
                    constexpr bool c_wide     = decltype(wide_tag)::value && c_promoted;
                    const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                                  nsamples_dst, warp_size, table_id, c_small_k, c_wide);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_wide>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                        channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                        stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                        stream);
                };
                if (rdna4_wide && calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_RDNA4, false, true) > 1) {
                    // wide (8 warps, 1 row/block) when the 8-way K split still gives every warp a full trip and
                    // the matrix is under 2^25 weights; otherwise upstream (1 wave, 1 row). In situ on the R9700
                    // (v3s decode, n=3, op timer): 24x16384 30.8 -> 14.8 us, 2048x4096 39.1 -> 28.6,
                    // 4096x2048 38.7 -> 27.8; >= 2^25 weights unchanged (enough waves, reduction costs 3-5%);
                    // K=256/512 shapes LOST 1.3-2.1x under wide (idle warps + reduction) and K=128 gained nothing
                    // from a rows-per-block variant, so short K keeps the upstream launch.
                    // gfx1201 has a dispatch cliff when a launch's wave count lands within a few of the part's
                    // 2048 wave slots (32 WGP x 4 SIMD32 x 16): 2048x4096 at one wave per row took 19.5 us where
                    // 2032 rows took 9.8, and 256 rows x 8 warps took 19.7 where 252 took 3.7. The APU (1280
                    // slots) has no such cliff. A shape that would land there takes the other launch.
                    // HIP reports RDNA WGPs as multiProcessorCount (32 on gfx1201): 4 SIMD32 x 16 waves each.
                    const int64_t slots    = (int64_t) ggml_cuda_info().devices[device].nsm * 64;
                    const int64_t launches = (int64_t) nchannels_dst * nsamples_dst;
                    const auto on_cliff = [&](int64_t waves) { return waves >= slots - 12 && waves <= slots + 3; };
                    const bool long_k = blocks_per_row_x >= 8 * blocks_per_iter_1warp;
                    // halo-hybrid experiment: GGML_CUDA_MMVQ_BIG_WIDE=1 keeps the wide launch for >= 2^25 weights too
                    // (in situ the 8192-row K=4096 KDA q projection ran at ~170 GB/s under the one-wave-per-row launch)
                    static const bool big_wide = getenv("GGML_CUDA_MMVQ_BIG_WIDE") != nullptr;
                    const bool big    = !big_wide && (int64_t) nrows_x * ncols_x >= (1 << 25);
                    const int64_t w_wide = (int64_t) nrows_x * 8 * launches;
                    const int64_t w_up   = (int64_t) nrows_x * launches;
                    bool wide = long_k && !big;
                    if (wide ? on_cliff(w_wide) : on_cliff(w_up)) {
                        wide = !wide;
                    }
                    // tall: <= 64 rows over K >= 8192 (q8_0), 32 warps split the K loop 4x deeper than wide
                    static const bool no_tall = getenv("GGML_CUDA_MMVQ_NO_TALL") != nullptr;
                    const bool tall = !no_tall && wide && nrows_x <= 64 && blocks_per_row_x >= 32 * blocks_per_iter_1warp &&
                        calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_RDNA4, true, false) > 1 && !on_cliff((int64_t) nrows_x * 32 * launches);
                    static const bool dbg = getenv("GGML_CUDA_MMVQ_DEBUG") != nullptr;
                    if (dbg) {
                        fprintf(stderr, "mmvq rdna4: rows=%d K=%d n=%d ch*s=%lld long_k=%d big=%d -> %s\n",
                            nrows_x, ncols_x, c_ncols_dst, (long long) launches, long_k, big, tall ? "tall" : wide ? "wide" : "upstream");
                    }
                    if (tall) {
                        launch(std::true_type{},  std::false_type{});
                    } else if (wide) {
                        launch(std::false_type{}, std::true_type{});
                    } else {
                        launch(std::false_type{}, std::false_type{});
                    }
                } else {
                    launch(std::false_type{}, std::false_type{});
                }
            };
            switch (ncols_dst) {
                case 2: launch_n(std::integral_constant<int, 2>{}); break;
                case 3: launch_n(std::integral_constant<int, 3>{}); break;
                default: launch_n(std::integral_constant<int, 4>{}); break;
            }
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// halo-hybrid: ONE launch for several GEMVs that read the same activation (q/k/v, qkv+gate, the shared expert's
// gate/up, the per-layer-embedding key/value pair, the hyper-connection down-projection with its inject). The caller
// has already quantized the activation once; this also removes the per-matrix dispatch and lets the combined row
// space fill the GPU in a single ramp.
//
// Blocks are 8 waves. A quantized member gets 8 rows per block, one wave per row (measured on gfx1151: 1..16 rows
// per workgroup all stream at the same 225 GB/s, so this costs nothing). An f32 member gets ONE row per block with
// all 8 waves splitting K and an LDS reduction, which is what mul_mat_vec_f gives it; one wave per f32 row was a
// 7.6 ms/token regression on the 4-row hyper-connection inject. The block's member is found by a short
// block-uniform scan over the per-member block prefix sums.
#define MMVQ_GROUP_MAX 8
#define MMVQ_GROUP_WAVES 8

struct mmvq_group_args {
    const void * vx[MMVQ_GROUP_MAX];
    float *      dst[MMVQ_GROUP_MAX];
    int32_t      stride_row[MMVQ_GROUP_MAX];       // weight row stride, in blocks (f32 members: in floats)
    int32_t      stride_col_dst[MMVQ_GROUP_MAX];   // dst column stride, in floats
    int32_t      nrows[MMVQ_GROUP_MAX];
    int32_t      block_end[MMVQ_GROUP_MAX];        // exclusive prefix sums of the block counts
    int32_t      n;
    uint32_t     f32_mask;                         // members whose weights are f32: dot the f32 activation instead
    uint32_t     glu_mask;                         // members that are a (gate, up) pair: vx is the gate, vx2 the up
    const void * vx2[MMVQ_GROUP_MAX];
    int32_t      glu_op;                           // ggml_glu_op of the pairs (all pairs in a group share it)
    const float * vy_f32;                          // the unquantized activation, for those members
    int32_t      stride_col_y_f32;
};

template <ggml_type type, int ncols_dst>
__global__ void __launch_bounds__(MMVQ_GROUP_WAVES * ggml_cuda_get_physical_warp_size(), 1)
mul_mat_vec_q_group(const mmvq_group_args a, const void * __restrict__ vy,
                    const uint32_t ncols_x, const uint32_t stride_col_y) {
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int qi        = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr       = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);
    constexpr int blocks_per_iter = vdr * warp_size / qi;
    constexpr int nwarps = MMVQ_GROUP_WAVES;
    constexpr int block_threads = nwarps * warp_size;

    __shared__ float s_part[nwarps][ncols_dst];

    const int blk = blockIdx.x;
    int e = 0;
#pragma unroll
    for (int k = 1; k < MMVQ_GROUP_MAX; ++k) {
        if (k < a.n && blk >= a.block_end[k - 1]) {
            e = k;
        }
    }
    const int lblk = blk - (e == 0 ? 0 : a.block_end[e - 1]);   // block index within the member
    const int tid  = threadIdx.x;
    const int lane = tid % warp_size, wave = tid / warp_size;

    if (a.f32_mask & (1u << e)) {   // block-uniform: an f32 member, one row per block, the 8 waves split K
        const int row = lblk;
        float acc[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[j] = 0.0f;
        }
        const float * w = (const float *) a.vx[e] + (int64_t) row * a.stride_row[e];
        const bool vec4 = (ncols_x % 4 == 0) && (((uintptr_t) w | (uintptr_t) a.vy_f32) % 16 == 0) && (a.stride_col_y_f32 % 4 == 0);
        if (vec4) {
            const int n4 = (int) ncols_x / 4;
            for (int c4 = tid; c4 < n4; c4 += block_threads) {
                const float4 wv = ((const float4 *) w)[c4];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const float4 yv = ((const float4 *) (a.vy_f32 + j*a.stride_col_y_f32))[c4];
                    acc[j] += wv.x*yv.x + wv.y*yv.y + wv.z*yv.z + wv.w*yv.w;
                }
            }
        } else {
            for (int col = tid; col < (int) ncols_x; col += block_threads) {
                const float wv = w[col];
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    acc[j] += wv * a.vy_f32[j*a.stride_col_y_f32 + col];
                }
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            acc[j] = warp_reduce_sum<warp_size>(acc[j]);
            if (lane == 0) {
                s_part[wave][j] = acc[j];
            }
        }
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                float v = 0.0f;
#pragma unroll
                for (int k = 0; k < nwarps; ++k) {
                    v += s_part[k][j];
                }
                a.dst[e][j*a.stride_col_dst[e] + row] = v;
            }
        }
        return;
    }

    // a quantized member: 8 rows per block, this wave owns one of them
    const int row = lblk * nwarps + wave;
    if (row >= a.nrows[e]) {
        return;
    }
    const int blocks_per_row_x = ncols_x / qk;
    const block_q8_1 * y = (const block_q8_1 *) vy;
    float tmp[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = 0.0f;
    }
    const int kqs = vdr * (lane % (qi/vdr));
    if (a.glu_mask & (1u << e)) {   // a (gate, up) pair: both rows in this wave, the GLU applied on the way out
        float tmp2[ncols_dst];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp2[j] = 0.0f;
        }
        for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
            const int kby = kbx * (qk/QK8_1);
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j]  += vec_dot_q_cuda(a.vx[e],  &y[j*stride_col_y + kby], row*a.stride_row[e] + kbx, kqs);
                tmp2[j] += vec_dot_q_cuda(a.vx2[e], &y[j*stride_col_y + kby], row*a.stride_row[e] + kbx, kqs);
            }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp[j]  = warp_reduce_sum<warp_size>(tmp[j]);
            tmp2[j] = warp_reduce_sum<warp_size>(tmp2[j]);
            if (lane == 0) {
                const float g = a.glu_op == GGML_GLU_OP_GEGLU ? ggml_cuda_op_gelu_single(tmp[j]) : ggml_cuda_op_silu_single(tmp[j]);
                a.dst[e][j*a.stride_col_dst[e] + row] = g * tmp2[j];
            }
        }
        return;
    }
    for (int kbx = lane / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            tmp[j] += vec_dot_q_cuda(a.vx[e], &y[j*stride_col_y + kby], row*a.stride_row[e] + kbx, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        tmp[j] = warp_reduce_sum<warp_size>(tmp[j]);
        if (lane == 0) {
            a.dst[e][j*a.stride_col_dst[e] + row] = tmp[j];
        }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_group_ncols(const mmvq_group_args & a, const void * vy, int64_t ncols_x,
                                      int64_t stride_col_y, int64_t ncols_dst, int blocks, int warp_size, cudaStream_t stream) {
    const dim3 block_nums(blocks, 1, 1);
    const dim3 block_dims(MMVQ_GROUP_WAVES * warp_size, 1, 1);
    switch (ncols_dst) {
        case 1: mul_mat_vec_q_group<type, 1><<<block_nums, block_dims, 0, stream>>>(a, vy, ncols_x, stride_col_y); break;
        case 2: mul_mat_vec_q_group<type, 2><<<block_nums, block_dims, 0, stream>>>(a, vy, ncols_x, stride_col_y); break;
        case 3: mul_mat_vec_q_group<type, 3><<<block_nums, block_dims, 0, stream>>>(a, vy, ncols_x, stride_col_y); break;
        case 4: mul_mat_vec_q_group<type, 4><<<block_nums, block_dims, 0, stream>>>(a, vy, ncols_x, stride_col_y); break;
        default: GGML_ABORT("mul_mat_vec_q_group: ncols_dst %d", (int) ncols_dst);
    }
}

// true when the group ran; the caller falls back to one launch per matrix otherwise
bool ggml_cuda_mul_mat_vec_q_group(ggml_backend_cuda_context & ctx, ggml_tensor ** nodes, int n,
                                   const ggml_tensor * src1, const char * src1_q8_1) {
    static const bool disabled = getenv("GGML_CUDA_NO_GEMV_GROUP") != nullptr && std::atoi(getenv("GGML_CUDA_NO_GEMV_GROUP"));
    if (disabled || n < 2 || n > MMVQ_GROUP_MAX || !src1_q8_1) {
        return false;
    }
    ggml_type type = GGML_TYPE_COUNT;              // the one quantized type of the group (f32 entries ride along)
    for (int k = 0; k < n; ++k) {
        // a GLU node stands for its (gate, up) pair, both quantized alike (the detector checked the shape)
        const ggml_tensor * mm = nodes[k]->op == GGML_OP_GLU ? nodes[k]->src[0] : nodes[k];
        const ggml_type t = mm->src[0]->type;
        if (t == GGML_TYPE_F32) {
            continue;
        }
        if (type != GGML_TYPE_COUNT && t != type) {
            return false;
        }
        type = t;
    }
    if (type == GGML_TYPE_COUNT) {
        return false;                              // all-f32 groups keep the existing mul_mat_vec_f path
    }
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    if (ne11 < 1 || ne11 > 4 || ne10 % ggml_blck_size(type) != 0) {
        return false;
    }
    mmvq_group_args a{};
    a.n = n;
    int blocks = 0;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * d = nodes[k];
        const ggml_tensor * mm = d;
        if (d->op == GGML_OP_GLU) {
            const ggml_tensor * gate = d->src[0], * up = d->src[1];
            const ggml_glu_op op = ggml_get_glu_op(d);
            if ((op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU) || ggml_get_op_params_i32(d, 1) != 0 ||
                gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT || gate->src[1] != src1 || up->src[1] != src1 ||
                up->src[0]->type != gate->src[0]->type || !ggml_is_contiguous(up->src[0]) ||
                up->src[0]->ne[0] != gate->src[0]->ne[0] || up->src[0]->ne[1] != gate->src[0]->ne[1] ||
                up->src[0]->nb[1] != gate->src[0]->nb[1] || (a.glu_mask && a.glu_op != (int32_t) op)) {
                return false;
            }
            mm = gate;
            a.glu_mask |= 1u << k;
            a.glu_op    = (int32_t) op;
            a.vx2[k]    = up->src[0]->data;
        }
        const ggml_tensor * w = mm->src[0];
        const bool is_f32 = w->type == GGML_TYPE_F32;
        if ((!is_f32 && w->type != type) || w->ne[0] != ne10 || !ggml_is_contiguous(w)) {
            return false;
        }
        if (is_f32) {
            if (!ggml_is_contiguous(src1) || src1->nb[0] != sizeof(float)) {
                return false;
            }
            a.f32_mask |= 1u << k;
        }
        if (d->type != GGML_TYPE_F32 || d->nb[0] != sizeof(float) || d->ne[1] != ne11 || d->ne[2] != 1 || d->ne[3] != 1) {
            return false;
        }
        if (w->ne[1] > INT32_MAX/2) {
            return false;
        }
        a.vx[k]             = w->data;
        a.dst[k]            = (float *) d->data;
        a.stride_row[k]     = (int32_t) (w->nb[1] / ggml_type_size(w->type));
        a.stride_col_dst[k] = (int32_t) (d->nb[1] / sizeof(float));
        a.nrows[k]          = (int32_t) w->ne[1];
        blocks             += is_f32 ? (int32_t) w->ne[1] : (int32_t) ((w->ne[1] + MMVQ_GROUP_WAVES - 1) / MMVQ_GROUP_WAVES);
        a.block_end[k]      = blocks;
    }
    const int64_t ne10_padded  = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t stride_col_y = ne10_padded / QK8_1;
    a.vy_f32           = (const float *) src1->data;
    a.stride_col_y_f32 = (int32_t) (src1->nb[1] / sizeof(float));

    ggml_cuda_set_device(ctx.device);
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    switch (type) {
        case GGML_TYPE_Q4_0: mul_mat_vec_q_group_ncols<GGML_TYPE_Q4_0>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q4_1: mul_mat_vec_q_group_ncols<GGML_TYPE_Q4_1>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q5_0: mul_mat_vec_q_group_ncols<GGML_TYPE_Q5_0>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q5_1: mul_mat_vec_q_group_ncols<GGML_TYPE_Q5_1>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q8_0: mul_mat_vec_q_group_ncols<GGML_TYPE_Q8_0>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q4_K: mul_mat_vec_q_group_ncols<GGML_TYPE_Q4_K>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q5_K: mul_mat_vec_q_group_ncols<GGML_TYPE_Q5_K>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_Q6_K: mul_mat_vec_q_group_ncols<GGML_TYPE_Q6_K>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_IQ4_NL: mul_mat_vec_q_group_ncols<GGML_TYPE_IQ4_NL>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        case GGML_TYPE_IQ4_XS: mul_mat_vec_q_group_ncols<GGML_TYPE_IQ4_XS>(a, src1_q8_1, ne10, stride_col_y, ne11, blocks, warp_size, ctx.stream()); break;
        default: return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, const char * src1_q8_1_pre) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        // halo-hybrid: up to 4 columns (the kernel's epilogue is generic over ncols_dst; a fused ADD operand is
        // either a [rows] bias or a [rows, n_tokens] tensor, see x_bias_stride_col)
        GGML_ASSERT(  ids || dst->ne[1] <= 4);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            GGML_ASSERT( ids || fusion->x_bias->ne[1] == 1 || fusion->x_bias->ne[1] == dst->ne[1]);
            fusion_local.x_bias = fusion->x_bias->data;
            fusion_local.x_bias_stride_col = (!ids && fusion->x_bias->ne[1] > 1) ? (uint32_t) (fusion->x_bias->nb[1] / sizeof(float)) : 0;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            GGML_ASSERT( ids || fusion->gate_bias->ne[1] == 1 || fusion->gate_bias->ne[1] == dst->ne[1]);
            fusion_local.gate_bias = fusion->gate_bias->data;
            fusion_local.gate_bias_stride_col = (!ids && fusion->gate_bias->ne[1] > 1) ? (uint32_t) (fusion->gate_bias->nb[1] / sizeof(float)) : 0;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
        if (fusion->tail_act || fusion->aux_dst) {   // halo-hybrid: KDA gate prologue, plain MUL_MAT only
            GGML_ASSERT(!ids && fusion->gate == nullptr);
            fusion_local.tail_act  = fusion->tail_act;
            fusion_local.tail_s0   = fusion->tail_s0;
            fusion_local.tail_b0   = fusion->tail_b0;
            fusion_local.tail_s1   = fusion->tail_s1;
            fusion_local.tail_b1   = fusion->tail_b1;
            fusion_local.x_mul     = fusion->x_mul ? (const float *) fusion->x_mul->data : nullptr;
            fusion_local.x_mul_div = fusion->x_mul_div;
            if (fusion->aux_dst) {
                GGML_ASSERT(fusion->aux_src && ggml_nelements(fusion->aux_src) == ggml_nelements(fusion->aux_dst));
                fusion_local.aux_src = (const float *) fusion->aux_src->data;
                fusion_local.aux_dst = (float *) fusion->aux_dst->data;
                fusion_local.aux_n   = (uint32_t) ggml_nelements(fusion->aux_dst);
            }
        }
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool());
    const char * src1_q8_1_d = src1_q8_1_pre;
    const bool side_ok = ne11 <= 8 && ne12 == 1 && ne13 == 1 && src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1);
    if (!src1_q8_1_d && side_ok) {
        src1_q8_1_d = ggml_cuda_q8_side_find(ctx, src1);   // a producer or an earlier GEMV already wrote the q8_1 copy
    }
    if (!src1_q8_1_d) {
        // halo-hybrid: at decode and verify widths the same activation feeds several GEMVs (q/k/v/f/g/beta of a KDA
        // layer, the q_a/kv_a/indexer projections of a DSA layer): quantize it once into the side arena and register it
        block_q8_1 * side = side_ok ? ggml_cuda_q8_side_reserve_rows(ctx, src1, ne10, ne11, ne10_padded) : nullptr;
        char * q8 = (char *) side;
        if (!q8) {
            src1_q8_1.alloc(ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
            q8 = src1_q8_1.get();
        }
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        {   // halo-hybrid: GGML_CUDA_GEMV_GROUPS=1 names any activation that still needs a standalone quantize launch
            // (none during decode on Qwen3.8: the producer-side q8_1 registry and the shared-activation group cover it)
            static const int who = getenv("GGML_CUDA_GEMV_GROUPS") ? atoi(getenv("GGML_CUDA_GEMV_GROUPS")) : 0;
            if (who) {
                GGML_LOG_WARN("q8-quantize: %s <- %s (%s) ne %lld x %lld, consumer %s\n",
                    src1->name, src1->src[0] ? src1->src[0]->name : "-", ggml_op_name(src1->op),
                    (long long) ne10, (long long) ne11, dst->name);
            }
        }
        quantize_row_q8_1_cuda(src1_d, nullptr, q8, src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        src1_q8_1_d = q8;
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    if (ids && ncols_dst > 1 && ne03 == 1 && ggml_cuda_mmvq_moe_grouped_enabled(src0->type) &&
            ncols_dst*nchannels_dst <= MMVQ_GRP_MAX_PAIRS) {
        ggml_cuda_mul_mat_vec_q_moe_grouped(ctx, src0->data, src0->type, src1_q8_1_d, ids_d, fusion_local, dst_d, ne00, ne01,
            ncols_dst, nchannels_dst, ids_stride, nchannels_y, s01, stride_col_y, stride_col_dst,
            s02, stride_channel_y, stride_channel_dst, stream);
        return;
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1_d, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
