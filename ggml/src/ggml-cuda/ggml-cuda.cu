#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/sgemm-tile.cuh"
#include "ggml-cuda/mmq-wmma.cuh"
#include "ggml-cuda/ewchain.cuh"
#include "ggml-cuda/persist.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/hc.cuh"
#include "ggml-cuda/kpool-compress.cuh"
#include "ggml-cuda/f32act.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/kq-mask.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

#ifdef GGML_USE_HIP
    // rocBLAS 7.x hands f32 GEMMs to hipBLASLt, whose gfx1201 sgemm solutions are 8x8 macro-tile fallbacks
    // (a [2560 -> 512] x 1024 sgemm runs at 2.5 TFLOPS; rocBLAS' own Tensile kernels reach 11).
    // Prefer rocBLAS unless the user chose explicitly.
    if (getenv("ROCBLAS_USE_HIPBLASLT") == nullptr) {
        setenv("ROCBLAS_USE_HIPBLASLT", "0", 1);
    }
#endif // GGML_USE_HIP

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    for (int i = 0; i < GGML_CUDA_COPY_EVENTS; ++i) {
        if (copy_events[i] != nullptr) {
            CUDA_CHECK(cudaEventDestroy(copy_events[i]));
        }
    }
    if (q8_arena != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(q8_arena));
    }
    if (hc_mix_scratch != nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(hc_mix_scratch));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
        // halo-hybrid: rocBLAS loads Tensile modules lazily, and on gfx1201 a first-choice load can return
        // hipErrorNoBinaryForGpu ("Cannot Find Global Var Sizes / Cannot create kernels" in an AMD_LOG_LEVEL=3
        // trace). rocBLAS HANDLES that by falling back to another solution, which launches and returns success -
        // but HIP's sticky per-thread error survives, and the cudaGetLastError() at the end of
        // ggml_cuda_compute_forward would attribute it to this op and abort a matmul that computed correctly.
        // CUBLAS_CHECK above has already validated the real outcome, so discard the handled error here.
        // Clearing at the START of compute_forward does NOT work (measured): the load happens during this call.
        // gfx1201 hits this and gfx1151 does not because gfx1201's rocBLAS ships far fewer tuned Tensile kernels
        // (56 files against 96), so first-choice loads miss and fall back far more often.
        // See docs/halo-hybrid/GFX1201-INDEXER-BUG.md.
        (void) cudaGetLastError();
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // halo-hybrid: rocBLAS loads Tensile modules lazily, and on gfx1201 a first-choice load can return
        // hipErrorNoBinaryForGpu ("Cannot Find Global Var Sizes / Cannot create kernels" in an AMD_LOG_LEVEL=3
        // trace). rocBLAS HANDLES that by falling back to another solution, which launches and returns success -
        // but HIP's sticky per-thread error survives, and the cudaGetLastError() at the end of
        // ggml_cuda_compute_forward would attribute it to this op and abort a matmul that computed correctly.
        // CUBLAS_CHECK above has already validated the real outcome, so discard the handled error here.
        // Clearing at the START of compute_forward does NOT work (measured): the load happens during this call.
        // gfx1201 hits this and gfx1151 does not because gfx1201's rocBLAS ships far fewer tuned Tensile kernels
        // (56 files against 96), so first-choice loads miss and fall back far more often.
        // See docs/halo-hybrid/GFX1201-INDEXER-BUG.md.
        (void) cudaGetLastError();
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // halo-hybrid: rocBLAS loads Tensile modules lazily, and on gfx1201 a first-choice load can return
        // hipErrorNoBinaryForGpu ("Cannot Find Global Var Sizes / Cannot create kernels" in an AMD_LOG_LEVEL=3
        // trace). rocBLAS HANDLES that by falling back to another solution, which launches and returns success -
        // but HIP's sticky per-thread error survives, and the cudaGetLastError() at the end of
        // ggml_cuda_compute_forward would attribute it to this op and abort a matmul that computed correctly.
        // CUBLAS_CHECK above has already validated the real outcome, so discard the handled error here.
        // Clearing at the START of compute_forward does NOT work (measured): the load happens during this call.
        // gfx1201 hits this and gfx1151 does not because gfx1201's rocBLAS ships far fewer tuned Tensile kernels
        // (56 files against 96), so first-choice loads miss and fall back far more often.
        // See docs/halo-hybrid/GFX1201-INDEXER-BUG.md.
        (void) cudaGetLastError();
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // halo-hybrid: same handled-rocBLAS-error clear as the other three GEMM entry points above. This is the
        // pointer-array path, taken when the batch is not uniformly strided; it has the same lazy Tensile load
        // underneath and would fail the same way. Note the CUDA_CHECK above is a check, not a clear, so a handled
        // error left by THIS call would surface at the end of ggml_cuda_compute_forward, which is the original bug.
        (void) cudaGetLastError();
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(ggml_cuda_info().devices[ctx.device].cc)) {
        compute_type = GGML_TYPE_F32;
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    static constexpr std::array<ggml_glu_op, 4> valid_glu_ops = { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP };

    if (std::find(valid_glu_ops.begin(), valid_glu_ops.end(), ggml_get_glu_op(glu)) == valid_glu_ops.end()) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // halo-hybrid: upstream fuses gate/up + GLU into mul_mat_vec_q only at ncols_dst = 1 and otherwise hands the
    // pair to the fused MMQ path, which at a 3-token verify batch read the 71 MB pair at ~350 GB/s on gfx1201
    // (201 us vs 2 x 74 us for the plain GEMVs). The mmvq kernel's fused epilogue is generic over ncols_dst, so
    // allow it up to 4 columns. GGML_CUDA_MMVQ_FUSE_N1=1 restores the upstream rule.
    static const bool fuse_n1_only = getenv("GGML_CUDA_MMVQ_FUSE_N1") != nullptr && atoi(getenv("GGML_CUDA_MMVQ_FUSE_N1")) != 0;
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1 && (fuse_n1_only || dst->ne[1] > 4)) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    // halo-hybrid: GGML_CUDA_TRACE_MM=<substring> logs the dispatch decision for matching weights. Added while
    // chasing a "no kernel image" on gfx1201 that only appears with whole-layer placement plus a draft head.
    {
        static const char * trace = getenv("GGML_CUDA_TRACE_MM");
        if (trace && src0->name[0] && strstr(src0->name, trace)) {
            const int dev_now = ggml_cuda_get_device();
            const int cc_now  = ggml_cuda_info().devices[dev_now].cc;
            const int cc_ctx  = ggml_cuda_info().devices[ctx.device].cc;
            GGML_LOG_ERROR("TRACE_MM %s: ctx.device=%d cc_ctx=%d | current_device=%d cc_now=%d | %s x %s "
                           "ne00=%lld ne01=%lld ne11=%lld | a0=%d a1=%d s01=%lld s11=%lld | mmf=%d mmvq=%d mmq=%d\n",
                src0->name, ctx.device, cc_ctx, dev_now, cc_now,
                ggml_type_name(src0->type), ggml_type_name(src1->type),
                (long long) ne00, (long long) ne01, (long long) ne11,
                (int) ((uintptr_t) src0->data % 256), (int) ((uintptr_t) src1->data % 256),
                (long long) (src0->nb[1]/ggml_type_size(src0->type)), (long long) (src1->nb[1]/sizeof(float)),
                (int) ggml_cuda_should_use_mmf(src0->type, cc_ctx, ggml_cuda_info().devices[ctx.device].warp_size, src0->ne, src0->nb, ne11, false),
                (int) ggml_cuda_should_use_mmvq(src0->type, cc_ctx, ne11),
                (int) ggml_cuda_should_use_mmq(src0->type, cc_ctx, ne11, 0));
        }
    }

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(ctx, src1, dst)) {
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    // A thin f32 matrix (a few output rows, many tokens) is a batched dot product over the activations:
    // run MMVF with the operands swapped (the activations are the "weights", the rows the batch) into a
    // [ne11, ne01] scratch and transpose it into dst. hipBLASLt runs this shape at a fifth of the bandwidth.
    static const bool no_mmvf_swap = getenv("GGML_CUDA_NO_MMVF_SWAP") != nullptr && atoi(getenv("GGML_CUDA_NO_MMVF_SWAP")) != 0;
    if (!no_mmvf_swap && ne01 > 1 && ne01 <= MMVF_MAX_BATCH_SIZE && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ ne01)) {
        ggml_cuda_pool_alloc<float> tmp(ctx.pool(), ne01*ne11);

        ggml_tensor dst_t = *dst;            // [ne11, ne01]: one row of dot products per token
        dst_t.data  = tmp.get();
        dst_t.ne[0] = ne11;
        dst_t.ne[1] = ne01;
        dst_t.nb[1] = dst_t.nb[0]*ne11;
        dst_t.nb[2] = dst_t.nb[1]*ne01;
        dst_t.nb[3] = dst_t.nb[2];
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_t);

        ggml_tensor view = *dst;             // the scratch seen as the transposed [ne01, ne11]
        view.data  = tmp.get();
        view.nb[0] = sizeof(float)*ne11;
        view.nb[1] = sizeof(float);
        view.nb[2] = sizeof(float)*ne01*ne11;
        view.nb[3] = view.nb[2];
        view.view_src = nullptr;
        ggml_cuda_cpy(ctx, &view, dst);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    // halo-hybrid: a thin f32 weight at prefill widths (the MoE router) stays in exact f32 on an LDS-tiled FMA
    //     kernel instead of rocBLAS's 32x32x8 tile (sgemm-tile.cu)
    if (GGML_CUDA_CC_IS_AMD(cc) && ggml_cuda_mul_mat_f32_tile(ctx, src0, src1, dst)) {
        return;
    }
    // halo-hybrid: a short-K q8_0 GEMV whose activation has no q8_1 side copy dots the f32 activation directly
    //     (f32act.cu) instead of paying a quantize launch for it (GGML_CUDA_F32ACT_K=0 disables)
    if (GGML_CUDA_CC_IS_AMD(cc) && ne11 <= 4 && !ggml_cuda_q8_side_find(ctx, src1) &&
            ggml_cuda_mul_mat_vec_q8_f32act(ctx, src0, src1, dst, nullptr)) {
        return;
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    // halo-hybrid: dense q8_0 at prefill widths on RDNA: dequantize-once f16 WMMA GEMM (mmq-wmma.cu)
    if (GGML_CUDA_CC_IS_AMD(cc) && ggml_cuda_mul_mat_q8_0_wmma(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                // halo-hybrid: the expert down-projection at decode width eats the f32 GLU output (f32act.cu)
                //     instead of quantizing the selected rows in a launch of their own
                if (GGML_CUDA_CC_IS_AMD(cc) && ggml_cuda_mul_mat_id_vec_q8_f32act(ctx, src0, src1, ids, dst)) {
                    return;
                }
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_MIX:
            ggml_cuda_op_dsv4_hc_mix(ctx, dst);
            break;
        case GGML_OP_KQ_MASK_BUILD:
            ggml_cuda_op_kq_mask_build(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // halo-hybrid: name the tensor, not just the op. "MUL_MAT failed" on a 46-layer two-host graph is not
        // actionable; the shapes and types are what identify which projection and which dispatch path.
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        GGML_LOG_ERROR("  dst  %s type=%s ne=[%lld %lld %lld %lld] nb=[%zu %zu] data=%p align=%d buft=%s\n",
            dst->name, ggml_type_name(dst->type),
            (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
            dst->nb[0], dst->nb[1], dst->data, (int) ((uintptr_t) dst->data % 256),
            dst->buffer ? ggml_backend_buffer_name(dst->buffer) : "none");
        for (int si = 0; si < GGML_MAX_SRC; ++si) {
            const ggml_tensor * s = dst->src[si];
            if (!s) continue;
            GGML_LOG_ERROR("  src%d %s type=%s ne=[%lld %lld %lld %lld] nb=[%zu %zu %zu] cont=%d data=%p align=%d view=%d buft=%s\n",
                si, s->name, ggml_type_name(s->type),
                (long long) s->ne[0], (long long) s->ne[1], (long long) s->ne[2], (long long) s->ne[3],
                s->nb[0], s->nb[1], s->nb[2], (int) ggml_is_contiguous(s),
                s->data, (int) ((uintptr_t) s->data % 256), s->view_src ? 1 : 0,
                s->buffer ? ggml_backend_buffer_name(s->buffer) : "none");
        }
        CUDA_CHECK(err);
    }

    return true;
}

// halo-hybrid: the persistent-region verifier (persist.cu, GGML_CUDA_PERSIST_VERIFY=1) re-runs nodes on the normal path
bool ggml_cuda_compute_forward_node(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    return ggml_cuda_compute_forward(ctx, dst);
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

// Small cross-device copies as a push KERNEL instead of an SDMA-engine transfer.
// The scheduler hands 12 KB activations between devices ~66 times per token on a two-GPU MoE
// layout; each SDMA copy costs ~28 us on the APU side, almost all of it engine setup latency.
// With peer access enabled the source device can write straight into the destination's
// memory over PCIe (posted writes), and a 12 KB kernel write takes a few microseconds.
// GGML_CUDA_KERNEL_COPY_MAX (bytes, default 262144) sets the threshold; 0 disables.
static __global__ void k_peer_copy_bytes(const uint4 * __restrict__ src, uint4 * __restrict__ dst, const size_t n16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n16) {
        dst[i] = src[i];
    }
}

static size_t ggml_cuda_kernel_copy_max() {
    static const size_t v = [] {
        const char * env = getenv("GGML_CUDA_KERNEL_COPY_MAX");
        return env ? (size_t) atoll(env) : (size_t) 262144;
    }();
    return v;
}

static bool ggml_cuda_ensure_peer_access(int src_device, int dst_device) {
    static bool enabled[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};
    static bool failed [GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};
    if (enabled[src_device][dst_device]) {
        return true;
    }
    if (failed[src_device][dst_device]) {
        return false;
    }
    int can = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, src_device, dst_device));
    if (!can) {
        failed[src_device][dst_device] = true;
        return false;
    }
    ggml_cuda_set_device(src_device);
    const cudaError_t err = cudaDeviceEnablePeerAccess(dst_device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        (void) cudaGetLastError();
        failed[src_device][dst_device] = true;
        return false;
    }
    (void) cudaGetLastError();
    enabled[src_device][dst_device] = true;
    return true;
}

static bool ggml_backend_cuda_cpy_tensor_async_impl(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst, const bool wait_dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            // (GGML_SCHED_OVERLAP_CUT orders its copies in the scheduler instead, only for cut graphs)
            static const bool lazy_inputs = getenv("GGML_SCHED_LAZY_INPUTS") != nullptr;
            if (lazy_inputs) {
                // With GGML_SCHED_LAZY_INPUTS a split may still be running on the destination device
                // when this copy is issued, and the destination buffer may alias memory that work is
                // using. Make the source stream wait for everything currently queued on the
                // destination before copying. The copy itself stays on the source stream (HIP is
                // happier that way), and the dst stream waits for it below as before.
                cudaEvent_t ev_dst = cuda_ctx_dst->next_copy_event();
                ggml_cuda_set_device(cuda_ctx_dst->device);
                CUDA_CHECK(cudaEventRecord(ev_dst, cuda_ctx_dst->stream()));
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_src->stream(), ev_dst, 0));
            }
            const size_t nbytes = ggml_nbytes(dst);
            const bool kernel_copy = nbytes > 0 && nbytes <= ggml_cuda_kernel_copy_max() && nbytes % 16 == 0 &&   // an empty copy would launch a zero-block grid
                ((uintptr_t) src->data % 16 == 0) && ((uintptr_t) dst->data % 16 == 0) &&
                ggml_cuda_ensure_peer_access(src_physical, dst_physical);
            if (kernel_copy) {
                ggml_cuda_set_device(cuda_ctx_src->device);
                const size_t n16 = nbytes / 16;
                const int block = 256;
                const int grid  = (int) ((n16 + block - 1) / block);
                k_peer_copy_bytes<<<grid, block, 0, cuda_ctx_src->stream()>>>((const uint4 *) src->data, (uint4 *) dst->data, n16);
                CUDA_CHECK(cudaGetLastError());
            } else {
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, nbytes, cuda_ctx_src->stream()));
            }
#endif // GGML_CUDA_NO_PEER_COPY
        }

        if (wait_dst) {
            // record event on src stream after the copy (a fresh event from the pool, see next_copy_event)
            cudaEvent_t ev_src = cuda_ctx_src->next_copy_event();
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventRecord(ev_src, cuda_ctx_src->stream()));

            // wait on dst stream for the copy to complete
            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), ev_src, 0));
        }
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    return ggml_backend_cuda_cpy_tensor_async_impl(backend_src, backend_dst, src, dst, /*wait_dst=*/true);
}

static bool ggml_backend_cuda_cpy_tensor_async_nowait(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    return ggml_backend_cuda_cpy_tensor_async_impl(backend_src, backend_dst, src, dst, /*wait_dst=*/false);
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_set_device(cuda_ctx->device);

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

// the key identifies both the split (its first node) and the shapes it was called with.
// a captured cuda graph hard-codes the shapes, so a caller that alternates shapes - a
// speculative verify batch, for example - needs a separate instance per shape. with a
// single key per split, every shape change resets the warmup and no graph is ever used.
//
// this stays O(1) on purpose: walking every node undoes the point of a cuda graph, which is
// to not touch per-node data on the hot path. the first and last node carry the batch
// dimension, which is what changes when a verify batch changes size. a shape this does not
// separate just shares an entry and re-captures, exactly as before, so it can only help.
static uint64_t ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    uint64_t key = (uint64_t) (uintptr_t) cgraph->nodes[0];

    auto mix = [&key](uint64_t v) {
        key = (key ^ v) * 0x100000001b3ull;
    };
    auto mix_str = [&mix](const char * s) {
        // tensor names live in char[GGML_MAX_NAME] and ggml_set_name always NUL-terminates
        for (; *s; ++s) { mix((uint64_t) (unsigned char) *s); }
    };

    mix(cgraph->n_nodes);

    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        mix(cgraph->nodes[0]->ne[d]);
        mix(cgraph->nodes[cgraph->n_nodes - 1]->ne[d]);
    }

    // halo-hybrid: the address of nodes[0] does not separate graphs on the rpc-server, which rebuilds every
    // graph in one persistent per-device arena from offset zero. Structurally identical per-layer splits
    // (the four-device GLM layouts send ~20 of them per token to one device) then share a key, every call
    // sees the previous layer's node properties, and the device never leaves warmup: permanent direct
    // execution instead of graph replay. Node names carry the layer index ("attn_norm-25"), so mixing the
    // first and last node's op and name separates them. Local backends already had distinct addresses;
    // for them this is a no-op.
    mix((uint64_t) cgraph->nodes[0]->op);
    mix((uint64_t) cgraph->nodes[cgraph->n_nodes - 1]->op);
    mix_str(cgraph->nodes[0]->name);
    mix_str(cgraph->nodes[cgraph->n_nodes - 1]->name);

    return key;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    bool res = false;

    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // Only norm/neox shaders have the fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles the norm/neox rope modes only
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) || dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        return 0;
    }

    fused_state_cpy.data        = (float *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / sizeof(float)) : 0;
    return skip;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS || n_tokens <= 0) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// ---- hyper-connection fusions (qwen4exp-style multi-stream residuals) ----------------------------
// Three chains of tiny element-wise ops on [hc*n_embd, n_tokens] activations become one kernel each.
// Every matcher checks data flow and use counts so an elided intermediate can never be read elsewhere.
// GGML_CUDA_NO_HC_FUSE=1 disables.
static bool hc_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}
static int hc_next(const ggml_cgraph * g, int j) {
    while (j < g->n_nodes && hc_is_view_op(g->nodes[j])) { j++; }
    return j;
}
static bool hc_unary(const ggml_tensor * t, ggml_unary_op u) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == u;
}
static bool hc_uses1(const ggml_cgraph * g, int j) {
    return ggml_node_get_use_count(g, j) == 1;
}
static const ggml_tensor * hc_root(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}
// The fused kernel reads `in` at indices other than the one it writes in `out`. If ggml-alloc placed
// `out` over memory that `in` occupied (legal in the unfused graph, where `in` is dead by then), the
// fused kernel would race with itself; decline the fusion in that case.
static bool hc_disjoint(const ggml_tensor * out, const ggml_tensor * in) {
    const char * a0 = (const char *) out->data; const char * a1 = a0 + ggml_nbytes(out);
    const char * b0 = (const char *) in->data;  const char * b1 = b0 + ggml_nbytes(in);
    return a1 <= b0 || b1 <= a0;
}
static float hc_param(const ggml_tensor * t, int k) {
    float v; memcpy(&v, (const float *) t->op_params + k, sizeof(float)); return v;
}
// all view-op nodes strictly inside (lo, hi) must be views of one of the given roots
static bool hc_views_belong(const ggml_cgraph * g, int lo, int hi, std::initializer_list<const ggml_tensor *> roots) {
    for (int q = lo + 1; q < hi; ++q) {
        const ggml_tensor * t = g->nodes[q];
        if (!hc_is_view_op(t)) { continue; }
        bool ok = false;
        for (const ggml_tensor * r : roots) { ok = ok || hc_root(t) == r; }
        if (!ok) { return false; }
    }
    return true;
}

static int ggml_cuda_try_fuse_hc(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    const int n = cgraph->n_nodes;
    ggml_tensor * node = cgraph->nodes[i];
    if (node->type != GGML_TYPE_F32) {
        return 0;
    }

    // (A) SCALE -> SILU
    if (node->op == GGML_OP_SCALE) {
        const int j = hc_next(cgraph, i + 1);
        if (j < n && hc_unary(cgraph->nodes[j], GGML_UNARY_OP_SILU) && cgraph->nodes[j]->src[0] == node &&
                hc_uses1(cgraph, i) && node->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[0]) &&
                ggml_is_contiguous(cgraph->nodes[j]) && hc_views_belong(cgraph, i, j, { node })) {
            // halo-hybrid: when the SILU's only reader is a short-K q8_0 GEMV, that GEMV applies scale+silu to the
            // activation itself (f32act.cu) and neither the element-wise kernel nor a q8_1 copy is needed
            const int k = hc_next(cgraph, j + 1);
            if (k < n && cgraph->nodes[k]->op == GGML_OP_MUL_MAT && cgraph->nodes[k]->src[1] == cgraph->nodes[j] &&
                    hc_uses1(cgraph, j) && hc_views_belong(cgraph, j, k, { cgraph->nodes[j] })) {
                ggml_cuda_f32act_prologue pro;
                memcpy(&pro.scale, (const float *) node->op_params + 0, sizeof(float));
                memcpy(&pro.bias,  (const float *) node->op_params + 1, sizeof(float));
                pro.silu = true;
                // the GEMV reads the SCALE's input; the activation view keeps the SILU node's shape
                ggml_tensor act = *cgraph->nodes[j];
                act.data = node->src[0]->data;
                for (int q = 0; q < 4; ++q) { act.nb[q] = node->src[0]->nb[q]; }
                if (ggml_cuda_mul_mat_vec_q8_f32act(*cuda_ctx, cgraph->nodes[k]->src[0], &act, cgraph->nodes[k], &pro)) {
                    return k - i;
                }
            }
            ggml_cuda_op_scale_silu(*cuda_ctx, node, cgraph->nodes[j]);
            return j - i;
        }
        return 0;
    }

    // (B) SIGMOID(g) -> MUL(xn, sig) -> ADD over hc stream views -> SCALE      => hc_mix
    //     otherwise SIGMOID(g) -> MUL(x, sig)                                   => mul_sigmoid
    if (hc_unary(node, GGML_UNARY_OP_SIGMOID)) {
        const ggml_tensor * g = node->src[0];
        const int j = hc_next(cgraph, i + 1);
        if (j >= n || cgraph->nodes[j]->op != GGML_OP_MUL || !hc_uses1(cgraph, i)) { return 0; }
        const ggml_tensor * mul = cgraph->nodes[j];
        const ggml_tensor * xn  = mul->src[0] == node ? mul->src[1] : (mul->src[1] == node ? mul->src[0] : nullptr);
        if (!xn || xn->type != GGML_TYPE_F32 || !ggml_is_contiguous(xn) || !ggml_is_contiguous(g) || g->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(mul) || !ggml_are_same_shape(xn, mul) || !hc_views_belong(cgraph, i, j, { node })) { return 0; }
        const bool same_shape = ggml_are_same_shape(xn, node);
        const bool per_col    = !same_shape && node->ne[0] == 1 && node->ne[1] == xn->ne[1] && node->ne[2] == xn->ne[2] && node->ne[3] == xn->ne[3];
        if (!same_shape && !per_col) { return 0; }
        const int k = hc_next(cgraph, j + 1);
        const bool chain = same_shape && hc_uses1(cgraph, j) && k < n && cgraph->nodes[k]->op == GGML_OP_ADD &&
                cgraph->nodes[k]->src[0]->op == GGML_OP_VIEW && hc_root(cgraph->nodes[k]->src[0]) == mul;
        if (!chain) {
            if (per_col && !hc_disjoint(cgraph->nodes[j], g)) { return 0; }
            // trailing same-shape ADD(y, mul) with no other consumer of mul: fold it in (the shared-expert residual)
            if (k < n && cgraph->nodes[k]->op == GGML_OP_ADD && hc_uses1(cgraph, j) && hc_views_belong(cgraph, j, k, { cgraph->nodes[j] })) {
                ggml_tensor * add = cgraph->nodes[k];
                const ggml_tensor * y = add->src[0] == mul ? add->src[1] : (add->src[1] == mul ? add->src[0] : nullptr);
                if (y && y->type == GGML_TYPE_F32 && ggml_is_contiguous(y) && ggml_are_same_shape(y, add) && ggml_are_same_shape(add, mul) &&
                        ggml_is_contiguous(add) && (per_col ? hc_disjoint(add, g) : true)) {
                    ggml_cuda_op_mul_sigmoid(*cuda_ctx, xn, g, y, add);
                    return k - i;
                }
            }
            ggml_cuda_op_mul_sigmoid(*cuda_ctx, xn, g, nullptr, cgraph->nodes[j]);
            return j - i;
        }
        const ggml_tensor * add = cgraph->nodes[k];
        const ggml_tensor * v0 = add->src[0];
        const ggml_tensor * v1 = add->src[1];
        if (v0->op != GGML_OP_VIEW || v1->op != GGML_OP_VIEW || hc_root(v0) != mul || hc_root(v1) != mul) { return 0; }
        const int64_t n_embd = v0->ne[0];
        const int64_t nt     = v0->ne[1];
        if (n_embd <= 0 || mul->ne[0] % n_embd != 0 || mul->ne[1] != nt || mul->ne[2] != 1 || mul->ne[3] != 1) { return 0; }
        const int64_t hc = mul->ne[0] / n_embd;
        if (hc < 2 || hc > 16) { return 0; }
        const size_t ts = sizeof(float);
        auto view_ok = [&](const ggml_tensor * v, int64_t c) {
            return v->op == GGML_OP_VIEW && hc_root(v) == mul && v->type == GGML_TYPE_F32 &&
                   v->ne[0] == n_embd && v->ne[1] == nt && v->ne[2] == 1 && v->ne[3] == 1 &&
                   v->nb[0] == ts && v->nb[1] == (size_t) hc * n_embd * ts && v->view_offs == (size_t) c * n_embd * ts;
        };
        if (!view_ok(v0, 0) || !view_ok(v1, 1)) { return 0; }
        int last_idx = k;
        for (int64_t c = 2; c < hc; ++c) {
            if (!hc_uses1(cgraph, last_idx)) { return 0; }
            const int m = hc_next(cgraph, last_idx + 1);
            if (m >= n || cgraph->nodes[m]->op != GGML_OP_ADD || cgraph->nodes[m]->src[0] != cgraph->nodes[last_idx] ||
                    !view_ok(cgraph->nodes[m]->src[1], c)) { return 0; }
            last_idx = m;
        }
        if (!hc_uses1(cgraph, last_idx)) { return 0; }
        const int sidx = hc_next(cgraph, last_idx + 1);
        if (sidx >= n || cgraph->nodes[sidx]->op != GGML_OP_SCALE || cgraph->nodes[sidx]->src[0] != cgraph->nodes[last_idx]) { return 0; }
        ggml_tensor * scale_node = cgraph->nodes[sidx];
        if (!ggml_is_contiguous(scale_node) || scale_node->ne[0] != n_embd || scale_node->ne[1] != nt ||
                !hc_views_belong(cgraph, i, sidx, { mul })) { return 0; }
        // ggml-alloc usually places the output in the slot freed by g (or xn) at the same base. With a single
        // token every thread reads only its own stream-0 element of that region before writing it, and the
        // grid covers n_embd in one pass, so a base-aligned alias is race-free; any other overlap is not.
        auto alias_ok = [&](const ggml_tensor * in) { return hc_disjoint(scale_node, in) || (nt == 1 && scale_node->data == in->data); };
        if (!alias_ok(xn) || !alias_ok(g)) { return 0; }
        ggml_cuda_op_hc_mix(*cuda_ctx, xn, g, n_embd, hc, nt, hc_param(scale_node, 0), hc_param(scale_node, 1), scale_node);
        return sidx - i;
    }

    // (D) MUL(e [ne0, m, nt], w [1, m, nt]) -> ADD over the m expert views      => weighted_sum
    if (node->op == GGML_OP_MUL) {
        const ggml_tensor * e = node->src[0];
        const ggml_tensor * w = node->src[1];
        if (e->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(e) || !ggml_is_contiguous(w) ||
                !ggml_are_same_shape(e, node) || w->ne[0] != 1 || w->ne[1] != e->ne[1] || w->ne[2] != e->ne[2] ||
                w->ne[3] != 1 || e->ne[3] != 1) { return 0; }
        const int64_t ne0 = e->ne[0];
        const int64_t m   = e->ne[1];
        const int64_t nt  = e->ne[2];
        if (m < 2 || m > 64 || ggml_node_get_use_count(cgraph, i) != (int32_t) m) { return 0; }
        const size_t ts = sizeof(float);
        auto view_ok = [&](const ggml_tensor * v, int64_t c) {
            return v->op == GGML_OP_VIEW && hc_root(v) == node && v->type == GGML_TYPE_F32 &&
                   v->ne[0] == ne0 && v->ne[1] == nt && v->ne[2] == 1 && v->ne[3] == 1 &&
                   v->nb[0] == ts && v->nb[1] == (size_t) m * ne0 * ts && v->view_offs == (size_t) c * ne0 * ts;
        };
        const int k = hc_next(cgraph, i + 1);
        if (k >= n || cgraph->nodes[k]->op != GGML_OP_ADD || !view_ok(cgraph->nodes[k]->src[0], 0) || !view_ok(cgraph->nodes[k]->src[1], 1)) { return 0; }
        int last_idx = k;
        for (int64_t c = 2; c < m; ++c) {
            if (!hc_uses1(cgraph, last_idx)) { return 0; }
            const int q = hc_next(cgraph, last_idx + 1);
            if (q >= n || cgraph->nodes[q]->op != GGML_OP_ADD || cgraph->nodes[q]->src[0] != cgraph->nodes[last_idx] ||
                    !view_ok(cgraph->nodes[q]->src[1], c)) { return 0; }
            last_idx = q;
        }
        ggml_tensor * out = cgraph->nodes[last_idx];
        if (!ggml_is_contiguous(out) || out->ne[0] != ne0 || out->ne[1] != nt || !hc_views_belong(cgraph, i, last_idx, { node }) ||
                !hc_disjoint(out, e) || !hc_disjoint(out, w)) { return 0; }
        ggml_cuda_op_weighted_sum(*cuda_ctx, e, w, ne0, m, nt, out);
        return last_idx - i;
    }

    // (E) ADD(x, b[ne0]) -> SOFTPLUS -> MUL(., a[ne0])      => gdn_gate
    if (node->op == GGML_OP_ADD) {
        const ggml_tensor * x = node->src[0];
        const ggml_tensor * b = node->src[1];
        if (x->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(b) ||
                !ggml_are_same_shape(x, node) || b->ne[0] != x->ne[0] || b->ne[1] != 1 || b->ne[2] != 1 || b->ne[3] != 1 ||
                !hc_uses1(cgraph, i)) { return 0; }
        const int j = hc_next(cgraph, i + 1);
        if (j >= n || !hc_unary(cgraph->nodes[j], GGML_UNARY_OP_SOFTPLUS) || cgraph->nodes[j]->src[0] != node || !hc_uses1(cgraph, j)) { return 0; }
        const int k = hc_next(cgraph, j + 1);
        if (k >= n || cgraph->nodes[k]->op != GGML_OP_MUL || cgraph->nodes[k]->src[0] != cgraph->nodes[j]) { return 0; }
        ggml_tensor * out = cgraph->nodes[k];
        const ggml_tensor * a = out->src[1];
        if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->ne[0] != x->ne[0] || a->ne[1] != 1 || a->ne[2] != 1 || a->ne[3] != 1 ||
                !ggml_are_same_shape(out, x) || !ggml_is_contiguous(out) || !hc_views_belong(cgraph, i, k, { node, cgraph->nodes[j] })) { return 0; }
        ggml_cuda_op_gdn_gate(*cuda_ctx, x, b, a, out);
        return k - i;
    }

    // (C) REPEAT(b) ; SCALE(inj) -> SIGMOID -> SCALE ; MUL(rep, w) ; ADD(x, mul)      => hc_combine
    if (node->op == GGML_OP_REPEAT) {
        const ggml_tensor * rep = node;
        const ggml_tensor * b   = rep->src[0];
        if (rep->ne[3] != 1 || b->type != GGML_TYPE_F32 || b->ne[1] != 1 || b->ne[3] != 1 ||
                b->ne[0] != rep->ne[0] || b->ne[2] != rep->ne[2] || !ggml_is_contiguous(b) || !hc_uses1(cgraph, i)) { return 0; }
        const int64_t n_embd = rep->ne[0];
        const int64_t hc     = rep->ne[1];
        const int64_t nt     = rep->ne[2];
        const int j1 = hc_next(cgraph, i + 1);
        if (j1 >= n || cgraph->nodes[j1]->op != GGML_OP_SCALE) { return 0; }
        const ggml_tensor * sc1 = cgraph->nodes[j1];
        const ggml_tensor * inj = sc1->src[0];
        if (inj->type != GGML_TYPE_F32 || !ggml_is_contiguous(inj) || inj->ne[0] != hc || inj->ne[1] != nt ||
                inj->ne[2] != 1 || inj->ne[3] != 1 || !hc_uses1(cgraph, j1)) { return 0; }
        const int j2 = hc_next(cgraph, j1 + 1);
        if (j2 >= n || !hc_unary(cgraph->nodes[j2], GGML_UNARY_OP_SIGMOID) || cgraph->nodes[j2]->src[0] != sc1 || !hc_uses1(cgraph, j2)) { return 0; }
        const int j3 = hc_next(cgraph, j2 + 1);
        if (j3 >= n || cgraph->nodes[j3]->op != GGML_OP_SCALE || cgraph->nodes[j3]->src[0] != cgraph->nodes[j2] || !hc_uses1(cgraph, j3)) { return 0; }
        const ggml_tensor * sc2 = cgraph->nodes[j3];
        const int j4 = hc_next(cgraph, j3 + 1);
        if (j4 >= n || cgraph->nodes[j4]->op != GGML_OP_MUL || !hc_uses1(cgraph, j4)) { return 0; }
        const ggml_tensor * mul = cgraph->nodes[j4];
        const ggml_tensor * w   = mul->src[0] == rep ? mul->src[1] : (mul->src[1] == rep ? mul->src[0] : nullptr);
        if (!w || hc_root(w) != sc2 || w->ne[0] != 1 || w->ne[1] != hc || w->ne[2] != nt || w->ne[3] != 1) { return 0; }
        const int j5 = hc_next(cgraph, j4 + 1);
        if (j5 >= n || cgraph->nodes[j5]->op != GGML_OP_ADD) { return 0; }
        ggml_tensor * add = cgraph->nodes[j5];
        const ggml_tensor * x = add->src[0] == mul ? add->src[1] : (add->src[1] == mul ? add->src[0] : nullptr);
        if (!x || x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, add) || !ggml_is_contiguous(x) || !ggml_is_contiguous(add) ||
                add->ne[0] != n_embd || add->ne[1] != hc || add->ne[2] != nt || add->ne[3] != 1 ||
                !hc_views_belong(cgraph, i, j5, { sc2 }) || !hc_disjoint(add, b) || !hc_disjoint(add, inj)) { return 0; }
        // halo-hybrid: qwen4exp hc boundary -- the next mix's RMS_NORM(add) -> MUL(gamma [n_embd, 1|hc]) joins the
        // combine as one launch that also writes the q8_1 copy of the normed rows (GGML_CUDA_NO_HC_BOUNDARY=2 disables)
        if (!ggml_cuda_hc_boundary_disabled(2) && n_embd % QK8_1 == 0) {
            const int j6 = hc_next(cgraph, j5 + 1);
            const int j7 = j6 < n ? hc_next(cgraph, j6 + 1) : n;
            if (j7 < n && cgraph->nodes[j6]->op == GGML_OP_RMS_NORM && cgraph->nodes[j6]->src[0] == add && hc_uses1(cgraph, j6) &&
                    cgraph->nodes[j7]->op == GGML_OP_MUL && cgraph->nodes[j7]->src[0] == cgraph->nodes[j6]) {
                const ggml_tensor * rms   = cgraph->nodes[j6];
                ggml_tensor *       xn    = cgraph->nodes[j7];
                const ggml_tensor * gamma = xn->src[1];
                auto same_or_disjoint = [](const ggml_tensor * o, const ggml_tensor * in) { return o->data == in->data || hc_disjoint(o, in); };
                if (gamma->type == GGML_TYPE_F32 && ggml_is_contiguous(gamma) && gamma->ne[0] == n_embd &&
                        (gamma->ne[1] == 1 || gamma->ne[1] == hc) && gamma->ne[2] == 1 && gamma->ne[3] == 1 &&
                        xn->type == GGML_TYPE_F32 && ggml_is_contiguous(xn) && ggml_are_same_shape(xn, add) &&
                        hc_views_belong(cgraph, i, j7, { sc2, add, rms, hc_root(gamma) }) &&
                        same_or_disjoint(add, x) && same_or_disjoint(xn, x) && hc_disjoint(xn, add) &&
                        hc_disjoint(xn, b) && hc_disjoint(xn, inj) && hc_disjoint(xn, gamma)) {
                    ggml_cuda_op_hc_combine_norm(*cuda_ctx, x, b, inj, gamma, n_embd, hc, nt,
                            hc_param(sc1, 0), hc_param(sc1, 1), hc_param(sc2, 0), hc_param(sc2, 1), hc_param(rms, 0), add, xn);
                    return j7 - i;
                }
            }
        }
        ggml_cuda_op_hc_combine(*cuda_ctx, x, b, inj, n_embd, hc, nt,
                hc_param(sc1, 0), hc_param(sc1, 1), hc_param(sc2, 0), hc_param(sc2, 1), add);
        return j5 - i;
    }

    return 0;
}


// ---- generic element-wise chain fusion (ewchain.cuh) ---------------------------------------------
static bool ew_unary_ok(const ggml_tensor * t) {
    switch (ggml_get_unary_op(t)) {
        case GGML_UNARY_OP_SIGMOID: case GGML_UNARY_OP_SILU: case GGML_UNARY_OP_EXP: case GGML_UNARY_OP_NEG:
        case GGML_UNARY_OP_RELU:    case GGML_UNARY_OP_TANH: case GGML_UNARY_OP_ABS:
            return true;
        default:
            return false;
    }
}
// a node the chain can absorb: f32, the chain input in src[0] (or src[1] for the commutative ops), a broadcastable
// f32 src1 for the binary ops
static bool ew_node_ok(const ggml_tensor * t, const ggml_tensor * prev, const ggml_tensor ** in, const ggml_tensor ** other) {
    if (t->type != GGML_TYPE_F32 || ggml_is_empty(t)) {
        return false;
    }
    *other = nullptr;
    switch (t->op) {
        case GGML_OP_MUL: case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_DIV: {
            const ggml_tensor * a = t->src[0];
            const ggml_tensor * b = t->src[1];
            if (prev && a != prev && b == prev && (t->op == GGML_OP_MUL || t->op == GGML_OP_ADD)) {
                std::swap(a, b);
            }
            if (prev && a != prev) {
                return false;
            }
            if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, t) || !ggml_can_repeat(b, t)) {
                return false;
            }
            *in = a; *other = b;
            return true;
        }
        case GGML_OP_SCALE: case GGML_OP_SQR: case GGML_OP_SQRT:
            if (prev && t->src[0] != prev) { return false; }
            if (t->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(t->src[0], t)) { return false; }
            *in = t->src[0];
            return true;
        case GGML_OP_UNARY:
            if (prev && t->src[0] != prev) { return false; }
            if (!ew_unary_ok(t) || t->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(t->src[0], t)) { return false; }
            *in = t->src[0];
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_ewchain_match(const ggml_cgraph * cgraph, int i, ggml_cuda_ew_match & m) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * in0 = nullptr;
    const ggml_tensor * other = nullptr;
    if (!ew_node_ok(cgraph->nodes[i], nullptr, &in0, &other)) {
        return false;
    }
    int idx[GGML_CUDA_EW_MAX_OPS];
    const ggml_tensor * others[GGML_CUDA_EW_MAX_OPS];
    idx[0] = i; others[0] = other;
    int cnt = 1;
    int last = i;
    while (cnt < GGML_CUDA_EW_MAX_OPS) {
        const ggml_tensor * prev = cgraph->nodes[last];
        if (!hc_uses1(cgraph, last) || (prev->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        const int j = hc_next(cgraph, last + 1);
        if (j >= n) {
            break;
        }
        const ggml_tensor * in = nullptr;
        if (!ew_node_ok(cgraph->nodes[j], prev, &in, &other)) {
            break;
        }
        // view nodes between the two must not view an intermediate of the chain (which the fused kernel never writes)
        bool views_ok = true;
        for (int q = last + 1; q < j && views_ok; ++q) {
            const ggml_tensor * v = cgraph->nodes[q];
            if (!hc_is_view_op(v)) { continue; }
            for (int c = 0; c < cnt; ++c) {
                if (hc_root(v) == cgraph->nodes[idx[c]]) { views_ok = false; }
            }
        }
        if (!views_ok) {
            break;
        }
        idx[cnt] = j; others[cnt] = other; ++cnt;
        last = j;
    }
    if (cnt < 2) {
        return false;
    }
    ggml_tensor * out = cgraph->nodes[last];
    if (!ggml_is_contiguous(out)) {
        return false;
    }
    // the fused kernel writes `out` while reading in0 and every src1: they must not overlap, except an exact
    // in-place elementwise alias of in0 (same address, same strides, which ggml-alloc produces for in-place ops)
    const bool inplace = out->data == in0->data && ggml_is_contiguous(in0) && ggml_are_same_shape(in0, out);
    if (!inplace && !hc_disjoint(out, in0)) {
        return false;
    }
    for (int c = 0; c < cnt; ++c) {
        if (others[c] && !hc_disjoint(out, others[c])) {
            return false;
        }
    }

    m = {};
    ggml_cuda_ew_chain & ch = m.ch;
    ch.n    = cnt;
    ch.src0 = (const char *) in0->data;
    ch.dst  = (float *) out->data;
    for (int d = 0; d < 4; ++d) {
        ch.nb0[d] = in0->nb[d];
        ch.ne[d]  = out->ne[d];
    }
    for (int c = 0; c < cnt; ++c) {
        const ggml_tensor * t = cgraph->nodes[idx[c]];
        ggml_cuda_ew_op & o = ch.ops[c];
        o.op = t->op;
        if (t->op == GGML_OP_UNARY) {
            o.unary = ggml_get_unary_op(t);
        } else if (t->op == GGML_OP_SCALE) {
            o.s = hc_param(t, 0);
            o.b = hc_param(t, 1);
        } else if (others[c]) {
            o.src1 = (const char *) others[c]->data;
            for (int d = 0; d < 4; ++d) {
                o.ne1[d] = others[c]->ne[d];
                o.nb1[d] = others[c]->nb[d];
            }
            m.others[m.n_others++] = others[c];
        }
    }
    m.last = last;
    m.in0  = in0;
    m.out  = out;
    return true;
}

// halo-hybrid: the hyper-connection boundary in one launch (dsv4-hc.cu, dsv4_hc_mix_fused): an optional DSV4_HC_POST
//     whose dst is the next DSV4_HC_MIX's input, the HC_MIX, and the RMS_NORM -> MUL(w) of its pre-mix row (the view
//     at offset 0), which also registers the q8_1 copy of the MUL's dst for the GEMVs behind it. The hc_post dst and
//     the post/comb part of the mix are still written (other consumers read them); the un-normed row is skipped when
//     the norm is its only reader. GGML_CUDA_NO_HC_NORM_FUSE=1 disables; GGML_CUDA_NO_HC_POST_FUSE=1 keeps the
//     norm tail but leaves hc_post as its own launch.
static bool hc_f32_rows16(const ggml_tensor * t) {   // f32, contiguous dim 0, 16-byte aligned rows (float4 loads)
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ((uintptr_t) t->data) % 16 == 0 &&
           t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}
static int ggml_cuda_try_fuse_hc_norm(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_HC_NORM_FUSE") != nullptr && std::atoi(getenv("GGML_CUDA_NO_HC_NORM_FUSE"));
    if (disabled) {
        return 0;
    }
    const int n = cgraph->n_nodes;
    ggml_tensor * node = cgraph->nodes[i];
    ggml_tensor * post = nullptr;
    int im = i;
    static const bool no_post = getenv("GGML_CUDA_NO_HC_POST_FUSE") != nullptr && std::atoi(getenv("GGML_CUDA_NO_HC_POST_FUSE"));
    if (node->op == GGML_OP_DSV4_HC_POST) {
        if (no_post) {
            return 0;
        }
        im = hc_next(cgraph, i + 1);
        if (im >= n || cgraph->nodes[im]->op != GGML_OP_DSV4_HC_MIX || cgraph->nodes[im]->src[0] != node) {
            return 0;
        }
        post = node;
    } else if (node->op != GGML_OP_DSV4_HC_MIX) {
        return 0;
    }
    ggml_tensor * mix = cgraph->nodes[im];
    const ggml_tensor * x = mix->src[0];
    const int64_t n_embd = x->ne[0];
    const int64_t nt     = x->ne[2];
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 4 || 4*n_embd != 16384 || x->ne[3] != 1 || mix->type != GGML_TYPE_F32 ||
            (mix->src[1]->type != GGML_TYPE_Q8_0 && mix->src[1]->type != GGML_TYPE_F32) || !hc_f32_rows16(x)) {
        return 0;
    }
    if (post) {
        const ggml_tensor * px = post->src[0], * pr = post->src[1], * pp = post->src[2], * pc = post->src[3];
        const bool ok = post->type == GGML_TYPE_F32 && ggml_are_same_shape(post, x) &&
            px->ne[0] == n_embd && px->ne[1] == nt && px->ne[2] == 1 && px->ne[3] == 1 && hc_f32_rows16(px) &&
            pr->ne[0] == n_embd && pr->ne[1] == 4 && pr->ne[2] == nt && pr->ne[3] == 1 && hc_f32_rows16(pr) &&
            pp->type == GGML_TYPE_F32 && pp->ne[0] == 4 && pp->ne[1] == nt && pp->ne[2] == 1 && pp->ne[3] == 1 &&
            pc->type == GGML_TYPE_F32 && pc->ne[0] == 4 && pc->ne[1] == 4 && pc->ne[2] == nt && pc->ne[3] == 1 &&
            hc_disjoint(post, px) && hc_disjoint(post, pr) && hc_disjoint(post, pp) && hc_disjoint(post, pc) &&
            hc_disjoint(mix, px) && hc_disjoint(mix, pr) && hc_disjoint(mix, pp) && hc_disjoint(mix, pc);
        if (!ok) {
            return 0;
        }
    }
    // the norm tail: VIEW(mix, offset 0, n_embd x nt) -> RMS_NORM -> MUL(w [n_embd])
    ggml_tensor * rn  = nullptr;
    ggml_tensor * mul = nullptr;
    bool write_out = true;
    const int ir = hc_next(cgraph, im + 1);
    if (ir + 1 < n && cgraph->nodes[ir]->op == GGML_OP_RMS_NORM && cgraph->nodes[ir + 1]->op == GGML_OP_MUL) {
        ggml_tensor * r = cgraph->nodes[ir];
        ggml_tensor * m = cgraph->nodes[ir + 1];
        const ggml_tensor * v  = r->src[0];
        const ggml_tensor * wv = m->src[0] == r ? m->src[1] : (m->src[1] == r ? m->src[0] : nullptr);
        const bool ok = v->view_src == mix && v->view_offs == 0 && v->ne[0] == n_embd && v->ne[1] == nt &&
            v->ne[2] == 1 && v->ne[3] == 1 && v->nb[0] == sizeof(float) && v->nb[1] == mix->nb[1] &&
            r->type == GGML_TYPE_F32 && hc_uses1(cgraph, ir) &&
            wv && wv->type == GGML_TYPE_F32 && ggml_is_contiguous(wv) && wv->ne[0] == n_embd && ggml_nrows(wv) == 1 &&
            m->type == GGML_TYPE_F32 && ggml_are_same_shape(m, r) && m->nb[0] == sizeof(float) &&
            hc_disjoint(m, x) && hc_disjoint(m, mix) && hc_disjoint(m, wv) &&
            (!post || (hc_disjoint(m, post->src[0]) && hc_disjoint(m, post->src[1]) && hc_disjoint(m, post->src[2]) && hc_disjoint(m, post->src[3])));
        if (ok) {
            rn = r; mul = m;
            for (int q = im + 1; q < ir; ++q) {
                if (cgraph->nodes[q] == v) { write_out = !hc_uses1(cgraph, q); }
            }
        }
    }
    if (!post && !mul) {
        return 0;
    }
    ggml_cuda_op_dsv4_hc_mix_fused(*cuda_ctx, mix, post, rn, mul, write_out);
    return (mul ? ir + 1 : im) - i;
}

static int ggml_cuda_try_fuse_ewchain(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_EWCHAIN") != nullptr && std::atoi(getenv("GGML_CUDA_NO_EWCHAIN"));
    if (disabled) {
        return 0;
    }
    ggml_cuda_ew_match m;
    if (!ggml_cuda_ewchain_match(cgraph, i, m)) {
        return 0;
    }
    ggml_cuda_op_ew_chain(*cuda_ctx, m.ch);
    return m.last - i;
}

// halo-hybrid: KDA conv tail at decode (glm5next build_kda_layer): concat(concat(w_q, w_k, 1), w_v, 1) -> ssm_conv ->
// silu -> l2_norm(Q-head view), l2_norm(K-head view) becomes one launch that reads the three conv weights directly
// (the two concats only rebuild a constant) and normalises the Q and K heads in-block. The SiLU output is still
// written in full, V is read from it through a view. Decode widths only (n_t <= 8); prefill keeps the unfused path.
// GGML_CUDA_NO_KDA_CONV_L2=1 disables.
static int ggml_cuda_try_fuse_kda_conv_l2(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_KDA_CONV_L2") != nullptr && std::atoi(getenv("GGML_CUDA_NO_KDA_CONV_L2"));
    if (disabled) {
        return 0;
    }
    const int n = cgraph->n_nodes;
    auto is_w = [](const ggml_tensor * w, const ggml_tensor * ref) {
        return w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_are_same_shape(w, ref);
    };
    auto concat_dim1 = [](const ggml_tensor * t) {
        return t->op == GGML_OP_CONCAT && t->type == GGML_TYPE_F32 && ggml_get_op_params_i32(t, 0) == 1;
    };

    ggml_tensor * c1 = cgraph->nodes[i];
    if (!concat_dim1(c1) || !hc_uses1(cgraph, i)) {
        return 0;
    }
    const ggml_tensor * w_q = c1->src[0];
    const ggml_tensor * w_k = c1->src[1];
    if (w_q->ne[2] != 1 || w_q->ne[3] != 1 || !is_w(w_q, w_q) || !is_w(w_k, w_q)) {
        return 0;
    }
    const int64_t d_conv  = w_q->ne[0];
    const int64_t d_inner = w_q->ne[1];

    const int j1 = hc_next(cgraph, i + 1);
    if (j1 >= n || !concat_dim1(cgraph->nodes[j1]) || cgraph->nodes[j1]->src[0] != c1 || !hc_uses1(cgraph, j1)) {
        return 0;
    }
    ggml_tensor * c2 = cgraph->nodes[j1];
    const ggml_tensor * w_v = c2->src[1];
    if (!is_w(w_v, w_q)) {
        return 0;
    }

    const int j2 = hc_next(cgraph, j1 + 1);
    if (j2 >= n || cgraph->nodes[j2]->op != GGML_OP_SSM_CONV || cgraph->nodes[j2]->src[1] != c2 || !hc_uses1(cgraph, j2)) {
        return 0;
    }
    ggml_tensor * conv = cgraph->nodes[j2];
    const ggml_tensor * conv_in = conv->src[0];
    const int j3 = hc_next(cgraph, j2 + 1);
    if (j3 >= n || !hc_unary(cgraph->nodes[j3], GGML_UNARY_OP_SILU) || cgraph->nodes[j3]->src[0] != conv) {
        return 0;
    }
    ggml_tensor * silu = cgraph->nodes[j3];
    if (conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32 || conv_in->type != GGML_TYPE_F32 ||
        conv_in->nb[0] != sizeof(float) || conv_in->nb[1] != conv_in->ne[0]*sizeof(float) ||
        silu->nb[0] != sizeof(float) || silu->ne[0] != 3*d_inner ||
        !ggml_cuda_ssm_conv_kda_l2_supported(d_conv, d_inner, silu->ne[1])) {
        return 0;
    }

    // the two L2 norms: views of the SiLU output, one 128-wide head per row, at the Q and K channel offsets
    ggml_tensor * l2[2] = { nullptr, nullptr };
    int j = j3;
    for (int r = 0; r < 2; ++r) {
        j = hc_next(cgraph, j + 1);
        if (j >= n || cgraph->nodes[j]->op != GGML_OP_L2_NORM) {
            return 0;
        }
        ggml_tensor * t = cgraph->nodes[j];
        const ggml_tensor * v = t->src[0];
        if (v->op != GGML_OP_VIEW || v->view_src != silu || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
            v->ne[0] != 128 || v->ne[1] != d_inner/128 || v->ne[2] != silu->ne[1] || v->ne[3] != silu->ne[2] ||
            v->nb[0] != sizeof(float) || v->nb[1] != 128*sizeof(float) || v->nb[2] != silu->nb[1] || v->nb[3] != silu->nb[2]) {
            return 0;
        }
        const size_t k = v->view_offs == 0 ? 0 : v->view_offs == (size_t) d_inner*sizeof(float) ? 1 : 2;
        if (k > 1 || l2[k] != nullptr) {
            return 0;
        }
        l2[k] = t;
    }
    const float eps = ggml_get_op_params_f32(l2[0], 0);
    if (eps != ggml_get_op_params_f32(l2[1], 0) || !(eps >= 0.0f)) {
        return 0;
    }
    // the fused kernel reads conv_in while writing silu/q/k from other blocks: they must not alias
    if (!hc_disjoint(silu, conv_in) || !hc_disjoint(l2[0], conv_in) || !hc_disjoint(l2[1], conv_in) ||
        !hc_disjoint(l2[0], silu) || !hc_disjoint(l2[1], silu) || !hc_disjoint(l2[0], l2[1])) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s aliases its conv input or outputs, KDA conv-l2 fusion declined\n", __func__, silu->name);
        }
        return 0;
    }

    ggml_cuda_op_ssm_conv_kda_l2(*cuda_ctx, conv_in, w_q, w_k, w_v, silu, l2[0], l2[1], eps);
    return j - i;
}

// halo-hybrid: KDA conv-input assembly (build_conv_state of glm5next/kimi-linear at decode) as one launch:
//     c1 = CONCAT(q, k, 0), [one MUL_MAT producing v when the GEMV hoist did not run], c2 = CONCAT(c1, v, 0),
//     c3 = CONCAT(states, TRANSPOSE(RESHAPE(c2)), 0), then up to KDA_CONV_ROWS_MAX_DST CPY(VIEW(c3), slot view).
//     2 + 1 + K launches -> 1 (K = n_rs_seq + 1). Decode only (n_seqs == 1, nt <= 8); structure-only here so that
//     graph_optimize can call it before allocation; the aliasing checks are in ggml_cuda_kda_conv_rows_disjoint.
//     GGML_CUDA_NO_KDA_CONV_ROWS=1 disables.
struct ggml_cuda_kda_conv_rows_match {
    ggml_tensor * c1;
    ggml_tensor * c2;
    ggml_tensor * c3;
    ggml_tensor * mid;   // the v MUL_MAT between c1 and c2, or nullptr
    int           n_cpy;
    ggml_tensor * cpy[KDA_CONV_ROWS_MAX_DST];
    int           s_idx[KDA_CONV_ROWS_MAX_DST];
    int           last;  // index of the last fused node
};

static bool ggml_cuda_kda_conv_rows_disabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_KDA_CONV_ROWS") != nullptr && std::atoi(getenv("GGML_CUDA_NO_KDA_CONV_ROWS"));
    return disabled;
}

// use count of any tensor in the graph (views included), -1 when unknown
static int kda_use_count(const ggml_cgraph * g, const ggml_tensor * t) {
    if (!g->use_counts || !ggml_hash_contains(&g->visited_hash_set, (ggml_tensor *) t)) {
        return -1;
    }
    return g->use_counts[ggml_hash_find(&g->visited_hash_set, t)];
}

static bool kda_f32_cols(const ggml_tensor * t, int64_t nt) {
    return t->type == GGML_TYPE_F32 && t->ne[1] == nt && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
}

static bool ggml_cuda_kda_conv_rows_find(const ggml_cgraph * g, int i, ggml_cuda_kda_conv_rows_match & m) {
    const int n = g->n_nodes;
    ggml_tensor * c1 = g->nodes[i];
    auto concat0 = [](const ggml_tensor * t) {
        return t->op == GGML_OP_CONCAT && ggml_get_op_params_i32(t, 0) == 0 && t->type == GGML_TYPE_F32 &&
               !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    if (!concat0(c1)) {
        return false;
    }
    const ggml_tensor * q = c1->src[0];
    const ggml_tensor * k = c1->src[1];
    const int64_t nt = q->ne[1];
    if (nt < 1 || nt > 8 || !kda_f32_cols(q, nt) || !kda_f32_cols(k, nt) || kda_use_count(g, c1) != 1) {
        return false;
    }
    int j = hc_next(g, i + 1);
    m.mid = nullptr;
    if (j < n && g->nodes[j]->op == GGML_OP_MUL_MAT) {
        m.mid = g->nodes[j];
        if (m.mid->src[0] == c1 || m.mid->src[1] == c1) {
            return false;
        }
        j = hc_next(g, j + 1);
    }
    if (j >= n || !concat0(g->nodes[j])) {
        return false;
    }
    ggml_tensor * c2 = g->nodes[j];
    const ggml_tensor * v = c2->src[1];
    if (c2->src[0] != c1 || !kda_f32_cols(v, nt) || kda_use_count(g, c2) != 1 || (m.mid && v != m.mid)) {
        return false;
    }
    const int64_t C = c2->ne[0];
    if (C > INT_MAX / 2) {
        return false;
    }
    const int k3 = hc_next(g, j + 1);
    if (k3 >= n || g->nodes[k3]->op != GGML_OP_CONCAT || ggml_get_op_params_i32(g->nodes[k3], 0) != 0 ||
            g->nodes[k3]->type != GGML_TYPE_F32) {
        return false;
    }
    ggml_tensor * c3 = g->nodes[k3];
    const ggml_tensor * st = c3->src[0];
    const ggml_tensor * xt = c3->src[1];
    const int64_t ns = st->ne[0];
    // x^T: element (t, c) at c2 + (t*C + c)*4, reached only through single-use view nodes
    if (xt->view_src != c2 || xt->view_offs != 0 || xt->ne[0] != nt || xt->ne[1] != C || xt->ne[2] != 1 || xt->ne[3] != 1 ||
            xt->nb[0] != (size_t) C*sizeof(float) || xt->nb[1] != sizeof(float)) {
        return false;
    }
    for (const ggml_tensor * t = xt; t != c2; t = t->src[0]) {
        if (!hc_is_view_op(t) || !t->src[0] || kda_use_count(g, t) != 1) {
            return false;
        }
    }
    if (st->type != GGML_TYPE_F32 || ns < 1 || ns > KDA_CONV_ROWS_MAX_DST || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 ||
            !ggml_is_contiguous(c3) || c3->ne[0] != ns + nt || c3->ne[1] != C) {
        return false;
    }
    m.c1 = c1; m.c2 = c2; m.c3 = c3;
    m.n_cpy = 0;
    m.last  = k3;
    for (int p = hc_next(g, k3 + 1); p < n && m.n_cpy < KDA_CONV_ROWS_MAX_DST; p = hc_next(g, p + 1)) {
        ggml_tensor * cp = g->nodes[p];
        if (cp->op != GGML_OP_CPY) {
            break;
        }
        const ggml_tensor * s = cp->src[0];
        const ggml_tensor * d = cp->src[1];
        if (s->view_src != c3 || s->type != GGML_TYPE_F32 || s->ne[0] != ns || s->ne[1] != C || s->ne[2] != 1 || s->ne[3] != 1 ||
                s->nb[0] != sizeof(float) || s->nb[1] != c3->nb[1] || s->view_offs % sizeof(float) != 0 ||
                (int64_t) (s->view_offs / sizeof(float)) + ns > ns + nt) {
            break;
        }
        if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != ns*C) {
            break;
        }
        m.cpy[m.n_cpy]   = cp;
        m.s_idx[m.n_cpy] = (int) (s->view_offs / sizeof(float));
        m.n_cpy++;
        m.last = p;
    }
    return true;
}

// the fused kernel reads q/k/v while it writes conv_input and the slots; the unfused graph let the allocator reuse
//     q/k/v memory for conv_input (and, without the hoist, v's for q/k). Decline on any overlap except the one the
//     kernel handles: a slot that IS the contiguous states memory (build_rs's single-slot view path).
static bool ggml_cuda_kda_conv_rows_disjoint(const ggml_cuda_kda_conv_rows_match & m) {
    auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    const ggml_tensor * q  = m.c1->src[0];
    const ggml_tensor * k  = m.c1->src[1];
    const ggml_tensor * v  = m.c2->src[1];
    const ggml_tensor * st = m.c3->src[0];
    const ggml_tensor * in[4] = { q, k, v, st };
    for (const ggml_tensor * t : in) {
        if (overlap(m.c3, t)) {
            return false;
        }
    }
    if (m.mid && (overlap(v, q) || overlap(v, k))) {
        return false;
    }
    for (int a = 0; a < m.n_cpy; ++a) {
        const ggml_tensor * d = m.cpy[a]->src[1];
        if (overlap(d, q) || overlap(d, k) || overlap(d, v) || overlap(d, m.c3)) {
            return false;
        }
        if (overlap(d, st) && !(d->data == st->data && ggml_is_contiguous(st))) {
            return false;
        }
        for (int b = 0; b < a; ++b) {
            if (overlap(d, m.cpy[b]->src[1])) {
                return false;
            }
        }
    }
    return true;
}

static int ggml_cuda_try_fuse_kda_conv_rows(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_kda_conv_rows_disabled()) {
        return 0;
    }
    ggml_cuda_kda_conv_rows_match m;
    if (!ggml_cuda_kda_conv_rows_find(cgraph, i, m)) {
        return 0;
    }
    if (!ggml_cuda_kda_conv_rows_disjoint(m)) {
        // expected only where graph_optimize's alloc deps did not run (an RPC server's graphs are allocated by the client)
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s: conv_input or a slot overlaps q/k/v, unfused path (logged once)\n", __func__, m.c3->name);
        }
        return 0;
    }
    // v's GEMV (no hoist: it sits between the two concats) runs first, as in the graph; the check above made sure
    // it does not land on q or k, which the unfused graph had already consumed at that point
    if (m.mid && !ggml_cuda_compute_forward(*cuda_ctx, m.mid)) {
        GGML_ABORT("%s: op not supported %s (%s)", __func__, m.mid->name, ggml_op_name(m.mid->op));
    }
    ggml_cuda_kda_conv_rows_args a = {};
    a.q = m.c1->src[0];
    a.k = m.c1->src[1];
    a.v = m.c2->src[1];
    a.states = m.c3->src[0];
    a.conv_input = m.c3;
    a.n_dst = m.n_cpy;
    for (int kk = 0; kk < m.n_cpy; ++kk) {
        a.dst[kk]   = (float *) m.cpy[kk]->src[1]->data;
        a.s_idx[kk] = m.s_idx[kk];
    }
    ggml_cuda_op_kda_conv_rows(*cuda_ctx, a);
    return m.last - i;
}

// halo-hybrid: the GDN conv front at decode (qwen4exp build_conv_state_at + ssm_conv + silu + l2_norm):
//     c3 = CONCAT(states, TRANSPOSE(x), 0), K x CPY(VIEW(c3), slot view), SSM_CONV(c3, w) -> SILU, L2_NORM(VIEW(silu))
//     4 + (K - 1) launches -> 1 (ggml_cuda_op_gdn_conv_front, ssm-conv.cu). The L2_NORM view must start at channel 0
//     and cover whole 128-wide heads (qwen4exp normalises Q and K, adjacent, in one node). conv_input is not written:
//     the copies and the conv must be its only readers. The builder puts the other layer's state gather between the
//     copies and the conv; graph_optimize moves the conv..l2 run up behind the copies (dependency-safe: its inputs are
//     conv_input and a weight) and keeps x and the states alive until the l2 output is allocated. Decode/verify only
//     (n_seqs == 1, nt <= 8), f32, a stored conv weight (glm5next's is a concat, so glm never matches).
//     GGML_CUDA_NO_GDN_CONV_FRONT=1 disables.
struct ggml_cuda_gdn_conv_front_match {
    ggml_tensor * c3;
    ggml_tensor * conv;
    ggml_tensor * silu;
    ggml_tensor * l2;
    int           n_cpy;
    ggml_tensor * cpy[GDN_CONV_FRONT_MAX_DST];
    int           s_idx[GDN_CONV_FRONT_MAX_DST];
    int           last_cpy;   // index of the last slot copy (i when there is none)
    int           i_conv;
    int           i_l2;
};

static bool ggml_cuda_gdn_conv_front_disabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_GDN_CONV_FRONT") != nullptr && std::atoi(getenv("GGML_CUDA_NO_GDN_CONV_FRONT"));
    return disabled;
}

// adjacent: the conv must follow the copies (view ops aside), as at dispatch; otherwise it may sit up to 64 nodes
// later, the order graph_optimize is handed
static bool ggml_cuda_gdn_conv_front_find(const ggml_cgraph * g, int i, ggml_cuda_gdn_conv_front_match & m, bool adjacent) {
    const int n = g->n_nodes;
    ggml_tensor * c3 = g->nodes[i];
    if (c3->op != GGML_OP_CONCAT || ggml_get_op_params_i32(c3, 0) != 0 || c3->type != GGML_TYPE_F32 ||
            (c3->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * st = c3->src[0];
    const ggml_tensor * xt = c3->src[1];
    const int64_t ns = st->ne[0];
    const int64_t C  = st->ne[1];
    const int64_t nt = xt->ne[0];
    if (st->type != GGML_TYPE_F32 || st->ne[2] != 1 || st->ne[3] != 1 || ns < 2 || ns > 3 ||
            st->nb[0] % sizeof(float) != 0 || st->nb[1] % sizeof(float) != 0) {
        return false;
    }
    if (xt->type != GGML_TYPE_F32 || xt->ne[1] != C || xt->ne[2] != 1 || xt->ne[3] != 1 ||
            xt->nb[1] != sizeof(float) || xt->nb[0] % sizeof(float) != 0 || !ggml_cuda_gdn_conv_front_supported(ns + 1, C, nt)) {
        return false;
    }
    if (c3->ne[0] != ns + nt || c3->ne[1] != C || c3->ne[2] != 1 || c3->ne[3] != 1 || !ggml_is_contiguous(c3)) {
        return false;
    }

    m.c3 = c3;
    m.n_cpy = 0;
    m.last_cpy = i;
    for (int p = hc_next(g, i + 1); p < n && m.n_cpy < GDN_CONV_FRONT_MAX_DST; p = hc_next(g, p + 1)) {
        ggml_tensor * cp = g->nodes[p];
        if (cp->op != GGML_OP_CPY) {
            break;
        }
        const ggml_tensor * s = cp->src[0];
        const ggml_tensor * d = cp->src[1];
        if (s->op != GGML_OP_VIEW || s->src[0] != c3 || kda_use_count(g, s) != 1 || s->type != GGML_TYPE_F32 ||
                s->ne[0] != ns || s->ne[1] != C || s->ne[2] != 1 || s->ne[3] != 1 ||
                s->nb[0] != sizeof(float) || s->nb[1] != c3->nb[1] || s->view_offs % sizeof(float) != 0 ||
                (int64_t) (s->view_offs / sizeof(float)) + ns > ns + nt) {
            return false;
        }
        if (d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != ns*C) {
            return false;
        }
        m.cpy[m.n_cpy]   = cp;
        m.s_idx[m.n_cpy] = (int) (s->view_offs / sizeof(float));
        m.n_cpy++;
        m.last_cpy = p;
    }

    // the conv
    int jc = -1;
    if (adjacent) {
        jc = hc_next(g, m.last_cpy + 1);
    } else {
        for (int p = m.last_cpy + 1; p < n && p <= m.last_cpy + 64; ++p) {
            if (g->nodes[p]->op == GGML_OP_SSM_CONV && g->nodes[p]->src[0] == c3) {
                jc = p;
                break;
            }
        }
    }
    if (jc < 0 || jc >= n || g->nodes[jc]->op != GGML_OP_SSM_CONV || g->nodes[jc]->src[0] != c3 || !hc_uses1(g, jc)) {
        return false;
    }
    // conv_input is read by the copies' views and the conv, nothing else: it is never written
    if (kda_use_count(g, c3) != m.n_cpy + 1) {
        return false;
    }
    ggml_tensor * conv = g->nodes[jc];
    const ggml_tensor * w = conv->src[1];
    if (w->op != GGML_OP_NONE || w->view_src != nullptr || w->type != GGML_TYPE_F32 || w->ne[0] != ns + 1 || w->ne[1] != C ||
            w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != sizeof(float) || conv->type != GGML_TYPE_F32) {
        return false;
    }
    const int js = jc + 1;
    if (js >= n || !hc_unary(g->nodes[js], GGML_UNARY_OP_SILU) || g->nodes[js]->src[0] != conv) {
        return false;
    }
    ggml_tensor * silu = g->nodes[js];
    if (silu->type != GGML_TYPE_F32 || silu->ne[0] != C || silu->ne[1] != nt || silu->ne[2] != 1 || silu->ne[3] != 1 ||
            silu->nb[0] != sizeof(float) || silu->nb[1] % sizeof(float) != 0) {
        return false;
    }
    const int jl = hc_next(g, js + 1);
    if (jl >= n || g->nodes[jl]->op != GGML_OP_L2_NORM || !hc_views_belong(g, js, jl, { silu })) {
        return false;
    }
    ggml_tensor * l2 = g->nodes[jl];
    const ggml_tensor * v = l2->src[0];
    if (v->op != GGML_OP_VIEW || v->view_src != silu || v->view_offs != 0 || v->type != GGML_TYPE_F32 ||
            v->ne[0] != GDN_CONV_FRONT_HEAD || v->ne[1] < 1 || v->ne[1]*GDN_CONV_FRONT_HEAD > C || v->ne[2] != nt || v->ne[3] != 1 ||
            v->nb[0] != sizeof(float) || v->nb[1] != GDN_CONV_FRONT_HEAD*sizeof(float) || v->nb[2] != silu->nb[1] ||
            l2->type != GGML_TYPE_F32 || !ggml_is_contiguous(l2) || !(ggml_get_op_params_f32(l2, 0) >= 0.0f)) {
        return false;
    }
    m.conv = conv;
    m.silu = silu;
    m.l2   = l2;
    m.i_conv = jc;
    m.i_l2   = jl;
    return true;
}

// the kernel reads x and the states while it writes the SiLU output, the l2 output and the slots from every block:
// decline on any overlap except a slot that IS the contiguous states memory (build_rs's single-slot view path, safe
// channel by channel)
static bool ggml_cuda_gdn_conv_front_disjoint(const ggml_cuda_gdn_conv_front_match & m) {
    const ggml_tensor * st = m.c3->src[0];
    const ggml_tensor * xt = m.c3->src[1];
    if (!hc_disjoint(m.silu, xt) || !hc_disjoint(m.l2, xt) || !hc_disjoint(m.silu, st) || !hc_disjoint(m.l2, st) ||
            !hc_disjoint(m.l2, m.silu)) {
        return false;
    }
    const bool st_cont = st->nb[0] == sizeof(float) && st->nb[1] == st->ne[0]*sizeof(float);
    for (int a = 0; a < m.n_cpy; ++a) {
        const ggml_tensor * d = m.cpy[a]->src[1];
        if (!hc_disjoint(d, xt) || !hc_disjoint(d, m.silu) || !hc_disjoint(d, m.l2)) {
            return false;
        }
        if (!hc_disjoint(d, st) && !(d->data == st->data && st_cont)) {
            return false;
        }
        for (int b = 0; b < a; ++b) {
            if (!hc_disjoint(d, m.cpy[b]->src[1])) {
                return false;
            }
        }
    }
    return true;
}

static int ggml_cuda_try_fuse_gdn_conv_front(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_gdn_conv_front_disabled()) {
        return 0;
    }
    ggml_cuda_gdn_conv_front_match m;
    if (!ggml_cuda_gdn_conv_front_find(cgraph, i, m, /*adjacent =*/ true)) {
        return 0;
    }
    if (!ggml_cuda_gdn_conv_front_disjoint(m)) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s: an output overlaps x, the states or a slot, unfused path (logged once)\n", __func__, m.silu->name);
        }
        return 0;
    }
    ggml_cuda_gdn_conv_front_args a = {};
    a.states = m.c3->src[0];
    a.xt     = m.c3->src[1];
    a.w      = m.conv->src[1];
    a.y      = m.silu;
    a.l2     = m.l2;
    a.eps    = ggml_get_op_params_f32(m.l2, 0);
    a.n_dst  = m.n_cpy;
    for (int k = 0; k < m.n_cpy; ++k) {
        a.dst[k]   = (float *) m.cpy[k]->src[1]->data;
        a.s_idx[k] = m.s_idx[k];
    }
    ggml_cuda_op_gdn_conv_front(*cuda_ctx, a);
    return m.i_l2 - i;
}

static bool ggml_cuda_fusion_disabled() {
    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    return disable_fusion;
}

// halo-hybrid: the recurrent-state gather at decode. build_rs gathers one cache row (get_rows by the s_copy
// index: the rollback snapshot plane the sequence resumes from) into a scratch tensor that only the layer's
// gated_delta_net reads, through a reshape. Leave the get_rows out and let the gdn kernel read the cache row
// through the index (gated_delta_net.cu, gather_t): one ~12 us launch and a 4 MB copy per KDA layer at decode.
// Gated to one sequence (a single index; n_rs == n_seqs, so build_rs's extra-state shuffle is empty) and to no
// node between the two, nor the gdn's own output, writing into the cache tensor or the index, so reading both late
// reads the same bytes. The index is the one that matters under real allocation: ggml-alloc frees s_copy after its
// last reader (the last layer's gather) and hands its bytes to a later node. graph_optimize keeps it alive until the
// gdn (alloc dep); where that did not run (an RPC server's graphs are allocated by the client) the range check
// keeps that layer's get_rows. The kernel reads each (head, column) of the row before the same thread writes that
// column's snapshots, so the gdn -> cache cpy fusion writing into the row it reads from is safe.
// GGML_CUDA_NO_KDA_STATE_GATHER=1 disables it.
static bool ggml_cuda_gdn_state_gather_disabled() {
    static const bool disabled = getenv("GGML_CUDA_NO_KDA_STATE_GATHER") != nullptr && std::atoi(getenv("GGML_CUDA_NO_KDA_STATE_GATHER"));
    return disabled;
}

// structural match (no data pointers: graph_optimize runs it before allocation); the gdn's node index or -1
static int ggml_cuda_gdn_state_gather_find(const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * rows = cgraph->nodes[i];
    if (rows->op != GGML_OP_GET_ROWS) {
        return -1;
    }
    const ggml_tensor * src = rows->src[0];
    const ggml_tensor * idx = rows->src[1];
    // one row of a 2D, row-contiguous f32 source by a one-element index that is the whole index tensor
    // (a view of a longer s_copy means n_rs > n_seqs: the extra-state shuffle then writes other rows)
    if (rows->type != GGML_TYPE_F32 || src->type != GGML_TYPE_F32 || idx->type != GGML_TYPE_I32 ||
            (rows->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_nelements(idx) != 1 ||
            (idx->view_src != nullptr && ggml_nelements(idx->view_src) != 1) ||
            src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(rows) || rows->ne[0] != src->ne[0] || rows->ne[1] != 1 ||
            ggml_node_get_use_count(cgraph, i) != 1) {
        return -1;
    }
    const ggml_tensor * cur = rows; // the gathered state, through its reshapes
    const int n_scan = std::min(cgraph->n_nodes, i + 512);
    for (int j = i + 1; j < n_scan; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_RESHAPE && n->src[0] == cur) {
            if (ggml_node_get_use_count(cgraph, j) != 1 || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return -1;
            }
            cur = n;
            continue;
        }
        if (n->op == GGML_OP_GATED_DELTA_NET && n->src[5] == cur) {
            const bool ok = n->type == GGML_TYPE_F32 && n->src[2]->ne[3] == 1 && cur->ne[3] == 1 && ggml_is_contiguous(cur) &&
                cur->ne[0]*cur->ne[1]*cur->ne[2] == src->ne[0] && (n->flags & GGML_TENSOR_FLAG_COMPUTE);
            return ok ? j : -1;
        }
        for (int k = 0; k < GGML_MAX_SRC; ++k) {
            if (n->src[k] == cur) {
                return -1; // another reader of the gathered state
            }
        }
    }
    return -1;
}

static bool ggml_cuda_try_defer_gdn_state_gather(ggml_backend_cuda_context * cuda_ctx, const ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_gdn_state_gather_disabled()) {
        return false;
    }
    const int ig = ggml_cuda_gdn_state_gather_find(cgraph, i);
    if (ig < 0) {
        return false;
    }
    const ggml_tensor * rows = cgraph->nodes[i];
    const ggml_tensor * gdn  = cgraph->nodes[ig];
    const ggml_tensor * src  = rows->src[0];
    const ggml_tensor * idx  = rows->src[1];
    if (src->data == nullptr || idx->data == nullptr || gdn->data == nullptr) {
        return false;
    }
    const char * c0 = (const char *) src->data; // the cache
    const char * c1 = c0 + ggml_nbytes(src);
    const char * x0 = (const char *) idx->data; // the index
    const char * x1 = x0 + ggml_nbytes(idx);
    auto writes_either = [&](const ggml_tensor * n) {
        const char * a0 = (const char *) n->data;
        const char * a1 = a0 + ggml_nbytes(n);
        return (a0 < c1 && c0 < a1) || (a0 < x1 && x0 < a1);
    };
    // nothing in between, nor the gdn's own output, may write into the cache tensor or the index
    for (int j = i + 1; j <= ig; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_is_empty(n) && n->data != nullptr && !ggml_cuda_is_view_or_noop(n) && writes_either(n)) {
            return false;
        }
    }
    for (auto & e : cuda_ctx->gdn_state_gather) {
        if (e.gdn == nullptr) {
            e = { gdn, rows };
            return true;
        }
    }
    return false;
}

// halo-hybrid: the KDA gate prologue at decode widths (n <= 4) as ONE mul_mat_vec_q launch. The graph is
//     MUL_MAT(f_b) -> ADD(dt_b) -> RESHAPE -> MUL(A, per head) -> SCALE(-1) -> SIGMOID -> SCALE(lower bound)
//     and, with the GEMV hoist (graph_optimize) having pulled the beta GEMV into the grouped launch, the beta's
//     RESHAPE -> SIGMOID comes right behind it. Unfused: GEMV + bin_bcast (the reshape of the ADD's output stops the
//     element-wise chain fuser) + ewchain + sigmoid. The GEMV's fused epilogue takes the bias (broadcast over the
//     columns), the per-head multiplier, both scales and the sigmoid; block 0 does the beta sigmoid on the side.
//     Every piece is optional except the MUL_MAT and the sigmoid. GGML_CUDA_NO_KDA_GATE_PROLOGUE=1 disables.
static bool kda_gate_rowshape(const ggml_tensor * t, int64_t nrows, int64_t ncols) {
    // a contiguous f32 reshape of the [nrows, ncols] GEMV output that keeps the column in dims >= 1 or >= 2
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return false;
    }
    return (t->ne[0] == nrows && t->ne[1]*t->ne[2]*t->ne[3] == ncols) ||
           (t->ne[0]*t->ne[1] == nrows && t->ne[2]*t->ne[3] == ncols);
}

static int ggml_cuda_try_fuse_kda_gate(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_KDA_GATE_PROLOGUE") != nullptr && atoi(getenv("GGML_CUDA_NO_KDA_GATE_PROLOGUE")) != 0;
    const int n = cgraph->n_nodes;
    ggml_tensor * mm = cgraph->nodes[i];
    if (disabled || mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) ||
            mm->ne[2] != 1 || mm->ne[3] != 1 || mm->src[1]->type != GGML_TYPE_F32 || !ggml_is_quantized(mm->src[0]->type) ||
            mm->src[0]->ne[2] != 1 || mm->src[0]->ne[3] != 1 || !ggml_cuda_should_fuse_mul_mat_vec_q(mm)) {
        return 0;
    }
    const int64_t nrows = mm->ne[0], ncols = mm->ne[1];

    const ggml_tensor * chain[8];   // mm + the absorbed nodes whose results are never written
    int n_chain = 0;
    chain[n_chain++] = mm;
    ggml_cuda_mm_fusion_args_host f{};
    const ggml_tensor * bias = nullptr;
    int last = i;
    int stage = 0;   // 0: bias, 1: mul, 2: scale0, 3: sigmoid, 4: scale1, 5: done
    bool has_tail = false;
    while (stage < 5 && n_chain < 8) {
        const ggml_tensor * prev = cgraph->nodes[last];
        if (!hc_uses1(cgraph, last) || (prev->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        const int j = hc_next(cgraph, last + 1);
        if (j >= n) {
            break;
        }
        ggml_tensor * t = cgraph->nodes[j];
        // the node's chain input: prev itself or a reshape of it
        auto from_prev = [&](const ggml_tensor * s) {
            return s == prev || (s->op == GGML_OP_RESHAPE && s->src[0] == prev);
        };
        if (!kda_gate_rowshape(t, nrows, ncols)) {
            break;
        }
        bool took = false;
        if (stage <= 0 && t->op == GGML_OP_ADD) {
            const ggml_tensor * b = t->src[0] == prev ? t->src[1] : t->src[1] == prev ? t->src[0] : nullptr;
            // [rows] broadcast over the columns, or a per-column [rows, ncols] operand (x_bias_stride_col)
            if (b && b->type == GGML_TYPE_F32 && ggml_is_contiguous(b) && ggml_are_same_shape(t, prev) &&
                    ((b->ne[0] == nrows && ggml_nelements(b) == nrows) || ggml_are_same_shape(b, mm))) {
                bias = b; stage = 1; took = true;
            }
        } else if (stage <= 1 && t->op == GGML_OP_MUL && t->src[0] != t->src[1]) {
            const ggml_tensor * a = from_prev(t->src[0]) ? t->src[0] : from_prev(t->src[1]) ? t->src[1] : nullptr;
            const ggml_tensor * m = a == t->src[0] ? t->src[1] : t->src[0];
            if (a && ggml_are_same_shape(a, t) && m->type == GGML_TYPE_F32 && ggml_is_contiguous(m) &&
                    m->ne[2] == 1 && m->ne[3] == 1) {
                // m broadcasts over the columns; per row r it reads m[r / div]
                uint32_t div = 0;
                if (ggml_nelements(m) == 1) {
                    div = (uint32_t) nrows;
                } else if (t->ne[0] == nrows && m->ne[0] == nrows && m->ne[1] == 1) {
                    div = 1;
                } else if (t->ne[0] != nrows && t->ne[0]*t->ne[1] == nrows && m->ne[1] == t->ne[1] && (m->ne[0] == 1 || m->ne[0] == t->ne[0])) {
                    div = m->ne[0] == 1 ? (uint32_t) t->ne[0] : 1;
                }
                if (div) {
                    f.x_mul = m; f.x_mul_div = div; stage = 2; took = true;
                }
            }
        } else if (stage <= 2 && t->op == GGML_OP_SCALE && from_prev(t->src[0])) {
            f.tail_s0 = hc_param(t, 0); f.tail_b0 = hc_param(t, 1); stage = 3; took = true;
        } else if (stage <= 3 && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID && from_prev(t->src[0])) {
            f.tail_act = 1; stage = 4; took = true;
        } else if (stage == 4 && t->op == GGML_OP_SCALE && from_prev(t->src[0])) {
            f.tail_s1 = hc_param(t, 0); f.tail_b1 = hc_param(t, 1); stage = 5; took = true;
        }
        if (!took) {
            break;
        }
        // views between the two must not view an intermediate (the fused kernel never writes one), except the
        // reshape of prev that t itself consumes
        for (int q = last + 1; q < j; ++q) {
            const ggml_tensor * v = cgraph->nodes[q];
            for (int c = 0; c < n_chain; ++c) {
                if (hc_root(v) == chain[c] && !(v->op == GGML_OP_RESHAPE && v->src[0] == prev && hc_uses1(cgraph, q) && (t->src[0] == v || t->src[1] == v))) {
                    return 0;
                }
            }
        }
        has_tail = has_tail || stage >= 2;
        chain[n_chain++] = t;
        last = j;
    }
    // only gate-shaped tails (a sigmoid in them): a plain bias is the MUL_MAT + ADD fusion's, and a GEMV with a
    // bare scale or multiplier behind it keeps whatever path it had
    if (last == i || !has_tail || f.tail_act != 1) {
        return 0;
    }
    // a chain that ends in the sigmoid read by a MUL is the SIGMOID+MUL fusion's (the KDA output gate g_b: taking
    // the sigmoid here would only move the launch to a bare MUL)
    if (stage == 4) {
        const int j2 = hc_next(cgraph, last + 1);
        if (j2 < n && cgraph->nodes[j2]->op == GGML_OP_MUL &&
                (cgraph->nodes[j2]->src[0] == cgraph->nodes[last] || cgraph->nodes[j2]->src[1] == cgraph->nodes[last])) {
            return 0;
        }
    }
    f.x_bias = bias;
    ggml_tensor * out = cgraph->nodes[last];

    // the beta sigmoid right behind: SIGMOID(RESHAPE(x)) with x computed before i (topological order: nothing
    // between i and here but the chain and views), contiguous, small
    int end = last;
    {
        const int k = hc_next(cgraph, last + 1);
        if (k < n) {
            ggml_tensor * sg = cgraph->nodes[k];
            const ggml_tensor * src = sg->src[0];
            const ggml_tensor * x = src && src->op == GGML_OP_RESHAPE ? src->src[0] : src;
            bool ok = sg->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID && sg->type == GGML_TYPE_F32 &&
                      x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(sg) &&
                      ggml_nelements(sg) == ggml_nelements(x) && ggml_nelements(sg) <= 4096 &&
                      x->op != GGML_OP_NONE && !hc_is_view_op(x);
            for (int c = 0; ok && c < n_chain; ++c) {
                ok = hc_root(x) != chain[c];
            }
            // views between the chain's end and the sigmoid: of the output, of x, or of the sigmoid's input
            for (int q = last + 1; ok && q < k; ++q) {
                const ggml_tensor * v = cgraph->nodes[q];
                for (int c = 0; ok && c < n_chain - 1; ++c) {
                    ok = hc_root(v) != chain[c];
                }
            }
            // the sigmoid is in place over x or apart from it; it never lands on the gate output
            ok = ok && (sg->data == x->data || hc_disjoint(sg, x)) && hc_disjoint(sg, out) && hc_disjoint(out, x);
            if (ok) {
                f.aux_src = x;
                f.aux_dst = sg;
                end = k;
            }
        }
    }
    // weights, but check: the kernel writes out while reading them
    if ((bias && !hc_disjoint(out, bias)) || (f.x_mul && !hc_disjoint(out, f.x_mul))) {
        return 0;
    }
    // out may sit on src1's memory (dead after the GEMV in the unfused graph): mul_mat_vec_q reads only the q8_1
    // copy that the quantize launch (or an earlier producer) made before the kernel starts, never src1 itself

    ggml_tensor d = *mm;   // the GEMV's own [nrows, ncols] view of the output
    d.data = out->data;
    d.view_src = nullptr;
    ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm->src[0], mm->src[1], nullptr, &d, &f);
    return end - i;
}

// halo-hybrid: the MLA/DSA attention tail at decode widths: the batched per-head GEMV (wv_b over the flash-attention
//     output) -> PERMUTE(0,2,1,3) -> CONT is one mul_mat_vec_q launch that writes the permuted layout through its
//     dst strides (token stride = the cont's nb[2], head stride = its nb[1]); the cpy launch goes away. Same kernel
//     and arguments as the unfused GEMV otherwise. GGML_CUDA_NO_MLA_V_PERMUTE=1 disables.
static int ggml_cuda_try_fuse_mla_v_permute(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_MLA_V_PERMUTE") != nullptr && atoi(getenv("GGML_CUDA_NO_MLA_V_PERMUTE")) != 0;
    const int n = cgraph->n_nodes;
    ggml_tensor * mm = cgraph->nodes[i];
    if (disabled || mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || mm->ne[3] != 1 ||
            !ggml_is_quantized(mm->src[0]->type) || mm->src[0]->ne[3] != 1 || mm->src[1]->type != GGML_TYPE_F32 ||
            mm->ne[1] < 1 || mm->ne[1] > 4 || !hc_uses1(cgraph, i) || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            mm->ne[0] % 64 != 0) {
        // (whole row blocks: mul_mat_vec_q bounds a partial row block by stride_col_dst, which is not the row
        // count once the columns are strided)
        return 0;
    }
    const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
    const int warp_size = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
    // only where the unfused MUL_MAT takes mul_mat_vec_q itself (ggml_cuda_mul_mat's order)
    if (!ggml_cuda_should_use_mmvq(mm->src[0]->type, cc, mm->ne[1]) ||
            ggml_cuda_should_use_mmf(mm->src[0]->type, cc, warp_size, mm->src[0]->ne, mm->src[0]->nb, mm->ne[1], false)) {
        return 0;
    }
    const int j = hc_next(cgraph, i + 1);
    if (j >= n) {
        return 0;
    }
    ggml_tensor * cont = cgraph->nodes[j];
    const ggml_tensor * p = cont->src[0];
    if (cont->op != GGML_OP_CONT || cont->type != GGML_TYPE_F32 || !ggml_is_contiguous(cont) ||
            !p || p->op != GGML_OP_PERMUTE || p->src[0] != mm) {
        return 0;
    }
    const int32_t * ax = (const int32_t *) p->op_params;
    if (ax[0] != 0 || ax[1] != 2 || ax[2] != 1 || ax[3] != 3) {
        return 0;
    }
    // the CONT must have the permute's own shape: the strides below are the permuted layout (the non-FA MLA branch
    // uses ggml_cont_2d(permute(...)), a [v*H, T] shape whose nb[2] spans the tensor: writes would run past it)
    for (int k = 0; k < GGML_MAX_DIMS; ++k) {
        if (cont->ne[k] != p->ne[k]) {
            return 0;
        }
    }
    // the permute is the only node between, and nothing else reads it
    for (int q = i + 1; q < j; ++q) {
        const ggml_tensor * v = cgraph->nodes[q];
        if (v != p && hc_root(v) == mm) {
            return 0;
        }
        if (v == p && !hc_uses1(cgraph, q)) {
            return 0;
        }
    }
    ggml_tensor d = *mm;
    d.data  = cont->data;
    d.nb[1] = cont->nb[2];   // token
    d.nb[2] = cont->nb[1];   // head
    d.nb[3] = cont->nb[3];
    d.view_src = nullptr;
    ggml_cuda_mul_mat_vec_q(*cuda_ctx, mm->src[0], mm->src[1], nullptr, &d);
    return j - i;
}

// halo-hybrid: MoE tail at decode widths (mmvq.cu ggml_cuda_mmvq_moe_tail), matched from the expert down GEMV:
//   MUL_MAT_ID(down_exps, act, ids) -> [MUL expert_scale] -> MUL weights -> VIEW x k -> ADD x (k-1)   => moe_out
//   MUL_MAT(down_shexp, sact) ; UNARY SIGMOID(g) ; MUL(shexp, sig) ; ADD(moe_out, gated)               => ffn_out
// with only view-type no-ops (any root) between the pieces and every intermediate used once, inside the chain.
struct ggml_cuda_moe_tail_match {
    const ggml_tensor * mmid   = nullptr;
    const ggml_tensor * shexp  = nullptr;   // MUL_MAT(down_shexp, sact)
    const ggml_tensor * g      = nullptr;   // pre-sigmoid gate logits
    ggml_cuda_moe_weighted_reduction_match red;
    ggml_tensor *       out    = nullptr;   // ffn_out
    int                 last   = -1;
};

static bool ggml_cuda_moe_tail_find(int device, const ggml_cgraph * cgraph, int i, ggml_cuda_moe_tail_match & m) {
    const int n = cgraph->n_nodes;
    const ggml_tensor * mmid = cgraph->nodes[i];
    if (mmid->op != GGML_OP_MUL_MAT_ID || mmid->type != GGML_TYPE_F32 || !ggml_is_quantized(mmid->src[0]->type) ||
            mmid->src[1]->type != GGML_TYPE_F32 || mmid->src[2]->type != GGML_TYPE_I32 || !ggml_is_contiguous(mmid) ||
            mmid->ne[3] != 1 || mmid->ne[2] < 1 || mmid->ne[2] > 4 || mmid->src[1]->ne[1] != mmid->ne[1] ||
            mmid->src[1]->ne[2] != mmid->ne[2] || mmid->src[2]->ne[0] != mmid->ne[1] || mmid->src[2]->ne[1] != mmid->ne[2] ||
            mmid->src[2]->nb[0] != sizeof(int32_t) || !hc_uses1(cgraph, i) || (mmid->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int j = hc_next(cgraph, i + 1);
    if (j >= n || cgraph->nodes[j]->op != GGML_OP_MUL || !ggml_cuda_match_moe_weighted_reduction(cgraph, j, m.red) ||
            m.red.experts != mmid) {
        return false;
    }
    const int red_last = j + m.red.node_count - 1;
    const ggml_tensor * moe_out = m.red.dst;
    const int64_t nt = mmid->ne[2];
    if (cgraph->nodes[red_last] != moe_out || !hc_uses1(cgraph, red_last) || (moe_out->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            !ggml_is_contiguous(m.red.weights) || moe_out->ne[0] != mmid->ne[0] || moe_out->ne[1] != nt) {
        return false;
    }
    for (int q = j; q < red_last; ++q) {   // the reduction's own nodes feed nothing outside it
        if (cgraph->nodes[q]->flags & GGML_TENSOR_FLAG_OUTPUT) { return false; }
    }
    const int k0 = hc_next(cgraph, red_last + 1);   // shared-expert down
    if (k0 >= n) { return false; }
    const ggml_tensor * sh = cgraph->nodes[k0];
    if (sh->op != GGML_OP_MUL_MAT || sh->type != GGML_TYPE_F32 || !ggml_is_quantized(sh->src[0]->type) ||
            sh->src[1]->type != GGML_TYPE_F32 || !ggml_is_contiguous(sh) || !ggml_are_same_shape(sh, moe_out) ||
            sh->src[1]->ne[1] != nt || sh->src[1]->ne[2] != 1 || sh->src[1]->ne[3] != 1 ||
            sh->src[0]->ne[2] != 1 || sh->src[0]->ne[3] != 1 || !hc_uses1(cgraph, k0) || (sh->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int k1 = hc_next(cgraph, k0 + 1);   // SIGMOID(g), g one value per token
    if (k1 >= n || !hc_unary(cgraph->nodes[k1], GGML_UNARY_OP_SIGMOID)) { return false; }
    const ggml_tensor * sig = cgraph->nodes[k1];
    const ggml_tensor * g   = sig->src[0];
    if (g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) || g->ne[0] != 1 || g->ne[1] != nt || g->ne[2] != 1 || g->ne[3] != 1 ||
            !hc_uses1(cgraph, k1) || (sig->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int k2 = hc_next(cgraph, k1 + 1);   // MUL(shexp, sig)
    if (k2 >= n) { return false; }
    const ggml_tensor * mul = cgraph->nodes[k2];
    if (mul->op != GGML_OP_MUL || !((mul->src[0] == sh && mul->src[1] == sig) || (mul->src[1] == sh && mul->src[0] == sig)) ||
            !ggml_are_same_shape(mul, sh) || !ggml_is_contiguous(mul) || !hc_uses1(cgraph, k2) || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int k3 = hc_next(cgraph, k2 + 1);   // ADD(moe_out, gated)
    if (k3 >= n) { return false; }
    ggml_tensor * add = cgraph->nodes[k3];
    if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) || !ggml_are_same_shape(add, moe_out) ||
            !((add->src[0] == moe_out && add->src[1] == mul) || (add->src[1] == moe_out && add->src[0] == mul))) {
        return false;
    }
    if (!ggml_cuda_mmvq_moe_tail_supported(device, mmid->src[0], mmid->src[1], sh->src[0], sh->src[1], mmid->ne[1], nt)) {
        return false;
    }
    m.mmid  = mmid;
    m.shexp = sh;
    m.g     = g;
    m.out   = add;
    m.last  = k3;
    return true;
}

static bool ggml_cuda_moe_tail_disjoint(const ggml_tensor * out, const ggml_tensor * in) {
    if (in == nullptr) { return true; }
    const char * a0 = (const char *) out->data; const char * a1 = a0 + ggml_nbytes(out);
    const char * b0 = (const char *) in->data;  const char * b1 = b0 + ggml_nbytes(in);
    return a1 <= b0 || b1 <= a0;
}

static int ggml_cuda_try_fuse_moe_tail(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    ggml_cuda_moe_tail_match m;
    if (!ggml_cuda_moe_tail_find(cuda_ctx->device, cgraph, i, m)) {
        return 0;
    }
    // every block reads ids / router weights / the gate logit and writes its own ffn_out element: ffn_out must not
    // sit over them (graph_optimize keeps them alive until ffn_out is allocated; this is the runtime guard)
    if (!ggml_cuda_moe_tail_disjoint(m.out, m.mmid->src[2]) || !ggml_cuda_moe_tail_disjoint(m.out, m.red.weights) ||
            !ggml_cuda_moe_tail_disjoint(m.out, m.red.expert_scale) || !ggml_cuda_moe_tail_disjoint(m.out, m.g)) {
        return 0;
    }
    ggml_cuda_mmvq_moe_tail(*cuda_ctx, m.mmid->src[0], m.mmid->src[1], m.mmid->src[2], m.red.weights, m.red.expert_scale,
        m.shexp->src[0], m.shexp->src[1], m.g, m.out);
    return m.last - i;
}

// try and fuse nodes and return the number of nodes to skip
static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    if (ggml_cuda_fusion_disabled()) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    if (node->op == GGML_OP_MUL_MAT_ID) {   // halo-hybrid: MoE tail at decode widths
        const int skip = ggml_cuda_try_fuse_moe_tail(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // halo-hybrid: glm5next DSA indexer pool compressor, 9 launches -> 1 (kpool-compress.cu)
    if (node->op == GGML_OP_GET_ROWS) {
        static const bool no_kpool = getenv("GGML_CUDA_NO_KPOOL_COMPRESS") != nullptr && std::atoi(getenv("GGML_CUDA_NO_KPOOL_COMPRESS"));
        ggml_cuda_kpool_compress_match km;
        if (!no_kpool && ggml_cuda_kpool_compress_match_graph(cgraph, i, km)) {
            ggml_cuda_op_kpool_compress(*cuda_ctx, km);
            return km.last - i;
        }
    }

    if (node->op == GGML_OP_CONCAT) {
        const int skip = ggml_cuda_try_fuse_kda_conv_l2(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_CONCAT) {
        const int skip = ggml_cuda_try_fuse_gdn_conv_front(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_DSV4_HC_POST || node->op == GGML_OP_DSV4_HC_MIX) {
        const int skip = ggml_cuda_try_fuse_hc_norm(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    static const bool no_hc_fuse = getenv("GGML_CUDA_NO_HC_FUSE") != nullptr && std::atoi(getenv("GGML_CUDA_NO_HC_FUSE"));
    if (!no_hc_fuse) {
        const int skip = ggml_cuda_try_fuse_hc(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_CONCAT) {
        const int skip = ggml_cuda_try_fuse_kda_conv_rows(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        int skip = ggml_cuda_try_fuse_kda_gate(cuda_ctx, cgraph, i);
        if (skip == 0) {
            skip = ggml_cuda_try_fuse_mla_v_permute(cuda_ctx, cgraph, i);
        }
        if (skip > 0) {
            return skip;
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // grouped mul_mat_vec_q: consecutive MUL_MATs that read the SAME activation vector
    // (q/k/v of an attention layer, qkv/gate/beta/alpha of a Gated DeltaNet layer). Each would
    // otherwise quantize src1 to q8_1 in its own launch; quantize once and reuse. On ROCm every
    // launch costs ~10 us of host time, so this is worth 3-4 launches per layer.
    // Gate/up pairs feeding a GLU are left to the dedicated fusion.
    if (node->op == GGML_OP_MUL_MAT && node->src[1] && node->src[1]->type == GGML_TYPE_F32 && node->type == GGML_TYPE_F32 && !cuda_ctx->mmvq_group_disabled) {
        const int cc = ggml_cuda_info().devices[cuda_ctx->device].cc;
        const int warp_size = ggml_cuda_info().devices[cuda_ctx->device].warp_size;
        const ggml_tensor * src1 = node->src[1];
        auto eligible = [&](const ggml_tensor * t, bool leader) {
            const ggml_tensor * w = t->src[0];
            if (t->op != GGML_OP_MUL_MAT || t->src[1] != src1 || t->type != GGML_TYPE_F32) {
                return false;
            }
            if (w->ne[2] != 1 || w->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                return false;
            }
            if (!w->buffer || !ggml_backend_buft_is_cuda(w->buffer->buft)) {
                return false;
            }
            // An f32 matrix (the hyper-connection inject, the DeltaNet alpha/beta) can ride along in the grouped
            // kernel, which dots it against the unquantized activation, but that is OFF by default: a row there is
            // one wave where mul_mat_vec_f gives it 8 warps, and no f32 matrix in these models has the rows to make
            // that back (the inject, 4 rows of K=10240, cost 7.6 ms per token). The member must be excluded HERE and
            // not in the launcher: a group the launcher refuses falls back to one launch per matrix and loses the
            // quantized merges too.
            if (w->type == GGML_TYPE_F32) {
                // on by default now that an f32 member gets a whole block per row (GGML_CUDA_GEMV_GROUP_F32ROWS=<n>
                // admits only members with at least n rows; set it huge to exclude them). An f32 matrix may also
                // lead (the MoE router precedes the shared expert's gate/up); the launcher still needs a quantized
                // member somewhere in the group, and falls back to the plain paths otherwise.
                static const int64_t f32_rows = getenv("GGML_CUDA_GEMV_GROUP_F32ROWS") ? atoll(getenv("GGML_CUDA_GEMV_GROUP_F32ROWS")) : 0;
                GGML_UNUSED(leader);
                return w->ne[0] == src1->ne[0] && w->ne[1] >= f32_rows;
            }
            return ggml_is_quantized(w->type) &&
                   ggml_cuda_should_use_mmvq(w->type, cc, src1->ne[1]) &&
                   !ggml_cuda_should_use_mmf(w->type, cc, warp_size, w->ne, w->nb, src1->ne[1], false);
        };
        if (eligible(node, true)) {
            int idx[8];
            int n = 1;
            idx[0] = i;
            for (int j = i + 1; j < cgraph->n_nodes && n < 8; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE) {
                    continue; // reshapes between the matmuls are free
                }
                if (!eligible(t, false)) {
                    break;
                }
                // a (gate, up, GLU) triple joins as ONE pair member, represented by its GLU node: the kernel dots
                // both rows in the same wave and applies the GLU on the way out. Anything else next to a GLU is
                // left to the dedicated fused path.
                static const bool no_pair = getenv("GGML_CUDA_NO_GEMV_PAIR") != nullptr;
                if (!no_pair && j + 2 < cgraph->n_nodes && cgraph->nodes[j + 2]->op == GGML_OP_GLU && eligible(cgraph->nodes[j + 1], false) &&
                    cgraph->nodes[j + 2]->src[0] == t && cgraph->nodes[j + 2]->src[1] == cgraph->nodes[j + 1] &&
                    ggml_is_quantized(t->src[0]->type) && cgraph->nodes[j + 1]->src[0]->type == t->src[0]->type &&
                    ggml_get_op_params_i32(cgraph->nodes[j + 2], 1) == 0 &&
                    (ggml_get_glu_op(cgraph->nodes[j + 2]) == GGML_GLU_OP_SWIGLU || ggml_get_glu_op(cgraph->nodes[j + 2]) == GGML_GLU_OP_GEGLU)) {
                    idx[n++] = j + 2;
                    j += 2;
                    continue;
                }
                if (j + 1 < cgraph->n_nodes && cgraph->nodes[j + 1]->op == GGML_OP_GLU) {
                    break;
                }
                idx[n++] = j;
            }
            // the leader itself may be the gate of a triple: only when it started the run
            if (n == 1 && i + 2 < cgraph->n_nodes && cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
                n = 0;   // let the fused path have it (the loop above broke on the GLU)
            }
            {   // halo-hybrid: GGML_CUDA_GEMV_GROUPS=1 reports how many consecutive GEMVs share this activation, i.e. how
                // many launches a grouped GEMV kernel would replace with one
                static const int who = getenv("GGML_CUDA_GEMV_GROUPS") ? atoi(getenv("GGML_CUDA_GEMV_GROUPS")) : 0;
                if (who) {
                    std::string names;
                    for (int k = 0; k < n; ++k) { names += " "; names += cgraph->nodes[idx[k]]->src[0]->name; }
                    GGML_LOG_WARN("gemv-group: n=%d src1=%s rows=%lld k=%lld ->%s\n", n, src1->name,
                        (long long) node->ne[0], (long long) src1->ne[0], names.c_str());
                }
            }
            bool has_q = false;   // an all-f32 run has nothing to quantize for; leave it to mul_mat_vec_f
            for (int k = 0; k < n; ++k) {
                const ggml_tensor * mm = cgraph->nodes[idx[k]]->op == GGML_OP_GLU ? cgraph->nodes[idx[k]]->src[0] : cgraph->nodes[idx[k]];
                has_q |= ggml_is_quantized(mm->src[0]->type);
            }
            if (n >= 2 && has_q) {
                cudaStream_t stream = cuda_ctx->stream();
                const int64_t ne10 = src1->ne[0];
                const int64_t ne11 = src1->ne[1];
                const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
                const size_t  ts_src1 = ggml_type_size(src1->type);
                ggml_cuda_pool_alloc<char> src1_q8_1(cuda_ctx->pool());
                const char * pre = ggml_cuda_q8_side_find(*cuda_ctx, src1);   // producer-side q8_1 copy, if any
                if (!pre) {
                    src1_q8_1.alloc(ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
                    ggml_type qtype = GGML_TYPE_F32;   // any quantized member's type: the q8_1 layout is the same for all of them
                    for (int k = 0; k < n && qtype == GGML_TYPE_F32; ++k) {
                        const ggml_tensor * mm = cgraph->nodes[idx[k]]->op == GGML_OP_GLU ? cgraph->nodes[idx[k]]->src[0] : cgraph->nodes[idx[k]];
                        qtype = mm->src[0]->type;
                    }
                    quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(), qtype,
                            ne10, src1->nb[1]/ts_src1, src1->nb[2]/ts_src1, src1->nb[3]/ts_src1, ne10_padded, ne11, 1, 1, stream);
                    pre = src1_q8_1.get();
                }
                ggml_tensor * group[8];
                for (int k = 0; k < n; ++k) {
                    group[k] = cgraph->nodes[idx[k]];
                }
                // one launch for the whole group when the weights share a type; otherwise one per matrix as before
                if (!ggml_cuda_mul_mat_vec_q_group(*cuda_ctx, group, n, src1, pre)) {
                    for (int k = 0; k < n; ++k) {
                        if (group[k]->op == GGML_OP_GLU) {   // the pair: gate, up, then the GLU node itself
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, group[k]->src[0]->src[0], src1, nullptr, group[k]->src[0], nullptr, pre);
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, group[k]->src[1]->src[0], src1, nullptr, group[k]->src[1], nullptr, pre);
                            ggml_cuda_compute_forward_node(*cuda_ctx, group[k]);
                        } else if (group[k]->src[0]->type == GGML_TYPE_F32) {
                            ggml_cuda_compute_forward_node(*cuda_ctx, group[k]);   // f32 matrix: its own path
                        } else {
                            ggml_cuda_mul_mat_vec_q(*cuda_ctx, group[k]->src[0], src1, nullptr, group[k], nullptr, pre);
                        }
                    }
                }
                return idx[n - 1] - i;
            }
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                if (!gate_scale || !up_scale) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if (ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            if (!scale) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    {
        const int skip = ggml_cuda_try_fuse_ewchain(cuda_ctx, cgraph, i);
        if (skip > 0) {
            return skip;
        }
    }

    return 0;
}

// halo-hybrid: GGML_CUDA_TIME_OPS=1 (use with GGML_CUDA_DISABLE_GRAPHS=1): hipEvent pairs around every launch of
// the direct evaluation path, accumulated per device and per op class (op, and for matmuls the weight type and
// shape), dumped to stderr every 200 graphs. Tracer-free in-situ kernel time: rocprofv3's per-dispatch completion
// handling inflates sub-100 us kernels several-fold (a 50 us GEMV read 391 us under it on gfx1201), events on
// the compute stream do not. Host cost ~2 us per node, so never use it for a throughput number.
#if defined(GGML_USE_HIP)
#define ggml_cuda_optimer_elapsed hipEventElapsedTime
#define ggml_cuda_optimer_create  hipEventCreate
#else
#define ggml_cuda_optimer_elapsed cudaEventElapsedTime
#define ggml_cuda_optimer_create  cudaEventCreate
#endif
struct ggml_cuda_optimer {
    struct acc { uint64_t n = 0; double us = 0; };
    std::map<std::string, acc> by_key;
    std::vector<cudaEvent_t> ev;      // pairs
    std::vector<std::string> keys;    // key per pair in this graph
    size_t used = 0;
    uint64_t graphs = 0;
    static bool enabled() { static const bool e = getenv("GGML_CUDA_TIME_OPS") != nullptr; return e; }
    cudaEvent_t next_event() {
        if (used >= ev.size()) { cudaEvent_t e; CUDA_CHECK(ggml_cuda_optimer_create(&e)); ev.push_back(e); }
        return ev[used++];
    }
    std::string tag;                  // "D " decode / "P " prefill, from the graph's widest MUL_MAT activation
    void begin(cudaStream_t st, const std::string & key) { keys.push_back(tag + key); CUDA_CHECK(cudaEventRecord(next_event(), st)); }
    void set_tag(const ggml_cgraph * cgraph) {
        int64_t n = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * t = cgraph->nodes[i];
            if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[1]) { n = std::max(n, t->src[1]->ne[1]); }
        }
        tag = n <= 8 ? "D " : "P ";
    }
    void end(cudaStream_t st) { CUDA_CHECK(cudaEventRecord(next_event(), st)); }
    void flush(int device) {
        if (used == 0) return;
        CUDA_CHECK(cudaEventSynchronize(ev[used - 1]));
        for (size_t i = 0; i < keys.size(); i++) {
            float ms = 0; CUDA_CHECK(ggml_cuda_optimer_elapsed(&ms, ev[2*i], ev[2*i + 1]));
            acc & a = by_key[keys[i]]; a.n++; a.us += ms * 1000.0;
        }
        used = 0; keys.clear(); graphs++;
        if (graphs % 200 == 0) {
            std::vector<std::pair<std::string, acc>> v(by_key.begin(), by_key.end());
            std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.us > b.second.us; });
            double tot = 0; uint64_t n = 0; for (auto & e : v) { tot += e.second.us; n += e.second.n; }
            fprintf(stderr, "\n=== OPTIMER dev %d after %llu graphs: %llu launches, %.1f ms total (%.3f ms/graph)\n",
                    device, (unsigned long long) graphs, (unsigned long long) n, tot / 1000.0, tot / 1000.0 / graphs);
            static const size_t top = getenv("GGML_CUDA_TIME_OPS_TOP") ? (size_t) atoi(getenv("GGML_CUDA_TIME_OPS_TOP")) : 40;
            for (size_t i = 0; i < v.size() && i < top; i++) {
                fprintf(stderr, "OPTIMER dev %d %-64s n=%8llu  %9.2f ms  mean %8.2f us\n", device, v[i].first.c_str(),
                        (unsigned long long) v[i].second.n, v[i].second.us / 1000.0, v[i].second.us / v[i].second.n);
            }
            fflush(stderr);
        }
    }
};
static ggml_cuda_optimer & ggml_cuda_optimer_for(int device) { static ggml_cuda_optimer t[GGML_CUDA_MAX_DEVICES]; return t[device]; }
static std::string ggml_cuda_optimer_key(const ggml_tensor * node, int n_fused) {
    std::string k = ggml_op_name(node->op);
    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        char b[160]; snprintf(b, sizeof(b), " %s %lldx%lld n=%lld", ggml_type_name(node->src[0]->type),
                 (long long) node->src[0]->ne[0], (long long) node->src[0]->ne[1], (long long) node->src[1]->ne[1]);
        k += b;
        static const bool names = getenv("GGML_CUDA_TIME_OPS_NAMES") != nullptr;
        if (names) {   // weight name with the layer number stripped, so the classes still merge across layers
            std::string nm = node->src[0]->name; size_t p0 = nm.find("blk."); 
            if (p0 != std::string::npos) { size_t p1 = nm.find('.', p0 + 4); if (p1 != std::string::npos) { nm = nm.substr(p1 + 1); } }
            k += " [" + nm + "]";
        }
    } else if (node->op == GGML_OP_FLASH_ATTN_EXT || node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_CONCAT || node->op == GGML_OP_CPY) {
        char b[64]; snprintf(b, sizeof(b), " %lldx%lld", (long long) node->ne[0], (long long) node->ne[1]); k += b;
    }
    if (n_fused > 0) { k += " +fused" + std::to_string(n_fused); }
    return k;
}

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, uint64_t graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            if (ggml_cuda_optimer::enabled() && !use_cuda_graph) { ggml_cuda_optimer_for(cuda_ctx->device).set_tag(cgraph); }
            for (auto & e : cuda_ctx->gdn_state_gather) { e = {}; } // deferrals never outlive one evaluation
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        cuda_ctx->curr_stream_no = 0;
                        for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                            // Wait on join events of forked streams in the main stream
                            CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[i - 1],
                                                       cuda_ctx->stream(cuda_ctx->device, i)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]));
                        }

                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
                        GGML_ASSERT (concurrent_event->stream_mapping.find(node) != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                } else if (i - prev_i > 1) {
                    //the previous node was fused
                    const ggml_tensor * prev_node = cgraph->nodes[i - 1];
                    try_launch_concurrent_event(prev_node);

                    if (is_concurrent_event_active) {
                        cuda_ctx->curr_stream_no = concurrent_event->stream_mapping[node];
                        GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", cuda_ctx->curr_stream_no, node->name);
                    }
                }

                prev_i = i;

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // halo-hybrid: a decode recurrent-state gather that the gated_delta_net reads through its index
                if (node->op == GGML_OP_GET_ROWS && !ggml_cuda_fusion_disabled() && stream_ctx.concurrent_events.empty() &&
                        ggml_cuda_try_defer_gdn_state_gather(cuda_ctx, cgraph, i)) {
                    continue;
                }

                // halo-hybrid: a run of supported nodes as one resident kernel (persist.cu, GGML_CUDA_PERSIST=1)
                {
                    const int consumed = ggml_cuda_persist_region(*cuda_ctx, cgraph, i);
                    if (consumed > 0) {
                        i += consumed - 1;
                        continue;
                    }
                }

                const bool optimer_on = ggml_cuda_optimer::enabled() && !use_cuda_graph;
                if (optimer_on) { ggml_cuda_optimer_for(cuda_ctx->device).begin(cuda_ctx->stream(), ggml_cuda_optimer_key(node, 0)); }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (nodes_to_skip != 0) {
                    if (optimer_on) { auto & t = ggml_cuda_optimer_for(cuda_ctx->device); t.keys.back() = t.tag + ggml_cuda_optimer_key(node, nodes_to_skip); t.end(cuda_ctx->stream()); }
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                if (optimer_on) { ggml_cuda_optimer_for(cuda_ctx->device).end(cuda_ctx->stream()); }

                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
            if (ggml_cuda_optimer::enabled()) { ggml_cuda_optimer_for(cuda_ctx->device).flush(cuda_ctx->device); }
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, uint64_t graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    ggml_cuda_q8_side_reset(*cuda_ctx);   // per-graph registry of producer-side q8_1 activation copies
    ggml_cuda_dsv4_hc_mix_scratch_init(*cuda_ctx);   // once per context, before any capture

    {   // halo-hybrid: GGML_CUDA_PAD_KERNELS=<n> appends n empty dependent kernels to every graph. The slope of
        // tok/s against n is the marginal cost of one kernel boundary in the real decode loop, which is what any
        // "fuse to fewer kernels" plan is actually buying.
        static const int pad = getenv("GGML_CUDA_PAD_KERNELS") ? atoi(getenv("GGML_CUDA_PAD_KERNELS")) : 0;
        if (pad > 0) {
            ggml_cuda_pad_kernels(*cuda_ctx, pad);
        }
    }

    {   // halo-hybrid: GGML_CUDA_DUMP_GRAPH=<n> prints the n-th graph's node list once (op, shapes, sources), which
        // is how the per-layer kernel inventory is counted
        static const int dump = getenv("GGML_CUDA_DUMP_GRAPH") ? atoi(getenv("GGML_CUDA_DUMP_GRAPH")) : 0;
        static int seen = 0;
        if (dump && ++seen == dump) {
            for (int i = 0; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                char srcs[256]; srcs[0] = 0;
                for (int j = 0; j < GGML_MAX_SRC && n->src[j]; ++j) {
                    char one[64];
                    snprintf(one, sizeof(one), "%s%s[%s]", j ? "," : "", n->src[j]->name, ggml_type_name(n->src[j]->type));
                    strncat(srcs, one, sizeof(srcs) - strlen(srcs) - 1);
                }
                GGML_LOG_WARN("gnode %4d %-14s %-22s %5lldx%-5lld <- %s\n", i, ggml_op_name(n->op), n->name,
                    (long long) n->ne[0], (long long) n->ne[1], srcs);
            }
        }
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    uint64_t graph_key = 0;

#ifdef USE_CUDA_GRAPH
    graph_key = ggml_cuda_graph_get_key(cgraph);

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    if (graph->is_enabled()) {
        const bool graph_compatible = ggml_cuda_graph_check_compability(cgraph);
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);

            if (!graph->warmup_complete) {
                // Warmup: need at least 2 calls with no property change on the 2nd call
                if (!properties_changed) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                }
                // else: properties changed or first call - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                if (properties_changed) {
                    // Properties changed - reset warmup, execute directly until stable again
                    graph->warmup_complete = false;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
                } else {
                    use_cuda_graph = true;
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        }
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    ggml_cuda_persist_debug_after(*cuda_ctx);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_set_device(cuda_ctx->device);

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_cuda_set_device(cuda_ctx->device);

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

    // halo-hybrid: keep q/k/v of the KDA conv-input assembly alive until conv_input is allocated, so the fused kernel
    // (ggml_cuda_try_fuse_kda_conv_rows) never finds conv_input placed over them and has to fall back
    if (!disable_fusion && !ggml_cuda_kda_conv_rows_disabled()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_kda_conv_rows_match m;
            if (cgraph->nodes[i]->op != GGML_OP_CONCAT || !ggml_cuda_kda_conv_rows_find(cgraph, i, m)) {
                continue;
            }
            params->add_alloc_dep(params->user_data, m.c1->src[0], m.c3);
            params->add_alloc_dep(params->user_data, m.c1->src[1], m.c3);
            params->add_alloc_dep(params->user_data, m.c2->src[1], m.c3);
            i = m.last;
        }
    }

    // halo-hybrid: keep the state-gather index alive until the gdn that reads it in place of the gather
    // (ggml_cuda_try_defer_gdn_state_gather); otherwise the last KDA layer of a split finds s_copy's bytes reused
    if (!disable_fusion && !ggml_cuda_gdn_state_gather_disabled()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const int ig = ggml_cuda_gdn_state_gather_find(cgraph, i);
            if (ig >= 0) {
                params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[1], cgraph->nodes[ig]);
            }
        }
    }

    // halo-hybrid: hoist GEMVs that read the same activation next to each other so the grouped GEMV (see the
    // MUL_MAT group in ggml_cuda_compute_forward_group) sees them as one run. Qwen3.8 puts a sigmoid between the
    // DeltaNet beta and alpha projections, and the shared expert's gate/up/GLU (plus its 1-row gate) 37 nodes after
    // the router that reads the same vector. Moving a node EARLIER is always dependency-safe when its inputs are a
    // weight and the anchor's own activation; this runs before allocation, so lifetimes follow the new order.
    // Decode only (activation of at most 4 columns). GGML_CUDA_NO_GEMV_HOIST=1 disables it.
    static const bool no_hoist = getenv("GGML_CUDA_NO_GEMV_HOIST") != nullptr;
    if (!disable_fusion && !no_hoist && !cuda_ctx->mmvq_group_disabled) {
        auto is_weight_gemv = [&](const ggml_tensor * t) {
            if (t->op != GGML_OP_MUL_MAT || !t->src[0] || !t->src[1] || t->type != GGML_TYPE_F32) return false;
            const ggml_tensor * w = t->src[0], * x = t->src[1];
            if (w->op != GGML_OP_NONE || !w->buffer || !ggml_backend_buft_is_cuda(w->buffer->buft)) return false;   // a stored weight, not a computed tensor
            if (w->ne[2] != 1 || w->ne[3] != 1 || x->type != GGML_TYPE_F32 || x->ne[1] > 4 || x->ne[2] != 1 || x->ne[3] != 1) return false;
            return ggml_is_quantized(w->type) || w->type == GGML_TYPE_F32;
        };
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_tensor * anchor = cgraph->nodes[i];
            if (!is_weight_gemv(anchor)) {
                continue;
            }
            const ggml_tensor * x = anchor->src[1];
            // end of the run already adjacent to the anchor (reshapes between GEMVs are free, as in the detector)
            int last = i;
            for (int j = i + 1; j < cgraph->n_nodes; ++j) {
                const ggml_tensor * t = cgraph->nodes[j];
                if (t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE) continue;
                if (is_weight_gemv(t) && t->src[1] == x) { last = j; continue; }
                if (t->op == GGML_OP_GLU && j >= 2 && t->src[0] == cgraph->nodes[j-2] && t->src[1] == cgraph->nodes[j-1] && last == j-1) { last = j; continue; }
                break;
            }
            // pull later GEMVs on the same activation up to just after the run
            for (int j = last + 1; j < cgraph->n_nodes; ++j) {
                ggml_tensor * t = cgraph->nodes[j];
                if (!is_weight_gemv(t) || t->src[1] != x) {
                    continue;
                }
                int len = 1;   // a (gate, up, GLU) triple moves together so the fused GLU path still sees it
                if (j + 2 < cgraph->n_nodes && is_weight_gemv(cgraph->nodes[j+1]) && cgraph->nodes[j+1]->src[1] == x &&
                    cgraph->nodes[j+2]->op == GGML_OP_GLU && cgraph->nodes[j+2]->src[0] == t && cgraph->nodes[j+2]->src[1] == cgraph->nodes[j+1]) {
                    len = 3;
                } else if (j + 1 < cgraph->n_nodes && cgraph->nodes[j+1]->op == GGML_OP_GLU && cgraph->nodes[j+1]->src[0] == t) {
                    continue;   // half of a fused pair we did not recognise; leave it
                }
                // the GLU triple's own outputs must not be read by anything between last+1 and j (they are not: those
                // nodes precede it in the original order), and its inputs are x and weights: safe to rotate up
                std::rotate(cgraph->nodes + last + 1, cgraph->nodes + j, cgraph->nodes + j + len);
                last += len;
                j = last;
            }
            i = last;
        }
    }

    // halo-hybrid: GDN conv front (ggml_cuda_try_fuse_gdn_conv_front). Move the conv..l2 run up behind the slot copies
    //     so the whole front is one contiguous run at dispatch, and keep x and the states alive until the l2 output is
    //     allocated, so neither output lands on memory the fused kernel still reads. Runs after the GEMV hoist, which
    //     never moves anything into this run (it only pulls GEMVs up to a GEMV run).
    if (!disable_fusion && !ggml_cuda_gdn_conv_front_disabled()) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_gdn_conv_front_match m;
            if (cgraph->nodes[i]->op != GGML_OP_CONCAT || !ggml_cuda_gdn_conv_front_find(cgraph, i, m, /*adjacent =*/ false)) {
                continue;
            }
            const int to = hc_next(cgraph, m.last_cpy + 1);
            if (to < m.i_conv) {
                // [to, i_conv) precede the run in the original order, so none of them reads it; the run reads only
                // conv_input and the weight, both available at `to`
                std::rotate(cgraph->nodes + to, cgraph->nodes + m.i_conv, cgraph->nodes + m.i_l2 + 1);
            }
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(hc_root(m.c3->src[0])), m.l2);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(hc_root(m.c3->src[1])), m.l2);
            i = to + (m.i_l2 - m.i_conv);
        }
    }

    // halo-hybrid: the fused MoE tail reads ids / router weights / the shared gate logit in every block while writing
    // ffn_out; keep them alive until ffn_out is allocated so ggml-alloc never places ffn_out over them. AFTER the GEMV
    // hoist: the matcher needs the shared expert's gate/up/GLU pulled next to the router (in builder order it sits
    // between the reduction and the shared-expert down, and nothing would match here)
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            ggml_cuda_moe_tail_match m;
            if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT_ID || !ggml_cuda_moe_tail_find(cuda_ctx->device, cgraph, i, m)) {
                continue;
            }
            params->add_alloc_dep(params->user_data, m.mmid->src[2], m.out);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(m.red.weights), m.out);
            if (m.red.expert_scale != nullptr) {
                params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(m.red.expert_scale), m.out);
            }
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(m.g), m.out);
            i = m.last;
        }
    }

#ifdef USE_CUDA_GRAPH
    const uint64_t graph_key = ggml_cuda_graph_get_key(cgraph);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    static bool enable_graph_optimization = [] {
        const char * env     = getenv("GGML_CUDA_GRAPH_OPT");
        return env != nullptr && atoi(env) == 1;
    }();

    if (!enable_graph_optimization) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    // The multi-device gate is lifted when GGML_CUDA_GRAPH_OPT_MULTI=1: the optimisation works
    // per backend context on that context's own graph, and on a two-GPU MoE layout the
    // ~1900 per-token dispatch gaps are the dominant idle (experiment, 2026-08-26).
    static const bool allow_multi = getenv("GGML_CUDA_GRAPH_OPT_MULTI") != nullptr;
    if (!use_cuda_graph || (!allow_multi && ggml_backend_cuda_get_device_count() != 1)) {
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
    /* .cpy_tensor_async_nowait = */ ggml_backend_cuda_cpy_tensor_async_nowait,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                        // gathered with the generic 32-block kernel (e.g. the 160-wide PLE n-gram table of qwen4exp)
                        return op->src[0]->ne[0] % QK4_NL == 0;
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_KQ_MASK_BUILD:
            return (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) && op->ne[1] <= 65535;
        case GGML_OP_DSV4_HC_MIX:
            return op->src[0]->type == GGML_TYPE_F32 && (op->src[1]->type == GGML_TYPE_Q8_0 || op->src[1]->type == GGML_TYPE_F32) &&
                op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                4*op->src[0]->ne[0] == 16*1024 && op->src[0]->nb[1] % 16 == 0 && op->src[0]->nb[2] % 16 == 0;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft));
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
