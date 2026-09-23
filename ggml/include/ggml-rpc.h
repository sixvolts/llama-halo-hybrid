#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// 7: halo-hybrid wire (GRAPH_COMPUTE / GRAPH_RECOMPUTE return an empty reply that the client waits for); an upstream
//    6.x peer would hang on the missing reply or desync on the extra one, so the handshake must reject the pairing
#define RPC_PROTO_MAJOR_VERSION    7
// 7.4: GGML_OP_DSV4_HC_MIX appended to the op enum (ids before it unchanged) and the HELLO reply carries
//      GGML_OP_COUNT, which the client compares: a peer built with a different op table is refused at the
//      handshake instead of executing shifted op ids (the patch field is never compared, so a patch bump alone
//      would not have protected the pairing)
// 7.5: GGML_OP_KQ_MASK_BUILD appended (masks built on the device; both hosts must carry the op)
#define RPC_PROTO_MINOR_VERSION 6
// 7.4.1: rpc_tensor.flags carries RPC_TENSOR_FLAG_WEIGHTS (client buffer usage WEIGHTS); the server marks the buffer so
//        its scheduler places ops by their weights. Compatible both ways (an older peer ignores or never sets the bit).
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT has changed - bump RPC_PROTO_MINOR_VERSION (the handshake compares major/minor only)");
static_assert(GGML_OP_COUNT <= 255, "GGML_OP_COUNT no longer fits the HELLO op_count byte - widen it");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
