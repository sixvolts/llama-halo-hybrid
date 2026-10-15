#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// halo-hybrid: the fork's RPC protocol. Its major (107) is outside upstream's number space, so an upstream rpc-server or
// client is refused at the handshake instead of misreading this wire; the client also requires the same minor and the
// same GGML_OP_COUNT, so both hosts must run the same build of this tree (there is no capability negotiation).
// What the fork's wire adds to upstream 7.0: GRAPH_COMPUTE / GRAPH_RECOMPUTE return an empty reply that the client
// waits for; the HELLO reply carries GGML_OP_COUNT; the fork ops (DSV4_HC_MIX, KQ_MASK_BUILD, QSA_TOP_K, QSA_HEAD_SUM);
// SET_TENSOR's cache_flag byte (from upstream); rpc_tensor.flags carries RPC_TENSOR_FLAG_WEIGHTS and
// GGML_TENSOR_FLAG_BOUNDARY; the commands COPY_TENSOR_ASYNC, GRAPH_COMPUTE_SCHED and SUPPORTS_OP (ids from 128).
// The protocol was 7.0 to 7.10 until 2026-10-15; bump the minor for every wire change.
#define RPC_PROTO_MAJOR_VERSION  107
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 105, "GGML_OP_COUNT has changed - bump RPC_PROTO_MINOR_VERSION");
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
