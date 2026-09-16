#pragma once

#include "common.cuh"

// halo-hybrid: persistent decode regions (docs/halo-hybrid/PERSISTENT-DECODE.md).
//
// A run of consecutive graph nodes that the persistent path supports is compiled into a task list and executed by
// ONE resident kernel (one 1024-thread workgroup per WGP) instead of one launch per node: workgroups walk the list
// in order, wait on per-task dependency counters (data-flow edges plus memory-range hazards from ggml-alloc's buffer
// reuse), take work items by static striding, and retire with one release atomic per block. Any node the path does
// not implement ends the region and runs as today.
//
// GGML_CUDA_PERSIST=1 enables (off by default: it is measurably slower than the per-node path on gfx1151, see the
// stage 3 section of the doc for why). GGML_CUDA_PERSIST_MIN=<n> is the minimum region length (default 4; the loss
// shrinks as this grows, and test-backend-ops uses 1 so every op body is compared against the CPU on its own).
// Other knobs: _OPS=<bitmask of pk_op> , _NEL=<max elements per node>, _MAX=<max nodes>, _GRID, _SERIAL=1,
// _MMVQ=rpw,ku,w, _VERIFY=1 (re-run each region on the normal path and compare), _TRACE=<n> (per-task timeline of
// the n-th execution), _DEBUG=1|2 (stuck regions / region composition), _NOGLUFUSE=1.
//
// Returns the number of nodes consumed starting at cgraph->nodes[i], or 0 when the region path does not apply.
int ggml_cuda_persist_region(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);

// GGML_CUDA_PERSIST_DEBUG=1: after a graph compute, report regions whose launch hit the bounded wait
void ggml_cuda_persist_debug_after(ggml_backend_cuda_context & ctx);

bool ggml_cuda_compute_forward_node(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst);   // defined in ggml-cuda.cu
