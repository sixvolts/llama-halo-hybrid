#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <map>
#include <utility>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;

// GLM-5-Next indexer pooling. no input may hold a negative index (ggml_set_rows asserts
// i1 >= 0, ggml_get_rows has no sentinel), so unusable entries are clamped and masked.

// n_kv/kpool (exact only while the sequences' cells are disjoint) plus 2 per seq for rebasing
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of Glm5NextTextIndexer.forward: must run over POOLS, a cell cut takes partial pools
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   pool_cells  pool member -> cell, 0 if not resident
//   pool_bias   computed, NOT gathered at the last member, which an incomplete pool lacks
//   cand_mask   bounds top-k spills a partial seq_rm would let escape
// pool_reps / new_pool_cells / new_pool_reps are nullptr when the cache is off, and an entry
// is emitted only for filled == kpool: cell 0 is real, so writing its 0 slot would clobber
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
              ggml_tensor    * pool_reps,
              ggml_tensor    * new_pool_cells,
              ggml_tensor    * new_pool_reps,
        // halo-hybrid: device-built masks - when sel_mask is NULL these receive the per-cell / per-query ints instead
              ggml_tensor    * dev_pos_at,
              ggml_tensor    * dev_pool_of,
              ggml_tensor    * dev_q,
              ggml_tensor    * dev_tail_start,
              ggml_tensor    * dev_bo_vis,
        // a global row is strm_of[s]*kv_size + cell; only read when pool_reps is set
        const uint32_t       * strm_of,
              int64_t          kv_size,
        // re-emit every complete pool, not only those this ubatch closed; set after a position mutation
              bool             rebuild,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// one map per ubatch; valid only while every indexer layer sees the same candidate set
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    // graph reuse: the tensor shapes are a function of (n_kv, ubatch, kpool_dirty); everything else is re-set per step
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // pooled-key cache: a pool's value lives in the row of its LAST member, a pure function of
    // cell content (seq_cp shares it, rebase leaves it alone)
    ggml_tensor * pool_reps      = nullptr; // I32 [n_pools, n_stream]  stream-local rep cell
    ggml_tensor * new_pool_cells = nullptr; // I32 [kpool*n_new_max, n_stream] members to (re)pool
    ggml_tensor * new_pool_reps  = nullptr; // I64 [n_new_max*n_stream] GLOBAL dest row

    // fixed for the decode phase: a shape tracking pools-closed-this-step would flip the graph
    // topology every kpool tokens and force CUDA-graph recapture
    uint32_t n_new_max = 0;

    // set at build time: re-emit every pool after a position mutation
    bool rebuild = false;

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    // halo-hybrid: device-built masks (ggml_kq_mask_build): per-cell / per-query ints replace the uploads
    ggml_tensor * dev_pos_at     = nullptr; // I32 [n_kv]  cell position, -1 unusable
    ggml_tensor * dev_pool_of    = nullptr; // I32 [n_kv]  pool index of the cell, -1 none
    ggml_tensor * dev_q          = nullptr; // I32 [n_tps] query position
    ggml_tensor * dev_tail_start = nullptr; // I32 [n_tps]
    ggml_tensor * dev_bo_vis     = nullptr; // I32 [n_tps]
    ggml_tensor * sel_cur  = nullptr;       // the instances of the last selected device (host mode: sel_mask/cand_mask)
    ggml_tensor * cand_cur = nullptr;
    std::map<ggml_backend_t, std::pair<ggml_tensor *, ggml_tensor *>> dev_masks; // keyed by the backend the pin assigned
    void select_device(ggml_context * ctx0, ggml_backend_sched_t sched, const std::function<void(ggml_tensor *, const char *, int)> & pin, int il);

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
