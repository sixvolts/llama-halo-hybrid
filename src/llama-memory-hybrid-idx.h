#pragma once

#include "llama-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

// TODO: this memory module is pending complete reimplementation - do not use for model other than Qwen4

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    // Defined out of line because kpool_layout is incomplete here.
    ~llama_memory_hybrid_idx();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // halo-hybrid (C6): per QSA layer, the finished (pooled, normed, roped) indexer key of every complete block of each
    // stream's one sequence, [idx_dim, qsa_blk_max, n_stream] f32, row = block index (position / ratio). A decode step
    // then recomputes only the trailing blocks its tokens touch instead of re-pooling the whole window. nullptr when
    // absent (no indexer, LLAMA_QSA_BLK_CACHE=0).
    ggml_tensor * get_qsa_blk_k(int32_t il) const;

    // what the next graph can do with that cache, for a ubatch already applied to the cells. The graph has one stream
    // per sequence of the ubatch when the cache keeps a stream per sequence (otherwise one); group s of the ubatch is
    // the tokens [s*n_tps, (s+1)*n_tps).
    struct qsa_plan {
        bool     usable = false;    // every group: one sequence, alone in its stream's cells, hole-free from position 0,
                                    // and the groups' streams consecutive (the stream range the graph's views cover)
        bool     incr   = false;    // usable, and in every stream the blocks before first[s] are current
        uint32_t s0     = 0;        // physical stream of group 0
        std::vector<int32_t> n_bid; // complete blocks after this ubatch, per group
        std::vector<int32_t> first; // first block the incremental graph recomputes (n_bid - n_re), per group

        // a unified cache holding several sequences (the KV window, llama_kv_cache::get_kv_window): the groups are
        // the ubatch's sequences, each laid out position p -> window cell base[g] + p with no other sequence in its
        // span; s0 is then the first sequence's slab of the block-key cache (one slab per sequence) and the graph
        // scores n_blocks blocks per group instead of every block of the window
        bool                 shared   = false;
        std::vector<int32_t> base;
        int64_t              n_blocks = 0;
    };
    // usable = false without the cache. lo, n_kv: the ubatch's KV window
    qsa_plan qsa_blk_plan(const llama_ubatch & ubatch, uint32_t ratio, int32_t n_re, uint32_t lo, uint32_t n_kv) const;

    // halo-hybrid (C4): usable (see above) and every used cell of each group holds its position (from base[g]) - the
    // layout ggml_qsa_top_k assumes; works without the block-key cache
    qsa_plan qsa_identity(const llama_ubatch & ubatch, uint32_t ratio, uint32_t lo, uint32_t n_kv) const;

    // the cache layout of shared_pool(): several sequences in one unified cache, attended through the KV window
    bool qsa_shared_pool() const;

    // block-key cache slab (and qsa_valid_pos entry) of a sequence: its stream, or the sequence in a shared pool
    uint32_t qsa_slab(llama_seq_id seq_id) const;

    // per slab: blocks whose first qsa_valid_pos[s] positions hold cells that the cache reflects; raised after
    // a graph computes them, lowered by seq_rm, zeroed by any other edit of the cells or their positions
    mutable std::vector<llama_pos> qsa_valid_pos;
    void qsa_valid_reset() const { std::fill(qsa_valid_pos.begin(), qsa_valid_pos.end(), 0); }

    // block-compressed sparse attention (qwen4exp QSA) over the cells of the indexer cache.
    // Blocks cut the position line, not the cell array, so no caller assumes a contiguous layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    // blk_bias asks for the bias per block instead: [n_blocks, n_tokens/ns, ns]
    // the caller then adds the attention mask, the only part of the bias that varies within a block
    // causal_attn selects the rule: causal forces the query's own block on, non-causal lets every visible block compete on score
    // cell j of all of these is cell lo + j of the cache, the start of the KV window (llama_kv_cache::get_kv_window)
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn, uint32_t lo) const;

    // The model's indexer pool size.
    uint32_t get_kpool() const { return hparams_idx.indexer_kpool; }

    // Which cells of a sequence make up which pool of kpool consecutive positions.
    // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
    // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
    struct kpool_layout;

    const kpool_layout & kpool_layout_update();
    const kpool_layout & kpool_layout_get() const;

    // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
    // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
    using stale_pos_t = std::array<llama_pos, LLAMA_MAX_SEQ>;

    static constexpr llama_pos POS_CLEAN = std::numeric_limits<llama_pos>::max();

    static stale_pos_t stale_pos_clean() {
        stale_pos_t res;
        res.fill(POS_CLEAN);
        return res;
    }

    const stale_pos_t & mem_idx_stale_get() const { return mem_idx_stale; }
    void mem_idx_stale_clear() { mem_idx_stale.fill(POS_CLEAN); }

private:
    // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
    // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
    void state_drop(llama_seq_id seq_id);

    // the indexer cache holds one key head per layer, so it needs its own hparams:
    // llama_kv_cache keeps a reference to what it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    qsa_plan qsa_blk_plan_cells(const llama_ubatch & ubatch, uint32_t ratio, int32_t n_re, uint32_t lo, uint32_t n_kv) const;
    qsa_plan qsa_blk_plan_shared(const llama_ubatch & ubatch, uint32_t ratio, int32_t n_re, uint32_t lo, uint32_t n_kv) const;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> qsa_ctxs_bufs;
    std::unordered_map<int32_t, ggml_tensor *> qsa_blk_k;
    // unique_ptr because kpool_layout is incomplete here
    std::unique_ptr<kpool_layout> kpool_lay;

    // whether the current layout has cells shared between sequences (kpool_layout is incomplete here, so out of line)
    bool kpool_layout_shared() const;

    // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
    void mem_idx_stale_set(llama_seq_id seq_id, llama_pos p0);

    // the position an edit at p0 stales the sequence from
    llama_pos mem_idx_stale_pos(llama_seq_id seq_id, llama_pos p0) const;

    stale_pos_t mem_idx_stale = stale_pos_clean();
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    class kpool_access {
    public:
        ggml_tensor * gather_key_gate(ggml_tensor * idxs) const;
        ggml_tensor * scatter_pooled(ggml_tensor * values, ggml_tensor * idxs) const;
        ggml_tensor * gather_pooled(ggml_tensor * idxs) const;

    private:
        friend class llama_memory_hybrid_idx_context;

        kpool_access(ggml_context * ctx, ggml_tensor * k, int64_t n_embd);

        ggml_context * ctx;
        ggml_tensor  * key_gate;
        ggml_tensor  * pooled;
    };

    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context(); // Defined out of line because kpool_state is incomplete here.

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    bool needs_reserve() const override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr with no indexer
    const llama_kv_cache_context * get_idx() const;

    // the memory itself (the QSA block-key cache and its plan live there)
    const llama_memory_hybrid_idx * get_mem() const { return mem; }

    // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
    uint32_t get_n_stream() const;

    // glm5-next, complete pools of kpool consecutive positions per sequence, scored as whole pools.
    uint32_t get_n_kpool    () const; // Padded pool count, where the last pool is always unused.
    uint32_t get_n_kpool_new() const; // Exact count of pools completed by the current ubatch.
    bool get_kpool_cache_safe() const;
    kpool_access get_kpool_access(ggml_context * ctx, int32_t il, int64_t n_embd) const;
    ggml_tensor * gather_mla_rows(ggml_context * ctx, ggml_tensor * idxs, int64_t n_rows, int64_t n_embd, int32_t il) const;
    void set_input_kpool(ggml_tensor * pool_cells, ggml_tensor * pool_idxs, ggml_tensor * pool_mask, ggml_tensor * tail_idxs,
                         ggml_tensor * gather_mask, bool gather, ggml_tensor * new_pool_idxs, ggml_tensor * new_pool_rep,
                         const llama_ubatch * ubatch) const;
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio,
                       bool blk_bias, bool causal_attn) const;

private:
    llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, read from the slot infos before ctx_idx takes them
    // declared first, so it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;

    // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
    struct kpool_state;
    kpool_state kpool_build_sizes() const;
    void kpool_build_state(const llama_ubatch & ubatch);
    const kpool_state & kpool_cur() const;

    // unique_ptr because kpool_state is incomplete here.
    std::unique_ptr<kpool_state> kpool_st;

    // The ubatch kpool_st was built for, guards against reads before apply.
    size_t i_kpool = SIZE_MAX;

    // Whether this context tracks k-pool states.
    bool kpool_track() const;

    // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
    llama_memory_hybrid_idx::stale_pos_t mem_idx_stale_batch = llama_memory_hybrid_idx::stale_pos_clean();
};
